from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel
from jarvis_v2.memory.store import MemoryRecord, MemoryStore
from jarvis_v2.scripts.test_runtime import (
    approve_pending_runtime_approval,
    make_temp_runtime,
)
from jarvis_v2.tools.registry import ToolArgumentType


class StaticPlanner:
    def __init__(self, tool_name: str, args: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.args = dict(args)

    def plan(self, user_input: str) -> Plan:
        return Plan(
            user_input,
            [PlannedAction(self.tool_name, dict(self.args), "memory approval binding smoke")],
            notes="static_memory_approval_binding_fixture",
        )


def _seed(
    runtime,
    title: str,
    body: str = "body",
    *,
    category: str = "fact",
    confidence: float = 0.8,
) -> int:
    return runtime.store.add_memory(
        MemoryRecord(category, title, body, source="smoke", confidence=confidence)
    )


def _memory_rows(store: MemoryStore) -> list[tuple[Any, ...]]:
    with store.connect() as conn:
        return [
            tuple(row)
            for row in conn.execute(
                """
                SELECT id, category, title, body, source, confidence,
                       created_at, updated_at, revision
                FROM memories ORDER BY id
                """
            )
        ]


def _queue(
    runtime,
    command: str,
    tool_name: str,
    *,
    raw_args: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any], Any]:
    if raw_args is not None:
        runtime.planner = StaticPlanner(tool_name, raw_args)
    held = runtime.handle(command)
    approvals = [
        result
        for result in held.tool_results
        if result.tool_name == tool_name
        and result.metadata.get("requires_confirmation") is True
    ]
    if len(approvals) != 1:
        raise SystemExit(
            f"{tool_name} did not queue exactly one approval: {command!r} -> {held}"
        )
    result = approvals[0]
    approval_id = result.metadata.get("approval_id")
    public_args = result.metadata.get("planned_args")
    if type(approval_id) is not int or type(public_args) is not dict:
        raise SystemExit(f"{tool_name} approval missed its exact stored arguments: {result}")
    row = runtime.store.get_pending_approval(approval_id)
    planned_args = json.loads(row["planned_args"]) if row is not None else None
    if type(planned_args) is not dict:
        raise SystemExit(f"{tool_name} approval did not persist its exact bound arguments")
    return approval_id, planned_args, held


def _execute(runtime, command: str, approval_id: int):
    transition = approve_pending_runtime_approval(runtime, approval_id)
    public_command = re.sub(
        r"(?i)(\btoken\s+)[0-9a-f]{64}(?=\s+to\s+(?:a\s+)?(?:decision|preference)\s*:)",
        r"\1<review-token>",
        command,
    )
    if transition.metadata.get("rerun_user_input") != public_command:
        raise SystemExit("approval transition changed the stored memory request")
    return runtime.handle(command, approved=True, approved_approval_id=approval_id)


def _assert_bound_shape(tool_name: str, args: dict[str, Any], raw: dict[str, Any]) -> None:
    expected = {
        "delete_memory": set(raw) | {"target_revision", "target_binding"},
        "edit_memory": set(raw) | {"target_revision", "target_binding"},
        "merge_memories": set(raw)
        | {"keep_revision", "keep_binding", "delete_revision", "delete_binding"},
        "promote_memory_to_decision": (set(raw) - {"review_token"})
        | {"review_binding", "target_revision", "target_binding"},
        "promote_memory_to_preference": (set(raw) - {"review_token"})
        | {"review_binding", "target_revision", "target_binding"},
    }[tool_name]
    if set(args) != expected:
        raise SystemExit(f"{tool_name} approval stored the wrong argument shape: {args}")
    binding_keys = [key for key in args if key.endswith("_binding")]
    revision_keys = [key for key in args if key.endswith("_revision")]
    if any(
        type(args[key]) is not str
        or len(args[key]) != 64
        or any(char not in "0123456789abcdef" for char in args[key])
        for key in binding_keys
    ):
        raise SystemExit(f"{tool_name} approval stored a malformed opaque binding: {args}")
    if any(type(args[key]) is not int or args[key] < 1 for key in revision_keys):
        raise SystemExit(f"{tool_name} approval stored a malformed target revision: {args}")


def _assert_binding_hidden(runtime, held, approval_id: int, args: dict[str, Any]) -> None:
    bindings = [value for key, value in args.items() if key.endswith("_binding")]
    held_result = next(
        result for result in held.tool_results if result.metadata.get("approval_id") == approval_id
    )
    displays: list[tuple[str, Any]] = [
        ("approval hold output", held_result.output),
        ("runtime approval response", held.response),
        ("approval hold metadata", held_result.metadata),
        ("runtime approval plan", held.plan),
        ("runtime approval metadata", held.metadata),
        ("approval hold display metadata", held_result.metadata.get("planned_args_display", {})),
    ]
    for tool_name in (
        "list_pending_approvals",
        "approval_readiness_packet",
        "approval_execution_packet",
    ):
        packet_args = (
            {} if tool_name == "list_pending_approvals" else {"approval_id": approval_id}
        )
        packet = runtime.registry.get(tool_name).handler(packet_args)
        if not packet.ok:
            raise SystemExit(f"could not render {tool_name} for approval #{approval_id}: {packet}")
        displays.extend(
            [
                (f"{tool_name} output", packet.output),
                (f"{tool_name} display metadata", packet.metadata),
            ]
        )
    transition = approve_pending_runtime_approval(runtime, approval_id)
    displays.append(("approval transition metadata", transition.metadata))
    for label, display in displays:
        rendered = json.dumps(display, sort_keys=True, default=str)
        leaked = [binding for binding in bindings if binding in rendered]
        if leaked:
            raise SystemExit(f"{label} exposed an opaque memory binding")


def _assert_contracts(runtime) -> None:
    expected = {
        "delete_memory": (
            {"memory_id"},
            {"memory_id", "target_revision", "target_binding"},
        ),
        "edit_memory": (
            {"memory_id", "category", "title", "body", "confidence"},
            {
                "memory_id",
                "category",
                "title",
                "body",
                "confidence",
                "target_revision",
                "target_binding",
            },
        ),
        "merge_memories": (
            {"keep_id", "delete_id"},
            {
                "keep_id",
                "delete_id",
                "keep_revision",
                "keep_binding",
                "delete_revision",
                "delete_binding",
            },
        ),
        "promote_memory_to_decision": (
            {
                "memory_id",
                "reviewed_revision",
                "review_token",
                "title",
                "rationale",
                "impact",
            },
            {
                "memory_id",
                "reviewed_revision",
                "review_binding",
                "title",
                "rationale",
                "impact",
                "target_revision",
                "target_binding",
            },
        ),
        "promote_memory_to_preference": (
            {
                "memory_id",
                "reviewed_revision",
                "review_token",
                "category",
                "key",
                "value",
            },
            {
                "memory_id",
                "reviewed_revision",
                "review_binding",
                "category",
                "key",
                "value",
                "target_revision",
                "target_binding",
            },
        ),
    }
    integer = frozenset({ToolArgumentType.INTEGER})
    string = frozenset({ToolArgumentType.STRING})
    for tool_name, (raw_names, bound_names) in expected.items():
        tool = runtime.registry.get(tool_name)
        raw = tool.argument_contract
        bound = tool.approval_argument_contract
        if raw is None or bound is None or tool.approval_argument_resolver is None:
            raise SystemExit(f"{tool_name} is missing its two-phase approval contract")
        raw_fields = {field.name: field for field in raw.fields}
        bound_fields = {field.name: field for field in bound.fields}
        if set(raw_fields) != raw_names or set(bound_fields) != bound_names:
            raise SystemExit(
                f"{tool_name} raw/bound contract drifted: "
                f"raw={set(raw_fields)}, bound={set(bound_fields)}"
            )
        required_raw = {
            "delete_memory": {"memory_id"},
            "edit_memory": {"memory_id"},
            "merge_memories": {"keep_id", "delete_id"},
            "promote_memory_to_decision": raw_names,
            "promote_memory_to_preference": raw_names,
        }[tool_name]
        if {name for name, field in raw_fields.items() if field.required} != required_raw:
            raise SystemExit(f"{tool_name} raw required arguments drifted")
        binding_names = {name for name in bound_fields if name.endswith("_binding")}
        revision_names = {name for name in bound_fields if name.endswith("_revision")}
        if any(
            not bound_fields[name].required or bound_fields[name].types != string
            for name in binding_names
        ):
            raise SystemExit(f"{tool_name} binding fields are not required strings")
        if any(
            not bound_fields[name].required or bound_fields[name].types != integer
            for name in revision_names
        ):
            raise SystemExit(f"{tool_name} revision fields are not required integers")


def test_contracts_natural_queue_and_private_display() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-contracts-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _assert_contracts(runtime)

        delete_id = _seed(runtime, "Natural delete")
        delete_command = f"delete memory {delete_id}"
        approval_id, args, held = _queue(runtime, delete_command, "delete_memory")
        _assert_bound_shape("delete_memory", args, {"memory_id": delete_id})
        _assert_binding_hidden(runtime, held, approval_id, args)

    with TemporaryDirectory(prefix="jarvis-memory-binding-merge-queue-") as temp:
        runtime = make_temp_runtime(Path(temp))
        keep_id = _seed(runtime, "Natural merge keep", "keep body")
        delete_id = _seed(runtime, "Natural merge delete", "delete body")
        merge_command = f"merge memory {delete_id} into {keep_id}"
        raw = {"keep_id": keep_id, "delete_id": delete_id}
        approval_id, args, held = _queue(runtime, merge_command, "merge_memories")
        _assert_bound_shape("merge_memories", args, raw)
        _assert_binding_hidden(runtime, held, approval_id, args)

    with TemporaryDirectory(prefix="jarvis-memory-binding-edit-queue-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(runtime, "Static edit", "before")
        raw = {
            "memory_id": memory_id,
            "category": "preference",
            "title": "Static edit updated",
            "body": "after",
            "confidence": 0.95,
        }
        approval_id, args, held = _queue(
            runtime,
            "edit the static memory fixture",
            "edit_memory",
            raw_args=raw,
        )
        _assert_bound_shape("edit_memory", args, raw)
        _assert_binding_hidden(runtime, held, approval_id, args)


def _assert_approved_once(runtime, command: str, approval_id: int):
    result = _execute(runtime, command, approval_id)
    if (
        not result.verified
        or len(result.tool_results) != 1
        or result.tool_results[0].ok is not True
        or result.tool_results[0].metadata.get("handler_invoked") is not True
    ):
        raise SystemExit(f"approved memory mutation did not execute exactly once: {result}")
    return result


def _assert_private_promotion_surface(
    label: str,
    value: Any,
    bound_args: dict[str, Any],
) -> None:
    rendered = json.dumps(value, sort_keys=True, default=str)
    if "review_token" in rendered:
        raise SystemExit(f"{label} retained the raw review_token argument")
    leaked = [
        key
        for key in ("review_binding", "target_binding")
        if bound_args[key] in rendered
    ]
    if leaked:
        raise SystemExit(f"{label} exposed opaque promotion bindings: {leaked}")


def _approve_private_promotion(
    runtime,
    command: str,
    tool_name: str,
    raw_args: dict[str, Any],
):
    approval_id, bound_args, held = _queue(
        runtime,
        command,
        tool_name,
        raw_args=raw_args,
    )
    _assert_bound_shape(tool_name, bound_args, raw_args)
    if "review_token" in bound_args:
        raise SystemExit(f"{tool_name} persisted raw review_token in approval-bound args")
    if (
        bound_args.get("review_binding") != raw_args["review_token"]
        or bound_args.get("target_binding") != raw_args["review_token"]
    ):
        raise SystemExit(f"{tool_name} did not bind the exact reviewed candidate")

    held_result = next(
        result for result in held.tool_results if result.metadata.get("approval_id") == approval_id
    )
    for label, display_args in (
        ("approval hold planned args", held_result.metadata.get("planned_args")),
        ("approval hold display args", held_result.metadata.get("planned_args_display")),
    ):
        if type(display_args) is not dict or set(display_args) != set(bound_args):
            raise SystemExit(f"{tool_name} {label} changed the bound argument keys")

    surfaces: list[tuple[str, Any]] = [
        ("approval hold output", held_result.output),
        ("approval hold metadata", held_result.metadata),
        ("runtime approval response", held.response),
    ]
    for surface_name in (
        "list_pending_approvals",
        "approval_readiness_packet",
        "approval_execution_packet",
    ):
        packet_args = (
            {} if surface_name == "list_pending_approvals" else {"approval_id": approval_id}
        )
        packet = runtime.registry.get(surface_name).handler(packet_args)
        if not packet.ok:
            raise SystemExit(
                f"could not render {surface_name} for {tool_name} approval #{approval_id}: {packet}"
            )
        surfaces.extend(
            [
                (f"{surface_name} output", packet.output),
                (f"{surface_name} metadata", packet.metadata),
            ]
        )

    transition = approve_pending_runtime_approval(runtime, approval_id)
    if transition.metadata.get("rerun_user_input") != command:
        raise SystemExit(f"{tool_name} approval transition changed the stored request")
    surfaces.extend(
        [
            ("approval transition output", transition.output),
            ("approval transition metadata", transition.metadata),
        ]
    )

    approved = runtime.handle(command, approved=True, approved_approval_id=approval_id)
    if (
        not approved.verified
        or len(approved.plan.actions) != 1
        or approved.plan.actions[0].tool_name != tool_name
        or set(approved.plan.actions[0].args) != set(bound_args)
        or len(approved.tool_results) != 1
        or approved.tool_results[0].tool_name != tool_name
        or approved.tool_results[0].ok is not True
        or approved.tool_results[0].metadata.get("handler_invoked") is not True
        or set(approved.tool_results[0].metadata.get("rerun_arg_keys", ()))
        != set(bound_args)
    ):
        raise SystemExit(f"approved {tool_name} did not execute exactly once: {approved}")
    surfaces.extend(
        [
            ("approved execution response", approved.response),
            ("approved execution public plan", approved.plan),
            ("approved execution result output", approved.tool_results[0].output),
            ("approved execution result metadata", approved.tool_results[0].metadata),
            ("approved execution runtime metadata", approved.metadata),
        ]
    )
    for label, surface in surfaces:
        _assert_private_promotion_surface(label, surface, bound_args)
    return approved


def test_promotions_execute_with_private_approval_bindings() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-promote-decision-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(
            runtime,
            "Decision promotion candidate",
            "The reviewed decision candidate body.",
            category="decision",
        )
        target = runtime.store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None or target.target_kind != "decision":
            raise SystemExit("decision promotion fixture did not resolve")
        raw = {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_token": target.binding,
            "title": "Approved decision promotion",
            "rationale": "The candidate was reviewed and approved explicitly.",
            "impact": "The exact memory becomes the durable decision record.",
        }
        approved = _approve_private_promotion(
            runtime,
            "promote the reviewed decision fixture",
            "promote_memory_to_decision",
            raw,
        )
        decision_id = approved.tool_results[0].metadata.get("decision_id")
        decision = runtime.store.get_decision(decision_id) if type(decision_id) is int else None
        memory = runtime.store.get_memory(memory_id)
        if (
            decision is None
            or decision["title"] != raw["title"]
            or decision["rationale"] != raw["rationale"]
            or decision["impact"] != raw["impact"]
            or memory is None
            or memory["category"] != "decisions"
            or int(memory["revision"]) != target.revision + 1
        ):
            raise SystemExit("approved decision promotion produced the wrong durable state")

    with TemporaryDirectory(prefix="jarvis-memory-binding-promote-preference-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(
            runtime,
            "Preference promotion candidate",
            "The reviewed preference candidate body.",
            category="preference",
        )
        target = runtime.store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None or target.target_kind != "preference":
            raise SystemExit("preference promotion fixture did not resolve")
        raw = {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_token": target.binding,
            "category": "communication",
            "key": "approval smoke tone",
            "value": "Warm, concise, and direct",
        }
        approved = _approve_private_promotion(
            runtime,
            "promote the reviewed preference fixture",
            "promote_memory_to_preference",
            raw,
        )
        preference_id = approved.tool_results[0].metadata.get("preference_id")
        preference = (
            runtime.store.get_preference_by_id(preference_id)
            if type(preference_id) is int
            else None
        )
        memory = runtime.store.get_memory(memory_id)
        if (
            preference is None
            or preference["category"] != raw["category"]
            or preference["key"] != raw["key"]
            or preference["value"] != raw["value"]
            or preference["status"] != "active"
            or memory is None
            or memory["category"] != "preferences"
            or int(memory["revision"]) != target.revision + 1
        ):
            raise SystemExit("approved preference promotion produced the wrong durable state")


def _promotion_structured_state(store: MemoryStore) -> tuple[Any, ...]:
    tables = (
        "decisions",
        "decision_memory_links",
        "decision_projection_jobs",
        "preferences",
        "preference_identity_owners",
        "preference_memory_links",
        "preference_projection_jobs",
        "memory_projection_jobs",
    )
    with store.connect() as conn:
        return tuple(
            (table, tuple(tuple(row) for row in conn.execute(f"SELECT * FROM {table}")))
            for table in tables
        )


def _natural_promotion_request(
    tool_name: str,
    memory_id: int,
    target,
) -> tuple[str, dict[str, Any]]:
    if tool_name == "promote_memory_to_preference":
        raw = {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_token": target.binding,
            "category": "communication",
            "key": "natural drift preference",
            "value": "Keep approval replay one-shot",
        }
        command = (
            f"promote memory {memory_id} revision {target.revision} token {target.binding} "
            f"to preference: {raw['category']} | {raw['key']} | {raw['value']}"
        )
        return command, raw
    if tool_name == "promote_memory_to_decision":
        raw = {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_token": target.binding,
            "title": "Natural drift decision",
            "rationale": "The strict natural command was reviewed explicitly",
            "impact": "A stale source must consume approval without promotion",
        }
        command = (
            f"promote memory {memory_id} revision {target.revision} token {target.binding} "
            f"to decision: {raw['title']} | {raw['rationale']} | {raw['impact']}"
        )
        return command, raw
    raise SystemExit(f"unsupported natural promotion fixture: {tool_name}")


def test_natural_preference_promotion_token_stays_private() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-natural-private-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(
            runtime,
            "Natural private preference candidate",
            "The opaque review token must remain internal to approval binding.",
            category="preference",
        )
        target = runtime.store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None or target.target_kind != "preference":
            raise SystemExit("natural private preference fixture did not resolve")
        command, raw = _natural_promotion_request(
            "promote_memory_to_preference",
            memory_id,
            target,
        )

        parsed = RuleBasedPlanner().plan(command)
        if (
            len(parsed.actions) != 1
            or parsed.actions[0].tool_name != "promote_memory_to_preference"
            or parsed.actions[0].args != raw
        ):
            raise SystemExit(f"strict natural preference command parsed incorrectly: {parsed}")
        tool = runtime.registry.get("promote_memory_to_preference")
        if tool.risk != RiskLevel.HIGH_RISK:
            raise SystemExit(
                "natural preference promotion no longer preserves its HIGH_RISK boundary"
            )

        approval_id, bound_args, held = _queue(
            runtime,
            command,
            "promote_memory_to_preference",
        )
        expected_bound_args = {
            "memory_id": memory_id,
            "reviewed_revision": target.revision,
            "review_binding": target.binding,
            "category": raw["category"],
            "key": raw["key"],
            "value": raw["value"],
            "target_revision": target.revision,
            "target_binding": target.binding,
        }
        if bound_args != expected_bound_args:
            raise SystemExit(
                "natural preference approval did not preserve its exact bound arguments"
            )

        listing = runtime.registry.get("list_pending_approvals").handler({})
        if not listing.ok:
            raise SystemExit(f"natural preference approval listing failed: {listing}")
        projection_path = listing.metadata.get("path")
        if not isinstance(projection_path, str) or not Path(projection_path).is_file():
            raise SystemExit(
                f"natural preference approval missed its pending projection: {listing.metadata}"
            )
        projection = Path(projection_path).read_text(encoding="utf-8")

        transition = approve_pending_runtime_approval(runtime, approval_id)
        approved = runtime.handle(
            command,
            approved=True,
            approved_approval_id=approval_id,
        )
        if (
            not approved.verified
            or len(approved.tool_results) != 1
            or approved.tool_results[0].tool_name != "promote_memory_to_preference"
            or approved.tool_results[0].ok is not True
            or approved.tool_results[0].metadata.get("handler_invoked") is not True
        ):
            raise SystemExit(
                f"natural private preference promotion did not execute once: {approved}"
            )

        public_surfaces: list[tuple[str, Any]] = [
            ("pending approval listing output", listing.output),
            ("pending approval listing metadata", listing.metadata),
            ("approval transition output", transition.output),
            ("approval transition metadata", transition.metadata),
            ("approval hold public plan", held.plan),
            ("approval hold runtime trace", held.metadata.get("runtime_trace")),
            ("approved execution public plan", approved.plan),
            ("approved execution runtime trace", approved.metadata.get("runtime_trace")),
            ("pending approvals projection", projection),
        ]
        leaking_surfaces = [
            label
            for label, surface in public_surfaces
            if target.binding in json.dumps(surface, sort_keys=True, default=str)
        ]
        if leaking_surfaces:
            raise SystemExit(
                "opaque preference review token reached public surfaces: "
                + ", ".join(leaking_surfaces)
            )


def _run_natural_promotion_drift_case(tool_name: str, drift_mode: str) -> None:
    target_kind = (
        "preference" if tool_name == "promote_memory_to_preference" else "decision"
    )
    with TemporaryDirectory(
        prefix=f"jarvis-memory-binding-{target_kind}-{drift_mode}-"
    ) as temp:
        runtime = make_temp_runtime(Path(temp))
        title = f"Natural {target_kind} {drift_mode} candidate"
        memory_id = _seed(
            runtime,
            title,
            "Source body before queued approval drift.",
            category=target_kind,
        )
        target = runtime.store.resolve_knowledge_promotion_approval_target(memory_id)
        if target is None or target.target_kind != target_kind:
            raise SystemExit(f"{target_kind} {drift_mode} fixture did not resolve")
        command, raw = _natural_promotion_request(tool_name, memory_id, target)
        approval_id, bound_args, held = _queue(runtime, command, tool_name)
        _assert_bound_shape(tool_name, bound_args, raw)
        if (
            len(held.plan.actions) != 1
            or held.plan.actions[0].tool_name != tool_name
            or held.plan.actions[0].args
            != {**raw, "review_token": "bound to reviewed memory version"}
        ):
            raise SystemExit(
                f"strict natural {target_kind} command did not queue its exact raw request"
            )

        if drift_mode == "same_revision_sql":
            with runtime.store.connect() as conn:
                conn.execute(
                    "UPDATE memories SET body = ? WHERE id = ?",
                    ("Direct SQL body drift without a revision bump.", memory_id),
                )
        elif drift_mode == "revision_bumped":
            if not runtime.store.update_memory(
                memory_id,
                target_kind,
                title,
                "Ordinary source drift with a revision bump.",
                0.63,
            ):
                raise SystemExit(f"could not revision-bump the {target_kind} fixture")
        else:
            raise SystemExit(f"unsupported promotion drift fixture: {drift_mode}")

        drifted = runtime.store.get_memory(memory_id)
        expected_revision = target.revision + (drift_mode == "revision_bumped")
        if drifted is None or int(drifted["revision"]) != expected_revision:
            raise SystemExit(f"{target_kind} {drift_mode} fixture drifted incorrectly")
        before_memories = _memory_rows(runtime.store)
        before_structured = _promotion_structured_state(runtime.store)

        approved = _execute(runtime, command, approval_id)
        if (
            approved.verified
            or len(approved.plan.actions) != 1
            or approved.plan.actions[0].tool_name != tool_name
            or len(approved.tool_results) != 1
        ):
            raise SystemExit(
                f"stale approved {target_kind} promotion returned the wrong runtime shape: {approved}"
            )
        result = approved.tool_results[0]
        if (
            result.ok
            or result.metadata.get("reason") != "exact_promotion_refused"
            or result.metadata.get("handler_invoked") is not True
            or result.metadata.get("state_changed") is not False
            or result.metadata.get("promotion_committed") is not False
        ):
            raise SystemExit(
                f"stale approved {target_kind} promotion did not fail closed: {approved}"
            )
        if (
            _memory_rows(runtime.store) != before_memories
            or _promotion_structured_state(runtime.store) != before_structured
        ):
            raise SystemExit(
                f"stale approved {target_kind} promotion changed source or structured state"
            )

        approval = runtime.store.get_approval(approval_id)
        claim = runtime.store.get_approval_execution_claim(approval_id)
        runs = runtime.store.approved_tool_runs_for_approvals([approval_id], limit=10)
        if (
            approval is None
            or approval["status"] != "approved"
            or claim is None
            or claim["outcome"] != "failed"
            or claim["completed_at"] is None
            or len(runs) != 1
            or runs[0]["tool_name"] != tool_name
            or int(runs[0]["ok"]) != 0
            or runtime.store.list_pending_approvals(limit=100)
        ):
            raise SystemExit(
                f"stale {target_kind} approval did not reach one terminal failed claim"
            )

        replay = runtime.handle(
            command,
            approved=True,
            approved_approval_id=approval_id,
        )
        if (
            replay.verified
            or replay.plan.actions
            or replay.tool_results
            or replay.plan.metadata.get("approval_rerun_binding_failure")
            != "approval_already_used"
            or _memory_rows(runtime.store) != before_memories
            or _promotion_structured_state(runtime.store) != before_structured
            or len(
                runtime.store.approved_tool_runs_for_approvals([approval_id], limit=10)
            )
            != 1
        ):
            raise SystemExit(f"terminal stale {target_kind} approval replayed: {replay}")

        repeated = runtime.registry.get("approve_pending_approval").handler(
            {"approval_id": approval_id}
        )
        if (
            repeated.ok
            or repeated.metadata.get("reason") != "approval_already_used"
            or _memory_rows(runtime.store) != before_memories
            or _promotion_structured_state(runtime.store) != before_structured
        ):
            raise SystemExit(
                f"terminal stale {target_kind} approval could be approved again: {repeated}"
            )


def test_natural_promotion_source_drift_is_terminal_without_replay() -> None:
    for tool_name in (
        "promote_memory_to_preference",
        "promote_memory_to_decision",
    ):
        for drift_mode in ("same_revision_sql", "revision_bumped"):
            _run_natural_promotion_drift_case(tool_name, drift_mode)


def _assert_replay_blocked(runtime, command: str, approval_id: int, before_rows) -> None:
    replay = runtime.handle(command, approved=True, approved_approval_id=approval_id)
    if replay.plan.actions or any(
        result.metadata.get("handler_invoked") is True for result in replay.tool_results
    ):
        raise SystemExit(f"completed memory approval replayed its handler: {replay}")
    if _memory_rows(runtime.store) != before_rows:
        raise SystemExit("blocked approval replay changed memory state")


def test_approved_exact_mutations_and_replay() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-delete-ok-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(runtime, "Delete once")
        command = f"delete memory {memory_id}"
        approval_id, _args, _held = _queue(runtime, command, "delete_memory")
        _assert_approved_once(runtime, command, approval_id)
        if runtime.store.get_memory(memory_id) is not None:
            raise SystemExit("approved exact delete left its target row")
        _assert_replay_blocked(runtime, command, approval_id, _memory_rows(runtime.store))

    with TemporaryDirectory(prefix="jarvis-memory-binding-edit-ok-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(runtime, "Edit once", "before")
        raw = {"memory_id": memory_id, "body": "after", "confidence": 0.91}
        command = "edit memory fixture exactly once"
        approval_id, _args, _held = _queue(
            runtime, command, "edit_memory", raw_args=raw
        )
        _assert_approved_once(runtime, command, approval_id)
        row = runtime.store.get_memory(memory_id)
        if (
            row is None
            or row["body"] != "after"
            or float(row["confidence"]) != 0.91
            or int(row["revision"]) != 2
        ):
            raise SystemExit(
                f"approved exact edit produced the wrong row: {dict(row) if row else None}"
            )
        _assert_replay_blocked(runtime, command, approval_id, _memory_rows(runtime.store))

    with TemporaryDirectory(prefix="jarvis-memory-binding-merge-ok-") as temp:
        runtime = make_temp_runtime(Path(temp))
        keep_id = _seed(runtime, "Merge once keep", "keep body", confidence=0.4)
        delete_id = _seed(runtime, "Merge once delete", "delete body", confidence=0.9)
        command = f"merge memory {delete_id} into {keep_id}"
        approval_id, _args, _held = _queue(runtime, command, "merge_memories")
        _assert_approved_once(runtime, command, approval_id)
        keep = runtime.store.get_memory(keep_id)
        if (
            keep is None
            or runtime.store.get_memory(delete_id) is not None
            or "delete body" not in str(keep["body"])
            or float(keep["confidence"]) != 0.9
            or int(keep["revision"]) != 2
        ):
            raise SystemExit("approved exact merge produced the wrong durable state")
        _assert_replay_blocked(runtime, command, approval_id, _memory_rows(runtime.store))


def _assert_stale_zero_mutation(runtime, command: str, approval_id: int, before_rows) -> None:
    result = _execute(runtime, command, approval_id)
    if len(result.tool_results) != 1:
        raise SystemExit(f"stale memory approval returned the wrong result count: {result}")
    tool_result = result.tool_results[0]
    reason = tool_result.metadata.get("reason") or tool_result.metadata.get(
        "memory_refusal_handoff", {}
    ).get("reason")
    if tool_result.ok or reason != "stale_memory_binding":
        raise SystemExit(f"changed memory did not stale its approval: {result}")
    if _memory_rows(runtime.store) != before_rows:
        raise SystemExit("stale memory approval changed memory state")


def test_same_id_mutations_are_stale() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-delete-stale-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(runtime, "Delete stale", "before")
        command = f"delete memory {memory_id}"
        approval_id, _args, _held = _queue(runtime, command, "delete_memory")
        if not runtime.store.update_memory(memory_id, "fact", "Delete stale", "changed", 0.8):
            raise SystemExit("could not mutate delete target fixture")
        _assert_stale_zero_mutation(
            runtime, command, approval_id, _memory_rows(runtime.store)
        )

    with TemporaryDirectory(prefix="jarvis-memory-binding-edit-stale-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(runtime, "Edit stale", "before")
        command = "edit a memory whose row changes before approval"
        approval_id, _args, _held = _queue(
            runtime,
            command,
            "edit_memory",
            raw_args={"memory_id": memory_id, "body": "approved body"},
        )
        if not runtime.store.update_memory(memory_id, "fact", "Edit stale", "concurrent body", 0.7):
            raise SystemExit("could not mutate edit target fixture")
        _assert_stale_zero_mutation(
            runtime, command, approval_id, _memory_rows(runtime.store)
        )

    with TemporaryDirectory(prefix="jarvis-memory-binding-merge-stale-") as temp:
        runtime = make_temp_runtime(Path(temp))
        keep_id = _seed(runtime, "Merge stale keep", "keep before")
        delete_id = _seed(runtime, "Merge stale delete", "delete before")
        command = f"merge memory {delete_id} into {keep_id}"
        approval_id, _args, _held = _queue(runtime, command, "merge_memories")
        if not runtime.store.update_memory(
            delete_id, "fact", "Merge stale delete", "delete changed", 0.8
        ):
            raise SystemExit("could not mutate merge target fixture")
        _assert_stale_zero_mutation(
            runtime, command, approval_id, _memory_rows(runtime.store)
        )


def test_binding_catches_direct_sql_content_drift() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-sql-drift-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(runtime, "SQL drift", "before")
        command = f"delete memory {memory_id}"
        approval_id, args, _held = _queue(runtime, command, "delete_memory")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE memories SET body = ? WHERE id = ?",
                ("changed without revision", memory_id),
            )
        row = runtime.store.get_memory(memory_id)
        if row is None or int(row["revision"]) != args["target_revision"]:
            raise SystemExit("direct SQL drift fixture unexpectedly advanced revision")
        _assert_stale_zero_mutation(
            runtime, command, approval_id, _memory_rows(runtime.store)
        )


def _assert_preapproval_refusal(runtime, command: str, tool_name: str, raw_args=None) -> None:
    before_rows = _memory_rows(runtime.store)
    before_pending = len(runtime.store.list_pending_approvals(limit=100))
    if raw_args is not None:
        runtime.planner = StaticPlanner(tool_name, raw_args)
    else:
        runtime.planner = RuleBasedPlanner()
    result = runtime.handle(command)
    matches = [item for item in result.tool_results if item.tool_name == tool_name]
    if (
        len(matches) != 1
        or matches[0].ok
        or matches[0].metadata.get("requires_confirmation") is not False
        or matches[0].metadata.get("handler_invoked") is not False
    ):
        raise SystemExit(f"{tool_name} did not refuse before approval: {result}")
    if len(runtime.store.list_pending_approvals(limit=100)) != before_pending:
        raise SystemExit(f"{tool_name} refusal queued an approval")
    if _memory_rows(runtime.store) != before_rows:
        raise SystemExit(f"{tool_name} pre-approval refusal changed memory state")


def test_missing_and_same_id_refuse_before_approval() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-refusals-") as temp:
        runtime = make_temp_runtime(Path(temp))
        existing_id = _seed(runtime, "Refusal survivor")
        missing_id = existing_id + 1000
        _assert_preapproval_refusal(
            runtime, f"delete memory {missing_id}", "delete_memory"
        )
        _assert_preapproval_refusal(
            runtime,
            "edit missing memory fixture",
            "edit_memory",
            {"memory_id": missing_id, "body": "must not write"},
        )
        _assert_preapproval_refusal(
            runtime,
            f"merge memory {missing_id} into {existing_id}",
            "merge_memories",
        )
        _assert_preapproval_refusal(
            runtime,
            f"merge memory {existing_id} into {existing_id}",
            "merge_memories",
        )


def test_legacy_unbound_approvals_fail_validation() -> None:
    cases = (
        ("delete_memory", lambda ids: {"memory_id": ids[0]}),
        ("edit_memory", lambda ids: {"memory_id": ids[0], "body": "legacy edit"}),
        ("merge_memories", lambda ids: {"keep_id": ids[0], "delete_id": ids[1]}),
    )
    for tool_name, args_builder in cases:
        with TemporaryDirectory(prefix=f"jarvis-memory-binding-legacy-{tool_name}-") as temp:
            runtime = make_temp_runtime(Path(temp))
            ids = (
                _seed(runtime, f"{tool_name} legacy first", "first"),
                _seed(runtime, f"{tool_name} legacy second", "second"),
            )
            command = f"legacy unbound {tool_name} approval"
            approval_id = runtime.store.add_pending_approval(
                runtime.session_id,
                command,
                tool_name,
                "legacy unbound memory approval",
                args_builder(ids),
            )
            before_rows = _memory_rows(runtime.store)
            result = _execute(runtime, command, approval_id)
            if len(result.tool_results) != 1:
                raise SystemExit(f"legacy {tool_name} approval returned the wrong result count")
            tool_result = result.tool_results[0]
            if (
                tool_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or tool_result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"legacy {tool_name} approval reached its handler: {result}")
            if _memory_rows(runtime.store) != before_rows:
                raise SystemExit(f"legacy {tool_name} approval changed memory state")
            with runtime.store.connect() as conn:
                claim = conn.execute(
                    "SELECT outcome FROM approval_execution_claims WHERE approval_id = ?",
                    (approval_id,),
                ).fetchone()
            if claim is None or claim["outcome"] != "failed":
                raise SystemExit(f"legacy {tool_name} approval claim did not fail closed")


def test_bound_edit_without_changes_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-empty-edit-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = _seed(runtime, "Empty bound edit", "must remain")
        target = runtime.store.resolve_memory_approval_target(memory_id)
        if target is None:
            raise SystemExit("empty bound edit fixture did not resolve")
        command = "malformed bound edit with no changes"
        approval_id = runtime.store.add_pending_approval(
            runtime.session_id,
            command,
            "edit_memory",
            "malformed empty edit fixture",
            {
                "memory_id": memory_id,
                "target_revision": target.revision,
                "target_binding": target.binding,
            },
        )
        before = _memory_rows(runtime.store)
        result = _execute(runtime, command, approval_id)
        if len(result.tool_results) != 1:
            raise SystemExit("empty bound edit returned the wrong result count")
        tool_result = result.tool_results[0]
        reason = tool_result.metadata.get("reason") or tool_result.metadata.get(
            "memory_refusal_handoff", {}
        ).get("reason")
        if tool_result.ok or reason != "no_changes":
            raise SystemExit(f"empty bound edit did not fail closed: {result}")
        if _memory_rows(runtime.store) != before:
            raise SystemExit("empty bound edit changed the memory row or revision")


def test_migration_and_ordinary_revision_increments() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-migration-") as temp:
        path = Path(temp) / "legacy.sqlite"
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO memories(
                    category, title, body, source, confidence, created_at, updated_at
                ) VALUES ('fact', 'Legacy memory', 'body', 'legacy', 0.8,
                          '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )
        store = MemoryStore(path)
        store.init()
        legacy = store.get_memory(1)
        if legacy is None or int(legacy["revision"]) != 1:
            raise SystemExit("legacy memory migration did not initialize revision 1")

    with TemporaryDirectory(prefix="jarvis-memory-binding-revisions-") as temp:
        store = MemoryStore(Path(temp) / "memory.sqlite")
        store.init()
        update_id = store.add_memory(MemoryRecord("fact", "Update target", "before"))
        if not store.update_memory(
            update_id, "fact", "Update target", "updated", 0.9
        ):
            raise SystemExit("ordinary memory update failed")
        updated = store.get_memory(update_id)
        if updated is None or int(updated["revision"]) != 2:
            raise SystemExit("ordinary memory update did not increment revision")

        keep_id = store.add_memory(MemoryRecord("fact", "Merge keep", "keep"))
        delete_id = store.add_memory(MemoryRecord("fact", "Merge delete", "delete"))
        if not store.merge_memories(keep_id, delete_id):
            raise SystemExit("ordinary memory merge failed")
        keep = store.get_memory(keep_id)
        if keep is None or int(keep["revision"]) != 2 or store.get_memory(delete_id) is not None:
            raise SystemExit("ordinary memory merge did not increment the kept revision")


def test_concurrent_exact_delete_has_one_winner() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-binding-concurrent-") as temp:
        store = MemoryStore(Path(temp) / "memory.sqlite")
        store.init()
        memory_id = store.add_memory(MemoryRecord("fact", "Concurrent target", "body"))
        target = store.resolve_memory_approval_target(memory_id)
        if target is None:
            raise SystemExit("concurrent exact delete fixture did not resolve")
        barrier = threading.Barrier(2)
        statuses: list[str] = []
        errors: list[BaseException] = []

        def run_delete() -> None:
            try:
                barrier.wait(timeout=5)
                statuses.append(
                    store.delete_memory_exact(
                        memory_id, target.revision, target.binding
                    ).status
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_delete) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        if errors or any(thread.is_alive() for thread in threads):
            raise SystemExit(f"concurrent exact memory delete failed: {errors}")
        if sorted(statuses) != ["deleted", "not_found"]:
            raise SystemExit(
                f"concurrent exact memory delete did not have one winner: {statuses}"
            )


def main() -> None:
    test_contracts_natural_queue_and_private_display()
    test_promotions_execute_with_private_approval_bindings()
    test_natural_preference_promotion_token_stays_private()
    test_natural_promotion_source_drift_is_terminal_without_replay()
    test_approved_exact_mutations_and_replay()
    test_same_id_mutations_are_stale()
    test_binding_catches_direct_sql_content_drift()
    test_missing_and_same_id_refuse_before_approval()
    test_legacy_unbound_approvals_fail_validation()
    test_bound_edit_without_changes_fails_closed()
    test_migration_and_ordinary_revision_increments()
    test_concurrent_exact_delete_has_one_winner()
    print("Memory approval binding smoke passed")


if __name__ == "__main__":
    main()
