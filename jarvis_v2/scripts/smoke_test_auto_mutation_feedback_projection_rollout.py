from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.feedback import (
    MAX_FEEDBACK_LIMIT,
    save_feedback_actions_auto_mutation_operation_key,
    save_feedback_report_auto_mutation_operation_key,
)
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    ToolArgumentType,
)


@dataclass(frozen=True)
class ProjectionCase:
    tool_name: str
    projection: str
    vault_method: str
    relative_path: str
    operation_key_builder: Callable[[dict[str, Any]], dict[str, Any]]


CASES = (
    ProjectionCase(
        "save_feedback_report",
        "feedback_report",
        "write_feedback_report_with_evidence",
        "Automations/Feedback Report.md",
        save_feedback_report_auto_mutation_operation_key,
    ),
    ProjectionCase(
        "save_feedback_actions",
        "feedback_actions",
        "write_feedback_actions_with_evidence",
        "Automations/Feedback Actions.md",
        save_feedback_actions_auto_mutation_operation_key,
    ),
)


class StaticPlanner:
    def __init__(self, tool_name: str, args: Any):
        self.tool_name = tool_name
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise one fixed feedback projection.",
            [PlannedAction(self.tool_name, self.args, "feedback projection receipt smoke")],
            needs_model=False,
        )


def _runtime(root: Path, case: ProjectionCase, args: Any) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(case.tool_name, args)
    return runtime


def _receipt_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [
            dict(row)
            for row in conn.execute("SELECT * FROM auto_mutation_receipts ORDER BY id")
        ]


def _tool_run_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM tool_runs ORDER BY id")]


def _install_counted_handler(runtime: JarvisRuntime, case: ProjectionCase) -> list[int]:
    tool = runtime.registry.get(case.tool_name)
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools[case.tool_name] = replace(tool, handler=counted)
    return calls


def _seed_feedback(runtime: JarvisRuntime, marker: str, count: int = 3) -> None:
    for index in range(count):
        runtime.store.add_memory(
            MemoryRecord(
                "feedback",
                f"Projection feedback {index}",
                f"{marker} item {index}",
                "feedback-projection-rollout-smoke",
            )
        )


def _assert_replay(result: Any, failure_kind: str, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} did not return one tool result: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != failure_kind:
        raise SystemExit(f"{label} returned the wrong replay state: {item}")


def _assert_completed_audit_links(runtime: JarvisRuntime, expected: int, label: str) -> None:
    receipts = _receipt_rows(runtime)
    successful_runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
    if len(receipts) != expected or len(successful_runs) != expected:
        raise SystemExit(f"{label} expected {expected} receipt/audit pairs: {receipts} / {successful_runs}")
    runs_by_id = {int(row["id"]): row for row in successful_runs}
    linked_ids: set[int] = set()
    for receipt in receipts:
        run = runs_by_id.get(int(receipt["tool_run_id"]))
        if (
            receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
            or run is None
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(f"{label} receipt lost its ordinary approved=0 audit: {receipt} / {run}")
        linked_ids.add(int(run["id"]))
    if len(linked_ids) != expected:
        raise SystemExit(f"{label} receipts reused an ordinary audit: {linked_ids}")


def _assert_receipt_privacy(
    runtime: JarvisRuntime,
    *,
    private_values: list[str],
    label: str,
) -> None:
    receipt_text = json.dumps(_receipt_rows(runtime), ensure_ascii=False, sort_keys=True, default=str)
    for private in private_values:
        if private and private in receipt_text:
            raise SystemExit(f"{label} receipt leaked private request, content, or path data: {private!r}")


def test_exact_registry_contracts_and_limit_independent_projection_keys() -> None:
    with TemporaryDirectory(prefix="jarvis-feedback-projection-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        expected_keys: list[dict[str, Any]] = []
        integer = frozenset({ToolArgumentType.INTEGER})
        for case in CASES:
            tool = runtime.registry.get(case.tool_name)
            contract = tool.auto_mutation_contract
            if (
                tool.risk is not RiskLevel.LOCAL_SAFE
                or contract is None
                or contract.version != AUTO_MUTATION_CONTRACT_VERSION
                or contract.effects != frozenset({AutoMutationEffect.OBSIDIAN_VAULT})
                or contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
                or contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
                or contract.operation_key_builder is not case.operation_key_builder
                or contract.semantic_preflight is not None
            ):
                raise SystemExit(f"{case.tool_name} auto-mutation contract drifted: {tool}")
            argument_contract = tool.argument_contract
            shape = (
                tuple(
                    (field.name, field.types, field.required)
                    for field in argument_contract.fields
                )
                if argument_contract is not None
                else ()
            )
            if argument_contract is None or argument_contract.allow_unknown or shape != (
                ("limit", integer, False),
            ):
                raise SystemExit(f"{case.tool_name} strict limit contract drifted: {argument_contract}")

            keys = [
                case.operation_key_builder({}),
                case.operation_key_builder({"limit": 1}),
                case.operation_key_builder({"limit": 12}),
                case.operation_key_builder({"limit": MAX_FEEDBACK_LIMIT}),
            ]
            expected = {"projection": case.projection}
            if any(key != expected for key in keys):
                raise SystemExit(f"{case.tool_name} limit variants changed projection identity: {keys}")
            expected_keys.append(expected)

        if expected_keys[0] == expected_keys[1]:
            raise SystemExit("feedback report and actions projections share one operation identity")
        for name in ("feedback_report", "feedback_actions"):
            if runtime.registry.get(name).auto_mutation_contract is not None:
                raise SystemExit(f"read-only feedback tool unexpectedly joined the receipt rollout: {name}")


def test_planner_empty_args_and_direct_handler_legacy_coercion() -> None:
    legacy_cases = (
        ({"limit": "3"}, 3),
        ({"limit": True}, 12),
        ({"limit": 0}, 1),
        ({"limit": MAX_FEEDBACK_LIMIT + 50}, MAX_FEEDBACK_LIMIT),
        ({"limit": "not-an-integer"}, 12),
    )
    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-empty-args-") as temp:
            runtime = _runtime(Path(temp), case, {})
            calls = _install_counted_handler(runtime, case)
            result = runtime.handle(
                "save the fixed feedback projection",
                request_token=f"{case.projection}-empty-args-private-token",
            )
            item = result.tool_results[0]
            if not item.ok or item.metadata.get("limit") != 12 or calls[0] != 1:
                raise SystemExit(f"{case.tool_name} planner {{}} compatibility drifted: {item}")

        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-legacy-limit-") as temp:
            runtime = make_temp_runtime(Path(temp))
            handler = runtime.registry.get(case.tool_name).handler
            for args, expected_limit in legacy_cases:
                result = handler(args)
                if not result.ok or result.metadata.get("limit") != expected_limit:
                    raise SystemExit(
                        f"{case.tool_name} direct-handler coercion drifted for {args}: {result}"
                    )
            if _receipt_rows(runtime):
                raise SystemExit(f"direct {case.tool_name} calls unexpectedly created runtime receipts")


def test_typed_invalid_args_stop_before_handler_and_receipt() -> None:
    invalid_args: tuple[Any, ...] = (
        {"limit": "12"},
        {"limit": True},
        {"limit": 1.5},
        {"limit": None},
        {"limit": []},
        {"limit": {}},
        {"limit": 12, "extra": "rejected"},
        None,
        [],
    )
    for case in CASES:
        for index, args in enumerate(invalid_args):
            with TemporaryDirectory(prefix=f"jarvis-{case.projection}-typed-") as temp:
                runtime = _runtime(Path(temp), case, args)
                calls = _install_counted_handler(runtime, case)
                result = runtime.handle(
                    "invalid feedback projection args",
                    request_token=f"{case.projection}-typed-private-{index}",
                )
                item = result.tool_results[0]
                if (
                    item.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or item.metadata.get("handler_invoked") is not False
                    or calls[0] != 0
                    or _receipt_rows(runtime)
                ):
                    raise SystemExit(f"{case.tool_name} typed case {index} crossed validation: {item}")
                if (runtime.vault.root_path / case.relative_path).exists():
                    raise SystemExit(f"{case.tool_name} typed case {index} published its projection")


def test_completed_replay_fresh_rewrite_audit_and_receipt_privacy() -> None:
    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-completed-") as temp:
            runtime = _runtime(Path(temp), case, {"limit": 1})
            marker = f"PRIVATE-{case.projection.upper()}-FEEDBACK-CONTENT"
            _seed_feedback(runtime, marker)
            calls = _install_counted_handler(runtime, case)
            first_token = f"{case.projection}-completed-private-token-one"
            fresh_token = f"{case.projection}-completed-private-token-two"

            first = runtime.handle("first projection write", request_token=first_token)
            same = runtime.handle("same-token projection replay", request_token=first_token)
            fresh = runtime.handle("fresh-token intentional rewrite", request_token=fresh_token)
            if not first.tool_results[0].ok or not fresh.tool_results[0].ok:
                raise SystemExit(f"{case.tool_name} valid projection write failed")
            _assert_replay(same, "auto_mutation_completed_replay", f"{case.tool_name} same-token replay")
            if calls[0] != 2:
                raise SystemExit(f"{case.tool_name} replay/rewrite invocation count drifted: {calls[0]}")
            projection_path = runtime.vault.root_path / case.relative_path
            if not projection_path.is_file():
                raise SystemExit(f"{case.tool_name} did not publish its fixed projection")
            _assert_completed_audit_links(runtime, 2, case.tool_name)
            _assert_receipt_privacy(
                runtime,
                private_values=[
                    first_token,
                    fresh_token,
                    marker,
                    str(runtime.vault.root_path),
                    str(runtime.store.db_path.parent),
                ],
                label=case.tool_name,
            )


def test_post_publication_exception_uncertain_replay_and_limit_fencing() -> None:
    for case in CASES:
        with TemporaryDirectory(prefix=f"jarvis-{case.projection}-uncertain-") as temp:
            runtime = _runtime(Path(temp), case, {"limit": 1})
            marker = f"PRIVATE-{case.projection.upper()}-UNCERTAIN-CONTENT"
            _seed_feedback(runtime, marker)
            handler_calls = _install_counted_handler(runtime, case)
            original_publish = getattr(runtime.vault, case.vault_method)
            publication_calls = [0]
            published_paths: list[Path] = []

            def publish_then_fail(*args: Any, **kwargs: Any) -> Any:
                publication_calls[0] += 1
                result = original_publish(*args, **kwargs)
                published_paths.append(Path(result[0]))
                raise OSError("representative post-publication feedback projection failure")

            setattr(runtime.vault, case.vault_method, publish_then_fail)
            owner_token = f"{case.projection}-uncertain-private-owner"
            changed_token = f"{case.projection}-uncertain-private-changed-limit"
            failed = runtime.handle("publish then fail", request_token=owner_token)
            if failed.tool_results[0].metadata.get("failure_kind") != "tool_error":
                raise SystemExit(f"{case.tool_name} post-publication exception was not a tool error")
            receipts = _receipt_rows(runtime)
            if (
                handler_calls[0] != 1
                or publication_calls[0] != 1
                or len(published_paths) != 1
                or not published_paths[0].is_file()
                or len(receipts) != 1
                or receipts[0]["state"] != "uncertain"
                or receipts[0]["result"] != "unknown"
            ):
                raise SystemExit(
                    f"{case.tool_name} did not retain one published uncertain outcome: "
                    f"{handler_calls} / {publication_calls} / {receipts}"
                )

            same = runtime.handle("same-token uncertain replay", request_token=owner_token)
            _assert_replay(
                same,
                "auto_mutation_outcome_uncertain",
                f"{case.tool_name} same-token uncertain replay",
            )
            runtime.planner = StaticPlanner(case.tool_name, {"limit": MAX_FEEDBACK_LIMIT})
            changed = runtime.handle("changed-limit fresh-token retry", request_token=changed_token)
            _assert_replay(
                changed,
                "auto_mutation_unresolved_action",
                f"{case.tool_name} changed-limit unresolved fence",
            )
            if handler_calls[0] != 1 or publication_calls[0] != 1 or len(_receipt_rows(runtime)) != 1:
                raise SystemExit(f"{case.tool_name} unresolved limit variant invoked the handler again")
            _assert_receipt_privacy(
                runtime,
                private_values=[
                    owner_token,
                    changed_token,
                    marker,
                    str(runtime.vault.root_path),
                    str(runtime.store.db_path.parent),
                ],
                label=f"{case.tool_name} uncertain",
            )


def main() -> None:
    test_exact_registry_contracts_and_limit_independent_projection_keys()
    test_planner_empty_args_and_direct_handler_legacy_coercion()
    test_typed_invalid_args_stop_before_handler_and_receipt()
    test_completed_replay_fresh_rewrite_audit_and_receipt_privacy()
    test_post_publication_exception_uncertain_replay_and_limit_fencing()
    print("Auto-mutation feedback projection rollout smoke test passed.")


if __name__ == "__main__":
    main()
