from __future__ import annotations

import json
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
from unittest.mock import patch

import jarvis_v2.agent.executor as executor_module
import jarvis_v2.agent.model_planner as model_planner_module
from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.model_planner import ModelBackedPlanner
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import (
    ApprovalArgumentResolution,
    Plan,
    PlannedAction,
    RiskLevel,
    ToolResult,
)
from jarvis_v2.agent.verifier import Verifier
from jarvis_v2.memory.store import MemoryRecord, MemoryStore
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import browser
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationContract,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    Tool,
    ToolArgumentContract,
    ToolArgumentSpec,
    ToolArgumentType,
    ToolRegistry,
    tool_argument_contract_summary,
)


INTEGER = frozenset({ToolArgumentType.INTEGER})
NUMBER = frozenset({ToolArgumentType.NUMBER})
STRING = frozenset({ToolArgumentType.STRING})
AUTO_CONTRACT = AutoMutationContract(
    AUTO_MUTATION_CONTRACT_VERSION,
    frozenset({AutoMutationEffect.LOCAL_DATABASE}),
    AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
    AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
)
STRICT_SCHEMA = ToolArgumentContract(
    TOOL_ARGUMENT_CONTRACT_VERSION,
    (
        ToolArgumentSpec("item_id", INTEGER, True),
        ToolArgumentSpec("body", STRING, True),
        ToolArgumentSpec("note", STRING, False),
    ),
    False,
)


class StaticPlanner:
    def __init__(self, actions: list[PlannedAction]):
        self.actions = actions

    def plan(self, _user_input: str) -> Plan:
        return Plan("Exercise typed tool arguments.", list(self.actions), needs_model=False)


class ModelOnlyPlanner:
    def plan(self, user_input: str) -> Plan:
        return Plan(user_input, [], needs_model=True)


def _runtime(temp: str, tools: list[Tool], actions: list[PlannedAction]) -> JarvisRuntime:
    runtime = object.__new__(JarvisRuntime)
    runtime.session_id = uuid.uuid4().hex[:8]
    runtime.storage_fallback = None
    runtime.store = MemoryStore(Path(temp) / "runtime.sqlite")
    runtime.store.init()
    runtime.registry = ToolRegistry()
    for tool in tools:
        runtime.registry.register(tool)
    runtime.planner = StaticPlanner(actions)
    runtime.executor = Executor(runtime.registry, PermissionPolicy())
    runtime.verifier = Verifier()
    runtime.chat = None
    runtime.vault = None
    return runtime


def _tool(
    name: str,
    risk: RiskLevel,
    handler: Callable[[dict[str, Any]], ToolResult],
    *,
    auto: bool = False,
    schema: ToolArgumentContract | None = STRICT_SCHEMA,
) -> Tool:
    return Tool(
        name,
        "typed fixture",
        risk,
        handler,
        "fixture",
        AUTO_CONTRACT if auto else None,
        schema,
    )


def _receipt_count(store: MemoryStore) -> int:
    with store.connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM auto_mutation_receipts").fetchone()[0])


def test_registry_rejects_malformed_contract_definitions() -> None:
    malformed = [
        "wrong-type",
        ToolArgumentContract(TOOL_ARGUMENT_CONTRACT_VERSION + 1, (), False),
        ToolArgumentContract(TOOL_ARGUMENT_CONTRACT_VERSION, [], False),  # type: ignore[arg-type]
        ToolArgumentContract(
            TOOL_ARGUMENT_CONTRACT_VERSION,
            (ToolArgumentSpec("same", STRING), ToolArgumentSpec("same", INTEGER)),
            False,
        ),
        ToolArgumentContract(
            TOOL_ARGUMENT_CONTRACT_VERSION,
            (ToolArgumentSpec("bad-name!", STRING),),
            False,
        ),
        ToolArgumentContract(
            TOOL_ARGUMENT_CONTRACT_VERSION,
            (ToolArgumentSpec("field", frozenset({"string"})),),  # type: ignore[arg-type]
            False,
        ),
        ToolArgumentContract(
            TOOL_ARGUMENT_CONTRACT_VERSION,
            (ToolArgumentSpec("field", STRING, 1),),  # type: ignore[arg-type]
            False,
        ),
        ToolArgumentContract(
            TOOL_ARGUMENT_CONTRACT_VERSION,
            (ToolArgumentSpec("field", STRING, False, 1, 2),),
            False,
        ),
        ToolArgumentContract(
            TOOL_ARGUMENT_CONTRACT_VERSION,
            (ToolArgumentSpec("field", INTEGER, False, 3, 2),),
            False,
        ),
        ToolArgumentContract(
            TOOL_ARGUMENT_CONTRACT_VERSION,
            (ToolArgumentSpec("field", INTEGER, False, True, 2),),
            False,
        ),
    ]
    for index, contract in enumerate(malformed):
        registry = ToolRegistry()
        fixture = Tool(
            f"bad_{index}",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            lambda _args: ToolResult(f"bad_{index}", True, "ran"),
            argument_contract=contract,  # type: ignore[arg-type]
        )
        try:
            registry.register(fixture)
        except ValueError:
            if registry.list():
                raise SystemExit("malformed argument contract remained registered")
        else:
            raise SystemExit("malformed argument contract registered")


def test_executor_enforces_shape_before_handler_and_permission() -> None:
    calls: list[dict[str, Any]] = []

    def handler(args: dict[str, Any]) -> ToolResult:
        calls.append(dict(args))
        return ToolResult("typed", True, "ran")

    registry = ToolRegistry()
    registry.register(_tool("typed", RiskLevel.LOCAL_SAFE, handler))
    executor = Executor(registry, PermissionPolicy())
    invalid_args: list[Any] = [
        {"body": "ok"},
        {"item_id": 1, "body": "ok", "extra": "private"},
        {"item_id": "1", "body": "ok"},
        {"item_id": True, "body": "ok"},
        {"item_id": 1, "body": 9},
        None,
        ["not", "an", "object"],
    ]
    for args in invalid_args:
        result = executor.execute(PlannedAction("typed", args, "typed smoke"))  # type: ignore[arg-type]
        if result.ok or result.metadata.get("failure_kind") != "tool_arguments_invalid":
            raise SystemExit("invalid typed arguments were accepted")
        if result.metadata.get("handler_invoked") is not False or result.metadata.get("executed_handler") is not False:
            raise SystemExit("argument rejection claimed handler execution")
        if result.metadata.get("requires_confirmation") is not False:
            raise SystemExit("argument rejection reached permission handling")
    if calls:
        raise SystemExit("invalid typed arguments invoked the handler")

    valid_args = {"item_id": 1, "body": "ok", "note": "kept"}
    valid = executor.execute(PlannedAction("typed", valid_args, "valid typed smoke"))
    if not valid.ok or calls != [valid_args]:
        raise SystemExit("valid typed arguments did not reach the handler unchanged")


def test_cancel_reminders_production_contract_rejects_before_handler() -> None:
    with TemporaryDirectory(prefix="jarvis-cancel-reminder-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        production_tool = runtime.registry.get("cancel_reminders")
        contract = production_tool.argument_contract
        shape = (
            tuple(
                (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        if (
            production_tool.risk is not RiskLevel.LOCAL_SAFE
            or contract is None
            or contract.allow_unknown is not False
            or shape != (("reminder_id", ("string",), False),)
        ):
            raise SystemExit(f"cancel_reminders contract is not exact optional-string/reject-unknown: {contract}")

        handler_calls: list[dict[str, Any]] = []

        def handler(args: dict[str, Any]) -> ToolResult:
            handler_calls.append(dict(args))
            return ToolResult("cancel_reminders", True, "ran")

        registry = ToolRegistry()
        registry.register(
            Tool(
                name=production_tool.name,
                description=production_tool.description,
                risk=production_tool.risk,
                handler=handler,
                toolset=production_tool.toolset,
                auto_mutation_contract=production_tool.auto_mutation_contract,
                argument_contract=contract,
                approval_argument_resolver=production_tool.approval_argument_resolver,
                approval_argument_contract=production_tool.approval_argument_contract,
            )
        )
        executor = Executor(registry, PermissionPolicy())
        secret = "PRIVATE-REMINDER-CONTRACT-VALUE"
        invalid_args: list[Any] = [
            {"reminder_id": 7},
            {"reminder_id": True},
            {"reminder_id": None},
            {"unknown": secret},
            ["reminder_id", "11111111"],
        ]
        for args in invalid_args:
            result = executor.execute(
                PlannedAction("cancel_reminders", args, "cancel reminder contract smoke")  # type: ignore[arg-type]
            )
            if result.ok or result.metadata.get("failure_kind") != "tool_arguments_invalid":
                raise SystemExit(f"cancel_reminders accepted malformed arguments: {args!r} -> {result}")
            if result.metadata.get("handler_invoked") is not False or result.metadata.get("executed_handler") is not False:
                raise SystemExit(f"cancel_reminders rejection claimed handler execution: {result}")
            if secret in result.output or secret in json.dumps(result.metadata, sort_keys=True, default=str):
                raise SystemExit("cancel_reminders argument rejection leaked an unknown value")
        if handler_calls:
            raise SystemExit(f"cancel_reminders malformed arguments reached the handler: {handler_calls}")


def test_pre_approval_argument_resolver_contract() -> None:
    raw_contract = ToolArgumentContract(
        TOOL_ARGUMENT_CONTRACT_VERSION,
        (ToolArgumentSpec("name", STRING, True),),
        False,
    )
    bound_contract = ToolArgumentContract(
        TOOL_ARGUMENT_CONTRACT_VERSION,
        (
            ToolArgumentSpec("name", STRING, True),
            ToolArgumentSpec("target_id", INTEGER, True),
        ),
        False,
    )
    for label, risk, resolver, approval_contract in (
        ("resolver_without_contract", RiskLevel.HIGH_RISK, lambda args: args, None),
        ("contract_without_resolver", RiskLevel.HIGH_RISK, None, bound_contract),
        ("resolver_on_safe_tool", RiskLevel.LOCAL_SAFE, lambda args: args, bound_contract),
    ):
        registry = ToolRegistry()
        try:
            registry.register(
                Tool(
                    label,
                    "invalid approval resolver fixture",
                    risk,
                    lambda _args: ToolResult(label, True, "ran"),
                    argument_contract=raw_contract,
                    approval_argument_resolver=resolver,  # type: ignore[arg-type]
                    approval_argument_contract=approval_contract,
                )
            )
        except ValueError:
            pass
        else:
            raise SystemExit(f"registry accepted malformed approval resolver contract: {label}")

    resolver_calls: list[dict[str, Any]] = []
    handler_calls: list[dict[str, Any]] = []

    def resolver(args: dict[str, Any]) -> ApprovalArgumentResolution:
        resolver_calls.append(dict(args))
        return ApprovalArgumentResolution({"name": args["name"], "target_id": 7})

    def handler(args: dict[str, Any]) -> ToolResult:
        handler_calls.append(dict(args))
        return ToolResult("bound_delete", True, "ran")

    registry = ToolRegistry()
    registry.register(
        Tool(
            "bound_delete",
            "approval resolver fixture",
            RiskLevel.HIGH_RISK,
            handler,
            argument_contract=raw_contract,
            approval_argument_resolver=resolver,
            approval_argument_contract=bound_contract,
        )
    )
    executor = Executor(registry, PermissionPolicy())
    held = executor.execute(PlannedAction("bound_delete", {"name": "target"}))
    if (
        held.metadata.get("failure_kind") != "approval_required"
        or held.metadata.get("planned_args") != {"name": "target", "target_id": 7}
        or resolver_calls != [{"name": "target"}]
        or handler_calls
    ):
        raise SystemExit(f"pre-approval resolver did not bind before the approval hold: {held}")
    legacy = executor.execute(
        PlannedAction("bound_delete", {"name": "target"}), approved=True
    )
    if (
        legacy.metadata.get("failure_kind") != "tool_arguments_invalid"
        or legacy.metadata.get("missing_arg_keys") != ["target_id"]
        or legacy.metadata.get("handler_invoked") is not False
        or resolver_calls != [{"name": "target"}]
    ):
        raise SystemExit(f"legacy approved arguments escaped the bound contract: {legacy}")
    completed = executor.execute(
        PlannedAction("bound_delete", {"name": "target", "target_id": 7}),
        approved=True,
    )
    if not completed.ok or handler_calls != [{"name": "target", "target_id": 7}]:
        raise SystemExit("bound approved arguments did not reach the handler exactly once")

    for label, bad_resolver, expected_status in (
        ("malformed", lambda _args: {"name": "target"}, "resolver_result_malformed"),
        (
            "reserved",
            lambda _args: ApprovalArgumentResolution(
                {"name": "target", "target_id": 7},
                {"planned_args": {"name": "attacker"}},
            ),
            "resolver_result_malformed",
        ),
        ("exception", lambda _args: (_ for _ in ()).throw(RuntimeError("private")), "resolver_unavailable"),
        (
            "requesting",
            lambda _args: ToolResult(
                "bad_requesting",
                False,
                "hold",
                {
                    "requires_confirmation": True,
                    "executed_handler": False,
                    "handler_invoked": False,
                    "authorizes_execution": False,
                    "approval_granted": False,
                },
            ),
            "resolver_refusal_malformed",
        ),
    ):
        bad_registry = ToolRegistry()
        bad_registry.register(
            Tool(
                f"bad_{label}",
                "bad resolver fixture",
                RiskLevel.HIGH_RISK,
                lambda _args: ToolResult(f"bad_{label}", True, "ran"),
                argument_contract=raw_contract,
                approval_argument_resolver=bad_resolver,  # type: ignore[arg-type]
                approval_argument_contract=bound_contract,
            )
        )
        refused = Executor(bad_registry, PermissionPolicy()).execute(
            PlannedAction(f"bad_{label}", {"name": "target"})
        )
        if (
            refused.metadata.get("failure_kind") != "approval_argument_resolution_failed"
            or refused.metadata.get("approval_argument_resolution_status") != expected_status
            or refused.metadata.get("requires_confirmation") is not False
            or "private" in refused.output
        ):
            raise SystemExit(f"bad approval resolver did not fail closed: {refused}")


def test_number_contract_rejects_non_finite_values() -> None:
    calls: list[dict[str, Any]] = []

    def handler(args: dict[str, Any]) -> ToolResult:
        calls.append(dict(args))
        return ToolResult("finite_number", True, "ran")

    contract = ToolArgumentContract(
        TOOL_ARGUMENT_CONTRACT_VERSION,
        (ToolArgumentSpec("value", NUMBER, True),),
        False,
    )
    registry = ToolRegistry()
    registry.register(_tool("finite_number", RiskLevel.READ_ONLY, handler, schema=contract))
    executor = Executor(registry, PermissionPolicy())
    for value in (float("nan"), float("inf"), float("-inf")):
        result = executor.execute(PlannedAction("finite_number", {"value": value}))
        if result.metadata.get("argument_validation_status") != "type_mismatch":
            raise SystemExit("non-finite value escaped the JSON-compatible number contract")
    for value in (0, 1.5, -2):
        result = executor.execute(PlannedAction("finite_number", {"value": value}))
        if not result.ok:
            raise SystemExit("finite JSON number was rejected")
    if calls != [{"value": 0}, {"value": 1.5}, {"value": -2}]:
        raise SystemExit("non-finite number reached the handler")


def test_legacy_unschematized_tools_remain_compatible() -> None:
    calls: list[dict[str, Any]] = []

    def legacy(args: dict[str, Any]) -> ToolResult:
        calls.append(dict(args))
        return ToolResult("legacy", True, "ran")

    registry = ToolRegistry()
    registry.register(_tool("legacy", RiskLevel.LOCAL_SAFE, legacy, schema=None))
    payload = {"arbitrary": [1, True, None], "nested": {"still": "accepted"}}
    result = Executor(registry, PermissionPolicy()).execute(PlannedAction("legacy", payload))
    if not result.ok or calls != [payload]:
        raise SystemExit("unschematized tool behavior changed")


def test_invalid_high_risk_arguments_never_queue_approval() -> None:
    calls = 0

    def risky(_args: dict[str, Any]) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult("typed_risky", True, "ran")

    with TemporaryDirectory(prefix="jarvis-typed-risky-") as temp:
        action = PlannedAction("typed_risky", {"item_id": True, "body": "private"})
        runtime = _runtime(temp, [_tool("typed_risky", RiskLevel.HIGH_RISK, risky)], [action])
        before = runtime.store.list_pending_approvals(limit=20)
        result = runtime.handle("typed risky malformed")
        after = runtime.store.list_pending_approvals(limit=20)
        item = result.tool_results[0]
        if calls or before or after:
            raise SystemExit("invalid high-risk arguments invoked or queued approval")
        if item.metadata.get("failure_kind") != "tool_arguments_invalid":
            raise SystemExit("invalid high-risk arguments missed the central contract")
        if item.metadata.get("requires_confirmation") is not False or item.metadata.get("approval_id") is not None:
            raise SystemExit("invalid high-risk arguments looked approval-held")


def test_invalid_auto_mutation_plan_stops_before_receipts_and_handlers() -> None:
    calls: list[str] = []

    def named(name: str) -> Callable[[dict[str, Any]], ToolResult]:
        def handler(_args: dict[str, Any]) -> ToolResult:
            calls.append(name)
            return ToolResult(name, True, "ran")

        return handler

    with TemporaryDirectory(prefix="jarvis-typed-auto-") as temp:
        tools = [
            _tool("first", RiskLevel.LOCAL_SAFE, named("first"), auto=True),
            _tool("second", RiskLevel.LOCAL_SAFE, named("second"), auto=True),
        ]
        actions = [
            PlannedAction("first", {"item_id": 1, "body": "valid"}),
            PlannedAction("second", {"item_id": False, "body": "invalid"}),
        ]
        runtime = _runtime(temp, tools, actions)
        result = runtime.handle("typed auto malformed", request_token="typed-auto-request")
        if calls or _receipt_count(runtime.store) != 0:
            raise SystemExit("invalid auto-mutation plan reached receipt preparation or a handler")
        if result.tool_results[0].metadata.get("failure_kind") != "tool_argument_plan_preflight_blocked":
            raise SystemExit("valid sibling did not receive all-or-none preflight block")
        if result.tool_results[1].metadata.get("failure_kind") != "tool_arguments_invalid":
            raise SystemExit("invalid sibling did not receive typed rejection")
        if result.verified or result.metadata["runtime_trace"].get("ran_tool_handlers") is not False:
            raise SystemExit("invalid auto-mutation plan claimed execution or verification")
        if result.metadata["runtime_trace"].get("executed_handler_count") != 0:
            raise SystemExit("invalid auto-mutation plan trace counted a handler")


def test_model_planner_advertises_and_enforces_contracts() -> None:
    calls = 0

    def handler(_args: dict[str, Any]) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult("typed_model", True, "ran")

    registry = ToolRegistry()
    registry.register(_tool("typed_model", RiskLevel.LOCAL_SAFE, handler))
    registry.register(_tool("typed_personal_model", RiskLevel.PERSONAL_DATA, handler))
    registry.register(_tool("typed_external_model", RiskLevel.EXTERNAL_SIDE_EFFECT, handler))
    planner = ModelBackedPlanner("fixture", registry, base=ModelOnlyPlanner())
    description = planner._tool_descriptions()
    if "item_id:integer" not in description or "body:string" not in description or "unknown rejected" not in description:
        raise SystemExit("model planner did not advertise the registered argument contract")
    if "typed_external_model(fixture, EXTERNAL_SIDE_EFFECT, requires explicit approval)" not in description:
        raise SystemExit("model planner described an external-side-effect tool as safe")
    if "typed_personal_model(fixture, PERSONAL_DATA, personal data; requires explicit approval)" not in description:
        raise SystemExit("model planner described a personal-data tool as optionally gated")

    original_generate = model_planner_module.generate_model_text
    try:
        for args in [None, [], "bad", {}, {"item_id": True, "body": "x"}, {"item_id": 1, "body": "x", "extra": 1}]:
            payload = {
                "mode": "tool",
                "goal": "typed model smoke",
                "actions": [{"tool_name": "typed_model", "args": args, "reason": "fixture"}],
            }
            model_planner_module.generate_model_text = lambda **_kwargs: json.dumps(payload)  # type: ignore[assignment]
            plan = planner.plan("model typed request")
            if len(plan.actions) != 1:
                raise SystemExit("model planner lost the known typed action")
            result = Executor(registry, PermissionPolicy()).execute(plan.actions[0])
            if result.metadata.get("failure_kind") != "tool_arguments_invalid":
                raise SystemExit("malformed model arguments escaped the executor boundary")
        if calls:
            raise SystemExit("malformed model arguments invoked the handler")
    finally:
        model_planner_module.generate_model_text = original_generate  # type: ignore[assignment]


def test_model_planner_preserves_malformed_optional_only_argument_roots() -> None:
    calls: list[dict[str, Any]] = []

    def handler(args: dict[str, Any]) -> ToolResult:
        calls.append(dict(args))
        return ToolResult("optional_model", True, "ran")

    contract = ToolArgumentContract(
        TOOL_ARGUMENT_CONTRACT_VERSION,
        (ToolArgumentSpec("note", STRING, False),),
        False,
    )
    registry = ToolRegistry()
    registry.register(_tool("optional_model", RiskLevel.READ_ONLY, handler, schema=contract))
    planner = ModelBackedPlanner("fixture", registry, base=ModelOnlyPlanner())
    original_generate = model_planner_module.generate_model_text
    try:
        for malformed in (None, [], "private malformed root"):
            payload = {
                "mode": "tool",
                "goal": "optional typed model smoke",
                "actions": [{"tool_name": "optional_model", "args": malformed}],
            }
            model_planner_module.generate_model_text = lambda **_kwargs: json.dumps(payload)  # type: ignore[assignment]
            plan = planner.plan("optional typed request")
            if (
                len(plan.actions) != 1
                or type(plan.actions[0].args) is not type(malformed)
                or plan.actions[0].args != malformed
            ):
                raise SystemExit("model planner normalized a malformed optional-only argument root")
            result = Executor(registry, PermissionPolicy()).execute(plan.actions[0])
            if result.metadata.get("argument_validation_status") != "arguments_not_object":
                raise SystemExit("malformed optional-only model arguments escaped rejection")

        payload = {
            "mode": "tool",
            "goal": "missing args means empty object",
            "actions": [{"tool_name": "optional_model"}],
        }
        model_planner_module.generate_model_text = lambda **_kwargs: json.dumps(payload)  # type: ignore[assignment]
        plan = planner.plan("optional typed request without args")
        result = Executor(registry, PermissionPolicy()).execute(plan.actions[0])
        if not result.ok or calls != [{}]:
            raise SystemExit("missing model args did not preserve the empty-object compatibility path")
    finally:
        model_planner_module.generate_model_text = original_generate  # type: ignore[assignment]


def test_model_fallback_metadata_does_not_store_response_content() -> None:
    marker = "PRIVATE_MODEL_RESPONSE_MARKER"
    registry = ToolRegistry()
    registry.register(_tool("optional_model", RiskLevel.READ_ONLY, lambda _args: ToolResult("optional_model", True, "ran")))
    planner = ModelBackedPlanner("fixture", registry, base=ModelOnlyPlanner())
    original_generate = model_planner_module.generate_model_text
    try:
        model_planner_module.generate_model_text = lambda **_kwargs: marker  # type: ignore[assignment]
        plan = planner.plan("private request")
    finally:
        model_planner_module.generate_model_text = original_generate  # type: ignore[assignment]
    encoded = json.dumps(plan.metadata, ensure_ascii=False, default=str)
    if marker in encoded or plan.metadata.get("model_planner_fallback_detail") != "response_not_recorded":
        raise SystemExit("model fallback metadata retained response content")
    if plan.metadata.get("model_planner_response_content_in_metadata") is not False:
        raise SystemExit("model fallback privacy receipt drifted")


def test_runtime_rejects_malformed_reconciliation_roots_before_route_guard() -> None:
    calls = 0

    def handler(_args: dict[str, Any]) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult("resolve_auto_mutation_receipt", True, "ran")

    contract = ToolArgumentContract(
        TOOL_ARGUMENT_CONTRACT_VERSION,
        (
            ToolArgumentSpec("receipt_id", INTEGER, True),
            ToolArgumentSpec("disposition", STRING, True),
        ),
        False,
    )
    for malformed in (None, [], "private malformed root"):
        with TemporaryDirectory(prefix="jarvis-typed-reconciliation-root-") as temp:
            action = PlannedAction("resolve_auto_mutation_receipt", malformed)  # type: ignore[arg-type]
            runtime = _runtime(
                temp,
                [_tool("resolve_auto_mutation_receipt", RiskLevel.LOCAL_SAFE, handler, schema=contract)],
                [action],
            )
            result = runtime.handle("malformed reconciliation root")
            if len(result.tool_results) != 1:
                raise SystemExit("malformed reconciliation root lost its typed failure")
            item = result.tool_results[0]
            if item.metadata.get("argument_validation_status") != "arguments_not_object":
                raise SystemExit("malformed reconciliation root reached the explicit-route guard")
    if calls:
        raise SystemExit("malformed reconciliation root invoked the resolver")


def test_invalid_arguments_stop_before_contact_policy_and_handler() -> None:
    class ExplodingPolicy:
        def check(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("permission policy was reached")

    def handler(_args: dict[str, Any]) -> ToolResult:
        raise AssertionError("handler was reached")

    registry = ToolRegistry()
    registry.register(_tool("send_email", RiskLevel.HIGH_RISK, handler))
    original_resolver = executor_module._resolve_contact_action_before_approval
    executor_module._resolve_contact_action_before_approval = lambda _action: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("contact resolution was reached")
    )
    try:
        result = Executor(registry, ExplodingPolicy()).execute(  # type: ignore[arg-type]
            PlannedAction("send_email", {"item_id": True, "body": "private"})
        )
    finally:
        executor_module._resolve_contact_action_before_approval = original_resolver  # type: ignore[assignment]
    if result.metadata.get("failure_kind") != "tool_arguments_invalid":
        raise SystemExit("invalid arguments did not stop at the typed boundary")


def test_contract_failures_are_content_private_and_auditable() -> None:
    marker = "PRIVATE_ARGUMENT_VALUE_SHOULD_NOT_APPEAR"
    path_marker = "/\x55sers/example/private/typed-secret.txt"
    calls = 0

    def handler(_args: dict[str, Any]) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult("private_typed", True, "ran")

    with TemporaryDirectory(prefix="jarvis-typed-private-") as temp:
        action = PlannedAction(
            "private_typed",
            {"item_id": 1, "body": marker, path_marker: marker},
            "typed privacy smoke",
        )
        runtime = _runtime(
            temp,
            [_tool("private_typed", RiskLevel.LOCAL_SAFE, handler, auto=True)],
            [action],
        )
        result = runtime.handle("typed privacy request", request_token="typed-private-request")
        if calls or _receipt_count(runtime.store):
            raise SystemExit("private malformed arguments reached mutation execution")
        item = result.tool_results[0]
        if item.metadata.get("unknown_arg_keys") != ["<unknown>"]:
            raise SystemExit("unknown argument names were not redacted")
        with runtime.store.connect() as conn:
            run = conn.execute("SELECT output, metadata FROM tool_runs ORDER BY id DESC LIMIT 1").fetchone()
        exposed = json.dumps(
            {
                "output": item.output,
                "metadata": item.metadata,
                "response": result.response,
                "runtime_trace": result.metadata.get("runtime_trace"),
                "audit_output": run["output"] if run else "",
                "audit_metadata": run["metadata"] if run else "",
            },
            ensure_ascii=False,
            default=str,
        )
        if marker in exposed or path_marker in exposed:
            raise SystemExit("argument contract rejection leaked private argument content")
        if not run:
            raise SystemExit("argument contract rejection was not recorded in the ordinary audit")


def test_read_only_batch_real_command_paths() -> None:
    cases = {
        "what time is it": ("current_time", {}),
        "time difference between seoul and tokyo": (
            "time_difference",
            {"source": "seoul", "target": "tokyo"},
        ),
        "what date is tomorrow": ("relative_date", {"target": "tomorrow"}),
        "calculate 2 + 2": ("calculate", {"expression": "2 + 2"}),
        "secure password 16 without symbols": (
            "generate_password",
            {"length": 16, "include_symbols": False},
        ),
        "generate uuid": ("generate_uuid", {}),
        "spell hello": ("spell_word", {"word": "hello"}),
        "count words hello world": ("count_text", {"text": "hello world"}),
        "uppercase hello world": (
            "transform_text",
            {"text": "hello world", "mode": "uppercase"},
        ),
        "bmi 70 kg 180 cm": (
            "calculate_bmi",
            {"weight": 70.0, "weight_unit": "kg", "height": 180.0, "height_unit": "cm"},
        ),
        "convert 5 miles to km": (
            "convert_units",
            {"value": 5.0, "from_unit": "miles", "to_unit": "km"},
        ),
    }
    with TemporaryDirectory(prefix="jarvis-typed-read-only-") as temp:
        runtime = make_temp_runtime(Path(temp))
        before_approvals = runtime.store.list_pending_approvals(limit=100)
        for command, (tool_name, expected_args) in cases.items():
            plan = runtime.planner.plan(command)
            if len(plan.actions) != 1 or plan.actions[0].tool_name != tool_name or plan.actions[0].args != expected_args:
                raise SystemExit(f"typed read-only command route drifted: {command!r} -> {plan.actions}")
            result = runtime.handle(command)
            if len(result.tool_results) != 1 or result.tool_results[0].tool_name != tool_name:
                raise SystemExit(f"typed read-only runtime binding drifted: {command!r}")
            item = result.tool_results[0]
            if item.ok is not True or item.metadata.get("handler_invoked") is not True:
                raise SystemExit(f"typed read-only command did not execute successfully: {command!r}")
            if item.metadata.get("requires_confirmation") is True:
                raise SystemExit(f"typed read-only command reached approval handling: {command!r}")
        after_approvals = runtime.store.list_pending_approvals(limit=100)
        with runtime.store.connect() as conn:
            run_count = int(conn.execute("SELECT COUNT(*) FROM tool_runs").fetchone()[0])
            receipt_count = int(conn.execute("SELECT COUNT(*) FROM auto_mutation_receipts").fetchone()[0])
        if before_approvals != after_approvals or run_count != len(cases) or receipt_count != 0:
            raise SystemExit("typed read-only command paths changed approval, audit, or mutation-receipt boundaries")


def test_tool_search_contract_stops_malformed_queries_before_handler() -> None:
    with TemporaryDirectory(prefix="jarvis-typed-tool-search-") as temp:
        runtime = make_temp_runtime(Path(temp))
        production_tool = runtime.registry.get("tool_search")
        contract = production_tool.argument_contract
        shape = (
            tuple(
                (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        if (
            production_tool.risk is not RiskLevel.READ_ONLY
            or contract is None
            or contract.allow_unknown
            or shape != (("query", ("string",), True), ("limit", ("integer",), False))
        ):
            raise SystemExit(f"tool_search contract drifted: {contract}")

        calls: list[dict[str, Any]] = []

        def counted_handler(args: dict[str, Any]) -> ToolResult:
            calls.append(dict(args))
            return production_tool.handler(args)

        runtime.registry._tools["tool_search"] = Tool(
            production_tool.name,
            production_tool.description,
            production_tool.risk,
            counted_handler,
            production_tool.toolset,
            production_tool.auto_mutation_contract,
            contract,
        )
        malformed: list[Any] = [
            {},
            {"query": True},
            {"query": ["approval"]},
            {"query": "approval", "limit": "12"},
            {"query": "approval", "limit": False},
            {"query": "approval", "extra": "private"},
            None,
        ]
        for index, args in enumerate(malformed):
            executor_result = runtime.executor.execute(
                PlannedAction("tool_search", args, "tool-search executor contract smoke")  # type: ignore[arg-type]
            )
            if (
                executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or executor_result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"tool_search executor accepted malformed arguments #{index}")
            runtime.planner = StaticPlanner(
                [PlannedAction("tool_search", args, "tool-search runtime contract smoke")]  # type: ignore[arg-type]
            )
            runtime_result = runtime.handle(f"exercise tool_search argument contract {index}")
            item = runtime_result.tool_results[0] if len(runtime_result.tool_results) == 1 else None
            if (
                item is None
                or item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
            ):
                raise SystemExit(f"tool_search runtime accepted malformed arguments #{index}")
        if calls:
            raise SystemExit("malformed tool_search arguments reached the handler")

        planner_plan = RuleBasedPlanner().plan("tool search: approval")
        if (
            len(planner_plan.actions) != 1
            or planner_plan.actions[0].tool_name != "tool_search"
            or planner_plan.actions[0].args != {"query": "approval"}
        ):
            raise SystemExit("tool_search rule-based planner route drifted")
        runtime.planner = StaticPlanner(
            [PlannedAction("tool_search", {"query": "approval", "limit": 2}, "valid tool-search contract smoke")]
        )
        valid = runtime.handle("exercise valid tool_search argument contract")
        item = valid.tool_results[0] if len(valid.tool_results) == 1 else None
        if (
            item is None
            or not item.ok
            or item.metadata.get("handler_invoked") is not True
            or calls != [{"query": "approval", "limit": 2}]
        ):
            raise SystemExit("valid tool_search arguments did not reach the handler unchanged")


def test_capability_cockpit_contract_stops_arguments_before_audit_reads() -> None:
    with TemporaryDirectory(prefix="jarvis-typed-capability-cockpit-") as temp:
        runtime = make_temp_runtime(Path(temp))
        production_tool = runtime.registry.get("capability_cockpit")
        contract = production_tool.argument_contract
        if (
            production_tool.risk is not RiskLevel.READ_ONLY
            or contract is None
            or contract.allow_unknown
            or contract.fields
        ):
            raise SystemExit(f"capability_cockpit contract drifted: {contract}")

        calls: list[dict[str, Any]] = []

        def counted_handler(args: dict[str, Any]) -> ToolResult:
            calls.append(dict(args))
            return production_tool.handler(args)

        runtime.registry._tools["capability_cockpit"] = Tool(
            production_tool.name,
            production_tool.description,
            production_tool.risk,
            counted_handler,
            production_tool.toolset,
            production_tool.auto_mutation_contract,
            contract,
        )
        malformed: list[Any] = [
            {"limit": 1},
            {"include_history": False},
            [],
            None,
        ]
        for index, args in enumerate(malformed):
            executor_result = runtime.executor.execute(
                PlannedAction("capability_cockpit", args, "capability-cockpit executor contract smoke")  # type: ignore[arg-type]
            )
            if (
                executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or executor_result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"capability_cockpit executor accepted malformed arguments #{index}")
            runtime.planner = StaticPlanner(
                [PlannedAction("capability_cockpit", args, "capability-cockpit runtime contract smoke")]  # type: ignore[arg-type]
            )
            runtime_result = runtime.handle(f"exercise capability_cockpit argument contract {index}")
            item = runtime_result.tool_results[0] if len(runtime_result.tool_results) == 1 else None
            if (
                item is None
                or item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
            ):
                raise SystemExit(f"capability_cockpit runtime accepted malformed arguments #{index}")
        if calls:
            raise SystemExit("malformed capability_cockpit arguments reached the handler")

        runtime.planner = RuleBasedPlanner()
        valid = runtime.handle("cockpit summary")
        item = valid.tool_results[0] if len(valid.tool_results) == 1 else None
        if (
            item is None
            or not item.ok
            or item.metadata.get("handler_invoked") is not True
            or calls != [{}]
        ):
            raise SystemExit("capability_cockpit exact route or valid empty arguments drifted")


def test_model_routing_status_contract_stops_arguments_before_probe() -> None:
    with TemporaryDirectory(prefix="jarvis-typed-model-routing-status-") as temp:
        runtime = make_temp_runtime(Path(temp))
        production_tool = runtime.registry.get("model_routing_status")
        contract = production_tool.argument_contract
        if (
            production_tool.risk is not RiskLevel.READ_ONLY
            or contract is None
            or contract.allow_unknown
            or contract.fields
        ):
            raise SystemExit(f"model_routing_status contract drifted: {contract}")

        calls: list[dict[str, Any]] = []

        def counted_handler(args: dict[str, Any]) -> ToolResult:
            calls.append(dict(args))
            return ToolResult("model_routing_status", True, "Model readiness fixture.", {})

        runtime.registry._tools["model_routing_status"] = Tool(
            production_tool.name,
            production_tool.description,
            production_tool.risk,
            counted_handler,
            production_tool.toolset,
            production_tool.auto_mutation_contract,
            contract,
        )
        malformed: list[Any] = [
            {"refresh": True},
            {"provider": "ollama"},
            [],
            None,
        ]
        for index, args in enumerate(malformed):
            executor_result = runtime.executor.execute(
                PlannedAction("model_routing_status", args, "model-status executor contract smoke")  # type: ignore[arg-type]
            )
            if (
                executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or executor_result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"model_routing_status executor accepted malformed arguments #{index}")
            runtime.planner = StaticPlanner(
                [PlannedAction("model_routing_status", args, "model-status runtime contract smoke")]  # type: ignore[arg-type]
            )
            runtime_result = runtime.handle(f"exercise model_routing_status argument contract {index}")
            item = runtime_result.tool_results[0] if len(runtime_result.tool_results) == 1 else None
            if (
                item is None
                or item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
            ):
                raise SystemExit(f"model_routing_status runtime accepted malformed arguments #{index}")
        if calls:
            raise SystemExit("malformed model_routing_status arguments reached the readiness handler")

        runtime.planner = RuleBasedPlanner()
        valid = runtime.handle("model status please")
        item = valid.tool_results[0] if len(valid.tool_results) == 1 else None
        if (
            item is None
            or not item.ok
            or item.metadata.get("handler_invoked") is not True
            or calls != [{}]
        ):
            raise SystemExit("model_routing_status exact route or valid empty arguments drifted")


def test_voice_setup_check_contract_stops_arguments_before_asr_inspection() -> None:
    with TemporaryDirectory(prefix="jarvis-typed-voice-setup-check-") as temp:
        runtime = make_temp_runtime(Path(temp))
        production_tool = runtime.registry.get("voice_setup_check")
        contract = production_tool.argument_contract
        if (
            production_tool.risk is not RiskLevel.READ_ONLY
            or contract is None
            or contract.allow_unknown
            or contract.fields
        ):
            raise SystemExit(f"voice_setup_check contract drifted: {contract}")

        calls: list[dict[str, Any]] = []

        def counted_handler(args: dict[str, Any]) -> ToolResult:
            calls.append(dict(args))
            return ToolResult("voice_setup_check", True, "Voice readiness fixture.", {})

        runtime.registry._tools["voice_setup_check"] = Tool(
            production_tool.name,
            production_tool.description,
            production_tool.risk,
            counted_handler,
            production_tool.toolset,
            production_tool.auto_mutation_contract,
            contract,
        )
        malformed: list[Any] = [
            {"refresh": True},
            {"whisper_model": "private"},
            [],
            None,
        ]
        for index, args in enumerate(malformed):
            executor_result = runtime.executor.execute(
                PlannedAction("voice_setup_check", args, "voice-setup executor contract smoke")  # type: ignore[arg-type]
            )
            if (
                executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or executor_result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"voice_setup_check executor accepted malformed arguments #{index}")
            runtime.planner = StaticPlanner(
                [PlannedAction("voice_setup_check", args, "voice-setup runtime contract smoke")]  # type: ignore[arg-type]
            )
            runtime_result = runtime.handle(f"exercise voice_setup_check argument contract {index}")
            item = runtime_result.tool_results[0] if len(runtime_result.tool_results) == 1 else None
            if (
                item is None
                or item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
            ):
                raise SystemExit(f"voice_setup_check runtime accepted malformed arguments #{index}")
        if calls:
            raise SystemExit("malformed voice_setup_check arguments reached the ASR readiness handler")

        runtime.planner = RuleBasedPlanner()
        valid = runtime.handle("voice setup check please")
        item = valid.tool_results[0] if len(valid.tool_results) == 1 else None
        if (
            item is None
            or not item.ok
            or item.metadata.get("handler_invoked") is not True
            or calls != [{}]
        ):
            raise SystemExit("voice_setup_check exact route or valid empty arguments drifted")


def test_search_memory_production_contract_stops_malformed_queries() -> None:
    with TemporaryDirectory(prefix="jarvis-typed-search-memory-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.add_memory(
            MemoryRecord(
                category="contract-smoke",
                title="Copper lighthouse",
                body="The copper lighthouse marks the typed memory contract seed.",
                source="contract-smoke",
            )
        )
        production_tool = runtime.registry.get("search_memory")
        calls: list[dict[str, Any]] = []

        def counted_handler(args: dict[str, Any]) -> ToolResult:
            calls.append(dict(args))
            return production_tool.handler(args)

        runtime.registry._tools["search_memory"] = Tool(
            production_tool.name,
            production_tool.description,
            production_tool.risk,
            counted_handler,
            production_tool.toolset,
            production_tool.auto_mutation_contract,
            production_tool.argument_contract,
        )
        malformed = [
            {"limit": 1},
            {"query": ["copper", "lighthouse"]},
            {"query": {"term": "copper lighthouse"}},
            {"query": True},
            {"query": "copper lighthouse", "unexpected": "private"},
        ]
        for index, args in enumerate(malformed):
            executor_result = runtime.executor.execute(
                PlannedAction("search_memory", args, "search memory executor contract smoke")
            )
            if (
                executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or executor_result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"search_memory executor accepted malformed arguments #{index}")

            runtime.planner = StaticPlanner(
                [PlannedAction("search_memory", args, "search memory runtime contract smoke")]
            )
            runtime_result = runtime.handle(f"exercise search memory argument contract {index}")
            if len(runtime_result.tool_results) != 1:
                raise SystemExit("search_memory runtime lost its central argument rejection")
            item = runtime_result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
            ):
                raise SystemExit(f"search_memory runtime accepted malformed arguments #{index}")
        if calls:
            raise SystemExit("malformed search_memory arguments reached the production handler")

        plan = RuleBasedPlanner().plan("search memory for copper lighthouse")
        if (
            len(plan.actions) != 1
            or plan.actions[0].tool_name != "search_memory"
            or plan.actions[0].args != {"query": "copper lighthouse"}
        ):
            raise SystemExit("search_memory rule-based planner route drifted")

        runtime.planner = StaticPlanner(
            [
                PlannedAction(
                    "search_memory",
                    {"query": "copper lighthouse", "limit": 1},
                    "valid search memory runtime contract smoke",
                )
            ]
        )
        valid = runtime.handle("exercise valid search memory argument contract")
        if len(valid.tool_results) != 1:
            raise SystemExit("valid search_memory runtime result was lost")
        valid_item = valid.tool_results[0]
        if (
            not valid_item.ok
            or valid_item.metadata.get("handler_invoked") is not True
            or valid_item.metadata.get("count") != 1
            or valid_item.metadata.get("limit") != 1
            or calls != [{"query": "copper lighthouse", "limit": 1}]
            or "Copper lighthouse" not in valid_item.output
        ):
            raise SystemExit("valid typed search_memory arguments did not find the seeded memory")

        legacy = production_tool.handler({"query": True, "limit": "not-an-integer"})
        if not legacy.ok or legacy.metadata.get("query") != "True" or legacy.metadata.get("limit") != 5:
            raise SystemExit("search_memory direct-handler legacy coercion changed")


def test_recent_memories_production_contract_stops_malformed_limits() -> None:
    private_content = "PRIVATE_RECENT_MEMORY_ARGUMENT_MUST_NOT_LEAK"
    with TemporaryDirectory(prefix="jarvis-typed-recent-memories-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.add_memory(
            MemoryRecord(
                category="contract-smoke",
                title="Silver compass",
                body="The silver compass marks the recent-memory contract seed.",
                source="contract-smoke",
            )
        )
        production_tool = runtime.registry.get("recent_memories")
        calls: list[dict[str, Any]] = []

        def counted_handler(args: dict[str, Any]) -> ToolResult:
            calls.append(dict(args))
            return production_tool.handler(args)

        runtime.registry._tools["recent_memories"] = Tool(
            production_tool.name,
            production_tool.description,
            production_tool.risk,
            counted_handler,
            production_tool.toolset,
            production_tool.auto_mutation_contract,
            production_tool.argument_contract,
        )
        malformed = [
            {"limit": "1"},
            {"limit": True},
            {"limit": 1.0},
            {"unexpected": private_content},
            {"limit": 1, "unexpected": private_content},
        ]
        rejected_surfaces: list[dict[str, Any]] = []
        for index, args in enumerate(malformed):
            executor_result = runtime.executor.execute(
                PlannedAction(
                    "recent_memories",
                    args,
                    "recent memories executor contract smoke",
                )
            )
            if (
                executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or executor_result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(
                    f"recent_memories executor accepted malformed arguments #{index}"
                )
            rejected_surfaces.append(
                {"output": executor_result.output, "metadata": executor_result.metadata}
            )

            runtime.planner = StaticPlanner(
                [
                    PlannedAction(
                        "recent_memories",
                        args,
                        "recent memories runtime contract smoke",
                    )
                ]
            )
            runtime_result = runtime.handle(
                f"exercise recent memories argument contract {index}"
            )
            if len(runtime_result.tool_results) != 1:
                raise SystemExit(
                    "recent_memories runtime lost its central argument rejection"
                )
            item = runtime_result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or runtime_result.metadata["runtime_trace"].get(
                    "executed_handler_count"
                )
                != 0
            ):
                raise SystemExit(
                    f"recent_memories runtime accepted malformed arguments #{index}"
                )
            rejected_surfaces.append(
                {
                    "output": item.output,
                    "metadata": item.metadata,
                    "response": runtime_result.response,
                    "runtime_trace": runtime_result.metadata.get("runtime_trace"),
                }
            )
        if calls:
            raise SystemExit(
                "malformed recent_memories arguments reached the production handler"
            )

        with runtime.store.connect() as conn:
            audit_rows = conn.execute(
                "SELECT output, metadata FROM tool_runs ORDER BY id"
            ).fetchall()
        rejected_text = json.dumps(
            {
                "rejections": rejected_surfaces,
                "audit": [dict(row) for row in audit_rows],
            },
            ensure_ascii=False,
            default=str,
        )
        if private_content in rejected_text:
            raise SystemExit(
                "recent_memories argument rejection leaked an unknown argument value"
            )

        plan = RuleBasedPlanner().plan("show latest memories")
        if (
            len(plan.actions) != 1
            or plan.actions[0].tool_name != "recent_memories"
            or plan.actions[0].args != {"limit": 10}
        ):
            raise SystemExit("recent_memories rule-based planner route drifted")

        runtime.planner = StaticPlanner(
            [
                PlannedAction(
                    "recent_memories",
                    {"limit": 1},
                    "valid recent memories runtime contract smoke",
                )
            ]
        )
        valid = runtime.handle("exercise valid recent memories argument contract")
        if len(valid.tool_results) != 1:
            raise SystemExit("valid recent_memories runtime result was lost")
        valid_item = valid.tool_results[0]
        if (
            not valid_item.ok
            or valid_item.metadata.get("handler_invoked") is not True
            or valid_item.metadata.get("count") != 1
            or valid_item.metadata.get("limit") != 1
            or calls != [{"limit": 1}]
            or "Silver compass" not in valid_item.output
        ):
            raise SystemExit(
                "valid typed recent_memories arguments did not return the seeded memory"
            )

        legacy = production_tool.handler({"limit": False})
        if not legacy.ok or legacy.metadata.get("limit") != 10:
            raise SystemExit("recent_memories direct-handler legacy coercion changed")


def test_memory_curator_review_contracts_stop_malformed_limits_before_private_reads() -> None:
    expected_shapes = {
        "list_weak_memories": (("limit", ("integer",), False, 1, 200),),
        "list_duplicate_memories": (("limit", ("integer",), False, 1, 200),),
    }
    malformed = {
        "list_weak_memories": [
            {"limit": "25"},
            {"limit": True},
            {"limit": 0},
            {"limit": 201},
            {"extra": "private"},
        ],
        "list_duplicate_memories": [
            {"limit": "200"},
            {"limit": False},
            {"limit": -1},
            {"limit": 201},
            {"extra": "private"},
        ],
    }
    with TemporaryDirectory(prefix="jarvis-typed-memory-curator-review-") as temp:
        runtime = make_temp_runtime(Path(temp))
        handler_calls: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_shapes}
        for name, expected_shape in expected_shapes.items():
            production_tool = runtime.registry.get(name)
            contract = production_tool.argument_contract
            shape = (
                tuple(
                    (
                        field.name,
                        tuple(sorted(kind.value for kind in field.types)),
                        field.required,
                        field.minimum,
                        field.maximum,
                    )
                    for field in contract.fields
                )
                if contract is not None
                else ()
            )
            if (
                production_tool.risk is not RiskLevel.READ_ONLY
                or contract is None
                or contract.allow_unknown
                or shape != expected_shape
            ):
                raise SystemExit(f"{name} memory curator review contract drifted: {contract}")

            def counted_handler(args: dict[str, Any], *, _name: str = name, _handler=production_tool.handler) -> ToolResult:
                handler_calls[_name].append(dict(args))
                return _handler(args)

            runtime.registry._tools[name] = Tool(
                production_tool.name,
                production_tool.description,
                production_tool.risk,
                counted_handler,
                production_tool.toolset,
                production_tool.auto_mutation_contract,
                contract,
                production_tool.approval_argument_resolver,
                production_tool.approval_argument_contract,
            )

        for name, cases in malformed.items():
            for index, args in enumerate(cases):
                executor_result = runtime.executor.execute(
                    PlannedAction(name, args, f"{name} executor argument contract smoke")
                )
                if (
                    executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or executor_result.metadata.get("handler_invoked") is not False
                ):
                    raise SystemExit(f"{name} executor accepted malformed arguments #{index}")
                runtime.planner = StaticPlanner(
                    [PlannedAction(name, args, f"{name} runtime argument contract smoke")]
                )
                runtime_result = runtime.handle(f"exercise {name} argument contract {index}")
                if len(runtime_result.tool_results) != 1:
                    raise SystemExit(f"{name} runtime lost its central argument rejection")
                item = runtime_result.tool_results[0]
                if (
                    item.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or item.metadata.get("handler_invoked") is not False
                    or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
                ):
                    raise SystemExit(f"{name} runtime accepted malformed arguments #{index}")

        if any(handler_calls.values()):
            raise SystemExit("malformed memory curator review arguments reached a private-memory handler")

        for name in expected_shapes:
            args = {"limit": 1}
            runtime.planner = StaticPlanner(
                [PlannedAction(name, args, f"valid {name} runtime argument contract smoke")]
            )
            result = runtime.handle(f"exercise valid {name} argument contract")
            if (
                len(result.tool_results) != 1
                or not result.tool_results[0].ok
                or result.tool_results[0].metadata.get("handler_invoked") is not True
            ):
                raise SystemExit(f"valid {name} arguments did not reach the handler")
        if any(handler_calls[name] != [{"limit": 1}] for name in expected_shapes):
            raise SystemExit("memory curator review contracts did not preserve valid handler behavior")


def test_feedback_review_contracts_stop_malformed_limits_before_private_reads() -> None:
    expected_shapes = {
        "feedback_report": (("limit", ("integer",), False, 1, 200),),
        "feedback_actions": (("limit", ("integer",), False, 1, 200),),
    }
    malformed = [
        {"limit": "12"},
        {"limit": True},
        {"limit": 0},
        {"limit": 201},
        {"extra": "private"},
    ]
    with TemporaryDirectory(prefix="jarvis-typed-feedback-review-") as temp:
        runtime = make_temp_runtime(Path(temp))
        handler_calls: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_shapes}
        for name, expected_shape in expected_shapes.items():
            production_tool = runtime.registry.get(name)
            contract = production_tool.argument_contract
            shape = (
                tuple(
                    (
                        field.name,
                        tuple(sorted(kind.value for kind in field.types)),
                        field.required,
                        field.minimum,
                        field.maximum,
                    )
                    for field in contract.fields
                )
                if contract is not None
                else ()
            )
            if (
                production_tool.risk is not RiskLevel.READ_ONLY
                or contract is None
                or contract.allow_unknown
                or shape != expected_shape
            ):
                raise SystemExit(f"{name} feedback review contract drifted: {contract}")

            def counted_handler(args: dict[str, Any], *, _name: str = name, _handler=production_tool.handler) -> ToolResult:
                handler_calls[_name].append(dict(args))
                return _handler(args)

            runtime.registry._tools[name] = Tool(
                production_tool.name,
                production_tool.description,
                production_tool.risk,
                counted_handler,
                production_tool.toolset,
                production_tool.auto_mutation_contract,
                contract,
                production_tool.approval_argument_resolver,
                production_tool.approval_argument_contract,
            )

        for name in expected_shapes:
            for index, args in enumerate(malformed):
                executor_result = runtime.executor.execute(
                    PlannedAction(name, args, f"{name} executor argument contract smoke")
                )
                if (
                    executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or executor_result.metadata.get("handler_invoked") is not False
                ):
                    raise SystemExit(f"{name} executor accepted malformed arguments #{index}")
                runtime.planner = StaticPlanner(
                    [PlannedAction(name, args, f"{name} runtime argument contract smoke")]
                )
                runtime_result = runtime.handle(f"exercise {name} argument contract {index}")
                if len(runtime_result.tool_results) != 1:
                    raise SystemExit(f"{name} runtime lost its central argument rejection")
                item = runtime_result.tool_results[0]
                if (
                    item.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or item.metadata.get("handler_invoked") is not False
                    or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
                ):
                    raise SystemExit(f"{name} runtime accepted malformed arguments #{index}")

        if any(handler_calls.values()):
            raise SystemExit("malformed feedback-review arguments reached a private-feedback handler")

        for name in expected_shapes:
            args = {"limit": 1}
            runtime.planner = StaticPlanner(
                [PlannedAction(name, args, f"valid {name} runtime argument contract smoke")]
            )
            result = runtime.handle(f"exercise valid {name} argument contract")
            if (
                len(result.tool_results) != 1
                or not result.tool_results[0].ok
                or result.tool_results[0].metadata.get("handler_invoked") is not True
            ):
                raise SystemExit(f"valid {name} arguments did not reach the handler")
        if any(handler_calls[name] != [{"limit": 1}] for name in expected_shapes):
            raise SystemExit("feedback review contracts did not preserve valid handler behavior")


def test_audit_overview_contracts_stop_malformed_limits_before_history_reads() -> None:
    expected_shapes = {
        "recent_tool_runs": (("limit", ("integer",), False, 1, 200),),
        "execution_health_report": (("limit", ("integer",), False, 1, 200),),
    }
    malformed = [
        {"limit": "20"},
        {"limit": True},
        {"limit": 0},
        {"limit": 201},
        {"extra": "private"},
    ]
    with TemporaryDirectory(prefix="jarvis-typed-audit-overview-") as temp:
        runtime = make_temp_runtime(Path(temp))
        handler_calls: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_shapes}
        for name, expected_shape in expected_shapes.items():
            production_tool = runtime.registry.get(name)
            contract = production_tool.argument_contract
            shape = (
                tuple(
                    (
                        field.name,
                        tuple(sorted(kind.value for kind in field.types)),
                        field.required,
                        field.minimum,
                        field.maximum,
                    )
                    for field in contract.fields
                )
                if contract is not None
                else ()
            )
            if (
                production_tool.risk is not RiskLevel.READ_ONLY
                or contract is None
                or contract.allow_unknown
                or shape != expected_shape
            ):
                raise SystemExit(f"{name} audit overview contract drifted: {contract}")

            def counted_handler(args: dict[str, Any], *, _name: str = name, _handler=production_tool.handler) -> ToolResult:
                handler_calls[_name].append(dict(args))
                return _handler(args)

            runtime.registry._tools[name] = Tool(
                production_tool.name,
                production_tool.description,
                production_tool.risk,
                counted_handler,
                production_tool.toolset,
                production_tool.auto_mutation_contract,
                contract,
                production_tool.approval_argument_resolver,
                production_tool.approval_argument_contract,
            )

        for name in expected_shapes:
            for index, args in enumerate(malformed):
                executor_result = runtime.executor.execute(
                    PlannedAction(name, args, f"{name} executor argument contract smoke")
                )
                if (
                    executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or executor_result.metadata.get("handler_invoked") is not False
                ):
                    raise SystemExit(f"{name} executor accepted malformed arguments #{index}")
                runtime.planner = StaticPlanner(
                    [PlannedAction(name, args, f"{name} runtime argument contract smoke")]
                )
                runtime_result = runtime.handle(f"exercise {name} argument contract {index}")
                if len(runtime_result.tool_results) != 1:
                    raise SystemExit(f"{name} runtime lost its central argument rejection")
                item = runtime_result.tool_results[0]
                if (
                    item.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or item.metadata.get("handler_invoked") is not False
                    or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
                ):
                    raise SystemExit(f"{name} runtime accepted malformed arguments #{index}")

        if any(handler_calls.values()):
            raise SystemExit("malformed audit-overview arguments reached an audit-history handler")

        for name in expected_shapes:
            args = {"limit": 1}
            runtime.planner = StaticPlanner(
                [PlannedAction(name, args, f"valid {name} runtime argument contract smoke")]
            )
            result = runtime.handle(f"exercise valid {name} argument contract")
            if (
                len(result.tool_results) != 1
                or not result.tool_results[0].ok
                or result.tool_results[0].metadata.get("handler_invoked") is not True
            ):
                raise SystemExit(f"valid {name} arguments did not reach the handler")
        if any(handler_calls[name] != [{"limit": 1}] for name in expected_shapes):
            raise SystemExit("audit overview contracts did not preserve valid handler behavior")


def test_audit_receipt_contracts_stop_malformed_selectors_before_handlers() -> None:
    malformed = {
        "verification_receipt": [
            {"run_id": True},
            {"run_id": ["latest"]},
            {"expectation": 7},
            {"unexpected": "private"},
        ],
        "runtime_trace_receipt": [
            {"message_id": False},
            {"trace_id": [1]},
            {"session_id": 7},
            {"limit": "40"},
            {"unexpected": "private"},
        ],
    }
    valid = {
        "verification_receipt": {"run_id": "latest", "expectation": ""},
        "runtime_trace_receipt": {},
    }
    with TemporaryDirectory(prefix="jarvis-typed-audit-receipts-") as temp:
        runtime = make_temp_runtime(Path(temp))
        handler_calls: dict[str, list[dict[str, Any]]] = {name: [] for name in malformed}
        for name in malformed:
            production_tool = runtime.registry.get(name)
            contract = production_tool.argument_contract
            if production_tool.risk is not RiskLevel.READ_ONLY or contract is None or contract.allow_unknown:
                raise SystemExit(f"{name} audit receipt contract drifted: {contract}")

            def counted_handler(args: dict[str, Any], *, _name: str = name, _handler=production_tool.handler) -> ToolResult:
                handler_calls[_name].append(dict(args))
                return _handler(args)

            runtime.registry._tools[name] = Tool(
                production_tool.name,
                production_tool.description,
                production_tool.risk,
                counted_handler,
                production_tool.toolset,
                production_tool.auto_mutation_contract,
                contract,
                production_tool.approval_argument_resolver,
                production_tool.approval_argument_contract,
            )

        for name, cases in malformed.items():
            for index, args in enumerate(cases):
                result = runtime.executor.execute(
                    PlannedAction(name, args, f"{name} selector contract smoke")
                )
                if (
                    result.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or result.metadata.get("handler_invoked") is not False
                ):
                    raise SystemExit(f"{name} accepted malformed selector #{index}")
        if any(handler_calls.values()):
            raise SystemExit("malformed audit receipt selectors reached a handler")

        for name, args in valid.items():
            result = runtime.executor.execute(
                PlannedAction(name, args, f"valid {name} selector contract smoke")
            )
            if result.metadata.get("handler_invoked") is not True:
                raise SystemExit(f"valid {name} selector did not reach the handler")


def test_execution_audit_packet_contracts_stop_malformed_history_reads() -> None:
    selector_shape = (
        ("run_id", ("integer", "string"), False, None, None),
        ("id", ("integer", "string"), False, None, None),
        ("tool_run_id", ("integer", "string"), False, None, None),
        ("limit", ("integer",), False, 1, 200),
    )
    limit_shape = (("limit", ("integer",), False, 1, 200),)
    expected_shapes = {
        "execution_audit_gate": limit_shape,
        "execution_recovery_packet": selector_shape,
        "after_action_learning_packet": selector_shape,
        "recovery_closure_checklist": limit_shape,
        "execution_learning_closure_packet": selector_shape,
    }
    malformed = {
        "execution_audit_gate": [
            {"limit": "30"},
            {"limit": True},
            {"limit": 0},
            {"limit": 201},
            {"unexpected": "private"},
        ],
        "recovery_closure_checklist": [
            {"limit": "80"},
            {"limit": False},
            {"limit": 0},
            {"limit": 201},
            {"unexpected": "private"},
        ],
        "execution_recovery_packet": [
            {"run_id": True},
            {"id": [1]},
            {"tool_run_id": {"id": 1}},
            {"limit": "30"},
            {"unexpected": "private"},
        ],
        "after_action_learning_packet": [
            {"run_id": False},
            {"id": [1]},
            {"tool_run_id": {"id": 1}},
            {"limit": 0},
            {"unexpected": "private"},
        ],
        "execution_learning_closure_packet": [
            {"run_id": True},
            {"id": [1]},
            {"tool_run_id": {"id": 1}},
            {"limit": 201},
            {"unexpected": "private"},
        ],
    }
    valid = {
        "execution_audit_gate": {"limit": 1},
        "execution_recovery_packet": {"run_id": "1", "limit": 1},
        "after_action_learning_packet": {"run_id": "1", "limit": 1},
        "recovery_closure_checklist": {"limit": 1},
        "execution_learning_closure_packet": {"run_id": "1", "limit": 1},
    }
    with TemporaryDirectory(prefix="jarvis-typed-execution-audit-") as temp:
        runtime = make_temp_runtime(Path(temp))
        handler_calls: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_shapes}
        for name, expected_shape in expected_shapes.items():
            production_tool = runtime.registry.get(name)
            contract = production_tool.argument_contract
            shape = (
                tuple(
                    (
                        field.name,
                        tuple(sorted(kind.value for kind in field.types)),
                        field.required,
                        field.minimum,
                        field.maximum,
                    )
                    for field in contract.fields
                )
                if contract is not None
                else ()
            )
            if (
                production_tool.risk is not RiskLevel.READ_ONLY
                or contract is None
                or contract.allow_unknown
                or shape != expected_shape
            ):
                raise SystemExit(f"{name} execution-audit contract drifted: {contract}")

            def counted_handler(args: dict[str, Any], *, _name: str = name, _handler=production_tool.handler) -> ToolResult:
                handler_calls[_name].append(dict(args))
                return _handler(args)

            runtime.registry._tools[name] = Tool(
                production_tool.name,
                production_tool.description,
                production_tool.risk,
                counted_handler,
                production_tool.toolset,
                production_tool.auto_mutation_contract,
                contract,
                production_tool.approval_argument_resolver,
                production_tool.approval_argument_contract,
            )

        for name, cases in malformed.items():
            for index, args in enumerate(cases):
                result = runtime.executor.execute(
                    PlannedAction(name, args, f"{name} execution-audit contract smoke")
                )
                if (
                    result.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or result.metadata.get("handler_invoked") is not False
                ):
                    raise SystemExit(f"{name} accepted malformed execution-audit input #{index}")
        if any(handler_calls.values()):
            raise SystemExit("malformed execution-audit inputs reached a history handler")

        for name, args in valid.items():
            result = runtime.executor.execute(
                PlannedAction(name, args, f"valid {name} execution-audit contract smoke")
            )
            if result.metadata.get("handler_invoked") is not True:
                raise SystemExit(f"valid {name} arguments did not reach the handler")


def test_approval_review_contracts_stop_malformed_queue_reads() -> None:
    selector_shape = (
        ("approval_id", ("integer", "string"), False, None, None),
        ("id", ("integer", "string"), False, None, None),
        ("target", ("integer", "string"), False, None, None),
    )
    status_limit_100_shape = (
        ("status", ("string",), False, None, None),
        ("limit", ("integer",), False, 1, 100),
    )
    expected_shapes = {
        "list_pending_approvals": status_limit_100_shape,
        "inspect_pending_approval": (
            ("status", ("string",), False, None, None),
            *selector_shape,
        ),
        "approval_execution_packet": (
            ("status", ("string",), False, None, None),
            *selector_shape,
        ),
        "approval_resume_packet": selector_shape,
        "approval_history": (("limit", ("integer",), False, 1, 50),),
        "approval_chain_proof": selector_shape,
        "approval_queue_summary": (
            ("status", ("string",), False, None, None),
            ("limit", ("integer",), False, 1, 20),
        ),
        "approval_readiness_packet": selector_shape,
        "review_pending_approvals": status_limit_100_shape,
    }
    malformed = {
        "list_pending_approvals": [
            {"status": 7},
            {"limit": "20"},
            {"limit": True},
            {"limit": 0},
            {"limit": 101},
            {"unexpected": "private"},
        ],
        "inspect_pending_approval": [
            {"approval_id": True},
            {"id": [1]},
            {"target": {"id": 1}},
            {"status": 7},
            {"unexpected": "private"},
        ],
        "approval_execution_packet": [
            {"approval_id": False},
            {"id": [1]},
            {"target": {"id": 1}},
            {"status": 7},
            {"unexpected": "private"},
        ],
        "approval_resume_packet": [
            {"approval_id": True},
            {"id": [1]},
            {"target": {"id": 1}},
            {"unexpected": "private"},
        ],
        "approval_history": [
            {"limit": "15"},
            {"limit": False},
            {"limit": 0},
            {"limit": 51},
            {"unexpected": "private"},
        ],
        "approval_chain_proof": [
            {"approval_id": True},
            {"id": [1]},
            {"target": {"id": 1}},
            {"unexpected": "private"},
        ],
        "approval_queue_summary": [
            {"status": 7},
            {"limit": "5"},
            {"limit": False},
            {"limit": 0},
            {"limit": 21},
            {"unexpected": "private"},
        ],
        "approval_readiness_packet": [
            {"approval_id": True},
            {"id": [1]},
            {"target": {"id": 1}},
            {"unexpected": "private"},
        ],
        "review_pending_approvals": [
            {"status": 7},
            {"limit": "10"},
            {"limit": False},
            {"limit": 0},
            {"limit": 101},
            {"unexpected": "private"},
        ],
    }
    valid = {
        "list_pending_approvals": {"status": "pending", "limit": 1},
        "inspect_pending_approval": {"approval_id": "latest", "status": "pending"},
        "approval_execution_packet": {"approval_id": "latest", "status": "pending"},
        "approval_resume_packet": {"approval_id": "latest"},
        "approval_history": {"limit": 1},
        "approval_chain_proof": {"approval_id": "latest"},
        "approval_queue_summary": {"status": "pending", "limit": 1},
        "approval_readiness_packet": {"approval_id": "latest"},
        "review_pending_approvals": {"status": "pending", "limit": 1},
    }
    with TemporaryDirectory(prefix="jarvis-typed-approval-review-") as temp:
        runtime = make_temp_runtime(Path(temp))
        handler_calls: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_shapes}
        for name, expected_shape in expected_shapes.items():
            production_tool = runtime.registry.get(name)
            contract = production_tool.argument_contract
            shape = (
                tuple(
                    (
                        field.name,
                        tuple(sorted(kind.value for kind in field.types)),
                        field.required,
                        field.minimum,
                        field.maximum,
                    )
                    for field in contract.fields
                )
                if contract is not None
                else ()
            )
            if (
                production_tool.risk is not RiskLevel.READ_ONLY
                or contract is None
                or contract.allow_unknown
                or shape != expected_shape
            ):
                raise SystemExit(f"{name} approval-review contract drifted: {contract}")

            def counted_handler(args: dict[str, Any], *, _name: str = name, _handler=production_tool.handler) -> ToolResult:
                handler_calls[_name].append(dict(args))
                return _handler(args)

            runtime.registry._tools[name] = Tool(
                production_tool.name,
                production_tool.description,
                production_tool.risk,
                counted_handler,
                production_tool.toolset,
                production_tool.auto_mutation_contract,
                contract,
                production_tool.approval_argument_resolver,
                production_tool.approval_argument_contract,
            )

        for name, cases in malformed.items():
            for index, args in enumerate(cases):
                result = runtime.executor.execute(
                    PlannedAction(name, args, f"{name} approval-review contract smoke")
                )
                if (
                    result.metadata.get("failure_kind") != "tool_arguments_invalid"
                    or result.metadata.get("handler_invoked") is not False
                ):
                    raise SystemExit(f"{name} accepted malformed approval-review input #{index}")
        if any(handler_calls.values()):
            raise SystemExit("malformed approval-review inputs reached a review handler")

        for name, args in valid.items():
            result = runtime.executor.execute(
                PlannedAction(name, args, f"valid {name} approval-review contract smoke")
            )
            if result.metadata.get("handler_invoked") is not True:
                raise SystemExit(f"valid {name} arguments did not reach the handler")
        if any(handler_calls[name] != [valid[name]] for name in valid):
            raise SystemExit("audit receipt contracts did not preserve valid handler behavior")


def test_browser_egress_contracts_stop_malformed_arguments_before_fetch() -> None:
    html = "<html><head><title>Contract Browser</title></head><body><a href='/next'>Next</a></body></html>"
    expected_shapes = {
        "fetch_page": (("url", ("string",), True), ("max_chars", ("integer",), False)),
        "extract_links": (("url", ("string",), True),),
        "web_search": (
            ("query", ("string",), True),
            ("engine", ("string",), False),
            ("max_chars", ("integer",), False),
        ),
    }
    malformed = {
        "fetch_page": [{}, {"url": True}, {"url": ["https://example.test"]}, {"url": "https://example.test", "extra": "private"}],
        "extract_links": [{}, {"url": 7}, {"url": {"value": "https://example.test"}}, {"url": "https://example.test", "extra": "private"}],
        "web_search": [{}, {"query": False}, {"query": ["jarvis"]}, {"query": "jarvis", "engine": 7}, {"query": "jarvis", "extra": "private"}],
    }
    valid = {
        "fetch_page": {"url": "https://example.test/page", "max_chars": 200},
        "extract_links": {"url": "https://example.test/page"},
        "web_search": {"query": "jarvis typed browser contract", "engine": "google", "max_chars": 200},
    }
    with TemporaryDirectory(prefix="jarvis-typed-browser-egress-") as temp:
        runtime = make_temp_runtime(Path(temp))
        handler_calls: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_shapes}
        for name in expected_shapes:
            production_tool = runtime.registry.get(name)
            contract = production_tool.argument_contract
            shape = (
                tuple(
                    (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                    for field in contract.fields
                )
                if contract is not None
                else ()
            )
            if production_tool.risk is not RiskLevel.LOCAL_SAFE or shape != expected_shapes[name] or contract.allow_unknown:
                raise SystemExit(f"{name} browser egress contract drifted: {contract}")

            def counted_handler(args: dict[str, Any], *, _name: str = name, _handler=production_tool.handler) -> ToolResult:
                handler_calls[_name].append(dict(args))
                return _handler(args)

            runtime.registry._tools[name] = Tool(
                production_tool.name,
                production_tool.description,
                production_tool.risk,
                counted_handler,
                production_tool.toolset,
                production_tool.auto_mutation_contract,
                contract,
                production_tool.approval_argument_resolver,
                production_tool.approval_argument_contract,
            )

        fetch_calls: list[str] = []

        def fake_fetch(url: str, timeout: int = 12) -> tuple[str, str]:
            fetch_calls.append(url)
            return html, url

        with patch.object(browser, "_fetch", side_effect=fake_fetch):
            for name, cases in malformed.items():
                for index, args in enumerate(cases):
                    executor_result = runtime.executor.execute(
                        PlannedAction(name, args, f"{name} executor argument contract smoke")
                    )
                    if (
                        executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                        or executor_result.metadata.get("handler_invoked") is not False
                    ):
                        raise SystemExit(f"{name} executor accepted malformed arguments #{index}")

                    runtime.planner = StaticPlanner(
                        [PlannedAction(name, args, f"{name} runtime argument contract smoke")]
                    )
                    runtime_result = runtime.handle(f"exercise {name} argument contract {index}")
                    if len(runtime_result.tool_results) != 1:
                        raise SystemExit(f"{name} runtime lost its central argument rejection")
                    item = runtime_result.tool_results[0]
                    if (
                        item.metadata.get("failure_kind") != "tool_arguments_invalid"
                        or item.metadata.get("handler_invoked") is not False
                        or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
                    ):
                        raise SystemExit(f"{name} runtime accepted malformed arguments #{index}")

            if fetch_calls or any(handler_calls.values()):
                raise SystemExit("malformed browser egress arguments reached a handler or fetch path")

            for name, args in valid.items():
                runtime.planner = StaticPlanner(
                    [PlannedAction(name, args, f"valid {name} runtime argument contract smoke")]
                )
                result = runtime.handle(f"exercise valid {name} argument contract")
                if (
                    len(result.tool_results) != 1
                    or not result.tool_results[0].ok
                    or result.tool_results[0].metadata.get("handler_invoked") is not True
                ):
                    raise SystemExit(f"valid {name} arguments did not reach the handler")

        if len(fetch_calls) != len(valid) or any(handler_calls[name] != [valid[name]] for name in valid):
            raise SystemExit("browser egress contract did not preserve valid handler/fetch behavior")
        history_rows = runtime.store.recent_browser_pages(limit=10)
        if len(history_rows) != 2:
            raise SystemExit("browser egress contract valid paths did not retain expected distinct history")


def test_browser_history_contracts_stop_malformed_arguments_before_handler_or_write() -> None:
    expected_shapes = {
        "recent_browser_pages": (("limit", ("integer",), False),),
        "summarize_page": (("url", ("string",), False), ("page_id", ("integer",), False)),
        "save_page_note": (("page_id", ("integer",), False),),
    }
    malformed = {
        "recent_browser_pages": [{"limit": "10"}, {"limit": True}, {"extra": "private"}],
        "summarize_page": [
            {"page_id": "latest"},
            {"page_id": True},
            {"url": 7},
            {"page_id": 1, "extra": "private"},
        ],
        "save_page_note": [{"page_id": "latest"}, {"page_id": False}, {"extra": "private"}],
    }
    valid = {
        "recent_browser_pages": {"limit": 1},
        "summarize_page": {"page_id": 1},
        "save_page_note": {"page_id": 1},
    }
    with TemporaryDirectory(prefix="jarvis-typed-browser-history-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.save_browser_page("https://example.test/page", "Typed browser page", "A saved browser page.")
        handler_calls: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_shapes}
        for name, expected_shape in expected_shapes.items():
            production_tool = runtime.registry.get(name)
            contract = production_tool.argument_contract
            shape = (
                tuple(
                    (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                    for field in contract.fields
                )
                if contract is not None
                else ()
            )
            if (
                contract is None
                or contract.allow_unknown
                or shape != expected_shape
                or production_tool.risk
                is not (RiskLevel.LOCAL_SAFE if name == "save_page_note" else RiskLevel.READ_ONLY)
            ):
                raise SystemExit(f"{name} browser history contract drifted: {contract}")

            def counted_handler(args: dict[str, Any], *, _name: str = name, _handler=production_tool.handler) -> ToolResult:
                handler_calls[_name].append(dict(args))
                return _handler(args)

            runtime.registry._tools[name] = Tool(
                production_tool.name,
                production_tool.description,
                production_tool.risk,
                counted_handler,
                production_tool.toolset,
                production_tool.auto_mutation_contract,
                contract,
                production_tool.approval_argument_resolver,
                production_tool.approval_argument_contract,
            )

        with patch.object(runtime.vault, "write_source", wraps=runtime.vault.write_source) as write_note:
            for name, cases in malformed.items():
                for index, args in enumerate(cases):
                    executor_result = runtime.executor.execute(
                        PlannedAction(name, args, f"{name} executor argument contract smoke")
                    )
                    if (
                        executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                        or executor_result.metadata.get("handler_invoked") is not False
                    ):
                        raise SystemExit(f"{name} executor accepted malformed arguments #{index}")
                    runtime.planner = StaticPlanner(
                        [PlannedAction(name, args, f"{name} runtime argument contract smoke")]
                    )
                    runtime_result = runtime.handle(f"exercise {name} argument contract {index}")
                    if len(runtime_result.tool_results) != 1:
                        raise SystemExit(f"{name} runtime lost its central argument rejection")
                    item = runtime_result.tool_results[0]
                    if (
                        item.metadata.get("failure_kind") != "tool_arguments_invalid"
                        or item.metadata.get("handler_invoked") is not False
                        or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
                    ):
                        raise SystemExit(f"{name} runtime accepted malformed arguments #{index}")

            if any(handler_calls.values()) or write_note.call_count:
                raise SystemExit("malformed browser history arguments reached a handler or note write")

            for name, args in valid.items():
                runtime.planner = StaticPlanner(
                    [PlannedAction(name, args, f"valid {name} runtime argument contract smoke")]
                )
                result = runtime.handle(f"exercise valid {name} argument contract")
                if (
                    len(result.tool_results) != 1
                    or not result.tool_results[0].ok
                    or result.tool_results[0].metadata.get("handler_invoked") is not True
                ):
                    raise SystemExit(f"valid {name} arguments did not reach the handler")

        if any(handler_calls[name] != [valid[name]] for name in valid) or write_note.call_count != 1:
            raise SystemExit("browser history contracts did not preserve valid handler or note-write behavior")


def test_brain_search_production_contract_stops_malformed_queries() -> None:
    private_content = "PRIVATE_BRAIN_CONTENT_MUST_NOT_LEAK"
    private_citation = "PRIVATE_BRAIN_CITATION_MUST_NOT_LEAK"
    with TemporaryDirectory(prefix="jarvis-typed-brain-search-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.add_memory(
            MemoryRecord(
                category="contract-smoke",
                title=private_citation,
                body=f"The quartz compass marks the brain contract seed. {private_content}",
                source="contract-smoke",
            )
        )
        production_tool = runtime.registry.get("brain_search")
        calls: list[dict[str, Any]] = []

        def counted_handler(args: dict[str, Any]) -> ToolResult:
            calls.append(dict(args))
            return production_tool.handler(args)

        runtime.registry._tools["brain_search"] = Tool(
            production_tool.name,
            production_tool.description,
            production_tool.risk,
            counted_handler,
            production_tool.toolset,
            production_tool.auto_mutation_contract,
            production_tool.argument_contract,
        )
        malformed = [
            {"limit": 1},
            {"query": ["quartz", private_content]},
            {"query": {"term": private_citation}},
            {"query": True},
            {"query": "quartz compass", "limit": "1"},
            {"query": "quartz compass", "limit": False},
            {"query": "quartz compass", "limit": 1.0},
            {"query": "quartz compass", "unexpected": private_content},
        ]
        rejected_surfaces: list[dict[str, Any]] = []
        for index, args in enumerate(malformed):
            executor_result = runtime.executor.execute(
                PlannedAction("brain_search", args, "brain search executor contract smoke")
            )
            if (
                executor_result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or executor_result.metadata.get("handler_invoked") is not False
            ):
                raise SystemExit(f"brain_search executor accepted malformed arguments #{index}")
            rejected_surfaces.append(
                {"output": executor_result.output, "metadata": executor_result.metadata}
            )

            runtime.planner = StaticPlanner(
                [PlannedAction("brain_search", args, "brain search runtime contract smoke")]
            )
            runtime_result = runtime.handle(f"exercise brain search argument contract {index}")
            if len(runtime_result.tool_results) != 1:
                raise SystemExit("brain_search runtime lost its central argument rejection")
            item = runtime_result.tool_results[0]
            if (
                item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or runtime_result.metadata["runtime_trace"].get("executed_handler_count") != 0
            ):
                raise SystemExit(f"brain_search runtime accepted malformed arguments #{index}")
            rejected_surfaces.append(
                {
                    "output": item.output,
                    "metadata": item.metadata,
                    "response": runtime_result.response,
                    "runtime_trace": runtime_result.metadata.get("runtime_trace"),
                }
            )
        if calls:
            raise SystemExit("malformed brain_search arguments reached the production handler")

        with runtime.store.connect() as conn:
            audit_rows = conn.execute("SELECT output, metadata FROM tool_runs ORDER BY id").fetchall()
        rejected_text = json.dumps(
            {
                "rejections": rejected_surfaces,
                "audit": [dict(row) for row in audit_rows],
            },
            ensure_ascii=False,
            default=str,
        )
        if private_content in rejected_text or private_citation in rejected_text:
            raise SystemExit("brain_search argument rejection leaked seeded private content or citation")

        plan = RuleBasedPlanner().plan("brain search quartz compass")
        if (
            len(plan.actions) != 1
            or plan.actions[0].tool_name != "brain_search"
            or plan.actions[0].args != {"query": "quartz compass", "limit": 8}
        ):
            raise SystemExit("brain_search rule-based planner route drifted")

        runtime.planner = StaticPlanner(
            [
                PlannedAction(
                    "brain_search",
                    {"query": "quartz compass", "limit": 1},
                    "valid brain search runtime contract smoke",
                )
            ]
        )
        valid = runtime.handle("exercise valid brain search argument contract")
        if len(valid.tool_results) != 1:
            raise SystemExit("valid brain_search runtime result was lost")
        valid_item = valid.tool_results[0]
        if (
            not valid_item.ok
            or valid_item.metadata.get("handler_invoked") is not True
            or valid_item.metadata.get("count") != 1
            or calls != [{"query": "quartz compass", "limit": 1}]
            or private_content not in valid_item.output
            or private_citation not in valid_item.output
        ):
            raise SystemExit("valid typed brain_search arguments did not find the seeded memory")

        legacy = production_tool.handler({"query": "quartz compass", "limit": False})
        if (
            not legacy.ok
            or legacy.metadata.get("query") != "quartz compass"
            or legacy.metadata.get("count") != 1
            or private_citation not in legacy.output
        ):
            raise SystemExit("brain_search direct-handler legacy limit coercion changed")


def test_production_schema_rollout_is_exact() -> None:
    original_batch = {
        "remember",
        "repair_decision_projections",
        "repair_memory_projections",
        "repair_person_projections",
        "repair_preference_projections",
        "write_daily_note",
        "record_feedback",
        "save_feedback_report",
        "save_feedback_actions",
        "record_decision",
        "record_decision_outcome",
        "set_preference_status",
        "set_preference",
        "add_person",
        "log_interaction",
        "set_decision_status",
        "save_skill",
        "memory_tree_summary",
        "queue_learning_tasks",
        "set_reminder",
        "cancel_reminders",
        "add_task",
        "complete_task",
        "complete_task_with_evidence",
        "update_task_status",
        "update_task_details",
        "import_tasks_from_note",
        "create_goal",
        "add_goal_step",
        "complete_goal_step",
        "set_goal_status",
        "export_goal",
        "export_tasks",
        "export_state_snapshot",
        "add_profile_note",
        "organize_note",
        "write_jarvis_note",
        "auto_mutation_reconciliation_queue",
        "auto_mutation_reconciliation_status",
        "resolve_auto_mutation_receipt",
        "delete_skill",
        "delete_memory",
        "edit_memory",
        "merge_memories",
        "promote_memory_to_decision",
        "promote_memory_to_preference",
        "promote_memory_to_profile",
        "clear_obsidian_inbox",
        "ingest_obsidian_inbox",
        "recent_file_digest",
        "schedule_assistant_basics",
        "schedule_morning_brief",
        "run_due_jobs",
        "resume_job",
        "run_job_now",
        "open_application",
        "open_jarvis_vault",
        "daily_brief",
        "daily_plan",
        "goal_nudge",
        "read_emails",
        "search_emails",
        "read_email_body",
        "create_event",
        "update_event",
        "delete_event",
        "research",
        "define",
        "wiki_summary",
        "get_weather",
        "get_news",
        "convert_currency",
        "translate",
        "get_crypto_price",
        "get_stock_price",
        "get_markets_overview",
        "get_air_quality",
        "next_holidays",
        "get_sun_times",
        "on_this_day",
        "send_kakao",
        "send_instagram_dm",
        "apple_reminders",
    }
    read_only_batch = {
        "current_time",
        "time_difference",
        "relative_date",
        "calculate",
        "generate_password",
        "generate_uuid",
        "spell_word",
        "count_text",
        "transform_text",
        "calculate_bmi",
        "convert_units",
        "search_memory",
        "recent_memories",
        "brain_search",
        "startup_recovery_report",
        "learning_review",
        "knowledge_promotion_packet",
        "list_tasks",
        "inspect_task",
        "search_tasks",
        "overdue_tasks",
        "task_overview",
        "next_task",
        "task_board",
        "read_profile",
        "list_preferences",
        "get_preference",
        "list_people",
        "get_person",
        "list_decisions",
        "get_decision",
        "list_goals",
        "goal_status",
        "next_actions",
        "get_memory",
        "list_jarvis_notes",
        "search_jarvis_notes",
        "read_jarvis_note",
        "outline_jarvis_note",
        "list_skills",
        "search_skills",
        "get_skill",
        "list_calendars",
        "list_events",
        "check_availability",
        "channel_health",
        "memory_stats",
        "recent_tool_runs",
        "execution_health_report",
        "verification_receipt",
        "runtime_trace_receipt",
        "execution_audit_gate",
        "execution_recovery_packet",
        "after_action_learning_packet",
        "recovery_closure_checklist",
        "execution_learning_closure_packet",
        "list_pending_approvals",
        "inspect_pending_approval",
        "approval_execution_packet",
        "approval_resume_packet",
        "approval_history",
        "approval_chain_proof",
        "approval_queue_summary",
        "approval_readiness_packet",
        "review_pending_approvals",
        "feedback_report",
        "feedback_actions",
        "list_weak_memories",
        "list_duplicate_memories",
        "list_scheduled_jobs",
        "web_lookup",
        "tool_search",
        "capability_cockpit",
        "restart_target_clarification",
        "personal_context_status",
        "model_routing_status",
        "voice_setup_check",
        "speak",
    }
    browser_batch = {
        "fetch_page",
        "extract_links",
        "web_search",
        "recent_browser_pages",
        "summarize_page",
        "save_page_note",
    }
    call_batch = {
        "call_contact",
        "call_kakao",
        "call_instagram",
        "call_telegram",
    }
    expected = original_batch | read_only_batch | browser_batch | call_batch
    string = ("string",)
    integer_string = ("integer", "string")
    number_string = ("number", "string")
    expected_read_only_shapes = {
        "current_time": (
            ("location", string, False),
            ("city", string, False),
            ("timezone", string, False),
            ("locale", string, False),
            ("show_timezone", ("boolean",), False),
        ),
        "time_difference": tuple(
            (name, string, False) for name in ("source", "from", "left", "target", "to", "right")
        ),
        "relative_date": tuple((name, string, False) for name in ("target", "date", "day")),
        "calculate": (("expression", string, True),),
        "generate_password": (
            ("length", integer_string, False),
            ("include_symbols", ("boolean",), False),
        ),
        "generate_uuid": (),
        "channel_health": (),
        "memory_stats": (),
        "recent_tool_runs": (("limit", ("integer",), False),),
        "execution_health_report": (("limit", ("integer",), False),),
        "execution_audit_gate": (("limit", ("integer",), False),),
        "execution_recovery_packet": (
            ("run_id", integer_string, False),
            ("id", integer_string, False),
            ("tool_run_id", integer_string, False),
            ("limit", ("integer",), False),
        ),
        "after_action_learning_packet": (
            ("run_id", integer_string, False),
            ("id", integer_string, False),
            ("tool_run_id", integer_string, False),
            ("limit", ("integer",), False),
        ),
        "recovery_closure_checklist": (("limit", ("integer",), False),),
        "execution_learning_closure_packet": (
            ("run_id", integer_string, False),
            ("id", integer_string, False),
            ("tool_run_id", integer_string, False),
            ("limit", ("integer",), False),
        ),
        "list_pending_approvals": (
            ("status", string, False),
            ("limit", ("integer",), False),
        ),
        "inspect_pending_approval": (
            ("status", string, False),
            ("approval_id", integer_string, False),
            ("id", integer_string, False),
            ("target", integer_string, False),
        ),
        "approval_execution_packet": (
            ("status", string, False),
            ("approval_id", integer_string, False),
            ("id", integer_string, False),
            ("target", integer_string, False),
        ),
        "approval_resume_packet": (
            ("approval_id", integer_string, False),
            ("id", integer_string, False),
            ("target", integer_string, False),
        ),
        "approval_history": (("limit", ("integer",), False),),
        "approval_chain_proof": (
            ("approval_id", integer_string, False),
            ("id", integer_string, False),
            ("target", integer_string, False),
        ),
        "approval_queue_summary": (
            ("status", string, False),
            ("limit", ("integer",), False),
        ),
        "approval_readiness_packet": (
            ("approval_id", integer_string, False),
            ("id", integer_string, False),
            ("target", integer_string, False),
        ),
        "review_pending_approvals": (
            ("status", string, False),
            ("limit", ("integer",), False),
        ),
        "verification_receipt": (
            ("expectation", string, False),
            ("expected", string, False),
            ("target", string, False),
            ("run_id", integer_string, False),
            ("id", integer_string, False),
            ("tool_run_id", integer_string, False),
        ),
        "runtime_trace_receipt": (
            ("session_id", string, False),
            ("message_id", integer_string, False),
            ("id", integer_string, False),
            ("trace_id", integer_string, False),
            ("limit", ("integer",), False),
        ),
        "feedback_report": (("limit", ("integer",), False),),
        "feedback_actions": (("limit", ("integer",), False),),
        "list_weak_memories": (("limit", ("integer",), False),),
        "list_duplicate_memories": (("limit", ("integer",), False),),
        "list_scheduled_jobs": (),
        "web_lookup": (("query", string, False), ("text", string, False)),
        "tool_search": (("query", string, True), ("limit", ("integer",), False)),
        "capability_cockpit": (),
        "restart_target_clarification": (),
        "personal_context_status": (),
        "model_routing_status": (),
        "voice_setup_check": (),
        "spell_word": (("word", string, False), ("text", string, False)),
        "count_text": (("text", string, False), ("phrase", string, False)),
        "transform_text": (
            ("mode", string, False),
            ("text", string, False),
            ("phrase", string, False),
            ("count", integer_string, False),
        ),
        "calculate_bmi": (
            ("weight_unit", string, True),
            ("height_unit", string, True),
            ("weight", number_string, True),
            ("height", number_string, True),
        ),
        "convert_units": (
            ("from_unit", string, True),
            ("to_unit", string, True),
            ("value", number_string, True),
        ),
        "search_memory": (
            ("query", string, True),
            ("limit", ("integer",), False),
        ),
        "recent_memories": (("limit", ("integer",), False),),
        "brain_search": (
            ("query", string, True),
            ("limit", ("integer",), False),
        ),
        "startup_recovery_report": (("limit", ("integer",), False),),
        "learning_review": (("limit", ("integer",), False),),
        "knowledge_promotion_packet": (("memory_id", ("integer",), True),),
        "list_tasks": (
            ("status", string, False),
            ("limit", integer_string, False),
        ),
        "inspect_task": (("task_id", integer_string, True),),
        "search_tasks": (
            ("query", string, True),
            ("status", string, False),
            ("limit", integer_string, False),
        ),
        "overdue_tasks": (("limit", integer_string, False),),
        "task_overview": (("limit", integer_string, False),),
        "next_task": (),
        "task_board": (("limit", integer_string, False),),
        "read_profile": (("max_chars", ("integer",), False),),
        "list_preferences": (
            ("category", string, False),
            ("status", string, False),
            ("limit", ("integer",), False),
        ),
        "get_preference": (
            ("key", string, True),
            ("category", string, False),
        ),
        "list_people": (("limit", ("integer",), False),),
        "get_person": (
            ("name", string, False),
            ("person_id", ("integer",), False),
            ("limit", ("integer",), False),
        ),
        "list_decisions": (
            ("status", string, False),
            ("limit", ("integer",), False),
        ),
        "get_decision": (("decision_id", ("integer",), True),),
        "list_goals": (
            ("status", string, False),
            ("limit", ("integer",), False),
        ),
        "goal_status": (("goal_id", ("integer",), True),),
        "next_actions": (("limit", ("integer",), False),),
        "get_memory": (("memory_id", ("integer",), True),),
        "list_jarvis_notes": (
            ("folder", string, False),
            ("limit", ("integer",), False),
        ),
        "search_jarvis_notes": (
            ("query", string, True),
            ("limit", ("integer",), False),
        ),
        "read_jarvis_note": (
            ("path", string, True),
            ("max_chars", ("integer",), False),
        ),
        "outline_jarvis_note": (("path", string, True),),
        "list_skills": (("limit", ("integer",), False),),
        "search_skills": (
            ("query", string, True),
            ("limit", ("integer",), False),
        ),
        "get_skill": (("name", string, True),),
    }
    expected_browser_egress_shapes = {
        "fetch_page": (("url", string, True), ("max_chars", ("integer",), False)),
        "extract_links": (("url", string, True),),
        "web_search": (
            ("query", string, True),
            ("engine", string, False),
            ("max_chars", ("integer",), False),
        ),
    }
    expected_browser_history_shapes = {
        "recent_browser_pages": (("limit", ("integer",), False),),
        "summarize_page": (("url", string, False), ("page_id", ("integer",), False)),
        "save_page_note": (("page_id", ("integer",), False),),
    }
    expected_call_shapes = {
        "call_contact": tuple(
            (name, string, False)
            for name in ("to", "recipient", "contact", "name", "mode", "kind", "type")
        ),
        "call_kakao": tuple(
            (name, string, False)
            for name in ("to", "recipient", "contact", "name", "mode", "kind", "type")
        ),
        "call_instagram": tuple(
            (name, string, False)
            for name in ("to", "recipient", "contact", "name", "mode", "kind", "type")
        ),
        "call_telegram": tuple(
            (name, string, False)
            for name in ("to", "recipient", "name", "mode")
        ),
    }
    with TemporaryDirectory(prefix="jarvis-typed-production-") as temp:
        runtime = make_temp_runtime(Path(temp))
        marked = {tool.name for tool in runtime.registry.list() if tool.argument_contract is not None}
        if marked != expected:
            raise SystemExit("production typed argument rollout drifted")
        for name in expected:
            tool = runtime.registry.get(name)
            if tool.argument_contract is None or not tool_argument_contract_summary(tool):
                raise SystemExit("production typed tool lacked a machine-readable contract")
        for name, expected_shape in expected_read_only_shapes.items():
            tool = runtime.registry.get(name)
            contract = tool.argument_contract
            if tool.risk is not RiskLevel.READ_ONLY or contract is None:
                raise SystemExit("read-only rollout risk or contract drifted")
            actual_shape = tuple(
                (
                    field.name,
                    tuple(sorted(argument_type.value for argument_type in field.types)),
                    field.required,
                )
                for field in contract.fields
            )
            if (
                contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
                or contract.allow_unknown is not False
                or actual_shape != expected_shape
            ):
                raise SystemExit(f"production schema shape drifted for {name}")
        apple_tool = runtime.registry.get("apple_reminders")
        apple_contract = apple_tool.argument_contract
        apple_shape = tuple(
            (
                field.name,
                tuple(sorted(argument_type.value for argument_type in field.types)),
                field.required,
            )
            for field in (apple_contract.fields if apple_contract is not None else ())
        )
        if (
            apple_tool.risk is not RiskLevel.PERSONAL_DATA
            or apple_contract is None
            or apple_contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or apple_contract.allow_unknown is not False
            or apple_shape
            != (("list_name", ("string",), False), ("limit", ("integer",), False))
        ):
            raise SystemExit("Apple Reminders personal-data schema or risk drifted")
        for name, expected_shape in expected_browser_egress_shapes.items():
            tool = runtime.registry.get(name)
            contract = tool.argument_contract
            if tool.risk is not RiskLevel.LOCAL_SAFE or contract is None:
                raise SystemExit("browser egress rollout risk or contract drifted")
            actual_shape = tuple(
                (
                    field.name,
                    tuple(sorted(argument_type.value for argument_type in field.types)),
                    field.required,
                )
                for field in contract.fields
            )
            if (
                contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
                or contract.allow_unknown is not False
                or actual_shape != expected_shape
            ):
                raise SystemExit(f"browser egress schema shape drifted for {name}")
        for name, expected_shape in expected_browser_history_shapes.items():
            tool = runtime.registry.get(name)
            contract = tool.argument_contract
            expected_risk = RiskLevel.LOCAL_SAFE if name == "save_page_note" else RiskLevel.READ_ONLY
            if tool.risk is not expected_risk or contract is None:
                raise SystemExit("browser history rollout risk or contract drifted")
            actual_shape = tuple(
                (
                    field.name,
                    tuple(sorted(argument_type.value for argument_type in field.types)),
                    field.required,
                )
                for field in contract.fields
            )
            if (
                contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
                or contract.allow_unknown is not False
                or actual_shape != expected_shape
            ):
                raise SystemExit(f"browser history schema shape drifted for {name}")
        for name, expected_shape in expected_call_shapes.items():
            tool = runtime.registry.get(name)
            contract = tool.argument_contract
            if tool.risk is not RiskLevel.HIGH_RISK or contract is None:
                raise SystemExit("call argument-contract rollout risk or contract drifted")
            actual_shape = tuple(
                (
                    field.name,
                    tuple(sorted(argument_type.value for argument_type in field.types)),
                    field.required,
                )
                for field in contract.fields
            )
            if (
                contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
                or contract.allow_unknown is not False
                or actual_shape != expected_shape
            ):
                raise SystemExit(f"call schema shape drifted for {name}")
        decision_contract = runtime.registry.get("record_decision").argument_contract
        if decision_contract is None:
            raise SystemExit("record_decision missed its strict argument contract")
        decision_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in decision_contract.fields
        )
        if decision_shape != (
            ("title", ("string",), True),
            ("rationale", ("string",), False),
            ("impact", ("string",), False),
        ):
            raise SystemExit(f"record_decision schema shape drifted: {decision_shape}")
        outcome_contract = runtime.registry.get("record_decision_outcome").argument_contract
        if outcome_contract is None:
            raise SystemExit("record_decision_outcome missed its strict argument contract")
        outcome_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in outcome_contract.fields
        )
        if outcome_shape != (
            ("summary", ("string",), True),
            ("decision_id", ("integer",), True),
        ):
            raise SystemExit(
                f"record_decision_outcome schema shape drifted: {outcome_shape}"
            )
        person_contract = runtime.registry.get("add_person").argument_contract
        if person_contract is None:
            raise SystemExit("add_person missed its strict argument contract")
        person_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in person_contract.fields
        )
        if person_shape != (
            ("name", ("string",), True),
            ("relation", ("string",), False),
            ("notes", ("string",), False),
        ):
            raise SystemExit(f"add_person schema shape drifted: {person_shape}")
        interaction_contract = runtime.registry.get("log_interaction").argument_contract
        if interaction_contract is None:
            raise SystemExit("log_interaction missed its strict argument contract")
        interaction_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in interaction_contract.fields
        )
        if (
            interaction_contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or interaction_contract.allow_unknown is not False
            or interaction_shape
            != (
                ("summary", ("string",), True),
                ("name", ("string",), False),
                ("happened_at", ("string",), False),
                ("person_id", ("integer", "string"), False),
            )
        ):
            raise SystemExit(f"log_interaction schema shape drifted: {interaction_shape}")
        export_goal_contract = runtime.registry.get("export_goal").argument_contract
        if export_goal_contract is None:
            raise SystemExit("export_goal missed its strict argument contract")
        export_goal_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in export_goal_contract.fields
        )
        if export_goal_shape != (("goal_id", ("integer", "string"), True),):
            raise SystemExit(f"export_goal schema shape drifted: {export_goal_shape}")
        export_tasks_contract = runtime.registry.get("export_tasks").argument_contract
        if export_tasks_contract is None:
            raise SystemExit("export_tasks missed its strict argument contract")
        export_tasks_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in export_tasks_contract.fields
        )
        if export_tasks_shape != (("limit", ("integer",), False),):
            raise SystemExit(f"export_tasks schema shape drifted: {export_tasks_shape}")
        learning_queue_contract = runtime.registry.get("queue_learning_tasks").argument_contract
        if learning_queue_contract is None:
            raise SystemExit("queue_learning_tasks missed its strict argument contract")
        learning_queue_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in learning_queue_contract.fields
        )
        if learning_queue_shape != (("limit", ("integer",), False),):
            raise SystemExit(f"queue_learning_tasks schema shape drifted: {learning_queue_shape}")
        for feedback_projection_name in (
            "save_feedback_report",
            "save_feedback_actions",
        ):
            feedback_projection_contract = runtime.registry.get(
                feedback_projection_name
            ).argument_contract
            if feedback_projection_contract is None:
                raise SystemExit(
                    f"{feedback_projection_name} missed its strict argument contract"
                )
            feedback_projection_shape = tuple(
                (
                    field.name,
                    tuple(sorted(kind.value for kind in field.types)),
                    field.required,
                )
                for field in feedback_projection_contract.fields
            )
            if feedback_projection_shape != (("limit", ("integer",), False),):
                raise SystemExit(
                    f"{feedback_projection_name} schema shape drifted: "
                    f"{feedback_projection_shape}"
                )
        set_reminder_tool = runtime.registry.get("set_reminder")
        set_reminder_contract = set_reminder_tool.argument_contract
        set_reminder_bound_contract = set_reminder_tool.approval_argument_contract
        if (
            set_reminder_tool.risk is not RiskLevel.EXTERNAL_SIDE_EFFECT
            or set_reminder_contract is None
            or set_reminder_contract.allow_unknown
            or set_reminder_tool.approval_argument_resolver is None
            or set_reminder_bound_contract is None
            or set_reminder_bound_contract.allow_unknown
        ):
            raise SystemExit("set_reminder missed its immutable pre-approval binding contract")
        set_reminder_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in set_reminder_contract.fields
        )
        if set_reminder_shape != (("text", ("string",), True),):
            raise SystemExit(f"set_reminder schema shape drifted: {set_reminder_shape}")
        set_reminder_bound_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in set_reminder_bound_contract.fields
        )
        if set_reminder_bound_shape != (
            ("due_epoch", ("number",), True),
            ("message", ("string",), True),
            ("owner_fingerprint", ("string",), True),
        ):
            raise SystemExit(f"set_reminder approval schema drifted: {set_reminder_bound_shape}")
        task_import_contract = runtime.registry.get("import_tasks_from_note").argument_contract
        if task_import_contract is None:
            raise SystemExit("import_tasks_from_note missed its strict argument contract")
        task_import_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in task_import_contract.fields
        )
        if task_import_shape != (
            ("path", ("string",), True),
            ("priority", ("string",), False),
        ):
            raise SystemExit(f"import_tasks_from_note schema shape drifted: {task_import_shape}")
        task_details_contract = runtime.registry.get("update_task_details").argument_contract
        if task_details_contract is None:
            raise SystemExit("update_task_details missed its strict argument contract")
        task_details_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in task_details_contract.fields
        )
        if task_details_shape != (
            ("body", ("string",), False),
            ("due", ("string",), False),
            ("priority", ("string",), False),
            ("task_id", ("integer",), True),
        ):
            raise SystemExit(f"update_task_details schema shape drifted: {task_details_shape}")
        state_snapshot_tool = runtime.registry.get("export_state_snapshot")
        if (
            state_snapshot_tool.risk is not RiskLevel.LOCAL_SAFE
            or state_snapshot_tool.auto_mutation_contract is None
            or state_snapshot_tool.argument_contract is None
            or state_snapshot_tool.argument_contract.fields != ()
        ):
            raise SystemExit("export_state_snapshot missed its empty local-mutation contract")
        profile_note_tool = runtime.registry.get("add_profile_note")
        profile_note_contract = profile_note_tool.argument_contract
        if profile_note_tool.risk is not RiskLevel.LOCAL_SAFE or profile_note_contract is None:
            raise SystemExit("add_profile_note missed its local-safe strict argument contract")
        profile_note_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in profile_note_contract.fields
        )
        if profile_note_shape != (
            ("body", ("string",), True),
            ("heading", ("string",), False),
            ("category", ("string",), False),
        ):
            raise SystemExit(f"add_profile_note schema shape drifted: {profile_note_shape}")
        organize_note_tool = runtime.registry.get("organize_note")
        organize_note_contract = organize_note_tool.argument_contract
        if organize_note_tool.risk is not RiskLevel.LOCAL_SAFE or organize_note_contract is None:
            raise SystemExit("organize_note missed its local-safe strict argument contract")
        organize_note_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in organize_note_contract.fields
        )
        if organize_note_shape != (("text", ("string",), True),):
            raise SystemExit(f"organize_note schema shape drifted: {organize_note_shape}")
        note_write_tool = runtime.registry.get("write_jarvis_note")
        note_write_contract = note_write_tool.argument_contract
        if note_write_tool.risk is not RiskLevel.LOCAL_SAFE or note_write_contract is None:
            raise SystemExit("write_jarvis_note missed its local-safe strict argument contract")
        note_write_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in note_write_contract.fields
        )
        if note_write_shape != (
            ("path", ("string",), True),
            ("body", ("string",), True),
            ("mode", ("string",), False),
        ):
            raise SystemExit(f"write_jarvis_note schema shape drifted: {note_write_shape}")
        delete_skill_tool = runtime.registry.get("delete_skill")
        delete_skill_raw = delete_skill_tool.argument_contract
        delete_skill_bound = delete_skill_tool.approval_argument_contract
        if (
            delete_skill_tool.risk is not RiskLevel.HIGH_RISK
            or delete_skill_tool.approval_argument_resolver is None
            or delete_skill_raw is None
            or delete_skill_bound is None
        ):
            raise SystemExit("delete_skill missed its immutable pre-approval binding contract")
        raw_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in delete_skill_raw.fields
        )
        bound_shape = tuple(
            (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
            for field in delete_skill_bound.fields
        )
        if raw_shape != (("name", ("string",), True),):
            raise SystemExit(f"delete_skill raw schema drifted: {raw_shape}")
        if bound_shape != (
            ("name", ("string",), True),
            ("target_skill_name", ("string",), True),
            ("target_skill_id", ("integer",), True),
            ("target_skill_revision", ("integer",), True),
        ):
            raise SystemExit(f"delete_skill approval schema drifted: {bound_shape}")
        expected_memory_shapes = {
            "delete_memory": (
                (("memory_id", ("integer",), True),),
                (
                    ("target_binding", ("string",), True),
                    ("memory_id", ("integer",), True),
                    ("target_revision", ("integer",), True),
                ),
            ),
            "edit_memory": (
                (
                    ("category", ("string",), False),
                    ("title", ("string",), False),
                    ("body", ("string",), False),
                    ("memory_id", ("integer",), True),
                    ("confidence", ("number",), False),
                ),
                (
                    ("target_binding", ("string",), True),
                    ("category", ("string",), False),
                    ("title", ("string",), False),
                    ("body", ("string",), False),
                    ("memory_id", ("integer",), True),
                    ("target_revision", ("integer",), True),
                    ("confidence", ("number",), False),
                ),
            ),
            "merge_memories": (
                (
                    ("keep_id", ("integer",), True),
                    ("delete_id", ("integer",), True),
                ),
                (
                    ("keep_binding", ("string",), True),
                    ("delete_binding", ("string",), True),
                    ("keep_id", ("integer",), True),
                    ("keep_revision", ("integer",), True),
                    ("delete_id", ("integer",), True),
                    ("delete_revision", ("integer",), True),
                ),
            ),
            "promote_memory_to_decision": (
                (
                    ("review_token", ("string",), True),
                    ("title", ("string",), True),
                    ("rationale", ("string",), True),
                    ("impact", ("string",), True),
                    ("memory_id", ("integer",), True),
                    ("reviewed_revision", ("integer",), True),
                ),
                (
                    ("review_binding", ("string",), True),
                    ("title", ("string",), True),
                    ("rationale", ("string",), True),
                    ("impact", ("string",), True),
                    ("target_binding", ("string",), True),
                    ("memory_id", ("integer",), True),
                    ("reviewed_revision", ("integer",), True),
                    ("target_revision", ("integer",), True),
                ),
            ),
            "promote_memory_to_preference": (
                (
                    ("review_token", ("string",), True),
                    ("category", ("string",), True),
                    ("key", ("string",), True),
                    ("value", ("string",), True),
                    ("memory_id", ("integer",), True),
                    ("reviewed_revision", ("integer",), True),
                ),
                (
                    ("review_binding", ("string",), True),
                    ("category", ("string",), True),
                    ("key", ("string",), True),
                    ("value", ("string",), True),
                    ("target_binding", ("string",), True),
                    ("memory_id", ("integer",), True),
                    ("reviewed_revision", ("integer",), True),
                    ("target_revision", ("integer",), True),
                ),
            ),
            "promote_memory_to_profile": (
                (
                    ("review_token", ("string",), True),
                    ("heading", ("string",), True),
                    ("category", ("string",), True),
                    ("body", ("string",), True),
                    ("memory_id", ("integer",), True),
                    ("reviewed_revision", ("integer",), True),
                ),
                (
                    ("review_binding", ("string",), True),
                    ("heading", ("string",), True),
                    ("category", ("string",), True),
                    ("body", ("string",), True),
                    ("target_binding", ("string",), True),
                    ("memory_id", ("integer",), True),
                    ("reviewed_revision", ("integer",), True),
                    ("target_revision", ("integer",), True),
                ),
            ),
        }
        for name, (expected_raw, expected_bound) in expected_memory_shapes.items():
            tool = runtime.registry.get(name)
            if tool.approval_argument_resolver is None or tool.approval_argument_contract is None:
                raise SystemExit(f"{name} missed its immutable pre-approval binding contract")
            actual_raw = tuple(
                (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                for field in tool.argument_contract.fields
            )
            actual_bound = tuple(
                (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                for field in tool.approval_argument_contract.fields
            )
            if actual_raw != expected_raw or actual_bound != expected_bound:
                raise SystemExit(
                    f"{name} approval argument shapes drifted: raw={actual_raw}, bound={actual_bound}"
                )
        for name in ["send_email", "run_shell_command", "create_reminder"]:
            if runtime.registry.get(name).argument_contract is not None:
                raise SystemExit("unverified side-effect or conditional tool entered typed rollout")


def main() -> None:
    test_registry_rejects_malformed_contract_definitions()
    test_executor_enforces_shape_before_handler_and_permission()
    test_cancel_reminders_production_contract_rejects_before_handler()
    test_pre_approval_argument_resolver_contract()
    test_number_contract_rejects_non_finite_values()
    test_legacy_unschematized_tools_remain_compatible()
    test_invalid_high_risk_arguments_never_queue_approval()
    test_invalid_auto_mutation_plan_stops_before_receipts_and_handlers()
    test_model_planner_advertises_and_enforces_contracts()
    test_model_planner_preserves_malformed_optional_only_argument_roots()
    test_model_fallback_metadata_does_not_store_response_content()
    test_runtime_rejects_malformed_reconciliation_roots_before_route_guard()
    test_invalid_arguments_stop_before_contact_policy_and_handler()
    test_contract_failures_are_content_private_and_auditable()
    test_read_only_batch_real_command_paths()
    test_tool_search_contract_stops_malformed_queries_before_handler()
    test_capability_cockpit_contract_stops_arguments_before_audit_reads()
    test_model_routing_status_contract_stops_arguments_before_probe()
    test_voice_setup_check_contract_stops_arguments_before_asr_inspection()
    test_search_memory_production_contract_stops_malformed_queries()
    test_recent_memories_production_contract_stops_malformed_limits()
    test_memory_curator_review_contracts_stop_malformed_limits_before_private_reads()
    test_feedback_review_contracts_stop_malformed_limits_before_private_reads()
    test_audit_overview_contracts_stop_malformed_limits_before_history_reads()
    test_audit_receipt_contracts_stop_malformed_selectors_before_handlers()
    test_execution_audit_packet_contracts_stop_malformed_history_reads()
    test_approval_review_contracts_stop_malformed_queue_reads()
    test_browser_egress_contracts_stop_malformed_arguments_before_fetch()
    test_browser_history_contracts_stop_malformed_arguments_before_handler_or_write()
    test_brain_search_production_contract_stops_malformed_queries()
    test_production_schema_rollout_is_exact()
    print("Tool argument contract smoke passed")


if __name__ == "__main__":
    main()
