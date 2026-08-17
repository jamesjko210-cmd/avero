from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    AUTO_MUTATION_META_TOOL_EXCEPTIONS,
    AutoMutationContract,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    Tool,
    ToolRegistry,
)


READ_ONLY_FLAGS = [
    "calls_model",
    "calls_external_service",
    "executes_tools",
    "reads_personal_data",
    "reads_private_data",
    "executes_side_effect",
    "external_side_effect",
    "writes_files",
    "writes_database",
    "writes_memory",
    "writes_notes",
    "queues_approval",
    "requires_approval",
    "controls_computer",
    "speaks",
    "completes_tasks",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
]


def _noop_tool(_: dict) -> ToolResult:
    return ToolResult("fixture", True, "ok")


def _valid_auto_mutation_contract() -> AutoMutationContract:
    return AutoMutationContract(
        version=AUTO_MUTATION_CONTRACT_VERSION,
        effects=frozenset({AutoMutationEffect.LOCAL_DATABASE}),
        replay_policy=AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
        crash_policy=AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
    )


def _assert_auto_mutation_registry_contract(runtime) -> None:
    default_registry = ToolRegistry()
    default_registry.register(Tool("default_fixture", "fixture", RiskLevel.LOCAL_SAFE, _noop_tool))
    if default_registry.get("default_fixture").auto_mutation_contract is not None:
        raise SystemExit("Tool auto-mutation eligibility must default to disabled")

    marked_names = {
        tool.name
        for tool in runtime.registry.list()
        if tool.auto_mutation_contract is not None
    }
    expected_effects = {
        "remember": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "write_daily_note": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "record_feedback": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "save_feedback_report": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "save_feedback_actions": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "record_decision": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "record_decision_outcome": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "set_preference_status": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "set_preference": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "add_person": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "log_interaction": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "set_decision_status": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "save_skill": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "memory_tree_summary": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "queue_learning_tasks": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "add_task": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "complete_task": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "complete_task_with_evidence": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "update_task_status": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "update_task_details": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "import_tasks_from_note": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "create_goal": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "add_goal_step": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "complete_goal_step": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "set_goal_status": frozenset({AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}),
        "export_goal": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "export_tasks": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "export_state_snapshot": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "daily_brief": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "daily_plan": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "goal_nudge": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "add_profile_note": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "organize_note": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
        "write_jarvis_note": frozenset({AutoMutationEffect.OBSIDIAN_VAULT}),
        "ingest_obsidian_inbox": frozenset(
            {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
        ),
    }
    expected_marked = set(expected_effects)
    if marked_names != expected_marked:
        raise SystemExit(f"Unexpected auto-mutation rollout: {sorted(marked_names)}")

    for name in expected_marked:
        contract = runtime.registry.get(name).auto_mutation_contract
        if contract is None or not contract.effects:
            raise SystemExit(f"Marked tool lacks a positive auto-mutation contract: {name}")
        if contract.effects != expected_effects[name]:
            raise SystemExit(f"Marked tool has incorrect local effects: {name} {contract.effects}")
        if contract.version != AUTO_MUTATION_CONTRACT_VERSION:
            raise SystemExit(f"Marked tool has unsupported auto-mutation version: {name}")
        if contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY:
            raise SystemExit(f"Marked tool has unsafe replay policy: {name}")
        if contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN:
            raise SystemExit(f"Marked tool has unsafe crash policy: {name}")

    task_status_aliases = frozenset(
        {"complete_task", "complete_task_with_evidence", "update_task_status"}
    )
    for name in task_status_aliases:
        contract = runtime.registry.get(name).auto_mutation_contract
        if (
            contract is None
            or contract.operation_scope != "task_status"
            or contract.legacy_operation_aliases != task_status_aliases
        ):
            raise SystemExit(f"Task-status shared receipt scope drifted: {name} {contract}")

    if AUTO_MUTATION_META_TOOL_EXCEPTIONS != frozenset(
        {("queue_learning_tasks", "learning")}
    ):
        raise SystemExit(
            "Auto-mutation meta-tool exceptions must remain queue_learning_tasks/learning-only: "
            f"{sorted(AUTO_MUTATION_META_TOOL_EXCEPTIONS)}"
        )
    exception_registry = ToolRegistry()
    exception_registry.register(
        Tool(
            "queue_learning_tasks",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            "learning",
            _valid_auto_mutation_contract(),
        )
    )
    if exception_registry.get("queue_learning_tasks").toolset != "learning":
        raise SystemExit("queue_learning_tasks meta-tool exception changed its public toolset")
    for wrong_toolset in ("approvals", "audit", "continuity", "safety", "scheduler"):
        wrong_registry = ToolRegistry()
        try:
            wrong_registry.register(
                Tool(
                    "queue_learning_tasks",
                    "fixture",
                    RiskLevel.LOCAL_SAFE,
                    _noop_tool,
                    wrong_toolset,
                    _valid_auto_mutation_contract(),
                )
            )
        except ValueError:
            if wrong_registry.list():
                raise SystemExit(
                    f"Wrong-category queue exception was retained: {wrong_toolset}"
                )
        else:
            raise SystemExit(
                f"queue_learning_tasks exception escaped into meta toolset: {wrong_toolset}"
            )

    for name in [
        "current_time",
        "save_learning_review",
        "schedule_daily_brief",
        "create_reminder",
        "system_info",
        "enable_computer_control",
        "append_execution_case_evidence",
        "approve_pending_approval",
        "send_email",
        "move_mouse",
    ]:
        if runtime.registry.get(name).auto_mutation_contract is not None:
            raise SystemExit(f"Forbidden or conditional tool became auto-mutation eligible: {name}")

    invalid_tools = [
        Tool("read_only_fixture", "fixture", RiskLevel.READ_ONLY, _noop_tool, auto_mutation_contract=_valid_auto_mutation_contract()),
        Tool("high_risk_fixture", "fixture", RiskLevel.HIGH_RISK, _noop_tool, auto_mutation_contract=_valid_auto_mutation_contract()),
        Tool("external_risk_fixture", "fixture", RiskLevel.EXTERNAL_SIDE_EFFECT, _noop_tool, auto_mutation_contract=_valid_auto_mutation_contract()),
        Tool("external_fixture", "fixture", RiskLevel.LOCAL_SAFE, _noop_tool, "personal", _valid_auto_mutation_contract()),
        Tool("computer_fixture", "fixture", RiskLevel.LOCAL_SAFE, _noop_tool, "computer", _valid_auto_mutation_contract()),
        Tool("meta_fixture", "fixture", RiskLevel.LOCAL_SAFE, _noop_tool, "approvals", _valid_auto_mutation_contract()),
        Tool("learning_meta_fixture", "fixture", RiskLevel.LOCAL_SAFE, _noop_tool, "learning", _valid_auto_mutation_contract()),
        Tool(
            "empty_effects_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset(),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
        ),
        Tool(
            "unknown_effect_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({"network"}),  # type: ignore[arg-type]
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
        ),
        Tool(
            "unsupported_version_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION + 1,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
        ),
        Tool(
            "unsupported_replay_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                "blind_retry",  # type: ignore[arg-type]
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
            ),
        ),
        Tool(
            "unsupported_crash_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                "retry",  # type: ignore[arg-type]
            ),
        ),
        Tool(
            "noncallable_operation_key_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_key_builder="not callable",  # type: ignore[arg-type]
            ),
        ),
        Tool(
            "noncallable_semantic_preflight_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                semantic_preflight="not callable",  # type: ignore[arg-type]
            ),
        ),
        Tool(
            "noncallable_execution_args_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                execution_args_builder="not callable",  # type: ignore[arg-type]
            ),
        ),
        Tool(
            "noncallable_semantic_result_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                semantic_preflight=lambda _args: "fixture",
                semantic_preflight_result_builder="not callable",  # type: ignore[arg-type]
            ),
        ),
        Tool(
            "orphaned_semantic_result_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                semantic_preflight_result_builder=lambda _args, _reason: ToolResult(
                    "orphaned_semantic_result_fixture", False, "blocked"
                ),
            ),
        ),
        Tool(
            "nonset_definite_no_effect_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                definite_no_effect_failure_reasons={"missing"},  # type: ignore[arg-type]
            ),
        ),
        Tool(
            "malformed_definite_no_effect_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                definite_no_effect_failure_reasons=frozenset({"bad reason"}),
            ),
        ),
        Tool(
            "legacy_alias_without_scope_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                legacy_operation_aliases=frozenset(
                    {"legacy_alias_without_scope_fixture"}
                ),
            ),
        ),
        Tool(
            "legacy_alias_omits_tool_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_scope="shared_fixture",
                legacy_operation_aliases=frozenset({"other_fixture"}),
            ),
        ),
        Tool(
            "malformed_legacy_alias_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_scope="shared_fixture",
                legacy_operation_aliases=frozenset(
                    {"malformed_legacy_alias_fixture", "bad alias"}
                ),
            ),
        ),
        Tool(
            "excessive_legacy_alias_fixture",
            "fixture",
            RiskLevel.LOCAL_SAFE,
            _noop_tool,
            auto_mutation_contract=AutoMutationContract(
                AUTO_MUTATION_CONTRACT_VERSION,
                frozenset({AutoMutationEffect.LOCAL_DATABASE}),
                AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY,
                AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN,
                operation_scope="shared_fixture",
                legacy_operation_aliases=frozenset(
                    {"excessive_legacy_alias_fixture"}
                    | {f"legacy_alias_{index}" for index in range(32)}
                ),
            ),
        ),
    ]
    for tool in invalid_tools:
        registry = ToolRegistry()
        try:
            registry.register(tool)
        except ValueError:
            if registry.list():
                raise SystemExit(f"Invalid contract was retained after rejection: {tool.name}")
        else:
            raise SystemExit(f"Invalid auto-mutation contract registered: {tool.name}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-architecture-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _assert_auto_mutation_registry_contract(runtime)
        cases = ["architecture map", "brain architecture", "architecture please"]
        for case in cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in [
                "Jarvis V2 architecture map",
                "Perception and input handling",
                "Natural language understanding",
                "Reasoning and planning",
                "Memory",
                "Action and tool execution",
                "Learning and continuous improvement",
                "agent harness",
                "Harness completion gates",
                "Safety spine",
                "pending approvals",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Architecture map missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("layers") != 6 or metadata.get("harness_gates") != 5:
                raise SystemExit(f"Architecture metadata missed core counts: {metadata}")
            for key in READ_ONLY_FLAGS:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Architecture map unsafe metadata {key}: {metadata}")


if __name__ == "__main__":
    main()
