from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.goals import MAX_GOAL_TITLE_CHARS
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    ToolArgumentType,
)


TOOL_NAME = "create_goal"
RETRY_COMMAND = "create goal <title> because <purpose>"
AUTHORIZATION_FLAGS = (
    "authorizes_retry",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "requires_confirmation",
    "requires_approval",
    "queues_approval",
)


class StaticPlanner:
    def __init__(self, args: dict[str, Any]):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise create-goal auto-mutation preflight.",
            [PlannedAction(TOOL_NAME, dict(self.args), "create-goal preflight smoke")],
            needs_model=False,
        )


def _runtime(root: Path, args: dict[str, Any]) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args)
    return runtime


def _set_args(runtime: JarvisRuntime, args: dict[str, Any]) -> None:
    runtime.planner = StaticPlanner(args)


def _rows(
    runtime: JarvisRuntime,
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute(sql, params)]


def _receipt_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM auto_mutation_receipts ORDER BY id")


def _tool_run_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM tool_runs ORDER BY id")


def _goal_rows(runtime: JarvisRuntime) -> list[dict[str, Any]]:
    return _rows(runtime, "SELECT * FROM goals ORDER BY id")


def _vault_snapshot(runtime: JarvisRuntime) -> dict[str, bytes]:
    return {
        str(path.relative_to(runtime.vault.root_path)): path.read_bytes()
        for path in runtime.vault.root_path.rglob("*")
        if path.is_file()
    }


def _install_counted_handler(runtime: JarvisRuntime) -> list[int]:
    tool = runtime.registry.get(TOOL_NAME)
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools[TOOL_NAME] = replace(tool, handler=counted)
    return calls


def _assert_no_approvals(runtime: JarvisRuntime, label: str) -> None:
    approvals = _rows(runtime, "SELECT * FROM pending_approvals ORDER BY id")
    runs = _tool_run_rows(runtime)
    if approvals:
        raise SystemExit(f"{label} queued an approval: {approvals}")
    if any(
        row["approved"] != 0
        or row["approval_id"] is not None
        or row["approval_action_digest"] is not None
        for row in runs
    ):
        raise SystemExit(f"{label} created approval evidence: {runs}")


def _assert_refusal(result: Any, reason: str, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} returned an unexpected result count: {result.tool_results}")
    item = result.tool_results[0]
    metadata = item.metadata
    handoff = metadata.get("goal_refusal_handoff")
    if (
        item.ok
        or metadata.get("failure_kind") != "auto_mutation_semantic_preflight_rejected"
        or metadata.get("reason") != reason
        or metadata.get("handler_invoked") is not False
        or metadata.get("executed_handler") is not False
        or metadata.get("state_changed") is not False
        or metadata.get("auto_mutation_effects_started") is not False
        or metadata.get("writes_files") is not False
        or metadata.get("writes_database") is not False
        or metadata.get("writes_memory") is not False
        or metadata.get("writes_notes") is not False
        or metadata.get("external_side_effect") is not False
        or metadata.get("controls_computer") is not False
        or metadata.get("goal_refusal_handoff_ready") is not True
        or metadata.get("goal_mutation_handoff_ready") is not False
        or tuple(metadata.get("next_safe_commands") or ())
        != (RETRY_COMMAND, "list goals", "next actions")
        or metadata.get("next_safe_command") != RETRY_COMMAND
        or metadata.get("next_safe_command_count") != 3
        or not isinstance(handoff, dict)
        or handoff.get("source") != TOOL_NAME
        or handoff.get("reason") != reason
        or handoff.get("mutation") != "goal_create"
        or handoff.get("refused") is not True
        or handoff.get("handoff_ready") is not True
        or handoff.get("next_commands", {}).get("retry") != RETRY_COMMAND
    ):
        raise SystemExit(f"{label} lost rich create-goal recovery metadata: {item}")
    if any(metadata.get(key) is not False for key in AUTHORIZATION_FLAGS):
        raise SystemExit(f"{label} granted authority: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    if not boundaries or any(value is not False for value in boundaries.values()):
        raise SystemExit(f"{label} crossed a refusal boundary: {handoff}")


def _assert_private_refusal_audits(
    runtime: JarvisRuntime,
    *,
    results: tuple[Any, ...],
    private_values: tuple[str, ...],
    expected_runs: int,
    label: str,
) -> None:
    runs = _tool_run_rows(runtime)
    if len(runs) != expected_runs:
        raise SystemExit(f"{label} audit count drifted: {runs}")
    for run in runs:
        metadata = json.loads(run["metadata"] or "{}")
        if (
            run["tool_name"] != TOOL_NAME
            or run["ok"] != 0
            or metadata.get("failure_kind")
            != "auto_mutation_semantic_preflight_rejected"
            or metadata.get("handler_invoked") is not False
            or metadata.get("executed_handler") is not False
        ):
            raise SystemExit(f"{label} refusal audit looked executed: {run}")
    assistant_messages = _rows(
        runtime,
        "SELECT content, metadata FROM messages WHERE role = 'assistant' ORDER BY id",
    )
    audit_text = json.dumps(
        {
            "runs": [
                {"output": row["output"], "metadata": row["metadata"]}
                for row in runs
            ],
            "runtime_results": [
                {
                    "response": result.response,
                    "metadata": result.metadata,
                    "plan_actions": [
                        {
                            "tool_name": action.tool_name,
                            "args": action.args,
                        }
                        for action in result.plan.actions
                    ],
                }
                for result in results
            ],
            "assistant_messages": assistant_messages,
        },
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )
    for private in private_values:
        if private and (private in audit_text or private.encode("unicode_escape").decode() in audit_text):
            raise SystemExit(f"{label} audit exposed refused title content")


def _assert_completed_receipt_audits(runtime: JarvisRuntime, expected: int) -> None:
    receipts = _receipt_rows(runtime)
    successful = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
    runs_by_id = {row["id"]: row for row in successful}
    if len(receipts) != expected or len(successful) != expected:
        raise SystemExit(
            f"create_goal expected {expected} receipt/audit pairs: {receipts} / {successful}"
        )
    linked: set[int] = set()
    for receipt in receipts:
        run = runs_by_id.get(receipt["tool_run_id"])
        if (
            receipt["tool_name"] != TOOL_NAME
            or receipt["state"] != "completed"
            or receipt["result"] != "succeeded"
            or receipt["resolution"] != "recorded"
            or run is None
            or run["tool_name"] != TOOL_NAME
            or run["approved"] != 0
            or run["approval_id"] is not None
            or run["approval_action_digest"] is not None
        ):
            raise SystemExit(f"create_goal receipt lost ordinary audit linkage: {receipt} / {run}")
        linked.add(int(run["id"]))
    if len(linked) != expected:
        raise SystemExit("create_goal receipts reused an ordinary audit row")


def test_exact_contract_and_title_only_identity() -> None:
    with TemporaryDirectory(prefix="jarvis-create-goal-contract-") as temp:
        runtime = _runtime(Path(temp), {"title": "Contract goal"})
        tool = runtime.registry.get(TOOL_NAME)
        contract = tool.auto_mutation_contract
        if (
            tool.risk is not RiskLevel.LOCAL_SAFE
            or tool.toolset != "goals"
            or contract is None
            or contract.version != AUTO_MUTATION_CONTRACT_VERSION
            or contract.effects
            != frozenset(
                {AutoMutationEffect.LOCAL_DATABASE, AutoMutationEffect.OBSIDIAN_VAULT}
            )
            or contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
            or contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
            or not callable(contract.operation_key_builder)
            or not callable(contract.semantic_preflight)
            or not callable(contract.semantic_preflight_result_builder)
        ):
            raise SystemExit(f"create_goal auto-mutation contract drifted: {tool}")

        arguments = tool.argument_contract
        string = frozenset({ToolArgumentType.STRING})
        shape = (
            tuple((field.name, field.types, field.required) for field in arguments.fields)
            if arguments is not None
            else ()
        )
        if (
            arguments is None
            or arguments.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or arguments.allow_unknown
            or shape
            != (
                ("title", string, True),
                ("purpose", string, False),
                ("horizon", string, False),
            )
        ):
            raise SystemExit(f"create_goal strict argument contract drifted: {arguments}")

        builder = contract.operation_key_builder
        equivalent = (
            builder({"title": "  Stable   goal  ", "purpose": "first", "horizon": "week"}),
            builder({"title": "Stable goal", "purpose": "changed", "horizon": "year"}),
            builder({"title": "Stable goal", "purpose": "", "horizon": ""}),
            builder({"title": "ＳＴＡＢＬＥ ＧＯＡＬ", "purpose": "unicode equivalent"}),
        )
        if len(set(json.dumps(item, sort_keys=True) for item in equivalent)) != 1:
            raise SystemExit(f"create_goal identity is not canonical title-only identity: {equivalent}")
        if set(equivalent[0]) != {"title"}:
            raise SystemExit(f"create_goal operation identity retained mutable fields: {equivalent[0]}")
        if builder({"title": "Different goal", "purpose": "changed"}) == equivalent[0]:
            raise SystemExit("create_goal collapsed different titles into one operation target")
        expanding_title = "\ufdfa" * MAX_GOAL_TITLE_CHARS
        normalized_title = builder({"title": expanding_title})["title"]
        if len(normalized_title) > MAX_GOAL_TITLE_CHARS:
            raise SystemExit(
                "create_goal operation identity exceeded the title bound after NFKC expansion"
            )


def test_refusals_re_evaluate_and_correct_with_same_token() -> None:
    cases = (
        ({"title": "", "purpose": "private missing purpose"}, "missing_title", ("private missing purpose",)),
        (
            {"title": "/\x55sers/private/PRIVATE-CREATE-GOAL-PATH", "purpose": "path purpose"},
            "invalid_title",
            ("/\x55sers/private/PRIVATE-CREATE-GOAL-PATH", "path purpose"),
        ),
        (
            {"title": "/users/private/PRIVATE-LOWERCASE-PATH", "purpose": "lower path purpose"},
            "invalid_title",
            ("/users/private/PRIVATE-LOWERCASE-PATH", "lower path purpose"),
        ),
        (
            {"title": "/Volumes/PRIVATE-VOLUME/goal", "purpose": "volume purpose"},
            "invalid_title",
            ("/Volumes/PRIVATE-VOLUME/goal", "volume purpose"),
        ),
        (
            {"title": "~/PRIVATE-HOME/goal", "purpose": "home purpose"},
            "invalid_title",
            ("~/PRIVATE-HOME/goal", "home purpose"),
        ),
        (
            {"title": "C:\\PRIVATE-WINDOWS\\goal", "purpose": "windows purpose"},
            "invalid_title",
            ("C:\\PRIVATE-WINDOWS\\goal", "windows purpose"),
        ),
        (
            {"title": "./PRIVATE-RELATIVE/goal", "purpose": "relative purpose"},
            "invalid_title",
            ("./PRIVATE-RELATIVE/goal", "relative purpose"),
        ),
        (
            {"title": "../PRIVATE-PARENT/goal", "purpose": "parent purpose"},
            "invalid_title",
            ("../PRIVATE-PARENT/goal", "parent purpose"),
        ),
        (
            {"title": "Open '~/PRIVATE-QUOTED/goal'", "purpose": "quoted purpose"},
            "invalid_title",
            ("Open '~/PRIVATE-QUOTED/goal'", "quoted purpose"),
        ),
        (
            {"title": "PRIVATE-CREATE-\ud800-GOAL", "purpose": "unicode purpose"},
            "invalid_unicode",
            ("PRIVATE-CREATE-\ud800-GOAL", "unicode purpose"),
        ),
    )
    for index, (invalid_args, reason, private_values) in enumerate(cases):
        with TemporaryDirectory(prefix=f"jarvis-create-goal-preflight-{index}-") as temp:
            runtime = _runtime(Path(temp), invalid_args)
            calls = _install_counted_handler(runtime)
            vault_before = _vault_snapshot(runtime)
            token = f"create-goal-refused-{index}"

            first = runtime.handle("refused create goal", request_token=token)
            same = runtime.handle("same-token refused create goal", request_token=token)
            fresh = runtime.handle(
                "fresh-token refused create goal",
                request_token=f"{token}-fresh",
            )
            for result, suffix in ((first, "first"), (same, "same"), (fresh, "fresh")):
                _assert_refusal(result, reason, f"{reason} {suffix}")
            if (
                calls[0] != 0
                or _receipt_rows(runtime)
                or _goal_rows(runtime)
                or _vault_snapshot(runtime) != vault_before
            ):
                raise SystemExit(f"{reason} preflight crossed a mutation boundary")
            _assert_private_refusal_audits(
                runtime,
                results=(first, same, fresh),
                private_values=private_values,
                expected_runs=3,
                label=reason,
            )
            _assert_no_approvals(runtime, reason)

            corrected = {
                "title": f"Corrected create goal {index}",
                "purpose": f"corrected purpose {index}",
                "horizon": "this month",
            }
            _set_args(runtime, corrected)
            succeeded = runtime.handle("corrected create goal", request_token=token)
            goals = _goal_rows(runtime)
            if (
                not succeeded.tool_results[0].ok
                or calls[0] != 1
                or len(goals) != 1
                or goals[0]["title"] != corrected["title"]
            ):
                raise SystemExit(f"{reason} corrected same-token create failed: {succeeded.tool_results}")
            _assert_completed_receipt_audits(runtime, 1)

            replay = runtime.handle("completed create-goal replay", request_token=token)
            if (
                replay.tool_results[0].metadata.get("failure_kind")
                != "auto_mutation_completed_replay"
                or calls[0] != 1
                or len(_goal_rows(runtime)) != 1
            ):
                raise SystemExit(f"{reason} completed same-token replay did not coalesce")

            _set_args(runtime, corrected)
            repeated = runtime.handle(
                "intentional repeated create goal",
                request_token=f"{token}-intentional-repeat",
            )
            if not repeated.tool_results[0].ok or calls[0] != 2 or len(_goal_rows(runtime)) != 2:
                raise SystemExit(f"{reason} fresh intentional create did not repeat")
            _assert_completed_receipt_audits(runtime, 2)
            _assert_no_approvals(runtime, f"{reason} success/replay")


def test_direct_handler_rejects_invalid_fields() -> None:
    cases = (
        ({"title": 123}, "invalid_type"),
        ({"title": "Typed goal", "purpose": ["not", "text"]}, "invalid_type"),
        ({"title": "Typed goal", "horizon": {"not": "text"}}, "invalid_type"),
        ({"title": "Malformed \ud800 title"}, "invalid_unicode"),
        ({"title": "Typed goal", "purpose": "Malformed \ud800 purpose"}, "invalid_unicode"),
        ({"title": "Typed goal", "horizon": "Malformed \ud800 horizon"}, "invalid_unicode"),
    )
    for index, (args, reason) in enumerate(cases):
        with TemporaryDirectory(prefix=f"jarvis-create-goal-direct-type-{index}-") as temp:
            runtime = _runtime(Path(temp), {"title": "unused"})
            before = _vault_snapshot(runtime)
            result = runtime.registry.get(TOOL_NAME).handler(args)
            if (
                result.ok
                or result.metadata.get("reason") != reason
                or result.metadata.get("state_changed") is not False
                or result.metadata.get("writes_files") is not False
                or result.metadata.get("writes_memory") is not False
                or result.metadata.get("writes_notes") is not False
                or _goal_rows(runtime)
                or _vault_snapshot(runtime) != before
            ):
                raise SystemExit(f"direct create_goal accepted an invalid field: {result}")


def test_uncertain_title_fences_changed_purpose_but_not_different_title() -> None:
    owner_args = {
        "title": "Stable \uff27oal Target",
        "purpose": "PRIVATE-OWNER-PURPOSE",
        "horizon": "this week",
    }
    with TemporaryDirectory(prefix="jarvis-create-goal-uncertain-") as temp:
        runtime = _runtime(Path(temp), owner_args)
        calls = _install_counted_handler(runtime)
        original_publish = runtime.vault.write_goal_with_evidence
        publication_calls = [0]

        def fail_after_database_commit(
            _goal: Any,
            _steps: Any,
            *,
            store_identity: str,
            expected_prior_content_digest: str | None = None,
        ) -> tuple[Path, str, str]:
            publication_calls[0] += 1
            raise OSError("representative create-goal mirror failure")

        runtime.vault.write_goal_with_evidence = fail_after_database_commit  # type: ignore[method-assign]
        owner = runtime.handle(
            "create goal before mirror failure",
            request_token="create-goal-uncertain-owner",
        )
        receipts = _receipt_rows(runtime)
        if (
            owner.tool_results[0].ok
            or owner.tool_results[0].metadata.get("auto_mutation_outcome_uncertain") is not True
            or calls[0] != 1
            or publication_calls[0] != 1
            or len(_goal_rows(runtime)) != 1
            or len(receipts) != 1
            or receipts[0]["state"] != "uncertain"
            or receipts[0]["result"] != "unknown"
        ):
            raise SystemExit(f"create_goal database-before-mirror failure lost uncertainty: {owner.tool_results}")

        _set_args(
            runtime,
            {
                "title": " Stable Goal Target ",
                "purpose": "PRIVATE-CHANGED-PURPOSE",
                "horizon": "next year",
            },
        )
        blocked = runtime.handle(
            "changed purpose for unresolved title",
            request_token="create-goal-uncertain-contender",
        )
        if (
            blocked.tool_results[0].metadata.get("failure_kind")
            != "auto_mutation_unresolved_action"
            or calls[0] != 1
            or publication_calls[0] != 1
            or len(_goal_rows(runtime)) != 1
            or len(_receipt_rows(runtime)) != 1
        ):
            raise SystemExit("changed create-goal purpose bypassed unresolved title custody")

        runtime.vault.write_goal_with_evidence = original_publish  # type: ignore[method-assign]
        _set_args(
            runtime,
            {
                "title": "A separate goal target",
                "purpose": "PRIVATE-SEPARATE-PURPOSE",
                "horizon": "this quarter",
            },
        )
        separate = runtime.handle(
            "create a separate title while another is unresolved",
            request_token="create-goal-separate-target",
        )
        receipts = _receipt_rows(runtime)
        goals = _goal_rows(runtime)
        if (
            not separate.tool_results[0].ok
            or calls[0] != 2
            or len(goals) != 2
            or {row["title"] for row in goals}
            != {"Stable Goal Target", "A separate goal target"}
            or len(receipts) != 2
            or [row["state"] for row in receipts] != ["uncertain", "completed"]
        ):
            raise SystemExit(f"different create-goal title did not remain separate: {goals} / {receipts}")
        successful = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
        if len(successful) != 1 or receipts[1]["tool_run_id"] != successful[0]["id"]:
            raise SystemExit("separate create-goal receipt lost its ordinary audit linkage")
        _assert_no_approvals(runtime, "create-goal uncertainty fencing")


def main() -> None:
    test_exact_contract_and_title_only_identity()
    test_refusals_re_evaluate_and_correct_with_same_token()
    test_direct_handler_rejects_invalid_fields()
    test_uncertain_title_fences_changed_purpose_but_not_different_title()
    print("Auto mutation create-goal preflight smoke passed")


if __name__ == "__main__":
    main()
