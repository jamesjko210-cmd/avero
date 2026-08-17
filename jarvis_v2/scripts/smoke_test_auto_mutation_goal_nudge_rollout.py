from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.memory.store import GoalRecord, auto_mutation_request_digest
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    ToolArgumentType,
)


TOOL_NAME = "goal_nudge"
PRIVATE_GOAL_TITLE = "PRIVATE-GOAL-NUDGE-CONTENT-72149"
UPDATED_GOAL_TITLE = "PRIVATE-GOAL-NUDGE-UPDATED-38126"


class StaticPlanner:
    def __init__(self, args: Any):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise aggregate-wired goal-nudge mutation custody.",
            [PlannedAction(TOOL_NAME, self.args, "goal-nudge rollout smoke")],
            needs_model=False,
        )


def _runtime(root: Path, args: Any) -> JarvisRuntime:
    runtime = make_temp_runtime(root)
    runtime.planner = StaticPlanner(args)
    return runtime


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
    return _rows(
        runtime,
        "SELECT * FROM tool_runs WHERE tool_name = ? ORDER BY id",
        (TOOL_NAME,),
    )


def _daily_note_text(runtime: JarvisRuntime) -> str:
    daily = runtime.vault.root_path / "Daily"
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(daily.glob("*.md"))
        if path.is_file()
    )


def _append_count(runtime: JarvisRuntime) -> int:
    return _daily_note_text(runtime).count("\n## Jarvis Goal Nudge\n")


def _vault_snapshot(runtime: JarvisRuntime) -> dict[str, bytes]:
    return {
        str(path.relative_to(runtime.vault.root_path)): path.read_bytes()
        for path in runtime.vault.root_path.rglob("*")
        if path.is_file()
    }


def _seed_stale_goal(runtime: JarvisRuntime) -> int:
    goal_id = runtime.store.create_goal(GoalRecord(PRIVATE_GOAL_TITLE))
    runtime.store.add_goal_step(goal_id, "PRIVATE-GOAL-NUDGE-NEXT-STEP")
    with runtime.store.connect() as conn:
        conn.execute(
            "UPDATE goals SET updated_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00Z", goal_id),
        )
    return goal_id


def _install_counted_handler(runtime: JarvisRuntime) -> list[int]:
    tool = runtime.registry.get(TOOL_NAME)
    calls = [0]

    def counted(args: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        return tool.handler(args)

    runtime.registry._tools[TOOL_NAME] = replace(tool, handler=counted)
    return calls


def _assert_failure(result: Any, failure_kind: str, label: str) -> None:
    if len(result.tool_results) != 1:
        raise SystemExit(f"{label} did not return one result: {result.tool_results}")
    item = result.tool_results[0]
    if item.ok or item.metadata.get("failure_kind") != failure_kind:
        raise SystemExit(f"{label} returned the wrong failure: {item}")


def _assert_uncertain_receipt(runtime: JarvisRuntime, resolution: str) -> None:
    receipts = _receipt_rows(runtime)
    if (
        len(receipts) != 1
        or receipts[0]["state"] != "uncertain"
        or receipts[0]["result"] != "unknown"
        or receipts[0]["resolution"] != resolution
        or receipts[0]["tool_run_id"] is not None
    ):
        raise SystemExit(f"goal-nudge uncertain receipt drifted: {receipts}")


def test_exact_contract_and_argument_schema() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-contract-") as temp:
        runtime = _runtime(Path(temp), {})
        tool = runtime.registry.get(TOOL_NAME)
        contract = tool.auto_mutation_contract
        if (
            tool.risk is not RiskLevel.LOCAL_SAFE
            or tool.toolset != "proactive"
            or contract is None
            or contract.version != AUTO_MUTATION_CONTRACT_VERSION
            or contract.effects != frozenset({AutoMutationEffect.OBSIDIAN_VAULT})
            or contract.replay_policy is not AutoMutationReplayPolicy.COALESCE_BY_OPERATION_KEY
            or contract.crash_policy is not AutoMutationCrashPolicy.STOP_AS_OUTCOME_UNKNOWN
            or not callable(contract.operation_key_builder)
            or contract.semantic_preflight is not None
            or contract.semantic_preflight_result_builder is not None
            or contract.definite_no_effect_failure_reasons != frozenset()
            or contract.operation_scope is not None
            or contract.legacy_operation_aliases != frozenset()
        ):
            raise SystemExit(f"{TOOL_NAME} auto-mutation contract drifted: {tool}")
        for scheduler_tool_name in ("schedule_goal_nudge", "run_job_now"):
            if runtime.registry.get(scheduler_tool_name).auto_mutation_contract is not None:
                raise SystemExit(
                    f"interactive receipt custody leaked into {scheduler_tool_name}"
                )
        identities = (
            contract.operation_key_builder({}),
            contract.operation_key_builder({"stale_days": 7}),
            contract.operation_key_builder({"stale_days": 365}),
        )
        if identities != ({"projection": TOOL_NAME},) * 3:
            raise SystemExit(f"goal-nudge destination identity drifted: {identities}")

        arguments = tool.argument_contract
        integer = frozenset({ToolArgumentType.INTEGER})
        shape = (
            tuple(
                (
                    field.name,
                    field.types,
                    field.required,
                    field.minimum,
                    field.maximum,
                )
                for field in arguments.fields
            )
            if arguments is not None
            else ()
        )
        if (
            arguments is None
            or arguments.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or arguments.allow_unknown
            or shape != (("stale_days", integer, False, 1, 365),)
        ):
            raise SystemExit(f"{TOOL_NAME} argument contract drifted: {arguments}")


def test_success_replay_repeat_and_audit_linkage() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-success-") as temp:
        runtime = _runtime(Path(temp), {})
        goal_id = _seed_stale_goal(runtime)
        first_source_digest = runtime.store.read_goal_nudge_snapshot(
            stale_days=7
        ).source_manifest_digest
        calls = _install_counted_handler(runtime)
        owner = "PRIVATE-GOAL-NUDGE-SUCCESS-OWNER"

        first = runtime.handle("render the first goal nudge", request_token=owner)
        item = first.tool_results[0]
        expected_date = datetime.now().strftime("%Y-%m-%d")
        if (
            not first.verified
            or not item.ok
            or calls[0] != 1
            or item.metadata.get("stale_days") != 7
            or item.metadata.get("target_date") != expected_date
            or item.metadata.get("path_display") != f"Daily/{expected_date}.md"
            or PRIVATE_GOAL_TITLE not in item.output
            or _append_count(runtime) != 1
        ):
            raise SystemExit(f"first goal-nudge mutation failed: {item}")

        same = runtime.handle("coalesce the same goal nudge", request_token=owner)
        _assert_failure(same, "auto_mutation_completed_replay", "same-token replay")
        if calls[0] != 1 or _append_count(runtime) != 1:
            raise SystemExit("same-token goal-nudge replay wrote again")

        runtime.planner = StaticPlanner({"stale_days": 7})
        canonical_collision = runtime.handle(
            "reuse the owner token with explicit default arguments",
            request_token=owner,
        )
        _assert_failure(
            canonical_collision,
            "auto_mutation_request_collision",
            "same-token explicit-default collision",
        )
        if calls[0] != 1 or _append_count(runtime) != 1:
            raise SystemExit("same-token argument-shape collision crossed the handler")

        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE goals SET title = ?, updated_at = ? WHERE id = ?",
                (UPDATED_GOAL_TITLE, "2000-01-01T00:00:00Z", goal_id),
            )
        refreshed_source_digest = runtime.store.read_goal_nudge_snapshot(
            stale_days=365
        ).source_manifest_digest
        if refreshed_source_digest == first_source_digest:
            raise SystemExit("goal-nudge source digest ignored changed source evidence")

        runtime.planner = StaticPlanner({"stale_days": 365})
        fresh = runtime.handle(
            "repeat goal nudge intentionally",
            request_token="PRIVATE-GOAL-NUDGE-SUCCESS-FRESH",
        )
        if (
            not fresh.verified
            or not fresh.tool_results[0].ok
            or calls[0] != 2
            or UPDATED_GOAL_TITLE not in fresh.tool_results[0].output
        ):
            raise SystemExit(f"fresh goal-nudge request did not run: {fresh.tool_results}")
        if _append_count(runtime) != 2:
            raise SystemExit("fresh goal-nudge request did not append exactly once")

        receipts = _receipt_rows(runtime)
        runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
        if len(receipts) != 2 or len(runs) != 2:
            raise SystemExit(f"goal-nudge receipt/audit count drifted: {receipts} / {runs}")
        runs_by_id = {row["id"]: row for row in runs}
        for receipt in receipts:
            run = runs_by_id.get(receipt["tool_run_id"])
            if (
                receipt["state"] != "completed"
                or receipt["result"] != "succeeded"
                or receipt["resolution"] != "recorded"
                or run is None
                or run["risk"] != "LOCAL_SAFE"
                or run["approved"] != 0
            ):
                raise SystemExit(f"goal-nudge audit linkage drifted: {receipt} / {run}")


def test_invalid_arguments_stop_before_receipt_handler_or_write() -> None:
    cases: tuple[Any, ...] = (
        {"stale_days": True},
        {"stale_days": "7"},
        {"stale_days": 1.5},
        {"stale_days": 0},
        {"stale_days": 366},
        {"stale_days": None},
        {"stale_days": []},
        {"stale_days": 7, "unknown": PRIVATE_GOAL_TITLE},
        [7],
    )
    for index, args in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-goal-nudge-invalid-") as temp:
            runtime = _runtime(Path(temp), args)
            calls = _install_counted_handler(runtime)
            before = _vault_snapshot(runtime)
            result = runtime.handle(
                "reject invalid goal nudge",
                request_token=f"goal-nudge-invalid-{index}",
            )
            _assert_failure(result, "tool_arguments_invalid", f"invalid case {index}")
            if calls[0] or _receipt_rows(runtime) or _vault_snapshot(runtime) != before:
                raise SystemExit(f"invalid goal-nudge case {index} crossed a write boundary")


def test_post_publication_uncertainty_fences_every_direct_nudge() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-uncertain-") as temp:
        runtime = _runtime(Path(temp), {"stale_days": 7})
        _seed_stale_goal(runtime)
        calls = _install_counted_handler(runtime)
        original_append = runtime.vault.append_daily_for_date
        append_calls = [0]

        def append_then_error(target_date: str, heading: str, body: str) -> Path:
            append_calls[0] += 1
            path = original_append(target_date, heading, body)
            raise OSError("representative failure after goal-nudge append")

        runtime.vault.append_daily_for_date = append_then_error  # type: ignore[method-assign]
        owner = "PRIVATE-GOAL-NUDGE-UNCERTAIN-OWNER"
        failed = runtime.handle("append before goal-nudge failure", request_token=owner)
        if (
            failed.tool_results[0].metadata.get("auto_mutation_outcome_uncertain") is not True
            or calls[0] != 1
            or append_calls[0] != 1
            or _append_count(runtime) != 1
        ):
            raise SystemExit(f"goal-nudge failure lost uncertain custody: {failed.tool_results}")
        _assert_uncertain_receipt(runtime, "manual_review")

        same = runtime.handle("same uncertain replay", request_token=owner)
        _assert_failure(same, "auto_mutation_outcome_uncertain", "same uncertain replay")
        fresh = runtime.handle(
            "fresh uncertain replay",
            request_token="PRIVATE-GOAL-NUDGE-UNCERTAIN-FRESH",
        )
        _assert_failure(fresh, "auto_mutation_unresolved_action", "fresh uncertain replay")
        runtime.planner = StaticPlanner({"stale_days": 365})
        collision = runtime.handle(
            "same-owner changed-window uncertain replay",
            request_token=owner,
        )
        _assert_failure(
            collision,
            "auto_mutation_request_collision",
            "same-owner changed-window uncertain replay",
        )
        changed = runtime.handle(
            "changed-window uncertain replay",
            request_token="PRIVATE-GOAL-NUDGE-UNCERTAIN-CHANGED",
        )
        _assert_failure(
            changed,
            "auto_mutation_unresolved_action",
            "changed-window uncertain replay",
        )
        if calls[0] != 1 or _append_count(runtime) != 1:
            raise SystemExit("unresolved goal-nudge replay invoked the handler")


def test_direct_nudge_binds_append_and_receipt_path_to_one_date() -> None:
    class FixedDatetime:
        @classmethod
        def now(cls) -> datetime:
            return datetime(2099, 12, 31, 23, 59, 59)

    with TemporaryDirectory(prefix="jarvis-goal-nudge-date-binding-") as temp:
        runtime = _runtime(Path(temp), {})
        _seed_stale_goal(runtime)
        with patch("jarvis_v2.tools.proactive.datetime", FixedDatetime):
            result = runtime.handle(
                "bind goal nudge to one date",
                request_token="PRIVATE-GOAL-NUDGE-DATE-BINDING",
            )
        item = result.tool_results[0]
        expected_date = "2099-12-31"
        expected_path = runtime.vault.root_path / "Daily" / f"{expected_date}.md"
        daily_paths = sorted(
            path.relative_to(runtime.vault.root_path).as_posix()
            for path in (runtime.vault.root_path / "Daily").glob("*.md")
        )
        if (
            not result.verified
            or not item.ok
            or item.metadata.get("target_date") != expected_date
            or item.metadata.get("path_display") != f"Daily/{expected_date}.md"
            or not expected_path.is_file()
            or daily_paths != [f"Daily/{expected_date}.md"]
            or PRIVATE_GOAL_TITLE not in expected_path.read_text(encoding="utf-8")
        ):
            raise SystemExit(
                "direct goal nudge split its append and receipt path dates: "
                f"{item} / {daily_paths}"
            )


def test_stale_running_recovery_remains_non_replayable() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-stale-") as temp:
        root = Path(temp)
        runtime = _runtime(root, {})
        _seed_stale_goal(runtime)
        tool = runtime.registry.get(TOOL_NAME)

        def append_then_interrupt(args: dict[str, Any]) -> ToolResult:
            tool.handler(args)
            raise KeyboardInterrupt

        runtime.registry._tools[TOOL_NAME] = replace(tool, handler=append_then_interrupt)
        owner = "PRIVATE-GOAL-NUDGE-STALE-OWNER"
        try:
            runtime.handle("interrupt after goal-nudge append", request_token=owner)
        except KeyboardInterrupt:
            pass
        else:
            raise SystemExit("goal-nudge stale-running fixture did not interrupt")
        receipts = _receipt_rows(runtime)
        before = _daily_note_text(runtime)
        if len(receipts) != 1 or receipts[0]["state"] != "running" or not before:
            raise SystemExit(f"goal-nudge stale-running fixture drifted: {receipts}")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET running_at = ?, updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z", receipts[0]["id"]),
            )

        recovered = _runtime(root, {})
        _assert_uncertain_receipt(recovered, "stale_recovery")
        calls = _install_counted_handler(recovered)
        same = recovered.handle("same stale replay", request_token=owner)
        fresh = recovered.handle(
            "fresh stale replay",
            request_token="PRIVATE-GOAL-NUDGE-STALE-FRESH",
        )
        _assert_failure(same, "auto_mutation_outcome_uncertain", "same stale replay")
        _assert_failure(fresh, "auto_mutation_unresolved_action", "fresh stale replay")
        if calls[0] or _daily_note_text(recovered) != before:
            raise SystemExit("stale goal-nudge recovery allowed a duplicate write")


def test_receipt_and_outward_runtime_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-nudge-privacy-") as temp:
        runtime = _runtime(Path(temp), {"stale_days": 7})
        _seed_stale_goal(runtime)
        request_token = "PRIVATE-GOAL-NUDGE-PRIVACY-TOKEN"
        result = runtime.handle("render privacy goal nudge", request_token=request_token)
        if not result.tool_results[0].ok or PRIVATE_GOAL_TITLE not in result.response:
            raise SystemExit("goal-nudge privacy fixture did not render its intended nudge")

        receipts = _receipt_rows(runtime)
        receipt_text = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
        absolute_paths = (
            str(Path(temp)),
            str(runtime.vault.root_path),
            str(runtime.store.db_path.parent),
        )
        for private in (PRIVATE_GOAL_TITLE, request_token, *absolute_paths):
            if private and private in receipt_text:
                raise SystemExit(f"goal-nudge receipt exposed raw private evidence: {private!r}")

        digest_values = [auto_mutation_request_digest(request_token)]
        digest_values.extend(
            str(row[key])
            for row in receipts
            for key in ("request_digest", "action_digest", "operation_digest", "run_token")
            if row.get(key)
        )
        outward = json.dumps(
            {"response": result.response, "metadata": result.metadata},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        for private in (request_token, *digest_values, *absolute_paths):
            if private and private in outward:
                raise SystemExit("goal-nudge outward runtime exposed receipt-private evidence")


def main() -> None:
    test_exact_contract_and_argument_schema()
    test_success_replay_repeat_and_audit_linkage()
    test_invalid_arguments_stop_before_receipt_handler_or_write()
    test_post_publication_uncertainty_fences_every_direct_nudge()
    test_direct_nudge_binds_append_and_receipt_path_to_one_date()
    test_stale_running_recovery_remains_non_replayable()
    test_receipt_and_outward_runtime_privacy()
    print("Goal-nudge auto-mutation rollout smoke test passed.")


if __name__ == "__main__":
    main()
