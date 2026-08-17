from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.memory.store import TaskRecord, auto_mutation_request_digest
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
    ToolArgumentType,
)


TOOL_NAME = "daily_plan"
TARGET_DATE = "2026-08-08"
OTHER_DATE = "2026-08-09"
PRIVATE_PLAN_CONTENT = "PRIVATE-DAILY-PLAN-CONTENT-84271"


class StaticPlanner:
    def __init__(self, args: Any):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise aggregate-wired daily-plan mutation custody.",
            [PlannedAction(TOOL_NAME, self.args, "daily-plan rollout smoke")],
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


def _note_path(runtime: JarvisRuntime, target_date: str) -> Path:
    return runtime.vault.root_path / "Daily" / f"{target_date}.md"


def _note_text(runtime: JarvisRuntime, target_date: str) -> str:
    path = _note_path(runtime, target_date)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _append_count(runtime: JarvisRuntime, target_date: str) -> int:
    return _note_text(runtime, target_date).count("\n## Jarvis Daily Plan\n")


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
        raise SystemExit(f"daily-plan uncertain receipt drifted: {receipts}")


def test_exact_date_scoped_contract_and_schema() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-contract-") as temp:
        runtime = _runtime(Path(temp), {"target_date": TARGET_DATE})
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
            or not callable(contract.semantic_preflight)
            or contract.semantic_preflight_result_builder is not None
            or contract.definite_no_effect_failure_reasons != frozenset()
            or contract.operation_scope is not None
            or contract.legacy_operation_aliases != frozenset()
        ):
            raise SystemExit(f"{TOOL_NAME} auto-mutation contract drifted: {tool}")
        if contract.operation_key_builder({"target_date": TARGET_DATE}) != {
            "target_date": TARGET_DATE
        }:
            raise SystemExit("daily-plan operation identity lost the exact target date")
        if contract.operation_key_builder({"target_date": OTHER_DATE}) == {
            "target_date": TARGET_DATE
        }:
            raise SystemExit("different daily-plan target dates collapsed to one identity")
        if contract.semantic_preflight({"target_date": TARGET_DATE}) is not None:
            raise SystemExit("valid daily-plan target date failed semantic preflight")
        if contract.semantic_preflight({"target_date": "2026-02-30"}) != "invalid_target_date":
            raise SystemExit("impossible daily-plan date passed semantic preflight")

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
            or shape != (("target_date", string, True),)
        ):
            raise SystemExit(f"{TOOL_NAME} argument contract drifted: {arguments}")


def test_success_replay_repeat_and_exact_bound_destination() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-success-") as temp:
        runtime = _runtime(Path(temp), {"target_date": TARGET_DATE})
        runtime.store.add_task(TaskRecord(PRIVATE_PLAN_CONTENT))
        calls = _install_counted_handler(runtime)
        owner = "PRIVATE-DAILY-PLAN-SUCCESS-OWNER"
        fresh_token = "PRIVATE-DAILY-PLAN-SUCCESS-FRESH"

        first = runtime.handle("render the first daily plan", request_token=owner)
        item = first.tool_results[0]
        if (
            not first.verified
            or not item.ok
            or calls[0] != 1
            or item.metadata.get("target_date") != TARGET_DATE
            or item.metadata.get("path_display") != f"Daily/{TARGET_DATE}.md"
            or f"Daily plan saved: Daily/{TARGET_DATE}.md" not in item.output
            or f"Plan date: {TARGET_DATE}" not in item.output
            or _append_count(runtime, TARGET_DATE) != 1
            or PRIVATE_PLAN_CONTENT not in _note_text(runtime, TARGET_DATE)
        ):
            raise SystemExit(f"first bound daily-plan mutation failed: {item}")
        if any(
            path.name != f"{TARGET_DATE}.md"
            for path in (runtime.vault.root_path / "Daily").glob("*.md")
        ):
            raise SystemExit("bound daily plan wrote a current-date or rollover note")

        same = runtime.handle("coalesce same daily plan", request_token=owner)
        _assert_failure(same, "auto_mutation_completed_replay", "same-token replay")
        if calls[0] != 1 or _append_count(runtime, TARGET_DATE) != 1:
            raise SystemExit("same-token daily-plan replay wrote again")

        fresh = runtime.handle("repeat daily plan intentionally", request_token=fresh_token)
        if not fresh.verified or not fresh.tool_results[0].ok or calls[0] != 2:
            raise SystemExit(f"fresh-token daily plan did not run: {fresh.tool_results}")
        if _append_count(runtime, TARGET_DATE) != 2:
            raise SystemExit("fresh-token daily plan did not append exactly once")

        receipts = _receipt_rows(runtime)
        runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
        if len(receipts) != 2 or len(runs) != 2:
            raise SystemExit(f"daily-plan receipt/audit count drifted: {receipts} / {runs}")
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
                raise SystemExit(f"daily-plan audit linkage drifted: {receipt} / {run}")


def test_invalid_arguments_stop_before_receipt_handler_or_write() -> None:
    cases = (
        ({}, "tool_arguments_invalid"),
        ({"target_date": TARGET_DATE, "unknown": PRIVATE_PLAN_CONTENT}, "tool_arguments_invalid"),
        ({"target_date": 20260808}, "tool_arguments_invalid"),
        ({"target_date": "2026-02-30"}, "auto_mutation_semantic_preflight_rejected"),
        ({"target_date": "２０２６-０８-０８"}, "auto_mutation_semantic_preflight_rejected"),
    )
    for index, (args, failure_kind) in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-daily-plan-invalid-") as temp:
            runtime = _runtime(Path(temp), args)
            calls = _install_counted_handler(runtime)
            before = _vault_snapshot(runtime)
            result = runtime.handle("reject invalid daily plan", request_token=f"invalid-{index}")
            _assert_failure(result, failure_kind, f"invalid case {index}")
            if calls[0] or _receipt_rows(runtime) or _vault_snapshot(runtime) != before:
                raise SystemExit(f"invalid daily-plan case {index} crossed a write boundary")


def test_uncertain_date_fences_only_that_date() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-uncertain-") as temp:
        runtime = _runtime(Path(temp), {"target_date": TARGET_DATE})
        runtime.store.add_task(TaskRecord(PRIVATE_PLAN_CONTENT))
        handler_calls = _install_counted_handler(runtime)
        original_fence = runtime.vault.daily_append_fence
        append_calls = [0]

        @contextmanager
        def fence_then_error(target_date: str):
            with original_fence(target_date) as append:
                def append_then_error(heading: str, body: str) -> Path:
                    append_calls[0] += 1
                    path = append(heading, body)
                    if target_date == TARGET_DATE:
                        raise OSError("representative failure after daily-plan append")
                    return path

                yield append_then_error

        runtime.vault.daily_append_fence = fence_then_error  # type: ignore[method-assign]
        owner = "PRIVATE-DAILY-PLAN-UNCERTAIN-OWNER"
        failed = runtime.handle("append before daily-plan failure", request_token=owner)
        if (
            failed.tool_results[0].metadata.get("auto_mutation_outcome_uncertain") is not True
            or handler_calls[0] != 1
            or append_calls[0] != 1
            or _append_count(runtime, TARGET_DATE) != 1
        ):
            raise SystemExit(f"daily-plan failure lost uncertain custody: {failed.tool_results}")
        _assert_uncertain_receipt(runtime, "manual_review")

        same = runtime.handle("same uncertain replay", request_token=owner)
        _assert_failure(same, "auto_mutation_outcome_uncertain", "same uncertain replay")
        same_date_fresh = runtime.handle(
            "fresh uncertain same-date replay",
            request_token="PRIVATE-DAILY-PLAN-UNCERTAIN-FRESH",
        )
        _assert_failure(
            same_date_fresh,
            "auto_mutation_unresolved_action",
            "fresh same-date uncertain replay",
        )
        if handler_calls[0] != 1 or _append_count(runtime, TARGET_DATE) != 1:
            raise SystemExit("uncertain same-date replay invoked the handler")

        runtime.planner = StaticPlanner({"target_date": OTHER_DATE})
        other = runtime.handle(
            "render a distinct date",
            request_token="PRIVATE-DAILY-PLAN-OTHER-DATE",
        )
        if not other.tool_results[0].ok or handler_calls[0] != 2:
            raise SystemExit(f"uncertain prior date blocked a distinct date: {other.tool_results}")
        if _append_count(runtime, OTHER_DATE) != 1:
            raise SystemExit("distinct daily-plan date was not written exactly once")


def test_stale_running_recovery_remains_non_replayable() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-stale-") as temp:
        root = Path(temp)
        runtime = _runtime(root, {"target_date": TARGET_DATE})
        runtime.store.add_task(TaskRecord(PRIVATE_PLAN_CONTENT))
        tool = runtime.registry.get(TOOL_NAME)

        def append_then_interrupt(args: dict[str, Any]) -> ToolResult:
            tool.handler(args)
            raise KeyboardInterrupt

        runtime.registry._tools[TOOL_NAME] = replace(tool, handler=append_then_interrupt)
        owner = "PRIVATE-DAILY-PLAN-STALE-OWNER"
        try:
            runtime.handle("interrupt after daily-plan append", request_token=owner)
        except KeyboardInterrupt:
            pass
        else:
            raise SystemExit("daily-plan stale-running fixture did not interrupt")
        receipts = _receipt_rows(runtime)
        before = _note_text(runtime, TARGET_DATE)
        if len(receipts) != 1 or receipts[0]["state"] != "running" or not before:
            raise SystemExit(f"daily-plan stale-running fixture drifted: {receipts}")
        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE auto_mutation_receipts SET running_at = ?, updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z", receipts[0]["id"]),
            )

        recovered = _runtime(root, {"target_date": TARGET_DATE})
        _assert_uncertain_receipt(recovered, "stale_recovery")
        calls = _install_counted_handler(recovered)
        same = recovered.handle("same stale replay", request_token=owner)
        fresh = recovered.handle("fresh stale replay", request_token="PRIVATE-STALE-FRESH")
        _assert_failure(same, "auto_mutation_outcome_uncertain", "same stale replay")
        _assert_failure(fresh, "auto_mutation_unresolved_action", "fresh stale replay")
        if calls[0] or _note_text(recovered, TARGET_DATE) != before:
            raise SystemExit("stale daily-plan recovery allowed a duplicate write")


def test_receipt_and_outward_runtime_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-plan-privacy-") as temp:
        runtime = _runtime(Path(temp), {"target_date": TARGET_DATE})
        runtime.store.add_task(TaskRecord(PRIVATE_PLAN_CONTENT))
        request_token = "PRIVATE-DAILY-PLAN-PRIVACY-TOKEN"
        result = runtime.handle("render privacy daily plan", request_token=request_token)
        if not result.tool_results[0].ok or PRIVATE_PLAN_CONTENT not in result.response:
            raise SystemExit("daily-plan privacy fixture did not render its intended plan")

        receipts = _receipt_rows(runtime)
        receipt_text = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
        absolute_paths = (str(Path(temp)), str(runtime.vault.root_path), str(runtime.store.db_path.parent))
        for private in (PRIVATE_PLAN_CONTENT, request_token, *absolute_paths):
            if private and private in receipt_text:
                raise SystemExit(f"daily-plan receipt exposed raw private evidence: {private!r}")

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
                raise SystemExit("daily-plan outward runtime exposed receipt-private evidence")


def main() -> None:
    test_exact_date_scoped_contract_and_schema()
    test_success_replay_repeat_and_exact_bound_destination()
    test_invalid_arguments_stop_before_receipt_handler_or_write()
    test_uncertain_date_fences_only_that_date()
    test_stale_running_recovery_remains_non_replayable()
    test_receipt_and_outward_runtime_privacy()
    print("Daily-plan auto-mutation rollout smoke test passed.")


if __name__ == "__main__":
    main()
