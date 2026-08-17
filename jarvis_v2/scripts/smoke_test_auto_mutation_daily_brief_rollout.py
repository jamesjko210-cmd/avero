from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.memory.store import MemoryRecord, auto_mutation_request_digest
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import (
    AUTO_MUTATION_CONTRACT_VERSION,
    TOOL_ARGUMENT_CONTRACT_VERSION,
    AutoMutationCrashPolicy,
    AutoMutationEffect,
    AutoMutationReplayPolicy,
)


TOOL_NAME = "daily_brief"
PRIVATE_MEMORY_TITLE = "PRIVATE-DAILY-BRIEF-CONTENT-72149"
UPDATED_MEMORY_TITLE = "PRIVATE-DAILY-BRIEF-UPDATED-38126"


class StaticPlanner:
    def __init__(self, args: Any):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise aggregate-wired daily-brief mutation custody.",
            [PlannedAction(TOOL_NAME, self.args, "daily-brief rollout smoke")],
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
    return _daily_note_text(runtime).count("\n## Jarvis Proactive Brief\n")


def _vault_snapshot(runtime: JarvisRuntime) -> dict[str, bytes]:
    return {
        str(path.relative_to(runtime.vault.root_path)): path.read_bytes()
        for path in runtime.vault.root_path.rglob("*")
        if path.is_file()
    }


def _seed_memory(runtime: JarvisRuntime) -> int:
    return runtime.store.add_memory(
        MemoryRecord("daily-brief", PRIVATE_MEMORY_TITLE, "PRIVATE-DAILY-BRIEF-BODY")
    )


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
        raise SystemExit(f"daily-brief uncertain receipt drifted: {receipts}")


def test_exact_contract_and_argument_schema() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-brief-contract-") as temp:
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
            or contract.operation_key_builder({}) != {"projection": TOOL_NAME}
        ):
            raise SystemExit(f"{TOOL_NAME} auto-mutation contract drifted: {tool}")
        for other in ("schedule_daily_brief", "run_job_now", "daily_briefing"):
            if runtime.registry.get(other).auto_mutation_contract is not None:
                raise SystemExit(f"interactive receipt custody leaked into {other}")

        arguments = tool.argument_contract
        if (
            arguments is None
            or arguments.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or arguments.allow_unknown
            or arguments.fields
        ):
            raise SystemExit(f"{TOOL_NAME} argument contract drifted: {arguments}")


def test_success_replay_repeat_and_audit_linkage() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-brief-success-") as temp:
        runtime = _runtime(Path(temp), {})
        memory_id = _seed_memory(runtime)
        calls = _install_counted_handler(runtime)
        owner = "PRIVATE-DAILY-BRIEF-SUCCESS-OWNER"

        first = runtime.handle("render the first daily brief", request_token=owner)
        item = first.tool_results[0]
        expected_date = datetime.now().strftime("%Y-%m-%d")
        if (
            not first.verified
            or not item.ok
            or calls[0] != 1
            or item.metadata.get("target_date") != expected_date
            or item.metadata.get("path_display") != f"Daily/{expected_date}.md"
            or PRIVATE_MEMORY_TITLE not in item.output
            or _append_count(runtime) != 1
        ):
            raise SystemExit(f"first daily-brief mutation failed: {item}")

        same = runtime.handle("coalesce the same daily brief", request_token=owner)
        _assert_failure(same, "auto_mutation_completed_replay", "same-token replay")
        if calls[0] != 1 or _append_count(runtime) != 1:
            raise SystemExit("same-token daily-brief replay wrote again")

        with runtime.store.connect() as conn:
            conn.execute(
                "UPDATE memories SET title = ?, body = ?, updated_at = ? WHERE id = ?",
                (UPDATED_MEMORY_TITLE, "UPDATED-DAILY-BRIEF-BODY", "2099-01-01T00:00:00Z", memory_id),
            )
        fresh = runtime.handle(
            "repeat daily brief intentionally",
            request_token="PRIVATE-DAILY-BRIEF-SUCCESS-FRESH",
        )
        if (
            not fresh.verified
            or not fresh.tool_results[0].ok
            or calls[0] != 2
            or UPDATED_MEMORY_TITLE not in fresh.tool_results[0].output
            or _append_count(runtime) != 2
        ):
            raise SystemExit(f"fresh daily-brief request did not run: {fresh.tool_results}")

        receipts = _receipt_rows(runtime)
        runs = [row for row in _tool_run_rows(runtime) if row["ok"] == 1]
        if len(receipts) != 2 or len(runs) != 2:
            raise SystemExit(f"daily-brief receipt/audit count drifted: {receipts} / {runs}")
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
                raise SystemExit(f"daily-brief audit linkage drifted: {receipt} / {run}")


def test_invalid_arguments_stop_before_receipt_handler_or_write() -> None:
    cases: tuple[Any, ...] = (
        {"unknown": PRIVATE_MEMORY_TITLE},
        {"target_date": "2042-08-15"},
        {"stale_days": 7},
        [],
        None,
        "daily brief",
    )
    for index, args in enumerate(cases):
        with TemporaryDirectory(prefix="jarvis-daily-brief-invalid-") as temp:
            runtime = _runtime(Path(temp), args)
            calls = _install_counted_handler(runtime)
            before = _vault_snapshot(runtime)
            result = runtime.handle(
                "reject invalid daily brief",
                request_token=f"daily-brief-invalid-{index}",
            )
            _assert_failure(result, "tool_arguments_invalid", f"invalid case {index}")
            if calls[0] or _receipt_rows(runtime) or _vault_snapshot(runtime) != before:
                raise SystemExit(f"invalid daily-brief case {index} crossed a write boundary")


def test_post_publication_uncertainty_fences_every_direct_brief() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-brief-uncertain-") as temp:
        runtime = _runtime(Path(temp), {})
        _seed_memory(runtime)
        calls = _install_counted_handler(runtime)
        original_fence = runtime.vault.daily_append_fence
        append_calls = [0]

        @contextmanager
        def append_then_error_fence(target_date: str):
            with original_fence(target_date) as append:
                def append_then_error(heading: str, body: str) -> Path:
                    append_calls[0] += 1
                    path = append(heading, body)
                    raise OSError("representative failure after daily-brief append")

                yield append_then_error

        runtime.vault.daily_append_fence = append_then_error_fence  # type: ignore[method-assign]
        owner = "PRIVATE-DAILY-BRIEF-UNCERTAIN-OWNER"
        failed = runtime.handle("append before daily-brief failure", request_token=owner)
        if (
            failed.tool_results[0].metadata.get("auto_mutation_outcome_uncertain") is not True
            or calls[0] != 1
            or append_calls[0] != 1
            or _append_count(runtime) != 1
        ):
            raise SystemExit(f"daily-brief failure lost uncertain custody: {failed.tool_results}")
        _assert_uncertain_receipt(runtime, "manual_review")

        same = runtime.handle("same uncertain replay", request_token=owner)
        _assert_failure(same, "auto_mutation_outcome_uncertain", "same uncertain replay")
        fresh = runtime.handle(
            "fresh uncertain replay",
            request_token="PRIVATE-DAILY-BRIEF-UNCERTAIN-FRESH",
        )
        _assert_failure(fresh, "auto_mutation_unresolved_action", "fresh uncertain replay")
        if calls[0] != 1 or _append_count(runtime) != 1:
            raise SystemExit("unresolved daily-brief replay invoked the handler")


def test_direct_brief_binds_append_and_receipt_path_to_one_date() -> None:
    class FixedDatetime:
        @classmethod
        def now(cls) -> datetime:
            return datetime(2099, 12, 31, 23, 59, 59)

    with TemporaryDirectory(prefix="jarvis-daily-brief-date-binding-") as temp:
        runtime = _runtime(Path(temp), {})
        _seed_memory(runtime)
        with patch("jarvis_v2.tools.proactive.datetime", FixedDatetime):
            result = runtime.handle(
                "bind daily brief to one date",
                request_token="PRIVATE-DAILY-BRIEF-DATE-BINDING",
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
            or PRIVATE_MEMORY_TITLE not in expected_path.read_text(encoding="utf-8")
        ):
            raise SystemExit(
                "direct daily brief split its append and receipt path dates: "
                f"{item} / {daily_paths}"
            )


def test_stale_running_recovery_remains_non_replayable() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-brief-stale-") as temp:
        root = Path(temp)
        runtime = _runtime(root, {})
        _seed_memory(runtime)
        tool = runtime.registry.get(TOOL_NAME)

        def append_then_interrupt(args: dict[str, Any]) -> ToolResult:
            tool.handler(args)
            raise KeyboardInterrupt

        runtime.registry._tools[TOOL_NAME] = replace(tool, handler=append_then_interrupt)
        owner = "PRIVATE-DAILY-BRIEF-STALE-OWNER"
        try:
            runtime.handle("interrupt after daily-brief append", request_token=owner)
        except KeyboardInterrupt:
            pass
        else:
            raise SystemExit("daily-brief stale-running fixture did not interrupt")
        receipts = _receipt_rows(runtime)
        before = _daily_note_text(runtime)
        if len(receipts) != 1 or receipts[0]["state"] != "running" or not before:
            raise SystemExit(f"daily-brief stale-running fixture drifted: {receipts}")
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
            request_token="PRIVATE-DAILY-BRIEF-STALE-FRESH",
        )
        _assert_failure(same, "auto_mutation_outcome_uncertain", "same stale replay")
        _assert_failure(fresh, "auto_mutation_unresolved_action", "fresh stale replay")
        if calls[0] or _daily_note_text(recovered) != before:
            raise SystemExit("stale daily-brief recovery allowed a duplicate write")


def test_receipt_and_outward_runtime_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-daily-brief-privacy-") as temp:
        runtime = _runtime(Path(temp), {})
        _seed_memory(runtime)
        request_token = "PRIVATE-DAILY-BRIEF-PRIVACY-TOKEN"
        result = runtime.handle("render privacy daily brief", request_token=request_token)
        if not result.tool_results[0].ok or PRIVATE_MEMORY_TITLE not in result.response:
            raise SystemExit("daily-brief privacy fixture did not render its intended source")

        receipts = _receipt_rows(runtime)
        receipt_text = json.dumps(receipts, ensure_ascii=False, sort_keys=True, default=str)
        absolute_paths = (
            str(Path(temp)),
            str(runtime.vault.root_path),
            str(runtime.store.db_path.parent),
        )
        for private in (PRIVATE_MEMORY_TITLE, request_token, *absolute_paths):
            if private and private in receipt_text:
                raise SystemExit(f"daily-brief receipt exposed raw private evidence: {private!r}")

        digests = [auto_mutation_request_digest(request_token)]
        digests.extend(
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
        for private in (request_token, *digests, *absolute_paths):
            if private and private in outward:
                raise SystemExit("daily-brief outward runtime exposed receipt-private evidence")


def main() -> None:
    test_exact_contract_and_argument_schema()
    test_success_replay_repeat_and_audit_linkage()
    test_invalid_arguments_stop_before_receipt_handler_or_write()
    test_post_publication_uncertainty_fences_every_direct_brief()
    test_direct_brief_binds_append_and_receipt_path_to_one_date()
    test_stale_running_recovery_remains_non_replayable()
    test_receipt_and_outward_runtime_privacy()
    print("Daily-brief auto-mutation rollout smoke test passed.")


if __name__ == "__main__":
    main()
