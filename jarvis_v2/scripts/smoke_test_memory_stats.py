from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import Plan, PlannedAction
from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION


def test_planner_routes_memory_stats_aliases() -> None:
    # Real gap found live 2026-07-09: "show memory stats" / "show my memory
    # stats" fell through to chat while bare "memory stats" worked.
    p = RuleBasedPlanner()
    for q in ("memory stats", "show memory stats", "show my memory stats"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["memory_stats"]:
            raise SystemExit(f"memory_stats route missed: {q!r} -> {[a.tool_name for a in actions]}")


def assert_runtime_routes_memory_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-memory-count-alias-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for text in (
            "how many memories do i have",
            "how many memories does jarvis have",
            "count memories",
            "memory count",
            "memories count",
            "기억 몇 개",
            "기억 개수",
            "메모리 몇 개",
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [action.get("tool_name") for action in planned] != ["memory_stats"]:
                raise SystemExit(f"runtime missed memory-count alias {text!r}: {planned} / {result.response!r}")
            if len(result.tool_results) != 1 or result.tool_results[0].tool_name != "memory_stats":
                raise SystemExit(f"memory-count alias should execute exactly one memory_stats tool for {text!r}: {result.tool_results}")
            metadata = result.tool_results[0].metadata
            if "Count: 0" not in result.response or metadata.get("total_memories") != 0:
                raise SystemExit(f"empty memory-count alias should answer with zero count for {text!r}: {result.response!r} / {metadata}")
            for key in (
                "writes_files",
                "writes_memory",
                "writes_notes",
                "queues_approval",
                "controls_computer",
                "reads_private_data",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
            ):
                if metadata.get(key):
                    raise SystemExit(f"memory-count alias unexpectedly set {key} for {text!r}: {metadata}")
        if runtime.store.list_memories(limit=100):
            raise SystemExit("memory count aliases must not create or mutate memories.")


class HostileRow:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-row {self.marker}>"


class _StaticPlanner:
    def __init__(self, args: object) -> None:
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise the memory-stats runtime argument contract.",
            [PlannedAction("memory_stats", self.args, "memory-stats contract smoke")],
            needs_model=False,
        )


def test_memory_stats_invalid_runtime_arguments_stop_before_memory_reads_or_handler() -> None:
    invalid_args = (
        {"limit": 1},
        {"category": "facts"},
        [],
        None,
        "memory stats",
    )
    for index, args in enumerate(invalid_args):
        with TemporaryDirectory(prefix="jarvis-memory-stats-arguments-") as temp:
            runtime = make_temp_runtime(Path(temp))
            tool = runtime.registry.get("memory_stats")
            contract = tool.argument_contract
            if (
                contract is None
                or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
                or contract.allow_unknown
                or contract.fields
            ):
                raise SystemExit(f"memory_stats should accept only an empty argument object: {contract}")
            calls = [0]

            def counted_handler(value: dict[str, object]):
                calls[0] += 1
                return tool.handler(value)

            runtime.registry._tools["memory_stats"] = replace(tool, handler=counted_handler)
            runtime.planner = _StaticPlanner(args)
            with (
                mock.patch.object(
                    runtime.store,
                    "memory_stats",
                    side_effect=AssertionError("invalid memory-stats arguments read memory state"),
                ),
                mock.patch.object(
                    runtime.store,
                    "conversation_compaction_legacy_review",
                    side_effect=AssertionError("invalid memory-stats arguments read legacy review state"),
                ),
            ):
                result = runtime.handle(
                    "reject invalid memory-stats arguments",
                    request_token=f"memory-stats-invalid-{index}",
                )
            if len(result.tool_results) != 1:
                raise SystemExit(f"invalid memory-stats case {index} lost its result")
            item = result.tool_results[0]
            if (
                item.ok
                or item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(
                    f"invalid memory-stats case {index} crossed the memory boundary: {item}"
                )


def test_personal_context_status_is_count_only_and_command_first() -> None:
    with TemporaryDirectory(prefix="jarvis-personal-context-status-") as temp:
        runtime = make_temp_runtime(Path(temp))
        marker = "PERSONAL CONTEXT SECRET MUST NOT APPEAR"
        runtime.store.add_memory(
            MemoryRecord(
                "facts",
                marker,
                marker,
                "personal-context-status-smoke",
            )
        )
        for command in ("personal context status", "개인 컨텍스트 상태"):
            result = runtime.handle(command)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [action.get("tool_name") for action in planned] != ["personal_context_status"]:
                raise SystemExit(f"personal-context status route missed {command!r}: {planned}")
            if len(result.tool_results) != 1 or result.tool_results[0].tool_name != "personal_context_status":
                raise SystemExit(f"personal-context status should execute exactly once for {command!r}: {result.tool_results}")
            item = result.tool_results[0]
            metadata = item.metadata
            if not item.ok or metadata.get("indexed_memories") != 1:
                raise SystemExit(f"personal-context status missed indexed-memory count: {item}")
            if marker in result.response or marker in json.dumps(metadata, sort_keys=True):
                raise SystemExit(f"personal-context status leaked content for {command!r}: {result.response!r}")
            if "Jarvis personal context status:" not in result.response or "indexed memories: 1" not in result.response:
                raise SystemExit(f"personal-context status missed count-only coverage view: {result.response!r}")
            for key in (
                "writes_files",
                "writes_database",
                "writes_memory",
                "writes_notes",
                "queues_approval",
                "requires_approval",
                "controls_computer",
                "calls_external_service",
                "reads_private_data",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
            ):
                if metadata.get(key):
                    raise SystemExit(f"personal-context status unexpectedly set {key} for {command!r}: {metadata}")
            if metadata.get("content_free") is not True or metadata.get("unavailable_components") != []:
                raise SystemExit(f"personal-context status metadata was not bounded and healthy: {metadata}")


def assert_no_future_authority(metadata: dict, label: str) -> None:
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False: {metadata}")


def assert_memory_stats_handoff(metadata: dict, label: str) -> None:
    assert_no_future_authority(metadata, label)
    handoff = metadata.get("memory_stats_handoff")
    if not metadata.get("memory_stats_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready memory_stats_handoff: {metadata}")
    for key in ("ready_for_operator", "content_in_handoff"):
        if handoff.get(key) is not True:
            raise SystemExit(f"{label} handoff should mark {key}=True: {handoff}")
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if handoff.get(key) is not False:
            raise SystemExit(f"{label} handoff should keep {key}=False: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "read_only": True,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} handoff boundary {key} should be {value}: {boundaries}")
    review = handoff.get("conversation_compaction_legacy_review")
    if not isinstance(review, dict):
        raise SystemExit(f"{label} handoff missed conversation compaction legacy review: {handoff}")
    for key in (
        "read_only",
        "ranges_inferred",
        "memory_edited",
        "memory_deleted",
        "repair_performed",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "requires_approval",
        "calls_model",
        "executes_tools",
        "external_side_effect",
        "controls_computer",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        expected_value = key == "read_only"
        if review.get(key) is not expected_value:
            raise SystemExit(f"{label} legacy review boundary {key} should be {expected_value}: {review}")


def main() -> None:
    test_planner_routes_memory_stats_aliases()
    assert_runtime_routes_memory_count_aliases()
    test_memory_stats_invalid_runtime_arguments_stop_before_memory_reads_or_handler()
    test_personal_context_status_is_count_only_and_command_first()
    with TemporaryDirectory(prefix="jarvis-memory-stats-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            "remember that memory stats summarize memory health",
            "add profile note the operator likes concise status summaries.",
            "memory stats",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1200])
            print()
            if case == "memory stats":
                if "Total memories:" not in result.response:
                    raise SystemExit(f"memory_stats should include explicit total-memory count: {result.response!r}")
                if result.tool_results[0].metadata.get("total_memories", 0) < 1:
                    raise SystemExit(f"memory_stats missed total_memories metadata: {result.tool_results[0].metadata}")
                assert_memory_stats_handoff(result.tool_results[0].metadata, case)

        review_id = runtime.store.add_memory(
            MemoryRecord(
                "conversation-digest",
                "Private malformed title /\x55sers/operator SHOULD NOT APPEAR",
                "Private digest body SHOULD NOT APPEAR",
                "conversation_compaction",
                0.8,
            )
        )
        review_result = runtime.handle("memory stats")
        review_metadata = review_result.tool_results[0].metadata
        review = review_metadata.get("conversation_compaction_legacy_review") or {}
        if not review_result.verified or not review.get("available") or review.get("review_required") is not True:
            raise SystemExit(f"memory stats missed required legacy review: {review_result.response!r} / {review_metadata}")
        if review.get("review_memory_ids") != [review_id] or review.get("next_command") != f"show memory {review_id}":
            raise SystemExit(f"memory stats legacy review was not actionable by id: {review!r}")
        for expected_text in (
            "Conversation compaction legacy review (limit 20):",
            "- unparseable_rows: 1",
            f"- review_memory_ids: {review_id}",
            f"- Manual next command: `show memory {review_id}`.",
            "No ranges are inferred. No memory is edited or deleted. No repair is performed.",
        ):
            if expected_text not in review_result.response:
                raise SystemExit(f"memory stats legacy review missed {expected_text!r}: {review_result.response!r}")
        review_text = review_result.response + json.dumps(review_metadata, sort_keys=True)
        for forbidden in ("SHOULD NOT APPEAR", "/\x55sers/operator", "Private digest body", "Private malformed title"):
            if forbidden in review_text:
                raise SystemExit(f"memory stats legacy review leaked content {forbidden!r}: {review_text}")
        assert_memory_stats_handoff(review_metadata, "legacy review memory stats")

        original_legacy_review = runtime.store.conversation_compaction_legacy_review
        runtime.store.conversation_compaction_legacy_review = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("/\x55sers/example/private/review SHOULD NOT APPEAR")
        )
        try:
            unavailable = runtime.registry.get("memory_stats").handler({})
        finally:
            runtime.store.conversation_compaction_legacy_review = original_legacy_review
        if not unavailable.ok or "Conversation compaction legacy review: unavailable." not in unavailable.output:
            raise SystemExit(f"memory stats should keep working when legacy review is unavailable: {unavailable}")
        unavailable_text = unavailable.output + json.dumps(unavailable.metadata, sort_keys=True)
        for forbidden in ("SHOULD NOT APPEAR", "/\x55sers/operator", "RuntimeError"):
            if forbidden in unavailable_text:
                raise SystemExit(f"unavailable legacy review leaked exception detail {forbidden!r}: {unavailable_text}")

        runtime.store.conversation_compaction_legacy_review = lambda **kwargs: {
            "unexpected_private_key": "/private/tmp/SHOULD NOT APPEAR"
        }
        try:
            malformed_review = runtime.registry.get("memory_stats").handler({})
        finally:
            runtime.store.conversation_compaction_legacy_review = original_legacy_review
        malformed_text = malformed_review.output + json.dumps(malformed_review.metadata, sort_keys=True)
        if "Conversation compaction legacy review: unavailable." not in malformed_review.output:
            raise SystemExit(f"malformed legacy review should fail closed: {malformed_review}")
        for forbidden in ("SHOULD NOT APPEAR", "/private/tmp", "unexpected_private_key"):
            if forbidden in malformed_text:
                raise SystemExit(f"malformed legacy review leaked payload detail {forbidden!r}: {malformed_text}")

        marker = "MEMORY_STATS_HOSTILE_ROW_SHOULD_NOT_LEAK /\x55sers/example/private/memory.db"
        original_memory_stats = runtime.store.memory_stats
        runtime.store.memory_stats = lambda: [
            HostileRow(marker),
            {
                "category": "facts",
                "source": "manual",
                "count": 2,
                "latest": "/private/tmp/memory-stats-latest.md",
            },
        ]
        try:
            hostile = runtime.registry.get("memory_stats").handler({})
        finally:
            runtime.store.memory_stats = original_memory_stats
        if not hostile.ok:
            raise SystemExit(f"memory_stats should tolerate malformed stat rows: {hostile.output}")
        for expected in [
            "hidden malformed memory stats rows: 1",
            "facts / manual: 2 latest <local-path>",
        ]:
            if expected not in hostile.output:
                raise SystemExit(f"memory_stats missed hostile-row fragment {expected!r}: {hostile.output}")
        leak_text = hostile.output + json.dumps(hostile.metadata, sort_keys=True)
        for leaked in ["MEMORY_STATS_HOSTILE_ROW_SHOULD_NOT_LEAK", "/\x55sers/example/private", "memory.db", "/private/tmp/memory-stats-latest.md"]:
            if leaked in leak_text:
                raise SystemExit(f"memory_stats leaked hostile row detail {leaked!r}: {leak_text}")
        if hostile.metadata.get("count") != 1:
            raise SystemExit(f"memory_stats should count readable stat rows only: {hostile.metadata}")
        if hostile.metadata.get("readable_memory_stat_rows") != 1 or hostile.metadata.get("unreadable_memory_stat_rows") != 1:
            raise SystemExit(f"memory_stats missed readable/unreadable counters: {hostile.metadata}")
        handoff = hostile.metadata.get("memory_stats_handoff") or {}
        if handoff.get("count") != 1 or handoff.get("total_memories") != 2:
            raise SystemExit(f"memory_stats handoff should preserve readable stats only: {handoff}")
        assert_memory_stats_handoff(hostile.metadata, "hostile memory stats")


if __name__ == "__main__":
    main()
