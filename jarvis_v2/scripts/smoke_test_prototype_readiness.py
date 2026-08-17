from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.automations.compaction import COMPACTION_JOB_NAME, COMPACTION_JOB_TYPE
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def test_planner_routes_prototype_readiness_aliases() -> None:
    # Real gap found live 2026-07-09: "prototype readiness checklist" (the
    # combined phrase) fell through to chat while "prototype readiness" and
    # "show prototype readiness" both worked.
    p = RuleBasedPlanner()
    for q in ("prototype readiness", "prototype readiness checklist", "show prototype readiness"):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["prototype_readiness_checklist"]:
            raise SystemExit(f"prototype_readiness_checklist route missed: {q!r} -> {[a.tool_name for a in actions]}")


def main() -> None:
    test_planner_routes_prototype_readiness_aliases()
    with TemporaryDirectory(prefix="jarvis-prototype-readiness-") as temp:
        runtime = make_temp_runtime(Path(temp))
        result = runtime.handle("prototype readiness")
        if not result.verified:
            raise SystemExit("prototype readiness route was not verified")
        if "Jarvis prototype readiness checklist" not in result.response:
            raise SystemExit("prototype readiness output missing checklist heading")
        if "Safe to try now:" not in result.response:
            raise SystemExit("prototype readiness output missing safe-to-try section")
        if "Approval-gated before real execution:" not in result.response:
            raise SystemExit("prototype readiness output missing approval-gated section")
        if "Priority goals do not override the operator's explicit stop times" not in result.response:
            raise SystemExit("prototype readiness output missing operator-limit section")
        if "This checklist is read-only" not in result.response:
            raise SystemExit("prototype readiness output missing safety boundary")
        for expected in [
            "setup check",
            "voice setup check",
            "computer control status",
            "computer task plan:",
            "approval readiness",
        ]:
            if expected not in result.response:
                raise SystemExit(f"prototype readiness output missing safe command or approval guidance: {expected}")

        direct = runtime.registry.get("prototype_readiness_checklist").handler({})
        if not direct.ok:
            raise SystemExit("prototype readiness tool failed")
        metadata = direct.metadata
        expected_false = [
            "calls_model",
            "executes_tools",
            "queues_approval",
            "approves_request",
            "dismisses_request",
            "controls_computer",
            "reads_private_data",
            "writes_files",
            "writes_notes",
            "writes_memory",
            "external_side_effect",
            "requires_approval",
            "speaks",
            "authorizes_execution",
            "authorizes_completion_claim",
            "approval_granted",
        ]
        for key in expected_false:
            if metadata.get(key) is not False:
                raise SystemExit(f"prototype readiness metadata did not keep {key}=False")
        if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
            raise SystemExit(f"prototype readiness metadata missed operator limits: {metadata}")

        readiness = runtime.handle("readiness report")
        if not readiness.verified:
            raise SystemExit("readiness report route was not verified")
        for expected in [
            "Conversation latency proof:",
            "mixed conversation proof matrix",
            "model routing status",
            "live_check",
            "JARVIS_CHAT_MAX_REPLY_TOKENS",
            "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
        ]:
            if expected not in readiness.response:
                raise SystemExit(f"readiness report missed mixed-conversation proof guidance: {expected}")
        readiness_metadata = runtime.registry.get("readiness_report").handler({}).metadata
        if readiness_metadata.get("conversation_latency_status") != "proof_required":
            raise SystemExit(f"readiness report missed latency proof status: {readiness_metadata}")
        if readiness_metadata.get("conversation_latency_target_ms") != 8000:
            raise SystemExit(f"readiness report missed latency target: {readiness_metadata}")
        if readiness_metadata.get("conversation_latency_required_turns") != 10:
            raise SystemExit(f"readiness report missed turn-count target: {readiness_metadata}")
        if readiness_metadata.get("conversation_latency_required_chat_samples") != 3:
            raise SystemExit(f"readiness report missed chat-sample target: {readiness_metadata}")
        if readiness_metadata.get("conversation_latency_next_required_command") != "mixed conversation proof matrix":
            raise SystemExit(f"readiness report missed next proof command: {readiness_metadata}")
        if readiness_metadata.get("conversation_latency_tuning_knobs") != [
            "JARVIS_CHAT_MAX_REPLY_TOKENS",
            "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
        ]:
            raise SystemExit(f"readiness report missed tuning knobs: {readiness_metadata}")
        if readiness_metadata.get("chat_max_reply_tokens") != 300:
            raise SystemExit(f"readiness report missed default reply token cap: {readiness_metadata}")
        if readiness_metadata.get("chat_max_history_messages") != 16:
            raise SystemExit(f"readiness report missed default history window: {readiness_metadata}")
        for phrase in ["readiness", "준비 상태"]:
            routed = runtime.handle(phrase)
            if "Conversation latency proof:" not in routed.response or "model routing status" not in routed.response:
                raise SystemExit(f"{phrase!r} readiness route missed latency proof guidance: {routed.response[:220]!r}")

        for case in ["can i test jarvis", "what can i try now", "prototype readiness please", "show latest prototype readiness"]:
            routed = runtime.handle(case)
            if not routed.verified or "Jarvis prototype readiness checklist" not in routed.response:
                raise SystemExit(f"prototype readiness phrase did not route: {case}")
            if "setup check" not in routed.response or "computer control status" not in routed.response:
                raise SystemExit(f"prototype readiness phrase missed setup/computer status guidance: {case}")
            if "Priority goals do not override the operator's explicit stop times" not in routed.response:
                raise SystemExit(f"prototype readiness phrase missed operator-limit guidance: {case}")

    class ExplodingRow:
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("secret scheduled row read /\x55sers/example/private/jarvis.sqlite")

        def __str__(self) -> str:
            raise RuntimeError("secret scheduled row string /\x55sers/example/private/schedule")

    with TemporaryDirectory(prefix="jarvis-prototype-readiness-jobs-") as temp:
        runtime = make_temp_runtime(Path(temp))
        original_list_jobs = runtime.store.list_jobs
        original_recent_runs = runtime.store.recent_tool_runs
        runtime.store.list_jobs = lambda: [
            ExplodingRow(),
            {"name": "/\x55sers/example/private/State Snapshot", "job_type": "state_snapshot", "enabled": True},
            {"name": COMPACTION_JOB_NAME, "job_type": COMPACTION_JOB_TYPE, "enabled": True},
        ]
        runtime.store.recent_tool_runs = lambda limit=10: [
            ExplodingRow(),
            {
                "id": 77,
                "tool_name": "send_telegram /\x55sers/example/private/token",
                "ok": False,
                "risk": "HIGH_RISK",
                "approved": False,
                "metadata": "{}",
            },
            {
                "id": 79,
                "tool_name": "dispatch_decision_packet",
                "ok": False,
                "risk": "HIGH_RISK",
                "approved": False,
                "approval_id": 79,
                "metadata": '{"failure_kind":"approval-gate"}',
            },
            {"id": 78, "tool_name": "prototype_readiness_checklist", "ok": True, "risk": "READ_ONLY", "metadata": "{}"},
        ]
        try:
            direct = runtime.registry.get("prototype_readiness_checklist").handler({})
        finally:
            runtime.store.list_jobs = original_list_jobs
            runtime.store.recent_tool_runs = original_recent_runs

        if not direct.ok:
            raise SystemExit(f"prototype readiness malformed scheduled-job fixture failed: {direct}")
        if "scheduled job table has 1 unreadable row(s); inspect with `list scheduled jobs`" not in direct.output:
            raise SystemExit(f"prototype readiness missed unreadable scheduled-job guidance: {direct.output}")
        if "Recent tool-run audit has 1 unreadable row(s); inspect with `recent tool runs`." not in direct.output:
            raise SystemExit(f"prototype readiness missed unreadable recent-run guidance: {direct.output}")
        if "Recent failed tool runs exist; inspect with `recent tool runs`." not in direct.output:
            raise SystemExit(f"prototype readiness missed failed recent-run guidance: {direct.output}")
        if (
            "Recent approval-held tool runs exist; review `approval readiness 79` -> `approval packet 79` -> `approval chain proof 79`."
            not in direct.output
        ):
            raise SystemExit(f"prototype readiness missed approval-held review chain: {direct.output}")
        if "- recent tool runs" not in direct.output:
            raise SystemExit(f"prototype readiness missed recent tool-run command guidance: {direct.output}")
        if "state snapshot schedule is not enabled" in direct.output:
            raise SystemExit(f"prototype readiness lost readable state snapshot job: {direct.output}")
        if "conversation compaction schedule is not enabled" in direct.output:
            raise SystemExit(f"prototype readiness lost readable conversation compaction job: {direct.output}")
        for leaked in ["/\x55sers/operator", "jarvis.sqlite", "secret scheduled row", "send_telegram"]:
            if leaked in direct.output:
                raise SystemExit(f"prototype readiness leaked malformed local row text: {leaked}")
        metadata = direct.metadata
        if metadata.get("scheduled_jobs") != 3:
            raise SystemExit(f"prototype readiness should preserve scheduled row count: {metadata}")
        if metadata.get("readable_scheduled_jobs") != 2:
            raise SystemExit(f"prototype readiness missed readable scheduled row count: {metadata}")
        if metadata.get("unreadable_scheduled_job_rows") != 1:
            raise SystemExit(f"prototype readiness missed unreadable scheduled row count: {metadata}")
        if metadata.get("recent_tool_runs") != 4:
            raise SystemExit(f"prototype readiness should preserve recent row count: {metadata}")
        if metadata.get("readable_recent_tool_runs") != 3:
            raise SystemExit(f"prototype readiness missed readable recent row count: {metadata}")
        if metadata.get("unreadable_recent_tool_run_rows") != 1:
            raise SystemExit(f"prototype readiness missed unreadable recent row count: {metadata}")
        if metadata.get("recent_failed_runs") != 1:
            raise SystemExit(f"prototype readiness missed failed recent-run count: {metadata}")
        if metadata.get("recent_approval_held_runs") != 1:
            raise SystemExit(f"prototype readiness missed approval-held recent-run count: {metadata}")
        expected_approval_review = [
            "approval readiness 79",
            "approval packet 79",
            "approval chain proof 79",
            "verification receipt <approved run id from approval chain proof 79>",
        ]
        if metadata.get("approval_held_review_commands") != expected_approval_review:
            raise SystemExit(f"prototype readiness missed approval-held review commands: {metadata}")
        if metadata.get("approval_held_review_next_command") != "approval readiness 79":
            raise SystemExit(f"prototype readiness missed approval-held next review command: {metadata}")
        attention = metadata.get("optional_attention", [])
        if "scheduled job table has 1 unreadable row(s); inspect with `list scheduled jobs`" not in attention:
            raise SystemExit(f"prototype readiness metadata missed scheduled-job guidance: {metadata}")
        commands = metadata.get("prototype_commands", [])
        if "recent tool runs" not in commands:
            raise SystemExit(f"prototype readiness metadata missed recent tool-run command: {metadata}")
        for leaked in ["/\x55sers/operator", "jarvis.sqlite", "secret scheduled row", "send_telegram"]:
            if leaked in str(metadata):
                raise SystemExit(f"prototype readiness metadata leaked malformed local row text: {leaked}")

        print("[ok] prototype readiness checklist")


if __name__ == "__main__":
    main()
