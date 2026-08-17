from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import channel_health as channel_health_module
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION


class _HostileValue:
    def __bool__(self) -> bool:
        raise RuntimeError("bool trap /\x55sers/example/private SHOULD NOT APPEAR")

    def __str__(self) -> str:
        raise RuntimeError("str trap /\x55sers/example/private SHOULD NOT APPEAR")


class _HostileRow:
    def __getitem__(self, key: str):
        if key in {"metadata", "created_at", "tool_name", "id", "ok"}:
            return _HostileValue()
        raise KeyError(key)


class _StaticPlanner:
    def __init__(self, args: object):
        self.args = args

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise the channel-health runtime argument contract.",
            [PlannedAction("channel_health", self.args, "channel-health contract smoke")],
            needs_model=False,
        )


def _seed_channel_runs(runtime) -> None:
    store = runtime.store
    store.log_tool_run(
        "smoke",
        "send_telegram",
        "HIGH_RISK",
        True,
        True,
        "LIVE OUTPUT SHOULD NOT APPEAR",
        metadata={
            "to": "BotFather SHOULD NOT APPEAR",
            "message": "secret message SHOULD NOT APPEAR",
        },
    )
    store.log_tool_run(
        "smoke",
        "send_telegram",
        "HIGH_RISK",
        False,
        True,
        "FAILURE OUTPUT SHOULD NOT APPEAR",
        metadata={
            "telegram_send_stage": "telegram_enter_did_not_send",
            "telegram_send_detail": "detail SHOULD NOT APPEAR",
            "to": "BotFather SHOULD NOT APPEAR",
        },
    )
    old_id = store.log_tool_run(
        "smoke",
        "send_kakao",
        "HIGH_RISK",
        True,
        True,
        "OLD OUTPUT SHOULD NOT APPEAR",
        metadata={"to": "Fixture SHOULD NOT APPEAR"},
    )
    old_time = (datetime.now(UTC).replace(tzinfo=None) - timedelta(days=8)).replace(microsecond=0).isoformat() + "Z"
    with store.connect() as conn:
        conn.execute("UPDATE tool_runs SET created_at = ? WHERE id = ?", (old_time, old_id))

    store.log_tool_run(
        "smoke",
        "send_kakao",
        "HIGH_RISK",
        False,
        True,
        "KAKAO OUTPUT SHOULD NOT APPEAR",
        metadata={"reason": "chat_not_verified", "to": "Fixture SHOULD NOT APPEAR"},
    )
    store.log_tool_run(
        "smoke",
        "call_instagram",
        "HIGH_RISK",
        False,
        True,
        "CALL OUTPUT SHOULD NOT APPEAR",
        metadata={"instagram_call_stage": "instagram_call_button_missing"},
    )
    store.log_tool_run(
        "smoke",
        "send_imessage",
        "HIGH_RISK",
        True,
        True,
        "IMESSAGE OUTPUT SHOULD NOT APPEAR",
        metadata={"resolved_handle": "+15555550123 SHOULD NOT APPEAR"},
    )


def test_channel_health_tool_reports_metadata_only() -> None:
    with TemporaryDirectory(prefix="jarvis-channel-health-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_channel_runs(runtime)
        tool = runtime.registry.get("channel_health")
        if tool.risk != RiskLevel.READ_ONLY:
            raise SystemExit(f"channel_health should be read-only: {tool.risk}")
        contract = tool.argument_contract
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown
            or contract.fields
        ):
            raise SystemExit(f"channel_health should accept only an empty argument object: {contract}")
        result = tool.handler({})
        if not result.ok:
            raise SystemExit(f"channel_health should return ok: {result}")
        output = result.output
        expected_fragments = [
            "Jarvis channel health report:",
            "No recipient names, message text",
            "Telegram messages (send_telegram)",
            "last failure stage telegram_enter_did_not_send",
            "KakaoTalk messages (send_kakao)",
            "last failure stage chat_not_verified",
            "Instagram calls (call_instagram)",
            "last failure stage instagram_call_button_missing",
            "iMessage (send_imessage)",
            "7d successes 1",
        ]
        for fragment in expected_fragments:
            if fragment not in output:
                raise SystemExit(f"channel_health missed {fragment!r}:\n{output}")
        forbidden = [
            "SHOULD NOT APPEAR",
            "secret message",
            "BotFather",
            "Fixture",
            "+15555550123",
            "LIVE OUTPUT",
            "FAILURE OUTPUT",
        ]
        for fragment in forbidden:
            if fragment in output:
                raise SystemExit(f"channel_health leaked content/recipient fragment {fragment!r}:\n{output}")
        if result.metadata.get("content_suppressed") is not True or result.metadata.get("reads_message_content") is not False:
            raise SystemExit(f"channel_health metadata should state content suppression: {result.metadata}")
        channels = {row["channel"]: row for row in result.metadata.get("channels", [])}
        if channels["telegram_message"]["last_failure_stage"] != "telegram_enter_did_not_send":
            raise SystemExit(f"telegram stage metadata wrong: {channels['telegram_message']}")
        if channels["kakao_message"]["success_count_7d"] != 0:
            raise SystemExit(f"old Kakao success should not count in 7d window: {channels['kakao_message']}")
        if channels["imessage"]["success_count_7d"] != 1:
                raise SystemExit(f"iMessage success count wrong: {channels['imessage']}")


def test_channel_health_invalid_runtime_arguments_stop_before_audit_read_or_handler() -> None:
    invalid_args = (
        {"channel": "telegram"},
        {"limit": 1},
        [],
        None,
        "health",
    )
    for index, args in enumerate(invalid_args):
        with TemporaryDirectory(prefix="jarvis-channel-health-arguments-") as temp:
            runtime = make_temp_runtime(Path(temp))
            tool = runtime.registry.get("channel_health")
            calls = [0]

            def counted_handler(value: dict[str, object]):
                calls[0] += 1
                return tool.handler(value)

            runtime.registry._tools["channel_health"] = replace(
                tool,
                handler=counted_handler,
            )
            runtime.planner = _StaticPlanner(args)
            with mock.patch.object(
                runtime.store,
                "recent_tool_runs",
                side_effect=AssertionError("invalid channel-health arguments read audit state"),
            ):
                result = runtime.handle(
                    "reject invalid channel-health arguments",
                    request_token=f"channel-health-invalid-{index}",
                )
            if len(result.tool_results) != 1:
                raise SystemExit(f"invalid channel-health case {index} lost its result")
            item = result.tool_results[0]
            if (
                item.ok
                or item.metadata.get("failure_kind") != "tool_arguments_invalid"
                or item.metadata.get("handler_invoked") is not False
                or calls[0] != 0
            ):
                raise SystemExit(
                    f"invalid channel-health case {index} crossed the audit boundary: {item}"
                )


def test_channel_health_separates_approval_holds_from_real_failures() -> None:
    with TemporaryDirectory(prefix="jarvis-channel-health-approval-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.log_tool_run(
            "smoke",
            "send_telegram",
            "HIGH_RISK",
            False,
            True,
            "REAL FAILURE OUTPUT SHOULD NOT APPEAR",
            metadata={"telegram_send_stage": "telegram_enter_did_not_send"},
        )
        runtime.store.log_tool_run(
            "smoke",
            "send_telegram",
            "HIGH_RISK",
            False,
            False,
            "APPROVAL OUTPUT SHOULD NOT APPEAR",
            metadata={
                "failure_kind": "approval-gate",
                "requires_confirmation": True,
                "to": "BotFather SHOULD NOT APPEAR",
                "message": "secret approval message SHOULD NOT APPEAR",
            },
        )

        result = runtime.registry.get("channel_health").handler({})
        if not result.ok:
            raise SystemExit(f"channel_health should handle approval holds: {result}")
        channels = {row["channel"]: row for row in result.metadata.get("channels", [])}
        telegram = channels["telegram_message"]
        if telegram["last_failure_stage"] != "telegram_enter_did_not_send":
            raise SystemExit(f"approval hold should not mask real failure stage: {telegram}")
        if telegram["last_approval_hold_stage"] != "approval_required":
            raise SystemExit(f"approval hold stage should be explicit and safe: {telegram}")
        if telegram["last_approval_hold_at"] == "never":
            raise SystemExit(f"approval hold timestamp missing: {telegram}")
        if "last approval hold" not in result.output:
            raise SystemExit(f"approval hold should be visible in the health report:\n{result.output}")
        payload = result.output + "\n" + str(result.metadata)
        for fragment in [
            "APPROVAL OUTPUT",
            "REAL FAILURE OUTPUT",
            "BotFather",
            "secret approval message",
            "SHOULD NOT APPEAR",
        ]:
            if fragment in payload:
                raise SystemExit(f"channel_health leaked approval fixture detail {fragment!r}: {payload}")


def test_channel_health_approval_only_channel_has_no_failure() -> None:
    with TemporaryDirectory(prefix="jarvis-channel-health-approval-only-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.log_tool_run(
            "smoke",
            "send_kakao",
            "HIGH_RISK",
            False,
            False,
            "KAKAO APPROVAL OUTPUT SHOULD NOT APPEAR",
            metadata={"failure_stage": "explicit approval required"},
        )

        result = runtime.registry.get("channel_health").handler({})
        if not result.ok:
            raise SystemExit(f"channel_health should handle approval-only channels: {result}")
        channels = {row["channel"]: row for row in result.metadata.get("channels", [])}
        kakao = channels["kakao_message"]
        if kakao["last_failure_at"] != "never" or kakao["last_failure_stage"] != "none":
            raise SystemExit(f"approval-only channel should not report a failure: {kakao}")
        if kakao["last_approval_hold_stage"] != "approval_required" or kakao["last_approval_hold_at"] == "never":
            raise SystemExit(f"approval-only channel should report the held approval: {kakao}")
        payload = result.output + "\n" + str(result.metadata)
        if "KAKAO APPROVAL OUTPUT" in payload or "SHOULD NOT APPEAR" in payload:
            raise SystemExit(f"channel_health leaked approval-only fixture detail: {payload}")


def test_channel_health_separates_preexecution_stops_from_channel_failures() -> None:
    with TemporaryDirectory(prefix="jarvis-channel-health-preexecution-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.log_tool_run(
            "smoke",
            "send_imessage",
            "HIGH_RISK",
            True,
            True,
            "SUCCESS OUTPUT SHOULD NOT APPEAR",
            metadata={"resolved_handle": "+15555550123 SHOULD NOT APPEAR"},
        )
        runtime.store.log_tool_run(
            "smoke",
            "send_imessage",
            "HIGH_RISK",
            False,
            True,
            "PREFLIGHT OUTPUT SHOULD NOT APPEAR",
            metadata={
                "executed_handler": False,
                "failure_kind": "contact_resolution_not_found",
                "planned_arg_keys": ["to", "message"],
                "to": "Private Recipient SHOULD NOT APPEAR",
            },
        )

        result = runtime.registry.get("channel_health").handler({})
        if not result.ok:
            raise SystemExit(f"channel_health should handle pre-execution stops: {result}")
        channels = {row["channel"]: row for row in result.metadata.get("channels", [])}
        imessage = channels["imessage"]
        if imessage["last_failure_at"] != "never" or imessage["last_failure_stage"] != "none":
            raise SystemExit(
                f"pre-execution contact stop must not regress iMessage health: {imessage}"
            )
        if (
            imessage["last_preexecution_stop_at"] == "never"
            or imessage["last_preexecution_stop_stage"]
            != "contact_resolution_not_found"
        ):
            raise SystemExit(
                f"pre-execution contact stop should remain visible separately: {imessage}"
            )
        if "last pre-execution stage contact_resolution_not_found" not in result.output:
            raise SystemExit(
                f"pre-execution stop should be visible in the safe report:\n{result.output}"
            )
        payload = result.output + "\n" + str(result.metadata)
        for fragment in [
            "SUCCESS OUTPUT",
            "PREFLIGHT OUTPUT",
            "+15555550123",
            "Private Recipient",
            "SHOULD NOT APPEAR",
        ]:
            if fragment in payload:
                raise SystemExit(
                    f"channel_health leaked pre-execution fixture detail {fragment!r}: {payload}"
                )


def test_channel_health_registered_in_runtime() -> None:
    with TemporaryDirectory(prefix="jarvis-channel-health-registry-") as temp:
        runtime = make_temp_runtime(Path(temp))
        names = {tool.name for tool in runtime.registry.list("personal")}
        if "channel_health" not in names:
            raise SystemExit(f"channel_health should be registered in personal tools: {sorted(names)}")


def test_channel_health_counts_timezone_aware_timestamps() -> None:
    with TemporaryDirectory(prefix="jarvis-channel-health-tz-") as temp:
        runtime = make_temp_runtime(Path(temp))
        run_id = runtime.store.log_tool_run(
            "smoke",
            "call_telegram",
            "HIGH_RISK",
            True,
            True,
            "TIMEZONE OUTPUT SHOULD NOT APPEAR",
            metadata={"to": "Timezone Recipient SHOULD NOT APPEAR"},
        )
        aware_time = datetime.now(UTC).replace(microsecond=0).isoformat()
        with runtime.store.connect() as conn:
            conn.execute("UPDATE tool_runs SET created_at = ? WHERE id = ?", (aware_time, run_id))

        result = runtime.registry.get("channel_health").handler({})
        if not result.ok:
            raise SystemExit(f"channel_health should handle timezone-aware timestamps: {result}")
        channels = {row["channel"]: row for row in result.metadata.get("channels", [])}
        if channels["telegram_call"]["success_count_7d"] != 1:
            raise SystemExit(f"timezone-aware success should count in 7d window: {channels['telegram_call']}")
        payload = result.output + "\n" + str(result.metadata)
        for fragment in ["TIMEZONE OUTPUT", "Timezone Recipient", "SHOULD NOT APPEAR"]:
            if fragment in payload:
                raise SystemExit(f"channel_health leaked timezone regression fixture {fragment!r}: {payload}")


def test_channel_health_suppresses_verbose_failure_reasons() -> None:
    with TemporaryDirectory(prefix="jarvis-channel-health-reason-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.log_tool_run(
            "smoke",
            "call_telegram",
            "HIGH_RISK",
            False,
            True,
            "VERBOSE FAILURE OUTPUT SHOULD NOT APPEAR",
            metadata={
                "reason": "could not open BotFather SHOULD NOT APPEAR for secret message",
                "telegram_call_detail": "detail SHOULD NOT APPEAR",
            },
        )

        result = runtime.registry.get("channel_health").handler({})
        if not result.ok:
            raise SystemExit(f"channel_health should handle verbose failure reasons: {result}")
        channels = {row["channel"]: row for row in result.metadata.get("channels", [])}
        if channels["telegram_call"]["last_failure_stage"] != "reason_detail_suppressed":
            raise SystemExit(f"verbose reason should be suppressed: {channels['telegram_call']}")
        payload = result.output + "\n" + str(result.metadata)
        for fragment in ["VERBOSE FAILURE OUTPUT", "BotFather", "secret message", "SHOULD NOT APPEAR"]:
            if fragment in payload:
                raise SystemExit(f"channel_health leaked verbose failure detail {fragment!r}: {payload}")


def test_channel_health_suppresses_malformed_timestamps() -> None:
    with TemporaryDirectory(prefix="jarvis-channel-health-created-at-") as temp:
        runtime = make_temp_runtime(Path(temp))
        run_id = runtime.store.log_tool_run(
            "smoke",
            "send_telegram",
            "HIGH_RISK",
            False,
            True,
            "MALFORMED TIMESTAMP OUTPUT SHOULD NOT APPEAR",
            metadata={"telegram_send_stage": "telegram_enter_did_not_send"},
        )
        poisoned_created_at = (
            "not-a-timestamp /\x55sers/example/private BotFather "
            "secret message SHOULD NOT APPEAR"
        )
        with runtime.store.connect() as conn:
            conn.execute("UPDATE tool_runs SET created_at = ? WHERE id = ?", (poisoned_created_at, run_id))

        result = runtime.registry.get("channel_health").handler({})
        if not result.ok:
            raise SystemExit(f"channel_health should handle malformed created_at values: {result}")
        channels = {row["channel"]: row for row in result.metadata.get("channels", [])}
        telegram = channels["telegram_message"]
        if telegram["last_failure_at"] != "invalid_timestamp":
            raise SystemExit(f"malformed timestamp should be suppressed: {telegram}")
        if "last failure invalid_timestamp" not in result.output:
            raise SystemExit(f"output should show invalid timestamp marker:\n{result.output}")
        payload = result.output + "\n" + str(result.metadata)
        for fragment in [
            "not-a-timestamp",
            "/\x55sers/operator",
            "BotFather",
            "secret message",
            "SHOULD NOT APPEAR",
            "MALFORMED TIMESTAMP OUTPUT",
        ]:
            if fragment in payload:
                raise SystemExit(f"channel_health leaked malformed timestamp detail {fragment!r}: {payload}")


def test_channel_health_marks_bounded_row_sample() -> None:
    with TemporaryDirectory(prefix="jarvis-channel-health-bounded-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for index in range(501):
            runtime.store.log_tool_run(
                "smoke",
                "send_telegram",
                "HIGH_RISK",
                True,
                True,
                f"BOUNDED OUTPUT {index} SHOULD NOT APPEAR",
                metadata={"to": f"Bounded Recipient {index} SHOULD NOT APPEAR"},
            )

        result = runtime.registry.get("channel_health").handler({})
        if not result.ok:
            raise SystemExit(f"channel_health should handle bounded row samples: {result}")
        if result.metadata.get("rows_reviewed") != 500:
            raise SystemExit(f"channel_health should cap reviewed rows at 500: {result.metadata}")
        if result.metadata.get("bounded_rows_limit") != 500:
            raise SystemExit(f"channel_health should expose bounded row limit: {result.metadata}")
        if result.metadata.get("row_sample_truncated") is not True:
            raise SystemExit(f"channel_health should mark truncated samples: {result.metadata}")
        if "Rows reviewed: 500 / 500 max (bounded sample; older matching rows may exist)." not in result.output:
            raise SystemExit(f"channel_health should explain bounded samples:\n{result.output}")
        payload = result.output + "\n" + str(result.metadata)
        for fragment in ["BOUNDED OUTPUT", "Bounded Recipient", "SHOULD NOT APPEAR"]:
            if fragment in payload:
                raise SystemExit(f"channel_health leaked bounded row fixture {fragment!r}: {payload}")


def test_channel_health_helpers_tolerate_hostile_row_values() -> None:
    row = _HostileRow()
    if channel_health_module._short(_HostileValue()) != "<unreadable>":
        raise SystemExit("hostile value should collapse to unreadable placeholder")
    if channel_health_module._metadata(row) != {}:
        raise SystemExit("hostile metadata payload should be ignored safely")
    if channel_health_module._created_at(row) != "invalid_timestamp":
        raise SystemExit("hostile timestamp should become invalid_timestamp")
    if channel_health_module._parse_created_at(_HostileValue()) is not None:
        raise SystemExit("hostile timestamp parser input should not parse")
    if channel_health_module._normalized_metadata_token(_HostileValue()) != "unreadable":
        raise SystemExit("hostile token should normalize to unreadable placeholder")
    if channel_health_module._failure_stage(row) != "unknown_failure":
        raise SystemExit("hostile metadata row should not invent a failure stage")
    if channel_health_module._is_approval_hold(row):
        raise SystemExit("hostile metadata row should not be treated as approval-held")
    if channel_health_module._is_preexecution_stop(row):
        raise SystemExit("hostile metadata row should not be treated as a pre-execution stop")
    if channel_health_module._safe_int(_HostileValue(), default=-1) != -1:
        raise SystemExit("hostile integer coercion should use default")
    payload = "\n".join(
        [
            channel_health_module._short(_HostileValue()),
            channel_health_module._created_at(row),
            channel_health_module._failure_stage_value("reason", _HostileValue()),
        ]
    )
    for fragment in ["/\x55sers/operator", "SHOULD NOT APPEAR", "bool trap", "str trap"]:
        if fragment in payload:
            raise SystemExit(f"hostile helper payload leaked {fragment!r}: {payload}")


def test_channel_health_routes_from_natural_phrases() -> None:
    from jarvis_v2.agent.planner import RuleBasedPlanner

    planner = RuleBasedPlanner()
    for phrase in (
        "channel status",
        "channel health",
        "messaging health",
        "channel health report",
        # Real gap found live 2026-07-09: "check channel health" fell through
        # to chat while bare "channel health" worked.
        "check channel health",
    ):
        plan = planner.plan(phrase)
        tools = [a.tool_name for a in (plan.actions or [])]
        if tools != ["channel_health"]:
            raise SystemExit(f"{phrase!r} should route to channel_health: {tools}")
    # must not shadow neighbours
    for phrase, expected in (("model status", "model_routing_status"), ("how is the weather today", "get_weather")):
        plan = planner.plan(phrase)
        tools = [a.tool_name for a in (plan.actions or [])]
        if tools != [expected]:
            raise SystemExit(f"{phrase!r} regressed: {tools}")


def main() -> None:
    test_channel_health_tool_reports_metadata_only()
    test_channel_health_invalid_runtime_arguments_stop_before_audit_read_or_handler()
    test_channel_health_separates_approval_holds_from_real_failures()
    test_channel_health_approval_only_channel_has_no_failure()
    test_channel_health_separates_preexecution_stops_from_channel_failures()
    test_channel_health_registered_in_runtime()
    test_channel_health_counts_timezone_aware_timestamps()
    test_channel_health_suppresses_verbose_failure_reasons()
    test_channel_health_suppresses_malformed_timestamps()
    test_channel_health_marks_bounded_row_sample()
    test_channel_health_helpers_tolerate_hostile_row_values()
    test_channel_health_routes_from_natural_phrases()
    print("Channel health smoke passed")


if __name__ == "__main__":
    main()
