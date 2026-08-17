"""Smoke tests for the countdown/days-until connector (pure date math)."""

from __future__ import annotations

from datetime import date

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools import countdown_connector as cc

NO_AUTHORITY_FLAGS = (
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
)


def _tools():
    return {t.name: t for t in cc.make_countdown_tools(load_config())}


def _assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def _assert_countdown_handoff(metadata: dict, label: str, *, status: str, reason: str = "") -> dict:
    handoff = metadata.get("countdown_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed countdown handoff: {metadata}")
    if metadata.get("countdown_handoff_ready") is not True:
        raise SystemExit(f"{label} countdown handoff readiness flag missing: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} countdown nested handoff_ready missing: {handoff}")
    if metadata.get("ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} countdown handoff should be ready for operator clients: {metadata}")
    if metadata.get("countdown_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} countdown ready alias parity failed: {metadata} vs {handoff}")
    if metadata.get("state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} countdown should report no state change: {metadata}")
    if metadata.get("changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} countdown should expose an empty changed list: {metadata}")
    if metadata.get("countdown_state_changed") != handoff.get("state_changed") or metadata.get("countdown_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} countdown state alias parity failed: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} countdown should keep {key}=False in flat and nested metadata: {metadata}")
        alias = f"countdown_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} countdown no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if metadata.get("content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} countdown should not put prose content in handoff metadata: {metadata}")
    if metadata.get("countdown_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} countdown content alias parity failed: {metadata} vs {handoff}")
    if handoff.get("source") != "days_until" or handoff.get("status") != status:
        raise SystemExit(f"{label} countdown handoff source/status wrong: {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} countdown handoff reason wrong: {handoff}")
    if handoff.get("target_length") != metadata.get("target_length"):
        raise SystemExit(f"{label} countdown handoff target length parity failed: {metadata}")
    if handoff.get("local_path_target") is not metadata.get("local_path_target"):
        raise SystemExit(f"{label} countdown handoff local-path parity failed: {metadata}")
    next_safe_command = handoff.get("next_safe_command")
    if not isinstance(next_safe_command, str) or not next_safe_command:
        raise SystemExit(f"{label} countdown next safe command missing: {handoff}")
    expected_commands = [next_safe_command]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} countdown safe command list parity failed: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} countdown safe command count failed: {handoff}")
    if metadata.get("countdown_next_safe_command") != next_safe_command:
        raise SystemExit(f"{label} countdown next command alias failed: {metadata} vs {handoff}")
    if metadata.get("countdown_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} countdown next commands alias failed: {metadata} vs {handoff}")
    if metadata.get("countdown_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} countdown command count alias failed: {metadata} vs {handoff}")
    if status == "ok":
        if handoff.get("retry_safe") is not False:
            raise SystemExit(f"{label} successful countdown should not advertise retry as needed: {handoff}")
        if handoff.get("days") != metadata.get("days"):
            raise SystemExit(f"{label} countdown handoff days parity failed: {metadata}")
        if not handoff.get("label") or not handoff.get("target_date"):
            raise SystemExit(f"{label} countdown handoff missed parsed target fields: {handoff}")
        if handoff.get("timing") not in {"past", "today", "tomorrow", "future"}:
            raise SystemExit(f"{label} countdown handoff timing wrong: {handoff}")
    else:
        if handoff.get("retry_safe") is not True:
            raise SystemExit(f"{label} refused countdown should be safe to retry after correction: {handoff}")
        if handoff.get("days") is not None or handoff.get("target_date"):
            raise SystemExit(f"{label} refused countdown handoff should not claim parsed target: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    if metadata.get("countdown_boundaries") != boundaries:
        raise SystemExit(f"{label} countdown boundary alias parity failed: {metadata} vs {handoff}")
    for key in [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        *NO_AUTHORITY_FLAGS,
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
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} countdown should not perform {key}: {metadata}")
    _assert_no_local_path(handoff, f"{label} countdown handoff")
    return handoff


def test_is_read_only() -> None:
    if _tools()["days_until"].risk != RiskLevel.READ_ONLY:
        raise SystemExit("days_until should be READ_ONLY")


def test_parses_named_and_explicit() -> None:
    today = date(2026, 6, 16)
    t, label = cc.parse_target("christmas", today)
    if t != date(2026, 12, 25) or label != "christmas":
        raise SystemExit(f"christmas parse wrong: {t} {label}")
    t, _ = cc.parse_target("new year", today)
    if t != date(2027, 1, 1):  # already past this year -> next year
        raise SystemExit(f"new year should roll to next year: {t}")
    t, _ = cc.parse_target("dec 25", today)
    if t != date(2026, 12, 25):
        raise SystemExit(f"month-day parse wrong: {t}")
    fixed_cases = {
        "juneteenth": (date(2026, 6, 19), "juneteenth"),
        "veterans day": (date(2026, 11, 11), "veterans day"),
        "veteran's day": (date(2026, 11, 11), "veteran's day"),
        "groundhog day": (date(2026, 2, 2), "groundhog day"),
    }
    for target, (expected_date, expected_label) in fixed_cases.items():
        t, label = cc.parse_target(target, date(2026, 1, 1))
        if t != expected_date or label != expected_label:
            raise SystemExit(f"fixed holiday parse wrong for {target!r}: {t} {label}")
    t, label = cc.parse_target("juneteenth", date(2026, 6, 20))
    if t != date(2027, 6, 19) or label != "juneteenth":
        raise SystemExit(f"juneteenth should roll after passing: {t} {label}")
    t, label = cc.parse_target("easter", date(2026, 1, 1))
    if t != date(2026, 4, 5) or label != "easter":
        raise SystemExit(f"easter parse wrong for 2026: {t} {label}")
    t, label = cc.parse_target("easter sunday", date(2026, 4, 6))
    if t != date(2027, 3, 28) or label != "easter":
        raise SystemExit(f"easter should roll to next year after passing: {t} {label}")
    movable_cases = {
        "mlk day": (date(2026, 1, 19), "mlk day"),
        "martin luther king day": (date(2026, 1, 19), "mlk day"),
        "martin luther king jr day": (date(2026, 1, 19), "mlk day"),
        "presidents day": (date(2026, 2, 16), "presidents day"),
        "president's day": (date(2026, 2, 16), "presidents day"),
        "washington's birthday": (date(2026, 2, 16), "presidents day"),
        "columbus day": (date(2026, 10, 12), "columbus day"),
        "indigenous peoples day": (date(2026, 10, 12), "indigenous peoples day"),
        "indigenous people's day": (date(2026, 10, 12), "indigenous peoples day"),
        "thanksgiving": (date(2026, 11, 26), "thanksgiving"),
        "thanksgiving day": (date(2026, 11, 26), "thanksgiving"),
        "black friday": (date(2026, 11, 27), "black friday"),
        "mothers day": (date(2026, 5, 10), "mother's day"),
        "mother's day": (date(2026, 5, 10), "mother's day"),
        "fathers day": (date(2026, 6, 21), "father's day"),
        "father's day": (date(2026, 6, 21), "father's day"),
        "memorial day": (date(2026, 5, 25), "memorial day"),
        "labor day": (date(2026, 9, 7), "labor day"),
        "labour day": (date(2026, 9, 7), "labor day"),
    }
    for target, (expected_date, expected_label) in movable_cases.items():
        t, label = cc.parse_target(target, date(2026, 1, 1))
        if t != expected_date or label != expected_label:
            raise SystemExit(f"movable holiday parse wrong for {target!r}: {t} {label}")
    t, label = cc.parse_target("thanksgiving", date(2026, 11, 27))
    if t != date(2027, 11, 25) or label != "thanksgiving":
        raise SystemExit(f"thanksgiving should roll after passing: {t} {label}")
    t, label = cc.parse_target("black friday", date(2026, 11, 28))
    if t != date(2027, 11, 26) or label != "black friday":
        raise SystemExit(f"black friday should roll after passing: {t} {label}")
    t, label = cc.parse_target("mlk day", date(2026, 1, 20))
    if t != date(2027, 1, 18) or label != "mlk day":
        raise SystemExit(f"mlk day should roll after passing: {t} {label}")
    if cc.parse_target("nonsense words", today) is not None:
        raise SystemExit("unparseable target should be None")


def test_rejects_impossible_dates() -> None:
    today = date(2026, 6, 16)
    for target in ["feb 31", "2/31", "2026-02-31"]:
        if cc.parse_target(target, today) is not None:
            raise SystemExit(f"impossible date should be rejected: {target!r}")
    out = _tools()["days_until"].handler({"target": "feb 31"})
    if out.ok or "tell me a date" not in out.output.lower():
        raise SystemExit(f"impossible date should ask for a valid target: {out.output}")
    _assert_countdown_handoff(out.metadata, "impossible date", status="refused", reason="invalid_target")


def test_days_until_output() -> None:
    out = _tools()["days_until"].handler({"target": "christmas"})
    if not out.ok or "until christmas" not in out.output.lower():
        raise SystemExit(f"days_until output wrong: {out.output}")
    if out.metadata.get("target_length") != len("christmas") or out.metadata.get("local_path_target"):
        raise SystemExit(f"days_until should preserve bounded target metadata: {out.metadata}")
    handoff = _assert_countdown_handoff(out.metadata, "christmas countdown", status="ok")
    if handoff.get("target_preview") != "christmas" or handoff.get("label") != "christmas":
        raise SystemExit(f"days_until handoff should preserve sanitized target and label: {handoff}")


def test_days_until_easter_output() -> None:
    out = _tools()["days_until"].handler({"target": "easter"})
    if not out.ok or "until easter" not in out.output.lower():
        raise SystemExit(f"easter countdown output wrong: {out.output}")
    handoff = _assert_countdown_handoff(out.metadata, "easter countdown", status="ok")
    if handoff.get("label") != "easter" or not handoff.get("target_date"):
        raise SystemExit(f"easter countdown should expose parsed target fields: {handoff}")


def test_days_until_movable_holiday_output() -> None:
    out = _tools()["days_until"].handler({"target": "thanksgiving"})
    if not out.ok or "until thanksgiving" not in out.output.lower():
        raise SystemExit(f"thanksgiving countdown output wrong: {out.output}")
    handoff = _assert_countdown_handoff(out.metadata, "thanksgiving countdown", status="ok")
    if handoff.get("label") != "thanksgiving" or not handoff.get("target_date"):
        raise SystemExit(f"thanksgiving countdown should expose parsed target fields: {handoff}")


def test_days_until_unit_aware_output() -> None:
    """Real gap found live 2026-07-09: "how many weeks/months until X" always
    answered in days regardless of the unit asked for -- the tool now accepts
    a `unit` arg and reports in that unit (with the exact day count alongside,
    since weeks/months are inherently approximate for "months")."""
    weeks_out = _tools()["days_until"].handler({"target": "christmas", "unit": "weeks"})
    if not weeks_out.ok or "week" not in weeks_out.output.lower() or "day" not in weeks_out.output.lower():
        raise SystemExit(f"days_until weeks output wrong: {weeks_out.output}")
    if weeks_out.metadata.get("unit") != "weeks" or weeks_out.metadata.get("weeks") is None:
        raise SystemExit(f"days_until weeks metadata missed unit: {weeks_out.metadata}")
    weeks_handoff = _assert_countdown_handoff(weeks_out.metadata, "weeks countdown", status="ok")
    if weeks_handoff.get("unit") != "weeks" or weeks_handoff.get("weeks") != weeks_out.metadata.get("weeks"):
        raise SystemExit(f"days_until weeks handoff missed unit metadata: {weeks_handoff}")
    months_out = _tools()["days_until"].handler({"target": "christmas", "unit": "months"})
    if not months_out.ok or "month" not in months_out.output.lower():
        raise SystemExit(f"days_until months output wrong: {months_out.output}")
    if "~" in months_out.output:
        raise SystemExit(f"days_until months output should use calendar months, not rough approximation: {months_out.output}")
    if months_out.metadata.get("unit") != "months" or months_out.metadata.get("months") is None:
        raise SystemExit(f"days_until months metadata missed unit: {months_out.metadata}")
    months_handoff = _assert_countdown_handoff(months_out.metadata, "months countdown", status="ok")
    if months_handoff.get("unit") != "months" or months_handoff.get("months") != months_out.metadata.get("months"):
        raise SystemExit(f"days_until months handoff missed unit metadata: {months_handoff}")
    month_text, months, remainder = cc._format_months(date(2026, 7, 9), date(2026, 12, 25), 169)
    if month_text != "5 months and 16 days (169 days)" or months != 5 or remainder != 16:
        raise SystemExit(f"calendar-month formatter should preserve full months plus leftover days: {month_text}, {months}, {remainder}")
    if cc._format_weeks(169) != "24 weeks and 1 day (169 days)":
        raise SystemExit(f"week formatter should preserve leftover days: {cc._format_weeks(169)}")
    # Default (no unit, or explicit "days") must stay exactly as before.
    days_out = _tools()["days_until"].handler({"target": "christmas"})
    if "week" in days_out.output.lower() or "month" in days_out.output.lower():
        raise SystemExit(f"days_until default output should stay in days: {days_out.output}")


def test_requires_target() -> None:
    out = _tools()["days_until"].handler({"target": ""})
    if out.ok or "until what" not in out.output.lower():
        raise SystemExit(f"empty target should ask: {out.output}")
    _assert_countdown_handoff(out.metadata, "missing countdown target", status="refused", reason="missing_target")


def test_rejects_local_path_targets() -> None:
    for target in [
        "/\x55sers/example/private/date",
        "/private/tmp/jarvis/date",
        "/var/folders/zc/jarvis/date",
        "/tmp/jarvis/date",
    ]:
        out = _tools()["days_until"].handler({"target": target})
        if out.ok or "not a local file path" not in out.output.lower():
            raise SystemExit(f"path-shaped target should be rejected locally: {out.output}")
        if out.metadata.get("reason") != "invalid_target" or not out.metadata.get("local_path_target"):
            raise SystemExit(f"path-shaped target should include bounded invalid metadata: {out.metadata}")
        if any(fragment in out.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
            raise SystemExit(f"path-shaped target should not be echoed: {out.output}")
        handoff = _assert_countdown_handoff(out.metadata, "path-shaped countdown target", status="refused", reason="invalid_target")
        if handoff.get("target_preview") != "<local-path>":
            raise SystemExit(f"path-shaped countdown handoff should redact target preview: {handoff}")


def test_planner_routes_countdown() -> None:
    p = RuleBasedPlanner()
    cases = {
        "how many days until christmas": {"target": "christmas"},
        # Real gap found live 2026-07-09, fixed: the unit word (weeks/months)
        # is now detected and passed through so the tool answers in the unit
        # actually asked for, instead of always answering in days.
        "how many weeks until christmas": {"target": "christmas", "unit": "weeks"},
        "how many months until christmas": {"target": "christmas", "unit": "months"},
        "how many sleeps until christmas": {"target": "christmas"},
        "how many days left until christmas": {"target": "christmas"},
        "days left until christmas": {"target": "christmas"},
        "weeks left until christmas": {"target": "christmas", "unit": "weeks"},
        "months left until christmas": {"target": "christmas", "unit": "months"},
        "time left until christmas": {"target": "christmas"},
        "time remaining until christmas": {"target": "christmas"},
        "time until christmas": {"target": "christmas"},
        "time to christmas": {"target": "christmas"},
        "how much time until christmas": {"target": "christmas"},
        "how much time left until christmas": {"target": "christmas"},
        "how much longer until christmas": {"target": "christmas"},
        "how much longer before christmas": {"target": "christmas"},
        "how long before christmas": {"target": "christmas"},
        "days until new year": {"target": "new year"},
        "weeks to new year": {"target": "new year", "unit": "weeks"},
        "countdown to halloween": {"target": "halloween"},
        "countdown for halloween": {"target": "halloween"},
        "countdown halloween": {"target": "halloween"},
        "count down to halloween": {"target": "halloween"},
        "count down halloween": {"target": "halloween"},
        "halloween countdown": {"target": "halloween"},
        "christmas countdown": {"target": "christmas"},
        "christmas days left": {"target": "christmas"},
        "christmas weeks left": {"target": "christmas", "unit": "weeks"},
        "christmas months left": {"target": "christmas", "unit": "months"},
        "new year days left": {"target": "new year"},
        "days until easter": {"target": "easter"},
        "how long until easter": {"target": "easter"},
        "countdown to easter": {"target": "easter"},
        "easter countdown": {"target": "easter"},
        "when is easter": {"target": "easter"},
        "when is easter sunday": {"target": "easter sunday"},
        "days until mlk day": {"target": "mlk day"},
        "when is mlk day": {"target": "mlk day"},
        "when is martin luther king day": {"target": "martin luther king day"},
        "when is martin luther king jr day": {"target": "martin luther king jr day"},
        "when is presidents day": {"target": "presidents day"},
        "when is president's day": {"target": "president's day"},
        "when is washington's birthday": {"target": "washington's birthday"},
        "when is juneteenth": {"target": "juneteenth"},
        "days until juneteenth": {"target": "juneteenth"},
        "when is veterans day": {"target": "veterans day"},
        "when is veteran's day": {"target": "veteran's day"},
        "when is groundhog day": {"target": "groundhog day"},
        "when is columbus day": {"target": "columbus day"},
        "when is indigenous peoples day": {"target": "indigenous peoples day"},
        "when is indigenous people's day": {"target": "indigenous people's day"},
        "days until thanksgiving": {"target": "thanksgiving"},
        "how long until thanksgiving": {"target": "thanksgiving"},
        "thanksgiving countdown": {"target": "thanksgiving"},
        "countdown to black friday": {"target": "black friday"},
        "count down to black friday": {"target": "black friday"},
        "when is thanksgiving": {"target": "thanksgiving"},
        "when is thanksgiving day": {"target": "thanksgiving day"},
        "when is mothers day": {"target": "mothers day"},
        "when is mother's day": {"target": "mother's day"},
        "when is fathers day": {"target": "fathers day"},
        "when is father's day": {"target": "father's day"},
        "when is memorial day": {"target": "memorial day"},
        "when is labor day": {"target": "labor day"},
        "when is black friday": {"target": "black friday"},
        "when is christmas": {"target": "christmas"},
        "when is valentines day": {"target": "valentines day"},
        "when is 2027-01-01": {"target": "2027-01-01"},
    }
    for q, expected_args in cases.items():
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["days_until"] or actions[0].args != expected_args:
            raise SystemExit(f"countdown route missed: {q!r} -> {actions}")
    meeting_actions = p.plan("when is my meeting").actions
    if [a.tool_name for a in meeting_actions] == ["days_until"]:
        raise SystemExit("countdown route should not hijack general when-is questions")
    malformed_reverse_actions = p.plan("christmas days until").actions
    if [a.tool_name for a in malformed_reverse_actions] == ["days_until"]:
        raise SystemExit("countdown route should not use a bare preposition as the target")
    for q in ["time in london", "what time is it in london"]:
        if [a.tool_name for a in p.plan(q).actions] != ["current_time"]:
            raise SystemExit(f"countdown route should not hijack world-time question: {q!r}")
    dst_actions = p.plan("when is daylight saving time").actions
    if [a.tool_name for a in dst_actions] == ["current_time"]:
        raise SystemExit("world-time route should not hijack daylight-saving questions")


def main() -> None:
    test_is_read_only()
    test_parses_named_and_explicit()
    test_rejects_impossible_dates()
    test_days_until_output()
    test_days_until_easter_output()
    test_days_until_movable_holiday_output()
    test_days_until_unit_aware_output()
    test_requires_target()
    test_rejects_local_path_targets()
    test_planner_routes_countdown()
    print("Countdown connector smoke passed")


if __name__ == "__main__":
    main()
