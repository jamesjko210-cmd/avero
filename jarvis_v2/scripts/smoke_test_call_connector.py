"""Mocked smoke tests for the calling connector (FaceTime/phone, Kakao, Instagram, Telegram).

Every OS seam is mocked so no real call is ever placed.
"""

from __future__ import annotations

import tempfile
import subprocess
from pathlib import Path

from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools import call_connector
from jarvis_v2.tools.registry import validate_tool_arguments


def _config() -> JarvisConfig:
    root = Path(tempfile.mkdtemp())
    return JarvisConfig(data_dir=root, db_path=root / "jarvis.sqlite", obsidian_vault=root / "Vault")


def _tools():
    return {t.name: t for t in call_connector.make_call_tools(_config())}


def test_call_tools_reject_untyped_or_injected_arguments_before_execution() -> None:
    tools = _tools()
    expected_fields = {
        "call_contact": {"to", "recipient", "contact", "name", "mode", "kind", "type"},
        "call_kakao": {"to", "recipient", "contact", "name", "mode", "kind", "type"},
        "call_instagram": {"to", "recipient", "contact", "name", "mode", "kind", "type"},
        "call_telegram": {"to", "recipient", "name", "mode"},
    }
    for name, fields in expected_fields.items():
        tool = tools[name]
        contract = tool.argument_contract
        if (
            contract is None
            or contract.allow_unknown is not False
            or {field.name for field in contract.fields} != fields
            or any(field.required for field in contract.fields)
        ):
            raise SystemExit(f"{name} strict call argument contract drifted: {contract}")

        valid = validate_tool_arguments(tool, {"to": "Fixture", "mode": "audio"})
        if not valid.valid or valid.status != "valid":
            raise SystemExit(f"{name} rejected valid bounded call arguments: {valid}")

        wrong_type = validate_tool_arguments(tool, {"to": ["Fixture"], "mode": "audio"})
        if wrong_type.valid or wrong_type.status != "type_mismatch" or wrong_type.type_mismatch_keys != ("to",):
            raise SystemExit(f"{name} accepted a non-text recipient: {wrong_type}")

        injected = validate_tool_arguments(
            tool,
            {"to": "Fixture", "mode": "audio", "approval_granted": True},
        )
        if injected.valid or injected.status != "unknown_arguments" or injected.unknown_keys != ("<unknown>",):
            raise SystemExit(f"{name} accepted an injected call argument: {injected}")


def _assert_kakao_clipboard_read_boundaries(result, *, label: str, expected: bool) -> None:
    handoff = result.metadata.get("call_handoff")
    boundaries = handoff.get("boundaries") if isinstance(handoff, dict) else None
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} missed nested call boundaries: {result.metadata}")
    if result.metadata.get("call_handoff_boundaries") != boundaries:
        raise SystemExit(f"{label} call boundary alias drifted: {result.metadata}")
    for key in ("reads_clipboard", "reads_private_data", "reads_personal_data"):
        if result.metadata.get(key, False) is not expected:
            raise SystemExit(f"{label} flat {key} should be {expected}: {result.metadata}")
        if boundaries.get(key) is not expected:
            raise SystemExit(f"{label} nested {key} should be {expected}: {boundaries}")
    if handoff.get("reads_clipboard") is not expected:
        raise SystemExit(f"{label} handoff clipboard-read flag should be {expected}: {handoff}")


def _assert_unknown_call_failure(result, *, channel: str, label: str) -> None:
    expected_action = call_connector._call_attempt_recovery_guidance(channel)
    if result.ok or result.output != expected_action:
        raise SystemExit(f"{label} did not expose the fixed outcome-unknown guidance: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": expected_action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    if (
        result.metadata.get("next_command") != "setup check"
        or result.metadata.get("recovery_commands") != ["setup check"]
        or result.metadata.get("outcome_known") is not False
        or result.metadata.get("outcome_unknown") is not True
        or result.metadata.get("execution_outcome_unknown") is not True
        or result.metadata.get("side_effect_possible") is not True
        or result.metadata.get("retry_safe") is not False
        or result.metadata.get("automatic_retry_allowed") is not False
        or result.metadata.get("authorizes_retry") is not False
    ):
        raise SystemExit(f"{label} outcome/retry aliases drifted: {result.metadata}")
    handoff = result.metadata.get("call_handoff")
    if (
        not isinstance(handoff, dict)
        or handoff.get("status") != "outcome_unknown"
        or handoff.get("reason") != "call_outcome_unknown"
        or handoff.get("call_attempted") is not True
        or handoff.get("next_safe_command") != "recent tool runs"
        or handoff.get("next_safe_commands") != ["recent tool runs", "execution recovery"]
    ):
        raise SystemExit(f"{label} handoff overclaimed a failed outcome: {result.metadata}")
    if channel == "KakaoTalk":
        _assert_kakao_clipboard_read_boundaries(result, label=label, expected=True)


def _assert_known_not_started_instagram_call(result, *, label: str, stage: str) -> None:
    handoff = result.metadata.get("call_handoff")
    if result.ok or "did not start" not in result.output:
        raise SystemExit(f"{label} overclaimed an Instagram call: {result}")
    expected = {
        "failure_stage": stage,
        "instagram_call_stage": stage,
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "executes_side_effect": False,
        "external_side_effect": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    if (
        not isinstance(handoff, dict)
        or handoff.get("status") != "failed"
        or handoff.get("reason") != "call_not_started"
        or handoff.get("call_attempted") is not False
        or handoff.get("state_changed") is not False
        or handoff.get("next_safe_command") != "setup check"
    ):
        raise SystemExit(f"{label} handoff contradicted known-not-started truth: {result.metadata}")


def test_instagram_call_trusted_click_requires_verified_call_surface() -> None:
    from jarvis_v2.tools import instagram_connector

    opened: list[str] = []
    scripts: list[str] = []
    clicks: list[tuple[int, int]] = []
    original_open = instagram_connector.open_instagram_thread
    original_js = instagram_connector._chrome_js
    original_snapshot = call_connector._instagram_call_surface_snapshot
    original_wait = call_connector._wait_for_instagram_call_surface
    original_click = call_connector._trusted_instagram_click_at
    try:
        instagram_connector.open_instagram_thread = lambda recipient: opened.append(recipient) or recipient
        instagram_connector._chrome_js = lambda script: scripts.append(script) or "TARGET:420,315"
        call_connector._instagram_call_surface_snapshot = lambda: (1, 1, 0, 0)
        call_connector._wait_for_instagram_call_surface = lambda before: before == (1, 1, 0, 0)
        call_connector._trusted_instagram_click_at = lambda x, y: clicks.append((x, y))
        placed = call_connector._place_instagram_call("Fixture Example", "audio")
    finally:
        instagram_connector.open_instagram_thread = original_open
        instagram_connector._chrome_js = original_js
        call_connector._instagram_call_surface_snapshot = original_snapshot
        call_connector._wait_for_instagram_call_surface = original_wait
        call_connector._trusted_instagram_click_at = original_click
    if placed != "Fixture Example" or opened != ["Fixture Example"] or clicks != [(420, 315)]:
        raise SystemExit(f"Instagram trusted-click confirmation drifted: {placed!r} {opened} {clicks}")
    if len(scripts) != 1 or "Audio call" not in scripts[0] or "elementFromPoint" not in scripts[0]:
        raise SystemExit(f"Instagram did not revalidate the exact visible audio button: {scripts}")
    if "dispatchEvent" in scripts[0]:
        raise SystemExit("Instagram call setup must perform exactly one trusted activation, not a synthetic click")


def test_instagram_popup_blocked_or_no_surface_never_claims_success() -> None:
    from jarvis_v2.tools import instagram_connector

    clicks: list[tuple[int, int]] = []
    original_open = instagram_connector.open_instagram_thread
    original_js = instagram_connector._chrome_js
    original_snapshot = call_connector._instagram_call_surface_snapshot
    original_wait = call_connector._wait_for_instagram_call_surface
    original_click = call_connector._trusted_instagram_click_at
    try:
        instagram_connector.open_instagram_thread = lambda recipient: recipient
        instagram_connector._chrome_js = lambda _script: "TARGET:300,200"
        call_connector._instagram_call_surface_snapshot = lambda: (1, 1, 0, 0)
        call_connector._wait_for_instagram_call_surface = lambda _before: False
        call_connector._trusted_instagram_click_at = lambda x, y: clicks.append((x, y))
        try:
            call_connector._place_instagram_call("Fixture Example", "audio")
        except call_connector.InstagramCallError as exc:
            if (
                exc.stage != "instagram_call_popup_blocked_or_missing"
                or exc.outcome_unknown is not True
                or exc.call_attempted is not True
            ):
                raise SystemExit(f"Instagram missing-popup truth drifted: {exc.__dict__}")
        else:
            raise SystemExit("Instagram no-surface path claimed call_requested")
    finally:
        instagram_connector.open_instagram_thread = original_open
        instagram_connector._chrome_js = original_js
        call_connector._instagram_call_surface_snapshot = original_snapshot
        call_connector._wait_for_instagram_call_surface = original_wait
        call_connector._trusted_instagram_click_at = original_click
    if clicks != [(300, 200)]:
        raise SystemExit(f"Instagram no-surface path retried its trusted click: {clicks}")


def test_instagram_timeout_before_click_is_known_not_started() -> None:
    from jarvis_v2.tools import instagram_connector

    clicks: list[tuple[int, int]] = []
    original_open = instagram_connector.open_instagram_thread
    original_click = call_connector._trusted_instagram_click_at
    try:
        instagram_connector.open_instagram_thread = lambda _recipient: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["osascript"], 10)
        )
        call_connector._trusted_instagram_click_at = lambda x, y: clicks.append((x, y))
        try:
            call_connector._place_instagram_call("Fixture Example", "audio")
        except call_connector.InstagramCallError as exc:
            if exc.stage != "automation_timeout" or exc.outcome_unknown or exc.call_attempted:
                raise SystemExit(f"Instagram pre-click timeout truth drifted: {exc.__dict__}")
        else:
            raise SystemExit("Instagram pre-click timeout claimed success")
    finally:
        instagram_connector.open_instagram_thread = original_open
        call_connector._trusted_instagram_click_at = original_click
    if clicks:
        raise SystemExit(f"Instagram pre-click timeout performed a trusted click: {clicks}")


def test_instagram_timeout_after_click_is_outcome_unknown() -> None:
    from jarvis_v2.tools import instagram_connector

    clicks: list[tuple[int, int]] = []
    original_open = instagram_connector.open_instagram_thread
    original_js = instagram_connector._chrome_js
    original_snapshot = call_connector._instagram_call_surface_snapshot
    original_wait = call_connector._wait_for_instagram_call_surface
    original_click = call_connector._trusted_instagram_click_at
    try:
        instagram_connector.open_instagram_thread = lambda recipient: recipient
        instagram_connector._chrome_js = lambda _script: "TARGET:250,175"
        call_connector._instagram_call_surface_snapshot = lambda: (1, 1, 0, 0)
        call_connector._wait_for_instagram_call_surface = lambda _before: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["osascript"], 10)
        )
        call_connector._trusted_instagram_click_at = lambda x, y: clicks.append((x, y))
        try:
            call_connector._place_instagram_call("Fixture Example", "video")
        except call_connector.InstagramCallError as exc:
            if (
                exc.stage != "instagram_call_confirmation_timeout"
                or exc.outcome_unknown is not True
                or exc.call_attempted is not True
            ):
                raise SystemExit(f"Instagram post-click timeout truth drifted: {exc.__dict__}")
        else:
            raise SystemExit("Instagram post-click timeout claimed success")
    finally:
        instagram_connector.open_instagram_thread = original_open
        instagram_connector._chrome_js = original_js
        call_connector._instagram_call_surface_snapshot = original_snapshot
        call_connector._wait_for_instagram_call_surface = original_wait
        call_connector._trusted_instagram_click_at = original_click
    if clicks != [(250, 175)]:
        raise SystemExit(f"Instagram post-click timeout retried activation: {clicks}")


def test_instagram_active_call_guard_preserves_existing_call() -> None:
    from jarvis_v2.tools import instagram_connector

    calls: list[str] = []
    original_open = instagram_connector.open_instagram_thread
    original_js = instagram_connector._chrome_js
    original_snapshot = call_connector._instagram_call_surface_snapshot
    original_click = call_connector._trusted_instagram_click_at
    try:
        instagram_connector.open_instagram_thread = lambda _recipient: (_ for _ in ()).throw(
            instagram_connector.InstagramWebError("instagram_active_call")
        )
        instagram_connector._chrome_js = lambda _script: calls.append("js") or "TARGET:1,1"
        call_connector._instagram_call_surface_snapshot = lambda: calls.append("snapshot") or (1, 1, 0, 0)
        call_connector._trusted_instagram_click_at = lambda _x, _y: calls.append("click")
        try:
            call_connector._place_instagram_call("Fixture Example", "audio")
        except call_connector.InstagramCallError as exc:
            if exc.stage != "instagram_active_call" or exc.outcome_unknown or exc.call_attempted:
                raise SystemExit(f"Instagram active-call guard truth drifted: {exc.__dict__}")
        else:
            raise SystemExit("Instagram active-call guard allowed a new call")
    finally:
        instagram_connector.open_instagram_thread = original_open
        instagram_connector._chrome_js = original_js
        call_connector._instagram_call_surface_snapshot = original_snapshot
        call_connector._trusted_instagram_click_at = original_click
    if calls:
        raise SystemExit(f"Instagram active-call guard touched browser state after stopping: {calls}")


def test_instagram_call_surface_proof_rejects_unrelated_tabs() -> None:
    baseline = (1, 1, 0, 0)
    if call_connector._instagram_call_surface_started(baseline, (2, 2, 0, 0)):
        raise SystemExit("An unrelated Instagram tab/window was accepted as call proof")
    if not call_connector._instagram_call_surface_started(baseline, (2, 2, 1, 0)):
        raise SystemExit("Visible outgoing/ringing/end-call UI was not accepted as call proof")
    if not call_connector._instagram_call_surface_started(baseline, (2, 2, 0, 1)):
        raise SystemExit("A new call-specific Instagram URL was not accepted as call proof")


def test_instagram_call_handler_preserves_pre_and_post_click_truth() -> None:
    tools = _tools()
    original_place = call_connector._place_instagram_call
    try:
        call_connector._place_instagram_call = lambda _to, _mode: (_ for _ in ()).throw(
            call_connector.InstagramCallError(
                "instagram_active_call",
                outcome_unknown=False,
                call_attempted=False,
            )
        )
        pre_click = tools["call_instagram"].handler({"to": "Fixture Example", "mode": "audio"})
        _assert_known_not_started_instagram_call(
            pre_click,
            label="Instagram pre-click active-call guard",
            stage="instagram_active_call",
        )

        call_connector._place_instagram_call = lambda _to, _mode: (_ for _ in ()).throw(
            call_connector.InstagramCallError(
                "instagram_call_popup_blocked_or_missing",
                outcome_unknown=True,
                call_attempted=True,
            )
        )
        post_click = tools["call_instagram"].handler({"to": "Fixture Example", "mode": "audio"})
        if (
            post_click.ok
            or post_click.output != call_connector._instagram_call_no_surface_recovery_guidance()
            or post_click.metadata.get("failure_stage") != "instagram_call_popup_blocked_or_missing"
            or post_click.metadata.get("instagram_call_stage") != "instagram_call_popup_blocked_or_missing"
            or post_click.metadata.get("executes_side_effect") is not True
            or post_click.metadata.get("outcome_unknown") is not True
            or post_click.metadata.get("retry_safe") is not False
            or post_click.metadata.get("automatic_retry_allowed") is not False
        ):
            raise SystemExit(f"Instagram post-click stage/attempt truth drifted: {post_click.metadata}")
    finally:
        call_connector._place_instagram_call = original_place


def _assert_unknown_telegram_send_failure(result, *, owner_self: bool, label: str) -> None:
    expected_action = call_connector._post_attempt_telegram_send_recovery_guidance(owner_self)
    if result.ok or result.output != expected_action:
        raise SystemExit(f"{label} did not expose fixed outcome-unknown guidance: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": expected_action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    if (
        result.metadata.get("next_command") != "setup check"
        or result.metadata.get("recovery_commands") != ["setup check"]
        or result.metadata.get("outcome_known") is not False
        or result.metadata.get("outcome_unknown") is not True
        or result.metadata.get("execution_outcome_unknown") is not True
        or result.metadata.get("side_effect_possible") is not True
        or result.metadata.get("retry_safe") is not False
        or result.metadata.get("automatic_retry_allowed") is not False
        or result.metadata.get("authorizes_retry") is not False
    ):
        raise SystemExit(f"{label} outcome/retry aliases drifted: {result.metadata}")
    handoff = result.metadata.get("telegram_send_handoff")
    if (
        not isinstance(handoff, dict)
        or handoff.get("status") != "outcome_unknown"
        or handoff.get("reason") != "send_outcome_unknown"
        or handoff.get("send_attempted") is not True
        or handoff.get("next_safe_command") != "recent tool runs"
        or handoff.get("next_safe_commands") != ["recent tool runs", "execution recovery"]
    ):
        raise SystemExit(f"{label} handoff overclaimed a failed outcome: {result.metadata}")


def _assert_known_not_sent_telegram_send_failure(result, *, label: str) -> None:
    action = call_connector.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid the canonical recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} recovery field {key} drifted: {result.metadata}")


def test_all_call_tools_are_high_risk() -> None:
    for name, tool in _tools().items():
        if tool.risk != RiskLevel.HIGH_RISK:
            raise SystemExit(f"{name} must be HIGH_RISK (approval-gated): {tool.risk}")


def test_call_contact_resolves_and_places_native_call() -> None:
    tools = _tools()
    calls = []
    orig_place = call_connector._place_native_call
    orig_resolve = call_connector.contacts_connector.resolve_contact
    try:
        call_connector._place_native_call = lambda mode, handle: calls.append((mode, handle)) or ""
        call_connector.contacts_connector.resolve_contact = lambda _q: [
            call_connector.contacts_connector.ContactMatch("Fixture Example", phone="+821012345678")
        ]
        result = tools["call_contact"].handler({"to": "fixture", "mode": "video"})
    finally:
        call_connector._place_native_call = orig_place
        call_connector.contacts_connector.resolve_contact = orig_resolve
    if not result.ok or calls != [("video", "+821012345678")]:
        raise SystemExit(f"call_contact should resolve name and place FaceTime video: {result} {calls}")
    if not result.metadata.get("external_side_effect") or result.metadata.get("resolved_to") != "+821012345678":
        raise SystemExit(f"call_contact metadata should mark side effect + resolved handle: {result.metadata}")


def test_call_contact_default_mode_is_phone() -> None:
    # A plain "call X" places a PHONE call (tel:// via iPhone Continuity), not
    # FaceTime — FaceTime is only used when explicitly requested.
    tools = _tools()
    calls = []
    orig_place = call_connector._place_native_call
    orig_resolve = call_connector.contacts_connector.resolve_contact
    try:
        call_connector._place_native_call = lambda mode, handle: calls.append((mode, handle)) or ""
        call_connector.contacts_connector.resolve_contact = lambda _q: [
            call_connector.contacts_connector.ContactMatch("Mom", phone="+821011112222")
        ]
        result = tools["call_contact"].handler({"to": "mom"})
    finally:
        call_connector._place_native_call = orig_place
        call_connector.contacts_connector.resolve_contact = orig_resolve
    if not result.ok or calls != [("phone", "+821011112222")]:
        raise SystemExit(f"call_contact default mode should be a phone call: {result} {calls}")


def test_call_contact_explicit_handle_skips_resolution() -> None:
    tools = _tools()
    calls = []
    orig_place = call_connector._place_native_call
    orig_resolve = call_connector.contacts_connector.resolve_contact
    try:
        call_connector._place_native_call = lambda mode, handle: calls.append((mode, handle)) or ""

        def fail_resolve(_q):
            raise AssertionError("explicit handle should not resolve contacts")

        call_connector.contacts_connector.resolve_contact = fail_resolve
        result = tools["call_contact"].handler({"to": "+821099998888", "mode": "phone"})
    finally:
        call_connector._place_native_call = orig_place
        call_connector.contacts_connector.resolve_contact = orig_resolve
    if not result.ok or calls != [("phone", "+821099998888")]:
        raise SystemExit(f"call_contact should place an explicit-handle call as-is: {result} {calls}")


def test_phone_mode_rejects_email_but_facetime_audio_accepts_it() -> None:
    tools = _tools()
    calls = []
    original_place = call_connector._place_native_call
    original_resolve = call_connector.contacts_connector.resolve_contact
    try:
        call_connector._place_native_call = lambda mode, handle: calls.append((mode, handle)) or ""
        direct_phone = tools["call_contact"].handler(
            {"to": "fixture@example.com", "mode": "phone"}
        )
        if direct_phone.ok or calls or "requires a phone number" not in direct_phone.output:
            raise SystemExit(f"Direct email phone call must fail before placement: {direct_phone} {calls}")

        call_connector.contacts_connector.resolve_contact = lambda _query: [
            call_connector.contacts_connector.ContactMatch(
                "Email Fixture",
                email="fixture@example.com",
            )
        ]
        named_phone = tools["call_contact"].handler(
            {"to": "Email Fixture", "mode": "phone"}
        )
        if named_phone.ok or calls or named_phone.metadata.get("reason") != "missing_phone_handle":
            raise SystemExit(f"Email-only contact phone call must fail before placement: {named_phone} {calls}")

        audio = tools["call_contact"].handler(
            {"to": "Email Fixture", "mode": "audio"}
        )
    finally:
        call_connector._place_native_call = original_place
        call_connector.contacts_connector.resolve_contact = original_resolve
    if not audio.ok or calls != [("audio", "fixture@example.com")]:
        raise SystemExit(f"FaceTime audio should accept an email-only contact: {audio} {calls}")


def test_native_call_failure_is_outcome_unknown_and_private() -> None:
    tools = _tools()
    original_place = call_connector._place_native_call
    try:
        call_connector._place_native_call = lambda _mode, _handle: (_ for _ in ()).throw(
            RuntimeError("SECRET SHOULD NOT APPEAR /\x55sers/example/private/call.log")
        )
        result = tools["call_contact"].handler({"to": "+821099998888", "mode": "audio"})
    finally:
        call_connector._place_native_call = original_place
    _assert_unknown_call_failure(
        result,
        channel="FaceTime/phone",
        label="native call failure",
    )
    for forbidden in ("SECRET SHOULD NOT APPEAR", "/\x55sers/", "/private/"):
        if forbidden in result.output or forbidden in str(result.metadata):
            raise SystemExit(f"Native call failure leaked {forbidden!r}: {result}")


def test_call_contact_missing_and_notfound_refuse_without_calling() -> None:
    tools = _tools()
    calls = []
    orig_place = call_connector._place_native_call
    orig_resolve = call_connector.contacts_connector.resolve_contact
    try:
        call_connector._place_native_call = lambda mode, handle: calls.append((mode, handle)) or ""
        # missing recipient
        missing = tools["call_contact"].handler({"mode": "audio"})
        if missing.ok or calls:
            raise SystemExit(f"missing recipient must refuse without calling: {missing} {calls}")
        # not found
        call_connector.contacts_connector.resolve_contact = lambda _q: []
        nf = tools["call_contact"].handler({"to": "nobody"})
        if nf.ok or calls or "couldn't find" not in nf.output.lower():
            raise SystemExit(f"unknown contact must refuse without calling: {nf} {calls}")
        # ambiguous
        call_connector.contacts_connector.resolve_contact = lambda _q: [
            call_connector.contacts_connector.ContactMatch("Fixture Example", phone="+8210001"),
            call_connector.contacts_connector.ContactMatch("Fixture Kim", phone="+8210002"),
        ]
        amb = tools["call_contact"].handler({"to": "fixture"})
        if amb.ok or calls or "Fixture Example" not in amb.output:
            raise SystemExit(f"ambiguous contact must ask which one: {amb} {calls}")
    finally:
        call_connector._place_native_call = orig_place
        call_connector.contacts_connector.resolve_contact = orig_resolve


def test_call_kakao_and_instagram_are_name_based_gui() -> None:
    tools = _tools()
    kakao_calls, ig_calls = [], []
    orig_k = call_connector._place_kakao_call
    orig_i = call_connector._place_instagram_call
    try:
        call_connector._place_kakao_call = lambda to, mode: kakao_calls.append((to, mode)) or "true"
        call_connector._place_instagram_call = lambda to, mode: ig_calls.append((to, mode)) or "true"
        k = tools["call_kakao"].handler({"to": "fixture example", "mode": "video"})
        i = tools["call_instagram"].handler({"to": "henry", "mode": "audio"})
    finally:
        call_connector._place_kakao_call = orig_k
        call_connector._place_instagram_call = orig_i
    if not k.ok or kakao_calls != [("fixture example", "video")]:
        raise SystemExit(f"call_kakao should pass the raw name to GUI automation: {k} {kakao_calls}")
    if not i.ok or ig_calls != [("henry", "audio")]:
        raise SystemExit(f"call_instagram should pass the raw handle to GUI automation: {i} {ig_calls}")
    if not k.metadata.get("external_side_effect") or not k.metadata.get("controls_computer"):
        raise SystemExit(f"call_kakao metadata should mark GUI side effect: {k.metadata}")
    _assert_kakao_clipboard_read_boundaries(
        k,
        label="successful KakaoTalk call request",
        expected=True,
    )
    if (
        k.metadata.get("clipboard_snapshot_captured") is not True
        or k.metadata.get("clipboard_replacement_attempted") is not True
        or k.metadata.get("clipboard_text_restored") is not True
        or k.metadata.get("clipboard_fully_restored") is not False
        or k.metadata.get("clipboard_restoration_scope") != "plain_text_value_only"
        or k.metadata.get("clipboard_private_content_possible") is not False
    ):
        raise SystemExit(f"call_kakao success overclaimed its clipboard boundary: {k.metadata}")
    missing_kakao = tools["call_kakao"].handler({"mode": "audio"})
    _assert_kakao_clipboard_read_boundaries(
        missing_kakao,
        label="pre-snapshot KakaoTalk call refusal",
        expected=False,
    )


def test_gui_call_phone_mode_downgrades_to_audio() -> None:
    tools = _tools()
    kakao_calls = []
    orig_k = call_connector._place_kakao_call
    try:
        call_connector._place_kakao_call = lambda to, mode: kakao_calls.append((to, mode)) or "true"
        tools["call_kakao"].handler({"to": "x", "mode": "phone"})
    finally:
        call_connector._place_kakao_call = orig_k
    if kakao_calls != [("x", "audio")]:
        raise SystemExit(f"kakao/ig have no phone mode; should downgrade to audio: {kakao_calls}")


def test_kakao_false_click_result_cannot_claim_call_started() -> None:
    tools = _tools()
    original_kakao = call_connector._place_kakao_call
    try:
        call_connector._place_kakao_call = lambda _to, _mode: "false"
        result = tools["call_kakao"].handler({"to": "Fixture", "mode": "audio"})
    finally:
        call_connector._place_kakao_call = original_kakao
    if result.ok or result.metadata.get("failure_stage") != "kakao_call_button_missing":
        raise SystemExit(f"KakaoTalk false click result overclaimed a call: {result}")
    if (
        result.metadata.get("clipboard_text_restored") is not True
        or result.metadata.get("clipboard_outcome_known") is not True
        or result.metadata.get("clipboard_private_content_possible") is not False
    ):
        raise SystemExit(f"KakaoTalk false-click result lost clipboard boundary truth: {result.metadata}")
    _assert_unknown_call_failure(
        result,
        channel="KakaoTalk",
        label="KakaoTalk missing call button",
    )


def test_kakao_call_clipboard_restore_failure_is_private() -> None:
    from jarvis_v2.tools import kakao_connector

    tools = _tools()
    original_kakao = call_connector._place_kakao_call
    try:
        call_connector._place_kakao_call = lambda *_args: (_ for _ in ()).throw(
            kakao_connector.KakaoSendError(
                "clipboard_restore_failed",
                "PRIVATE /\x55sers/example/kakao-call-clipboard.log",
                clipboard_metadata=call_connector.clipboard_safety.privacy_boundary_metadata(
                    snapshot_attempted=True,
                    snapshot_captured=True,
                    replacement_attempted=True,
                    text_restored=False,
                ),
            )
        )
        result = tools["call_kakao"].handler({"to": "Fixture", "mode": "audio"})
    finally:
        call_connector._place_kakao_call = original_kakao
    _assert_unknown_call_failure(
        result,
        channel="KakaoTalk",
        label="KakaoTalk clipboard restore failure",
    )
    if (
        result.metadata.get("clipboard_text_restored") is not False
        or result.metadata.get("clipboard_outcome_known") is not False
        or result.metadata.get("clipboard_private_content_possible") is not True
        or "/\x55sers/" in str(result.metadata)
        or "PRIVATE" in result.output
    ):
        raise SystemExit(f"KakaoTalk call lost clipboard privacy truth: {result}")


def test_gui_call_failure_is_friendly() -> None:
    tools = _tools()
    orig_k = call_connector._place_kakao_call
    try:
        def boom(_to, _mode):
            raise RuntimeError("accessibility not granted")

        call_connector._place_kakao_call = boom
        result = tools["call_kakao"].handler({"to": "fixture", "mode": "audio"})
    finally:
        call_connector._place_kakao_call = orig_k
    if result.ok or "accessibility not granted" in result.output:
        raise SystemExit(f"kakao call failure should be friendly and not leak raw errors: {result}")
    if result.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"kakao call failure should keep a bounded diagnostic type: {result.metadata}")
    _assert_unknown_call_failure(
        result,
        channel="KakaoTalk",
        label="generic KakaoTalk call failure",
    )


def test_gui_call_failures_are_actionable_and_private() -> None:
    from jarvis_v2.tools import instagram_connector, kakao_connector

    tools = _tools()
    private_detail = "No Instagram DM thread named '가상연락처일'. Threads I can see: Private Friend | Family"
    original_instagram = call_connector._place_instagram_call
    original_kakao = call_connector._place_kakao_call
    try:
        call_connector._place_instagram_call = lambda _to, _mode: (_ for _ in ()).throw(
            instagram_connector.InstagramWebError("instagram_thread_not_found", private_detail)
        )
        instagram_result = tools["call_instagram"].handler({"to": "@the operator", "mode": "audio"})
        call_connector._place_kakao_call = lambda _to, _mode: (_ for _ in ()).throw(
            kakao_connector.KakaoSendError(
                "macos_automation_permission",
                "In System Settings > Privacy & Security > Automation, allow the Jarvis Python process "
                "to control System Events and KakaoTalk, then retry.",
            )
        )
        kakao_result = tools["call_kakao"].handler({"to": "Fixture Example", "mode": "audio"})
    finally:
        call_connector._place_instagram_call = original_instagram
        call_connector._place_kakao_call = original_kakao

    if instagram_result.ok or instagram_result.metadata.get("failure_stage") != "instagram_thread_not_found":
        raise SystemExit(f"Instagram call receipt missed the safe failure stage: {instagram_result}")
    if any(value in instagram_result.output or value in str(instagram_result.metadata) for value in ("가상연락처일", "Private Friend", "Family")):
        raise SystemExit(f"Instagram call failure leaked private page state: {instagram_result.output} {instagram_result.metadata}")
    _assert_unknown_call_failure(
        instagram_result,
        channel="Instagram",
        label="Instagram call failure",
    )
    if kakao_result.ok or kakao_result.metadata.get("failure_stage") != "macos_automation_permission":
        raise SystemExit(f"Kakao call receipt missed the safe failure stage: {kakao_result}")
    if "Automation and Accessibility" not in kakao_result.output or "Fixture Example" in kakao_result.output:
        raise SystemExit(f"Kakao call recovery guidance was incomplete or leaked a recipient: {kakao_result.output}")
    _assert_unknown_call_failure(
        kakao_result,
        channel="KakaoTalk",
        label="KakaoTalk permission call failure",
    )


def test_telegram_call_and_send_work_via_web() -> None:
    tools = _tools()
    # Calling goes through the chat-menu flow on web.telegram.org.
    call_calls = []
    orig_call = call_connector._place_telegram_call
    try:
        call_connector._place_telegram_call = lambda to, mode: call_calls.append((to, mode)) or to
        call = tools["call_telegram"].handler({"to": "BotFather", "mode": "audio"})
    finally:
        call_connector._place_telegram_call = orig_call
    if not call.ok or call_calls != [("BotFather", "audio")]:
        raise SystemExit(f"call_telegram should place a Telegram Web call: {call} {call_calls}")
    # A failure surfaces the specific stage, never a raw error.
    orig_call = call_connector._place_telegram_call
    try:
        call_connector._place_telegram_call = lambda _to, _mode: (_ for _ in ()).throw(
            call_connector.TelegramWebSendError("telegram_call_item_missing", "no Call option")
        )
        failed = tools["call_telegram"].handler({"to": "BotFather", "mode": "audio"})
    finally:
        call_connector._place_telegram_call = orig_call
    if (
        failed.ok
        or failed.metadata.get("failure_stage") != "telegram_call_item_missing"
        or failed.metadata.get("outcome_known") is not True
    ):
        raise SystemExit(f"call_telegram known-not-started stage drifted: {failed}")
    # Sending works via web.telegram.org — just mock the place function
    send_calls = []
    orig_place = call_connector._place_telegram_send
    try:
        call_connector._place_telegram_send = lambda to, msg: send_calls.append((to, msg)) or "sent"
        send = tools["send_telegram"].handler({"to": "fixture", "message": "hi"})
    finally:
        call_connector._place_telegram_send = orig_place
    if not send.ok or send_calls != [("fixture", "hi")]:
        raise SystemExit(f"send_telegram should send via Telegram Web: {send} {send_calls}")


def test_telegram_post_click_call_failure_is_outcome_unknown() -> None:
    tools = _tools()
    original_call = call_connector._place_telegram_call
    try:
        call_connector._place_telegram_call = lambda _to, _mode: (_ for _ in ()).throw(
            call_connector.TelegramWebSendError(
                "telegram_call_click_failed",
                "PRIVATE SHOULD NOT APPEAR /\x55sers/example/private/telegram-call.log",
            )
        )
        result = tools["call_telegram"].handler({"to": "Fixture", "mode": "audio"})
    finally:
        call_connector._place_telegram_call = original_call
    if result.metadata.get("failure_stage") != "telegram_call_click_failed":
        raise SystemExit(f"Telegram post-click stage drifted: {result}")
    _assert_unknown_call_failure(
        result,
        channel="Telegram",
        label="Telegram post-click call failure",
    )
    if "PRIVATE SHOULD NOT APPEAR" in result.output or "/\x55sers/" in str(result.metadata):
        raise SystemExit(f"Telegram post-click call failure leaked private detail: {result}")


def test_telegram_post_click_confirmation_stage_cannot_be_known_safe() -> None:
    original_open = call_connector._open_telegram_chat
    original_chrome = call_connector._chrome_js
    original_sleep = call_connector.time.sleep
    responses = iter(("TOGGLED", "CLICKED:100,200"))
    try:
        call_connector._open_telegram_chat = lambda _recipient: "Fixture"

        def fake_chrome(_script):
            try:
                return next(responses)
            except StopIteration:
                raise call_connector.TelegramWebSendError(
                    "chrome_javascript_permission",
                    "PRIVATE /\x55sers/example/post-click-call.log",
                )

        call_connector._chrome_js = fake_chrome
        call_connector.time.sleep = lambda _seconds: None
        try:
            call_connector._place_telegram_call("Fixture", "audio")
        except call_connector.TelegramWebSendError as exc:
            if exc.stage != "chrome_javascript_permission" or exc.outcome_unknown is not True:
                raise SystemExit(f"post-click Telegram call bridge failure was misclassified: {exc}")
            if "PRIVATE" in exc.detail or "/\x55sers/" in exc.detail:
                raise SystemExit(f"post-click Telegram call bridge failure leaked detail: {exc.detail}")
        else:
            raise SystemExit("post-click Telegram call bridge failure was accepted")
    finally:
        call_connector._open_telegram_chat = original_open
        call_connector._chrome_js = original_chrome
        call_connector.time.sleep = original_sleep

    tools = _tools()
    original_call = call_connector._place_telegram_call
    try:
        call_connector._place_telegram_call = lambda _to, _mode: (_ for _ in ()).throw(
            call_connector.TelegramWebSendError(
                "chrome_javascript_permission",
                "PRIVATE /\x55sers/example/post-click-handler.log",
                outcome_unknown=True,
            )
        )
        result = tools["call_telegram"].handler({"to": "Fixture", "mode": "audio"})
    finally:
        call_connector._place_telegram_call = original_call
    _assert_unknown_call_failure(
        result,
        channel="Telegram",
        label="Telegram post-click permission-stage failure",
    )
    if "PRIVATE" in result.output or "/\x55sers/" in str(result.metadata):
        raise SystemExit(f"Telegram post-click permission-stage failure leaked detail: {result}")


def test_telegram_failures_are_actionable_and_private() -> None:
    tools = _tools()
    private_detail = "No Telegram chat titled '가상연락처일'. Chats I can see: Private Friend | Family"
    original_send = call_connector._place_telegram_send
    original_call = call_connector._place_telegram_call
    try:
        call_connector._place_telegram_send = lambda _to, _message: (_ for _ in ()).throw(
            call_connector.TelegramWebSendError("telegram_chat_not_found", private_detail)
        )
        send_result = tools["send_telegram"].handler({"to": "가상연락처일", "message": "hello"})
        call_connector._place_telegram_call = lambda _to, _mode: (_ for _ in ()).throw(
            call_connector.TelegramWebSendError("telegram_call_item_missing", "Call | Video Call | Private Friend")
        )
        call_result = tools["call_telegram"].handler({"to": "가상연락처일", "mode": "audio"})
    finally:
        call_connector._place_telegram_send = original_send
        call_connector._place_telegram_call = original_call

    if send_result.ok or send_result.metadata.get("failure_stage") != "telegram_chat_not_found":
        raise SystemExit(f"Telegram send receipt missed the safe failure stage: {send_result}")
    if call_result.ok or call_result.metadata.get("failure_stage") != "telegram_call_item_missing":
        raise SystemExit(f"Telegram call receipt missed the safe failure stage: {call_result}")
    for result in (send_result, call_result):
        if any(value in result.output or value in str(result.metadata) for value in ("Private Friend", "Family")):
            raise SystemExit(f"Telegram failure leaked private page state: {result.output} {result.metadata}")
    if "Open or create the exact Telegram chat" not in send_result.output:
        raise SystemExit(f"Telegram send recovery guidance drifted: {send_result.output}")
    if "did not offer the requested call option" not in call_result.output:
        raise SystemExit(f"Telegram call recovery guidance drifted: {call_result.output}")


def test_telegram_send_verifies_chat_and_carries_korean_via_js() -> None:
    """Full flow with every osascript call mocked: Korean must travel raw inside
    the JS bridge (never via keystroke typing), the chat header must be verified
    before the message is staged, and the ONE trusted Enter (key code 36) must
    fire only after the composer is armed."""
    if "\\u" in call_connector._as_script_string("가상연락처이"):
        raise SystemExit("AppleScript string helper must preserve Korean instead of JSON \\u escapes")

    sequence: list[str] = []
    commands: list[str] = []
    orig_run = call_connector.subprocess.run
    try:
        def fake_run(command, capture_output, text, timeout):
            joined = "\n".join(str(part) for part in command)
            commands.append(joined)

            class Result:
                returncode = 0
                stderr = ""
                stdout = ""

            if "on run argv" in joined:
                js = str(command[3])
                if "'MISS:'" in js:
                    sequence.append("find")
                    Result.stdout = "PEER:777"
                elif "location.hash" in js:
                    sequence.append("open")
                    Result.stdout = "reloading"
                elif "location.reload" in js:
                    sequence.append("reload")
                    Result.stdout = "reloading"
                elif "'HEADER:'" in js:
                    sequence.append("verify_header")
                    Result.stdout = "HEADER:가상연락처이"
                elif "'NO_COMPOSER'" in js:
                    sequence.append("stage")
                    Result.stdout = "ARMED"
                elif "'COMPOSER_FOCUSED'" in js:
                    sequence.append("focus_check")
                    Result.stdout = "COMPOSER_FOCUSED"
                elif "'CLEARED'" in js:
                    sequence.append("confirm")
                    Result.stdout = "CLEARED"
                else:
                    sequence.append("boot")
                    Result.stdout = "ready"
            elif "key code 36" in joined:
                sequence.append("enter")
            else:
                sequence.append("focus_tab")
            return Result()

        call_connector.subprocess.run = fake_run
        header = call_connector._place_telegram_send("가상연락처이", "안녕하세요")
    finally:
        call_connector.subprocess.run = orig_run

    if header != "가상연락처이":
        raise SystemExit(f"telegram send should return the VERIFIED chat header: {header!r}")
    everything = "\n".join(commands)
    if "가상연락처이" not in everything or "안녕하세요" not in everything:
        raise SystemExit("Korean recipient/message must travel raw through the JS bridge")
    if "\\uc11c" in everything or "\\uc548" in everything:
        raise SystemExit("Telegram JS leaked JSON unicode escapes instead of Hangul")
    if 'keystroke "가상연락처이"' in everything or 'keystroke "안녕하세요"' in everything:
        raise SystemExit("Telegram send must never TYPE text via keystroke (Hangul cannot be typed)")
    if "active tab of front window" in everything:
        raise SystemExit("Telegram JS must target the Telegram tab by URL, never 'active tab of front window'")
    if ".normalize('NFKC')" not in everything:
        raise SystemExit("Telegram exact-title matching must normalize Unicode identity safely")
    expected_order = [
        "focus_tab", "boot", "reload", "boot", "find", "open",
        "verify_header", "stage", "focus_check", "enter", "confirm",
    ]
    if sequence != expected_order:
        raise SystemExit(f"telegram send steps out of order: {sequence} != {expected_order}")


def test_telegram_send_handler_accepts_korean_recipient_and_message() -> None:
    tools = _tools()
    send_calls = []
    orig_place = call_connector._place_telegram_send
    try:
        call_connector._place_telegram_send = lambda to, msg: send_calls.append((to, msg)) or "sent"
        send = tools["send_telegram"].handler({"to": "가상연락처이", "message": "안녕하세요"})
    finally:
        call_connector._place_telegram_send = orig_place
    if not send.ok or send_calls != [("가상연락처이", "안녕하세요")]:
        raise SystemExit(f"send_telegram should preserve Korean recipient/message: {send} {send_calls}")
    if send.metadata.get("message_chars") != len("안녕하세요"):
        raise SystemExit(f"send_telegram should count Korean message chars correctly: {send.metadata}")
    handoff = send.metadata.get("telegram_send_handoff")
    if (
        not isinstance(handoff, dict)
        or handoff.get("status") != "sender_confirmed"
        or send.metadata.get("sender_side_send_confirmed") is not True
        or send.metadata.get("recipient_delivery_confirmed") is not False
        or send.metadata.get("delivery_confirmed") is not False
        or send.metadata.get("delivery_evidence") != "sender_side_only"
        or send.metadata.get("recipient_confirmation_required") is not True
        or handoff.get("sender_side_send_confirmed") is not True
        or handoff.get("recipient_delivery_confirmed") is not False
        or "recipient delivery is not confirmed" not in send.output
        or "Telegram message sent" in send.output
    ):
        raise SystemExit(f"Telegram Web sender receipt overclaimed recipient delivery: {send}")


def test_telegram_owner_self_uses_bot_api_without_web_control() -> None:
    tools = _tools()
    owner_calls = []
    web_calls = []
    orig_owner = call_connector._send_owner_telegram
    orig_web = call_connector._place_telegram_send
    try:
        call_connector._send_owner_telegram = lambda msg: owner_calls.append(msg)
        call_connector._place_telegram_send = lambda to, msg: web_calls.append((to, msg))
        for recipient in ("me", "myself", "나에게", "나한테"):
            result = tools["send_telegram"].handler({"to": recipient, "message": "owner proof"})
            if not result.ok:
                raise SystemExit(f"owner-self Telegram send should use the configured owner channel: {result}")
            if result.metadata.get("telegram_delivery_path") != "owner_bot_api":
                raise SystemExit(f"owner-self Telegram send exposed the wrong delivery path: {result.metadata}")
            if result.metadata.get("controls_computer") is not False:
                raise SystemExit(f"owner-self Bot API send must not claim browser control: {result.metadata}")
    finally:
        call_connector._send_owner_telegram = orig_owner
        call_connector._place_telegram_send = orig_web
    if owner_calls != ["owner proof"] * 4 or web_calls:
        raise SystemExit(f"owner-self Telegram send crossed the wrong adapter: owner={owner_calls}, web={web_calls}")


def test_telegram_owner_self_failure_is_safe_and_bounded() -> None:
    tools = _tools()
    orig_owner = call_connector._send_owner_telegram
    try:
        call_connector._send_owner_telegram = lambda _msg: (_ for _ in ()).throw(
            call_connector.TelegramOwnerSendError(
                "telegram_api_rejected",
                "Check the Telegram bot connection and owner configuration, then retry.",
            )
        )
        result = tools["send_telegram"].handler({"to": "me", "message": "owner proof"})
    finally:
        call_connector._send_owner_telegram = orig_owner
    if result.ok or result.metadata.get("telegram_delivery_path") != "owner_bot_api":
        raise SystemExit(f"owner-self Telegram failure lost its bounded path: {result}")
    if result.metadata.get("telegram_send_stage") != "telegram_api_rejected":
        raise SystemExit(f"owner-self Telegram failure lost its safe stage: {result.metadata}")
    if result.metadata.get("controls_computer") is not False:
        raise SystemExit(f"owner-self Telegram failure must not claim browser control: {result.metadata}")
    if result.metadata.get("outcome_known") is not True or result.metadata.get("outcome_unknown") is not False:
        raise SystemExit(f"authoritative Telegram rejection should be known: {result.metadata}")
    _assert_known_not_sent_telegram_send_failure(
        result,
        label="owner Telegram rejection",
    )
    handoff = result.metadata.get("telegram_send_handoff")
    if not isinstance(handoff, dict) or handoff.get("status") != "failed":
        raise SystemExit(f"authoritative Telegram rejection should hand off as failed: {result.metadata}")
    if handoff.get("next_safe_commands") != ["setup check"]:
        raise SystemExit(f"known Telegram failure exposed stale recovery commands: {handoff}")


def test_telegram_owner_unknown_acknowledgement_is_quarantined() -> None:
    tools = _tools()
    private_detail = "PRIVATE /\x55sers/example/telegram-owner.log"
    orig_owner = call_connector._send_owner_telegram
    try:
        call_connector._send_owner_telegram = lambda _msg: (_ for _ in ()).throw(
            call_connector.TelegramOwnerSendError(
                "telegram_delivery_outcome_unknown",
                private_detail,
                outcome_unknown=True,
            )
        )
        result = tools["send_telegram"].handler({"to": "me", "message": "owner proof"})
    finally:
        call_connector._send_owner_telegram = orig_owner
    _assert_unknown_telegram_send_failure(
        result,
        owner_self=True,
        label="owner Telegram ambiguous acknowledgement",
    )
    if private_detail in result.output or "/\x55sers/" in str(result.metadata):
        raise SystemExit(f"owner Telegram ambiguous acknowledgement leaked detail: {result}")


def test_owner_telegram_api_response_classification() -> None:
    from jarvis_v2.automations import telegram_control

    orig_owner = telegram_control._owner_chat_id
    orig_send = telegram_control.send_message
    try:
        telegram_control._owner_chat_id = lambda: "-1008605"

        telegram_control.send_message = lambda _owner, _message: {
            "ok": True,
            "result": {
                "message_id": 8605,
                "chat": {"id": -1008605},
            },
        }
        call_connector._send_owner_telegram("proof")

        ambiguous_responses = (
            {"error": "network_outcome_unknown"},
            {"error": "malformed_response"},
            "not-a-dict",
            {"ok": True},
            {"ok": True, "result": {"chat": {"id": -1008605}}},
            {"ok": True, "result": {"message_id": 8605, "chat": {"id": -1009999}}},
            {
                "ok": False,
                "results": [
                    {
                        "ok": True,
                        "result": {
                            "message_id": 8605,
                            "chat": {"id": -1008605},
                        },
                    },
                    {"ok": False, "error": "http_400"},
                ],
            },
        )
        for index, response in enumerate(ambiguous_responses):
            telegram_control.send_message = lambda _owner, _message, value=response: value
            result = _tools()["send_telegram"].handler({"to": "me", "message": "proof"})
            _assert_unknown_telegram_send_failure(
                result,
                owner_self=True,
                label=f"owner Telegram ambiguous acknowledgement #{index}",
            )
            if (
                result.metadata.get("telegram_send_stage") != "telegram_delivery_outcome_unknown"
                or "-1008605" in result.output
                or "-1009999" in result.output
                or "-1008605" in str(result.metadata)
                or "-1009999" in str(result.metadata)
            ):
                raise SystemExit(
                    f"ambiguous owner response leaked identity or lost its safe stage: {result}"
                )

        telegram_control.send_message = lambda _owner, _message: {"ok": False, "error": "http_400"}
        try:
            call_connector._send_owner_telegram("proof")
        except call_connector.TelegramOwnerSendError as exc:
            if exc.stage != "telegram_api_rejected" or exc.outcome_unknown is not False:
                raise SystemExit(f"authoritative owner rejection was misclassified: {exc}")
        else:
            raise SystemExit("authoritative owner rejection was accepted")
    finally:
        telegram_control._owner_chat_id = orig_owner
        telegram_control.send_message = orig_send


def test_telegram_send_failure_reports_safe_stage() -> None:
    tools = _tools()
    orig_place = call_connector._place_telegram_send
    try:
        call_connector._place_telegram_send = lambda _to, _msg: (_ for _ in ()).throw(
            call_connector.TelegramWebSendError("chrome_javascript_permission", "Enable Chrome JavaScript from Apple Events.")
        )
        result = tools["send_telegram"].handler({"to": "가상연락처이", "message": "hello"})
    finally:
        call_connector._place_telegram_send = orig_place
    if result.ok or "chrome_javascript_permission" not in result.output:
        raise SystemExit(f"send_telegram should report the safe failure stage: {result.output}")
    if result.metadata.get("telegram_send_stage") != "chrome_javascript_permission":
        raise SystemExit(f"send_telegram should keep bounded stage metadata: {result.metadata}")
    if result.metadata.get("error_type") != "TelegramWebSendError":
        raise SystemExit(f"send_telegram should keep bounded error type metadata: {result.metadata}")
    if result.metadata.get("outcome_known") is not True or result.metadata.get("outcome_unknown") is not False:
        raise SystemExit(f"pre-send Telegram permission failure should be known: {result.metadata}")
    _assert_known_not_sent_telegram_send_failure(
        result,
        label="Telegram Web permission failure",
    )
    handoff = result.metadata.get("telegram_send_handoff")
    if not isinstance(handoff, dict) or handoff.get("next_safe_commands") != ["setup check"]:
        raise SystemExit(f"known Telegram Web failure exposed stale recovery commands: {result.metadata}")


def test_telegram_post_enter_send_failure_is_outcome_unknown() -> None:
    tools = _tools()
    private_detail = "PRIVATE /\x55sers/example/telegram-send.log"
    orig_place = call_connector._place_telegram_send
    try:
        call_connector._place_telegram_send = lambda _to, _msg: (_ for _ in ()).throw(
            call_connector.TelegramWebSendError(
                "telegram_send_unconfirmed",
                private_detail,
                outcome_unknown=True,
            )
        )
        result = tools["send_telegram"].handler({"to": "Fixture", "message": "hello"})
    finally:
        call_connector._place_telegram_send = orig_place
    _assert_unknown_telegram_send_failure(
        result,
        owner_self=False,
        label="Telegram Web post-Enter failure",
    )
    if private_detail in result.output or "/\x55sers/" in str(result.metadata):
        raise SystemExit(f"Telegram Web post-Enter failure leaked detail: {result}")


def test_telegram_post_enter_bridge_failure_becomes_unconfirmed() -> None:
    original_open = call_connector._open_telegram_chat
    original_chrome = call_connector._chrome_js
    original_enter = call_connector._trusted_enter
    original_sleep = call_connector.time.sleep
    calls = 0
    try:
        call_connector._open_telegram_chat = lambda _recipient: "Fixture"

        def fake_chrome(_script):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "ARMED"
            if calls == 2:
                return "COMPOSER_FOCUSED"
            raise RuntimeError("PRIVATE /\x55sers/example/post-enter.log")

        call_connector._chrome_js = fake_chrome
        call_connector._trusted_enter = lambda: None
        call_connector.time.sleep = lambda _seconds: None
        try:
            call_connector._place_telegram_send("Fixture", "hello")
        except call_connector.TelegramWebSendError as exc:
            if exc.stage != "telegram_send_unconfirmed" or exc.outcome_unknown is not True:
                raise SystemExit(f"post-Enter bridge failure was not quarantined: {exc}")
            if "PRIVATE" in exc.detail or "/\x55sers/" in exc.detail:
                raise SystemExit(f"post-Enter bridge failure leaked private detail: {exc.detail}")
        else:
            raise SystemExit("post-Enter bridge failure was accepted")
    finally:
        call_connector._open_telegram_chat = original_open
        call_connector._chrome_js = original_chrome
        call_connector._trusted_enter = original_enter
        call_connector.time.sleep = original_sleep


def test_telegram_osascript_failure_is_classified_without_raw_leak() -> None:
    captured = {}
    orig_run = call_connector.subprocess.run
    try:
        def fake_run(command, capture_output, text, timeout):
            captured["command"] = command

            class Result:
                returncode = 1
                stdout = ""
                stderr = "execution error: System Events got an error: osascript is not allowed assistive access. (-25211)"

            return Result()

        call_connector.subprocess.run = fake_run
        try:
            call_connector._place_telegram_send("가상연락처이", "hello")
        except call_connector.TelegramWebSendError as exc:
            if exc.stage != "macos_accessibility_permission":
                raise SystemExit(f"accessibility failure should be classified safely: {exc.stage} {exc.detail}")
            if "/\x55sers/" in exc.detail or "/private/" in exc.detail:
                raise SystemExit(f"telegram failure detail should not leak paths: {exc.detail}")
        else:
            raise SystemExit("expected TelegramWebSendError for osascript failure")
    finally:
        call_connector.subprocess.run = orig_run


def test_facetime_url_scheme_mapping() -> None:
    if call_connector._facetime_url("video", "+15551234") != "facetime://+15551234":
        raise SystemExit("video should map to facetime://")
    if call_connector._facetime_url("audio", "a@example.com") != "facetime-audio://a@example.com":
        raise SystemExit("audio should map to facetime-audio://")
    if call_connector._facetime_url("phone", "+15551234") != "tel://+15551234":
        raise SystemExit("phone should map to tel://")


def main() -> None:
    test_all_call_tools_are_high_risk()
    test_call_tools_reject_untyped_or_injected_arguments_before_execution()
    test_call_contact_resolves_and_places_native_call()
    test_call_contact_default_mode_is_phone()
    test_call_contact_explicit_handle_skips_resolution()
    test_phone_mode_rejects_email_but_facetime_audio_accepts_it()
    test_native_call_failure_is_outcome_unknown_and_private()
    test_call_contact_missing_and_notfound_refuse_without_calling()
    test_call_kakao_and_instagram_are_name_based_gui()
    test_instagram_call_trusted_click_requires_verified_call_surface()
    test_instagram_popup_blocked_or_no_surface_never_claims_success()
    test_instagram_timeout_before_click_is_known_not_started()
    test_instagram_timeout_after_click_is_outcome_unknown()
    test_instagram_active_call_guard_preserves_existing_call()
    test_instagram_call_surface_proof_rejects_unrelated_tabs()
    test_instagram_call_handler_preserves_pre_and_post_click_truth()
    test_gui_call_phone_mode_downgrades_to_audio()
    test_kakao_false_click_result_cannot_claim_call_started()
    test_kakao_call_clipboard_restore_failure_is_private()
    test_gui_call_failure_is_friendly()
    test_gui_call_failures_are_actionable_and_private()
    test_telegram_call_and_send_work_via_web()
    test_telegram_post_click_call_failure_is_outcome_unknown()
    test_telegram_post_click_confirmation_stage_cannot_be_known_safe()
    test_telegram_failures_are_actionable_and_private()
    test_telegram_send_verifies_chat_and_carries_korean_via_js()
    test_telegram_send_handler_accepts_korean_recipient_and_message()
    test_telegram_owner_self_uses_bot_api_without_web_control()
    test_telegram_owner_self_failure_is_safe_and_bounded()
    test_telegram_owner_unknown_acknowledgement_is_quarantined()
    test_owner_telegram_api_response_classification()
    test_telegram_send_failure_reports_safe_stage()
    test_telegram_post_enter_send_failure_is_outcome_unknown()
    test_telegram_post_enter_bridge_failure_becomes_unconfirmed()
    test_telegram_osascript_failure_is_classified_without_raw_leak()
    test_facetime_url_scheme_mapping()
    print("Call connector smoke passed")


if __name__ == "__main__":
    main()
