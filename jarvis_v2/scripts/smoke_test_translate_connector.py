"""Smoke tests for the translation connector (mocked fetch, no network)."""

from __future__ import annotations

from typing import Any

from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import translate_connector as tc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry


NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _tools():
    return {t.name: t for t in tc.make_translate_tools(load_config())}


def _assert_translate_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to mymemory",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "offline", "traceback", "/users/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in lowered:
            raise SystemExit(f"{label} leaked raw backend text {forbidden!r}: {output}")


def _assert_translate_handoff(
    metadata: dict[str, Any],
    label: str,
    *,
    status: str,
    reason: str = "",
    calls_external_service: bool = True,
    retry_safe: bool = False,
    source: str = "",
    target: str = "",
) -> dict[str, Any]:
    if metadata.get("translate_handoff_ready") is not True:
        raise SystemExit(f"{label} missed translate_handoff_ready: {metadata}")
    handoff = metadata.get("translate_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing translate_handoff: {metadata}")
    if handoff.get("source") != "mymemory" or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong source/status: {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} nested handoff_ready missing: {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong reason: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff should be operator-ready: {handoff}")
    if metadata.get("translate_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} handoff should declare no state changes: {handoff}")
    if metadata.get("translate_state_changed") != handoff.get("state_changed") or metadata.get("translate_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    if source and handoff.get("source_lang") != source:
        raise SystemExit(f"{label} has wrong source language: {handoff}")
    if target and handoff.get("target_lang") != target:
        raise SystemExit(f"{label} has wrong target language: {handoff}")
    for key in ("raw_from", "raw_to", "text_chars", "max_text_chars", "text_truncated", "exception_type"):
        if handoff.get(key) != metadata.get(key, ""):
            raise SystemExit(f"{label} should mirror {key}: {handoff} vs {metadata}")
    if handoff.get("content_in_handoff") is not False or handoff.get("input_text_in_metadata") is not False or handoff.get("translated_text_in_metadata") is not False:
        raise SystemExit(f"{label} should exclude input/output text from metadata: {handoff}")
    if metadata.get("translate_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content alias parity failed: {metadata} vs {handoff}")
    if metadata.get("translate_input_text_in_metadata") != handoff.get("input_text_in_metadata"):
        raise SystemExit(f"{label} input-text alias parity failed: {metadata} vs {handoff}")
    if metadata.get("translate_translated_text_in_metadata") != handoff.get("translated_text_in_metadata"):
        raise SystemExit(f"{label} translated-text alias parity failed: {metadata} vs {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
        alias = f"translate_{key}"
        if metadata.get(alias) is not expected_value or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if handoff.get("retry_safe") is not retry_safe:
        raise SystemExit(f"{label} has wrong retry safety: {handoff}")
    if handoff.get("next_safe_command") != "translate <text> to <language>":
        raise SystemExit(f"{label} has wrong next safe command: {handoff}")
    expected_commands = [handoff["next_safe_command"]]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} safe-command list parity failed: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} safe-command count failed: {handoff}")
    if metadata.get("translate_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("translate_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("translate_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command count alias parity failed: {metadata} vs {handoff}")
    boundaries = handoff.get("boundaries")
    expected = {
        "calls_model": False,
        "calls_external_service": calls_external_service,
        "calls_external_services": calls_external_service,
        "executes_tools": False,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        **NO_AUTHORITY_FLAGS,
    }
    if boundaries != expected:
        raise SystemExit(f"{label} has wrong boundaries: {boundaries}")
    if metadata.get("translate_boundaries") != boundaries:
        raise SystemExit(f"{label} boundary alias parity failed: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not calls_external_service or metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} flat external-call flag should match handoff: {metadata}")
    combined = f"{metadata}\n{handoff}".lower()
    for fragment in ("/users/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in combined:
            raise SystemExit(f"{label} leaked a local path fragment {fragment!r}: {metadata} / {handoff}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["translate"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("translate should be LOCAL_SAFE")


def test_translate_argument_contract_fences_malformed_input() -> None:
    fetch_calls: list[tuple[str, str, str]] = []
    original_fetch = tc._fetch

    def mocked_fetch(phrase: str, source: str, target: str) -> dict:
        fetch_calls.append((phrase, source, target))
        return {"responseData": {"translatedText": "mock translation"}}

    try:
        tc._fetch = mocked_fetch  # type: ignore[assignment]
        tool = _tools()["translate"]
        contract = tool.argument_contract
        actual_shape = (
            tuple(
                (
                    field.name,
                    tuple(sorted(kind.value for kind in field.types)),
                    field.required,
                    field.minimum,
                    field.maximum,
                )
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        expected_shape = tuple(
            (name, ("string",), False, None, None)
            for name in ("text", "phrase", "request", "from", "to")
        )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
            or actual_shape != expected_shape
        ):
            raise SystemExit(f"translate should have an exact strict argument contract: {contract}")

        registry = ToolRegistry()
        registry.register(tool)
        executor = Executor(registry, PermissionPolicy())
        private_sentinel = "private-translate-contract-sentinel"
        malformed_args: list[object] = [
            {"text": {"value": private_sentinel}},
            {"phrase": [private_sentinel]},
            {"request": True},
            {"from": 123},
            {"to": False},
            {"unknown": private_sentinel},
            ["text", private_sentinel],
        ]
        for args in malformed_args:
            result = executor.execute(
                PlannedAction("translate", args, "translate contract smoke")  # type: ignore[arg-type]
            )
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
            ):
                raise SystemExit(f"malformed translate arguments crossed the executor fence: {args!r} -> {result}")
            if private_sentinel in f"{result.output}\n{result.metadata}":
                raise SystemExit(f"translate rejection leaked an argument value: {result}")
        if fetch_calls:
            raise SystemExit(f"malformed translate arguments reached the network seam: {fetch_calls}")

        valid_args = (
            {"request": "translate hello to korean"},
            {"text": "hello", "to": "korean"},
            {"phrase": "안녕", "from": "korean", "to": "english"},
        )
        for args in valid_args:
            result = executor.execute(PlannedAction("translate", args, "valid translate contract smoke"))
            if not result.ok:
                raise SystemExit(f"valid translate arguments were rejected: {args!r} -> {result}")
        if len(fetch_calls) != len(valid_args):
            raise SystemExit(f"valid translate arguments did not each reach the mocked fetch seam: {fetch_calls}")
    finally:
        tc._fetch = original_fetch  # type: ignore[assignment]


def test_parse_targets_and_detection() -> None:
    cases = {
        "translate hello to korean": ("hello", "en", "ko"),
        "how do you say thank you in korean": ("thank you", "en", "ko"),
        "how to say good morning in korean": ("good morning", "en", "ko"),
        "say good morning in korean": ("good morning", "en", "ko"),
        "translate 안녕 to english": ("안녕", "ko", "en"),
        "what does 사랑 mean in english": ("사랑", "ko", "en"),
        "korean for good morning": ("good morning", "en", "ko"),
        "spanish for thank you": ("thank you", "en", "es"),
        "english for 사랑": ("사랑", "ko", "en"),
        "hello in korean": ("hello", "en", "ko"),
        "good morning in spanish": ("good morning", "en", "es"),
        "\"thank you\" in french": ("thank you", "en", "fr"),
        "translate hello": ("hello", "en", "ko"),  # default en->ko
    }
    for text, expected in cases.items():
        if tc.parse_translation(text) != expected:
            raise SystemExit(f"parse wrong for {text!r}: {tc.parse_translation(text)}")
    if tc.parse_translation("what time is it") is not None:
        raise SystemExit("non-translation text should not parse")
    if tc.parse_translation("weather in korean") is not None:
        raise SystemExit("bare command-looking phrase should not parse as translation")


def test_translate_with_mocked_api() -> None:
    seen = {}

    def fake_fetch(phrase, source, target):
        seen.update({"phrase": phrase, "source": source, "target": target})
        return {"responseData": {"translatedText": "좋은 아침"}}

    tc._fetch = fake_fetch  # type: ignore
    out = _tools()["translate"].handler({"request": "translate good morning to korean"})
    if not out.ok or "좋은 아침" not in out.output:
        raise SystemExit(f"translate output wrong: {out.output}")
    if seen != {"phrase": "good morning", "source": "en", "target": "ko"}:
        raise SystemExit(f"fetch args wrong: {seen}")
    handoff = _assert_translate_handoff(out.metadata, "translate success", status="ok", source="en", target="ko")
    if handoff.get("translated_text_available") is not True:
        raise SystemExit(f"translate success should mark translated text availability: {handoff}")
    if "good morning" in str(handoff) or "좋은 아침" in str(handoff):
        raise SystemExit(f"translate handoff should not copy input/output text: {handoff}")


def test_translate_avoids_length_mismatched_top_pick() -> None:
    """Real bug, found live via JarvisRuntime() 2026-07-08: MyMemory's free
    translation-memory API tied "hello" -> ko at quality=74 between an unrelated
    sentence ("My name is Azlan.") and the correct "안녕하세요" -- both scored
    identically, with the wrong one ranked first as `responseData.translatedText`.
    This pins the fix using the exact real API shape that exposed the bug."""

    def fake_fetch(phrase, source, target):
        return {
            "responseData": {"translatedText": "제 이름은 Azlan입니다.", "match": 1},
            "matches": [
                {"translation": "제 이름은 Azlan입니다.", "quality": "74", "match": 1},
                {"translation": "안녕하세요", "quality": "74", "match": 1},
                {"translation": "anyo", "quality": "74", "match": 0.99},
            ],
        }

    tc._fetch = fake_fetch  # type: ignore
    out = _tools()["translate"].handler({"request": "translate hello to korean"})
    if not out.ok or "안녕하세요" not in out.output:
        raise SystemExit(f"translate should pick the token-count-plausible alternate: {out.output}")
    if "Azlan" in out.output:
        raise SystemExit(f"translate should not keep the length-mismatched top pick: {out.output}")


def test_translate_keeps_top_pick_when_no_better_alternate_exists() -> None:
    """The re-ranking must not fire when the top pick is fine, or when no
    alternate actually qualifies (lower quality, or also implausible)."""

    def fake_fetch(phrase, source, target):
        return {
            "responseData": {"translatedText": "감사합니다", "match": 1},
            "matches": [
                {"translation": "감사합니다", "quality": "100", "match": 1},
                {"translation": "고맙습니다 정말로 너무너무", "quality": "74", "match": 0.9},
            ],
        }

    tc._fetch = fake_fetch  # type: ignore
    out = _tools()["translate"].handler({"request": "translate thank you to korean"})
    if not out.ok or "감사합니다" not in out.output:
        raise SystemExit(f"translate should keep a perfectly good top pick unchanged: {out.output}")


def test_translate_length_mismatch_check_ignores_longer_source_phrases() -> None:
    """The re-ranking is only for SHORT source phrases (<= 3 tokens) where a much
    longer reply is implausible. A longer source phrase legitimately producing a
    longer translation must not be second-guessed."""

    def fake_fetch(phrase, source, target):
        return {
            "responseData": {
                "translatedText": "이것은 상당히 길고 완전히 그럴듯한 번역 문장입니다 정말로",
                "match": 1,
            },
            "matches": [],
        }

    tc._fetch = fake_fetch  # type: ignore
    out = _tools()["translate"].handler(
        {"request": "translate this is a somewhat long and completely plausible source sentence to korean"}
    )
    if not out.ok or "이것은" not in out.output:
        raise SystemExit(f"translate should not second-guess a long source phrase's long translation: {out.output}")


def test_translate_falls_back_when_top_pick_is_empty() -> None:
    """Real bug found live 2026-07-09: MyMemory's `responseData.translatedText`
    can be empty ("") while `matches[]` still holds a perfectly good candidate
    -- observed for "thank you" -> ja, top pick "" tied at quality=74 with a
    real, correct non-empty match. Previously this always failed with "No
    translation returned" even though a usable answer was sitting right there
    in the same response."""

    def fake_fetch(phrase, source, target):
        return {
            "responseData": {"translatedText": "", "match": 1},
            "matches": [
                {"translation": "", "quality": "74", "match": 1},
                {"translation": "ありがとうございます", "quality": "74", "match": 1},
            ],
        }

    tc._fetch = fake_fetch  # type: ignore
    out = _tools()["translate"].handler({"request": "translate thank you to japanese"})
    if not out.ok or "ありがとうございます" not in out.output:
        raise SystemExit(f"translate should fall back to a real match when the top pick is empty: {out.output}")


def test_translate_stays_refused_when_no_usable_match_exists() -> None:
    """The empty-top-pick fallback must not manufacture an answer when every
    candidate in `matches[]` is also empty/unusable -- still a clean refusal."""

    def fake_fetch(phrase, source, target):
        return {
            "responseData": {"translatedText": "", "match": 1},
            "matches": [{"translation": "", "quality": "74", "match": 1}],
        }

    tc._fetch = fake_fetch  # type: ignore
    out = _tools()["translate"].handler({"request": "translate hello to korean"})
    if out.ok or "No translation returned" not in out.output:
        raise SystemExit(f"translate should still refuse cleanly when no match has real content: {out.output}")


def test_direct_text_is_capped_before_fetch() -> None:
    seen = {}

    def fake_fetch(phrase, source, target):
        seen.update({"phrase": phrase, "source": source, "target": target})
        return {"responseData": {"translatedText": "ok"}}

    tc._fetch = fake_fetch  # type: ignore
    out = _tools()["translate"].handler({"text": "x" * (tc.MAX_TEXT_CHARS + 25), "to": "korean"})
    if not out.ok:
        raise SystemExit(f"direct capped translation should succeed with mock: {out.output}")
    if len(seen.get("phrase", "")) != tc.MAX_TEXT_CHARS:
        raise SystemExit(f"direct text should be capped before fetch: {len(seen.get('phrase', ''))}")
    if seen.get("source") != "en" or seen.get("target") != "ko":
        raise SystemExit(f"direct text language args wrong: {seen}")
    if out.metadata.get("text_chars") != tc.MAX_TEXT_CHARS or not out.metadata.get("text_truncated"):
        raise SystemExit(f"direct capped translation should preserve bounded text metadata: {out.metadata}")
    _assert_translate_handoff(out.metadata, "direct capped translation", status="ok", source="en", target="ko")


def test_unsupported_direct_language_never_fetches() -> None:
    calls = []

    def fail_fetch(phrase, source, target):
        calls.append((phrase, source, target))
        raise AssertionError("unsupported language should not fetch")

    tc._fetch = fail_fetch  # type: ignore
    for args, expected in [
        ({"text": "hello", "to": "klingon"}, "target"),
        ({"text": "hello", "from": "elvish", "to": "korean"}, "source"),
    ]:
        out = _tools()["translate"].handler(args)
        if out.ok or expected not in out.output.lower():
            raise SystemExit(f"unsupported language should be rejected locally: {args!r} -> {out.output}")
        if out.metadata.get("reason") != "unsupported_language":
            raise SystemExit(f"unsupported language should include reason metadata: {out.metadata}")
        _assert_translate_handoff(
            out.metadata,
            f"unsupported {expected}",
            status="refused",
            reason="unsupported_language",
            calls_external_service=False,
        )
    if calls:
        raise SystemExit(f"unsupported language unexpectedly fetched: {calls}")

    path_target = _tools()["translate"].handler({"text": "hello", "to": "/\x55sers/example/private/lang"})
    if path_target.ok or path_target.metadata.get("raw_to") != "<local-path>":
        raise SystemExit(f"unsupported target language should redact path-shaped raw metadata: {path_target.metadata}")
    path_target_handoff = _assert_translate_handoff(
        path_target.metadata,
        "unsupported path target",
        status="refused",
        reason="unsupported_language",
        calls_external_service=False,
    )
    if path_target_handoff.get("raw_to") != "<local-path>":
        raise SystemExit(f"unsupported target should redact path in handoff: {path_target_handoff}")
    path_source = _tools()["translate"].handler({"text": "hello", "from": "/private/tmp/jarvis-translate-lang", "to": "korean"})
    if path_source.ok or path_source.metadata.get("raw_from") != "<local-path>":
        raise SystemExit(f"unsupported source language should redact path-shaped raw metadata: {path_source.metadata}")
    path_source_handoff = _assert_translate_handoff(
        path_source.metadata,
        "unsupported path source",
        status="refused",
        reason="unsupported_language",
        calls_external_service=False,
    )
    if path_source_handoff.get("raw_from") != "<local-path>":
        raise SystemExit(f"unsupported source should redact path in handoff: {path_source_handoff}")
    temp_path_target = _tools()["translate"].handler({"text": "hello", "to": "/var/folders/zc/jarvis-translate-lang"})
    if temp_path_target.ok or temp_path_target.metadata.get("raw_to") != "<local-path>":
        raise SystemExit(f"unsupported temp-root target language should redact path-shaped raw metadata: {temp_path_target.metadata}")
    tmp_path_source = _tools()["translate"].handler({"text": "hello", "from": "/tmp/jarvis-translate-lang", "to": "korean"})
    if tmp_path_source.ok or tmp_path_source.metadata.get("raw_from") != "<local-path>":
        raise SystemExit(f"unsupported tmp-root source language should redact path-shaped raw metadata: {tmp_path_source.metadata}")
    uppercase_path_target = _tools()["translate"].handler({"text": "hello", "to": "/USERS/OPERATOR/PRIVATE/LANG"})
    if uppercase_path_target.ok or uppercase_path_target.metadata.get("raw_to") != "<local-path>":
        raise SystemExit(f"uppercase path-shaped target language should redact raw metadata: {uppercase_path_target.metadata}")
    uppercase_path_handoff = _assert_translate_handoff(
        uppercase_path_target.metadata,
        "unsupported uppercase path target",
        status="refused",
        reason="unsupported_language",
        calls_external_service=False,
    )
    if uppercase_path_handoff.get("raw_to") != "<local-path>":
        raise SystemExit(f"uppercase path target should redact path in handoff: {uppercase_path_handoff}")


def test_path_shaped_translation_text_never_fetches() -> None:
    calls = []

    def fail_fetch(phrase, source, target):
        calls.append((phrase, source, target))
        raise AssertionError("path-shaped translation text should not fetch")

    tc._fetch = fail_fetch  # type: ignore
    cases = [
        {"text": "/\x55sers/example/private/phrase", "to": "korean"},
        {"text": "/private/tmp/jarvis-translate-phrase", "to": "korean"},
        {"request": "translate /var/folders/zc/jarvis-translate-phrase to korean"},
        {"request": "translate /tmp/jarvis-translate-phrase to korean"},
        {"request": "translate /USERS/OPERATOR/PRIVATE/PHRASE to korean"},
    ]
    for args in cases:
        out = _tools()["translate"].handler(args)
        if out.ok or "not a local file path" not in out.output.lower():
            raise SystemExit(f"path-shaped translation text should be rejected locally: {args!r} -> {out.output}")
        if out.metadata.get("reason") != "invalid_text" or out.metadata.get("local_path_text") is not True:
            raise SystemExit(f"path-shaped translation text should expose stable refusal metadata: {out.metadata}")
        if any(fragment in out.output for fragment in ["/\x55sers/", "/private/", "/var/folders", "/tmp/"]):
            raise SystemExit(f"path-shaped translation refusal should not echo raw local paths: {out.output}")
        handoff = _assert_translate_handoff(
            out.metadata,
            f"path-shaped translation text {args!r}",
            status="refused",
            reason="invalid_text",
            calls_external_service=False,
            target="ko",
        )
        if handoff.get("local_path_text") is not True:
            raise SystemExit(f"path-shaped translation text should preserve refusal flag in handoff: {handoff}")
    if calls:
        raise SystemExit(f"path-shaped translation text unexpectedly fetched: {calls}")


def test_handles_error() -> None:
    def boom(phrase, source, target):
        raise RuntimeError("offline")

    tc._fetch = boom  # type: ignore
    out = _tools()["translate"].handler({"request": "translate hi to korean"})
    if out.ok or not out.output.strip():
        raise SystemExit(f"error not handled: {out.output}")
    if "offline" in out.output or out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"error should be friendly with bounded exception metadata: {out.output} {out.metadata}")
    _assert_translate_recovery_message(out.output, "generic translate fetch error")
    _assert_translate_handoff(
        out.metadata,
        "translate fetch error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
        source="en",
        target="ko",
    )

    def server_error(phrase, source, target):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    tc._fetch = server_error  # type: ignore
    server_out = _tools()["translate"].handler({"request": "translate hi to korean"})
    if server_out.ok or server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"HTTP 500 should be handled with bounded exception metadata: {server_out.output} {server_out.metadata}")
    _assert_translate_recovery_message(server_out.output, "translate HTTP 500")
    _assert_translate_handoff(
        server_out.metadata,
        "translate HTTP 500",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
        source="en",
        target="ko",
    )

    def timeout_error(phrase, source, target):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    tc._fetch = timeout_error  # type: ignore
    timeout_out = _tools()["translate"].handler({"request": "translate hi to korean"})
    if timeout_out.ok or timeout_out.metadata.get("exception_type") != "TimeoutError":
        raise SystemExit(f"timeout should be handled with bounded exception metadata: {timeout_out.output} {timeout_out.metadata}")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"timeout recovery should preserve the timeout cause: {timeout_out.output}")
    _assert_translate_recovery_message(timeout_out.output, "translate timeout")
    _assert_translate_handoff(
        timeout_out.metadata,
        "translate timeout",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
        source="en",
        target="ko",
    )


def test_empty_translation_preserves_metadata() -> None:
    def fake_fetch(phrase, source, target):
        return {"responseData": {"translatedText": ""}}

    tc._fetch = fake_fetch  # type: ignore
    out = _tools()["translate"].handler({"request": "translate hello to korean"})
    if out.ok or "No translation returned" not in out.output:
        raise SystemExit(f"empty translation should be a clear refusal: {out.output}")
    _assert_translate_recovery_message(out.output, "empty translation")
    if out.metadata.get("source") != "en" or out.metadata.get("target") != "ko":
        raise SystemExit(f"empty translation metadata should preserve language pair: {out.metadata}")
    if out.metadata.get("text_chars") != len("hello"):
        raise SystemExit(f"empty translation metadata should preserve bounded text length: {out.metadata}")
    handoff = _assert_translate_handoff(
        out.metadata,
        "empty translation",
        status="empty",
        reason="no_translation",
        retry_safe=True,
        source="en",
        target="ko",
    )
    if handoff.get("translated_text_available") is not False:
        raise SystemExit(f"empty translation should mark no translated text: {handoff}")


def test_missing_translation_request_never_fetches() -> None:
    calls = []

    def fail_fetch(phrase, source, target):
        calls.append((phrase, source, target))
        raise AssertionError("missing translation request should not fetch")

    tc._fetch = fail_fetch  # type: ignore
    out = _tools()["translate"].handler({"request": ""})
    if out.ok or "what to translate" not in out.output.lower():
        raise SystemExit(f"missing translation request should ask for text: {out.output}")
    _assert_translate_handoff(
        out.metadata,
        "missing translation request",
        status="refused",
        reason="missing_text",
        calls_external_service=False,
    )
    if calls:
        raise SystemExit(f"missing translation request unexpectedly fetched: {calls}")


def test_planner_routes_translation() -> None:
    p = RuleBasedPlanner()
    for q in [
        "translate hello to korean",
        "how do you say thanks in korean",
        "how to say good morning in korean",
        "say good morning in korean",
        "what is 사랑 in english",
        "what does 사랑 mean in english",
        "korean for good morning",
        "spanish for thank you",
        "english for 사랑",
        "hello in korean",
        "good morning in spanish",
        "\"thank you\" in french",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["translate"]:
            raise SystemExit(f"planner missed translate route: {q!r}")
    unsupported = p.plan("klingon for hello")
    if unsupported and unsupported.actions:
        raise SystemExit("planner should not route unsupported terse language aliases to translate")
    command_like = p.plan("weather in korean")
    if [a.tool_name for a in command_like.actions] == ["translate"]:
        raise SystemExit("planner should not route command-looking bare phrase to translate")


def main() -> None:
    test_is_local_safe()
    test_translate_argument_contract_fences_malformed_input()
    test_parse_targets_and_detection()
    test_translate_with_mocked_api()
    test_translate_avoids_length_mismatched_top_pick()
    test_translate_keeps_top_pick_when_no_better_alternate_exists()
    test_translate_length_mismatch_check_ignores_longer_source_phrases()
    test_translate_falls_back_when_top_pick_is_empty()
    test_translate_stays_refused_when_no_usable_match_exists()
    test_direct_text_is_capped_before_fetch()
    test_unsupported_direct_language_never_fetches()
    test_path_shaped_translation_text_never_fetches()
    test_handles_error()
    test_empty_translation_preserves_metadata()
    test_missing_translation_request_never_fetches()
    test_planner_routes_translation()
    print("Translate connector smoke passed")


if __name__ == "__main__":
    main()
