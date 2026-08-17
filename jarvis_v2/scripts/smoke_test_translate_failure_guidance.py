"""Focused offline proof for translation failure guidance and privacy fences."""

from __future__ import annotations

from typing import Any, Callable

from jarvis_v2.agent.failure_guidance import EXTERNAL_INFORMATION_RECOVERY_ACTION
from jarvis_v2.config import load_config
from jarvis_v2.tools import translate_connector as tc
from jarvis_v2.tools._http import HttpError


PRIVATE_SENTINELS = (
    "/\x55sers/example/private/translation.txt",
    "sk_" + "test_translate_failure_guidance_private_123456",
    "api_key=translate-private-sentinel-123456",
)


def _tool():
    return {tool.name: tool for tool in tc.make_translate_tools(load_config())}["translate"]


def _assert_private_safe(result: Any, label: str) -> None:
    public = f"{result.output}\n{result.metadata}".lower()
    for sentinel in PRIVATE_SENTINELS:
        if sentinel.lower() in public:
            raise SystemExit(f"{label} leaked private-looking input: {result}")
    for fragment in ("/users/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in public:
            raise SystemExit(f"{label} leaked a local path fragment {fragment!r}: {result}")


def _assert_boundaries(result: Any, label: str, *, external: bool) -> None:
    metadata = result.metadata
    expected_false = (
        "calls_model",
        "executes_tools",
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
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    )
    for key in expected_false:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} has unsafe {key}: {metadata}")
    if metadata.get("calls_external_service") is not external:
        raise SystemExit(f"{label} has wrong external-call truth: {metadata}")
    if metadata.get("calls_external_services") is not external:
        raise SystemExit(f"{label} has wrong external-call alias: {metadata}")
    _assert_private_safe(result, label)


def _assert_input_guidance(result: Any, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    _assert_boundaries(result, label, external=False)
    guidance = result.metadata.get("recovery_guidance")
    expected = {
        "version": 1,
        "action": tc.TRANSLATION_INPUT_RECOVERY_ACTION,
        "commands": [],
    }
    if guidance != expected or tc.TRANSLATION_INPUT_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} lacks canonical input guidance: {result}")
    if result.metadata.get("next_command") is not None:
        raise SystemExit(f"{label} should not grant a retry command: {result.metadata}")


def _assert_external_guidance(result: Any, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    _assert_boundaries(result, label, external=True)
    guidance = result.metadata.get("recovery_guidance")
    expected = {
        "version": 1,
        "action": EXTERNAL_INFORMATION_RECOVERY_ACTION,
        "commands": ["setup check"],
    }
    if guidance != expected or EXTERNAL_INFORMATION_RECOVERY_ACTION not in result.output:
        raise SystemExit(f"{label} lacks canonical external guidance: {result}")
    if result.metadata.get("next_command") != "setup check":
        raise SystemExit(f"{label} lacks the bounded diagnostic command: {result.metadata}")
    for key, expected_value in {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }.items():
        if result.metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} has wrong {key}: {result.metadata}")


def _with_fetch(mock: Callable[[str, str, str], dict[str, Any]], args: dict[str, Any]):
    original = tc._fetch
    try:
        tc._fetch = mock  # type: ignore[assignment]
        return _tool().handler(args)
    finally:
        tc._fetch = original  # type: ignore[assignment]


def test_local_input_failures_never_fetch() -> None:
    calls: list[tuple[str, str, str]] = []

    def forbidden_fetch(phrase: str, source: str, target: str) -> dict[str, Any]:
        calls.append((phrase, source, target))
        raise AssertionError("local input refusal crossed the network seam")

    cases = (
        ({}, "missing text"),
        ({"text": "hello", "to": "klingon"}, "unsupported target"),
        ({"text": "hello", "from": "elvish", "to": "korean"}, "unsupported source"),
        ({"text": PRIVATE_SENTINELS[0], "to": "korean"}, "path-shaped text"),
        ({"text": PRIVATE_SENTINELS[1], "to": "korean"}, "secret-shaped text"),
        ({"text": PRIVATE_SENTINELS[2], "to": "korean"}, "key-shaped text"),
        ({"text": "hello", "to": PRIVATE_SENTINELS[2]}, "private target language"),
    )
    original = tc._fetch
    try:
        tc._fetch = forbidden_fetch  # type: ignore[assignment]
        for args, label in cases:
            result = _tool().handler(args)
            _assert_input_guidance(result, label)
    finally:
        tc._fetch = original  # type: ignore[assignment]
    if calls:
        raise SystemExit(f"input failures reached the network seam: {calls}")


def test_oversized_input_stays_bounded() -> None:
    seen: list[tuple[str, str, str]] = []

    def successful_fetch(phrase: str, source: str, target: str) -> dict[str, Any]:
        seen.append((phrase, source, target))
        return {"responseData": {"translatedText": "bounded result"}}

    raw = "x" * (tc.MAX_TEXT_CHARS + 137)
    result = _with_fetch(successful_fetch, {"text": raw, "to": "korean"})
    if not result.ok or len(seen) != 1 or len(seen[0][0]) != tc.MAX_TEXT_CHARS:
        raise SystemExit(f"oversized translation did not stay bounded: {seen} {result}")
    if result.metadata.get("text_chars") != tc.MAX_TEXT_CHARS:
        raise SystemExit(f"oversized translation reported the wrong bounded length: {result.metadata}")
    if result.metadata.get("text_truncated") is not True:
        raise SystemExit(f"oversized translation omitted truncation truth: {result.metadata}")
    _assert_boundaries(result, "oversized translation", external=True)


def test_external_failures_are_canonical_and_offline() -> None:
    def api_failure(_phrase: str, _source: str, _target: str) -> dict[str, Any]:
        raise HttpError(
            503,
            "private upstream body " + "sk_" + "test_translate_failure_guidance_private_123456",
        )

    def parse_failure(_phrase: str, _source: str, _target: str) -> dict[str, Any]:
        raise ValueError("/\x55sers/example/private/translation-response.json")

    def malformed_payload(_phrase: str, _source: str, _target: str) -> dict[str, Any]:
        return {"responseData": "not-an-object"}

    def no_result(_phrase: str, _source: str, _target: str) -> dict[str, Any]:
        return {"responseData": {"translatedText": ""}, "matches": []}

    for mock, label in (
        (api_failure, "API failure"),
        (parse_failure, "parse failure"),
        (malformed_payload, "malformed payload"),
        (no_result, "no result"),
    ):
        result = _with_fetch(mock, {"text": "hello", "to": "korean"})
        _assert_external_guidance(result, label)


def test_success_preserves_translation_and_privacy_contract() -> None:
    def successful_fetch(phrase: str, source: str, target: str) -> dict[str, Any]:
        if (phrase, source, target) != ("hello", "en", "ko"):
            raise AssertionError((phrase, source, target))
        return {"responseData": {"translatedText": "안녕하세요"}}

    result = _with_fetch(successful_fetch, {"text": "hello", "to": "korean"})
    if not result.ok or result.output != "hello → 안녕하세요":
        raise SystemExit(f"successful translation changed behavior: {result}")
    if result.metadata.get("recovery_guidance") is not None:
        raise SystemExit(f"successful translation should not declare failure guidance: {result.metadata}")
    handoff = result.metadata.get("translate_handoff")
    if not isinstance(handoff, dict) or handoff.get("translated_text_available") is not True:
        raise SystemExit(f"successful translation lacks bounded handoff truth: {result.metadata}")
    if handoff.get("input_text_in_metadata") is not False or handoff.get("translated_text_in_metadata") is not False:
        raise SystemExit(f"successful translation copied content into metadata: {handoff}")
    _assert_boundaries(result, "successful translation", external=True)


def main() -> None:
    test_local_input_failures_never_fetch()
    test_oversized_input_stays_bounded()
    test_external_failures_are_canonical_and_offline()
    test_success_preserves_translation_and_privacy_contract()
    print("Translate failure guidance smoke passed")


if __name__ == "__main__":
    main()
