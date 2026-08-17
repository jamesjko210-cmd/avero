"""Smoke tests for the dictionary connector (mocked fetch, no network)."""

from __future__ import annotations

from typing import Any

from jarvis_v2.agent.executor import Executor
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import PlannedAction, RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import dictionary_connector as dc
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION, ToolRegistry

NO_AUTHORITY_FLAGS = (
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
)


_SAMPLE = [{
    "word": "serendipity",
    "phonetic": "/ser.ən.ˈdɪp.ɪ.ti/",
    "meanings": [{"partOfSpeech": "noun", "definitions": [{"definition": "A fortunate happenstance."}]}],
}]


def _tools():
    return {t.name: t for t in dc.make_dictionary_tools(load_config())}


def _assert_dictionary_recovery_message(output: str, label: str) -> None:
    lowered = output.lower()
    for fragment in [
        "network access to dictionaryapi.dev",
        "setup check",
        "retry in a moment",
    ]:
        if fragment not in lowered:
            raise SystemExit(f"{label} missed actionable recovery guidance {fragment!r}: {output}")
    for forbidden in ["http error", "offline", "traceback", "/users/", "/private/", "/var/folders/", "/tmp/"]:
        if forbidden in lowered:
            raise SystemExit(f"{label} leaked raw backend text {forbidden!r}: {output}")


def _assert_no_local_path(value: Any, label: str) -> None:
    text = str(value)
    if "/\x55sers/" in text or "/private/" in text or "/var/folders" in text or "/tmp/" in text:
        raise SystemExit(f"{label} should not leak local paths: {value!r}")


def _assert_dictionary_handoff(
    metadata: dict[str, Any],
    label: str,
    *,
    status: str,
    reason: str = "",
    calls_external_service: bool = False,
) -> dict[str, Any]:
    if metadata.get("dictionary_handoff_ready") is not True:
        raise SystemExit(f"{label} missed dictionary_handoff_ready: {metadata}")
    handoff = metadata.get("dictionary_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing dictionary_handoff: {metadata}")
    if handoff.get("source") != "define" or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong handoff status/source: {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong handoff reason: {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} handoff should expose nested handoff_ready: {handoff}")
    if handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} handoff should be operator-ready: {handoff}")
    if metadata.get("dictionary_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} should mirror operator readiness: {metadata} vs {handoff}")
    if handoff.get("state_changed") is not False or handoff.get("changed") != []:
        raise SystemExit(f"{label} handoff should declare no state changes: {handoff}")
    if metadata.get("dictionary_state_changed") != handoff.get("state_changed") or metadata.get("dictionary_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} should mirror state-change aliases: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False in flat and nested metadata: {metadata}")
        alias = f"dictionary_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} should mirror no-authority alias {alias}: {metadata} vs {handoff}")
    for key in ("raw_word_length", "word_length", "word_truncated", "local_path_word"):
        if handoff.get(key) != metadata.get(key, False):
            raise SystemExit(f"{label} handoff should mirror {key}: {handoff} vs {metadata}")
    if handoff.get("content_in_handoff") is not False or handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} should mark content exclusion: {handoff}")
    if metadata.get("dictionary_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} should mirror content-in-handoff alias: {metadata} vs {handoff}")
    next_safe_command = handoff.get("next_safe_command")
    if not isinstance(next_safe_command, str) or not next_safe_command:
        raise SystemExit(f"{label} should include a next safe command: {handoff}")
    expected_commands = [next_safe_command]
    if handoff.get("next_safe_commands") != expected_commands or handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} should expose safe-command list/count: {handoff}")
    if metadata.get("dictionary_next_safe_command") != next_safe_command:
        raise SystemExit(f"{label} should mirror next safe command: {metadata} vs {handoff}")
    if metadata.get("dictionary_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} should mirror next safe commands: {metadata} vs {handoff}")
    if metadata.get("dictionary_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} should mirror next safe command count: {metadata} vs {handoff}")
    if "count" in metadata and handoff.get("definition_count") != metadata.get("count"):
        raise SystemExit(f"{label} should mirror definition count: {handoff} vs {metadata}")
    if handoff.get("exception_type", "") != metadata.get("exception_type", ""):
        raise SystemExit(f"{label} should mirror exception type: {handoff} vs {metadata}")
    _assert_no_local_path(handoff, label)

    boundaries = handoff.get("boundaries")
    expected = {
        "calls_model": False,
        "calls_external_service": calls_external_service,
        "calls_external_services": calls_external_service,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
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
    }
    if boundaries != expected:
        raise SystemExit(f"{label} has wrong boundaries: {boundaries}")
    if metadata.get("dictionary_boundaries") != boundaries:
        raise SystemExit(f"{label} should mirror dictionary boundaries: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not calls_external_service or metadata.get("calls_external_services") is not calls_external_service:
        raise SystemExit(f"{label} metadata should mirror external-service read state: {metadata}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["define"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("define should be LOCAL_SAFE")


def test_define_argument_contract_fences_malformed_input() -> None:
    fetch_calls: list[str] = []
    original_fetch = dc._fetch

    def mocked_fetch(word: str) -> list:
        fetch_calls.append(word)
        return _SAMPLE

    try:
        dc._fetch = mocked_fetch  # type: ignore[assignment]
        private_sentinel = "private-dictionary-contract-sentinel"
        tool = _tools()["define"]
        contract = tool.argument_contract
        actual_shape = (
            tuple(
                (field.name, tuple(sorted(kind.value for kind in field.types)), field.required)
                for field in contract.fields
            )
            if contract is not None
            else ()
        )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
            or actual_shape != (("word", ("string",), False), ("text", ("string",), False))
        ):
            raise SystemExit(f"define should have an exact strict word/text contract: {contract}")

        registry = ToolRegistry()
        registry.register(tool)
        executor = Executor(registry, PermissionPolicy())
        malformed_args: list[object] = [
            None,
            ["not", "an", "object"],
            {"word": {"private": private_sentinel}},
            {"word": True},
            {"word": [private_sentinel]},
            {"text": 7},
            {"word": "serendipity", "unexpected": private_sentinel},
        ]
        rejected_surfaces: list[object] = []
        for args in malformed_args:
            result = executor.execute(PlannedAction("define", args, "dictionary contract smoke"))  # type: ignore[arg-type]
            if (
                result.ok
                or result.metadata.get("failure_kind") != "tool_arguments_invalid"
                or result.metadata.get("handler_invoked") is not False
                or result.metadata.get("executed_handler") is not False
                or result.metadata.get("requires_confirmation") is not False
            ):
                raise SystemExit(f"malformed dictionary input passed the executor: {result}")
            rejected_surfaces.append({"output": result.output, "metadata": result.metadata})
        if fetch_calls:
            raise SystemExit(f"malformed dictionary input reached dictionaryapi.dev: {fetch_calls}")
        if private_sentinel in str(rejected_surfaces):
            raise SystemExit("dictionary argument rejection leaked a supplied private value")

        for args in ({"word": "serendipity"}, {"text": "serendipity"}):
            result = executor.execute(PlannedAction("define", args, "valid dictionary contract smoke"))
            if not result.ok or fetch_calls[-1] != "serendipity":
                raise SystemExit(f"valid dictionary input did not preserve the word alias: {args} {result}")
        empty = executor.execute(PlannedAction("define", {}, "empty dictionary contract smoke"))
        if (
            empty.ok
            or empty.metadata.get("handler_invoked") is not True
            or "which word should i define" not in empty.output.lower()
        ):
            raise SystemExit(f"empty dictionary compatibility changed: {empty}")
    finally:
        dc._fetch = original_fetch  # type: ignore[assignment]


def test_defines_word() -> None:
    dc._fetch = lambda word: _SAMPLE  # type: ignore
    out = _tools()["define"].handler({"word": "serendipity"})
    if not out.ok or "serendipity" not in out.output or "fortunate happenstance" not in out.output.lower():
        raise SystemExit(f"define output wrong: {out.output}")
    if "(noun)" not in out.output:
        raise SystemExit("define should include part of speech")
    handoff = _assert_dictionary_handoff(out.metadata, "define success", status="ok", calls_external_service=True)
    if handoff.get("word") != "serendipity" or handoff.get("definition_count") != 1:
        raise SystemExit(f"define success should include word/count handoff: {handoff}")


def test_strips_polite_suffix_from_word() -> None:
    seen = []

    def fake_fetch(word):
        seen.append(word)
        return _SAMPLE

    dc._fetch = fake_fetch  # type: ignore
    out = _tools()["define"].handler({"word": "serendipity please"})
    if not out.ok or "fortunate happenstance" not in out.output.lower():
        raise SystemExit(f"polite define word should succeed: {out.output} {out.metadata}")
    if seen != ["serendipity"]:
        raise SystemExit(f"polite define word should fetch cleaned word: {seen}")
    if out.metadata.get("raw_word_length") != len("serendipity please") or out.metadata.get("word_length") != len("serendipity"):
        raise SystemExit(f"polite define word should preserve raw and cleaned lengths: {out.metadata}")
    handoff = _assert_dictionary_handoff(out.metadata, "polite define word", status="ok", calls_external_service=True)
    if handoff.get("raw_word_preview") != "serendipity please" or handoff.get("word") != "serendipity":
        raise SystemExit(f"polite define handoff should preserve raw preview and clean word: {handoff}")


def test_unknown_word_is_clean() -> None:
    def not_found(word):
        raise RuntimeError("HTTP Error 404: Not Found")

    dc._fetch = not_found  # type: ignore
    out = _tools()["define"].handler({"word": "zzxqq"})
    if not out.ok or "No definition found" not in out.output:
        raise SystemExit(f"unknown word should be a clean message: {out.output}")
    _assert_dictionary_handoff(
        out.metadata,
        "unknown word",
        status="empty",
        reason="not_found",
        calls_external_service=True,
    )


def test_fetch_failure_preserves_bounded_exception_type() -> None:
    def offline(word):
        raise RuntimeError("offline")

    dc._fetch = offline  # type: ignore
    out = _tools()["define"].handler({"word": "serendipity"})
    if out.ok or not out.output.strip() or "offline" in out.output:
        raise SystemExit(f"fetch failure should stay friendly: {out.output}")
    if out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"fetch failure should preserve bounded exception type: {out.metadata}")
    _assert_dictionary_recovery_message(out.output, "dictionary generic fetch error")
    _assert_dictionary_handoff(
        out.metadata,
        "fetch failure",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
    )

    def server_error(word):
        raise HttpError(500, "HTTP Error 500: Internal Server Error")

    dc._fetch = server_error  # type: ignore
    server_out = _tools()["define"].handler({"word": "serendipity"})
    if server_out.ok or server_out.metadata.get("exception_type") != "HttpError":
        raise SystemExit(f"HTTP 500 should preserve bounded exception metadata: {server_out.output} {server_out.metadata}")
    _assert_dictionary_recovery_message(server_out.output, "dictionary HTTP 500")
    _assert_dictionary_handoff(
        server_out.metadata,
        "HTTP 500 fetch failure",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
    )

    def timeout_error(word):
        raise TimeoutError("request timed out near /\x55sers/example/private")

    dc._fetch = timeout_error  # type: ignore
    timeout_out = _tools()["define"].handler({"word": "serendipity"})
    if timeout_out.ok or timeout_out.metadata.get("exception_type") != "TimeoutError":
        raise SystemExit(f"timeout should preserve bounded exception metadata: {timeout_out.output} {timeout_out.metadata}")
    if "timed out" not in timeout_out.output.lower():
        raise SystemExit(f"timeout recovery should preserve timeout cause: {timeout_out.output}")
    _assert_dictionary_recovery_message(timeout_out.output, "dictionary timeout")
    _assert_dictionary_handoff(
        timeout_out.metadata,
        "timeout fetch failure",
        status="unavailable",
        reason="fetch_error",
        calls_external_service=True,
    )


def test_requires_word() -> None:
    out = _tools()["define"].handler({"word": ""})
    if out.ok or "which word" not in out.output.lower():
        raise SystemExit(f"empty word should ask: {out.output}")
    if out.metadata.get("raw_word_length") != 0 or out.metadata.get("max_word_chars") != dc.MAX_WORD_CHARS:
        raise SystemExit(f"empty word should preserve bounded metadata: {out.metadata}")
    if out.metadata.get("calls_external_service") is not False:
        raise SystemExit(f"empty word should not claim an external call: {out.metadata}")
    _assert_dictionary_handoff(
        out.metadata,
        "missing word",
        status="refused",
        reason="missing_word",
        calls_external_service=False,
    )


def test_invalid_word_never_fetches() -> None:
    calls = []

    def fail_fetch(word):
        calls.append(word)
        raise AssertionError("invalid word should not fetch")

    dc._fetch = fail_fetch  # type: ignore
    cases = [
        ("two words", False),
        ("12345", False),
        ("x" * (dc.MAX_WORD_CHARS + 1), False),
        ("/\x55sers/example/private/word", True),
        ("/private/tmp/jarvis/word", True),
        ("/var/folders/zc/jarvis-dictionary-word", True),
        ("/tmp/jarvis-dictionary-word", True),
    ]
    for bad, is_local_path in cases:
        out = _tools()["define"].handler({"word": bad})
        expected = "not a local file path" if is_local_path else "one english word"
        if out.ok or expected not in out.output.lower():
            raise SystemExit(f"invalid word should be rejected locally: {bad!r} -> {out.output}")
        if out.metadata.get("reason") != "invalid_word":
            raise SystemExit(f"invalid word should include reason metadata: {out.metadata}")
        if bool(out.metadata.get("local_path_word")) != is_local_path:
            raise SystemExit(f"define should mark only path-shaped invalid words: {out.metadata}")
        if out.metadata.get("calls_external_service") is not False:
            raise SystemExit(f"invalid word should not claim an external call: {out.metadata}")
        _assert_dictionary_handoff(
            out.metadata,
            f"invalid word {bad!r}",
            status="refused",
            reason="invalid_word",
            calls_external_service=False,
        )
        if "/\x55sers/" in out.output or "/private/" in out.output or "/var/folders" in out.output or "/tmp/" in out.output:
            raise SystemExit(f"define should not echo local paths: {out.output}")
    if calls:
        raise SystemExit(f"invalid word unexpectedly fetched: {calls}")


def test_planner_routes_define() -> None:
    p = RuleBasedPlanner()
    cases = {
        "define serendipity": "serendipity",
        "define serendipity please": "serendipity",
        "define the word ephemeral": "ephemeral",
        "define word ephemeral": "ephemeral",
        "dictionary serendipity": "serendipity",
        "what does ephemeral mean": "ephemeral",
        "what does ephemeral mean please": "ephemeral",
        "what does api stand for": "api",
        "what does api stand for please": "api",
        "meaning stoic": "stoic",
        "meaning of stoic please": "stoic",
        "meaning for ephemeral": "ephemeral",
        "what is the meaning of serendipity": "serendipity",
        "what is the meaning of serendipity please": "serendipity",
        "what is the definition of ephemeral": "ephemeral",
        "what's the definition of ephemeral please": "ephemeral",
        "definition for ephemeral": "ephemeral",
        "definition of the word ephemeral": "ephemeral",
        "dictionary meaning of serendipity": "serendipity",
        "dictionary definition of serendipity": "serendipity",
        "dictionary lookup serendipity": "serendipity",
        "dictionary look up serendipity": "serendipity",
        "look up the definition of serendipity": "serendipity",
        "lookup the meaning of serendipity": "serendipity",
        "search for the definition of serendipity": "serendipity",
        "look up serendipity in the dictionary": "serendipity",
        # Real gap found live 2026-07-10: "find the definition of X" fell
        # through to chat while the sibling "search for the definition of X"
        # already worked -- the verb alternation only recognized
        # "look up"/"lookup"/"search (for)", not "find".
        "find the definition of serendipity": "serendipity",
        "find the meaning of serendipity": "serendipity",
    }
    for q, expected_word in cases.items():
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["define"]:
            raise SystemExit(f"planner missed define route: {q!r}")
        if actions[0].args.get("word") != expected_word:
            raise SystemExit(f"planner should pass cleaned define word: {q!r} -> {actions[0].args}")
    if [a.tool_name for a in p.plan("look up Ada Lovelace").actions] == ["define"]:
        raise SystemExit("plain lookup should not be hijacked by dictionary routing")
    stand_for_actions = p.plan("what does api key stand for").actions
    if stand_for_actions and stand_for_actions[0].tool_name == "define":
        raise SystemExit(f"multi-word stand-for question should not pass a malformed dictionary word: {stand_for_actions}")
    if [a.tool_name for a in p.plan("what is the weather").actions] != ["get_weather"]:
        raise SystemExit("weather should not be hijacked by dictionary routing")


def main() -> None:
    test_is_local_safe()
    test_define_argument_contract_fences_malformed_input()
    test_defines_word()
    test_strips_polite_suffix_from_word()
    test_unknown_word_is_clean()
    test_fetch_failure_preserves_bounded_exception_type()
    test_requires_word()
    test_invalid_word_never_fetches()
    test_planner_routes_define()
    print("Dictionary connector smoke passed")


if __name__ == "__main__":
    main()
