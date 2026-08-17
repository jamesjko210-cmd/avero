"""Smoke tests for model-compose plus writer handoff."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.model_provider import ModelProviderError
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import compose_connector as cc
from jarvis_v2.tools import writer_connector as wc


NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _tools(config=None):
    return {tool.name: tool for tool in cc.make_compose_tools(config or load_config())}


def _assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def _assert_compose_handoff(
    metadata: dict,
    label: str,
    *,
    status: str,
    reason: str = "",
    calls_model: bool = False,
    calls_external_service: bool = False,
    executes_tools: bool = False,
    controls_computer: bool = False,
) -> dict:
    handoff = metadata.get("compose_write_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed compose handoff: {metadata}")
    if handoff.get("source") != "compose_and_write" or handoff.get("status") != status:
        raise SystemExit(f"{label} compose handoff source/status wrong: {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} compose flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} compose handoff {key} should be {expected_value}: {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} compose handoff reason wrong: {handoff}")
    if handoff.get("prompt_chars") != metadata.get("prompt_chars", handoff.get("prompt_chars")):
        raise SystemExit(f"{label} compose handoff prompt length parity failed: {metadata}")
    if handoff.get("generated_content_in_metadata") is not False:
        raise SystemExit(f"{label} compose handoff must not copy generated content: {handoff}")
    if handoff.get("approval_required_before_execution") is not controls_computer:
        raise SystemExit(f"{label} compose handoff approval flag wrong: {handoff}")
    if handoff.get("manual_review_required") is not controls_computer:
        raise SystemExit(f"{label} compose handoff manual review flag wrong: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "calls_model": calls_model,
        "calls_external_service": calls_external_service,
        "executes_tools": executes_tools,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": controls_computer,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": controls_computer,
        "controls_computer": controls_computer,
        **NO_AUTHORITY_FLAGS,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} compose handoff boundary {key} should be {value}: {handoff}")
        if metadata.get(key, value) is not value:
            raise SystemExit(f"{label} compose flat boundary {key} should be {value}: {metadata}")
    _assert_no_local_path(handoff, f"{label} compose handoff")
    return handoff


def _install_writer_fakes(calls: list, sleeps: list | None = None):
    change_count = 10
    clipboard_text = "user clipboard"

    def fake_run(command, **kwargs):
        nonlocal change_count, clipboard_text
        argv = list(command)
        calls.append(argv)
        if argv == wc.clipboard_safety._CHANGE_COUNT_COMMAND:
            return type(
                "FakeProc",
                (),
                {"returncode": 0, "stdout": str(change_count), "stderr": ""},
            )()
        if argv == ["osascript", "-e", "clipboard info"]:
            return type(
                "FakeProc",
                (),
                {
                    "returncode": 0,
                    "stdout": "«class ut16», 8, «class utf8», 4, string, 4",
                    "stderr": "",
                },
            )()
        if argv == ["pbpaste"]:
            return type(
                "FakeProc",
                (),
                {"returncode": 0, "stdout": clipboard_text, "stderr": ""},
            )()
        if wc.clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv:
            if int(argv[-1]) != change_count:
                return type(
                    "FakeProc",
                    (),
                    {"returncode": 0, "stdout": "STALE", "stderr": ""},
                )()
            clipboard_text = kwargs.get("input", "")
            change_count += 1
            return type(
                "FakeProc",
                (),
                {
                    "returncode": 0,
                    "stdout": f"WRITTEN:{change_count}",
                    "stderr": "",
                },
            )()
        return type(
            "FakeProc",
            (),
            {"returncode": 0, "stdout": "", "stderr": ""},
        )()

    wc.subprocess.run = fake_run  # type: ignore[assignment]
    wc.time.sleep = (lambda seconds: sleeps.append(seconds)) if sleeps is not None else (lambda *_: None)  # type: ignore[assignment]


def test_tool_is_high_risk() -> None:
    tool = _tools()["compose_and_write"]
    if tool.risk != RiskLevel.HIGH_RISK or tool.toolset != "personal":
        raise SystemExit("compose_and_write must be HIGH_RISK in the personal toolset.")
    missing = tool.handler({"prompt": ""})
    if missing.ok or missing.metadata.get("controls_computer") or missing.metadata.get("executes_tools"):
        raise SystemExit(f"Missing prompt should fail before writer handoff: {missing.metadata}")
    _assert_compose_handoff(missing.metadata, "missing prompt", status="refused", reason="missing_prompt")


def test_generate_then_human_write() -> None:
    calls: list = []
    sleeps: list = []
    old_generate = cc._generate_text
    old_run = wc.subprocess.run
    old_sleep = wc.time.sleep
    try:
        cc._generate_text = lambda _config, prompt: f"Generated from: {prompt}"  # type: ignore[assignment]
        _install_writer_fakes(calls, sleeps)
        out = _tools()["compose_and_write"].handler(
            {"prompt": "say hello", "mode": "human", "speed": "fast", "countdown": 0}
        )
    finally:
        cc._generate_text = old_generate  # type: ignore[assignment]
        wc.subprocess.run = old_run  # type: ignore[assignment]
        wc.time.sleep = old_sleep  # type: ignore[assignment]
    if not out.ok or out.metadata.get("writer_tool") != "human_write":
        raise SystemExit(f"compose_and_write human mode failed: {out.output} {out.metadata}")
    if not calls or sleeps[:1] != [0]:
        raise SystemExit(f"compose_and_write did not hand generated text to human_write: calls={calls}, sleeps={sleeps}")
    if not out.metadata.get("calls_model") or not out.metadata.get("controls_computer") or not out.metadata.get("requires_approval"):
        raise SystemExit(f"compose_and_write metadata missed model/control/approval flags: {out.metadata}")
    handoff = _assert_compose_handoff(
        out.metadata,
        "human compose",
        status="writer_ok",
        calls_model=True,
        executes_tools=True,
        controls_computer=True,
    )
    if handoff.get("writer_tool") != "human_write" or handoff.get("writer_ok") is not True:
        raise SystemExit(f"human compose handoff should preserve writer result: {handoff}")
    if handoff.get("generated_chars") != out.metadata.get("generated_chars"):
        raise SystemExit(f"human compose handoff generated char count diverged: {handoff}")


def test_generate_then_paste() -> None:
    calls: list = []
    old_generate = cc._generate_text
    old_run = wc.subprocess.run
    old_sleep = wc.time.sleep
    try:
        cc._generate_text = lambda *_: "Paste this generated block."  # type: ignore[assignment]
        _install_writer_fakes(calls)
        out = _tools()["compose_and_write"].handler({"prompt": "paste block", "mode": "paste", "countdown": 0})
    finally:
        cc._generate_text = old_generate  # type: ignore[assignment]
        wc.subprocess.run = old_run  # type: ignore[assignment]
        wc.time.sleep = old_sleep  # type: ignore[assignment]
    if not out.ok or out.metadata.get("writer_tool") != "paste_text":
        raise SystemExit(f"compose_and_write paste mode failed: {out.output} {out.metadata}")
    conditional_writes = [
        call
        for call in calls
        if wc.clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in call
    ]
    if len(conditional_writes) != 2 or any(call and call[0] == "pbcopy" for call in calls):
        raise SystemExit(
            "compose_and_write paste mode did not use the guarded shared "
            f"clipboard transaction: {calls}"
        )
    handoff = _assert_compose_handoff(
        out.metadata,
        "paste compose",
        status="writer_ok",
        calls_model=True,
        executes_tools=True,
        controls_computer=True,
    )
    if handoff.get("writer_tool") != "paste_text" or handoff.get("mode") != "paste":
        raise SystemExit(f"paste compose handoff should preserve writer/mode: {handoff}")


def test_openai_compose_uses_shared_provider_and_preserves_unicode() -> None:
    config = replace(
        load_config(),
        model_provider="openai",
        chat_model="gpt-5.6-terra",
        chat_reasoning_effort="medium",
        chat_timeout_seconds=60.0,
    )
    calls: list = []
    captured: dict = {}
    old_generate = cc.generate_model_text
    old_run = wc.subprocess.run
    old_sleep = wc.time.sleep
    try:
        def fake_generate(**kwargs):
            captured.update(kwargs)
            return "안녕하세요. 만나서 반갑습니다."

        cc.generate_model_text = fake_generate  # type: ignore[assignment]
        _install_writer_fakes(calls)
        out = _tools(config)["compose_and_write"].handler(
            {"prompt": "가상연락처이에게 따뜻한 인사말을 써줘", "mode": "paste", "countdown": 0}
        )
    finally:
        cc.generate_model_text = old_generate  # type: ignore[assignment]
        wc.subprocess.run = old_run  # type: ignore[assignment]
        wc.time.sleep = old_sleep  # type: ignore[assignment]

    if not out.ok or captured.get("provider") != "openai" or captured.get("model") != "gpt-5.6-terra":
        raise SystemExit(f"OpenAI compose missed shared provider routing: {captured} {out}")
    if captured.get("reasoning_effort") != "medium" or captured.get("timeout_seconds") != 60.0:
        raise SystemExit(f"OpenAI compose missed bounded model settings: {captured}")
    if captured.get("max_output_tokens") != config.openai_max_output_tokens:
        raise SystemExit(f"OpenAI compose missed its total reasoning/output ceiling: {captured}")
    messages = captured.get("messages") or []
    if not messages or messages[-1].get("content") != "가상연락처이에게 따뜻한 인사말을 써줘":
        raise SystemExit(f"OpenAI compose did not preserve Korean input: {messages}")
    if out.metadata.get("model_provider") != "openai" or out.metadata.get("calls_external_service") is not True:
        raise SystemExit(f"OpenAI compose metadata missed remote model boundary: {out.metadata}")
    handoff = _assert_compose_handoff(
        out.metadata,
        "OpenAI compose",
        status="writer_ok",
        calls_model=True,
        calls_external_service=True,
        executes_tools=True,
        controls_computer=True,
    )
    if handoff.get("model_provider") != "openai":
        raise SystemExit(f"OpenAI compose handoff missed provider: {handoff}")


def test_openai_compose_failure_names_openai_fix_and_never_writes() -> None:
    config = replace(
        load_config(),
        model_provider="openai",
        chat_model="gpt-5.6-terra",
        chat_timeout_seconds=60.0,
    )
    calls: list = []
    old_generate = cc.generate_model_text
    old_run = wc.subprocess.run
    old_sleep = wc.time.sleep
    try:
        def fail_generate(**_kwargs):
            raise ModelProviderError(
                "openai_api_key_missing",
                "Set OPENAI_API_KEY in Jarvis's local environment, then retry `model routing status`.",
            )

        cc.generate_model_text = fail_generate  # type: ignore[assignment]
        _install_writer_fakes(calls)
        out = _tools(config)["compose_and_write"].handler(
            {"prompt": "draft a short hello", "mode": "human", "countdown": 0}
        )
    finally:
        cc.generate_model_text = old_generate  # type: ignore[assignment]
        wc.subprocess.run = old_run  # type: ignore[assignment]
        wc.time.sleep = old_sleep  # type: ignore[assignment]

    if out.ok or calls:
        raise SystemExit(f"OpenAI compose failure must not type or paste: calls={calls}, result={out}")
    for expected in ("OPENAI_API_KEY", "model routing status", "compose_model_unavailable", "did not type or paste"):
        if expected not in out.output:
            raise SystemExit(f"OpenAI compose failure missed recovery {expected!r}: {out.output}")
    for forbidden in ("start Ollama", "ollama pull", "local model"):
        if forbidden in out.output or forbidden in str(out.metadata.get("model_recovery_hint") or ""):
            raise SystemExit(f"OpenAI compose failure emitted contradictory recovery {forbidden!r}: {out}")
    if out.metadata.get("calls_external_service") is not True or out.metadata.get("controls_computer") is not False:
        raise SystemExit(f"OpenAI compose failure boundaries are inaccurate: {out.metadata}")
    _assert_compose_handoff(
        out.metadata,
        "OpenAI compose failure",
        status="failed",
        reason="model_error",
        calls_model=True,
        calls_external_service=True,
    )


def test_model_failure_does_not_type() -> None:
    calls: list = []
    old_generate = cc._generate_text
    old_run = wc.subprocess.run
    old_sleep = wc.time.sleep
    try:
        def boom(_config, _prompt):
            raise RuntimeError("model unavailable near /\x55sers/example/private/compose-model")

        cc._generate_text = boom  # type: ignore[assignment]
        _install_writer_fakes(calls)
        out = _tools()["compose_and_write"].handler({"prompt": "should not type", "mode": "human", "countdown": 0})
    finally:
        cc._generate_text = old_generate  # type: ignore[assignment]
        wc.subprocess.run = old_run  # type: ignore[assignment]
        wc.time.sleep = old_sleep  # type: ignore[assignment]
    if out.ok or out.metadata.get("reason") != "model_error":
        raise SystemExit(f"Model failure should return a clean compose error: {out.output} {out.metadata}")
    if "model unavailable near" in out.output or "Compose error" in out.output:
        raise SystemExit(f"Model failure should not leak the raw model exception: {out.output}")
    for expected in (
        "model routing status",
        "start Ollama",
        "ollama pull",
        "Diagnostic: compose_model_unavailable",
        "I did not type or paste anything",
    ):
        if expected not in out.output:
            raise SystemExit(f"Model failure should name compose model recovery step {expected!r}: {out.output}")
    for forbidden in ("/\x55sers/", "/private/", "compose-model", "Traceback"):
        if forbidden in out.output or forbidden in str(out.metadata):
            raise SystemExit(f"Model failure leaked raw exception/local path {forbidden!r}: {out.output} {out.metadata}")
    if out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"Model failure should preserve bounded diagnostic metadata: {out.metadata}")
    if out.metadata.get("diagnostic") != "compose_model_unavailable":
        raise SystemExit(f"Model failure should preserve compose diagnostic metadata: {out.metadata}")
    recovery_hint = out.metadata.get("model_recovery_hint", "")
    for expected in ("model routing status", "start Ollama", "ollama pull", "compose_model_unavailable"):
        if expected not in recovery_hint:
            raise SystemExit(f"Model recovery hint metadata missed {expected!r}: {out.metadata}")
    if calls or out.metadata.get("controls_computer") or out.metadata.get("executes_tools"):
        raise SystemExit(f"Model failure must not type or paste: calls={calls}, metadata={out.metadata}")
    _assert_compose_handoff(out.metadata, "model failure", status="failed", reason="model_error", calls_model=True)


def test_path_shaped_prompt_never_generates_or_writes() -> None:
    path_samples = [
        "/\x55sers/example/Desktop/private-note",
        "/private/tmp/jarvis-secret",
        "/var/folders/zc/jarvis-secret",
        "/tmp/jarvis-secret",
    ]
    old_generate = cc._generate_text
    generated_prompts: list[str] = []

    def fake_generate(_config, prompt):
        generated_prompts.append(prompt)
        return "should not happen"

    try:
        cc._generate_text = fake_generate  # type: ignore[assignment]
        for sample in path_samples:
            out = _tools()["compose_and_write"].handler({"prompt": f"write about {sample}", "mode": "paste", "countdown": 0})
            if out.ok or out.metadata.get("reason") != "invalid_prompt" or not out.metadata.get("local_path_prompt"):
                raise SystemExit(f"compose should reject local-path-shaped prompts: {sample} -> {out.output} {out.metadata}")
            if out.metadata.get("calls_model") or out.metadata.get("executes_tools") or out.metadata.get("controls_computer"):
                raise SystemExit(f"compose local-path prompt must not model-call or write: {out.metadata}")
            if sample in out.output or sample in str(out.metadata):
                raise SystemExit(f"compose should not echo raw local path: {out.output} {out.metadata}")
            handoff = _assert_compose_handoff(out.metadata, "path-shaped prompt", status="refused", reason="invalid_prompt")
            if handoff.get("prompt_preview") != "write about <local-path>" or handoff.get("local_path_prompt") is not True:
                raise SystemExit(f"compose handoff should redact path-shaped prompt: {handoff}")
    finally:
        cc._generate_text = old_generate  # type: ignore[assignment]
    if generated_prompts:
        raise SystemExit(f"compose local-path prompts should fail before generation: {generated_prompts}")


def test_empty_generation_does_not_type() -> None:
    calls: list = []
    old_generate = cc._generate_text
    old_run = wc.subprocess.run
    old_sleep = wc.time.sleep
    try:
        cc._generate_text = lambda *_: "   "  # type: ignore[assignment]
        _install_writer_fakes(calls)
        out = _tools()["compose_and_write"].handler({"prompt": "draft empty", "mode": "paste", "countdown": 0})
    finally:
        cc._generate_text = old_generate  # type: ignore[assignment]
        wc.subprocess.run = old_run  # type: ignore[assignment]
        wc.time.sleep = old_sleep  # type: ignore[assignment]
    if out.ok or out.metadata.get("reason") != "empty_generation":
        raise SystemExit(f"empty generation should fail cleanly: {out.output} {out.metadata}")
    if calls or out.metadata.get("controls_computer") or out.metadata.get("executes_tools"):
        raise SystemExit(f"empty generation must not type or paste: calls={calls}, metadata={out.metadata}")
    _assert_compose_handoff(out.metadata, "empty generation", status="failed", reason="empty_generation", calls_model=True)


def test_malformed_args_are_bounded_and_do_not_crash() -> None:
    calls: list = []
    old_generate = cc._generate_text
    old_run = wc.subprocess.run
    old_sleep = wc.time.sleep
    try:
        cc._generate_text = lambda _config, prompt: f"Generated from malformed prompt: {prompt}"  # type: ignore[assignment]
        _install_writer_fakes(calls)
        out = _tools()["compose_and_write"].handler(
            {"prompt": ["hello", {"nested": "value"}], "mode": "/\x55sers/example/private/compose-mode", "countdown": False, "speed": {"bad": "value"}}
        )
    finally:
        cc._generate_text = old_generate  # type: ignore[assignment]
        wc.subprocess.run = old_run  # type: ignore[assignment]
        wc.time.sleep = old_sleep  # type: ignore[assignment]
    if not out.ok or out.metadata.get("writer_tool") != "human_write":
        raise SystemExit(f"malformed compose args should fall back cleanly: {out.output} {out.metadata}")
    if not out.metadata.get("raw_mode") or out.metadata.get("prompt_chars", 0) <= 0:
        raise SystemExit(f"malformed compose args should preserve sanitized metadata: {out.metadata}")
    if out.metadata.get("raw_mode") != "<local-path>" or "/\x55sers/" in str(out.metadata):
        raise SystemExit(f"malformed compose args should redact path-shaped raw mode metadata: {out.metadata}")
    if not calls:
        raise SystemExit("malformed compose args should still hand generated text to the safe mocked writer")
    handoff = _assert_compose_handoff(
        out.metadata,
        "malformed compose args",
        status="writer_ok",
        calls_model=True,
        executes_tools=True,
        controls_computer=True,
    )
    if handoff.get("raw_mode") != "<local-path>":
        raise SystemExit(f"malformed compose handoff should preserve redacted raw mode: {handoff}")


def test_planner_routes_and_runtime_gates() -> None:
    planner = RuleBasedPlanner()
    paragraph = planner.plan("write a paragraph about the moon and type it")
    if paragraph.actions[0].tool_name != "compose_and_write":
        raise SystemExit(f"Paragraph compose route missed: {paragraph}")
    if paragraph.actions[0].args.get("prompt") != "Write a paragraph about the moon.":
        raise SystemExit(f"Paragraph prompt wrong: {paragraph.actions[0].args}")
    email = planner.plan("draft an email about tomorrow's launch like a human")
    if email.actions[0].tool_name != "compose_and_write" or "Draft an email" not in email.actions[0].args.get("prompt", ""):
        raise SystemExit(f"Email compose route missed: {email.actions[0].args}")
    docs = planner.plan("compose a quick project intro in google docs")
    if docs.actions[0].tool_name != "compose_and_write" or docs.actions[0].args.get("prompt") != "a quick project intro":
        raise SystemExit(f"Google Docs compose route missed: {docs.actions[0].args}")

    with TemporaryDirectory(prefix="jarvis-compose-runtime-") as temp:
        runtime = make_temp_runtime(Path(temp))
        result = runtime.handle("compose a quick project intro in google docs")
        if "explicit approval required" not in result.response or not runtime.store.list_pending_approvals(limit=10):
            raise SystemExit(f"compose_and_write should stay approval-gated at runtime: {result.response}")


def main() -> None:
    test_tool_is_high_risk()
    test_generate_then_human_write()
    test_generate_then_paste()
    test_openai_compose_uses_shared_provider_and_preserves_unicode()
    test_openai_compose_failure_names_openai_fix_and_never_writes()
    test_model_failure_does_not_type()
    test_path_shaped_prompt_never_generates_or_writes()
    test_empty_generation_does_not_type()
    test_malformed_args_are_bounded_and_do_not_crash()
    test_planner_routes_and_runtime_gates()
    print("Compose connector smoke passed")


if __name__ == "__main__":
    main()
