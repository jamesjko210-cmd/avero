"""Smoke tests for the human-writer connector (subprocess + sleep mocked)."""

from __future__ import annotations

from types import SimpleNamespace

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools import writer_connector as wc


NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


def _tools():
    return {t.name: t for t in wc.make_writer_tools(load_config())}


def _assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def _assert_writer_recovery_message(output: str, label: str) -> None:
    expected_fragments = [
        "target app focused",
        "Accessibility access",
        "System Settings > Privacy & Security > Accessibility",
        "setup check",
        "after approval",
    ]
    for fragment in expected_fragments:
        if fragment not in output:
            raise SystemExit(f"{label} should name recovery step {fragment!r}: {output}")


def _assert_writer_handoff(
    metadata: dict,
    label: str,
    *,
    source: str,
    status: str,
    reason: str = "",
    controls_computer: bool = False,
) -> dict:
    handoff = metadata.get("writer_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed writer handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("status") != status:
        raise SystemExit(f"{label} writer handoff source/status wrong: {handoff}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} writer flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} writer handoff {key} should be {expected_value}: {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} writer handoff reason wrong: {handoff}")
    if handoff.get("chars") != metadata.get("chars", handoff.get("chars")):
        raise SystemExit(f"{label} writer handoff char parity failed: {metadata}")
    if handoff.get("text_content_in_metadata") is not False:
        raise SystemExit(f"{label} writer handoff should not copy text content: {handoff}")
    if handoff.get("local_path_text") is not bool(metadata.get("local_path_text", False)):
        raise SystemExit(f"{label} writer handoff local-path parity failed: {metadata}")
    requires_approval = bool(metadata.get("requires_approval", controls_computer))
    if handoff.get("approval_required_before_execution") is not requires_approval:
        raise SystemExit(f"{label} writer handoff approval flag wrong: {handoff}")
    if handoff.get("manual_review_required") is not requires_approval:
        raise SystemExit(f"{label} writer handoff manual review flag wrong: {handoff}")
    expected_commands = {
        "outcome_unknown": ["recent tool runs", "execution recovery"],
        "failed": ["setup check"],
        "written": ["recent tool runs"],
        "typing_command_completed": ["recent tool runs"],
        "paste_command_completed": ["recent tool runs"],
    }.get(status, ["safe next actions"])
    if (
        handoff.get("next_safe_command") != expected_commands[0]
        or handoff.get("next_safe_commands") != expected_commands
    ):
        raise SystemExit(f"{label} writer recovery commands drifted: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    reads_clipboard = metadata.get("reads_clipboard") is True
    expected = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "reads_personal_data": reads_clipboard,
        "reads_private_data": reads_clipboard,
        "reads_clipboard": reads_clipboard,
        "executes_side_effect": metadata.get("executes_side_effect") is True,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": requires_approval,
        "controls_computer": controls_computer,
        **NO_AUTHORITY_FLAGS,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} writer handoff boundary {key} should be {value}: {handoff}")
        if metadata.get(key, value) is not value:
            raise SystemExit(f"{label} writer flat boundary {key} should be {value}: {metadata}")
    _assert_no_local_path(handoff, f"{label} writer handoff")
    return handoff


def _assert_unknown_writer_failure(out, label: str, *, mode: str, reason: str) -> dict:
    expected_output = wc._post_attempt_writer_recovery_guidance(mode)
    if out.ok or out.output != expected_output:
        raise SystemExit(f"{label} did not expose fixed outcome-unknown guidance: {out}")
    if out.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": expected_output,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {out.metadata}")
    if (
        out.metadata.get("next_command") != "setup check"
        or out.metadata.get("recovery_commands") != ["setup check"]
        or out.metadata.get("outcome_known") is not False
        or out.metadata.get("outcome_unknown") is not True
        or out.metadata.get("execution_outcome_unknown") is not True
        or out.metadata.get("side_effect_possible") is not True
        or out.metadata.get("retry_safe") is not False
        or out.metadata.get("automatic_retry_allowed") is not False
        or out.metadata.get("authorizes_retry") is not False
    ):
        raise SystemExit(f"{label} outcome/retry aliases drifted: {out.metadata}")
    handoff = out.metadata.get("writer_handoff")
    if (
        not isinstance(handoff, dict)
        or handoff.get("status") != "outcome_unknown"
        or handoff.get("reason") != reason
        or handoff.get("next_safe_commands") != ["recent tool runs", "execution recovery"]
    ):
        raise SystemExit(f"{label} handoff overclaimed a failed outcome: {out.metadata}")
    return handoff


def _install_fakes(calls: list, sleeps: list | None = None):
    change_count = 10
    clipboard_text = "user clipboard"

    def fake_run(command, **kwargs):
        nonlocal change_count, clipboard_text
        argv = list(command)
        calls.append(argv)
        if argv == wc.clipboard_safety._CHANGE_COUNT_COMMAND:
            return SimpleNamespace(returncode=0, stdout=str(change_count), stderr="")
        if argv == ["osascript", "-e", "clipboard info"]:
            return SimpleNamespace(
                returncode=0,
                stdout="«class ut16», 8, «class utf8», 4, string, 4",
                stderr="",
            )
        if argv == ["pbpaste"]:
            return SimpleNamespace(returncode=0, stdout=clipboard_text, stderr="")
        if wc.clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv:
            if int(argv[-1]) != change_count:
                return SimpleNamespace(returncode=0, stdout="STALE", stderr="")
            clipboard_text = kwargs.get("input", "")
            change_count += 1
            return SimpleNamespace(
                returncode=0,
                stdout=f"WRITTEN:{change_count}",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    wc.subprocess.run = fake_run  # type: ignore
    wc.time.sleep = (lambda seconds: sleeps.append(seconds)) if sleeps is not None else (lambda *_: None)  # type: ignore


def test_risk_is_high() -> None:
    tools = _tools()
    if tools["human_write"].risk != RiskLevel.HIGH_RISK or tools["paste_text"].risk != RiskLevel.HIGH_RISK:
        raise SystemExit("writer tools must be HIGH_RISK")
    for name in ["human_write", "paste_text"]:
        meta = {"calls_model": False}
        # tools are gated: requires_approval + controls_computer set in metadata
    out = tools["human_write"].handler({"text": ""})
    if out.ok:
        raise SystemExit("empty text should fail")
    _assert_writer_handoff(out.metadata, "empty human_write", source="human_write", status="refused", reason="missing_text")


def test_plan_keystrokes_and_script() -> None:
    actions = wc.plan_keystrokes("Hi.\n", 88, 30, False)
    kinds = [a[0] for a in actions]
    if "enter" not in kinds:
        raise SystemExit("newline should map to an enter keystroke")
    script = wc.build_script(actions)
    if "key code 36" not in script or 'tell application "System Events"' not in script:
        raise SystemExit(f"AppleScript build wrong: {script[:80]}")


def test_typos_produce_correction_triples() -> None:
    wc.random.seed(1)
    # force typos by patching the rate high
    old = wc.TYPO_RATE
    wc.TYPO_RATE = 1.0
    try:
        actions = wc.plan_keystrokes("abc", 88, 30, True)
    finally:
        wc.TYPO_RATE = old
    kinds = [a[0] for a in actions]
    if "backspace" not in kinds:
        raise SystemExit(f"typos should insert backspace corrections: {kinds}")


def test_human_write_runs_with_mocked_io() -> None:
    calls: list = []
    sleeps: list = []
    _install_fakes(calls, sleeps)
    out = _tools()["human_write"].handler({"text": "Hello world.", "speed": "fast", "countdown": 0})
    if not out.ok or "Typing command completed" not in out.output or "confirm the target" not in out.output:
        raise SystemExit(f"human_write failed: {out.output}")
    if not calls:
        raise SystemExit("human_write did not invoke osascript")
    if sleeps[:1] != [0] or out.metadata.get("countdown") != 0:
        raise SystemExit(f"human_write should honor explicit zero countdown: sleeps={sleeps}, metadata={out.metadata}")
    if not out.metadata.get("controls_computer") or not out.metadata.get("requires_approval"):
        raise SystemExit(f"human_write metadata must mark computer control + approval: {out.metadata}")
    human_handoff = _assert_writer_handoff(
        out.metadata,
        "human_write success",
        source="human_write",
        status="typing_command_completed",
        controls_computer=True,
    )
    if human_handoff.get("speed") != "fast" or human_handoff.get("typos") is not True:
        raise SystemExit(f"human_write handoff should preserve speed and typos: {human_handoff}")

    calls.clear()
    sleeps.clear()
    bool_out = _tools()["human_write"].handler({"text": "Hello world.", "speed": "fast", "countdown": True})
    if not bool_out.ok or sleeps[:1] != [wc.DEFAULT_COUNTDOWN_SECONDS]:
        raise SystemExit(f"human_write should treat boolean countdown as malformed default: sleeps={sleeps}, output={bool_out.output}")
    if bool_out.metadata.get("countdown") != wc.DEFAULT_COUNTDOWN_SECONDS or bool_out.metadata.get("raw_countdown") != "True":
        raise SystemExit(f"human_write should preserve boolean raw countdown metadata: {bool_out.metadata}")
    bool_handoff = _assert_writer_handoff(
        bool_out.metadata,
        "human_write boolean countdown",
        source="human_write",
        status="typing_command_completed",
        controls_computer=True,
    )
    if bool_handoff.get("raw_countdown") != "True" or bool_handoff.get("countdown") != wc.DEFAULT_COUNTDOWN_SECONDS:
        raise SystemExit(f"human_write handoff should preserve sanitized boolean countdown: {bool_handoff}")

    calls.clear()
    sleeps.clear()
    path_out = _tools()["human_write"].handler({"text": "Hello world.", "speed": "fast", "countdown": "/\x55sers/example/private/countdown"})
    if not path_out.ok or sleeps[:1] != [wc.DEFAULT_COUNTDOWN_SECONDS]:
        raise SystemExit(f"human_write should treat path-shaped countdown as malformed default: sleeps={sleeps}, output={path_out.output}")
    if path_out.metadata.get("countdown") != wc.DEFAULT_COUNTDOWN_SECONDS or path_out.metadata.get("raw_countdown") != "<local-path>":
        raise SystemExit(f"human_write should redact path-shaped raw countdown metadata: {path_out.metadata}")
    path_handoff = _assert_writer_handoff(
        path_out.metadata,
        "human_write path countdown",
        source="human_write",
        status="typing_command_completed",
        controls_computer=True,
    )
    if path_handoff.get("raw_countdown") != "<local-path>":
        raise SystemExit(f"human_write handoff should redact path-shaped countdown: {path_handoff}")


def test_human_write_failure_is_clean() -> None:
    old_sleep = wc.time.sleep
    old_run_actions = wc._run_actions

    def boom(_actions):
        raise RuntimeError("raw osascript failure")

    try:
        wc.time.sleep = lambda *_: None  # type: ignore
        wc._run_actions = boom  # type: ignore
        out = _tools()["human_write"].handler({"text": "Hello world.", "countdown": 0})
    finally:
        wc.time.sleep = old_sleep  # type: ignore
        wc._run_actions = old_run_actions  # type: ignore
    if out.ok:
        raise SystemExit(f"human_write failure should fail closed: {out.output}")
    if "raw osascript failure" in out.output or "Human-write error" in out.output:
        raise SystemExit(f"human_write failure should not leak raw exceptions: {out.output}")
    if out.metadata.get("exception_type") != "RuntimeError" or out.metadata.get("error_type") != "RuntimeError":
        raise SystemExit(f"human_write failure should preserve bounded diagnostic metadata: {out.metadata}")
    fail_handoff = _assert_writer_handoff(
        out.metadata,
        "human_write failure",
        source="human_write",
        status="outcome_unknown",
        reason="write_outcome_unknown",
        controls_computer=True,
    )
    _assert_unknown_writer_failure(
        out,
        "human_write failure",
        mode="human",
        reason="write_outcome_unknown",
    )
    if fail_handoff.get("exception_type") != "RuntimeError" or fail_handoff.get("error_type") != "RuntimeError":
        raise SystemExit(f"human_write failure handoff should preserve exception type: {fail_handoff}")


def test_human_write_multichunk_failure_is_outcome_unknown() -> None:
    original_run = wc.subprocess.run
    original_sleep = wc.time.sleep
    calls = 0
    try:
        def fake_run(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                returncode=0 if calls == 1 else 1,
                stdout="",
                stderr="PRIVATE /\x55sers/example/writer-chunk.log",
            )

        wc.subprocess.run = fake_run  # type: ignore
        wc.time.sleep = lambda *_: None  # type: ignore
        out = _tools()["human_write"].handler(
            {"text": "a" * 401, "typos": False, "countdown": 0}
        )
    finally:
        wc.subprocess.run = original_run  # type: ignore
        wc.time.sleep = original_sleep  # type: ignore
    if calls != 2:
        raise SystemExit(f"human_write multi-chunk proof did not reach the second chunk: {calls}")
    _assert_unknown_writer_failure(
        out,
        "human_write second-chunk failure",
        mode="human",
        reason="write_outcome_unknown",
    )
    _assert_no_local_path(out.metadata, "human_write second-chunk failure metadata")


def test_human_write_preflight_failure_is_known() -> None:
    original_plan = wc.plan_keystrokes
    original_sleep = wc.time.sleep
    try:
        wc.plan_keystrokes = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("PRIVATE /\x55sers/example/writer-plan.log")
        )
        wc.time.sleep = lambda *_: None  # type: ignore
        out = _tools()["human_write"].handler(
            {"text": "Hello world.", "countdown": 0}
        )
    finally:
        wc.plan_keystrokes = original_plan
        wc.time.sleep = original_sleep  # type: ignore
    if (
        out.ok
        or out.metadata.get("failure_stage") != "typing_preflight"
        or out.metadata.get("outcome_known") is not True
        or out.metadata.get("outcome_unknown") is not False
        or out.metadata.get("side_effect_possible") is not False
        or out.metadata.get("retry_safe") is not True
        or out.metadata.get("controls_computer") is not False
    ):
        raise SystemExit(f"human_write preflight failure was not known/no-effect: {out}")
    _assert_writer_recovery_message(out.output, "human_write preflight failure")
    _assert_writer_handoff(
        out.metadata,
        "human_write preflight failure",
        source="human_write",
        status="failed",
        reason="write_error",
        controls_computer=False,
    )
    _assert_no_local_path(out.metadata, "human_write preflight failure metadata")


def test_path_shaped_text_never_writes() -> None:
    path_samples = [
        "/\x55sers/example/Desktop/private-note",
        "/private/tmp/jarvis-secret",
        "/var/folders/zc/jarvis-secret",
        "/tmp/jarvis-secret",
    ]
    tools = _tools()
    for tool_name in ["human_write", "paste_text"]:
        for sample in path_samples:
            calls: list = []
            sleeps: list = []
            _install_fakes(calls, sleeps)
            out = tools[tool_name].handler({"text": f"please write {sample}", "countdown": 0})
            if out.ok or out.metadata.get("reason") != "invalid_text" or not out.metadata.get("local_path_text"):
                raise SystemExit(f"{tool_name} should reject local-path-shaped text: {sample} -> {out.output} {out.metadata}")
            if calls or sleeps:
                raise SystemExit(f"{tool_name} should fail before sleep or write side effects: calls={calls}, sleeps={sleeps}")
            if sample in out.output or sample in str(out.metadata):
                raise SystemExit(f"{tool_name} should not echo raw local path: {out.output} {out.metadata}")
            _assert_writer_handoff(out.metadata, f"{tool_name} path-shaped text", source=tool_name, status="refused", reason="invalid_text")


def test_paste_text_runs_with_mocked_io() -> None:
    calls: list = []
    sleeps: list = []
    _install_fakes(calls, sleeps)
    out = _tools()["paste_text"].handler({"text": "A block of text", "countdown": 0})
    if not out.ok or "Paste command completed" not in out.output:
        raise SystemExit(f"paste_text failed: {out.output}")
    if sleeps[:1] != [0] or out.metadata.get("countdown") != 0:
        raise SystemExit(f"paste_text should honor explicit zero countdown: sleeps={sleeps}, metadata={out.metadata}")
    paste_handoff = _assert_writer_handoff(out.metadata, "paste_text success", source="paste_text", status="paste_command_completed", controls_computer=True)
    if paste_handoff.get("countdown") != 0:
        raise SystemExit(f"paste_text handoff should preserve countdown: {paste_handoff}")
    if (
        out.metadata.get("target_content_verified") is not False
        or out.metadata.get("clipboard_restored") is not None
        or out.metadata.get("clipboard_text_restored") is not True
        or out.metadata.get("clipboard_fully_restored") is not False
        or out.metadata.get("clipboard_restoration_scope") != "plain_text_value_only"
        or out.metadata.get("clipboard_snapshot_attempted") is not True
        or out.metadata.get("clipboard_snapshot_captured") is not True
        or out.metadata.get("clipboard_replacement_attempted") is not True
        or out.metadata.get("clipboard_private_content_possible") is not False
        or out.metadata.get("reads_clipboard") is not True
        or out.metadata.get("reads_private_data") is not True
        or out.metadata.get("reads_personal_data") is not True
    ):
        raise SystemExit(f"paste success overclaimed target or clipboard fidelity: {out.metadata}")

    calls.clear()
    sleeps.clear()
    bool_out = _tools()["paste_text"].handler({"text": "A block of text", "countdown": False})
    if not bool_out.ok or sleeps[:1] != [wc.DEFAULT_COUNTDOWN_SECONDS]:
        raise SystemExit(f"paste_text should treat boolean countdown as malformed default: sleeps={sleeps}, output={bool_out.output}")
    if bool_out.metadata.get("countdown") != wc.DEFAULT_COUNTDOWN_SECONDS or bool_out.metadata.get("raw_countdown") != "False":
        raise SystemExit(f"paste_text should preserve boolean raw countdown metadata: {bool_out.metadata}")
    bool_handoff = _assert_writer_handoff(bool_out.metadata, "paste_text boolean countdown", source="paste_text", status="paste_command_completed", controls_computer=True)
    if bool_handoff.get("raw_countdown") != "False" or bool_handoff.get("countdown") != wc.DEFAULT_COUNTDOWN_SECONDS:
        raise SystemExit(f"paste_text handoff should preserve sanitized boolean countdown: {bool_handoff}")

    calls.clear()
    sleeps.clear()
    path_out = _tools()["paste_text"].handler({"text": "A block of text", "countdown": "/private/tmp/jarvis-writer-countdown"})
    if not path_out.ok or sleeps[:1] != [wc.DEFAULT_COUNTDOWN_SECONDS]:
        raise SystemExit(f"paste_text should treat path-shaped countdown as malformed default: sleeps={sleeps}, output={path_out.output}")
    if path_out.metadata.get("countdown") != wc.DEFAULT_COUNTDOWN_SECONDS or path_out.metadata.get("raw_countdown") != "<local-path>":
        raise SystemExit(f"paste_text should redact path-shaped raw countdown metadata: {path_out.metadata}")
    path_handoff = _assert_writer_handoff(path_out.metadata, "paste_text path countdown", source="paste_text", status="paste_command_completed", controls_computer=True)
    if path_handoff.get("raw_countdown") != "<local-path>":
        raise SystemExit(f"paste_text handoff should redact path-shaped countdown: {path_handoff}")


def test_paste_text_failure_is_clean() -> None:
    old_sleep = wc.time.sleep
    old_paste_block = wc._paste_block

    def boom(_text):
        raise RuntimeError("raw paste failure")

    try:
        wc.time.sleep = lambda *_: None  # type: ignore
        wc._paste_block = boom  # type: ignore
        out = _tools()["paste_text"].handler({"text": "A block of text", "countdown": 0})
    finally:
        wc.time.sleep = old_sleep  # type: ignore
        wc._paste_block = old_paste_block  # type: ignore
    if out.ok:
        raise SystemExit(f"paste_text failure should fail closed: {out.output}")
    if "raw paste failure" in out.output or "Write error" in out.output:
        raise SystemExit(f"paste_text failure should not leak raw exceptions: {out.output}")
    if out.metadata.get("exception_type") != "RuntimeError" or out.metadata.get("error_type") != "RuntimeError":
        raise SystemExit(f"paste_text failure should preserve bounded diagnostic metadata: {out.metadata}")
    fail_handoff = _assert_writer_handoff(
        out.metadata,
        "paste_text failure",
        source="paste_text",
        status="outcome_unknown",
        reason="paste_outcome_unknown",
        controls_computer=True,
    )
    _assert_unknown_writer_failure(
        out,
        "paste_text failure",
        mode="paste",
        reason="paste_outcome_unknown",
    )
    if fail_handoff.get("exception_type") != "RuntimeError" or fail_handoff.get("error_type") != "RuntimeError":
        raise SystemExit(f"paste_text failure handoff should preserve exception type: {fail_handoff}")


def test_paste_text_checks_every_subprocess_stage() -> None:
    old_run = wc.subprocess.run
    old_sleep = wc.time.sleep

    def run_case(mode: str):
        calls: list[list[str]] = []
        change_count = 10
        clipboard_text = "old clipboard"
        conditional_writes = 0

        def fake_run(argv, **kwargs):
            nonlocal change_count, clipboard_text, conditional_writes
            argv = list(argv)
            calls.append(argv)
            if argv == wc.clipboard_safety._CHANGE_COUNT_COMMAND:
                return SimpleNamespace(
                    returncode=0,
                    stdout=str(change_count),
                    stderr="",
                )
            if argv == ["osascript", "-e", "clipboard info"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="«class ut16», 26, «class utf8», 13, string, 13",
                    stderr="",
                )
            if argv == ["pbpaste"]:
                return SimpleNamespace(
                    returncode=1 if mode == "clipboard_read" else 0,
                    stdout=clipboard_text,
                    stderr="PRIVATE /\x55sers/example/clipboard.log",
                )
            if wc.clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv:
                conditional_writes += 1
                if (
                    mode == "clipboard_write"
                    and conditional_writes == 1
                ) or (
                    mode == "clipboard_restore"
                    and conditional_writes == 2
                ):
                    return SimpleNamespace(
                        returncode=1,
                        stdout="",
                        stderr="PRIVATE /\x55sers/example/clipboard.log",
                    )
                if int(argv[-1]) != change_count:
                    return SimpleNamespace(returncode=0, stdout="STALE", stderr="")
                clipboard_text = kwargs.get("input", "")
                change_count += 1
                return SimpleNamespace(
                    returncode=0,
                    stdout=f"WRITTEN:{change_count}",
                    stderr="",
                )
            if argv[0] == "osascript":
                return SimpleNamespace(
                    returncode=1 if mode == "paste_command" else 0,
                    stdout="",
                    stderr="PRIVATE /\x55sers/example/paste.log",
                )
            raise AssertionError(f"unexpected writer subprocess: {argv}")

        def fake_sleep(seconds):
            nonlocal change_count, clipboard_text
            if mode == "clipboard_ownership_lost" and seconds == 0.3:
                clipboard_text = "newer user clipboard"
                change_count += 1

        try:
            wc.subprocess.run = fake_run  # type: ignore
            wc.time.sleep = fake_sleep  # type: ignore
            return _tools()["paste_text"].handler({"text": "A block of text", "countdown": 0}), calls
        finally:
            wc.subprocess.run = old_run  # type: ignore
            wc.time.sleep = old_sleep  # type: ignore

    read_failed, read_calls = run_case("clipboard_read")
    if read_failed.ok or read_failed.metadata.get("failure_stage") != "clipboard_read":
        raise SystemExit(f"clipboard read failure was not detected: {read_failed.output} {read_failed.metadata}")
    if (
        read_failed.metadata.get("retry_safe") is not True
        or read_failed.metadata.get("clipboard_replacement_attempted") is not False
        or any(
            wc.clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in call
            for call in read_calls
        )
    ):
        raise SystemExit(f"clipboard read failure should stop before mutation: {read_calls} {read_failed.metadata}")

    write_failed, write_calls = run_case("clipboard_write")
    if write_failed.ok or write_failed.metadata.get("failure_stage") != "clipboard_write":
        raise SystemExit(f"clipboard write failure was not detected: {write_failed.output} {write_failed.metadata}")
    if (
        write_failed.metadata.get("clipboard_replacement_attempted") is not False
        or write_failed.metadata.get("clipboard_outcome_known") is not True
        or write_failed.metadata.get("retry_safe") is not True
    ):
        raise SystemExit(f"clipboard write failure boundary drifted: {write_calls} {write_failed.metadata}")

    paste_failed, paste_calls = run_case("paste_command")
    if paste_failed.ok or paste_failed.metadata.get("failure_stage") != "paste_command":
        raise SystemExit(f"paste command failure was not detected: {paste_failed.output} {paste_failed.metadata}")
    if paste_failed.metadata.get("paste_outcome_possible") is not True or paste_failed.metadata.get("retry_safe"):
        raise SystemExit(f"paste command failure should be outcome-unknown: {paste_calls} {paste_failed.metadata}")
    if "may already be" not in paste_failed.output or "do not paste again automatically" not in paste_failed.output:
        raise SystemExit(f"paste uncertainty should prevent blind retry: {paste_failed.output}")
    _assert_writer_handoff(
        paste_failed.metadata,
        "paste command uncertainty",
        source="paste_text",
        status="outcome_unknown",
        reason="paste_outcome_unknown",
        controls_computer=True,
    )
    _assert_unknown_writer_failure(
        paste_failed,
        "paste command uncertainty",
        mode="paste",
        reason="paste_outcome_unknown",
    )

    restore_failed, restore_calls = run_case("clipboard_restore")
    if restore_failed.ok or restore_failed.metadata.get("failure_stage") != "clipboard_restore":
        raise SystemExit(f"clipboard restore failure was not detected: {restore_failed.output} {restore_failed.metadata}")
    if restore_failed.metadata.get("paste_command_completed") is not True or restore_failed.metadata.get("clipboard_restored") is not False:
        raise SystemExit(f"restore failure lost partial-success truth: {restore_calls} {restore_failed.metadata}")
    if restore_failed.metadata.get("clipboard_outcome_known") is not False:
        raise SystemExit(f"restore failure overclaimed clipboard outcome: {restore_failed.metadata}")
    if (
        restore_failed.metadata.get("clipboard_snapshot_captured") is not True
        or restore_failed.metadata.get("clipboard_replacement_attempted") is not True
        or restore_failed.metadata.get("clipboard_private_content_possible") is not True
    ):
        raise SystemExit(f"restore failure lost clipboard privacy truth: {restore_failed.metadata}")
    if "clipboard content may not have been restored" not in restore_failed.output:
        if "clipboard" not in restore_failed.output:
            raise SystemExit(f"restore failure should warn about clipboard state: {restore_failed.output}")
    _assert_unknown_writer_failure(
        restore_failed,
        "clipboard restore after paste",
        mode="paste",
        reason="paste_outcome_unknown",
    )

    write_restore_failed, write_restore_calls = run_case("clipboard_ownership_lost")
    if (
        write_restore_failed.ok
        or write_restore_failed.metadata.get("failure_stage") != "clipboard_ownership_lost"
        or write_restore_failed.metadata.get("clipboard_restored") is not False
        or write_restore_failed.metadata.get("paste_outcome_possible") is not True
    ):
        raise SystemExit(
            "clipboard ownership loss was not quarantined: "
            f"{write_restore_calls} {write_restore_failed.metadata}"
        )
    _assert_unknown_writer_failure(
        write_restore_failed,
        "clipboard ownership loss",
        mode="paste",
        reason="paste_outcome_unknown",
    )


def test_paste_text_refuses_rich_clipboard_before_mutation() -> None:
    original_run = wc.subprocess.run
    original_sleep = wc.time.sleep
    calls: list[list[str]] = []
    try:
        def fake_run(argv, **_kwargs):
            argv = list(argv)
            calls.append(argv)
            if argv == wc.clipboard_safety._CHANGE_COUNT_COMMAND:
                return SimpleNamespace(returncode=0, stdout="10", stderr="")
            if argv == ["osascript", "-e", "clipboard info"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="«class RTF », 200, «class utf8», 20, string, 20",
                    stderr="",
                )
            raise AssertionError(f"rich clipboard guard allowed mutation: {argv}")

        wc.subprocess.run = fake_run  # type: ignore
        wc.time.sleep = lambda *_: None  # type: ignore
        out = _tools()["paste_text"].handler(
            {"text": "A block of text", "countdown": 0}
        )
    finally:
        wc.subprocess.run = original_run  # type: ignore
        wc.time.sleep = original_sleep  # type: ignore
    if (
        out.ok
        or out.metadata.get("failure_stage") != "clipboard_rich_content"
        or out.metadata.get("outcome_known") is not True
        or out.metadata.get("outcome_unknown") is not False
        or out.metadata.get("retry_safe") is not True
        or out.metadata.get("clipboard_snapshot_attempted") is not True
        or out.metadata.get("clipboard_snapshot_captured") is not False
        or out.metadata.get("clipboard_replacement_attempted") is not False
        or out.metadata.get("clipboard_private_content_possible") is not False
        or out.metadata.get("reads_clipboard") is not True
        or out.metadata.get("reads_private_data") is not True
        or any(
            wc.clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in call
            for call in calls
        )
    ):
        raise SystemExit(f"rich clipboard guard did not fail closed: {calls} {out}")
    if "stopped before changing the clipboard" not in out.output:
        raise SystemExit(f"rich clipboard guard guidance drifted: {out.output}")
    _assert_writer_handoff(
        out.metadata,
        "rich clipboard refusal",
        source="paste_text",
        status="refused",
        reason="clipboard_requires_plain_text",
        controls_computer=False,
    )


def test_planner_routes_writer() -> None:
    p = RuleBasedPlanner()
    hw = p.plan("human write: Hello there friend.")
    if hw.actions[0].tool_name != "human_write" or hw.actions[0].args.get("text") != "Hello there friend.":
        raise SystemExit(f"human_write route wrong: {hw.actions[0].args}")
    fast = p.plan("autowrite fast: The quick brown fox.")
    if fast.actions[0].args.get("speed") != "fast" or fast.actions[0].args.get("text") != "The quick brown fox.":
        raise SystemExit(f"speed/text parse wrong: {fast.actions[0].args}")
    pt = p.plan("paste: Just a block")
    if pt.actions[0].tool_name != "paste_text" or pt.actions[0].args.get("text") != "Just a block":
        raise SystemExit(f"paste route wrong: {pt.actions[0].args}")
    # Real gap found live 2026-07-10: "type this out: X" (and "type it out: X")
    # left the leftover word "out:" INSIDE the captured paste text instead of
    # stripping it as part of the lead-in verb phrase -- the regex recognized
    # "type out" and "type this" as separate whole alternatives but not their
    # combination, so "this" consumed the "type this" branch and "out:" spilled
    # into the payload. A HIGH_RISK computer-control tool would have pasted the
    # wrong text ("out: hello world" instead of "hello world").
    for text, expected in {
        "type this out: hello world": "hello world",
        "type it out: hello world": "hello world",
        "type this: hello world": "hello world",
        "type out: hello world": "hello world",
    }.items():
        route = p.plan(text)
        if route.actions[0].tool_name != "paste_text" or route.actions[0].args.get("text") != expected:
            raise SystemExit(f"paste 'type ... out' route wrong for {text!r}: {route.actions}")


def main() -> None:
    test_risk_is_high()
    test_plan_keystrokes_and_script()
    test_typos_produce_correction_triples()
    test_human_write_runs_with_mocked_io()
    test_human_write_failure_is_clean()
    test_human_write_multichunk_failure_is_outcome_unknown()
    test_human_write_preflight_failure_is_known()
    test_path_shaped_text_never_writes()
    test_paste_text_runs_with_mocked_io()
    test_paste_text_failure_is_clean()
    test_paste_text_checks_every_subprocess_stage()
    test_paste_text_refuses_rich_clipboard_before_mutation()
    test_planner_routes_writer()
    print("Writer connector smoke passed")


if __name__ == "__main__":
    main()
