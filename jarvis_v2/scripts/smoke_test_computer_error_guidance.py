"""Offline canonical-guidance coverage for computer-control failures."""

from __future__ import annotations

from jarvis_v2.tools import computer


class _FailureGUI:
    def size(self):
        raise RuntimeError("private screen path /\x55sers/example/secret")

    def position(self):
        raise RuntimeError("private pointer path /\x55sers/example/secret")

    def moveTo(self, _x, _y, duration=0):
        raise RuntimeError("private move path /\x55sers/example/secret")

    def click(self, _x, _y, button="left"):
        raise RuntimeError("private click path /\x55sers/example/secret")

    def write(self, _text, interval=0):
        raise RuntimeError("private keyboard path /\x55sers/example/secret")

    def screenshot(self):
        raise RuntimeError("private screenshot path /\x55sers/example/secret")


class _FallbackFailure:
    returncode = 1
    stdout = ""
    stderr = "private fallback /\x55sers/example/secret"


def _assert_guidance(result, *, outcome_unknown: bool) -> None:
    if result.ok:
        raise SystemExit(f"computer failure unexpectedly succeeded: {result}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("version") != 1:
        raise SystemExit(f"computer failure missed canonical guidance: {result.metadata}")
    action = guidance.get("action")
    if not isinstance(action, str) or action not in result.output:
        raise SystemExit(f"computer guidance action is not public: {result}")
    if "/\x55sers/" in result.output or "private " in result.output:
        raise SystemExit(f"computer failure leaked private exception detail: {result.output}")
    expected = {
        "outcome_known": not outcome_unknown,
        "outcome_unknown": outcome_unknown,
        "side_effect_possible": outcome_unknown,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"computer failure has incorrect {key}: {result.metadata}")
    if outcome_unknown:
        if result.metadata.get("retry_safe") is not False or "do not retry automatically" not in result.output:
            raise SystemExit(f"attempted computer action lacks no-retry truth: {result}")


def main() -> None:
    original_gui = computer._pyautogui
    original_enabled = computer._enabled
    original_run = computer.subprocess.run
    try:
        computer._enabled = False
        disabled = computer.click({"x": 1, "y": 2})

        computer._enabled = True
        invalid = [
            computer.move_mouse({"x": "bad", "y": 2}),
            computer.click({"x": 1, "y": 2, "button": "invalid"}),
            computer.type_text({"text": ""}),
            computer.type_text({"text": "x" * (computer.MAX_TYPED_TEXT_CHARS + 1)}),
        ]

        computer._pyautogui = lambda: _FailureGUI()
        computer.subprocess.run = lambda *_args, **_kwargs: _FallbackFailure()
        reads = [computer.screen_size({}), computer.mouse_position({})]
        attempts = [
            computer.move_mouse({"x": 1, "y": 2}),
            computer.click({"x": 1, "y": 2}),
            computer.type_text({"text": "hello"}),
            computer.screenshot({"filename": "proof.png"}),
        ]
    finally:
        computer._pyautogui = original_gui
        computer._enabled = original_enabled
        computer.subprocess.run = original_run

    _assert_guidance(disabled, outcome_unknown=False)
    for result in invalid:
        _assert_guidance(result, outcome_unknown=False)
    for result in reads:
        _assert_guidance(result, outcome_unknown=False)
        if result.metadata.get("retry_safe") is not True:
            raise SystemExit(f"side-effect-free computer read is not retry-safe: {result.metadata}")
    for result in attempts:
        _assert_guidance(result, outcome_unknown=True)

    print("Computer error guidance smoke passed")


if __name__ == "__main__":
    main()
