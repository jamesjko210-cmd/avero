"""Mocked smoke tests for real observe-act-verify computer primitives."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import computer


class FakeImage:
    def __init__(self, calls: list[tuple]):
        self.calls = calls

    def save(self, path):
        self.calls.append(("save", str(path)))


class FakePyAutoGUI:
    def __init__(self):
        self.calls: list[tuple] = []
        self.FAILSAFE = True
        self.PAUSE = 0

    def screenshot(self):
        self.calls.append(("screenshot",))
        return FakeImage(self.calls)

    def size(self):
        self.calls.append(("size",))
        return SimpleNamespace(width=1440, height=900)

    def position(self):
        self.calls.append(("position",))
        return SimpleNamespace(x=12, y=34)

    def locateCenterOnScreen(self, path, confidence=None):
        self.calls.append(("locate", path, confidence))
        return SimpleNamespace(x=321, y=654)

    def click(self, x, y, button="left"):
        self.calls.append(("click", x, y, button))

    def write(self, text, interval=0):
        self.calls.append(("write", text, interval))

    def moveTo(self, x, y, duration=0):
        self.calls.append(("moveTo", x, y, duration))


class FailingPyAutoGUI:
    def size(self):
        raise RuntimeError("screen denied near /\x55sers/example/private/screen")

    def position(self):
        raise RuntimeError("mouse denied near /\x55sers/example/private/mouse")

    def moveTo(self, x, y, duration=0):
        raise RuntimeError("move denied near /\x55sers/example/private/mouse")

    def click(self, x, y, button="left"):
        raise RuntimeError("click denied near /\x55sers/example/private/mouse")

    def write(self, text, interval=0):
        raise RuntimeError("type denied near /\x55sers/example/private/keyboard")

    def screenshot(self):
        raise RuntimeError("screenshot denied near /\x55sers/example/private/screen")


def test_computer_control_primitives_are_high_risk() -> None:
    runtime = make_temp_runtime(Path(tempfile.mkdtemp()))
    for name in ("screenshot", "observe_screen", "verify_screen", "click", "type_text", "move_mouse", "observe_act_verify"):
        tool = runtime.registry.get(name)
        if tool.risk != RiskLevel.HIGH_RISK:
            raise SystemExit(f"{name} must be HIGH_RISK, got {tool.risk}")


def test_disabled_oav_stops_before_observing() -> None:
    fake = FakePyAutoGUI()
    original_pyautogui = computer._pyautogui
    original_enabled = computer._enabled
    try:
        computer._enabled = False
        computer._pyautogui = lambda: fake
        result = computer.observe_act_verify({"action": "click", "x": 10, "y": 20, "expectation": "menu opens"})
    finally:
        computer._pyautogui = original_pyautogui
        computer._enabled = original_enabled
    if result.ok:
        raise SystemExit("Disabled OAV should not run.")
    if fake.calls:
        raise SystemExit(f"Disabled OAV should not observe or act: {fake.calls}")
    if result.metadata.get("computer_control_enabled") is not False:
        raise SystemExit(f"Disabled OAV metadata should report disabled state: {result.metadata}")


def test_failed_before_observation_stops_before_primitive() -> None:
    fake = FakePyAutoGUI()
    original_pyautogui = computer._pyautogui
    original_observe_screen = computer.observe_screen
    original_enabled = computer._enabled
    try:
        computer._enabled = True
        computer._pyautogui = lambda: fake
        computer.observe_screen = lambda _args: ToolResult(
            "observe_screen",
            False,
            "Before observation unavailable.",
            computer._safe_metadata(
                observes_screen=True,
                takes_screenshot=True,
                writes_files=False,
                exception_type="RuntimeError",
            ),
        )
        result = computer.observe_act_verify(
            {"action": "click", "x": 10, "y": 20, "expectation": "menu opens"}
        )
    finally:
        computer._pyautogui = original_pyautogui
        computer.observe_screen = original_observe_screen
        computer._enabled = original_enabled
    if result.ok:
        raise SystemExit("OAV should fail when its before observation fails.")
    if fake.calls:
        raise SystemExit(f"Failed before observation should cause zero primitive calls: {fake.calls}")
    if result.metadata.get("stopped_before_action") is not True or result.metadata.get("action_attempted") is not False:
        raise SystemExit(f"Failed before observation metadata missed fail-closed state: {result.metadata}")
    if result.metadata.get("before", {}).get("exception_type") != "RuntimeError":
        raise SystemExit(f"Failed before observation metadata missed safe diagnostics: {result.metadata}")


def test_oav_click_locates_target_and_verifies_after() -> None:
    fake = FakePyAutoGUI()
    original_pyautogui = computer._pyautogui
    original_enabled = computer._enabled
    try:
        computer._enabled = True
        computer._pyautogui = lambda: fake
        result = computer.observe_act_verify(
            {
                "action": "click",
                "target_image": "/tmp/fake_button.png",
                "expectation": "settings panel opens",
            }
        )
    finally:
        computer._pyautogui = original_pyautogui
        computer._enabled = original_enabled
    if not result.ok:
        raise SystemExit(f"Mocked OAV click failed: {result}")
    call_names = [call[0] for call in fake.calls]
    expected_order = ["screenshot", "save", "size", "position", "locate", "click", "screenshot", "save", "size", "position"]
    if call_names != expected_order:
        raise SystemExit(f"OAV click sequence changed: {fake.calls}")
    if ("click", 321, 654, "left") not in fake.calls:
        raise SystemExit(f"OAV click did not use located target center: {fake.calls}")
    if result.metadata.get("target_located") is not True or result.metadata.get("located_x") != 321 or result.metadata.get("located_y") != 654:
        raise SystemExit(f"OAV locate metadata missing: {result.metadata}")
    if result.metadata.get("controls_computer") is not True or result.metadata.get("takes_screenshot") is not True:
        raise SystemExit(f"OAV execution metadata should show control and screenshots: {result.metadata}")


def test_oav_type_text_observes_types_and_verifies() -> None:
    fake = FakePyAutoGUI()
    original_pyautogui = computer._pyautogui
    original_enabled = computer._enabled
    try:
        computer._enabled = True
        computer._pyautogui = lambda: fake
        result = computer.observe_act_verify(
            {
                "action": "type_text",
                "text": "hello",
                "expectation": "hello appears in the selected field",
            }
        )
    finally:
        computer._pyautogui = original_pyautogui
        computer._enabled = original_enabled
    if not result.ok:
        raise SystemExit(f"Mocked OAV type_text failed: {result}")
    call_names = [call[0] for call in fake.calls]
    expected_order = ["screenshot", "save", "size", "position", "write", "screenshot", "save", "size", "position"]
    if call_names != expected_order:
        raise SystemExit(f"OAV type_text sequence changed: {fake.calls}")
    if ("write", "hello", 0.02) not in fake.calls:
        raise SystemExit(f"OAV type_text did not type expected text: {fake.calls}")


def test_oav_type_text_preserves_exact_approved_text() -> None:
    fake = FakePyAutoGUI()
    approved_text = "  first line\n\ud55c\uad6d\uc5b4 /\x55sers/example/private/literal path  \nlast line\t"
    original_pyautogui = computer._pyautogui
    original_enabled = computer._enabled
    try:
        computer._enabled = True
        computer._pyautogui = lambda: fake
        result = computer.observe_act_verify(
            {
                "action": "type_text",
                "text": approved_text,
                "expectation": "exact text appears in the selected field",
            }
        )
    finally:
        computer._pyautogui = original_pyautogui
        computer._enabled = original_enabled
    if not result.ok:
        raise SystemExit(f"Mocked exact OAV type_text failed: {result}")
    writes = [call for call in fake.calls if call[0] == "write"]
    if writes != [("write", approved_text, 0.02)]:
        raise SystemExit(f"OAV type_text changed the approved text: {writes}")
    if approved_text in result.output or approved_text in repr(result.metadata):
        raise SystemExit("OAV type_text leaked raw approved text in output or metadata.")


def test_oversized_type_text_is_refused_without_typing() -> None:
    fake = FakePyAutoGUI()
    oversized_text = "x" * (computer.MAX_TYPED_TEXT_CHARS + 1)
    original_pyautogui = computer._pyautogui
    original_enabled = computer._enabled
    try:
        computer._enabled = True
        computer._pyautogui = lambda: fake
        result = computer.type_text({"text": oversized_text})
    finally:
        computer._pyautogui = original_pyautogui
        computer._enabled = original_enabled
    if result.ok:
        raise SystemExit("Oversized type_text should be refused.")
    if fake.calls:
        raise SystemExit(f"Oversized type_text should never call the GUI primitive: {fake.calls}")
    if result.metadata.get("text_chars") != len(oversized_text):
        raise SystemExit(f"Oversized type_text should report only the safe character count: {result.metadata}")
    if oversized_text in result.output or oversized_text in repr(result.metadata):
        raise SystemExit("Oversized type_text leaked raw approved text in output or metadata.")


def _assert_clean_computer_failure(result, expected: str, label: str, *, attempted: bool) -> None:
    if result.ok or expected not in result.output:
        raise SystemExit(f"{label} should return friendly output: {result.output}")
    for recovery in [
        "System Settings > Privacy & Security > Accessibility",
        "Screen Recording",
        "computer control readiness",
    ]:
        if recovery not in result.output:
            raise SystemExit(f"{label} missed actionable recovery guidance {recovery}: {result.output}")
    if attempted:
        if "outcome is unknown" not in result.output or "do not retry automatically" not in result.output:
            raise SystemExit(f"{label} missed outcome-unknown recovery guidance: {result.output}")
        for key, expected_value in {
            "outcome_known": False,
            "outcome_unknown": True,
            "side_effect_possible": True,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }.items():
            if result.metadata.get(key) is not expected_value:
                raise SystemExit(f"{label} has unsafe {key} metadata: {result.metadata}")
    elif "then retry" not in result.output:
        raise SystemExit(f"{label} missed read-recovery guidance: {result.output}")
    if "/\x55sers/operator" in result.output or "denied near" in result.output:
        raise SystemExit(f"{label} leaked raw exception text: {result.output}")
    if result.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"{label} missed diagnostic exception type: {result.metadata}")
    if result.metadata.get("operator_timeboxes_override_priority") is not True or result.metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed operator-limit metadata: {result.metadata}")


def test_computer_failures_are_non_leaky() -> None:
    fake = FailingPyAutoGUI()

    class FakeFallbackFailure:
        returncode = 1
        stdout = ""
        stderr = "osascript denied /\x55sers/example/private/screen"

    original_pyautogui = computer._pyautogui
    original_enabled = computer._enabled
    original_subprocess_run = computer.subprocess.run
    try:
        computer._enabled = True
        computer._pyautogui = lambda: fake
        computer.subprocess.run = lambda *_args, **_kwargs: FakeFallbackFailure()  # type: ignore[assignment]
        cases = [
            (computer.screen_size({}), "Could not read screen size.", "screen_size", False),
            (computer.mouse_position({}), "Could not read mouse position.", "mouse_position", False),
            (computer.move_mouse({"x": 10, "y": 20}), "Could not move mouse.", "move_mouse", True),
            (computer.click({"x": 10, "y": 20}), "Could not click.", "click", True),
            (computer.type_text({"text": "hello"}), "Could not type text.", "type_text", True),
            (computer.screenshot({"filename": "failure.png"}), "Could not take screenshot.", "screenshot", True),
        ]
    finally:
        computer._pyautogui = original_pyautogui
        computer._enabled = original_enabled
        computer.subprocess.run = original_subprocess_run  # type: ignore[assignment]
    for result, expected, label, attempted in cases:
        _assert_clean_computer_failure(result, expected, label, attempted=attempted)


def main() -> None:
    test_computer_control_primitives_are_high_risk()
    test_disabled_oav_stops_before_observing()
    test_failed_before_observation_stops_before_primitive()
    test_oav_click_locates_target_and_verifies_after()
    test_oav_type_text_observes_types_and_verifies()
    test_oav_type_text_preserves_exact_approved_text()
    test_oversized_type_text_is_refused_without_typing()
    test_computer_failures_are_non_leaky()
    print("Computer OAV smoke passed")


if __name__ == "__main__":
    main()
