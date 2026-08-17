"""Smoke tests for local image OCR (mocked Vision runner — no Swift, no images)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools import ocr


def _tools():
    return {t.name: t for t in ocr.make_ocr_tools(load_config())}


def _tool(name: str = "ocr_image"):
    return _tools()[name]


def _image(tmp: str, name: str = "x.png") -> str:
    p = Path(tmp) / name
    p.write_bytes(b"not-a-real-image")  # OCR runner is mocked; contents don't matter
    return str(p)


def _assert_known_local_recovery(out, label: str, *, action: str, commands: list[str]) -> None:
    if out.ok or action not in out.output:
        raise SystemExit(f"{label} hid the canonical recovery action: {out}")
    if out.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": commands,
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {out.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    if commands:
        expected["next_command"] = commands[0]
        expected["recovery_commands"] = commands
    for key, value in expected.items():
        if out.metadata.get(key) != value:
            raise SystemExit(f"{label} recovery field {key} drifted: {out.metadata}")


def test_tool_is_local_safe() -> None:
    if _tool().risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("ocr_image should be LOCAL_SAFE")
    if _tool("photo_document_intake_plan").risk != RiskLevel.READ_ONLY:
        raise SystemExit("photo_document_intake_plan should be READ_ONLY")


def test_is_image_path() -> None:
    if not ocr.is_image_path("/x/y.PNG") or not ocr.is_image_path("a.jpeg"):
        raise SystemExit("image extensions should be recognized")
    if ocr.is_image_path("notes.txt") or ocr.is_image_path("clip.oga"):
        raise SystemExit("non-image extensions must be rejected")


def test_extract_text_uses_runner_and_caps_length() -> None:
    ocr._OCR_RUNNER = lambda path: "Receipt total 42.50 USD"  # type: ignore
    try:
        if ocr.extract_text_from_image(Path("/tmp/x.png")) != "Receipt total 42.50 USD":
            raise SystemExit("OCR runner result not returned")
        ocr._OCR_RUNNER = lambda path: "A" * (ocr.MAX_OCR_TEXT_CHARS + 50)  # type: ignore
        out = ocr.extract_text_from_image(Path("/tmp/x.png"))
        if len(out) > ocr.MAX_OCR_TEXT_CHARS + 1 or not out.endswith("…"):
            raise SystemExit(f"OCR text not capped: len={len(out)}")
    finally:
        ocr._OCR_RUNNER = None  # type: ignore


def test_ocr_image_tool_success_and_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        img = _image(tmp)
        ocr._OCR_RUNNER = lambda path: "Hello world"  # type: ignore
        try:
            out = _tool().handler({"path": img})
            if not out.ok or "Hello world" not in out.output or out.metadata.get("text_found") is not True:
                raise SystemExit(f"ocr success wrong: {out.output} / {out.metadata}")
            ocr._OCR_RUNNER = lambda path: ""  # type: ignore
            out = _tool().handler({"path": img})
            if not out.ok or out.metadata.get("text_found") is not False:
                raise SystemExit(f"empty-OCR should be a clean no-text success: {out.output} / {out.metadata}")
        finally:
            ocr._OCR_RUNNER = None  # type: ignore


def test_ocr_image_tool_rejects_bad_inputs() -> None:
    ocr._OCR_RUNNER = lambda path: "should not run"  # type: ignore
    try:
        missing_path = _tool().handler({})
        if missing_path.ok:
            raise SystemExit("missing path should fail")
        _assert_known_local_recovery(
            missing_path,
            "OCR missing path",
            action=ocr.LOCAL_READ_INPUT_RECOVERY_ACTION,
            commands=[],
        )
        if missing_path.metadata.get("reads_image_text") is not False:
            raise SystemExit(f"OCR missing path should not claim an image read: {missing_path.metadata}")
        if _tool().handler({"path": "notes.txt"}).ok:
            raise SystemExit("non-image path should fail")
        non_image = _tool().handler({"path": "notes.txt"})
        if non_image.metadata.get("reads_image_text") is not False:
            raise SystemExit(f"OCR non-image refusal should not claim an image read: {non_image.metadata}")
        oversized_path = _tool().handler({"path": "x" * (ocr.MAX_OCR_PATH_CHARS + 1) + ".png"})
        if oversized_path.ok or oversized_path.metadata.get("reason") != "path_too_long":
            raise SystemExit(f"overlong OCR path should fail closed: {oversized_path}")
        _assert_known_local_recovery(
            oversized_path,
            "OCR oversized path",
            action=ocr.LOCAL_READ_INPUT_RECOVERY_ACTION,
            commands=[],
        )
        if oversized_path.metadata.get("reads_image_text") is not False:
            raise SystemExit(f"OCR oversized path should not claim an image read: {oversized_path.metadata}")
        missing_file = _tool().handler({"path": "/no/such/image.png"})
        if missing_file.ok:
            raise SystemExit("missing file should fail")
        _assert_known_local_recovery(
            missing_file,
            "OCR missing file",
            action=ocr.LOCAL_READ_INPUT_RECOVERY_ACTION,
            commands=[],
        )
        if missing_file.metadata.get("reads_image_text") is not False:
            raise SystemExit(f"OCR missing file should not claim an image read: {missing_file.metadata}")
    finally:
        ocr._OCR_RUNNER = None  # type: ignore


def test_ocr_unavailable_names_recovery() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        img = _image(tmp)
        original = ocr.ocr_available
        ocr.ocr_available = lambda: False  # type: ignore[assignment]
        try:
            out = _tool().handler({"path": img})
            expected = ["setup check", "swift --version", "Apple Command Line Tools", "ocr image: <path>"]
            missing = [item for item in expected if item not in out.output]
            if out.ok or missing:
                raise SystemExit(f"OCR unavailable guidance wrong: missing={missing} output={out.output!r}")
            if out.metadata.get("reason") != "ocr_unavailable":
                raise SystemExit(f"OCR unavailable metadata wrong: {out.metadata}")
            _assert_known_local_recovery(
                out,
                "OCR unavailable",
                action=ocr.LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                commands=["setup check"],
            )
        finally:
            ocr.ocr_available = original  # type: ignore[assignment]


def test_ocr_error_is_clean_and_actionable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        img = _image(tmp)

        def boom(path):
            raise RuntimeError("/\x55sers/example/secret-receipt.png vision exploded")

        ocr._OCR_RUNNER = boom  # type: ignore
        try:
            out = _tool().handler({"path": img})
            expected = ["opens in Preview", "PNG or JPG", "setup check", "ocr image: <path>"]
            missing = [item for item in expected if item not in out.output]
            if out.ok or missing:
                raise SystemExit(f"OCR error guidance wrong: missing={missing} output={out.output!r}")
            if "vision exploded" in out.output or "/\x55sers/operator" in out.output:
                raise SystemExit(f"OCR error should be clean/non-leaky: {out.output}")
            if out.metadata.get("exception_type") != "RuntimeError":
                raise SystemExit(f"OCR error should keep bounded diagnostic: {out.metadata}")
            if out.metadata.get("error_type") != "RuntimeError":
                raise SystemExit(f"OCR error should keep allowlisted diagnostic: {out.metadata}")
        finally:
            ocr._OCR_RUNNER = None  # type: ignore


def test_photo_document_intake_plan_is_read_only_and_redacts() -> None:
    out = _tool("photo_document_intake_plan").handler({"path": "/tmp/private-receipt.png"})
    if not out.ok or "Jarvis photo/document intake plan" not in out.output:
        raise SystemExit(f"photo/document plan failed: {out.output}")
    required = [
        "read-only",
        "does not open the image",
        "does not run OCR",
        "summary, memory draft, or task draft",
        "confirmation-gated",
        "recognized image extension: yes",
    ]
    missing = [item for item in required if item not in out.output]
    if missing:
        raise SystemExit(f"photo/document plan missing expected text: {missing}")
    if "/tmp/private-receipt.png" in out.output or "/tmp/private-receipt.png" in str(out.metadata):
        raise SystemExit(f"photo/document plan leaked a local path: {out.output} / {out.metadata}")
    metadata = out.metadata
    handoff = metadata.get("photo_document_intake_plan_handoff") or {}
    if metadata.get("reads_personal_data") is not False or metadata.get("reads_image_text") is not False:
        raise SystemExit(f"photo/document plan should not read image content: {metadata}")
    if handoff.get("authorizes_execution") or handoff.get("approval_granted") or handoff.get("content_in_handoff"):
        raise SystemExit(f"photo/document plan handoff must be proof-only: {handoff}")
    for key in [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "executes_side_effect",
        "writes_memory",
        "writes_notes",
        "writes_database",
        "queues_approval",
        "controls_computer",
    ]:
        if metadata.get(key):
            raise SystemExit(f"photo/document plan unsafe metadata {key}: {metadata}")


def test_planner_routes_photo_document_intake() -> None:
    planner = RuleBasedPlanner()
    canonical_path = "/tmp/jarvis-v3-nonexistent-ocr-route-proof.png"
    canonical_plan = planner.plan(f"ocr_image path={canonical_path}")
    if len(canonical_plan.actions) != 1:
        raise SystemExit(f"canonical OCR command did not route deterministically: {canonical_plan}")
    canonical_action = canonical_plan.actions[0]
    if canonical_action.tool_name != "ocr_image" or canonical_action.args != {"path": canonical_path}:
        raise SystemExit(f"canonical OCR path binding drifted: {canonical_action}")
    for malformed in (
        f"ocr_image image={canonical_path}",
        f"ocr_image path={canonical_path} extra=ignored",
        "ocr_image path=",
    ):
        malformed_plan = planner.plan(malformed)
        if any(action.tool_name == "ocr_image" for action in malformed_plan.actions):
            raise SystemExit(f"malformed canonical OCR command reached ocr_image: {malformed!r}")

    plan_cases = {
        "photo intake plan: /tmp/private-receipt.png": "photo_document_intake_plan",
        "review this photo": "photo_document_intake_plan",
        "read this picture": "photo_document_intake_plan",
        "scan this picture": "photo_document_intake_plan",
        "scan this receipt": "photo_document_intake_plan",
        "scan this screenshot": "photo_document_intake_plan",
        "describe this photo": "photo_document_intake_plan",
        "describe this screenshot": "photo_document_intake_plan",
        "summarize this receipt": "photo_document_intake_plan",
        "what does this receipt say": "photo_document_intake_plan",
        "what does this picture say": "photo_document_intake_plan",
        "what's in this image": "photo_document_intake_plan",
        "what is in this image": "photo_document_intake_plan",
        "what's in this photo": "photo_document_intake_plan",
        "what does this screenshot say": "photo_document_intake_plan",
        "extract text from this screenshot": "photo_document_intake_plan",
        "extract text from this receipt": "photo_document_intake_plan",
        "extract text from this picture": "photo_document_intake_plan",
        "ocr this receipt": "photo_document_intake_plan",
        "ocr this screenshot": "photo_document_intake_plan",
        "ocr this picture": "photo_document_intake_plan",
        "ocr image: /tmp/private-receipt.png": "ocr_image",
        "ocr receipt: /tmp/private-receipt.png": "ocr_image",
        "ocr picture: /tmp/private-receipt.png": "ocr_image",
        "read text from photo: /tmp/private-receipt.png": "ocr_image",
        "read receipt photo: /tmp/private-receipt.png": "photo_document_intake_plan",
    }
    for command, expected_tool in plan_cases.items():
        plan = planner.plan(command)
        tools = [action.tool_name for action in plan.actions]
        if tools != [expected_tool]:
            raise SystemExit(f"{command!r} routed to {tools}, expected {expected_tool!r}")
    plan = planner.plan("review this photo: /tmp/private-receipt.png")
    args = plan.actions[0].args if plan.actions else {}
    if args.get("path") != "/tmp/private-receipt.png":
        raise SystemExit(f"photo intake planner missed path: {args}")
    receipt_photo_plan = planner.plan("read receipt photo: /tmp/private-receipt.png")
    receipt_photo_args = receipt_photo_plan.actions[0].args if receipt_photo_plan.actions else {}
    if receipt_photo_args.get("path") != "/tmp/private-receipt.png":
        raise SystemExit(f"receipt photo intake planner missed clean path: {receipt_photo_args}")
    dictionary_guard = planner.plan("what does serendipity mean")
    if [action.tool_name for action in dictionary_guard.actions] != ["define"]:
        raise SystemExit(f"dictionary guard was hijacked by OCR intake: {dictionary_guard.actions}")


def main() -> None:
    test_tool_is_local_safe()
    test_is_image_path()
    test_extract_text_uses_runner_and_caps_length()
    test_ocr_image_tool_success_and_empty()
    test_ocr_image_tool_rejects_bad_inputs()
    test_ocr_unavailable_names_recovery()
    test_ocr_error_is_clean_and_actionable()
    test_photo_document_intake_plan_is_read_only_and_redacts()
    test_planner_routes_photo_document_intake()
    print("OCR smoke passed")


if __name__ == "__main__":
    main()
