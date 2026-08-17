"""Local, free image OCR for Jarvis V2 using the macOS Vision framework.

No cloud / no API key: a tiny bundled Swift helper runs Apple's on-device text
recognition. It is compiled once and cached, then reused. `extract_text_from_image`
is the entry point (mockable in tests via `_OCR_RUNNER`); `ocr_image` exposes it as
a READ_ONLY tool, and the Telegram bridge uses the same path for photo/document
intake. Falls back gracefully (returns ""/clear errors) when Swift/Vision is absent.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    declare_retryable_local_read_failure,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult


MAX_OCR_TEXT_CHARS = 8000
MAX_OCR_PATH_CHARS = 2000
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".heic", ".webp"}
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)

# Test seam: set to a callable(Path)->str to bypass the real Vision helper.
_OCR_RUNNER: Callable[[Path], str] | None = None

_SWIFT_OCR_SOURCE = r'''
import Vision
import Foundation
import AppKit

guard CommandLine.arguments.count > 1,
      let img = NSImage(contentsOfFile: CommandLine.arguments[1]),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("ocr-load-failed\n".data(using: .utf8)!)
    exit(2)
}
let request = VNRecognizeTextRequest { req, _ in
    let observations = (req.results as? [VNRecognizedTextObservation]) ?? []
    for obs in observations {
        if let line = obs.topCandidates(1).first {
            print(line.string)
        }
    }
}
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
// Default to Korean + English (the operator's context); override via JARVIS_OCR_LANGUAGES.
let langs = (ProcessInfo.processInfo.environment["JARVIS_OCR_LANGUAGES"] ?? "ko-KR,en-US")
    .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
if !langs.isEmpty { request.recognitionLanguages = langs }
do {
    try VNImageRequestHandler(cgImage: cg, options: [:]).perform([request])
} catch {
    FileHandle.standardError.write("ocr-failed\n".data(using: .utf8)!)
    exit(3)
}
'''


def _cache_dir() -> Path:
    base = Path(os.environ.get("JARVIS_CACHE_DIR") or (Path.home() / ".cache" / "jarvis-v3"))
    out = base / "ocr"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _compiled_ocr_binary() -> Path | None:
    """Compile the Swift Vision helper once and cache it; return the binary path."""
    if not shutil.which("swiftc") and not shutil.which("swift"):
        return None
    digest = hashlib.sha256(_SWIFT_OCR_SOURCE.encode("utf-8")).hexdigest()[:16]
    cache = _cache_dir()
    binary = cache / f"ocr_{digest}"
    if binary.is_file():
        return binary
    src = cache / f"ocr_{digest}.swift"
    if not src.is_file():
        src.write_text(_SWIFT_OCR_SOURCE, encoding="utf-8")
    if shutil.which("swiftc"):
        result = subprocess.run(
            ["swiftc", "-O", str(src), "-o", str(binary)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and binary.is_file():
            return binary
    return None  # caller falls back to interpreting the source


def ocr_available() -> bool:
    if _OCR_RUNNER is not None:
        return True
    return bool(shutil.which("swift") or shutil.which("swiftc"))


def _run_vision_ocr(image_path: Path) -> str:
    """Run the macOS Vision OCR helper on an image. Real implementation."""
    binary = _compiled_ocr_binary()
    if binary is not None:
        result = subprocess.run([str(binary), str(image_path)], capture_output=True, text=True, timeout=120)
    elif shutil.which("swift"):
        # Fallback: interpret the source directly (slower, no compile cache).
        cache = _cache_dir()
        src = cache / "ocr_interp.swift"
        if not src.is_file():
            src.write_text(_SWIFT_OCR_SOURCE, encoding="utf-8")
        result = subprocess.run(["swift", str(src), str(image_path)], capture_output=True, text=True, timeout=180)
    else:
        raise RuntimeError("Swift/Vision OCR is not available on this machine.")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "OCR failed")
    return result.stdout.strip()


def extract_text_from_image(image_path: Path) -> str:
    """Extract text from an image via local Vision OCR. Mockable through `_OCR_RUNNER`."""
    runner = _OCR_RUNNER or _run_vision_ocr
    text = (runner(Path(image_path)) or "").strip()
    if len(text) > MAX_OCR_TEXT_CHARS:
        text = text[:MAX_OCR_TEXT_CHARS].rstrip() + "…"
    return text


def is_image_path(value: Any) -> bool:
    try:
        return Path(str(value)).suffix.lower() in IMAGE_EXTENSIONS
    except Exception:
        return False


def _display(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").strip().split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    return text[:limit] if len(text) > limit else text


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "executes_tools": False,
        "reads_personal_data": True,
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
        "reads_image_text": True,
    }
    base.update(extra)
    return base


def _ocr_unavailable_message() -> str:
    return (
        "On-device OCR isn't available on this Mac. Run `setup check`, confirm Swift is installed "
        "with `swift --version`, or install Apple Command Line Tools, then retry `ocr image: <path>`. "
        f"{LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION}"
    )


def _ocr_read_error_message() -> str:
    return (
        "I couldn't read text from that image. Make sure it opens in Preview, try a PNG or JPG "
        "screenshot if the format is unusual, run `setup check`, then retry `ocr image: <path>`."
    )


def photo_document_intake_plan(args: dict[str, Any]) -> ToolResult:
    raw = str(args.get("path") or args.get("image") or args.get("file") or args.get("request") or "").strip()
    if len(raw) > 2000:
        return ToolResult(
            "photo_document_intake_plan",
            False,
            "Photo/document intake request is too long to review safely.",
            _safe_metadata(
                reads_personal_data=False,
                reads_image_text=False,
                reason="request_too_long",
                request_chars=len(raw),
            ),
        )

    attachment = raw
    path = Path(attachment).expanduser() if attachment else None
    suffix = path.suffix.lower() if path else ""
    recognized_extension = suffix in IMAGE_EXTENSIONS if suffix else False
    path_hash = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16] if path else ""
    handoff = {
        "source": "photo_document_intake_plan",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "attachment_supplied": bool(attachment),
        "attachment_display": _display(attachment) if attachment else "",
        "path_hash": path_hash,
        "suffix": suffix,
        "recognized_extension": recognized_extension,
        "future_ocr_requires_explicit_attachment": True,
        "future_memory_or_task_write_requires_confirmation": True,
        "next_safe_command": "ocr image: <path>",
        "boundaries": [
            "does_not_open_image",
            "does_not_run_ocr",
            "does_not_call_model",
            "does_not_write_memory",
            "does_not_create_task",
            "does_not_queue_approval",
            "does_not_send_message",
            "does_not_grant_future_approval",
        ],
    }
    lines = [
        "Jarvis photo/document intake plan:",
        "",
        "Purpose:",
        "- Prepare a safe phone-photo/document flow for receipts, screenshots, and document photos.",
        "- This plan is read-only and does not open the image, run OCR, call a model, write memory, create tasks, send messages, queue approvals, or grant future approval.",
        "- It does not run OCR; it only prepares the review lane.",
        "",
        "Requested attachment:",
    ]
    if attachment:
        lines.extend(
            [
                f"- reference: {_display(attachment)}",
                f"- path hash: `{path_hash}`",
                f"- extension: {suffix or 'none'}",
                f"- recognized image extension: {'yes' if recognized_extension else 'no'}",
            ]
        )
    else:
        lines.append("- none supplied; send/attach a receipt, screenshot, or document photo, or use `ocr image: <path>` for an explicit local image.")
    lines.extend(
        [
            "",
            "Safe future flow:",
            "1. the operator intentionally supplies a photo/document attachment or local image path.",
            "2. Jarvis runs local OCR only on that explicit image and returns a temporary text preview.",
            "3. the operator chooses whether the preview should become a summary, memory draft, or task draft.",
            "4. Any memory/task write stays separate and confirmation-gated.",
            "",
            "Privacy boundary:",
            "- Photos and screenshots can contain contacts, account numbers, locations, notifications, and private messages.",
            "- OCR text should stay in the review preview unless the operator explicitly confirms a durable note/task.",
            "- This packet proves readiness only; it is not permission to read this or any future image.",
        ]
    )
    return ToolResult(
        "photo_document_intake_plan",
        True,
        "\n".join(lines),
        _safe_metadata(
            reads_personal_data=False,
            reads_image_text=False,
            photo_document_intake_plan_handoff_ready=True,
            photo_document_intake_plan_handoff=handoff,
            photo_document_intake_plan_ready_for_operator=handoff["ready_for_operator"],
            photo_document_intake_plan_state_changed=handoff["state_changed"],
            photo_document_intake_plan_changed=handoff["changed"],
            photo_document_intake_plan_content_in_handoff=handoff["content_in_handoff"],
            photo_document_intake_plan_boundaries=handoff["boundaries"],
            photo_document_intake_plan_next_safe_command=handoff["next_safe_command"],
            ready_for_operator=True,
            state_changed=False,
            changed=[],
            content_in_handoff=False,
            attachment_display=handoff["attachment_display"],
            path_hash=path_hash,
            suffix=suffix,
            recognized_extension=recognized_extension,
            future_ocr_requires_explicit_attachment=True,
            future_memory_or_task_write_requires_confirmation=True,
        ),
    )


def make_ocr_tools(config):
    def ocr_image(args: dict[str, Any]) -> ToolResult:
        raw = args.get("path") or args.get("image") or args.get("file")
        path = str(raw or "").strip()
        if not path:
            failure_output = (
                "Give me the path to an image to read. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "ocr_image",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _safe_metadata(reads_personal_data=False, reads_image_text=False, reason="missing_path"),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        if len(path) > MAX_OCR_PATH_CHARS:
            failure_output = (
                "Image path is too long to read safely. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "ocr_image",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _safe_metadata(
                        reads_personal_data=False,
                        reads_image_text=False,
                        reason="path_too_long",
                        path_chars=len(path),
                        max_path_chars=MAX_OCR_PATH_CHARS,
                    ),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        if not is_image_path(path):
            return ToolResult("ocr_image", False, "That doesn't look like an image file.",
                              _safe_metadata(reads_personal_data=False, reads_image_text=False, reason="not_an_image"))
        try:
            image_exists = Path(path).expanduser().is_file()
        except (OSError, ValueError):
            image_exists = False
        if not image_exists:
            failure_output = (
                "I couldn't find that image file. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "ocr_image",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _safe_metadata(reads_personal_data=False, reads_image_text=False, reason="missing_file"),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        if not ocr_available():
            failure_output = _ocr_unavailable_message()
            return ToolResult(
                "ocr_image",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(reason="ocr_unavailable"),
                    output=failure_output,
                    action=LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )
        try:
            text = extract_text_from_image(Path(path).expanduser())
        except Exception as exc:
            return ToolResult("ocr_image", False, _ocr_read_error_message(),
                              _safe_metadata(
                                  reason="ocr_error",
                                  exception_type=type(exc).__name__,
                                  error_type=type(exc).__name__,
                              ))
        if not text:
            return ToolResult("ocr_image", True, "I didn't find any readable text in that image.",
                              _safe_metadata(text_found=False, text_chars=0))
        return ToolResult("ocr_image", True, f"📄 Text from the image:\n{text}",
                          _safe_metadata(text_found=True, text_chars=len(text)))

    from jarvis_v2.tools.registry import Tool

    return [
        Tool(
            "photo_document_intake_plan",
            "Prepare a read-only photo/document intake plan before OCR, summaries, memory, or tasks. Args: optional path/request.",
            RiskLevel.READ_ONLY,
            photo_document_intake_plan,
            "personal",
        ),
        Tool(
            "ocr_image",
            "Read text from a local image file using on-device macOS OCR (no cloud). Args: path.",
            RiskLevel.LOCAL_SAFE,
            ocr_image,
            "personal",
        )
    ]
