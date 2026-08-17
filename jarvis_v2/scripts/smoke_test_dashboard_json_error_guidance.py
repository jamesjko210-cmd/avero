"""Focused offline checks for dashboard JSON-body recovery guidance.

Scope is intentionally limited to the six ``do_POST`` request-body validation
egresses identified by the public error-egress inventory.  All six are
actionable input failures; this does not classify or prove the other dashboard
JSON errors.
"""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path

from jarvis_v2.ui.status_server import (
    StatusHandler,
    _dashboard_json_body_error_payload,
)


PRIVATE_MARKERS = (
    "/\x55sers/example/private/dashboard-request.json",
    "/private/dashboard-request.json",
    "/var/folders/dashboard-request.json",
    "/tmp/dashboard-request.json",
)
EXPECTED_RECOVERY = {
    "invalid Content-Length": "Retry with Content-Length set to the request body's byte length.",
    "empty request body": "Retry with a non-empty JSON object as the request body.",
    "request body is too large": (
        "Reduce the JSON request body to the endpoint's documented size limit, then retry."
    ),
    "request body must be valid JSON": "Correct the JSON syntax, send it as UTF-8, then retry.",
    "request body must be a JSON object": "Wrap the request fields in a JSON object, then retry.",
}
FALSE_BOUNDARIES = (
    "calls_model",
    "executes_tools",
    "queues_approval",
    "approves_request",
    "dismisses_request",
    "controls_computer",
    "reads_private_data",
    "writes_files",
    "external_side_effect",
)


def _read_body_error(raw_length: str, body: bytes, *, max_bytes: int) -> ValueError:
    handler = object.__new__(StatusHandler)
    handler.headers = {"Content-Length": raw_length}
    handler.rfile = BytesIO(body)
    try:
        handler._read_json_body(max_bytes=max_bytes)
    except ValueError as exc:
        return exc
    raise SystemExit(
        f"dashboard JSON fixture unexpectedly parsed: length={raw_length!r}, body={body!r}"
    )


def _assert_private_safe(value: object, label: str) -> None:
    text = repr(value)
    for marker in PRIVATE_MARKERS:
        if marker in text:
            raise SystemExit(f"{label} leaked private marker {marker!r}: {text}")


def test_exact_parser_failures_have_actionable_guidance() -> None:
    cases = (
        ("invalid Content-Length", _read_body_error("invalid", b"{}", max_bytes=32)),
        ("empty request body", _read_body_error("0", b"", max_bytes=32)),
        ("request body is too large", _read_body_error("33", b"{}", max_bytes=32)),
        ("request body must be valid JSON", _read_body_error("1", b"{", max_bytes=32)),
        ("request body must be a JSON object", _read_body_error("2", b"[]", max_bytes=32)),
    )
    for expected_error, exc in cases:
        payload = _dashboard_json_body_error_payload(exc)
        if payload.get("error") != expected_error:
            raise SystemExit(f"dashboard JSON error changed: {payload}")
        if payload.get("error_code") != "invalid_dashboard_json_request":
            raise SystemExit(f"dashboard JSON error code changed: {payload}")
        if payload.get("recovery") != EXPECTED_RECOVERY[expected_error]:
            raise SystemExit(f"dashboard JSON recovery guidance changed: {payload}")
        metadata = payload.get("metadata") or {}
        if metadata.get("error_category") != "request_validation":
            raise SystemExit(f"dashboard JSON category changed: {payload}")
        if metadata.get("retryable_after_correction") is not True:
            raise SystemExit(f"dashboard JSON recovery is not actionable: {payload}")
        for key in FALSE_BOUNDARIES:
            if metadata.get(key) is not False:
                raise SystemExit(f"dashboard JSON boundary {key} changed: {payload}")
        _assert_private_safe(payload, expected_error)


def test_unrecognized_parser_failure_is_private_safe() -> None:
    marker = PRIVATE_MARKERS[0]
    payload = _dashboard_json_body_error_payload(ValueError(marker))
    if payload.get("error") != "request body was rejected":
        raise SystemExit(f"unknown dashboard JSON error was not canonicalized: {payload}")
    if payload.get("recovery") != (
        "Retry with a non-empty UTF-8 JSON object that satisfies the endpoint's size limit."
    ):
        raise SystemExit(f"unknown dashboard JSON error missed recovery guidance: {payload}")
    _assert_private_safe(payload, "unknown dashboard JSON error")


def test_all_six_post_egresses_use_the_canonical_payload() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "jarvis_v2/ui/status_server.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    post_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "do_POST"
    )
    body_validation_handlers: list[ast.ExceptHandler] = []
    for node in ast.walk(post_method):
        if not isinstance(node, ast.Try):
            continue
        if "_read_json_body" not in ast.unparse(node):
            continue
        body_validation_handlers.extend(
            handler
            for handler in node.handlers
            if isinstance(handler.type, ast.Name) and handler.type.id == "ValueError"
        )
    if len(body_validation_handlers) != 6:
        raise SystemExit(
            f"dashboard JSON-body egress count drifted: {len(body_validation_handlers)}"
        )
    for handler in body_validation_handlers:
        rendered = ast.unparse(handler)
        if "_dashboard_json_body_error_payload(exc)" not in rendered:
            raise SystemExit(f"dashboard JSON egress bypassed canonical guidance: {rendered}")
        if "str(exc)" in rendered:
            raise SystemExit(f"dashboard JSON egress reflects exception text directly: {rendered}")


def main() -> None:
    test_exact_parser_failures_have_actionable_guidance()
    test_unrecognized_parser_failure_is_private_safe()
    test_all_six_post_egresses_use_the_canonical_payload()
    print(
        "dashboard JSON error guidance smoke test passed "
        "(6 actionable egresses; 0 neutral/internal egresses in this bounded slice)"
    )


if __name__ == "__main__":
    main()
