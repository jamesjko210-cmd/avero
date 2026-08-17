"""Offline smoke test for the conservative non-ToolResult egress inventory."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

from jarvis_v2.scripts.public_error_egress_inventory import (
    CLI_PATHS,
    HUD_PATH,
    STATUS_SERVER_PATH,
    TELEGRAM_ADAPTER_PATH,
    build_public_error_egress_inventory,
)


# These pins deliberately describe only the first-phase static shapes documented
# by the inventory module. They must not be presented as whole-product coverage.
EXPECTED_COUNTS_BY_CATEGORY = {
    "dashboard_client_error": 7,
    "dashboard_json_error": 110,
    "hud_error_reply": 3,
    "telegram_dynamic_reply": 8,
    "terminal_error_output": 21,
}
EXPECTED_COUNTS_BY_PATH = {
    "jarvis_v2/automations/telegram_control.py": 8,
    "jarvis_v2/scripts/ask.py": 3,
    "jarvis_v2/scripts/chat.py": 1,
    "jarvis_v2/scripts/hud.py": 3,
    "jarvis_v2/scripts/run_status_server.py": 9,
    "jarvis_v2/scripts/run_telegram_control.py": 7,
    "jarvis_v2/scripts/telegram_whoami.py": 1,
    "jarvis_v2/ui/status_server.py": 117,
}
EXPECTED_FINGERPRINT = "beee5705d401450827df4108ae96041b69d9e144f0d305a71d00ca71c7f43685"


def test_synthetic_shapes_and_exclusions() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        sources = {
            STATUS_SERVER_PATH: """
class Handler:
    def emit(self, exc):
        self._send_json({"ok": False, "error": str(exc)}, status=400)
        self._send_json(_dashboard_json_body_error_payload(exc), status=400)
        self._send_json({"ok": True, "message": "error is only prose"})

DASHBOARD_HTML = '''
preflightOutput.textContent = 'Risk preview could not be loaded.';
brainResultOutput.textContent = data.output || data.error || JSON.stringify(data, null, 2);
brainResultOutput.textContent = String(error);
throw new Error(data.error || data.output || 'raw server failure');
'''
""",
            TELEGRAM_ADAPTER_PATH: """
class Bridge:
    def reply(self, text):
        self.send_func("owner", text, None)
        self.answer_callback_func("id", text)
        send_message("owner", text)
""",
            HUD_PATH: """
def submit(exc):
    HudReply(response="failed", error=type(exc).__name__)
    HudReply(response="ok", error="")
""",
            CLI_PATHS[0]: """
def main(exc):
    print("Request failed; retry.")
    print("ordinary output")
    print_startup_failure(exc)
""",
        }
        for path in (STATUS_SERVER_PATH, TELEGRAM_ADAPTER_PATH, HUD_PATH, *CLI_PATHS):
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(sources.get(path, ""), encoding="utf-8")

        inventory = build_public_error_egress_inventory(root)
        if dict(inventory.counts_by_category) != {
            "dashboard_client_error": 1,
            "dashboard_client_raw_error": 3,
            "dashboard_json_error": 2,
            "hud_error_reply": 1,
            "telegram_dynamic_reply": 2,
            "terminal_error_output": 2,
        }:
            raise SystemExit(f"synthetic category inventory drifted: {inventory}")
        if len(inventory.candidates) != 11:
            raise SystemExit(f"synthetic exclusions drifted: {inventory}")


def test_production_inventory_is_exact_and_non_toolresult() -> None:
    root = Path(__file__).resolve().parents[2]
    inventory = build_public_error_egress_inventory(root)
    actual_categories = dict(inventory.counts_by_category)
    actual_paths = dict(inventory.counts_by_path)
    if actual_categories != EXPECTED_COUNTS_BY_CATEGORY:
        raise SystemExit(
            f"public error-egress category counts drifted: {actual_categories}"
        )
    if actual_paths != EXPECTED_COUNTS_BY_PATH:
        raise SystemExit(f"public error-egress path counts drifted: {actual_paths}")
    if inventory.fingerprint != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f"public error-egress location fingerprint drifted: {inventory.fingerprint}"
        )
    if not inventory.candidates:
        raise SystemExit("public error-egress inventory unexpectedly empty")
    if any("ToolResult" in item.shape for item in inventory.candidates):
        raise SystemExit("non-ToolResult inventory overlapped ToolResult constructor sites")
    if any(not item.qualname or item.line <= 0 for item in inventory.candidates):
        raise SystemExit(f"inventory emitted an unstable candidate: {inventory.candidates}")
    if actual_categories.get("dashboard_client_raw_error", 0):
        raise SystemExit(
            "production dashboard still contains a known raw client-error reflection"
        )


def test_inventory_source_makes_no_exhaustive_claim() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "jarvis_v2/scripts/public_error_egress_inventory.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    required = ("first-phase inventory", "not an exhaustive proof", "separate")
    if any(fragment not in docstring for fragment in required):
        raise SystemExit(f"inventory scope disclaimer drifted: {docstring!r}")


def main() -> None:
    test_synthetic_shapes_and_exclusions()
    test_production_inventory_is_exact_and_non_toolresult()
    test_inventory_source_makes_no_exhaustive_claim()
    print("public error egress inventory smoke test passed")


if __name__ == "__main__":
    main()
