from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.scripts.test_runtime import handle_runtime_case
from jarvis_v2.tools import browser


def assert_vault_relative_receipt(result, root: Path, prefix: str, label: str) -> None:
    metadata = result.tool_results[0].metadata
    path_text = str(metadata.get("path") or "")
    path_display = metadata.get("path_display")
    receipt_line = result.response.split("\n", 1)[0]
    if not path_text or not Path(path_text).exists():
        raise SystemExit(f"{label} should preserve exact saved path metadata: {metadata}")
    if path_text in receipt_line:
        raise SystemExit(f"{label} should not print the raw local note path.")
    if str(root) in receipt_line or "/private/" in receipt_line or "/\x55sers/" in receipt_line:
        raise SystemExit(f"{label} receipt should not expose local temp or user paths.")
    if not isinstance(path_display, str) or not path_display.startswith(prefix):
        raise SystemExit(f"{label} missed vault-relative display metadata: {metadata}")
    if path_display not in receipt_line:
        raise SystemExit(f"{label} should print the vault-relative saved-note label.")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-next-layer-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)
        page_url = "https://example.test/tmp_browser_test.html"
        page_html = (
            "<html><head><title>Jarvis Browser Test</title></head>"
            "<body><h1>Hello Browser</h1><a href='https://example.com/path'>Example Link</a></body></html>"
        )
        original_fetch = browser._fetch
        browser._fetch = lambda _url, timeout=12: (page_html, _url)  # type: ignore[assignment]

        cases = [
            f"fetch page {page_url}",
            f"extract links from {page_url}",
            "recent browser pages",
            "summarize latest page",
            "save latest page to obsidian",
            "open url https://example.test/",
            "setup check",
            "computer control status",
            "computer task plan: open settings, click battery, and verify the battery panel is visible",
            "screen size",
            "weak memories",
            "duplicate memories",
            "show memory 1",
            "schedule daily brief",
            "list scheduled jobs",
            "pause job Daily Brief",
            "resume job Daily Brief",
            "run job Daily Brief now",
            "run due jobs",
        ]
        for case in cases:
            result = handle_runtime_case(
                runtime,
                case,
                approved=case in {"resume job Daily Brief", "run job Daily Brief now", "run due jobs"},
            )
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1600])
            print()
            if case.startswith("computer task plan"):
                required = [
                    "Computer task plan",
                    "Safety boundary",
                    "read-only",
                    "Observe-act-verify loop",
                    "Likely primitive actions",
                    "observe act verify action click",
                    "Stop conditions",
                    "explicitly approves screen observation",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Computer task plan missing expected text: {missing}")
            if case == "setup check":
                v3_shell_prefix = 'JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env"'
                dashboard_command = (
                    f"{v3_shell_prefix} ./launch_jarvis_v3_dashboard.py"
                )
                dashboard_status_command = (
                    f'{v3_shell_prefix} ./launch_jarvis_v3.py "status dashboard"'
                )
                required = [
                    "Jarvis V3 setup check",
                    "pbpaste",
                    "not executed; clipboard privacy",
                    "Safe next Terminal commands",
                    "run from the Jarvis V3 project folder",
                    dashboard_command,
                    "JARVIS_STATUS_HOST",
                    "JARVIS_STATUS_PORT",
                    dashboard_status_command,
                    "prototype readiness",
                    "explicit stop times",
                ]
                missing = [item for item in required if item not in result.response]
                if missing:
                    raise SystemExit(f"Setup check missing clipboard privacy boundary: {missing}")
                metadata = result.tool_results[0].metadata
                if metadata.get("operator_timeboxes_override_priority") is not True or metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"Setup check missed operator-limit metadata: {metadata}")
                for command in [dashboard_command, dashboard_status_command]:
                    if command not in metadata.get("next_commands", []):
                        raise SystemExit(f"Setup check missed dashboard command metadata: {metadata}")
            if case == "save latest page to obsidian":
                metadata = result.tool_results[0].metadata
                if not metadata.get("writes_notes") or not metadata.get("writes_files"):
                    raise SystemExit(f"Saved page note should report note/file writes: {metadata}")
                assert_vault_relative_receipt(result, root, "Sources/", case)
            if case == "open url https://example.test/":
                metadata = result.tool_results[0].metadata
                if result.verified or metadata.get("reason") != "os_browser_launch_disabled":
                    raise SystemExit(f"OS browser launch should fail closed: {result.response} {metadata}")
                if metadata.get("next_command") != "fetch page https://example.test/":
                    raise SystemExit(f"OS browser refusal missed pinned-fetch recovery: {metadata}")
                if metadata.get("external_network") or metadata.get("controls_computer"):
                    raise SystemExit(f"OS browser refusal should be inert: {metadata}")
            if case in {
                "schedule daily brief",
                "list scheduled jobs",
                "pause job Daily Brief",
                "resume job Daily Brief",
                "run job Daily Brief now",
                "run due jobs",
            }:
                metadata = result.tool_results[0].metadata
                if metadata.get("operator_timeboxes_override_priority") is not True:
                    raise SystemExit(f"{case} missed scheduler operator timebox metadata: {metadata}")
                if metadata.get("stop_times_override_priority") is not True:
                    raise SystemExit(f"{case} missed scheduler stop-time metadata: {metadata}")
                if "explicit stop times" not in result.response or "approval-gated" not in result.response:
                    raise SystemExit(f"{case} missed scheduler safety boundary.")
        browser._fetch = original_fetch  # type: ignore[assignment]


if __name__ == "__main__":
    main()
