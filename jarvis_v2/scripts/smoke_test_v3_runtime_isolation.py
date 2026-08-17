"""Offline proof that V3 defaults cannot silently reuse V2 runtime state."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.env import V3PathCustodyError, validate_v3_path_custody
from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile


ROOT = Path(__file__).resolve().parents[2]
IS_PUBLIC_CANDIDATE = structural_public_candidate_profile(ROOT) is not None


def test_v3_default_paths_in_fresh_process() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-isolation-") as temp:
        root = Path(temp)
        home = root / "home"
        home.mkdir()
        empty_v3_env = root / "v3.env"
        empty_v3_env.write_text("", encoding="utf-8")
        empty_v3_env.chmod(0o600)
        poisoned_v2_env = root / "v2.env"
        poisoned_v2_env.write_text("JARVIS_V2_SENTINEL=must-not-load\n", encoding="utf-8")
        child_env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "JARVIS_V2_ENV": str(poisoned_v2_env),
            "JARVIS_V3_ENV": str(empty_v3_env),
        }
        code = """
import os
import pwd
from pathlib import Path
from jarvis_v2.config import load_config
from jarvis_v2.automations import imessage_control, reminders, telegram_control
from jarvis_v2.tools import calendar_connector
from jarvis_v2.ui.status_config import status_port_from_env

home = Path(os.environ['HOME'])
account_home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
config = load_config()
assert 'JARVIS_V2_SENTINEL' not in os.environ
assert config.data_dir == home / '.jarvis_v3'
assert config.db_path == home / '.jarvis_v3' / 'jarvis.sqlite'
assert config.obsidian_vault == home / '.jarvis_v3' / 'Vault'
assert telegram_control._state_file() == account_home / '.jarvis_v3' / 'telegram_control.json'
assert imessage_control._state_file() == home / '.jarvis_v3' / 'imessage_control.json'
assert reminders._reminders_file() == home / '.jarvis_v3' / 'telegram_reminders.json'
assert calendar_connector._creds_file() == home / '.jarvis_v3' / 'google_credentials.json'
assert calendar_connector._readonly_token_file() == home / '.jarvis_v3' / 'google_calendar_readonly_token.json'
assert calendar_connector._token_file() == home / '.jarvis_v3' / 'google_token.json'
assert status_port_from_env() == 8766
print('fresh-process V3 defaults are isolated')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        if result.returncode != 0:
            raise SystemExit(
                "fresh-process V3 isolation failed: "
                f"stdout={result.stdout!r}; stderr={result.stderr!r}"
            )


def test_production_defaults_have_no_v2_targets() -> None:
    production_paths = (
        "jarvis_v2/config.py",
        "jarvis_v2/env.py",
        "jarvis_v2/agent/runtime.py",
        "jarvis_v2/automations/telegram_control.py",
        "jarvis_v2/automations/imessage_control.py",
        "jarvis_v2/automations/reminders.py",
        "jarvis_v2/scripts/live_check.py",
        "jarvis_v2/scripts/run_status_server.py",
        "jarvis_v2/scripts/setup_status_auth.py",
        "jarvis_v2/tools/calendar_connector.py",
        "jarvis_v2/tools/ocr.py",
        "jarvis_v2/tools/storage.py",
        "jarvis_v2/tools/system.py",
        "jarvis_v2/ui/status_config.py",
        "jarvis_v2/ui/status_server.py",
    )
    forbidden = (
        "JARVIS_V2_ENV",
        "JARVIS_V2_IMESSAGE_STATE",
        '"~/.jarvis_v2"',
        '".jarvis_v2_runtime"',
        '".jarvis_v2_durable"',
        "com.jarvis-v2.telegram",
        "com.jarvis-v2.imessage",
        "com.jarvis-v2.dashboard",
        "launch_jarvis_v2_dashboard.py",
    )
    for relative in production_paths:
        text = (ROOT / relative).read_text(encoding="utf-8")
        found = [marker for marker in forbidden if marker in text]
        if found:
            raise SystemExit(f"{relative} retains V2 runtime target(s): {found}")


def test_explicit_v3_paths_cannot_alias_v2_state() -> None:
    direct_cases = (
        {"JARVIS_DATA_DIR": "~/.jarvis_v2"},
        {"JARVIS_DB_PATH": "~/.jarvis_v2/jarvis.sqlite"},
        {"JARVIS_OBSIDIAN_VAULT": "/private/tmp/jarvis-v2/Vault"},
        {"JARVIS_GOOGLE_READONLY_TOKEN": ".jarvis_v2_runtime/token.json"},
        {"JARVIS_WATCHED_DIRS": f"/private/tmp/safe{os.pathsep}~/.jarvis_v2/Vault"},
    )
    for values in direct_cases:
        expected_key = next(iter(values))
        try:
            validate_v3_path_custody(values)
        except V3PathCustodyError as exc:
            if exc.key != expected_key or exc.reason_code != "overlaps_v2":
                raise SystemExit(f"V2 custody refusal lost bounded identity: {exc!r}")
            if ".jarvis_v2" in str(exc) or "/private/" in str(exc):
                raise SystemExit(f"V2 custody refusal leaked a supplied path: {exc}")
        else:
            raise SystemExit(f"V3 accepted a direct V2 custody target: {tuple(values)}")

    with TemporaryDirectory(prefix="jarvis-v3-custody-") as temp:
        root = Path(temp)
        v2 = root / ".jarvis_v2"
        v2.mkdir()
        alias = root / "apparently-v3"
        alias.symlink_to(v2, target_is_directory=True)
        try:
            validate_v3_path_custody({"JARVIS_DATA_DIR": str(alias)})
        except V3PathCustodyError as exc:
            if exc.key != "JARVIS_DATA_DIR" or exc.reason_code != "overlaps_v2":
                raise SystemExit(f"symlinked V2 custody refusal lost bounded identity: {exc!r}")
        else:
            raise SystemExit("V3 accepted a symlink alias into V2 state")

    synthetic_path = Path("/private/tmp/synthetic-unresolvable-v3-state")
    with patch.object(Path, "resolve", side_effect=OSError("private path detail")):
        try:
            validate_v3_path_custody({"JARVIS_DATA_DIR": str(synthetic_path)})
        except V3PathCustodyError as exc:
            rendered = str(exc)
            if exc.key != "JARVIS_DATA_DIR" or exc.reason_code != "unresolvable_alias":
                raise SystemExit(f"unresolvable V3 path refusal lost bounded identity: {exc!r}")
            if "private path detail" in rendered or str(synthetic_path) in rendered:
                raise SystemExit("unresolvable V3 path refusal leaked local detail")
        else:
            raise SystemExit("V3 accepted a persistent path whose aliases could not be resolved")

    try:
        validate_v3_path_custody({"JARVIS_DB_PATH": "synthetic-invalid\x00private-detail"})
    except V3PathCustodyError as exc:
        rendered = str(exc)
        if exc.key != "JARVIS_DB_PATH" or exc.reason_code != "invalid_path":
            raise SystemExit(f"invalid V3 path refusal lost bounded identity: {exc!r}")
        if "private-detail" in rendered or "embedded null" in rendered:
            raise SystemExit("invalid V3 path refusal leaked local detail")
    else:
        raise SystemExit("V3 accepted an invalid persistent path")

    hostile = V3PathCustodyError("SECRET_PATH_VALUE", "raw-private-reason")
    if (
        hostile.key != "V3_PERSISTENT_PATH"
        or hostile.reason_code != "invalid_path"
        or "SECRET_PATH_VALUE" in str(hostile)
        or "raw-private-reason" in str(hostile)
    ):
        raise SystemExit(f"path-custody exception accepted unallowlisted diagnostics: {hostile!r}")


def test_selected_environment_cannot_redirect_v3_into_v2() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-selected-custody-") as temp:
        root = Path(temp)
        home = root / "home"
        home.mkdir()
        selected = root / "v3.env"
        selected.write_text("JARVIS_DATA_DIR=~/.jarvis_v2\n", encoding="utf-8")
        selected.chmod(0o600)
        child_env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "JARVIS_V3_ENV": str(selected),
        }
        code = "from jarvis_v2.config import load_config; load_config()"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        combined = result.stdout + result.stderr
        if result.returncode == 0 or "V3PathCustodyError" not in combined:
            raise SystemExit(f"selected V3 env redirected into V2 state: {combined!r}")
        if os.fspath(root) in combined or os.fspath(home) in combined:
            raise SystemExit("selected V2 custody refusal leaked a local path")


def test_v3_service_templates_are_inert_by_default() -> None:
    services = ("telegram", "imessage", "dashboard")
    template_paths = tuple(ROOT / f"com.jarvis-v3.{service}.plist" for service in services)
    if IS_PUBLIC_CANDIDATE:
        if any(path.exists() for path in template_paths):
            raise SystemExit("history-free public candidate included a private service plist")
        return
    for service in services:
        text = (ROOT / f"com.jarvis-v3.{service}.plist").read_text(encoding="utf-8")
        if "JARVIS_V3_ENABLE_DAEMONS" in text or "JARVIS_V3_ENABLE_SCHEDULER" in text:
            raise SystemExit(f"{service} template bypasses a supervised daemon gate")
        if f"com.jarvis-v3.{service}" not in text or "AI agents/jarvis-v3" not in text:
            raise SystemExit(f"{service} template missed V3 identity")
        if "AI agents/jarvis-v2" in text or "com.jarvis-v2" in text:
            raise SystemExit(f"{service} template points to V2")
    dashboard_entrypoint = (ROOT / "jarvis_v2" / "scripts" / "run_status_server.py").read_text(
        encoding="utf-8"
    )
    if 'require_v3_daemon_enable("dashboard")' not in dashboard_entrypoint:
        raise SystemExit("dashboard entrypoint bypasses the supervised V3 daemon gate")


def main() -> None:
    test_v3_default_paths_in_fresh_process()
    test_production_defaults_have_no_v2_targets()
    test_explicit_v3_paths_cannot_alias_v2_state()
    test_selected_environment_cannot_redirect_v3_into_v2()
    test_v3_service_templates_are_inert_by_default()
    print("V3 runtime isolation smoke passed")


if __name__ == "__main__":
    main()
