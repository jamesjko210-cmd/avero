"""Pin the Google Calendar auth-failure diagnostics.

Found 2026-07-06: the OAuth token had been dead (`invalid_grant` on refresh)
since 06-17, but every calendar tool reported only the generic "Google
Calendar is having trouble right now. Try again in a moment." — a message
that actively suggested retrying an unrecoverable failure, hiding three weeks
of silently degraded morning briefs. This smoke pins that:

1. invalid_grant / RefreshError failures map to an actionable message naming
   the exact re-auth command (and explicitly saying retrying will not help),
2. the other error branches (missing packages, missing credentials, generic)
   still map exactly as before,
3. the interactive re-auth script imports cleanly but REFUSES to run without
   a TTY, so it can never be triggered from an autonomous/daemon context.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import call, patch

from jarvis_v2.tools.calendar_connector import _atomic_write_private_text, _calendar_error


class _FakeRefreshError(Exception):
    pass


# Mirror google.auth.exceptions.RefreshError's type name without importing it,
# so this smoke runs even if google packages are absent.
_FakeRefreshError.__name__ = "RefreshError"


def test_private_atomic_writer_preserves_exact_bytes_and_mode() -> None:
    payload = '{"token":"synthetic","escaped":"\\u2603"}\n'
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "nested" / "token.json"
        previous_umask = os.umask(0)
        try:
            _atomic_write_private_text(target, payload)
        finally:
            os.umask(previous_umask)

        if target.read_bytes() != payload.encode("utf-8"):
            raise SystemExit("private atomic writer must preserve the exact UTF-8 bytes")
        if stat.S_IMODE(target.stat().st_mode) != 0o600:
            raise SystemExit(f"private atomic writer mode must be 0600, got {stat.S_IMODE(target.stat().st_mode):04o}")


def test_private_atomic_writer_preserves_old_bytes_on_pre_replace_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "token.json"
        old_bytes = b'{"token":"old-synthetic"}\n'
        target.write_bytes(old_bytes)

        with patch("jarvis_v2.tools.calendar_connector.os.replace", side_effect=OSError("injected replace failure")):
            try:
                _atomic_write_private_text(target, '{"token":"new-synthetic"}\n')
            except OSError as exc:
                if "injected replace failure" not in str(exc):
                    raise
            else:
                raise SystemExit("private atomic writer hid the injected pre-replace failure")

        if target.read_bytes() != old_bytes:
            raise SystemExit("private atomic writer changed old bytes before replacement succeeded")
        leftovers = [path.name for path in target.parent.iterdir() if path != target]
        if leftovers:
            raise SystemExit(f"private atomic writer leaked temporary files after failure: {leftovers!r}")


class _TTYInput:
    def isatty(self) -> bool:
        return True


def test_reauth_backup_and_new_token_use_private_writer() -> None:
    from jarvis_v2.scripts import google_calendar_reauth
    from jarvis_v2.tools import calendar_connector

    class _FakeCredentials:
        expiry = "synthetic-expiry"

        def to_json(self) -> str:
            return '{"token":"new-synthetic"}'

    credentials = _FakeCredentials()

    class _FakeFlow:
        @classmethod
        def from_client_secrets_file(cls, _path: str, _scopes: list[str]) -> "_FakeFlow":
            return cls()

        def run_local_server(self, *, port: int) -> _FakeCredentials:
            if port != 0:
                raise AssertionError("synthetic flow expected an ephemeral port")
            return credentials

    class _FakeCalendarList:
        def list(self, *, maxResults: int, fields: str) -> "_FakeCalendarList":
            if maxResults != 1 or fields != "items(id)":
                raise AssertionError("unexpected synthetic verification request")
            return self

        def execute(self) -> dict[str, list[dict[str, str]]]:
            return {"items": []}

    class _FakeService:
        def calendarList(self) -> _FakeCalendarList:
            return _FakeCalendarList()

    oauth_package = ModuleType("google_auth_oauthlib")
    oauth_flow = ModuleType("google_auth_oauthlib.flow")
    oauth_flow.InstalledAppFlow = _FakeFlow  # type: ignore[attr-defined]
    api_package = ModuleType("googleapiclient")
    api_discovery = ModuleType("googleapiclient.discovery")
    api_discovery.build = lambda *_args, **_kwargs: _FakeService()  # type: ignore[attr-defined]
    fake_modules = {
        "google_auth_oauthlib": oauth_package,
        "google_auth_oauthlib.flow": oauth_flow,
        "googleapiclient": api_package,
        "googleapiclient.discovery": api_discovery,
    }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        creds_file = root / "credentials.json"
        token_file = root / "token.json"
        creds_file.write_text("synthetic credentials marker", encoding="utf-8")
        token_file.write_text('{"token":"old-synthetic"}', encoding="utf-8")

        with patch.dict(sys.modules, fake_modules), patch.object(sys, "stdin", _TTYInput()), patch.object(
            calendar_connector, "_creds_file", return_value=creds_file
        ), patch.object(calendar_connector, "_token_file", return_value=token_file), patch.object(
            calendar_connector, "_atomic_write_private_text"
        ) as private_writer:
            result = google_calendar_reauth.main()

        if result != 0:
            raise SystemExit(f"synthetic re-auth flow returned {result}, expected 0")
        if private_writer.call_count != 2:
            raise SystemExit(f"re-auth must route backup and new token through the private writer: {private_writer.mock_calls!r}")
        backup_path = private_writer.call_args_list[0].args[0]
        if backup_path.parent != token_file.parent or not backup_path.name.startswith("token.json.bak-"):
            raise SystemExit(f"re-auth backup path drifted: {backup_path!r}")
        expected_calls = [
            call(backup_path, '{"token":"old-synthetic"}'),
            call(token_file, '{"token":"new-synthetic"}'),
        ]
        if private_writer.call_args_list != expected_calls:
            raise SystemExit(f"re-auth private-writer calls drifted: {private_writer.call_args_list!r}")


def test_invalid_grant_maps_to_actionable_reauth_message() -> None:
    for exc in (
        Exception("('invalid_grant: Bad Request', {'error': 'invalid_grant'})"),
        _FakeRefreshError("token has been expired or revoked"),
    ):
        message = _calendar_error(exc, mutation=True)
        if (
            "expired" not in message
            or "./launch_jarvis_v3_calendar_auth.py full-access" not in message
            or "normal macOS Terminal shell prompt" not in message
            or "not in Telegram" not in message
        ):
            raise SystemExit(f"invalid_grant must map to the re-auth instructions, got: {message!r}")
        if "Retrying will not help" not in message:
            raise SystemExit(f"dead-token message must warn that retries are useless: {message!r}")
        if "try again in a moment" in message.lower():
            raise SystemExit("dead-token failures must NOT suggest retrying")


def test_other_error_branches_unchanged() -> None:
    cases = {
        "Google API packages not installed. Run: pip install ...": "not installed on this Mac",
        "Google credentials not found at /some/path. Run setup first.": "not connected yet",
        "totally unexpected socket timeout": "having trouble right now",
    }
    for raw, expected_fragment in cases.items():
        message = _calendar_error(Exception(raw))
        if expected_fragment not in message:
            raise SystemExit(f"error branch drifted for {raw!r}: {message!r}")
        if "having trouble right now" in message and (
            "./launch_jarvis_v3_calendar_auth.py readonly" not in message
            or "./launch_jarvis_v3_calendar_auth.py full-access" in message
        ):
            raise SystemExit("generic read-lane recovery broadened Calendar authorization")

    readonly_expired = _calendar_error(
        Exception("invalid_grant: synthetic read token expired")
    )
    if (
        "./launch_jarvis_v3_calendar_auth.py readonly" not in readonly_expired
        or "./launch_jarvis_v3_calendar_auth.py full-access" in readonly_expired
    ):
        raise SystemExit("read-lane invalid_grant suggested full-access authorization")


def test_reauth_script_refuses_without_tty() -> None:
    # Run the script with stdin as a pipe (not a TTY); it must exit 2 without
    # touching credentials or opening a browser.
    proc = subprocess.run(
        [sys.executable, "-m", "jarvis_v2.scripts.google_calendar_reauth"],
        stdin=subprocess.PIPE,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 2:
        raise SystemExit(
            f"re-auth script must refuse non-interactive runs (exit 2), got {proc.returncode}: {proc.stderr[:200]}"
        )
    if "Refusing to run" not in proc.stderr:
        raise SystemExit(f"re-auth refusal message missing: {proc.stderr[:200]}")


def main() -> None:
    test_private_atomic_writer_preserves_exact_bytes_and_mode()
    test_private_atomic_writer_preserves_old_bytes_on_pre_replace_failure()
    test_reauth_backup_and_new_token_use_private_writer()
    test_invalid_grant_maps_to_actionable_reauth_message()
    test_other_error_branches_unchanged()
    test_reauth_script_refuses_without_tty()
    print("Calendar auth smoke passed")


if __name__ == "__main__":
    main()
