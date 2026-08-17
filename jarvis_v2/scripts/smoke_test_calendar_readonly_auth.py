"""Offline proof for the separate least-privilege Calendar read lane."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from jarvis_v2.config import load_config
from jarvis_v2.scripts import google_calendar_readonly_auth
from jarvis_v2.tools import calendar_connector as cc
from jarvis_v2.tools import system as system_tools


class _TTYInput:
    def isatty(self) -> bool:
        return True


def _fake_google_modules(credentials, verification_marker: dict[str, object]) -> dict[str, ModuleType]:
    class FakeFlow:
        received_scopes: list[str] = []

        @classmethod
        def from_client_secrets_file(cls, _path: str, scopes: list[str]):
            cls.received_scopes = list(scopes)
            return cls()

        def run_local_server(self, *, port: int):
            if port != 0:
                raise AssertionError("read-only OAuth must use an ephemeral local port")
            return credentials

    class FakeCalendarList:
        def list(self, *, maxResults: int, fields: str):
            verification_marker["request"] = (maxResults, fields)
            return self

        def execute(self):
            verification_marker["executed"] = True
            return {"items": [{"id": "PRIVATE-ID-MUST-NOT-BE-PRINTED"}]}

    class FakeService:
        def calendarList(self):
            return FakeCalendarList()

    oauth_package = ModuleType("google_auth_oauthlib")
    oauth_flow = ModuleType("google_auth_oauthlib.flow")
    oauth_flow.InstalledAppFlow = FakeFlow  # type: ignore[attr-defined]
    api_package = ModuleType("googleapiclient")
    api_discovery = ModuleType("googleapiclient.discovery")
    api_discovery.build = lambda *_args, **_kwargs: FakeService()  # type: ignore[attr-defined]
    return {
        "google_auth_oauthlib": oauth_package,
        "google_auth_oauthlib.flow": oauth_flow,
        "googleapiclient": api_package,
        "googleapiclient.discovery": api_discovery,
    }


def test_scope_and_path_separation() -> None:
    expected = {
        "https://www.googleapis.com/auth/calendar.events.readonly",
        "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    }
    if set(cc._READONLY_SCOPES) != expected or len(cc._READONLY_SCOPES) != 2:
        raise SystemExit(f"Calendar read scopes are not the exact least-privilege pair: {cc._READONLY_SCOPES}")
    if set(cc._READONLY_SCOPES) & set(cc._SCOPES):
        raise SystemExit("read-only and mutation scope sets must remain distinct")
    if cc._readonly_token_file() == cc._token_file():
        raise SystemExit("read-only and mutation token paths must remain distinct")


def test_interactive_auth_writes_only_read_token() -> None:
    class FakeCredentials:
        scopes = list(cc._READONLY_SCOPES)
        granted_scopes = list(cc._READONLY_SCOPES)

        def to_json(self) -> str:
            return json.dumps({"token": "synthetic-read-token", "scopes": self.scopes})

    marker: dict[str, object] = {}
    fake_modules = _fake_google_modules(FakeCredentials(), marker)
    with tempfile.TemporaryDirectory(prefix="jarvis-calendar-read-auth-") as temp:
        root = Path(temp)
        creds_file = root / "credentials.json"
        read_token = root / "readonly-token.json"
        write_token = root / "write-token.json"
        creds_file.write_text("synthetic desktop client", encoding="utf-8")
        creds_file.chmod(0o600)
        write_bytes = b'{"token":"broad-token-must-not-change"}'
        write_token.write_bytes(write_bytes)
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output), patch.dict(sys.modules, fake_modules), patch.object(
            sys, "stdin", _TTYInput()
        ), patch.object(cc, "_creds_file", return_value=creds_file), patch.object(
            cc, "_readonly_token_file", return_value=read_token
        ), patch.object(cc, "_token_file", return_value=write_token):
            result = google_calendar_readonly_auth.main()
        if result != 0 or not read_token.is_file():
            raise SystemExit("interactive read-only auth did not create its separate token")
        if stat.S_IMODE(read_token.stat().st_mode) != 0o600:
            raise SystemExit("read-only token must be owner-only mode 0600")
        payload = json.loads(read_token.read_text(encoding="utf-8"))
        if set(payload.get("scopes", [])) != set(cc._READONLY_SCOPES):
            raise SystemExit(f"saved read-only token scopes drifted: {payload.get('scopes')}")
        if write_token.read_bytes() != write_bytes:
            raise SystemExit("read-only auth changed the full-access token")
        if marker != {"request": (1, "items(id)"), "executed": True}:
            raise SystemExit(f"content-free verification request drifted: {marker}")
        if "PRIVATE-ID-MUST-NOT-BE-PRINTED" in output.getvalue():
            raise SystemExit("read-only authorization printed calendar content")


def test_auth_rejects_scope_expansion_without_writing() -> None:
    class BroadCredentials:
        scopes = [*cc._READONLY_SCOPES, "https://www.googleapis.com/auth/calendar"]
        granted_scopes = list(scopes)

        def to_json(self) -> str:
            return json.dumps({"token": "must-not-save", "scopes": self.scopes})

    marker: dict[str, object] = {}
    fake_modules = _fake_google_modules(BroadCredentials(), marker)
    with tempfile.TemporaryDirectory(prefix="jarvis-calendar-read-scope-") as temp:
        root = Path(temp)
        creds_file = root / "credentials.json"
        read_token = root / "readonly-token.json"
        creds_file.write_text("synthetic desktop client", encoding="utf-8")
        creds_file.chmod(0o600)
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output), patch.dict(sys.modules, fake_modules), patch.object(
            sys, "stdin", _TTYInput()
        ), patch.object(cc, "_creds_file", return_value=creds_file), patch.object(
            cc, "_readonly_token_file", return_value=read_token
        ):
            result = google_calendar_readonly_auth.main()
        if result != 1 or read_token.exists() or marker:
            raise SystemExit("expanded OAuth scopes must fail before token write or API verification")


def test_runtime_rejects_missing_and_wrong_scope_tokens_before_api() -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-calendar-read-runtime-") as temp:
        root = Path(temp)
        missing = root / "missing.json"
        with patch.object(cc, "_readonly_token_file", return_value=missing):
            try:
                cc._get_readonly_service()
            except cc.GoogleCalendarReadonlyAuthError:
                pass
            else:
                raise SystemExit("missing read-only token must fail closed")

        wrong = root / "wrong.json"
        wrong.write_text(
            json.dumps({"token": "synthetic", "scopes": ["https://www.googleapis.com/auth/calendar"]}),
            encoding="utf-8",
        )
        wrong.chmod(0o600)
        with patch.object(cc, "_readonly_token_file", return_value=wrong):
            try:
                cc._get_readonly_service()
            except cc.GoogleCalendarReadonlyAuthError:
                pass
            else:
                raise SystemExit("broad token in read-only path must fail closed")


def test_auth_refuses_without_tty() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "jarvis_v2.scripts.google_calendar_readonly_auth"],
        stdin=subprocess.PIPE,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 2 or "Refusing to run" not in proc.stderr:
        raise SystemExit(f"read-only auth must refuse autonomous execution: {proc.returncode} {proc.stderr[:200]}")


def test_setup_file_status_rejects_public_and_symlink_tokens() -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-calendar-read-status-") as temp:
        root = Path(temp)
        token = root / "readonly.json"
        token.write_text("synthetic", encoding="utf-8")
        token.chmod(0o644)
        public_status = system_tools._private_required_file_status("JARVIS_GOOGLE_READONLY_TOKEN", token)
        if public_status["valid"] or public_status["owner_only"]:
            raise SystemExit("setup must reject a group/world-readable read-only token")
        token.chmod(0o600)
        private_status = system_tools._private_required_file_status("JARVIS_GOOGLE_READONLY_TOKEN", token)
        if not private_status["valid"] or private_status["is_symlink"]:
            raise SystemExit("setup must accept an owner-only regular read-only token")
        link = root / "linked.json"
        link.symlink_to(token)
        link_status = system_tools._private_required_file_status("JARVIS_GOOGLE_READONLY_TOKEN", link)
        if link_status["valid"] or not link_status["is_symlink"]:
            raise SystemExit("setup must reject symlinked read-only tokens")


def test_read_and_mutation_tools_use_separate_services() -> None:
    class Exec:
        def __init__(self, payload):
            self.payload = payload

        def execute(self):
            return self.payload

    class CalendarList:
        def list(self):
            return Exec({"items": []})

    class Events:
        def list(self, **_kwargs):
            return Exec({"items": []})

        def insert(self, *, calendarId, body):
            return Exec({"id": "synthetic", "summary": body["summary"], "start": body["start"]})

        def patch(self, *, calendarId, eventId, body):
            return Exec({"id": eventId, "summary": body.get("summary", "synthetic")})

        def delete(self, *, calendarId, eventId):
            return Exec({})

    class Service:
        def calendarList(self):
            return CalendarList()

        def events(self):
            return Events()

    calls = {"read": 0, "write": 0}

    def read_service():
        calls["read"] += 1
        return Service()

    def write_service():
        calls["write"] += 1
        return Service()

    with patch.object(cc, "_get_readonly_service", side_effect=read_service), patch.object(
        cc, "_get_service", side_effect=write_service
    ):
        tools = {tool.name: tool for tool in cc.make_calendar_tools(load_config())}
        if not tools["list_calendars"].handler({}).ok:
            raise SystemExit("calendar-list read failed on the read-only service")
        if not tools["list_events"].handler({"range": "today"}).ok:
            raise SystemExit("event-list read failed on the read-only service")
        if not tools["check_availability"].handler({"text": "am I free today"}).ok:
            raise SystemExit("availability read failed on the read-only service")
        if calls != {"read": 3, "write": 0}:
            raise SystemExit(f"Calendar reads crossed into mutation service: {calls}")

        mutation_args = {
            "create_event": {"title": "Synthetic", "start": "2030-01-01", "calendar_id": "synthetic-calendar"},
            "update_event": {"event_id": "synthetic", "title": "Synthetic", "calendar_id": "synthetic-calendar"},
            "delete_event": {"event_id": "synthetic", "calendar_id": "synthetic-calendar"},
        }
        for tool_name, args in mutation_args.items():
            if not tools[tool_name].handler(args).ok:
                raise SystemExit(f"{tool_name} failed on the mutation service")
        if calls != {"read": 3, "write": 3}:
            raise SystemExit(f"Calendar mutations crossed into read-only service: {calls}")


def main() -> None:
    test_scope_and_path_separation()
    test_interactive_auth_writes_only_read_token()
    test_auth_rejects_scope_expansion_without_writing()
    test_runtime_rejects_missing_and_wrong_scope_tokens_before_api()
    test_auth_refuses_without_tty()
    test_setup_file_status_rejects_public_and_symlink_tokens()
    test_read_and_mutation_tools_use_separate_services()
    print("Calendar read-only auth smoke passed")


if __name__ == "__main__":
    main()
