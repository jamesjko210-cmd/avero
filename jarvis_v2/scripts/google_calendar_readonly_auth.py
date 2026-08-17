"""Create Jarvis's separate, least-privilege Google Calendar read token.

This is intentionally human-only. Ordinary Jarvis reads never open a browser,
never fall back to the full-access Calendar token, and never broaden scopes.

Run from a normal terminal:
    JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_calendar_auth.py readonly
"""

from __future__ import annotations

import json
import stat
import sys
from datetime import datetime, timezone


def _exact_granted_scopes(credentials, expected: list[str]) -> bool:
    granted = getattr(credentials, "granted_scopes", None)
    if granted is None:
        granted = getattr(credentials, "scopes", None)
    return (
        isinstance(granted, (list, tuple, set, frozenset))
        and len(granted) == len(expected)
        and set(granted) == set(expected)
    )


def main() -> int:
    if not sys.stdin.isatty():
        print(
            "Refusing to run: read-only Calendar authorization opens an interactive "
            "Google consent page and must be run by the operator in a real terminal.",
            file=sys.stderr,
        )
        return 2

    from jarvis_v2.tools import calendar_connector as cc

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        print(
            "Google API packages are not installed. Run `setup check`, install the "
            "reported packages, then retry.",
            file=sys.stderr,
        )
        return 1

    creds_file = cc._creds_file()
    token_file = cc._readonly_token_file()
    try:
        creds_stat = creds_file.stat()
    except OSError:
        print(
            "Google OAuth desktop credentials are not configured. Add the owner-only "
            "credentials file reported by `setup check`, then retry.",
            file=sys.stderr,
        )
        return 1
    if creds_file.is_symlink() or not stat.S_ISREG(creds_stat.st_mode) or stat.S_IMODE(creds_stat.st_mode) & 0o077:
        print(
            "Google OAuth desktop credentials must be a non-symlink, owner-only file. "
            "Run `chmod 600` on the credentials file, then retry.",
            file=sys.stderr,
        )
        return 1

    if token_file.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = token_file.with_name(token_file.name + f".bak-{stamp}")
        try:
            cc._atomic_write_private_text(backup, token_file.read_text(encoding="utf-8"))
        except OSError:
            print("The existing read-only token could not be backed up safely; nothing changed.", file=sys.stderr)
            return 1
        print("Backed up the existing read-only token.")

    print("Opening Google's read-only Calendar consent page in your browser...")
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), cc._READONLY_SCOPES)
    credentials = flow.run_local_server(port=0)
    if not _exact_granted_scopes(credentials, cc._READONLY_SCOPES):
        print(
            "Google returned a scope set that was not exactly the requested read-only "
            "Calendar scopes. No token was saved.",
            file=sys.stderr,
        )
        return 1

    try:
        serialized = json.loads(credentials.to_json())
        if not isinstance(serialized, dict):
            raise ValueError("credential serialization was not an object")
        serialized["scopes"] = list(cc._READONLY_SCOPES)
        token_json = json.dumps(serialized, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError):
        print("Google returned an unreadable credential response. No token was saved.", file=sys.stderr)
        return 1

    print("Verifying read-only Calendar access without displaying calendar content...")
    try:
        service = build("calendar", "v3", credentials=credentials)
        service.calendarList().list(maxResults=1, fields="items(id)").execute()
    except Exception:
        print("Google Calendar verification failed. No new token was saved.", file=sys.stderr)
        return 1
    cc._atomic_write_private_text(token_file, token_json)
    print("OK — Google Calendar read-only access is connected. No calendar content was printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
