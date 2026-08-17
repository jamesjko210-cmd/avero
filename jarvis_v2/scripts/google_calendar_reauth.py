"""Re-authenticate Jarvis's Google Calendar access (interactive; the operator runs this).

Why this exists: the stored OAuth token can die permanently (`invalid_grant`)
— e.g. Google Cloud "Testing"-mode apps expire refresh tokens after ~7 days,
or the grant is revoked by a password/security change. When that happens no
automatic refresh will ever succeed; the only fix is a fresh browser consent,
which must be a human action. Jarvis never runs this on its own — the calendar
tools fail closed with a message pointing here instead.

Usage (in a normal terminal, NOT autonomous):
    JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_calendar_auth.py full-access

What it does:
1. Backs up the current token file next to itself (token.json.bak-<UTC stamp>).
2. Opens the Google consent page in your browser (InstalledAppFlow).
3. Writes the fresh full-access mutation token, then verifies it without printing calendar content.

If this keeps expiring every ~7 days, open the Google Cloud Console for this
OAuth client and publish the app from "Testing" to "In production" — Testing
mode is what puts a 7-day lifetime on refresh tokens.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone


def main() -> int:
    if not sys.stdin.isatty():
        print(
            "Refusing to run: this script opens an interactive Google consent "
            "page and must be run by a human in a real terminal.",
            file=sys.stderr,
        )
        return 2

    from jarvis_v2.tools import calendar_connector as cc

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        print(
            "Google API packages are not installed. Run:\n"
            "  pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client",
            file=sys.stderr,
        )
        return 1

    creds_file = cc._creds_file()
    token_file = cc._token_file()
    if not creds_file.exists():
        print(f"Google credentials file not found at {creds_file}. Add it first.", file=sys.stderr)
        return 1

    if token_file.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = token_file.with_name(token_file.name + f".bak-{stamp}")
        cc._atomic_write_private_text(backup, token_file.read_text(encoding="utf-8"))
        print(f"Backed up existing token to {backup.name}")

    print("WARNING: this flow requests full Google Calendar read/write access for approval-gated mutations.")
    print("Opening Google consent page in your browser...")
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), cc._SCOPES)
    creds = flow.run_local_server(port=0)
    cc._atomic_write_private_text(token_file, creds.to_json())
    print(f"New token written. Expiry: {creds.expiry}")

    print("Verifying access without displaying calendar content...")
    service = build("calendar", "v3", credentials=creds)
    service.calendarList().list(maxResults=1, fields="items(id)").execute()
    print("OK — full-access Calendar authorization verified. No calendar content was printed.")
    print("Done. Approval-gated Calendar mutations can use this separate token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
