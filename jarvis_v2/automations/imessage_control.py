"""Owner-locked iMessage command channel for Jarvis V2.

The bridge is intentionally narrow:
- only the owner identifier from JARVIS_OWNER_IMESSAGE is accepted
- only inbound messages starting with "Jarvis" are treated as commands
- a persisted ROWID cursor prevents replay on restart
- a fresh install seeds past existing history (never replays old texts)
- same-Apple-ID self-thread is supported (commands arrive is_from_me=1)
- commands run through the V2 JarvisRuntime so the normal safety harness
  (planner, registry, permission policy, approval queue, audit log) still gates
  every side effect
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterable


STATE_FILE: Path | None = None
MESSAGES_DB = os.path.expanduser("~/Library/Messages/chat.db")
POLL_INTERVAL_SECONDS = 3.0
TRIGGER = "jarvis"
# Same-Apple-ID self-threads record one message twice (a sent copy is_from_me=1
# and a received copy is_from_me=0). Collapse identical commands seen within this
# window so the owner gets exactly one reply per command.
DEDUP_WINDOW_SECONDS = 12.0
MAX_TRANSPORT_ROWID = (1 << 63) - 1
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
_STOP = threading.Event()
_THREAD: threading.Thread | None = None


def _clean(value) -> str:
    return "" if value is None else str(value).strip()


def _safe_outbound_text(value: str) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", _clean(value))


def _has_local_path(value: str) -> bool:
    raw = _clean(value)
    return bool(raw and (Path(raw).is_absolute() or "/" in raw or "\\" in raw))


def _owner_status() -> dict[str, object]:
    raw = _clean(os.getenv("JARVIS_OWNER_IMESSAGE", ""))
    configured = bool(raw)
    valid = configured and not _has_local_path(raw) and any(ch.isalnum() for ch in raw)
    return {
        "value": raw,
        "configured": configured,
        "valid": valid,
        "reason": "missing_owner" if not configured else "invalid_owner",
    }


def _owner_identifier() -> str:
    status = _owner_status()
    return str(status["value"]) if status["valid"] else ""


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _owner_matches(sender: str, owner: str) -> bool:
    sender = _clean(sender)
    owner = _clean(owner)
    if not sender or not owner:
        return False
    if sender.casefold() == owner.casefold():
        return True
    sender_digits = _digits(sender)
    owner_digits = _digits(owner)
    # Compare the last 8 digits so +82 / 010 country-code variants of one phone
    # number still match (Korean mobile numbers share the trailing subscriber id).
    if sender_digits and owner_digits:
        if sender_digits == owner_digits:
            return True
        tail = min(len(sender_digits), len(owner_digits), 8)
        if tail >= 7 and sender_digits[-tail:] == owner_digits[-tail:]:
            return True
    return False


def _command_from_text(text: str) -> str | None:
    text = _clean(text)
    if not text:
        return None
    low = text.casefold()
    if low == TRIGGER:
        return "How can I help?"
    prefix = TRIGGER + " "
    if not low.startswith(prefix):
        return None
    return text[len(prefix):].strip() or "How can I help?"


def _state_file() -> Path:
    if STATE_FILE is not None:
        return STATE_FILE
    raw = os.getenv("JARVIS_V3_IMESSAGE_STATE", "").strip()
    return Path(raw or Path.home() / ".jarvis_v3" / "imessage_control.json").expanduser()


def _load_state() -> dict:
    """Return the persisted cursor. last_rowid is None when never initialized
    (fresh install) so process_once can seed past existing history instead of
    replaying old "Jarvis ..." texts."""
    try:
        state_file = _state_file()
        if state_file.exists():
            data = json.loads(state_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                last_rowid = data.get("last_rowid", 0)
                if isinstance(last_rowid, int) and last_rowid >= 0:
                    return {"last_rowid": last_rowid}
    except Exception:
        try:
            state_file.replace(state_file.with_suffix(state_file.suffix + ".corrupt"))
        except Exception:
            pass
    return {"last_rowid": None}


def _save_state(state: dict) -> None:
    try:
        state_file = _state_file()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_suffix(state_file.suffix + ".tmp")
        tmp.write_text(json.dumps({"last_rowid": int(state.get("last_rowid", 0))}), encoding="utf-8")
        tmp.replace(state_file)
    except Exception:
        pass


def _current_max_rowid(db_path: str) -> int | None:
    """Highest message ROWID, or None if the database is unreadable.

    None (not 0) on error is critical: if the bridge starts before it has Full
    Disk Access and we seeded 0, the next readable poll would fetch every
    historical message > 0 and replay old "Jarvis ..." commands. Returning None
    lets process_once defer seeding until the DB is actually readable."""
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message").fetchone()
        finally:
            conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return None


def _fetch_new_messages(db_path: str, after_rowid: int, limit: int = 50) -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT m.ROWID, m.text, m.is_from_me, COALESCE(h.id, ''),
                   COALESCE(c.chat_identifier, '')
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.ROWID > ?
            ORDER BY m.ROWID ASC
            LIMIT ?
            """,
            (int(after_rowid), int(limit)),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "rowid": int(rowid),
            "text": text or "",
            "is_from_me": int(is_from_me or 0),
            "sender": sender or "",
            "chat_identifier": chat_id or "",
        }
        for rowid, text, is_from_me, sender, chat_id in rows
    ]


def _rowid_request_token(rowid: object) -> str | None:
    if type(rowid) is not int or rowid <= 0 or rowid > MAX_TRANSPORT_ROWID:
        return None
    return f"imessage-row:v1:{rowid}"


def _safe_reply_text(reply: str) -> str:
    """Guarantee a reply never begins with the trigger word, so Jarvis's own
    outgoing messages are never re-read as commands in a same-Apple-ID
    self-thread (where replies arrive with is_from_me=1)."""
    text = _safe_outbound_text(reply)
    if text.casefold().startswith(TRIGGER):
        return "↩ " + text  # leading arrow breaks the trigger prefix
    return text


def _runtime_error_reply(exc: Exception) -> str:
    return (
        "Jarvis hit an internal error while handling that iMessage command. "
        f"Try again or check the local Jarvis logs. ({type(exc).__name__})"
    )


def _send_imessage(recipient: str, message: str) -> str:
    safe = message.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy "{recipient}" of targetService
    send "{safe}" to targetBuddy
end tell'''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "osascript failed")
    return result.stdout.strip()


class IMessageCommandBridge:
    def __init__(
        self,
        *,
        runtime_factory: Callable[[], object] | None = None,
        send_func: Callable[[str, str], str] = _send_imessage,
        fetch_func: Callable[[str, int, int], Iterable[dict]] = _fetch_new_messages,
        db_path: str = MESSAGES_DB,
    ):
        self.runtime_factory = runtime_factory
        self.send_func = send_func
        self.fetch_func = fetch_func
        self.db_path = db_path
        self._runtime = None
        # command text -> monotonic time of last handling, for self-thread dedup
        self._recent_commands: dict[str, float] = {}

    def _is_duplicate(self, command: str, now: float) -> bool:
        """True if this exact command was handled within DEDUP_WINDOW_SECONDS.

        Same-Apple-ID self-threads store one message as both a sent and a
        received row; without this the owner would get one reply per copy."""
        # Drop stale entries so the map cannot grow without bound.
        for text in [t for t, ts in self._recent_commands.items() if now - ts > DEDUP_WINDOW_SECONDS]:
            self._recent_commands.pop(text, None)
        if command in self._recent_commands and now - self._recent_commands[command] <= DEDUP_WINDOW_SECONDS:
            self._recent_commands[command] = now
            return True
        self._recent_commands[command] = now
        return False

    def _runtime_instance(self):
        if self._runtime is None:
            if self.runtime_factory is None:
                from jarvis_v2.agent.runtime import JarvisRuntime
                self._runtime = JarvisRuntime()
            else:
                self._runtime = self.runtime_factory()
        return self._runtime

    def process_once(self) -> int:
        owner = _owner_identifier()
        if not owner:
            return 0

        state = _load_state()
        if state.get("last_rowid") is None:
            seed = _current_max_rowid(self.db_path)
            # Defer seeding if the DB is unreadable (e.g. Full Disk Access not yet
            # granted) so we never seed 0 and later replay full history.
            if seed is None:
                return 0
            _save_state({"last_rowid": seed})
            return 0
        last_rowid = int(state.get("last_rowid", 0))
        processed = 0
        max_seen = last_rowid
        try:
            messages = list(self.fetch_func(self.db_path, last_rowid, 50))
        except Exception:
            return 0

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            rowid = msg.get("rowid")
            request_token = _rowid_request_token(rowid)
            if request_token is None:
                continue
            max_seen = max(max_seen, rowid)
            sender = _clean(msg.get("sender", ""))
            chat_id = _clean(msg.get("chat_identifier", ""))
            if not (_owner_matches(sender, owner) or _owner_matches(chat_id, owner)):
                continue
            command = _command_from_text(msg.get("text", ""))
            if command is None:
                continue
            # Collapse the duplicate sent/received copies of a self-thread message.
            if self._is_duplicate(command, time.monotonic()):
                continue
            reply = self._handle_command(command, request_token=request_token)
            if reply:
                try:
                    self.send_func(owner, _safe_reply_text(reply))
                except Exception:
                    pass
            processed += 1

        if max_seen != last_rowid:
            state["last_rowid"] = max_seen
            _save_state(state)
        return processed

    def _handle_command(self, command: str, *, request_token: str) -> str:
        try:
            result = self._runtime_instance().handle(
                command,
                request_token=request_token,
            )
            return str(getattr(result, "response", "") or "Done.")
        except Exception as e:
            return _runtime_error_reply(e)

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event or _STOP
        while not stop.is_set():
            self.process_once()
            stop.wait(POLL_INTERVAL_SECONDS)


def start_imessage_control() -> str:
    """Start the owner-locked iMessage control bridge if configured."""
    global _THREAD
    owner_status = _owner_status()
    if not owner_status["configured"]:
        return "iMessage control disabled: JARVIS_OWNER_IMESSAGE is not set."
    if not owner_status["valid"]:
        return "iMessage control disabled: JARVIS_OWNER_IMESSAGE is invalid (value hidden)."
    if _THREAD is not None and _THREAD.is_alive():
        return "iMessage control already running."
    _STOP.clear()
    bridge = IMessageCommandBridge()
    _THREAD = threading.Thread(target=bridge.run_forever, name="JarvisV2IMessageControl", daemon=True)
    _THREAD.start()
    return "iMessage control started."


def stop_imessage_control() -> str:
    _STOP.set()
    return "iMessage control stopped."
