"""Smoke tests for the Jarvis V2 owner-locked iMessage control bridge.

Uses a fake chat.db so it runs without Full Disk Access.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from jarvis_v2.automations import imessage_control as ic


class FakeRuntime:
    def __init__(self, replies):
        self.replies = list(replies)
        self.inputs = []
        self.request_tokens = []

    def handle(self, text, request_token=None):
        self.inputs.append(text)
        self.request_tokens.append(request_token)
        reply = self.replies.pop(0) if self.replies else "ok"

        class R:
            response = reply

        return R()


def _make_db(path: str, rows: list[tuple]) -> None:
    """rows: (sender, text, is_from_me[, chat_identifier])."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
        conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT, is_from_me INTEGER, handle_id INTEGER)")
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        handles, chats = {}, {}
        for row in rows:
            sender, text, is_from_me = row[0], row[1], row[2]
            chat_id_value = row[3] if len(row) > 3 else sender
            if sender not in handles:
                conn.execute("INSERT INTO handle (id) VALUES (?)", (sender,))
                handles[sender] = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            if chat_id_value not in chats:
                conn.execute("INSERT INTO chat (chat_identifier) VALUES (?)", (chat_id_value,))
                chats[chat_id_value] = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("INSERT INTO message (text, is_from_me, handle_id) VALUES (?, ?, ?)", (text, is_from_me, handles[sender]))
            mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)", (chats[chat_id_value], mid))
        conn.commit()
    finally:
        conn.close()


def _fresh_state() -> None:
    ic.STATE_FILE = Path(tempfile.mkdtemp()) / "cursor.json"


def _seed_cursor_zero() -> None:
    _fresh_state()
    ic._save_state({"last_rowid": 0})


def test_owner_trigger_runs_and_cursor_prevents_replay() -> None:
    _seed_cursor_zero()
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "chat.db")
        _make_db(db, [("+15551234567", "Jarvis status", 0)])
        rt = FakeRuntime(["All systems nominal."])
        sent = []
        bridge = ic.IMessageCommandBridge(runtime_factory=lambda: rt, send_func=lambda to, t: sent.append((to, t)), db_path=db)
        os.environ["JARVIS_OWNER_IMESSAGE"] = "+15551234567"
        if bridge.process_once() != 1:
            raise SystemExit("owner trigger not processed")
        if bridge.process_once() != 0:
            raise SystemExit("cursor did not prevent replay")
    if rt.inputs != ["status"]:
        raise SystemExit(f"trigger not stripped: {rt.inputs}")
    if rt.request_tokens != ["imessage-row:v1:1"]:
        raise SystemExit(f"owner command missed its stable ROWID token: {rt.request_tokens}")
    if sent != [("+15551234567", "All systems nominal.")]:
        raise SystemExit(f"reply not sent to owner: {sent}")


def test_non_owner_and_no_trigger_ignored() -> None:
    _seed_cursor_zero()
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "chat.db")
        _make_db(db, [
            ("+19998887777", "Jarvis ignore me", 0),
            ("+15551234567", "hello no trigger", 0),
        ])
        rt = FakeRuntime(["nope"])
        sent = []
        bridge = ic.IMessageCommandBridge(runtime_factory=lambda: rt, send_func=lambda to, t: sent.append((to, t)), db_path=db)
        os.environ["JARVIS_OWNER_IMESSAGE"] = "+15551234567"
        if bridge.process_once() != 0:
            raise SystemExit("non-owner / no-trigger messages must be ignored")
    if rt.inputs or sent:
        raise SystemExit(f"ignored messages leaked: {rt.inputs} {sent}")


def test_malformed_fetch_rowid_is_ignored() -> None:
    _seed_cursor_zero()
    os.environ["JARVIS_OWNER_IMESSAGE"] = "+15551234567"
    rt = FakeRuntime(["ok"])
    sent = []
    bridge = ic.IMessageCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda to, t: sent.append((to, t)),
        fetch_func=lambda db_path, after_rowid, limit: [
            {"rowid": "not-an-int", "sender": "+15551234567", "chat_identifier": "+15551234567", "text": "Jarvis bad", "is_from_me": 0},
            {"rowid": 2, "sender": "+15551234567", "chat_identifier": "+15551234567", "text": "Jarvis good", "is_from_me": 0},
        ],
        db_path="/unused/chat.db",
    )
    if bridge.process_once() != 1:
        raise SystemExit("malformed rowid should be skipped while valid command is processed")
    if rt.inputs != ["good"] or sent != [("+15551234567", "ok")]:
        raise SystemExit(f"malformed rowid handling wrong: inputs={rt.inputs} sent={sent}")
    if ic._load_state()["last_rowid"] != 2:
        raise SystemExit(f"cursor should advance to valid max rowid: {ic._load_state()}")


def test_transport_row_tokens_are_exact_stable_and_content_free() -> None:
    _seed_cursor_zero()
    os.environ["JARVIS_OWNER_IMESSAGE"] = "+15551234567"
    rt = FakeRuntime(["first", "replay", "distinct"])
    original_save_state = ic._save_state
    try:
        ic._save_state = lambda state: None
        first = ic.IMessageCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda to, text: None,
            fetch_func=lambda db_path, after_rowid, limit: [
                {
                    "rowid": 77,
                    "sender": "+15551234567",
                    "chat_identifier": "+15551234567",
                    "text": "Jarvis private command text",
                    "is_from_me": 0,
                }
            ],
            db_path="/unused/chat.db",
        )
        restarted = ic.IMessageCommandBridge(
            runtime_factory=lambda: rt,
            send_func=lambda to, text: None,
            fetch_func=lambda db_path, after_rowid, limit: [
                {
                    "rowid": 77,
                    "sender": "+15551234567",
                    "chat_identifier": "+15551234567",
                    "text": "Jarvis changed replay text",
                    "is_from_me": 0,
                }
            ],
            db_path="/unused/chat.db",
        )
        if first.process_once() != 1 or restarted.process_once() != 1:
            raise SystemExit("same iMessage transport row did not replay through the bridge")
    finally:
        ic._save_state = original_save_state

    third = ic.IMessageCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda to, text: None,
        fetch_func=lambda db_path, after_rowid, limit: [
            {
                "rowid": 78,
                "sender": "+15551234567",
                "chat_identifier": "+15551234567",
                "text": "Jarvis private command text",
                "is_from_me": 0,
            }
        ],
        db_path="/unused/chat.db",
    )
    if third.process_once() != 1:
        raise SystemExit("distinct iMessage transport row did not run")
    expected = [
        "imessage-row:v1:77",
        "imessage-row:v1:77",
        "imessage-row:v1:78",
    ]
    if rt.request_tokens != expected:
        raise SystemExit(f"iMessage transport tokens were not deterministic and distinct: {rt.request_tokens}")
    if any("private" in token or "command" in token or len(token) > 64 for token in rt.request_tokens):
        raise SystemExit(f"iMessage transport token contained command content or was unbounded: {rt.request_tokens}")


def test_malformed_transport_rowid_types_fail_closed() -> None:
    _seed_cursor_zero()
    os.environ["JARVIS_OWNER_IMESSAGE"] = "+15551234567"
    rt = FakeRuntime(["ok"])
    invalid_ids = [
        True,
        1.0,
        "1",
        0,
        -1,
        ic.MAX_TRANSPORT_ROWID + 1,
        None,
    ]
    rows = [
        {
            "rowid": value,
            "sender": "+15551234567",
            "chat_identifier": "+15551234567",
            "text": "Jarvis must not run",
            "is_from_me": 0,
        }
        for value in invalid_ids
    ]
    rows.append(
        {
            "rowid": 19,
            "sender": "+15551234567",
            "chat_identifier": "+15551234567",
            "text": "Jarvis valid command",
            "is_from_me": 0,
        }
    )
    bridge = ic.IMessageCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda to, text: None,
        fetch_func=lambda db_path, after_rowid, limit: rows,
        db_path="/unused/chat.db",
    )
    if bridge.process_once() != 1:
        raise SystemExit("malformed iMessage identities affected valid row processing")
    if rt.inputs != ["valid command"] or rt.request_tokens != ["imessage-row:v1:19"]:
        raise SystemExit(
            f"malformed iMessage identities reached runtime: {rt.inputs} / {rt.request_tokens}"
        )


def test_same_apple_id_self_thread() -> None:
    _seed_cursor_zero()
    owner = "the operator@example.com"
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "chat.db")
        _make_db(db, [(owner, "Jarvis what's my battery", 1, owner)])  # is_from_me=1 self-thread
        rt = FakeRuntime(["Battery at 80%."])
        sent = []
        bridge = ic.IMessageCommandBridge(runtime_factory=lambda: rt, send_func=lambda to, t: sent.append((to, t)), db_path=db)
        os.environ["JARVIS_OWNER_IMESSAGE"] = owner
        if bridge.process_once() != 1:
            raise SystemExit("self-thread owner command (is_from_me=1) not processed")
    if rt.inputs != ["what's my battery"]:
        raise SystemExit(f"self-thread command wrong: {rt.inputs}")


def test_fresh_install_skips_history() -> None:
    _fresh_state()  # no state = fresh install
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "chat.db")
        _make_db(db, [("+15551234567", "Jarvis delete everything", 0)])
        rt = FakeRuntime(["should never run"])
        sent = []
        bridge = ic.IMessageCommandBridge(runtime_factory=lambda: rt, send_func=lambda to, t: sent.append((to, t)), db_path=db)
        os.environ["JARVIS_OWNER_IMESSAGE"] = "+15551234567"
        if bridge.process_once() != 0 or rt.inputs or sent:
            raise SystemExit("fresh install replayed history")
        # a genuinely new message after seeding IS processed
        conn = sqlite3.connect(db)
        try:
            conn.execute("INSERT INTO message (text, is_from_me, handle_id) VALUES ('Jarvis status', 0, 1)")
            mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, ?)", (mid,))
            conn.commit()
        finally:
            conn.close()
        if bridge.process_once() != 1:
            raise SystemExit("new message after seed not processed")


def test_unreadable_db_defers_seed() -> None:
    _fresh_state()
    os.environ["JARVIS_OWNER_IMESSAGE"] = "+15551234567"
    bridge = ic.IMessageCommandBridge(runtime_factory=lambda: None, send_func=lambda a, b: None, db_path="/nonexistent/chat.db")
    if bridge.process_once() != 0:
        raise SystemExit("unreadable DB should process nothing")
    if ic._load_state()["last_rowid"] is not None:
        raise SystemExit("unreadable DB must NOT seed a cursor (replay risk)")


def test_path_shaped_owner_is_rejected_before_fetch_runtime_or_send() -> None:
    _seed_cursor_zero()
    raw_owner = "/\x55sers/example/private/imessage-owner"
    os.environ["JARVIS_OWNER_IMESSAGE"] = raw_owner
    rt = FakeRuntime(["should not run"])
    touched = []
    bridge = ic.IMessageCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda to, text: touched.append(("send", to, text)) or (_ for _ in ()).throw(AssertionError("iMessage send should not start")),
        fetch_func=lambda db_path, after_rowid, limit: touched.append(("fetch", db_path, after_rowid)) or (_ for _ in ()).throw(AssertionError("Messages fetch should not start")),
        db_path="/unused/chat.db",
    )
    if bridge.process_once() != 0:
        raise SystemExit("path-shaped owner should disable iMessage processing")
    if rt.inputs or touched:
        raise SystemExit(f"path-shaped owner reached runtime or side effects: inputs={rt.inputs} touched={touched}")
    message = ic.start_imessage_control()
    if "invalid" not in message or "value hidden" not in message:
        raise SystemExit(f"startup should report hidden invalid iMessage owner: {message}")
    if raw_owner in message:
        raise SystemExit(f"startup leaked path-shaped iMessage owner: {message}")


def test_self_thread_duplicate_command_replies_once() -> None:
    """A same-Apple-ID self-thread stores one command as both a sent (is_from_me=1)
    and received (is_from_me=0) row. The owner must get exactly ONE reply."""
    _seed_cursor_zero()
    owner = "the operator@example.com"
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "chat.db")
        # Same command recorded twice: received copy and sent copy.
        _make_db(db, [
            (owner, "Jarvis what time is it", 0, owner),
            (owner, "Jarvis what time is it", 1, owner),
        ])
        rt = FakeRuntime(["1:41 PM", "1:41 PM"])
        sent = []
        bridge = ic.IMessageCommandBridge(runtime_factory=lambda: rt, send_func=lambda to, t: sent.append((to, t)), db_path=db)
        os.environ["JARVIS_OWNER_IMESSAGE"] = owner
        bridge.process_once()
    if len(sent) != 1:
        raise SystemExit(f"self-thread duplicate produced {len(sent)} replies, expected 1: {sent}")
    if rt.inputs != ["what time is it"]:
        raise SystemExit(f"duplicate command handled more than once: {rt.inputs}")


def test_reply_loop_prevention() -> None:
    if ic._safe_reply_text("Jarvis at your service").casefold().startswith("jarvis"):
        raise SystemExit("reply starting with trigger not sanitized")
    if ic._safe_reply_text("Battery 80%") != "Battery 80%":
        raise SystemExit("normal reply altered")
    path_reply = ic._safe_reply_text(
        "Jarvis saved /\x55sers/example/private/report.md, /private/tmp/jarvis-proof.txt, "
        "/var/folders/zc/jarvis-proof.txt, and /tmp/jarvis-proof.txt"
    )
    if (
        path_reply.casefold().startswith("jarvis")
        or "/\x55sers/operator" in path_reply
        or "/private/tmp" in path_reply
        or "/var/folders" in path_reply
        or "/tmp/" in path_reply
        or "<local-path>" not in path_reply
    ):
        raise SystemExit(f"path-shaped reply should be redacted and trigger-safe: {path_reply}")


def test_runtime_exception_reply_is_non_leaky() -> None:
    class FailingRuntime:
        def handle(self, _text, request_token=None):
            raise RuntimeError("secret stack path /\x55sers/example/private and /var/folders/zc/jarvis")

    _seed_cursor_zero()
    os.environ["JARVIS_OWNER_IMESSAGE"] = "+15551234567"
    sent = []
    bridge = ic.IMessageCommandBridge(
        runtime_factory=lambda: FailingRuntime(),
        send_func=lambda to, text: sent.append((to, text)),
        fetch_func=lambda db_path, after_rowid, limit: [
            {"rowid": 1, "sender": "+15551234567", "chat_identifier": "+15551234567", "text": "Jarvis status", "is_from_me": 0}
        ],
        db_path="/unused/chat.db",
    )
    if bridge.process_once() != 1:
        raise SystemExit("runtime exception command should still be processed once")
    if len(sent) != 1:
        raise SystemExit(f"runtime exception should send one safe reply: {sent}")
    reply = sent[0][1]
    if "internal error" not in reply or "RuntimeError" not in reply:
        raise SystemExit(f"runtime exception reply should be friendly and diagnostic: {reply}")
    if "secret stack path" in reply or "/\x55sers/operator" in reply or "/var/folders" in reply or "iMessage control error" in reply:
        raise SystemExit(f"runtime exception reply leaked raw error text: {reply}")


def test_runtime_reply_redacts_local_paths_before_send() -> None:
    _seed_cursor_zero()
    os.environ["JARVIS_OWNER_IMESSAGE"] = "+15551234567"
    rt = FakeRuntime([
        "Saved report at /\x55sers/example/private/report.md, /private/tmp/jarvis-proof.txt, "
        "/var/folders/zc/jarvis-proof.txt, and /tmp/jarvis-proof.txt"
    ])
    sent = []
    bridge = ic.IMessageCommandBridge(
        runtime_factory=lambda: rt,
        send_func=lambda to, text: sent.append((to, text)),
        fetch_func=lambda db_path, after_rowid, limit: [
            {"rowid": 1, "sender": "+15551234567", "chat_identifier": "+15551234567", "text": "Jarvis where is the report", "is_from_me": 0}
        ],
        db_path="/unused/chat.db",
    )
    if bridge.process_once() != 1:
        raise SystemExit("path-shaped runtime reply command should still process")
    if len(sent) != 1:
        raise SystemExit(f"path-shaped runtime reply should send once: {sent}")
    reply = sent[0][1]
    if (
        "/\x55sers/operator" in reply
        or "/private/tmp" in reply
        or "/var/folders" in reply
        or "/tmp/" in reply
        or "<local-path>" not in reply
    ):
        raise SystemExit(f"iMessage runtime reply did not redact local paths before send: {reply}")


def test_state_file_reads_environment_at_call_time() -> None:
    old_file = ic.STATE_FILE
    old_env = os.environ.get("JARVIS_V3_IMESSAGE_STATE")
    temp = Path(tempfile.mkdtemp()) / "env-imessage.json"
    try:
        ic.STATE_FILE = None
        os.environ["JARVIS_V3_IMESSAGE_STATE"] = str(temp)
        ic._save_state({"last_rowid": 123})
        if not temp.exists():
            raise SystemExit("JARVIS_V3_IMESSAGE_STATE was not honored at save time")
        if ic._load_state()["last_rowid"] != 123:
            raise SystemExit(f"JARVIS_V3_IMESSAGE_STATE was not honored at load time: {ic._load_state()}")
    finally:
        ic.STATE_FILE = old_file
        if old_env is None:
            os.environ.pop("JARVIS_V3_IMESSAGE_STATE", None)
        else:
            os.environ["JARVIS_V3_IMESSAGE_STATE"] = old_env


def test_blank_state_file_env_uses_default() -> None:
    old_file = ic.STATE_FILE
    old_env = os.environ.get("JARVIS_V3_IMESSAGE_STATE")
    try:
        ic.STATE_FILE = None
        os.environ["JARVIS_V3_IMESSAGE_STATE"] = "   "
        expected = Path.home() / ".jarvis_v3" / "imessage_control.json"
        if ic._state_file() != expected:
            raise SystemExit(f"blank JARVIS_V3_IMESSAGE_STATE should use default: {ic._state_file()}")
    finally:
        ic.STATE_FILE = old_file
        if old_env is None:
            os.environ.pop("JARVIS_V3_IMESSAGE_STATE", None)
        else:
            os.environ["JARVIS_V3_IMESSAGE_STATE"] = old_env


def main() -> None:
    test_owner_trigger_runs_and_cursor_prevents_replay()
    test_non_owner_and_no_trigger_ignored()
    test_malformed_fetch_rowid_is_ignored()
    test_transport_row_tokens_are_exact_stable_and_content_free()
    test_malformed_transport_rowid_types_fail_closed()
    test_same_apple_id_self_thread()
    test_fresh_install_skips_history()
    test_unreadable_db_defers_seed()
    test_path_shaped_owner_is_rejected_before_fetch_runtime_or_send()
    test_self_thread_duplicate_command_replies_once()
    test_reply_loop_prevention()
    test_runtime_exception_reply_is_non_leaky()
    test_runtime_reply_redacts_local_paths_before_send()
    test_state_file_reads_environment_at_call_time()
    test_blank_state_file_env_uses_default()
    print("iMessage control smoke passed")


if __name__ == "__main__":
    main()
