"""Offline reconstruction proof for the two primary Jarvis LaunchAgents.

This harness performs no socket bind, network request, process control, job
execution, approval, or external side effect.
"""

from __future__ import annotations

import os
import plistlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.automations import telegram_control
from jarvis_v2.memory.store import TaskRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile
from jarvis_v2.ui import status_server


ROOT = Path(__file__).resolve().parents[2]
IS_PUBLIC_CANDIDATE = structural_public_candidate_profile(ROOT) is not None
AUTH_TOKEN = "daemon-reconstruction-smoke-token-0001"
OWNER_CHAT_ID = "555001"
TASK_MARKER = "daemon reconstruction durable task"


class _FakeBoundServer:
    def __init__(self, address, handler_cls) -> None:
        self.server_address = address
        self.handler_cls = handler_cls
        self.closed = False

    def server_close(self) -> None:
        self.closed = True


def _assert_primary_launch_contracts() -> None:
    expected = {
        "com.jarvis-v3.telegram.plist": (
            "com.jarvis-v3.telegram",
            "jarvis_v2.scripts.run_telegram_control",
        ),
        "com.jarvis-v3.dashboard.plist": (
            "com.jarvis-v3.dashboard",
            "jarvis_v2.scripts.run_status_server",
        ),
    }
    if IS_PUBLIC_CANDIDATE:
        if any((ROOT / filename).exists() for filename in expected):
            raise SystemExit("history-free public candidate included a private service plist")
        return
    for filename, (label, module) in expected.items():
        with (ROOT / filename).open("rb") as handle:
            contract = plistlib.load(handle)
        if contract.get("Label") != label:
            raise SystemExit(f"{filename} label drifted from the primary daemon contract")
        if contract.get("ProgramArguments") != ["/opt/homebrew/bin/python3", "-m", module]:
            raise SystemExit(f"{filename} module launch contract drifted")
        if contract.get("RunAtLoad") is not True or contract.get("KeepAlive") is not True:
            raise SystemExit(f"{filename} no longer declares reboot/restart recovery")
        if contract.get("WorkingDirectory") != str(ROOT):
            raise SystemExit(f"{filename} working directory drifted from the checked-in project")


def _make_dashboard_handler(runtime):
    server = status_server.make_status_server(
        runtime,
        "127.0.0.1",
        8765,
        auth_token=AUTH_TOKEN,
    )
    return server, server.handler_cls.__new__(server.handler_cls)


def _assert_reconstruction_contracts() -> None:
    previous_state_file = telegram_control.STATE_FILE
    previous_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    previous_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        with TemporaryDirectory(prefix="jarvis-daemon-reconstruction-") as temp:
            root = Path(temp)
            telegram_control.STATE_FILE = root / "telegram-offset.json"
            os.environ["TELEGRAM_BOT_TOKEN"] = "123456:test-token"
            os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER_CHAT_ID

            runtime_before = make_temp_runtime(root / "runtime")
            task_id = runtime_before.store.add_task(TaskRecord(body=TASK_MARKER))
            telegram_control._save_offset(41)

            server_before, handler_before = _make_dashboard_handler(runtime_before)
            preview_token = handler_before._register_approval_preview_token(task_id)
            if preview_token not in type(handler_before).approval_preview_tokens:
                raise SystemExit("first dashboard process did not retain its preview authority")

            first_offsets: list[int | None] = []
            first_replies: list[str] = []

            def first_fetch(_token: str, offset: int | None, _timeout: int) -> list[dict]:
                first_offsets.append(offset)
                return [
                    {
                        "update_id": 41,
                        "message": {
                            "chat": {"id": int(OWNER_CHAT_ID)},
                            "text": "list tasks",
                        },
                    }
                ]

            bridge_before = telegram_control.TelegramCommandBridge(
                runtime_factory=lambda: runtime_before,
                fetch_func=first_fetch,
                send_func=lambda _chat, text, _markup=None: first_replies.append(text) or {"ok": True},
                chat_action_func=lambda _chat, _action: {"ok": True},
            )
            bridge_before._deliver_due_reminders = lambda: None
            if bridge_before.process_once(poll_timeout=0) != 1:
                raise SystemExit("first Telegram daemon process did not consume its owner update")
            if first_offsets != [41] or telegram_control._load_offset() != 42:
                raise SystemExit("first Telegram daemon process did not durably advance its offset")
            if not first_replies or TASK_MARKER not in first_replies[-1]:
                raise SystemExit("first Telegram daemon process could not read the durable task")

            runtime_after = make_temp_runtime(root / "runtime")
            if runtime_after.session_id == runtime_before.session_id:
                raise SystemExit("runtime reconstruction reused process-ephemeral session authority")
            durable_tasks = runtime_after.store.list_tasks(status="open", limit=25)
            if not any(row["id"] == task_id and row["body"] == TASK_MARKER for row in durable_tasks):
                raise SystemExit("runtime reconstruction lost durable SQLite task state")

            server_after, handler_after = _make_dashboard_handler(runtime_after)
            if (
                type(handler_before).approval_preview_tokens
                is type(handler_after).approval_preview_tokens
            ):
                raise SystemExit("dashboard reconstruction reused the old preview-token store")
            token_ok, token_error = handler_after._consume_approval_preview_token(
                task_id,
                preview_token,
            )
            if token_ok or "missing, expired, or already used" not in token_error:
                raise SystemExit("dashboard reconstruction accepted old ephemeral preview authority")
            if not any(row["id"] == task_id for row in runtime_after.store.list_tasks(limit=25)):
                raise SystemExit("dashboard reconstruction invalidated durable state with authority")

            second_offsets: list[int | None] = []
            second_replies: list[str] = []

            def second_fetch(_token: str, offset: int | None, _timeout: int) -> list[dict]:
                second_offsets.append(offset)
                return [
                    {
                        "update_id": 42,
                        "message": {
                            "chat": {"id": int(OWNER_CHAT_ID)},
                            "text": "list tasks",
                        },
                    }
                ]

            bridge_after = telegram_control.TelegramCommandBridge(
                runtime_factory=lambda: runtime_after,
                fetch_func=second_fetch,
                send_func=lambda _chat, text, _markup=None: second_replies.append(text) or {"ok": True},
                chat_action_func=lambda _chat, _action: {"ok": True},
            )
            bridge_after._deliver_due_reminders = lambda: None
            if bridge_after.process_once(poll_timeout=0) != 1:
                raise SystemExit("reconstructed Telegram daemon did not consume the next owner update")
            if second_offsets != [42] or telegram_control._load_offset() != 43:
                raise SystemExit("reconstructed Telegram daemon replayed or skipped its durable offset")
            if not second_replies or TASK_MARKER not in second_replies[-1]:
                raise SystemExit("reconstructed Telegram daemon could not read durable runtime state")

            for server in (server_before, server_after):
                server.server_close()
                if not server.closed:
                    raise SystemExit("fake dashboard server did not close cleanly")
    finally:
        telegram_control.STATE_FILE = previous_state_file
        if previous_token is None:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        else:
            os.environ["TELEGRAM_BOT_TOKEN"] = previous_token
        if previous_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = previous_owner


def main() -> None:
    _assert_primary_launch_contracts()
    with (
        patch.object(status_server, "ThreadingHTTPServer", _FakeBoundServer),
        patch.object(status_server, "_start_dashboard_voice_warmup"),
    ):
        _assert_reconstruction_contracts()
    print("Primary daemon reconstruction smoke tests passed.")


if __name__ == "__main__":
    main()
