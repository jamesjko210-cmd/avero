"""Offline adversarial smoke for identity-bound Telegram offset custody."""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_v2.automations import telegram_control as control
from jarvis_v2.automations import telegram_offset_state as state


TOKEN = "12345:" + "A" * 32
OTHER_TOKEN = "67890:" + "B" * 32
OWNER = "4827001"
OTHER_OWNER = "4827002"
IDENTITY = state._identity(TOKEN, OWNER)


def _path(root: Path) -> Path:
    return root / "telegram-offset.json"


def _expect(reason: str, callback) -> None:
    try:
        callback()
    except state.TelegramOffsetStateError as exc:
        if exc.reason != reason or str(exc) != reason:
            raise SystemExit(
                f"offset refusal reason drifted: expected {reason}, received {exc.reason}"
            )
    else:
        raise SystemExit(f"offset custody accepted an invalid case: {reason}")


def _load(path: Path, *, token: str = TOKEN, owner: str = OWNER) -> state.OffsetSnapshot:
    return state.load_offset(path, bot_token=token, owner_id=owner)


def _advance(
    path: Path,
    offset: int,
    expected: state.OffsetSnapshot,
    *,
    token: str = TOKEN,
    owner: str = OWNER,
) -> state.OffsetSnapshot:
    return state.advance_offset(
        path,
        bot_token=token,
        owner_id=owner,
        new_offset=offset,
        expected=expected,
    )


def _write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.write_bytes(payload)
    path.chmod(mode)


def test_fresh_success_is_canonical_private_bound_and_durable() -> None:
    with TemporaryDirectory(prefix="jarvis-telegram-offset-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        fresh = _load(path)
        if not fresh.fresh or fresh.offset is not None or fresh.revision != 0:
            raise SystemExit("truly absent offset state was not classified as fresh")
        saved = _advance(path, 41, fresh)
        if saved.offset != 41 or saved.revision != 1 or saved.fresh:
            raise SystemExit("first offset publication lost monotonic state")
        raw = path.read_bytes()
        value = json.loads(raw)
        if raw != state._canonical(value) or set(value) != {
            "identity_sha256", "offset", "revision", "version"
        }:
            raise SystemExit("offset state was not exact canonical schema")
        if TOKEN.encode() in raw or OWNER.encode() in raw:
            raise SystemExit("offset state exposed Telegram credentials")
        if stat.S_IMODE(path.stat().st_mode) != 0o600 or path.stat().st_nlink != 1:
            raise SystemExit("offset state lost owner-only single-link custody")
        lock = root / f".{path.name}.lock"
        if stat.S_IMODE(lock.stat().st_mode) != 0o600 or lock.stat().st_nlink != 1:
            raise SystemExit("offset lock lost owner-only single-link custody")
        if any(item.name.startswith(f".{path.name}.tmp.") for item in root.iterdir()):
            raise SystemExit("successful publication left a temporary file")
        if _load(path) != saved:
            raise SystemExit("published offset did not survive a clean restart")


def test_malformed_unversioned_identity_and_bounds_fail_closed() -> None:
    cases = (
        b"not-json\n",
        b'{"offset":7}\n',
        b'{"identity_sha256":"' + IDENTITY.encode("ascii") + b'","offset":true,"revision":1,"version":1}\n',
        b'{"identity_sha256":"' + IDENTITY.encode("ascii") + b'","offset":1,"revision":1,"version":2}\n',
        b'{"identity_sha256":"' + IDENTITY.encode("ascii") + b'","offset":1,"offset":2,"revision":1,"version":1}\n',
    )
    for payload in cases:
        with TemporaryDirectory(prefix="jarvis-telegram-offset-malformed-", dir="/private/tmp") as temp:
            path = _path(Path(temp))
            _write(path, payload)
            _expect("state_malformed", lambda path=path: _load(path))
    with TemporaryDirectory(prefix="jarvis-telegram-offset-bound-", dir="/private/tmp") as temp:
        path = _path(Path(temp))
        snapshot = _load(path)
        _advance(path, 3, snapshot)
        _expect("state_identity_mismatch", lambda: _load(path, token=OTHER_TOKEN))
        _expect("state_identity_mismatch", lambda: _load(path, owner=OTHER_OWNER))
        _expect("state_identity_invalid", lambda: _load(path, token="\udcff"))
        _expect(
            "state_path_invalid",
            lambda: state.load_offset(
                "/private/tmp/invalid\x00state",
                bot_token=TOKEN,
                owner_id=OWNER,
            ),
        )
        _expect("state_malformed", lambda: _advance(path, -1, _load(path)))
        _expect("state_malformed", lambda: _advance(path, state.MAX_OFFSET + 1, _load(path)))


def test_symlink_hardlink_mode_parent_and_oversize_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-telegram-offset-custody-", dir="/private/tmp") as temp:
        root = Path(temp)
        target = root / "target"
        _write(target, b"{}\n")
        alias = _path(root)
        alias.symlink_to(target)
        _expect("state_unsafe", lambda: _load(alias))

    with TemporaryDirectory(prefix="jarvis-telegram-offset-hardlink-", dir="/private/tmp") as temp:
        root = Path(temp)
        target = root / "target"
        _write(target, b"{}\n")
        alias = _path(root)
        os.link(target, alias)
        _expect("state_unsafe", lambda: _load(alias))

    with TemporaryDirectory(prefix="jarvis-telegram-offset-mode-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        _write(path, b"{}\n", 0o644)
        _expect("state_unsafe", lambda: _load(path))
        path.chmod(0o000)
        _expect("state_unavailable", lambda: _load(path))

    with TemporaryDirectory(prefix="jarvis-telegram-offset-parent-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        root.chmod(0o755)
        try:
            _expect("state_parent_invalid", lambda: _load(path))
        finally:
            root.chmod(0o700)

    with TemporaryDirectory(prefix="jarvis-telegram-offset-unwritable-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        root.chmod(0o500)
        try:
            _expect("state_parent_invalid", lambda: _load(path))
        finally:
            root.chmod(0o700)

    with TemporaryDirectory(prefix="jarvis-telegram-offset-large-", dir="/private/tmp") as temp:
        path = _path(Path(temp))
        _write(path, b"x" * (state.MAX_STATE_BYTES + 1))
        _expect("state_unsafe", lambda: _load(path))


def test_random_exclusive_temp_collision_and_failures_leave_no_regression() -> None:
    with TemporaryDirectory(prefix="jarvis-telegram-offset-temp-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        fresh = _load(path)
        collision = root / f".{path.name}.tmp.collision"
        _write(collision, b"sentinel")
        with patch.object(state.secrets, "token_hex", side_effect=["collision", "unique"]):
            saved = _advance(path, 5, fresh)
        if saved.offset != 5 or collision.read_bytes() != b"sentinel":
            raise SystemExit("exclusive temp collision overwrote an existing entry")

    with TemporaryDirectory(prefix="jarvis-telegram-offset-collide-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        fresh = _load(path)
        collision = root / f".{path.name}.tmp.collision"
        _write(collision, b"sentinel")
        with patch.object(state.secrets, "token_hex", return_value="collision"):
            _expect("state_temp_collision", lambda: _advance(path, 5, fresh))
        if path.exists() or collision.read_bytes() != b"sentinel":
            raise SystemExit("exhausted temp collisions changed offset state")

    with TemporaryDirectory(prefix="jarvis-telegram-offset-rename-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        fresh = _load(path)
        real_rename = state._renameatx_np

        def fail_publish(source_fd, source_name, destination_fd, destination_name, flags):
            if destination_name == path.name:
                raise OSError("synthetic")
            return real_rename(source_fd, source_name, destination_fd, destination_name, flags)

        with patch.object(state, "_renameatx_np", side_effect=fail_publish):
            _expect("state_write_failed", lambda: _advance(path, 5, fresh))
        if path.exists() or any(".tmp." in item.name for item in root.iterdir()):
            raise SystemExit("rename failure published or leaked temporary state")

    with TemporaryDirectory(prefix="jarvis-telegram-offset-fsync-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        fresh = _load(path)
        with patch.object(state.os, "fsync", side_effect=OSError("synthetic")):
            _expect("state_write_failed", lambda: _advance(path, 5, fresh))
        if path.exists() or any(".tmp." in item.name for item in root.iterdir()):
            raise SystemExit("file fsync failure published or leaked temporary state")


def test_directory_fsync_failure_is_uncertain_and_replay_loads_published_state() -> None:
    with TemporaryDirectory(prefix="jarvis-telegram-offset-dir-fsync-", dir="/private/tmp") as temp:
        path = _path(Path(temp))
        fresh = _load(path)
        real_fsync = state.os.fsync
        calls = 0

        def fail_second(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic")
            real_fsync(descriptor)

        with patch.object(state.os, "fsync", side_effect=fail_second):
            _expect("state_persistence_uncertain", lambda: _advance(path, 9, fresh))
        if _load(path).offset != 9:
            raise SystemExit("uncertain directory fsync lost the atomically published replay state")


def test_cas_concurrency_and_regression_never_overwrite_newer_offset() -> None:
    with TemporaryDirectory(prefix="jarvis-telegram-offset-cas-", dir="/private/tmp") as temp:
        path = _path(Path(temp))
        initial = _load(path)
        current = _advance(path, 10, initial)
        _expect("state_conflict", lambda: _advance(path, 20, initial))
        _expect("state_regression", lambda: _advance(path, 9, current))
        if _load(path).offset != 10:
            raise SystemExit("stale CAS or regression changed the stored offset")

        expected = _load(path)
        outcomes: list[str] = []

        def writer(value: int) -> None:
            try:
                _advance(path, value, expected)
                outcomes.append("saved")
            except state.TelegramOffsetStateError as exc:
                outcomes.append(exc.reason)

        threads = [threading.Thread(target=writer, args=(value,)) for value in (11, 12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        if sorted(outcomes) != ["saved", "state_conflict"] or _load(path).offset not in {11, 12}:
            raise SystemExit("concurrent CAS writers did not produce exactly one winner")


def test_injected_fresh_existing_cleanup_and_lock_replacement_races_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-telegram-offset-fresh-race-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        fresh = _load(path)
        attacker = b"attacker-sentinel"
        real_rename = state._renameatx_np

        def inject_fresh(source_fd, source_name, destination_fd, destination_name, flags):
            if destination_name == path.name and flags == state.RENAME_EXCL:
                _write(path, attacker)
            return real_rename(source_fd, source_name, destination_fd, destination_name, flags)

        with patch.object(state, "_renameatx_np", side_effect=inject_fresh):
            _expect("state_conflict", lambda: _advance(path, 5, fresh))
        if path.read_bytes() != attacker:
            raise SystemExit("fresh exclusive publication overwrote a raced state entry")

    with TemporaryDirectory(prefix="jarvis-telegram-offset-swap-race-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        current = _advance(path, 10, _load(path))
        attacker_path = root / "attacker"
        _write(attacker_path, path.read_bytes())
        attacker_inode = attacker_path.stat().st_ino
        real_rename = state._renameatx_np
        injected = False

        def inject_swap(source_fd, source_name, destination_fd, destination_name, flags):
            nonlocal injected
            if not injected and destination_name == path.name and flags == state.RENAME_SWAP:
                injected = True
                os.replace(attacker_path, path)
            return real_rename(source_fd, source_name, destination_fd, destination_name, flags)

        with patch.object(state, "_renameatx_np", side_effect=inject_swap):
            _expect("state_persistence_uncertain", lambda: _advance(path, 11, current))
        if path.stat().st_ino != attacker_inode or _load(path).offset != 10:
            raise SystemExit("swap CAS did not restore the exact raced current state")

    with TemporaryDirectory(prefix="jarvis-telegram-offset-cleanup-race-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        fresh = _load(path)
        real_rename = state._renameatx_np
        cleanup_injected = False

        def inject_cleanup(source_fd, source_name, destination_fd, destination_name, flags):
            nonlocal cleanup_injected
            if destination_name == path.name:
                raise OSError("synthetic publication failure")
            if ".quarantine." in destination_name and not cleanup_injected:
                cleanup_injected = True
                replacement = root / "replacement"
                _write(replacement, b"replacement-sentinel")
                os.replace(replacement, root / source_name)
            return real_rename(source_fd, source_name, destination_fd, destination_name, flags)

        with patch.object(state, "_renameatx_np", side_effect=inject_cleanup):
            _expect("state_persistence_uncertain", lambda: _advance(path, 5, fresh))
        if path.exists():
            raise SystemExit("cleanup race published a state entry")
        quarantines = [item for item in root.iterdir() if ".quarantine." in item.name]
        if len(quarantines) != 1 or quarantines[0].read_bytes() != b"replacement-sentinel":
            raise SystemExit("cleanup race did not isolate the untrusted replacement")

    with TemporaryDirectory(prefix="jarvis-telegram-offset-lock-race-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        _load(path)
        lock = root / f".{path.name}.lock"
        real_flock = state.fcntl.flock

        def replace_after_flock(descriptor: int, operation: int) -> None:
            real_flock(descriptor, operation)
            replacement = root / "replacement-lock"
            _write(replacement, b"")
            os.replace(replacement, lock)

        with patch.object(state.fcntl, "flock", side_effect=replace_after_flock):
            _expect("state_lock_invalid", lambda: _load(path))


def test_parent_path_replacement_before_and_during_publication_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-telegram-offset-parent-before-", dir="/private/tmp") as temp:
        outer = Path(temp)
        parent = outer / "state-parent"
        parent.mkdir(mode=0o700)
        path = _path(parent)
        fresh = _load(path)
        detached = outer / "detached-parent"
        real_validate = state._validate_parent_identity
        injected = False

        def replace_before_publish(state_path: Path, parent_fd: int) -> None:
            nonlocal injected
            if not injected:
                injected = True
                parent.rename(detached)
                parent.mkdir(mode=0o700)
            real_validate(state_path, parent_fd)

        with patch.object(state, "_validate_parent_identity", side_effect=replace_before_publish):
            _expect("state_parent_invalid", lambda: _advance(path, 5, fresh))
        if path.exists() or (detached / path.name).exists():
            raise SystemExit("parent replacement before publication created offset state")

    with TemporaryDirectory(prefix="jarvis-telegram-offset-parent-during-", dir="/private/tmp") as temp:
        outer = Path(temp)
        parent = outer / "state-parent"
        parent.mkdir(mode=0o700)
        path = _path(parent)
        fresh = _load(path)
        detached = outer / "detached-parent"
        real_rename = state._renameatx_np
        injected = False

        def replace_during_publish(source_fd, source_name, destination_fd, destination_name, flags):
            nonlocal injected
            if not injected and destination_name == path.name and flags == state.RENAME_EXCL:
                injected = True
                parent.rename(detached)
                parent.mkdir(mode=0o700)
            return real_rename(source_fd, source_name, destination_fd, destination_name, flags)

        with patch.object(state, "_renameatx_np", side_effect=replace_during_publish):
            _expect("state_persistence_uncertain", lambda: _advance(path, 6, fresh))
        if path.exists() or not (detached / path.name).exists():
            raise SystemExit("parent replacement during publication reported false path success")


def test_noop_advance_revalidates_parent_lock_and_closing_state() -> None:
    with TemporaryDirectory(prefix="jarvis-telegram-offset-noop-parent-", dir="/private/tmp") as temp:
        outer = Path(temp)
        parent = outer / "state-parent"
        parent.mkdir(mode=0o700)
        path = _path(parent)
        current = _advance(path, 42, _load(path))
        detached = outer / "detached-parent"
        real_read = state._read_state
        reads = 0

        def replace_parent_after_read(parent_fd: int, name: str, identity: str):
            nonlocal reads
            result = real_read(parent_fd, name, identity)
            reads += 1
            if reads == 1:
                parent.rename(detached)
                parent.mkdir(mode=0o700)
            return result

        with patch.object(state, "_read_state", side_effect=replace_parent_after_read):
            _expect("state_parent_invalid", lambda: _advance(path, 42, current))
        if path.exists() or _load(path).offset is not None or _load(detached / path.name).offset != 42:
            raise SystemExit("no-op advance reported success after parent replacement")

    with TemporaryDirectory(prefix="jarvis-telegram-offset-noop-lock-", dir="/private/tmp") as temp:
        root = Path(temp)
        path = _path(root)
        current = _advance(path, 42, _load(path))
        lock = root / f".{path.name}.lock"
        real_read = state._read_state
        reads = 0

        def replace_lock_after_read(parent_fd: int, name: str, identity: str):
            nonlocal reads
            result = real_read(parent_fd, name, identity)
            reads += 1
            if reads == 1:
                replacement = root / "replacement-lock"
                _write(replacement, b"")
                os.replace(replacement, lock)
            return result

        with patch.object(state, "_read_state", side_effect=replace_lock_after_read):
            _expect("state_lock_invalid", lambda: _advance(path, 42, current))
        if _load(path).offset != 42:
            raise SystemExit("no-op lock replacement changed the durable offset")


class _Runtime:
    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.tokens: list[str] = []

    def handle(self, text: str, request_token: str | None = None) -> SimpleNamespace:
        self.inputs.append(text)
        self.tokens.append(request_token or "")
        return SimpleNamespace(response="ok", tool_results=[])


def _update(update_id: int, text: str) -> dict:
    return {"update_id": update_id, "message": {"chat": {"id": int(OWNER)}, "text": text}}


def test_bridge_stops_before_poll_on_bad_state_and_preserves_fresh_drain() -> None:
    old_state = control.STATE_FILE
    old_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ["TELEGRAM_BOT_TOKEN"] = TOKEN
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER
        with TemporaryDirectory(prefix="jarvis-telegram-offset-bridge-", dir="/private/tmp") as temp:
            root = Path(temp)
            path = _path(root)
            control.STATE_FILE = path
            _write(path, b'{"offset":1}\n')
            fetched: list[bool] = []
            bridge = control.TelegramCommandBridge(
                runtime_factory=_Runtime,
                fetch_func=lambda *_args: fetched.append(True) or [],
                send_func=lambda *_args: {"ok": True},
                chat_action_func=None,
            )
            _expect("state_malformed", lambda: bridge.process_once(poll_timeout=0))
            if fetched:
                raise SystemExit("malformed state reached Telegram polling")

        with TemporaryDirectory(prefix="jarvis-telegram-offset-fresh-", dir="/private/tmp") as temp:
            path = _path(Path(temp))
            control.STATE_FILE = path
            runtime = _Runtime()
            bridge = control.TelegramCommandBridge(
                runtime_factory=lambda: runtime,
                fetch_func=lambda _token, offset, _timeout: [_update(40, "stale")] if offset is None else [],
                send_func=lambda *_args: {"ok": True},
                chat_action_func=None,
            )
            if bridge.process_once(poll_timeout=0) != 0 or runtime.inputs:
                raise SystemExit("truly fresh state executed backlog")
            if control._load_offset(TOKEN, OWNER) != 41:
                raise SystemExit("fresh backlog drain did not durably advance the bound offset")
    finally:
        control.STATE_FILE = old_state
        if old_token is None:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        else:
            os.environ["TELEGRAM_BOT_TOKEN"] = old_token
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_bridge_persistence_failure_surfaces_and_replay_token_is_stable() -> None:
    old_state = control.STATE_FILE
    old_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ["TELEGRAM_BOT_TOKEN"] = TOKEN
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER
        with TemporaryDirectory(prefix="jarvis-telegram-offset-replay-", dir="/private/tmp") as temp:
            path = _path(Path(temp))
            control.STATE_FILE = path
            control._save_offset(0, TOKEN, OWNER)
            runtime = _Runtime()
            bridge = control.TelegramCommandBridge(
                runtime_factory=lambda: runtime,
                fetch_func=lambda *_args: [_update(77, "command")],
                send_func=lambda *_args: {"ok": True},
                chat_action_func=None,
            )
            real_rename = state._renameatx_np

            def fail_publish(source_fd, source_name, destination_fd, destination_name, flags):
                if destination_name == path.name:
                    raise OSError("synthetic")
                return real_rename(source_fd, source_name, destination_fd, destination_name, flags)

            with patch.object(state, "_renameatx_np", side_effect=fail_publish):
                _expect("state_write_failed", lambda: bridge.process_once(poll_timeout=0))
            if control._load_offset(TOKEN, OWNER) != 0:
                raise SystemExit("failed offset publication changed replay position")
            if bridge.process_once(poll_timeout=0) != 1:
                raise SystemExit("restart replay did not complete after persistence recovered")
            if runtime.tokens != ["telegram-update:v1:77", "telegram-update:v1:77"]:
                raise SystemExit("crash replay lost stable content-free request identity")
            if control._load_offset(TOKEN, OWNER) != 78:
                raise SystemExit("successful replay did not persist the new offset")
    finally:
        control.STATE_FILE = old_state
        if old_token is None:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        else:
            os.environ["TELEGRAM_BOT_TOKEN"] = old_token
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def test_bridge_carries_exact_prepoll_snapshot_across_execution() -> None:
    old_state = control.STATE_FILE
    old_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    try:
        os.environ["TELEGRAM_BOT_TOKEN"] = TOKEN
        os.environ["JARVIS_OWNER_TELEGRAM"] = OWNER
        for mutation in ("delete", "replace", "advance"):
            with TemporaryDirectory(
                prefix=f"jarvis-telegram-offset-{mutation}-",
                dir="/private/tmp",
            ) as temp:
                root = Path(temp)
                path = _path(root)
                control.STATE_FILE = path
                control._save_offset(0, TOKEN, OWNER)
                original = path.read_bytes()
                runtime = _Runtime()
                fetch_calls = 0
                runtime_mutated = False

                def mutate_state() -> None:
                    if mutation == "delete":
                        path.unlink()
                    elif mutation == "replace":
                        replacement = root / "replacement"
                        _write(replacement, original)
                        os.replace(replacement, path)
                    else:
                        _advance(path, 1, _load(path))

                def fetch(_token, _offset, _timeout):
                    nonlocal fetch_calls
                    if fetch_calls == 0 and mutation == "delete":
                        mutate_state()
                    fetch_calls += 1
                    return [_update(77, "command")]

                real_handle = runtime.handle

                def handle(text: str, request_token: str | None = None):
                    nonlocal runtime_mutated
                    if not runtime_mutated and mutation != "delete":
                        runtime_mutated = True
                        mutate_state()
                    return real_handle(text, request_token=request_token)

                runtime.handle = handle

                bridge = control.TelegramCommandBridge(
                    runtime_factory=lambda: runtime,
                    fetch_func=fetch,
                    send_func=lambda *_args: {"ok": True},
                    chat_action_func=None,
                )
                bridge._deliver_due_reminders = lambda: None
                _expect("state_conflict", lambda: bridge.process_once(poll_timeout=0))
                if runtime.inputs != ["command"]:
                    raise SystemExit(f"{mutation} race did not reach the deterministic execution point")

                if mutation == "delete":
                    _write(path, original)
                if bridge.process_once(poll_timeout=0) != 1:
                    raise SystemExit(f"{mutation} race did not replay after custody recovery")
                if runtime.inputs != ["command", "command"] or runtime.tokens != [
                    "telegram-update:v1:77",
                    "telegram-update:v1:77",
                ]:
                    raise SystemExit(f"{mutation} race lost stable replay identity")
                if control._load_offset(TOKEN, OWNER) != 78:
                    raise SystemExit(f"{mutation} race falsely reported offset persistence")
    finally:
        control.STATE_FILE = old_state
        if old_token is None:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        else:
            os.environ["TELEGRAM_BOT_TOKEN"] = old_token
        if old_owner is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner


def main() -> int:
    test_fresh_success_is_canonical_private_bound_and_durable()
    test_malformed_unversioned_identity_and_bounds_fail_closed()
    test_symlink_hardlink_mode_parent_and_oversize_fail_closed()
    test_random_exclusive_temp_collision_and_failures_leave_no_regression()
    test_directory_fsync_failure_is_uncertain_and_replay_loads_published_state()
    test_cas_concurrency_and_regression_never_overwrite_newer_offset()
    test_injected_fresh_existing_cleanup_and_lock_replacement_races_fail_closed()
    test_parent_path_replacement_before_and_during_publication_fails_closed()
    test_noop_advance_revalidates_parent_lock_and_closing_state()
    test_bridge_stops_before_poll_on_bad_state_and_preserves_fresh_drain()
    test_bridge_persistence_failure_surfaces_and_replay_token_is_stable()
    test_bridge_carries_exact_prepoll_snapshot_across_execution()
    print("telegram offset state smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
