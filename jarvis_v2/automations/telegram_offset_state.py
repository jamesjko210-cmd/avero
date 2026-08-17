"""Fail-closed, identity-bound Telegram update-offset persistence."""

from __future__ import annotations

import fcntl
import ctypes
import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn


STATE_VERSION = 1
MAX_OFFSET = (1 << 63) - 1
MAX_STATE_BYTES = 4096
MAX_IDENTITY_BYTES = 4096
TEMP_ATTEMPTS = 8
IDENTITY_RE = re.compile(r"[0-9a-f]{64}")
RENAME_SWAP = 0x00000002
RENAME_EXCL = 0x00000004

_REASONS = frozenset(
    {
        "state_path_invalid",
        "state_parent_invalid",
        "state_lock_invalid",
        "state_unavailable",
        "state_unsafe",
        "state_malformed",
        "state_identity_invalid",
        "state_identity_mismatch",
        "state_conflict",
        "state_regression",
        "state_temp_collision",
        "state_write_failed",
        "state_persistence_uncertain",
        "state_platform_unsupported",
    }
)


class TelegramOffsetStateError(RuntimeError):
    """A path- and credential-free offset-state refusal."""

    def __init__(self, reason: str) -> None:
        self.reason = reason if reason in _REASONS else "state_unavailable"
        super().__init__(self.reason)


def _fail(reason: str) -> NoReturn:
    raise TelegramOffsetStateError(reason)


@dataclass(frozen=True)
class OffsetSnapshot:
    offset: int | None
    revision: int
    identity_sha256: str
    file_token: tuple[int, ...] | None

    @property
    def fresh(self) -> bool:
        return self.offset is None


def _token(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _cleanup_token(value: os.stat_result) -> tuple[int, ...]:
    return _token(value)[:-1]


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        _fail("state_parent_invalid")
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )


def _file_flags(*, writable: bool = False, create: bool = False, exclusive: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        _fail("state_unsafe")
    flags = (os.O_RDWR if writable else os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0) | nofollow
    if not writable:
        flags |= getattr(os, "O_NONBLOCK", 0)
    if create:
        flags |= os.O_CREAT
    if exclusive:
        flags |= os.O_EXCL
    return flags


def _identity(token: str, owner: str) -> str:
    if type(token) is not str or type(owner) is not str or not token or not owner:
        _fail("state_identity_invalid")
    try:
        token_bytes = token.encode("utf-8", errors="strict")
        owner_bytes = owner.encode("utf-8", errors="strict")
    except UnicodeError:
        _fail("state_identity_invalid")
    if (
        len(token_bytes) > MAX_IDENTITY_BYTES
        or len(owner_bytes) > MAX_IDENTITY_BYTES
        or b"\x00" in token_bytes
        or b"\x00" in owner_bytes
    ):
        _fail("state_identity_invalid")
    digest = hashlib.sha256()
    digest.update(b"jarvis-telegram-offset-identity-v1\0")
    digest.update(len(token_bytes).to_bytes(4, "big"))
    digest.update(token_bytes)
    digest.update(len(owner_bytes).to_bytes(4, "big"))
    digest.update(owner_bytes)
    return digest.hexdigest()


def _validate_path(path: Path | str) -> Path:
    try:
        value = Path(path)
        encoded_value = os.fsencode(value)
        encoded_name = os.fsencode(value.name)
    except (TypeError, ValueError, UnicodeError, OSError):
        _fail("state_path_invalid")
    if value.is_absolute() and len(value.parts) > 1 and value.parts[1] in {"etc", "tmp", "var"}:
        alias = Path("/") / value.parts[1]
        expected = Path("/private") / value.parts[1]
        try:
            if alias.is_symlink() and Path(os.path.realpath(alias)) == expected:
                value = expected.joinpath(*value.parts[2:])
        except OSError:
            _fail("state_path_invalid")
    if (
        not value.is_absolute()
        or not value.name
        or value.name in {".", ".."}
        or len(encoded_value) > 2048
        or len(encoded_name) > 180
        or b"\x00" in encoded_value
    ):
        _fail("state_path_invalid")
    return value


def _open_parent(path: Path) -> int:
    descriptor: int | None = None
    try:
        descriptor = os.open("/", _directory_flags())
        for component in path.parent.parts[1:]:
            if component in {"", ".", ".."}:
                _fail("state_parent_invalid")
            next_descriptor = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        info = os.fstat(descriptor)
        named = os.stat(path.parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _token(info) != _token(named)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            _fail("state_parent_invalid")
        return descriptor
    except TelegramOffsetStateError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except (OSError, ValueError):
        if descriptor is not None:
            os.close(descriptor)
        _fail("state_parent_invalid")


def _validate_parent_identity(path: Path, parent_fd: int) -> None:
    """Re-resolve the complete configured parent and match the held directory."""

    reopened: int | None = None
    try:
        reopened = _open_parent(path)
        if _token(os.fstat(parent_fd)) != _token(os.fstat(reopened)):
            _fail("state_parent_invalid")
    except TelegramOffsetStateError:
        raise
    except OSError:
        _fail("state_parent_invalid")
    finally:
        if reopened is not None:
            os.close(reopened)


def _read_bounded(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    remaining = MAX_STATE_BYTES + 1
    try:
        while remaining:
            chunk = os.read(descriptor, min(4096, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except OSError:
        _fail("state_unavailable")
    value = b"".join(chunks)
    if len(value) > MAX_STATE_BYTES:
        _fail("state_malformed")
    return value


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _read_state(parent_fd: int, name: str, identity_sha256: str) -> OffsetSnapshot:
    descriptor: int | None = None
    try:
        try:
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return OffsetSnapshot(None, 0, identity_sha256, None)
        if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode):
            _fail("state_unsafe")
        descriptor = os.open(name, _file_flags(), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _token(named) != _token(opened)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size < 1
            or opened.st_size > MAX_STATE_BYTES
        ):
            _fail("state_unsafe")
        raw = _read_bounded(descriptor)
        after = os.fstat(descriptor)
        named_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _token(opened) != _token(after) or _token(after) != _token(named_after):
            _fail("state_conflict")
    except TelegramOffsetStateError:
        raise
    except (OSError, ValueError):
        _fail("state_unavailable")
    finally:
        if descriptor is not None:
            os.close(descriptor)

    duplicate = False

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except Exception:
        _fail("state_malformed")
    expected_keys = {"identity_sha256", "offset", "revision", "version"}
    if duplicate or type(value) is not dict or set(value) != expected_keys or raw != _canonical(value):
        _fail("state_malformed")
    stored_identity = value.get("identity_sha256")
    offset = value.get("offset")
    revision = value.get("revision")
    if value.get("version") != STATE_VERSION:
        _fail("state_malformed")
    if type(stored_identity) is not str or IDENTITY_RE.fullmatch(stored_identity) is None:
        _fail("state_malformed")
    if not secrets.compare_digest(stored_identity, identity_sha256):
        _fail("state_identity_mismatch")
    if type(offset) is not int or offset < 0 or offset > MAX_OFFSET:
        _fail("state_malformed")
    if type(revision) is not int or revision < 1 or revision > MAX_OFFSET:
        _fail("state_malformed")
    return OffsetSnapshot(offset, revision, stored_identity, _token(after))


def _open_lock(parent_fd: int, name: str) -> int:
    lock_name = f".{name}.lock"
    descriptor: int | None = None
    created = False
    try:
        try:
            descriptor = os.open(
                lock_name,
                _file_flags(writable=True, create=True, exclusive=True),
                0o600,
                dir_fd=parent_fd,
            )
            created = True
            os.fchmod(descriptor, 0o600)
        except FileExistsError:
            descriptor = os.open(lock_name, _file_flags(writable=True), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        named = os.stat(lock_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _token(opened) != _token(named)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            _fail("state_lock_invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        closing = os.stat(lock_name, dir_fd=parent_fd, follow_symlinks=False)
        if _token(os.fstat(descriptor)) != _token(closing):
            _fail("state_lock_invalid")
        if created:
            os.fsync(descriptor)
            os.fsync(parent_fd)
        return descriptor
    except TelegramOffsetStateError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        _fail("state_lock_invalid")


def _same_snapshot(left: OffsetSnapshot, right: OffsetSnapshot) -> bool:
    return (
        left.offset == right.offset
        and left.revision == right.revision
        and secrets.compare_digest(left.identity_sha256, right.identity_sha256)
        and left.file_token == right.file_token
    )


def _same_renamed_snapshot(left: OffsetSnapshot, right: OffsetSnapshot) -> bool:
    left_token = left.file_token[:-1] if left.file_token is not None else None
    right_token = right.file_token[:-1] if right.file_token is not None else None
    return (
        left.offset == right.offset
        and left.revision == right.revision
        and secrets.compare_digest(left.identity_sha256, right.identity_sha256)
        and left_token == right_token
    )


def _renameatx_np(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
    flags: int,
) -> None:
    try:
        function = ctypes.CDLL(None, use_errno=True).renameatx_np
    except (AttributeError, OSError):
        _fail("state_platform_unsupported")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(
        source_fd,
        os.fsencode(source_name),
        destination_fd,
        os.fsencode(destination_name),
        flags,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, "renameatx_np failed")


def _validate_lock_identity(parent_fd: int, state_name: str, lock_fd: int) -> None:
    try:
        named = os.stat(f".{state_name}.lock", dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(lock_fd)
    except OSError:
        _fail("state_lock_invalid")
    if _token(named) != _token(opened):
        _fail("state_lock_invalid")


def load_offset(path: Path | str, *, bot_token: str, owner_id: str) -> OffsetSnapshot:
    state_path = _validate_path(path)
    identity_sha256 = _identity(bot_token, owner_id)
    parent_fd = _open_parent(state_path)
    lock_fd: int | None = None
    try:
        lock_fd = _open_lock(parent_fd, state_path.name)
        result = _read_state(parent_fd, state_path.name, identity_sha256)
        _validate_parent_identity(state_path, parent_fd)
        return result
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(parent_fd)


def _quarantine_remove(
    parent_fd: int,
    state_name: str,
    name: str,
    token: tuple[int, ...] | None,
) -> bool:
    if token is None:
        return True
    for _attempt in range(TEMP_ATTEMPTS):
        quarantine = f".{state_name}.quarantine.{secrets.token_hex(16)}"
        try:
            _renameatx_np(parent_fd, name, parent_fd, quarantine, RENAME_EXCL)
        except FileNotFoundError:
            return True
        except FileExistsError:
            continue
        except (OSError, TelegramOffsetStateError):
            return False
        try:
            moved = os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
            if _cleanup_token(moved) != token:
                return False
            os.unlink(quarantine, dir_fd=parent_fd)
            return True
        except OSError:
            return False
    return False


def advance_offset(
    path: Path | str,
    *,
    bot_token: str,
    owner_id: str,
    new_offset: int,
    expected: OffsetSnapshot,
) -> OffsetSnapshot:
    if type(new_offset) is not int or new_offset < 0 or new_offset > MAX_OFFSET:
        _fail("state_malformed")
    if not isinstance(expected, OffsetSnapshot):
        _fail("state_conflict")
    state_path = _validate_path(path)
    identity_sha256 = _identity(bot_token, owner_id)
    if not secrets.compare_digest(expected.identity_sha256, identity_sha256):
        _fail("state_identity_mismatch")
    parent_fd = _open_parent(state_path)
    lock_fd: int | None = None
    temp_name: str | None = None
    temp_token: tuple[int, ...] | None = None
    published = False
    try:
        lock_fd = _open_lock(parent_fd, state_path.name)
        current = _read_state(parent_fd, state_path.name, identity_sha256)
        if not _same_snapshot(current, expected):
            _fail("state_conflict")
        if current.offset is not None and new_offset < current.offset:
            _fail("state_regression")
        if current.offset == new_offset:
            closing = _read_state(parent_fd, state_path.name, identity_sha256)
            if not _same_snapshot(closing, current):
                _fail("state_conflict")
            _validate_lock_identity(parent_fd, state_path.name, lock_fd)
            _validate_parent_identity(state_path, parent_fd)
            return closing
        revision = current.revision + 1
        if revision > MAX_OFFSET:
            _fail("state_malformed")
        payload = _canonical(
            {
                "identity_sha256": identity_sha256,
                "offset": new_offset,
                "revision": revision,
                "version": STATE_VERSION,
            }
        )
        temp_fd: int | None = None
        for _attempt in range(TEMP_ATTEMPTS):
            candidate = f".{state_path.name}.tmp.{secrets.token_hex(16)}"
            try:
                temp_fd = os.open(
                    candidate,
                    _file_flags(writable=True, create=True, exclusive=True),
                    0o600,
                    dir_fd=parent_fd,
                )
                temp_name = candidate
                temp_token = _cleanup_token(os.fstat(temp_fd))
                break
            except FileExistsError:
                continue
            except OSError:
                _fail("state_write_failed")
        if temp_fd is None or temp_name is None:
            _fail("state_temp_collision")
        try:
            os.fchmod(temp_fd, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(temp_fd, view)
                if written <= 0:
                    _fail("state_write_failed")
                view = view[written:]
            temp_token = _cleanup_token(os.fstat(temp_fd))
            os.fsync(temp_fd)
            temp_token = _cleanup_token(os.fstat(temp_fd))
        except TelegramOffsetStateError:
            raise
        except OSError:
            _fail("state_write_failed")
        finally:
            os.close(temp_fd)
        closing = _read_state(parent_fd, state_path.name, identity_sha256)
        if not _same_snapshot(closing, current):
            _fail("state_conflict")
        _validate_lock_identity(parent_fd, state_path.name, lock_fd)
        _validate_parent_identity(state_path, parent_fd)
        try:
            if current.fresh:
                _renameatx_np(
                    parent_fd,
                    temp_name,
                    parent_fd,
                    state_path.name,
                    RENAME_EXCL,
                )
                published = True
                temp_name = None
                temp_token = None
            else:
                _renameatx_np(
                    parent_fd,
                    temp_name,
                    parent_fd,
                    state_path.name,
                    RENAME_SWAP,
                )
                published = True
                displaced = _read_state(parent_fd, temp_name, identity_sha256)
                if not _same_renamed_snapshot(displaced, current):
                    try:
                        _renameatx_np(
                            parent_fd,
                            state_path.name,
                            parent_fd,
                            temp_name,
                            RENAME_SWAP,
                        )
                        published = False
                        restored = _read_state(parent_fd, state_path.name, identity_sha256)
                    except (OSError, TelegramOffsetStateError):
                        _fail("state_persistence_uncertain")
                    if not _same_renamed_snapshot(restored, current):
                        _fail("state_persistence_uncertain")
                    _fail("state_conflict")
                displaced_token = (
                    displaced.file_token[:-1] if displaced.file_token is not None else None
                )
                if not _quarantine_remove(
                    parent_fd,
                    state_path.name,
                    temp_name,
                    displaced_token,
                ):
                    _fail("state_persistence_uncertain")
                temp_name = None
                temp_token = None
            os.fsync(parent_fd)
        except FileExistsError:
            _fail("state_conflict")
        except OSError:
            _fail("state_persistence_uncertain" if published else "state_write_failed")
        try:
            result = _read_state(parent_fd, state_path.name, identity_sha256)
        except TelegramOffsetStateError:
            _fail("state_persistence_uncertain")
        if result.offset != new_offset or result.revision != revision:
            _fail("state_persistence_uncertain")
        try:
            _validate_parent_identity(state_path, parent_fd)
        except TelegramOffsetStateError:
            _fail("state_persistence_uncertain")
        return result
    except TelegramOffsetStateError:
        if temp_name is not None and not _quarantine_remove(
            parent_fd,
            state_path.name,
            temp_name,
            temp_token,
        ):
            raise TelegramOffsetStateError("state_persistence_uncertain") from None
        raise
    except BaseException:
        if temp_name is not None:
            _quarantine_remove(parent_fd, state_path.name, temp_name, temp_token)
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(parent_fd)
