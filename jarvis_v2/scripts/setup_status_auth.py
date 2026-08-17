"""Securely create or validate the local status-dashboard authentication token.

This operator utility only manages the selected environment file.  It never
starts, stops, installs, or reloads a service, and it never prints the token.
"""

from __future__ import annotations

import argparse
import errno
import os
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from jarvis_v2.ui.status_config import STATUS_AUTH_ENV, status_auth_token_is_valid


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAX_ENV_BYTES = 1024 * 1024
_TOKEN_ENTROPY_BYTES = 32
_OWNER_MODE = 0o600


class StatusAuthSetupError(RuntimeError):
    """Bounded operator-facing refusal with no secret or local-path detail."""


@dataclass(frozen=True)
class StatusAuthSetupResult:
    state: str
    token_generated: bool
    custody_repaired: bool


def _selected_env_path(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser()
    configured = os.getenv("JARVIS_V3_ENV", "").strip()
    return Path(configured).expanduser() if configured else _PROJECT_ROOT / ".env"


def _validate_regular_owner(env_stat: os.stat_result) -> None:
    if not stat.S_ISREG(env_stat.st_mode):
        raise StatusAuthSetupError("the environment path is not a regular file")
    if env_stat.st_uid != os.geteuid():
        raise StatusAuthSetupError("the environment file is not owned by the current user")


def _read_existing_once(path: Path) -> tuple[int, os.stat_result, bytes] | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise StatusAuthSetupError("environment file symlinks are not allowed") from None
        raise StatusAuthSetupError("the environment file could not be opened securely") from None

    try:
        env_stat = os.fstat(fd)
        _validate_regular_owner(env_stat)
        if env_stat.st_size > _MAX_ENV_BYTES:
            raise StatusAuthSetupError("the environment file is too large to update safely")
        contents = bytearray()
        while True:
            chunk = os.read(fd, min(65536, _MAX_ENV_BYTES + 1 - len(contents)))
            if not chunk:
                break
            contents.extend(chunk)
            if len(contents) > _MAX_ENV_BYTES:
                raise StatusAuthSetupError("the environment file is too large to update safely")
        return fd, env_stat, bytes(contents)
    except Exception:
        os.close(fd)
        raise


def _token_values(contents: bytes) -> list[str]:
    values: list[str] = []
    expected_key = STATUS_AUTH_ENV.encode("ascii")
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(b"#") or b"=" not in line:
            continue
        if line.startswith(b"export "):
            line = line.removeprefix(b"export ").strip()
        key, _, raw_value = line.partition(b"=")
        if key.strip() != expected_key:
            continue
        value = raw_value.strip()
        quoted = (
            len(value) >= 2
            and ((value.startswith(b'"') and value.endswith(b'"')) or (value.startswith(b"'") and value.endswith(b"'")))
        )
        if not quoted and b" #" in value:
            value = value.split(b" #", 1)[0].rstrip()
        value = value.strip(b'"').strip(b"'")
        try:
            values.append(value.decode("ascii"))
        except UnicodeDecodeError:
            values.append("")
    return values


def _new_contents(contents: bytes, token: str) -> bytes:
    separator = b"" if not contents or contents.endswith((b"\n", b"\r")) else b"\n"
    return contents + separator + f"{STATUS_AUTH_ENV}={token}\n".encode("ascii")


def _open_parent(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(path.parent, flags)
    except OSError:
        raise StatusAuthSetupError("the environment directory is unavailable") from None
    try:
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise StatusAuthSetupError("the environment parent is not a directory")
    except Exception:
        os.close(directory_fd)
        raise
    return directory_fd


def _write_temp(path: Path, contents: bytes) -> tuple[int, Path]:
    if len(contents) > _MAX_ENV_BYTES:
        raise StatusAuthSetupError("the updated environment file would be too large")
    fd = -1
    temp_path: Path | None = None
    prepared = False
    try:
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.status-auth-",
            suffix=".tmp",
            dir=path.parent,
        )
        temp_path = Path(raw_path)
        os.fchmod(fd, _OWNER_MODE)
        temp_stat = os.fstat(fd)
        _validate_regular_owner(temp_stat)
        view = memoryview(contents)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short environment-file write")
            view = view[written:]
        os.fsync(fd)
        prepared = True
        return fd, temp_path
    except StatusAuthSetupError:
        raise
    except OSError:
        raise StatusAuthSetupError("the secure environment update could not be prepared") from None
    finally:
        if not prepared and temp_path is not None and fd >= 0:
            # Ownership transfers to the caller only on a successful return.
            # During exception unwinding, close and remove the unpublished file.
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _same_file_now(path: Path, original: os.stat_result, fd: int) -> bool:
    try:
        current_path = os.stat(path, follow_symlinks=False)
        current_fd = os.fstat(fd)
    except OSError:
        return False
    original_identity = (original.st_dev, original.st_ino)
    return (
        stat.S_ISREG(current_path.st_mode)
        and (current_path.st_dev, current_path.st_ino) == original_identity
        and (current_fd.st_dev, current_fd.st_ino) == original_identity
        and current_fd.st_size == original.st_size
        and current_fd.st_mtime_ns == original.st_mtime_ns
    )


def _publish(
    path: Path,
    contents: bytes,
    *,
    existing_fd: int | None,
    existing_stat: os.stat_result | None,
) -> None:
    directory_fd = _open_parent(path)
    temp_fd = -1
    temp_path: Path | None = None
    try:
        temp_fd, temp_path = _write_temp(path, contents)
        os.close(temp_fd)
        temp_fd = -1
        if existing_stat is None:
            try:
                os.link(temp_path, path, follow_symlinks=False)
            except FileExistsError:
                raise StatusAuthSetupError(
                    "the environment file appeared during setup; rerun after inspecting it"
                ) from None
            except OSError:
                raise StatusAuthSetupError("the environment file could not be published securely") from None
            try:
                os.unlink(temp_path)
            except OSError:
                raise StatusAuthSetupError(
                    "the environment update was published but temporary-file cleanup failed"
                ) from None
            temp_path = None
        else:
            if existing_fd is None or not _same_file_now(path, existing_stat, existing_fd):
                raise StatusAuthSetupError(
                    "the environment file changed during setup; rerun without concurrent edits"
                )
            try:
                os.replace(temp_path, path)
            except OSError:
                raise StatusAuthSetupError("the environment file could not be replaced atomically") from None
            temp_path = None
        try:
            os.fsync(directory_fd)
        except OSError:
            raise StatusAuthSetupError(
                "the environment update was published but directory durability was not confirmed"
            ) from None
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        os.close(directory_fd)


def setup_status_auth(env_path: Path | None = None) -> StatusAuthSetupResult:
    """Generate or validate the dashboard token without exposing its value."""

    selected = _selected_env_path(env_path)
    existing = _read_existing_once(selected)
    existing_fd: int | None = None
    try:
        if existing is None:
            original_stat = None
            contents = b""
        else:
            existing_fd, original_stat, contents = existing

        values = _token_values(contents)
        if len(values) > 1:
            raise StatusAuthSetupError("multiple dashboard authentication entries are not allowed")
        if values:
            if not status_auth_token_is_valid(values[0]):
                raise StatusAuthSetupError(
                    "the existing dashboard authentication token is invalid; replace it deliberately"
                )
            custody_repaired = False
            if original_stat is not None and stat.S_IMODE(original_stat.st_mode) != _OWNER_MODE:
                try:
                    os.fchmod(existing_fd, _OWNER_MODE)
                    os.fsync(existing_fd)
                except OSError:
                    raise StatusAuthSetupError(
                        "owner-only environment-file permissions could not be applied"
                    ) from None
                custody_repaired = True
            return StatusAuthSetupResult(
                state="existing_valid",
                token_generated=False,
                custody_repaired=custody_repaired,
            )

        token = secrets.token_urlsafe(_TOKEN_ENTROPY_BYTES)
        if not status_auth_token_is_valid(token):
            raise StatusAuthSetupError("secure token generation returned an invalid value")
        _publish(
            selected,
            _new_contents(contents, token),
            existing_fd=existing_fd,
            existing_stat=original_stat,
        )
        return StatusAuthSetupResult(
            state="generated",
            token_generated=True,
            custody_repaired=(
                original_stat is not None and stat.S_IMODE(original_stat.st_mode) != _OWNER_MODE
            ),
        )
    finally:
        if existing_fd is not None:
            os.close(existing_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or validate the Jarvis status-dashboard token in an owner-only "
            "environment file. The token is never printed."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Environment file to update (defaults to JARVIS_V3_ENV or the project .env).",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        result = setup_status_auth(args.env_file)
    except StatusAuthSetupError as exc:
        print(f"Dashboard authentication setup refused: {exc}.", file=os.sys.stderr)
        raise SystemExit(2) from None
    if result.token_generated:
        print(
            "Dashboard authentication is configured in an owner-only environment file. "
            "A new token was generated and was not displayed."
        )
    else:
        custody = " Owner-only permissions were applied." if result.custody_repaired else ""
        print(
            "Dashboard authentication is already configured and valid."
            f"{custody} The token was not displayed."
        )


if __name__ == "__main__":
    main()
