from __future__ import annotations

import contextlib
import io
import os
import stat
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.scripts import setup_status_auth as auth_setup
from jarvis_v2.ui.status_config import status_auth_token_is_valid


GENERATED_TOKEN = "generated-status-auth-token-" + ("x" * 40)
EXISTING_TOKEN = "existing-status-auth-token-" + ("y" * 40)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _run_main(path: Path) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    with (
        mock.patch.object(sys, "argv", ["setup_status_auth", "--env-file", str(path)]),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        try:
            auth_setup.main()
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_setup_status_auth() -> None:
    if not status_auth_token_is_valid(GENERATED_TOKEN) or not status_auth_token_is_valid(EXISTING_TOKEN):
        raise SystemExit("status-auth smoke tokens no longer satisfy the production contract")

    with TemporaryDirectory(prefix="jarvis-status-auth-setup-") as temp:
        root = Path(temp)

        missing = root / "missing.env"
        with mock.patch.object(auth_setup.secrets, "token_urlsafe", return_value=GENERATED_TOKEN) as generator:
            code, stdout, stderr = _run_main(missing)
        if code != 0 or stderr or generator.call_args_list != [mock.call(32)]:
            raise SystemExit(f"missing-file setup failed safely: {code} {stdout!r} {stderr!r}")
        if missing.read_bytes() != f"JARVIS_STATUS_AUTH_TOKEN={GENERATED_TOKEN}\n".encode():
            raise SystemExit("missing-file setup did not publish the exact generated entry")
        if _mode(missing) != 0o600:
            raise SystemExit("generated environment file is not owner-only")
        if GENERATED_TOKEN in stdout or GENERATED_TOKEN in stderr:
            raise SystemExit("setup CLI printed the generated dashboard token")
        before = (missing.read_bytes(), missing.stat().st_ino, missing.stat().st_mtime_ns)
        with mock.patch.object(
            auth_setup.secrets,
            "token_urlsafe",
            side_effect=AssertionError("idempotent setup regenerated the token"),
        ):
            code, stdout, stderr = _run_main(missing)
        after = (missing.read_bytes(), missing.stat().st_ino, missing.stat().st_mtime_ns)
        if code != 0 or stderr or before != after or "already configured and valid" not in stdout:
            raise SystemExit("idempotent setup changed an existing valid token")
        if GENERATED_TOKEN in stdout:
            raise SystemExit("idempotent setup printed the existing dashboard token")

        preserved = root / "preserved.env"
        original = b"# exact bytes stay exact\nALPHA=1\nexport BETA='two words'\nTAIL=no-newline"
        preserved.write_bytes(original)
        preserved.chmod(0o644)
        replace_calls: list[tuple[object, object]] = []
        real_replace = auth_setup.os.replace

        def tracked_replace(source: object, destination: object) -> None:
            replace_calls.append((source, destination))
            real_replace(source, destination)

        fsync_calls: list[int] = []
        real_fsync = auth_setup.os.fsync

        def tracked_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            real_fsync(fd)

        with (
            mock.patch.object(auth_setup.secrets, "token_urlsafe", return_value=GENERATED_TOKEN),
            mock.patch.object(auth_setup.os, "replace", side_effect=tracked_replace),
            mock.patch.object(auth_setup.os, "fsync", side_effect=tracked_fsync),
        ):
            result = auth_setup.setup_status_auth(preserved)
        expected = original + f"\nJARVIS_STATUS_AUTH_TOKEN={GENERATED_TOKEN}\n".encode()
        if preserved.read_bytes() != expected:
            raise SystemExit("setup did not preserve all pre-existing environment bytes")
        if _mode(preserved) != 0o600 or not result.token_generated or not result.custody_repaired:
            raise SystemExit(f"atomic update did not establish owner-only custody: {result}")
        if len(replace_calls) != 1 or len(fsync_calls) < 2:
            raise SystemExit("existing-file update missed atomic replace or durable fsync")
        if list(root.glob(f".{preserved.name}.status-auth-*.tmp")):
            raise SystemExit("successful setup left a temporary environment file")

        valid_loose = root / "valid-loose.env"
        valid_bytes = f"KEEP=unchanged\nJARVIS_STATUS_AUTH_TOKEN={EXISTING_TOKEN}\n".encode()
        valid_loose.write_bytes(valid_bytes)
        valid_loose.chmod(0o644)
        loose_inode = valid_loose.stat().st_ino
        result = auth_setup.setup_status_auth(valid_loose)
        if (
            result.state != "existing_valid"
            or result.token_generated
            or not result.custody_repaired
            or valid_loose.read_bytes() != valid_bytes
            or valid_loose.stat().st_ino != loose_inode
            or _mode(valid_loose) != 0o600
        ):
            raise SystemExit(f"valid-token custody repair changed content or identity: {result}")

        invalid = root / "invalid.env"
        invalid.write_bytes(b"KEEP=1\nJARVIS_STATUS_AUTH_TOKEN=short\n")
        invalid.chmod(0o600)
        invalid_before = invalid.read_bytes()
        code, stdout, stderr = _run_main(invalid)
        if code != 2 or stdout or "existing dashboard authentication token is invalid" not in stderr:
            raise SystemExit("invalid existing token did not fail closed")
        if invalid.read_bytes() != invalid_before or "short" in stderr:
            raise SystemExit("invalid-token refusal changed or exposed the environment value")

        duplicate = root / "duplicate.env"
        duplicate.write_text(
            f"JARVIS_STATUS_AUTH_TOKEN={EXISTING_TOKEN}\n"
            f"export JARVIS_STATUS_AUTH_TOKEN={GENERATED_TOKEN}\n",
            encoding="utf-8",
        )
        duplicate.chmod(0o600)
        duplicate_before = duplicate.read_bytes()
        try:
            auth_setup.setup_status_auth(duplicate)
        except auth_setup.StatusAuthSetupError as exc:
            if "multiple dashboard authentication entries" not in str(exc):
                raise SystemExit(f"duplicate-token refusal was not bounded: {exc}")
        else:
            raise SystemExit("duplicate dashboard tokens were accepted")
        if duplicate.read_bytes() != duplicate_before:
            raise SystemExit("duplicate-token refusal changed the environment file")

        target = root / "target.env"
        target.write_bytes(b"KEEP=target\n")
        target.chmod(0o600)
        symlink = root / "symlink.env"
        symlink.symlink_to(target)
        try:
            auth_setup.setup_status_auth(symlink)
        except auth_setup.StatusAuthSetupError as exc:
            if "symlinks are not allowed" not in str(exc):
                raise SystemExit(f"symlink refusal was not bounded: {exc}")
        else:
            raise SystemExit("environment-file symlink was accepted")
        if target.read_bytes() != b"KEEP=target\n":
            raise SystemExit("symlink refusal changed its target")

        directory = root / "directory.env"
        directory.mkdir()
        try:
            auth_setup.setup_status_auth(directory)
        except auth_setup.StatusAuthSetupError as exc:
            if "not a regular file" not in str(exc):
                raise SystemExit(f"non-regular refusal was not bounded: {exc}")
        else:
            raise SystemExit("non-regular environment path was accepted")

        synthetic = os.stat_result((stat.S_IFREG | 0o600, 1, 1, 1, os.geteuid() + 1, 0, 0, 0, 0, 0))
        try:
            auth_setup._validate_regular_owner(synthetic)
        except auth_setup.StatusAuthSetupError as exc:
            if "not owned by the current user" not in str(exc):
                raise SystemExit(f"wrong-owner refusal was not bounded: {exc}")
        else:
            raise SystemExit("wrong-owner environment metadata was accepted")

        atomic = root / "atomic.env"
        atomic_before = b"KEEP=atomic\n"
        atomic.write_bytes(atomic_before)
        atomic.chmod(0o600)
        with (
            mock.patch.object(auth_setup.secrets, "token_urlsafe", return_value=GENERATED_TOKEN),
            mock.patch.object(auth_setup.os, "replace", side_effect=OSError("seeded private detail")),
        ):
            try:
                auth_setup.setup_status_auth(atomic)
            except auth_setup.StatusAuthSetupError as exc:
                if str(exc) != "the environment file could not be replaced atomically":
                    raise SystemExit(f"atomic publication failure leaked detail: {exc}")
            else:
                raise SystemExit("atomic publication failure was accepted")
        if atomic.read_bytes() != atomic_before:
            raise SystemExit("failed atomic publication changed the original environment file")
        if list(root.glob(f".{atomic.name}.status-auth-*.tmp")):
            raise SystemExit("failed atomic publication left a temporary environment file")

    source = Path(auth_setup.__file__).read_text(encoding="utf-8")
    for forbidden in ("launchctl", "subprocess", "run_status_server", "serve_forever"):
        if forbidden in source:
            raise SystemExit(f"status-auth setup gained service/process authority: {forbidden}")


def main() -> None:
    test_setup_status_auth()
    print("status auth setup smoke ok")


if __name__ == "__main__":
    main()
