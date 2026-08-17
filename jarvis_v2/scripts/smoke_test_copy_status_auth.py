from __future__ import annotations

import contextlib
import io
import os
import stat
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from jarvis_v2.scripts import copy_status_auth as auth_copy
from jarvis_v2.ui.status_config import status_auth_token_is_valid


TOKEN = "synthetic-dashboard-password-" + ("q" * 40)
SECOND_TOKEN = "second-synthetic-dashboard-password-" + ("r" * 40)


def _run_main(path: Path, runner: object) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = 0
    with (
        mock.patch.object(sys, "argv", ["copy_status_auth", "--env-file", str(path)]),
        mock.patch.object(auth_copy.subprocess, "run", side_effect=runner),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        try:
            auth_copy.main()
        except SystemExit as exc:
            code = int(exc.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


def _write_owner_only(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o600)


def test_copy_status_auth() -> None:
    if not status_auth_token_is_valid(TOKEN) or not status_auth_token_is_valid(SECOND_TOKEN):
        raise SystemExit("synthetic copy smoke passwords no longer satisfy the production contract")

    with TemporaryDirectory(prefix="jarvis-status-auth-copy-") as temp:
        root = Path(temp)
        valid = root / "valid.env"
        _write_owner_only(
            valid,
            "# quoted values and inline comments use the shared env parser\n"
            f"export JARVIS_STATUS_AUTH_TOKEN='{TOKEN}' # local-only note\n",
        )
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fake_copy(*args: object, **kwargs: object) -> SimpleNamespace:
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=0)

        before = (valid.read_bytes(), valid.stat().st_ino, valid.stat().st_mtime_ns)
        code, stdout, stderr = _run_main(valid, fake_copy)
        after = (valid.read_bytes(), valid.stat().st_ino, valid.stat().st_mtime_ns)
        if code != 0 or stderr or "copied to the local clipboard" not in stdout:
            raise SystemExit(f"valid dashboard password copy failed: {code} {stdout!r} {stderr!r}")
        if before != after:
            raise SystemExit("dashboard password copy changed the selected environment file")
        if TOKEN in stdout or TOKEN in stderr:
            raise SystemExit("dashboard password copy CLI displayed the password")
        if len(calls) != 1:
            raise SystemExit("valid dashboard password copy did not invoke one clipboard process")
        args, kwargs = calls[0]
        if args != (["/usr/bin/pbcopy"],):
            raise SystemExit(f"dashboard password copy invoked an unexpected program: {args!r}")
        if kwargs.get("input") != TOKEN.encode("utf-8"):
            raise SystemExit("dashboard password copy changed the parsed quoted password")
        if (
            kwargs.get("stdout") is not subprocess.DEVNULL
            or kwargs.get("stderr") is not subprocess.DEVNULL
            or kwargs.get("timeout") != 5
            or kwargs.get("check") is not False
        ):
            raise SystemExit("dashboard password copy did not suppress child output or bound execution")

        quoted_hash = root / "quoted-hash.env"
        hash_token = TOKEN + "#inside"
        _write_owner_only(
            quoted_hash,
            f'JARVIS_STATUS_AUTH_TOKEN="{hash_token}"\n',
        )
        quoted_inputs: list[bytes] = []

        def fake_quoted(*args: object, **kwargs: object) -> SimpleNamespace:
            quoted_inputs.append(kwargs["input"])  # type: ignore[arg-type]
            return SimpleNamespace(returncode=0)

        with mock.patch.object(auth_copy.subprocess, "run", side_effect=fake_quoted):
            result = auth_copy.copy_status_auth(quoted_hash)
        if not result.copied or quoted_inputs != [hash_token.encode("utf-8")]:
            raise SystemExit("shared env parsing changed a quoted hash password")

        refused_cases: list[tuple[str, Path, str]] = []

        duplicate = root / "duplicate.env"
        _write_owner_only(
            duplicate,
            f"JARVIS_STATUS_AUTH_TOKEN={TOKEN}\n"
            f"export JARVIS_STATUS_AUTH_TOKEN={SECOND_TOKEN}\n",
        )
        refused_cases.append(("duplicate", duplicate, "multiple dashboard authentication entries"))

        unsafe = root / "unsafe.env"
        _write_owner_only(unsafe, f"JARVIS_STATUS_AUTH_TOKEN={TOKEN}\n")
        unsafe.chmod(0o640)
        refused_cases.append(("unsafe mode", unsafe, "unsafe custody"))

        missing = root / "missing.env"
        _write_owner_only(missing, "UNRELATED=value\n")
        refused_cases.append(("missing entry", missing, "missing or invalid"))

        invalid = root / "invalid.env"
        _write_owner_only(invalid, "JARVIS_STATUS_AUTH_TOKEN=short\n")
        refused_cases.append(("invalid entry", invalid, "missing or invalid"))

        absent = root / "absent.env"
        refused_cases.append(("missing file", absent, "does not exist"))

        target = root / "target.env"
        _write_owner_only(target, f"JARVIS_STATUS_AUTH_TOKEN={TOKEN}\n")
        symlink = root / "symlink.env"
        symlink.symlink_to(target)
        refused_cases.append(("symlink", symlink, "unsafe custody"))

        directory = root / "directory.env"
        directory.mkdir()
        refused_cases.append(("directory", directory, "secure regular file"))

        fifo = root / "fifo.env"
        os.mkfifo(fifo, mode=0o600)
        refused_cases.append(("named pipe", fifo, "secure regular file"))

        for label, path, expected in refused_cases:
            def forbidden_copy(*args: object, **kwargs: object) -> object:
                raise AssertionError(f"{label} reached the clipboard subprocess")

            code, stdout, stderr = _run_main(path, forbidden_copy)
            if code != 2 or stdout or expected not in stderr:
                raise SystemExit(f"{label} did not fail closed: {code} {stdout!r} {stderr!r}")
            for secret in (TOKEN, SECOND_TOKEN, "short"):
                if secret in stdout or secret in stderr:
                    raise SystemExit(f"{label} refusal displayed secret material")

        wrong_owner = root / "wrong-owner.env"
        _write_owner_only(wrong_owner, f"JARVIS_STATUS_AUTH_TOKEN={TOKEN}\n")
        with mock.patch("jarvis_v2.env.os.geteuid", return_value=os.geteuid() + 1):
            code, stdout, stderr = _run_main(
                wrong_owner,
                AssertionError("wrong-owner file reached the clipboard subprocess"),
            )
        if code != 2 or stdout or "unsafe custody" not in stderr or TOKEN in stderr:
            raise SystemExit("wrong-owner dashboard password file did not fail closed")

        process_failure = root / "process-failure.env"
        _write_owner_only(process_failure, f"JARVIS_STATUS_AUTH_TOKEN={TOKEN}\n")
        code, stdout, stderr = _run_main(
            process_failure,
            OSError(f"seeded failure containing {TOKEN}"),
        )
        if code != 2 or stdout or "clipboard could not be updated" not in stderr or TOKEN in stderr:
            raise SystemExit("clipboard process failure leaked secret or detail")

        nonzero_calls = 0

        def fake_nonzero(*args: object, **kwargs: object) -> SimpleNamespace:
            nonlocal nonzero_calls
            nonzero_calls += 1
            return SimpleNamespace(returncode=9)

        code, stdout, stderr = _run_main(process_failure, fake_nonzero)
        if (
            code != 2
            or stdout
            or "clipboard could not be updated" not in stderr
            or TOKEN in stderr
            or nonzero_calls != 1
        ):
            raise SystemExit("nonzero clipboard process did not fail without disclosure")

    source = Path(auth_copy.__file__).read_text(encoding="utf-8")
    for forbidden in ("import logging", "shell=True", "check_output", "os.system"):
        if forbidden in source:
            raise SystemExit(f"dashboard password copy gained an unsafe output/process path: {forbidden}")


def main() -> None:
    test_copy_status_auth()
    print("status auth clipboard copy smoke ok")


if __name__ == "__main__":
    main()
