from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, NoReturn

from jarvis_v2.env import V3PathCustodyError, read_env_values, validate_v3_path_custody


MINIMUM_V3_PYTHON = (3, 11)
V3_REQUIRED_LAUNCH_MODULES = ("sqlite3", "ssl")
V3_CONFIGURED_PYTHON_ENV = "JARVIS_V3_PYTHON"
MAX_CONFIGURED_PYTHON_PATH_BYTES = 1024
_CUSTODY_UNSET = object()
_V3_PYTHON_REEXEC_GUARD = "_JARVIS_V3_PYTHON_REEXEC_GUARD"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMON_V3_PYTHONS = (
    Path("/opt/homebrew/bin/python3"),
    Path("/usr/local/bin/python3"),
)
_MODULE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}")
_INTERPRETER_PROBE_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": os.defpath,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
}
_INTERPRETER_PROBE = """
import importlib
import json
import sys

required = json.loads(sys.argv[1])
preferred = json.loads(sys.argv[2])

def missing(names):
    result = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            result.append(name)
    return result

print(json.dumps({
    "version": [sys.version_info.major, sys.version_info.minor],
    "missing_required": missing(required),
    "missing_preferred": missing(preferred),
}, sort_keys=True))
""".strip()
V3_INHERITED_CUSTODY_KEYS = frozenset(
    {
        "GMAIL_ADDRESS",
        "GMAIL_APP_PASSWORD",
        "JARVIS_CACHE_DIR",
        "JARVIS_DATA_DIR",
        "JARVIS_DB_PATH",
        "JARVIS_GOOGLE_CREDS",
        "JARVIS_GOOGLE_READONLY_TOKEN",
        "JARVIS_GOOGLE_TOKEN",
        "JARVIS_OAV_VISION_REVIEWER_COMMAND",
        "JARVIS_OBSIDIAN_ROOT",
        "JARVIS_OBSIDIAN_VAULT",
        "JARVIS_OWNER_IMESSAGE",
        "JARVIS_OWNER_TELEGRAM",
        "JARVIS_REMINDERS_FILE",
        "JARVIS_STATUS_AUTH_TOKEN",
        "JARVIS_STORAGE_FALLBACK_DIR",
        "JARVIS_TELEGRAM_STATE",
        "JARVIS_V3_IMESSAGE_STATE",
        "JARVIS_V3_PYTHON",
        "JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH",
        "JARVIS_VOICE_WHISPER_CLI",
        "JARVIS_VOICE_WHISPER_MODEL_PATH",
        "JARVIS_WATCHED_DIRS",
        "OBSIDIAN_VAULT_PATH",
        "OPENAI_API_KEY",
        "TELEGRAM_BOT_TOKEN",
    }
)


class V3InheritedEnvironmentConflict(RuntimeError):
    """A selected V3 env file conflicts with inherited runtime custody."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys
        super().__init__("conflicting inherited V3 runtime custody")


class V3EnvironmentUnavailable(RuntimeError):
    """The explicitly selected V3 environment could not be read safely."""


class V3PythonUnavailable(RuntimeError):
    """No compatible interpreter can start the V3 source safely."""


@dataclass(frozen=True)
class V3PythonProbe:
    path: Path
    version: tuple[int, int]
    missing_required: tuple[str, ...]
    missing_preferred: tuple[str, ...]

    @property
    def launch_ready(self) -> bool:
        return self.version >= MINIMUM_V3_PYTHON and not self.missing_required

    @property
    def preferred_ready(self) -> bool:
        return self.launch_ready and not self.missing_preferred


@dataclass(frozen=True, repr=False)
class _V3EnvironmentCustody:
    selected_environment: bool
    configured_python: str | None


def _selected_v3_environment_values() -> dict[str, str] | None:
    """Return one safely read selected environment snapshot, when configured."""

    configured_path = os.getenv("JARVIS_V3_ENV", "").strip()
    if not configured_path:
        return None
    validate_v3_path_custody({"JARVIS_V3_ENV": configured_path})
    try:
        return read_env_values(
            Path(configured_path),
            require_owner_only=True,
        )
    except (OSError, UnicodeError, ValueError):
        raise V3EnvironmentUnavailable("selected V3 environment is unavailable") from None


def assert_v3_environment_custody() -> _V3EnvironmentCustody:
    """Refuse inherited custody overrides that differ from the selected V3 env."""

    selected_values = _selected_v3_environment_values()
    if selected_values is None:
        validate_v3_path_custody(dict(os.environ))
        return _V3EnvironmentCustody(False, None)
    conflicts = tuple(
        sorted(
            key
            for key in V3_INHERITED_CUSTODY_KEYS
            if (inherited := os.getenv(key)) is not None
            and selected_values.get(key) != inherited
        )
    )
    if conflicts:
        raise V3InheritedEnvironmentConflict(conflicts)
    effective = dict(os.environ)
    effective.update({key: value for key, value in selected_values.items() if key not in effective})
    validate_v3_path_custody(effective)
    selected_python = selected_values.get(V3_CONFIGURED_PYTHON_ENV)
    return _V3EnvironmentCustody(
        True,
        selected_python if selected_python else None,
    )


def _safe_module_names(names: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(name).strip() for name in names))
    if any(not _MODULE_NAME_RE.fullmatch(name) for name in selected):
        raise ValueError("invalid interpreter dependency name")
    return selected


def inspect_v3_python(
    path: Path,
    required_modules: tuple[str, ...],
    preferred_modules: tuple[str, ...],
) -> V3PythonProbe | None:
    """Inspect one interpreter without importing Jarvis or reading private state."""

    required = _safe_module_names(required_modules)
    preferred = _safe_module_names(preferred_modules)
    try:
        result = subprocess.run(
            [
                str(path),
                "-I",
                "-c",
                _INTERPRETER_PROBE,
                json.dumps(required),
                json.dumps(preferred),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=dict(_INTERPRETER_PROBE_ENV),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
        version_raw = payload["version"]
        version = (int(version_raw[0]), int(version_raw[1]))
        missing_required = tuple(str(name) for name in payload["missing_required"])
        missing_preferred = tuple(str(name) for name in payload["missing_preferred"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        len(version) != 2
        or version[0] < 0
        or version[1] < 0
        or any(name not in required for name in missing_required)
        or any(name not in preferred for name in missing_preferred)
    ):
        return None
    return V3PythonProbe(path, version, missing_required, missing_preferred)


def _same_executable(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except (OSError, RuntimeError, ValueError):
        try:
            return left.resolve() == right.resolve()
        except (OSError, RuntimeError, ValueError):
            return False


def _reexec_guard(candidate: Path, script: Path) -> str:
    """Bind one re-exec hop to exact current candidate and launcher identities."""

    try:
        candidate.resolve(strict=True)
        script.resolve(strict=True)
        candidate_stat = candidate.stat()
        script_stat = script.stat()
    except (OSError, RuntimeError, ValueError):
        raise V3PythonUnavailable("interpreter re-exec identity is unavailable") from None
    material = {
        "domain": "jarvis-v3-python-reexec-guard-v1",
        "candidate_device": candidate_stat.st_dev,
        "candidate_inode": candidate_stat.st_ino,
        "candidate_uid": candidate_stat.st_uid,
        "candidate_mode": candidate_stat.st_mode,
        "candidate_size": candidate_stat.st_size,
        "candidate_mtime_ns": candidate_stat.st_mtime_ns,
        "candidate_ctime_ns": candidate_stat.st_ctime_ns,
        "script_device": script_stat.st_dev,
        "script_inode": script_stat.st_ino,
        "script_uid": script_stat.st_uid,
        "script_mode": script_stat.st_mode,
        "script_size": script_stat.st_size,
        "script_mtime_ns": script_stat.st_mtime_ns,
        "script_ctime_ns": script_stat.st_ctime_ns,
    }
    try:
        encoded = json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError):
        raise V3PythonUnavailable("interpreter re-exec identity is unavailable") from None
    return hashlib.sha256(encoded).hexdigest()


def _stop_reexec(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    print(
        "Review JARVIS_V3_PYTHON in the selected owner-only environment, then retry.",
        file=sys.stderr,
    )
    raise SystemExit(78)


def _reasonable_python_executable(path: Path) -> bool:
    """Accept regular, executable Python candidates owned by root or this user."""

    try:
        if not path.is_absolute():
            return False
        candidate_stat = path.stat()
        executable = os.access(path, os.X_OK)
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(
        stat.S_ISREG(candidate_stat.st_mode)
        and candidate_stat.st_uid in {0, os.geteuid()}
        and not candidate_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        and executable
    )


def _configured_python(
    custody: _V3EnvironmentCustody | object = _CUSTODY_UNSET,
) -> Path | None:
    """Resolve the authoritative configured interpreter without exposing its value.

    A selected owner-only V3 environment is authoritative, including when it
    intentionally omits this optional setting.  The inherited shell value is
    consulted only when no selected V3 environment exists.
    """

    if custody is _CUSTODY_UNSET:
        custody = assert_v3_environment_custody()
    if not isinstance(custody, _V3EnvironmentCustody):
        raise V3PythonUnavailable("configured interpreter is invalid")
    raw = (
        custody.configured_python
        if custody.selected_environment
        else os.getenv(V3_CONFIGURED_PYTHON_ENV, "")
    )
    raw = str(raw or "").strip()
    if not raw:
        return None
    try:
        encoded = raw.encode("utf-8", errors="strict")
        selected = Path(raw)
    except (UnicodeError, ValueError):
        raise V3PythonUnavailable("configured interpreter is invalid") from None
    if (
        len(encoded) > MAX_CONFIGURED_PYTHON_PATH_BYTES
        or not selected.is_absolute()
        or raw.startswith("~")
        or "\x00" in raw
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw)
        or any(part in {"", ".", ".."} for part in selected.parts[1:])
        or selected.as_posix() != raw
    ):
        raise V3PythonUnavailable("configured interpreter is not absolute")
    try:
        validate_v3_path_custody({V3_CONFIGURED_PYTHON_ENV: raw})
    except V3PathCustodyError:
        raise V3PythonUnavailable("configured interpreter custody is invalid") from None
    return selected


def preferred_v3_python(
    *,
    current_executable: str | Path | None = None,
    project_root: Path | None = None,
    configured: Path | None = None,
    _configured_is_authoritative: bool = False,
    common_candidates: Iterable[Path] = COMMON_V3_PYTHONS,
    required_modules: tuple[str, ...] = V3_REQUIRED_LAUNCH_MODULES,
    preferred_modules: tuple[str, ...] = (),
    inspector: Callable[
        [Path, tuple[str, ...], tuple[str, ...]], V3PythonProbe | None
    ] = inspect_v3_python,
) -> Path | None:
    """Choose a compatible V3 interpreter using exact required/optional modules."""

    required = _safe_module_names(required_modules)
    preferred = _safe_module_names(preferred_modules)
    current = Path(current_executable or sys.executable)
    configured_path = (
        configured
        if configured is not None or _configured_is_authoritative
        else _configured_python()
    )
    if configured_path is not None:
        candidates = (configured_path,)
    else:
        local_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        candidates = (
            local_root / ".venv" / "bin" / "python3",
            current,
            *tuple(common_candidates),
        )

    unique: list[Path] = []
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if any(_same_executable(candidate, existing) for existing in unique):
            continue
        unique.append(candidate)

    fallback: V3PythonProbe | None = None
    for candidate in unique:
        if not _reasonable_python_executable(candidate):
            continue
        probe = inspector(candidate, required, preferred)
        if probe is None or not probe.launch_ready:
            continue
        if fallback is None:
            fallback = probe
        if probe.preferred_ready:
            return None if _same_executable(probe.path, current) else probe.path

    if fallback is None:
        raise V3PythonUnavailable("no compatible interpreter")
    return None if _same_executable(fallback.path, current) else fallback.path


def reexec_with_v3_python(script: Path) -> None:
    """Re-exec a user-invoked V3 launcher with its dependency-complete interpreter."""
    try:
        custody = assert_v3_environment_custody()
    except V3EnvironmentUnavailable:
        print(
            "Jarvis V3 could not read the selected environment safely.",
            file=sys.stderr,
        )
        print(
            "Correct or recreate the owner-only V3 environment using QUICKSTART.md, "
            "then retry.",
            file=sys.stderr,
        )
        raise SystemExit(78) from None
    except V3PathCustodyError as exc:
        reason_guidance = {
            "invalid_path": "the configured path is invalid",
            "overlaps_v2": "the configured path overlaps V2 rollback custody",
            "unresolvable_alias": "a link or alias in the configured path could not be resolved",
        }
        reason = reason_guidance[exc.reason_code]
        print(
            f"Jarvis V3 could not validate setting {exc.key} safely: {reason}.",
            file=sys.stderr,
        )
        print(
            f"Review only {exc.key}; correct its path, or remove that setting if it is optional. "
            "Then retry `setup check`. See QUICKSTART.md under Launcher path recovery.",
            file=sys.stderr,
        )
        raise SystemExit(78) from None
    except V3InheritedEnvironmentConflict as exc:
        names = ", ".join(exc.keys)
        print(
            "Jarvis V3 refused inherited state or credential overrides that differ from "
            f"the selected V3 environment file: {names}.",
            file=sys.stderr,
        )
        print(
            "Unset those variables or put the exact intended values in the selected V3 "
            "environment file, then retry.",
            file=sys.stderr,
        )
        raise SystemExit(78) from None
    try:
        configured = _configured_python(custody)
        candidate = preferred_v3_python(
            project_root=script.resolve().parent,
            configured=configured,
            _configured_is_authoritative=True,
        )
    except V3PythonUnavailable:
        print(
            "Jarvis V3 needs Python 3.11 or newer with the standard SSL and SQLite modules.",
            file=sys.stderr,
        )
        print(
            "Install a supported Python, create `.venv` in the Jarvis V3 folder, install "
            "requirements.txt there, then retry. You may instead set JARVIS_V3_PYTHON to "
            "an absolute compatible interpreter path.",
            file=sys.stderr,
        )
        raise SystemExit(78) from None
    inherited_guard = os.getenv(_V3_PYTHON_REEXEC_GUARD)
    if candidate is None and inherited_guard is None:
        return
    try:
        script_path = script.resolve(strict=True)
        guard = _reexec_guard(
            candidate if candidate is not None else Path(sys.executable),
            script_path,
        )
    except V3PythonUnavailable:
        _stop_reexec(
            "Jarvis V3 could not bind the interpreter restart safely.",
        )
    if candidate is None:
        if (
            _SHA256_RE.fullmatch(str(inherited_guard)) is None
            or inherited_guard != guard
        ):
            _stop_reexec("Jarvis V3 refused an invalid interpreter restart marker.")
        os.environ.pop(_V3_PYTHON_REEXEC_GUARD, None)
        return
    if inherited_guard is not None:
        valid_marker = _SHA256_RE.fullmatch(inherited_guard) is not None
        _stop_reexec(
            "Jarvis V3 stopped because the interpreter restart did not converge."
            if valid_marker and inherited_guard == guard
            else "Jarvis V3 refused an invalid interpreter restart marker.",
        )
    child_environment = dict(os.environ)
    child_environment[_V3_PYTHON_REEXEC_GUARD] = guard
    try:
        os.execve(
            str(candidate),
            [str(candidate), str(script_path), *sys.argv[1:]],
            child_environment,
        )
    except OSError:
        _stop_reexec("Jarvis V3 could not complete the interpreter restart safely.")
