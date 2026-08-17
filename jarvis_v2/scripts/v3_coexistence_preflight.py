"""Content-free, read-only V2/V3 coexistence preflight.

The preflight accepts two explicit custody roots: the V2 environment file and
the already-built V3 active-contract manifest.  It never discovers an
environment file, changes launchd state, starts a process, or renders local
paths, credentials, digests, state offsets, or raw exception text.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import pwd
import re
import secrets
import stat
from pathlib import Path
from typing import Callable

from jarvis_v2.automations.telegram_control import MAX_TRANSPORT_UPDATE_ID
from jarvis_v2.automations import telegram_offset_state
from jarvis_v2.env import read_env_values, validate_v3_path_custody
from jarvis_v2.scripts import v3_active_launchagent_contracts as contracts
from jarvis_v2.scripts import v3_recovery_receipt as recovery
from jarvis_v2.ui.status_config import DEFAULT_STATUS_PORT


SCHEMA_VERSION = 2
V2_TELEGRAM_LABEL = "com.jarvis-v2.telegram"
V2_DEFAULT_STATUS_PORT = 8765
MAX_STATE_BYTES = 4096
RELEVANT_ENV_KEYS = frozenset(
    {
        "JARVIS_DATA_DIR",
        "JARVIS_DB_PATH",
        "JARVIS_STATUS_PORT",
        "TELEGRAM_BOT_TOKEN",
        "JARVIS_OWNER_TELEGRAM",
        "JARVIS_TELEGRAM_STATE",
    }
)
TOKEN_RE = re.compile(r"[1-9][0-9]{4,19}:[A-Za-z0-9_-]{20,200}")
OWNER_RE = re.compile(r"-?[1-9][0-9]{0,19}")


def _empty_report() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "jarvis_v3_coexistence_preflight",
        "ready_for_parallel_activation": False,
        "serial_handoff_required": False,
        "checks": {
            "manifest": "unknown",
            "scheduler": "unknown",
            "v2_environment": "unknown",
            "v3_environment": "unknown",
            "persistent_paths": "unknown",
            "dashboard_port": "unknown",
            "telegram_token": "unknown",
            "telegram_owner": "unknown",
            "telegram_state_relation": "unknown",
            "v2_telegram_state": "unknown",
            "v3_telegram_state": "unknown",
            "v2_telegram_service": "unknown",
        },
        "blockers": [],
        "warnings": [],
        "values_included": False,
        "paths_included": False,
        "hashes_included": False,
        "mutates_state": False,
        "launchctl_control_performed": False,
    }


def _blocker(report: dict[str, object], code: str) -> None:
    blockers = report["blockers"]
    assert isinstance(blockers, list)
    blockers.append(code)


def _warning(report: dict[str, object], code: str) -> None:
    warnings = report["warnings"]
    assert isinstance(warnings, list)
    warnings.append(code)


def _checks(report: dict[str, object]) -> dict[str, str]:
    checks = report["checks"]
    assert isinstance(checks, dict)
    return checks


def _account_home() -> Path:
    account = pwd.getpwuid(os.geteuid())
    raw = getattr(account, "pw_dir", None)
    if type(raw) is not str:
        raise ValueError("account")
    home = Path(raw)
    if not home.is_absolute() or not home.name:
        raise ValueError("account")
    return home


def _expand_account_path(raw: str, home: Path) -> Path:
    value = raw.strip()
    if value == "~":
        path = home
    elif value.startswith("~/"):
        path = home / value[2:]
    elif value.startswith("~"):
        raise ValueError("path")
    else:
        path = Path(value)
    if not path.is_absolute() or not path.name:
        raise ValueError("path")
    return path


def _effective_paths(values: dict[str, str], *, generation: str, home: Path) -> dict[str, Path]:
    default_root = home / (".jarvis_v2" if generation == "v2" else ".jarvis_v3")
    data = _expand_account_path(values.get("JARVIS_DATA_DIR", os.fspath(default_root)), home)
    database = _expand_account_path(
        values.get("JARVIS_DB_PATH", os.fspath(data / "jarvis.sqlite")), home
    )
    state = _expand_account_path(
        values.get("JARVIS_TELEGRAM_STATE", os.fspath(default_root / "telegram_control.json")),
        home,
    )
    return {"data": data, "database": database, "telegram_state": state}


def _resolved(path: Path) -> Path:
    return path.resolve(strict=False)


def _same_inode(left: Path, right: Path) -> bool:
    try:
        left_info = os.stat(left, follow_symlinks=False)
        right_info = os.stat(right, follow_symlinks=False)
    except OSError:
        return False
    return (left_info.st_dev, left_info.st_ino) == (right_info.st_dev, right_info.st_ino)


def _paths_overlap(left: Path, right: Path) -> bool:
    resolved_left = _resolved(left)
    resolved_right = _resolved(right)
    return (
        resolved_left == resolved_right
        or resolved_left in resolved_right.parents
        or resolved_right in resolved_left.parents
        or _same_inode(left, right)
        or _same_inode(resolved_left, resolved_right)
    )


def _port(values: dict[str, str], default: int) -> int | None:
    raw = values.get("JARVIS_STATUS_PORT")
    if raw is None:
        return default
    try:
        value = int(raw.strip(), 10)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 65535 else None


def _credential_status(value: str | None, pattern: re.Pattern[str]) -> str:
    if value is None or not value.strip():
        return "missing"
    candidate = value.strip()
    if "\x00" in candidate or "\n" in candidate or "\r" in candidate:
        return "invalid"
    return "valid" if pattern.fullmatch(candidate) is not None else "invalid"


def _same_secret(left: str, right: str) -> bool:
    return secrets.compare_digest(left.strip().encode("utf-8"), right.strip().encode("utf-8"))


def _state_stat_token(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class _StateFileMissing(RuntimeError):
    """The final state entry alone is absent under a verified parent."""


def _read_stable_state(path: Path, *, allow_v2_legacy_readable: bool) -> tuple[bytes, str]:
    if not path.is_absolute() or not path.name:
        raise ValueError("state")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    parent_fd: int | None = None
    descriptor: int | None = None
    try:
        parent_fd = os.open("/", directory_flags)
        for component in path.parent.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("state")
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        opened_parent = os.fstat(parent_fd)
        named_parent = os.stat(path.parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened_parent.st_mode)
            or stat.S_ISLNK(named_parent.st_mode)
            or _state_stat_token(opened_parent) != _state_stat_token(named_parent)
            or (
                not allow_v2_legacy_readable
                and (
                    opened_parent.st_uid != os.geteuid()
                    or stat.S_IMODE(opened_parent.st_mode) != 0o700
                )
            )
        ):
            raise ValueError("state")
        try:
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            named_after = os.stat(path.parent, follow_symlinks=False)
            if _state_stat_token(opened_parent) != _state_stat_token(named_after):
                raise ValueError("state") from None
            raise _StateFileMissing from None
        mode = stat.S_IMODE(named.st_mode)
        strict_owner_only = mode == 0o600
        legacy_readable = allow_v2_legacy_readable and mode == 0o644
        if (
            not stat.S_ISREG(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or named.st_uid != os.geteuid()
            or named.st_nlink != 1
            or named.st_size < 1
            or named.st_size > MAX_STATE_BYTES
            or mode & 0o7000
            or not (strict_owner_only or legacy_readable)
        ):
            raise ValueError("state")
        descriptor = os.open(path.name, file_flags, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if _state_stat_token(named) != _state_stat_token(opened):
            raise ValueError("state")
        chunks: list[bytes] = []
        remaining = MAX_STATE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_STATE_BYTES:
            raise ValueError("state")
        after = os.fstat(descriptor)
        named_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _state_stat_token(opened) != _state_stat_token(after)
            or _state_stat_token(after) != _state_stat_token(named_after)
            or stat.S_ISLNK(named_after.st_mode)
        ):
            raise ValueError("state")
        closing_parent = os.stat(path.parent, follow_symlinks=False)
        if _state_stat_token(opened_parent) != _state_stat_token(closing_parent):
            raise ValueError("state")
        return raw, "valid_legacy_readable" if legacy_readable else "valid"
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def _state_status(
    path: Path,
    *,
    allow_v2_legacy_readable: bool,
    bot_token: str | None = None,
    owner_id: str | None = None,
) -> str:
    try:
        raw, custody = _read_stable_state(
            path, allow_v2_legacy_readable=allow_v2_legacy_readable
        )
        duplicate = False

        def pairs(pairs_value: list[tuple[str, object]]) -> dict[str, object]:
            nonlocal duplicate
            result: dict[str, object] = {}
            for key, value in pairs_value:
                if key in result:
                    duplicate = True
                result[key] = value
            return result

        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
        )
        if duplicate or type(value) is not dict:
            return "invalid"
        if allow_v2_legacy_readable:
            offset = value.get("offset") if set(value) == {"offset"} else None
            if type(offset) is not int or offset < 0 or offset > MAX_TRANSPORT_UPDATE_ID:
                return "invalid"
            return custody
        if (
            set(value) != {"identity_sha256", "offset", "revision", "version"}
            or raw != telegram_offset_state._canonical(value)
            or value.get("version") != telegram_offset_state.STATE_VERSION
            or type(value.get("offset")) is not int
            or value["offset"] < 0
            or value["offset"] > telegram_offset_state.MAX_OFFSET
            or type(value.get("revision")) is not int
            or value["revision"] < 1
            or value["revision"] > telegram_offset_state.MAX_OFFSET
            or type(value.get("identity_sha256")) is not str
            or telegram_offset_state.IDENTITY_RE.fullmatch(value["identity_sha256"]) is None
            or type(bot_token) is not str
            or type(owner_id) is not str
        ):
            return "invalid"
        expected_identity = telegram_offset_state._identity(bot_token, owner_id)
        if not secrets.compare_digest(value["identity_sha256"], expected_identity):
            return "invalid"
        return custody
    except _StateFileMissing:
        return "fresh"
    except Exception:
        return "invalid"


def _v3_env_path(evidence: recovery.ContractEvidence) -> Path:
    if not evidence.artifacts:
        raise ValueError("manifest")
    payload = plistlib.loads(evidence.artifacts[0][1])
    environment = payload.get("EnvironmentVariables")
    if type(environment) is not dict:
        raise ValueError("manifest")
    value = environment.get(contracts.ENV_FILE_KEY)
    if type(value) is not str or not Path(value).is_absolute():
        raise ValueError("manifest")
    return Path(value)


def observe_v2_telegram_loaded() -> bool | None:
    """Observe one exact V2 label without controlling launchd."""

    try:
        result = recovery._run_observer(
            ["/bin/launchctl", "print", f"gui/{os.geteuid()}/{V2_TELEGRAM_LABEL}"],
            "v2_telegram_observation_unknown",
            timeout=5.0,
            max_output_bytes=4096,
        )
    except Exception:
        return None
    combined = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode == 0:
        return True
    if combined == "Could not find service":
        return False
    pattern = (
        r'(?:Bad request\.\s*)?Could not find service "com\.jarvis-v2\.telegram" '
        r'in domain for user gui:\s*[0-9]+\.?'
    )
    return False if re.fullmatch(pattern, combined) is not None else None


def inspect_coexistence(
    v2_env: Path | str,
    active_contract_manifest: Path | str,
    *,
    launchd_reader: Callable[[], bool | None] = observe_v2_telegram_loaded,
) -> dict[str, object]:
    report = _empty_report()
    checks = _checks(report)
    try:
        v2_env_path = Path(v2_env)
        manifest_path = Path(active_contract_manifest)
        if not v2_env_path.is_absolute() or not manifest_path.is_absolute():
            raise ValueError("explicit_path")
        evidence = recovery.validate_active_contract_manifest(manifest_path)
        checks["manifest"] = "valid"
        checks["scheduler"] = "disabled"
    except Exception:
        checks["manifest"] = "invalid"
        _blocker(report, "manifest_invalid")
        return report

    try:
        v2_values = read_env_values(
            v2_env_path,
            reject_duplicate_keys=RELEVANT_ENV_KEYS,
            require_owner_only=True,
        )
        checks["v2_environment"] = "valid"
    except Exception:
        checks["v2_environment"] = "invalid"
        _blocker(report, "v2_environment_invalid")
        return report

    try:
        v3_values = read_env_values(
            _v3_env_path(evidence),
            reject_duplicate_keys=RELEVANT_ENV_KEYS,
            require_owner_only=True,
        )
        checks["v3_environment"] = "valid"
    except Exception:
        checks["v3_environment"] = "invalid"
        _blocker(report, "v3_environment_invalid")
        return report

    v2_paths: dict[str, Path] | None = None
    v3_paths: dict[str, Path] | None = None
    try:
        home = _account_home()
        v2_paths = _effective_paths(v2_values, generation="v2", home=home)
        v3_paths = _effective_paths(v3_values, generation="v3", home=home)
        validate_v3_path_custody(
            {
                "JARVIS_DATA_DIR": os.fspath(v3_paths["data"]),
                "JARVIS_DB_PATH": os.fspath(v3_paths["database"]),
                "JARVIS_TELEGRAM_STATE": os.fspath(v3_paths["telegram_state"]),
            }
        )
        overlap = any(
            _paths_overlap(v2_path, v3_path)
            for v2_path in v2_paths.values()
            for v3_path in v3_paths.values()
        )
    except Exception:
        checks["persistent_paths"] = "invalid"
        _blocker(report, "persistent_path_invalid")
        overlap = True
    if overlap and checks["persistent_paths"] != "invalid":
        checks["persistent_paths"] = "overlap"
        _blocker(report, "persistent_path_overlap")
    elif not overlap:
        checks["persistent_paths"] = "distinct"

    v2_port = _port(v2_values, V2_DEFAULT_STATUS_PORT)
    v3_port = _port(v3_values, DEFAULT_STATUS_PORT)
    if v2_port is None or v3_port is None:
        checks["dashboard_port"] = "invalid"
        _blocker(report, "dashboard_port_invalid")
    elif v2_port == v3_port:
        checks["dashboard_port"] = "same"
        _blocker(report, "dashboard_port_collision")
    else:
        checks["dashboard_port"] = "distinct"

    token_statuses = (
        _credential_status(v2_values.get("TELEGRAM_BOT_TOKEN"), TOKEN_RE),
        _credential_status(v3_values.get("TELEGRAM_BOT_TOKEN"), TOKEN_RE),
    )
    if "missing" in token_statuses:
        checks["telegram_token"] = "missing"
        _blocker(report, "telegram_token_missing")
    elif "invalid" in token_statuses:
        checks["telegram_token"] = "invalid"
        _blocker(report, "telegram_token_invalid")
    else:
        assert v2_values.get("TELEGRAM_BOT_TOKEN") is not None
        assert v3_values.get("TELEGRAM_BOT_TOKEN") is not None
        checks["telegram_token"] = (
            "same"
            if _same_secret(v2_values["TELEGRAM_BOT_TOKEN"], v3_values["TELEGRAM_BOT_TOKEN"])
            else "distinct"
        )

    owner_statuses = (
        _credential_status(v2_values.get("JARVIS_OWNER_TELEGRAM"), OWNER_RE),
        _credential_status(v3_values.get("JARVIS_OWNER_TELEGRAM"), OWNER_RE),
    )
    if "missing" in owner_statuses:
        checks["telegram_owner"] = "missing"
        _blocker(report, "telegram_owner_missing")
    elif "invalid" in owner_statuses:
        checks["telegram_owner"] = "invalid"
        _blocker(report, "telegram_owner_invalid")
    else:
        assert v2_values.get("JARVIS_OWNER_TELEGRAM") is not None
        assert v3_values.get("JARVIS_OWNER_TELEGRAM") is not None
        checks["telegram_owner"] = (
            "same"
            if _same_secret(v2_values["JARVIS_OWNER_TELEGRAM"], v3_values["JARVIS_OWNER_TELEGRAM"])
            else "distinct"
        )

    if v2_paths is None or v3_paths is None:
        state_overlap = False
        checks["telegram_state_relation"] = "unknown"
        checks["v2_telegram_state"] = "invalid"
        checks["v3_telegram_state"] = "invalid"
        _blocker(report, "telegram_state_path_invalid")
    else:
        try:
            state_overlap = _paths_overlap(
                v2_paths["telegram_state"], v3_paths["telegram_state"]
            )
            checks["telegram_state_relation"] = "overlap" if state_overlap else "distinct"
        except Exception:
            state_overlap = False
            checks["telegram_state_relation"] = "unknown"
            _blocker(report, "telegram_state_path_invalid")
    if state_overlap:
        _blocker(report, "telegram_state_overlap")
    if v2_paths is not None and v3_paths is not None:
        v2_state = _state_status(
            v2_paths["telegram_state"], allow_v2_legacy_readable=True
        )
        v3_state = _state_status(
            v3_paths["telegram_state"],
            allow_v2_legacy_readable=False,
            bot_token=v3_values.get("TELEGRAM_BOT_TOKEN"),
            owner_id=v3_values.get("JARVIS_OWNER_TELEGRAM"),
        )
        checks["v2_telegram_state"] = v2_state
        checks["v3_telegram_state"] = v3_state
        if v2_state == "invalid":
            _blocker(report, "v2_telegram_state_invalid")
        elif v2_state == "valid_legacy_readable":
            _warning(report, "v2_telegram_state_legacy_permissions")
        if v3_state == "invalid":
            _blocker(report, "v3_telegram_state_invalid")

    try:
        loaded = launchd_reader()
    except Exception:
        loaded = None
    if type(loaded) is not bool:
        checks["v2_telegram_service"] = "unknown"
        _blocker(report, "v2_telegram_load_state_unknown")
    else:
        checks["v2_telegram_service"] = "loaded" if loaded else "unloaded"

    if checks["telegram_token"] == "same" and type(loaded) is bool:
        if loaded:
            _blocker(report, "shared_telegram_transport_active")
        else:
            report["serial_handoff_required"] = True
            _blocker(report, "shared_telegram_transport_requires_serial_handoff")

    blockers = report["blockers"]
    warnings = report["warnings"]
    assert isinstance(blockers, list) and isinstance(warnings, list)
    report["blockers"] = sorted(set(blockers))
    report["warnings"] = sorted(set(warnings))
    report["ready_for_parallel_activation"] = not report["blockers"]
    return report


class _BoundedParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise ValueError("arguments_invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _BoundedParser(description="Read-only V2/V3 coexistence preflight.")
    parser.add_argument("--v2-env", required=True, type=Path)
    parser.add_argument("--active-contract-manifest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except (TypeError, ValueError):
        report = _empty_report()
        _blocker(report, "arguments_invalid")
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 2
    report = inspect_coexistence(args.v2_env, args.active_contract_manifest)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["ready_for_parallel_activation"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
