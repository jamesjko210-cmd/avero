"""Minimal .env loader (no external dependency).

Reads KEY=VALUE lines from the project .env into os.environ without
overwriting variables already set in the real environment.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OWNER_ONLY_ENV_KEYS = frozenset({"JARVIS_STATUS_AUTH_TOKEN"})
_V3_PERSISTENT_PATH_KEYS = frozenset(
    {
        "JARVIS_CACHE_DIR",
        "JARVIS_DATA_DIR",
        "JARVIS_DB_PATH",
        "JARVIS_GOOGLE_CREDS",
        "JARVIS_GOOGLE_READONLY_TOKEN",
        "JARVIS_GOOGLE_TOKEN",
        "JARVIS_OBSIDIAN_VAULT",
        "JARVIS_REMINDERS_FILE",
        "JARVIS_STORAGE_FALLBACK_DIR",
        "JARVIS_TELEGRAM_STATE",
        "JARVIS_V3_ENV",
        "JARVIS_V3_IMESSAGE_STATE",
        "JARVIS_V3_PYTHON",
        "OBSIDIAN_VAULT_PATH",
    }
)
_V2_PATH_COMPONENTS = frozenset(
    {
        ".jarvis_v2",
        ".jarvis_" + "v2_runtime",
        ".jarvis_" + "v2_durable",
        "jarvis-v2",
    }
)
V3_PATH_CUSTODY_REASON_CODES = frozenset(
    {"invalid_path", "overlaps_v2", "unresolvable_alias"}
)
_V3_PATH_CUSTODY_KEYS = _V3_PERSISTENT_PATH_KEYS | frozenset({"JARVIS_WATCHED_DIRS"})
_V3_PATH_CUSTODY_UNKNOWN_KEY = "V3_PERSISTENT_PATH"
_MAX_ENV_FILE_BYTES = 1024 * 1024


class V3PathCustodyError(RuntimeError):
    """A path-free V3 custody refusal with allowlisted operator diagnostics."""

    def __init__(self, key: str, reason_code: str) -> None:
        self.key = key if key in _V3_PATH_CUSTODY_KEYS else _V3_PATH_CUSTODY_UNKNOWN_KEY
        self.reason_code = (
            reason_code
            if reason_code in V3_PATH_CUSTODY_REASON_CODES
            else "invalid_path"
        )
        super().__init__(
            f"V3 persistent path custody refused for {self.key}: {self.reason_code}"
        )


def _normalize_macos_system_alias(path: Path) -> Path:
    """Normalize only Apple's fixed root aliases used by ordinary local paths."""

    if path.is_absolute() and len(path.parts) > 1:
        first = path.parts[1]
        if first in {"etc", "tmp", "var"}:
            alias = Path("/") / first
            expected = Path("/private") / first
            try:
                if alias.is_symlink() and Path(os.path.realpath(alias)) == expected:
                    return expected.joinpath(*path.parts[2:])
            except OSError:
                pass
    return path


def _inspect_v3_path_components(path: Path, *, key: str) -> None:
    """Reject existing aliases while allowing an absent future suffix.

    The walk anchors every existing component to a directory descriptor and
    never follows symlinks.  A first missing component ends the inspection:
    its nearest existing parent has been validated and no later component can
    exist below that spelling yet.  Existing regular-file targets must have a
    single link so a differently named V2 file cannot be reused as V3 state.

    This is an honest-operator custody check, not complete protection against
    a same-UID process that deliberately races path mutation after validation.
    Runtime writers must retain their own atomic/no-follow boundaries.
    """

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    components = path.parts[1:] if path.is_absolute() else path.parts
    if any(component == ".." for component in components):
        raise V3PathCustodyError(key, "invalid_path")
    descriptor: int | None = None
    try:
        descriptor = os.open("/" if path.is_absolute() else ".", directory_flags)
        for index, component in enumerate(components):
            if component in {"", "."}:
                continue
            try:
                named = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return
            except OSError:
                raise V3PathCustodyError(key, "unresolvable_alias") from None
            if stat.S_ISLNK(named.st_mode):
                raise V3PathCustodyError(key, "unresolvable_alias")
            final = index == len(components) - 1
            if final:
                if stat.S_ISREG(named.st_mode) and named.st_nlink != 1:
                    raise V3PathCustodyError(key, "unresolvable_alias")
                if not (stat.S_ISREG(named.st_mode) or stat.S_ISDIR(named.st_mode)):
                    raise V3PathCustodyError(key, "invalid_path")
                return
            if not stat.S_ISDIR(named.st_mode):
                raise V3PathCustodyError(key, "invalid_path")
            try:
                next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            except OSError:
                raise V3PathCustodyError(key, "unresolvable_alias") from None
            opened = os.fstat(next_descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not _same_entry(named, opened)
            ):
                os.close(next_descriptor)
                raise V3PathCustodyError(key, "unresolvable_alias")
            os.close(descriptor)
            descriptor = next_descriptor
    except V3PathCustodyError:
        raise
    except OSError:
        raise V3PathCustodyError(key, "unresolvable_alias") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _has_v2_path_component(path: Path, *, key: str) -> bool:
    candidates = [path]
    try:
        candidates.append(path.resolve(strict=False))
    except ValueError:
        raise V3PathCustodyError(key, "invalid_path") from None
    except (OSError, RuntimeError):
        # An alias that cannot be resolved cannot prove that it stays outside
        # V2 custody.  Fail closed with a bounded domain exception instead of
        # accepting the raw spelling or leaking pathlib's local-path detail.
        raise V3PathCustodyError(key, "unresolvable_alias") from None
    return any(
        component.casefold() in _V2_PATH_COMPONENTS
        for candidate in candidates
        for component in candidate.parts
    )


def validate_v3_path_custody(values: dict[str, str]) -> None:
    """Reject V3 state/token/cache paths that directly or indirectly target V2."""

    configured_paths: list[tuple[str, str]] = []
    for key in sorted(_V3_PERSISTENT_PATH_KEYS):
        value = str(values.get(key) or "").strip()
        if value:
            configured_paths.append((key, value))
    watched = str(values.get("JARVIS_WATCHED_DIRS") or "").strip()
    if watched:
        configured_paths.extend(
            ("JARVIS_WATCHED_DIRS", item.strip())
            for item in watched.split(os.pathsep)
            if item.strip()
        )
    for key, raw in configured_paths:
        try:
            path = _normalize_macos_system_alias(Path(raw).expanduser())
        except (OSError, RuntimeError, ValueError):
            raise V3PathCustodyError(key, "invalid_path") from None
        if _has_v2_path_component(path, key=key):
            raise V3PathCustodyError(key, "overlaps_v2")
        # The selected environment is immediately opened by the stronger
        # stable-file reader below, which already rejects symlink components,
        # multi-link files, replacement, and partial reads while preserving its
        # established PermissionError guidance. Other persistent targets need
        # this standalone custody walk before their later consumers open them.
        if key not in {"JARVIS_V3_ENV", "JARVIS_V3_PYTHON"}:
            _inspect_v3_path_components(path, key=key)


def _valid_key(key: str) -> bool:
    return bool(key) and (key[0].isalpha() or key[0] == "_") and all(char.isalnum() or char == "_" for char in key)


def _validate_owner_only_env_stat(env_stat: os.stat_result) -> None:
    file_mode = stat.S_IMODE(env_stat.st_mode)
    if (
        env_stat.st_uid != os.geteuid()
        or env_stat.st_nlink != 1
        or file_mode & 0o077
        or not file_mode & stat.S_IRUSR
    ):
        raise PermissionError(
            "Environment file containing dashboard authentication must be an owner-only, "
            "single-link regular file; set its permissions to 0600."
        )


def _same_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _read_stable_env_file(
    env_path: Path,
    *,
    missing_ok: bool,
    require_owner_only: bool,
) -> tuple[str, os.stat_result | None]:
    """Read an env file through one component-wise, no-follow custody chain."""

    # macOS exposes these fixed root aliases as compatibility paths. Normalize
    # only those OS-owned aliases before the no-follow walk; arbitrary parent
    # symlinks remain forbidden.
    env_path = _normalize_macos_system_alias(env_path)

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
        if env_path.is_absolute():
            parent_fd = os.open("/", directory_flags)
            components = env_path.parent.parts[1:]
        else:
            parent_fd = os.open(".", directory_flags)
            components = env_path.parent.parts
        for component in components:
            if component in {"", "."}:
                continue
            if component == "..":
                raise PermissionError("Environment file parent traversal is not allowed.")
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd

        named_before = os.stat(
            env_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(named_before.st_mode):
            raise PermissionError("Environment file symlinks are not allowed.")
        if not stat.S_ISREG(named_before.st_mode):
            raise OSError("Environment path is not a regular file")
        descriptor = os.open(env_path.name, file_flags, dir_fd=parent_fd)
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or not _same_entry(named_before, opened_before)
        ):
            raise PermissionError("Environment file changed during validation.")
        if require_owner_only:
            _validate_owner_only_env_stat(opened_before)
        if opened_before.st_size > _MAX_ENV_FILE_BYTES:
            raise OSError("Environment file exceeds the bounded size limit")

        remaining = _MAX_ENV_FILE_BYTES + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        opened_after = os.fstat(descriptor)
        named_after = os.stat(
            env_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        stable_fields_before = (
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_size,
            opened_before.st_mtime_ns,
            opened_before.st_nlink,
        )
        stable_fields_after = (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_size,
            opened_after.st_mtime_ns,
            opened_after.st_nlink,
        )
        if (
            len(raw) > _MAX_ENV_FILE_BYTES
            or stable_fields_before != stable_fields_after
            or not _same_entry(opened_after, named_after)
        ):
            raise PermissionError("Environment file changed during validation.")
        return raw.decode("utf-8"), opened_after
    except FileNotFoundError:
        if missing_ok:
            return "", None
        raise FileNotFoundError("Explicit environment file does not exist") from None
    except PermissionError as exc:
        if str(exc).startswith("Environment file "):
            raise
        raise PermissionError("Environment file custody could not be verified.") from None
    except OSError as exc:
        if getattr(exc, "errno", None) in {20, 40}:
            raise PermissionError("Environment file path aliases are not allowed.") from None
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def read_env_values(
    env_path: Path,
    *,
    missing_ok: bool = False,
    reject_duplicate_keys: frozenset[str] = frozenset(),
    require_owner_only: bool = False,
) -> dict[str, str]:
    """Read one env file with the same custody and parsing rules as ``load_env``."""

    env_path = env_path.expanduser()
    contents, env_stat = _read_stable_env_file(
        env_path,
        missing_ok=missing_ok,
        require_owner_only=require_owner_only,
    )
    if env_stat is None:
        return {}

    parsed: dict[str, str] = {}
    file_keys: set[str] = set()
    duplicate_keys: set[str] = set()
    for raw in contents.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        quoted = (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'"))
        if not quoted and " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        value = value.strip('"').strip("'")
        if _valid_key(key):
            if key in file_keys:
                duplicate_keys.add(key)
            file_keys.add(key)
        if _valid_key(key) and key not in parsed:
            parsed[key] = value
    rejected_duplicates = duplicate_keys & reject_duplicate_keys
    if rejected_duplicates:
        names = ", ".join(sorted(rejected_duplicates))
        raise ValueError(f"Duplicate environment entries are not allowed: {names}")
    if file_keys & _OWNER_ONLY_ENV_KEYS:
        _validate_owner_only_env_stat(env_stat)
    return parsed


def load_env(path: Path | None = None) -> None:
    configured_path = os.getenv("JARVIS_V3_ENV", "").strip()
    if configured_path:
        validate_v3_path_custody({"JARVIS_V3_ENV": configured_path})
    explicit_path = path is not None or bool(configured_path)
    env_path = (path or Path(configured_path or _PROJECT_ROOT / ".env")).expanduser()
    values = read_env_values(
        env_path,
        missing_ok=not explicit_path,
        # A nonblank JARVIS_V3_ENV is the custody boundary for V3 state and
        # credentials.  Require owner-only custody even when the selected file
        # is empty or does not contain the dashboard-auth key.  Keep the
        # historical project .env fallback compatible when no V3 file is
        # explicitly selected.
        require_owner_only=bool(configured_path),
    )
    additions = {key: value for key, value in values.items() if key not in os.environ}
    effective = dict(os.environ)
    effective.update(additions)
    validate_v3_path_custody(effective)
    os.environ.update(additions)
