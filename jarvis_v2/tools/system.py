from __future__ import annotations

import platform
import importlib.util
import os
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any
from math import isfinite

from jarvis_v2.agent.model_provider import (
    OLLAMA_UNVERIFIED_PERSONAL_CONTEXT_ENV,
    ollama_local_only_policy,
    resolve_ollama_destination,
)
from jarvis_v2.agent.failure_guidance import (
    declare_failure_guidance,
    declare_outcome_unknown_failure,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import ApprovalArgumentResolution, ToolResult
from jarvis_v2.config import (
    DEFAULT_CHAT_MAX_HISTORY_MESSAGES,
    DEFAULT_CHAT_MAX_REPLY_TOKENS,
    DEFAULT_OPENAI_CHAT_MODEL,
    DEFAULT_OPENAI_PLANNER_MODEL,
    MAX_CHAT_MAX_HISTORY_MESSAGES,
    MAX_CHAT_MAX_REPLY_TOKENS,
    MAX_OLLAMA_CHAT_TIMEOUT_SECONDS,
    MAX_OLLAMA_MODEL_TIMEOUT_SECONDS,
    MAX_OPENAI_CHAT_TIMEOUT_SECONDS,
    MAX_OPENAI_MODEL_TIMEOUT_SECONDS,
    MIN_CHAT_MAX_HISTORY_MESSAGES,
    MIN_CHAT_MAX_REPLY_TOKENS,
)
from jarvis_v2.tools.calendar_connector import _creds_file as _google_creds_file
from jarvis_v2.tools.calendar_connector import _readonly_token_file as _google_readonly_token_file
from jarvis_v2.tools.calendar_connector import _token_file as _google_token_file
from jarvis_v2.ui.status_config import (
    DEFAULT_STATUS_HOST,
    DEFAULT_STATUS_PORT,
    StatusHostConfig,
    StatusPortConfig,
    status_host_is_loopback,
)
from jarvis_v2.v3_commands import (
    V3_CALENDAR_AUTH_READONLY_COMMAND,
    V3_CALENDAR_AUTH_TERMINAL_LOCATION,
)


MAX_CLIPBOARD_CHARS = 20000
MAX_CLIPBOARD_WRITE_CHARS = 20000
MAX_APP_NAME_CHARS = 120
DEFAULT_SMOKE_MODULE_TIMEOUT_SECONDS = 180
MIN_SMOKE_MODULE_TIMEOUT_SECONDS = 5
MAX_SMOKE_MODULE_TIMEOUT_SECONDS = 3600
DEFAULT_MODEL_TIMEOUT_SECONDS = 2.5
MIN_MODEL_TIMEOUT_SECONDS = 0.5
DEFAULT_CHAT_TIMEOUT_SECONDS = 20.0
MIN_CHAT_TIMEOUT_SECONDS = 1.0
DEFAULT_OBSIDIAN_ROOT = "Jarvis"
DEFAULT_MODEL_ALIAS = "llama3.1"
MODEL_ALIAS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
MODEL_PROVIDERS = {"ollama", "openai"}
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
SECRET_ENV_KEYS = {
    "TELEGRAM_BOT_TOKEN",
    "GMAIL_ADDRESS",
    "GMAIL_APP_PASSWORD",
    "OPENAI_API_KEY",
    "JARVIS_STATUS_AUTH_TOKEN",
}
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}
OPERATOR_LIMIT_RULE = (
    "the operator's explicit stop times, work windows, pause commands, and newer instructions "
    "override system actions, clipboard actions, app launches, and priority goals."
)
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
SYSTEM_BOUNDARY = (
    "\n\nSystem boundary:\n"
    f"- {OPERATOR_LIMIT_RULE}\n"
    "- Clipboard reads, private app/window context, shell/code, computer control, destructive changes, "
    "and outside-world side effects remain approval-gated."
)
SYSTEM_READ_RECOVERY_ACTION = (
    "Run `setup check`, correct the reported macOS permission or configuration issue, then retry the read."
)
SYSTEM_INPUT_RECOVERY_ACTION = (
    "Correct the reported input, then submit a new request through the normal approval and policy checks."
)
APPLICATION_OUTCOME_UNKNOWN_ACTION = (
    "The application-launch outcome is unknown. Check whether the app opened before issuing a new request; "
    "do not retry automatically."
)
GOOGLE_CALENDAR_READONLY_AUTH_COMMAND = V3_CALENDAR_AUTH_READONLY_COMMAND
VOLUME_OUTCOME_UNKNOWN_ACTION = (
    "The volume-change outcome is unknown. Check the current volume before issuing a new request; "
    "do not retry automatically."
)
CLIPBOARD_OUTCOME_UNKNOWN_ACTION = (
    "The clipboard-write outcome is unknown. Check the clipboard before issuing a new request; "
    "do not retry automatically."
)


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _raw_int_metadata(value: Any, *, key: str, sanitized: int) -> dict[str, Any]:
    if value is None:
        return {key: sanitized}
    if isinstance(value, bool):
        return {key: sanitized, f"raw_{key}": str(value)}
    try:
        int(value)
    except (TypeError, ValueError):
        return {key: sanitized, f"raw_{key}": _short_raw(value, limit=80)}
    return {key: sanitized}


def _smoke_timeout_status() -> dict[str, Any]:
    raw = os.getenv("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "effective_seconds": float(DEFAULT_SMOKE_MODULE_TIMEOUT_SECONDS),
        }
    try:
        value = float(raw)
    except ValueError:
        return {
            "configured": True,
            "valid": False,
            "source": "env-invalid",
            "effective_seconds": float(DEFAULT_SMOKE_MODULE_TIMEOUT_SECONDS),
        }
    valid = isfinite(value)
    effective = float(DEFAULT_SMOKE_MODULE_TIMEOUT_SECONDS)
    source = "env-invalid"
    if valid:
        effective = min(float(MAX_SMOKE_MODULE_TIMEOUT_SECONDS), max(float(MIN_SMOKE_MODULE_TIMEOUT_SECONDS), value))
        source = "env-clamped" if effective != value else "env"
    return {
        "configured": True,
        "valid": valid,
        "source": source,
        "effective_seconds": effective,
    }


def _runtime_timeout_status(
    env_key: str, default: float, minimum: float, maximum: float
) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "effective_seconds": float(default),
            "minimum_seconds": float(minimum),
            "maximum_seconds": float(maximum),
        }
    try:
        value = float(raw)
    except ValueError:
        return {
            "configured": True,
            "valid": False,
            "source": "env-invalid",
            "effective_seconds": float(default),
            "minimum_seconds": float(minimum),
            "maximum_seconds": float(maximum),
        }
    valid = isfinite(value)
    effective = float(default)
    source = "env-invalid"
    if valid:
        effective = min(float(maximum), max(float(minimum), value))
        source = "env-clamped" if effective != value else "env"
    return {
        "configured": True,
        "valid": valid,
        "source": source,
        "effective_seconds": effective,
        "minimum_seconds": float(minimum),
        "maximum_seconds": float(maximum),
    }


def _runtime_timeout_validation_detail(status: dict[str, Any], default: float) -> str:
    if not status["valid"]:
        return f"invalid; using default {default:g}s"
    if status["source"] == "env-clamped":
        return (
            f"clamped to {status['effective_seconds']:g}s "
            f"(maximum {status['maximum_seconds']:g}s)"
        )
    return "ok"


def _runtime_int_status(env_key: str, default: int, minimum: int, maximum: int) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "effective_value": int(default),
        }
    try:
        value = int(raw)
    except ValueError:
        return {
            "configured": True,
            "valid": False,
            "source": "env-invalid",
            "effective_value": int(default),
        }
    effective = max(int(minimum), min(int(maximum), value))
    return {
        "configured": True,
        "valid": True,
        "source": "env-clamped" if effective != value else "env",
        "effective_value": effective,
    }


def _runtime_int_validation_detail(status: dict[str, Any], default: int, maximum: int) -> str:
    if not status["valid"]:
        return f"invalid; using default {default}"
    if status["source"] == "env-clamped" and status["effective_value"] == maximum:
        return f"ok; clamped to maximum {maximum}"
    return "ok"


def _model_planner_toggle_status() -> dict[str, Any]:
    raw = os.getenv("JARVIS_USE_MODEL_PLANNER", "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "effective_enabled": True,
            "fallback_enabled": True,
        }
    normalized = raw.lower()
    if normalized in TRUE_ENV_VALUES:
        return {
            "configured": True,
            "valid": True,
            "source": "env",
            "effective_enabled": True,
            "fallback_enabled": True,
        }
    if normalized in FALSE_ENV_VALUES:
        return {
            "configured": True,
            "valid": True,
            "source": "env",
            "effective_enabled": False,
            "fallback_enabled": True,
        }
    return {
        "configured": True,
        "valid": False,
        "source": "env-invalid",
        "effective_enabled": True,
        "fallback_enabled": True,
    }


def _remote_compaction_toggle_status() -> dict[str, Any]:
    raw = os.getenv("JARVIS_ALLOW_REMOTE_COMPACTION", "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "effective_enabled": False,
            "fallback_enabled": False,
        }
    normalized = raw.lower()
    if normalized in TRUE_ENV_VALUES:
        return {
            "configured": True,
            "valid": True,
            "source": "env",
            "effective_enabled": True,
            "fallback_enabled": False,
        }
    if normalized in FALSE_ENV_VALUES:
        return {
            "configured": True,
            "valid": True,
            "source": "env",
            "effective_enabled": False,
            "fallback_enabled": False,
        }
    return {
        "configured": True,
        "valid": False,
        "source": "env-invalid",
        "effective_enabled": False,
        "fallback_enabled": False,
    }


def _remote_personal_context_toggle_status() -> dict[str, Any]:
    raw = os.getenv("JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT", "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "effective_enabled": False,
            "fallback_enabled": False,
        }
    normalized = raw.lower()
    if normalized in TRUE_ENV_VALUES:
        return {
            "configured": True,
            "valid": True,
            "source": "env",
            "effective_enabled": True,
            "fallback_enabled": False,
        }
    if normalized in FALSE_ENV_VALUES:
        return {
            "configured": True,
            "valid": True,
            "source": "env",
            "effective_enabled": False,
            "fallback_enabled": False,
        }
    return {
        "configured": True,
        "valid": False,
        "source": "env-invalid",
        "effective_enabled": False,
        "fallback_enabled": False,
    }


def _model_alias_status(env_key: str, fallback_alias: str, *, default_source: str) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": default_source,
            "alias_chars": len(fallback_alias),
            "alias_truncated": False,
            "fallback_alias": fallback_alias,
            "effective_alias": fallback_alias,
        }
    compact = " ".join(raw.split())
    alias = _short_raw(raw, limit=128)
    valid = (
        bool(alias)
        and not bool(LOCAL_PATH_RE.search(raw))
        and not Path(raw).is_absolute()
        and "/" not in raw
        and "\\" not in raw
        and bool(MODEL_ALIAS_RE.fullmatch(raw))
    )
    return {
        "configured": True,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
        "alias_chars": len(alias),
        "alias_truncated": len(compact) > 128,
        "fallback_alias": fallback_alias,
        "effective_alias": raw if valid else fallback_alias,
    }


def _enum_env_status(env_key: str, default: str, choices: set[str]) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "effective_value": default,
            "fallback_value": default,
        }
    normalized = raw.lower()
    valid = normalized in choices
    return {
        "configured": True,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
        "effective_value": normalized if valid else default,
        "fallback_value": default,
    }


def _status_host_config_without_env_load() -> StatusHostConfig:
    raw = (os.getenv("JARVIS_STATUS_HOST") or "").strip()
    if not raw:
        return StatusHostConfig(DEFAULT_STATUS_HOST, configured=False, valid=True, source="default")
    if (
        any(char.isspace() for char in raw)
        or any(ord(char) < 32 for char in raw)
        or not status_host_is_loopback(raw)
    ):
        return StatusHostConfig(DEFAULT_STATUS_HOST, configured=True, valid=False, source="env-invalid")
    return StatusHostConfig(raw, configured=True, valid=True, source="env")


def _status_port_config_without_env_load() -> StatusPortConfig:
    raw = (os.getenv("JARVIS_STATUS_PORT") or "").strip()
    if not raw:
        return StatusPortConfig(DEFAULT_STATUS_PORT, configured=False, valid=True, source="default")
    try:
        port = int(raw)
    except ValueError:
        return StatusPortConfig(DEFAULT_STATUS_PORT, configured=True, valid=False, source="env-invalid")
    if not 1 <= port <= 65535:
        return StatusPortConfig(DEFAULT_STATUS_PORT, configured=True, valid=False, source="env-invalid")
    return StatusPortConfig(port, configured=True, valid=True, source="env")


def _env_file_status() -> dict[str, Any]:
    raw = os.getenv("JARVIS_V3_ENV", "").strip()
    if not raw:
        return {
            "configured": False,
            "exists": False,
            "is_file": False,
            "is_dir": False,
            "parent_exists": False,
            "is_symlink": False,
            "owner_matches": None,
            "owner_only": None,
            "valid": True,
            "source": "default",
        }
    path = Path(raw).expanduser()
    parent_exists = path.parent.exists()
    try:
        file_stat = path.lstat()
    except OSError:
        exists = False
        is_file = False
        is_dir = False
        is_symlink = False
        owner_matches = None
        owner_only = None
    else:
        exists = True
        is_symlink = stat.S_ISLNK(file_stat.st_mode)
        is_file = stat.S_ISREG(file_stat.st_mode)
        is_dir = stat.S_ISDIR(file_stat.st_mode)
        owner_matches = file_stat.st_uid == os.geteuid()
        mode = stat.S_IMODE(file_stat.st_mode)
        owner_only = bool(owner_matches and mode & stat.S_IRUSR and not mode & 0o077)
    valid = bool(is_file and not is_symlink and owner_only)
    return {
        "configured": True,
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "parent_exists": parent_exists,
        "is_symlink": is_symlink,
        "owner_matches": owner_matches,
        "owner_only": owner_only,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
    }


def _watched_dirs_status() -> dict[str, Any]:
    raw = os.getenv("JARVIS_WATCHED_DIRS", "")
    entries = [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    if not entries:
        return {
            "configured": False,
            "count": 0,
            "existing_dirs": 0,
            "missing": 0,
            "not_dirs": 0,
            "valid": True,
            "source": "default",
        }
    existing_dirs = 0
    missing = 0
    not_dirs = 0
    for entry in entries:
        path = Path(entry).expanduser()
        if not path.exists():
            missing += 1
        elif path.is_dir():
            existing_dirs += 1
        else:
            not_dirs += 1
    valid = missing == 0 and not_dirs == 0
    return {
        "configured": True,
        "count": len(entries),
        "existing_dirs": existing_dirs,
        "missing": missing,
        "not_dirs": not_dirs,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
    }


def _weather_location_status() -> dict[str, Any]:
    raw = os.getenv("JARVIS_WEATHER_LOCATION", "").strip()
    location = _short_raw(raw, limit=80)
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "location_chars": 5,
            "location_truncated": False,
            "fallback_location": "Seoul",
        }
    valid = bool(location and any(ch.isalnum() for ch in location)) and not bool(LOCAL_PATH_RE.search(raw))
    return {
        "configured": True,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
        "location_chars": len(location),
        "location_truncated": len(" ".join(raw.split())) > 80,
        "fallback_location": "Seoul",
    }


def _news_locale_status() -> dict[str, Any]:
    raw = os.getenv("JARVIS_NEWS_LOCALE", "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "raw_chars": 0,
            "raw_truncated": False,
            "fallback_hl": "en-US",
            "fallback_gl": "US",
        }
    compact = " ".join(raw.split())
    hl, _, gl = raw.partition(":")
    hl = hl.strip()
    gl = gl.strip()
    valid_hl = bool(re.fullmatch(r"[A-Za-z]{2}(?:-[A-Za-z]{2})?", hl))
    valid_gl = bool(re.fullmatch(r"[A-Za-z]{2}", gl))
    valid = valid_hl and valid_gl and not bool(LOCAL_PATH_RE.search(raw))
    return {
        "configured": True,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
        "raw_chars": min(len(_short_raw(raw, limit=80)), 80),
        "raw_truncated": len(compact) > 80,
        "fallback_hl": "en-US",
        "fallback_gl": "US",
    }


def _owner_allowlist_status(env_key: str, *, required: bool) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": not required,
            "source": "missing" if required else "default",
            "raw_chars": 0,
            "raw_truncated": False,
            "required": required,
        }
    compact = " ".join(raw.split())
    cleaned = _short_raw(raw, limit=80)
    valid = bool(cleaned and any(ch.isalnum() for ch in cleaned)) and not bool(LOCAL_PATH_RE.search(raw))
    return {
        "configured": True,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
        "raw_chars": len(cleaned),
        "raw_truncated": len(compact) > 80,
        "required": required,
    }


def _secret_env_status(env_key: str, *, required: bool, kind: str = "secret") -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": not required,
            "source": "missing" if required else "default",
            "raw_chars": 0,
            "raw_truncated": False,
            "required": required,
            "kind": kind,
        }
    compact = " ".join(raw.split())
    cleaned = _short_raw(raw, limit=80)
    valid = (
        bool(cleaned)
        and not bool(LOCAL_PATH_RE.search(raw))
        and not Path(raw).is_absolute()
        and "/" not in raw
        and "\\" not in raw
    )
    if kind == "email":
        valid = valid and bool(re.fullmatch(r"[^@\s/\\]+@[^@\s/\\]+\.[^@\s/\\]+", raw))
    else:
        valid = valid and any(ch.isalnum() for ch in cleaned)
    return {
        "configured": True,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
        "raw_chars": len(cleaned),
        "raw_truncated": len(compact) > 80,
        "required": required,
        "kind": kind,
    }


def _secret_env_presence_status(env_key: str, *, required: bool, kind: str = "secret") -> dict[str, Any]:
    present = env_key in os.environ
    return {
        "configured": present,
        "present": present,
        "valid": None,
        "source": "environment-key" if present else "missing" if required else "default",
        "required": required,
        "kind": kind,
        "value_inspected": False,
        "validation_performed": False,
    }


def _local_command_env_status(env_key: str, *, required: bool = False) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": not required,
            "source": "missing" if required else "default",
            "required": required,
            "raw_chars": 0,
            "raw_truncated": False,
            "token_count": 0,
            "executable_resolved": False,
            "executable_kind": "missing",
            "executable_exists": False,
            "executable_is_file": False,
            "executable_is_dir": False,
            "executable_parent_exists": False,
            "executable_on_path": False,
            "executable_is_executable": False,
            "parse_error": False,
        }
    compact = " ".join(raw.split())
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return {
            "configured": True,
            "valid": False,
            "source": "env-invalid",
            "required": required,
            "raw_chars": len(_short_raw(raw, limit=80)),
            "raw_truncated": len(compact) > 80,
            "token_count": 0,
            "executable_resolved": False,
            "executable_kind": "parse_error",
            "executable_exists": False,
            "executable_is_file": False,
            "executable_is_dir": False,
            "executable_parent_exists": False,
            "executable_on_path": False,
            "executable_is_executable": False,
            "parse_error": True,
        }
    if not tokens:
        return {
            "configured": True,
            "valid": False,
            "source": "env-invalid",
            "required": required,
            "raw_chars": 0,
            "raw_truncated": False,
            "token_count": 0,
            "executable_resolved": False,
            "executable_kind": "empty",
            "executable_exists": False,
            "executable_is_file": False,
            "executable_is_dir": False,
            "executable_parent_exists": False,
            "executable_on_path": False,
            "executable_is_executable": False,
            "parse_error": False,
        }
    executable = tokens[0]
    if "/" in executable or executable.startswith("~"):
        path = Path(executable).expanduser()
        exists = path.exists()
        is_file = path.is_file()
        is_dir = path.is_dir()
        parent_exists = path.parent.exists()
        executable_ok = is_file and os.access(path, os.X_OK)
        kind = "path"
        resolved = executable_ok
        on_path = False
    else:
        resolved_path = shutil.which(executable)
        exists = bool(resolved_path)
        is_file = bool(resolved_path and Path(resolved_path).is_file())
        is_dir = False
        parent_exists = bool(resolved_path and Path(resolved_path).parent.exists())
        executable_ok = bool(resolved_path)
        kind = "path_command"
        resolved = bool(resolved_path)
        on_path = bool(resolved_path)
    return {
        "configured": True,
        "valid": resolved,
        "source": "env" if resolved else "env-invalid",
        "required": required,
        "raw_chars": len(_short_raw(raw, limit=80)),
        "raw_truncated": len(compact) > 80,
        "token_count": len(tokens),
        "executable_resolved": resolved,
        "executable_kind": kind,
        "executable_exists": exists,
        "executable_is_file": is_file,
        "executable_is_dir": is_dir,
        "executable_parent_exists": parent_exists,
        "executable_on_path": on_path,
        "executable_is_executable": executable_ok,
        "parse_error": False,
    }


def _local_path_env_status(env_key: str, *, expected: str, required: bool = False) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": not required,
            "source": "missing" if required else "default",
            "required": required,
            "expected": expected,
            "raw_chars": 0,
            "raw_truncated": False,
            "exists": False,
            "is_file": False,
            "is_dir": False,
            "parent_exists": False,
        }
    compact = " ".join(raw.split())
    path = Path(raw).expanduser()
    exists = path.exists()
    is_file = path.is_file()
    is_dir = path.is_dir()
    parent_exists = path.parent.exists()
    valid = False
    if expected == "file":
        valid = is_file
    elif expected == "directory":
        valid = is_dir
    elif expected == "existing_path":
        valid = exists
    else:
        valid = exists
    return {
        "configured": True,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
        "required": required,
        "expected": expected,
        "raw_chars": len(_short_raw(raw, limit=80)),
        "raw_truncated": len(compact) > 80,
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "parent_exists": parent_exists,
    }


def _state_file_target_status(env_key: str) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return {
            "configured": False,
            "exists": False,
            "is_file": False,
            "is_dir": False,
            "parent_exists": False,
            "valid": True,
            "source": "default",
        }
    path = Path(raw).expanduser()
    exists = path.exists()
    is_file = path.is_file()
    is_dir = path.is_dir()
    parent_exists = path.parent.exists()
    valid = parent_exists and not is_dir
    return {
        "configured": True,
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "parent_exists": parent_exists,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
    }


def _directory_target_status(env_key: str) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return {
            "configured": False,
            "exists": False,
            "is_dir": False,
            "is_file": False,
            "parent_exists": False,
            "valid": True,
            "source": "default",
        }
    path = Path(raw).expanduser()
    exists = path.exists()
    is_dir = path.is_dir()
    is_file = path.is_file()
    parent_exists = path.parent.exists()
    valid = parent_exists and not is_file
    return {
        "configured": True,
        "exists": exists,
        "is_dir": is_dir,
        "is_file": is_file,
        "parent_exists": parent_exists,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
    }


def _obsidian_vault_target_status() -> dict[str, Any]:
    primary = os.getenv("JARVIS_OBSIDIAN_VAULT", "").strip()
    secondary = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
    env_key = "JARVIS_OBSIDIAN_VAULT" if primary else "OBSIDIAN_VAULT_PATH" if secondary else ""
    if not env_key:
        status = _directory_target_status("JARVIS_OBSIDIAN_VAULT")
        status["env_key"] = ""
        return status
    status = _directory_target_status(env_key)
    status["env_key"] = env_key
    return status


def _obsidian_root_status() -> dict[str, Any]:
    raw = os.getenv("JARVIS_OBSIDIAN_ROOT", "").strip()
    configured = bool(raw)
    if not configured:
        return {
            "configured": False,
            "valid": True,
            "source": "default",
            "root_chars": len(DEFAULT_OBSIDIAN_ROOT),
            "root_truncated": False,
            "fallback_root": DEFAULT_OBSIDIAN_ROOT,
        }
    root = _short_raw(raw, limit=80)
    valid = (
        bool(root)
        and not bool(LOCAL_PATH_RE.search(raw))
        and not Path(raw).is_absolute()
        and "/" not in raw
        and "\\" not in raw
        and root not in {".", ".."}
    )
    return {
        "configured": True,
        "valid": valid,
        "source": "env" if valid else "env-invalid",
        "root_chars": len(root),
        "root_truncated": len(" ".join(raw.split())) > 80,
        "fallback_root": DEFAULT_OBSIDIAN_ROOT,
    }


def _required_file_status(env_key: str, path: Path) -> dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    exists = path.exists()
    is_file = path.is_file()
    is_dir = path.is_dir()
    parent_exists = path.parent.exists()
    return {
        "configured": bool(raw),
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "parent_exists": parent_exists,
        "valid": is_file,
        "source": "env" if raw and is_file else "env-invalid" if raw else "default" if is_file else "default-missing",
    }


def _private_required_file_status(env_key: str, path: Path) -> dict[str, Any]:
    """Presence-only status for credential files that must be private and non-symlinked."""

    raw = os.getenv(env_key, "").strip()
    parent_exists = path.parent.exists()
    is_symlink = path.is_symlink()
    exists = path.exists() or is_symlink
    is_file = False
    is_dir = False
    owner_only = False
    if exists and not is_symlink:
        try:
            file_stat = path.stat()
            is_file = stat.S_ISREG(file_stat.st_mode)
            is_dir = stat.S_ISDIR(file_stat.st_mode)
            owner_only = is_file and stat.S_IMODE(file_stat.st_mode) & 0o077 == 0
        except OSError:
            pass
    valid = is_file and owner_only and not is_symlink
    return {
        "configured": bool(raw),
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "is_symlink": is_symlink,
        "owner_only": owner_only,
        "parent_exists": parent_exists,
        "valid": valid,
        "source": "env" if raw and valid else "env-invalid" if raw else "default" if valid else "default-missing",
    }


def _storage_fallback_status() -> dict[str, Any]:
    disable_raw = os.getenv("JARVIS_DISABLE_STORAGE_FALLBACK", "").strip()
    disable_configured = bool(disable_raw)
    disable_normalized = disable_raw.lower()
    disable_valid = not disable_configured or disable_normalized in TRUE_ENV_VALUES or disable_normalized in FALSE_ENV_VALUES
    disabled = disable_configured and disable_normalized in TRUE_ENV_VALUES
    disable_source = "default" if not disable_configured else "env" if disable_valid else "env-invalid"
    raw = os.getenv("JARVIS_STORAGE_FALLBACK_DIR", "").strip()
    path = Path(raw or Path.cwd() / ".jarvis_v3_runtime").expanduser()
    exists = path.exists()
    is_dir = path.is_dir()
    is_file = path.is_file()
    parent_exists = path.parent.exists()
    parent_writable = parent_exists and os.access(path.parent, os.W_OK)
    valid = disabled or (parent_writable and not is_file)
    source = "disabled" if disabled else "env" if raw and valid else "env-invalid" if raw else "default" if valid else "default-invalid"
    return {
        "configured": bool(raw),
        "disabled": disabled,
        "disable_configured": disable_configured,
        "disable_valid": disable_valid,
        "disable_source": disable_source,
        "disable_raw_chars": len(disable_raw),
        "disable_raw_truncated": len(disable_raw) > 80,
        "disable_fallback_disabled": False,
        "exists": exists,
        "is_dir": is_dir,
        "is_file": is_file,
        "parent_exists": parent_exists,
        "parent_writable": parent_writable,
        "valid": valid,
        "source": source,
    }


def _short(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_raw(value: Any, *, limit: int) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "queues_approval": False,
        "requires_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "speaks": False,
        "completes_tasks": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    metadata.update(extra)
    return metadata


def _system_boundaries(
    *,
    reads_clipboard: bool = False,
    writes_clipboard: bool = False,
    reads_private_data: bool = False,
    executes_side_effect: bool = False,
) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": False,
        "executes_tools": False,
        "queues_approval": False,
        "requires_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": reads_private_data,
        "reads_personal_data": reads_private_data,
        "reads_clipboard": reads_clipboard,
        "writes_clipboard": writes_clipboard,
        "executes_side_effect": executes_side_effect,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "speaks": False,
        "completes_tasks": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _system_handoff(
    *,
    source: str,
    status: str,
    reason: str | None = None,
    app: str | None = None,
    level: int | None = None,
    available: bool | None = None,
    chars: int | None = None,
    max_chars: int | None = None,
    truncated: bool | None = None,
    returncode: int | None = None,
    error_type: str | None = None,
    action_attempted: bool = False,
    reads_clipboard: bool = False,
    writes_clipboard: bool = False,
    reads_private_data: bool = False,
    executes_side_effect: bool = False,
    state_changed: bool = False,
    changed: list[str] | None = None,
) -> dict[str, Any]:
    changed_identities = list(changed or []) if state_changed else []
    handoff: dict[str, Any] = {
        "source": source,
        "status": status,
        "reason": reason,
        "refused": status == "refused",
        "ready_for_operator": True,
        "state_changed": state_changed,
        "changed": changed_identities,
        "action_attempted": action_attempted,
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "operator_limit_rule": OPERATOR_LIMIT_RULE,
        "next_commands": [
            "system info",
            "what volume",
            "get clipboard",
            "set clipboard <text>",
            "open <app>",
        ],
        "boundaries": _system_boundaries(
            reads_clipboard=reads_clipboard,
            writes_clipboard=writes_clipboard,
            reads_private_data=reads_private_data,
            executes_side_effect=executes_side_effect,
        ),
    }
    if app is not None:
        handoff["app"] = _short_raw(app, limit=MAX_APP_NAME_CHARS)
    if level is not None:
        handoff["level"] = level
    if available is not None:
        handoff["available"] = available
    if chars is not None:
        handoff["chars"] = chars
    if max_chars is not None:
        handoff["max_chars"] = max_chars
    if truncated is not None:
        handoff["truncated"] = truncated
    if returncode is not None:
        handoff["returncode"] = returncode
    if error_type:
        handoff["error_type"] = error_type
    return handoff


def _system_metadata(*, source: str, status: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    state_changed = extra.get("state_changed") is True
    changed = list(extra.get("changed") or []) if state_changed else []
    handoff = _system_handoff(
        source=source,
        status=status,
        reason=reason,
        app=extra.get("app") if isinstance(extra.get("app"), str) else None,
        level=extra.get("level") if isinstance(extra.get("level"), int) else None,
        available=extra.get("available") if isinstance(extra.get("available"), bool) else None,
        chars=extra.get("chars") if isinstance(extra.get("chars"), int) else None,
        max_chars=extra.get("max_chars") if isinstance(extra.get("max_chars"), int) else None,
        truncated=extra.get("truncated") if isinstance(extra.get("truncated"), bool) else None,
        returncode=extra.get("returncode") if isinstance(extra.get("returncode"), int) else None,
        error_type=extra.get("error_type") if isinstance(extra.get("error_type"), str) else None,
        action_attempted=bool(extra.get("action_attempted")),
        reads_clipboard=bool(extra.get("reads_clipboard")),
        writes_clipboard=bool(extra.get("writes_clipboard")),
        reads_private_data=bool(extra.get("reads_private_data") or extra.get("reads_personal_data")),
        executes_side_effect=bool(extra.get("executes_side_effect")),
        state_changed=state_changed,
        changed=changed,
    )
    metadata_extra = dict(extra)
    metadata_extra["state_changed"] = state_changed
    metadata_extra["changed"] = changed
    if reason is not None and "reason" not in metadata_extra:
        metadata_extra["reason"] = reason
    return _safe_metadata(
        **metadata_extra,
        system_handoff=handoff,
        system_handoff_ready=True,
        system_status=status,
        system_state_changed=state_changed,
        system_changed=list(changed),
        refusal_reason=reason,
    )


def _with_system_boundary(body: str) -> str:
    return f"{body}{SYSTEM_BOUNDARY}"


def _system_error(action: str, exc: Exception | None = None) -> str:
    detail = f" ({type(exc).__name__})" if exc is not None else ""
    return (
        f"Could not {action}. Check macOS permissions in System Settings > Privacy & Security "
        f"(especially Accessibility for app/window actions), run `setup check`, then retry.{detail}"
    )


def _known_no_action_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
    action: str = SYSTEM_INPUT_RECOVERY_ACTION,
) -> dict[str, Any]:
    """Declare a refusal that occurred before any system side effect."""

    declared = dict(metadata)
    declared.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        declared,
        output=output,
        action=action,
    )


def _system_read_failure_metadata(
    metadata: dict[str, Any],
    *,
    output: str,
) -> dict[str, Any]:
    """Declare a side-effect-free private system read failure."""

    return declare_retryable_personal_read_failure(
        metadata,
        output=output,
        action=SYSTEM_READ_RECOVERY_ACTION,
        commands=("setup check",),
    )


def _system_attempt_outcome_unknown_metadata(
    metadata: dict[str, Any],
    *,
    action: str,
) -> dict[str, Any]:
    """Declare a system mutation attempt that must be verified before replay."""

    return declare_outcome_unknown_failure(
        metadata,
        output=action,
    )


def _run(args: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def system_info(_: dict[str, Any]) -> ToolResult:
    parts = [
        f"System: {platform.platform()}",
        f"Machine: {platform.machine()}",
        f"Python: {platform.python_version()}",
    ]
    try:
        uptime = _run(["uptime"]).stdout.strip()
        if uptime:
            parts.append(f"Uptime: {uptime}")
    except Exception:
        pass
    try:
        disk = _run(["df", "-h", "/"]).stdout.strip()
        if disk:
            parts.append("Disk:\n" + disk)
    except Exception:
        pass
    return ToolResult("system_info", True, _with_system_boundary("\n".join(parts)), _safe_metadata(reads_system_info=True))


def list_running_apps(_: dict[str, Any]) -> ToolResult:
    script = 'tell application "System Events" to get name of every process whose background only is false'
    try:
        result = _run(["osascript", "-e", script])
    except Exception as exc:
        output = f"{_system_error('list apps', exc)} {SYSTEM_READ_RECOVERY_ACTION}"
        return ToolResult(
            "list_running_apps",
            False,
            _with_system_boundary(output),
            _system_read_failure_metadata(
                _safe_metadata(reads_running_apps=True, reads_personal_data=True, reads_private_data=True, exception_type=type(exc).__name__),
                output=output,
            ),
        )
    if result.returncode != 0:
        output = f"{_system_error('list apps')} {SYSTEM_READ_RECOVERY_ACTION}"
        return ToolResult(
            "list_running_apps",
            False,
            _with_system_boundary(output),
            _system_read_failure_metadata(
                _safe_metadata(reads_running_apps=True, reads_personal_data=True, reads_private_data=True, returncode=result.returncode),
                output=output,
            ),
        )
    apps = sorted(app.strip() for app in result.stdout.strip().split(",") if app.strip())
    return ToolResult("list_running_apps", True, _with_system_boundary("\n".join(apps) if apps else "No visible apps found."), _safe_metadata(count=len(apps), reads_running_apps=True, reads_personal_data=True, reads_private_data=True))


def frontmost_app(_: dict[str, Any]) -> ToolResult:
    script = 'tell application "System Events" to get name of first process whose frontmost is true'
    try:
        result = _run(["osascript", "-e", script])
    except Exception as exc:
        output = f"{_system_error('determine frontmost app', exc)} {SYSTEM_READ_RECOVERY_ACTION}"
        return ToolResult(
            "frontmost_app",
            False,
            _with_system_boundary(output),
            _system_read_failure_metadata(
                _safe_metadata(reads_running_apps=True, reads_personal_data=True, reads_private_data=True, exception_type=type(exc).__name__),
                output=output,
            ),
        )
    if result.returncode == 0:
        name = _short(result.stdout, limit=MAX_APP_NAME_CHARS)
        return ToolResult("frontmost_app", True, _with_system_boundary(name), _safe_metadata(app=name, reads_running_apps=True, reads_personal_data=True, reads_private_data=True))
    output = f"{_system_error('determine frontmost app')} {SYSTEM_READ_RECOVERY_ACTION}"
    return ToolResult(
        "frontmost_app",
        False,
        _with_system_boundary(output),
        _system_read_failure_metadata(
            _safe_metadata(reads_running_apps=True, reads_personal_data=True, reads_private_data=True, returncode=result.returncode),
            output=output,
        ),
    )


def stop_jarvis(_: dict[str, Any]) -> ToolResult:
    output = (
        "Stop acknowledged. I did not delete the message, approve anything, dismiss approvals, "
        "run tools, control the computer, read private data, or change external state. "
        "In the GUI, Stop Jarvis also cancels local speech/output display while keeping the conversation and approval queue visible. "
        f"{OPERATOR_LIMIT_RULE}"
    )
    return ToolResult(
        "stop_jarvis",
        True,
        output,
        _safe_metadata(),
    )


def telegram_control_restart_guidance(_: dict[str, Any]) -> ToolResult:
    """Explain the owner-only boundary for the Telegram control daemon."""
    output = (
        "Telegram control daemon restart was not performed. Jarvis cannot restart local daemons from chat. "
        "For a read-only capability and recovery overview, run `telegram control daemon status`; "
        "it does not inspect live process state. "
        "Use your approved local operator runbook to restart it when you choose. "
        "No process control, message send, approval change, account access, or private-data read occurred."
    )
    return ToolResult(
        "telegram_control_restart_guidance",
        True,
        output,
        _safe_metadata(
            restart_requested=True,
            restart_performed=False,
            status_command="telegram control daemon status",
        ),
    )


def restart_target_clarification(_: dict[str, Any]) -> ToolResult:
    """Keep ambiguous restart follow-ups from becoming guessed computer control."""
    output = (
        "Jarvis did not restart anything because the target is ambiguous. Name the exact service first. "
        "For the Telegram control daemon, say `restart the Telegram control daemon`; Jarvis will only give the approved local operator guidance. "
        "For the local model, run `model status` before choosing any operator action. "
        "No process control, message send, approval change, account access, or private-data read occurred."
    )
    return ToolResult(
        "restart_target_clarification",
        True,
        output,
        _safe_metadata(
            restart_requested=True,
            restart_performed=False,
            restart_target_identified=False,
            status_command="model status",
        ),
    )


def resolve_open_application_approval(
    args: dict[str, Any],
) -> ApprovalArgumentResolution | ToolResult:
    name = _short(args.get("name"), limit=MAX_APP_NAME_CHARS)
    if LOCAL_PATH_RE.search(name):
        output = f"Application name cannot be a local file path. {SYSTEM_INPUT_RECOVERY_ACTION}"
        metadata = _known_no_action_failure_metadata(
            _system_metadata(
                source="open_application",
                status="refused",
                reason="invalid_app_name",
                app="<local-path>",
                executes_side_effect=False,
            ),
            output=output,
        )
        metadata.update(
            {
                "requires_confirmation": False,
                "executed_handler": False,
                "handler_invoked": False,
            }
        )
        return ToolResult("open_application", False, _with_system_boundary(output), metadata)
    return ApprovalArgumentResolution({"name": name}, {})


def open_application(args: dict[str, Any]) -> ToolResult:
    name = _short(args.get("name"), limit=MAX_APP_NAME_CHARS)
    if not name:
        output = f"No application name provided. {SYSTEM_INPUT_RECOVERY_ACTION}"
        return ToolResult(
            "open_application",
            False,
            _with_system_boundary(output),
            _known_no_action_failure_metadata(
                _system_metadata(source="open_application", status="refused", reason="missing_app_name", executes_side_effect=True, app=""),
                output=output,
            ),
        )
    if LOCAL_PATH_RE.search(name):
        output = f"Application name cannot be a local file path. {SYSTEM_INPUT_RECOVERY_ACTION}"
        return ToolResult(
            "open_application",
            False,
            _with_system_boundary(output),
            _known_no_action_failure_metadata(
                _system_metadata(source="open_application", status="refused", reason="invalid_app_name", app="<local-path>", executes_side_effect=False),
                output=output,
            ),
        )
    try:
        result = _run(["open", "-a", name])
    except Exception as exc:
        output = (
            f"Could not open {name}. Check macOS permissions in System Settings > Privacy & Security "
            "(especially Accessibility for app/window actions), and run `setup check` if the app did not open. "
            f"({type(exc).__name__}) {APPLICATION_OUTCOME_UNKNOWN_ACTION}"
        )
        return ToolResult(
            "open_application",
            False,
            _with_system_boundary(output),
            _system_attempt_outcome_unknown_metadata(
                _system_metadata(source="open_application", status="error", app=name, executes_side_effect=True, action_attempted=True, exception_type=type(exc).__name__, error_type=type(exc).__name__),
                action=APPLICATION_OUTCOME_UNKNOWN_ACTION,
            ),
        )
    if result.returncode == 0:
        return ToolResult(
            "open_application",
            True,
            _with_system_boundary(f"Opened {name}."),
            _system_metadata(
                source="open_application",
                status="ok",
                app=name,
                executes_side_effect=True,
                action_attempted=True,
                state_changed=True,
                changed=["application_launch"],
            ),
        )
    output = (
        f"Could not open {name}. Check macOS permissions in System Settings > Privacy & Security "
        "(especially Accessibility for app/window actions), and run `setup check` if the app did not open. "
        f"{APPLICATION_OUTCOME_UNKNOWN_ACTION}"
    )
    return ToolResult(
        "open_application",
        False,
        _with_system_boundary(output),
        _system_attempt_outcome_unknown_metadata(
            _system_metadata(source="open_application", status="error", app=name, executes_side_effect=True, action_attempted=True, returncode=result.returncode),
            action=APPLICATION_OUTCOME_UNKNOWN_ACTION,
        ),
    )


def volume(args: dict[str, Any]) -> ToolResult:
    level = args.get("level")
    if level is None:
        scripts = [
            "output volume of (get volume settings)",
            "get output volume of (get volume settings)",
        ]
        errors = []
        for script in scripts:
            result = _run(["osascript", "-e", script])
            if result.returncode == 0 and result.stdout.strip():
                return ToolResult(
                    "volume",
                    True,
                    _with_system_boundary(f"Current volume: {result.stdout.strip()}%"),
                    _system_metadata(source="volume", status="ok", available=True),
                )
            errors.append((result.stderr or result.stdout or "").strip())
        detail = next((error for error in errors if error), "macOS did not return a volume value.")
        return ToolResult(
            "volume",
            True,
            _with_system_boundary(
                "Current volume is unavailable in this environment. Check macOS audio permissions and output settings "
                "in System Settings > Sound and Privacy & Security, run `setup check`, then retry."
            ),
            _system_metadata(source="volume", status="empty", available=False, error_type="volume_read_unavailable", error_detail_chars=len(detail)),
        )

    try:
        value = max(0, min(100, int(level)))
    except (TypeError, ValueError):
        output = f"Volume level must be a number from 0 to 100. {SYSTEM_INPUT_RECOVERY_ACTION}"
        return ToolResult(
            "volume",
            False,
            _with_system_boundary(output),
            _known_no_action_failure_metadata(
                _system_metadata(source="volume", status="refused", reason="invalid_level", level=None, raw_level=_short_raw(level, limit=80)),
                output=output,
            ),
        )
    result = _run(["osascript", "-e", f"set volume output volume {value}"])
    if result.returncode == 0:
        return ToolResult(
            "volume",
            True,
            _with_system_boundary(f"Volume set to {value}%."),
            _system_metadata(
                source="volume",
                status="ok",
                level=value,
                available=True,
                executes_side_effect=True,
                action_attempted=True,
                state_changed=True,
                changed=["output_volume"],
            ),
        )
    output = (
        "Could not set volume. Check macOS audio permissions and output settings in System Settings > Sound "
        "and Privacy & Security, and run `setup check` if the volume did not change. "
        f"{VOLUME_OUTCOME_UNKNOWN_ACTION}"
    )
    return ToolResult(
        "volume",
        False,
        _with_system_boundary(output),
        _system_attempt_outcome_unknown_metadata(
            _system_metadata(source="volume", status="error", level=value, available=False, executes_side_effect=True, action_attempted=True, returncode=result.returncode),
            action=VOLUME_OUTCOME_UNKNOWN_ACTION,
        ),
    )


def get_clipboard(args: dict[str, Any]) -> ToolResult:
    max_chars = _bounded_int(args.get("max_chars"), 1000, 1, MAX_CLIPBOARD_CHARS)
    result = _run(["pbpaste"])
    if result.returncode != 0:
        output = (
            "Could not read clipboard. Check macOS clipboard and Automation permissions in System Settings > "
            "Privacy & Security, run `setup check`, then retry. "
            f"{SYSTEM_READ_RECOVERY_ACTION}"
        )
        return ToolResult(
            "get_clipboard",
            False,
            _with_system_boundary(output),
            _system_read_failure_metadata(
                _system_metadata(
                    source="get_clipboard",
                    status="error",
                    reads_private_data=True,
                    reads_personal_data=True,
                    reads_clipboard=True,
                    returncode=result.returncode,
                    **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars),
                ),
                output=output,
            ),
        )
    content = result.stdout
    if not content:
        return ToolResult(
            "get_clipboard",
            True,
            _with_system_boundary("Clipboard is empty."),
            _system_metadata(source="get_clipboard", status="empty", chars=0, truncated=False, reads_private_data=True, reads_personal_data=True, reads_clipboard=True, **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars)),
        )
    truncated = len(content) > max_chars
    output = content[:max_chars]
    if truncated:
        output += f"\n\n... truncated {len(content) - max_chars} chars"
    return ToolResult(
        "get_clipboard",
        True,
        _with_system_boundary(output),
        _system_metadata(source="get_clipboard", status="ok", chars=len(content), truncated=truncated, reads_private_data=True, reads_personal_data=True, reads_clipboard=True, **_raw_int_metadata(args.get("max_chars"), key="max_chars", sanitized=max_chars)),
    )


def set_clipboard(args: dict[str, Any]) -> ToolResult:
    text = str(args.get("text") or "")
    if len(text) > MAX_CLIPBOARD_WRITE_CHARS:
        output = (
            f"Refusing to copy {len(text)} chars to clipboard; limit is {MAX_CLIPBOARD_WRITE_CHARS}. "
            f"{SYSTEM_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "set_clipboard",
            False,
            _with_system_boundary(output),
            _known_no_action_failure_metadata(
                _system_metadata(source="set_clipboard", status="refused", reason="too_large", chars=len(text), max_chars=MAX_CLIPBOARD_WRITE_CHARS, writes_clipboard=False),
                output=output,
            ),
        )
    process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    process.communicate(text.encode("utf-8"))
    if process.returncode == 0:
        return ToolResult(
            "set_clipboard",
            True,
            _with_system_boundary(f"Copied {len(text)} chars to clipboard."),
            _system_metadata(
                source="set_clipboard",
                status="ok",
                chars=len(text),
                writes_clipboard=True,
                executes_side_effect=True,
                action_attempted=True,
                state_changed=True,
                changed=["clipboard"],
            ),
        )
    output = f"Could not set clipboard. {CLIPBOARD_OUTCOME_UNKNOWN_ACTION}"
    return ToolResult(
        "set_clipboard",
        False,
        _with_system_boundary(output),
        _system_attempt_outcome_unknown_metadata(
            _system_metadata(source="set_clipboard", status="error", chars=len(text), writes_clipboard=False, executes_side_effect=True, action_attempted=True),
            action=CLIPBOARD_OUTCOME_UNKNOWN_ACTION,
        ),
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _ollama_destination_policy_detail(destination: Any) -> str:
    family = destination.address_family or "unavailable"
    port = str(destination.port) if destination.port else "unavailable"
    return (
        f"source: {destination.source}; family: {family}; port: {port}; "
        "redirects: blocked; proxy policy: stripped from accepted probes"
    )


def _ollama_no_cloud_policy_detail(
    policy: Any,
    *,
    required: bool,
    provider_valid: bool = True,
) -> str:
    if not provider_valid:
        status = "not evaluated; model provider is invalid; stored context withheld"
    elif not required:
        status = "not required; OpenAI provider selected"
    elif not policy.valid:
        status = (
            "needs attention; invalid setting; personalized Ollama operation is not ready; "
            "current-message-only model route may remain usable; stored context withheld"
        )
    elif not policy.unverified_context_consent_valid:
        status = (
            "needs attention; invalid unverified personal-context consent; personalized Ollama "
            "operation is not ready; current-message-only model route may remain usable; stored context withheld"
        )
    elif policy.cloud_model_alias:
        status = (
            "needs attention; cloud model alias blocked; personalized Ollama operation is not ready; "
            "stored context withheld"
        )
    elif policy.personal_context_allowed:
        status = "ok; stored personal context allowed for the non-cloud model alias"
    else:
        status = "current-message-only model route may remain usable; stored context withheld"
    return (
        f"{status} | configured: {'yes' if policy.configured else 'no'}; "
        f"valid: {'yes' if policy.valid else 'no'}; "
        f"requested: {'yes' if policy.requested else 'no'}; "
        f"consent configured: {'yes' if policy.unverified_context_consent_configured else 'no'}; "
        f"consent valid: {'yes' if policy.unverified_context_consent_valid else 'no'}; "
        f"consent granted: {'yes' if policy.unverified_context_consent_allowed else 'no'}; "
        "execution locality: unknown; "
        "daemon cloud-disabled state: independently unverified; "
        "on-device execution: independently unverified; value hidden"
    )


def setup_check(_: dict[str, Any]) -> ToolResult:
    env_file = _env_file_status()
    provider_status = _enum_env_status("JARVIS_MODEL_PROVIDER", "ollama", MODEL_PROVIDERS)
    model_provider = provider_status["effective_value"] if provider_status["valid"] else "invalid"
    openai_provider = model_provider == "openai"
    ollama_provider = model_provider == "ollama"
    ollama_destination = resolve_ollama_destination()
    checks = []
    missing: list[str] = []
    attention: list[str] = []
    modules = ["pyautogui", "PIL"]
    if ollama_provider:
        modules.insert(0, "ollama")
    elif openai_provider:
        checks.append("- ollama: not required (OpenAI provider selected)")
    else:
        checks.append("- ollama: not evaluated (model provider invalid)")
    for module in modules:
        available = _module_available(module)
        label = "pillow" if module == "PIL" else module
        if not available:
            missing.append(label)
        checks.append(f"- {label}: {'ok' if available else 'missing'}")

    google_dependency_specs = [
        ("google.oauth2.credentials", "google-auth"),
        ("google_auth_oauthlib.flow", "google-auth-oauthlib"),
        ("google_auth_httplib2", "google-auth-httplib2"),
        ("googleapiclient.discovery", "google-api-python-client"),
    ]
    google_dependency_status: dict[str, bool] = {}
    for module, label in google_dependency_specs:
        available = _module_available(module)
        google_dependency_status[label] = available
        if not available:
            missing.append(label)
            attention.append(label)
        checks.append(f"- Google connector dependency ({label}): {'ok' if available else 'missing'}")

    for command in [["osascript", "-e", "return 1"], ["open", "--help"], ["say", "-v", "?"]]:
        name = command[0]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            ok = result.returncode == 0 or name == "open"
        except Exception:
            ok = False
        if not ok:
            missing.append(name)
        checks.append(f"- {name}: {'ok' if ok else 'missing or blocked'}")

    pbpaste_available = shutil.which("pbpaste") is not None
    if not pbpaste_available:
        missing.append("pbpaste")
    checks.append(f"- pbpaste: {'ok' if pbpaste_available else 'missing or blocked'} (not executed; clipboard privacy)")

    env_checks = [
        ("JARVIS_V3_ENV", "custom env file"),
        ("JARVIS_DATA_DIR", "Jarvis data directory"),
        ("JARVIS_DB_PATH", "Jarvis SQLite database file"),
        ("JARVIS_OBSIDIAN_VAULT", "Obsidian vault path"),
        ("OBSIDIAN_VAULT_PATH", "secondary Obsidian vault path"),
        ("JARVIS_OBSIDIAN_ROOT", "Obsidian Jarvis root folder"),
        ("TELEGRAM_BOT_TOKEN", "Telegram bot token"),
        ("JARVIS_OWNER_TELEGRAM", "Telegram owner allowlist"),
        ("JARVIS_TELEGRAM_STATE", "Telegram command state file"),
        ("GMAIL_ADDRESS", "Gmail address"),
        ("GMAIL_APP_PASSWORD", "Gmail app password"),
        ("JARVIS_GOOGLE_READONLY_TOKEN", "Google Calendar read-only token file"),
        ("JARVIS_OWNER_IMESSAGE", "legacy iMessage owner allowlist"),
        ("JARVIS_V3_IMESSAGE_STATE", "legacy iMessage command state file"),
        ("JARVIS_MODEL_PROVIDER", "model provider"),
        ("OPENAI_API_KEY", "OpenAI API key"),
        ("OLLAMA_NO_CLOUD", "Ollama no-cloud setting"),
        (OLLAMA_UNVERIFIED_PERSONAL_CONTEXT_ENV, "Ollama unverified personal-context consent"),
        ("OLLAMA_MODEL", "fallback Ollama model alias"),
        ("JARVIS_CHAT_MODEL", "chat model alias"),
        ("JARVIS_PLANNER_MODEL", "planner model alias"),
        ("JARVIS_CHAT_REASONING_EFFORT", "chat reasoning effort"),
        ("JARVIS_PLANNER_REASONING_EFFORT", "planner reasoning effort"),
        ("JARVIS_OPENAI_MAX_OUTPUT_TOKENS", "OpenAI total reasoning/output ceiling"),
        ("JARVIS_MODEL_TIMEOUT_SECONDS", "planner timeout override"),
        ("JARVIS_CHAT_TIMEOUT_SECONDS", "free-form chat timeout override"),
        ("JARVIS_CHAT_MAX_REPLY_TOKENS", "chat reply token cap"),
        ("JARVIS_CHAT_MAX_HISTORY_MESSAGES", "chat history window"),
        ("JARVIS_USE_MODEL_PLANNER", "model planner toggle"),
        ("JARVIS_ALLOW_REMOTE_COMPACTION", "remote conversation compaction opt-in"),
        ("JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT", "remote interactive personal-context opt-in"),
        ("JARVIS_DISABLE_STORAGE_FALLBACK", "storage fallback disable toggle"),
        ("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", "smoke module timeout override"),
        ("JARVIS_WEATHER_LOCATION", "default weather location"),
        ("JARVIS_NEWS_LOCALE", "news locale"),
        ("JARVIS_STATUS_HOST", "status dashboard host"),
        ("JARVIS_STATUS_PORT", "status dashboard port"),
        ("JARVIS_STATUS_AUTH_TOKEN", "status dashboard authentication token"),
        ("JARVIS_WATCHED_DIRS", "watched directories"),
        ("JARVIS_REMINDERS_FILE", "Telegram reminders state file"),
        ("JARVIS_OAV_VISION_REVIEWER_COMMAND", "local OAV vision reviewer command"),
        ("JARVIS_VOICE_WHISPER_MODEL_PATH", "local Whisper model file"),
        ("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH", "local faster-whisper model directory"),
    ]
    env_configured: dict[str, bool] = {}
    for env_key, label in env_checks:
        configured = env_key in os.environ if env_key in SECRET_ENV_KEYS else bool(os.getenv(env_key, "").strip())
        env_configured[env_key] = configured
        status = "key present" if configured else "key not present"
        detail = "value not inspected" if env_key in SECRET_ENV_KEYS else "value hidden"
        if env_key in {"TELEGRAM_BOT_TOKEN", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"} and not configured:
            attention.append(env_key)
        checks.append(f"- {label} ({env_key}): {status} ({detail})")

    status_host_config = _status_host_config_without_env_load()
    status_port_config = _status_port_config_without_env_load()
    chat_reasoning = _enum_env_status(
        "JARVIS_CHAT_REASONING_EFFORT", "low" if openai_provider else "medium", REASONING_EFFORTS
    )
    planner_reasoning = _enum_env_status("JARVIS_PLANNER_REASONING_EFFORT", "low", REASONING_EFFORTS)
    planner_timeout_default = 12.0 if openai_provider else DEFAULT_MODEL_TIMEOUT_SECONDS
    chat_timeout_default = 60.0 if openai_provider else DEFAULT_CHAT_TIMEOUT_SECONDS
    planner_timeout_maximum = (
        MAX_OPENAI_MODEL_TIMEOUT_SECONDS if openai_provider else MAX_OLLAMA_MODEL_TIMEOUT_SECONDS
    )
    chat_timeout_maximum = (
        MAX_OPENAI_CHAT_TIMEOUT_SECONDS if openai_provider else MAX_OLLAMA_CHAT_TIMEOUT_SECONDS
    )
    planner_timeout = _runtime_timeout_status(
        "JARVIS_MODEL_TIMEOUT_SECONDS",
        planner_timeout_default,
        MIN_MODEL_TIMEOUT_SECONDS,
        planner_timeout_maximum,
    )
    chat_timeout = _runtime_timeout_status(
        "JARVIS_CHAT_TIMEOUT_SECONDS",
        chat_timeout_default,
        MIN_CHAT_TIMEOUT_SECONDS,
        chat_timeout_maximum,
    )
    chat_max_reply_tokens = _runtime_int_status(
        "JARVIS_CHAT_MAX_REPLY_TOKENS",
        DEFAULT_CHAT_MAX_REPLY_TOKENS,
        MIN_CHAT_MAX_REPLY_TOKENS,
        MAX_CHAT_MAX_REPLY_TOKENS,
    )
    chat_max_history_messages = _runtime_int_status(
        "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
        DEFAULT_CHAT_MAX_HISTORY_MESSAGES,
        MIN_CHAT_MAX_HISTORY_MESSAGES,
        MAX_CHAT_MAX_HISTORY_MESSAGES,
    )
    smoke_timeout = _smoke_timeout_status()
    model_planner_toggle = _model_planner_toggle_status()
    remote_compaction_toggle = _remote_compaction_toggle_status()
    remote_personal_context_toggle = _remote_personal_context_toggle_status()
    fallback_model_alias = _model_alias_status("OLLAMA_MODEL", DEFAULT_MODEL_ALIAS, default_source="default")
    chat_model_fallback = DEFAULT_OPENAI_CHAT_MODEL if openai_provider else fallback_model_alias["effective_alias"]
    chat_model_default_source = "provider-default" if openai_provider else fallback_model_alias["source"]
    chat_model_alias = _model_alias_status(
        "JARVIS_CHAT_MODEL",
        chat_model_fallback,
        default_source=chat_model_default_source,
    )
    planner_model_fallback = DEFAULT_OPENAI_PLANNER_MODEL if openai_provider else chat_model_alias["effective_alias"]
    planner_model_default_source = "provider-default" if openai_provider else chat_model_alias["source"]
    planner_model_alias = _model_alias_status(
        "JARVIS_PLANNER_MODEL",
        planner_model_fallback,
        default_source=planner_model_default_source,
    )
    ollama_no_cloud = ollama_local_only_policy(chat_model_alias["effective_alias"])
    data_dir_target = _directory_target_status("JARVIS_DATA_DIR")
    db_path_target = _state_file_target_status("JARVIS_DB_PATH")
    obsidian_vault_target = _obsidian_vault_target_status()
    obsidian_root = _obsidian_root_status()
    watched_dirs = _watched_dirs_status()
    weather_location = _weather_location_status()
    news_locale = _news_locale_status()
    telegram_bot_token = _secret_env_presence_status("TELEGRAM_BOT_TOKEN", required=True)
    gmail_address = _secret_env_presence_status("GMAIL_ADDRESS", required=True, kind="email")
    gmail_app_password = _secret_env_presence_status("GMAIL_APP_PASSWORD", required=True)
    openai_api_key = _secret_env_presence_status("OPENAI_API_KEY", required=openai_provider)
    telegram_owner = _owner_allowlist_status("JARVIS_OWNER_TELEGRAM", required=True)
    imessage_owner = _owner_allowlist_status("JARVIS_OWNER_IMESSAGE", required=False)
    telegram_state_file = _state_file_target_status("JARVIS_TELEGRAM_STATE")
    imessage_state_file = _state_file_target_status("JARVIS_V3_IMESSAGE_STATE")
    reminders_state_file = _state_file_target_status("JARVIS_REMINDERS_FILE")
    oav_vision_reviewer = _local_command_env_status("JARVIS_OAV_VISION_REVIEWER_COMMAND")
    voice_whisper_model = _local_path_env_status("JARVIS_VOICE_WHISPER_MODEL_PATH", expected="file")
    voice_faster_whisper_model = _local_path_env_status("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH", expected="directory")
    storage_fallback = _storage_fallback_status()
    google_creds = _google_creds_file()
    google_readonly_token = _google_readonly_token_file()
    google_token = _google_token_file()
    google_credentials_status = _required_file_status("JARVIS_GOOGLE_CREDS", google_creds)
    google_readonly_token_status = _private_required_file_status(
        "JARVIS_GOOGLE_READONLY_TOKEN", google_readonly_token
    )
    google_token_status = _required_file_status("JARVIS_GOOGLE_TOKEN", google_token)
    if not status_host_config.valid:
        attention.append("JARVIS_STATUS_HOST")
    if not status_port_config.valid:
        attention.append("JARVIS_STATUS_PORT")
    if not env_configured["JARVIS_STATUS_AUTH_TOKEN"]:
        attention.append("JARVIS_STATUS_AUTH_TOKEN")
    if not planner_timeout["valid"]:
        attention.append("JARVIS_MODEL_TIMEOUT_SECONDS")
    if not chat_timeout["valid"]:
        attention.append("JARVIS_CHAT_TIMEOUT_SECONDS")
    if not chat_max_reply_tokens["valid"]:
        attention.append("JARVIS_CHAT_MAX_REPLY_TOKENS")
    if not chat_max_history_messages["valid"]:
        attention.append("JARVIS_CHAT_MAX_HISTORY_MESSAGES")
    if not smoke_timeout["valid"]:
        attention.append("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS")
    if not model_planner_toggle["valid"]:
        attention.append("JARVIS_USE_MODEL_PLANNER")
    if not remote_compaction_toggle["valid"]:
        attention.append("JARVIS_ALLOW_REMOTE_COMPACTION")
    if not remote_personal_context_toggle["valid"]:
        attention.append("JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT")
    if not provider_status["valid"]:
        attention.append("JARVIS_MODEL_PROVIDER")
    if ollama_provider and not ollama_destination.allowed:
        attention.append("OLLAMA_HOST")
    if ollama_provider and (not ollama_no_cloud.valid or not ollama_no_cloud.requested):
        attention.append("OLLAMA_NO_CLOUD")
    if ollama_provider and (
        not ollama_no_cloud.unverified_context_consent_valid
        or (ollama_no_cloud.requested and not ollama_no_cloud.unverified_context_consent_allowed)
    ):
        attention.append(OLLAMA_UNVERIFIED_PERSONAL_CONTEXT_ENV)
    if ollama_provider and ollama_no_cloud.cloud_model_alias:
        attention.append("JARVIS_CHAT_MODEL")
    if not chat_reasoning["valid"]:
        attention.append("JARVIS_CHAT_REASONING_EFFORT")
    if not planner_reasoning["valid"]:
        attention.append("JARVIS_PLANNER_REASONING_EFFORT")
    if openai_api_key["required"] and not openai_api_key["present"]:
        attention.append("OPENAI_API_KEY")
    if not storage_fallback["disable_valid"]:
        attention.append("JARVIS_DISABLE_STORAGE_FALLBACK")
    if ollama_provider and not fallback_model_alias["valid"]:
        attention.append("OLLAMA_MODEL")
    if not chat_model_alias["valid"]:
        attention.append("JARVIS_CHAT_MODEL")
    if not planner_model_alias["valid"]:
        attention.append("JARVIS_PLANNER_MODEL")
    if not env_file["valid"]:
        attention.append("JARVIS_V3_ENV")
    if not data_dir_target["valid"]:
        attention.append("JARVIS_DATA_DIR")
    if not db_path_target["valid"]:
        attention.append("JARVIS_DB_PATH")
    if not obsidian_vault_target["valid"]:
        attention.append(obsidian_vault_target["env_key"] or "JARVIS_OBSIDIAN_VAULT")
    if not obsidian_root["valid"]:
        attention.append("JARVIS_OBSIDIAN_ROOT")
    if not watched_dirs["valid"]:
        attention.append("JARVIS_WATCHED_DIRS")
    if not weather_location["valid"]:
        attention.append("JARVIS_WEATHER_LOCATION")
    if not news_locale["valid"]:
        attention.append("JARVIS_NEWS_LOCALE")
    if not telegram_bot_token["present"]:
        attention.append("TELEGRAM_BOT_TOKEN")
    if not gmail_address["present"]:
        attention.append("GMAIL_ADDRESS")
    if not gmail_app_password["present"]:
        attention.append("GMAIL_APP_PASSWORD")
    if not telegram_owner["valid"]:
        attention.append("JARVIS_OWNER_TELEGRAM")
    if not imessage_owner["valid"]:
        attention.append("JARVIS_OWNER_IMESSAGE")
    if not telegram_state_file["valid"]:
        attention.append("JARVIS_TELEGRAM_STATE")
    if not imessage_state_file["valid"]:
        attention.append("JARVIS_V3_IMESSAGE_STATE")
    if not reminders_state_file["valid"]:
        attention.append("JARVIS_REMINDERS_FILE")
    if not oav_vision_reviewer["valid"]:
        attention.append("JARVIS_OAV_VISION_REVIEWER_COMMAND")
    if not voice_whisper_model["valid"]:
        attention.append("JARVIS_VOICE_WHISPER_MODEL_PATH")
    if not voice_faster_whisper_model["valid"]:
        attention.append("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH")
    if not storage_fallback["valid"]:
        attention.append("JARVIS_STORAGE_FALLBACK_DIR")
    if not google_credentials_status["valid"]:
        attention.append("JARVIS_GOOGLE_CREDS")
    if not google_readonly_token_status["valid"]:
        attention.append("JARVIS_GOOGLE_READONLY_TOKEN")
    if google_token_status["configured"] and not google_token_status["valid"]:
        attention.append("JARVIS_GOOGLE_TOKEN")
    checks.append(
        "- Custom env file validation: "
        + (
            "not set; project .env contents not inspected"
            if not env_file["configured"]
            else "ok; file exists (contents not inspected)"
            if env_file["valid"]
            else "invalid; symlinks are not allowed (path hidden)"
            if env_file["is_symlink"]
            else "invalid; file must be owner-only; run chmod 600 on it (path hidden)"
            if env_file["is_file"] and not env_file["owner_only"]
            else "invalid; path is a directory (path hidden)"
            if env_file["is_dir"]
            else "invalid; parent folder is missing (path hidden)"
            if not env_file["parent_exists"]
            else "invalid; file not found (path hidden)"
        )
    )
    checks.append("- Secret environment values: not inspected; environment-key presence only")
    checks.append(
        "- Ollama destination policy: "
        + (
            "not required; OpenAI provider selected"
            if openai_provider
            else "not evaluated; model provider invalid"
            if not ollama_provider
            else "ok; local loopback accepted"
            if ollama_destination.allowed
            else "needs attention; configured destination is invalid or non-loopback; probe skipped"
        )
        + " | "
        + _ollama_destination_policy_detail(ollama_destination)
    )
    checks.append(
        "- Ollama no-cloud policy: "
        + _ollama_no_cloud_policy_detail(
            ollama_no_cloud,
            required=ollama_provider,
            provider_valid=provider_status["valid"],
        )
    )
    checks.append(
        "- Jarvis data directory validation: "
        + (
            "not set; using default local data dir (path hidden; not created)"
            if not data_dir_target["configured"]
            else "ok; directory exists (contents not scanned)"
            if data_dir_target["is_dir"]
            else "ok; directory can be created later (path hidden; not created)"
            if data_dir_target["valid"]
            else "invalid; target is a file (path hidden)"
            if data_dir_target["is_file"]
            else "invalid; parent folder is missing (path hidden)"
        )
    )
    checks.append(
        "- Jarvis SQLite database path validation: "
        + (
            "not set; using data directory default (path hidden)"
            if not db_path_target["configured"]
            else "ok; target file path usable (contents not read)"
            if db_path_target["valid"]
            else "invalid; path is a directory (path hidden)"
            if db_path_target["is_dir"]
            else "invalid; parent folder is missing (path hidden)"
        )
    )
    checks.append(
        "- Obsidian vault path validation: "
        + (
            "not set; using detected/default vault path (path hidden; not created)"
            if not obsidian_vault_target["configured"]
            else "ok; directory exists (contents not scanned)"
            if obsidian_vault_target["is_dir"]
            else "ok; directory can be created later (path hidden; not created)"
            if obsidian_vault_target["valid"]
            else "invalid; target is a file (path hidden)"
            if obsidian_vault_target["is_file"]
            else "invalid; parent folder is missing (path hidden)"
        )
    )
    checks.append(
        "- Obsidian Jarvis root folder validation: "
        + (
            f"not set; using {DEFAULT_OBSIDIAN_ROOT}"
            if not obsidian_root["configured"]
            else "ok"
            if obsidian_root["valid"]
            else f"invalid; using {DEFAULT_OBSIDIAN_ROOT} (value hidden)"
        )
    )
    checks.append(
        "- Watched directories validation: "
        + (
            "not set; recent-file digest disabled"
            if not watched_dirs["configured"]
            else f"ok; {watched_dirs['existing_dirs']} configured director{'y' if watched_dirs['existing_dirs'] == 1 else 'ies'} available"
            if watched_dirs["valid"]
            else (
                f"invalid; {watched_dirs['missing']} missing, "
                f"{watched_dirs['not_dirs']} not directories "
                "(paths hidden; contents not scanned)"
            )
        )
    )
    checks.append(
        "- Default weather location validation: "
        + (
            "not set; using Seoul"
            if not weather_location["configured"]
            else "ok"
            if weather_location["valid"]
            else "invalid; using Seoul (value hidden)"
        )
    )
    checks.append(
        "- News locale validation: "
        + (
            "not set; using en-US:US"
            if not news_locale["configured"]
            else "ok"
            if news_locale["valid"]
            else "invalid; using en-US:US (value hidden)"
        )
    )
    checks.append(
        "- Telegram bot token presence: "
        + (
            "key present (value not inspected)"
            if telegram_bot_token["present"]
            else "key not present; set TELEGRAM_BOT_TOKEN (value not inspected)"
        )
    )
    checks.append(
        "- Gmail address presence: "
        + (
            "key present (value not inspected)"
            if gmail_address["present"]
            else "key not present; set GMAIL_ADDRESS (value not inspected)"
        )
    )
    checks.append(
        "- Gmail app password presence: "
        + (
            "key present (value not inspected)"
            if gmail_app_password["present"]
            else "key not present; set GMAIL_APP_PASSWORD (value not inspected)"
        )
    )
    checks.append(
        "- OpenAI API key presence: "
        + (
            "not required; Ollama provider selected (value not inspected)"
            if not openai_provider
            else "key present (value not inspected; live access not probed)"
            if openai_api_key["present"]
            else "key not present; set OPENAI_API_KEY locally (value not inspected)"
        )
    )
    if openai_provider:
        checks.append(
            "- OpenAI safety identifier: stable single-owner pseudonym sent with model requests "
            "(value hidden; not derived from personal data or the API key)"
        )
        checks.append(
            "- OpenAI retention boundary: Jarvis sends store=false, but account retention controls "
            "are not checked; default abuse-monitoring logs may retain prompts/responses for up to "
            "30 days unless approved controls apply"
        )
    checks.append(
        "- Telegram owner allowlist validation: "
        + (
            "ok"
            if telegram_owner["valid"] and telegram_owner["configured"]
            else "missing; set JARVIS_OWNER_TELEGRAM (value hidden)"
            if not telegram_owner["configured"]
            else "invalid; value hidden"
        )
    )
    checks.append(
        "- iMessage owner allowlist validation: "
        + (
            "not set; iMessage control disabled"
            if not imessage_owner["configured"]
            else "ok"
            if imessage_owner["valid"]
            else "invalid; value hidden"
        )
    )
    checks.append(
        "- Telegram command state file validation: "
        + (
            "not set; using default local state path"
            if not telegram_state_file["configured"]
            else "ok; target file path usable (contents not read)"
            if telegram_state_file["valid"]
            else "invalid; path is a directory (path hidden)"
            if telegram_state_file["is_dir"]
            else "invalid; parent folder is missing (path hidden)"
        )
    )
    checks.append(
        "- iMessage command state file validation: "
        + (
            "not set; using default local state path"
            if not imessage_state_file["configured"]
            else "ok; target file path usable (contents not read)"
            if imessage_state_file["valid"]
            else "invalid; path is a directory (path hidden)"
            if imessage_state_file["is_dir"]
            else "invalid; parent folder is missing (path hidden)"
        )
    )
    checks.append(
        "- Telegram reminders state file validation: "
        + (
            "not set; using default local state path"
            if not reminders_state_file["configured"]
            else "ok; target file path usable (contents not read)"
            if reminders_state_file["valid"]
            else "invalid; path is a directory (path hidden)"
            if reminders_state_file["is_dir"]
            else "invalid; parent folder is missing (path hidden)"
        )
    )
    checks.append(
        "- Local OAV vision reviewer command validation: "
        + (
            "not set; vision model review held until configured"
            if not oav_vision_reviewer["configured"]
            else "ok; executable resolved without running it"
            if oav_vision_reviewer["valid"]
            else "invalid; executable not resolvable (value hidden)"
            if not oav_vision_reviewer["parse_error"]
            else "invalid; command could not be parsed (value hidden)"
        )
    )
    checks.append(
        "- Local Whisper model path validation: "
        + (
            "not set; audio-file transcription held until configured"
            if not voice_whisper_model["configured"]
            else "ok; model file exists (not loaded)"
            if voice_whisper_model["valid"]
            else "invalid; path is a directory (path hidden)"
            if voice_whisper_model["is_dir"]
            else "invalid; parent folder is missing (path hidden)"
            if not voice_whisper_model["parent_exists"]
            else "invalid; model file not found (path hidden)"
        )
    )
    checks.append(
        "- Local faster-whisper model path validation: "
        + (
            "not set; faster-whisper audio-file transcription held until configured"
            if not voice_faster_whisper_model["configured"]
            else "ok; model directory exists (not loaded)"
            if voice_faster_whisper_model["valid"]
            else "invalid; target is a file (path hidden)"
            if voice_faster_whisper_model["is_file"]
            else "invalid; parent folder is missing (path hidden)"
            if not voice_faster_whisper_model["parent_exists"]
            else "invalid; model directory not found (path hidden)"
        )
    )
    checks.append(
        "- Status dashboard host validation: "
        f"{'ok' if status_host_config.valid else f'invalid; using default {DEFAULT_STATUS_HOST}'} "
        "(value hidden)"
    )
    checks.append(
        "- Status dashboard port validation: "
        f"{'ok' if status_port_config.valid else f'invalid; using default {DEFAULT_STATUS_PORT}'} "
        "(value hidden)"
    )
    checks.append(
        "- Smoke module timeout validation: "
        + (
            "ok"
            if smoke_timeout["valid"]
            else f"invalid; using default {DEFAULT_SMOKE_MODULE_TIMEOUT_SECONDS:g}s"
        )
        + " (value hidden)"
    )
    checks.append(
        "- Planner timeout validation: "
        + _runtime_timeout_validation_detail(planner_timeout, planner_timeout_default)
        + " (value hidden)"
    )
    checks.append(
        "- Chat timeout validation: "
        + _runtime_timeout_validation_detail(chat_timeout, chat_timeout_default)
        + " (value hidden)"
    )
    checks.append(
        "- Chat reply token cap validation: "
        + _runtime_int_validation_detail(
            chat_max_reply_tokens, DEFAULT_CHAT_MAX_REPLY_TOKENS, MAX_CHAT_MAX_REPLY_TOKENS
        )
        + " (value hidden)"
    )
    checks.append(
        "- Chat history window validation: "
        + _runtime_int_validation_detail(
            chat_max_history_messages,
            DEFAULT_CHAT_MAX_HISTORY_MESSAGES,
            MAX_CHAT_MAX_HISTORY_MESSAGES,
        )
        + " (value hidden)"
    )

    checks.append(
        "- Model provider validation: "
        + (
            f"not set; using {provider_status['effective_value']}"
            if not provider_status["configured"]
            else f"ok; using {provider_status['effective_value']}"
            if provider_status["valid"]
            else f"invalid; using {provider_status['fallback_value']} (value hidden)"
        )
    )
    checks.append(
        "- Chat reasoning effort validation: "
        + (
            f"not set; using {chat_reasoning['effective_value']}"
            if not chat_reasoning["configured"]
            else f"ok; using {chat_reasoning['effective_value']}"
            if chat_reasoning["valid"]
            else f"invalid; using {chat_reasoning['fallback_value']} (value hidden)"
        )
    )
    checks.append(
        "- Planner reasoning effort validation: "
        + (
            f"not set; using {planner_reasoning['effective_value']}"
            if not planner_reasoning["configured"]
            else f"ok; using {planner_reasoning['effective_value']}"
            if planner_reasoning["valid"]
            else f"invalid; using {planner_reasoning['fallback_value']} (value hidden)"
        )
    )
    checks.append(
        "- Fallback Ollama model alias validation: "
        + (
            "not required; OpenAI provider selected"
            if openai_provider
            else
            f"not set; using {DEFAULT_MODEL_ALIAS}"
            if not fallback_model_alias["configured"]
            else "ok"
            if fallback_model_alias["valid"]
            else f"invalid; using {DEFAULT_MODEL_ALIAS} (value hidden)"
        )
    )
    checks.append(
        "- Chat model alias validation: "
        + (
            "not set; using OpenAI provider default"
            if openai_provider and not chat_model_alias["configured"]
            else "not set; using fallback Ollama model alias"
            if not chat_model_alias["configured"]
            else "ok"
            if chat_model_alias["valid"]
            else "invalid; using fallback model alias (value hidden)"
        )
    )
    checks.append(
        "- Planner model alias validation: "
        + (
            "not set; using OpenAI provider default"
            if openai_provider and not planner_model_alias["configured"]
            else "not set; using chat model alias"
            if not planner_model_alias["configured"]
            else "ok"
            if planner_model_alias["valid"]
            else "invalid; using chat model alias (value hidden)"
        )
    )
    checks.append(
        "- Model planner toggle validation: "
        + (
            "not set; enabled by default"
            if not model_planner_toggle["configured"]
            else "ok; enabled"
            if model_planner_toggle["valid"] and model_planner_toggle["effective_enabled"]
            else "ok; disabled"
            if model_planner_toggle["valid"]
            else "invalid; enabled by default (value hidden)"
        )
    )
    checks.append(
        "- Remote interactive personal-context validation: "
        + (
            "not set; stored profile, preferences, memory, skills, and prior chat history stay local by default"
            if not remote_personal_context_toggle["configured"]
            else "ok; enabled for ordinary OpenAI chat"
            if remote_personal_context_toggle["valid"] and remote_personal_context_toggle["effective_enabled"]
            else "ok; disabled; current OpenAI chat message may still be sent"
            if remote_personal_context_toggle["valid"]
            else "invalid; stored personal context stays local by default (value hidden)"
        )
    )
    checks.append(
        "- Remote conversation compaction validation: "
        + (
            "not set; remote history summarization disabled by default"
            if not remote_compaction_toggle["configured"]
            else "ok; enabled for the selected remote model"
            if remote_compaction_toggle["valid"] and remote_compaction_toggle["effective_enabled"]
            else "ok; disabled"
            if remote_compaction_toggle["valid"]
            else "invalid; remote history summarization disabled (value hidden)"
        )
    )
    checks.append(
        "- Google credentials file (JARVIS_GOOGLE_CREDS): "
        + (
            "present"
            if google_credentials_status["valid"]
            else "invalid; path is a directory (path hidden)"
            if google_credentials_status["is_dir"]
            else "not found; parent folder missing (path hidden)"
            if not google_credentials_status["parent_exists"]
            else "not found"
        )
        + " (contents not read)"
    )
    checks.append(
        "- Google Calendar read-only token file (JARVIS_GOOGLE_READONLY_TOKEN): "
        + (
            "present; owner-only"
            if google_readonly_token_status["valid"]
            else "invalid; symlinks are not allowed (path hidden)"
            if google_readonly_token_status["is_symlink"]
            else "invalid; run chmod 600 on the file (path hidden)"
            if google_readonly_token_status["is_file"] and not google_readonly_token_status["owner_only"]
            else "invalid; path is a directory (path hidden)"
            if google_readonly_token_status["is_dir"]
            else "not found; parent folder missing (path hidden)"
            if not google_readonly_token_status["parent_exists"]
            else "not found; Calendar reads are disconnected"
        )
        + " (contents not read)"
    )
    checks.append(
        "- Google Calendar full-access mutation token file (JARVIS_GOOGLE_TOKEN): "
        + (
            "present"
            if google_token_status["valid"]
            else "invalid; path is a directory (path hidden)"
            if google_token_status["is_dir"]
            else "not found; parent folder missing (path hidden)"
            if not google_token_status["parent_exists"]
            else "not configured; Calendar writes remain disabled"
        )
        + " (contents not read)"
    )

    checks.append(
        "- Storage fallback (JARVIS_DISABLE_STORAGE_FALLBACK): "
        + (
            "not set; fallback enabled by default"
            if not storage_fallback["disable_configured"]
            else "ok; fallback disabled"
            if storage_fallback["disable_valid"] and storage_fallback["disabled"]
            else "ok; fallback enabled"
            if storage_fallback["disable_valid"]
            else "invalid; fallback enabled by default (value hidden)"
        )
        + ("" if not storage_fallback["disable_valid"] else " (value hidden)")
    )
    checks.append(
        "- Storage fallback directory parent: "
        f"{'writable' if storage_fallback['parent_writable'] else 'not writable or missing'} "
        "(path hidden; not created)"
    )
    checks.append(
        "- Storage fallback directory target: "
        + (
            "disabled; not checked"
            if storage_fallback["disabled"]
            else "ok; target directory already exists (contents not scanned)"
            if storage_fallback["is_dir"]
            else "ok; target can be created later (not created)"
            if storage_fallback["valid"]
            else "invalid; target is a file (path hidden)"
            if storage_fallback["is_file"]
            else "invalid; parent folder is missing or not writable (path hidden)"
        )
    )

    app_checks = [
        ("KakaoTalk", Path("/Applications/KakaoTalk.app")),
        ("Google Chrome", Path("/Applications/Google Chrome.app")),
    ]
    for app_name, app_path in app_checks:
        available = app_path.exists()
        if not available:
            attention.append(app_name)
        checks.append(f"- {app_name} app: {'present' if available else 'not found'} (not launched)")

    checks.append("- Accessibility permission for GUI messaging: manual check required (not probed)")
    v3_shell_prefix = 'JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env"'
    next_commands = [
        f"{v3_shell_prefix} ./launch_jarvis_v3_dashboard.py",
        f'{v3_shell_prefix} ./launch_jarvis_v3.py "status dashboard"',
        f'{v3_shell_prefix} ./launch_jarvis_v3.py "jarvis doctor"',
        f'{v3_shell_prefix} ./launch_jarvis_v3.py "prototype readiness"',
        f'{v3_shell_prefix} ./launch_jarvis_v3.py "voice setup check"',
        f'{v3_shell_prefix} ./launch_jarvis_v3.py "computer control status"',
        f'{v3_shell_prefix} ./launch_jarvis_v3.py "readiness report"',
    ]
    if google_credentials_status["valid"] and not google_readonly_token_status["valid"]:
        next_commands.append(GOOGLE_CALENDAR_READONLY_AUTH_COMMAND)
    lines = [
        "Jarvis V3 setup check:",
        *checks,
        "",
        "Safe next Terminal commands (run from the Jarvis V3 project folder "
        f"{V3_CALENDAR_AUTH_TERMINAL_LOCATION}):",
    ]
    lines.extend(f"- `{command}`" for command in next_commands)
    lines.extend(["", "System boundary:", f"- {OPERATOR_LIMIT_RULE}"])

    return ToolResult(
        "setup_check",
        True,
        "\n".join(lines),
        _safe_metadata(
            checks=len(checks),
            missing=missing,
            setup_attention=attention,
            next_commands=next_commands,
            reads_clipboard=False,
            reads_environment=True,
            reads_secret_values=False,
            inspects_secret_values=False,
            secret_key_presence_only=True,
            env_configured=env_configured,
            model_provider=model_provider,
            model_provider_configured=provider_status["configured"],
            model_provider_valid=provider_status["valid"],
            model_provider_source=provider_status["source"],
            model_provider_fallback=provider_status["fallback_value"],
            ollama_required=ollama_provider,
            openai_api_key_configured=openai_api_key["configured"],
            openai_api_key_valid=openai_api_key["valid"],
            openai_api_key_required=openai_api_key["required"],
            openai_api_key_value_inspected=openai_api_key["value_inspected"],
            openai_api_key_validation_performed=openai_api_key["validation_performed"],
            openai_api_key_value_exposed=False,
            openai_safety_identifier_sent=openai_provider,
            openai_safety_identifier_scope="single_owner" if openai_provider else "not_applicable",
            openai_safety_identifier_value_exposed=False,
            openai_safety_identifier_uses_personal_data=False,
            openai_safety_identifier_uses_api_key=False,
            openai_live_access_probed=False,
            openai_request_store_flag=False,
            openai_account_retention_controls_checked=False,
            openai_zero_data_retention_verified=False,
            openai_default_abuse_monitoring_may_retain_content=openai_provider,
            openai_default_abuse_monitoring_max_days=30 if openai_provider else 0,
            chat_reasoning_effort=chat_reasoning["effective_value"],
            chat_reasoning_effort_valid=chat_reasoning["valid"],
            planner_reasoning_effort=planner_reasoning["effective_value"],
            planner_reasoning_effort_valid=planner_reasoning["valid"],
            env_file_configured=env_file["configured"],
            env_file_exists=env_file["exists"],
            env_file_is_file=env_file["is_file"],
            env_file_is_dir=env_file["is_dir"],
            env_file_parent_exists=env_file["parent_exists"],
            env_file_is_symlink=env_file["is_symlink"],
            env_file_owner_matches=env_file["owner_matches"],
            env_file_owner_only=env_file["owner_only"],
            env_file_valid=env_file["valid"],
            env_file_source=env_file["source"],
            reads_env_file_contents=False,
            data_dir_configured=data_dir_target["configured"],
            data_dir_exists=data_dir_target["exists"],
            data_dir_is_dir=data_dir_target["is_dir"],
            data_dir_is_file=data_dir_target["is_file"],
            data_dir_parent_exists=data_dir_target["parent_exists"],
            data_dir_valid=data_dir_target["valid"],
            data_dir_source=data_dir_target["source"],
            creates_data_dir=False,
            scans_data_dir=False,
            db_path_configured=db_path_target["configured"],
            db_path_exists=db_path_target["exists"],
            db_path_is_file=db_path_target["is_file"],
            db_path_is_dir=db_path_target["is_dir"],
            db_path_parent_exists=db_path_target["parent_exists"],
            db_path_valid=db_path_target["valid"],
            db_path_source=db_path_target["source"],
            reads_db_file_contents=False,
            creates_db_file=False,
            obsidian_vault_configured=obsidian_vault_target["configured"],
            obsidian_vault_env_key=obsidian_vault_target["env_key"],
            obsidian_vault_exists=obsidian_vault_target["exists"],
            obsidian_vault_is_dir=obsidian_vault_target["is_dir"],
            obsidian_vault_is_file=obsidian_vault_target["is_file"],
            obsidian_vault_parent_exists=obsidian_vault_target["parent_exists"],
            obsidian_vault_valid=obsidian_vault_target["valid"],
            obsidian_vault_source=obsidian_vault_target["source"],
            creates_obsidian_vault=False,
            scans_obsidian_vault=False,
            obsidian_root_configured=obsidian_root["configured"],
            obsidian_root_valid=obsidian_root["valid"],
            obsidian_root_source=obsidian_root["source"],
            obsidian_root_chars=obsidian_root["root_chars"],
            obsidian_root_truncated=obsidian_root["root_truncated"],
            obsidian_root_fallback=obsidian_root["fallback_root"],
            watched_dirs_configured=watched_dirs["configured"],
            watched_dirs_count=watched_dirs["count"],
            watched_dirs_existing_dirs=watched_dirs["existing_dirs"],
            watched_dirs_missing=watched_dirs["missing"],
            watched_dirs_not_dirs=watched_dirs["not_dirs"],
            watched_dirs_valid=watched_dirs["valid"],
            watched_dirs_source=watched_dirs["source"],
            scans_watched_dirs=False,
            weather_location_configured=weather_location["configured"],
            weather_location_valid=weather_location["valid"],
            weather_location_source=weather_location["source"],
            weather_location_chars=weather_location["location_chars"],
            weather_location_truncated=weather_location["location_truncated"],
            weather_location_fallback=weather_location["fallback_location"],
            news_locale_configured=news_locale["configured"],
            news_locale_valid=news_locale["valid"],
            news_locale_source=news_locale["source"],
            news_locale_raw_chars=news_locale["raw_chars"],
            news_locale_raw_truncated=news_locale["raw_truncated"],
            news_locale_fallback_hl=news_locale["fallback_hl"],
            news_locale_fallback_gl=news_locale["fallback_gl"],
            telegram_bot_token_configured=telegram_bot_token["configured"],
            telegram_bot_token_valid=telegram_bot_token["valid"],
            telegram_bot_token_source=telegram_bot_token["source"],
            telegram_bot_token_required=telegram_bot_token["required"],
            telegram_bot_token_value_inspected=telegram_bot_token["value_inspected"],
            telegram_bot_token_validation_performed=telegram_bot_token["validation_performed"],
            gmail_address_configured=gmail_address["configured"],
            gmail_address_valid=gmail_address["valid"],
            gmail_address_source=gmail_address["source"],
            gmail_address_required=gmail_address["required"],
            gmail_address_value_inspected=gmail_address["value_inspected"],
            gmail_address_validation_performed=gmail_address["validation_performed"],
            gmail_app_password_configured=gmail_app_password["configured"],
            gmail_app_password_valid=gmail_app_password["valid"],
            gmail_app_password_source=gmail_app_password["source"],
            gmail_app_password_required=gmail_app_password["required"],
            gmail_app_password_value_inspected=gmail_app_password["value_inspected"],
            gmail_app_password_validation_performed=gmail_app_password["validation_performed"],
            telegram_owner_allowlist_configured=telegram_owner["configured"],
            telegram_owner_allowlist_valid=telegram_owner["valid"],
            telegram_owner_allowlist_source=telegram_owner["source"],
            telegram_owner_allowlist_required=telegram_owner["required"],
            telegram_owner_allowlist_raw_chars=telegram_owner["raw_chars"],
            telegram_owner_allowlist_raw_truncated=telegram_owner["raw_truncated"],
            imessage_owner_allowlist_configured=imessage_owner["configured"],
            imessage_owner_allowlist_valid=imessage_owner["valid"],
            imessage_owner_allowlist_source=imessage_owner["source"],
            imessage_owner_allowlist_required=imessage_owner["required"],
            imessage_owner_allowlist_raw_chars=imessage_owner["raw_chars"],
            imessage_owner_allowlist_raw_truncated=imessage_owner["raw_truncated"],
            telegram_state_file_configured=telegram_state_file["configured"],
            telegram_state_file_exists=telegram_state_file["exists"],
            telegram_state_file_is_file=telegram_state_file["is_file"],
            telegram_state_file_is_dir=telegram_state_file["is_dir"],
            telegram_state_file_parent_exists=telegram_state_file["parent_exists"],
            telegram_state_file_valid=telegram_state_file["valid"],
            telegram_state_file_source=telegram_state_file["source"],
            reads_telegram_state_file_contents=False,
            creates_telegram_state_file=False,
            imessage_state_file_configured=imessage_state_file["configured"],
            imessage_state_file_exists=imessage_state_file["exists"],
            imessage_state_file_is_file=imessage_state_file["is_file"],
            imessage_state_file_is_dir=imessage_state_file["is_dir"],
            imessage_state_file_parent_exists=imessage_state_file["parent_exists"],
            imessage_state_file_valid=imessage_state_file["valid"],
            imessage_state_file_source=imessage_state_file["source"],
            reads_imessage_state_file_contents=False,
            creates_imessage_state_file=False,
            reminders_state_file_configured=reminders_state_file["configured"],
            reminders_state_file_exists=reminders_state_file["exists"],
            reminders_state_file_is_file=reminders_state_file["is_file"],
            reminders_state_file_is_dir=reminders_state_file["is_dir"],
            reminders_state_file_parent_exists=reminders_state_file["parent_exists"],
            reminders_state_file_valid=reminders_state_file["valid"],
            reminders_state_file_source=reminders_state_file["source"],
            reads_reminders_state_file_contents=False,
            creates_reminders_state_file=False,
            oav_vision_reviewer_command_configured=oav_vision_reviewer["configured"],
            oav_vision_reviewer_command_valid=oav_vision_reviewer["valid"],
            oav_vision_reviewer_command_source=oav_vision_reviewer["source"],
            oav_vision_reviewer_command_required=oav_vision_reviewer["required"],
            oav_vision_reviewer_command_raw_chars=oav_vision_reviewer["raw_chars"],
            oav_vision_reviewer_command_raw_truncated=oav_vision_reviewer["raw_truncated"],
            oav_vision_reviewer_command_token_count=oav_vision_reviewer["token_count"],
            oav_vision_reviewer_command_executable_resolved=oav_vision_reviewer["executable_resolved"],
            oav_vision_reviewer_command_executable_kind=oav_vision_reviewer["executable_kind"],
            oav_vision_reviewer_command_executable_exists=oav_vision_reviewer["executable_exists"],
            oav_vision_reviewer_command_executable_is_file=oav_vision_reviewer["executable_is_file"],
            oav_vision_reviewer_command_executable_is_dir=oav_vision_reviewer["executable_is_dir"],
            oav_vision_reviewer_command_executable_parent_exists=oav_vision_reviewer["executable_parent_exists"],
            oav_vision_reviewer_command_executable_on_path=oav_vision_reviewer["executable_on_path"],
            oav_vision_reviewer_command_executable_is_executable=oav_vision_reviewer["executable_is_executable"],
            oav_vision_reviewer_command_parse_error=oav_vision_reviewer["parse_error"],
            executes_oav_vision_reviewer_command=False,
            voice_whisper_model_path_configured=voice_whisper_model["configured"],
            voice_whisper_model_path_valid=voice_whisper_model["valid"],
            voice_whisper_model_path_source=voice_whisper_model["source"],
            voice_whisper_model_path_required=voice_whisper_model["required"],
            voice_whisper_model_path_expected=voice_whisper_model["expected"],
            voice_whisper_model_path_raw_chars=voice_whisper_model["raw_chars"],
            voice_whisper_model_path_raw_truncated=voice_whisper_model["raw_truncated"],
            voice_whisper_model_path_exists=voice_whisper_model["exists"],
            voice_whisper_model_path_is_file=voice_whisper_model["is_file"],
            voice_whisper_model_path_is_dir=voice_whisper_model["is_dir"],
            voice_whisper_model_path_parent_exists=voice_whisper_model["parent_exists"],
            voice_faster_whisper_model_path_configured=voice_faster_whisper_model["configured"],
            voice_faster_whisper_model_path_valid=voice_faster_whisper_model["valid"],
            voice_faster_whisper_model_path_source=voice_faster_whisper_model["source"],
            voice_faster_whisper_model_path_required=voice_faster_whisper_model["required"],
            voice_faster_whisper_model_path_expected=voice_faster_whisper_model["expected"],
            voice_faster_whisper_model_path_raw_chars=voice_faster_whisper_model["raw_chars"],
            voice_faster_whisper_model_path_raw_truncated=voice_faster_whisper_model["raw_truncated"],
            voice_faster_whisper_model_path_exists=voice_faster_whisper_model["exists"],
            voice_faster_whisper_model_path_is_file=voice_faster_whisper_model["is_file"],
            voice_faster_whisper_model_path_is_dir=voice_faster_whisper_model["is_dir"],
            voice_faster_whisper_model_path_parent_exists=voice_faster_whisper_model["parent_exists"],
            loads_voice_whisper_model=False,
            loads_voice_faster_whisper_model=False,
            reads_audio_for_voice_model_validation=False,
            status_dashboard_host_configured=status_host_config.configured,
            status_dashboard_host_valid=status_host_config.valid,
            status_dashboard_host_source=status_host_config.source,
            status_dashboard_host_default=DEFAULT_STATUS_HOST,
            status_dashboard_port_configured=env_configured["JARVIS_STATUS_PORT"],
            status_dashboard_port_valid=status_port_config.valid,
            status_dashboard_port_source=status_port_config.source,
            status_dashboard_port_default=DEFAULT_STATUS_PORT,
            status_dashboard_auth_token_configured=env_configured["JARVIS_STATUS_AUTH_TOKEN"],
            status_dashboard_auth_token_value_inspected=False,
            status_dashboard_auth_token_validation_performed=False,
            planner_timeout_configured=planner_timeout["configured"],
            planner_timeout_valid=planner_timeout["valid"],
            planner_timeout_source=planner_timeout["source"],
            planner_timeout_default=planner_timeout_default,
            planner_timeout_minimum=MIN_MODEL_TIMEOUT_SECONDS,
            planner_timeout_maximum=planner_timeout_maximum,
            planner_timeout_effective_seconds=planner_timeout["effective_seconds"],
            chat_timeout_configured=chat_timeout["configured"],
            chat_timeout_valid=chat_timeout["valid"],
            chat_timeout_source=chat_timeout["source"],
            chat_timeout_default=chat_timeout_default,
            chat_timeout_minimum=MIN_CHAT_TIMEOUT_SECONDS,
            chat_timeout_maximum=chat_timeout_maximum,
            chat_timeout_effective_seconds=chat_timeout["effective_seconds"],
            chat_max_reply_tokens_configured=chat_max_reply_tokens["configured"],
            chat_max_reply_tokens_valid=chat_max_reply_tokens["valid"],
            chat_max_reply_tokens_source=chat_max_reply_tokens["source"],
            chat_max_reply_tokens_default=DEFAULT_CHAT_MAX_REPLY_TOKENS,
            chat_max_reply_tokens_minimum=MIN_CHAT_MAX_REPLY_TOKENS,
            chat_max_reply_tokens_maximum=MAX_CHAT_MAX_REPLY_TOKENS,
            chat_max_reply_tokens_effective=chat_max_reply_tokens["effective_value"],
            chat_max_history_messages_configured=chat_max_history_messages["configured"],
            chat_max_history_messages_valid=chat_max_history_messages["valid"],
            chat_max_history_messages_source=chat_max_history_messages["source"],
            chat_max_history_messages_default=DEFAULT_CHAT_MAX_HISTORY_MESSAGES,
            chat_max_history_messages_minimum=MIN_CHAT_MAX_HISTORY_MESSAGES,
            chat_max_history_messages_maximum=MAX_CHAT_MAX_HISTORY_MESSAGES,
            chat_max_history_messages_effective=chat_max_history_messages["effective_value"],
            smoke_module_timeout_configured=smoke_timeout["configured"],
            smoke_module_timeout_valid=smoke_timeout["valid"],
            smoke_module_timeout_source=smoke_timeout["source"],
            smoke_module_timeout_default=DEFAULT_SMOKE_MODULE_TIMEOUT_SECONDS,
            smoke_module_timeout_effective_seconds=smoke_timeout["effective_seconds"],
            fallback_model_alias_configured=fallback_model_alias["configured"],
            fallback_model_alias_valid=fallback_model_alias["valid"],
            fallback_model_alias_source=fallback_model_alias["source"],
            fallback_model_alias_chars=fallback_model_alias["alias_chars"],
            fallback_model_alias_truncated=fallback_model_alias["alias_truncated"],
            fallback_model_alias_fallback=fallback_model_alias["fallback_alias"],
            chat_model_alias_configured=chat_model_alias["configured"],
            chat_model_alias_valid=chat_model_alias["valid"],
            chat_model_alias_source=chat_model_alias["source"],
            chat_model_alias_chars=chat_model_alias["alias_chars"],
            chat_model_alias_truncated=chat_model_alias["alias_truncated"],
            chat_model_alias_fallback=chat_model_alias["fallback_alias"],
            planner_model_alias_configured=planner_model_alias["configured"],
            planner_model_alias_valid=planner_model_alias["valid"],
            planner_model_alias_source=planner_model_alias["source"],
            planner_model_alias_chars=planner_model_alias["alias_chars"],
            planner_model_alias_truncated=planner_model_alias["alias_truncated"],
            planner_model_alias_fallback=planner_model_alias["fallback_alias"],
            model_planner_toggle_configured=model_planner_toggle["configured"],
            model_planner_toggle_valid=model_planner_toggle["valid"],
            model_planner_toggle_source=model_planner_toggle["source"],
            model_planner_toggle_effective_enabled=model_planner_toggle["effective_enabled"],
            model_planner_toggle_fallback_enabled=model_planner_toggle["fallback_enabled"],
            remote_compaction_toggle_configured=remote_compaction_toggle["configured"],
            remote_compaction_toggle_valid=remote_compaction_toggle["valid"],
            remote_compaction_toggle_source=remote_compaction_toggle["source"],
            remote_conversation_compaction_enabled=remote_compaction_toggle["effective_enabled"],
            remote_compaction_sends_history_to_external_model=bool(
                openai_provider and remote_compaction_toggle["effective_enabled"]
            ),
            remote_compaction_live_access_probed=False,
            remote_personal_context_toggle_configured=remote_personal_context_toggle["configured"],
            remote_personal_context_toggle_valid=remote_personal_context_toggle["valid"],
            remote_personal_context_toggle_source=remote_personal_context_toggle["source"],
            remote_personal_context_allowed=remote_personal_context_toggle["effective_enabled"],
            remote_personal_context_sends_stored_context_to_external_model=bool(
                openai_provider and remote_personal_context_toggle["effective_enabled"]
            ),
            remote_personal_context_current_message_may_be_external=bool(openai_provider),
            remote_personal_context_content_in_metadata=False,
            google_credentials_present=google_credentials_status["valid"],
            google_credentials_configured=google_credentials_status["configured"],
            google_credentials_exists=google_credentials_status["exists"],
            google_credentials_is_file=google_credentials_status["is_file"],
            google_credentials_is_dir=google_credentials_status["is_dir"],
            google_credentials_parent_exists=google_credentials_status["parent_exists"],
            google_credentials_valid=google_credentials_status["valid"],
            google_credentials_source=google_credentials_status["source"],
            google_readonly_token_present=google_readonly_token_status["valid"],
            google_readonly_token_configured=google_readonly_token_status["configured"],
            google_readonly_token_exists=google_readonly_token_status["exists"],
            google_readonly_token_is_file=google_readonly_token_status["is_file"],
            google_readonly_token_is_dir=google_readonly_token_status["is_dir"],
            google_readonly_token_is_symlink=google_readonly_token_status["is_symlink"],
            google_readonly_token_owner_only=google_readonly_token_status["owner_only"],
            google_readonly_token_parent_exists=google_readonly_token_status["parent_exists"],
            google_readonly_token_valid=google_readonly_token_status["valid"],
            google_readonly_token_source=google_readonly_token_status["source"],
            google_token_present=google_token_status["valid"],
            google_token_configured=google_token_status["configured"],
            google_token_exists=google_token_status["exists"],
            google_token_is_file=google_token_status["is_file"],
            google_token_is_dir=google_token_status["is_dir"],
            google_token_parent_exists=google_token_status["parent_exists"],
            google_token_valid=google_token_status["valid"],
            google_token_source=google_token_status["source"],
            google_connector_dependencies=google_dependency_status,
            reads_google_credentials=False,
            storage_fallback_disabled=storage_fallback["disabled"],
            storage_fallback_disable_configured=storage_fallback["disable_configured"],
            storage_fallback_disable_valid=storage_fallback["disable_valid"],
            storage_fallback_disable_source=storage_fallback["disable_source"],
            storage_fallback_disable_chars=storage_fallback["disable_raw_chars"],
            storage_fallback_disable_truncated=storage_fallback["disable_raw_truncated"],
            storage_fallback_disable_fallback_disabled=storage_fallback["disable_fallback_disabled"],
            storage_fallback_configured=storage_fallback["configured"],
            storage_fallback_exists=storage_fallback["exists"],
            storage_fallback_is_dir=storage_fallback["is_dir"],
            storage_fallback_is_file=storage_fallback["is_file"],
            storage_fallback_parent_exists=storage_fallback["parent_exists"],
            storage_fallback_parent_writable=storage_fallback["parent_writable"],
            storage_fallback_valid=storage_fallback["valid"],
            storage_fallback_source=storage_fallback["source"],
            creates_storage_fallback_dir=False,
            scans_storage_fallback_dir=False,
            launches_apps=False,
            ollama_destination_probe_attempted=False,
            ollama_probe_uses_explicit_loopback_env=False,
            ollama_probe_proxy_environment_stripped=False,
            ollama_probe_uses_validated_loopback_http=False,
            ollama_probe_redirects_blocked=False,
            ollama_probe_proxy_bypassed=False,
            ollama_probe_uses_subprocess=False,
            ollama_probe_diagnostic="ollama_probe_not_attempted",
            ollama_probe_model_count=0,
            ollama_daemon_cloud_disabled_verified=False,
            ollama_on_device_model_execution_verified=False,
            ollama_cloud_features_disabled_verified=False,
            **ollama_destination.receipt(),
            **ollama_no_cloud.receipt(),
        ),
    )
