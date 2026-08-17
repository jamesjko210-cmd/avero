from __future__ import annotations

import os
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
DEFAULT_OBSIDIAN_ROOT = "Jarvis"
DEFAULT_MODEL_ALIAS = "llama3.1"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-5.6-terra"
DEFAULT_OPENAI_PLANNER_MODEL = "gpt-5.6-luna"
DEFAULT_OPENAI_MAX_OUTPUT_TOKENS = 25_000
MIN_OPENAI_MAX_OUTPUT_TOKENS = 1_000
MAX_OPENAI_MAX_OUTPUT_TOKENS = 128_000
DEFAULT_CHAT_MAX_REPLY_TOKENS = 300
MIN_CHAT_MAX_REPLY_TOKENS = 40
MAX_CHAT_MAX_REPLY_TOKENS = 4_096
DEFAULT_CHAT_MAX_HISTORY_MESSAGES = 16
MIN_CHAT_MAX_HISTORY_MESSAGES = 0
MAX_CHAT_MAX_HISTORY_MESSAGES = 64
MAX_OLLAMA_MODEL_TIMEOUT_SECONDS = 60.0
MAX_OPENAI_MODEL_TIMEOUT_SECONDS = 180.0
MAX_OLLAMA_CHAT_TIMEOUT_SECONDS = 300.0
MAX_OPENAI_CHAT_TIMEOUT_SECONDS = 600.0
MODEL_ALIAS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
MODEL_PROVIDERS = {"ollama", "openai"}
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}
DEFAULT_VAULTS = [
    Path("~/.jarvis_v3/Vault"),
]


@dataclass(frozen=True)
class JarvisConfig:
    data_dir: Path
    db_path: Path
    obsidian_vault: Path
    obsidian_root: str = "Jarvis"
    chat_model: str = "llama3.1"
    planner_model: str = "llama3.1"
    model_timeout_seconds: float = 2.5
    chat_timeout_seconds: float = 20.0
    chat_max_reply_tokens: int = DEFAULT_CHAT_MAX_REPLY_TOKENS
    chat_max_history_messages: int = DEFAULT_CHAT_MAX_HISTORY_MESSAGES
    use_model_planner: bool = True
    model_provider: str = DEFAULT_MODEL_PROVIDER
    chat_reasoning_effort: str = "medium"
    planner_reasoning_effort: str = "low"
    openai_max_output_tokens: int = DEFAULT_OPENAI_MAX_OUTPUT_TOKENS
    allow_remote_conversation_compaction: bool = False
    allow_remote_personal_context: bool = False
    watched_dirs: tuple[Path, ...] = ()


def detect_obsidian_vault() -> Path:
    env_path = (os.getenv("JARVIS_OBSIDIAN_VAULT") or "").strip()
    if not env_path:
        env_path = (os.getenv("OBSIDIAN_VAULT_PATH") or "").strip()
    if env_path:
        return Path(env_path).expanduser()

    expanded_defaults = [vault.expanduser() for vault in DEFAULT_VAULTS]
    for vault in expanded_defaults:
        if (vault / ".obsidian").exists():
            return vault

    return expanded_defaults[0]


def _finite_timeout_from_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if not isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


def _int_from_env(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value)


def _bounded_int_from_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _obsidian_root_from_env() -> str:
    root = _env_text("JARVIS_OBSIDIAN_ROOT", DEFAULT_OBSIDIAN_ROOT)
    if (
        LOCAL_PATH_RE.search(root)
        or Path(root).is_absolute()
        or "/" in root
        or "\\" in root
        or root in {".", ".."}
    ):
        return DEFAULT_OBSIDIAN_ROOT
    return root


def _valid_model_alias(value: str) -> bool:
    return (
        bool(value)
        and not LOCAL_PATH_RE.search(value)
        and not Path(value).is_absolute()
        and "/" not in value
        and "\\" not in value
        and bool(MODEL_ALIAS_RE.fullmatch(value))
    )


def _model_alias_from_env(name: str, default: str) -> str:
    alias = _env_text(name, default)
    if _valid_model_alias(alias):
        return alias
    if _valid_model_alias(default):
        return default
    return DEFAULT_MODEL_ALIAS


def _choice_from_env(name: str, default: str, choices: set[str]) -> str:
    value = _env_text(name, default).lower()
    return value if value in choices else default


def _model_provider_from_env() -> str:
    raw = os.getenv("JARVIS_MODEL_PROVIDER")
    if raw is None or not raw.strip():
        return DEFAULT_MODEL_PROVIDER
    value = raw.strip().lower()
    return value if value in MODEL_PROVIDERS else "invalid"


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:
        return default
    if value in TRUE_ENV_VALUES:
        return True
    if value in FALSE_ENV_VALUES:
        return False
    return default


def load_config() -> JarvisConfig:
    from jarvis_v2.env import load_env
    load_env()
    data_dir = Path(_env_text("JARVIS_DATA_DIR", "~/.jarvis_v3")).expanduser()
    db_path = Path(_env_text("JARVIS_DB_PATH", str(data_dir / "jarvis.sqlite"))).expanduser()
    root = _obsidian_root_from_env()
    model_provider = _model_provider_from_env()
    if model_provider == "openai":
        chat_model = _model_alias_from_env("JARVIS_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL)
        planner_model = _model_alias_from_env("JARVIS_PLANNER_MODEL", DEFAULT_OPENAI_PLANNER_MODEL)
        default_planner_timeout = 12.0
        default_chat_timeout = 60.0
        maximum_planner_timeout = MAX_OPENAI_MODEL_TIMEOUT_SECONDS
        maximum_chat_timeout = MAX_OPENAI_CHAT_TIMEOUT_SECONDS
    else:
        ollama_model = _model_alias_from_env("OLLAMA_MODEL", DEFAULT_MODEL_ALIAS)
        chat_model = _model_alias_from_env("JARVIS_CHAT_MODEL", ollama_model)
        planner_model = _model_alias_from_env("JARVIS_PLANNER_MODEL", chat_model)
        default_planner_timeout = 2.5
        default_chat_timeout = 20.0
        maximum_planner_timeout = MAX_OLLAMA_MODEL_TIMEOUT_SECONDS
        maximum_chat_timeout = MAX_OLLAMA_CHAT_TIMEOUT_SECONDS
    model_timeout_seconds = _finite_timeout_from_env(
        "JARVIS_MODEL_TIMEOUT_SECONDS", default_planner_timeout, 0.5, maximum_planner_timeout
    )
    chat_timeout_seconds = _finite_timeout_from_env(
        "JARVIS_CHAT_TIMEOUT_SECONDS", default_chat_timeout, 1.0, maximum_chat_timeout
    )
    chat_max_reply_tokens = _bounded_int_from_env(
        "JARVIS_CHAT_MAX_REPLY_TOKENS",
        DEFAULT_CHAT_MAX_REPLY_TOKENS,
        MIN_CHAT_MAX_REPLY_TOKENS,
        MAX_CHAT_MAX_REPLY_TOKENS,
    )
    chat_max_history_messages = _bounded_int_from_env(
        "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
        DEFAULT_CHAT_MAX_HISTORY_MESSAGES,
        MIN_CHAT_MAX_HISTORY_MESSAGES,
        MAX_CHAT_MAX_HISTORY_MESSAGES,
    )
    use_model_planner = _bool_from_env("JARVIS_USE_MODEL_PLANNER", True)
    default_chat_reasoning_effort = "low" if model_provider == "openai" else "medium"
    chat_reasoning_effort = _choice_from_env(
        "JARVIS_CHAT_REASONING_EFFORT", default_chat_reasoning_effort, REASONING_EFFORTS
    )
    planner_reasoning_effort = _choice_from_env(
        "JARVIS_PLANNER_REASONING_EFFORT", "low", REASONING_EFFORTS
    )
    openai_max_output_tokens = _bounded_int_from_env(
        "JARVIS_OPENAI_MAX_OUTPUT_TOKENS",
        DEFAULT_OPENAI_MAX_OUTPUT_TOKENS,
        MIN_OPENAI_MAX_OUTPUT_TOKENS,
        MAX_OPENAI_MAX_OUTPUT_TOKENS,
    )
    allow_remote_conversation_compaction = _bool_from_env(
        "JARVIS_ALLOW_REMOTE_COMPACTION", False
    )
    allow_remote_personal_context = _bool_from_env(
        "JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT", False
    )
    watched_dirs_env = os.getenv("JARVIS_WATCHED_DIRS", "")
    watched_dirs = tuple(Path(item.strip()).expanduser() for item in watched_dirs_env.split(":") if item.strip())
    return JarvisConfig(
        data_dir=data_dir,
        db_path=db_path,
        obsidian_vault=detect_obsidian_vault(),
        obsidian_root=root,
        chat_model=chat_model,
        planner_model=planner_model,
        model_timeout_seconds=model_timeout_seconds,
        chat_timeout_seconds=chat_timeout_seconds,
        chat_max_reply_tokens=chat_max_reply_tokens,
        chat_max_history_messages=chat_max_history_messages,
        use_model_planner=use_model_planner,
        model_provider=model_provider,
        chat_reasoning_effort=chat_reasoning_effort,
        planner_reasoning_effort=planner_reasoning_effort,
        openai_max_output_tokens=openai_max_output_tokens,
        allow_remote_conversation_compaction=allow_remote_conversation_compaction,
        allow_remote_personal_context=allow_remote_personal_context,
        watched_dirs=watched_dirs,
    )
