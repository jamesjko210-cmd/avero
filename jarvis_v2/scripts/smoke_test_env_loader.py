from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.config import DEFAULT_MODEL_ALIAS, load_config
import jarvis_v2.env as env_module
from jarvis_v2.env import load_env


class EnvGuard:
    def __init__(self, keys: list[str], updates: dict[str, str] | None = None):
        self.keys = keys
        self.updates = updates or {}
        self.previous: dict[str, str | None] = {}

    def __enter__(self):
        watched = set(self.keys) | set(self.updates)
        self.previous = {key: os.environ.get(key) for key in watched}
        os.environ.update(self.updates)
        for key in self.keys:
            if key not in self.updates:
                os.environ.pop(key, None)
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_load_env_expands_custom_env_path() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        home = root / "home"
        home.mkdir()
        env_file = home / "custom.env"
        env_file.write_text(
            "\n".join(
                [
                    "# local test file",
                    "JARVIS_CHAT_TIMEOUT_SECONDS=31",
                    "JARVIS_PLANNER_MODEL='planner-smoke'",
                    'JARVIS_CHAT_MODEL="chat-smoke"',
                    "export JARVIS_USE_MODEL_PLANNER=0",
                    "JARVIS_MODEL_TIMEOUT_SECONDS=4.5 # planner timeout",
                    'JARVIS_NEWS_LOCALE="en-US:#local"',
                    "BAD KEY=ignored",
                    "1BAD=ignored",
                ]
            ),
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        with EnvGuard(
            [
                "JARVIS_CHAT_TIMEOUT_SECONDS",
                "JARVIS_PLANNER_MODEL",
                "JARVIS_CHAT_MODEL",
                "JARVIS_USE_MODEL_PLANNER",
                "export JARVIS_USE_MODEL_PLANNER",
                "JARVIS_MODEL_TIMEOUT_SECONDS",
                "JARVIS_NEWS_LOCALE",
                "BAD KEY",
                "1BAD",
            ],
            {"HOME": str(home), "JARVIS_V3_ENV": "~/custom.env"},
        ):
            load_env()
            if os.environ.get("JARVIS_CHAT_TIMEOUT_SECONDS") != "31":
                raise SystemExit("load_env did not expand JARVIS_V3_ENV with ~")
            if os.environ.get("JARVIS_PLANNER_MODEL") != "planner-smoke":
                raise SystemExit("load_env did not strip single quotes")
            if os.environ.get("JARVIS_CHAT_MODEL") != "chat-smoke":
                raise SystemExit("load_env did not strip double quotes")
            if os.environ.get("JARVIS_USE_MODEL_PLANNER") != "0":
                raise SystemExit("load_env did not parse export-prefixed env lines")
            if "export JARVIS_USE_MODEL_PLANNER" in os.environ:
                raise SystemExit("load_env created a literal export-prefixed env key")
            if os.environ.get("JARVIS_MODEL_TIMEOUT_SECONDS") != "4.5":
                raise SystemExit("load_env did not strip unquoted inline comments")
            if os.environ.get("JARVIS_NEWS_LOCALE") != "en-US:#local":
                raise SystemExit("load_env stripped # from a quoted value")
            if "BAD KEY" in os.environ or "1BAD" in os.environ:
                raise SystemExit("load_env accepted malformed environment keys")


def test_load_env_does_not_overwrite_existing_environment() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        env_file = Path(temp) / "custom.env"
        env_file.write_text("JARVIS_CHAT_TIMEOUT_SECONDS=99\n", encoding="utf-8")
        env_file.chmod(0o600)
        with EnvGuard(
            ["JARVIS_CHAT_TIMEOUT_SECONDS"],
            {"JARVIS_V3_ENV": str(env_file), "JARVIS_CHAT_TIMEOUT_SECONDS": "12"},
        ):
            load_env()
            if os.environ.get("JARVIS_CHAT_TIMEOUT_SECONDS") != "12":
                raise SystemExit("load_env overwrote an existing environment value")


def test_load_env_blank_custom_path_uses_project_default() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        default_env = root / ".env"
        default_env.write_text("JARVIS_CHAT_TIMEOUT_SECONDS=44\n", encoding="utf-8")
        old_project_root = env_module._PROJECT_ROOT
        env_module._PROJECT_ROOT = root
        try:
            with EnvGuard(
                ["JARVIS_CHAT_TIMEOUT_SECONDS"],
                {"JARVIS_V3_ENV": "   "},
            ):
                load_env()
                if os.environ.get("JARVIS_CHAT_TIMEOUT_SECONDS") != "44":
                    raise SystemExit("blank JARVIS_V3_ENV did not fall back to project .env")
        finally:
            env_module._PROJECT_ROOT = old_project_root


def test_load_env_missing_project_default_is_optional() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        old_project_root = env_module._PROJECT_ROOT
        env_module._PROJECT_ROOT = Path(temp)
        try:
            with EnvGuard([], {"JARVIS_V3_ENV": "   "}):
                load_env()
        finally:
            env_module._PROJECT_ROOT = old_project_root


def test_load_env_explicit_missing_paths_fail() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        missing = Path(temp) / "missing.env"
        with EnvGuard(["JARVIS_V3_ENV"]):
            try:
                load_env(missing)
            except FileNotFoundError:
                pass
            else:
                raise SystemExit("explicit function path did not fail when missing")

        with EnvGuard([], {"JARVIS_V3_ENV": str(missing)}):
            try:
                load_config()
            except FileNotFoundError:
                pass
            else:
                raise SystemExit("nonblank JARVIS_V3_ENV did not fail when missing")


def test_load_env_explicit_directory_fails() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        with EnvGuard([], {"JARVIS_V3_ENV": temp}):
            try:
                load_env()
            except OSError:
                pass
            else:
                raise SystemExit("explicit environment directory did not fail")


def test_load_env_explicit_unreadable_file_fails() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        env_file = Path(temp) / "unreadable.env"
        env_file.write_text("JARVIS_ENV_UNREADABLE_SMOKE=hidden\n", encoding="utf-8")
        with EnvGuard(["JARVIS_ENV_UNREADABLE_SMOKE", "JARVIS_V3_ENV"]):
            with patch.object(os, "open", side_effect=PermissionError("environment file is unreadable")):
                try:
                    load_env(env_file)
                except PermissionError:
                    pass
                else:
                    raise SystemExit("explicit unreadable environment file did not fail")
            if "JARVIS_ENV_UNREADABLE_SMOKE" in os.environ:
                raise SystemExit("unreadable environment file partially updated os.environ")


def test_load_env_invalid_utf8_fails_without_partial_updates() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        env_file = Path(temp) / "invalid.env"
        env_file.write_bytes(b"JARVIS_ENV_PARTIAL_SMOKE=must-not-leak\n\xff")
        env_file.chmod(0o600)
        with EnvGuard(["JARVIS_ENV_PARTIAL_SMOKE"], {"JARVIS_V3_ENV": str(env_file)}):
            try:
                load_env()
            except UnicodeDecodeError:
                pass
            else:
                raise SystemExit("invalid UTF-8 environment file did not fail")
            if "JARVIS_ENV_PARTIAL_SMOKE" in os.environ:
                raise SystemExit("invalid UTF-8 environment file partially updated os.environ")


def test_dashboard_auth_env_requires_owner_only_file() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-auth-") as temp:
        env_file = Path(temp) / "dashboard.env"
        env_file.write_text(
            "JARVIS_ENV_PARTIAL_SMOKE=must-not-load\n"
            "JARVIS_STATUS_AUTH_TOKEN=status-env-smoke-secret-32-characters\n",
            encoding="utf-8",
        )
        for insecure_mode in (0o644, 0o640, 0o666):
            env_file.chmod(insecure_mode)
            with EnvGuard(
                ["JARVIS_ENV_PARTIAL_SMOKE", "JARVIS_STATUS_AUTH_TOKEN"],
                {"JARVIS_V3_ENV": str(env_file)},
            ):
                try:
                    load_env()
                except PermissionError as exc:
                    diagnostic = str(exc)
                else:
                    raise SystemExit(
                        f"dashboard auth loaded from insecure environment mode {insecure_mode:o}"
                    )
                if "0600" not in diagnostic:
                    raise SystemExit("dashboard auth environment permission failure missed its bounded fix")
                if "status-env-smoke-secret" in diagnostic or str(env_file) in diagnostic:
                    raise SystemExit("dashboard auth environment permission failure leaked secret/path material")
                if "JARVIS_ENV_PARTIAL_SMOKE" in os.environ or "JARVIS_STATUS_AUTH_TOKEN" in os.environ:
                    raise SystemExit("insecure dashboard auth environment file partially updated os.environ")

        env_file.chmod(0o600)
        with EnvGuard(
            ["JARVIS_ENV_PARTIAL_SMOKE", "JARVIS_STATUS_AUTH_TOKEN"],
            {"JARVIS_V3_ENV": str(env_file)},
        ), patch.object(os, "geteuid", return_value=os.geteuid() + 1):
            try:
                load_env()
            except PermissionError:
                pass
            else:
                raise SystemExit("dashboard auth loaded from an environment file owned by another uid")
            if "JARVIS_ENV_PARTIAL_SMOKE" in os.environ or "JARVIS_STATUS_AUTH_TOKEN" in os.environ:
                raise SystemExit("foreign-owned dashboard auth environment partially updated os.environ")

        with EnvGuard(
            ["JARVIS_ENV_PARTIAL_SMOKE", "JARVIS_STATUS_AUTH_TOKEN"],
            {"JARVIS_V3_ENV": str(env_file)},
        ):
            load_env()
            if os.environ.get("JARVIS_STATUS_AUTH_TOKEN") != "status-env-smoke-secret-32-characters":
                raise SystemExit("owner-only dashboard auth environment file did not load")

        symlink_path = Path(temp) / "dashboard-link.env"
        symlink_path.symlink_to(env_file)
        with EnvGuard(
            ["JARVIS_ENV_PARTIAL_SMOKE", "JARVIS_STATUS_AUTH_TOKEN"],
            {"JARVIS_V3_ENV": str(symlink_path)},
        ):
            try:
                load_env()
            except PermissionError as exc:
                diagnostic = str(exc)
            else:
                raise SystemExit("dashboard auth loaded through an environment-file symlink")
            if "symlink" not in diagnostic.lower():
                raise SystemExit("environment symlink refusal missed bounded guidance")
            if str(env_file) in diagnostic or str(symlink_path) in diagnostic:
                raise SystemExit("environment symlink refusal leaked a local path")


def test_selected_v3_env_always_requires_owner_only_custody() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-selected-custody-") as temp:
        root = Path(temp)
        selected = root / "selected-private-name.env"
        secret_marker = "synthetic-selected-env-secret-must-not-leak"
        cases = (
            ("", ()),
            (
                f"OPENAI_API_KEY={secret_marker}\n"
                "JARVIS_ENV_PARTIAL_SMOKE=must-not-load\n",
                ("OPENAI_API_KEY", "JARVIS_ENV_PARTIAL_SMOKE"),
            ),
        )
        for content, keys in cases:
            selected.write_text(content, encoding="utf-8")
            selected.chmod(0o644)
            with EnvGuard(list(keys), {"JARVIS_V3_ENV": str(selected)}):
                try:
                    load_env()
                except PermissionError as exc:
                    diagnostic = str(exc)
                else:
                    raise SystemExit(
                        "explicit V3 environment loaded without owner-only custody"
                    )
                if "0600" not in diagnostic:
                    raise SystemExit(
                        "selected V3 environment permission refusal missed bounded recovery"
                    )
                if secret_marker in diagnostic or str(selected) in diagnostic:
                    raise SystemExit(
                        "selected V3 environment permission refusal leaked secret/path material"
                    )
                if any(key in os.environ for key in keys):
                    raise SystemExit(
                        "insecure selected V3 environment partially updated os.environ"
                    )

        selected.write_text(
            "JARVIS_ENV_OWNER_ONLY_SMOKE=loaded\n",
            encoding="utf-8",
        )
        selected.chmod(0o600)
        with EnvGuard(
            ["JARVIS_ENV_OWNER_ONLY_SMOKE"],
            {"JARVIS_V3_ENV": str(selected)},
        ):
            load_env()
            if os.environ.get("JARVIS_ENV_OWNER_ONLY_SMOKE") != "loaded":
                raise SystemExit("owner-only selected V3 environment did not load")

        hardlink = root / "selected-hardlink.env"
        os.link(selected, hardlink)
        with EnvGuard(
            ["JARVIS_ENV_OWNER_ONLY_SMOKE"],
            {"JARVIS_V3_ENV": str(selected)},
        ):
            try:
                load_env()
            except PermissionError as exc:
                diagnostic = str(exc)
            else:
                raise SystemExit("hard-linked selected V3 environment did not fail closed")
            if "single-link" not in diagnostic or str(selected) in diagnostic:
                raise SystemExit("hard-linked environment refusal was not bounded")
            if "JARVIS_ENV_OWNER_ONLY_SMOKE" in os.environ:
                raise SystemExit("hard-linked environment partially updated os.environ")

        real_parent = root / "real-parent"
        real_parent.mkdir()
        parent_selected = real_parent / "selected.env"
        parent_selected.write_text(
            "JARVIS_ENV_PARENT_ALIAS_SMOKE=must-not-load\n",
            encoding="utf-8",
        )
        parent_selected.chmod(0o600)
        parent_alias = root / "parent-alias"
        parent_alias.symlink_to(real_parent, target_is_directory=True)
        with EnvGuard(
            ["JARVIS_ENV_PARENT_ALIAS_SMOKE"],
            {"JARVIS_V3_ENV": str(parent_alias / "selected.env")},
        ):
            try:
                load_env()
            except PermissionError as exc:
                diagnostic = str(exc)
            else:
                raise SystemExit("symlink-parent selected V3 environment did not fail closed")
            if "alias" not in diagnostic.lower() or str(parent_alias) in diagnostic:
                raise SystemExit("symlink-parent environment refusal was not bounded")
            if "JARVIS_ENV_PARENT_ALIAS_SMOKE" in os.environ:
                raise SystemExit("symlink-parent environment partially updated os.environ")

        fallback_root = root / "fallback"
        fallback_root.mkdir()
        fallback = fallback_root / ".env"
        fallback.write_text("JARVIS_ENV_FALLBACK_SMOKE=loaded\n", encoding="utf-8")
        fallback.chmod(0o644)
        old_project_root = env_module._PROJECT_ROOT
        env_module._PROJECT_ROOT = fallback_root
        try:
            with EnvGuard(
                ["JARVIS_ENV_FALLBACK_SMOKE"],
                {"JARVIS_V3_ENV": ""},
            ):
                load_env()
                if os.environ.get("JARVIS_ENV_FALLBACK_SMOKE") != "loaded":
                    raise SystemExit(
                        "project fallback .env compatibility was not preserved"
                    )
        finally:
            env_module._PROJECT_ROOT = old_project_root


def test_persistent_path_custody_rejects_hardlinks_and_symlink_components() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-path-custody-") as temp:
        root = Path(temp)
        v2_root = root / ".jarvis_v2"
        v2_root.mkdir()
        v2_state = v2_root / "state.sqlite"
        v2_state.write_text("synthetic-v2-state\n", encoding="utf-8")
        v3_root = root / "v3"
        v3_root.mkdir()
        hardlink_alias = v3_root / "state.sqlite"
        os.link(v2_state, hardlink_alias)

        try:
            env_module.validate_v3_path_custody(
                {"JARVIS_DB_PATH": str(hardlink_alias)}
            )
        except env_module.V3PathCustodyError as exc:
            if exc.key != "JARVIS_DB_PATH" or exc.reason_code != "unresolvable_alias":
                raise SystemExit(f"hardlink refusal lost bounded metadata: {exc}")
            if str(root) in str(exc):
                raise SystemExit("hardlink refusal leaked a local path")
        else:
            raise SystemExit("existing multi-link persistent file was accepted")

        single_link = v3_root / "single-link.sqlite"
        single_link.write_text("synthetic-v3-state\n", encoding="utf-8")
        env_module.validate_v3_path_custody({"JARVIS_DB_PATH": str(single_link)})

        safe_target = root / "safe-target"
        safe_target.mkdir()
        component_alias = root / "component-alias"
        component_alias.symlink_to(safe_target, target_is_directory=True)
        for values, expected_key in (
            ({"JARVIS_DATA_DIR": str(component_alias / "future")}, "JARVIS_DATA_DIR"),
            (
                {"JARVIS_WATCHED_DIRS": str(component_alias / "watched")},
                "JARVIS_WATCHED_DIRS",
            ),
        ):
            try:
                env_module.validate_v3_path_custody(values)
            except env_module.V3PathCustodyError as exc:
                if exc.key != expected_key or exc.reason_code != "unresolvable_alias":
                    raise SystemExit(f"symlink-component refusal drifted: {exc}")
            else:
                raise SystemExit(f"symlink component was accepted for {expected_key}")

        component_alias.unlink()
        component_alias.symlink_to(v2_root, target_is_directory=True)
        try:
            env_module.validate_v3_path_custody(
                {"JARVIS_DATA_DIR": str(component_alias / "retargeted")}
            )
        except env_module.V3PathCustodyError:
            pass
        else:
            raise SystemExit("retargeted symlink component was accepted")


def test_persistent_path_custody_allows_future_suffixes_and_macos_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-path-future-") as temp:
        root = Path(temp)
        existing_parent = root / "existing"
        existing_parent.mkdir()
        env_module.validate_v3_path_custody(
            {
                "JARVIS_DATA_DIR": str(existing_parent / "future" / "data"),
                "JARVIS_DB_PATH": str(
                    existing_parent / "future" / "data" / "jarvis.sqlite"
                ),
                "JARVIS_WATCHED_DIRS": str(
                    existing_parent / "future" / "watched"
                ),
            }
        )
        traversal_after_missing = (
            existing_parent / "future-missing" / ".." / "retargeted"
        )
        try:
            env_module.validate_v3_path_custody(
                {"JARVIS_DATA_DIR": str(traversal_after_missing)}
            )
        except env_module.V3PathCustodyError as exc:
            if exc.reason_code != "invalid_path":
                raise SystemExit(f"future traversal refusal drifted: {exc}")
        else:
            raise SystemExit("parent traversal after a missing component was accepted")

        for alias_name in ("tmp", "var", "etc"):
            alias = Path("/") / alias_name
            expected = Path("/private") / alias_name
            if not (
                alias.is_symlink()
                and Path(os.path.realpath(alias)) == expected
            ):
                continue
            env_module.validate_v3_path_custody(
                {
                    "JARVIS_CACHE_DIR": str(
                        alias / f"jarvis-v3-system-alias-{alias_name}" / "future"
                    )
                }
            )


def test_load_config_trims_watched_dirs() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        watched_one = root / "Watched One"
        watched_two = root / "Watched Two"
        env_file = root / "custom.env"
        env_file.write_text(
            f"JARVIS_WATCHED_DIRS={watched_one}: {watched_two} :\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        with EnvGuard(
            [
                "JARVIS_WATCHED_DIRS",
                "JARVIS_DATA_DIR",
                "JARVIS_DB_PATH",
                "JARVIS_OBSIDIAN_VAULT",
                "OBSIDIAN_VAULT_PATH",
            ],
            {
                "JARVIS_V3_ENV": str(env_file),
                "JARVIS_DATA_DIR": str(root / "data"),
                "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "vault"),
            },
        ):
            config = load_config()
            if config.watched_dirs != (watched_one, watched_two):
                raise SystemExit(f"load_config did not trim watched dirs: {config.watched_dirs!r}")


def test_load_config_trims_string_values_and_falls_back_on_blanks() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        data_dir = root / "data"
        db_path = data_dir / "jarvis.sqlite"
        vault = root / "vault"
        with EnvGuard(
            [
                "JARVIS_DATA_DIR",
                "JARVIS_DB_PATH",
                "JARVIS_OBSIDIAN_VAULT",
                "OBSIDIAN_VAULT_PATH",
                "JARVIS_OBSIDIAN_ROOT",
                "JARVIS_CHAT_MODEL",
                "JARVIS_PLANNER_MODEL",
                "OLLAMA_MODEL",
            ],
            {
                "JARVIS_DATA_DIR": f"  {data_dir}  ",
                "JARVIS_DB_PATH": f"  {db_path}  ",
                "JARVIS_OBSIDIAN_VAULT": str(vault),
                "JARVIS_OBSIDIAN_ROOT": "  Jarvis Notes  ",
                "JARVIS_CHAT_MODEL": "   ",
                "OLLAMA_MODEL": "  ollama-smoke  ",
                "JARVIS_PLANNER_MODEL": "  planner-smoke  ",
            },
        ):
            config = load_config()
            if config.data_dir != data_dir:
                raise SystemExit(f"load_config did not trim JARVIS_DATA_DIR: {config.data_dir!r}")
            if config.db_path != db_path:
                raise SystemExit(f"load_config did not trim JARVIS_DB_PATH: {config.db_path!r}")
            if config.obsidian_root != "Jarvis Notes":
                raise SystemExit(f"load_config did not trim JARVIS_OBSIDIAN_ROOT: {config.obsidian_root!r}")
            if config.chat_model != "ollama-smoke":
                raise SystemExit(f"load_config did not fall back from blank chat model to OLLAMA_MODEL: {config.chat_model!r}")
            if config.planner_model != "planner-smoke":
                raise SystemExit(f"load_config did not trim JARVIS_PLANNER_MODEL: {config.planner_model!r}")


def test_load_config_blank_paths_use_defaults() -> None:
    with EnvGuard(
        [
            "JARVIS_DATA_DIR",
            "JARVIS_DB_PATH",
            "JARVIS_OBSIDIAN_VAULT",
            "OBSIDIAN_VAULT_PATH",
        ],
        {
            "JARVIS_DATA_DIR": "   ",
            "JARVIS_DB_PATH": "   ",
        },
    ):
        config = load_config()
        expected_data_dir = Path("~/.jarvis_v3").expanduser()
        if config.data_dir != expected_data_dir:
            raise SystemExit(f"blank JARVIS_DATA_DIR should use default: {config.data_dir!r}")
        if config.db_path != expected_data_dir / "jarvis.sqlite":
            raise SystemExit(f"blank JARVIS_DB_PATH should use default under data dir: {config.db_path!r}")


def test_load_config_invalid_obsidian_root_falls_back() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        with EnvGuard(
            [
                "JARVIS_DATA_DIR",
                "JARVIS_DB_PATH",
                "JARVIS_OBSIDIAN_VAULT",
                "OBSIDIAN_VAULT_PATH",
                "JARVIS_OBSIDIAN_ROOT",
            ],
            {
                "JARVIS_DATA_DIR": str(root / "data"),
                "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "vault"),
                "JARVIS_OBSIDIAN_ROOT": "/\x55sers/example/private/obsidian-root",
            },
        ):
            config = load_config()
            if config.obsidian_root != "Jarvis":
                raise SystemExit(f"path-shaped JARVIS_OBSIDIAN_ROOT should fall back to Jarvis: {config.obsidian_root!r}")


def test_load_config_invalid_model_planner_toggle_falls_back() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        with EnvGuard(
            [
                "JARVIS_DATA_DIR",
                "JARVIS_DB_PATH",
                "JARVIS_OBSIDIAN_VAULT",
                "OBSIDIAN_VAULT_PATH",
                "JARVIS_USE_MODEL_PLANNER",
            ],
            {
                "JARVIS_DATA_DIR": str(root / "data"),
                "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "vault"),
                "JARVIS_USE_MODEL_PLANNER": "/\x55sers/example/private/planner-toggle",
            },
        ):
            config = load_config()
            if config.use_model_planner is not True:
                raise SystemExit("invalid JARVIS_USE_MODEL_PLANNER should fall back to enabled")


def test_remote_compaction_requires_explicit_valid_opt_in() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        base = {
            "JARVIS_DATA_DIR": str(root / "data"),
            "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
            "JARVIS_OBSIDIAN_VAULT": str(root / "vault"),
        }
        keys = [
            "JARVIS_DATA_DIR",
            "JARVIS_DB_PATH",
            "JARVIS_OBSIDIAN_VAULT",
            "OBSIDIAN_VAULT_PATH",
            "JARVIS_ALLOW_REMOTE_COMPACTION",
        ]
        with EnvGuard(keys, base):
            if load_config().allow_remote_conversation_compaction is not False:
                raise SystemExit("remote conversation compaction must default to disabled")
        with EnvGuard(keys, {**base, "JARVIS_ALLOW_REMOTE_COMPACTION": "1"}):
            if load_config().allow_remote_conversation_compaction is not True:
                raise SystemExit("explicit JARVIS_ALLOW_REMOTE_COMPACTION=1 opt-in was ignored")
        with EnvGuard(keys, {**base, "JARVIS_ALLOW_REMOTE_COMPACTION": "/\x55sers/example/private/unsafe-toggle"}):
            if load_config().allow_remote_conversation_compaction is not False:
                raise SystemExit("invalid remote compaction opt-in must fail closed")


def test_remote_personal_context_requires_explicit_valid_opt_in() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        base = {
            "JARVIS_DATA_DIR": str(root / "data"),
            "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
            "JARVIS_OBSIDIAN_VAULT": str(root / "vault"),
        }
        keys = [
            "JARVIS_DATA_DIR",
            "JARVIS_DB_PATH",
            "JARVIS_OBSIDIAN_VAULT",
            "OBSIDIAN_VAULT_PATH",
            "JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT",
        ]
        with EnvGuard(keys, base):
            if load_config().allow_remote_personal_context is not False:
                raise SystemExit("remote personal context must default to disabled")
        with EnvGuard(keys, {**base, "JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT": "0"}):
            if load_config().allow_remote_personal_context is not False:
                raise SystemExit("JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT=0 must stay disabled")
        with EnvGuard(
            keys,
            {**base, "JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT": "/\x55sers/example/private/unsafe-toggle"},
        ):
            if load_config().allow_remote_personal_context is not False:
                raise SystemExit("invalid remote personal context opt-in must fail closed")
        with EnvGuard(keys, {**base, "JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT": "1"}):
            if load_config().allow_remote_personal_context is not True:
                raise SystemExit("explicit JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT=1 opt-in was ignored")


def test_load_config_invalid_model_aliases_fall_back() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        empty_env = root / "empty.env"
        empty_env.write_text("", encoding="utf-8")
        empty_env.chmod(0o600)
        with EnvGuard(
            [
                "JARVIS_V3_ENV",
                "JARVIS_DATA_DIR",
                "JARVIS_DB_PATH",
                "JARVIS_OBSIDIAN_VAULT",
                "OBSIDIAN_VAULT_PATH",
                "OLLAMA_MODEL",
                "JARVIS_CHAT_MODEL",
                "JARVIS_PLANNER_MODEL",
            ],
            {
                "JARVIS_V3_ENV": str(empty_env),
                "JARVIS_DATA_DIR": str(root / "data"),
                "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "vault"),
                "OLLAMA_MODEL": "  fallback-smoke:latest  ",
                "JARVIS_CHAT_MODEL": "/\x55sers/example/private/chat-model",
                "JARVIS_PLANNER_MODEL": "/\x55sers/example/private/planner-model",
            },
        ):
            config = load_config()
            if config.chat_model != "fallback-smoke:latest":
                raise SystemExit(f"path-shaped JARVIS_CHAT_MODEL should fall back to OLLAMA_MODEL: {config.chat_model!r}")
            if config.planner_model != "fallback-smoke:latest":
                raise SystemExit(f"path-shaped JARVIS_PLANNER_MODEL should fall back to chat model: {config.planner_model!r}")
        with EnvGuard(
            [
                "JARVIS_V3_ENV",
                "JARVIS_DATA_DIR",
                "JARVIS_DB_PATH",
                "JARVIS_OBSIDIAN_VAULT",
                "OBSIDIAN_VAULT_PATH",
                "OLLAMA_MODEL",
                "JARVIS_CHAT_MODEL",
                "JARVIS_PLANNER_MODEL",
            ],
            {
                "JARVIS_V3_ENV": str(empty_env),
                "JARVIS_DATA_DIR": str(root / "data"),
                "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "vault"),
                "OLLAMA_MODEL": "/\x55sers/example/private/ollama-model",
            },
        ):
            config = load_config()
            if config.chat_model != DEFAULT_MODEL_ALIAS or config.planner_model != DEFAULT_MODEL_ALIAS:
                raise SystemExit(f"path-shaped OLLAMA_MODEL should fall back to {DEFAULT_MODEL_ALIAS}: {config!r}")


def test_load_config_blank_vault_env_falls_through_to_secondary() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        secondary_vault = root / "secondary-vault"
        with EnvGuard(
            [
                "JARVIS_DATA_DIR",
                "JARVIS_DB_PATH",
                "JARVIS_OBSIDIAN_VAULT",
                "OBSIDIAN_VAULT_PATH",
            ],
            {
                "JARVIS_DATA_DIR": str(root / "data"),
                "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": "   ",
                "OBSIDIAN_VAULT_PATH": f"  {secondary_vault}  ",
            },
        ):
            config = load_config()
            if config.obsidian_vault != secondary_vault:
                raise SystemExit(f"blank primary vault env should fall through to secondary: {config.obsidian_vault!r}")


def test_load_config_default_vault_is_home_relative() -> None:
    with TemporaryDirectory(prefix="jarvis-env-loader-") as temp:
        root = Path(temp)
        home = root / "home"
        home.mkdir()
        empty_env = root / "empty.env"
        empty_env.write_text("", encoding="utf-8")
        empty_env.chmod(0o600)
        expected = home / ".jarvis_v3" / "Vault"
        with EnvGuard(
            [
                "JARVIS_DATA_DIR",
                "JARVIS_DB_PATH",
                "JARVIS_OBSIDIAN_VAULT",
                "OBSIDIAN_VAULT_PATH",
            ],
            {"HOME": str(home), "JARVIS_V3_ENV": str(empty_env)},
        ):
            config = load_config()
            if config.obsidian_vault != expected:
                raise SystemExit(f"default Obsidian vault should follow HOME: {config.obsidian_vault!r}")
            if "/\x55sers/operator" in str(config.obsidian_vault):
                raise SystemExit(f"default Obsidian vault should not bake in the operator's home: {config.obsidian_vault!r}")


def main() -> None:
    test_load_env_expands_custom_env_path()
    test_load_env_does_not_overwrite_existing_environment()
    test_load_env_blank_custom_path_uses_project_default()
    test_load_env_missing_project_default_is_optional()
    test_load_env_explicit_missing_paths_fail()
    test_load_env_explicit_directory_fails()
    test_load_env_explicit_unreadable_file_fails()
    test_load_env_invalid_utf8_fails_without_partial_updates()
    test_dashboard_auth_env_requires_owner_only_file()
    test_selected_v3_env_always_requires_owner_only_custody()
    test_persistent_path_custody_rejects_hardlinks_and_symlink_components()
    test_persistent_path_custody_allows_future_suffixes_and_macos_aliases()
    test_load_config_trims_watched_dirs()
    test_load_config_trims_string_values_and_falls_back_on_blanks()
    test_load_config_blank_paths_use_defaults()
    test_load_config_invalid_obsidian_root_falls_back()
    test_load_config_invalid_model_planner_toggle_falls_back()
    test_remote_compaction_requires_explicit_valid_opt_in()
    test_remote_personal_context_requires_explicit_valid_opt_in()
    test_load_config_invalid_model_aliases_fall_back()
    test_load_config_blank_vault_env_falls_through_to_secondary()
    test_load_config_default_vault_is_home_relative()
    print("Env loader smoke passed")


if __name__ == "__main__":
    main()
