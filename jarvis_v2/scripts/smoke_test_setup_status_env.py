from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import (
    MAX_OPENAI_CHAT_TIMEOUT_SECONDS,
    MAX_OPENAI_MODEL_TIMEOUT_SECONDS,
    JarvisConfig,
    load_config,
)
from jarvis_v2.env import load_env
from jarvis_v2.tools.system import (
    _local_command_env_status,
    _local_path_env_status,
    _model_alias_status,
    _news_locale_status,
    _obsidian_root_status,
    _owner_allowlist_status,
    _raw_int_metadata,
    _secret_env_status,
    setup_check,
    _weather_location_status,
    open_application,
    volume,
)
from jarvis_v2.ui.status_config import StatusHostConfig, StatusPortConfig


def _restore_env(key: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = old_value


def assert_temp_root_system_redaction() -> None:
    env_cases = [
        ("OLLAMA_MODEL", lambda key: _model_alias_status(key, "llama3.1", default_source="default"), ("/var/folders/zc/jarvis-model", "/tmp/jarvis-model")),
        ("JARVIS_WEATHER_LOCATION", lambda _: _weather_location_status(), ("/var/folders/zc/jarvis-weather", "/tmp/jarvis-weather")),
        ("JARVIS_NEWS_LOCALE", lambda _: _news_locale_status(), ("/var/folders/zc/jarvis-news:KR", "/tmp/jarvis-news:KR")),
        ("JARVIS_OWNER_TELEGRAM", lambda _: _owner_allowlist_status("JARVIS_OWNER_TELEGRAM", required=True), ("/var/folders/zc/jarvis-owner", "/tmp/jarvis-owner")),
        ("TELEGRAM_BOT_TOKEN", lambda _: _secret_env_status("TELEGRAM_BOT_TOKEN", required=True), ("/var/folders/zc/jarvis-token", "/tmp/jarvis-token")),
        ("JARVIS_OBSIDIAN_ROOT", lambda _: _obsidian_root_status(), ("/var/folders/zc/jarvis-root", "/tmp/jarvis-root")),
        ("JARVIS_OAV_VISION_REVIEWER_COMMAND", lambda _: _local_command_env_status("JARVIS_OAV_VISION_REVIEWER_COMMAND"), ("/var/folders/zc/missing-oav-reviewer --json", "/tmp/missing-oav-reviewer --json")),
        ("JARVIS_VOICE_WHISPER_MODEL_PATH", lambda _: _local_path_env_status("JARVIS_VOICE_WHISPER_MODEL_PATH", expected="file"), ("/var/folders/zc/missing-whisper-model.bin", "/tmp/missing-whisper-model.bin")),
        ("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH", lambda _: _local_path_env_status("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH", expected="directory"), ("/var/folders/zc/missing-faster-whisper-model", "/tmp/missing-faster-whisper-model")),
    ]
    for env_key, status_func, raw_values in env_cases:
        old_value = os.environ.get(env_key)
        try:
            for raw_value in raw_values:
                os.environ[env_key] = raw_value
                status = status_func(env_key)
                if status.get("valid") is not False or status.get("source") != "env-invalid":
                    raise SystemExit(f"system setup status should reject temp-root {env_key}: {status}")
        finally:
            _restore_env(env_key, old_value)

    for raw_value in ("/var/folders/zc/jarvis-volume", "/tmp/jarvis-volume"):
        metadata = _raw_int_metadata(raw_value, key="level", sanitized=0)
        if metadata.get("raw_level") != "<local-path>":
            raise SystemExit(f"system raw numeric metadata should redact temp-root paths: {metadata}")
        volume_result = volume({"level": raw_value})
        if volume_result.ok or volume_result.metadata.get("raw_level") != "<local-path>":
            raise SystemExit(f"volume should redact temp-root malformed levels before side effects: {volume_result.metadata}")

    for raw_value in ("/var/folders/zc/jarvis-app", "/tmp/jarvis-app"):
        app_result = open_application({"name": raw_value})
        if app_result.ok or app_result.metadata.get("app") != "<local-path>" or app_result.metadata.get("executes_side_effect"):
            raise SystemExit(f"open_application should reject temp-root app names before OS action: {app_result.metadata}")


def assert_doctor_no_future_authority(metadata: dict, label: str) -> None:
    for key in [
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "calls_model",
        "executes_tools",
        "queues_approval",
        "approves_request",
        "reads_private_data",
        "reads_personal_data",
        "external_side_effect",
        "controls_computer",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "speaks",
        "completes_tasks",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} unsafe diagnostic metadata {key}: {metadata}")
    boundaries = metadata.get("doctor_handoff", {}).get("boundaries", {})
    for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} unsafe nested doctor boundary {key}: {metadata}")


def assert_setup_check_does_not_read_env_or_secret_values() -> None:
    secret_keys = {
        "TELEGRAM_BOT_TOKEN",
        "GMAIL_ADDRESS",
        "GMAIL_APP_PASSWORD",
        "OPENAI_API_KEY",
        "JARVIS_STATUS_AUTH_TOKEN",
    }
    secret_marker = "synthetic-secret-value-must-not-be-read"
    env_file_marker = "synthetic-env-file-value-must-not-be-loaded"
    real_getenv = os.getenv

    def guarded_getenv(key: str, default: str | None = None) -> str | None:
        if key in secret_keys:
            raise AssertionError(f"setup check retrieved secret value for {key}")
        return real_getenv(key, default)

    with TemporaryDirectory(prefix="jarvis-setup-privacy-") as temp:
        env_file = Path(temp) / "synthetic.env"
        env_file.write_text(
            f"ENV_FILE_MARKER={env_file_marker}\nOPENAI_API_KEY={env_file_marker}\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        synthetic_env = {
            "JARVIS_V3_ENV": str(env_file),
            "JARVIS_MODEL_PROVIDER": "openai",
            "TELEGRAM_BOT_TOKEN": secret_marker,
            "GMAIL_ADDRESS": secret_marker,
            "GMAIL_APP_PASSWORD": secret_marker,
            "OPENAI_API_KEY": secret_marker,
            "JARVIS_STATUS_AUTH_TOKEN": secret_marker,
        }
        with (
            patch.dict(os.environ, synthetic_env, clear=False),
            patch("jarvis_v2.tools.system.os.getenv", side_effect=guarded_getenv),
            patch("jarvis_v2.env.load_env", side_effect=AssertionError("setup check loaded env file")),
            patch(
                "jarvis_v2.tools.system._secret_env_status",
                side_effect=AssertionError("setup check inspected a secret value"),
            ),
        ):
            result = setup_check({})

        combined = f"{result.output} {result.metadata}"
        if secret_marker in combined or env_file_marker in combined:
            raise SystemExit(f"setup check exposed a synthetic secret or env-file value: {combined}")
        for expected in (
            "file exists (contents not inspected)",
            "Secret environment values: not inspected; environment-key presence only",
            "key present (value not inspected)",
        ):
            if expected not in result.output:
                raise SystemExit(f"setup check missed privacy disclosure {expected!r}: {result.output}")
        metadata = result.metadata
        if metadata.get("reads_env_file_contents") is not False:
            raise SystemExit(f"setup check overclaimed env-file privacy: {metadata}")
        if metadata.get("reads_secret_values") is not False or metadata.get("inspects_secret_values") is not False:
            raise SystemExit(f"setup check overclaimed secret-value privacy: {metadata}")
        if metadata.get("secret_key_presence_only") is not True:
            raise SystemExit(f"setup check missed key-presence-only metadata: {metadata}")
        for prefix in ("telegram_bot_token", "gmail_address", "gmail_app_password", "openai_api_key"):
            if metadata.get(f"{prefix}_configured") is not True:
                raise SystemExit(f"setup check missed synthetic {prefix} key presence: {metadata}")
            if metadata.get(f"{prefix}_valid") is not None:
                raise SystemExit(f"setup check claimed unperformed {prefix} validation: {metadata}")
            if metadata.get(f"{prefix}_value_inspected") is not False:
                raise SystemExit(f"setup check claimed it inspected {prefix}: {metadata}")
            if metadata.get(f"{prefix}_validation_performed") is not False:
                raise SystemExit(f"setup check claimed it validated {prefix}: {metadata}")
        if metadata.get("status_dashboard_auth_token_configured") is not True:
            raise SystemExit("setup check missed dashboard authentication key presence.")
        if metadata.get("status_dashboard_auth_token_value_inspected") is not False:
            raise SystemExit("setup check inspected the dashboard authentication token.")
        if metadata.get("status_dashboard_auth_token_validation_performed") is not False:
            raise SystemExit("setup check claimed dashboard token validation without reading it.")
        if "ENV_FILE_MARKER" in os.environ:
            raise SystemExit("setup check loaded the synthetic env file into the process environment")


def assert_model_timeout_caps_match_setup_and_doctor() -> None:
    cases = [
        (
            str(MAX_OPENAI_MODEL_TIMEOUT_SECONDS),
            str(MAX_OPENAI_CHAT_TIMEOUT_SECONDS),
            "env",
        ),
        ("1000000000", "1000000000", "env-clamped"),
    ]
    for planner_raw, chat_raw, expected_source in cases:
        with TemporaryDirectory(prefix="jarvis-timeout-diagnostic-cap-") as temp:
            root = Path(temp)
            env = {
                "JARVIS_DATA_DIR": str(root),
                "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                "JARVIS_MODEL_PROVIDER": "openai",
                "JARVIS_MODEL_TIMEOUT_SECONDS": planner_raw,
                "JARVIS_CHAT_TIMEOUT_SECONDS": chat_raw,
            }
            with patch.dict(os.environ, env, clear=False):
                config = load_config()
                setup = setup_check({})
                doctor = JarvisRuntime(config).handle("jarvis doctor").tool_results[0]
        if config.model_timeout_seconds != MAX_OPENAI_MODEL_TIMEOUT_SECONDS:
            raise SystemExit(f"OpenAI planner timeout diagnostic/runtime parity drifted: {config}")
        if config.chat_timeout_seconds != MAX_OPENAI_CHAT_TIMEOUT_SECONDS:
            raise SystemExit(f"OpenAI chat timeout diagnostic/runtime parity drifted: {config}")
        for label, result in (("setup", setup), ("doctor", doctor)):
            metadata = result.metadata
            expected = {
                "planner_timeout_valid": True,
                "planner_timeout_source": expected_source,
                "planner_timeout_effective_seconds": MAX_OPENAI_MODEL_TIMEOUT_SECONDS,
                "planner_timeout_maximum": MAX_OPENAI_MODEL_TIMEOUT_SECONDS,
                "chat_timeout_valid": True,
                "chat_timeout_source": expected_source,
                "chat_timeout_effective_seconds": MAX_OPENAI_CHAT_TIMEOUT_SECONDS,
                "chat_timeout_maximum": MAX_OPENAI_CHAT_TIMEOUT_SECONDS,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise SystemExit(f"{label} timeout cap metadata drifted at {key}: {metadata}")
            if expected_source == "env-clamped":
                for expected_text in (
                    "Planner timeout validation: clamped to 180s (maximum 180s) (value hidden)",
                    "Chat timeout validation: clamped to 600s (maximum 600s) (value hidden)",
                ):
                    if expected_text.lower() not in result.output.lower():
                        raise SystemExit(f"{label} missed timeout clamp disclosure: {result.output}")
            elif "timeout validation: clamped" in result.output.lower():
                raise SystemExit(f"{label} clamped an exact maximum timeout: {result.output}")
            if "1000000000" in result.output or "1000000000" in str(metadata):
                raise SystemExit(f"{label} exposed the raw oversized timeout: {result.output} {metadata}")


def assert_ollama_destination_diagnostics_are_loopback_only() -> None:
    def completed(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="local probe ok", stderr="")

    def assert_receipt(metadata: dict, *, allowed: bool, source: str, family: str, port: int) -> None:
        expected = {
            "ollama_destination_allowed": allowed,
            "ollama_destination_source": source,
            "ollama_destination_address_family": family,
            "ollama_destination_port": port,
            "ollama_destination_value_exposed": False,
            "ollama_redirects_allowed": False,
            "ollama_proxy_environment_allowed": False,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise SystemExit(f"Ollama destination receipt drifted at {key}: {metadata}")

    with TemporaryDirectory(prefix="jarvis-ollama-diagnostic-policy-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
            model_provider="ollama",
        )
        runtime = JarvisRuntime(config)
        proxy_keys = (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        )
        accepted_env = {
            "JARVIS_MODEL_PROVIDER": "ollama",
            "OLLAMA_HOST": "localhost:22445",
            "OLLAMA_NO_CLOUD": "1",
            "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
            **{key: f"http://proxy-marker-{key}" for key in proxy_keys},
        }
        hidden_probe_value = "private-ollama-probe-payload"
        with patch.dict(os.environ, accepted_env, clear=False):
            with patch("jarvis_v2.tools.system.subprocess.run", side_effect=lambda args, **_: completed(args)):
                setup = setup_check({})

            doctor_subprocess_calls: list[list[str]] = []
            doctor_probe_calls: list[dict] = []

            def doctor_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                doctor_subprocess_calls.append(args)
                return completed(args)

            def doctor_probe(**kwargs: object) -> tuple[bool, list[str], str, str]:
                doctor_probe_calls.append(dict(kwargs))
                return True, [hidden_probe_value], "ollama_probe_ok", hidden_probe_value

            with (
                patch("jarvis_v2.tools.doctor.subprocess.run", side_effect=doctor_run),
                patch("jarvis_v2.tools.doctor.probe_ollama_models", side_effect=doctor_probe),
            ):
                doctor = runtime.handle("jarvis doctor").tool_results[0]

            readiness_probe_calls: list[dict] = []

            def readiness_probe(**kwargs: object) -> tuple[bool, list[str], str, str]:
                readiness_probe_calls.append(dict(kwargs))
                return True, [hidden_probe_value], "ollama_probe_ok", hidden_probe_value

            with patch("jarvis_v2.tools.readiness.probe_ollama_models", side_effect=readiness_probe):
                readiness = runtime.handle("readiness report").tool_results[0]

        for result in (setup, doctor, readiness):
            assert_receipt(
                result.metadata,
                allowed=True,
                source="env",
                family="localhost_pinned_ipv4",
                port=22445,
            )
            for expected_text in (
                "source: env",
                "family: localhost_pinned_ipv4",
                "port: 22445",
                "redirects: blocked",
                "proxy policy: stripped from accepted probes",
            ):
                if expected_text not in result.output:
                    raise SystemExit(f"Ollama diagnostic missed {expected_text!r}: {result.output}")
            expected_policy = {
                "ollama_no_cloud_configured": True,
                "ollama_no_cloud_valid": True,
                "ollama_no_cloud_requested": True,
                "ollama_cloud_model_alias": False,
                "ollama_personal_context_allowed": True,
                "ollama_unverified_context_consent_configured": True,
                "ollama_unverified_context_consent_valid": True,
                "ollama_unverified_context_consent_allowed": True,
                "ollama_execution_locality_verified": False,
                "ollama_daemon_cloud_disabled_verified": False,
                "ollama_on_device_model_execution_verified": False,
                "ollama_cloud_policy_value_exposed": False,
            }
            for key, value in expected_policy.items():
                if result.metadata.get(key) != value:
                    raise SystemExit(f"Ollama no-cloud receipt drifted at {key}: {result.metadata}")
            for expected_text in (
                "configured: yes",
                "valid: yes",
                "requested: yes",
                "consent configured: yes",
                "consent valid: yes",
                "consent granted: yes",
                "execution locality: unknown",
                "daemon cloud-disabled state: independently unverified",
                "on-device execution: independently unverified",
            ):
                if expected_text not in result.output:
                    raise SystemExit(f"Ollama no-cloud diagnostic missed {expected_text!r}: {result.output}")

        if doctor_probe_calls != [{"timeout_seconds": 5.0}]:
            raise SystemExit(f"doctor did not use the shared Ollama HTTP probe once: {doctor_probe_calls}")
        if readiness_probe_calls != [{"timeout_seconds": 5.0}]:
            raise SystemExit(f"readiness did not use the shared Ollama HTTP probe once: {readiness_probe_calls}")
        if ["ollama", "list"] in doctor_subprocess_calls:
            raise SystemExit(f"doctor retained an Ollama subprocess probe: {doctor_subprocess_calls}")
        for result in (doctor, readiness):
            expected_probe_receipt = {
                "ollama_probe_uses_explicit_loopback_env": False,
                "ollama_probe_proxy_environment_stripped": False,
                "ollama_probe_uses_validated_loopback_http": True,
                "ollama_probe_redirects_blocked": True,
                "ollama_probe_proxy_bypassed": True,
                "ollama_probe_uses_subprocess": False,
                "ollama_probe_diagnostic": "ollama_probe_ok",
                "ollama_probe_model_count": 1,
            }
            for key, value in expected_probe_receipt.items():
                if result.metadata.get(key) != value:
                    raise SystemExit(f"{result.tool_name} helper receipt drifted at {key}: {result.metadata}")
            if hidden_probe_value in result.output or hidden_probe_value in str(result.metadata):
                raise SystemExit(f"{result.tool_name} exposed helper model/error content")

        rejected_host = "http://private-rejected-ollama.example:11434"
        rejected_env = {
            "JARVIS_MODEL_PROVIDER": "ollama",
            "OLLAMA_HOST": rejected_host,
            "OLLAMA_NO_CLOUD": "1",
            "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
        }
        with patch.dict(os.environ, rejected_env, clear=False):
            setup_calls: list[list[str]] = []

            def setup_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                setup_calls.append(args)
                return completed(args)

            with patch("jarvis_v2.tools.system.subprocess.run", side_effect=setup_run):
                rejected_setup = setup_check({})

            rejected_doctor_calls: list[list[str]] = []

            def rejected_doctor_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                rejected_doctor_calls.append(args)
                return completed(args)

            with (
                patch("jarvis_v2.tools.doctor.subprocess.run", side_effect=rejected_doctor_run),
                patch(
                    "jarvis_v2.tools.doctor.probe_ollama_models",
                    side_effect=AssertionError("doctor probed rejected Ollama destination"),
                ),
            ):
                rejected_doctor = runtime.handle("jarvis doctor").tool_results[0]

            with patch(
                "jarvis_v2.tools.readiness.probe_ollama_models",
                side_effect=AssertionError("readiness probed rejected Ollama destination"),
            ):
                rejected_readiness = runtime.handle("readiness report").tool_results[0]

        for label, calls in (
            ("setup", setup_calls),
            ("doctor", rejected_doctor_calls),
        ):
            if ["ollama", "list"] in calls:
                raise SystemExit(f"{label} probed a rejected Ollama destination: {calls}")
        for result in (rejected_setup, rejected_doctor, rejected_readiness):
            assert_receipt(result.metadata, allowed=False, source="env-invalid", family="", port=0)
            combined = f"{result.output} {result.metadata}"
            if rejected_host in combined or "private-rejected-ollama" in combined:
                raise SystemExit(f"Ollama diagnostic exposed a rejected host through {result.tool_name}: {combined}")
            if result.metadata.get("ollama_destination_probe_attempted") is not False:
                raise SystemExit(f"Ollama diagnostic claimed a rejected probe through {result.tool_name}: {result.metadata}")
        if "OLLAMA_HOST" not in rejected_setup.metadata.get("setup_attention", []):
            raise SystemExit(f"setup check did not flag rejected OLLAMA_HOST: {rejected_setup.metadata}")
        if rejected_readiness.metadata.get("status") != "not ready":
            raise SystemExit(f"readiness did not block a rejected Ollama destination: {rejected_readiness.metadata}")


def assert_ollama_no_cloud_diagnostics_are_truthful() -> None:
    invalid_marker = "private-invalid-no-cloud-value"
    invalid_consent_marker = "private-invalid-consent-value"
    cases = (
        ("absent", None, None, "llama3.1", False, True, False, False, True, False, False, True),
        ("explicit false", "0", None, "llama3.1", True, True, False, False, True, False, False, True),
        ("consent required", "1", None, "llama3.1", True, True, True, False, True, False, False, True),
        ("explicit true", "1", "1", "llama3.1", True, True, True, True, True, True, True, False),
        ("invalid", invalid_marker, None, "llama3.1", True, False, False, False, True, False, False, True),
        (
            "invalid consent",
            "1",
            invalid_consent_marker,
            "llama3.1",
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
        ),
        ("cloud alias", "1", "1", "gpt-oss:120b-cloud", True, True, True, True, True, True, False, False),
    )

    def completed(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="local probe ok", stderr="")

    for (
        label,
        raw_setting,
        raw_consent,
        model,
        configured,
        valid,
        requested,
        consent_configured,
        consent_valid,
        consent_allowed,
        personal_context_allowed,
        current_message_only_usable,
    ) in cases:
        with TemporaryDirectory(prefix="jarvis-ollama-no-cloud-diagnostic-") as temp:
            root = Path(temp)
            empty_env = root / "empty.env"
            empty_env.write_text("", encoding="utf-8")
            empty_env.chmod(0o600)
            config = JarvisConfig(
                data_dir=root,
                db_path=root / "jarvis.sqlite",
                obsidian_vault=root / "Vault",
                obsidian_root="Jarvis",
                use_model_planner=False,
                model_provider="ollama",
                chat_model=model,
                planner_model=model,
            )
            env = {
                "JARVIS_V3_ENV": str(empty_env),
                "JARVIS_MODEL_PROVIDER": "ollama",
                "OLLAMA_HOST": "127.0.0.1:22445",
                "JARVIS_CHAT_MODEL": model,
                "JARVIS_PLANNER_MODEL": model,
            }
            if raw_setting is not None:
                env["OLLAMA_NO_CLOUD"] = raw_setting
            if raw_consent is not None:
                env["JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"] = raw_consent
            with patch.dict(os.environ, env, clear=True):
                runtime = JarvisRuntime(config)
                with patch("jarvis_v2.tools.system.subprocess.run", side_effect=lambda args, **_: completed(args)):
                    setup = setup_check({})
                with (
                    patch("jarvis_v2.tools.doctor.subprocess.run", side_effect=lambda args, **_: completed(args)),
                    patch(
                        "jarvis_v2.tools.doctor.probe_ollama_models",
                        return_value=(True, ["private-model-alias"], "ollama_probe_ok", "private-error"),
                    ),
                ):
                    doctor = runtime.handle("jarvis doctor").tool_results[0]
                with patch(
                    "jarvis_v2.tools.readiness.probe_ollama_models",
                    return_value=(True, ["private-model-alias"], "ollama_probe_ok", "private-error"),
                ):
                    readiness = runtime.handle("readiness report").tool_results[0]

        expected = {
            "ollama_no_cloud_configured": configured,
            "ollama_no_cloud_valid": valid,
            "ollama_no_cloud_requested": requested,
            "ollama_cloud_model_alias": label == "cloud alias",
            "ollama_personal_context_allowed": personal_context_allowed,
            "ollama_unverified_context_consent_configured": consent_configured,
            "ollama_unverified_context_consent_valid": consent_valid,
            "ollama_unverified_context_consent_allowed": consent_allowed,
            "ollama_execution_locality_verified": False,
            "ollama_daemon_cloud_disabled_verified": False,
            "ollama_on_device_model_execution_verified": False,
            "ollama_cloud_policy_value_exposed": False,
        }
        for result in (setup, doctor, readiness):
            for key, value in expected.items():
                if result.metadata.get(key) != value:
                    raise SystemExit(f"{label} {result.tool_name} drifted at {key}: {result.metadata}")
            for expected_text in (
                f"configured: {'yes' if configured else 'no'}",
                f"valid: {'yes' if valid else 'no'}",
                f"requested: {'yes' if requested else 'no'}",
                f"consent configured: {'yes' if consent_configured else 'no'}",
                f"consent valid: {'yes' if consent_valid else 'no'}",
                f"consent granted: {'yes' if consent_allowed else 'no'}",
                "execution locality: unknown",
                "daemon cloud-disabled state: independently unverified",
                "on-device execution: independently unverified",
            ):
                if expected_text not in result.output:
                    raise SystemExit(
                        f"{label} {result.tool_name} missed no-cloud disclosure {expected_text!r}: {result.output}"
                    )
            combined = f"{result.output} {result.metadata}"
            if invalid_marker in combined or invalid_consent_marker in combined:
                raise SystemExit(f"{label} {result.tool_name} exposed a raw Ollama policy value")
            if "private-model-alias" in combined or "private-error" in combined:
                raise SystemExit(f"{label} {result.tool_name} exposed helper response content")

        if readiness.metadata.get("ollama_personalized_context_ready") is not personal_context_allowed:
            raise SystemExit(f"{label} readiness personalized state drifted: {readiness.metadata}")
        if readiness.metadata.get("ollama_current_message_only_usable") is not current_message_only_usable:
            raise SystemExit(f"{label} readiness current-message state drifted: {readiness.metadata}")
        if readiness.metadata.get("ollama_stored_context_withheld") is personal_context_allowed:
            raise SystemExit(f"{label} readiness stored-context state drifted: {readiness.metadata}")
        if not personal_context_allowed and "stored context withheld" not in readiness.output:
            raise SystemExit(f"{label} readiness missed stored-context withholding: {readiness.output}")
        if label in {"absent", "explicit false", "consent required", "invalid", "invalid consent"}:
            if readiness.metadata.get("model_route_ready") is not True:
                raise SystemExit(f"{label} should preserve the current-message route: {readiness.metadata}")
            if "current-message-only model route may remain usable" not in readiness.output:
                raise SystemExit(f"{label} missed current-message-only disclosure: {readiness.output}")
        if label == "invalid":
            if "OLLAMA_NO_CLOUD" not in setup.metadata.get("setup_attention", []):
                raise SystemExit(f"invalid no-cloud setting was not setup attention: {setup.metadata}")
            if "Ollama personalized context policy: needs attention" not in doctor.output:
                raise SystemExit(f"invalid no-cloud setting was not doctor attention: {doctor.output}")
            if "personalized Ollama operation is not ready" not in readiness.output:
                raise SystemExit(f"invalid no-cloud setting was not personalized-readiness attention: {readiness.output}")
        if label == "invalid consent":
            consent_key = "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"
            if consent_key not in setup.metadata.get("setup_attention", []):
                raise SystemExit(f"invalid consent was not setup attention: {setup.metadata}")
            if readiness.metadata.get("ollama_personalized_context_ready") is not False:
                raise SystemExit(f"invalid consent did not fail closed: {readiness.metadata}")
            if "invalid consent setting" not in readiness.output:
                raise SystemExit(f"invalid consent was not reported content-free: {readiness.output}")


def assert_invalid_provider_blocks_without_provider_probes() -> None:
    invalid_provider = "private-invalid-provider"
    rejected_host = "http://private-invalid-provider-host.example:11434"
    openai_secret = "private-openai-key-value"

    def completed(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="diagnostic command ok", stderr="")

    with TemporaryDirectory(prefix="jarvis-invalid-provider-diagnostic-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
            model_provider="invalid",
        )
        runtime = JarvisRuntime(config)
        env = {
            "JARVIS_MODEL_PROVIDER": invalid_provider,
            "OLLAMA_HOST": rejected_host,
            "OPENAI_API_KEY": openai_secret,
        }
        with patch.dict(os.environ, env, clear=True):
            setup_subprocess_calls: list[list[str]] = []

            def setup_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                setup_subprocess_calls.append(args)
                return completed(args)

            with patch("jarvis_v2.tools.system.subprocess.run", side_effect=setup_run):
                setup = setup_check({})

            doctor_subprocess_calls: list[list[str]] = []

            def doctor_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                doctor_subprocess_calls.append(args)
                return completed(args)

            with (
                patch("jarvis_v2.tools.doctor.subprocess.run", side_effect=doctor_run),
                patch("jarvis_v2.tools.doctor.probe_ollama_models") as doctor_ollama_probe,
            ):
                doctor = runtime.handle("jarvis doctor").tool_results[0]

            with (
                patch("jarvis_v2.tools.readiness.probe_ollama_models") as readiness_ollama_probe,
                patch("jarvis_v2.tools.readiness.openai_api_key_configured") as readiness_openai_probe,
            ):
                readiness = runtime.handle("readiness report").tool_results[0]

        doctor_ollama_probe.assert_not_called()
        readiness_ollama_probe.assert_not_called()
        readiness_openai_probe.assert_not_called()
        for label, calls in (("setup", setup_subprocess_calls), ("doctor", doctor_subprocess_calls)):
            if ["ollama", "list"] in calls:
                raise SystemExit(f"{label} used an Ollama subprocess for an invalid provider: {calls}")
        for result in (setup, doctor, readiness):
            metadata = result.metadata
            if metadata.get("model_provider") != "invalid" or metadata.get("model_provider_valid") is not False:
                raise SystemExit(f"{result.tool_name} did not preserve invalid provider state: {metadata}")
            if metadata.get("ollama_destination_probe_attempted") is not False:
                raise SystemExit(f"{result.tool_name} probed Ollama for an invalid provider: {metadata}")
            if metadata.get("ollama_probe_uses_subprocess") is not False:
                raise SystemExit(f"{result.tool_name} claimed subprocess use for an invalid provider: {metadata}")
            combined = f"{result.output} {metadata}"
            for hidden_value in (invalid_provider, rejected_host, openai_secret):
                if hidden_value in combined:
                    raise SystemExit(f"{result.tool_name} exposed invalid-provider configuration content")
        if "JARVIS_MODEL_PROVIDER" not in setup.metadata.get("setup_attention", []):
            raise SystemExit(f"setup did not flag the invalid provider: {setup.metadata}")
        if "model provider: needs attention" not in doctor.output:
            raise SystemExit(f"doctor did not flag the invalid provider: {doctor.output}")
        if readiness.metadata.get("status") != "not ready":
            raise SystemExit(f"readiness did not block the invalid provider: {readiness.metadata}")
        if "no Ollama or OpenAI connectivity probe attempted" not in readiness.output:
            raise SystemExit(f"readiness missed the invalid-provider probe boundary: {readiness.output}")


def main() -> None:
    assert_setup_check_does_not_read_env_or_secret_values()
    assert_model_timeout_caps_match_setup_and_doctor()
    assert_ollama_destination_diagnostics_are_loopback_only()
    assert_ollama_no_cloud_diagnostics_are_truthful()
    assert_invalid_provider_blocks_without_provider_probes()
    old_host = os.environ.get("JARVIS_STATUS_HOST")
    old_env_file = os.environ.get("JARVIS_V3_ENV")
    old_data_dir = os.environ.get("JARVIS_DATA_DIR")
    old_db_path = os.environ.get("JARVIS_DB_PATH")
    old_obsidian_vault = os.environ.get("JARVIS_OBSIDIAN_VAULT")
    old_secondary_obsidian_vault = os.environ.get("OBSIDIAN_VAULT_PATH")
    old_obsidian_root = os.environ.get("JARVIS_OBSIDIAN_ROOT")
    old_ollama_model = os.environ.get("OLLAMA_MODEL")
    old_chat_model = os.environ.get("JARVIS_CHAT_MODEL")
    old_planner_model = os.environ.get("JARVIS_PLANNER_MODEL")
    old_port = os.environ.get("JARVIS_STATUS_PORT")
    old_model_timeout = os.environ.get("JARVIS_MODEL_TIMEOUT_SECONDS")
    old_chat_timeout = os.environ.get("JARVIS_CHAT_TIMEOUT_SECONDS")
    old_chat_max_reply_tokens = os.environ.get("JARVIS_CHAT_MAX_REPLY_TOKENS")
    old_chat_max_history_messages = os.environ.get("JARVIS_CHAT_MAX_HISTORY_MESSAGES")
    old_model_planner_toggle = os.environ.get("JARVIS_USE_MODEL_PLANNER")
    old_smoke_timeout = os.environ.get("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS")
    old_weather_location = os.environ.get("JARVIS_WEATHER_LOCATION")
    old_news_locale = os.environ.get("JARVIS_NEWS_LOCALE")
    old_telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    old_gmail_address = os.environ.get("GMAIL_ADDRESS")
    old_gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD")
    old_owner_telegram = os.environ.get("JARVIS_OWNER_TELEGRAM")
    old_owner_imessage = os.environ.get("JARVIS_OWNER_IMESSAGE")
    old_google_creds = os.environ.get("JARVIS_GOOGLE_CREDS")
    old_google_readonly_token = os.environ.get("JARVIS_GOOGLE_READONLY_TOKEN")
    old_google_token = os.environ.get("JARVIS_GOOGLE_TOKEN")
    old_watched_dirs = os.environ.get("JARVIS_WATCHED_DIRS")
    old_telegram_state = os.environ.get("JARVIS_TELEGRAM_STATE")
    old_imessage_state = os.environ.get("JARVIS_V3_IMESSAGE_STATE")
    old_reminders_file = os.environ.get("JARVIS_REMINDERS_FILE")
    old_storage_fallback = os.environ.get("JARVIS_STORAGE_FALLBACK_DIR")
    old_storage_fallback_disabled = os.environ.get("JARVIS_DISABLE_STORAGE_FALLBACK")
    old_oav_vision_reviewer = os.environ.get("JARVIS_OAV_VISION_REVIEWER_COMMAND")
    old_voice_whisper_model = os.environ.get("JARVIS_VOICE_WHISPER_MODEL_PATH")
    old_voice_faster_whisper_model = os.environ.get("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH")
    old_home = os.environ.get("HOME")
    try:
        os.environ.pop("JARVIS_GOOGLE_READONLY_TOKEN", None)
        assert_temp_root_system_redaction()
        os.environ["JARVIS_STATUS_HOST"] = "bad host"
        os.environ["JARVIS_STATUS_PORT"] = "not-a-port"
        os.environ["JARVIS_MODEL_TIMEOUT_SECONDS"] = "not-a-planner-timeout"
        os.environ["JARVIS_CHAT_TIMEOUT_SECONDS"] = "not-a-chat-timeout"
        os.environ["JARVIS_CHAT_MAX_REPLY_TOKENS"] = "/\x55sers/example/private/chat-token-cap"
        os.environ["JARVIS_CHAT_MAX_HISTORY_MESSAGES"] = "/\x55sers/example/private/chat-history-window"
        os.environ["JARVIS_USE_MODEL_PLANNER"] = "/\x55sers/example/private/planner-toggle"
        os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = "nan"
        with TemporaryDirectory(prefix="jarvis-setup-status-env-") as temp:
            root = Path(temp)
            home = root / "home"
            env_dir = root / "env-dir"
            data_dir_file = root / "data-dir-file"
            db_path_missing_parent = root / "missing-db-parent" / "jarvis.sqlite"
            obsidian_vault_file = root / "obsidian-vault-file"
            watched_ok = root / "watched-ok"
            watched_missing = root / "watched-missing"
            weather_location_path = "/\x55sers/example/private/weather-default"
            telegram_owner_path = "/\x55sers/example/private/telegram-owner"
            imessage_owner_path = "/\x55sers/example/private/imessage-owner"
            storage_disable_path = "/\x55sers/example/private/storage-disable-toggle"
            telegram_state_dir = root / "telegram-state-dir"
            imessage_state_missing_parent = root / "missing-imessage-parent" / "imessage_state.json"
            reminders_missing_parent = root / "missing-reminders-parent" / "reminders.json"
            storage_fallback_file = root / "fallback-file"
            google_creds_dir = root / "google-creds-dir"
            google_token_missing_parent = root / "missing-google-token-parent" / "google_token.json"
            oav_reviewer_missing = root / "missing-oav-reviewer"
            voice_whisper_model_dir = root / "voice-whisper-model-dir"
            voice_faster_whisper_model_file = root / "voice-faster-whisper-model-file"
            home.mkdir()
            env_dir.mkdir()
            watched_ok.mkdir()
            telegram_state_dir.mkdir()
            google_creds_dir.mkdir()
            voice_whisper_model_dir.mkdir()
            data_dir_file.write_text("not a directory", encoding="utf-8")
            obsidian_vault_file.write_text("not a directory", encoding="utf-8")
            storage_fallback_file.write_text("not a directory", encoding="utf-8")
            voice_faster_whisper_model_file.write_text("not a model directory", encoding="utf-8")
            os.environ["HOME"] = str(home)
            os.environ["JARVIS_V3_ENV"] = str(env_dir)
            os.environ["JARVIS_DATA_DIR"] = str(data_dir_file)
            os.environ["JARVIS_DB_PATH"] = str(db_path_missing_parent)
            os.environ["JARVIS_OBSIDIAN_VAULT"] = str(obsidian_vault_file)
            os.environ["OBSIDIAN_VAULT_PATH"] = "/\x55sers/example/private/secondary-vault"
            os.environ["JARVIS_OBSIDIAN_ROOT"] = "/\x55sers/example/private/obsidian-root"
            os.environ["OLLAMA_MODEL"] = "/\x55sers/example/private/ollama-model"
            os.environ["JARVIS_CHAT_MODEL"] = "/\x55sers/example/private/chat-model"
            os.environ["JARVIS_PLANNER_MODEL"] = "/\x55sers/example/private/planner-model"
            os.environ["JARVIS_WATCHED_DIRS"] = f"{watched_ok}:{watched_missing}"
            os.environ["JARVIS_WEATHER_LOCATION"] = weather_location_path
            os.environ["JARVIS_NEWS_LOCALE"] = "/\x55sers/example/private/news-locale:KR"
            os.environ["TELEGRAM_BOT_TOKEN"] = "/\x55sers/example/private/telegram-token"
            os.environ["GMAIL_ADDRESS"] = "/\x55sers/example/private/gmail-address"
            os.environ["GMAIL_APP_PASSWORD"] = "/\x55sers/example/private/gmail-password"
            os.environ["JARVIS_OWNER_TELEGRAM"] = telegram_owner_path
            os.environ["JARVIS_OWNER_IMESSAGE"] = imessage_owner_path
            os.environ["JARVIS_TELEGRAM_STATE"] = str(telegram_state_dir)
            os.environ["JARVIS_V3_IMESSAGE_STATE"] = str(imessage_state_missing_parent)
            os.environ["JARVIS_REMINDERS_FILE"] = str(reminders_missing_parent)
            os.environ["JARVIS_STORAGE_FALLBACK_DIR"] = str(storage_fallback_file)
            os.environ["JARVIS_DISABLE_STORAGE_FALLBACK"] = storage_disable_path
            os.environ["JARVIS_GOOGLE_CREDS"] = str(google_creds_dir)
            os.environ["JARVIS_GOOGLE_TOKEN"] = str(google_token_missing_parent)
            os.environ["JARVIS_OAV_VISION_REVIEWER_COMMAND"] = f"{oav_reviewer_missing} --json"
            os.environ["JARVIS_VOICE_WHISPER_MODEL_PATH"] = str(voice_whisper_model_dir)
            os.environ["JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH"] = str(voice_faster_whisper_model_file)
            try:
                load_env()
            except OSError:
                pass
            else:
                raise SystemExit("strict env loading should reject an explicit directory path")
            runtime = JarvisRuntime(
                JarvisConfig(
                    data_dir=root,
                    db_path=root / "jarvis.sqlite",
                    obsidian_vault=root / "Vault",
                    obsidian_root="Jarvis",
                    use_model_planner=False,
                )
            )
            with patch("jarvis_v2.env.load_env", side_effect=AssertionError("setup check called strict load_env")):
                result = runtime.handle("setup check")
            if not result.verified:
                raise SystemExit(f"setup check should remain verified with invalid dashboard port: {result.response}")
            output = result.response
            if "Status dashboard host validation: invalid; using default 127.0.0.1 (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid host diagnostic: {output}")
            if "Status dashboard port validation: invalid; using default 8766 (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid port diagnostic: {output}")
            if "Smoke module timeout validation: invalid; using default 180s (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid smoke timeout diagnostic: {output}")
            if "Planner timeout validation: invalid; using default 2.5s (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid planner timeout diagnostic: {output}")
            if "Chat timeout validation: invalid; using default 20s (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid chat timeout diagnostic: {output}")
            if "Chat reply token cap validation: invalid; using default 300 (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid chat reply token cap diagnostic: {output}")
            if "Chat history window validation: invalid; using default 16 (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid chat history window diagnostic: {output}")
            if "Fallback Ollama model alias validation: invalid; using llama3.1 (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid fallback model diagnostic: {output}")
            if "Chat model alias validation: invalid; using fallback model alias (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid chat model diagnostic: {output}")
            if "Planner model alias validation: invalid; using chat model alias (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid planner model diagnostic: {output}")
            if "Model planner toggle validation: invalid; enabled by default (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid model planner toggle diagnostic: {output}")
            if "Custom env file validation: invalid; path is a directory (path hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid env-file diagnostic: {output}")
            if "Jarvis data directory validation: invalid; target is a file (path hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid data-dir diagnostic: {output}")
            if "Jarvis SQLite database path validation: invalid; parent folder is missing (path hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid db-path diagnostic: {output}")
            if "Obsidian vault path validation: invalid; target is a file (path hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid Obsidian vault diagnostic: {output}")
            if "Obsidian Jarvis root folder validation: invalid; using Jarvis (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid Obsidian root diagnostic: {output}")
            if "Watched directories validation: invalid; 1 missing, 0 not directories (paths hidden; contents not scanned)" not in output:
                raise SystemExit(f"setup check missed hidden invalid watched-dirs diagnostic: {output}")
            if "Default weather location validation: invalid; using Seoul (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid weather location diagnostic: {output}")
            if "News locale validation: invalid; using en-US:US (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid news locale diagnostic: {output}")
            if "Telegram bot token presence: key present (value not inspected)" not in output:
                raise SystemExit(f"setup check missed Telegram token presence-only diagnostic: {output}")
            if "Gmail address presence: key present (value not inspected)" not in output:
                raise SystemExit(f"setup check missed Gmail address presence-only diagnostic: {output}")
            if "Gmail app password presence: key present (value not inspected)" not in output:
                raise SystemExit(f"setup check missed Gmail password presence-only diagnostic: {output}")
            if "Telegram owner allowlist validation: invalid; value hidden" not in output:
                raise SystemExit(f"setup check missed hidden invalid Telegram owner diagnostic: {output}")
            if "iMessage owner allowlist validation: invalid; value hidden" not in output:
                raise SystemExit(f"setup check missed hidden invalid iMessage owner diagnostic: {output}")
            if "Telegram command state file validation: invalid; path is a directory (path hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid Telegram state diagnostic: {output}")
            if "iMessage command state file validation: invalid; parent folder is missing (path hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid iMessage state diagnostic: {output}")
            if "Telegram reminders state file validation: invalid; parent folder is missing (path hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid reminders state diagnostic: {output}")
            if "Storage fallback directory target: invalid; target is a file (path hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid storage fallback target diagnostic: {output}")
            if "Storage fallback (JARVIS_DISABLE_STORAGE_FALLBACK): invalid; fallback enabled by default (value hidden)" not in output:
                raise SystemExit(f"setup check missed hidden invalid storage fallback disable diagnostic: {output}")
            if "Google credentials file (JARVIS_GOOGLE_CREDS): invalid; path is a directory (path hidden) (contents not read)" not in output:
                raise SystemExit(f"setup check missed hidden invalid Google credentials diagnostic: {output}")
            if "Google Calendar full-access mutation token file (JARVIS_GOOGLE_TOKEN): not found; parent folder missing (path hidden) (contents not read)" not in output:
                raise SystemExit(f"setup check missed hidden invalid Google token diagnostic: {output}")
            for raw_value in [
                "bad host",
                "not-a-port",
                "not-a-planner-timeout",
                "not-a-chat-timeout",
                "/\x55sers/example/private/chat-token-cap",
                "/\x55sers/example/private/chat-history-window",
                "/\x55sers/example/private/planner-toggle",
                "nan",
                str(env_dir),
                str(data_dir_file),
                str(db_path_missing_parent),
                str(obsidian_vault_file),
                "/\x55sers/example/private/secondary-vault",
                "/\x55sers/example/private/obsidian-root",
                "/\x55sers/example/private/ollama-model",
                "/\x55sers/example/private/chat-model",
                "/\x55sers/example/private/planner-model",
                str(watched_ok),
                str(watched_missing),
                weather_location_path,
                "/\x55sers/example/private/news-locale:KR",
                "/\x55sers/example/private/telegram-token",
                "/\x55sers/example/private/gmail-address",
                "/\x55sers/example/private/gmail-password",
                telegram_owner_path,
                imessage_owner_path,
                storage_disable_path,
                str(telegram_state_dir),
                str(imessage_state_missing_parent),
                str(reminders_missing_parent),
                str(storage_fallback_file),
                str(google_creds_dir),
                str(google_token_missing_parent),
                str(oav_reviewer_missing),
            ]:
                if raw_value in output:
                    raise SystemExit(f"setup check leaked raw invalid env value: {output}")
            metadata = result.tool_results[0].metadata
            if metadata.get("env_file_valid") is not False:
                raise SystemExit(f"setup check missed invalid env-file metadata: {metadata}")
            if metadata.get("env_file_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid env-file source: {metadata}")
            if metadata.get("env_file_is_dir") is not True:
                raise SystemExit(f"setup check missed env-file directory metadata: {metadata}")
            if metadata.get("reads_env_file_contents") is not False:
                raise SystemExit(f"setup check should not read env-file contents: {metadata}")
            if metadata.get("data_dir_valid") is not False:
                raise SystemExit(f"setup check missed invalid data-dir metadata: {metadata}")
            if metadata.get("data_dir_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid data-dir source: {metadata}")
            if metadata.get("data_dir_is_file") is not True:
                raise SystemExit(f"setup check missed data-dir file metadata: {metadata}")
            if metadata.get("creates_data_dir") is not False or metadata.get("scans_data_dir") is not False:
                raise SystemExit(f"setup check should not create or scan data dirs: {metadata}")
            if metadata.get("db_path_valid") is not False:
                raise SystemExit(f"setup check missed invalid db-path metadata: {metadata}")
            if metadata.get("db_path_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid db-path source: {metadata}")
            if metadata.get("db_path_parent_exists") is not False:
                raise SystemExit(f"setup check missed db-path missing-parent metadata: {metadata}")
            if metadata.get("reads_db_file_contents") is not False or metadata.get("creates_db_file") is not False:
                raise SystemExit(f"setup check should not read/create db files: {metadata}")
            if metadata.get("obsidian_vault_valid") is not False:
                raise SystemExit(f"setup check missed invalid Obsidian vault metadata: {metadata}")
            if metadata.get("obsidian_vault_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid Obsidian vault source: {metadata}")
            if metadata.get("obsidian_vault_is_file") is not True:
                raise SystemExit(f"setup check missed Obsidian vault file metadata: {metadata}")
            if metadata.get("obsidian_vault_env_key") != "JARVIS_OBSIDIAN_VAULT":
                raise SystemExit(f"setup check should report primary Obsidian env key: {metadata}")
            if metadata.get("creates_obsidian_vault") is not False or metadata.get("scans_obsidian_vault") is not False:
                raise SystemExit(f"setup check should not create or scan Obsidian vaults: {metadata}")
            if metadata.get("obsidian_root_valid") is not False:
                raise SystemExit(f"setup check missed invalid Obsidian root metadata: {metadata}")
            if metadata.get("obsidian_root_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid Obsidian root source: {metadata}")
            if metadata.get("obsidian_root_fallback") != "Jarvis":
                raise SystemExit(f"setup check missed safe Obsidian root fallback metadata: {metadata}")
            if metadata.get("watched_dirs_valid") is not False:
                raise SystemExit(f"setup check missed invalid watched-dirs metadata: {metadata}")
            if metadata.get("watched_dirs_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid watched-dirs source: {metadata}")
            if metadata.get("watched_dirs_count") != 2:
                raise SystemExit(f"setup check missed watched-dirs count metadata: {metadata}")
            if metadata.get("watched_dirs_existing_dirs") != 1:
                raise SystemExit(f"setup check missed watched-dirs existing count: {metadata}")
            if metadata.get("watched_dirs_missing") != 1:
                raise SystemExit(f"setup check missed watched-dirs missing count: {metadata}")
            if metadata.get("watched_dirs_not_dirs") != 0:
                raise SystemExit(f"setup check missed watched-dirs not-dir count: {metadata}")
            if metadata.get("scans_watched_dirs") is not False:
                raise SystemExit(f"setup check should not scan watched directory contents: {metadata}")
            if metadata.get("weather_location_valid") is not False:
                raise SystemExit(f"setup check missed invalid weather location metadata: {metadata}")
            if metadata.get("weather_location_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid weather location source: {metadata}")
            if metadata.get("weather_location_fallback") != "Seoul":
                raise SystemExit(f"setup check missed safe weather fallback metadata: {metadata}")
            if metadata.get("news_locale_valid") is not False:
                raise SystemExit(f"setup check missed invalid news locale metadata: {metadata}")
            if metadata.get("news_locale_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid news locale source: {metadata}")
            if metadata.get("news_locale_fallback_hl") != "en-US" or metadata.get("news_locale_fallback_gl") != "US":
                raise SystemExit(f"setup check missed safe news locale fallback metadata: {metadata}")
            if metadata.get("telegram_bot_token_valid") is not None:
                raise SystemExit(f"setup check should not validate Telegram bot token values: {metadata}")
            if metadata.get("telegram_bot_token_source") != "environment-key":
                raise SystemExit(f"setup check missed Telegram bot token key-presence source: {metadata}")
            if metadata.get("telegram_bot_token_value_inspected") is not False:
                raise SystemExit(f"setup check inspected Telegram bot token value: {metadata}")
            if metadata.get("telegram_bot_token_required") is not True:
                raise SystemExit(f"setup check missed required Telegram bot token metadata: {metadata}")
            if metadata.get("gmail_address_valid") is not None:
                raise SystemExit(f"setup check should not validate Gmail address values: {metadata}")
            if metadata.get("gmail_address_source") != "environment-key":
                raise SystemExit(f"setup check missed Gmail address key-presence source: {metadata}")
            if metadata.get("gmail_address_value_inspected") is not False:
                raise SystemExit(f"setup check inspected Gmail address value: {metadata}")
            if metadata.get("gmail_address_required") is not True:
                raise SystemExit(f"setup check missed required Gmail address metadata: {metadata}")
            if metadata.get("gmail_app_password_valid") is not None:
                raise SystemExit(f"setup check should not validate Gmail app password values: {metadata}")
            if metadata.get("gmail_app_password_source") != "environment-key":
                raise SystemExit(f"setup check missed Gmail password key-presence source: {metadata}")
            if metadata.get("gmail_app_password_value_inspected") is not False:
                raise SystemExit(f"setup check inspected Gmail app password value: {metadata}")
            if metadata.get("gmail_app_password_required") is not True:
                raise SystemExit(f"setup check missed required Gmail app password metadata: {metadata}")
            if metadata.get("telegram_owner_allowlist_valid") is not False:
                raise SystemExit(f"setup check missed invalid Telegram owner metadata: {metadata}")
            if metadata.get("telegram_owner_allowlist_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid Telegram owner source: {metadata}")
            if metadata.get("telegram_owner_allowlist_configured") is not True:
                raise SystemExit(f"setup check missed configured Telegram owner metadata: {metadata}")
            if metadata.get("telegram_owner_allowlist_required") is not True:
                raise SystemExit(f"setup check missed required Telegram owner metadata: {metadata}")
            if metadata.get("imessage_owner_allowlist_valid") is not False:
                raise SystemExit(f"setup check missed invalid iMessage owner metadata: {metadata}")
            if metadata.get("imessage_owner_allowlist_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid iMessage owner source: {metadata}")
            if metadata.get("imessage_owner_allowlist_configured") is not True:
                raise SystemExit(f"setup check missed configured iMessage owner metadata: {metadata}")
            if metadata.get("imessage_owner_allowlist_required") is not False:
                raise SystemExit(f"setup check missed optional iMessage owner metadata: {metadata}")
            if metadata.get("telegram_state_file_valid") is not False:
                raise SystemExit(f"setup check missed invalid Telegram state metadata: {metadata}")
            if metadata.get("telegram_state_file_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid Telegram state source: {metadata}")
            if metadata.get("telegram_state_file_is_dir") is not True:
                raise SystemExit(f"setup check missed Telegram state directory metadata: {metadata}")
            if metadata.get("reads_telegram_state_file_contents") is not False:
                raise SystemExit(f"setup check should not read Telegram state contents: {metadata}")
            if metadata.get("creates_telegram_state_file") is not False:
                raise SystemExit(f"setup check should not create Telegram state files: {metadata}")
            if metadata.get("imessage_state_file_valid") is not False:
                raise SystemExit(f"setup check missed invalid iMessage state metadata: {metadata}")
            if metadata.get("imessage_state_file_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid iMessage state source: {metadata}")
            if metadata.get("imessage_state_file_parent_exists") is not False:
                raise SystemExit(f"setup check missed iMessage state missing-parent metadata: {metadata}")
            if metadata.get("reads_imessage_state_file_contents") is not False:
                raise SystemExit(f"setup check should not read iMessage state contents: {metadata}")
            if metadata.get("creates_imessage_state_file") is not False:
                raise SystemExit(f"setup check should not create iMessage state files: {metadata}")
            if metadata.get("reminders_state_file_valid") is not False:
                raise SystemExit(f"setup check missed invalid reminders state metadata: {metadata}")
            if metadata.get("reminders_state_file_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid reminders state source: {metadata}")
            if metadata.get("reminders_state_file_parent_exists") is not False:
                raise SystemExit(f"setup check missed reminders missing-parent metadata: {metadata}")
            if metadata.get("reads_reminders_state_file_contents") is not False:
                raise SystemExit(f"setup check should not read reminders state contents: {metadata}")
            if metadata.get("creates_reminders_state_file") is not False:
                raise SystemExit(f"setup check should not create reminders state files: {metadata}")
            if metadata.get("storage_fallback_valid") is not False:
                raise SystemExit(f"setup check missed invalid storage fallback metadata: {metadata}")
            if metadata.get("storage_fallback_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid storage fallback source: {metadata}")
            if metadata.get("storage_fallback_disabled") is not False:
                raise SystemExit(f"setup check should keep fallback enabled for invalid disable toggle: {metadata}")
            if metadata.get("storage_fallback_disable_configured") is not True:
                raise SystemExit(f"setup check missed configured storage fallback disable metadata: {metadata}")
            if metadata.get("storage_fallback_disable_valid") is not False:
                raise SystemExit(f"setup check missed invalid storage fallback disable metadata: {metadata}")
            if metadata.get("storage_fallback_disable_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid storage fallback disable source: {metadata}")
            if metadata.get("storage_fallback_disable_fallback_disabled") is not False:
                raise SystemExit(f"setup check missed safe storage fallback disable fallback metadata: {metadata}")
            if metadata.get("storage_fallback_is_file") is not True:
                raise SystemExit(f"setup check missed storage fallback file metadata: {metadata}")
            if metadata.get("storage_fallback_parent_writable") is not True:
                raise SystemExit(f"setup check should still see storage fallback parent as writable: {metadata}")
            if metadata.get("creates_storage_fallback_dir") is not False:
                raise SystemExit(f"setup check should not create storage fallback dirs: {metadata}")
            if metadata.get("scans_storage_fallback_dir") is not False:
                raise SystemExit(f"setup check should not scan storage fallback dirs: {metadata}")
            if metadata.get("status_dashboard_host_valid") is not False:
                raise SystemExit(f"setup check missed invalid status host metadata: {metadata}")
            if metadata.get("status_dashboard_host_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid status host source: {metadata}")
            if metadata.get("status_dashboard_port_valid") is not False:
                raise SystemExit(f"setup check missed invalid status port metadata: {metadata}")
            if metadata.get("status_dashboard_port_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid status port source: {metadata}")
            if metadata.get("smoke_module_timeout_valid") is not False:
                raise SystemExit(f"setup check missed invalid smoke timeout metadata: {metadata}")
            if metadata.get("smoke_module_timeout_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid smoke timeout source: {metadata}")
            if metadata.get("smoke_module_timeout_default") != 180:
                raise SystemExit(f"setup check missed smoke timeout default metadata: {metadata}")
            if metadata.get("planner_timeout_valid") is not False:
                raise SystemExit(f"setup check missed invalid planner timeout metadata: {metadata}")
            if metadata.get("planner_timeout_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid planner timeout source: {metadata}")
            if metadata.get("planner_timeout_default") != 2.5:
                raise SystemExit(f"setup check missed planner timeout default metadata: {metadata}")
            if metadata.get("chat_timeout_valid") is not False:
                raise SystemExit(f"setup check missed invalid chat timeout metadata: {metadata}")
            if metadata.get("chat_timeout_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid chat timeout source: {metadata}")
            if metadata.get("chat_timeout_default") != 20.0:
                raise SystemExit(f"setup check missed chat timeout default metadata: {metadata}")
            if metadata.get("chat_max_reply_tokens_valid") is not False:
                raise SystemExit(f"setup check missed invalid chat reply token cap metadata: {metadata}")
            if metadata.get("chat_max_reply_tokens_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid chat reply token cap source: {metadata}")
            if metadata.get("chat_max_reply_tokens_default") != 300:
                raise SystemExit(f"setup check missed chat reply token cap default metadata: {metadata}")
            if metadata.get("chat_max_reply_tokens_effective") != 300:
                raise SystemExit(f"setup check missed chat reply token cap effective fallback: {metadata}")
            if metadata.get("chat_max_history_messages_valid") is not False:
                raise SystemExit(f"setup check missed invalid chat history window metadata: {metadata}")
            if metadata.get("chat_max_history_messages_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid chat history window source: {metadata}")
            if metadata.get("chat_max_history_messages_default") != 16:
                raise SystemExit(f"setup check missed chat history window default metadata: {metadata}")
            if metadata.get("chat_max_history_messages_effective") != 16:
                raise SystemExit(f"setup check missed chat history window effective fallback: {metadata}")
            if metadata.get("fallback_model_alias_valid") is not False:
                raise SystemExit(f"setup check missed invalid fallback model alias metadata: {metadata}")
            if metadata.get("fallback_model_alias_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid fallback model alias source: {metadata}")
            if metadata.get("fallback_model_alias_fallback") != "llama3.1":
                raise SystemExit(f"setup check missed fallback model alias fallback metadata: {metadata}")
            if metadata.get("chat_model_alias_valid") is not False:
                raise SystemExit(f"setup check missed invalid chat model alias metadata: {metadata}")
            if metadata.get("chat_model_alias_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid chat model alias source: {metadata}")
            if metadata.get("chat_model_alias_fallback") != "llama3.1":
                raise SystemExit(f"setup check missed chat model alias fallback metadata: {metadata}")
            if metadata.get("planner_model_alias_valid") is not False:
                raise SystemExit(f"setup check missed invalid planner model alias metadata: {metadata}")
            if metadata.get("planner_model_alias_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid planner model alias source: {metadata}")
            if metadata.get("planner_model_alias_fallback") != "llama3.1":
                raise SystemExit(f"setup check missed planner model alias fallback metadata: {metadata}")
            if metadata.get("model_planner_toggle_valid") is not False:
                raise SystemExit(f"setup check missed invalid model planner toggle metadata: {metadata}")
            if metadata.get("model_planner_toggle_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid model planner toggle source: {metadata}")
            if metadata.get("model_planner_toggle_effective_enabled") is not True:
                raise SystemExit(f"setup check missed model planner toggle enabled fallback metadata: {metadata}")
            if metadata.get("model_planner_toggle_fallback_enabled") is not True:
                raise SystemExit(f"setup check missed model planner toggle fallback metadata: {metadata}")
            if metadata.get("google_credentials_present") is not False:
                raise SystemExit(f"directory Google creds env should not make setup check report credentials present: {metadata}")
            if metadata.get("google_credentials_valid") is not False:
                raise SystemExit(f"setup check missed invalid Google credentials metadata: {metadata}")
            if metadata.get("google_credentials_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid Google credentials source: {metadata}")
            if metadata.get("google_credentials_is_dir") is not True:
                raise SystemExit(f"setup check missed Google credentials directory metadata: {metadata}")
            if metadata.get("google_credentials_is_file") is not False:
                raise SystemExit(f"setup check should require Google credentials to be a file: {metadata}")
            if metadata.get("google_token_present") is not False:
                raise SystemExit(f"missing-parent Google token env should not make setup check report token present: {metadata}")
            if metadata.get("google_token_valid") is not False:
                raise SystemExit(f"setup check missed invalid Google token metadata: {metadata}")
            if metadata.get("google_token_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid Google token source: {metadata}")
            if metadata.get("google_token_parent_exists") is not False:
                raise SystemExit(f"setup check missed Google token missing-parent metadata: {metadata}")
            for env_key in [
                "JARVIS_STATUS_HOST",
                "JARVIS_STATUS_PORT",
                "JARVIS_MODEL_TIMEOUT_SECONDS",
                "JARVIS_CHAT_TIMEOUT_SECONDS",
                "JARVIS_CHAT_MAX_REPLY_TOKENS",
                "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
                "OLLAMA_MODEL",
                "JARVIS_CHAT_MODEL",
                "JARVIS_PLANNER_MODEL",
                "JARVIS_USE_MODEL_PLANNER",
                "JARVIS_DISABLE_STORAGE_FALLBACK",
                "JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS",
                "JARVIS_DATA_DIR",
                "JARVIS_DB_PATH",
                "JARVIS_OBSIDIAN_VAULT",
                "JARVIS_OBSIDIAN_ROOT",
                "JARVIS_WEATHER_LOCATION",
                "JARVIS_NEWS_LOCALE",
                "JARVIS_OWNER_TELEGRAM",
                "JARVIS_OWNER_IMESSAGE",
                "JARVIS_V3_ENV",
                "JARVIS_WATCHED_DIRS",
                "JARVIS_TELEGRAM_STATE",
                "JARVIS_V3_IMESSAGE_STATE",
                "JARVIS_REMINDERS_FILE",
                "JARVIS_STORAGE_FALLBACK_DIR",
                "JARVIS_GOOGLE_CREDS",
                "JARVIS_GOOGLE_TOKEN",
                "JARVIS_OAV_VISION_REVIEWER_COMMAND",
                "JARVIS_VOICE_WHISPER_MODEL_PATH",
                "JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH",
            ]:
                if env_key not in metadata.get("setup_attention", []):
                    raise SystemExit(f"setup check should flag invalid {env_key} for attention: {metadata}")
            if "Local OAV vision reviewer command validation: invalid; executable not resolvable (value hidden)" not in output:
                raise SystemExit(f"setup check missed invalid OAV reviewer diagnostic: {output}")
            if metadata.get("oav_vision_reviewer_command_valid") is not False:
                raise SystemExit(f"setup check missed invalid OAV reviewer metadata: {metadata}")
            if metadata.get("oav_vision_reviewer_command_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid OAV reviewer source: {metadata}")
            if metadata.get("oav_vision_reviewer_command_configured") is not True:
                raise SystemExit(f"setup check missed configured OAV reviewer metadata: {metadata}")
            if metadata.get("oav_vision_reviewer_command_executable_resolved") is not False:
                raise SystemExit(f"setup check should not resolve missing OAV reviewer executable: {metadata}")
            if metadata.get("executes_oav_vision_reviewer_command") is not False:
                raise SystemExit(f"setup check must not execute the OAV reviewer command: {metadata}")
            if "Local Whisper model path validation: invalid; path is a directory (path hidden)" not in output:
                raise SystemExit(f"setup check missed invalid Whisper model diagnostic: {output}")
            if "Local faster-whisper model path validation: invalid; target is a file (path hidden)" not in output:
                raise SystemExit(f"setup check missed invalid faster-whisper model diagnostic: {output}")
            if metadata.get("voice_whisper_model_path_valid") is not False or metadata.get("voice_whisper_model_path_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid Whisper model metadata: {metadata}")
            if metadata.get("voice_whisper_model_path_configured") is not True or metadata.get("voice_whisper_model_path_is_dir") is not True:
                raise SystemExit(f"setup check missed Whisper model path shape metadata: {metadata}")
            if metadata.get("voice_faster_whisper_model_path_valid") is not False or metadata.get("voice_faster_whisper_model_path_source") != "env-invalid":
                raise SystemExit(f"setup check missed invalid faster-whisper model metadata: {metadata}")
            if metadata.get("voice_faster_whisper_model_path_configured") is not True or metadata.get("voice_faster_whisper_model_path_is_file") is not True:
                raise SystemExit(f"setup check missed faster-whisper model path shape metadata: {metadata}")
            for key in ["loads_voice_whisper_model", "loads_voice_faster_whisper_model", "reads_audio_for_voice_model_validation"]:
                if metadata.get(key) is not False:
                    raise SystemExit(f"setup check should not load models or read audio during model path validation ({key}): {metadata}")

            with (
                patch(
                    "jarvis_v2.tools.doctor.status_host_config_from_env",
                    return_value=StatusHostConfig("127.0.0.1", configured=True, valid=False, source="env-invalid"),
                ),
                patch(
                    "jarvis_v2.tools.doctor.status_port_config_from_env",
                    return_value=StatusPortConfig(8766, configured=True, valid=False, source="env-invalid"),
                ),
            ):
                doctor_result = runtime.handle("jarvis doctor")
            if not doctor_result.verified:
                raise SystemExit(f"jarvis doctor should remain verified with invalid dashboard port: {doctor_result.response}")
            doctor_output = doctor_result.response
            if "dashboard host validation: invalid; using default 127.0.0.1 (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid host diagnostic: {doctor_output}")
            if "dashboard port validation: invalid; using default 8766 (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid port diagnostic: {doctor_output}")
            if "smoke module timeout validation: invalid; using default 180s (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid smoke timeout diagnostic: {doctor_output}")
            if "planner timeout validation: invalid; using default 2.5s (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid planner timeout diagnostic: {doctor_output}")
            if "chat timeout validation: invalid; using default 20s (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid chat timeout diagnostic: {doctor_output}")
            if "chat reply token cap validation: invalid; using default 300 (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid chat reply token cap diagnostic: {doctor_output}")
            if "chat history window validation: invalid; using default 16 (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid chat history window diagnostic: {doctor_output}")
            if "fallback Ollama model alias validation: invalid; using llama3.1 (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid fallback model diagnostic: {doctor_output}")
            if "chat model alias validation: invalid; using fallback model alias (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid chat model diagnostic: {doctor_output}")
            if "planner model alias validation: invalid; using chat model alias (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid planner model diagnostic: {doctor_output}")
            if "model planner toggle validation: invalid; enabled by default (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid model planner toggle diagnostic: {doctor_output}")
            if "storage fallback disable validation: invalid; fallback enabled by default (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid storage fallback disable diagnostic: {doctor_output}")
            if "Telegram bot token validation: invalid (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid Telegram bot token diagnostic: {doctor_output}")
            if "Gmail address validation: invalid (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid Gmail address diagnostic: {doctor_output}")
            if "Gmail app password validation: invalid (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid Gmail app password diagnostic: {doctor_output}")
            if "local OAV vision reviewer command validation: invalid; executable not resolvable (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid OAV reviewer diagnostic: {doctor_output}")
            if "local Whisper model path validation: invalid; path is a directory (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid Whisper model diagnostic: {doctor_output}")
            if "local faster-whisper model path validation: invalid; target is a file (value hidden)" not in doctor_output:
                raise SystemExit(f"jarvis doctor missed hidden invalid faster-whisper model diagnostic: {doctor_output}")
            for raw_value in [
                "bad host",
                "not-a-port",
                "not-a-planner-timeout",
                "not-a-chat-timeout",
                "/\x55sers/example/private/chat-token-cap",
                "/\x55sers/example/private/chat-history-window",
                "/\x55sers/example/private/planner-toggle",
                "nan",
                str(env_dir),
                str(data_dir_file),
                str(db_path_missing_parent),
                str(obsidian_vault_file),
                "/\x55sers/example/private/secondary-vault",
                "/\x55sers/example/private/obsidian-root",
                "/\x55sers/example/private/ollama-model",
                "/\x55sers/example/private/chat-model",
                "/\x55sers/example/private/planner-model",
                str(watched_ok),
                str(watched_missing),
                weather_location_path,
                "/\x55sers/example/private/news-locale:KR",
                "/\x55sers/example/private/telegram-token",
                "/\x55sers/example/private/gmail-address",
                "/\x55sers/example/private/gmail-password",
                telegram_owner_path,
                imessage_owner_path,
                storage_disable_path,
                str(telegram_state_dir),
                str(imessage_state_missing_parent),
                str(reminders_missing_parent),
                str(storage_fallback_file),
                str(google_creds_dir),
                str(google_token_missing_parent),
                str(oav_reviewer_missing),
                str(voice_whisper_model_dir),
                str(voice_faster_whisper_model_file),
            ]:
                if raw_value in doctor_output:
                    raise SystemExit(f"jarvis doctor leaked raw invalid env value: {doctor_output}")
            doctor_metadata = doctor_result.tool_results[0].metadata
            assert_doctor_no_future_authority(doctor_metadata, "jarvis doctor")
            if doctor_metadata.get("status_dashboard_host_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid status host metadata: {doctor_metadata}")
            if doctor_metadata.get("status_dashboard_host_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid status host source: {doctor_metadata}")
            if doctor_metadata.get("status_dashboard_host_default") != "127.0.0.1":
                raise SystemExit(f"jarvis doctor missed status host default metadata: {doctor_metadata}")
            if doctor_metadata.get("status_dashboard_port_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid status port metadata: {doctor_metadata}")
            if doctor_metadata.get("status_dashboard_port_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid status port source: {doctor_metadata}")
            if doctor_metadata.get("status_dashboard_port_default") != 8766:
                raise SystemExit(f"jarvis doctor missed status port default metadata: {doctor_metadata}")
            if doctor_metadata.get("smoke_module_timeout_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid smoke timeout metadata: {doctor_metadata}")
            if doctor_metadata.get("smoke_module_timeout_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid smoke timeout source: {doctor_metadata}")
            if doctor_metadata.get("smoke_module_timeout_default") != 180:
                raise SystemExit(f"jarvis doctor missed smoke timeout default metadata: {doctor_metadata}")
            if doctor_metadata.get("planner_timeout_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid planner timeout metadata: {doctor_metadata}")
            if doctor_metadata.get("planner_timeout_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid planner timeout source: {doctor_metadata}")
            if doctor_metadata.get("planner_timeout_default") != 2.5:
                raise SystemExit(f"jarvis doctor missed planner timeout default metadata: {doctor_metadata}")
            if doctor_metadata.get("chat_timeout_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid chat timeout metadata: {doctor_metadata}")
            if doctor_metadata.get("chat_timeout_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid chat timeout source: {doctor_metadata}")
            if doctor_metadata.get("chat_timeout_default") != 20.0:
                raise SystemExit(f"jarvis doctor missed chat timeout default metadata: {doctor_metadata}")
            if doctor_metadata.get("chat_max_reply_tokens_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid chat reply token cap metadata: {doctor_metadata}")
            if doctor_metadata.get("chat_max_reply_tokens_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid chat reply token cap source: {doctor_metadata}")
            if doctor_metadata.get("chat_max_reply_tokens_default") != 300:
                raise SystemExit(f"jarvis doctor missed chat reply token cap default metadata: {doctor_metadata}")
            if doctor_metadata.get("chat_max_reply_tokens_effective") != 300:
                raise SystemExit(f"jarvis doctor missed chat reply token cap effective fallback: {doctor_metadata}")
            if doctor_metadata.get("chat_max_history_messages_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid chat history window metadata: {doctor_metadata}")
            if doctor_metadata.get("chat_max_history_messages_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid chat history window source: {doctor_metadata}")
            if doctor_metadata.get("chat_max_history_messages_default") != 16:
                raise SystemExit(f"jarvis doctor missed chat history window default metadata: {doctor_metadata}")
            if doctor_metadata.get("chat_max_history_messages_effective") != 16:
                raise SystemExit(f"jarvis doctor missed chat history window effective fallback: {doctor_metadata}")
            if doctor_metadata.get("fallback_model_alias_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid fallback model alias metadata: {doctor_metadata}")
            if doctor_metadata.get("fallback_model_alias_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid fallback model alias source: {doctor_metadata}")
            if doctor_metadata.get("fallback_model_alias_fallback") != "llama3.1":
                raise SystemExit(f"jarvis doctor missed fallback model alias fallback metadata: {doctor_metadata}")
            if doctor_metadata.get("chat_model_alias_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid chat model alias metadata: {doctor_metadata}")
            if doctor_metadata.get("chat_model_alias_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid chat model alias source: {doctor_metadata}")
            if doctor_metadata.get("chat_model_alias_fallback") != "llama3.1":
                raise SystemExit(f"jarvis doctor missed chat model alias fallback metadata: {doctor_metadata}")
            if doctor_metadata.get("planner_model_alias_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid planner model alias metadata: {doctor_metadata}")
            if doctor_metadata.get("planner_model_alias_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid planner model alias source: {doctor_metadata}")
            if doctor_metadata.get("planner_model_alias_fallback") != "llama3.1":
                raise SystemExit(f"jarvis doctor missed planner model alias fallback metadata: {doctor_metadata}")
            if doctor_metadata.get("model_planner_toggle_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid model planner toggle metadata: {doctor_metadata}")
            if doctor_metadata.get("model_planner_toggle_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid model planner toggle source: {doctor_metadata}")
            if doctor_metadata.get("model_planner_toggle_effective_enabled") is not True:
                raise SystemExit(f"jarvis doctor missed model planner toggle enabled fallback metadata: {doctor_metadata}")
            if doctor_metadata.get("model_planner_toggle_fallback_enabled") is not True:
                raise SystemExit(f"jarvis doctor missed model planner toggle fallback metadata: {doctor_metadata}")
            if doctor_metadata.get("storage_fallback_disabled") is not False:
                raise SystemExit(f"jarvis doctor should keep fallback enabled for invalid disable toggle: {doctor_metadata}")
            if doctor_metadata.get("storage_fallback_disable_configured") is not True:
                raise SystemExit(f"jarvis doctor missed configured storage fallback disable metadata: {doctor_metadata}")
            if doctor_metadata.get("storage_fallback_disable_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid storage fallback disable metadata: {doctor_metadata}")
            if doctor_metadata.get("storage_fallback_disable_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid storage fallback disable source: {doctor_metadata}")
            if doctor_metadata.get("storage_fallback_disable_fallback_disabled") is not False:
                raise SystemExit(f"jarvis doctor missed safe storage fallback disable fallback metadata: {doctor_metadata}")
            if doctor_metadata.get("telegram_bot_token_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid Telegram bot token metadata: {doctor_metadata}")
            if doctor_metadata.get("telegram_bot_token_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid Telegram bot token source: {doctor_metadata}")
            if doctor_metadata.get("gmail_address_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid Gmail address metadata: {doctor_metadata}")
            if doctor_metadata.get("gmail_address_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid Gmail address source: {doctor_metadata}")
            if doctor_metadata.get("gmail_app_password_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid Gmail app password metadata: {doctor_metadata}")
            if doctor_metadata.get("gmail_app_password_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid Gmail app password source: {doctor_metadata}")
            if doctor_metadata.get("oav_vision_reviewer_command_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid OAV reviewer metadata: {doctor_metadata}")
            if doctor_metadata.get("oav_vision_reviewer_command_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid OAV reviewer source: {doctor_metadata}")
            if doctor_metadata.get("oav_vision_reviewer_command_configured") is not True:
                raise SystemExit(f"jarvis doctor missed configured OAV reviewer metadata: {doctor_metadata}")
            if doctor_metadata.get("oav_vision_reviewer_command_executable_resolved") is not False:
                raise SystemExit(f"jarvis doctor should not resolve missing OAV reviewer executable: {doctor_metadata}")
            if doctor_metadata.get("executes_oav_vision_reviewer_command") is not False:
                raise SystemExit(f"jarvis doctor must not execute the OAV reviewer command: {doctor_metadata}")
            if doctor_metadata.get("voice_whisper_model_path_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid Whisper model metadata: {doctor_metadata}")
            if doctor_metadata.get("voice_whisper_model_path_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid Whisper model source: {doctor_metadata}")
            if doctor_metadata.get("voice_whisper_model_path_configured") is not True or doctor_metadata.get("voice_whisper_model_path_is_dir") is not True:
                raise SystemExit(f"jarvis doctor missed Whisper model path shape metadata: {doctor_metadata}")
            if doctor_metadata.get("voice_faster_whisper_model_path_valid") is not False:
                raise SystemExit(f"jarvis doctor missed invalid faster-whisper model metadata: {doctor_metadata}")
            if doctor_metadata.get("voice_faster_whisper_model_path_source") != "env-invalid":
                raise SystemExit(f"jarvis doctor missed invalid faster-whisper model source: {doctor_metadata}")
            if doctor_metadata.get("voice_faster_whisper_model_path_configured") is not True or doctor_metadata.get("voice_faster_whisper_model_path_is_file") is not True:
                raise SystemExit(f"jarvis doctor missed faster-whisper model path shape metadata: {doctor_metadata}")
            for key in ["loads_voice_whisper_model", "loads_voice_faster_whisper_model", "reads_audio_for_voice_model_validation"]:
                if doctor_metadata.get(key) is not False:
                    raise SystemExit(f"jarvis doctor should not load models or read audio during model path validation ({key}): {doctor_metadata}")

            oversized_reply_tokens = "9" * 120
            oversized_history_messages = "8" * 120
            os.environ["JARVIS_CHAT_MAX_REPLY_TOKENS"] = oversized_reply_tokens
            os.environ["JARVIS_CHAT_MAX_HISTORY_MESSAGES"] = oversized_history_messages
            bounded_setup_result = runtime.handle("setup check")
            with (
                patch(
                    "jarvis_v2.tools.doctor.status_host_config_from_env",
                    return_value=StatusHostConfig("127.0.0.1", configured=True, valid=False, source="env-invalid"),
                ),
                patch(
                    "jarvis_v2.tools.doctor.status_port_config_from_env",
                    return_value=StatusPortConfig(8766, configured=True, valid=False, source="env-invalid"),
                ),
            ):
                bounded_doctor_result = runtime.handle("jarvis doctor")
            for label, bounded_result in [
                ("setup check", bounded_setup_result),
                ("jarvis doctor", bounded_doctor_result),
            ]:
                bounded_output = bounded_result.response
                bounded_metadata = bounded_result.tool_results[0].metadata
                if oversized_reply_tokens in bounded_output or oversized_history_messages in bounded_output:
                    raise SystemExit(f"{label} leaked oversized chat resource env values: {bounded_output}")
                if "chat reply token cap validation: ok; clamped to maximum 4096 (value hidden)" not in bounded_output.lower():
                    raise SystemExit(f"{label} missed bounded reply token diagnostic: {bounded_output}")
                if "chat history window validation: ok; clamped to maximum 64 (value hidden)" not in bounded_output.lower():
                    raise SystemExit(f"{label} missed bounded history diagnostic: {bounded_output}")
                expected_metadata = {
                    "chat_max_reply_tokens_valid": True,
                    "chat_max_reply_tokens_source": "env-clamped",
                    "chat_max_reply_tokens_maximum": 4096,
                    "chat_max_reply_tokens_effective": 4096,
                    "chat_max_history_messages_valid": True,
                    "chat_max_history_messages_source": "env-clamped",
                    "chat_max_history_messages_maximum": 64,
                    "chat_max_history_messages_effective": 64,
                }
                for key, expected in expected_metadata.items():
                    if bounded_metadata.get(key) != expected:
                        raise SystemExit(f"{label} chat cap metadata mismatch for {key}: {bounded_metadata}")
                for metadata_value in bounded_metadata.values():
                    if metadata_value == oversized_reply_tokens or metadata_value == oversized_history_messages:
                        raise SystemExit(f"{label} exposed an oversized raw chat resource value: {bounded_metadata}")

            os.environ["JARVIS_DISABLE_STORAGE_FALLBACK"] = "on"
            disabled_result = runtime.handle("setup check")
            if not disabled_result.verified:
                raise SystemExit(f"setup check should remain verified with disabled storage fallback: {disabled_result.response}")
            disabled_output = disabled_result.response
            if "Storage fallback (JARVIS_DISABLE_STORAGE_FALLBACK): ok; fallback disabled (value hidden)" not in disabled_output:
                raise SystemExit(f"setup check missed valid storage fallback disable diagnostic: {disabled_output}")
            if "Storage fallback directory target: disabled; not checked" not in disabled_output:
                raise SystemExit(f"setup check should skip fallback target when disabled: {disabled_output}")
            disabled_metadata = disabled_result.tool_results[0].metadata
            if disabled_metadata.get("storage_fallback_disabled") is not True:
                raise SystemExit(f"setup check missed disabled storage fallback metadata: {disabled_metadata}")
            if disabled_metadata.get("storage_fallback_disable_valid") is not True:
                raise SystemExit(f"setup check missed valid storage fallback disable metadata: {disabled_metadata}")
            if disabled_metadata.get("storage_fallback_disable_source") != "env":
                raise SystemExit(f"setup check missed env storage fallback disable source: {disabled_metadata}")
            if disabled_metadata.get("storage_fallback_valid") is not True:
                raise SystemExit(f"disabled storage fallback should make target validity pass: {disabled_metadata}")
            if disabled_metadata.get("storage_fallback_source") != "disabled":
                raise SystemExit(f"setup check missed disabled storage fallback source: {disabled_metadata}")
            with (
                patch(
                    "jarvis_v2.tools.doctor.status_host_config_from_env",
                    return_value=StatusHostConfig("127.0.0.1", configured=True, valid=False, source="env-invalid"),
                ),
                patch(
                    "jarvis_v2.tools.doctor.status_port_config_from_env",
                    return_value=StatusPortConfig(8766, configured=True, valid=False, source="env-invalid"),
                ),
            ):
                disabled_doctor = runtime.handle("jarvis doctor")
            if not disabled_doctor.verified:
                raise SystemExit(f"jarvis doctor should remain verified with disabled storage fallback: {disabled_doctor.response}")
            if "storage fallback disable validation: ok; fallback disabled (value hidden)" not in disabled_doctor.response:
                raise SystemExit(f"jarvis doctor missed valid storage fallback disable diagnostic: {disabled_doctor.response}")
            disabled_doctor_metadata = disabled_doctor.tool_results[0].metadata
            assert_doctor_no_future_authority(disabled_doctor_metadata, "jarvis doctor disabled fallback")
            if disabled_doctor_metadata.get("storage_fallback_disabled") is not True:
                raise SystemExit(f"jarvis doctor missed disabled storage fallback metadata: {disabled_doctor_metadata}")
            if disabled_doctor_metadata.get("storage_fallback_disable_valid") is not True:
                raise SystemExit(f"jarvis doctor missed valid storage fallback disable metadata: {disabled_doctor_metadata}")
            if disabled_doctor_metadata.get("storage_fallback_disable_source") != "env":
                raise SystemExit(f"jarvis doctor missed env storage fallback disable source: {disabled_doctor_metadata}")
    finally:
        if old_env_file is None:
            os.environ.pop("JARVIS_V3_ENV", None)
        else:
            os.environ["JARVIS_V3_ENV"] = old_env_file
        if old_data_dir is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = old_data_dir
        if old_db_path is None:
            os.environ.pop("JARVIS_DB_PATH", None)
        else:
            os.environ["JARVIS_DB_PATH"] = old_db_path
        if old_obsidian_vault is None:
            os.environ.pop("JARVIS_OBSIDIAN_VAULT", None)
        else:
            os.environ["JARVIS_OBSIDIAN_VAULT"] = old_obsidian_vault
        if old_secondary_obsidian_vault is None:
            os.environ.pop("OBSIDIAN_VAULT_PATH", None)
        else:
            os.environ["OBSIDIAN_VAULT_PATH"] = old_secondary_obsidian_vault
        if old_obsidian_root is None:
            os.environ.pop("JARVIS_OBSIDIAN_ROOT", None)
        else:
            os.environ["JARVIS_OBSIDIAN_ROOT"] = old_obsidian_root
        if old_ollama_model is None:
            os.environ.pop("OLLAMA_MODEL", None)
        else:
            os.environ["OLLAMA_MODEL"] = old_ollama_model
        if old_chat_model is None:
            os.environ.pop("JARVIS_CHAT_MODEL", None)
        else:
            os.environ["JARVIS_CHAT_MODEL"] = old_chat_model
        if old_planner_model is None:
            os.environ.pop("JARVIS_PLANNER_MODEL", None)
        else:
            os.environ["JARVIS_PLANNER_MODEL"] = old_planner_model
        if old_host is None:
            os.environ.pop("JARVIS_STATUS_HOST", None)
        else:
            os.environ["JARVIS_STATUS_HOST"] = old_host
        if old_port is None:
            os.environ.pop("JARVIS_STATUS_PORT", None)
        else:
            os.environ["JARVIS_STATUS_PORT"] = old_port
        if old_model_timeout is None:
            os.environ.pop("JARVIS_MODEL_TIMEOUT_SECONDS", None)
        else:
            os.environ["JARVIS_MODEL_TIMEOUT_SECONDS"] = old_model_timeout
        if old_chat_timeout is None:
            os.environ.pop("JARVIS_CHAT_TIMEOUT_SECONDS", None)
        else:
            os.environ["JARVIS_CHAT_TIMEOUT_SECONDS"] = old_chat_timeout
        if old_chat_max_reply_tokens is None:
            os.environ.pop("JARVIS_CHAT_MAX_REPLY_TOKENS", None)
        else:
            os.environ["JARVIS_CHAT_MAX_REPLY_TOKENS"] = old_chat_max_reply_tokens
        if old_chat_max_history_messages is None:
            os.environ.pop("JARVIS_CHAT_MAX_HISTORY_MESSAGES", None)
        else:
            os.environ["JARVIS_CHAT_MAX_HISTORY_MESSAGES"] = old_chat_max_history_messages
        if old_model_planner_toggle is None:
            os.environ.pop("JARVIS_USE_MODEL_PLANNER", None)
        else:
            os.environ["JARVIS_USE_MODEL_PLANNER"] = old_model_planner_toggle
        if old_smoke_timeout is None:
            os.environ.pop("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", None)
        else:
            os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = old_smoke_timeout
        if old_weather_location is None:
            os.environ.pop("JARVIS_WEATHER_LOCATION", None)
        else:
            os.environ["JARVIS_WEATHER_LOCATION"] = old_weather_location
        if old_news_locale is None:
            os.environ.pop("JARVIS_NEWS_LOCALE", None)
        else:
            os.environ["JARVIS_NEWS_LOCALE"] = old_news_locale
        if old_telegram_bot_token is None:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        else:
            os.environ["TELEGRAM_BOT_TOKEN"] = old_telegram_bot_token
        if old_gmail_address is None:
            os.environ.pop("GMAIL_ADDRESS", None)
        else:
            os.environ["GMAIL_ADDRESS"] = old_gmail_address
        if old_gmail_app_password is None:
            os.environ.pop("GMAIL_APP_PASSWORD", None)
        else:
            os.environ["GMAIL_APP_PASSWORD"] = old_gmail_app_password
        if old_owner_telegram is None:
            os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
        else:
            os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner_telegram
        if old_owner_imessage is None:
            os.environ.pop("JARVIS_OWNER_IMESSAGE", None)
        else:
            os.environ["JARVIS_OWNER_IMESSAGE"] = old_owner_imessage
        if old_google_creds is None:
            os.environ.pop("JARVIS_GOOGLE_CREDS", None)
        else:
            os.environ["JARVIS_GOOGLE_CREDS"] = old_google_creds
        if old_google_readonly_token is None:
            os.environ.pop("JARVIS_GOOGLE_READONLY_TOKEN", None)
        else:
            os.environ["JARVIS_GOOGLE_READONLY_TOKEN"] = old_google_readonly_token
        if old_google_token is None:
            os.environ.pop("JARVIS_GOOGLE_TOKEN", None)
        else:
            os.environ["JARVIS_GOOGLE_TOKEN"] = old_google_token
        if old_watched_dirs is None:
            os.environ.pop("JARVIS_WATCHED_DIRS", None)
        else:
            os.environ["JARVIS_WATCHED_DIRS"] = old_watched_dirs
        if old_telegram_state is None:
            os.environ.pop("JARVIS_TELEGRAM_STATE", None)
        else:
            os.environ["JARVIS_TELEGRAM_STATE"] = old_telegram_state
        if old_imessage_state is None:
            os.environ.pop("JARVIS_V3_IMESSAGE_STATE", None)
        else:
            os.environ["JARVIS_V3_IMESSAGE_STATE"] = old_imessage_state
        if old_reminders_file is None:
            os.environ.pop("JARVIS_REMINDERS_FILE", None)
        else:
            os.environ["JARVIS_REMINDERS_FILE"] = old_reminders_file
        if old_storage_fallback is None:
            os.environ.pop("JARVIS_STORAGE_FALLBACK_DIR", None)
        else:
            os.environ["JARVIS_STORAGE_FALLBACK_DIR"] = old_storage_fallback
        if old_storage_fallback_disabled is None:
            os.environ.pop("JARVIS_DISABLE_STORAGE_FALLBACK", None)
        else:
            os.environ["JARVIS_DISABLE_STORAGE_FALLBACK"] = old_storage_fallback_disabled
        if old_oav_vision_reviewer is None:
            os.environ.pop("JARVIS_OAV_VISION_REVIEWER_COMMAND", None)
        else:
            os.environ["JARVIS_OAV_VISION_REVIEWER_COMMAND"] = old_oav_vision_reviewer
        if old_voice_whisper_model is None:
            os.environ.pop("JARVIS_VOICE_WHISPER_MODEL_PATH", None)
        else:
            os.environ["JARVIS_VOICE_WHISPER_MODEL_PATH"] = old_voice_whisper_model
        if old_voice_faster_whisper_model is None:
            os.environ.pop("JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH", None)
        else:
            os.environ["JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH"] = old_voice_faster_whisper_model
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home

    print("setup status env smoke passed")


if __name__ == "__main__":
    main()
