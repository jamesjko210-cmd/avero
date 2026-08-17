from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import DEFAULT_MODEL_ALIAS, DEFAULT_OBSIDIAN_ROOT, JarvisConfig, _model_alias_from_env, _obsidian_root_from_env
from jarvis_v2.scripts import run_status_server
from jarvis_v2.scripts.daemon_gate import V3_DAEMON_ENABLE_ENV
from jarvis_v2.scripts.test_v3_launcher_env import empty_v3_launcher_env, scrubbed_v3_launcher_env
from jarvis_v2.ui.status_config import (
    DEFAULT_STATUS_HOST,
    DEFAULT_STATUS_PORT,
    STATUS_AUTH_ENV,
    status_auth_config_from_env,
    status_base_url,
    status_host_config_from_env,
    status_host_from_env,
    status_port_config_from_env,
    status_port_from_env,
)


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "launch_jarvis_v3_dashboard.py"


def main() -> None:
    launcher_env_temp = TemporaryDirectory(prefix="jarvis-status-config-launcher-env-")
    old_host = os.environ.get("JARVIS_STATUS_HOST")
    old_port = os.environ.get("JARVIS_STATUS_PORT")
    old_auth = os.environ.get(STATUS_AUTH_ENV)
    old_obsidian_root = os.environ.get("JARVIS_OBSIDIAN_ROOT")
    old_chat_model = os.environ.get("JARVIS_CHAT_MODEL")
    try:
        for raw_root in (
            "/\x55sers/example/Desktop/Claude code/Jarvis",
            "/private/tmp/jarvis-root",
            "/var/folders/zc/jarvis-root",
            "/tmp/jarvis-root",
        ):
            os.environ["JARVIS_OBSIDIAN_ROOT"] = raw_root
            if _obsidian_root_from_env() != DEFAULT_OBSIDIAN_ROOT:
                raise SystemExit(f"config should reject path-shaped Obsidian roots: {raw_root!r}")
        os.environ["JARVIS_OBSIDIAN_ROOT"] = "JarvisNotes"
        if _obsidian_root_from_env() != "JarvisNotes":
            raise SystemExit("config should keep safe Obsidian root names.")

        for raw_model in (
            "/\x55sers/example/private/model",
            "/private/tmp/jarvis-model",
            "/var/folders/zc/jarvis-model",
            "/tmp/jarvis-model",
        ):
            os.environ["JARVIS_CHAT_MODEL"] = raw_model
            if _model_alias_from_env("JARVIS_CHAT_MODEL", DEFAULT_MODEL_ALIAS) != DEFAULT_MODEL_ALIAS:
                raise SystemExit(f"config should reject path-shaped model aliases: {raw_model!r}")
        os.environ["JARVIS_CHAT_MODEL"] = "llama3.1:8b"
        if _model_alias_from_env("JARVIS_CHAT_MODEL", DEFAULT_MODEL_ALIAS) != "llama3.1:8b":
            raise SystemExit("config should keep safe model aliases.")

        os.environ["JARVIS_STATUS_HOST"] = "127.0.0.2"
        os.environ["JARVIS_STATUS_PORT"] = "9876"
        # Keep these environment-contract cases isolated from a real project
        # .env, which may legitimately contain the production dashboard token.
        with mock.patch("jarvis_v2.env.load_env"):
            os.environ.pop(STATUS_AUTH_ENV, None)
            missing_auth = status_auth_config_from_env()
            if missing_auth.configured or missing_auth.valid or missing_auth.source != "env-missing":
                raise SystemExit(f"status auth should fail closed when missing: {missing_auth!r}")
            os.environ[STATUS_AUTH_ENV] = "too-short"
            invalid_auth = status_auth_config_from_env()
            if not invalid_auth.configured or invalid_auth.valid or invalid_auth.source != "env-invalid":
                raise SystemExit(f"status auth should reject short secrets: {invalid_auth!r}")
            os.environ[STATUS_AUTH_ENV] = "status-config-smoke-secret-32-characters"
            valid_auth = status_auth_config_from_env()
        if not valid_auth.configured or not valid_auth.valid or valid_auth.source != "env":
            raise SystemExit(f"status auth should accept a strong local secret: {valid_auth!r}")
        if valid_auth.token != os.environ[STATUS_AUTH_ENV]:
            raise SystemExit("status auth config lost the validated secret.")
        if valid_auth.token in repr(valid_auth):
            raise SystemExit("status auth config repr leaked the secret.")
        if status_host_from_env() != "127.0.0.2" or status_port_from_env() != 9876:
            raise SystemExit("status config did not honor env host/port defaults")
        if status_base_url() != "http://127.0.0.2:9876":
            raise SystemExit("status config did not build the env-backed URL")
        if status_base_url("::1", 9876) != "http://[::1]:9876":
            raise SystemExit("status config should bracket IPv6 hosts in URLs")

        for unsafe_host in ("bad host", "0.0.0.0", "192.168.1.5", "jarvis.local"):
            os.environ["JARVIS_STATUS_HOST"] = unsafe_host
            invalid_host = status_host_config_from_env()
            if invalid_host.host != DEFAULT_STATUS_HOST or invalid_host.valid is not False or invalid_host.source != "env-invalid":
                raise SystemExit(f"status config should reject non-loopback host env: {unsafe_host!r}")
        os.environ["JARVIS_STATUS_HOST"] = "127.0.0.2"

        os.environ["JARVIS_STATUS_PORT"] = "not-a-port"
        invalid_port = status_port_config_from_env()
        if invalid_port.port != DEFAULT_STATUS_PORT or invalid_port.valid is not False or invalid_port.source != "env-invalid":
            raise SystemExit("status config should fall back on invalid port env")
        os.environ["JARVIS_STATUS_PORT"] = "70000"
        out_of_range_port = status_port_config_from_env()
        if out_of_range_port.port != DEFAULT_STATUS_PORT or out_of_range_port.valid is not False or out_of_range_port.source != "env-invalid":
            raise SystemExit("status config should fall back on out-of-range port env")

        with TemporaryDirectory(prefix="jarvis-status-config-") as temp:
            root = Path(temp)
            runtime = JarvisRuntime(
                JarvisConfig(
                    data_dir=root,
                    db_path=root / "jarvis.sqlite",
                    obsidian_vault=root / "Vault",
                    obsidian_root="Jarvis",
                    use_model_planner=False,
                )
            )
            os.environ["JARVIS_STATUS_HOST"] = "localhost"
            os.environ["JARVIS_STATUS_PORT"] = "9877"
            result = runtime.registry.get("status_dashboard").handler({})
            if "http://localhost:9877" not in result.output or "http://localhost:9877/api/status" not in result.output:
                raise SystemExit(f"status dashboard tool missed env-backed URL: {result.output}")
            if "JARVIS_STATUS_HOST" not in result.output or "JARVIS_STATUS_PORT" not in result.output:
                raise SystemExit(f"status dashboard tool missed env override guidance: {result.output}")
            if result.metadata.get("calls_model") is not False or result.metadata.get("executes_tools") is not False:
                raise SystemExit("status dashboard guidance should remain read-only and model-free")
            os.environ["JARVIS_STATUS_HOST"] = "::1"
            os.environ["JARVIS_STATUS_PORT"] = "9880"
            ipv6_result = runtime.registry.get("status_dashboard").handler({})
            if "http://[::1]:9880" not in ipv6_result.output or "http://[::1]:9880/api/status" not in ipv6_result.output:
                raise SystemExit(f"status dashboard tool missed bracketed IPv6 URL: {ipv6_result.output}")

        env = os.environ.copy()
        env.update(
            {
                V3_DAEMON_ENABLE_ENV: "1",
                "JARVIS_STATUS_HOST": "localhost",
                "JARVIS_STATUS_PORT": "9877",
                STATUS_AUTH_ENV: "status-config-smoke-secret-32-characters",
            }
        )
        launcher_env = empty_v3_launcher_env(
            Path(launcher_env_temp.name) / "runtime.env",
            env,
        )
        launcher_env[V3_DAEMON_ENABLE_ENV] = "1"
        help_run = subprocess.run(
            [sys.executable, str(LAUNCHER), "--help"],
            text=True,
            capture_output=True,
            timeout=10,
            cwd=str(ROOT),
            env=launcher_env,
        )
        print(help_run.stdout)
        if help_run.stderr:
            print("stderr:")
            print(help_run.stderr)
        if help_run.returncode != 0:
            raise SystemExit(f"status launcher --help exited with {help_run.returncode}")
        if "JARVIS_STATUS_HOST" not in help_run.stdout or "JARVIS_STATUS_PORT" not in help_run.stdout:
            raise SystemExit(f"status launcher --help missed env default guidance: {help_run.stdout}")

        cli_override = subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    [
                        "import sys",
                        "from unittest.mock import patch",
                        "from jarvis_v2.scripts.run_status_server import main",
                        "sys.argv=['run_status_server','--host','127.0.0.2','--port','9878']",
                        "with patch('jarvis_v2.scripts.run_status_server.JarvisRuntime') as runtime, patch('jarvis_v2.scripts.run_status_server.make_status_server') as make_server:",
                        "    make_server.return_value.serve_forever.side_effect=KeyboardInterrupt",
                        "    main()",
                        "    args=make_server.call_args.args",
                        "    kwargs=make_server.call_args.kwargs",
                        "    print(f'{args[1]}:{args[2]}')",
                        "    print('auth-bound=' + str(bool(kwargs.get('auth_token'))))",
                    ]
                ),
            ],
            text=True,
            capture_output=True,
            timeout=10,
            cwd=str(ROOT),
            env=env,
        )
        print(cli_override.stdout)
        if cli_override.stderr:
            print("stderr:")
            print(cli_override.stderr)
        if cli_override.returncode != 0:
            raise SystemExit(f"status launcher CLI override check exited with {cli_override.returncode}")
        if "127.0.0.2:9878" not in cli_override.stdout:
            raise SystemExit(f"status launcher did not let CLI flags override env defaults: {cli_override.stdout}")
        if "auth-bound=True" not in cli_override.stdout:
            raise SystemExit("status launcher did not bind validated authentication to the server.")

        ipv6_cli = subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    [
                        "import sys",
                        "from unittest.mock import patch",
                        "from jarvis_v2.scripts.run_status_server import main",
                        "sys.argv=['run_status_server','--host','::1','--port','9880']",
                        "with patch('jarvis_v2.scripts.run_status_server.JarvisRuntime'), patch('jarvis_v2.scripts.run_status_server.make_status_server') as make_server:",
                        "    make_server.return_value.serve_forever.side_effect=KeyboardInterrupt",
                        "    main()",
                    ]
                ),
            ],
            text=True,
            capture_output=True,
            timeout=10,
            cwd=str(ROOT),
            env=env,
        )
        print(ipv6_cli.stdout)
        if ipv6_cli.stderr:
            print("stderr:")
            print(ipv6_cli.stderr)
        if ipv6_cli.returncode != 0:
            raise SystemExit(f"status launcher IPv6 CLI check exited with {ipv6_cli.returncode}")
        if "Jarvis status dashboard: http://[::1]:9880" not in ipv6_cli.stdout:
            raise SystemExit(f"status launcher missed bracketed IPv6 URL: {ipv6_cli.stdout}")

        bind_failure = subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    [
                        "import sys",
                        "from unittest.mock import patch",
                        "from jarvis_v2.scripts.run_status_server import main",
                        "sys.argv=['run_status_server','--host','127.0.0.1','--port','9879']",
                        "with patch('jarvis_v2.scripts.run_status_server.JarvisRuntime'), patch('jarvis_v2.scripts.run_status_server.make_status_server', side_effect=OSError('address already in use')):",
                        "    main()",
                    ]
                ),
            ],
            text=True,
            capture_output=True,
            timeout=10,
            cwd=str(ROOT),
            env=env,
        )
        print(bind_failure.stdout)
        if bind_failure.stderr:
            print("stderr:")
            print(bind_failure.stderr)
        if bind_failure.returncode != 4:
            raise SystemExit(f"status launcher bind failure should exit 4, got {bind_failure.returncode}")
        if "could not bind http://127.0.0.1:9879" not in bind_failure.stdout:
            raise SystemExit(f"status launcher bind failure missed friendly URL: {bind_failure.stdout}")
        if "Diagnostic: OSError" not in bind_failure.stdout:
            raise SystemExit(f"status launcher bind failure missed bounded diagnostic: {bind_failure.stdout}")
        if "Choose another local port with --port or JARVIS_STATUS_PORT." not in bind_failure.stdout:
            raise SystemExit(f"status launcher bind failure missed recovery hint: {bind_failure.stdout}")
        if "address already in use" in bind_failure.stdout:
            raise SystemExit(f"status launcher bind failure leaked raw bind exception: {bind_failure.stdout}")
        if "Traceback" in bind_failure.stderr:
            raise SystemExit(f"status launcher bind failure should not print a traceback: {bind_failure.stderr}")

        non_tty_run = subprocess.run(
            [
                sys.executable,
                str(LAUNCHER),
                "--host",
                "0.0.0.0",
                "--port",
                "9879",
            ],
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=10,
            cwd=str(ROOT),
            env=launcher_env,
        )
        if non_tty_run.returncode != 4:
            raise SystemExit(
                "root dashboard launcher should refuse redirected-input starts before config: "
                f"{non_tty_run.returncode}"
            )
        if "standard input is not an attached Terminal" not in non_tty_run.stdout:
            raise SystemExit("root dashboard launcher non-TTY refusal missed bounded guidance.")
        if "refused a non-loopback bind host" in non_tty_run.stdout or STATUS_AUTH_ENV in non_tty_run.stdout:
            raise SystemExit("root dashboard launcher crossed its foreground gate before config.")

        attestation_calls: list[Path] = []

        def record_attestation(wrapper_path, *, caller_frame) -> None:
            del caller_frame
            attestation_calls.append(Path(wrapper_path))

        non_loopback_output = io.StringIO()
        with (
            mock.patch.object(
                run_status_server,
                "_require_manual_foreground_attestation",
                side_effect=record_attestation,
            ),
            mock.patch.object(run_status_server, "JarvisRuntime", side_effect=AssertionError("runtime constructed")),
            mock.patch.object(
                sys,
                "argv",
                [str(LAUNCHER), "--host", "0.0.0.0", "--port", "9879"],
            ),
            contextlib.redirect_stdout(non_loopback_output),
        ):
            try:
                run_status_server.manual_foreground_main(wrapper_path=LAUNCHER)
            except SystemExit as exc:
                if exc.code != 6:
                    raise SystemExit(
                        f"attested dashboard should refuse non-loopback binds with 6, got {exc.code}"
                    ) from exc
            else:
                raise SystemExit("attested dashboard accepted a non-loopback HTTP Basic bind")
        if attestation_calls != [LAUNCHER]:
            raise SystemExit(f"non-loopback test did not cross exactly one attestation: {attestation_calls}")
        if "refused a non-loopback bind host" not in non_loopback_output.getvalue():
            raise SystemExit("attested dashboard non-loopback refusal missed bounded guidance.")

        with TemporaryDirectory(prefix="jarvis-status-missing-auth-") as temp:
            missing_auth_file = Path(temp) / "dashboard.env"
            missing_auth_file.write_text("JARVIS_STATUS_PORT=9879\n", encoding="utf-8")
            missing_auth_file.chmod(0o600)
            missing_auth_env = scrubbed_v3_launcher_env(env, selected_env=missing_auth_file)
            missing_auth_env[V3_DAEMON_ENABLE_ENV] = "1"
            missing_auth_run = subprocess.run(
                [sys.executable, "-m", "jarvis_v2.scripts.run_status_server", "--port", "9879"],
                text=True,
                capture_output=True,
                timeout=10,
                cwd=str(ROOT),
                env=missing_auth_env,
            )
        if missing_auth_run.returncode != 5:
            raise SystemExit(
                f"status launcher should fail closed before startup when auth is missing: "
                f"{missing_auth_run.returncode}"
            )
        if STATUS_AUTH_ENV not in missing_auth_run.stdout or "Traceback" in missing_auth_run.stderr:
            raise SystemExit("status launcher missing-auth guidance was not bounded and actionable.")

        with TemporaryDirectory(prefix="jarvis-status-insecure-env-") as temp:
            insecure_env_file = Path(temp) / "dashboard.env"
            insecure_env_file.write_text(
                f"{STATUS_AUTH_ENV}=status-config-file-secret-32-characters\n",
                encoding="utf-8",
            )
            insecure_env_file.chmod(0o644)
            insecure_env = scrubbed_v3_launcher_env(env, selected_env=insecure_env_file)
            insecure_env[V3_DAEMON_ENABLE_ENV] = "1"
            insecure_env_run = subprocess.run(
                [sys.executable, "-m", "jarvis_v2.scripts.run_status_server", "--port", "9879"],
                text=True,
                capture_output=True,
                timeout=10,
                cwd=str(ROOT),
                env=insecure_env,
            )
        if insecure_env_run.returncode != 7:
            raise SystemExit(
                f"daemon-gated status module should fail before runtime on insecure auth-file custody: "
                f"{insecure_env_run.returncode}"
            )
        combined_insecure_output = insecure_env_run.stdout + insecure_env_run.stderr
        if "chmod 600 .env" not in combined_insecure_output:
            raise SystemExit("status module insecure-env refusal missed its bounded fix.")
        for forbidden in ("status-config-file-secret", str(insecure_env_file), "Traceback"):
            if forbidden in combined_insecure_output:
                raise SystemExit("status launcher insecure-env refusal leaked secret/path/traceback material.")
    finally:
        if old_host is None:
            os.environ.pop("JARVIS_STATUS_HOST", None)
        else:
            os.environ["JARVIS_STATUS_HOST"] = old_host
        if old_port is None:
            os.environ.pop("JARVIS_STATUS_PORT", None)
        else:
            os.environ["JARVIS_STATUS_PORT"] = old_port
        if old_auth is None:
            os.environ.pop(STATUS_AUTH_ENV, None)
        else:
            os.environ[STATUS_AUTH_ENV] = old_auth
        if old_obsidian_root is None:
            os.environ.pop("JARVIS_OBSIDIAN_ROOT", None)
        else:
            os.environ["JARVIS_OBSIDIAN_ROOT"] = old_obsidian_root
        if old_chat_model is None:
            os.environ.pop("JARVIS_CHAT_MODEL", None)
        else:
            os.environ["JARVIS_CHAT_MODEL"] = old_chat_model
        launcher_env_temp.cleanup()

    print("status config smoke passed")


if __name__ == "__main__":
    main()
