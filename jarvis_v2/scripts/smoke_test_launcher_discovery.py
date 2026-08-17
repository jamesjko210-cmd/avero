from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import runpy
import shutil
import subprocess
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.agent.model_provider import ollama_local_only_policy
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.scripts.test_v3_launcher_env import empty_v3_launcher_env
from jarvis_v2.v3_commands import (
    V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND,
    V3_CALENDAR_AUTH_READONLY_COMMAND,
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DASHBOARD_MODULE_COMMAND,
)


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "launch_jarvis_v3_dashboard.py"
CALENDAR_AUTH_LAUNCHER = ROOT / "launch_jarvis_v3_calendar_auth.py"
README = ROOT / "README.md"
QUICKSTART = ROOT / "QUICKSTART.md"
ENV_EXAMPLE = ROOT / ".env.example"
IMESSAGE_LAUNCHER = ROOT / "jarvis_v2" / "scripts" / "run_imessage_control.py"
OLLAMA_LOOPBACK_HOST = "http://127.0.0.1:11434"
OLLAMA_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@contextlib.contextmanager
def _stateless_local_ollama_env():
    keys = (
        "OLLAMA_HOST",
        "OLLAMA_NO_CLOUD",
        "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT",
        *OLLAMA_PROXY_ENV_KEYS,
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["OLLAMA_HOST"] = OLLAMA_LOOPBACK_HOST
    os.environ.pop("OLLAMA_NO_CLOUD", None)
    os.environ.pop("JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT", None)
    for key in OLLAMA_PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert_warmup_failure_is_bounded(module_name: str) -> None:
    class Config:
        chat_model = "jarvis-v2-smoke-model"
        # _warm_model() checks the provider before touching Ollama; this stub
        # models the real Config's default so the daemon path still runs.
        model_provider = "ollama"

    def failing_generate(**_kwargs):
        raise RuntimeError("SECRET raw warmup failure")

    module = __import__(module_name, fromlist=["_warm_model"])
    original_generate = module.generate_model_text
    module.generate_model_text = failing_generate
    try:
        out = io.StringIO()
        with _stateless_local_ollama_env(), contextlib.redirect_stdout(out):
            module._warm_model(Config())
    finally:
        module.generate_model_text = original_generate
    output = out.getvalue()
    if "Ollama loopback daemon warmup did not complete. Diagnostic: RuntimeError." not in output:
        raise SystemExit(f"{module_name} warmup failure missed bounded diagnostic: {output}")
    if "SECRET raw warmup failure" in output:
        raise SystemExit(f"{module_name} warmup failure leaked raw exception text: {output}")


def _assert_warmup_success_is_daemon_bounded(module_name: str) -> None:
    calls: list[dict[str, object]] = []

    class Config:
        chat_model = "jarvis-v2-smoke-model"
        model_provider = "ollama"

    def successful_generate(**kwargs):
        calls.append(kwargs)
        return "ready"

    module = __import__(module_name, fromlist=["_warm_model"])
    original_generate = module.generate_model_text
    module.generate_model_text = successful_generate
    try:
        out = io.StringIO()
        with _stateless_local_ollama_env(), contextlib.redirect_stdout(out):
            module._warm_model(Config())
    finally:
        module.generate_model_text = original_generate
    output = out.getvalue()
    if "Ollama loopback daemon responded to the warmup request" not in output:
        raise SystemExit(f"{module_name} warmup success missed daemon receipt: {output}")
    if "does not verify where model execution occurred" not in output:
        raise SystemExit(f"{module_name} warmup success missed execution caveat: {output}")
    for forbidden in ("warmed and resident", "local execution", "executed locally"):
        if forbidden in output.lower():
            raise SystemExit(f"{module_name} warmup success overclaimed {forbidden!r}: {output}")
    if len(calls) != 1:
        raise SystemExit(f"{module_name} warmup success did not reach the fake model adapter exactly once")
    call = calls[0]
    if (
        call.get("provider") != "ollama"
        or call.get("model") != Config.chat_model
        or call.get("messages") != [{"role": "user", "content": "ready?"}]
        or call.get("timeout_seconds") != 60
        or call.get("max_output_tokens") != 1
        or call.get("temperature") != 0
        or call.get("keep_alive") != "30m"
    ):
        raise SystemExit(f"{module_name} warmup success used the wrong bounded model-adapter request")


def _assert_invalid_provider_skips_warmup(module_name: str) -> None:
    class Config:
        chat_model = "jarvis-v2-smoke-model"
        model_provider = "invalid"

    module = __import__(module_name, fromlist=["_warm_model"])
    original_generate = module.generate_model_text

    def fail_if_called(**_kwargs):
        raise AssertionError("invalid provider attempted a warmup request")

    module.generate_model_text = fail_if_called
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            module._warm_model(Config())
    finally:
        module.generate_model_text = original_generate
    output = out.getvalue()
    if "Ollama loopback daemon warmup skipped" not in output or "no request was sent" not in output:
        raise SystemExit(f"{module_name} invalid-provider skip was unclear: {output}")


def _assert_unverified_ollama_context_requires_both_flags() -> None:
    consent_env = "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"
    operator_intent_only = ollama_local_only_policy(
        "jarvis-v2-smoke-model",
        {"OLLAMA_NO_CLOUD": "1", consent_env: "0"},
    )
    consent_only = ollama_local_only_policy(
        "jarvis-v2-smoke-model",
        {"OLLAMA_NO_CLOUD": "0", consent_env: "1"},
    )
    both_flags = ollama_local_only_policy(
        "jarvis-v2-smoke-model",
        {"OLLAMA_NO_CLOUD": "1", consent_env: "1"},
    )
    if operator_intent_only.personal_context_allowed:
        raise SystemExit("OLLAMA_NO_CLOUD alone must not allow stored personal context")
    if consent_only.personal_context_allowed:
        raise SystemExit("unverified-context consent alone must not allow stored personal context")
    if not both_flags.personal_context_allowed:
        raise SystemExit("both Ollama policy flags should allow stored context after route/model checks")
    for label, policy in (
        ("operator intent only", operator_intent_only),
        ("consent only", consent_only),
        ("both flags", both_flags),
    ):
        if policy.receipt().get("ollama_execution_locality_verified") is not False:
            raise SystemExit(f"{label} incorrectly attested Ollama execution locality")


def _assert_documented_foreground_dashboard_lifecycle() -> None:
    """Exercise the root wrapper without granting background-daemon authority."""

    from jarvis_v2.scripts import run_status_server, v3_python_runtime

    events: list[object] = []

    class SyntheticRuntime:
        def __init__(self) -> None:
            events.append("runtime")

    class SyntheticServer:
        def serve_forever(self) -> None:
            events.append("serve")

        def server_close(self) -> None:
            events.append("close")

    def synthetic_server(runtime, host: str, port: int, *, auth_token: str):
        events.append(("server", runtime, host, port, auth_token))
        return SyntheticServer()

    def forbidden_daemon_gate(_service: str) -> None:
        raise AssertionError("manual foreground dashboard requested daemon authority")

    documented_command = (
        'JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" '
        "./launch_jarvis_v3_dashboard.py"
    )
    if documented_command not in QUICKSTART.read_text(encoding="utf-8"):
        raise SystemExit("QUICKSTART lost the canonical foreground dashboard command")

    output = io.StringIO()
    with (
        mock.patch.object(v3_python_runtime, "reexec_with_v3_python", return_value=None),
        mock.patch.object(run_status_server, "require_v3_daemon_enable", side_effect=forbidden_daemon_gate),
        mock.patch.object(run_status_server.os, "isatty", return_value=True),
        mock.patch.object(run_status_server.os, "tcgetpgrp", return_value=731),
        mock.patch.object(run_status_server.os, "getpgrp", return_value=731),
        mock.patch.object(run_status_server, "status_host_from_env", return_value="127.0.0.1"),
        mock.patch.object(run_status_server, "status_port_from_env", return_value=8766),
        mock.patch.object(
            run_status_server,
            "status_auth_config_from_env",
            return_value=types.SimpleNamespace(valid=True, token="synthetic-dashboard-auth"),
        ),
        mock.patch.object(run_status_server, "JarvisRuntime", SyntheticRuntime),
        mock.patch.object(run_status_server, "make_status_server", side_effect=synthetic_server),
        mock.patch.object(sys, "argv", [str(LAUNCHER)]),
        contextlib.redirect_stdout(output),
    ):
        runpy.run_path(str(LAUNCHER), run_name="__main__")

    if events[:2] != ["runtime", ("server", mock.ANY, "127.0.0.1", 8766, "synthetic-dashboard-auth")]:
        raise SystemExit(f"foreground dashboard construction drifted: {events}")
    if events[2:] != ["serve", "close"]:
        raise SystemExit(f"foreground dashboard did not serve and close cleanly: {events}")
    rendered = output.getvalue()
    if "Jarvis status dashboard: http://127.0.0.1:8766" not in rendered:
        raise SystemExit(f"foreground dashboard missed its loopback URL: {rendered}")
    if "Press Ctrl+C to stop." not in rendered:
        raise SystemExit(f"foreground dashboard missed bounded stop guidance: {rendered}")


def _assert_foreground_dashboard_refuses_unattested_callers() -> None:
    """Reject detached/imported paths before environment or runtime access."""

    from jarvis_v2.scripts import run_status_server, v3_python_runtime

    sentinel = "/private/tmp/jarvis-v3-selected-env-must-not-be-read"
    early_refusal = subprocess.run(
        [sys.executable, str(LAUNCHER)],
        cwd=str(ROOT),
        env={
            "HOME": "/private/tmp",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "JARVIS_V3_ENV": sentinel,
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rendered_early = early_refusal.stdout + early_refusal.stderr
    if (
        early_refusal.returncode != 4
        or "standard input is not an attached Terminal" not in rendered_early
        or sentinel in rendered_early
        or "selected environment" in rendered_early.casefold()
        or "Traceback" in rendered_early
    ):
        raise SystemExit(
            "dashboard wrapper did not refuse before selected-environment read/probe/re-exec"
        )

    def forbidden_state_access(*_args, **_kwargs):
        raise AssertionError("foreground refusal reached environment, auth, or runtime state")

    common_patches = (
        mock.patch.object(v3_python_runtime, "reexec_with_v3_python", return_value=None),
        mock.patch.object(run_status_server, "status_host_from_env", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "status_port_from_env", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "status_auth_config_from_env", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "JarvisRuntime", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "make_status_server", side_effect=forbidden_state_access),
        mock.patch.object(sys, "argv", [str(LAUNCHER)]),
    )

    output = io.StringIO()
    with contextlib.ExitStack() as stack:
        for patcher in common_patches:
            stack.enter_context(patcher)
        stack.enter_context(mock.patch.object(run_status_server.os, "isatty", return_value=False))
        stack.enter_context(
            mock.patch.object(
                run_status_server.os,
                "tcgetpgrp",
                side_effect=AssertionError("non-TTY queried process group"),
            )
        )
        stack.enter_context(contextlib.redirect_stdout(output))
        try:
            runpy.run_path(str(LAUNCHER), run_name="__main__")
        except SystemExit as exc:
            if exc.code != 4:
                raise SystemExit(f"non-TTY foreground refusal used unexpected exit {exc.code}") from exc
        else:
            raise SystemExit("non-TTY foreground dashboard launch was accepted")
    if "standard input is not an attached Terminal" not in output.getvalue():
        raise SystemExit(f"non-TTY foreground refusal was unclear: {output.getvalue()}")

    output = io.StringIO()
    with (
        mock.patch.object(v3_python_runtime, "reexec_with_v3_python", return_value=None),
        mock.patch.object(run_status_server, "status_host_from_env", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "status_port_from_env", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "status_auth_config_from_env", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "JarvisRuntime", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "make_status_server", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server.os, "isatty", return_value=True),
        mock.patch.object(run_status_server.os, "tcgetpgrp", return_value=731),
        mock.patch.object(run_status_server.os, "getpgrp", return_value=732),
        mock.patch.object(sys, "argv", [str(LAUNCHER)]),
        contextlib.redirect_stdout(output),
    ):
        try:
            runpy.run_path(str(LAUNCHER), run_name="__main__")
        except SystemExit as exc:
            if exc.code != 4:
                raise SystemExit(f"background foreground refusal used unexpected exit {exc.code}") from exc
        else:
            raise SystemExit("background-process dashboard launch was accepted")
    if "not the Terminal foreground process group" not in output.getvalue():
        raise SystemExit(f"background foreground refusal was unclear: {output.getvalue()}")

    for label, wrapper_path in (
        ("mismatched wrapper", ROOT / "copied_dashboard_launcher.py"),
        ("imported exact wrapper", LAUNCHER),
    ):
        output = io.StringIO()
        with (
            mock.patch.object(run_status_server, "status_host_from_env", side_effect=forbidden_state_access),
            mock.patch.object(run_status_server, "status_port_from_env", side_effect=forbidden_state_access),
            mock.patch.object(run_status_server, "status_auth_config_from_env", side_effect=forbidden_state_access),
            mock.patch.object(run_status_server, "JarvisRuntime", side_effect=forbidden_state_access),
            mock.patch.object(run_status_server, "make_status_server", side_effect=forbidden_state_access),
            mock.patch.object(run_status_server.os, "isatty", return_value=True),
            mock.patch.object(run_status_server.os, "tcgetpgrp", return_value=731),
            mock.patch.object(run_status_server.os, "getpgrp", return_value=731),
            mock.patch.object(sys, "argv", [str(LAUNCHER)]),
            contextlib.redirect_stdout(output),
        ):
            try:
                run_status_server.manual_foreground_main(wrapper_path=wrapper_path)
            except SystemExit as exc:
                if exc.code != 4:
                    raise SystemExit(f"{label} refusal used unexpected exit {exc.code}") from exc
            else:
                raise SystemExit(f"{label} dashboard call was accepted")
        if "root-wrapper provenance" not in output.getvalue():
            raise SystemExit(f"{label} refusal was unclear: {output.getvalue()}")


def _assert_direct_dashboard_module_remains_daemon_gated() -> None:
    from jarvis_v2.scripts import run_status_server

    events: list[object] = []

    def daemon_gate(service: str) -> None:
        events.append(("gate", service))
        raise SystemExit(4)

    def forbidden_state_access(*_args, **_kwargs):
        raise AssertionError("direct module reached state before the daemon gate")

    with (
        mock.patch.object(run_status_server, "require_v3_daemon_enable", side_effect=daemon_gate),
        mock.patch.object(run_status_server, "status_host_from_env", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "status_port_from_env", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "status_auth_config_from_env", side_effect=forbidden_state_access),
        mock.patch.object(run_status_server, "JarvisRuntime", side_effect=forbidden_state_access),
        mock.patch.object(sys, "argv", [str(run_status_server.__file__)]),
    ):
        try:
            run_status_server.main()
        except SystemExit as exc:
            if exc.code != 4:
                raise SystemExit(f"direct dashboard module gate used unexpected exit {exc.code}") from exc
        else:
            raise SystemExit("direct dashboard module bypassed the daemon gate")
    if events != [("gate", "dashboard")]:
        raise SystemExit(f"direct dashboard module gate drifted: {events}")


def _assert_root_wrappers_are_import_inert_and_copy_bound() -> None:
    """Imports do nothing, while copied main wrappers fail before env/auth."""

    from jarvis_v2 import env as env_module
    from jarvis_v2.scripts import (
        google_calendar_readonly_auth,
        google_calendar_reauth,
        run_status_server,
        v3_python_runtime,
    )

    wrappers = (
        ("dashboard", LAUNCHER, []),
        ("calendar", CALENDAR_AUTH_LAUNCHER, ["readonly"]),
    )
    for label, wrapper, arguments in wrappers:
        for mechanism in ("runpy", "module"):
            with (
                mock.patch.object(v3_python_runtime, "reexec_with_v3_python") as reexec,
                mock.patch.object(run_status_server, "manual_foreground_main") as dashboard,
                mock.patch.object(google_calendar_readonly_auth, "main") as readonly,
                mock.patch.object(google_calendar_reauth, "main") as full,
                mock.patch.object(env_module, "load_env") as load_selected,
                mock.patch.object(sys, "argv", [str(wrapper), *arguments]),
                mock.patch.object(os, "isatty", return_value=True),
                mock.patch.object(os, "tcgetpgrp", return_value=919),
                mock.patch.object(os, "getpgrp", return_value=919),
            ):
                if mechanism == "runpy":
                    runpy.run_path(str(wrapper))
                else:
                    spec = importlib.util.spec_from_file_location(
                        f"_jarvis_v3_{label}_wrapper_import_smoke",
                        wrapper,
                    )
                    if spec is None or spec.loader is None:
                        raise SystemExit(f"could not build {label} wrapper import spec")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
            if any(
                call.called
                for call in (reexec, dashboard, readonly, full, load_selected)
            ):
                raise SystemExit(f"{label} wrapper {mechanism} import performed launch work")

    imported_calendar = runpy.run_path(str(CALENDAR_AUTH_LAUNCHER))
    imported_output = io.StringIO()
    with (
        mock.patch.object(google_calendar_readonly_auth, "main") as readonly,
        mock.patch.object(google_calendar_reauth, "main") as full,
        mock.patch.object(env_module, "load_env") as load_selected,
        mock.patch.object(sys, "argv", [str(CALENDAR_AUTH_LAUNCHER), "readonly"]),
        contextlib.redirect_stderr(imported_output),
    ):
        try:
            imported_calendar["_authorized_main"](
                authority=imported_calendar["_DIRECT_EXECUTION_AUTHORITY"],
                runtime_module=v3_python_runtime,
            )
        except SystemExit as exc:
            if exc.code != 4:
                raise SystemExit("imported Calendar caller used an unexpected exit") from exc
        else:
            raise SystemExit("imported Calendar caller gained root-wrapper authority")
    if (
        "root-wrapper provenance" not in imported_output.getvalue()
        or readonly.called
        or full.called
        or load_selected.called
    ):
        raise SystemExit("imported Calendar caller reached environment or OAuth dispatch")

    sentinel = "/private/tmp/jarvis-v3-copied-wrapper-env-must-not-be-read"
    with TemporaryDirectory(prefix="jarvis-v3-copied-root-wrapper-") as temp:
        copy_root = Path(temp)
        for label, wrapper, arguments in wrappers:
            copied = copy_root / wrapper.name
            shutil.copy2(wrapper, copied)
            output = io.StringIO()
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "HOME": str(copy_root),
                        "PATH": os.environ.get("PATH", ""),
                        "PYTHONPATH": str(ROOT),
                        "JARVIS_V3_ENV": sentinel,
                    },
                    clear=True,
                ),
                mock.patch.object(v3_python_runtime, "reexec_with_v3_python") as reexec,
                mock.patch.object(run_status_server, "manual_foreground_main") as dashboard,
                mock.patch.object(google_calendar_readonly_auth, "main") as readonly,
                mock.patch.object(google_calendar_reauth, "main") as full,
                mock.patch.object(env_module, "load_env") as load_selected,
                mock.patch.object(sys, "argv", [str(copied), *arguments]),
                mock.patch.object(os, "isatty", return_value=True),
                mock.patch.object(os, "tcgetpgrp", return_value=920),
                mock.patch.object(os, "getpgrp", return_value=920),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(output),
            ):
                try:
                    runpy.run_path(str(copied), run_name="__main__")
                except SystemExit as exc:
                    if exc.code != 4:
                        raise SystemExit(f"copied {label} wrapper used exit {exc.code}") from exc
                else:
                    raise SystemExit(f"copied {label} root wrapper was accepted")
            rendered = output.getvalue()
            if (
                "root-wrapper provenance" not in rendered
                or sentinel in rendered
                or any(
                    call.called
                    for call in (reexec, dashboard, readonly, full, load_selected)
                )
            ):
                raise SystemExit(f"copied {label} wrapper reached environment or auth work")


def _assert_calendar_auth_loads_selected_custom_paths() -> None:
    """Selected owner-only auth paths reach connector helpers without OAuth."""

    from jarvis_v2.scripts import (
        google_calendar_readonly_auth,
        google_calendar_reauth,
        v3_python_runtime,
    )
    from jarvis_v2.tools import calendar_connector

    with TemporaryDirectory(prefix="jarvis-v3-calendar-selected-env-") as temp:
        root = Path(temp)
        selected_env = root / "runtime.env"
        expected = (
            root / "custom-credentials.json",
            root / "custom-full-token.json",
            root / "custom-readonly-token.json",
        )
        selected_env.write_text(
            f"JARVIS_GOOGLE_CREDS={expected[0]}\n"
            f"JARVIS_GOOGLE_TOKEN={expected[1]}\n"
            f"JARVIS_GOOGLE_READONLY_TOKEN={expected[2]}\n",
            encoding="utf-8",
        )
        selected_env.chmod(0o600)
        observed: list[tuple[Path, Path, Path]] = []

        def bounded_readonly_auth() -> int:
            observed.append(
                (
                    calendar_connector._creds_file(),
                    calendar_connector._token_file(),
                    calendar_connector._readonly_token_file(),
                )
            )
            return 0

        output = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "HOME": str(root),
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": str(ROOT),
                    "JARVIS_V3_ENV": str(selected_env),
                },
                clear=True,
            ),
            mock.patch.object(v3_python_runtime, "reexec_with_v3_python", return_value=None),
            mock.patch.object(os, "isatty", return_value=True),
            mock.patch.object(os, "tcgetpgrp", return_value=921),
            mock.patch.object(os, "getpgrp", return_value=921),
            mock.patch.object(
                google_calendar_readonly_auth,
                "main",
                side_effect=bounded_readonly_auth,
            ),
            mock.patch.object(google_calendar_reauth, "main") as full,
            mock.patch.object(
                sys,
                "argv",
                [str(CALENDAR_AUTH_LAUNCHER), "readonly"],
            ),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
        ):
            try:
                runpy.run_path(str(CALENDAR_AUTH_LAUNCHER), run_name="__main__")
            except SystemExit as exc:
                if exc.code != 0:
                    raise SystemExit("selected Calendar auth paths did not dispatch") from exc
            else:
                raise SystemExit("selected Calendar auth dispatch did not terminate")
        rendered = output.getvalue()
        if observed != [expected] or full.called:
            raise SystemExit("Calendar connector helpers ignored or crossed selected token paths")
        if any(str(value) in rendered for value in (selected_env, *expected)):
            raise SystemExit("Calendar selected-environment proof leaked a private path")

        missing_output = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "HOME": str(root),
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": str(ROOT),
                },
                clear=True,
            ),
            mock.patch.object(v3_python_runtime, "reexec_with_v3_python", return_value=None),
            mock.patch.object(os, "isatty", return_value=True),
            mock.patch.object(os, "tcgetpgrp", return_value=922),
            mock.patch.object(os, "getpgrp", return_value=922),
            mock.patch.object(google_calendar_readonly_auth, "main") as readonly,
            mock.patch.object(google_calendar_reauth, "main") as full,
            mock.patch.object(
                sys,
                "argv",
                [str(CALENDAR_AUTH_LAUNCHER), "readonly"],
            ),
            contextlib.redirect_stderr(missing_output),
        ):
            try:
                runpy.run_path(str(CALENDAR_AUTH_LAUNCHER), run_name="__main__")
            except SystemExit as exc:
                if exc.code != 78:
                    raise SystemExit("missing selected Calendar env used wrong exit") from exc
            else:
                raise SystemExit("Calendar auth accepted project-env fallback")
        if (
            readonly.called
            or full.called
            or "explicit owner-only JARVIS_V3_ENV" not in missing_output.getvalue()
        ):
            raise SystemExit("missing selected Calendar env reached OAuth or lost guidance")


def _assert_calendar_auth_root_launcher_dispatch() -> None:
    """Prove explicit, human-only dispatch without opening OAuth or reading private state."""

    from jarvis_v2.scripts import (
        google_calendar_readonly_auth,
        google_calendar_reauth,
        v3_python_runtime,
    )

    if not CALENDAR_AUTH_LAUNCHER.is_file():
        raise SystemExit("root Calendar authorization launcher is missing")
    source = CALENDAR_AUTH_LAUNCHER.read_text(encoding="utf-8")
    main_guard = source.find('if __name__ == "__main__":')
    precheck_call = source.find("    _require_human_foreground()\n", main_guard)
    runtime_import = source.find(
        "from jarvis_v2.scripts import v3_python_runtime",
        main_guard,
    )
    reexec_index = source.find("reexec_with_v3_python(Path(__file__))", main_guard)
    load_index = source.find("    _load_selected_v3_environment()", main_guard)
    dispatch_index = source.find(
        "authority=_DIRECT_EXECUTION_AUTHORITY",
        main_guard,
    )
    readonly_index = source.find(
        "from jarvis_v2.scripts.google_calendar_readonly_auth import main as auth_main"
    )
    full_index = source.find(
        "from jarvis_v2.scripts.google_calendar_reauth import main as auth_main"
    )
    if min(
        main_guard,
        precheck_call,
        runtime_import,
        reexec_index,
        load_index,
        dispatch_index,
        readonly_index,
        full_index,
    ) < 0 or not (
        main_guard < precheck_call < runtime_import < reexec_index < load_index < dispatch_index
    ):
        raise SystemExit(
            "Calendar auth launcher did not precheck human custody before V3 Python selection"
        )

    sentinel = "/private/tmp/jarvis-v3-calendar-selected-env-must-not-be-read"
    early_refusal = subprocess.run(
        [sys.executable, str(CALENDAR_AUTH_LAUNCHER), "readonly"],
        cwd=str(ROOT),
        env={
            "HOME": "/private/tmp",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "JARVIS_V3_ENV": sentinel,
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rendered_early = early_refusal.stdout + early_refusal.stderr
    if (
        early_refusal.returncode != 4
        or "foreground Terminal" not in rendered_early
        or sentinel in rendered_early
        or "selected environment" in rendered_early.casefold()
        or "Traceback" in rendered_early
    ):
        raise SystemExit(
            "Calendar auth launcher did not refuse before selected-environment read/probe/re-exec"
        )

    for selected in (None, sentinel):
        help_env = {
            "HOME": "/private/tmp",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if selected is not None:
            help_env["JARVIS_V3_ENV"] = selected
        help_run = subprocess.run(
            [sys.executable, str(CALENDAR_AUTH_LAUNCHER), "--help"],
            cwd=str(ROOT),
            env=help_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if (
            help_run.returncode != 0
            or "readonly" not in help_run.stdout
            or "full-access" not in help_run.stdout
            or "Traceback" in help_run.stderr
            or sentinel in help_run.stdout + help_run.stderr
        ):
            raise SystemExit("Calendar auth launcher help was unavailable or read environment")

    with TemporaryDirectory(prefix="jarvis-v3-calendar-auth-dispatch-") as temp:
        selected_env = Path(temp) / "runtime.env"
        selected_values = empty_v3_launcher_env(selected_env)
        for mode, expected_code, readonly_calls, full_calls in (
            ("readonly", 31, 1, 0),
            ("full-access", 32, 0, 1),
        ):
            with (
                mock.patch.dict(os.environ, selected_values, clear=True),
                mock.patch.object(v3_python_runtime, "reexec_with_v3_python", return_value=None),
                mock.patch.object(os, "isatty", return_value=True),
                mock.patch.object(os, "tcgetpgrp", return_value=841),
                mock.patch.object(os, "getpgrp", return_value=841),
                mock.patch.object(google_calendar_readonly_auth, "main", return_value=31) as readonly,
                mock.patch.object(google_calendar_reauth, "main", return_value=32) as full,
                mock.patch.object(sys, "argv", [str(CALENDAR_AUTH_LAUNCHER), mode]),
            ):
                try:
                    runpy.run_path(str(CALENDAR_AUTH_LAUNCHER), run_name="__main__")
                except SystemExit as exc:
                    if exc.code != expected_code:
                        raise SystemExit(f"{mode} Calendar dispatch returned {exc.code}") from exc
                else:
                    raise SystemExit(f"{mode} Calendar dispatch did not return through its auth flow")
            if readonly.call_count != readonly_calls or full.call_count != full_calls:
                raise SystemExit(f"{mode} Calendar dispatch crossed authorization modes")

    output = io.StringIO()
    with (
        mock.patch.object(v3_python_runtime, "reexec_with_v3_python", return_value=None),
        mock.patch.object(os, "isatty", return_value=False),
        mock.patch.object(
            os,
            "tcgetpgrp",
            side_effect=AssertionError("non-TTY Calendar auth queried process group"),
        ),
        mock.patch.object(google_calendar_readonly_auth, "main") as readonly,
        mock.patch.object(google_calendar_reauth, "main") as full,
        mock.patch.object(sys, "argv", [str(CALENDAR_AUTH_LAUNCHER), "readonly"]),
        contextlib.redirect_stderr(output),
    ):
        try:
            runpy.run_path(str(CALENDAR_AUTH_LAUNCHER), run_name="__main__")
        except SystemExit as exc:
            if exc.code != 4:
                raise SystemExit("non-TTY Calendar auth used an unexpected exit") from exc
        else:
            raise SystemExit("non-TTY Calendar authorization was accepted")
    if readonly.called or full.called or "foreground Terminal" not in output.getvalue():
        raise SystemExit("non-TTY Calendar auth reached OAuth dispatch or lost guidance")

    for command in (V3_CALENDAR_AUTH_READONLY_COMMAND, V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND):
        if not command.startswith('JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./'):
            raise SystemExit(f"Calendar auth command lost selected-environment custody: {command}")


def main() -> None:
    _assert_unverified_ollama_context_requires_both_flags()
    _assert_root_wrappers_are_import_inert_and_copy_bound()
    _assert_documented_foreground_dashboard_lifecycle()
    _assert_foreground_dashboard_refuses_unattested_callers()
    _assert_direct_dashboard_module_remains_daemon_gated()
    _assert_calendar_auth_root_launcher_dispatch()
    _assert_calendar_auth_loads_selected_custom_paths()
    if not LAUNCHER.exists():
        raise SystemExit("root dashboard launcher is missing")
    text = LAUNCHER.read_text()
    for expected in (
        "jarvis_v2.scripts.run_status_server",
        "manual_foreground_main(wrapper_path=Path(__file__))",
    ):
        if expected not in text:
            raise SystemExit(f"launcher missing expected wiring: {expected}")
    precheck_call = text.find("    _precheck_manual_foreground()\n")
    runtime_import = text.find(
        "from jarvis_v2.scripts import v3_python_runtime"
    )
    if precheck_call < 0 or runtime_import < 0 or precheck_call >= runtime_import:
        raise SystemExit("dashboard wrapper did not precheck foreground custody before runtime import")

    help_sentinel = "/private/tmp/jarvis-v3-dashboard-help-env-must-not-be-read"
    for selected in (None, help_sentinel):
        help_env = {
            "HOME": "/private/tmp",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if selected is not None:
            help_env["JARVIS_V3_ENV"] = selected
        help_run = subprocess.run(
            [sys.executable, str(LAUNCHER), "--help"],
            text=True,
            capture_output=True,
            timeout=10,
            cwd=str(ROOT),
            env=help_env,
        )
        if (
            help_run.returncode != 0
            or "Run the Jarvis V3 read-only status dashboard" not in help_run.stdout
            or "--port" not in help_run.stdout
            or "--host" not in help_run.stdout
            or "Traceback" in help_run.stderr
            or help_sentinel in help_run.stdout + help_run.stderr
        ):
            raise SystemExit("dashboard help was unavailable or read selected environment")
    print(help_run.stdout)

    readme = README.read_text()
    for expected in (
        "This is the Jarvis V4 release source",
        "not `jarvis-ollama`",
        "`QUICKSTART.md` is the authoritative fresh-start and daily-use path",
        "root dashboard launcher in `QUICKSTART.md` is the supported preview entrypoint",
        V3_DASHBOARD_INFO_COMMAND,
        "JARVIS_STATUS_HOST",
        "JARVIS_STATUS_PORT",
        "http://127.0.0.1:8766",
    ):
        if expected not in readme:
            raise SystemExit(f"README missing launcher discovery text: {expected}")
    for forbidden in (
        "python3 launch_jarvis_v3_dashboard.py",
        "python3 -m jarvis_v2.scripts.run_status_server",
    ):
        if forbidden in readme:
            raise SystemExit(f"README exposed a competing dashboard launcher: {forbidden}")
    for expected in (
        "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=1",
        "operator intent; it is not daemon attestation",
        "Model execution remains unverified",
    ):
        if expected not in readme:
            raise SystemExit(f"README missing Ollama trust-boundary text: {expected}")

    env_example = ENV_EXAMPLE.read_text()
    for expected in (
        "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=0",
        "Stored personal context is eligible for Ollama only when OLLAMA_NO_CLOUD=1",
        "daemon execution remains unverified",
    ):
        if expected not in env_example:
            raise SystemExit(f".env.example missing Ollama trust-boundary text: {expected}")

    imessage_launcher = IMESSAGE_LAUNCHER.read_text()
    for expected in (
        "legacy/dev-only",
        "Telegram is the primary remote command channel",
        "Every side effect still goes through the V2 safety harness",
    ):
        if expected not in imessage_launcher:
            raise SystemExit(f"iMessage launcher handoff text drifted: {expected}")
    for module_name in (
        "jarvis_v2.scripts.run_telegram_control",
        "jarvis_v2.scripts.run_imessage_control",
    ):
        _assert_warmup_success_is_daemon_bounded(module_name)
        _assert_warmup_failure_is_bounded(module_name)
        _assert_invalid_provider_skips_warmup(module_name)

    with TemporaryDirectory(prefix="jarvis-launcher-discovery-") as temp:
        root = Path(temp)
        runtime = JarvisRuntime(
            JarvisConfig(
                data_dir=root,
                db_path=root / "jarvis.sqlite",
                obsidian_vault=root / "Vault",
                obsidian_root="Jarvis",
                chat_model="jarvis-v2-missing-smoke-model",
                use_model_planner=False,
            )
        )
        dashboard_result = runtime.registry.get("status_dashboard").handler({})
        routed_dashboard_result = runtime.handle("status dashboard")
        runtime.handle("run command python3 --version")
        dashboard_diagnosis = runtime.diagnose_command("status dashboard")
        dashboard_action_rehearsal = runtime.registry.get("action_rehearsal").handler({"request": "status dashboard"})
        dashboard_turn_rehearsal = runtime.registry.get("assistant_turn_rehearsal").handler({"message": "status dashboard"})
        dashboard_command_diagnosis = runtime.registry.get("command_diagnosis").handler({"request": "status dashboard"})
    for expected in (
        V3_DASHBOARD_COMMAND,
        V3_DASHBOARD_MODULE_COMMAND,
        "http://127.0.0.1:8766",
        "http://127.0.0.1:8766/api/status",
    ):
        if expected not in dashboard_result.output:
            raise SystemExit(f"status_dashboard output missing launcher text: {expected}")
    if dashboard_result.metadata.get("calls_model") is not False or dashboard_result.metadata.get("executes_tools") is not False:
        raise SystemExit("status_dashboard metadata should stay read-only and model-free")
    if not routed_dashboard_result.verified:
        raise SystemExit("runtime did not route 'status dashboard' successfully")
    if routed_dashboard_result.tool_results[0].tool_name != "status_dashboard":
        raise SystemExit(f"runtime routed 'status dashboard' to {routed_dashboard_result.tool_results[0].tool_name}")
    if V3_DASHBOARD_COMMAND not in routed_dashboard_result.response:
        raise SystemExit("runtime 'status dashboard' response missed root launcher")
    if dashboard_diagnosis.get("route") != "auto_tool":
        raise SystemExit(f"status dashboard diagnosis should bypass recovery debt: {dashboard_diagnosis}")
    if dashboard_diagnosis.get("recovery_closure_blocks_auto_execution") is not True:
        raise SystemExit("status dashboard diagnosis test did not create recovery debt")
    if "send this command normally" not in dashboard_diagnosis.get("recommended_next_commands", []):
        raise SystemExit(f"status dashboard diagnosis missed safe next command: {dashboard_diagnosis}")
    for tool_name, result in (
        ("action_rehearsal", dashboard_action_rehearsal),
        ("assistant_turn_rehearsal", dashboard_turn_rehearsal),
        ("command_diagnosis", dashboard_command_diagnosis),
    ):
        if result.metadata.get("route") != "auto_tool":
            raise SystemExit(f"{tool_name} should keep read-only dashboard preview usable during recovery debt: {result.metadata}")
        if result.metadata.get("safe_to_execute_now") is not True:
            raise SystemExit(f"{tool_name} should mark read-only dashboard preview safe: {result.metadata}")
        if result.metadata.get("recovery_closure_blocks_auto_execution") is not True:
            raise SystemExit(f"{tool_name} regression did not preserve recovery debt metadata: {result.metadata}")
    diagnosis_handoff = dashboard_command_diagnosis.metadata.get("command_diagnosis_handoff") or {}
    if diagnosis_handoff.get("recovery_closure_proof_queue") != dashboard_command_diagnosis.metadata.get("recovery_closure_proof_queue"):
        raise SystemExit(f"command_diagnosis handoff missed recovery proof queue: {diagnosis_handoff}")
    if diagnosis_handoff.get("recovery_closure_proof_queue_count") != dashboard_command_diagnosis.metadata.get("recovery_closure_proof_queue_count"):
        raise SystemExit(f"command_diagnosis handoff missed recovery proof queue count: {diagnosis_handoff}")
    if diagnosis_handoff.get("recovery_closure_next_proof_command") != dashboard_command_diagnosis.metadata.get("recovery_closure_next_proof_command"):
        raise SystemExit(f"command_diagnosis handoff missed recovery next proof command: {diagnosis_handoff}")
    if diagnosis_handoff.get("execution_learning_proof_queue") != dashboard_command_diagnosis.metadata.get("execution_learning_proof_queue"):
        raise SystemExit(f"command_diagnosis handoff missed execution learning proof queue: {diagnosis_handoff}")
    if diagnosis_handoff.get("execution_learning_proof_queue_count") != dashboard_command_diagnosis.metadata.get("execution_learning_proof_queue_count"):
        raise SystemExit(f"command_diagnosis handoff missed execution learning proof queue count: {diagnosis_handoff}")
    if diagnosis_handoff.get("execution_learning_next_proof_command") != dashboard_command_diagnosis.metadata.get("execution_learning_next_proof_command"):
        raise SystemExit(f"command_diagnosis handoff missed execution learning next proof command: {diagnosis_handoff}")

    print("launcher discovery smoke passed")


if __name__ == "__main__":
    main()
