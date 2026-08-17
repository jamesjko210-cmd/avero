from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import jarvis_v2.scripts.smoke_test_all as smoke_test_all
from jarvis_v2.env import read_env_values
from jarvis_v2.scripts.smoke_test_all import (
    COMPLETED_PROCESS_GROUP_SETTLE_SECONDS,
    DEFAULT_MODULE_TIMEOUT_SECONDS,
    DEFAULT_SUITE_LOCK_TIMEOUT_SECONDS,
    MAX_MODULE_TIMEOUT_SECONDS,
    MAX_MODULE_CAPTURE_BYTES,
    MAX_FAILING_MODULE_OUTPUT_CHARS,
    MAX_MODULE_LABEL_CHARS,
    MAX_SUITE_LOCK_TIMEOUT_SECONDS,
    MIN_MODULE_TIMEOUT_SECONDS,
    MIN_SUITE_LOCK_TIMEOUT_SECONDS,
    MODULE_OUTPUT_LIMIT_MARKER,
    RESIDUAL_PROCESS_GROUP_MARKER,
    STATUS_SERVER_SMOKE_MODULE,
    STATUS_SERVER_DEFAULT_MODULE_TIMEOUT_SECONDS,
    SENSITIVE_CHILD_ENV_KEYS,
    SUITE_LOCK_ENV_KEY,
    SUITE_LOCK_TIMEOUT_ENV_KEY,
    TEST_MODULES,
    _aggregate_run_lock,
    _child_env,
    _discover_smoke_modules,
    _duplicate_modules,
    _failure_summary_with_duration,
    _failing_module_output,
    _format_duration,
    _failure_summary,
    _has_modules,
    _isolated_child_env,
    _missing_modules,
    _module_heading,
    _module_timeout_seconds,
    _module_timeout_seconds_for,
    _non_smoke_modules,
    _passing_module_output,
    _python_compile_failures,
    _retryable_module_failure,
    _run_module,
    _safe_module_label,
    _safe_stream_text,
    _should_strip_child_env_key,
    _stream_text,
    _success_summary,
    _success_summary_with_duration,
    _suite_summary,
    _suite_lock_path,
    _suite_lock_timeout_seconds,
    _suite_timeout_note,
    _timeout_for_timeout_message,
    _unclean_modules,
    _uninvoked_test_functions,
    _uninvoked_test_functions_from_source,
    _unlisted_smoke_modules,
)
from jarvis_v2.scripts.v3_python_runtime import (
    V3_INHERITED_CUSTODY_KEYS,
    assert_v3_environment_custody,
)


def _env_example_keys(root: Path) -> list[str]:
    keys: list[str] = []
    for line in (root / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if COMPLETED_PROCESS_GROUP_SETTLE_SECONDS != 5.0:
        raise SystemExit("aggregate completed-process settle contract drifted")
    old_live = os.environ.get("JARVIS_LIVE_GMAIL_SMOKE")
    old_live_other = os.environ.get("JARVIS_LIVE_CALENDAR_SMOKE")
    old_live_generic = os.environ.get("JARVIS_LIVE_SMOKE")
    old_safe = os.environ.get("JARVIS_DATA_DIR")
    old_timeout = os.environ.get("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS")
    old_suite_lock = os.environ.get(SUITE_LOCK_ENV_KEY)
    old_suite_lock_timeout = os.environ.get(SUITE_LOCK_TIMEOUT_ENV_KEY)
    old_env_source = os.environ.get("JARVIS_V3_ENV")
    old_daemon_enable = os.environ.get("JARVIS_V3_ENABLE_DAEMONS")
    old_scheduler_enable = os.environ.get("JARVIS_V3_ENABLE_SCHEDULER")
    marker_sensitive = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "KAKAO_ACCESS_TOKEN",
        "INSTAGRAM_ACCESS_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
    }
    sensitive_test_keys = set(SENSITIVE_CHILD_ENV_KEYS) | marker_sensitive
    old_sensitive = {key: os.environ.get(key) for key in sensitive_test_keys}
    try:
        os.environ["JARVIS_LIVE_GMAIL_SMOKE"] = "1"
        os.environ["JARVIS_LIVE_CALENDAR_SMOKE"] = "1"
        os.environ["JARVIS_LIVE_SMOKE"] = "1"
        for key in sensitive_test_keys:
            os.environ[key] = f"real-{key.lower()}-should-not-reach-child"
        os.environ["JARVIS_DATA_DIR"] = "/tmp/jarvis-smoke-runner-safe"
        os.environ["JARVIS_V3_ENABLE_DAEMONS"] = "1"
        os.environ["JARVIS_V3_ENABLE_SCHEDULER"] = "1"
        child = _child_env()
        if child.get("JARVIS_V3_ENABLE_DAEMONS") != "0":
            raise SystemExit("aggregate smoke runner did not disable V3 daemons")
        if child.get("JARVIS_V3_ENABLE_SCHEDULER") != "0":
            raise SystemExit("aggregate smoke runner did not disable V3 scheduler execution")
        for key in ["JARVIS_LIVE_GMAIL_SMOKE", "JARVIS_LIVE_CALENDAR_SMOKE", "JARVIS_LIVE_SMOKE"]:
            if key in child:
                raise SystemExit(f"aggregate smoke runner leaked live flag {key}")
        for key in sensitive_test_keys:
            if key in child:
                raise SystemExit(f"aggregate smoke runner leaked sensitive env {key}")
        if child.get("JARVIS_DATA_DIR") != "/tmp/jarvis-smoke-runner-safe":
            raise SystemExit("aggregate smoke runner stripped ordinary local-safe env unexpectedly")
        child_env_source = Path(child.get("JARVIS_V3_ENV", ""))
        if not child_env_source.is_file() or child_env_source.read_text(encoding="utf-8"):
            raise SystemExit("aggregate smoke runner did not force an empty child env source")
        with TemporaryDirectory(prefix="jarvis-smoke-bytecode-env-") as bytecode_temp:
            isolated_root = Path(bytecode_temp)
            with mock.patch.dict(
                os.environ,
                {
                    "JARVIS_V3_PYTHON": "/production/interpreter-must-not-reach-smokes",
                    "JARVIS_OWNER_IMESSAGE": "production-owner-must-not-reach-smokes",
                },
                clear=False,
            ):
                isolated = _isolated_child_env(isolated_root)
            if isolated.get("PYTHONDONTWRITEBYTECODE") != "1":
                raise SystemExit("aggregate child smokes may write bytecode into the candidate tree")
            selected_path = Path(isolated.get("JARVIS_V3_ENV", ""))
            if selected_path != isolated_root / "runtime.env":
                raise SystemExit("aggregate smoke isolation did not select its bounded runtime env")
            if not selected_path.is_file() or selected_path.stat().st_mode & 0o077:
                raise SystemExit("aggregate smoke isolation env is not owner-only")
            selected_values = read_env_values(selected_path, require_owner_only=True)
            inherited_custody = {
                key: isolated[key]
                for key in V3_INHERITED_CUSTODY_KEYS
                if key in isolated
            }
            if selected_values != inherited_custody:
                raise SystemExit("aggregate smoke selected and inherited custody values diverged")
            if (
                "JARVIS_V3_PYTHON" in isolated
                or "JARVIS_OWNER_IMESSAGE" in isolated
                or "production" in selected_path.read_text(encoding="utf-8")
            ):
                raise SystemExit("aggregate smoke isolation retained production custody")
            with mock.patch.dict(os.environ, isolated, clear=True):
                custody = assert_v3_environment_custody()
            if not custody.selected_environment or custody.configured_python is not None:
                raise SystemExit("aggregate smoke isolation was refused by V3 runtime custody")
        for key in marker_sensitive:
            if not _should_strip_child_env_key(key):
                raise SystemExit(f"aggregate smoke runner did not classify sensitive marker env {key}")
        for key in ["JARVIS_DATA_DIR", "JARVIS_STATUS_HOST", "OLLAMA_MODEL"]:
            if _should_strip_child_env_key(key):
                raise SystemExit(f"aggregate smoke runner classified local-safe env as sensitive: {key}")
        for key in _env_example_keys(root):
            value = key.upper()
            looks_sensitive = (
                value.startswith("GMAIL_")
                or value.startswith("TELEGRAM_")
                or value in SENSITIVE_CHILD_ENV_KEYS
                or any(marker in value for marker in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "CREDS", "API_KEY", "ACCESS_KEY", "PRIVATE_KEY"))
            )
            if looks_sensitive and not _should_strip_child_env_key(key):
                raise SystemExit(f".env.example sensitive env is not stripped from child smokes: {key}")

        with TemporaryDirectory(prefix="jarvis-smoke-child-env-") as temp_dir:
            fixture_env = Path(temp_dir) / "fixture.env"
            fixture_env.write_text(
                "OPENAI_API_KEY=fixture-secret-sentinel\n"
                "JARVIS_LIVE_SMOKE=fixture-live-sentinel\n",
                encoding="utf-8",
            )
            fixture_env.chmod(0o600)
            os.environ["JARVIS_V3_ENV"] = str(fixture_env)
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os; "
                        "from jarvis_v2.env import load_env; "
                        "from jarvis_v2.config import load_config; "
                        "load_env(); load_config(); "
                        "raise SystemExit(0 if all(os.getenv(key) is None for key in "
                        "('OPENAI_API_KEY', 'JARVIS_LIVE_SMOKE')) else 1)"
                    ),
                ],
                capture_output=True,
                text=True,
                env=_child_env(),
                timeout=30,
            )
            if probe.returncode != 0:
                raise SystemExit("aggregate child recovered sensitive values from a configured env source")

        duplicates = _duplicate_modules(["one", "two", "one", "three", "two", "two"])
        if duplicates != ["one", "two"]:
            raise SystemExit(f"aggregate smoke runner duplicate detection drifted: {duplicates!r}")
        if _duplicate_modules(["one", "two", "three"]):
            raise SystemExit("aggregate smoke runner reported duplicates for a unique module list")

        unclean = _unclean_modules(["jarvis_v2.scripts.smoke_test_core", " jarvis_v2.scripts.smoke_test_help", ""])
        if unclean != [" jarvis_v2.scripts.smoke_test_help", ""]:
            raise SystemExit(f"aggregate smoke runner unclean-module detection drifted: {unclean!r}")
        if _unclean_modules(["jarvis_v2.scripts.smoke_test_core", "jarvis_v2.scripts.smoke_test_help"]):
            raise SystemExit("aggregate smoke runner reported clean module names as unclean")

        missing = _missing_modules(["jarvis_v2.scripts.smoke_test_smoke_runner", "jarvis_v2.scripts.no_such_smoke"])
        if missing != ["jarvis_v2.scripts.no_such_smoke"]:
            raise SystemExit(f"aggregate smoke runner missing-module detection drifted: {missing!r}")
        if _missing_modules(["jarvis_v2.scripts.smoke_test_all"]):
            raise SystemExit("aggregate smoke runner reported a known module as missing")

        non_smoke = _non_smoke_modules(["jarvis_v2.scripts.smoke_test_all", "jarvis_v2.scripts.run_telegram_control"])
        if non_smoke != ["jarvis_v2.scripts.run_telegram_control"]:
            raise SystemExit(f"aggregate smoke runner non-smoke detection drifted: {non_smoke!r}")
        if _non_smoke_modules(["jarvis_v2.scripts.smoke_test_core", "jarvis_v2.scripts.smoke_test_safety"]):
            raise SystemExit("aggregate smoke runner reported smoke modules as non-smoke")

        discovered = _discover_smoke_modules()
        if "jarvis_v2.scripts.smoke_test_live" in discovered:
            raise SystemExit("aggregate smoke runner should exclude opt-in live smoke from default discovery")
        if "jarvis_v2.scripts.smoke_test_all" in discovered:
            raise SystemExit("aggregate smoke runner should exclude itself from default discovery")
        if "jarvis_v2.scripts.smoke_test_core" not in discovered:
            raise SystemExit("aggregate smoke runner discovery missed a normal smoke module")
        unlisted = _unlisted_smoke_modules(["jarvis_v2.scripts.smoke_test_core"], [
            "jarvis_v2.scripts.smoke_test_core",
            "jarvis_v2.scripts.smoke_test_new_feature",
        ])
        if unlisted != ["jarvis_v2.scripts.smoke_test_new_feature"]:
            raise SystemExit(f"aggregate smoke runner unlisted-module detection drifted: {unlisted!r}")
        if _unlisted_smoke_modules(TEST_MODULES):
            raise SystemExit("aggregate smoke runner TEST_MODULES missed a discovered default smoke")

        with TemporaryDirectory(prefix="jarvis-smoke-compile-") as temp_dir:
            compile_root = Path(temp_dir)
            package_root = compile_root / "jarvis_v2"
            package_root.mkdir()
            (compile_root / "entry.py").write_text("value = 1\n", encoding="utf-8")
            broken_path = package_root / "broken.py"
            broken_path.write_text("def broken(:\n    pass\n", encoding="utf-8")
            compile_failures = _python_compile_failures(compile_root)
            if len(compile_failures) != 1 or compile_failures[0][0] != "jarvis_v2/broken.py":
                raise SystemExit(f"aggregate compile preflight missed repository syntax error: {compile_failures!r}")
            broken_path.write_text("def fixed():\n    pass\n", encoding="utf-8")
            if _python_compile_failures(compile_root):
                raise SystemExit("aggregate compile preflight reported valid repository sources")

        synthetic_uninvoked = _uninvoked_test_functions_from_source(
            """
def test_called():
    pass

def test_missing():
    pass

def helper():
    test_called()

def main():
    helper()

if __name__ == "__main__":
    main()
""",
            "synthetic_smoke",
        )
        if synthetic_uninvoked != ["test_missing"]:
            raise SystemExit(f"aggregate smoke runner uninvoked-test detection drifted: {synthetic_uninvoked!r}")
        direct_async_uninvoked = _uninvoked_test_functions_from_source(
            """
async def test_async_called():
    pass

async def test_async_missing():
    pass

def main():
    test_async_called()

if __name__ == "__main__":
    main()
""",
            "synthetic_direct_async_smoke",
        )
        if direct_async_uninvoked != ["test_async_called", "test_async_missing"]:
            raise SystemExit(f"aggregate smoke runner direct-async detection drifted: {direct_async_uninvoked!r}")
        async_run_uninvoked = _uninvoked_test_functions_from_source(
            """
import asyncio

async def test_async_called():
    pass

async def test_async_missing():
    pass

def main():
    asyncio.run(test_async_called())

if __name__ == "__main__":
    main()
""",
            "synthetic_asyncio_run_smoke",
        )
        if async_run_uninvoked != ["test_async_missing"]:
            raise SystemExit(f"aggregate smoke runner asyncio.run async detection drifted: {async_run_uninvoked!r}")
        dead_helper_uninvoked = _uninvoked_test_functions_from_source(
            """
def test_dead_helper_only():
    pass

def helper():
    test_dead_helper_only()

def main():
    pass

if __name__ == "__main__":
    main()
""",
            "synthetic_dead_helper_smoke",
        )
        if dead_helper_uninvoked != ["test_dead_helper_only"]:
            raise SystemExit(f"aggregate smoke runner dead-helper detection drifted: {dead_helper_uninvoked!r}")
        self_call_uninvoked = _uninvoked_test_functions_from_source(
            """
def test_self_call_only():
    test_self_call_only()

def main():
    pass

if __name__ == "__main__":
    main()
""",
            "synthetic_self_call_smoke",
        )
        if self_call_uninvoked != ["test_self_call_only"]:
            raise SystemExit(f"aggregate smoke runner self-call detection drifted: {self_call_uninvoked!r}")
        false_guard_uninvoked = _uninvoked_test_functions_from_source(
            """
def test_false_guard_only():
    pass

def main():
    test_false_guard_only()

if False:
    main()
""",
            "synthetic_false_guard_smoke",
        )
        if false_guard_uninvoked != ["test_false_guard_only"]:
            raise SystemExit(f"aggregate smoke runner false-guard detection drifted: {false_guard_uninvoked!r}")
        env_guard_uninvoked = _uninvoked_test_functions_from_source(
            """
import os

def test_env_guard_only():
    pass

def main():
    test_env_guard_only()

if os.getenv("RUN_SMOKE_TESTS"):
    main()
""",
            "synthetic_env_guard_smoke",
        )
        if env_guard_uninvoked != ["test_env_guard_only"]:
            raise SystemExit(f"aggregate smoke runner env-guard detection drifted: {env_guard_uninvoked!r}")
        reversed_main_guard_uninvoked = _uninvoked_test_functions_from_source(
            """
def test_reversed_guard_called():
    pass

def test_reversed_guard_missing():
    pass

def main():
    test_reversed_guard_called()

if "__main__" == __name__:
    main()
""",
            "synthetic_reversed_main_guard_smoke",
        )
        if reversed_main_guard_uninvoked != ["test_reversed_guard_missing"]:
            raise SystemExit(f"aggregate smoke runner reversed-main-guard detection drifted: {reversed_main_guard_uninvoked!r}")
        class_method_uninvoked = _uninvoked_test_functions_from_source(
            """
class SmokeCase:
    def test_method_called(self):
        pass

    def test_method_missing(self):
        pass

def main():
    SmokeCase().test_method_called()

if __name__ == "__main__":
    main()
""",
            "synthetic_class_method_smoke",
        )
        if class_method_uninvoked != ["SmokeCase.test_method_missing"]:
            raise SystemExit(f"aggregate smoke runner class-method detection drifted: {class_method_uninvoked!r}")
        instance_method_uninvoked = _uninvoked_test_functions_from_source(
            """
class SmokeCase:
    def test_instance_method_called(self):
        pass

    def test_instance_method_missing(self):
        pass

def main():
    case = SmokeCase()
    case.test_instance_method_called()

if __name__ == "__main__":
    main()
""",
            "synthetic_instance_method_smoke",
        )
        if instance_method_uninvoked != ["SmokeCase.test_instance_method_missing"]:
            raise SystemExit(f"aggregate smoke runner instance-method detection drifted: {instance_method_uninvoked!r}")
        uncalled_class_methods = _uninvoked_test_functions_from_source(
            """
class SmokeCase:
    def test_pytest_style_only(self):
        pass

def main():
    pass

if __name__ == "__main__":
    main()
""",
            "synthetic_pytest_style_class_smoke",
        )
        if uncalled_class_methods != ["SmokeCase.test_pytest_style_only"]:
            raise SystemExit(f"aggregate smoke runner pytest-style class detection drifted: {uncalled_class_methods!r}")
        if _uninvoked_test_functions_from_source(
            """
def test_first():
    pass

def test_second():
    pass

def main():
    test_first()
    test_second()

if __name__ == "__main__":
    main()
""",
            "synthetic_smoke_clean",
        ):
            raise SystemExit("aggregate smoke runner reported invoked synthetic tests as uninvoked")
        uninvoked = _uninvoked_test_functions(TEST_MODULES)
        if uninvoked:
            raise SystemExit(f"aggregate smoke runner found uninvoked tests in configured suite: {uninvoked!r}")

        if _has_modules([]):
            raise SystemExit("aggregate smoke runner should reject an empty module list")
        if not _has_modules(["jarvis_v2.scripts.smoke_test_core"]):
            raise SystemExit("aggregate smoke runner should accept a non-empty module list")

        if _stream_text(None) != "":
            raise SystemExit("aggregate smoke runner should normalize empty timeout streams")
        if _stream_text(b"partial\xffstdout") != "partial\ufffdstdout":
            raise SystemExit("aggregate smoke runner should decode timeout byte streams safely")
        if _stream_text("plain stdout") != "plain stdout":
            raise SystemExit("aggregate smoke runner should preserve timeout text streams")
        synthetic_secret = "s" + "k-" + "live-aggregate-smoke-secret"
        safe_env = {
            "OPENAI_API_KEY": synthetic_secret,
            "JARVIS_DATA_DIR": "/tmp/jarvis-smoke-runner-safe",
        }
        redacted = _safe_stream_text(
            f"stdout leaked {synthetic_secret} but kept /tmp/jarvis-smoke-runner-safe",
            safe_env,
        )
        if synthetic_secret in redacted or "<redacted-env-value>" not in redacted:
            raise SystemExit(f"aggregate smoke runner should redact sensitive env values in output: {redacted!r}")
        if "/tmp/jarvis-smoke-runner-safe" not in redacted:
            raise SystemExit(f"aggregate smoke runner should preserve local-safe env values in output: {redacted!r}")
        redacted_bytes = _safe_stream_text(f"stderr {synthetic_secret}".encode("utf-8"), safe_env)
        if synthetic_secret in redacted_bytes or "<redacted-env-value>" not in redacted_bytes:
            raise SystemExit("aggregate smoke runner should redact sensitive env values in byte streams")
        if _passing_module_output("short stdout") != "short stdout":
            raise SystemExit("aggregate smoke runner should preserve short passing output")
        seeded_secret = os.environ["OPENAI_API_KEY"]
        passing_redacted = _passing_module_output(f"passing stdout {seeded_secret}")
        if seeded_secret in passing_redacted or "<redacted-env-value>" not in passing_redacted:
            raise SystemExit("aggregate smoke runner should redact sensitive env values in passing output")
        long_stdout = "x" * 1300
        preview = _passing_module_output(long_stdout)
        if len(preview) >= len(long_stdout):
            raise SystemExit("aggregate smoke runner should bound long passing output")
        if "omitted 100 chars" not in preview:
            raise SystemExit(f"aggregate smoke runner truncation summary drifted: {preview!r}")
        long_failure = "failure prefix " + os.environ["OPENAI_API_KEY"] + " " + ("y" * (MAX_FAILING_MODULE_OUTPUT_CHARS + 50))
        failure_preview = _failing_module_output(long_failure, stream_name="stdout")
        if os.environ["OPENAI_API_KEY"] in failure_preview or "<redacted-env-value>" not in failure_preview:
            raise SystemExit("aggregate smoke runner should redact sensitive env values in failing output")
        if len(failure_preview) >= len(long_failure):
            raise SystemExit("aggregate smoke runner should bound long failing output")
        if "stdout truncated for failing module" not in failure_preview or "omitted" not in failure_preview:
            raise SystemExit(f"aggregate smoke runner failing truncation summary drifted: {failure_preview!r}")
        failure_bytes = _failing_module_output(
            ("stderr leaked " + os.environ["OPENAI_API_KEY"] + " " + ("z" * (MAX_FAILING_MODULE_OUTPUT_CHARS + 50))).encode(),
            stream_name="stderr",
        )
        if os.environ["OPENAI_API_KEY"] in failure_bytes or "<redacted-env-value>" not in failure_bytes:
            raise SystemExit("aggregate smoke runner should redact sensitive env values in failing byte output")
        if "stderr truncated for failing module" not in failure_bytes:
            raise SystemExit("aggregate smoke runner should mark bounded failing stderr")
        if _safe_module_label("jarvis_v2.scripts.smoke_test_core") != "jarvis_v2.scripts.smoke_test_core":
            raise SystemExit("aggregate smoke runner should preserve normal module labels")
        secret_module = f"jarvis_v2.scripts.smoke_test_{os.environ['OPENAI_API_KEY']}"
        secret_label = _safe_module_label(secret_module)
        if os.environ["OPENAI_API_KEY"] in secret_label or "<redacted-env-value>" not in secret_label:
            raise SystemExit(f"aggregate smoke runner should redact sensitive env values in module labels: {secret_label!r}")
        path_label = _safe_module_label("/\x55sers/example/private/smoke_test_secret_module SHOULD NOT APPEAR")
        if "<local-path>" not in path_label:
            raise SystemExit(f"aggregate smoke runner should redact path-shaped module labels: {path_label!r}")
        for forbidden in ("/" + "Users/operator", "smoke_test_secret_module", "SHOULD NOT APPEAR"):
            if forbidden in path_label:
                raise SystemExit(f"aggregate smoke runner leaked path-shaped module label {forbidden!r}: {path_label!r}")
        long_label = _safe_module_label("jarvis_v2.scripts.smoke_test_" + ("x" * (MAX_MODULE_LABEL_CHARS + 50)))
        if len(long_label) > MAX_MODULE_LABEL_CHARS or not long_label.endswith("…"):
            raise SystemExit(f"aggregate smoke runner should bound long module labels: {long_label!r}")

        if _suite_summary(["jarvis_v2.scripts.smoke_test_core"], 7.0) != "Running 1 smoke module with 7s per-module timeout.":
            raise SystemExit("aggregate smoke runner summary should singularize one module")
        if _suite_summary(["a", "b"], 12.5) != "Running 2 smoke modules with 12.5s per-module timeout.":
            raise SystemExit("aggregate smoke runner summary should include module count and timeout")
        expected_timeout_note = (
            f"Status-server smoke timeout: {STATUS_SERVER_DEFAULT_MODULE_TIMEOUT_SECONDS}s "
            f"(default {DEFAULT_MODULE_TIMEOUT_SECONDS}s for other modules)."
        )
        os.environ.pop("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", None)
        if _suite_timeout_note([STATUS_SERVER_SMOKE_MODULE]) != expected_timeout_note:
            raise SystemExit("aggregate smoke runner missed status-server timeout note")
        if _suite_timeout_note(["jarvis_v2.scripts.smoke_test_core"]):
            raise SystemExit("aggregate smoke runner should not print timeout note without status-server smoke")
        os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = "240"
        if _suite_timeout_note([STATUS_SERVER_SMOKE_MODULE]):
            raise SystemExit("aggregate smoke runner should not print status timeout note when env override is set")
        os.environ.pop("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", None)
        if _success_summary(["jarvis_v2.scripts.smoke_test_core"]) != "All 1 smoke module passed.":
            raise SystemExit("aggregate smoke runner success summary should singularize one module")
        if _success_summary(["a", "b"]) != "All 2 smoke modules passed.":
            raise SystemExit("aggregate smoke runner success summary should include module count")
        if _failure_summary(["a"], ["a", "b"]) != "FAILED: 1 of 2 smoke modules failed.":
            raise SystemExit("aggregate smoke runner failure summary should include failure count")
        if _failure_summary(["a"], ["a"]) != "FAILED: 1 of 1 smoke module failed.":
            raise SystemExit("aggregate smoke runner failure summary should singularize one module")
        if _format_duration(-1.0) != "0.0s":
            raise SystemExit("aggregate smoke runner duration formatter should clamp negative elapsed time")
        if _format_duration(12.34) != "12.3s":
            raise SystemExit("aggregate smoke runner duration formatter should round to tenths")
        if _success_summary_with_duration(["a", "b"], 12.34) != "All 2 smoke modules passed. Duration: 12.3s.":
            raise SystemExit("aggregate smoke runner success duration summary drifted")
        if _failure_summary_with_duration(["a"], ["a", "b"], 1.25) != "FAILED: 1 of 2 smoke modules failed. Duration: 1.2s.":
            raise SystemExit("aggregate smoke runner failure duration summary drifted")
        if _module_heading(3, 72, "jarvis_v2.scripts.smoke_test_core") != "== [3/72] jarvis_v2.scripts.smoke_test_core ==":
            raise SystemExit("aggregate smoke runner module heading should include ordinal progress")
        unsafe_heading = _module_heading(1, 1, f"/private/tmp/smoke_test_secret {os.environ['OPENAI_API_KEY']} SHOULD NOT APPEAR")
        if "<local-path>" not in unsafe_heading or os.environ["OPENAI_API_KEY"] in unsafe_heading:
            raise SystemExit(f"aggregate smoke runner heading should use safe module labels: {unsafe_heading!r}")
        for forbidden in ("/private/tmp", "SHOULD NOT APPEAR"):
            if forbidden in unsafe_heading:
                raise SystemExit(f"aggregate smoke runner heading leaked unsafe module label {forbidden!r}: {unsafe_heading!r}")
        if TEST_MODULES.index(STATUS_SERVER_SMOKE_MODULE) > 5:
            raise SystemExit("aggregate smoke runner should run status-server smoke early to avoid late local bind flakes")
        retryable = subprocess.CompletedProcess(
            ["python3", "-m", STATUS_SERVER_SMOKE_MODULE],
            1,
            "Traceback...\nPermissionError: [Errno 1] Operation not permitted\n",
            "",
        )
        if not _retryable_module_failure(STATUS_SERVER_SMOKE_MODULE, retryable):
            raise SystemExit("aggregate smoke runner should retry transient status-server bind permission failures")
        non_retryable_status = subprocess.CompletedProcess(
            ["python3", "-m", STATUS_SERVER_SMOKE_MODULE],
            1,
            "AssertionError: real dashboard contract failure",
            "",
        )
        if _retryable_module_failure(STATUS_SERVER_SMOKE_MODULE, non_retryable_status):
            raise SystemExit("aggregate smoke runner should not retry ordinary status-server failures")
        other_module = subprocess.CompletedProcess(
            ["python3", "-m", "jarvis_v2.scripts.smoke_test_core"],
            1,
            "PermissionError: [Errno 1] Operation not permitted",
            "",
        )
        if _retryable_module_failure("jarvis_v2.scripts.smoke_test_core", other_module):
            raise SystemExit("aggregate smoke runner should only retry the status-server smoke permission edge")
        passing_status = subprocess.CompletedProcess(
            ["python3", "-m", STATUS_SERVER_SMOKE_MODULE],
            0,
            "ok",
            "",
        )
        if _retryable_module_failure(STATUS_SERVER_SMOKE_MODULE, passing_status):
            raise SystemExit("aggregate smoke runner should not retry a passing status-server smoke")

        retry_timeout = subprocess.TimeoutExpired(
            [sys.executable, "-m", STATUS_SERVER_SMOKE_MODULE],
            STATUS_SERVER_DEFAULT_MODULE_TIMEOUT_SECONDS,
            output="partial retry output",
            stderr="partial retry error",
        )
        retry_output = io.StringIO()
        with (
            mock.patch.object(smoke_test_all, "TEST_MODULES", [STATUS_SERVER_SMOKE_MODULE]),
            mock.patch.object(smoke_test_all, "_aggregate_run_lock", return_value=nullcontext()),
            mock.patch.object(smoke_test_all, "_duplicate_modules", return_value=[]),
            mock.patch.object(smoke_test_all, "_unclean_modules", return_value=[]),
            mock.patch.object(smoke_test_all, "_missing_modules", return_value=[]),
            mock.patch.object(smoke_test_all, "_non_smoke_modules", return_value=[]),
            mock.patch.object(smoke_test_all, "_unlisted_smoke_modules", return_value=[]),
            mock.patch.object(smoke_test_all, "_python_compile_failures", return_value=[]),
            mock.patch.object(smoke_test_all, "_uninvoked_test_functions", return_value={}),
            mock.patch.object(
                smoke_test_all,
                "_run_module",
                side_effect=[retryable, retry_timeout],
            ) as aggregate_run,
            redirect_stdout(retry_output),
        ):
            try:
                smoke_test_all.main()
            except SystemExit as exc:
                if exc.code != 1:
                    raise SystemExit(f"aggregate retry timeout exit drifted: {exc.code!r}") from exc
            else:
                raise SystemExit("aggregate retry timeout should fail the bounded suite")
        retry_text = retry_output.getvalue()
        for expected in ("RETRY:", "TIMEOUT:", "FAILED: 1 of 1 smoke module failed."):
            if expected not in retry_text:
                raise SystemExit(f"aggregate retry timeout missed bounded summary {expected!r}: {retry_text!r}")
        if aggregate_run.call_count != 2:
            raise SystemExit(f"aggregate retry timeout should stop after two attempts: {aggregate_run.call_count}")

        os.environ.pop("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", None)
        if _module_timeout_seconds() != float(DEFAULT_MODULE_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner default timeout drifted")
        if _module_timeout_seconds_for("jarvis_v2.scripts.smoke_test_core") != float(DEFAULT_MODULE_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner default module timeout drifted")
        if _module_timeout_seconds_for(STATUS_SERVER_SMOKE_MODULE) != float(STATUS_SERVER_DEFAULT_MODULE_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner status-server module timeout drifted")
        if _timeout_for_timeout_message(STATUS_SERVER_SMOKE_MODULE, 123.0) != float(STATUS_SERVER_DEFAULT_MODULE_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner timeout message should use status-server module timeout")
        env_example = (root / ".env.example").read_text(encoding="utf-8")
        expected_env_line = f"JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS={DEFAULT_MODULE_TIMEOUT_SECONDS}"
        if expected_env_line not in env_example:
            raise SystemExit(f".env.example smoke timeout default drifted; expected {expected_env_line}")
        os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = "bad"
        if _module_timeout_seconds() != float(DEFAULT_MODULE_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner should fall back on invalid timeout env")
        os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = "1"
        if _module_timeout_seconds() != float(MIN_MODULE_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner should clamp tiny timeout env")
        os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = "inf"
        if _module_timeout_seconds() != float(DEFAULT_MODULE_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner should fall back on non-finite timeout env")
        os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = str(MAX_MODULE_TIMEOUT_SECONDS + 100)
        if _module_timeout_seconds() != float(MAX_MODULE_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner should clamp huge timeout env")

        os.environ.pop(SUITE_LOCK_ENV_KEY, None)
        os.environ.pop(SUITE_LOCK_TIMEOUT_ENV_KEY, None)
        if _suite_lock_timeout_seconds() != float(DEFAULT_SUITE_LOCK_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner suite-lock default timeout drifted")
        os.environ[SUITE_LOCK_TIMEOUT_ENV_KEY] = "bad"
        if _suite_lock_timeout_seconds() != float(DEFAULT_SUITE_LOCK_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner should fall back on invalid suite-lock timeout env")
        os.environ[SUITE_LOCK_TIMEOUT_ENV_KEY] = "inf"
        if _suite_lock_timeout_seconds() != float(DEFAULT_SUITE_LOCK_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner should fall back on non-finite suite-lock timeout env")
        os.environ[SUITE_LOCK_TIMEOUT_ENV_KEY] = "-5"
        if _suite_lock_timeout_seconds() != float(MIN_SUITE_LOCK_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner should clamp tiny suite-lock timeout env")
        os.environ[SUITE_LOCK_TIMEOUT_ENV_KEY] = str(MAX_SUITE_LOCK_TIMEOUT_SECONDS + 100)
        if _suite_lock_timeout_seconds() != float(MAX_SUITE_LOCK_TIMEOUT_SECONDS):
            raise SystemExit("aggregate smoke runner should clamp huge suite-lock timeout env")
        with TemporaryDirectory(prefix="jarvis-smoke-suite-lock-") as temp_dir:
            lock_path = Path(temp_dir) / "suite.lock"
            os.environ[SUITE_LOCK_ENV_KEY] = str(lock_path)
            os.environ[SUITE_LOCK_TIMEOUT_ENV_KEY] = "0.25"
            if _suite_lock_path() != lock_path:
                raise SystemExit("aggregate smoke runner suite-lock path env override drifted")
            with _aggregate_run_lock():
                if not lock_path.exists():
                    raise SystemExit("aggregate smoke runner suite-lock should create the lock file")
                lock_text = lock_path.read_text(encoding="utf-8")
                if "pid=" not in lock_text or "started=" not in lock_text:
                    raise SystemExit(f"aggregate smoke runner suite-lock metadata drifted: {lock_text!r}")
            with mock.patch(
                "jarvis_v2.scripts.smoke_test_all.fcntl.flock",
                side_effect=[BlockingIOError(), None, None],
            ) as flock:
                with _aggregate_run_lock():
                    pass
                if flock.call_count != 3:
                    raise SystemExit(f"aggregate smoke runner suite-lock wait path drifted: {flock.call_count}")
            os.environ[SUITE_LOCK_TIMEOUT_ENV_KEY] = "0"
            with mock.patch(
                "jarvis_v2.scripts.smoke_test_all.fcntl.flock",
                side_effect=BlockingIOError(),
            ):
                try:
                    with _aggregate_run_lock():
                        pass
                except SystemExit as exc:
                    if exc.code != 1:
                        raise SystemExit(f"aggregate smoke runner suite-lock timeout exit drifted: {exc.code!r}") from exc
                else:
                    raise SystemExit("aggregate smoke runner suite-lock timeout should fail closed")

        os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = "7"
        if _module_timeout_seconds_for(STATUS_SERVER_SMOKE_MODULE) != 7.0:
            raise SystemExit("aggregate smoke runner env override should control status-server timeout")
        if _timeout_for_timeout_message(STATUS_SERVER_SMOKE_MODULE, 7.0) != 7.0:
            raise SystemExit("aggregate smoke runner timeout message should honor env override fallback")
        with TemporaryDirectory(prefix="jarvis-smoke-protected-storage-") as protected_dir:
            protected_root = Path(protected_dir)
            protected_paths = {
                "HOME": str(protected_root / "production-home"),
                "TMPDIR": str(protected_root / "production-tmp"),
                "XDG_CACHE_HOME": str(protected_root / "production-xdg-cache"),
                "XDG_CONFIG_HOME": str(protected_root / "production-xdg-config"),
                "XDG_DATA_HOME": str(protected_root / "production-xdg-data"),
                "JARVIS_DATA_DIR": str(protected_root / "production-data"),
                "JARVIS_DB_PATH": str(protected_root / "production.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(protected_root / "production-vault"),
                "OBSIDIAN_VAULT_PATH": str(protected_root / "secondary-production-vault"),
                "JARVIS_STORAGE_FALLBACK_DIR": str(protected_root / "production-fallback"),
                "JARVIS_WATCHED_DIRS": str(protected_root / "production-watched"),
                "JARVIS_TELEGRAM_STATE": str(protected_root / "production-telegram-state.json"),
                "JARVIS_V3_IMESSAGE_STATE": str(protected_root / "production-imessage-state.json"),
                "JARVIS_REMINDERS_FILE": str(protected_root / "production-reminders.json"),
                "JARVIS_CACHE_DIR": str(protected_root / "production-cache"),
            }
            with (
                mock.patch.dict(os.environ, protected_paths),
                mock.patch("jarvis_v2.scripts.smoke_test_all.subprocess.Popen") as popen,
            ):
                process = mock.Mock(pid=12345, returncode=0)
                popen.return_value = process
                isolated_roots_seen: list[Path] = []

                def complete_isolated_module(*, timeout):
                    child = popen.call_args.kwargs.get("env") or {}
                    current_root = Path(child["JARVIS_DATA_DIR"]).parent
                    if not current_root.is_dir() or current_root.stat().st_mode & 0o077:
                        raise SystemExit(
                            f"aggregate smoke runner module root was not private: {current_root}"
                        )
                    nested_socket_path = (
                        Path(child["TMPDIR"])
                        / "jarvis-subagent-runner-socket-12345678"
                        / "runner.sock"
                    )
                    if len(os.fsencode(nested_socket_path)) > 103:
                        raise SystemExit(
                            "aggregate smoke runner left insufficient AF_UNIX path budget: "
                            f"{nested_socket_path}"
                        )
                    isolated_roots_seen.append(current_root)
                    return "ok", ""

                process.communicate.side_effect = complete_isolated_module
                result = _run_module("fake.module")
                second_result = _run_module("fake.module")
                if result.stdout != "ok":
                    raise SystemExit("aggregate smoke runner test did not return mocked result")
                if second_result.stdout != "ok":
                    raise SystemExit("aggregate smoke runner path-reuse test did not return mocked result")
                if process.communicate.call_args_list != [mock.call(timeout=7.0), mock.call(timeout=7.0)]:
                    raise SystemExit("aggregate smoke runner changed the module timeout during path-reuse proof")
                if len(set(isolated_roots_seen)) != 2:
                    raise SystemExit(f"aggregate smoke runner reused a module storage root: {isolated_roots_seen}")
                kwargs = popen.call_args.kwargs
                if kwargs.get("start_new_session") is not True:
                    raise SystemExit(f"aggregate smoke runner did not isolate the module session: {kwargs}")
                child_env = kwargs.get("env") or {}
                if "JARVIS_LIVE_GMAIL_SMOKE" in child_env:
                    raise SystemExit("aggregate smoke runner _run_module leaked live env")
                for key in sensitive_test_keys:
                    if key in child_env:
                        raise SystemExit(f"aggregate smoke runner _run_module leaked sensitive env {key}")
                for key, protected_path in protected_paths.items():
                    if key == "JARVIS_WATCHED_DIRS":
                        if child_env.get(key) != "":
                            raise SystemExit("aggregate smoke runner did not disable production watched directories")
                    elif child_env.get(key) == protected_path:
                        raise SystemExit(f"aggregate smoke runner inherited production storage path {key}")
                isolated_data = Path(child_env.get("JARVIS_DATA_DIR", ""))
                isolated_root = isolated_data.parent
                if child_env.get("JARVIS_STATUS_PORT") != "8766":
                    raise SystemExit(f"aggregate smoke runner isolated status port drifted: {child_env}")
                expected_isolated = {
                    "JARVIS_DATA_DIR": isolated_root / "data",
                    "JARVIS_DB_PATH": isolated_root / "data" / "jarvis.sqlite",
                    "JARVIS_OBSIDIAN_VAULT": isolated_root / "vault",
                    "OBSIDIAN_VAULT_PATH": isolated_root / "vault",
                    "JARVIS_STORAGE_FALLBACK_DIR": isolated_root / "fallback",
                    "JARVIS_TELEGRAM_STATE": isolated_root / "state" / "telegram-control.json",
                    "JARVIS_V3_IMESSAGE_STATE": isolated_root / "state" / "imessage-control.json",
                    "JARVIS_REMINDERS_FILE": isolated_root / "state" / "telegram-reminders.json",
                    "JARVIS_CACHE_DIR": isolated_root / "cache",
                    "HOME": isolated_root / "home",
                    "TMPDIR": isolated_root / "tmp",
                    "XDG_CACHE_HOME": isolated_root / "cache",
                    "XDG_CONFIG_HOME": isolated_root / "config",
                    "XDG_DATA_HOME": isolated_root / "share",
                }
                for key, expected_path in expected_isolated.items():
                    if Path(child_env.get(key, "")) != expected_path:
                        raise SystemExit(f"aggregate smoke runner isolated {key} drifted: {child_env}")
            for completed_root in isolated_roots_seen:
                if completed_root.exists():
                    raise SystemExit(
                        f"aggregate smoke runner left isolated module storage behind: {completed_root}"
                    )
            for protected_path in protected_paths.values():
                if protected_path and Path(protected_path).exists():
                    raise SystemExit(f"aggregate smoke runner touched configured production path: {protected_path}")

        os.environ.pop("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", None)
        with mock.patch("jarvis_v2.scripts.smoke_test_all.subprocess.Popen") as popen:
            process = mock.Mock(pid=12345, returncode=0)
            process.communicate.return_value = ("ok", "")
            popen.return_value = process
            _run_module(STATUS_SERVER_SMOKE_MODULE)
            process.communicate.assert_called_once_with(timeout=float(STATUS_SERVER_DEFAULT_MODULE_TIMEOUT_SECONDS))

        os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = "9"
        with mock.patch("jarvis_v2.scripts.smoke_test_all.subprocess.Popen") as popen:
            process = mock.Mock(pid=12345, returncode=0)
            process.communicate.return_value = ("ok", "")
            popen.return_value = process
            _run_module("fake.module", 11.0)
            process.communicate.assert_called_once_with(timeout=11.0)

        timeout_marker = subprocess.TimeoutExpired(
            [sys.executable, "-m", "fake.module"],
            1.0,
            output="partial-timeout-output",
            stderr="partial-timeout-error",
        )
        with (
            mock.patch("jarvis_v2.scripts.smoke_test_all.subprocess.Popen") as popen,
            mock.patch.object(
                smoke_test_all,
                "_terminate_process_group",
                side_effect=RuntimeError("process-group cleanup failed"),
            ),
        ):
            process = mock.Mock(pid=12345, returncode=None)
            process.communicate.side_effect = [
                timeout_marker,
                ("direct-child-output", "direct-child-error"),
            ]
            popen.return_value = process
            try:
                _run_module("fake.module", 1.0)
            except subprocess.TimeoutExpired as exc:
                if exc.timeout != 1.0:
                    raise SystemExit(f"aggregate cleanup changed the timeout: {exc.timeout!r}") from exc
                if exc.stdout != "direct-child-output" or exc.stderr != "direct-child-error":
                    raise SystemExit(f"aggregate direct-child fallback lost cleanup output: {exc!r}") from exc
            except RuntimeError as exc:
                raise SystemExit(f"aggregate cleanup failure masked TimeoutExpired: {exc}") from exc
            else:
                raise SystemExit("aggregate cleanup failure should preserve TimeoutExpired")
            process.kill.assert_called_once_with()
            if process.communicate.call_count != 2:
                raise SystemExit("aggregate timeout did not reap through the direct-child fallback")

        interrupt_marker = "original-keyboard-interrupt"
        with (
            mock.patch("jarvis_v2.scripts.smoke_test_all.subprocess.Popen") as popen,
            mock.patch.object(
                smoke_test_all,
                "_terminate_process_group",
                side_effect=RuntimeError("process-group cleanup failed"),
            ),
        ):
            process = mock.Mock(pid=12345, returncode=None)
            process.communicate.side_effect = [
                KeyboardInterrupt(interrupt_marker),
                RuntimeError("direct-child reap failed"),
            ]
            process.kill.side_effect = RuntimeError("direct-child kill failed")
            process.wait.side_effect = RuntimeError("direct-child wait failed")
            popen.return_value = process
            try:
                _run_module("fake.module", 1.0)
            except KeyboardInterrupt as exc:
                if str(exc) != interrupt_marker:
                    raise SystemExit(f"aggregate cleanup changed KeyboardInterrupt: {exc}") from exc
            except RuntimeError as exc:
                raise SystemExit(f"aggregate cleanup failure masked KeyboardInterrupt: {exc}") from exc
            else:
                raise SystemExit("aggregate cleanup failure should preserve KeyboardInterrupt")
            process.kill.assert_called_once_with()
            process.wait.assert_called_once_with(
                timeout=smoke_test_all.PROCESS_GROUP_KILL_GRACE_SECONDS
            )

        with (
            mock.patch.object(smoke_test_all.os, "name", "unsupported"),
            mock.patch("jarvis_v2.scripts.smoke_test_all.subprocess.Popen") as popen,
        ):
            try:
                _run_module("fake.module", 1.0)
            except RuntimeError as exc:
                if "process-group isolation is unavailable" not in str(exc):
                    raise SystemExit(f"aggregate unsupported-platform failure drifted: {exc}") from exc
            else:
                raise SystemExit("aggregate runner should fail closed without process-group isolation")
            if popen.called:
                raise SystemExit("aggregate runner spawned a module without process-group isolation")

        with TemporaryDirectory(prefix="jarvis-smoke-completed-group-settle-") as temp_dir:
            fixture_root = Path(temp_dir)
            fixture_module = "jarvis_smoke_completed_group_settle_fixture"
            (fixture_root / f"{fixture_module}.py").write_text(
                "print('completed-group-settle-proof', flush=True)\n",
                encoding="utf-8",
            )
            fixture_env = {
                "PYTHONPATH": os.pathsep.join(
                    filter(None, [str(fixture_root), os.environ.get("PYTHONPATH", "")])
                )
            }
            with (
                mock.patch.dict(os.environ, fixture_env),
                mock.patch.object(
                    smoke_test_all,
                    "_process_group_exists",
                    return_value=True,
                ),
                mock.patch.object(
                    smoke_test_all,
                    "_wait_for_process_group_exit",
                    return_value=True,
                ) as wait_for_exit,
                mock.patch.object(
                    smoke_test_all,
                    "_cleanup_real_process_group_best_effort",
                ) as cleanup,
            ):
                result = _run_module(fixture_module, 10.0)
            if result.returncode != 0 or "completed-group-settle-proof" not in result.stdout:
                raise SystemExit("aggregate runner lost a successful module while settling its PGID")
            wait_for_exit.assert_called_once()
            if wait_for_exit.call_args.args[1] != COMPLETED_PROCESS_GROUP_SETTLE_SECONDS:
                raise SystemExit("aggregate completed-process settle deadline drifted")
            if cleanup.called:
                raise SystemExit("aggregate runner terminated a naturally settling module group")

            with (
                mock.patch.dict(os.environ, fixture_env),
                mock.patch.object(
                    smoke_test_all,
                    "_process_group_exists",
                    return_value=True,
                ),
                mock.patch.object(
                    smoke_test_all,
                    "_wait_for_process_group_exit",
                    return_value=False,
                ) as wait_for_exit,
                mock.patch.object(
                    smoke_test_all,
                    "_cleanup_real_process_group_best_effort",
                ) as cleanup,
            ):
                result = _run_module(fixture_module, 10.0)
            if result.returncode == 0 or RESIDUAL_PROCESS_GROUP_MARKER not in result.stderr:
                raise SystemExit("aggregate runner accepted a persistent completed process group")
            wait_for_exit.assert_called_once()
            if wait_for_exit.call_args.args[1] != COMPLETED_PROCESS_GROUP_SETTLE_SECONDS:
                raise SystemExit("aggregate persistent-process settle deadline drifted")
            cleanup.assert_called_once()

        with TemporaryDirectory(prefix="jarvis-smoke-descendant-timeout-") as temp_dir:
            fixture_root = Path(temp_dir)
            fixture_module = "jarvis_smoke_descendant_timeout_fixture"
            descendant_pid_path = fixture_root / "descendant.pid"
            survivor_marker_path = fixture_root / "descendant-survived"
            child_env_proof_path = fixture_root / "descendant-env.json"
            symlink_target_dir = fixture_root / "symlink-target"
            symlink_target_dir.mkdir()
            symlink_target_sentinel = symlink_target_dir / "keep.txt"
            symlink_target_sentinel.write_text("keep", encoding="utf-8")
            protected_child_paths = {
                "HOME": str(fixture_root / "must-not-use-home"),
                "JARVIS_DATA_DIR": str(fixture_root / "must-not-use-data"),
                "JARVIS_TELEGRAM_STATE": str(fixture_root / "must-not-use-telegram-state.json"),
                "JARVIS_REMINDERS_FILE": str(fixture_root / "must-not-use-reminders.json"),
                "JARVIS_CACHE_DIR": str(fixture_root / "must-not-use-cache"),
            }
            child_source = (
                "import json, os, signal, time\n"
                "from pathlib import Path\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "Path(os.environ['JARVIS_SMOKE_DESCENDANT_PID_PATH']).write_text(str(os.getpid()), encoding='utf-8')\n"
                "proof_keys = ('HOME', 'TMPDIR', 'JARVIS_DATA_DIR', 'JARVIS_DB_PATH', 'JARVIS_OBSIDIAN_VAULT', 'JARVIS_TELEGRAM_STATE', 'JARVIS_REMINDERS_FILE', 'JARVIS_CACHE_DIR')\n"
                "Path(os.environ['JARVIS_SMOKE_CHILD_ENV_PROOF_PATH']).write_text(json.dumps({key: os.environ.get(key, '') for key in proof_keys}), encoding='utf-8')\n"
                "Path(os.environ['JARVIS_DATA_DIR']).joinpath('descendant-touched').write_text('isolated', encoding='utf-8')\n"
                "Path(os.environ['JARVIS_DATA_DIR']).joinpath('outside-link').symlink_to(os.environ['JARVIS_SMOKE_SYMLINK_TARGET_PATH'], target_is_directory=True)\n"
                "time.sleep(10)\n"
                "Path(os.environ['JARVIS_SMOKE_DESCENDANT_SURVIVOR_PATH']).write_text('survived', encoding='utf-8')\n"
                "time.sleep(60)\n"
            )
            (fixture_root / f"{fixture_module}.py").write_text(
                "import os, subprocess, sys, time\n"
                "from pathlib import Path\n"
                f"subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
                "pid_path = Path(os.environ['JARVIS_SMOKE_DESCENDANT_PID_PATH'])\n"
                "deadline = time.monotonic() + 5\n"
                "while not pid_path.exists() and time.monotonic() < deadline:\n"
                "    time.sleep(0.01)\n"
                "if not pid_path.exists():\n"
                "    raise SystemExit('descendant failed to start')\n"
                "print('descendant-ready', flush=True)\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            fixture_env = {
                "PYTHONPATH": os.pathsep.join(filter(None, [str(fixture_root), os.environ.get("PYTHONPATH", "")])),
                "JARVIS_SMOKE_DESCENDANT_PID_PATH": str(descendant_pid_path),
                "JARVIS_SMOKE_DESCENDANT_SURVIVOR_PATH": str(survivor_marker_path),
                "JARVIS_SMOKE_CHILD_ENV_PROOF_PATH": str(child_env_proof_path),
                "JARVIS_SMOKE_SYMLINK_TARGET_PATH": str(symlink_target_dir),
                **protected_child_paths,
            }
            with (
                mock.patch.dict(os.environ, fixture_env),
                mock.patch.object(smoke_test_all, "PROCESS_GROUP_TERMINATE_GRACE_SECONDS", 0.2),
                mock.patch.object(smoke_test_all, "PROCESS_GROUP_KILL_GRACE_SECONDS", 1.0),
            ):
                try:
                    _run_module(fixture_module, 0.75)
                except subprocess.TimeoutExpired as exc:
                    if "descendant-ready" not in _stream_text(exc.stdout):
                        raise SystemExit(f"aggregate timeout lost descendant fixture output: {exc.stdout!r}") from exc
                else:
                    raise SystemExit("aggregate descendant fixture should time out")
            if not descendant_pid_path.is_file():
                raise SystemExit("aggregate descendant fixture did not publish its pid")
            if not child_env_proof_path.is_file():
                raise SystemExit("aggregate descendant fixture did not publish inherited env proof")
            child_env_proof = json.loads(child_env_proof_path.read_text(encoding="utf-8"))
            for key, protected_path in protected_child_paths.items():
                if child_env_proof.get(key) == protected_path:
                    raise SystemExit(f"aggregate descendant inherited production path {key}")
            isolated_timeout_root = Path(child_env_proof["JARVIS_DATA_DIR"]).parent
            if isolated_timeout_root.exists():
                raise SystemExit(
                    f"aggregate timeout left isolated module storage behind: {isolated_timeout_root}"
                )
            if Path(child_env_proof["JARVIS_DB_PATH"]).parent != Path(child_env_proof["JARVIS_DATA_DIR"]):
                raise SystemExit(f"aggregate descendant DB escaped isolated data dir: {child_env_proof}")
            if Path(child_env_proof["JARVIS_OBSIDIAN_VAULT"]).parent != isolated_timeout_root:
                raise SystemExit(f"aggregate descendant vault escaped isolated root: {child_env_proof}")
            for protected_path in protected_child_paths.values():
                if Path(protected_path).exists():
                    raise SystemExit(f"aggregate descendant touched production path: {protected_path}")
            if symlink_target_sentinel.read_text(encoding="utf-8") != "keep":
                raise SystemExit("aggregate timeout cleanup followed an isolated-storage symlink")
            descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2.0
            while _process_exists(descendant_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            if _process_exists(descendant_pid):
                raise SystemExit(f"aggregate timeout left descendant process running: pid={descendant_pid}")
            if survivor_marker_path.exists():
                raise SystemExit("aggregate timeout descendant survived long enough to perform delayed work")

        with TemporaryDirectory(prefix="jarvis-smoke-success-descendant-") as temp_dir:
            fixture_root = Path(temp_dir)
            fixture_module = "jarvis_smoke_success_descendant_fixture"
            descendant_pid_path = fixture_root / "descendant.pid"
            survivor_marker_path = fixture_root / "descendant-survived"
            child_source = (
                "import os, signal, time\n"
                "from pathlib import Path\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "Path(os.environ['JARVIS_SMOKE_DESCENDANT_PID_PATH']).write_text(str(os.getpid()), encoding='utf-8')\n"
                "time.sleep(10)\n"
                "Path(os.environ['JARVIS_SMOKE_DESCENDANT_SURVIVOR_PATH']).write_text('survived', encoding='utf-8')\n"
                "time.sleep(60)\n"
            )
            (fixture_root / f"{fixture_module}.py").write_text(
                "import subprocess, sys, time\n"
                "from pathlib import Path\n"
                f"subprocess.Popen([sys.executable, '-c', {child_source!r}], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                "pid_path = Path(__import__('os').environ['JARVIS_SMOKE_DESCENDANT_PID_PATH'])\n"
                "deadline = time.monotonic() + 5\n"
                "while not pid_path.exists() and time.monotonic() < deadline:\n"
                "    time.sleep(0.01)\n"
                "if not pid_path.exists():\n"
                "    raise SystemExit('descendant failed to start')\n"
                "print('module-parent-succeeded', flush=True)\n",
                encoding="utf-8",
            )
            fixture_env = {
                "PYTHONPATH": os.pathsep.join(filter(None, [str(fixture_root), os.environ.get("PYTHONPATH", "")])),
                "JARVIS_SMOKE_DESCENDANT_PID_PATH": str(descendant_pid_path),
                "JARVIS_SMOKE_DESCENDANT_SURVIVOR_PATH": str(survivor_marker_path),
            }
            with (
                mock.patch.dict(os.environ, fixture_env),
                mock.patch.object(smoke_test_all, "PROCESS_GROUP_TERMINATE_GRACE_SECONDS", 0.2),
                mock.patch.object(smoke_test_all, "PROCESS_GROUP_KILL_GRACE_SECONDS", 1.0),
            ):
                result = _run_module(fixture_module, 10.0)
            if result.returncode == 0 or RESIDUAL_PROCESS_GROUP_MARKER not in result.stderr:
                raise SystemExit(
                    "aggregate runner falsely passed a module that left a closed-pipe descendant"
                )
            if not descendant_pid_path.is_file():
                raise SystemExit("aggregate success-descendant fixture did not publish its pid")
            descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2.0
            while _process_exists(descendant_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            if _process_exists(descendant_pid):
                raise SystemExit(
                    f"aggregate success path left descendant running: pid={descendant_pid}"
                )
            if survivor_marker_path.exists():
                raise SystemExit(
                    "aggregate success-path descendant survived long enough to perform delayed work"
                )

        with TemporaryDirectory(prefix="jarvis-smoke-output-limit-") as temp_dir:
            fixture_root = Path(temp_dir)
            fixture_module = "jarvis_smoke_output_limit_fixture"
            descendant_pid_path = fixture_root / "descendant.pid"
            survivor_marker_path = fixture_root / "descendant-survived"
            child_source = (
                "import os, signal, time\n"
                "from pathlib import Path\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "Path(os.environ['JARVIS_SMOKE_DESCENDANT_PID_PATH']).write_text(str(os.getpid()), encoding='utf-8')\n"
                "time.sleep(10)\n"
                "Path(os.environ['JARVIS_SMOKE_DESCENDANT_SURVIVOR_PATH']).write_text('survived', encoding='utf-8')\n"
                "time.sleep(60)\n"
            )
            (fixture_root / f"{fixture_module}.py").write_text(
                "import os, subprocess, sys, time\n"
                "from pathlib import Path\n"
                f"subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
                "pid_path = Path(os.environ['JARVIS_SMOKE_DESCENDANT_PID_PATH'])\n"
                "deadline = time.monotonic() + 5\n"
                "while not pid_path.exists() and time.monotonic() < deadline:\n"
                "    time.sleep(0.01)\n"
                "if not pid_path.exists():\n"
                "    raise SystemExit('descendant failed to start')\n"
                "print('noisy-ready', flush=True)\n"
                "chunk = b'x' * 8192\n"
                "while True:\n"
                "    os.write(1, chunk)\n"
                "    os.write(2, chunk)\n",
                encoding="utf-8",
            )
            fixture_env = {
                "PYTHONPATH": os.pathsep.join(filter(None, [str(fixture_root), os.environ.get("PYTHONPATH", "")])),
                "JARVIS_SMOKE_DESCENDANT_PID_PATH": str(descendant_pid_path),
                "JARVIS_SMOKE_DESCENDANT_SURVIVOR_PATH": str(survivor_marker_path),
            }
            real_popen = subprocess.Popen
            spawned: list[subprocess.Popen[bytes]] = []

            def recording_popen(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                spawned.append(process)
                return process

            started = time.monotonic()
            with (
                mock.patch.dict(os.environ, fixture_env),
                mock.patch.object(smoke_test_all, "PROCESS_GROUP_TERMINATE_GRACE_SECONDS", 0.2),
                mock.patch.object(smoke_test_all, "PROCESS_GROUP_KILL_GRACE_SECONDS", 1.0),
                mock.patch("jarvis_v2.scripts.smoke_test_all.subprocess.Popen", side_effect=recording_popen),
            ):
                result = _run_module(fixture_module, 10.0)
            elapsed = time.monotonic() - started
            if result.returncode == 0:
                raise SystemExit("aggregate noisy-output fixture should fail closed at the capture limit")
            if MODULE_OUTPUT_LIMIT_MARKER not in result.stderr:
                raise SystemExit(f"aggregate noisy-output fixture lost its limit marker: {result.stderr[-200:]!r}")
            combined_capture_bytes = len(result.stdout.encode("utf-8")) + len(result.stderr.encode("utf-8"))
            if combined_capture_bytes > MAX_MODULE_CAPTURE_BYTES:
                raise SystemExit(
                    "aggregate noisy-output capture exceeded its hard bound: "
                    f"{combined_capture_bytes} > {MAX_MODULE_CAPTURE_BYTES}"
                )
            if elapsed >= 5.0:
                raise SystemExit(f"aggregate noisy-output cap did not terminate promptly: {elapsed:.2f}s")
            if len(spawned) != 1 or spawned[0].returncode is None:
                raise SystemExit("aggregate noisy-output cap did not reap the direct smoke process")
            if not descendant_pid_path.is_file():
                raise SystemExit("aggregate noisy-output fixture did not publish its descendant pid")
            descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2.0
            while _process_exists(descendant_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            if _process_exists(descendant_pid):
                raise SystemExit(f"aggregate noisy-output cap left descendant running: pid={descendant_pid}")
            if survivor_marker_path.exists():
                raise SystemExit("aggregate noisy-output descendant survived long enough to perform delayed work")

        with TemporaryDirectory(prefix="jarvis-smoke-descendant-cancel-") as temp_dir:
            fixture_root = Path(temp_dir)
            fixture_module = "jarvis_smoke_descendant_cancel_fixture"
            descendant_pid_path = fixture_root / "descendant.pid"
            survivor_marker_path = fixture_root / "descendant-survived"
            child_source = (
                "import os, signal, time\n"
                "from pathlib import Path\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "Path(os.environ['JARVIS_SMOKE_DESCENDANT_PID_PATH']).write_text(str(os.getpid()), encoding='utf-8')\n"
                "time.sleep(10)\n"
                "Path(os.environ['JARVIS_SMOKE_DESCENDANT_SURVIVOR_PATH']).write_text('survived', encoding='utf-8')\n"
                "time.sleep(60)\n"
            )
            (fixture_root / f"{fixture_module}.py").write_text(
                "import os, subprocess, sys, time\n"
                "from pathlib import Path\n"
                f"subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
                "pid_path = Path(os.environ['JARVIS_SMOKE_DESCENDANT_PID_PATH'])\n"
                "deadline = time.monotonic() + 5\n"
                "while not pid_path.exists() and time.monotonic() < deadline:\n"
                "    time.sleep(0.01)\n"
                "if not pid_path.exists():\n"
                "    raise SystemExit('descendant failed to start')\n"
                "print('descendant-ready', flush=True)\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            fixture_env = {
                "PYTHONPATH": os.pathsep.join(filter(None, [str(fixture_root), os.environ.get("PYTHONPATH", "")])),
                "JARVIS_SMOKE_DESCENDANT_PID_PATH": str(descendant_pid_path),
                "JARVIS_SMOKE_DESCENDANT_SURVIVOR_PATH": str(survivor_marker_path),
            }
            real_popen = subprocess.Popen
            real_selector_factory = smoke_test_all.selectors.DefaultSelector
            interrupt_marker = "aggregate-cancellation-propagated"

            class InterruptingSelector:
                def __init__(self):
                    self.selector = real_selector_factory()
                    self.interrupted = False

                def register(self, *args, **kwargs):
                    return self.selector.register(*args, **kwargs)

                def unregister(self, *args, **kwargs):
                    return self.selector.unregister(*args, **kwargs)

                def get_map(self):
                    return self.selector.get_map()

                def select(self, timeout: float | None = None):
                    if not self.interrupted:
                        deadline = time.monotonic() + 5.0
                        while not descendant_pid_path.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        if not descendant_pid_path.exists():
                            raise RuntimeError("cancellation fixture descendant failed to start")
                        self.interrupted = True
                        raise KeyboardInterrupt(interrupt_marker)
                    return self.selector.select(timeout)

                def close(self) -> None:
                    self.selector.close()

            spawned: list[subprocess.Popen[bytes]] = []

            def recording_popen(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                spawned.append(process)
                return process

            with (
                mock.patch.dict(os.environ, fixture_env),
                mock.patch.object(smoke_test_all, "PROCESS_GROUP_TERMINATE_GRACE_SECONDS", 0.2),
                mock.patch.object(smoke_test_all, "PROCESS_GROUP_KILL_GRACE_SECONDS", 1.0),
                mock.patch("jarvis_v2.scripts.smoke_test_all.subprocess.Popen", side_effect=recording_popen),
                mock.patch.object(smoke_test_all.selectors, "DefaultSelector", InterruptingSelector),
            ):
                try:
                    _run_module(fixture_module, 30.0)
                except KeyboardInterrupt as exc:
                    if str(exc) != interrupt_marker:
                        raise SystemExit(f"aggregate cancellation changed during propagation: {exc}") from exc
                else:
                    raise SystemExit("aggregate cancellation should propagate after process-group cleanup")
            if len(spawned) != 1 or spawned[0].returncode is None:
                raise SystemExit("aggregate cancellation did not reap the direct smoke process")
            if not descendant_pid_path.is_file():
                raise SystemExit("aggregate cancellation fixture did not publish its descendant pid")
            descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2.0
            while _process_exists(descendant_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            if _process_exists(descendant_pid):
                raise SystemExit(f"aggregate cancellation left descendant process running: pid={descendant_pid}")
            if survivor_marker_path.exists():
                raise SystemExit("aggregate cancellation descendant survived long enough to perform delayed work")
    finally:
        if old_live is None:
            os.environ.pop("JARVIS_LIVE_GMAIL_SMOKE", None)
        else:
            os.environ["JARVIS_LIVE_GMAIL_SMOKE"] = old_live
        if old_live_other is None:
            os.environ.pop("JARVIS_LIVE_CALENDAR_SMOKE", None)
        else:
            os.environ["JARVIS_LIVE_CALENDAR_SMOKE"] = old_live_other
        if old_live_generic is None:
            os.environ.pop("JARVIS_LIVE_SMOKE", None)
        else:
            os.environ["JARVIS_LIVE_SMOKE"] = old_live_generic
        if old_safe is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = old_safe
        if old_timeout is None:
            os.environ.pop("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", None)
        else:
            os.environ["JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS"] = old_timeout
        if old_suite_lock is None:
            os.environ.pop(SUITE_LOCK_ENV_KEY, None)
        else:
            os.environ[SUITE_LOCK_ENV_KEY] = old_suite_lock
        if old_suite_lock_timeout is None:
            os.environ.pop(SUITE_LOCK_TIMEOUT_ENV_KEY, None)
        else:
            os.environ[SUITE_LOCK_TIMEOUT_ENV_KEY] = old_suite_lock_timeout
        if old_env_source is None:
            os.environ.pop("JARVIS_V3_ENV", None)
        else:
            os.environ["JARVIS_V3_ENV"] = old_env_source
        if old_daemon_enable is None:
            os.environ.pop("JARVIS_V3_ENABLE_DAEMONS", None)
        else:
            os.environ["JARVIS_V3_ENABLE_DAEMONS"] = old_daemon_enable
        if old_scheduler_enable is None:
            os.environ.pop("JARVIS_V3_ENABLE_SCHEDULER", None)
        else:
            os.environ["JARVIS_V3_ENABLE_SCHEDULER"] = old_scheduler_enable
        for key, value in old_sensitive.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("smoke runner smoke passed")


if __name__ == "__main__":
    main()
