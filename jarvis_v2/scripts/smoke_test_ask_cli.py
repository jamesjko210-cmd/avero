from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.receipts import runtime_result_receipt
from jarvis_v2.agent.types import Plan, PlannedAction, RuntimeResult, ToolResult
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.scripts import ask, bootstrap_memory
from jarvis_v2.scripts.startup import (
    V3_BOOTSTRAP_CHECK_COMMAND,
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DIAGNOSE_COMMAND,
    V3_ENV_COMMAND_PREFIX,
)


class HostileReceiptValue:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __bool__(self) -> bool:
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-receipt {self.marker}>"


def assert_bounded_runtime_init_failures() -> None:
    original_argv = sys.argv
    try:
        for exception_type in (PermissionError, RuntimeError):
            marker = f"{exception_type.__name__} migration lock /\x55sers/example/private/jarvis.sqlite"
            runtime_error = exception_type(marker)
            if exception_type is RuntimeError:
                runtime_error.__cause__ = PermissionError(marker)
            for json_output in (False, True):
                stdout = StringIO()
                stderr = StringIO()
                sys.argv = ["jarvis-ask", *(["--json"] if json_output else []), "what time is it"]
                with (
                    patch.object(ask, "JarvisRuntime", side_effect=runtime_error),
                    redirect_stdout(stdout),
                    redirect_stderr(stderr),
                ):
                    try:
                        ask.main()
                    except SystemExit as exc:
                        if exc.code != 3:
                            raise SystemExit(
                                f"ask {exception_type.__name__} startup failure should exit 3, got {exc.code}"
                            ) from exc
                    else:
                        raise SystemExit(f"ask {exception_type.__name__} startup failure should exit")

                output = stdout.getvalue()
                error_output = stderr.getvalue()
                combined = output + error_output
                if marker in combined or "Traceback" in combined or "/\x55sers/example/private" in combined:
                    raise SystemExit(f"ask {exception_type.__name__} startup failure leaked raw details: {combined}")
                if json_output:
                    if error_output:
                        raise SystemExit(f"ask JSON startup failure should not write stderr: {error_output}")
                    receipt = json.loads(output)
                    if receipt.get("route") != "startup" or receipt.get("exception_type") != exception_type.__name__:
                        raise SystemExit(f"ask JSON startup failure missed bounded receipt: {receipt}")
                else:
                    if output:
                        raise SystemExit(f"ask text startup failure should not write stdout: {output}")
                    if "Jarvis ask could not start." not in error_output or f"- error: {exception_type.__name__}:" not in error_output:
                        raise SystemExit(f"ask text startup failure missed bounded guidance: {error_output}")

        sys.argv = ["jarvis-ask", "what time is it"]
        unrelated = RuntimeError("unrelated runtime bug")
        with patch.object(ask, "JarvisRuntime", side_effect=unrelated):
            try:
                ask.main()
            except RuntimeError as exc:
                if exc is not unrelated:
                    raise SystemExit("ask changed an unrelated RuntimeError") from exc
            else:
                raise SystemExit("ask should not hide unrelated RuntimeError startup bugs")
    finally:
        sys.argv = original_argv


def main() -> None:
    assert_bounded_runtime_init_failures()
    malformed_trace_result = RuntimeResult(
        user_input="receipt fixture",
        plan=Plan(goal="receipt fixture", actions=[]),
        tool_results=[],
        verified=True,
        response="ok",
        metadata={
            "runtime_route": "tools",
            "runtime_trace": {
                "approval_queue_before": "bad-before",
                "approval_queue_after": True,
                "approval_queue_delta": float("inf"),
                "approved_reruns": "bad-reruns",
            },
        },
    )
    malformed_trace_receipt = runtime_result_receipt(malformed_trace_result, pending_approvals=9)
    if (
        malformed_trace_receipt.get("approval_queue_before") != 0
        or malformed_trace_receipt.get("approval_queue_after") != 9
        or malformed_trace_receipt.get("approval_queue_delta") != 0
        or malformed_trace_receipt.get("approved_reruns") != 0
    ):
        raise SystemExit(f"runtime_result_receipt should default malformed trace counters safely: {malformed_trace_receipt}")
    malformed_planner = malformed_trace_receipt.get("planner_metadata") or {}
    expected_empty_planner = {
        "planner_type": None,
        "model_planner_attempted": False,
        "model_planner_state": None,
        "model_planner_used": False,
        "model_planner_fell_back": False,
        "model_planner_fallback_reason": None,
        "model_planner_fallback_detail": None,
        "model_planner_recovery_hint": None,
        "model_planner_exception_type": None,
        "model_planner_model": None,
        "model_planner_timeout_seconds": None,
        "model_planner_action_count": None,
        "model_planner_ignored_unknown_tools": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    if malformed_planner != expected_empty_planner:
        raise SystemExit(f"runtime_result_receipt should expose a stable empty planner contract: {malformed_trace_receipt}")

    hostile_marker = "RECEIPT_HOSTILE_SHOULD_NOT_LEAK /\x55sers/example/private/receipt.sqlite"
    hostile = HostileReceiptValue(hostile_marker)
    hostile_receipt_result = RuntimeResult(
        user_input=hostile,
        plan=Plan(
            goal=hostile,
            actions=[
                PlannedAction(
                    "hostile_tool",
                    {
                        hostile: hostile,
                        "normal": hostile,
                    },
                    hostile,
                )
            ],
        ),
        tool_results=[
            ToolResult(
                "hostile_tool",
                False,
                hostile,
                {
                    hostile: hostile,
                    "approval_id": 8,
                },
            )
        ],
        verified=False,
        response=hostile,
        metadata={
            "runtime_route": "tools",
            "runtime_trace": {
                "planner_notes": hostile,
                "risk_levels": [hostile, "READ_ONLY"],
                "verification": hostile,
                "planner_metadata": {
                    "planner_type": hostile,
                    "model_planner_attempted": hostile,
                    "model_planner_ignored_unknown_tools": [hostile, "safe_tool"],
                },
                hostile: hostile,
            },
            "chat_response": {
                hostile: hostile,
                "diagnostic": hostile,
            },
        },
    )
    hostile_receipt = runtime_result_receipt(hostile_receipt_result, pending_approvals=0)
    hostile_text = json.dumps(hostile_receipt, sort_keys=True)
    if hostile_marker in hostile_text or "receipt.sqlite" in hostile_text or "/\x55sers/example/private" in hostile_text:
        raise SystemExit(f"runtime_result_receipt leaked hostile value marker: {hostile_receipt}")
    if hostile_receipt.get("risk_levels") != ["READ_ONLY"]:
        raise SystemExit(f"runtime_result_receipt should skip hostile risk levels and keep safe ones: {hostile_receipt}")
    hostile_action = hostile_receipt.get("plan", {}).get("actions", [{}])[0]
    if "<unreadable>" not in hostile_action.get("arg_keys", []):
        raise SystemExit(f"runtime_result_receipt should preserve unreadable arg key placeholder: {hostile_receipt}")
    hostile_tool_metadata = hostile_receipt.get("tool_results", [{}])[0].get("metadata", {})
    if hostile_tool_metadata.get("approval_id") != 8 or "<unreadable>" not in hostile_tool_metadata:
        raise SystemExit(f"runtime_result_receipt should preserve safe metadata and placeholder hostile keys: {hostile_receipt}")
    hostile_planner = hostile_receipt.get("planner_metadata") or {}
    if hostile_planner.get("model_planner_ignored_unknown_tools") != ["safe_tool"]:
        raise SystemExit(f"runtime_result_receipt should skip hostile planner ignored tools: {hostile_receipt}")

    noisy_receipt_result = RuntimeResult(
        user_input="receipt noisy fixture /\x55sers/example/Desktop/Claude code/private-message.txt",
        plan=Plan(
            goal="receipt noisy fixture /tmp/receipt-goal-secret.log",
            actions=[
                PlannedAction(
                    "read_thing",
                    {
                        "path": "/\x55sers/example/Desktop/Claude code/private.txt",
                        "notes": "line one\n/var/folders/zc/receipt-action-secret.log /tmp/receipt-action-secret.log " + ("long detail " * 60),
                    },
                    "exercise receipt arg previews /var/folders/zc/receipt-reason-secret.log",
                )
            ],
        ),
        tool_results=[
            ToolResult(
                "read_thing",
                False,
                "/private/var/folders/secret/output.txt\n/var/folders/zc/receipt-output-secret.log /tmp/receipt-output-secret.log " + ("raw output " * 60),
                {
                    "approval_id": 7,
                    "planned_args": {
                        "path": "/\x55sers/example/Desktop/Claude code/private.txt",
                        "tmp": "/tmp/receipt-metadata-secret.log",
                        "var": "/var/folders/zc/receipt-metadata-secret.log",
                    },
                },
            )
        ],
        verified=False,
        response="noisy fixture /\x55sers/example/Desktop/Claude code/private-response.txt /tmp/receipt-response-secret.log",
        metadata={
            "runtime_route": "tools",
            "runtime_trace": {
                "risk_levels": ["READ_ONLY"],
                "verification": "failed /\x55sers/example/Desktop/Claude code/private-verification.txt",
                "planned_actions": [
                    {
                        "tool_name": "read_thing",
                        "args": {"path": "/tmp/receipt-trace-arg-secret.log"},
                        "reason": "/var/folders/zc/receipt-trace-reason-secret.log",
                    }
                ],
                "stages": [{"detail": "/private/receipt-stage-secret.log"}],
            },
            "chat_response": {
                "source": "fallback",
                "diagnostic": "/var/folders/zc/receipt-chat-secret.log",
            },
        },
    )
    noisy_receipt = runtime_result_receipt(noisy_receipt_result, pending_approvals=0)
    noisy_action = noisy_receipt["plan"]["actions"][0]
    if noisy_action.get("arg_keys") != ["notes", "path"]:
        raise SystemExit(f"runtime_result_receipt should expose stable arg keys: {noisy_receipt}")
    noisy_text = json.dumps(noisy_receipt, sort_keys=True)
    if any(fragment in noisy_text for fragment in ["/\x55sers/", "Desktop/Claude code", "/private/var", "/var/folders/", "/tmp/"]):
        raise SystemExit(f"runtime_result_receipt should scrub local paths from exported JSON: {noisy_receipt}")
    if "\n" in noisy_action.get("args", {}).get("notes", ""):
        raise SystemExit(f"runtime_result_receipt should normalize multiline arg previews: {noisy_receipt}")
    noisy_tool = noisy_receipt["tool_results"][0]
    if noisy_tool.get("metadata", {}).get("approval_id") != 7:
        raise SystemExit(f"runtime_result_receipt should preserve numeric metadata: {noisy_receipt}")
    if len(noisy_tool.get("output", "")) > 240 or len(noisy_action.get("args", {}).get("notes", "")) > 240:
        raise SystemExit(f"runtime_result_receipt should bound exported output and args: {noisy_receipt}")

    with TemporaryDirectory(prefix="jarvis-ask-cli-") as temp:
        root = Path(temp)
        env = os.environ.copy()
        env.update(
            {
                "JARVIS_DATA_DIR": str(root),
                "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                "JARVIS_USE_MODEL_PLANNER": "0",
            }
        )

        help_run = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--help"],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(help_run.stdout)
        if help_run.stderr:
            print("stderr:")
            print(help_run.stderr)
        if help_run.returncode != 0:
            raise SystemExit(f"ask help CLI exited with {help_run.returncode}")
        if (
            'JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env"' not in help_run.stdout
            or "./launch_jarvis_v3_dashboard.py" not in help_run.stdout
            or "Jarvis V3 project folder" not in help_run.stdout
        ):
            raise SystemExit(f"ask help CLI missed root dashboard launcher: {help_run.stdout}")

        missing_env_path = root / "private-missing.env"
        missing_env = env.copy()
        missing_env["JARVIS_V3_ENV"] = str(missing_env_path)
        missing_env_failure = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--json", "what", "time", "is", "it"],
            text=True,
            capture_output=True,
            timeout=20,
            env=missing_env,
        )
        if missing_env_failure.returncode != 3:
            raise SystemExit(f"ask missing explicit env should exit 3, got {missing_env_failure.returncode}")
        if missing_env_failure.stderr:
            raise SystemExit(f"ask missing explicit env should not write stderr: {missing_env_failure.stderr}")
        missing_env_receipt = json.loads(missing_env_failure.stdout)
        missing_env_text = json.dumps(missing_env_receipt, sort_keys=True)
        if missing_env_receipt.get("route") != "startup" or missing_env_receipt.get("ok") is not False:
            raise SystemExit(f"ask missing explicit env missed startup receipt: {missing_env_receipt}")
        if missing_env_receipt.get("exception_type") != "FileNotFoundError":
            raise SystemExit(f"ask missing explicit env missed bounded exception type: {missing_env_receipt}")
        if missing_env_receipt.get("configured_paths_inspected") is not False:
            raise SystemExit(f"ask missing explicit env should not reload configured paths: {missing_env_receipt}")
        for expected in ("JARVIS_V3_ENV", "JARVIS_DATA_DIR", "JARVIS_DB_PATH"):
            if expected not in missing_env_text:
                raise SystemExit(f"ask missing explicit env missed recovery for {expected}: {missing_env_receipt}")
        for forbidden in (str(missing_env_path), str(root), "Traceback", "Explicit environment file does not exist"):
            if forbidden in missing_env_text:
                raise SystemExit(f"ask missing explicit env leaked raw failure detail: {missing_env_receipt}")

        diagnosis = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--diagnose", "--json", "run", "command", "python3", "--version"],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(diagnosis.stdout)
        if diagnosis.stderr:
            print("stderr:")
            print(diagnosis.stderr)
        if diagnosis.returncode != 0:
            raise SystemExit(f"ask diagnosis CLI exited with {diagnosis.returncode}")
        diagnosis_data = json.loads(diagnosis.stdout)
        if diagnosis_data.get("route") != "approval" or not diagnosis_data.get("approval_required"):
            raise SystemExit(f"ask diagnosis CLI missed approval route: {diagnosis_data}")
        if "approval readiness latest" not in diagnosis_data.get("recommended_next_commands", []):
            raise SystemExit(f"ask diagnosis CLI missed approval readiness recommendation: {diagnosis_data}")
        if (
            diagnosis_data.get("forecast_queue_before") != 0
            or diagnosis_data.get("forecast_queue_after_if_sent") != 1
            or diagnosis_data.get("forecast_queue_delta_if_sent") != 1
            or diagnosis_data.get("forecast_new_approvals") != 1
            or diagnosis_data.get("forecast_reused_approval_ids") != []
        ):
            raise SystemExit(f"ask diagnosis CLI missed approval queue forecast: {diagnosis_data}")
        diagnosis_forecast = diagnosis_data.get("approval_queue_forecast") or []
        if not diagnosis_forecast or diagnosis_forecast[0].get("tool_name") != "run_shell_command" or diagnosis_forecast[0].get("would_queue_new_approval") is not True:
            raise SystemExit(f"ask diagnosis CLI missed per-action approval forecast: {diagnosis_data}")

        path_diagnosis = subprocess.run(
            [
                sys.executable,
                "-m",
                "jarvis_v2.scripts.ask",
                "--diagnose",
                "--json",
                "run",
                "command",
                "/\x55sers/example/Desktop/Claude code/private-diagnosis.txt",
            ],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(path_diagnosis.stdout)
        if path_diagnosis.stderr:
            print("stderr:")
            print(path_diagnosis.stderr)
        if path_diagnosis.returncode != 0:
            raise SystemExit(f"ask path diagnosis CLI exited with {path_diagnosis.returncode}")
        path_diagnosis_data = json.loads(path_diagnosis.stdout)
        path_diagnosis_text = json.dumps(path_diagnosis_data, sort_keys=True)
        if any(fragment in path_diagnosis_text for fragment in ["/\x55sers/", "Desktop/Claude code", "/private/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"ask diagnosis CLI should scrub local paths from exported JSON: {path_diagnosis_data}")
        if path_diagnosis_data.get("planned_actions", [{}])[0].get("args", {}).get("command") != "<local-path>":
            raise SystemExit(f"ask diagnosis CLI missed redacted planned arg: {path_diagnosis_data}")
        if "execution governor: run command <local-path>" not in path_diagnosis_data.get("recommended_next_commands", []):
            raise SystemExit(f"ask diagnosis CLI missed redacted next command: {path_diagnosis_data}")

        store = MemoryStore(root / "jarvis.sqlite")
        store.init()
        store.log_tool_run(
            session_id="ask-smoke",
            tool_name="create_event",
            risk="HIGH_RISK",
            ok=False,
            approved=False,
            output="blocked high-risk event creation",
            approval_id=4,
            metadata={"approval_id": 4},
        )
        build_diagnosis = subprocess.run(
            [
                sys.executable,
                "-m",
                "jarvis_v2.scripts.ask",
                "--diagnose",
                "--json",
                "continue",
                "building",
                "Jarvis",
                "V2",
                "as",
                "an",
                "agent",
                "harness",
            ],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(build_diagnosis.stdout)
        if build_diagnosis.stderr:
            print("stderr:")
            print(build_diagnosis.stderr)
        if build_diagnosis.returncode != 0:
            raise SystemExit(f"ask build diagnosis CLI exited with {build_diagnosis.returncode}")
        build_diagnosis_data = json.loads(build_diagnosis.stdout)
        if build_diagnosis_data.get("route") != "auto_tool" or build_diagnosis_data.get("safe_to_execute_now") is not True:
            raise SystemExit(f"ask build diagnosis should allow the read-only build selector: {build_diagnosis_data}")
        if build_diagnosis_data.get("recovery_closure_blocks_auto_execution") is not True:
            raise SystemExit(f"ask build diagnosis missed outstanding recovery debt: {build_diagnosis_data}")
        if build_diagnosis_data.get("recovery_closure_blocks_current_command") is not False:
            raise SystemExit(f"ask build diagnosis should not apply recovery hold to read-only diagnostics: {build_diagnosis_data}")
        if build_diagnosis_data.get("recovery_closure_allows_read_only_diagnostics") is not True:
            raise SystemExit(f"ask build diagnosis missed read-only recovery exception metadata: {build_diagnosis_data}")
        if not build_diagnosis_data.get("planned_actions") or build_diagnosis_data["planned_actions"][0].get("tool_name") != "harness_build_slice":
            raise SystemExit(f"ask build diagnosis missed harness build slice action: {build_diagnosis_data}")

        delimited_diagnosis = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--diagnose", "--", "remember", "--json", "as", "literal", "text"],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(delimited_diagnosis.stdout)
        if delimited_diagnosis.stderr:
            print("stderr:")
            print(delimited_diagnosis.stderr)
        if delimited_diagnosis.returncode != 0:
            raise SystemExit(f"ask delimited diagnosis CLI exited with {delimited_diagnosis.returncode}")
        if delimited_diagnosis.stdout.lstrip().startswith("{"):
            raise SystemExit("ask CLI treated a post-delimiter --json token as a script flag.")
        if "remember --json as literal text" not in delimited_diagnosis.stdout:
            raise SystemExit(f"ask CLI lost post-delimiter message text: {delimited_diagnosis.stdout}")

        blocked = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--json", "run", "command", "python3", "--version"],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(blocked.stdout)
        if blocked.stderr:
            print("stderr:")
            print(blocked.stderr)
        if blocked.returncode != 0:
            raise SystemExit(f"ask CLI should not fail for a blocked approval-gated turn without --strict: {blocked.returncode}")
        blocked_data = json.loads(blocked.stdout)
        if blocked_data.get("ok") is not False or blocked_data.get("approval_required") is not True:
            raise SystemExit(f"ask JSON CLI missed blocked approval state: {blocked_data}")
        if blocked_data.get("pending_approvals") != 1 or blocked_data.get("queued_approval_ids") != [1]:
            raise SystemExit(f"ask JSON CLI missed queued approval ids: {blocked_data}")
        if blocked_data.get("queued_approval_count") != 1 or blocked_data.get("risk_levels") != ["HIGH_RISK"]:
            raise SystemExit(f"ask JSON CLI missed top-level risk/queue receipt fields: {blocked_data}")
        if (
            blocked_data.get("approval_queue_before") != 0
            or blocked_data.get("approval_queue_after") != 1
            or blocked_data.get("approval_queue_delta") != 1
            or blocked_data.get("new_approval_ids") != [1]
            or blocked_data.get("reused_approval_ids") != []
        ):
            raise SystemExit(f"ask JSON CLI missed first approval queue ledger: {blocked_data}")
        if blocked_data.get("ran_tool_handlers") or blocked_data.get("verification") != "run_shell_command: 'run_shell_command' is risk level HIGH_RISK; explicit approval required.":
            raise SystemExit(f"ask JSON CLI missed top-level verification/handler receipt fields: {blocked_data}")
        if blocked_data.get("runtime_trace", {}).get("route") != "tools":
            raise SystemExit(f"ask JSON CLI missed runtime trace route: {blocked_data}")
        if not any(action.get("tool_name") == "run_shell_command" for action in blocked_data.get("plan", {}).get("actions", [])):
            raise SystemExit(f"ask JSON CLI missed planned shell action: {blocked_data}")
        if not any(item.get("metadata", {}).get("approval_id") == 1 for item in blocked_data.get("tool_results", [])):
            raise SystemExit(f"ask JSON CLI missed tool-result approval id: {blocked_data}")
        if not any(item.get("tool") == "run_shell_command" and item.get("tool_name") == "run_shell_command" for item in blocked_data.get("tool_results", [])):
            raise SystemExit(f"ask JSON CLI missed shared receipt tool aliases: {blocked_data}")
        for expected in ["Safety receipt", "approval readiness 1", "approval packet 1", "approval chain proof 1"]:
            if expected not in blocked_data.get("response", ""):
                raise SystemExit(f"ask CLI missed blocked approval text: {expected}")

        duplicate_blocked = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--json", "run", "command", "python3", "--version"],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(duplicate_blocked.stdout)
        if duplicate_blocked.stderr:
            print("stderr:")
            print(duplicate_blocked.stderr)
        if duplicate_blocked.returncode != 0:
            raise SystemExit(f"ask duplicate approval CLI should not fail without --strict: {duplicate_blocked.returncode}")
        duplicate_data = json.loads(duplicate_blocked.stdout)
        if duplicate_data.get("pending_approvals") != 1 or duplicate_data.get("queued_approval_ids") != [1]:
            raise SystemExit(f"ask duplicate approval CLI missed reused approval id: {duplicate_data}")
        if (
            duplicate_data.get("approval_queue_before") != 1
            or duplicate_data.get("approval_queue_after") != 1
            or duplicate_data.get("approval_queue_delta") != 0
            or duplicate_data.get("new_approval_ids") != []
            or duplicate_data.get("reused_approval_ids") != [1]
        ):
            raise SystemExit(f"ask duplicate approval CLI missed reuse ledger: {duplicate_data}")
        if "already queued as approval #1" not in duplicate_data.get("response", ""):
            raise SystemExit(f"ask duplicate approval CLI missed reused receipt text: {duplicate_data}")

        duplicate_diagnosis = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--diagnose", "--json", "run", "command", "python3", "--version"],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(duplicate_diagnosis.stdout)
        if duplicate_diagnosis.stderr:
            print("stderr:")
            print(duplicate_diagnosis.stderr)
        if duplicate_diagnosis.returncode != 0:
            raise SystemExit(f"ask duplicate diagnosis CLI exited with {duplicate_diagnosis.returncode}")
        duplicate_diagnosis_data = json.loads(duplicate_diagnosis.stdout)
        if (
            duplicate_diagnosis_data.get("forecast_queue_before") != 1
            or duplicate_diagnosis_data.get("forecast_queue_after_if_sent") != 1
            or duplicate_diagnosis_data.get("forecast_queue_delta_if_sent") != 0
            or duplicate_diagnosis_data.get("forecast_new_approvals") != 0
            or duplicate_diagnosis_data.get("forecast_reused_approval_ids") != [1]
        ):
            raise SystemExit(f"ask duplicate diagnosis CLI missed approval reuse forecast: {duplicate_diagnosis_data}")
        duplicate_diagnosis_forecast = duplicate_diagnosis_data.get("approval_queue_forecast") or []
        if not duplicate_diagnosis_forecast or duplicate_diagnosis_forecast[0].get("existing_approval_id") != 1 or duplicate_diagnosis_forecast[0].get("would_reuse_pending_approval") is not True:
            raise SystemExit(f"ask duplicate diagnosis CLI missed per-action approval reuse: {duplicate_diagnosis_data}")

        structured = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--json", "what", "time", "is", "it"],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(structured.stdout)
        if structured.stderr:
            print("stderr:")
            print(structured.stderr)
        if structured.returncode != 0:
            raise SystemExit(f"ask JSON CLI exited with {structured.returncode}")
        structured_data = json.loads(structured.stdout)
        if not structured_data.get("ok") or structured_data.get("route") != "tools":
            raise SystemExit(f"ask JSON CLI missed verified tool route: {structured_data}")
        if "response" not in structured_data or "runtime_trace" not in structured_data:
            raise SystemExit("ask JSON CLI missed response/runtime trace.")
        if structured_data.get("approval_required") or structured_data.get("queued_approval_ids"):
            raise SystemExit(f"ask JSON CLI should not report approvals for safe time read: {structured_data}")
        if structured_data.get("queued_approval_count") != 0 or structured_data.get("risk_levels") != ["READ_ONLY"]:
            raise SystemExit(f"ask JSON CLI missed safe top-level risk/queue fields: {structured_data}")
        if structured_data.get("ran_tool_handlers") is not True or structured_data.get("verification") != "All planned actions completed.":
            raise SystemExit(f"ask JSON CLI missed safe top-level verification/handler fields: {structured_data}")
        if structured_data.get("pending_approvals") != 1:
            raise SystemExit(f"ask JSON CLI should preserve earlier pending approval count: {structured_data}")
        if set(structured_data.get("planner_metadata", {})) != set(expected_empty_planner):
            raise SystemExit(f"ask JSON CLI missed stable planner metadata keys: {structured_data}")
        if structured_data.get("planner_metadata", {}).get("model_planner_ignored_unknown_tools") != []:
            raise SystemExit(f"ask JSON CLI should expose empty ignored-tool diagnostics as a list: {structured_data}")
        if (
            structured_data.get("approval_queue_before") != 1
            or structured_data.get("approval_queue_after") != 1
            or structured_data.get("approval_queue_delta") != 0
            or structured_data.get("new_approval_ids") != []
            or structured_data.get("reused_approval_ids") != []
        ):
            raise SystemExit(f"ask JSON CLI missed safe approval-neutral ledger: {structured_data}")
        if not any(item.get("tool") == "current_time" for item in structured_data.get("tool_results", [])):
            raise SystemExit(f"ask JSON CLI missed current_time tool result: {structured_data}")

        stdin_structured = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--json"],
            input="what time is it\n",
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(stdin_structured.stdout)
        if stdin_structured.stderr:
            print("stderr:")
            print(stdin_structured.stderr)
        if stdin_structured.returncode != 0:
            raise SystemExit(f"ask JSON stdin CLI exited with {stdin_structured.returncode}")
        stdin_data = json.loads(stdin_structured.stdout)
        if stdin_data.get("message") != "what time is it" or not stdin_data.get("ok"):
            raise SystemExit(f"ask JSON stdin CLI missed message or verification: {stdin_data}")

        non_directory_home = root / "not-a-directory-home"
        non_directory_home.write_text("not a directory", encoding="utf-8")
        # The aggregate runner selects an owner-only runtime.env whose
        # synthetic primary storage paths are intentionally writable.  This
        # scenario is specifically proving the default-primary-storage
        # failure path, so select a separate, explicitly empty owner-only env
        # file instead of depending on the aggregate parent's selected file
        # (or falling through to the project's .env).
        fallback_selected_env = root / "fallback-runtime.env"
        fallback_selected_env.write_text("", encoding="utf-8")
        fallback_selected_env.chmod(0o600)
        fallback_env = env.copy()
        fallback_env.pop("JARVIS_DATA_DIR", None)
        fallback_env.pop("JARVIS_DB_PATH", None)
        fallback_env.pop("JARVIS_DISABLE_STORAGE_FALLBACK", None)
        fallback_env["JARVIS_V3_ENV"] = str(fallback_selected_env)
        fallback_env["HOME"] = str(non_directory_home)
        fallback_env["JARVIS_STORAGE_FALLBACK_DIR"] = str(root / "fallback-storage")
        default_storage_fallback = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "from jarvis_v2.agent.receipts import runtime_result_receipt; "
                    "from jarvis_v2.agent.runtime import JarvisRuntime; "
                    "runtime = JarvisRuntime(); "
                    "result = runtime.handle('save work block checkpoint'); "
                    "print(json.dumps(runtime_result_receipt(result, len(runtime.store.list_pending_approvals(limit=100))), sort_keys=True))"
                ),
            ],
            text=True,
            capture_output=True,
            timeout=20,
            env=fallback_env,
        )
        print(default_storage_fallback.stdout)
        if default_storage_fallback.stderr:
            print("stderr:")
            print(default_storage_fallback.stderr)
        if default_storage_fallback.returncode != 0:
            raise SystemExit(f"ask default storage fallback should exit 0, got {default_storage_fallback.returncode}")
        fallback_data = json.loads(default_storage_fallback.stdout)
        fallback_trace = fallback_data.get("runtime_trace", {})
        if not fallback_data.get("ok") or fallback_data.get("route") != "tools":
            raise SystemExit(f"ask default storage fallback should still run read-only tools: {fallback_data}")
        if fallback_trace.get("storage_fallback_active") is not True:
            raise SystemExit(f"ask default storage fallback missed trace marker: {fallback_data}")
        if fallback_trace.get("storage_fallback_reason") != "primary_storage_not_writable":
            raise SystemExit(f"ask default storage fallback leaked raw fallback reason: {fallback_data}")
        if not fallback_trace.get("storage_fallback_exception_type"):
            raise SystemExit(f"ask default storage fallback missed bounded exception type: {fallback_data}")
        fallback_trace_text = json.dumps(fallback_trace, sort_keys=True)
        for forbidden in ["attempt to write", "readonly database", "OperationalError:"]:
            if forbidden in fallback_trace_text:
                raise SystemExit(f"ask default storage fallback trace leaked raw storage exception text: {fallback_data}")
        if str(root / "fallback-storage" / "jarvis.sqlite") != fallback_trace.get("storage_fallback_db_path"):
            raise SystemExit(f"ask default storage fallback used wrong DB path: {fallback_data}")
        expected_fallback_vault = root / "fallback-storage" / "Vault"
        if str(expected_fallback_vault) != fallback_trace.get("storage_fallback_vault_path"):
            raise SystemExit(f"ask default storage fallback used wrong vault path: {fallback_data}")
        if fallback_trace.get("storage_fallback_db_path_display") != "workspace-local fallback database":
            raise SystemExit(f"ask default storage fallback missed safe DB display label: {fallback_data}")
        if fallback_trace.get("storage_fallback_vault_path_display") != "workspace-local fallback notes":
            raise SystemExit(f"ask default storage fallback missed safe vault display label: {fallback_data}")
        if not (expected_fallback_vault / "Jarvis" / "Reflections").exists():
            raise SystemExit(f"ask default storage fallback did not initialize the local fallback vault: {fallback_data}")
        if not list((expected_fallback_vault / "Jarvis" / "Reflections").glob("* Work Block Checkpoint.md")):
            raise SystemExit(f"ask default storage fallback did not save checkpoint into the fallback vault: {fallback_data}")
        if not fallback_trace.get("ran_tool_handlers"):
            raise SystemExit(f"ask default storage fallback did not run the read-only handler: {fallback_data}")
        if "Storage fallback:" not in fallback_data.get("response", ""):
            raise SystemExit(f"ask default storage fallback missed user-visible note: {fallback_data}")
        if "workspace-local fallback notes" not in fallback_data.get("response", ""):
            raise SystemExit(f"ask default storage fallback missed safe fallback notes label: {fallback_data}")
        fallback_response = fallback_data.get("response", "")
        for forbidden_path in [str(root / "fallback-storage" / "jarvis.sqlite"), str(expected_fallback_vault)]:
            if forbidden_path in fallback_response:
                raise SystemExit(f"ask default storage fallback response leaked raw path {forbidden_path}: {fallback_data}")

        blank_fallback_cwd = root / "blank-fallback-cwd"
        blank_fallback_cwd.mkdir()
        blank_fallback_env = fallback_env.copy()
        blank_fallback_env["JARVIS_STORAGE_FALLBACK_DIR"] = "   "
        blank_fallback_env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + blank_fallback_env.get("PYTHONPATH", "")
        blank_storage_fallback = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "from jarvis_v2.agent.receipts import runtime_result_receipt; "
                    "from jarvis_v2.agent.runtime import JarvisRuntime; "
                    "runtime = JarvisRuntime(); "
                    "result = runtime.handle('save work block checkpoint'); "
                    "print(json.dumps(runtime_result_receipt(result, len(runtime.store.list_pending_approvals(limit=100))), sort_keys=True))"
                ),
            ],
            text=True,
            capture_output=True,
            timeout=20,
            cwd=blank_fallback_cwd,
            env=blank_fallback_env,
        )
        print(blank_storage_fallback.stdout)
        if blank_storage_fallback.stderr:
            print("stderr:")
            print(blank_storage_fallback.stderr)
        if blank_storage_fallback.returncode != 0:
            raise SystemExit(f"ask blank storage fallback should exit 0, got {blank_storage_fallback.returncode}")
        blank_fallback_data = json.loads(blank_storage_fallback.stdout)
        blank_fallback_trace = blank_fallback_data.get("runtime_trace", {})
        expected_blank_fallback = blank_fallback_cwd / ".jarvis_v3_runtime"
        if blank_fallback_trace.get("storage_fallback_active") is not True:
            raise SystemExit(f"ask blank storage fallback missed trace marker: {blank_fallback_data}")
        if blank_fallback_trace.get("storage_fallback_reason") != "primary_storage_not_writable":
            raise SystemExit(f"ask blank storage fallback leaked raw fallback reason: {blank_fallback_data}")
        if not blank_fallback_trace.get("storage_fallback_exception_type"):
            raise SystemExit(f"ask blank storage fallback missed bounded exception type: {blank_fallback_data}")
        if set(blank_fallback_data.get("planner_metadata", {})) != set(expected_empty_planner):
            raise SystemExit(f"ask blank storage fallback missed stable planner metadata keys: {blank_fallback_data}")
        if blank_fallback_data.get("planner_metadata", {}).get("model_planner_used") is not False:
            raise SystemExit(f"ask blank storage fallback planner metadata should report no model planner use: {blank_fallback_data}")
        actual_blank_db = Path(str(blank_fallback_trace.get("storage_fallback_db_path") or "")).resolve()
        actual_blank_vault = Path(str(blank_fallback_trace.get("storage_fallback_vault_path") or "")).resolve()
        if (expected_blank_fallback / "jarvis.sqlite").resolve() != actual_blank_db:
            raise SystemExit(f"ask blank storage fallback used wrong DB path: {blank_fallback_data}")
        if (expected_blank_fallback / "Vault").resolve() != actual_blank_vault:
            raise SystemExit(f"ask blank storage fallback used wrong vault path: {blank_fallback_data}")
        if blank_fallback_trace.get("storage_fallback_db_path_display") != "workspace-local fallback database":
            raise SystemExit(f"ask blank storage fallback missed safe DB display label: {blank_fallback_data}")
        if blank_fallback_trace.get("storage_fallback_vault_path_display") != "workspace-local fallback notes":
            raise SystemExit(f"ask blank storage fallback missed safe vault display label: {blank_fallback_data}")
        blank_response = blank_fallback_data.get("response", "")
        for forbidden_path in [str(expected_blank_fallback / "jarvis.sqlite"), str(expected_blank_fallback / "Vault")]:
            if forbidden_path in blank_response:
                raise SystemExit(f"ask blank storage fallback response leaked raw path {forbidden_path}: {blank_fallback_data}")

        blank_primary_env = fallback_env.copy()
        blank_primary_env["JARVIS_DATA_DIR"] = "   "
        blank_primary_env["JARVIS_DB_PATH"] = "   "
        blank_primary_env["JARVIS_STORAGE_FALLBACK_DIR"] = str(root / "blank-primary-fallback")
        blank_primary_fallback = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "from jarvis_v2.agent.receipts import runtime_result_receipt; "
                    "from jarvis_v2.agent.runtime import JarvisRuntime; "
                    "runtime = JarvisRuntime(); "
                    "result = runtime.handle('save work block checkpoint'); "
                    "print(json.dumps(runtime_result_receipt(result, len(runtime.store.list_pending_approvals(limit=100))), sort_keys=True))"
                ),
            ],
            text=True,
            capture_output=True,
            timeout=20,
            env=blank_primary_env,
        )
        print(blank_primary_fallback.stdout)
        if blank_primary_fallback.stderr:
            print("stderr:")
            print(blank_primary_fallback.stderr)
        if blank_primary_fallback.returncode != 0:
            raise SystemExit(f"ask blank primary storage env fallback should exit 0, got {blank_primary_fallback.returncode}")
        blank_primary_data = json.loads(blank_primary_fallback.stdout)
        blank_primary_trace = blank_primary_data.get("runtime_trace", {})
        if blank_primary_trace.get("storage_fallback_active") is not True:
            raise SystemExit(f"blank primary storage env should not block fallback: {blank_primary_data}")
        if blank_primary_trace.get("storage_fallback_reason") != "primary_storage_not_writable":
            raise SystemExit(f"blank primary storage env fallback leaked raw reason: {blank_primary_data}")
        if not blank_primary_trace.get("storage_fallback_exception_type"):
            raise SystemExit(f"blank primary storage env fallback missed bounded exception type: {blank_primary_data}")
        if str(root / "blank-primary-fallback" / "jarvis.sqlite") != blank_primary_trace.get("storage_fallback_db_path"):
            raise SystemExit(f"blank primary storage env fallback used wrong DB path: {blank_primary_data}")
        if blank_primary_trace.get("storage_fallback_db_path_display") != "workspace-local fallback database":
            raise SystemExit(f"blank primary storage env fallback missed safe DB display label: {blank_primary_data}")
        if blank_primary_trace.get("storage_fallback_vault_path_display") != "workspace-local fallback notes":
            raise SystemExit(f"blank primary storage env fallback missed safe vault display label: {blank_primary_data}")
        blank_primary_response = blank_primary_data.get("response", "")
        for forbidden_path in [str(root / "blank-primary-fallback" / "jarvis.sqlite"), str(root / "blank-primary-fallback" / "Vault")]:
            if forbidden_path in blank_primary_response:
                raise SystemExit(f"blank primary storage env fallback response leaked raw path {forbidden_path}: {blank_primary_data}")

        fallback_recovery = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "from jarvis_v2.agent.receipts import runtime_result_receipt; "
                    "from jarvis_v2.agent.runtime import JarvisRuntime; "
                    "runtime = JarvisRuntime(); "
                    "result = runtime.handle('checkpoint recovery: fallback checkpoint'); "
                    "print(json.dumps(runtime_result_receipt(result, len(runtime.store.list_pending_approvals(limit=100))), sort_keys=True))"
                ),
            ],
            text=True,
            capture_output=True,
            timeout=20,
            env=fallback_env,
        )
        print(fallback_recovery.stdout)
        if fallback_recovery.stderr:
            print("stderr:")
            print(fallback_recovery.stderr)
        if fallback_recovery.returncode != 0:
            raise SystemExit(f"ask default storage fallback recovery should exit 0, got {fallback_recovery.returncode}")
        fallback_recovery_data = json.loads(fallback_recovery.stdout)
        recovery_results = fallback_recovery_data.get("tool_results") or []
        recovery_metadata = recovery_results[0].get("metadata", {}) if recovery_results else {}
        if recovery_metadata.get("checkpoint_found") is not True:
            raise SystemExit(f"ask default storage fallback recovery did not find saved checkpoint: {fallback_recovery_data}")
        if recovery_metadata.get("checkpoint_freshness") != "fresh":
            raise SystemExit(f"ask default storage fallback recovery did not report fresh checkpoint: {fallback_recovery_data}")

        existing_db_path = root / "jarvis.sqlite"
        existing_db_path.chmod(0o400)
        try:
            readonly_existing = subprocess.run(
                [sys.executable, "-m", "jarvis_v2.scripts.ask", "--json", "what", "time", "is", "it"],
                text=True,
                capture_output=True,
                timeout=20,
                env=env,
            )
        finally:
            existing_db_path.chmod(0o600)
            for sidecar_suffix in ("-shm", "-wal"):
                sidecar_path = Path(f"{existing_db_path}{sidecar_suffix}")
                if sidecar_path.exists():
                    sidecar_path.chmod(0o600)
        print(readonly_existing.stdout)
        if readonly_existing.stderr:
            print("stderr:")
            print(readonly_existing.stderr)
        if readonly_existing.returncode != 0:
            raise SystemExit(f"ask existing read-only DB should return a safe receipt, got {readonly_existing.returncode}")
        readonly_existing_data = json.loads(readonly_existing.stdout)
        if readonly_existing_data.get("route") != "storage_degraded" or readonly_existing_data.get("ok") is not False:
            raise SystemExit(f"ask existing read-only DB missed storage-degraded receipt: {readonly_existing_data}")
        readonly_trace = readonly_existing_data.get("runtime_trace", {})
        if readonly_trace.get("ran_tool_handlers") is not False or readonly_trace.get("storage_degraded") is not True:
            raise SystemExit(f"ask existing read-only DB did not prove execution was held: {readonly_existing_data}")
        if readonly_trace.get("tool_results") != [] or readonly_existing_data.get("tool_results") != []:
            raise SystemExit(f"ask existing read-only DB should not run tools without audit storage: {readonly_existing_data}")
        readonly_response = str(readonly_existing_data.get("response") or "")
        if "Retry the command after storage is writable" not in readonly_response:
            raise SystemExit(f"ask existing read-only DB missed recovery guidance: {readonly_existing_data}")
        for expected in [
            "JARVIS_DATA_DIR",
            "JARVIS_DB_PATH",
            "python3 -m jarvis_v2.scripts.bootstrap_memory --check",
            "then retry",
        ]:
            if expected not in readonly_response:
                raise SystemExit(f"ask existing read-only DB warning missed actionable guidance {expected}: {readonly_existing_data}")
        readonly_storage_text = json.dumps(
            {
                "response": readonly_existing_data.get("response", ""),
                "trace": readonly_trace,
            },
            sort_keys=True,
        )
        for forbidden in ["attempt to write", "readonly database", "OperationalError:"]:
            if forbidden in readonly_storage_text:
                raise SystemExit(f"ask existing read-only DB leaked raw storage exception text: {readonly_existing_data}")

        incomplete_dir = root / "incomplete-readonly-db"
        incomplete_dir.mkdir()
        incomplete_db_path = incomplete_dir / "jarvis.sqlite"
        with sqlite3.connect(incomplete_db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
                    metadata TEXT, created_at TEXT
                );
                CREATE TABLE history_policy_epochs (
                    epoch_id TEXT PRIMARY KEY, policy_fingerprint TEXT, provider TEXT,
                    model_identifier TEXT, destination_class TEXT, session_generation INTEGER,
                    explicit_consent_satisfied INTEGER, created_at TEXT
                );
                CREATE TABLE active_history_policy_epoch (
                    singleton_id INTEGER PRIMARY KEY, active_epoch_id TEXT, activated_at TEXT
                );
                CREATE TABLE pending_approvals (id INTEGER PRIMARY KEY);
                CREATE TABLE store_instance_state (singleton_id INTEGER PRIMARY KEY);
                CREATE TABLE tool_runs (id INTEGER PRIMARY KEY);
                INSERT INTO history_policy_epochs VALUES (
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                    'local', 'unconfigured', 'local-only', 0, 0, '2026-01-01T00:00:00Z'
                );
                INSERT INTO active_history_policy_epoch VALUES (
                    1,
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    '2026-01-01T00:00:00Z'
                );
                """
            )
        incomplete_db_path.chmod(0o400)
        incomplete_dir.chmod(0o500)
        try:
            incomplete_env = env.copy()
            incomplete_env["JARVIS_DB_PATH"] = str(incomplete_db_path)
            incomplete_startup = subprocess.run(
                [sys.executable, "-m", "jarvis_v2.scripts.ask", "--json", "what", "time", "is", "it"],
                text=True,
                capture_output=True,
                timeout=20,
                env=incomplete_env,
            )
        finally:
            incomplete_dir.chmod(0o700)
            incomplete_db_path.chmod(0o600)
        print(incomplete_startup.stdout)
        if incomplete_startup.stderr:
            print("stderr:")
            print(incomplete_startup.stderr)
        if incomplete_startup.returncode != 3:
            raise SystemExit(
                "ask incomplete read-only DB must fail startup instead of entering degraded mode: "
                f"{incomplete_startup.returncode}"
            )
        incomplete_data = json.loads(incomplete_startup.stdout)
        if incomplete_data.get("route") != "startup" or incomplete_data.get("ok") is not False:
            raise SystemExit(
                f"ask incomplete read-only DB returned a compatibility receipt: {incomplete_data}"
            )

        readonly_dir = root / "readonly-db"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o500)
        try:
            readonly_env = env.copy()
            readonly_env["JARVIS_DB_PATH"] = str(readonly_dir / "jarvis.sqlite")
            startup_failure = subprocess.run(
                [sys.executable, "-m", "jarvis_v2.scripts.ask", "--json", "what", "time", "is", "it"],
                text=True,
                capture_output=True,
                timeout=20,
                env=readonly_env,
            )
        finally:
            readonly_dir.chmod(0o700)
        print(startup_failure.stdout)
        if startup_failure.stderr:
            print("stderr:")
            print(startup_failure.stderr)
        if startup_failure.returncode != 3:
            raise SystemExit(f"ask JSON startup failure should exit 3: {startup_failure.returncode}")
        startup_data = json.loads(startup_failure.stdout)
        if startup_data.get("route") != "startup" or startup_data.get("ok") is not False:
            raise SystemExit(f"ask JSON startup failure missed startup receipt: {startup_data}")
        if startup_data.get("safety_boundary", {}).get("started_runtime") is not False:
            raise SystemExit(f"ask JSON startup failure should prove runtime did not start: {startup_data}")
        if startup_data.get("db_path") != "<local-path>":
            raise SystemExit(f"ask JSON startup failure should redact database path: {startup_data}")
        if startup_data.get("exception_type") != startup_data.get("error"):
            raise SystemExit(f"ask JSON startup failure missed exception type diagnostic: {startup_data}")
        startup_message = str(startup_data.get("message") or "")
        for expected in [
            "JARVIS_DATA_DIR",
            "JARVIS_DB_PATH",
            V3_BOOTSTRAP_CHECK_COMMAND,
            "retry startup",
        ]:
            if expected not in startup_message:
                raise SystemExit(f"ask JSON startup failure missed actionable guidance {expected}: {startup_data}")
        startup_text = json.dumps(startup_data, sort_keys=True)
        if "Permission denied" in startup_message or str(readonly_dir) in startup_text or "/var/folders/" in startup_text or "/tmp/" in startup_text:
            raise SystemExit(f"ask JSON startup failure message should not include raw exception text: {startup_data}")
        if "Set JARVIS_DATA_DIR" not in " ".join(startup_data.get("safe_recovery", [])):
            raise SystemExit(f"ask JSON startup failure missed recovery hint: {startup_data}")
        recovery_text = " ".join(startup_data.get("safe_recovery", []))
        for expected in [
            V3_BOOTSTRAP_CHECK_COMMAND,
            "only after the check reports configured storage ready",
            V3_DIAGNOSE_COMMAND,
            V3_DASHBOARD_COMMAND,
            V3_DASHBOARD_INFO_COMMAND,
        ]:
            if expected not in recovery_text:
                raise SystemExit(f"ask JSON startup failure missed dashboard recovery hint {expected}: {startup_data}")
        if recovery_text.count(V3_ENV_COMMAND_PREFIX) != 5:
            raise SystemExit(f"ask JSON startup recovery lost V3 environment custody: {startup_data}")
        for forbidden in (
            "`python3 -m jarvis_v2",
            "`python3 launch_jarvis_v3",
            "`./launch_jarvis_v3",
        ):
            if forbidden in recovery_text:
                raise SystemExit(
                    f"ask JSON startup recovery exposed bare command {forbidden!r}: {startup_data}"
                )

        bootstrap_stdout = StringIO()
        bootstrap_stderr = StringIO()
        bootstrap_marker = "bootstrap permission failure /\x55sers/example/private/jarvis.sqlite"
        bootstrap_env = root / "bootstrap-empty.env"
        bootstrap_env.write_text("", encoding="utf-8")
        bootstrap_env.chmod(0o600)
        with (
            patch.dict(
                os.environ,
                {
                    "JARVIS_V3_ENV": str(bootstrap_env),
                    "JARVIS_DATA_DIR": str(root / "bootstrap-data"),
                    "JARVIS_DB_PATH": str(root / "bootstrap-data" / "jarvis.sqlite"),
                    "JARVIS_OBSIDIAN_VAULT": str(root / "bootstrap-vault"),
                },
            ),
            patch.object(MemoryStore, "init", side_effect=PermissionError(bootstrap_marker)),
            redirect_stdout(bootstrap_stdout),
            redirect_stderr(bootstrap_stderr),
        ):
            try:
                bootstrap_memory.main([])
            except SystemExit as exc:
                if exc.code != 3:
                    raise SystemExit(f"bootstrap startup failure should exit 3: {exc.code}") from exc
            else:
                raise SystemExit("bootstrap startup failure should exit")
        if bootstrap_stdout.getvalue():
            raise SystemExit(f"bootstrap startup failure should not write stdout: {bootstrap_stdout.getvalue()}")
        bootstrap_error = bootstrap_stderr.getvalue()
        if "Jarvis bootstrap could not start." not in bootstrap_error:
            raise SystemExit(f"bootstrap startup failure missed readable error: {bootstrap_error}")
        if "- database: <local-path>" not in bootstrap_error:
            raise SystemExit(f"bootstrap startup failure missed redacted database path: {bootstrap_error}")
        if bootstrap_marker in bootstrap_error or "Traceback" in bootstrap_error or "/\x55sers/example/private" in bootstrap_error:
            raise SystemExit(f"bootstrap startup failure leaked raw storage exception text: {bootstrap_error}")

        if not any(item.get("tool") == "current_time" for item in stdin_data.get("tool_results", [])):
            raise SystemExit(f"ask JSON stdin CLI missed current_time tool result: {stdin_data}")

        strict_blocked = subprocess.run(
            [sys.executable, "-m", "jarvis_v2.scripts.ask", "--strict", "run", "command", "python3", "--version"],
            text=True,
            capture_output=True,
            timeout=20,
            env=env,
        )
        print(strict_blocked.stdout)
        if strict_blocked.returncode != 2:
            raise SystemExit(f"ask strict CLI should exit 2 for blocked unverified turns, got {strict_blocked.returncode}")


if __name__ == "__main__":
    main()
