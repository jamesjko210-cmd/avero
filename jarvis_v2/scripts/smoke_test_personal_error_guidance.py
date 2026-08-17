from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import personal as personal_tools


PRIVATE_MARKERS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")
MISSING_ACTION_TOOLS = (
    "integration_action_preview",
    "integration_scope_packet",
    "integration_dry_run_contract",
    "integration_runbook",
    "integration_promotion_gate",
    "integration_implementation_spec",
    "integration_preflight_contract",
    "integration_enablement_gate",
    "integration_rehearsal_receipt",
    "integration_metadata_preview",
    "integration_proof_bundle",
    "integration_implementation_review",
    "integration_route_lock",
    "integration_adapter_probe",
)


def _assert_public(result: Any, label: str) -> None:
    public = f"{result.output}\n{result.metadata}"
    if any(marker in public for marker in PRIVATE_MARKERS):
        raise SystemExit(f"{label} leaked a private local path: {public}")


def _assert_guided_preflight_failure(
    result: Any,
    *,
    label: str,
    reason: str,
    retry_safe: bool,
) -> None:
    if result.ok or result.metadata.get("reason") != reason:
        raise SystemExit(f"{label} did not fail before execution: {result}")
    declaration = result.metadata.get("recovery_guidance")
    if not isinstance(declaration, dict) or declaration.get("version") != 1:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    action = declaration.get("action")
    if not isinstance(action, str) or action not in result.output:
        raise SystemExit(f"{label} recovery action was not visible: {result}")
    if declaration.get("commands") != []:
        raise SystemExit(f"{label} unexpectedly exposed an execution command: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": retry_safe,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "executes_tools": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "controls_computer": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "approval_granted": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} truth field {key} drifted: {result.metadata}")
    _assert_public(result, label)


def _assert_unknown_open(result: Any, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded")
    for phrase in (
        "may already be open",
        "check first",
        "do not automatically retry",
        "setup check",
        "then retry through normal policy",
    ):
        if phrase not in result.output:
            raise SystemExit(f"{label} omitted {phrase!r}: {result.output}")
    declaration = result.metadata.get("recovery_guidance")
    if not isinstance(declaration, dict) or declaration.get("version") != 1:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    if declaration.get("action") != result.output or declaration.get("commands") != ["setup check"]:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": False,
        "outcome_unknown": True,
        "execution_outcome_unknown": True,
        "side_effect_possible": True,
        "retry_safe": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "executes_tools": True,
        "executes_side_effect": True,
        "external_side_effect": False,
        "controls_computer": True,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} truth field {key} drifted: {result.metadata}")
    _assert_public(result, label)


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-personal-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))

        for tool_name in MISSING_ACTION_TOOLS:
            result = runtime.registry.get(tool_name).handler({})
            _assert_guided_preflight_failure(
                result,
                label=f"{tool_name} missing action",
                reason="missing_action",
                retry_safe=True,
            )

        original_run = personal_tools.subprocess.run
        try:
            def must_not_run(*_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError("rejected reminder input must not launch AppleScript")

            personal_tools.subprocess.run = must_not_run  # type: ignore[assignment]
            missing_title = runtime.registry.get("create_reminder").handler({})
            path_title = runtime.registry.get("create_reminder").handler(
                {"title": "/\x55sers/owner/private/reminder"}
            )
        finally:
            personal_tools.subprocess.run = original_run  # type: ignore[assignment]

        _assert_guided_preflight_failure(
            missing_title,
            label="reminder missing title",
            reason="missing_title",
            retry_safe=False,
        )
        _assert_guided_preflight_failure(
            path_title,
            label="reminder path title",
            reason="invalid_title",
            retry_safe=False,
        )

        class FailedOpen:
            returncode = 1
            stdout = ""
            stderr = "/\x55sers/owner/private/vault"

        try:
            personal_tools.subprocess.run = lambda *_args, **_kwargs: FailedOpen()  # type: ignore[assignment]
            nonzero = runtime.registry.get("open_jarvis_vault").handler({})

            def raise_open(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("/private/var/folders/private-vault")

            personal_tools.subprocess.run = raise_open  # type: ignore[assignment]
            exception = runtime.registry.get("open_jarvis_vault").handler({})
        finally:
            personal_tools.subprocess.run = original_run  # type: ignore[assignment]

        _assert_unknown_open(nonzero, "vault open nonzero")
        if nonzero.metadata.get("returncode") != 1:
            raise SystemExit(f"vault open nonzero missed return code: {nonzero.metadata}")
        _assert_unknown_open(exception, "vault open exception")
        if exception.metadata.get("exception_type") != "RuntimeError":
            raise SystemExit(f"vault open exception missed exception type: {exception.metadata}")

    print("Personal error-guidance smoke passed: 18 validation and local-action failures")


if __name__ == "__main__":
    main()
