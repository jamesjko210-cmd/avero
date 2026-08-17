"""Focused offline proof for core-registry failure recovery declarations."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfoNotFoundError

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import registry as registry_module


PRIVATE_MARKERS = (
    "/\x55sers/example/private/input.txt",
    "/private/input.txt",
    "/var/folders/input.txt",
    "/tmp/input.txt",
    "sk_" + "live_SUPERSECRET123",
    "traceback",
)


def _assert_guided_failure(
    result,
    *,
    label: str,
    action: str,
    commands: list[str] | None = None,
    side_effect_possible: bool = False,
    retry_safe: bool = True,
) -> None:
    expected_commands = commands or []
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
    if action not in result.output:
        raise SystemExit(f"{label} hid its recovery action: {result.output}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": expected_commands,
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    expected_flags = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": side_effect_possible,
        "retry_safe": retry_safe,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, expected in expected_flags.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    if expected_commands:
        if result.metadata.get("next_command") != expected_commands[0]:
            raise SystemExit(f"{label} next command drifted: {result.metadata}")
        if result.metadata.get("recovery_commands") != expected_commands:
            raise SystemExit(f"{label} recovery commands drifted: {result.metadata}")
    public_proof = f"{result.output}\n{result.metadata}".lower()
    for marker in PRIVATE_MARKERS:
        if marker.lower() in public_proof:
            raise SystemExit(f"{label} leaked private failure detail: {public_proof}")
    for key in (
        "calls_model",
        "executes_tools",
        "queues_approval",
        "external_side_effect",
        "controls_computer",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if result.metadata.get(key):
            raise SystemExit(f"{label} unexpectedly set {key}: {result.metadata}")


class _OffsetlessDateTime:
    @classmethod
    def now(cls, _timezone=None):
        return SimpleNamespace(utcoffset=lambda: None)


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-registry-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        get = runtime.registry.get

        input_cases = (
            (
                "current time unknown timezone",
                get("current_time").handler(
                    {"location": "/\x55sers/example/private/input.txt"}
                ),
            ),
            (
                "time difference unknown timezone",
                get("time_difference").handler(
                    {
                        "source": "/private/input.txt",
                        "target": "Seoul",
                    }
                ),
            ),
            (
                "relative date unsupported",
                get("relative_date").handler({"target": "/tmp/input.txt"}),
            ),
            ("remember empty", get("remember").handler({})),
            (
                "remember local category",
                get("remember").handler(
                    {"category": "/private/input.txt", "body": "safe body"}
                ),
            ),
            (
                "remember local title",
                get("remember").handler(
                    {"title": "/tmp/input.txt", "body": "safe body"}
                ),
            ),
            (
                "remember local body",
                get("remember").handler({"body": "/var/folders/input.txt"}),
            ),
            ("search memory empty", get("search_memory").handler({})),
            ("daily note empty", get("write_daily_note").handler({})),
            ("command diagnosis empty", get("command_diagnosis").handler({})),
        )
        for label, result in input_cases:
            _assert_guided_failure(
                result,
                label=label,
                action=registry_module.CORE_INPUT_RECOVERY_ACTION,
            )
        if runtime.store.list_memories(limit=20):
            raise SystemExit("rejected registry inputs mutated memory")

        timezone_action = registry_module._timezone_database_recovery_guidance()
        timezone_commands = ["setup check", "time in UTC"]
        with patch.object(
            registry_module,
            "ZoneInfo",
            side_effect=ZoneInfoNotFoundError("private timezone database detail"),
        ):
            timezone_cases = (
                (
                    "current time timezone database",
                    get("current_time").handler({"location": "Seoul"}),
                ),
                (
                    "time difference timezone database",
                    get("time_difference").handler(
                        {"source": "Seoul", "target": "Tokyo"}
                    ),
                ),
            )
        with patch.object(registry_module, "datetime", _OffsetlessDateTime):
            offset_result = get("time_difference").handler(
                {"source": "Seoul", "target": "Tokyo"}
            )
        timezone_cases += (("time difference offset unavailable", offset_result),)
        for label, result in timezone_cases:
            _assert_guided_failure(
                result,
                label=label,
                action=timezone_action,
                commands=timezone_commands,
            )

        pending_projection = SimpleNamespace(
            status="pending_error",
            path_display="",
        )
        with patch.object(
            registry_module,
            "reconcile_memory_projection",
            return_value=pending_projection,
        ):
            projection_result = get("remember").handler(
                {"category": "facts", "title": "Projection proof", "body": "safe"}
            )
        _assert_guided_failure(
            projection_result,
            label="memory projection pending",
            action=registry_module.MEMORY_PROJECTION_RECOVERY_ACTION,
            commands=["repair memory projections"],
            side_effect_possible=True,
            retry_safe=False,
        )
        if projection_result.metadata.get("projection_pending") is not True:
            raise SystemExit(
                f"projection failure lost repair custody: {projection_result.metadata}"
            )
        if len(runtime.store.list_memories(limit=20)) != 1:
            raise SystemExit("projection failure did not preserve exactly one saved memory")
        for key in ("writes_files", "writes_memory", "writes_notes"):
            if projection_result.metadata.get(key) is not True:
                raise SystemExit(
                    f"projection failure hid committed write field {key}: "
                    f"{projection_result.metadata}"
                )

    print("Core registry failure-guidance smoke passed: 14 offline branches.")


if __name__ == "__main__":
    main()
