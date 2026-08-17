"""Offline smoke coverage for people-tool recovery declarations."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.failure_guidance import LOCAL_READ_INPUT_RECOVERY_ACTION
from jarvis_v2.memory.store import PersonIdentityUnavailable
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.people import (
    PEOPLE_IDENTITY_RECOVERY_ACTION,
    PEOPLE_INPUT_RECOVERY_ACTION,
    PEOPLE_NOT_FOUND_RECOVERY_ACTION,
    PEOPLE_PROJECTION_RECOVERY_ACTION,
    PEOPLE_UNREADABLE_RECOVERY_ACTION,
    _people_projection_pending_failure,
)


PRIVATE_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")
KNOWN_NO_REPLAY_FIELDS = {
    "outcome_known": True,
    "outcome_unknown": False,
    "execution_outcome_unknown": False,
    "side_effect_possible": False,
    "automatic_retry_allowed": False,
    "authorizes_retry": False,
}


class UnreadablePerson:
    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"unreadable:{key}")


def _assert_guidance(
    result: object,
    *,
    action: str,
    commands: list[str],
    reason: str | None,
    retry_safe: bool,
    label: str,
) -> None:
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid its recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": commands,
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    if reason is not None and result.metadata.get("reason") != reason:
        raise SystemExit(f"{label} refusal reason drifted: {result.metadata}")
    for key, expected in KNOWN_NO_REPLAY_FIELDS.items():
        if result.metadata.get(key) != expected:
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    if result.metadata.get("retry_safe") is not retry_safe:
        raise SystemExit(f"{label} retry safety drifted: {result.metadata}")
    for key in (
        "queues_approval",
        "requires_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "external_side_effect",
        "controls_computer",
    ):
        if result.metadata.get(key):
            raise SystemExit(f"{label} unexpectedly set {key}: {result.metadata}")
    public = f"{result.output}\n{result.metadata}"
    if any(fragment in public for fragment in PRIVATE_FRAGMENTS):
        raise SystemExit(f"{label} leaked private local detail: {public}")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-people-guidance-") as temp:
        runtime = make_temp_runtime(Path(temp))
        add_person = runtime.registry.get("add_person").handler
        get_person = runtime.registry.get("get_person").handler
        log_interaction = runtime.registry.get("log_interaction").handler

        cases = (
            (
                add_person({"name": ""}),
                PEOPLE_INPUT_RECOVERY_ACTION,
                [],
                "missing_name",
                "missing add-person name",
            ),
            (
                add_person({"name": "/\x55sers/example/private/person"}),
                PEOPLE_INPUT_RECOVERY_ACTION,
                [],
                "invalid_name",
                "private-looking add-person name",
            ),
            (
                log_interaction({"name": "Maya", "summary": ""}),
                PEOPLE_INPUT_RECOVERY_ACTION,
                [],
                "missing_summary",
                "missing interaction summary",
            ),
            (
                get_person({}),
                LOCAL_READ_INPUT_RECOVERY_ACTION,
                [],
                "missing_person",
                "missing read selector",
            ),
            (
                get_person({"person_id": "bad"}),
                LOCAL_READ_INPUT_RECOVERY_ACTION,
                [],
                "bad_person_id",
                "invalid read selector",
            ),
            (
                get_person({"name": "Nobody"}),
                PEOPLE_NOT_FOUND_RECOVERY_ACTION,
                [],
                "not_found",
                "missing saved person",
            ),
        )
        for result, action, commands, reason, label in cases:
            _assert_guidance(
                result,
                action=action,
                commands=commands,
                reason=reason,
                retry_safe=True,
                label=label,
            )

        with patch.object(runtime.store, "get_person", return_value=UnreadablePerson()):
            unreadable = get_person({"name": "Unreadable"})
        _assert_guidance(
            unreadable,
            action=PEOPLE_UNREADABLE_RECOVERY_ACTION,
            commands=["people"],
            reason="unreadable_person",
            retry_safe=True,
            label="unreadable saved person",
        )

        with patch.object(
            runtime.store,
            "get_person",
            side_effect=PersonIdentityUnavailable("private index detail"),
        ):
            identity_unavailable = get_person({"name": "Maya"})
        _assert_guidance(
            identity_unavailable,
            action=PEOPLE_IDENTITY_RECOVERY_ACTION,
            commands=[],
            reason="identity_index_unavailable",
            retry_safe=True,
            label="identity index unavailable",
        )

        pending = _people_projection_pending_failure(
            "add_person",
            "Saved person #7, but person note publication remains pending. "
            "Run `repair person projections`; do not add the person again.",
            {
                "state_changed": True,
                "writes_files": True,
                "writes_memory": True,
                "writes_notes": True,
                "queues_approval": False,
                "requires_approval": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
                "external_side_effect": False,
                "controls_computer": False,
            },
        )
        _assert_guidance(
            pending,
            action=PEOPLE_PROJECTION_RECOVERY_ACTION,
            commands=[],
            reason=None,
            retry_safe=False,
            label="committed person with pending projection",
        )
        if pending.metadata.get("state_changed") is not True:
            raise SystemExit(f"pending projection lost committed-state truth: {pending.metadata}")
        if "do not add the person again" not in pending.output:
            raise SystemExit(f"pending projection invited a duplicate mutation: {pending.output}")

    print("People error-guidance smoke test passed.")


if __name__ == "__main__":
    main()
