"""Machine-readable recovery guidance for user-visible tool failures.

This module deliberately validates explicit declarations. It does not guess
whether arbitrary English sounds actionable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


FAILURE_GUIDANCE_VERSION = 1
MAX_FAILURE_GUIDANCE_ACTION_CHARS = 240
MAX_FAILURE_GUIDANCE_COMMANDS = 5
MAX_FAILURE_GUIDANCE_COMMAND_CHARS = 120
KNOWN_NOT_SENT_SEND_RECOVERY_ACTION = (
    "Run `setup check`, correct the reported issue, then submit a new approved send."
)
PERSONAL_READ_RECOVERY_ACTION = (
    "Run `setup check`, correct the reported integration issue, then retry the read."
)
LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION = (
    "Run `setup check`, correct the reported configuration or storage issue, then retry the read."
)
LOCAL_READ_INPUT_RECOVERY_ACTION = (
    "Correct the reported local read input, then retry through the normal policy."
)
RESOURCE_NOT_FOUND_RECOVERY_ACTION = (
    "Refresh the relevant list, correct the missing identifier or path, then retry through the normal policy."
)
EXTERNAL_INFORMATION_RECOVERY_ACTION = (
    "Run `setup check`, confirm network access, then retry the lookup."
)
_LOCAL_PATH_RE = re.compile(
    r"(?:/" r"Users/|/private/|/var/folders/|/tmp/)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?:"
    r"\bsk_(?:live|test)_[A-Za-z0-9_-]+"
    r"|\bgh[pousr]_[A-Za-z0-9_-]+"
    r"|\bxox[baprs]-[A-Za-z0-9-]+"
    r"|\bAIza[A-Za-z0-9_-]{16,}"
    r"|\b\d{6,}:[A-Za-z0-9_-]{20,}"
    r")",
    re.IGNORECASE,
)


def _guidance_text(value: object, *, field: str, limit: int) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    if text != value or "\n" in text or "\r" in text:
        raise ValueError(f"{field} must be one normalized line")
    if len(text) > limit:
        raise ValueError(f"{field} exceeds its public-text bound")
    if _LOCAL_PATH_RE.search(text) or _SECRET_VALUE_RE.search(text):
        raise ValueError(f"{field} contains private-looking detail")
    return text


def declare_failure_guidance(
    metadata: dict[str, Any] | None,
    *,
    output: str,
    action: str,
    commands: Sequence[str] = (),
) -> dict[str, Any]:
    """Return copied metadata with one bounded, public recovery declaration.

    ``action`` must already appear in the user-visible ``output``. Commands are
    mirrored to the established ``next_command`` / ``recovery_commands`` keys
    so existing executor and audit consumers remain compatible.
    """

    if metadata is not None and type(metadata) is not dict:
        raise ValueError("metadata must be an exact dict")
    if type(output) is not str:
        raise ValueError("output must be an exact string")
    public_action = _guidance_text(
        action,
        field="action",
        limit=MAX_FAILURE_GUIDANCE_ACTION_CHARS,
    )
    if public_action not in output:
        raise ValueError("action must appear verbatim in user-visible output")
    if type(commands) not in {tuple, list}:
        raise ValueError("commands must be a tuple or list")
    if len(commands) > MAX_FAILURE_GUIDANCE_COMMANDS:
        raise ValueError("commands exceed their count bound")
    public_commands = [
        _guidance_text(
            command,
            field=f"commands[{index}]",
            limit=MAX_FAILURE_GUIDANCE_COMMAND_CHARS,
        )
        for index, command in enumerate(commands)
    ]
    if len(set(public_commands)) != len(public_commands):
        raise ValueError("commands must be unique")
    if any(command not in output for command in public_commands):
        raise ValueError("every command must appear in user-visible output")

    result = dict(metadata or {})
    declaration = {
        "version": FAILURE_GUIDANCE_VERSION,
        "action": public_action,
        "commands": public_commands,
    }
    existing = result.get("recovery_guidance")
    if existing is not None and existing != declaration:
        raise ValueError("recovery_guidance conflicts with the declaration")
    if public_commands:
        existing_next = result.get("next_command")
        if existing_next is not None and existing_next != public_commands[0]:
            raise ValueError("next_command conflicts with the declaration")
        existing_commands = result.get("recovery_commands")
        if existing_commands is not None and existing_commands != public_commands:
            raise ValueError("recovery_commands conflict with the declaration")
        result["next_command"] = public_commands[0]
        result["recovery_commands"] = public_commands
    result["recovery_guidance"] = declaration
    return result


def declare_outcome_unknown_failure(
    metadata: dict[str, Any] | None,
    *,
    output: str,
    commands: Sequence[str] = (),
) -> dict[str, Any]:
    """Declare a possible side effect whose final outcome must be checked first."""

    result = dict(metadata or {})
    result.update(
        {
            "outcome_known": False,
            "outcome_unknown": True,
            "execution_outcome_unknown": True,
            "side_effect_possible": True,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        result,
        output=output,
        action=output,
        commands=commands,
    )


def declare_known_not_sent_failure(
    metadata: dict[str, Any] | None,
    *,
    output: str,
    action: str,
    commands: Sequence[str] = (),
) -> dict[str, Any]:
    """Declare a failed send whose transport did not accept the side effect."""

    result = dict(metadata or {})
    result.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        result,
        output=output,
        action=action,
        commands=commands,
    )


def declare_retryable_personal_read_failure(
    metadata: dict[str, Any] | None,
    *,
    output: str,
    action: str,
    commands: Sequence[str] = (),
) -> dict[str, Any]:
    """Declare a read-only integration failure with no possible side effect."""

    result = dict(metadata or {})
    result.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        result,
        output=output,
        action=action,
        commands=commands,
    )


def declare_retryable_local_read_failure(
    metadata: dict[str, Any] | None,
    *,
    output: str,
    action: str,
    commands: Sequence[str] = (),
) -> dict[str, Any]:
    """Declare a rejected side-effect-free local read or search request."""

    result = dict(metadata or {})
    result.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        result,
        output=output,
        action=action,
        commands=commands,
    )


def declare_resource_not_found_failure(
    metadata: dict[str, Any] | None,
    *,
    output: str,
    action: str,
) -> dict[str, Any]:
    """Declare a missing local resource without granting a retry."""

    result = dict(metadata or {})
    result.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    # Resource-specific commands remain in the established next/recovery
    # aliases. The canonical declaration carries the bounded public action.
    return declare_failure_guidance(
        result,
        output=output,
        action=action,
    )


def declare_retryable_external_information_failure(
    metadata: dict[str, Any] | None,
    *,
    output: str,
    action: str,
    commands: Sequence[str] = (),
) -> dict[str, Any]:
    """Declare a failed read-only public-information lookup."""

    result = dict(metadata or {})
    result.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": False,
            "retry_safe": True,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return declare_failure_guidance(
        result,
        output=output,
        action=action,
        commands=commands,
    )
