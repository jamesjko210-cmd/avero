"""Mocked smoke tests for KakaoTalk and Instagram DM connectors."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import ApprovalArgumentResolution, Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.store import approval_action_digest
from jarvis_v2.scripts.test_runtime import (
    make_temp_runtime,
    review_pending_runtime_approval,
)
from jarvis_v2.tools import (
    clipboard_safety,
    contacts_connector,
    instagram_connector,
    kakao_connector,
)
from jarvis_v2.tools.registry import validate_tool_arguments


LOCAL_PATH_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")
NO_AUTHORITY_FLAGS = {
    "authorizes_execution": False,
    "authorizes_completion_claim": False,
    "approval_granted": False,
}


class _InstagramClipboardRun:
    def __init__(self, text: str = "user clipboard") -> None:
        self.text = text
        self.change_count = 10
        self.calls: list[tuple[list[str], dict]] = []
        self.write_count = 0
        self.fail_write_number: int | None = None
        self.fail_inspection = False
        self.newer_text_during_paste: str | None = None
        self.clipboard_info = "«class ut16», 8, «class utf8», 4, string, 4"

    def __call__(self, command, **kwargs):
        argv = list(command)
        self.calls.append((argv, dict(kwargs)))
        if argv == ["osascript", "-"]:
            script = kwargs.get("input", "")
            if 'return "CHAT_VERIFIED"' in script:
                return SimpleNamespace(returncode=0, stdout="CHAT_VERIFIED", stderr="")
            if 'return "ENTER_PRESSED"' in script:
                return SimpleNamespace(returncode=0, stdout="ENTER_PRESSED", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv == clipboard_safety._CHANGE_COUNT_COMMAND:
            return SimpleNamespace(
                returncode=0,
                stdout=str(self.change_count),
                stderr="",
            )
        if argv == ["osascript", "-e", "clipboard info"]:
            return SimpleNamespace(
                returncode=1 if self.fail_inspection else 0,
                stdout=(
                    ""
                    if self.fail_inspection
                    else self.clipboard_info
                ),
                stderr="PRIVATE /\x55sers/example/clipboard-inspection.log",
            )
        if argv == ["pbpaste"]:
            return SimpleNamespace(returncode=0, stdout=self.text, stderr="")
        if clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv:
            expected = int(argv[-1])
            if expected != self.change_count:
                return SimpleNamespace(returncode=0, stdout="STALE", stderr="")
            self.write_count += 1
            if self.fail_write_number == self.write_count:
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="PRIVATE /\x55sers/example/clipboard-write.log",
                )
            self.text = kwargs.get("input", "")
            self.change_count += 1
            return SimpleNamespace(
                returncode=0,
                stdout=f"WRITTEN:{self.change_count}",
                stderr="",
            )
        if (
            argv[:2] == ["osascript", "-e"]
            and argv != ["osascript", "-e", "clipboard info"]
            and self.newer_text_during_paste is not None
        ):
            self.text = self.newer_text_during_paste
            self.change_count += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class _KakaoClipboardRun(_InstagramClipboardRun):
    """Deterministic Kakao GUI/clipboard interleaving fixture."""

    def __init__(self, mode: str = "normal") -> None:
        super().__init__("user clipboard")
        self.mode = mode
        self.gui_phase = 0
        self.enter_observed = False
        self.newer_text = "newer user clipboard"

    def _external_copy(self) -> None:
        self.text = self.newer_text
        self.change_count += 1

    def __call__(self, command, **kwargs):
        argv = list(command)
        if argv != ["osascript", "-"]:
            return super().__call__(command, **kwargs)
        self.calls.append((argv, dict(kwargs)))
        self.gui_phase += 1
        if self.gui_phase == 1:
            if self.mode in {
                "preopened_wrong_case",
                "preopened_wrong_diacritic",
                "preopened_duplicate_initial",
            }:
                reason = {
                    "preopened_wrong_case": "the preopened exact KakaoTalk chat has the wrong case",
                    "preopened_wrong_diacritic": "the preopened exact KakaoTalk chat has a different diacritic",
                    "preopened_duplicate_initial": "the preopened KakaoTalk target is ambiguous",
                }[self.mode]
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr=f"execution error: JARVIS_GUARD: {reason}, so no message was sent (-2700)",
                )
            if self.mode == "recipient_wrong_marker":
                return SimpleNamespace(returncode=0, stdout="NOT_VERIFIED", stderr="")
            if self.mode == "guard_abort":
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "execution error: JARVIS_GUARD: no exact KakaoTalk chat matched "
                        "fixture example in the first 5 search results, so no message was typed. "
                        "Confirm the exact KakaoTalk display name and retry (-2700)"
                    ),
                )
            if self.mode == "before_recipient_paste":
                self._external_copy()
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "execution error: JARVIS_GUARD: clipboard ownership changed before "
                        "recipient paste, so no chat was opened (-2700)"
                    ),
                )
            if self.mode == "between_recipient_and_message":
                self._external_copy()
            return SimpleNamespace(returncode=0, stdout="CHAT_VERIFIED", stderr="")
        if self.gui_phase == 2:
            if self.mode in {
                "preopened_duplicate_before_paste",
                "preopened_duplicate_before_enter",
            }:
                boundary = (
                    "message paste"
                    if self.mode == "preopened_duplicate_before_paste"
                    else "Enter"
                )
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "execution error: JARVIS_GUARD: the exact KakaoTalk target became "
                        f"ambiguous before {boundary}, so no message was sent (-2700)"
                    ),
                )
            if self.mode == "message_wrong_marker":
                self.enter_observed = True
                return SimpleNamespace(returncode=0, stdout="NOT_ENTER_PROOF", stderr="")
            if self.mode == "immediately_before_enter":
                self._external_copy()
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "execution error: JARVIS_GUARD: clipboard ownership changed before Enter, "
                        "so no message was sent (-2700)"
                    ),
                )
            if self.mode == "after_enter":
                self.enter_observed = True
                self._external_copy()
            else:
                self.enter_observed = True
            return SimpleNamespace(returncode=0, stdout="ENTER_PRESSED", stderr="")
        raise AssertionError(f"Unexpected Kakao GUI phase: {self.gui_phase}")


def _config() -> JarvisConfig:
    root = Path(tempfile.mkdtemp())
    return JarvisConfig(data_dir=root, db_path=root / "jarvis.sqlite", obsidian_vault=root / "Vault")


def _leaks_local_path(value) -> bool:
    return any(fragment in str(value) for fragment in LOCAL_PATH_FRAGMENTS)


def assert_outcome_unknown_recovery(result, *, expected_output: str, label: str) -> None:
    if result.ok or result.output != expected_output:
        raise SystemExit(f"{label} missed fixed outcome-unknown output: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": expected_output,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": False,
        "outcome_unknown": True,
        "execution_outcome_unknown": True,
        "side_effect_possible": True,
        "retry_safe": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} outcome/retry field {key} drifted: {result.metadata}")


def assert_known_not_sent_recovery(result, *, action: str, label: str) -> None:
    if action not in result.output:
        raise SystemExit(f"{label} hid the canonical recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} recovery field {key} drifted: {result.metadata}")


def assert_kakao_send_handoff(metadata: dict, label: str, *, status: str) -> dict:
    if metadata.get("kakao_send_handoff_ready") is not True:
        raise SystemExit(f"{label} missed flat Kakao handoff readiness: {metadata}")
    handoff = metadata.get("kakao_send_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed Kakao send handoff: {metadata}")
    if handoff.get("source") != "send_kakao":
        raise SystemExit(f"{label} Kakao handoff source wrong: {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} Kakao nested handoff readiness missing: {handoff}")
    if handoff.get("status") != status:
        raise SystemExit(f"{label} Kakao handoff status wrong: {handoff}")
    send_attempted = bool(handoff.get("send_attempted"))
    expected_changed = ["kakao_send_attempt"] if send_attempted else []
    for key, expected_value in [
        ("ready_for_operator", True),
        ("state_changed", send_attempted),
        ("content_in_handoff", False),
        *NO_AUTHORITY_FLAGS.items(),
    ]:
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
    if metadata.get("kakao_send_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} Kakao ready alias parity failed: {metadata} vs {handoff}")
    if metadata.get("kakao_send_state_changed") != handoff.get("state_changed"):
        raise SystemExit(f"{label} Kakao state alias parity failed: {metadata} vs {handoff}")
    if metadata.get("kakao_send_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} Kakao content alias parity failed: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        alias = f"kakao_send_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} Kakao no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if metadata.get("changed") != expected_changed or handoff.get("changed") != expected_changed:
        raise SystemExit(f"{label} changed should reflect attempted Kakao send only: {handoff} vs {metadata}")
    if metadata.get("kakao_send_changed") != expected_changed:
        raise SystemExit(f"{label} Kakao changed alias parity failed: {metadata} vs {handoff}")
    if handoff.get("to") != metadata.get("to", handoff.get("to")):
        raise SystemExit(f"{label} Kakao handoff recipient parity failed: {metadata}")
    if handoff.get("message_chars") != metadata.get("message_chars"):
        raise SystemExit(f"{label} Kakao handoff message length parity failed: {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} Kakao handoff should not carry message content: {handoff}")
    if metadata.get("kakao_send_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} Kakao content metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("approval_required_before_execution") is not True or handoff.get("manual_review_required") is not True:
        raise SystemExit(f"{label} Kakao handoff should preserve approval/manual-review gate: {handoff}")
    expected_commands = {
        "outcome_unknown": ["recent tool runs", "execution recovery"],
        "failed": ["setup check"],
        "requested": ["recent tool runs"],
    }.get(status, ["safe next actions"])
    if handoff.get("next_safe_command") != expected_commands[0]:
        raise SystemExit(f"{label} Kakao next safe command wrong: {handoff}")
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} Kakao safe-command list wrong: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} Kakao safe-command count wrong: {handoff}")
    if metadata.get("kakao_send_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} Kakao next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("kakao_send_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} Kakao next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("kakao_send_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} Kakao next-command count alias parity failed: {metadata} vs {handoff}")
    contact_lookup_attempted = handoff.get("contact_lookup_attempted") is True
    reads_clipboard = handoff.get("reads_clipboard") is True
    if metadata.get("send_attempted", False) is not send_attempted:
        raise SystemExit(f"{label} Kakao handoff/metadata send-attempt parity failed: {metadata}")
    if metadata.get("requires_approval", False) is not send_attempted:
        raise SystemExit(f"{label} Kakao flat approval boundary should match attempted send: {metadata}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} Kakao handoff leaked local path: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} Kakao handoff missed boundaries: {handoff}")
    if metadata.get("kakao_send_boundaries") != boundaries:
        raise SystemExit(f"{label} Kakao boundary alias parity failed: {metadata} vs {handoff}")
    if boundaries.get("requires_approval") is not send_attempted:
        raise SystemExit(f"{label} Kakao handoff approval boundary should match attempted send: {handoff}")
    if boundaries.get("executes_side_effect") is not send_attempted or boundaries.get("external_side_effect") is not send_attempted:
        raise SystemExit(f"{label} Kakao handoff side-effect boundary should match attempted send: {handoff}")
    for key in [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} Kakao send should not perform {key}: {metadata}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} Kakao flat {key} should be {expected_value}: {metadata}")
        if boundaries.get(key) is not expected_value:
            raise SystemExit(f"{label} Kakao boundary {key} should be {expected_value}: {boundaries}")
    expected_personal_read = contact_lookup_attempted or reads_clipboard
    for key, expected_value in {
        "reads_clipboard": reads_clipboard,
        "reads_private_data": reads_clipboard,
        "reads_personal_data": expected_personal_read,
    }.items():
        if boundaries.get(key) is not expected_value:
            raise SystemExit(f"{label} Kakao {key} boundary drifted: {handoff}")
        if metadata.get(key, False) is not expected_value:
            raise SystemExit(f"{label} Kakao flat {key} drifted: {metadata}")
    return handoff


def assert_instagram_dm_handoff(metadata: dict, label: str, *, status: str) -> dict:
    if metadata.get("instagram_dm_handoff_ready") is not True:
        raise SystemExit(f"{label} missed flat Instagram handoff readiness: {metadata}")
    handoff = metadata.get("instagram_dm_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed Instagram DM handoff: {metadata}")
    if handoff.get("source") != "send_instagram_dm":
        raise SystemExit(f"{label} Instagram handoff source wrong: {handoff}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} Instagram nested handoff readiness missing: {handoff}")
    if handoff.get("status") != status:
        raise SystemExit(f"{label} Instagram handoff status wrong: {handoff}")
    send_attempted = bool(handoff.get("send_attempted"))
    expected_changed = ["instagram_dm_send_attempt"] if send_attempted else []
    for key, expected_value in [
        ("ready_for_operator", True),
        ("state_changed", send_attempted),
        ("content_in_handoff", False),
        *NO_AUTHORITY_FLAGS.items(),
    ]:
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} flat {key} should be {expected_value}: {metadata}")
        if handoff.get(key) is not expected_value:
            raise SystemExit(f"{label} handoff {key} should be {expected_value}: {handoff}")
    if metadata.get("instagram_dm_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} Instagram ready alias parity failed: {metadata} vs {handoff}")
    if metadata.get("instagram_dm_state_changed") != handoff.get("state_changed"):
        raise SystemExit(f"{label} Instagram state alias parity failed: {metadata} vs {handoff}")
    if metadata.get("instagram_dm_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} Instagram content alias parity failed: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        alias = f"instagram_dm_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} Instagram no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if metadata.get("changed") != expected_changed or handoff.get("changed") != expected_changed:
        raise SystemExit(f"{label} changed should reflect attempted Instagram DM only: {handoff} vs {metadata}")
    if metadata.get("instagram_dm_changed") != expected_changed:
        raise SystemExit(f"{label} Instagram changed alias parity failed: {metadata} vs {handoff}")
    if handoff.get("to") != metadata.get("to", handoff.get("to")):
        raise SystemExit(f"{label} Instagram handoff recipient parity failed: {metadata}")
    if handoff.get("message_chars") != metadata.get("message_chars"):
        raise SystemExit(f"{label} Instagram handoff message length parity failed: {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} Instagram handoff should not carry message content: {handoff}")
    if metadata.get("instagram_dm_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} Instagram content metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("approval_required_before_execution") is not True or handoff.get("manual_review_required") is not True:
        raise SystemExit(f"{label} Instagram handoff should preserve approval/manual-review gate: {handoff}")
    expected_commands = {
        "outcome_unknown": ["recent tool runs", "execution recovery"],
        "failed": ["setup check"],
        "sender_confirmed": ["recent tool runs"],
    }.get(status, ["safe next actions"])
    if handoff.get("next_safe_command") != expected_commands[0]:
        raise SystemExit(f"{label} Instagram next safe command wrong: {handoff}")
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} Instagram safe-command list wrong: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} Instagram safe-command count wrong: {handoff}")
    if metadata.get("instagram_dm_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} Instagram next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("instagram_dm_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} Instagram next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("instagram_dm_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} Instagram next-command count alias parity failed: {metadata} vs {handoff}")
    contact_lookup_attempted = handoff.get("contact_lookup_attempted") is True
    if metadata.get("send_attempted", False) is not send_attempted:
        raise SystemExit(f"{label} Instagram handoff/metadata send-attempt parity failed: {metadata}")
    if metadata.get("requires_approval", False) is not send_attempted:
        raise SystemExit(f"{label} Instagram flat approval boundary should match attempted send: {metadata}")
    if "automation" not in str(handoff.get("automation_warning", "")).lower():
        raise SystemExit(f"{label} Instagram handoff should preserve automation warning: {handoff}")
    if _leaks_local_path(handoff):
        raise SystemExit(f"{label} Instagram handoff leaked local path: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict):
        raise SystemExit(f"{label} Instagram handoff missed boundaries: {handoff}")
    if metadata.get("instagram_dm_boundaries") != boundaries:
        raise SystemExit(f"{label} Instagram boundary alias parity failed: {metadata} vs {handoff}")
    if boundaries.get("requires_approval") is not send_attempted:
        raise SystemExit(f"{label} Instagram handoff approval boundary should match attempted send: {handoff}")
    if boundaries.get("executes_side_effect") is not send_attempted or boundaries.get("external_side_effect") is not send_attempted:
        raise SystemExit(f"{label} Instagram handoff side-effect boundary should match attempted send: {handoff}")
    for key in [
        "calls_model",
        "executes_tools",
        "reads_private_data",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
    ]:
        if boundaries.get(key) or metadata.get(key):
            raise SystemExit(f"{label} Instagram send should not perform {key}: {metadata}")
    for key, expected_value in NO_AUTHORITY_FLAGS.items():
        if metadata.get(key) is not expected_value:
            raise SystemExit(f"{label} Instagram flat {key} should be {expected_value}: {metadata}")
        if boundaries.get(key) is not expected_value:
            raise SystemExit(f"{label} Instagram boundary {key} should be {expected_value}: {boundaries}")
    if boundaries.get("calls_external_service") is not True:
        raise SystemExit(f"{label} Instagram handoff should preserve external-service browser target: {handoff}")
    if boundaries.get("reads_personal_data") is not contact_lookup_attempted:
        raise SystemExit(f"{label} Instagram personal-data boundary should match contact lookup: {handoff}")
    if metadata.get("reads_personal_data", False) is not contact_lookup_attempted:
        raise SystemExit(f"{label} Instagram flat personal-data boundary should match contact lookup: {metadata}")
    return handoff


def test_kakao_tool_is_high_risk_and_validates_args() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    if tool.name != "send_kakao" or tool.risk != RiskLevel.HIGH_RISK:
        raise SystemExit(f"Kakao tool risk registration is wrong: {tool}")
    contract = tool.argument_contract
    if (
        contract is None
        or contract.allow_unknown is not False
        or {
            field.name: (
                tuple(sorted(kind.value for kind in field.types)),
                field.required,
            )
            for field in contract.fields
        }
        != {
            "to": (("string",), True),
            "message": (("string",), True),
            "target_mode": (("string",), False),
        }
    ):
        raise SystemExit(f"Kakao strict argument contract drifted: {contract}")
    for args, expected_status in (
        ({"to": ["Mom"], "message": "hi"}, "type_mismatch"),
        ({"to": "Mom", "message": "hi", "approval_granted": True}, "unknown_arguments"),
        (
            {
                "to": "Mom",
                "message": "hi",
                kakao_connector.KAKAO_APPROVAL_BINDING_KEY: "0" * 64,
            },
            "unknown_arguments",
        ),
    ):
        validation = validate_tool_arguments(tool, args)
        if validation.valid or validation.status != expected_status:
            raise SystemExit(f"Kakao malformed arguments did not fail closed: {args} -> {validation}")
    valid = validate_tool_arguments(tool, {"to": "Mom", "message": "hi"})
    if not valid.valid or valid.status != "valid":
        raise SystemExit(f"Kakao rejected its canonical reviewed argument shape: {valid}")
    approval_contract = tool.approval_argument_contract
    if (
        approval_contract is None
        or approval_contract.allow_unknown is not False
        or {field.name for field in approval_contract.fields}
        != {
            "to",
            "message",
            kakao_connector.KAKAO_TARGET_MODE_KEY,
            kakao_connector.KAKAO_APPROVAL_BINDING_KEY,
        }
    ):
        raise SystemExit(f"Kakao approval-bound contract drifted: {approval_contract}")
    bound_validation = validate_tool_arguments(
        tool,
        {
            "to": "Mom",
            "message": "hi",
            kakao_connector.KAKAO_TARGET_MODE_KEY:
                kakao_connector.KAKAO_SEARCH_EXACT_TITLE_MODE,
            kakao_connector.KAKAO_APPROVAL_BINDING_KEY:
                kakao_connector._kakao_approval_binding("Mom", "hi"),
        },
        approved=True,
    )
    if not bound_validation.valid:
        raise SystemExit(
            f"Kakao rejected its resolver-bound approved shape: {bound_validation}"
        )
    forged_bound = tool.handler(
        {
            "to": "Mom",
            "message": "changed after binding",
            kakao_connector.KAKAO_APPROVAL_BINDING_KEY:
                kakao_connector._kakao_approval_binding("Mom", "original"),
        }
    )
    if (
        forged_bound.ok
        or forged_bound.metadata.get("send_attempted")
        or forged_bound.metadata.get("reason") != "approval_binding_invalid"
    ):
        raise SystemExit(
            f"Kakao handler accepted a forged approved binding: {forged_bound}"
        )
    forged_mode = tool.handler(
        {
            "to": "Mom",
            "message": "original",
            kakao_connector.KAKAO_TARGET_MODE_KEY:
                kakao_connector.KAKAO_SEARCH_EXACT_TITLE_MODE,
            kakao_connector.KAKAO_APPROVAL_BINDING_KEY:
                kakao_connector._kakao_approval_binding(
                    "Mom",
                    "original",
                    kakao_connector.KAKAO_PREOPENED_EXACT_CHAT_MODE,
                ),
        }
    )
    if (
        forged_mode.ok
        or forged_mode.metadata.get("send_attempted")
        or forged_mode.metadata.get("reason") != "approval_binding_invalid"
    ):
        raise SystemExit(
            f"Kakao handler accepted a target-mode change after binding: {forged_mode}"
        )
    missing_to = tool.handler({"message": "hi"})
    if missing_to.ok or "recipient" not in missing_to.output.lower():
        raise SystemExit(f"Kakao validation should reject missing recipient: {missing_to}")
    missing_to_handoff = assert_kakao_send_handoff(missing_to.metadata, "Kakao missing-recipient", status="refused")
    if missing_to_handoff.get("reason") != "missing_recipient" or missing_to_handoff.get("send_attempted"):
        raise SystemExit(f"Kakao missing-recipient handoff should be inert refusal: {missing_to_handoff}")
    missing = tool.handler({"to": "Mom"})
    if missing.ok or "message" not in missing.output.lower():
        raise SystemExit(f"Kakao validation should reject missing message: {missing}")
    if missing.metadata.get("executes_side_effect"):
        raise SystemExit(f"Kakao validation must not claim a send occurred: {missing.metadata}")
    if missing.metadata.get("to") != "Mom" or missing.metadata.get("message_chars") != 0:
        raise SystemExit(f"Kakao missing-message metadata should preserve sanitized recipient only: {missing.metadata}")
    handoff = assert_kakao_send_handoff(missing.metadata, "Kakao missing-message", status="refused")
    if handoff.get("reason") != "missing_message" or handoff.get("send_attempted"):
        raise SystemExit(f"Kakao missing-message handoff should be inert refusal: {handoff}")


def test_kakao_tool_uses_mocked_sender() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    calls = []
    original = kakao_connector._send_kakao
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector.contacts_connector.resolve_contact = lambda _query: [
            kakao_connector.contacts_connector.ContactMatch("Mom", phone="+821012345678")
        ]
        def mocked_send(to, message):
            calls.append((to, message))
            return {
                **clipboard_safety.privacy_boundary_metadata(
                    snapshot_attempted=True,
                    snapshot_captured=True,
                    replacement_attempted=True,
                    text_restored=True,
                ),
                "kakao_recipient_phase_completed": True,
                "kakao_message_phase_completed": True,
                "kakao_enter_pressed": True,
            }

        kakao_connector._send_kakao = mocked_send
        result = tool.handler({"to": "Mom", "message": "On my way"})
    finally:
        kakao_connector._send_kakao = original
        kakao_connector.contacts_connector.resolve_contact = original_resolve
    # Kakao searches its own friend list by NAME, so a single Contacts match
    # resolves to the canonical name typed into Kakao search — not a phone handle
    # (that handle model is iMessage/email only).
    if not result.ok or calls != [("Mom", "On my way")]:
        raise SystemExit(f"Kakao mocked send failed: {result} {calls}")
    if not result.metadata.get("executes_side_effect") or not result.metadata.get("external_side_effect"):
        raise SystemExit(f"Kakao successful send metadata should mark the side effect: {result.metadata}")
    handoff = assert_kakao_send_handoff(result.metadata, "Kakao mocked send", status="outcome_unknown")
    if handoff.get("send_attempted") is not True or not handoff.get("boundaries", {}).get("controls_computer"):
        raise SystemExit(f"Kakao successful send handoff should preserve GUI-send attempt: {handoff}")
    if (
        result.metadata.get("send_requested") is not True
        or result.metadata.get("gui_automation_completed") is not True
        or result.metadata.get("recipient_window_title_verified") is not True
        or result.metadata.get("one_to_one_chat_verified") is not False
        or result.metadata.get("send_key_requested") is not True
        or result.metadata.get("target_content_verified") is not False
        or result.metadata.get("delivery_verified") is not False
        or result.metadata.get("confirmation_required") is not True
        or result.metadata.get("operator_confirmation_required") is not True
        or result.metadata.get("execution_outcome_unknown") is not True
        or result.metadata.get("outcome_known") is not False
        or result.metadata.get("side_effect_possible") is not True
        or result.metadata.get("retry_safe") is not False
        or result.metadata.get("automatic_retry_allowed") is not False
        or result.metadata.get("authorizes_retry") is not False
        or result.metadata.get("authorizes_completion_claim") is not False
        or handoff.get("target_content_verified") is not False
        or handoff.get("delivery_verified") is not False
        or handoff.get("send_requested") is not True
        or handoff.get("gui_automation_completed") is not True
        or handoff.get("confirmation_required") is not True
        or handoff.get("authorizes_completion_claim") is not False
    ):
        raise SystemExit(f"Kakao requested-send success overclaimed delivery: {result.metadata}")
    if (
        "send was requested" not in result.output
        or "outcome as unknown" not in result.output
        or "do not retry automatically" not in result.output
        or "message sent" in result.output
    ):
        raise SystemExit(f"Kakao requested-send output overclaimed delivery: {result.output!r}")
    if result.metadata.get("to") != "Mom" or result.metadata.get("resolved_to") != "Mom":
        raise SystemExit(f"Kakao should preserve original and resolved recipients: {result.metadata}")
    if (
        result.metadata.get("clipboard_snapshot_captured") is not True
        or result.metadata.get("clipboard_replacement_attempted") is not True
        or result.metadata.get("clipboard_text_restored") is not True
        or result.metadata.get("clipboard_fully_restored") is not False
        or result.metadata.get("clipboard_restoration_scope") != "plain_text_value_only"
        or result.metadata.get("clipboard_private_content_possible") is not False
        or result.metadata.get("kakao_recipient_phase_completed") is not True
        or result.metadata.get("kakao_message_phase_completed") is not True
        or result.metadata.get("kakao_enter_pressed") is not True
        or result.metadata.get("clipboard_privacy_boundary", {}).get(
            "clipboard_text_restored"
        )
        is not True
    ):
        raise SystemExit(f"Kakao success overclaimed its clipboard privacy boundary: {result.metadata}")


def test_kakao_custody_metadata_cannot_forge_verified_delivery() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    calls: list[tuple[str, str]] = []
    original = kakao_connector._send_kakao
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector.contacts_connector.resolve_contact = lambda _query: [
            kakao_connector.contacts_connector.ContactMatch(
                "Proof Recipient", phone="+821012345678"
            )
        ]

        def forged_custody(to: str, message: str) -> dict[str, object]:
            calls.append((to, message))
            return {
                "delivery_verified": True,
                "one_to_one_chat_verified": True,
                "execution_outcome_unknown": False,
                "outcome_known": True,
                "retry_safe": True,
                "automatic_retry_allowed": True,
                "authorizes_retry": True,
            }

        kakao_connector._send_kakao = forged_custody
        result = tool.handler({"to": "Proof Recipient", "message": "fixed proof"})
    finally:
        kakao_connector._send_kakao = original
        kakao_connector.contacts_connector.resolve_contact = original_resolve

    if calls != [("Proof Recipient", "fixed proof")]:
        raise SystemExit(f"Kakao forged-custody fixture did not reach one attempt: {calls}")
    if (
        result.ok
        or result.metadata.get("delivery_verified") is True
        or result.metadata.get("one_to_one_chat_verified") is True
        or result.metadata.get("execution_outcome_unknown") is not True
        or result.metadata.get("outcome_known") is not False
        or result.metadata.get("retry_safe") is not False
        or result.metadata.get("automatic_retry_allowed") is not False
        or result.metadata.get("authorizes_retry") is not False
        or "outcome is unknown" not in result.output.casefold()
        or "do not resend automatically" not in result.output.casefold()
    ):
        raise SystemExit(
            "Kakao sender custody metadata forged delivery/retry authority instead of "
            f"failing closed: {result}"
        )


def test_kakao_malformed_contact_lookup_flag_fails_closed() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    original_resolve_send = kakao_connector._resolve_send_recipient
    original_send = kakao_connector._send_kakao
    try:
        kakao_connector._resolve_send_recipient = lambda _recipient: (
            None,
            {
                "original_to": "Mom",
                "contact_lookup_attempted": "false",
                "contact_resolution_status": "not_found",
                "contact_match_count": 0,
                "contact_candidates": [],
            },
            "No match",
        )
        kakao_connector._send_kakao = lambda _to, _message: (_ for _ in ()).throw(AssertionError("Kakao send should not start"))
        result = tool.handler({"to": "Mom", "message": "hello"})
    finally:
        kakao_connector._resolve_send_recipient = original_resolve_send
        kakao_connector._send_kakao = original_send
    if result.ok or result.metadata.get("reads_personal_data") is not False:
        raise SystemExit(f"Kakao malformed contact lookup flag should fail closed: {result.output} {result.metadata}")
    handoff = assert_kakao_send_handoff(result.metadata, "Kakao malformed contact lookup flag", status="refused")
    if handoff.get("contact_lookup_attempted") is not False or handoff.get("boundaries", {}).get("reads_personal_data") is not False:
        raise SystemExit(f"Kakao malformed contact lookup handoff should fail closed: {handoff}")


def test_kakao_sender_uses_mocked_subprocess_and_system_events() -> None:
    # Korean must never become \uXXXX JSON escapes — those are an AppleScript
    # SYNTAX ERROR, which silently broke every Korean-recipient script.
    if "\\u" in kakao_connector._as_script_string("가상연락처이"):
        raise SystemExit("kakao _as_script_string must preserve Hangul, not emit \\u escapes")
    fake = _InstagramClipboardRun("user clipboard")
    original = kakao_connector.subprocess.run
    try:
        kakao_connector.subprocess.run = fake
        receipt = kakao_connector._send_kakao('Mom "K"', 'On my way \\ soon')
    finally:
        kakao_connector.subprocess.run = original
    phase_calls = [
        (argv, kwargs)
        for argv, kwargs in fake.calls
        if argv == ["osascript", "-"]
    ]
    if len(phase_calls) != 2:
        raise SystemExit(f"Kakao sender did not run exactly two GUI phases: {phase_calls}")
    scripts: list[str] = []
    for cmd, kwargs in phase_calls:
        script = kwargs.get("input")
        if (
            cmd != ["osascript", "-"]
            or kwargs.get("capture_output") is not True
            or kwargs.get("text") is not True
            or kwargs.get("timeout") != 30
            or not isinstance(script, str)
        ):
            raise SystemExit(f"Kakao send phase arguments drifted: {cmd} {kwargs}")
        scripts.append(script)
    recipient_script, message_script = scripts
    combined_script = "\n".join(scripts)
    argv_text = repr([argv for argv, _kwargs in fake.calls])
    if 'Mom "K"' in argv_text or "On my way" in argv_text:
        raise SystemExit(f"Kakao recipient/message leaked into subprocess argv: {argv_text}")
    if "set the clipboard" in combined_script or "savedClip" in combined_script:
        raise SystemExit("Kakao send phases must never set, save, or restore the clipboard")
    expected_recipient_parts = [
        'tell application "KakaoTalk" to activate',
        'tell application "System Events"',
        'tell process "KakaoTalk"',
        'set stagedRecipient to (the clipboard as text)',
        'if stagedRecipient is not "Mom \\"K\\""',
        'keystroke "v" using {command down}',
        'keystroke "2" using {command down}',
        "key code 125",
        "JARVIS_GUARD:",
        "name of front window",
        "set maxSearchResults to 5",
        "repeat with candidateIndex from 1 to maxSearchResults",
        "repeat with selectionStep from 1 to candidateIndex",
        "ignoring case",
        'set titlesMatch to (chatTitle is "Mom \\"K\\"")',
        "set chatMatched to true",
    ]
    for expected in expected_recipient_parts:
        if expected not in recipient_script:
            raise SystemExit(f"Kakao recipient phase missing {expected!r}: {recipient_script}")
    expected_message_parts = [
        'set titlesMatch to (chatTitle is "Mom \\"K\\"")',
        'set stagedMessage to (the clipboard as text)',
        'if stagedMessage is not "On my way \\\\ soon"',
        'set stagedMessageBeforeEnter to (the clipboard as text)',
        'if stagedMessageBeforeEnter is not "On my way \\\\ soon"',
        'set titlesMatchBeforeEnter to (chatTitleBeforeEnter is "Mom \\"K\\"")',
        'keystroke "v" using {command down}',
        "key code 36",
        'return "ENTER_PRESSED"',
    ]
    for expected in expected_message_parts:
        if expected not in message_script:
            raise SystemExit(f"Kakao message phase missing {expected!r}: {message_script}")
    if 'keystroke "Mom \\"K\\""' in combined_script or 'keystroke "On my way \\\\ soon"' in combined_script:
        raise SystemExit("Kakao sender must never TYPE text via keystroke (Korean cannot be typed)")
    if "chatTitle contains" in combined_script or "contains chatTitle" in combined_script:
        raise SystemExit("Kakao sender must require an exact chat title, never a substring match")
    if message_script.index("titlesMatch is false") > message_script.index('keystroke "v"'):
        raise SystemExit("Kakao sender must verify the exact chat before message paste")
    if message_script.index("titlesMatchBeforeEnter is false") > message_script.index("key code 36"):
        raise SystemExit("Kakao sender must reverify the exact chat before Enter")
    writes = [
        kwargs.get("input")
        for argv, kwargs in fake.calls
        if kakao_connector.clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv
    ]
    if writes != ['Mom "K"', 'On my way \\ soon', "user clipboard"]:
        raise SystemExit(f"Kakao clipboard transaction order drifted: {writes}")
    if (
        fake.text != "user clipboard"
        or receipt.get("clipboard_text_restored") is not True
        or receipt.get("kakao_recipient_phase_completed") is not True
        or receipt.get("kakao_message_phase_completed") is not True
        or receipt.get("kakao_enter_pressed") is not True
    ):
        raise SystemExit(f"Kakao sender custody receipt drifted: {fake.text!r} {receipt}")


def test_kakao_sender_preserves_hangul_recipient_and_message_via_clipboard() -> None:
    """Hangul must reach AppleScript as text literals and only via clipboard paste."""
    fake = _InstagramClipboardRun("user clipboard")
    original = kakao_connector.subprocess.run
    recipient = "가상연락처일"
    message = "자비스 한글 전달 확인"
    try:
        kakao_connector.subprocess.run = fake
        kakao_connector._send_kakao(recipient, message)
    finally:
        kakao_connector.subprocess.run = original
    script_calls = [
        (argv, kwargs)
        for argv, kwargs in fake.calls
        if argv == ["osascript", "-"]
    ]
    if len(script_calls) != 2:
        raise SystemExit(f"Hangul Kakao send should make two GUI phase calls: {script_calls}")
    scripts = [kwargs.get("input") for _argv, kwargs in script_calls]
    if not all(isinstance(script, str) for script in scripts):
        raise SystemExit(f"Hangul Kakao phases must travel over stdin: {script_calls}")
    combined = "\n".join(scripts)
    if recipient in repr([argv for argv, _kwargs in fake.calls]) or message in repr([argv for argv, _kwargs in fake.calls]):
        raise SystemExit("Hangul Kakao recipient/message leaked into subprocess argv")
    for value in (recipient, message):
        if value not in combined or "\\u" in combined:
            raise SystemExit(f"Hangul Kakao text must remain literal in AppleScript: {combined}")
        if f'keystroke "{value}"' in combined:
            raise SystemExit("Hangul Kakao text must never be injected with keystroke")
    if combined.count('keystroke "v" using {command down}') < 2:
        raise SystemExit("Hangul recipient and message must both enter through clipboard paste")
    if "set the clipboard" in combined or "savedClip" in combined:
        raise SystemExit("Hangul Kakao send phases mutated clipboard outside the transaction")


def test_kakao_preopened_sender_never_navigates_or_searches() -> None:
    fake = _InstagramClipboardRun("user clipboard")
    original = kakao_connector.subprocess.run
    recipient = "Exact Proof Friend"
    message = "Jarvis V3 Kakao supervised proof 8605"
    try:
        kakao_connector.subprocess.run = fake
        receipt = kakao_connector._send_kakao_preopened_exact_chat(
            recipient,
            message,
        )
    finally:
        kakao_connector.subprocess.run = original

    phase_calls = [
        (argv, kwargs)
        for argv, kwargs in fake.calls
        if argv == ["osascript", "-"]
    ]
    if len(phase_calls) != 2:
        raise SystemExit(f"Preopened Kakao mode did not run two exact phases: {phase_calls}")
    recipient_script = str(phase_calls[0][1].get("input") or "")
    message_script = str(phase_calls[1][1].get("input") or "")
    forbidden_navigation = (
        'keystroke "2"',
        'keystroke "f"',
        "candidateIndex",
        "maxSearchResults",
        'tell application "KakaoTalk" to reopen',
        'tell application "KakaoTalk" to activate',
    )
    if any(item in "\n".join((recipient_script, message_script)) for item in forbidden_navigation):
        raise SystemExit(
            "Preopened Kakao mode navigated, searched, or used launch-capable activation "
            f"after operator review: {recipient_script}\n{message_script}"
        )
    required_recipient_guards = (
        'if not (exists process "KakaoTalk")',
        "name of front window",
        "set exactTitleWindowCount to 0",
        "if exactTitleWindowCount is not 1",
        "considering case, diacriticals",
        'return "CHAT_VERIFIED"',
    )
    if any(item not in recipient_script for item in required_recipient_guards):
        raise SystemExit(f"Preopened Kakao target guards drifted: {recipient_script}")
    if recipient not in recipient_script or recipient not in message_script:
        raise SystemExit("Preopened Kakao exact title was not preserved across both phases")
    if "ignoring case" in recipient_script or "ignoring case" in message_script:
        raise SystemExit("Preopened Kakao mode weakened exact title matching")
    for expected in (
        "set exactTitleWindowCountBeforePaste to 0",
        "if exactTitleWindowCountBeforePaste is not 1",
        "set exactTitleWindowCountBeforeEnter to 0",
        "if exactTitleWindowCountBeforeEnter is not 1",
    ):
        if expected not in message_script:
            raise SystemExit(f"Preopened Kakao repeated ambiguity guard missing {expected!r}")
    if message_script.index("titlesMatch is false") > message_script.index('keystroke "v"'):
        raise SystemExit("Preopened Kakao mode did not recheck title before paste")
    if message_script.index("titlesMatchBeforeEnter is false") > message_script.index("key code 36"):
        raise SystemExit("Preopened Kakao mode did not recheck title before Enter")
    writes = [
        kwargs.get("input")
        for argv, kwargs in fake.calls
        if clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in argv
    ]
    if writes != [message, "user clipboard"]:
        raise SystemExit(
            f"Preopened Kakao mode staged anything except message then restore: {writes}"
        )
    if (
        receipt.get("kakao_recipient_phase_completed") is not True
        or receipt.get("kakao_message_phase_completed") is not True
        or receipt.get("kakao_enter_pressed") is not True
        or receipt.get("kakao_preopened_navigation_performed") is not False
        or receipt.get("kakao_preopened_search_performed") is not False
    ):
        raise SystemExit(f"Preopened Kakao receipt overclaimed or lost phase truth: {receipt}")


def test_kakao_preopened_case_diacritic_duplicate_and_drift_stop_before_enter() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    original_run = kakao_connector.subprocess.run
    try:
        for mode in (
            "preopened_wrong_case",
            "preopened_wrong_diacritic",
            "preopened_duplicate_initial",
            "preopened_duplicate_before_paste",
            "preopened_duplicate_before_enter",
        ):
            fake = _KakaoClipboardRun(mode)
            kakao_connector.subprocess.run = fake
            result = tool.handler(
                {
                    "to": "Éxact Proof Friend",
                    "message": "fixed proof",
                    kakao_connector.KAKAO_TARGET_MODE_KEY:
                        kakao_connector.KAKAO_PREOPENED_EXACT_CHAT_MODE,
                }
            )
            if (
                result.ok
                or result.metadata.get("outcome_known") is not True
                or result.metadata.get("execution_outcome_unknown") is not False
                or fake.enter_observed
            ):
                raise SystemExit(
                    f"Preopened Kakao {mode} did not stop as known-not-sent: {result}"
                )
            assert_known_not_sent_recovery(
                result,
                action=kakao_connector.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
                label=f"Preopened Kakao {mode}",
            )
            gui_scripts = [
                str(kwargs.get("input") or "")
                for argv, kwargs in fake.calls
                if argv == ["osascript", "-"]
            ]
            if any(
                token in "\n".join(gui_scripts)
                for token in ('keystroke "2"', 'keystroke "f"', "candidateIndex")
            ):
                raise SystemExit(f"Preopened Kakao {mode} fell back to navigation/search")
    finally:
        kakao_connector.subprocess.run = original_run


def test_kakao_phase_receipts_must_match_exact_markers() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    original_run = kakao_connector.subprocess.run
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector.contacts_connector.resolve_contact = lambda _query: []
        for mode, expected_stage, unknown in (
            ("recipient_wrong_marker", "recipient_phase_receipt_invalid", False),
            ("message_wrong_marker", "message_phase_receipt_invalid", True),
        ):
            fake = _KakaoClipboardRun(mode)
            kakao_connector.subprocess.run = fake
            result = tool.handler(
                {"to": "Exact Proof Friend", "message": "fixed proof"}
            )
            if result.ok or result.metadata.get("failure_stage") != expected_stage:
                raise SystemExit(f"Kakao accepted a wrong {mode} phase receipt: {result}")
            if unknown:
                assert_outcome_unknown_recovery(
                    result,
                    expected_output=kakao_connector._post_attempt_send_recovery_guidance(),
                    label=f"Kakao {mode}",
                )
            else:
                assert_known_not_sent_recovery(
                    result,
                    action=kakao_connector.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
                    label=f"Kakao {mode}",
                )
    finally:
        kakao_connector.subprocess.run = original_run
        kakao_connector.contacts_connector.resolve_contact = original_resolve


def test_kakao_guard_abort_surfaces_friendly_diagnostic() -> None:
    # When the guarded script aborts (wrong window / no matching chat), the tool
    # output should carry the specific guard reason, not the generic error.
    tool = kakao_connector.make_kakao_tools(_config())[0]
    original_run = kakao_connector.subprocess.run
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    fake = _KakaoClipboardRun("guard_abort")
    try:
        kakao_connector.contacts_connector.resolve_contact = lambda _query: []
        kakao_connector.subprocess.run = fake
        result = tool.handler({"to": "fixture example", "message": "hello"})
    finally:
        kakao_connector.subprocess.run = original_run
        kakao_connector.contacts_connector.resolve_contact = original_resolve
    if result.ok:
        raise SystemExit(f"guard abort must fail the send: {result}")
    if "stopped safely" not in result.output or "no exact KakaoTalk chat matched fixture example" not in result.output:
        raise SystemExit(f"guard abort should surface the specific safe diagnostic: {result.output!r}")
    if "노가리" in result.output or "top_result_mismatch:" in result.output:
        raise SystemExit(f"guard abort must not leak an unrelated chat title or raw guard token: {result.output!r}")
    if result.metadata.get("guard_stopped") is not True:
        raise SystemExit(f"guard abort should mark guard_stopped: {result.metadata}")
    assert_known_not_sent_recovery(
        result,
        action=kakao_connector.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
        label="Kakao guard abort",
    )
    if fake.text != "user clipboard" or fake.write_count != 2:
        raise SystemExit(f"Kakao guard abort did not restore owned clipboard: {fake.calls}")


def test_kakao_clipboard_snapshot_and_restore_fail_closed() -> None:
    original_run = kakao_connector.subprocess.run
    try:
        rich_run = _KakaoClipboardRun()
        rich_run.clipboard_info = "«class RTF », 200, «class utf8», 20, string, 20"
        kakao_connector.subprocess.run = rich_run
        try:
            kakao_connector._send_kakao("Fixture", "private proof")
        except kakao_connector.KakaoSendError as exc:
            if exc.stage != "clipboard_rich_content" or rich_run.write_count != 0:
                raise SystemExit(f"Kakao rich clipboard guard drifted: {rich_run.calls} {exc}")
            if (
                exc.clipboard_metadata.get("clipboard_snapshot_attempted") is not True
                or exc.clipboard_metadata.get("clipboard_snapshot_captured") is not False
                or exc.clipboard_metadata.get("clipboard_replacement_attempted") is not False
            ):
                raise SystemExit(f"Kakao rich clipboard metadata drifted: {exc.clipboard_metadata}")
        else:
            raise SystemExit("Kakao rich clipboard guard accepted destructive replacement")

        restore_failure_run = _KakaoClipboardRun()
        restore_failure_run.fail_write_number = 3
        kakao_connector.subprocess.run = restore_failure_run
        try:
            kakao_connector._send_kakao("Fixture", "private proof")
        except kakao_connector.KakaoSendError as exc:
            if exc.stage != "clipboard_custody_after_enter" or restore_failure_run.write_count != 3:
                raise SystemExit(f"Kakao restore failure was not surfaced: {exc}")
            if "PRIVATE" in exc.detail or "/\x55sers/" in exc.detail:
                raise SystemExit(f"Kakao restore failure leaked private detail: {exc.detail}")
            if exc.enter_pressed is not True:
                raise SystemExit(f"Kakao restore failure lost the Enter boundary: {exc}")
        else:
            raise SystemExit("Kakao restore failure incorrectly reported success")
    finally:
        kakao_connector.subprocess.run = original_run

    tool = kakao_connector.make_kakao_tools(_config())[0]
    original_send = kakao_connector._send_kakao
    try:
        kakao_connector._send_kakao = lambda _to, _message: (_ for _ in ()).throw(
            kakao_connector.KakaoSendError(
                "clipboard_custody_after_enter",
                "PRIVATE /\x55sers/example/kakao-handler.log",
                clipboard_metadata=kakao_connector.clipboard_safety.privacy_boundary_metadata(
                    snapshot_attempted=True,
                    snapshot_captured=True,
                    replacement_attempted=True,
                    text_restored=False,
                ),
            )
        )
        result = tool.handler({"to": "Fixture", "message": "private proof"})
    finally:
        kakao_connector._send_kakao = original_send
    assert_outcome_unknown_recovery(
        result,
        expected_output=kakao_connector._post_attempt_send_recovery_guidance(),
        label="Kakao clipboard restore failure",
    )
    if _leaks_local_path(result.metadata):
        raise SystemExit(f"Kakao clipboard restore failure leaked a path: {result.metadata}")
    if (
        result.metadata.get("clipboard_snapshot_captured") is not True
        or result.metadata.get("clipboard_replacement_attempted") is not True
        or result.metadata.get("clipboard_text_restored") is not False
        or result.metadata.get("clipboard_outcome_known") is not False
        or result.metadata.get("clipboard_private_content_possible") is not True
        or result.metadata.get("clipboard_privacy_boundary", {}).get(
            "clipboard_private_content_possible"
        )
        is not True
    ):
        raise SystemExit(f"Kakao restore failure lost clipboard privacy truth: {result.metadata}")


def test_kakao_clipboard_custody_interleavings_are_fail_closed() -> None:
    cases = (
        ("before_recipient_paste", "clipboard_custody_before_enter", False),
        ("between_recipient_and_message", "clipboard_custody_before_enter", False),
        ("immediately_before_enter", "clipboard_custody_before_enter", False),
        ("after_enter", "clipboard_custody_after_enter", True),
    )
    tool = kakao_connector.make_kakao_tools(_config())[0]
    original_run = kakao_connector.subprocess.run
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector.contacts_connector.resolve_contact = lambda _query: []
        for mode, expected_stage, outcome_unknown in cases:
            fake = _KakaoClipboardRun(mode)
            kakao_connector.subprocess.run = fake
            result = tool.handler({"to": "Synthetic Friend", "message": "Synthetic proof"})
            if result.ok or result.metadata.get("failure_stage") != expected_stage:
                raise SystemExit(f"Kakao {mode} custody classification drifted: {result}")
            if fake.text != fake.newer_text:
                raise SystemExit(f"Kakao {mode} overwrote the newer clipboard: {fake.calls}")
            if fake.enter_observed is not outcome_unknown:
                raise SystemExit(f"Kakao {mode} Enter boundary drifted: {fake.calls}")
            if outcome_unknown:
                assert_outcome_unknown_recovery(
                    result,
                    expected_output=kakao_connector._post_attempt_send_recovery_guidance(),
                    label=f"Kakao {mode}",
                )
            else:
                assert_known_not_sent_recovery(
                    result,
                    action=kakao_connector.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
                    label=f"Kakao {mode}",
                )
            if (
                result.metadata.get("clipboard_text_restored") is not False
                or result.metadata.get("clipboard_outcome_known") is not False
                or result.metadata.get("clipboard_private_content_possible") is not True
                or _leaks_local_path(result.metadata)
            ):
                raise SystemExit(f"Kakao {mode} lost content-free custody truth: {result.metadata}")
    finally:
        kakao_connector.subprocess.run = original_run
        kakao_connector.contacts_connector.resolve_contact = original_resolve


def test_kakao_permission_failures_are_actionable_and_bounded() -> None:
    cases = (
        (
            "execution error: Not authorized to send Apple events to System Events. (-1743)",
            "macos_automation_permission",
            "Privacy & Security > Automation",
        ),
        (
            "execution error: osascript is not allowed assistive access. (-25211)",
            "macos_accessibility_permission",
            "Privacy & Security > Accessibility",
        ),
    )
    for raw_error, expected_stage, expected_guidance in cases:
        stage, detail = kakao_connector._kakao_error_stage(raw_error)
        if stage != expected_stage or expected_guidance not in detail:
            raise SystemExit(f"Kakao permission classifier drifted: {stage!r} {detail!r}")
        failure_stage, output = kakao_connector._kakao_failure(
            kakao_connector.KakaoSendError(stage, detail)
        )
        if failure_stage != expected_stage or expected_stage not in output or expected_guidance not in output:
            raise SystemExit(f"Kakao permission guidance was not surfaced safely: {failure_stage!r} {output!r}")
        if raw_error in output or "(-1743)" in output or "(-25211)" in output:
            raise SystemExit(f"Kakao permission guidance leaked raw AppleScript diagnostics: {output!r}")

    timeout_stage, timeout_output = kakao_connector._kakao_failure(
        kakao_connector.subprocess.TimeoutExpired(["osascript"], 30)
    )
    if timeout_stage != "automation_timeout" or "timed out" not in timeout_output:
        raise SystemExit(f"Kakao timeout classification drifted: {timeout_stage!r} {timeout_output!r}")

    tool = kakao_connector.make_kakao_tools(_config())[0]
    original_send = kakao_connector._send_kakao
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector.contacts_connector.resolve_contact = lambda _query: []
        kakao_connector._send_kakao = lambda _to, _message: (_ for _ in ()).throw(
            kakao_connector.KakaoSendError(
                "macos_automation_permission",
                "In System Settings > Privacy & Security > Automation, allow the Jarvis Python process "
                "to control System Events and KakaoTalk, then retry.",
            )
        )
        result = tool.handler({"to": "Fixture Example", "message": "hello"})
    finally:
        kakao_connector._send_kakao = original_send
        kakao_connector.contacts_connector.resolve_contact = original_resolve
    if (
        result.ok
        or result.metadata.get("failure_stage") != "macos_automation_permission"
        or result.metadata.get("kakao_send_stage") != "macos_automation_permission"
    ):
        raise SystemExit(f"Kakao tool receipt missed the safe failure stage: {result.output} {result.metadata}")
    assert_known_not_sent_recovery(
        result,
        action=kakao_connector.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
        label="Kakao permission failure",
    )
    if "Privacy & Security > Automation" not in result.output or "Fixture Example" in result.output:
        raise SystemExit(f"Kakao tool permission guidance is incomplete or leaks recipient data: {result.output!r}")
    if (
        result.metadata.get("outcome_known") is not True
        or result.metadata.get("outcome_unknown") is not False
        or result.metadata.get("side_effect_possible") is not False
    ):
        raise SystemExit(f"Kakao permission failure lost known-no-send truth: {result.metadata}")


def test_kakao_send_failure_preserves_attempt_metadata() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    original = kakao_connector._send_kakao
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector.contacts_connector.resolve_contact = lambda _query: [
            kakao_connector.contacts_connector.ContactMatch("Mom", phone="+821012345678")
        ]
        kakao_connector._send_kakao = lambda _to, _message: (_ for _ in ()).throw(RuntimeError("app unavailable"))
        result = tool.handler({"to": "Mom", "message": "On my way"})
    finally:
        kakao_connector._send_kakao = original
        kakao_connector.contacts_connector.resolve_contact = original_resolve
    assert_outcome_unknown_recovery(
        result,
        expected_output=kakao_connector._post_attempt_send_recovery_guidance(),
        label="generic KakaoTalk send failure",
    )
    if "app unavailable" in result.output or "Error:" in result.output:
        raise SystemExit(f"Kakao mocked failure should not leak raw errors: {result}")
    if not result.metadata.get("send_attempted") or not result.metadata.get("executes_side_effect"):
        raise SystemExit(f"Kakao failed send should preserve attempted side-effect metadata: {result.metadata}")
    if not result.metadata.get("external_side_effect") or result.metadata.get("to") != "Mom" or result.metadata.get("resolved_to") != "Mom":
        raise SystemExit(f"Kakao failed send metadata should preserve recipient and external attempt: {result.metadata}")
    if result.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"Kakao failed send metadata should preserve bounded diagnostic type: {result.metadata}")
    handoff = assert_kakao_send_handoff(result.metadata, "Kakao failed send", status="outcome_unknown")
    if handoff.get("reason") != "send_outcome_unknown" or handoff.get("send_attempted") is not True:
        raise SystemExit(f"Kakao failed send handoff should preserve outcome uncertainty: {handoff}")


def test_kakao_redacts_path_shaped_recipient_metadata() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    for recipient in (
        "/\x55sers/example/Desktop/Claude code/kakao-target",
        "/var/folders/zc/jarvis/kakao-target",
        "/tmp/jarvis/kakao-target",
    ):
        missing = tool.handler({"to": recipient})
        if missing.ok or missing.metadata.get("to") != "<local-path>":
            raise SystemExit(f"Kakao missing-message metadata should redact path-shaped recipients: {missing.metadata}")
        handoff = assert_kakao_send_handoff(missing.metadata, "Kakao path-shaped missing-message", status="refused")
        if handoff.get("to") != "<local-path>" or handoff.get("reason") != "missing_message":
            raise SystemExit(f"Kakao path-shaped handoff should redact recipient: {handoff}")
        if (
            _leaks_local_path(missing.output)
            or _leaks_local_path(missing.metadata)
        ):
            raise SystemExit(f"Kakao validation leaked path-shaped recipient: {missing.output} {missing.metadata}")

    original = kakao_connector._send_kakao
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector.contacts_connector.resolve_contact = lambda _query: [
            kakao_connector.contacts_connector.ContactMatch("Path Target", phone="+821012345678")
        ]
        kakao_connector._send_kakao = lambda _to, _message: (_ for _ in ()).throw(RuntimeError("app unavailable near /private/tmp/kakao"))
        failed = tool.handler({"to": "/var/folders/zc/jarvis/kakao-target", "message": "hello"})
    finally:
        kakao_connector._send_kakao = original
        kakao_connector.contacts_connector.resolve_contact = original_resolve
    if failed.ok or failed.metadata.get("to") != "<local-path>" or "<local-path>" in failed.output:
        raise SystemExit(f"Kakao failed-send recipient redaction wrong: {failed.output} {failed.metadata}")
    handoff = assert_kakao_send_handoff(
        failed.metadata,
        "Kakao path-shaped failed-send",
        status="outcome_unknown",
    )
    if handoff.get("to") != "<local-path>" or handoff.get("reason") != "send_outcome_unknown":
        raise SystemExit(f"Kakao failed-send handoff should redact recipient: {handoff}")
    if (
        _leaks_local_path(failed.output)
        or _leaks_local_path(failed.metadata)
    ):
        raise SystemExit(f"Kakao failed-send output/metadata leaked path-shaped recipient: {failed.output} {failed.metadata}")


def test_kakao_ambiguous_contact_refuses_without_live_send() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    calls = []
    original = kakao_connector._send_kakao
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector._send_kakao = lambda to, message: calls.append((to, message)) or ""
        kakao_connector.contacts_connector.resolve_contact = lambda _query: [
            kakao_connector.contacts_connector.ContactMatch("Fixture Example", phone="+821012345670"),
            kakao_connector.contacts_connector.ContactMatch("Fixture Kim", phone="+821012345671"),
        ]
        result = tool.handler({"to": "fixture", "message": "hello"})
    finally:
        kakao_connector._send_kakao = original
        kakao_connector.contacts_connector.resolve_contact = original_resolve
    if result.ok or calls:
        raise SystemExit(f"Kakao ambiguous contact must not send: {result} {calls}")
    if "Fixture Example" not in result.output or "Fixture Kim" not in result.output:
        raise SystemExit(f"Kakao ambiguous contact should ask which contact: {result.output}")
    handoff = assert_kakao_send_handoff(result.metadata, "Kakao ambiguous contact", status="refused")
    if handoff.get("contact_resolution_status") != "ambiguous" or handoff.get("send_attempted"):
        raise SystemExit(f"Kakao ambiguous handoff wrong: {handoff}")


def test_kakao_unknown_contact_falls_through_to_kakao_search() -> None:
    # A name that isn't in Apple Contacts is still a valid Kakao friend (Kakao
    # searches its own list by name). Instead of refusing, Jarvis proceeds with the
    # raw name typed into Kakao search; the approval gate is the safety net. This is
    # what lets a Kakao-only friend (e.g. "Fixture Example") be reached.
    tool = kakao_connector.make_kakao_tools(_config())[0]
    calls = []
    original = kakao_connector._send_kakao
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector._send_kakao = lambda to, message: calls.append((to, message)) or ""
        kakao_connector.contacts_connector.resolve_contact = lambda _query: []
        result = tool.handler({"to": "fixture example", "message": "hello"})
    finally:
        kakao_connector._send_kakao = original
        kakao_connector.contacts_connector.resolve_contact = original_resolve
    if not result.ok or calls != [("fixture example", "hello")]:
        raise SystemExit(f"Kakao unknown contact should send with the raw name: {result} {calls}")
    handoff = assert_kakao_send_handoff(result.metadata, "Kakao unknown contact", status="outcome_unknown")
    if handoff.get("contact_resolution_status") != "unverified_name" or handoff.get("send_attempted") is not True:
        raise SystemExit(f"Kakao unknown-contact handoff wrong: {handoff}")


def test_kakao_contact_lookup_failure_falls_back_to_exact_kakao_search() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    calls = []
    original = kakao_connector._send_kakao
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector._send_kakao = lambda to, message: calls.append((to, message)) or to
        kakao_connector.contacts_connector.resolve_contact = lambda _query: (_ for _ in ()).throw(
            RuntimeError("Contacts unavailable")
        )
        result = tool.handler({"to": "Fixture Example", "message": "hello"})
    finally:
        kakao_connector._send_kakao = original
        kakao_connector.contacts_connector.resolve_contact = original_resolve
    if not result.ok or calls != [("Fixture Example", "hello")]:
        raise SystemExit(f"Kakao Contacts failure should preserve exact Kakao search: {result} {calls}")
    handoff = assert_kakao_send_handoff(result.metadata, "Kakao unavailable contact lookup", status="outcome_unknown")
    if (
        handoff.get("contact_resolution_status") != "unavailable"
        or handoff.get("contact_lookup_attempted") is not True
        or handoff.get("contact_match_count") != 0
    ):
        raise SystemExit(f"Kakao unavailable-contact handoff wrong: {handoff}")


def test_kakao_explicit_handle_skips_contact_resolution() -> None:
    tool = kakao_connector.make_kakao_tools(_config())[0]
    calls = []
    original = kakao_connector._send_kakao
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    try:
        kakao_connector._send_kakao = lambda to, message: calls.append((to, message)) or ""

        def fail_resolve(_query):
            raise AssertionError("explicit handle should not resolve contacts")

        kakao_connector.contacts_connector.resolve_contact = fail_resolve
        result = tool.handler({"to": "+821012345678", "message": "hello"})
    finally:
        kakao_connector._send_kakao = original
        kakao_connector.contacts_connector.resolve_contact = original_resolve
    if not result.ok or calls != [("+821012345678", "hello")]:
        raise SystemExit(f"Kakao explicit handle should send as-is: {result} {calls}")
    handoff = assert_kakao_send_handoff(result.metadata, "Kakao explicit handle", status="outcome_unknown")
    if handoff.get("contact_resolution_status") != "skipped" or handoff.get("contact_lookup_attempted"):
        raise SystemExit(f"Kakao explicit handle handoff wrong: {handoff}")


def test_kakao_preopened_runtime_binds_exact_mode_without_contacts_or_replay() -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-kakao-preopened-binding-") as temp:
        runtime = make_temp_runtime(Path(temp))
        recipient = "Exact Proof Friend"
        message = "Jarvis V3 Kakao supervised proof 8605"
        command = f"kakao preopened_exact_chat {recipient}: {message}"
        contacts_calls: list[str] = []
        preopened_sends: list[tuple[str, str]] = []
        original_resolve = kakao_connector.contacts_connector.resolve_contact
        original_preopened_send = kakao_connector._send_kakao_preopened_exact_chat
        original_search_send = kakao_connector._send_kakao
        try:
            kakao_connector.contacts_connector.resolve_contact = (
                lambda query: contacts_calls.append(query)
                or (_ for _ in ()).throw(
                    AssertionError("preopened exact-chat mode must not read Contacts")
                )
            )
            kakao_connector._send_kakao = lambda _to, _message: (_ for _ in ()).throw(
                AssertionError("preopened exact-chat mode must not use search sender")
            )
            kakao_connector._send_kakao_preopened_exact_chat = (
                lambda to, body: preopened_sends.append((to, body))
                or {
                    **clipboard_safety.privacy_boundary_metadata(
                        snapshot_attempted=True,
                        snapshot_captured=True,
                        replacement_attempted=True,
                        text_restored=True,
                    ),
                    "kakao_recipient_phase_completed": True,
                    "kakao_message_phase_completed": True,
                    "kakao_enter_pressed": True,
                    "kakao_preopened_navigation_performed": False,
                    "kakao_preopened_search_performed": False,
                }
            )

            held = runtime.handle(command)
            approval_ids = [
                item.metadata.get("approval_id")
                for item in held.tool_results
                if item.metadata.get("requires_confirmation") is True
            ]
            if len(approval_ids) != 1 or type(approval_ids[0]) is not int:
                raise SystemExit(f"Preopened Kakao command did not queue one approval: {held}")
            approval_id = approval_ids[0]
            row = runtime.store.get_pending_approval(approval_id)
            stored_args = json.loads(str(row["planned_args"])) if row is not None else {}
            expected_args = {
                "to": recipient,
                "message": message,
                kakao_connector.KAKAO_TARGET_MODE_KEY:
                    kakao_connector.KAKAO_PREOPENED_EXACT_CHAT_MODE,
                kakao_connector.KAKAO_APPROVAL_BINDING_KEY:
                    kakao_connector._kakao_approval_binding(
                        recipient,
                        message,
                        kakao_connector.KAKAO_PREOPENED_EXACT_CHAT_MODE,
                    ),
            }
            if stored_args != expected_args or contacts_calls or preopened_sends:
                raise SystemExit(
                    "Preopened Kakao approval did not bind exact untouched arguments before "
                    f"execution: {stored_args} {contacts_calls} {preopened_sends}"
                )

            packet = review_pending_runtime_approval(runtime, approval_id)
            if (
                "bound to exact reviewed Kakao recipient, message, and target mode"
                not in packet.output
                or f"- target_mode: {kakao_connector.KAKAO_PREOPENED_EXACT_CHAT_MODE}"
                not in packet.output
            ):
                raise SystemExit(f"Preopened Kakao approval packet is ambiguous: {packet.output}")

            approved = runtime.handle(f"approve approval {approval_id}")
            if (
                approved.verified
                or contacts_calls
                or preopened_sends != [(recipient, message)]
                or "outcome is unknown" not in approved.response.casefold()
                or "do not retry" not in approved.response.casefold()
            ):
                raise SystemExit(
                    "Preopened Kakao exact one-shot execution lost target/no-replay truth: "
                    f"{approved} {contacts_calls} {preopened_sends}"
                )
            run = runtime.store.approved_tool_runs_for_approvals(
                [approval_id],
                limit=10,
            )
            metadata = json.loads(str(run[0]["metadata"])) if len(run) == 1 else {}
            if (
                metadata.get("approved_execution_outcome") != "outcome_unknown"
                or metadata.get("recipient_window_title_verified") is not True
                or metadata.get("one_to_one_chat_verified") is not False
                or metadata.get("delivery_verified") is not False
            ):
                raise SystemExit(f"Preopened Kakao durable truth drifted: {metadata}")

            replay = runtime.handle(f"approve approval {approval_id}")
            if replay.verified or preopened_sends != [(recipient, message)]:
                raise SystemExit(
                    f"Preopened Kakao one-shot approval replayed: {replay} {preopened_sends}"
                )
        finally:
            kakao_connector.contacts_connector.resolve_contact = original_resolve
            kakao_connector._send_kakao_preopened_exact_chat = original_preopened_send
            kakao_connector._send_kakao = original_search_send


def test_kakao_runtime_binds_canonical_recipient_once_and_replays_never() -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-kakao-approval-binding-") as temp:
        runtime = make_temp_runtime(Path(temp))
        raw_recipient = "  Proof Alias  "
        canonical_recipient = "Proof Canonical"
        message = "  Jarvis V3 Kakao approval binding proof  "
        request = "synthetic Kakao approval-binding request"
        runtime.planner = SimpleNamespace(
            plan=lambda _user_input: Plan(
                "Bind one synthetic Kakao request.",
                [
                    PlannedAction(
                        "send_kakao",
                        {"to": raw_recipient, "message": message},
                        "Synthetic approval-binding smoke.",
                    )
                ],
            )
        )

        contact_queries: list[str] = []
        sends: list[tuple[str, str]] = []
        original_resolve = kakao_connector.contacts_connector.resolve_contact
        original_send = kakao_connector._send_kakao
        try:
            def resolve_once(query: str):
                contact_queries.append(query)
                return [
                    kakao_connector.contacts_connector.ContactMatch(
                        canonical_recipient,
                        phone="+821011112222",
                    )
                ]

            kakao_connector.contacts_connector.resolve_contact = resolve_once
            kakao_connector._send_kakao = (
                lambda to, body: sends.append((to, body)) or canonical_recipient
            )
            held = runtime.handle(request)
            approval_ids = [
                item.metadata.get("approval_id")
                for item in held.tool_results
                if item.metadata.get("requires_confirmation") is True
            ]
            if len(approval_ids) != 1 or type(approval_ids[0]) is not int:
                raise SystemExit(f"Kakao request did not queue one exact approval: {held}")
            approval_id = approval_ids[0]
            row = runtime.store.get_pending_approval(approval_id)
            if row is None:
                raise SystemExit("Kakao exact approval row was not stored")
            stored_args = json.loads(str(row["planned_args"]))
            expected_binding = kakao_connector._kakao_approval_binding(
                canonical_recipient,
                message,
            )
            expected_args = {
                "to": canonical_recipient,
                "message": message,
                kakao_connector.KAKAO_TARGET_MODE_KEY:
                    kakao_connector.KAKAO_SEARCH_EXACT_TITLE_MODE,
                kakao_connector.KAKAO_APPROVAL_BINDING_KEY: expected_binding,
            }
            if stored_args != expected_args:
                raise SystemExit(
                    f"Kakao approval did not bind canonical recipient and unchanged message: {stored_args}"
                )
            if contact_queries != [raw_recipient.strip()] or sends:
                raise SystemExit(
                    f"Kakao resolution/send order drifted before approval: {contact_queries} {sends}"
                )

            def contact_drift(_query: str):
                raise AssertionError("Approved Kakao rerun must not resolve Contacts again")

            kakao_connector.contacts_connector.resolve_contact = contact_drift
            runtime.planner = RuleBasedPlanner()
            review_pending_runtime_approval(runtime, approval_id)
            approved = runtime.handle(f"approve approval {approval_id}")
            if approved.verified or sends != [(canonical_recipient, message)]:
                raise SystemExit(
                    "Kakao unverified delivery became top-level verified or changed the exact "
                    f"attempted target/message: {approved} {sends}"
                )
            if (
                "outcome is unknown" not in approved.response.casefold()
                or "do not retry" not in approved.response.casefold()
            ):
                raise SystemExit(
                    f"Kakao unverified delivery missed no-retry unknown guidance: {approved.response}"
                )

            claim = runtime.store.get_approval_execution_claim(approval_id)
            runs = runtime.store.approved_tool_runs_for_approvals([approval_id], limit=10)
            expected_digest = approval_action_digest("send_kakao", expected_args)
            durable_send_metadata = (
                json.loads(str(runs[0]["metadata"])) if len(runs) == 1 else {}
            )
            if (
                claim is None
                or claim["outcome"] != "outcome_unknown"
                or not claim["completed_at"]
                or len(runs) != 1
                or runs[0]["approval_action_digest"] != expected_digest
                or durable_send_metadata.get("recipient_window_title_verified") is not True
                or durable_send_metadata.get("one_to_one_chat_verified") is not False
                or durable_send_metadata.get("send_key_requested") is not True
                or durable_send_metadata.get("delivery_verified") is not False
                or durable_send_metadata.get("confirmation_required") is not True
                or durable_send_metadata.get("operator_confirmation_required") is not True
                or durable_send_metadata.get("execution_outcome_unknown") is not True
                or durable_send_metadata.get("outcome_known") is not False
                or durable_send_metadata.get("side_effect_possible") is not True
                or durable_send_metadata.get("retry_safe") is not False
                or durable_send_metadata.get("automatic_retry_allowed") is not False
                or durable_send_metadata.get("authorizes_retry") is not False
                or durable_send_metadata.get("approved_execution_outcome") != "outcome_unknown"
            ):
                raise SystemExit(
                    "Kakao approved rerun lost one-shot action linkage: "
                    f"claim={dict(claim) if claim is not None else None}, "
                    f"runs={[dict(run) for run in runs]}, metadata={durable_send_metadata}"
                )
            chain = runtime.handle(f"approval chain proof {approval_id}")
            chain_results = [
                item
                for item in chain.tool_results
                if item.tool_name == "approval_chain_proof"
            ]
            if (
                len(chain_results) != 1
                or chain_results[0].metadata.get("valid_execution_proof") is not False
                or chain_results[0].metadata.get("verdict")
                != "APPROVAL_EXECUTION_OUTCOME_UNKNOWN"
                or chain_results[0].metadata.get("approval_execution_outcome_unknown")
                is not True
                or chain_results[0].metadata.get("approval_execution_claim_present") is not True
                or chain_results[0].metadata.get("approval_execution_claim_completed") is not True
                or chain_results[0].metadata.get("linked_runs") != 1
                or chain_results[0].metadata.get("matching_tool_runs") != 1
                or "APPROVAL_CHAIN_PROVEN" in chain_results[0].output
                or "do not replay" not in chain_results[0].output.casefold()
            ):
                raise SystemExit(
                    "Kakao attempted execution was not linked without overclaiming delivery: "
                    f"{chain}"
                )

            replay = runtime.handle(f"approve approval {approval_id}")
            if replay.verified or sends != [(canonical_recipient, message)]:
                raise SystemExit(
                    f"Kakao one-shot approval replayed its send: {replay} {sends}"
                )
        finally:
            kakao_connector.contacts_connector.resolve_contact = original_resolve
            kakao_connector._send_kakao = original_send


def test_kakao_runtime_refuses_unreviewable_raw_arguments_before_approval() -> None:
    cases = (
        ("recipient-too-long", "recipient_too_long", {"to": "x" * (kakao_connector.MAX_RECIPIENT_CHARS + 1), "message": "ok"}),
        ("message-too-long", "message_too_long", {"to": "Proof Name", "message": "x" * (kakao_connector.MAX_MESSAGE_CHARS + 1)}),
        ("blank-recipient", "missing_recipient", {"to": "   ", "message": "ok"}),
        ("blank-message", "missing_message", {"to": "Proof Name", "message": "   "}),
        ("recipient-control", "recipient_has_control_characters", {"to": "Proof\nName", "message": "ok"}),
        ("message-control", "message_has_control_characters", {"to": "Proof Name", "message": "proof\ttext"}),
        ("at-handle", "recipient_handle_not_allowed", {"to": "@proof_handle", "message": "ok"}),
        ("phone-handle", "recipient_handle_not_allowed", {"to": "+821055501234", "message": "ok"}),
        ("email-handle", "recipient_handle_not_allowed", {"to": "proof@example.test", "message": "ok"}),
        (
            "target-mode-near-miss",
            "target_mode_invalid",
            {
                "to": "Proof Name",
                "message": "ok",
                kakao_connector.KAKAO_TARGET_MODE_KEY: "PREOPENED_EXACT_CHAT",
            },
        ),
        (
            "preopened-surrounding-space",
            "preopened_recipient_not_exact",
            {
                "to": "Proof Name ",
                "message": "ok",
                kakao_connector.KAKAO_TARGET_MODE_KEY:
                    kakao_connector.KAKAO_PREOPENED_EXACT_CHAT_MODE,
            },
        ),
    )
    original_resolve = kakao_connector.contacts_connector.resolve_contact
    original_send = kakao_connector._send_kakao
    try:
        kakao_connector.contacts_connector.resolve_contact = lambda _query: (_ for _ in ()).throw(
            AssertionError("Invalid raw Kakao arguments must stop before Contacts")
        )
        kakao_connector._send_kakao = lambda _to, _message: (_ for _ in ()).throw(
            AssertionError("Invalid raw Kakao arguments must stop before send")
        )
        for label, expected_reason, planned_args in cases:
            with tempfile.TemporaryDirectory(prefix=f"jarvis-kakao-invalid-{label}-") as temp:
                runtime = make_temp_runtime(Path(temp))
                runtime.planner = SimpleNamespace(
                    plan=lambda _user_input, planned_args=planned_args: Plan(
                        "Reject an unreviewable synthetic Kakao request.",
                        [
                            PlannedAction(
                                "send_kakao",
                                dict(planned_args),
                                "Synthetic invalid-argument smoke.",
                            )
                        ],
                    )
                )
                result = runtime.handle(f"synthetic Kakao invalid case {label}")
                if result.verified or len(result.tool_results) != 1:
                    raise SystemExit(f"Kakao {label} did not fail closed: {result}")
                item = result.tool_results[0]
                if (
                    item.metadata.get("requires_confirmation") is not False
                    or item.metadata.get("executed_handler") is not False
                    or item.metadata.get("handler_invoked") is not False
                    or item.metadata.get("failure_kind")
                    != "kakao_approval_arguments_invalid"
                    or item.metadata.get("reason") != expected_reason
                    or runtime.store.list_pending_approvals(limit=10)
                ):
                    raise SystemExit(
                        f"Kakao {label} queued authority or reached its handler: {item}"
                    )
    finally:
        kakao_connector.contacts_connector.resolve_contact = original_resolve
        kakao_connector._send_kakao = original_send


def test_instagram_tool_is_high_risk_and_validates_args() -> None:
    tool = instagram_connector.make_instagram_tools(_config())[0]
    if tool.name != "send_instagram_dm" or tool.risk != RiskLevel.HIGH_RISK:
        raise SystemExit(f"Instagram tool risk registration is wrong: {tool}")
    fields = {field.name: field for field in (tool.argument_contract.fields if tool.argument_contract else ())}
    if set(fields) != {"to", "message", "allow_new_recipient"}:
        raise SystemExit(f"Instagram argument contract fields drifted: {fields}")
    if not fields["to"].required or not fields["message"].required or fields["allow_new_recipient"].required:
        raise SystemExit(f"Instagram argument contract required flags drifted: {fields}")
    missing = tool.handler({"message": "hi"})
    if missing.ok or "recipient" not in missing.output.lower():
        raise SystemExit(f"Instagram validation should reject missing recipient: {missing}")
    if missing.metadata.get("executes_side_effect"):
        raise SystemExit(f"Instagram validation must not claim a send occurred: {missing.metadata}")
    if missing.metadata.get("message_chars") != 2 or missing.metadata.get("to"):
        raise SystemExit(f"Instagram missing-recipient metadata should preserve message length only: {missing.metadata}")
    if "automation" not in missing.metadata.get("automation_warning", "").lower():
        raise SystemExit(f"Instagram metadata should document automation risk: {missing.metadata}")
    missing_handoff = assert_instagram_dm_handoff(missing.metadata, "Instagram missing-recipient", status="refused")
    if missing_handoff.get("reason") != "missing_recipient" or missing_handoff.get("send_attempted"):
        raise SystemExit(f"Instagram missing-recipient handoff should be inert refusal: {missing_handoff}")
    missing_message = tool.handler({"to": "@example_user"})
    if missing_message.ok or "message" not in missing_message.output.lower():
        raise SystemExit(f"Instagram validation should reject missing message: {missing_message}")
    missing_message_handoff = assert_instagram_dm_handoff(missing_message.metadata, "Instagram missing-message", status="refused")
    if missing_message_handoff.get("reason") != "missing_message" or missing_message_handoff.get("send_attempted"):
        raise SystemExit(f"Instagram missing-message handoff should be inert refusal: {missing_message_handoff}")


def test_instagram_approval_binds_exact_username_and_rejects_target_drift() -> None:
    tool = instagram_connector.make_instagram_tools(_config())[0]
    resolver = tool.approval_argument_resolver
    if resolver is None or tool.approval_argument_contract is None:
        raise SystemExit("Instagram send must bind approval arguments before queueing")

    original_resolve = contacts_connector.resolve_contact
    original_send = instagram_connector._send_instagram_dm
    sends: list[tuple[str, str, bool]] = []
    try:
        contacts_connector.resolve_contact = lambda _query: (_ for _ in ()).throw(
            AssertionError("Instagram approval identity must not be retargeted through Contacts")
        )  # type: ignore[assignment]
        alias = resolver({"to": "Example Contact", "message": "hello"})
        if (
            not isinstance(alias, ToolResult)
            or alias.ok
            or alias.metadata.get("reason") != "exact_username_required_before_approval"
            or alias.metadata.get("executed_handler") is not False
            or alias.metadata.get("handler_invoked") is not False
        ):
            raise SystemExit(f"Instagram display-name approval did not fail closed: {alias}")

        resolution = resolver({"to": "@exampleproof8605", "message": "hello"})
        if not isinstance(resolution, ApprovalArgumentResolution):
            raise SystemExit(f"Exact Instagram username was not approval-bound: {resolution}")
        expected_args = {
            "to": "@exampleproof8605",
            "message": "hello",
            "allow_new_recipient": False,
            instagram_connector.INSTAGRAM_APPROVAL_BINDING_KEY:
                instagram_connector._instagram_approval_binding(
                    "@exampleproof8605", "hello", False
                ),
        }
        if resolution.args != expected_args:
            raise SystemExit(f"Instagram approval binding changed reviewed arguments: {resolution}")
        if not validate_tool_arguments(tool, resolution.args, approved=True).valid:
            raise SystemExit("Instagram approval-bound arguments missed the execution contract")
        if validate_tool_arguments(tool, resolution.args).valid:
            raise SystemExit("Instagram callers could inject the private approval binding")

        instagram_connector._send_instagram_dm = (
            lambda to, message, **kwargs: sends.append(
                (to, message, kwargs.get("allow_new_recipient") is True)
            ) or to
        )
        accepted = tool.handler(dict(resolution.args))
        if not accepted.ok or sends != [("@exampleproof8605", "hello", False)]:
            raise SystemExit(f"Bound Instagram execution changed its exact target: {accepted} {sends}")
        if accepted.metadata.get("contact_lookup_attempted") is not False:
            raise SystemExit(f"Bound Instagram execution re-read Contacts: {accepted.metadata}")

        for key, value in (
            ("to", "@different_account"),
            ("message", "changed message"),
            ("allow_new_recipient", True),
        ):
            drifted = dict(resolution.args)
            drifted[key] = value
            stopped = tool.handler(drifted)
            if (
                stopped.ok
                or stopped.metadata.get("reason") != "approval_binding_invalid"
                or stopped.metadata.get("send_attempted")
                or len(sends) != 1
            ):
                raise SystemExit(
                    f"Instagram approved target/message drift reached the sender ({key}): {stopped} {sends}"
                )
    finally:
        contacts_connector.resolve_contact = original_resolve  # type: ignore[assignment]
        instagram_connector._send_instagram_dm = original_send


def test_instagram_runtime_approval_is_exact_and_one_shot() -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-instagram-approval-binding-") as temp:
        runtime = make_temp_runtime(Path(temp))
        recipient = "@exampleproof8605"
        message = "Jarvis V3 exact Instagram approval proof"
        runtime.planner = SimpleNamespace(
            plan=lambda _user_input: Plan(
                "Bind one exact Instagram send.",
                [
                    PlannedAction(
                        "send_instagram_dm",
                        {"to": recipient, "message": message},
                        "Synthetic exact-username approval smoke.",
                    )
                ],
            )
        )
        original_resolve = contacts_connector.resolve_contact
        original_send = instagram_connector._send_instagram_dm
        sends: list[tuple[str, str, bool]] = []
        try:
            contacts_connector.resolve_contact = lambda _query: (_ for _ in ()).throw(
                AssertionError("Instagram runtime approval must not consult Contacts")
            )  # type: ignore[assignment]
            instagram_connector._send_instagram_dm = (
                lambda to, body, **kwargs: sends.append(
                    (to, body, kwargs.get("allow_new_recipient") is True)
                ) or to
            )
            held = runtime.handle("synthetic exact Instagram approval request")
            approval_ids = [
                item.metadata.get("approval_id")
                for item in held.tool_results
                if item.metadata.get("requires_confirmation") is True
            ]
            if len(approval_ids) != 1 or type(approval_ids[0]) is not int or sends:
                raise SystemExit(f"Instagram exact request did not queue one inert approval: {held} {sends}")
            approval_id = approval_ids[0]
            row = runtime.store.get_pending_approval(approval_id)
            stored_args = json.loads(str(row["planned_args"])) if row is not None else {}
            expected_args = {
                "to": recipient,
                "message": message,
                "allow_new_recipient": False,
                instagram_connector.INSTAGRAM_APPROVAL_BINDING_KEY:
                    instagram_connector._instagram_approval_binding(
                        recipient, message, False
                    ),
            }
            if stored_args != expected_args:
                raise SystemExit(f"Instagram queued approval was not exact-target bound: {stored_args}")
            packet = review_pending_runtime_approval(runtime, approval_id)
            if "bound to exact reviewed Instagram @username" not in packet.output:
                raise SystemExit(f"Instagram approval packet hid its identity binding: {packet.output}")

            runtime.planner = RuleBasedPlanner()
            approved = runtime.handle(f"approve approval {approval_id}")
            if sends != [(recipient, message, False)]:
                raise SystemExit(f"Instagram approved execution drifted or did not run once: {approved} {sends}")
            rerun_receipts = [
                item.metadata.get("approved_rerun_result")
                for item in approved.tool_results
                if item.tool_name == "approve_pending_approval"
            ]
            if (
                len(rerun_receipts) != 1
                or not isinstance(rerun_receipts[0], dict)
                or "recipient delivery is not confirmed"
                not in str(rerun_receipts[0].get("output_preview") or "")
            ):
                raise SystemExit(f"Instagram approved sender receipt overclaimed delivery: {approved}")
            runs = runtime.store.approved_tool_runs_for_approvals([approval_id], limit=10)
            metadata = json.loads(str(runs[0]["metadata"])) if len(runs) == 1 else {}
            if (
                len(runs) != 1
                or metadata.get("approved_execution_outcome") != "succeeded"
            ):
                raise SystemExit(f"Instagram durable one-shot linkage drifted: {runs} {metadata}")

            replay = runtime.handle(f"approve approval {approval_id}")
            if sends != [(recipient, message, False)] or replay.verified:
                raise SystemExit(f"Instagram one-shot approval replayed: {replay} {sends}")
        finally:
            contacts_connector.resolve_contact = original_resolve  # type: ignore[assignment]
            instagram_connector._send_instagram_dm = original_send


def test_instagram_tool_uses_mocked_sender() -> None:
    tool = instagram_connector.make_instagram_tools(_config())[0]
    calls = []
    original = instagram_connector._send_instagram_dm
    original_resolve = contacts_connector.resolve_contact
    try:
        contacts_connector.resolve_contact = lambda _query: [contacts_connector.ContactMatch(name="Fixture Example")]  # type: ignore[assignment]
        instagram_connector._send_instagram_dm = (
            lambda to, message, **kwargs: calls.append((to, message, kwargs.get("allow_new_recipient"))) or to
        )
        result = tool.handler({"to": "Fixture", "message": "hello"})
    finally:
        instagram_connector._send_instagram_dm = original
        contacts_connector.resolve_contact = original_resolve  # type: ignore[assignment]
    if not result.ok or calls != [("Fixture Example", "hello", False)]:
        raise SystemExit(f"Instagram mocked send failed: {result} {calls}")
    if not result.metadata.get("executes_side_effect") or not result.metadata.get("external_side_effect"):
        raise SystemExit(f"Instagram successful send metadata should mark the side effect: {result.metadata}")
    handoff = assert_instagram_dm_handoff(result.metadata, "Instagram mocked send", status="sender_confirmed")
    if handoff.get("send_attempted") is not True or not handoff.get("boundaries", {}).get("controls_computer"):
        raise SystemExit(f"Instagram successful send handoff should preserve GUI-send attempt: {handoff}")
    if result.metadata.get("to") != "Fixture" or result.metadata.get("resolved_to") != "Fixture Example":
        raise SystemExit(f"Instagram should preserve original and resolved recipients: {result.metadata}")
    if result.metadata.get("recipient_policy") != "existing_or_followed":
        raise SystemExit(f"Instagram default recipient policy metadata drifted: {result.metadata}")
    if (
        result.metadata.get("sender_side_send_confirmed") is not True
        or result.metadata.get("recipient_delivery_confirmed") is not False
        or result.metadata.get("delivery_confirmed") is not False
        or result.metadata.get("delivery_evidence") != "sender_side_only"
        or result.metadata.get("recipient_confirmation_required") is not True
        or "recipient delivery is not confirmed" not in result.output
        or "Instagram DM sent" in result.output
    ):
        raise SystemExit(f"Instagram sender receipt overclaimed recipient delivery: {result}")


def test_instagram_ambiguous_contact_refuses_without_browser_send() -> None:
    tool = instagram_connector.make_instagram_tools(_config())[0]
    original_resolve = contacts_connector.resolve_contact
    original_send = instagram_connector._send_instagram_dm
    try:
        contacts_connector.resolve_contact = lambda _query: [  # type: ignore[assignment]
            contacts_connector.ContactMatch(name="Fixture Example"),
            contacts_connector.ContactMatch(name="Fixture Kim"),
        ]
        instagram_connector._send_instagram_dm = lambda _to, _message, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Instagram send should not start for an ambiguous contact")
        )
        result = tool.handler({"to": "Fixture", "message": "hello"})
    finally:
        contacts_connector.resolve_contact = original_resolve  # type: ignore[assignment]
        instagram_connector._send_instagram_dm = original_send
    if result.ok or result.metadata.get("send_attempted") or result.metadata.get("contact_resolution_status") != "ambiguous":
        raise SystemExit(f"Instagram ambiguous contact should stop before browser automation: {result}")
    if "Fixture Example" not in result.output or "Fixture Kim" not in result.output:
        raise SystemExit(f"Instagram ambiguous contact should ask which contact: {result.output}")
    handoff = assert_instagram_dm_handoff(result.metadata, "Instagram ambiguous contact", status="refused")
    if handoff.get("send_attempted") or handoff.get("contact_resolution_status") != "ambiguous":
        raise SystemExit(f"Instagram ambiguous handoff wrong: {handoff}")


def test_instagram_preserves_a_verified_open_thread() -> None:
    original_guard = instagram_connector._guard_no_active_instagram_call
    original_focus = instagram_connector._focus_instagram_tab
    original_match = instagram_connector._open_thread_matches
    original_navigate = instagram_connector._navigate_instagram_inbox
    try:
        calls = []
        instagram_connector._guard_no_active_instagram_call = lambda: calls.append("call-guard")  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = lambda: calls.append("focus")  # type: ignore[assignment]
        instagram_connector._open_thread_matches = lambda recipient: calls.append(recipient) or True  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = lambda: (_ for _ in ()).throw(
            AssertionError("A verified manually opened DM must not be replaced with the inbox")
        )  # type: ignore[assignment]
        instagram_connector.open_instagram_thread("Fixture Example")
    finally:
        instagram_connector._guard_no_active_instagram_call = original_guard  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = original_focus  # type: ignore[assignment]
        instagram_connector._open_thread_matches = original_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = original_navigate  # type: ignore[assignment]
    if calls != ["call-guard", "focus", "Fixture Example"]:
        raise SystemExit(f"Instagram verified-thread preservation order drifted: {calls}")


def test_instagram_active_call_stops_before_focus_navigation_or_search() -> None:
    original_guard = instagram_connector._guard_no_active_instagram_call
    original_focus = instagram_connector._focus_instagram_tab
    original_navigate = instagram_connector._navigate_instagram_inbox
    original_search = instagram_connector._start_instagram_thread_search
    calls: list[str] = []
    try:
        def blocked_guard() -> None:
            calls.append("call-guard")
            raise instagram_connector.InstagramWebError("instagram_active_call")

        instagram_connector._guard_no_active_instagram_call = blocked_guard  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = lambda: calls.append("focus")  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = lambda: calls.append("navigate")  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = lambda _recipient: calls.append("search") or "SEARCHING"  # type: ignore[assignment]
        try:
            instagram_connector.open_instagram_thread("Fixture Example")
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_active_call":
                raise SystemExit(f"Instagram active-call guard stage drifted: {exc.stage}")
        else:
            raise SystemExit("Instagram active-call guard should stop before page control")
    finally:
        instagram_connector._guard_no_active_instagram_call = original_guard  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = original_focus  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = original_navigate  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = original_search  # type: ignore[assignment]
    if calls != ["call-guard"]:
        raise SystemExit(f"Instagram changed browser state after detecting an active call: {calls}")


def test_instagram_call_guard_checks_every_tab_and_fails_closed() -> None:
    wrapper = instagram_connector._CHROME_INSTAGRAM_CALL_GUARD_WRAPPER
    call_js = instagram_connector._ACTIVE_INSTAGRAM_CALL_JS
    required_fragments = (
        "repeat with w in windows",
        "repeat with t in tabs of w",
        'URL of t contains "instagram.com"',
        'if callState is "ACTIVE" then return "ACTIVE"',
    )
    for fragment in required_fragments:
        if fragment not in wrapper:
            raise SystemExit(f"Instagram call guard missed all-tab check {fragment!r}")
    for label in ("end call", "leave call", "hang up", "통화 종료", "전화 끊기"):
        if label not in call_js:
            raise SystemExit(f"Instagram call guard missed visible control label {label!r}")

    original_run = instagram_connector.subprocess.run
    try:
        instagram_connector.subprocess.run = lambda *_args, **_kwargs: SimpleNamespace(  # type: ignore[assignment]
            returncode=0,
            stdout="ACTIVE\n",
            stderr="",
        )
        try:
            instagram_connector._guard_no_active_instagram_call()
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_active_call":
                raise SystemExit(f"Instagram active-call subprocess guard stage drifted: {exc.stage}")
        else:
            raise SystemExit("Instagram active-call subprocess guard should stop")

        instagram_connector.subprocess.run = lambda *_args, **_kwargs: SimpleNamespace(  # type: ignore[assignment]
            returncode=0,
            stdout="UNKNOWN\n",
            stderr="",
        )
        try:
            instagram_connector._guard_no_active_instagram_call()
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_call_state_unavailable":
                raise SystemExit(f"Instagram unknown call-state guard stage drifted: {exc.stage}")
        else:
            raise SystemExit("Instagram unknown call state must fail closed")
    finally:
        instagram_connector.subprocess.run = original_run  # type: ignore[assignment]


def test_instagram_searches_beyond_recent_threads_before_refusing() -> None:
    original_guard = instagram_connector._guard_no_active_instagram_call
    original_focus = instagram_connector._focus_instagram_tab
    original_match = instagram_connector._open_thread_matches
    original_navigate = instagram_connector._navigate_instagram_inbox
    original_search = instagram_connector._start_instagram_thread_search
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    calls: list[str] = []
    find_calls = 0
    header_checks = 0

    def fake_js(script: str, timeout: int = 20) -> str:
        nonlocal find_calls
        del timeout
        if "input[name=\"username\"]" in script:
            calls.append("ready")
            return "ready"
        if "var exact = unique" in script:
            find_calls += 1
            calls.append(f"find-{find_calls}")
            return "MISS" if find_calls == 1 else "CLICKED:Fixture Example"
        if "location.pathname.indexOf('/direct/t/')" in script:
            calls.append("opened")
            return "OPEN"
        raise AssertionError(f"Unexpected Instagram JS during search smoke: {script[:80]}")

    def fake_match(_recipient: str) -> bool:
        nonlocal header_checks
        header_checks += 1
        return header_checks > 1

    try:
        instagram_connector._guard_no_active_instagram_call = lambda: calls.append("call-guard")  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = lambda: calls.append("focus")  # type: ignore[assignment]
        instagram_connector._open_thread_matches = fake_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = lambda: calls.append("inbox")  # type: ignore[assignment]
        def fake_search(recipient: str) -> str:
            calls.append(f"search-{recipient}")
            return "NO_SEARCH_INPUT" if recipient == "" else "SEARCHING"

        instagram_connector._start_instagram_thread_search = fake_search  # type: ignore[assignment]
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        instagram_connector.open_instagram_thread("Fixture Example")
    finally:
        instagram_connector._guard_no_active_instagram_call = original_guard  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = original_focus  # type: ignore[assignment]
        instagram_connector._open_thread_matches = original_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = original_navigate  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = original_search  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    expected = [
        "call-guard",
        "focus",
        "inbox",
        "ready",
        "find-1",
        "search-",
        "search-",
        "search-",
        "search-",
        "search-",
        "search-",
        "search-Fixture Example",
        "find-2",
        "opened",
    ]
    if calls != expected:
        raise SystemExit(f"Instagram exact-search fallback order drifted: {calls}")
    if header_checks != 2:
        raise SystemExit(f"Instagram opened-thread recipient verification drifted: {header_checks}")


def test_instagram_checks_empty_search_overlay_before_text_query() -> None:
    original_guard = instagram_connector._guard_no_active_instagram_call
    original_focus = instagram_connector._focus_instagram_tab
    original_match = instagram_connector._open_thread_matches
    original_navigate = instagram_connector._navigate_instagram_inbox
    original_search = instagram_connector._start_instagram_thread_search
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    calls: list[str] = []
    find_calls = 0
    header_checks = 0

    def fake_js(script: str, timeout: int = 20) -> str:
        nonlocal find_calls
        del timeout
        if "input[name=\"username\"]" in script:
            return "ready"
        if "var exact = unique" in script:
            find_calls += 1
            return "MISS" if find_calls == 1 else "CLICKED:Fixture Example"
        if "location.pathname.indexOf('/direct/t/')" in script:
            return "OPEN"
        raise AssertionError(f"Unexpected Instagram JS during empty-search smoke: {script[:80]}")

    def fake_match(_recipient: str) -> bool:
        nonlocal header_checks
        header_checks += 1
        return header_checks > 1

    try:
        instagram_connector._guard_no_active_instagram_call = lambda: None  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = lambda: None  # type: ignore[assignment]
        instagram_connector._open_thread_matches = fake_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = lambda: None  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = (
            lambda recipient: calls.append(f"search-{recipient}") or "SEARCHING"
        )  # type: ignore[assignment]
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        instagram_connector.open_instagram_thread("Fixture Example")
    finally:
        instagram_connector._guard_no_active_instagram_call = original_guard  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = original_focus  # type: ignore[assignment]
        instagram_connector._open_thread_matches = original_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = original_navigate  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = original_search  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if calls != ["search-"]:
        raise SystemExit(f"Instagram should select the exact empty-search suggestion before a text query: {calls}")


def test_instagram_search_result_advances_new_message_overlay() -> None:
    original_guard = instagram_connector._guard_no_active_instagram_call
    original_focus = instagram_connector._focus_instagram_tab
    original_match = instagram_connector._open_thread_matches
    original_navigate = instagram_connector._navigate_instagram_inbox
    original_search = instagram_connector._start_instagram_thread_search
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    states = iter(("LIST", "OPEN", "OPEN"))
    scripts: list[str] = []
    header_checks = 0

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        scripts.append(script)
        if "input[name=\"username\"]" in script:
            return "ready"
        if "var exact = unique" in script:
            return "CLICKED:Fixture Example"
        if "location.pathname.indexOf('/direct/t/')" in script:
            return next(states)
        if "return t==='Chat'||t==='Next'" in script:
            return "CLICKED"
        raise AssertionError(f"Unexpected Instagram JS during overlay advance smoke: {script[:80]}")

    def fake_match(_recipient: str) -> bool:
        nonlocal header_checks
        header_checks += 1
        return header_checks > 2

    try:
        instagram_connector._guard_no_active_instagram_call = lambda: None  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = lambda: None  # type: ignore[assignment]
        instagram_connector._open_thread_matches = fake_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = lambda: None  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = lambda _recipient: "SEARCHING"  # type: ignore[assignment]
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        matched = instagram_connector.open_instagram_thread("Fixture Example")
    finally:
        instagram_connector._guard_no_active_instagram_call = original_guard  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = original_focus  # type: ignore[assignment]
        instagram_connector._open_thread_matches = original_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = original_navigate  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = original_search  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if matched != "Fixture Example" or header_checks != 3:
        raise SystemExit(f"Instagram overlay advance lost exact recipient verification: {matched!r} {header_checks}")
    advances = [script for script in scripts if "return t==='Chat'||t==='Next'" in script]
    if len(advances) != 1:
        raise SystemExit(f"Instagram should advance the selected new-message overlay exactly once: {len(advances)}")


def test_instagram_search_activates_overlay_before_setting_query() -> None:
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    scripts: list[str] = []
    try:
        instagram_connector._chrome_js = (  # type: ignore[assignment]
            lambda script, timeout=20: scripts.append(script)
            or ("SEARCH_ACTIVATED" if "var el=input" in script else "SEARCHING")
        )
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        state = instagram_connector._start_instagram_thread_search("")
    finally:
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if state != "SEARCHING" or len(scripts) != 2:
        raise SystemExit(f"Instagram empty-search activation call drifted: {state!r} {scripts}")
    activation_script, value_script = scripts
    for fragment in ("var el=input", "'pointerdown'", "input.focus()", "SEARCH_ACTIVATED"):
        if fragment not in activation_script:
            raise SystemExit(f"Instagram search activation missed {fragment!r}: {activation_script}")
    if "setter.call(input" in activation_script:
        raise SystemExit("Instagram search must not set the soon-to-be-detached input")
    for fragment in ("setter.call(input, \"\")", "new InputEvent('input'"):
        if fragment not in value_script:
            raise SystemExit(f"Instagram search value update missed {fragment!r}: {value_script}")


def test_instagram_rejects_group_title_for_individual_recipient() -> None:
    finder = instagram_connector._instagram_thread_finder_js("Fixture")
    if "indexOf(q+' ')" in finder:
        raise SystemExit("Instagram finder still permits unsafe first-name prefix expansion")
    if instagram_connector._clicked_instagram_name("CLICKED:Fixture Example and Fixture Alternate", "Fixture"):
        raise SystemExit("Instagram accepted a group title for an individual recipient")
    if instagram_connector._clicked_instagram_name("CLICKED:Fixture Example", "Fixture Example") != "Fixture Example":
        raise SystemExit("Instagram exact individual recipient verification drifted")
    if instagram_connector._clicked_instagram_name("CLICKED:fixture_yoo", "@fixture_yoo") != "@fixture_yoo":
        raise SystemExit("Instagram exact handle identity marker was lost after account selection")
    if instagram_connector._clicked_instagram_name("CLICKED", "Fixture Example"):
        raise SystemExit("Instagram accepted a click result without verified recipient identity")

    followed_finder = instagram_connector._instagram_thread_finder_js("Fixture Example", require_following=True)
    required_fragments = (
        "var requireFollowing = true",
        "candidates[0].node.innerText",
        'rowTokens.indexOf(label)>=0',
        "return 'NOT_FOLLOWING'",
        '"following"',
        '"팔로잉"',
    )
    for fragment in required_fragments:
        if fragment not in followed_finder:
            raise SystemExit(f"Instagram followed-recipient selector missed row-scoped guard {fragment!r}")
    if "document.body.innerText" in followed_finder:
        raise SystemExit("Instagram followed-recipient selector must not trust a page-wide Following label")


def test_instagram_existing_search_ignores_account_results_and_group_rows() -> None:
    finder = instagram_connector._instagram_thread_finder_js(
        "Fixture Example",
        existing_only=True,
    )
    required_fragments = (
        "var existingOnly = true",
        "s.closest('a[href*=\"/direct/t/\"]')",
        '"messages"',
        '"메시지"',
        '"more accounts"',
        '"계정 더 보기"',
        "function inMessagesSection(s)",
        "function inAccountSection(s)",
        "function inUnfilteredInbox()",
        "location.pathname.indexOf('/direct/inbox/')!==0",
        "!(input.value||'').trim()",
        "!messageHeadings.length && !accountHeadings.length",
        "return y>=start&&y<=top",
        "function isMessageSearchSummary(lines)",
        r"/^\d+\s+matched messages?$/",
        r"/메시지/.test(detail)",
        r"/일치/.test(detail)",
        "inUnfilteredInbox()||(inMessagesSection(s)&&!isMessageSearchSummary(lines))",
        "account:inAccountSection(s)",
        "normalized(e.primary)===q",
        "var accounts = exact.filter(function(e) { return e.account; })",
        "var candidates = existingOnly ? existing : (accountHeadings.length ? accounts",
    )
    for fragment in required_fragments:
        if fragment not in finder:
            raise SystemExit(
                f"Instagram existing-thread selector missed conversation-only guard {fragment!r}"
            )
    if "var candidates = exact;" in finder:
        raise SystemExit(
            "Instagram existing-thread search still mixes existing conversations with account results"
        )
    if "existing:!!thread||inMessagesSection(s)" in finder:
        raise SystemExit(
            "Instagram existing-thread search still treats matched-message summaries as conversations"
        )

    new_recipient_finder = instagram_connector._instagram_thread_finder_js("@newperson")
    if "var existingOnly = false" not in new_recipient_finder:
        raise SystemExit(
            "Instagram explicit new-recipient selector unexpectedly requires an existing thread"
        )
    if "accountHeadings.length ? accounts" not in new_recipient_finder:
        raise SystemExit(
            "Instagram new-recipient selector still mixes account rows with message-search matches"
        )


def test_instagram_new_recipient_requires_explicit_mode() -> None:
    original_guard = instagram_connector._guard_no_active_instagram_call
    original_focus = instagram_connector._focus_instagram_tab
    original_match = instagram_connector._open_thread_matches
    original_navigate = instagram_connector._navigate_instagram_inbox
    original_search = instagram_connector._start_instagram_thread_search
    original_new = instagram_connector._open_new_instagram_recipient
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    new_calls: list[tuple[str, bool]] = []

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        if "input[name=\"username\"]" in script:
            return "ready"
        if "var exact = unique" in script:
            return "MISS"
        raise AssertionError(f"Unexpected Instagram JS during recipient-policy smoke: {script[:100]}")

    try:
        instagram_connector._guard_no_active_instagram_call = lambda: None  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = lambda: None  # type: ignore[assignment]
        instagram_connector._open_thread_matches = lambda _recipient: False  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = lambda: None  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = lambda _recipient: "SEARCHING"  # type: ignore[assignment]
        instagram_connector._open_new_instagram_recipient = (  # type: ignore[assignment]
            lambda recipient, *, require_following: new_calls.append((recipient, require_following)) or "New Person"
        )
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]

        try:
            instagram_connector.open_instagram_thread("New Person")
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_recipient_not_known":
                raise SystemExit(f"Default Instagram unknown-recipient stage drifted: {exc.stage}")
        else:
            raise SystemExit("Default Instagram send must refuse a recipient outside existing conversations")
        if new_calls:
            raise SystemExit(f"Default Instagram send entered new-recipient mode: {new_calls}")

        matched = instagram_connector.open_instagram_thread("New Person", allow_new_recipient=True)
        if matched != "New Person" or new_calls != [("New Person", False)]:
            raise SystemExit(f"Explicit Instagram new-recipient mode did not use its separate path: {matched!r} {new_calls}")
    finally:
        instagram_connector._guard_no_active_instagram_call = original_guard  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = original_focus  # type: ignore[assignment]
        instagram_connector._open_thread_matches = original_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = original_navigate  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = original_search  # type: ignore[assignment]
        instagram_connector._open_new_instagram_recipient = original_new  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]


def test_instagram_ambiguous_page_match_never_enters_new_recipient_mode() -> None:
    original_guard = instagram_connector._guard_no_active_instagram_call
    original_focus = instagram_connector._focus_instagram_tab
    original_match = instagram_connector._open_thread_matches
    original_navigate = instagram_connector._navigate_instagram_inbox
    original_search = instagram_connector._start_instagram_thread_search
    original_new = instagram_connector._open_new_instagram_recipient
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    try:
        instagram_connector._guard_no_active_instagram_call = lambda: None  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = lambda: None  # type: ignore[assignment]
        instagram_connector._open_thread_matches = lambda _recipient: False  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = lambda: None  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = lambda _recipient: "SEARCHING"  # type: ignore[assignment]
        instagram_connector._open_new_instagram_recipient = lambda _recipient, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Ambiguous Instagram result must not enter new-recipient mode")
        )  # type: ignore[assignment]
        instagram_connector._chrome_js = lambda script, timeout=20: (
            "ready" if "input[name=\"username\"]" in script else "AMBIGUOUS"
        )  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        try:
            instagram_connector.open_instagram_thread("Fixture", allow_new_recipient=True)
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_thread_not_found":
                raise SystemExit(f"Ambiguous Instagram result stage drifted: {exc.stage}")
        else:
            raise SystemExit("Ambiguous Instagram result must stop before typing")
    finally:
        instagram_connector._guard_no_active_instagram_call = original_guard  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = original_focus  # type: ignore[assignment]
        instagram_connector._open_thread_matches = original_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = original_navigate  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = original_search  # type: ignore[assignment]
        instagram_connector._open_new_instagram_recipient = original_new  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]


def test_instagram_default_fallback_requires_followed_account() -> None:
    original_guard = instagram_connector._guard_no_active_instagram_call
    original_focus = instagram_connector._focus_instagram_tab
    original_match = instagram_connector._open_thread_matches
    original_navigate = instagram_connector._navigate_instagram_inbox
    original_search = instagram_connector._start_instagram_thread_search
    original_new = instagram_connector._open_new_instagram_recipient
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    new_calls: list[tuple[str, bool]] = []

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        if "input[name=\"username\"]" in script:
            return "ready"
        if "var exact = unique" in script:
            return "MISS"
        raise AssertionError(f"Unexpected Instagram JS during followed-recipient fallback smoke: {script[:100]}")

    try:
        instagram_connector._guard_no_active_instagram_call = lambda: None  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = lambda: None  # type: ignore[assignment]
        instagram_connector._open_thread_matches = lambda _recipient: False  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = lambda: None  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = lambda _recipient: "SEARCHING"  # type: ignore[assignment]
        instagram_connector._open_new_instagram_recipient = (  # type: ignore[assignment]
            lambda recipient, *, require_following: new_calls.append((recipient, require_following)) or recipient
        )
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]

        matched = instagram_connector.open_instagram_thread(
            "Fixture Example",
            allow_followed_recipient=True,
        )
    finally:
        instagram_connector._guard_no_active_instagram_call = original_guard  # type: ignore[assignment]
        instagram_connector._focus_instagram_tab = original_focus  # type: ignore[assignment]
        instagram_connector._open_thread_matches = original_match  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox = original_navigate  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = original_search  # type: ignore[assignment]
        instagram_connector._open_new_instagram_recipient = original_new  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if matched != "Fixture Example" or new_calls != [("Fixture Example", True)]:
        raise SystemExit(f"Default Instagram fallback did not require followed-account proof: {matched!r} {new_calls}")


def test_instagram_display_name_without_row_proof_requires_exact_handle() -> None:
    original_search = instagram_connector._start_instagram_thread_search
    original_match = instagram_connector._open_thread_matches
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    scripts: list[str] = []

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        scripts.append(script)
        if "https://www.instagram.com/direct/new/" in script:
            if not script.startswith("(function(){") or "return 'NAVIGATING';})()" not in script:
                raise AssertionError(f"Instagram New Message navigation must use a valid IIFE: {script}")
            return "NAVIGATING"
        if "var exact = unique" in script:
            return "NOT_FOLLOWING"
        raise AssertionError(f"Following refusal should stop before opening a thread: {script[:100]}")

    try:
        instagram_connector._start_instagram_thread_search = lambda _recipient: "SEARCHING"  # type: ignore[assignment]
        instagram_connector._open_thread_matches = lambda _recipient: (_ for _ in ()).throw(
            AssertionError("Unfollowed recipient must not reach opened-thread verification")
        )  # type: ignore[assignment]
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        try:
            instagram_connector._open_new_instagram_recipient("New Person", require_following=True)
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_exact_handle_required":
                raise SystemExit(f"Instagram exact-handle requirement stage drifted: {exc.stage}")
        else:
            raise SystemExit("Default Instagram new-message path selected an account without Following proof")
    finally:
        instagram_connector._start_instagram_thread_search = original_search  # type: ignore[assignment]
        instagram_connector._open_thread_matches = original_match  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    finder_scripts = [script for script in scripts if "var exact = unique" in script]
    if len(finder_scripts) != 1 or "var requireFollowing = true" not in finder_scripts[0]:
        raise SystemExit(f"Default Instagram new-message path did not activate the follow guard: {scripts}")


def test_instagram_exact_handle_profile_following_verification() -> None:
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    states = ["FOLLOWING"]
    scripts: list[str] = []

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        scripts.append(script)
        if "location.href=" in script:
            return "NAVIGATING"
        return states.pop(0)

    try:
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        instagram_connector._verify_followed_instagram_profile("@fixture_yoo")
    finally:
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if len(scripts) != 2 or "/fixture_yoo/" not in scripts[0] or "Following" in scripts[1]:
        raise SystemExit(f"Instagram followed-profile verification script drifted: {scripts}")
    if '"following"' not in scripts[1] or '"팔로잉"' not in scripts[1]:
        raise SystemExit(f"Instagram profile verification missed bilingual Following labels: {scripts[1]}")
    required_scope_fragments = (
        "var main=document.querySelector('main,[role=\"main\"]')",
        "var header=main.querySelector('header')",
        "header.querySelectorAll('button,[role=\"button\"]')",
        "x.getAttribute('aria-label')",
        "x.querySelectorAll('[aria-label]')",
        "replace(/\\s+/g,' ')",
    )
    missing_scope = [fragment for fragment in required_scope_fragments if fragment not in scripts[1]]
    if missing_scope:
        raise SystemExit(f"Instagram profile verification is missing scoped relationship proof: {missing_scope}")
    if "document.querySelectorAll('button,[role=\"button\"]')" in scripts[1]:
        raise SystemExit("Instagram profile verification still trusts unrelated page-wide Follow buttons")

    try:
        instagram_connector._verify_followed_instagram_profile("Fixture Example")
    except instagram_connector.InstagramWebError as exc:
        if exc.stage != "instagram_exact_handle_required":
            raise SystemExit(f"Instagram display-name profile guard stage drifted: {exc.stage}")
    else:
        raise SystemExit("Instagram profile verification accepted a display name instead of an exact @username")

    states = ["NOT_FOLLOWING"]
    try:
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        try:
            instagram_connector._verify_followed_instagram_profile("@newperson")
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_recipient_not_followed":
                raise SystemExit(f"Instagram non-followed profile stage drifted: {exc.stage}")
        else:
            raise SystemExit("Instagram accepted an exact profile whose visible action was Follow")
    finally:
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]


def test_instagram_exact_handle_opened_thread_uses_profile_link() -> None:
    original_js = instagram_connector._chrome_js
    scripts: list[str] = []
    try:
        instagram_connector._chrome_js = lambda script, timeout=20: scripts.append(script) or "MATCH"  # type: ignore[assignment]
        matched = instagram_connector._open_thread_matches("@fixture_yoo")
    finally:
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
    if not matched or len(scripts) != 1:
        raise SystemExit(f"Instagram exact-handle thread verification call drifted: {matched} {scripts}")
    script = scripts[0]
    required_fragments = (
        "main.querySelectorAll('h1,h2,h3,[role=\"heading\"]')",
        "!h.closest('nav,[role=\"navigation\"]')",
        "header.querySelectorAll('a[href]')",
        "h.closest('a[href]')",
        "new URL(a.href,location.origin).pathname",
        "headerLinked||headingLinked",
        "headerMatched||headingMatched",
    )
    for fragment in required_fragments:
        if fragment not in script:
            raise SystemExit(
                f"Instagram opened-thread identity guard missed headerless-layout fallback {fragment!r}: {script}"
            )
    if "if (!header) return 'NO_HEADER'" in script:
        raise SystemExit("Instagram opened-thread identity guard still rejects the current headerless DM layout")
    if "header.querySelectorAll('a[href]')" not in script or "new URL(a.href,location.origin).pathname" not in script:
        raise SystemExit(f"Instagram exact-handle thread verification missed the profile-link identity guard: {script}")


def test_instagram_new_thread_preserves_exact_handle_for_opened_thread_proof() -> None:
    original_verify = instagram_connector._verify_followed_instagram_profile
    original_search = instagram_connector._start_instagram_thread_search
    original_match = instagram_connector._open_thread_matches
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    matched_recipients: list[str] = []

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        if "https://www.instagram.com/direct/new/" in script:
            return "NAVIGATING"
        if "var exact = unique" in script:
            return "CLICKED:proof_account_2029"
        if "location.pathname.indexOf('/direct/t/')" in script:
            return "OPEN"
        raise AssertionError(f"Unexpected Instagram exact-handle thread JS: {script[:100]}")

    try:
        instagram_connector._verify_followed_instagram_profile = lambda _recipient: None  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = lambda _recipient: "SEARCHING"  # type: ignore[assignment]
        instagram_connector._open_thread_matches = (  # type: ignore[assignment]
            lambda recipient: matched_recipients.append(recipient) or recipient == "@proof_account_2029"
        )
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        matched = instagram_connector._open_new_instagram_recipient("@proof_account_2029", require_following=True)
    finally:
        instagram_connector._verify_followed_instagram_profile = original_verify  # type: ignore[assignment]
        instagram_connector._start_instagram_thread_search = original_search  # type: ignore[assignment]
        instagram_connector._open_thread_matches = original_match  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if matched != "@proof_account_2029" or matched_recipients != ["@proof_account_2029"]:
        raise SystemExit(
            f"Instagram new-thread flow lost the exact @handle before opened-thread proof: {matched!r} {matched_recipients}"
        )


def test_instagram_inbox_navigation_uses_valid_javascript() -> None:
    original_js = instagram_connector._chrome_js
    scripts: list[str] = []
    try:
        instagram_connector._chrome_js = lambda script, timeout=20: scripts.append(script) or "NAVIGATING"  # type: ignore[assignment]
        instagram_connector._navigate_instagram_inbox()
    finally:
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
    if len(scripts) != 1:
        raise SystemExit(f"Instagram inbox navigation call count drifted: {scripts}")
    script = scripts[0]
    if not script.startswith("(function(){") or "return 'NAVIGATING';})()" not in script:
        raise SystemExit(f"Instagram inbox navigation must use a valid IIFE: {script}")


def test_instagram_tool_preserves_explicit_new_recipient_authorization() -> None:
    tool = instagram_connector.make_instagram_tools(_config())[0]
    original_resolve = contacts_connector.resolve_contact
    original_send = instagram_connector._send_instagram_dm
    calls: list[tuple[str, str, bool]] = []
    try:
        contacts_connector.resolve_contact = lambda _query: []  # type: ignore[assignment]
        instagram_connector._send_instagram_dm = (
            lambda to, message, **kwargs: calls.append((to, message, kwargs.get("allow_new_recipient") is True)) or to
        )
        result = tool.handler({
            "to": "@newperson",
            "message": "hello",
            "allow_new_recipient": True,
        })
    finally:
        contacts_connector.resolve_contact = original_resolve  # type: ignore[assignment]
        instagram_connector._send_instagram_dm = original_send
    if not result.ok or calls != [("@newperson", "hello", True)]:
        raise SystemExit(f"Instagram explicit new-recipient authorization was lost: {result} {calls}")
    if result.metadata.get("recipient_policy") != "explicit_new_allowed":
        raise SystemExit(f"Instagram explicit recipient policy metadata drifted: {result.metadata}")
    handoff = assert_instagram_dm_handoff(result.metadata, "Instagram explicit new recipient", status="sender_confirmed")
    if handoff.get("allow_new_recipient") is not True or handoff.get("recipient_policy") != "explicit_new_allowed":
        raise SystemExit(f"Instagram explicit new-recipient handoff drifted: {handoff}")


def test_instagram_lexical_composer_uses_verified_unicode_paste() -> None:
    original_open = instagram_connector.open_instagram_thread
    original_paste = instagram_connector._trusted_paste_with_clipboard_restored
    original_enter = instagram_connector._trusted_enter
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    pasted: list[str] = []
    trusted_enters: list[str] = []
    open_calls: list[tuple[str, bool, bool]] = []
    scripts: list[str] = []
    message = "자비스 인스타그램 전달 확인"

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        scripts.append(script)
        if "SEND_TARGET_READY" in script:
            return "SEND_TARGET_READY"
        if "return String(matches.length)" in script:
            return "0"
        if "return el ? 'TEXT:'" in script:
            return "TEXT:" + message
        if "document.activeElement === el" in script:
            return "COMPOSER_FOCUSED"
        if "EMPTY_UNCONFIRMED" in script and "var expected =" in script:
            return "CONFIRMED"
        if "var candidates" in script and "data-lexical-editor" in script:
            return "COMPOSER_FOCUSED"
        raise AssertionError(f"Unexpected Instagram JS during composer smoke: {script[:100]}")

    try:
        instagram_connector.open_instagram_thread = (  # type: ignore[assignment]
            lambda recipient, **kwargs: open_calls.append((
                recipient,
                kwargs.get("allow_new_recipient") is True,
                kwargs.get("allow_followed_recipient") is True,
            )) or recipient
        )
        instagram_connector._trusted_paste_with_clipboard_restored = lambda value: pasted.append(value)  # type: ignore[assignment]
        instagram_connector._trusted_enter = lambda: trusted_enters.append("enter")  # type: ignore[assignment]
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        sent_to = instagram_connector._send_instagram_dm("Fixture Example", message)
    finally:
        instagram_connector.open_instagram_thread = original_open  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = original_paste  # type: ignore[assignment]
        instagram_connector._trusted_enter = original_enter  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if (
        sent_to != "Fixture Example"
        or pasted != [message]
        or trusted_enters != ["enter"]
        or open_calls != [("Fixture Example", False, True)]
    ):
        raise SystemExit(
            "Instagram Unicode/followed compose flow drifted: "
            f"{sent_to!r} {pasted!r} {trusted_enters!r} {open_calls!r}"
        )
    if any("NO_SEND_BUTTON" in script for script in scripts):
        raise SystemExit("Instagram send flow regressed to an untrusted synthetic Send click")
    unsafe_fallback = "|| document.querySelector('[contenteditable=\"true\"]')"
    unsafe_candidate = ", [contenteditable=\"true\"]"
    if any(unsafe_fallback in script or unsafe_candidate in script for script in scripts):
        raise SystemExit("Instagram send flow regained a generic contenteditable composer fallback")
    confirmation_scripts = [script for script in scripts if "EMPTY_UNCONFIRMED" in script]
    baseline_scripts = [script for script in scripts if "return String(matches.length)" in script]
    if (
        len(baseline_scripts) != 1
        or len(confirmation_scripts) != 1
        or json.dumps(message, ensure_ascii=False) not in confirmation_scripts[0]
        or "var baseline = 0" not in confirmation_scripts[0]
        or "closest('[contenteditable=\"true\"]')" not in confirmation_scripts[0]
    ):
        raise SystemExit(
            "Instagram send confirmation did not require a new exact visible message: "
            f"baseline={baseline_scripts} confirmation={confirmation_scripts}"
        )


def test_instagram_revalidates_thread_and_composer_before_trusted_paste() -> None:
    original_open = instagram_connector.open_instagram_thread
    original_paste = instagram_connector._trusted_paste_with_clipboard_restored
    original_enter = instagram_connector._trusted_enter
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep

    def run_case(*, target_state: str, expected_stage: str) -> None:
        events: list[str] = []

        def fake_js(script: str, timeout: int = 20) -> str:
            del timeout
            if "return String(matches.length)" in script:
                events.append("baseline")
                return "0"
            if "SEND_TARGET_READY" in script:
                events.append("target-revalidation")
                return target_state
            if "var candidates" in script and "data-lexical-editor" in script:
                events.append("prepare")
                return "COMPOSER_FOCUSED"
            raise AssertionError(f"Unexpected Instagram JS during target revalidation: {script[:100]}")

        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = (  # type: ignore[assignment]
            lambda _value: events.append("paste")
        )
        instagram_connector._trusted_enter = lambda: events.append("enter")  # type: ignore[assignment]
        try:
            instagram_connector._send_instagram_dm("Fixture Example", "boundary proof")
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != expected_stage:
                raise SystemExit(
                    f"Instagram target revalidation returned {exc.stage}, expected {expected_stage}"
                )
        else:
            raise SystemExit("Instagram sent after its direct-thread/composer target changed")
        if "paste" in events or "enter" in events:
            raise SystemExit(f"Instagram crossed a trusted input boundary after revalidation failed: {events}")
        expected_prefix = ["baseline", "prepare", "target-revalidation"]
        if events[:3] != expected_prefix:
            raise SystemExit(f"Instagram target revalidation was not immediately before paste: {events}")
        if events != expected_prefix:
            raise SystemExit(f"Instagram target revalidation order drifted: {events}")

    try:
        instagram_connector.open_instagram_thread = lambda _recipient, **_kwargs: "Fixture Example"  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        run_case(
            target_state="NOT_MATCH",
            expected_stage="instagram_thread_not_found",
        )
        run_case(
            target_state="NO_COMPOSER",
            expected_stage="instagram_compose_failed",
        )
    finally:
        instagram_connector.open_instagram_thread = original_open  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = original_paste  # type: ignore[assignment]
        instagram_connector._trusted_enter = original_enter  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]


def test_instagram_trusted_enter_failure_is_outcome_unknown() -> None:
    original_open = instagram_connector.open_instagram_thread
    original_paste = instagram_connector._trusted_paste_with_clipboard_restored
    original_enter = instagram_connector._trusted_enter
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    events: list[str] = []

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        if "return String(matches.length)" in script:
            return "0"
        if "SEND_TARGET_READY" in script:
            return "SEND_TARGET_READY"
        if "return el ? 'TEXT:'" in script:
            return "TEXT:enter boundary proof"
        if "document.activeElement === el" in script:
            return "COMPOSER_FOCUSED"
        if "var candidates" in script and "data-lexical-editor" in script:
            return "COMPOSER_FOCUSED"
        raise AssertionError(f"Unexpected Instagram JS during Enter boundary smoke: {script[:100]}")

    def failed_enter() -> None:
        events.append("enter-attempted")
        raise instagram_connector.InstagramWebError("macos_accessibility_permission")

    try:
        instagram_connector.open_instagram_thread = lambda _recipient, **_kwargs: "Fixture Example"  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = lambda _value: events.append("paste")  # type: ignore[assignment]
        instagram_connector._trusted_enter = failed_enter  # type: ignore[assignment]
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        try:
            instagram_connector._send_instagram_dm("Fixture Example", "enter boundary proof")
        except instagram_connector.InstagramWebError as exc:
            if (
                exc.stage != "instagram_send_unconfirmed"
                or exc.send_attempted is not True
                or instagram_connector._send_outcome_is_known_not_sent(exc, exc.stage)
            ):
                raise SystemExit(f"Instagram Enter failure was not outcome-unknown: {exc}")
        else:
            raise SystemExit("Instagram treated a failed trusted Enter attempt as known-not-sent")
    finally:
        instagram_connector.open_instagram_thread = original_open  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = original_paste  # type: ignore[assignment]
        instagram_connector._trusted_enter = original_enter  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if events != ["paste", "enter-attempted"]:
        raise SystemExit(f"Instagram Enter boundary fixture crossed the wrong actions: {events}")


def test_instagram_trusted_paste_restores_clipboard() -> None:
    original_run = instagram_connector.subprocess.run
    fake_run = _InstagramClipboardRun("user clipboard")

    try:
        instagram_connector.subprocess.run = fake_run  # type: ignore[assignment]
        receipt = instagram_connector._trusted_paste_with_clipboard_restored("한글 proof")
    finally:
        instagram_connector.subprocess.run = original_run  # type: ignore[assignment]
    commands = [call[0] for call in fake_run.calls]
    paste_calls = [
        call
        for call in fake_run.calls
        if call[0][:2] == ["osascript", "-e"]
        and call[0] != ["osascript", "-e", "clipboard info"]
    ]
    if len(paste_calls) != 1:
        raise SystemExit(f"Instagram trusted-paste action count drifted: {commands}")
    paste_script = paste_calls[0][0][2]
    if (
        'tell application "Google Chrome" to activate' not in paste_script
        or "delay 0.2" not in paste_script
        or 'tell process "Google Chrome"' not in paste_script
        or "set frontmost to true" not in paste_script
        or "key code 9 using {command down}" not in paste_script
        or "delay 0.35" not in paste_script
        or paste_script.index("delay 0.35") < paste_script.index("key code 9 using {command down}")
    ):
        raise SystemExit(
            "Instagram trusted paste must activate Chrome immediately before Cmd-V "
            f"and wait for Chromium to consume the clipboard: {paste_script}"
        )
    writes = [
        call
        for call in fake_run.calls
        if clipboard_safety._CONDITIONAL_PLAIN_TEXT_WRITE_JXA in call[0]
    ]
    if (
        len(writes) != 2
        or writes[0][1].get("input") != "한글 proof"
        or writes[1][1].get("input") != "user clipboard"
        or fake_run.text != "user clipboard"
    ):
        raise SystemExit(
            "Instagram trusted paste did not stage and restore under ownership: "
            f"{fake_run.calls}"
        )
    if ["pbcopy"] in commands:
        raise SystemExit(f"Instagram trusted paste used legacy pbcopy: {commands}")
    if (
        receipt.get("clipboard_snapshot_captured") is not True
        or receipt.get("clipboard_replacement_attempted") is not True
        or receipt.get("clipboard_text_restored") is not True
        or receipt.get("clipboard_fully_restored") is not False
        or receipt.get("clipboard_restoration_scope") != "plain_text_value_only"
        or receipt.get("clipboard_private_content_possible") is not False
        or receipt.get("clipboard_privacy_boundary", {}).get(
            "clipboard_snapshot_captured"
        )
        is not True
    ):
        raise SystemExit(f"Instagram clipboard receipt overclaimed its privacy boundary: {receipt}")


def test_instagram_clipboard_snapshot_and_restore_fail_closed() -> None:
    original_run = instagram_connector.subprocess.run
    try:
        unreadable_run = _InstagramClipboardRun()
        unreadable_run.fail_inspection = True
        instagram_connector.subprocess.run = unreadable_run
        try:
            instagram_connector._trusted_paste_with_clipboard_restored("private proof")
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_clipboard_inspection":
                raise SystemExit(
                    "Instagram clipboard snapshot failure drifted: "
                    f"{unreadable_run.calls} {exc}"
                )
            if "PRIVATE" in exc.detail or "/\x55sers/" in exc.detail:
                raise SystemExit(f"Instagram clipboard snapshot failure leaked detail: {exc.detail}")
            if (
                exc.clipboard_metadata.get("clipboard_snapshot_captured") is not False
                or exc.clipboard_metadata.get("clipboard_replacement_attempted") is not False
                or exc.clipboard_metadata.get("clipboard_outcome_known") is not True
            ):
                raise SystemExit(f"Instagram snapshot failure metadata overclaimed mutation: {exc.clipboard_metadata}")
        else:
            raise SystemExit("Instagram clipboard snapshot failure allowed replacement")

        restore_failure_run = _InstagramClipboardRun("user clipboard")
        restore_failure_run.fail_write_number = 2
        instagram_connector.subprocess.run = restore_failure_run
        try:
            instagram_connector._trusted_paste_with_clipboard_restored("private proof")
        except instagram_connector.InstagramWebError as exc:
            if (
                exc.stage != "instagram_send_unconfirmed"
                or restore_failure_run.write_count != 2
            ):
                raise SystemExit(f"Instagram restore failure was not quarantined: {exc}")
            if "PRIVATE" in exc.detail or "/\x55sers/" in exc.detail:
                raise SystemExit(f"Instagram restore failure leaked detail: {exc.detail}")
            if (
                exc.clipboard_metadata.get("clipboard_snapshot_captured") is not True
                or exc.clipboard_metadata.get("clipboard_replacement_attempted") is not True
                or exc.clipboard_metadata.get("clipboard_text_restored") is not False
                or exc.clipboard_metadata.get("clipboard_outcome_known") is not False
                or exc.clipboard_metadata.get("clipboard_private_content_possible") is not True
            ):
                raise SystemExit(f"Instagram restore failure lost clipboard privacy truth: {exc.clipboard_metadata}")
        else:
            raise SystemExit("Instagram restore failure incorrectly reported success")

        ownership_loss_run = _InstagramClipboardRun("user clipboard")
        ownership_loss_run.newer_text_during_paste = "newer user clipboard"
        instagram_connector.subprocess.run = ownership_loss_run
        try:
            instagram_connector._trusted_paste_with_clipboard_restored("private proof")
        except instagram_connector.InstagramWebError as exc:
            if (
                exc.stage != "instagram_send_unconfirmed"
                or ownership_loss_run.text != "newer user clipboard"
                or ownership_loss_run.write_count != 1
            ):
                raise SystemExit(
                    "Instagram ownership loss overwrote a newer clipboard or "
                    f"became retryable: {ownership_loss_run.calls} {exc}"
                )
            if (
                exc.clipboard_metadata.get("clipboard_text_restored") is not False
                or exc.clipboard_metadata.get("clipboard_outcome_known") is not False
                or exc.clipboard_metadata.get("clipboard_private_content_possible")
                is not True
            ):
                raise SystemExit(
                    "Instagram ownership loss lost no-blind-retry custody truth: "
                    f"{exc.clipboard_metadata}"
                )
        else:
            raise SystemExit("Instagram ownership loss incorrectly reported success")
    finally:
        instagram_connector.subprocess.run = original_run

    tool = instagram_connector.make_instagram_tools(_config())[0]
    original_send = instagram_connector._send_instagram_dm
    try:
        instagram_connector._send_instagram_dm = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            instagram_connector.InstagramWebError(
                "instagram_send_unconfirmed",
                "PRIVATE /\x55sers/example/instagram-handler.log",
                clipboard_metadata=instagram_connector.clipboard_safety.privacy_boundary_metadata(
                    snapshot_attempted=True,
                    snapshot_captured=True,
                    replacement_attempted=True,
                    text_restored=False,
                ),
            )
        )
        result = tool.handler({"to": "@example_user", "message": "private proof"})
    finally:
        instagram_connector._send_instagram_dm = original_send
    assert_outcome_unknown_recovery(
        result,
        expected_output=instagram_connector._post_attempt_send_recovery_guidance(),
        label="Instagram clipboard restore failure",
    )
    if (
        result.metadata.get("clipboard_text_restored") is not False
        or result.metadata.get("clipboard_outcome_known") is not False
        or result.metadata.get("clipboard_private_content_possible") is not True
        or result.metadata.get("clipboard_privacy_boundary", {}).get(
            "clipboard_outcome_known"
        )
        is not False
        or _leaks_local_path(result.metadata)
    ):
        raise SystemExit(f"Instagram handler lost clipboard privacy truth: {result.metadata}")


def test_instagram_send_requires_one_new_visible_message() -> None:
    original_open = instagram_connector.open_instagram_thread
    original_paste = instagram_connector._trusted_paste_with_clipboard_restored
    original_enter = instagram_connector._trusted_enter
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    trusted_enters: list[str] = []
    confirmation_checks = 0

    def fake_js(script: str, timeout: int = 20) -> str:
        nonlocal confirmation_checks
        del timeout
        if "SEND_TARGET_READY" in script:
            return "SEND_TARGET_READY"
        if "return String(matches.length)" in script:
            return "1"
        if "return el ? 'TEXT:'" in script:
            return "TEXT:duplicate-safe proof"
        if "document.activeElement === el" in script:
            return "COMPOSER_FOCUSED"
        if "EMPTY_UNCONFIRMED" in script:
            confirmation_checks += 1
            if "var baseline = 1" not in script:
                raise AssertionError("Instagram confirmation lost its before-send baseline")
            return "EMPTY_UNCONFIRMED"
        if "var candidates" in script and "data-lexical-editor" in script:
            return "COMPOSER_FOCUSED"
        raise AssertionError(f"Unexpected Instagram JS during duplicate-proof smoke: {script[:100]}")

    try:
        instagram_connector.open_instagram_thread = lambda _recipient, **_kwargs: "Fixture Example"  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = lambda _value: None  # type: ignore[assignment]
        instagram_connector._trusted_enter = lambda: trusted_enters.append("enter")  # type: ignore[assignment]
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        try:
            instagram_connector._send_instagram_dm("Fixture Example", "duplicate-safe proof")
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_send_unconfirmed":
                raise SystemExit(f"Instagram duplicate-safe proof returned the wrong stage: {exc.stage}")
        else:
            raise SystemExit("Instagram treated an older identical message as proof of a new send")
    finally:
        instagram_connector.open_instagram_thread = original_open  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = original_paste  # type: ignore[assignment]
        instagram_connector._trusted_enter = original_enter  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if trusted_enters != ["enter"] or confirmation_checks != 10:
        raise SystemExit(
            "Instagram unconfirmed send retried or skipped bounded confirmation: "
            f"enters={trusted_enters!r} checks={confirmation_checks}"
        )


def test_instagram_post_enter_bridge_failure_becomes_unconfirmed() -> None:
    original_open = instagram_connector.open_instagram_thread
    original_paste = instagram_connector._trusted_paste_with_clipboard_restored
    original_enter = instagram_connector._trusted_enter
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    trusted_enters: list[str] = []

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        if "SEND_TARGET_READY" in script:
            return "SEND_TARGET_READY"
        if "return String(matches.length)" in script:
            return "0"
        if "return el ? 'TEXT:'" in script:
            return "TEXT:post-enter proof"
        if "document.activeElement === el" in script:
            return "COMPOSER_FOCUSED"
        if "EMPTY_UNCONFIRMED" in script:
            raise instagram_connector.InstagramWebError(
                "chrome_javascript_permission",
                "PRIVATE /\x55sers/example/private/post-enter.log",
            )
        if "var candidates" in script and "data-lexical-editor" in script:
            return "COMPOSER_FOCUSED"
        raise AssertionError(f"Unexpected Instagram JS during post-Enter smoke: {script[:100]}")

    try:
        instagram_connector.open_instagram_thread = lambda _recipient, **_kwargs: "Fixture Example"  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = lambda _value: None  # type: ignore[assignment]
        instagram_connector._trusted_enter = lambda: trusted_enters.append("enter")  # type: ignore[assignment]
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        try:
            instagram_connector._send_instagram_dm("Fixture Example", "post-enter proof")
        except instagram_connector.InstagramWebError as exc:
            if (
                exc.stage != "instagram_send_unconfirmed"
                or "PRIVATE" in exc.detail
                or "/\x55sers/" in exc.detail
            ):
                raise SystemExit(f"Instagram post-Enter failure was not safely quarantined: {exc}")
        else:
            raise SystemExit("Instagram post-Enter confirmation failure was treated as success")
    finally:
        instagram_connector.open_instagram_thread = original_open  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = original_paste  # type: ignore[assignment]
        instagram_connector._trusted_enter = original_enter  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if trusted_enters != ["enter"]:
        raise SystemExit(f"Instagram post-Enter fixture did not cross the boundary once: {trusted_enters}")


def test_instagram_empty_paste_never_sends() -> None:
    original_open = instagram_connector.open_instagram_thread
    original_paste = instagram_connector._trusted_paste_with_clipboard_restored
    original_enter = instagram_connector._trusted_enter
    original_js = instagram_connector._chrome_js
    original_sleep = instagram_connector.time.sleep
    trusted_enters: list[str] = []

    def fake_js(script: str, timeout: int = 20) -> str:
        del timeout
        if "SEND_TARGET_READY" in script:
            return "SEND_TARGET_READY"
        if "return String(matches.length)" in script:
            return "0"
        if "return el ? 'TEXT:'" in script:
            return "TEXT:"
        if "var candidates" in script and "data-lexical-editor" in script:
            return "COMPOSER_FOCUSED"
        raise AssertionError(f"Unexpected Instagram JS after an empty paste: {script[:100]}")

    try:
        instagram_connector.open_instagram_thread = lambda _recipient, **_kwargs: "Fixture Example"  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = lambda _value: None  # type: ignore[assignment]
        instagram_connector._trusted_enter = lambda: trusted_enters.append("enter")  # type: ignore[assignment]
        instagram_connector._chrome_js = fake_js  # type: ignore[assignment]
        instagram_connector.time.sleep = lambda _seconds: None  # type: ignore[assignment]
        try:
            instagram_connector._send_instagram_dm("Fixture Example", "must not send")
        except instagram_connector.InstagramWebError as exc:
            if exc.stage != "instagram_compose_failed":
                raise SystemExit(f"Instagram empty paste returned the wrong stage: {exc.stage}")
        else:
            raise SystemExit("Instagram sent after the trusted paste left the composer empty")
    finally:
        instagram_connector.open_instagram_thread = original_open  # type: ignore[assignment]
        instagram_connector._trusted_paste_with_clipboard_restored = original_paste  # type: ignore[assignment]
        instagram_connector._trusted_enter = original_enter  # type: ignore[assignment]
        instagram_connector._chrome_js = original_js  # type: ignore[assignment]
        instagram_connector.time.sleep = original_sleep  # type: ignore[assignment]
    if trusted_enters:
        raise SystemExit(f"Instagram pressed Enter after an empty paste: {trusted_enters}")


def test_instagram_send_failure_preserves_attempt_metadata() -> None:
    tool = instagram_connector.make_instagram_tools(_config())[0]
    original = instagram_connector._send_instagram_dm
    try:
        instagram_connector._send_instagram_dm = lambda _to, _message, **_kwargs: (_ for _ in ()).throw(RuntimeError("chrome unavailable"))
        result = tool.handler({"to": "@example_user", "message": "hello"})
    finally:
        instagram_connector._send_instagram_dm = original
    assert_outcome_unknown_recovery(
        result,
        expected_output=instagram_connector._post_attempt_send_recovery_guidance(),
        label="generic Instagram send failure",
    )
    if "chrome unavailable" in result.output or "Error:" in result.output:
        raise SystemExit(f"Instagram mocked failure should not leak raw errors: {result}")
    if not result.metadata.get("send_attempted") or not result.metadata.get("executes_side_effect"):
        raise SystemExit(f"Instagram failed send should preserve attempted side-effect metadata: {result.metadata}")
    if not result.metadata.get("external_side_effect") or result.metadata.get("to") != "@example_user":
        raise SystemExit(f"Instagram failed send metadata should preserve recipient and external attempt: {result.metadata}")
    if result.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"Instagram failed send metadata should preserve bounded diagnostic type: {result.metadata}")
    handoff = assert_instagram_dm_handoff(result.metadata, "Instagram failed send", status="outcome_unknown")
    if handoff.get("reason") != "send_outcome_unknown" or handoff.get("send_attempted") is not True:
        raise SystemExit(f"Instagram failed send handoff should preserve outcome uncertainty: {handoff}")


def test_instagram_failures_are_actionable_and_private() -> None:
    cases = (
        (
            "execution error: Not authorized to send Apple events to Google Chrome. (-1743)",
            "macos_automation_permission",
            "Privacy & Security > Automation",
        ),
        (
            "execution error: osascript is not allowed assistive access. (-25211)",
            "macos_accessibility_permission",
            "Privacy & Security > Accessibility",
        ),
    )
    for raw_error, expected_stage, expected_guidance in cases:
        stage, detail = instagram_connector._instagram_error_stage(raw_error)
        if stage != expected_stage or expected_guidance not in detail:
            raise SystemExit(f"Instagram permission classifier drifted: {stage!r} {detail!r}")
        failure_stage, output = instagram_connector._instagram_dm_failure(
            instagram_connector.InstagramWebError(stage, detail)
        )
        if failure_stage != expected_stage or expected_guidance not in output:
            raise SystemExit(f"Instagram permission guidance was not surfaced safely: {failure_stage!r} {output!r}")
        if raw_error in output or "(-1743)" in output or "(-25211)" in output:
            raise SystemExit(f"Instagram permission guidance leaked raw AppleScript diagnostics: {output!r}")

    private_detail = "No Instagram DM thread named '가상연락처일'. Threads I can see: Private Friend | Family"
    failure_stage, output = instagram_connector._instagram_dm_failure(
        instagram_connector.InstagramWebError("instagram_thread_not_found", private_detail)
    )
    if failure_stage != "instagram_thread_not_found" or "Open the exact conversation" not in output:
        raise SystemExit(f"Instagram thread guidance drifted: {failure_stage!r} {output!r}")
    if "가상연락처일" in output or "Private Friend" in output or "Family" in output:
        raise SystemExit(f"Instagram thread guidance leaked private page state: {output!r}")

    timeout_stage, timeout_output = instagram_connector._instagram_dm_failure(
        instagram_connector.subprocess.TimeoutExpired(["osascript"], 20)
    )
    if timeout_stage != "automation_timeout" or "nothing was retried" not in timeout_output:
        raise SystemExit(f"Instagram timeout classification drifted: {timeout_stage!r} {timeout_output!r}")

    tool = instagram_connector.make_instagram_tools(_config())[0]
    original = instagram_connector._send_instagram_dm
    try:
        instagram_connector._send_instagram_dm = lambda _to, _message, **_kwargs: (_ for _ in ()).throw(
            instagram_connector.InstagramWebError("instagram_thread_not_found", private_detail)
        )
        result = tool.handler({"to": "@example_user", "message": "hello"})
    finally:
        instagram_connector._send_instagram_dm = original
    if result.ok or result.metadata.get("failure_stage") != "instagram_thread_not_found":
        raise SystemExit(f"Instagram tool receipt missed the safe failure stage: {result.output} {result.metadata}")
    assert_known_not_sent_recovery(
        result,
        action=instagram_connector.KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
        label="Instagram pre-send thread failure",
    )
    if any(value in result.output or value in str(result.metadata) for value in ("가상연락처일", "Private Friend", "Family")):
        raise SystemExit(f"Instagram failure receipt leaked private page state: {result.output} {result.metadata}")
    if (
        result.metadata.get("outcome_known") is not True
        or result.metadata.get("outcome_unknown") is not False
        or result.metadata.get("side_effect_possible") is not False
    ):
        raise SystemExit(f"Instagram pre-send thread failure lost known-no-send truth: {result.metadata}")

    original = instagram_connector._send_instagram_dm
    try:
        instagram_connector._send_instagram_dm = lambda _to, _message, **_kwargs: (_ for _ in ()).throw(
            instagram_connector.InstagramWebError(
                "instagram_send_unconfirmed",
                "PRIVATE SHOULD NOT APPEAR /\x55sers/example/private/instagram.log",
            )
        )
        unconfirmed = tool.handler({"to": "@example_user", "message": "hello"})
    finally:
        instagram_connector._send_instagram_dm = original
    assert_outcome_unknown_recovery(
        unconfirmed,
        expected_output=instagram_connector._post_attempt_send_recovery_guidance(),
        label="Instagram unconfirmed delivery",
    )
    handoff = assert_instagram_dm_handoff(
        unconfirmed.metadata,
        "Instagram unconfirmed delivery",
        status="outcome_unknown",
    )
    if handoff.get("reason") != "send_outcome_unknown":
        raise SystemExit(f"Instagram unconfirmed handoff drifted: {handoff}")
    if "PRIVATE SHOULD NOT APPEAR" in unconfirmed.output or _leaks_local_path(unconfirmed.metadata):
        raise SystemExit(f"Instagram unconfirmed delivery leaked private detail: {unconfirmed}")


def test_instagram_redacts_path_shaped_recipient_metadata() -> None:
    tool = instagram_connector.make_instagram_tools(_config())[0]
    for recipient in (
        "/\x55sers/example/Desktop/Claude code/instagram-target",
        "/var/folders/zc/jarvis/instagram-target",
        "/tmp/jarvis/instagram-target",
    ):
        missing = tool.handler({"to": recipient})
        if missing.ok or missing.metadata.get("to") != "<local-path>":
            raise SystemExit(f"Instagram missing-message metadata should redact path-shaped recipients: {missing.metadata}")
        handoff = assert_instagram_dm_handoff(missing.metadata, "Instagram path-shaped missing-message", status="refused")
        if handoff.get("to") != "<local-path>" or handoff.get("reason") != "missing_message":
            raise SystemExit(f"Instagram path-shaped handoff should redact recipient: {handoff}")
        if _leaks_local_path(missing.output) or _leaks_local_path(missing.metadata):
            raise SystemExit(f"Instagram validation leaked path-shaped recipient: {missing.output} {missing.metadata}")

    original = instagram_connector._send_instagram_dm
    try:
        instagram_connector._send_instagram_dm = lambda _to, _message, **_kwargs: (_ for _ in ()).throw(RuntimeError("chrome unavailable near /private/tmp/instagram"))
        failed = tool.handler({"to": "/tmp/instagram-target", "message": "hello"})
    finally:
        instagram_connector._send_instagram_dm = original
    if failed.ok or failed.metadata.get("to") != "<local-path>" or "<local-path>" in failed.output:
        raise SystemExit(f"Instagram failed-send recipient redaction wrong: {failed.output} {failed.metadata}")
    handoff = assert_instagram_dm_handoff(
        failed.metadata,
        "Instagram path-shaped failed-send",
        status="outcome_unknown",
    )
    if handoff.get("to") != "<local-path>" or handoff.get("reason") != "send_outcome_unknown":
        raise SystemExit(f"Instagram failed-send handoff should redact recipient: {handoff}")
    if _leaks_local_path(failed.output) or _leaks_local_path(failed.metadata):
        raise SystemExit(f"Instagram failed-send output/metadata leaked path-shaped recipient: {failed.output} {failed.metadata}")


def test_planner_routes_social_send_intents_to_gated_tools() -> None:
    planner = RuleBasedPlanner()
    preopened_command = (
        "kakao preopened_exact_chat Exact Proof Friend: "
        "Jarvis V3 Kakao supervised proof 8605"
    )
    preopened_plan = planner.plan(preopened_command)
    if (
        len(preopened_plan.actions) != 1
        or preopened_plan.actions[0].tool_name != "send_kakao"
        or preopened_plan.actions[0].args
        != {
            "to": "Exact Proof Friend",
            "message": "Jarvis V3 Kakao supervised proof 8605",
            "target_mode": "preopened_exact_chat",
        }
    ):
        raise SystemExit(f"Preopened exact Kakao route drifted: {preopened_plan}")

    kakao_plan = planner.plan("kakao Mom: On my way")
    if kakao_plan.actions[0].tool_name != "send_kakao":
        raise SystemExit(f"Kakao route missed: {kakao_plan}")
    if kakao_plan.actions[0].args != {"to": "Mom", "message": "On my way"}:
        raise SystemExit(f"Kakao args wrong: {kakao_plan.actions[0].args}")

    kakao_saying_plan = planner.plan("send a kakao to Mom saying On my way")
    if kakao_saying_plan.actions[0].tool_name != "send_kakao":
        raise SystemExit(f"Kakao saying route missed: {kakao_saying_plan}")
    if kakao_saying_plan.actions[0].args != {"to": "Mom", "message": "On my way"}:
        raise SystemExit(f"Kakao saying args wrong: {kakao_saying_plan.actions[0].args}")

    korean_kakao_plan = planner.plan("카카오톡 엄마에게 곧 갈게")
    if korean_kakao_plan.actions[0].tool_name != "send_kakao":
        raise SystemExit(f"Korean Kakao route missed: {korean_kakao_plan}")
    if korean_kakao_plan.actions[0].args != {"to": "엄마", "message": "곧 갈게"}:
        raise SystemExit(f"Korean Kakao args wrong: {korean_kakao_plan.actions[0].args}")

    # Trailing-service phrasing: the service is named AFTER the recipient, e.g.
    # "text fixture to kakaotalk saying ...". These must route to Kakao (not
    # iMessage) with a clean recipient — the service words must not leak into `to`.
    for phrase in [
        "text fixture to kakaotalk saying hello from Jarvis",
        "message fixture on kakao saying hi",
        "text fixture to kakao: hello there",
        "send mom via kakaotalk saying call me",
    ]:
        plan = planner.plan(phrase)
        if plan.actions[0].tool_name != "send_kakao":
            raise SystemExit(f"Trailing-service Kakao route missed: {phrase!r} -> {plan.actions[0].tool_name}")
        to = plan.actions[0].args.get("to")
        if to not in {"fixture", "mom"}:
            raise SystemExit(f"Trailing-service Kakao leaked service into recipient: {phrase!r} -> to={to!r}")

    # Guard the inverse: a plain text/message with no service word stays iMessage.
    for phrase in ["text fixture saying hello", "message sam saying running late"]:
        plan = planner.plan(phrase)
        if plan.actions[0].tool_name != "send_imessage":
            raise SystemExit(f"Plain text should stay iMessage: {phrase!r} -> {plan.actions[0].tool_name}")

    instagram_plan = planner.plan("send instagram dm to @example_user saying hello there")
    if instagram_plan.actions[0].tool_name != "send_instagram_dm":
        raise SystemExit(f"Instagram route missed: {instagram_plan}")
    if instagram_plan.actions[0].args != {"to": "@example_user", "message": "hello there"}:
        raise SystemExit(f"Instagram args wrong: {instagram_plan.actions[0].args}")

    explicit_new_cases = (
        "send a new Instagram DM to @newperson saying hello there",
        "send Instagram message to @newperson saying hello there; they are not in my contacts",
        "send New Person an Instagram message saying hello there, I don't follow them",
    )
    for command in explicit_new_cases:
        plan = planner.plan(command)
        if not plan.actions or plan.actions[0].tool_name != "send_instagram_dm":
            raise SystemExit(f"Explicit new Instagram route missed: {command!r} -> {plan}")
        if plan.actions[0].args.get("message") != "hello there":
            raise SystemExit(f"Explicit new Instagram marker leaked into message: {command!r} -> {plan.actions[0].args}")
        if plan.actions[0].args.get("allow_new_recipient") is not True:
            raise SystemExit(f"Explicit new Instagram route lost authorization: {command!r} -> {plan.actions[0].args}")

    instagram_dm_plan = planner.plan("dm @example_user on instagram saying hello there")
    if instagram_dm_plan.actions[0].tool_name != "send_instagram_dm":
        raise SystemExit(f"Instagram platform-explicit DM route missed: {instagram_dm_plan}")
    if instagram_dm_plan.actions[0].args != {"to": "@example_user", "message": "hello there"}:
        raise SystemExit(f"Instagram platform-explicit DM args wrong: {instagram_dm_plan.actions[0].args}")

    ig_dm_plan = planner.plan("dm @example_user on ig saying hello there")
    if ig_dm_plan.actions[0].tool_name != "send_instagram_dm":
        raise SystemExit(f"IG platform-explicit DM route missed: {ig_dm_plan}")
    if ig_dm_plan.actions[0].args != {"to": "@example_user", "message": "hello there"}:
        raise SystemExit(f"IG platform-explicit DM args wrong: {ig_dm_plan.actions[0].args}")

    send_instagram_dm_plan = planner.plan("send a dm to @example_user on instagram saying hello there")
    if send_instagram_dm_plan.actions[0].tool_name != "send_instagram_dm":
        raise SystemExit(f"Send Instagram platform-explicit DM route missed: {send_instagram_dm_plan}")
    if send_instagram_dm_plan.actions[0].args != {"to": "@example_user", "message": "hello there"}:
        raise SystemExit(f"Send Instagram platform-explicit DM args wrong: {send_instagram_dm_plan.actions[0].args}")

    natural_imessage_cases = [
        ("text fixture saying hi", {"to": "fixture", "message": "hi"}),
        ("imessage fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("message fixture: hi", {"to": "fixture", "message": "hi"}),
        ("dm fixture saying hi", {"to": "fixture", "message": "hi"}),
        ("send text to fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("send message to fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("send imessage to fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("send a dm to fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("text +15551234567 saying hi", {"to": "+15551234567", "message": "hi"}),
        ("message +15551234567: hi", {"to": "+15551234567", "message": "hi"}),
        ("send text to +15551234567 saying hello", {"to": "+15551234567", "message": "hello"}),
    ]
    for command, expected_args in natural_imessage_cases:
        plan = planner.plan(command)
        if not plan.actions or plan.actions[0].tool_name != "send_imessage":
            raise SystemExit(f"Natural iMessage route should select send_imessage: {command!r} -> {plan}")
        if plan.actions[0].args != expected_args:
            raise SystemExit(f"Natural iMessage args wrong for {command!r}: {plan.actions[0].args}")


def test_dispatch_routes_natural_imessage_to_approval_gated_send() -> None:
    natural_imessage_cases = [
        ("text fixture saying hi", {"to": "fixture", "message": "hi"}),
        ("imessage fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("message fixture: hi", {"to": "fixture", "message": "hi"}),
        ("dm fixture saying hi", {"to": "fixture", "message": "hi"}),
        ("send text to fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("send message to fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("send imessage to fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("send a dm to fixture saying hello", {"to": "fixture", "message": "hello"}),
        ("text +15551234567 saying hi", {"to": "+15551234567", "message": "hi"}),
        ("message +15551234567: hi", {"to": "+15551234567", "message": "hi"}),
        ("send text to +15551234567 saying hello", {"to": "+15551234567", "message": "hello"}),
    ]
    with tempfile.TemporaryDirectory(prefix="jarvis-social-dispatch-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for command, expected_args in natural_imessage_cases:
            packet = runtime.registry.get("dispatch_decision_packet").handler({"request": command})
            metadata = packet.metadata
            planned_actions = metadata.get("planned_actions") or []
            if metadata.get("decision") != "QUEUE_APPROVAL_IF_SENT_FOR_REAL" or metadata.get("route") != "approval_gated_tools":
                raise SystemExit(f"Natural iMessage dispatch should be approval gated for {command!r}: {metadata}")
            if metadata.get("approval_required") is not True or metadata.get("can_auto_run") is not False or metadata.get("safe_to_execute_now") is not False:
                raise SystemExit(f"Natural iMessage dispatch should not auto-run for {command!r}: {metadata}")
            if "personal data" not in metadata.get("matched_risks", []) or "external side effect" not in metadata.get("matched_risks", []):
                raise SystemExit(f"Natural iMessage dispatch missed messaging risk signals for {command!r}: {metadata}")
            if not planned_actions or planned_actions[0].get("tool") != "send_imessage":
                raise SystemExit(f"Natural iMessage dispatch missed send_imessage plan for {command!r}: {metadata}")
            if planned_actions[0].get("args") != expected_args or planned_actions[0].get("requires_approval") is not True:
                raise SystemExit(f"Natural iMessage dispatch args/gate wrong for {command!r}: {metadata}")
            forecast = metadata.get("approval_queue_forecast") or []
            if not forecast or forecast[0].get("tool_name") != "send_imessage" or forecast[0].get("would_queue_new_approval") is not True:
                raise SystemExit(f"Natural iMessage dispatch missed approval forecast for {command!r}: {metadata}")


def test_dispatch_keeps_both_instagram_recipient_modes_approval_gated() -> None:
    cases = (
        (
            "send Fixture an instagram message saying hello",
            {"to": "Fixture", "message": "hello"},
        ),
        (
            "send a new Instagram DM to @newperson saying hello",
            {"to": "@newperson", "message": "hello", "allow_new_recipient": "True"},
        ),
    )
    with tempfile.TemporaryDirectory(prefix="jarvis-instagram-dispatch-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for command, expected_args in cases:
            packet = runtime.registry.get("dispatch_decision_packet").handler({"request": command})
            metadata = packet.metadata
            planned_actions = metadata.get("planned_actions") or []
            if metadata.get("decision") != "QUEUE_APPROVAL_IF_SENT_FOR_REAL":
                raise SystemExit(f"Instagram dispatch should queue approval for {command!r}: {metadata}")
            if metadata.get("route") != "approval_gated_tools" or metadata.get("can_auto_run") is not False:
                raise SystemExit(f"Instagram dispatch must remain approval-gated for {command!r}: {metadata}")
            if not planned_actions or planned_actions[0].get("tool") != "send_instagram_dm":
                raise SystemExit(f"Instagram dispatch missed send tool for {command!r}: {metadata}")
            if planned_actions[0].get("args") != expected_args or planned_actions[0].get("requires_approval") is not True:
                raise SystemExit(f"Instagram dispatch args/gate drifted for {command!r}: {metadata}")


def test_planner_routes_recipient_first_and_calls() -> None:
    planner = RuleBasedPlanner()

    # Recipient-first send phrasing ("send <name> a <service> saying <msg>") — the
    # most natural form, previously unrouted and diverted to a read-only packet.
    send_cases = [
        ("send fixture example a kakao saying Jarvis first live send", "send_kakao", "fixture example", "Jarvis first live send"),
        ("send fixture a kakaotalk message saying hi", "send_kakao", "fixture", "hi"),
        ("send mom an instagram dm saying hello", "send_instagram_dm", "mom", "hello"),
        ("send fixture a text saying running late", "send_imessage", "fixture", "running late"),
        ("send fixture a message saying hi", "send_imessage", "fixture", "hi"),
        ("send fixture a telegram saying hi", "send_telegram", "fixture", "hi"),
        ("send 가상연락처이 a telegram saying 안녕하세요", "send_telegram", "가상연락처이", "안녕하세요"),
    ]
    for command, tool, to, message in send_cases:
        plan = planner.plan(command)
        if not plan.actions or plan.actions[0].tool_name != tool:
            raise SystemExit(f"Recipient-first send should route to {tool}: {command!r} -> {plan}")
        if plan.actions[0].args != {"to": to, "message": message}:
            raise SystemExit(f"Recipient-first send args wrong for {command!r}: {plan.actions[0].args}")

    # Calling: channel-specific first, then generic FaceTime/phone.
    call_cases = [
        ("call fixture on kakao", "call_kakao", "fixture", "audio"),
        ("video call fixture on kakaotalk", "call_kakao", "fixture", "video"),
        ("call mom on instagram", "call_instagram", "mom", "audio"),
        ("call fixture on telegram", "call_telegram", "fixture", "audio"),
        ("call fixture", "call_contact", "fixture", "phone"),
        ("facetime fixture", "call_contact", "fixture", "video"),
        ("facetime audio fixture", "call_contact", "fixture", "audio"),
        ("phone mom", "call_contact", "mom", "phone"),
        ("call mom on her phone", "call_contact", "mom", "phone"),
    ]
    for command, tool, to, mode in call_cases:
        plan = planner.plan(command)
        if not plan.actions or plan.actions[0].tool_name != tool:
            raise SystemExit(f"Call should route to {tool}: {command!r} -> {plan}")
        if plan.actions[0].args.get("to") != to or plan.actions[0].args.get("mode") != mode:
            raise SystemExit(f"Call args wrong for {command!r}: {plan.actions[0].args}")


def main() -> None:
    test_kakao_tool_is_high_risk_and_validates_args()
    test_kakao_tool_uses_mocked_sender()
    test_kakao_custody_metadata_cannot_forge_verified_delivery()
    test_kakao_malformed_contact_lookup_flag_fails_closed()
    test_kakao_sender_uses_mocked_subprocess_and_system_events()
    test_kakao_sender_preserves_hangul_recipient_and_message_via_clipboard()
    test_kakao_preopened_sender_never_navigates_or_searches()
    test_kakao_preopened_case_diacritic_duplicate_and_drift_stop_before_enter()
    test_kakao_phase_receipts_must_match_exact_markers()
    test_kakao_guard_abort_surfaces_friendly_diagnostic()
    test_kakao_clipboard_snapshot_and_restore_fail_closed()
    test_kakao_clipboard_custody_interleavings_are_fail_closed()
    test_kakao_permission_failures_are_actionable_and_bounded()
    test_kakao_send_failure_preserves_attempt_metadata()
    test_kakao_redacts_path_shaped_recipient_metadata()
    test_kakao_ambiguous_contact_refuses_without_live_send()
    test_kakao_unknown_contact_falls_through_to_kakao_search()
    test_kakao_contact_lookup_failure_falls_back_to_exact_kakao_search()
    test_kakao_explicit_handle_skips_contact_resolution()
    test_kakao_preopened_runtime_binds_exact_mode_without_contacts_or_replay()
    test_kakao_runtime_binds_canonical_recipient_once_and_replays_never()
    test_kakao_runtime_refuses_unreviewable_raw_arguments_before_approval()
    test_instagram_tool_is_high_risk_and_validates_args()
    test_instagram_approval_binds_exact_username_and_rejects_target_drift()
    test_instagram_runtime_approval_is_exact_and_one_shot()
    test_instagram_tool_uses_mocked_sender()
    test_instagram_ambiguous_contact_refuses_without_browser_send()
    test_instagram_preserves_a_verified_open_thread()
    test_instagram_active_call_stops_before_focus_navigation_or_search()
    test_instagram_call_guard_checks_every_tab_and_fails_closed()
    test_instagram_searches_beyond_recent_threads_before_refusing()
    test_instagram_checks_empty_search_overlay_before_text_query()
    test_instagram_search_result_advances_new_message_overlay()
    test_instagram_search_activates_overlay_before_setting_query()
    test_instagram_rejects_group_title_for_individual_recipient()
    test_instagram_existing_search_ignores_account_results_and_group_rows()
    test_instagram_new_recipient_requires_explicit_mode()
    test_instagram_ambiguous_page_match_never_enters_new_recipient_mode()
    test_instagram_default_fallback_requires_followed_account()
    test_instagram_display_name_without_row_proof_requires_exact_handle()
    test_instagram_exact_handle_profile_following_verification()
    test_instagram_exact_handle_opened_thread_uses_profile_link()
    test_instagram_new_thread_preserves_exact_handle_for_opened_thread_proof()
    test_instagram_inbox_navigation_uses_valid_javascript()
    test_instagram_tool_preserves_explicit_new_recipient_authorization()
    test_instagram_lexical_composer_uses_verified_unicode_paste()
    test_instagram_revalidates_thread_and_composer_before_trusted_paste()
    test_instagram_trusted_enter_failure_is_outcome_unknown()
    test_instagram_trusted_paste_restores_clipboard()
    test_instagram_clipboard_snapshot_and_restore_fail_closed()
    test_instagram_empty_paste_never_sends()
    test_instagram_send_requires_one_new_visible_message()
    test_instagram_post_enter_bridge_failure_becomes_unconfirmed()
    test_instagram_send_failure_preserves_attempt_metadata()
    test_instagram_failures_are_actionable_and_private()
    test_instagram_redacts_path_shaped_recipient_metadata()
    test_planner_routes_social_send_intents_to_gated_tools()
    test_planner_routes_recipient_first_and_calls()
    test_dispatch_routes_natural_imessage_to_approval_gated_send()
    test_dispatch_keeps_both_instagram_recipient_modes_approval_gated()
    print("Social connector smoke passed")


if __name__ == "__main__":
    main()
