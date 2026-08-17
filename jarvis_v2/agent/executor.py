from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    declare_failure_guidance,
    declare_outcome_unknown_failure,
)
from jarvis_v2.agent.types import ApprovalArgumentResolution, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.tools import contacts_connector, contacts_fuzzy
from jarvis_v2.tools.permissions import PermissionPolicy
from jarvis_v2.tools.registry import (
    Tool,
    ToolArgumentValidation,
    ToolRegistry,
    validate_tool_arguments,
)


MAX_EXECUTOR_FIELD_CHARS = 160
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
UNREADABLE_TEXT = "<unreadable>"
# Only channels that genuinely need a phone/email *handle* go through pre-approval
# Contacts resolution and the stale-approval handle guard. Name-based channels
# (KakaoTalk, Instagram) type the recipient *name* into the app's own search box,
# so they never need an Apple Contacts handle — the recipient name shown on the
# approval screen is the verification step. Keeping Kakao/Instagram out of this set
# lets you reach app-only friends (e.g. a Kakao friend not in Apple Contacts).
CONTACT_RESOLVED_ACTION_TOOLS = {"send_imessage", "send_email", "call_contact"}
APPROVAL_RESOLVER_RESERVED_METADATA_KEYS = frozenset(
    {
        "failure_kind",
        "requires_confirmation",
        "risk_level",
        "risk_value",
        "toolset",
        "planned_args",
        "planned_args_display",
        "planner_reason",
        "executed_handler",
        "handler_invoked",
        "authorizes_retry",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approval_id",
        "reused_pending_approval",
    }
)
_AUTO_MUTATION_EXECUTION_SEAL = object()


@dataclass(frozen=True)
class _AutoMutationExecutionEnvelope:
    tool_name: str
    public_args: dict[str, Any]
    handler_args: dict[str, Any]
    private_values: tuple[str, ...]
    seal: object


def _short(value: object, *, limit: int = MAX_EXECUTOR_FIELD_CHARS) -> str:
    if value is None:
        text = ""
    else:
        try:
            text = str(value)
        except Exception:
            text = UNREADABLE_TEXT
    text = LOCAL_PATH_RE.sub("<local-path>", text).strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_arg_display(value: Any, *, key: str = "") -> Any:
    if key.endswith("_binding") or key == "target_binding":
        return "bound to reviewed memory version"
    if isinstance(value, str):
        return _short(value)
    if isinstance(value, dict):
        displayed: dict[str, Any] = {}
        for item_key, item in value.items():
            safe_key = _short(item_key)
            displayed[safe_key] = _safe_arg_display(item, key=safe_key)
        return displayed
    if isinstance(value, (list, tuple)):
        return [_safe_arg_display(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _short(value)


def _planned_arg_keys(args: Any) -> list[str]:
    if type(args) is not dict or not args:
        return []
    return ["<redacted>"]


def _contains_private_execution_value(value: Any, private_values: tuple[str, ...]) -> bool:
    if isinstance(value, str):
        return any(private and private in value for private in private_values)
    if isinstance(value, dict):
        return any(
            _contains_private_execution_value(key, private_values)
            or _contains_private_execution_value(item, private_values)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_private_execution_value(item, private_values) for item in value)
    return False


def _argument_contract_failure_result(
    action: PlannedAction,
    tool: Tool,
    validation: ToolArgumentValidation,
    *,
    approved: bool = False,
) -> ToolResult:
    output = (
        "The planned arguments do not match this tool's registered contract, so nothing ran. "
        "Correct the request fields and types before retrying."
    )
    recovery_action = "Correct the request fields and types before retrying."
    contract = (
        tool.approval_argument_contract
        if approved and tool.approval_argument_contract is not None
        else tool.argument_contract
    )
    allowed_keys = {
        field.name for field in (contract.fields if contract is not None else ())
    }
    known_planned_keys = (
        sorted(key for key in action.args if type(key) is str and key in allowed_keys)
        if type(action.args) is dict
        else []
    )
    if validation.unknown_keys:
        known_planned_keys.append("<unknown>")
    return ToolResult(
        action.tool_name,
        False,
        output,
        declare_failure_guidance(
            {
                "failure_kind": "tool_arguments_invalid",
                "argument_validation_status": validation.status,
                "argument_contract_version": contract.version if contract is not None else None,
                "planned_arg_keys": known_planned_keys,
                "required_arg_keys": sorted(
                    field.name for field in (contract.fields if contract is not None else ()) if field.required
                ),
                "allowed_arg_keys": sorted(
                    field.name for field in (contract.fields if contract is not None else ())
                ),
                "missing_arg_keys": list(validation.missing_keys),
                "unknown_arg_keys": list(validation.unknown_keys),
                "type_mismatch_arg_keys": list(validation.type_mismatch_keys),
                "requires_confirmation": False,
                "executed_handler": False,
                "handler_invoked": False,
                "risk_level": tool.risk.name,
                "risk_value": int(tool.risk),
                "toolset": tool.toolset,
                "authorizes_retry": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
            output=output,
            action=recovery_action,
        ),
    )


def _argument_contract_plan_blocked_result(action: PlannedAction) -> ToolResult:
    output = (
        "Another action in this mutation plan has invalid arguments, so nothing in the plan ran. "
        "Correct the invalid action's request fields and types, then submit the plan again."
    )
    recovery_action = (
        "Correct the invalid action's request fields and types, then submit the plan again."
    )
    return ToolResult(
        action.tool_name,
        False,
        output,
        declare_failure_guidance(
            {
                "failure_kind": "tool_argument_plan_preflight_blocked",
                "planned_arg_keys": _planned_arg_keys(action.args),
                "requires_confirmation": False,
                "executed_handler": False,
                "handler_invoked": False,
                "authorizes_retry": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
            output=output,
            action=recovery_action,
        ),
    )


def _contact_handle_for_action(
    action: PlannedAction,
    match: contacts_connector.ContactMatch,
) -> str:
    if action.tool_name == "send_email":
        emails = match.all_emails()
        return emails[0] if emails else match.email
    if action.tool_name == "call_contact":
        from jarvis_v2.tools import call_connector

        return call_connector.call_handle_for_mode(
            match,
            action.args.get("mode"),
        )
    return match.handle


def _contact_choices(
    action: PlannedAction,
    matches: list[contacts_connector.ContactMatch],
) -> list[dict[str, str]]:
    choices: list[dict[str, str]] = []
    for match in matches[:5]:
        choices.append(
            {
                "name": _short(match.name),
                "handle": _short(_contact_handle_for_action(action, match)),
            }
        )
    return choices


def _contact_resolution_failure(
    action: PlannedAction,
    *,
    query: str,
    status: str,
    output: str,
    matches: list[contacts_connector.ContactMatch] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    recovery_action: str | None = None,
    recovery_commands: tuple[str, ...] = (),
) -> ToolResult:
    metadata = {
        "failure_kind": f"contact_resolution_{status}",
        "requires_confirmation": False,
        "executed_handler": False,
        "planned_args": action.args,
        "planned_args_display": _safe_arg_display(action.args),
        "planner_reason": action.reason,
        "contact_resolution_status": status,
        "contact_resolution_query": _short(query),
        "contact_resolution_match_count": len(matches or []),
        "contact_resolution_matches": _contact_choices(action, matches or []),
        "handler_invoked": False,
        "authorizes_retry": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if recovery_action is not None:
        metadata = declare_failure_guidance(
            metadata,
            output=output,
            action=recovery_action,
            commands=recovery_commands,
        )
    return ToolResult(action.tool_name, False, output, metadata)


def _resolve_contact_action_before_approval(action: PlannedAction) -> tuple[PlannedAction, dict[str, Any]] | ToolResult:
    if action.tool_name not in CONTACT_RESOLVED_ACTION_TOOLS:
        return action, {}
    recipient = action.args.get("to")
    if not isinstance(recipient, str) or not recipient.strip():
        return action, {}
    query = recipient.strip()
    if contacts_connector.looks_like_handle(query):
        if action.tool_name == "call_contact":
            from jarvis_v2.tools import call_connector

            if not call_connector.call_handle_is_valid(
                action.args.get("mode"),
                query,
            ):
                return _contact_resolution_failure(
                    action,
                    query=query,
                    status="invalid_phone_handle",
                    output=(
                        "A phone call requires a phone number; an email address cannot be dialed. "
                        "I did not queue an approval. Give a phone number, then run "
                        "`call <phone number>`, or submit a fresh FaceTime audio/video request."
                    ),
                    recovery_action=(
                        "Give a phone number, then run `call <phone number>`, or submit a fresh "
                        "FaceTime audio/video request."
                    ),
                    recovery_commands=("call <phone number>",),
                )
        return action, {"contact_resolution_status": "already_handle"}

    try:
        matches = contacts_connector.resolve_contact(query)
    except Exception as exc:
        return _contact_resolution_failure(
            action,
            query=query,
            status="unavailable",
            output=(
                f"I couldn't verify the contact for '{_short(query)}' right now, "
                "so I did not queue an approval for the request. Run `setup check`, restore "
                "macOS Contacts permission, then run `find contact <name>` before trying "
                "the request again."
            ),
            matches=[],
            extra_metadata={"exception_type": type(exc).__name__},
            recovery_action=(
                "Run `setup check`, restore macOS Contacts permission, then run "
                "`find contact <name>` before trying the request again."
            ),
            recovery_commands=("setup check", "find contact <name>"),
        )
    if not matches:
        suggestion = contacts_fuzzy.suggestion_clause(query)
        return _contact_resolution_failure(
            action,
            query=query,
            status="not_found",
            output=(
                f"I couldn't find a contact matching '{_short(query)}', so I did not queue "
                f"an approval for the request.{suggestion} Correct or add the contact, then "
                "run `find contact <name>` before trying the request again."
            ),
            matches=[],
            extra_metadata={"contact_resolution_suggestion_clause": _short(suggestion)} if suggestion else None,
            recovery_action=(
                "Correct or add the contact, then run `find contact <name>` before trying "
                "the request again."
            ),
            recovery_commands=("find contact <name>",),
        )
    if len(matches) > 1:
        names = ", ".join(_short(match.name) for match in matches[:5])
        return _contact_resolution_failure(
            action,
            query=query,
            status="ambiguous",
            output=f"I found multiple contacts for '{_short(query)}' ({names}). Which one should I use?",
            matches=matches,
        )

    match = matches[0]
    handle = _contact_handle_for_action(action, match)
    if action.tool_name == "call_contact" and not handle:
        return _contact_resolution_failure(
            action,
            query=query,
            status="missing_phone_handle",
            output=(
                f"I found {_short(match.name)}, but that contact has no phone number. I did not "
                "queue an approval. Add a phone number, run `find contact <name>`, then submit "
                "a fresh phone-call request; or request FaceTime audio/video."
            ),
            matches=matches,
            recovery_action=(
                "Add a phone number, run `find contact <name>`, then submit a fresh phone-call "
                "request; or request FaceTime audio/video."
            ),
            recovery_commands=("find contact <name>",),
        )
    if not handle:
        return _contact_resolution_failure(
            action,
            query=query,
            status="missing_handle",
            output=(
                f"I found {_short(match.name)}, but not a usable handle for "
                f"{action.tool_name}. I did not queue an approval. Add the required phone "
                "number or email address in Contacts, then run `find contact <name>` before "
                "trying the request again."
            ),
            matches=matches,
            recovery_action=(
                "Add the required phone number or email address in Contacts, then run "
                "`find contact <name>` before trying the request again."
            ),
            recovery_commands=("find contact <name>",),
        )

    resolved_args = dict(action.args)
    resolved_args["to"] = handle
    return PlannedAction(action.tool_name, resolved_args, action.reason), {
        "contact_resolution_status": "resolved",
        "contact_resolution_query": _short(query),
        "contact_resolution_name": _short(match.name),
        "contact_resolution_handle": _short(handle),
    }


def _approval_required_result(action: PlannedAction, tool: Tool, reason: str, extra_metadata: dict[str, Any] | None = None) -> ToolResult:
    metadata = dict(extra_metadata or {})
    metadata.update({
        "failure_kind": "approval_required",
        "requires_confirmation": True,
        "risk_level": tool.risk.name,
        "risk_value": int(tool.risk),
        "toolset": tool.toolset,
        "planned_args": action.args,
        "planned_args_display": _safe_arg_display(action.args),
        "planner_reason": action.reason,
        "executed_handler": False,
    })
    output = reason
    if extra_metadata and extra_metadata.get("contact_resolution_status") == "resolved":
        output = (
            f"{reason} Recipient resolved as {extra_metadata.get('contact_resolution_name')} "
            f"({extra_metadata.get('contact_resolution_handle')})."
        )
    return ToolResult(action.tool_name, False, output, metadata)


def _stale_contact_approval_result(action: PlannedAction) -> ToolResult:
    recipient = _short(action.args.get("to"))
    request_kind = "call" if action.tool_name == "call_contact" else "send"
    recovery_action = (
        f"Reissue the {request_kind} request so Jarvis can queue a fresh approval with the "
        "resolved handle."
    )
    output = (
        f"This approved {request_kind} still has an unresolved recipient name ({recipient}). "
        f"I did not run it. {recovery_action}"
    )
    return ToolResult(
        action.tool_name,
        False,
        output,
        declare_failure_guidance(
            {
                "failure_kind": f"stale_{request_kind}_approval_unresolved_recipient",
                "requires_confirmation": False,
                "executed_handler": False,
                "planned_args": action.args,
                "planned_args_display": _safe_arg_display(action.args),
                "planner_reason": action.reason,
                "contact_resolution_status": "stale_unresolved_approval",
                "contact_resolution_query": recipient,
                "approval_rerun_blocked": True,
                "approval_rerun_block_reason": f"{request_kind}_recipient_not_resolved_before_approval",
                "handler_invoked": False,
                "authorizes_retry": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
            output=output,
            action=recovery_action,
        ),
    )


def _approval_argument_resolution_failure(
    action: PlannedAction,
    tool: Tool,
    *,
    status: str,
) -> ToolResult:
    output = (
        "Jarvis could not bind this destructive request to one immutable target, so no approval "
        "was queued and nothing ran. Reissue the request after checking the target."
    )
    recovery_action = "Reissue the request after checking the target."
    return ToolResult(
        action.tool_name,
        False,
        output,
        declare_failure_guidance(
            {
                "failure_kind": "approval_argument_resolution_failed",
                "approval_argument_resolution_status": status,
                "planned_arg_keys": _planned_arg_keys(action.args),
                "requires_confirmation": False,
                "executed_handler": False,
                "handler_invoked": False,
                "risk_level": tool.risk.name,
                "risk_value": int(tool.risk),
                "toolset": tool.toolset,
                "authorizes_retry": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
            output=output,
            action=recovery_action,
        ),
    )


def _internal_mutation_binding_failure(action: PlannedAction) -> ToolResult:
    output = (
        "Jarvis could not prepare the internal mutation binding, so nothing ran. "
        "Review the source state, then submit a fresh mutation request."
    )
    recovery_action = "Review the source state, then submit a fresh mutation request."
    return ToolResult(
        action.tool_name,
        False,
        output,
        declare_failure_guidance(
            {
                "failure_kind": "internal_mutation_binding_invalid",
                "requires_confirmation": False,
                "executed_handler": False,
                "handler_invoked": False,
                "authorizes_retry": False,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
            output=output,
            action=recovery_action,
        ),
    )


def _internal_mutation_binding_leak_result(action: PlannedAction) -> ToolResult:
    output = (
        "The mutation handler exposed private execution-binding data. Its outcome is blocked for "
        "review and it will not be retried automatically. Review the target state before issuing "
        "any new mutation request."
    )
    return ToolResult(
        action.tool_name,
        False,
        output,
        declare_outcome_unknown_failure(
            {
                "failure_kind": "internal_mutation_binding_leak",
                "requires_confirmation": False,
                "executed_handler": True,
                "handler_invoked": True,
                "authorizes_execution": False,
                "authorizes_completion_claim": False,
                "approval_granted": False,
            },
            output=output,
        ),
    )


class Executor:
    def __init__(self, registry: ToolRegistry, policy: PermissionPolicy):
        self.registry = registry
        self.policy = policy

    def validate_action_arguments(self, action: PlannedAction) -> ToolResult | None:
        try:
            tool = self.registry.get(action.tool_name)
        except KeyError:
            return None
        validation = validate_tool_arguments(tool, action.args)
        if validation.valid:
            return None
        return _argument_contract_failure_result(action, tool, validation)

    def prepare_auto_mutation_execution(
        self,
        action: PlannedAction,
        candidate_args: dict[str, Any],
    ) -> _AutoMutationExecutionEnvelope | ToolResult:
        """Seal a receipt-only argument extension before any mutation is claimed."""
        try:
            tool = self.registry.get(action.tool_name)
        except KeyError:
            return _internal_mutation_binding_failure(action)
        contract = tool.auto_mutation_contract
        private_contract = (
            contract.execution_argument_contract if contract is not None else None
        )
        if (
            type(candidate_args) is not dict
            or contract is None
            or contract.execution_args_builder is None
            or private_contract is None
            or set(action.args) - set(candidate_args)
            or any(candidate_args.get(key) != value for key, value in action.args.items())
        ):
            return _internal_mutation_binding_failure(action)
        private_args = {
            key: value for key, value in candidate_args.items() if key not in action.args
        }
        private_tool = replace(tool, argument_contract=private_contract)
        validation = validate_tool_arguments(private_tool, private_args)
        if not validation.valid:
            return _internal_mutation_binding_failure(action)
        private_values = tuple(
            value for value in private_args.values() if isinstance(value, str) and value
        )
        return _AutoMutationExecutionEnvelope(
            tool_name=action.tool_name,
            public_args=dict(action.args),
            handler_args=dict(candidate_args),
            private_values=private_values,
            seal=_AUTO_MUTATION_EXECUTION_SEAL,
        )

    @staticmethod
    def argument_contract_plan_blocked_result(action: PlannedAction) -> ToolResult:
        return _argument_contract_plan_blocked_result(action)

    def execute(
        self,
        action: PlannedAction,
        approved: bool = False,
        *,
        execution_envelope: _AutoMutationExecutionEnvelope | None = None,
    ) -> ToolResult:
        try:
            tool = self.registry.get(action.tool_name)
        except KeyError as exc:
            return ToolResult(
                action.tool_name,
                False,
                "Jarvis could not find the planned tool in the registry. Run `list tools`, then correct "
                "planner routing or the tool registration map before retry.",
                {
                    "failure_kind": "unknown_tool",
                    "requested_tool": _short(action.tool_name),
                    "planned_args": action.args,
                    "planned_args_display": _safe_arg_display(action.args),
                    "planner_reason": action.reason,
                    "requires_confirmation": False,
                    "executed_handler": False,
                    "error_type": type(exc).__name__,
                    "exception_type": type(exc).__name__,
                    "next_command": "list tools",
                    "recovery_commands": ["list tools"],
                    "retry_requires_registry_fix": True,
                    "authorizes_retry": False,
                    "authorizes_execution": False,
                    "authorizes_completion_claim": False,
                },
            )

        validation = validate_tool_arguments(tool, action.args, approved=approved)
        if not validation.valid:
            return _argument_contract_failure_result(
                action,
                tool,
                validation,
                approved=approved,
            )

        approval_metadata: dict[str, Any] = {}
        if approved and action.tool_name in CONTACT_RESOLVED_ACTION_TOOLS:
            recipient = action.args.get("to")
            if (
                action.tool_name == "call_contact"
                and isinstance(recipient, str)
                and recipient.strip()
                and contacts_connector.looks_like_handle(recipient)
            ):
                from jarvis_v2.tools import call_connector

                if not call_connector.call_handle_is_valid(
                    action.args.get("mode"),
                    recipient,
                ):
                    return _contact_resolution_failure(
                        action,
                        query=recipient,
                        status="approved_invalid_phone_handle",
                        output=(
                            "This approved phone call contains an email address instead of a phone "
                            "number, so I did not run it. Give a phone number, then run "
                            "`call <phone number>`, or submit a fresh FaceTime audio/video request."
                        ),
                        recovery_action=(
                            "Give a phone number, then run `call <phone number>`, or submit a fresh "
                            "FaceTime audio/video request."
                        ),
                        recovery_commands=("call <phone number>",),
                    )
            if isinstance(recipient, str) and recipient.strip() and not contacts_connector.looks_like_handle(recipient):
                return _stale_contact_approval_result(action)
        if not approved:
            resolved = _resolve_contact_action_before_approval(action)
            if isinstance(resolved, ToolResult):
                return resolved
            action, approval_metadata = resolved
            validation = validate_tool_arguments(tool, action.args)
            if not validation.valid:
                return _argument_contract_failure_result(action, tool, validation)

        if not approved and tool.approval_argument_resolver is not None:
            try:
                resolution = tool.approval_argument_resolver(dict(action.args))
            except Exception:
                return _approval_argument_resolution_failure(
                    action,
                    tool,
                    status="resolver_unavailable",
                )
            if isinstance(resolution, ToolResult):
                metadata = resolution.metadata
                if (
                    type(resolution.tool_name) is not str
                    or resolution.tool_name != action.tool_name
                    or resolution.ok is not False
                    or type(resolution.output) is not str
                    or type(metadata) is not dict
                    or metadata.get("requires_confirmation") is not False
                    or metadata.get("executed_handler") is not False
                    or metadata.get("handler_invoked") is not False
                    or metadata.get("authorizes_execution") is not False
                    or metadata.get("approval_granted") is not False
                ):
                    return _approval_argument_resolution_failure(
                        action,
                        tool,
                        status="resolver_refusal_malformed",
                    )
                return resolution
            if (
                not isinstance(resolution, ApprovalArgumentResolution)
                or type(resolution.args) is not dict
                or type(resolution.metadata) is not dict
                or APPROVAL_RESOLVER_RESERVED_METADATA_KEYS.intersection(
                    resolution.metadata
                )
            ):
                return _approval_argument_resolution_failure(
                    action,
                    tool,
                    status="resolver_result_malformed",
                )
            action = PlannedAction(action.tool_name, dict(resolution.args), action.reason)
            approval_metadata.update(resolution.metadata)
            validation = validate_tool_arguments(tool, action.args, approved=True)
            if not validation.valid:
                return _argument_contract_failure_result(
                    action,
                    tool,
                    validation,
                    approved=True,
                )

        if execution_envelope is not None:
            if (
                approved
                or type(execution_envelope) is not _AutoMutationExecutionEnvelope
                or execution_envelope.seal is not _AUTO_MUTATION_EXECUTION_SEAL
                or execution_envelope.tool_name != action.tool_name
                or execution_envelope.public_args != action.args
            ):
                return _internal_mutation_binding_failure(action)
        # Keep the public action intact for policy decisions and diagnostics.  Internal
        # receipt bindings are handler-only custody data and must never reach audits.
        handler_args = (
            execution_envelope.handler_args
            if execution_envelope is not None
            else action.args
        )

        decision = self.policy.check(tool, approved=approved)
        if not decision.allowed:
            return _approval_required_result(action, tool, decision.reason, approval_metadata)

        try:
            result = tool.handler(handler_args)
            if execution_envelope is not None and _contains_private_execution_value(
                result.output,
                execution_envelope.private_values,
            ):
                return _internal_mutation_binding_leak_result(action)
            if (
                execution_envelope is not None
                and isinstance(result.metadata, dict)
                and _contains_private_execution_value(
                    result.metadata,
                    execution_envelope.private_values,
                )
            ):
                return _internal_mutation_binding_leak_result(action)
            if not isinstance(result.tool_name, str) or result.tool_name != action.tool_name:
                return ToolResult(
                    action.tool_name,
                    False,
                    "Tool returned a receipt for a different planned action. Nothing was verified; run "
                    "`recent tool runs`, then inspect the tool registration before retry.",
                    {
                        "failure_kind": "tool_result_binding_mismatch",
                        "planned_args": action.args,
                        "planned_args_display": _safe_arg_display(action.args),
                        "planner_reason": action.reason,
                        "requires_confirmation": False,
                        "executed_handler": True,
                        "handler_invoked": True,
                        "side_effect_possible": True,
                        "execution_outcome_unknown": True,
                        "outcome_known": False,
                        "retry_safe": False,
                        "reported_tool_name": _short(result.tool_name) or "<unknown tool>",
                        "next_command": "recent tool runs",
                        "recovery_commands": ["recent tool runs", "list tools"],
                        "retry_requires_registry_fix": True,
                        "authorizes_retry": False,
                        "authorizes_execution": False,
                        "authorizes_completion_claim": False,
                    },
                )
            if result.ok is not True and result.ok is not False:
                return ToolResult(
                    action.tool_name,
                    False,
                    "Tool returned a malformed success flag. Nothing was verified; run `recent tool runs`, "
                    "then inspect the tool implementation before retry.",
                    {
                        "failure_kind": "tool_result_ok_malformed",
                        "planned_args": action.args,
                        "planned_args_display": _safe_arg_display(action.args),
                        "planner_reason": action.reason,
                        "requires_confirmation": False,
                        "executed_handler": True,
                        "handler_invoked": True,
                        "side_effect_possible": True,
                        "execution_outcome_unknown": True,
                        "outcome_known": False,
                        "retry_safe": False,
                        "next_command": "recent tool runs",
                        "recovery_commands": ["recent tool runs", "list tools"],
                        "retry_requires_registry_fix": True,
                        "authorizes_retry": False,
                        "authorizes_execution": False,
                        "authorizes_completion_claim": False,
                    },
                )
            if type(result.output) is not str:
                return ToolResult(
                    action.tool_name,
                    False,
                    "The tool handler ran, but its receipt output was malformed. The action outcome is "
                    "uncertain; do not retry automatically. Run `recent tool runs`, then `execution recovery`.",
                    {
                        "failure_kind": "tool_result_output_malformed",
                        "planned_args": action.args,
                        "planned_args_display": _safe_arg_display(action.args),
                        "planner_reason": action.reason,
                        "requires_confirmation": False,
                        "executed_handler": True,
                        "handler_invoked": True,
                        "side_effect_possible": True,
                        "execution_outcome_unknown": True,
                        "outcome_known": False,
                        "retry_safe": False,
                        "next_command": "recent tool runs",
                        "recovery_commands": ["recent tool runs", "execution recovery"],
                        "retry_requires_recovery_review": True,
                        "authorizes_retry": False,
                        "authorizes_execution": False,
                        "authorizes_completion_claim": False,
                    },
                )
            if type(result.metadata) is not dict:
                return ToolResult(
                    action.tool_name,
                    False,
                    "The tool handler ran, but its receipt metadata was malformed. The action outcome is "
                    "uncertain; do not retry automatically. Run `recent tool runs`, then `execution recovery`.",
                    {
                        "failure_kind": "tool_result_metadata_malformed",
                        "planned_args": action.args,
                        "planned_args_display": _safe_arg_display(action.args),
                        "planner_reason": action.reason,
                        "requires_confirmation": False,
                        "executed_handler": True,
                        "handler_invoked": True,
                        "side_effect_possible": True,
                        "execution_outcome_unknown": True,
                        "outcome_known": False,
                        "retry_safe": False,
                        "next_command": "recent tool runs",
                        "recovery_commands": ["recent tool runs", "execution recovery"],
                        "retry_requires_recovery_review": True,
                        "authorizes_retry": False,
                        "authorizes_execution": False,
                        "authorizes_completion_claim": False,
                    },
                )
            result.metadata["handler_invoked"] = True
            # Invocation truth is executor-owned. A handler must not be able to
            # forge a pre-execution receipt by returning this reserved field.
            result.metadata["executed_handler"] = True
            return result
        except Exception as exc:
            risky_exception = tool.risk in {
                RiskLevel.PERSONAL_DATA,
                RiskLevel.EXTERNAL_SIDE_EFFECT,
                RiskLevel.HIGH_RISK,
            }
            uncertainty_metadata = (
                {
                    "side_effect_possible": True,
                    "execution_outcome_unknown": True,
                    "outcome_known": False,
                    "retry_safe": False,
                }
                if risky_exception
                else {}
            )
            return ToolResult(
                action.tool_name,
                False,
                "Tool failed while running. Run `recent tool runs`, then `execution recovery` for the "
                f"latest failed run; retry only after recovery review. ({type(exc).__name__})",
                {
                    "failure_kind": "tool_error",
                    "planned_args": action.args,
                    "planned_args_display": _safe_arg_display(action.args),
                    "planner_reason": action.reason,
                    "requires_confirmation": False,
                    "executed_handler": True,
                    "handler_invoked": True,
                    "handler_exception_after_invocation": True,
                    "error_type": type(exc).__name__,
                    "exception_type": type(exc).__name__,
                    "next_command": "recent tool runs",
                    "recovery_commands": ["recent tool runs", "execution recovery", "execution health report"],
                    "retry_requires_recovery_review": True,
                    "authorizes_retry": False,
                    "authorizes_execution": False,
                    "authorizes_completion_claim": False,
                    **uncertainty_metadata,
                },
            )
