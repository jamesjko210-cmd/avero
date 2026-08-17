from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis_v2.agent.types import RiskLevel


APPROVED_EXECUTION_SUCCEEDED = "succeeded"
APPROVED_EXECUTION_FAILED = "failed"
APPROVED_EXECUTION_OUTCOME_UNKNOWN = "outcome_unknown"

RESERVED_OUTCOME_BOOLEAN_KEYS = frozenset(
    {
        "execution_outcome_unknown",
        "outcome_known",
        "timed_out",
        "post_start_failed",
        "output_capture_failed",
        "durability_uncertain",
        "side_effect_possible",
        "handler_invoked",
        "executed_handler",
        "handler_exception_after_invocation",
    }
)
POST_START_UNCERTAINTY_KEYS = (
    "timed_out",
    "post_start_failed",
    "output_capture_failed",
    "durability_uncertain",
)
UNCERTAIN_HANDLER_EXCEPTION_RISKS = frozenset(
    {
        RiskLevel.PERSONAL_DATA.name,
        RiskLevel.EXTERNAL_SIDE_EFFECT.name,
        RiskLevel.HIGH_RISK.name,
    }
)
MALFORMED_OUTCOME_REASONS = frozenset(
    {
        "ok_malformed",
        "metadata_malformed",
        "reserved_boolean_malformed",
    }
)


@dataclass(frozen=True)
class ApprovedExecutionOutcome:
    outcome: str
    succeeded: bool
    failed: bool
    outcome_unknown: bool
    reason: str


def _outcome(outcome: str, reason: str) -> ApprovedExecutionOutcome:
    return ApprovedExecutionOutcome(
        outcome=outcome,
        succeeded=outcome == APPROVED_EXECUTION_SUCCEEDED,
        failed=outcome == APPROVED_EXECUTION_FAILED,
        outcome_unknown=outcome == APPROVED_EXECUTION_OUTCOME_UNKNOWN,
        reason=reason,
    )


def _risk_name(risk: Any) -> str | None:
    if type(risk) is RiskLevel:
        return risk.name
    if type(risk) is str:
        normalized = risk.strip().upper()
        if normalized in RiskLevel.__members__:
            return normalized
    return None


def classify_approved_execution_outcome(
    *,
    ok: Any,
    metadata: Any,
    risk: Any,
) -> ApprovedExecutionOutcome:
    """Classify one invoked approved action without performing I/O or mutating input."""
    if type(ok) is not bool:
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "ok_malformed")
    if type(metadata) is not dict:
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "metadata_malformed")
    if any(
        key in metadata and type(metadata[key]) is not bool
        for key in RESERVED_OUTCOME_BOOLEAN_KEYS
    ):
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "reserved_boolean_malformed")

    handler_invoked = metadata.get("handler_invoked")
    executed_handler = metadata.get("executed_handler")
    handler_exception = metadata.get("handler_exception_after_invocation") is True or (
        metadata.get("failure_kind") == "tool_error" and handler_invoked is True
    )
    if (
        handler_invoked is not None
        and executed_handler is not None
        and handler_invoked is not executed_handler
    ):
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "invocation_truth_contradiction")
    if handler_exception and (
        ok is True or handler_invoked is False or executed_handler is False
    ):
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "handler_exception_contradiction")
    if (
        metadata.get("execution_outcome_unknown") is True
        and metadata.get("outcome_known") is True
    ):
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "outcome_truth_contradiction")

    if metadata.get("execution_outcome_unknown") is True:
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "explicit_outcome_unknown")
    if metadata.get("outcome_known") is False:
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "explicit_outcome_not_known")
    if any(metadata.get(key) is True for key in POST_START_UNCERTAINTY_KEYS):
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "post_start_uncertainty")
    if ok is False and metadata.get("side_effect_possible") is True:
        return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "failed_side_effect_possible")
    if handler_exception:
        risk_name = _risk_name(risk)
        if risk_name is None or risk_name in UNCERTAIN_HANDLER_EXCEPTION_RISKS:
            return _outcome(APPROVED_EXECUTION_OUTCOME_UNKNOWN, "risky_handler_exception")

    if ok is True:
        return _outcome(APPROVED_EXECUTION_SUCCEEDED, "exact_success")
    return _outcome(APPROVED_EXECUTION_FAILED, "exact_failure")
