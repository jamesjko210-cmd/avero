"""Offline, content-free closure verifier for the Jarvis V3 functional preview.

The verifier binds the private August 15 gate and supervised-proof dispositions
to one immutable, history-free public candidate.  It never publishes, installs
or starts services, changes schedules, reads account data, or performs a live
action.  With ``--run-aggregate`` it runs only the isolated smoke suite inside
the already-built candidate, using a credential-free temporary environment.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import hmac
import json
from math import isfinite
import os
from pathlib import Path
import re
import secrets
import selectors
import signal
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Iterable, Mapping

from jarvis_v2.scripts.public_release_candidate import (
    CandidateBlob,
    CandidateBuildError,
    PublicCandidateMarker,
    SOURCE_COMMIT_RE,
    _candidate_blobs,
    _candidate_state_fd,
    _candidate_tree_has_finalized_modes_fd,
    _manifest_digest,
    _open_candidate_root,
    _run_git,
    _source_commit,
    _validated_source_root,
)
from jarvis_v2.scripts.public_release_preflight import (
    EXTRA_DENY_ENV,
    PreflightConfigError,
    _OPEN_DIRECTORY_FLAGS,
    _read_regular_at,
    _scan_release_tree_fd,
    extra_deny_literals_from_env,
)
from jarvis_v2.scripts import v3_recovery_receipt


GATE_DOCUMENT = "AUGUST_15_FUNCTIONAL_PREVIEW.md"
PROOF_RUNBOOK = "V3_SUPERVISED_PROOF_RUNBOOK.md"
EVIDENCE_ONLY_PATHS = frozenset({GATE_DOCUMENT, PROOF_RUNBOOK})
MAX_EVIDENCE_DOCUMENT_BYTES = 512 * 1024
AGGREGATE_TIMEOUT_SECONDS = 3 * 60 * 60
MAX_AGGREGATE_DIAGNOSTIC_BYTES = 64 * 1024
MAX_AGGREGATE_CAPTURE_BYTES = 1024 * 1024

GATE_STATUS_BLOCK_START = "<!-- JARVIS_V3_PREVIEW_GATE_STATUS_V1"
GATE_STATUS_BLOCK_END = "END_JARVIS_V3_PREVIEW_GATE_STATUS_V1 -->"
PROOF_STATUS_BLOCK_START = "<!-- JARVIS_V3_PREVIEW_PROOF_STATUS_V1"
PROOF_STATUS_BLOCK_END = "END_JARVIS_V3_PREVIEW_PROOF_STATUS_V1 -->"
PROOF_ATTESTATION_BLOCK_START = "<!-- JARVIS_V3_PREVIEW_PROOF_ATTESTATION_V1"
PROOF_ATTESTATION_BLOCK_END = "END_JARVIS_V3_PREVIEW_PROOF_ATTESTATION_V1 -->"
VALID_DISPOSITIONS = frozenset(
    {"passed", "pending", "deferred", "disabled", "failed", "unknown"}
)
REQUIRED_GATE_IDS = (
    "v3_environment_isolation",
    "daemon_entrypoints_fail_closed",
    "isolated_regression_suite",
    "cli_and_dashboard_startup",
    "public_quickstart",
    "ordinary_conversation_route",
    "restart_persistence",
    "mixed_session_coherence",
    "safety_contracts",
    "bounded_failure_recovery",
)
PROOF_LANES = (
    "voice",
    "ocr",
    "google_calendar",
    "apple_reminders",
    "telegram",
    "imessage",
    "instagram",
    "kakaotalk",
    "calls",
    "recovery_services",
)
PROOF_DISPOSITION_POLICY = {
    # These are the minimum functional-preview demonstrations. Documentation
    # may not defer or disable them and still call the preview ready.
    "voice": frozenset({"passed"}),
    "ocr": frozenset({"passed"}),
    "google_calendar": frozenset({"passed"}),
    # Individual account connectors are optional, but each must either have a
    # real supervised pass or be explicitly disabled. ``deferred`` is too
    # ambiguous for a connector advertised by the preview.
    "apple_reminders": frozenset({"passed", "disabled"}),
    "telegram": frozenset({"passed", "disabled"}),
    "imessage": frozenset({"passed", "disabled"}),
    "instagram": frozenset({"passed", "disabled"}),
    "kakaotalk": frozenset({"passed", "disabled"}),
    # Call channels are present in the carried-forward capability inventory,
    # but they are explicitly outside this functional preview. Closure must
    # record that descope rather than silently omitting the advertised lane.
    "calls": frozenset({"disabled"}),
    # Recovery/service activation is explicitly outside the August 15 cut.
    "recovery_services": frozenset({"passed", "deferred", "disabled"}),
}
CONNECTOR_PROOF_LANES = (
    "apple_reminders",
    "telegram",
    "imessage",
    "instagram",
    "kakaotalk",
)
CONNECTOR_EVIDENCE_CLASSES = {
    "apple_reminders": frozenset({"bounded_read_observed"}),
    "telegram": frozenset({"approval_bound_sender_success_recipient_confirmed"}),
    "imessage": frozenset({"approval_bound_sender_success_recipient_confirmed"}),
    "instagram": frozenset({"approval_bound_sender_success_recipient_confirmed"}),
    "kakaotalk": frozenset({"approval_bound_target_verified_recipient_confirmed"}),
}
RECOVERY_PROOF_LANE = "recovery_services"
RECOVERY_ATTESTATION_LANES = ("recovery_network", "recovery_reboot")
RECOVERY_EVIDENCE_CLASSES = {
    # These names deliberately describe only what the receipt observer can
    # establish: route status, boot identity, and expected process presence.
    # They are not end-to-end service, Internet, delivery, or workload proof.
    "recovery_network": frozenset(
        {"supervised_route_and_process_transition_observed"}
    ),
    "recovery_reboot": frozenset(
        {"supervised_boot_and_process_transition_observed"}
    ),
}
ATTESTATION_LANES = CONNECTOR_PROOF_LANES + RECOVERY_ATTESTATION_LANES
ATTESTATION_DIGEST_DOMAIN = "jarvis-v3-preview-connector-proof-attestation-v1"
RECOVERY_ATTESTATION_DIGEST_DOMAIN = (
    "jarvis-v3-preview-recovery-proof-attestation-v3"
)
RECOVERY_RECEIPT_MAX_BYTES = 64 * 1024
PROCESS_GROUP_CLEANUP_SECONDS = 5.0
CLOSURE_RECEIPT_SCHEMA = "jarvis-v3-preview-closure-receipt"
CLOSURE_RECEIPT_VERSION = 2
CLOSURE_RECEIPT_DIGEST_DOMAIN = "jarvis-v3-preview-closure-receipt-v2"
CLOSURE_RECEIPT_ANCHOR_ENV = "JARVIS_V3_CLOSURE_RECEIPT_ANCHOR"
CLOSURE_RECEIPT_HMAC_DOMAIN = "jarvis-v3-preview-closure-receipt-hmac-v1"
CLOSURE_PRIVATE_DENY_COMMITMENT_DOMAIN = (
    "jarvis-v3-preview-private-deny-review-v1"
)
CLOSURE_RECEIPT_MAX_BYTES = 32 * 1024
CLOSURE_RECEIPT_FILE_MODE = 0o600
CLOSURE_RECEIPT_DIRECTORY_MODE = 0o700
CLOSURE_RECEIPT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")
CLOSURE_RUNNER_PATH = "jarvis_v2/scripts/v3_preview_closure.py"
CLOSURE_SMOKE_INVENTORY_PATH = "jarvis_v2/scripts/smoke_test_all.py"
CLOSURE_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "version",
        "recorded_at_utc",
        "candidate_source_commit",
        "source_head_commit",
        "candidate_manifest_digest",
        "candidate_files_scanned",
        "candidate_bytes_scanned",
        "candidate_finalized_immutable",
        "candidate_stable",
        "source_commit_is_ancestor",
        "selected_content_matches",
        "closure_evidence_sha256",
        "closure_runner_sha256",
        "smoke_module_inventory_sha256",
        "aggregate_environment_contract_sha256",
        "aggregate_module_count",
        "aggregate_duration_seconds",
        "aggregate_passed",
        "receipt_hmac_sha256",
        "private_deny_review_supplied",
        "private_deny_literal_count",
        "private_deny_review_commitment_sha256",
        "paths_included",
        "private_content_included",
        "publishes",
        "publication_authorized",
        "cutover_authorized",
        "starts_services",
        "changes_schedules",
        "runs_live_actions",
        "receipt_sha256",
    }
)

_AGGREGATE_ENVIRONMENT_CONTRACT = {
    "version": 1,
    "generated_private_paths": [
        "HOME",
        "JARVIS_SMOKE_SUITE_LOCK_PATH",
        "JARVIS_V3_ENV",
        "PYTHONPATH",
        "TMPDIR",
        "XDG_CACHE_HOME",
    ],
    "fixed_values": {
        "JARVIS_V3_ENABLE_DAEMONS": "0",
        "JARVIS_V3_ENABLE_SCHEDULER": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    },
    "bounded_inherited_keys": ["LANG", "LC_ALL", "PATH"],
}


class PreviewClosureError(ValueError):
    """Raised when closure inputs cannot be inspected without weakening gates."""


class ClosureReceiptOutcomeUnknown(PreviewClosureError):
    """Raised when a linked receipt could not be durably removed after failure."""


@dataclass(frozen=True)
class ClosureReceiptVerification:
    receipt_sha256: str
    self_consistent: bool
    aggregate_execution_authenticated: bool


@dataclass(frozen=True)
class ClosureReceiptCreation:
    receipt_sha256: str


@dataclass(frozen=True)
class CandidateSnapshot:
    marker: PublicCandidateMarker
    files_scanned: int
    bytes_scanned: int
    finalized_immutable: bool


@dataclass(frozen=True)
class AggregateResult:
    ran: bool
    passed: bool
    module_count: int
    duration_seconds: float | None
    failure_reason: str | None = None
    failed_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectorAttestationAudit:
    structurally_valid: bool
    attested_lanes: tuple[str, ...]
    missing_lanes: tuple[str, ...]
    invalid_lanes: tuple[str, ...]
    mismatched_lanes: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryAttestationAudit:
    structurally_valid: bool
    attested: bool
    missing: bool
    invalid: bool
    mismatched: bool


@dataclass(frozen=True)
class RecoveryReceiptSnapshot:
    sha256: str
    finalized_on: date | None


@dataclass(frozen=True)
class ClosureEvidenceBinding:
    bound: bool
    connector_proof_state_bound: bool
    recovery_proof_state_bound: bool
    required_gate_state_bound: bool


@dataclass(frozen=True)
class PreviewClosureReport:
    ready: bool
    preconditions_ready: bool
    required_gate_total: int
    required_gate_checked: int
    supervised_proof_total: int
    supervised_proof_passed: int
    supervised_proof_deferred: int
    supervised_proof_disabled: int
    supervised_proof_pending: int
    supervised_proof_failed: int
    supervised_proof_unknown: int
    pending_supervised_proofs: tuple[str, ...]
    failed_supervised_proofs: tuple[str, ...]
    unknown_supervised_proofs: tuple[str, ...]
    unaccepted_supervised_proofs: tuple[str, ...]
    connector_proof_attested: int
    unattested_connector_proofs: tuple[str, ...]
    invalid_connector_proof_attestations: tuple[str, ...]
    mismatched_connector_proof_attestations: tuple[str, ...]
    recovery_proof_attested: bool
    recovery_proof_attestation_missing: bool
    recovery_proof_attestation_invalid: bool
    recovery_proof_attestation_mismatched: bool
    connector_proof_state_candidate_bound: bool
    recovery_proof_state_candidate_bound: bool
    required_gate_state_candidate_bound: bool
    closure_evidence_state_candidate_bound: bool
    candidate_source_commit: str | None
    candidate_manifest_digest: str | None
    candidate_files_scanned: int | None
    candidate_bytes_scanned: int | None
    candidate_finalized_immutable: bool
    candidate_preflight_before: bool
    candidate_preflight_after: bool
    candidate_stable: bool
    source_commit_exists: bool
    source_commit_is_ancestor: bool
    selected_content_matches: bool
    evidence_only_head_drift: bool
    private_deny_review_supplied: bool
    private_deny_literal_count: int
    aggregate_requested: bool
    aggregate_ran: bool
    aggregate_passed: bool
    aggregate_module_count: int
    aggregate_duration_seconds: float | None
    aggregate_failure_reason: str | None
    aggregate_failed_modules: tuple[str, ...]
    blocking_codes: tuple[str, ...]
    private_deny_review_commitment_sha256: str | None = field(
        default=None,
        repr=False,
    )

    def summary_json_payload(self) -> dict[str, object]:
        return {
            "schema": "jarvis-v3-preview-closure",
            "version": 3,
            "ready": self.ready,
            "preconditions_ready": self.preconditions_ready,
            "required_gate_total": self.required_gate_total,
            "required_gate_checked": self.required_gate_checked,
            "supervised_proof_total": self.supervised_proof_total,
            "supervised_proof_passed": self.supervised_proof_passed,
            "supervised_proof_deferred": self.supervised_proof_deferred,
            "supervised_proof_disabled": self.supervised_proof_disabled,
            "supervised_proof_pending": self.supervised_proof_pending,
            "supervised_proof_failed": self.supervised_proof_failed,
            "supervised_proof_unknown": self.supervised_proof_unknown,
            "pending_supervised_proofs": list(self.pending_supervised_proofs),
            "failed_supervised_proofs": list(self.failed_supervised_proofs),
            "unknown_supervised_proofs": list(self.unknown_supervised_proofs),
            "unaccepted_supervised_proofs": list(self.unaccepted_supervised_proofs),
            "connector_proof_attested": self.connector_proof_attested,
            "unattested_connector_proofs": list(self.unattested_connector_proofs),
            "invalid_connector_proof_attestations": list(
                self.invalid_connector_proof_attestations
            ),
            "mismatched_connector_proof_attestations": list(
                self.mismatched_connector_proof_attestations
            ),
            "recovery_proof_attested": self.recovery_proof_attested,
            "recovery_proof_attestation_missing": (
                self.recovery_proof_attestation_missing
            ),
            "recovery_proof_attestation_invalid": (
                self.recovery_proof_attestation_invalid
            ),
            "recovery_proof_attestation_mismatched": (
                self.recovery_proof_attestation_mismatched
            ),
            "connector_proof_state_candidate_bound": (
                self.connector_proof_state_candidate_bound
            ),
            "recovery_proof_state_candidate_bound": (
                self.recovery_proof_state_candidate_bound
            ),
            "required_gate_state_candidate_bound": (
                self.required_gate_state_candidate_bound
            ),
            "closure_evidence_state_candidate_bound": (
                self.closure_evidence_state_candidate_bound
            ),
            "candidate_source_commit": self.candidate_source_commit,
            "candidate_manifest_digest": self.candidate_manifest_digest,
            "candidate_files_scanned": self.candidate_files_scanned,
            "candidate_bytes_scanned": self.candidate_bytes_scanned,
            "candidate_finalized_immutable": self.candidate_finalized_immutable,
            "candidate_preflight_before": self.candidate_preflight_before,
            "candidate_preflight_after": self.candidate_preflight_after,
            "candidate_stable": self.candidate_stable,
            "source_commit_exists": self.source_commit_exists,
            "source_commit_is_ancestor": self.source_commit_is_ancestor,
            "selected_content_matches": self.selected_content_matches,
            "evidence_only_head_drift": self.evidence_only_head_drift,
            "private_deny_review_supplied": self.private_deny_review_supplied,
            "private_deny_literal_count": self.private_deny_literal_count,
            "aggregate_requested": self.aggregate_requested,
            "aggregate_ran": self.aggregate_ran,
            "aggregate_passed": self.aggregate_passed,
            "aggregate_module_count": self.aggregate_module_count,
            "aggregate_duration_seconds": self.aggregate_duration_seconds,
            "aggregate_failure_reason": self.aggregate_failure_reason,
            "aggregate_failed_modules": list(self.aggregate_failed_modules),
            "blocking_codes": list(self.blocking_codes),
            "paths_included": False,
            "private_content_included": False,
            "read_only_except_isolated_aggregate": True,
            "publishes": False,
            "publication_authorized": False,
            "cutover_authorized": False,
            "starts_services": False,
            "changes_schedules": False,
            "runs_live_actions": False,
        }


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                dict(payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PreviewClosureError("closure receipt is invalid") from exc


def _domain_digest(domain: str, value: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(value)
    return digest.hexdigest()


def _receipt_digest(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    return _domain_digest(CLOSURE_RECEIPT_DIGEST_DOMAIN, _canonical_json_bytes(unsigned))


def _validated_receipt_anchor(anchor: str) -> bytes:
    if type(anchor) is not str or re.fullmatch(r"[0-9a-f]{64}", anchor) is None:
        raise PreviewClosureError("closure receipt anchor is invalid")
    return bytes.fromhex(anchor)


def _receipt_anchor_from_environment(
    environment: Mapping[str, str],
    *,
    required: bool,
) -> str | None:
    anchor = environment.get(CLOSURE_RECEIPT_ANCHOR_ENV)
    if anchor in {None, ""}:
        if required:
            raise PreviewClosureError("closure receipt anchor is missing")
        return None
    _validated_receipt_anchor(anchor)
    return anchor


def _receipt_hmac(payload: Mapping[str, object], anchor: str) -> str:
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    unsigned.pop("receipt_hmac_sha256", None)
    return hmac.new(
        _validated_receipt_anchor(anchor),
        CLOSURE_RECEIPT_HMAC_DOMAIN.encode("ascii")
        + b"\0"
        + _canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()


def _private_deny_review_commitment(
    anchor: str,
    literals: tuple[bytes, ...],
) -> str:
    if not literals:
        raise PreviewClosureError("private deny review is missing")
    anchor_bytes = _validated_receipt_anchor(anchor)
    serialized = _canonical_json_bytes(
        {"literals": [literal.decode("utf-8") for literal in literals]}
    )
    return hmac.new(
        anchor_bytes,
        CLOSURE_PRIVATE_DENY_COMMITMENT_DOMAIN.encode("ascii") + b"\0" + serialized,
        hashlib.sha256,
    ).hexdigest()


def _aggregate_environment_contract_digest() -> str:
    return _domain_digest(
        "jarvis-v3-preview-aggregate-environment-contract-v1",
        _canonical_json_bytes(_AGGREGATE_ENVIRONMENT_CONTRACT),
    )


def _candidate_receipt_bindings(
    candidate_root: str | Path,
) -> tuple[str, tuple[str, ...], PublicCandidateMarker]:
    opened = _open_candidate_root(candidate_root)
    if opened is None:
        raise PreviewClosureError("candidate receipt binding is invalid")
    _root, root_fd, _root_stat = opened
    try:
        state = _candidate_state_fd(root_fd)
        if state is None:
            raise PreviewClosureError("candidate receipt binding is invalid")
        by_path = {blob.path: blob.content for blob in state.blobs}
        runner = by_path.get(CLOSURE_RUNNER_PATH)
        inventory_source = by_path.get(CLOSURE_SMOKE_INVENTORY_PATH)
        if runner is None or inventory_source is None:
            raise PreviewClosureError("candidate receipt binding is invalid")
        try:
            tree = ast.parse(inventory_source, filename=CLOSURE_SMOKE_INVENTORY_PATH)
        except (SyntaxError, ValueError) as exc:
            raise PreviewClosureError("candidate smoke inventory is invalid") from exc
        assignments = [
            node
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and (
                (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "TEST_MODULES"
                )
                or (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "TEST_MODULES"
                )
            )
        ]
        if len(assignments) != 1:
            raise PreviewClosureError("candidate smoke inventory is invalid")
        value = assignments[0].value
        if not isinstance(value, (ast.List, ast.Tuple)):
            raise PreviewClosureError("candidate smoke inventory is invalid")
        modules: list[str] = []
        for element in value.elts:
            if (
                not isinstance(element, ast.Constant)
                or type(element.value) is not str
                or re.fullmatch(r"[A-Za-z0-9_.]+", element.value) is None
            ):
                raise PreviewClosureError("candidate smoke inventory is invalid")
            modules.append(element.value)
        if not modules or len(modules) != len(set(modules)):
            raise PreviewClosureError("candidate smoke inventory is invalid")
        return hashlib.sha256(runner).hexdigest(), tuple(modules), state.marker
    finally:
        os.close(root_fd)


def _smoke_module_inventory_digest(modules: tuple[str, ...]) -> str:
    return _domain_digest(
        "jarvis-v3-preview-smoke-module-inventory-v1",
        _canonical_json_bytes({"modules": list(modules)}),
    )


def _closure_evidence_digest(source: Path) -> str:
    digests = _proof_state_digests(
        _read_evidence_document(source, GATE_DOCUMENT),
        _read_evidence_document(source, PROOF_RUNBOOK),
    )
    if digests is None:
        raise PreviewClosureError("closure evidence is invalid")
    return digests[0]


def _receipt_payload(
    source_root: str | Path,
    candidate_root: str | Path,
    report: PreviewClosureReport,
    *,
    environ: Mapping[str, str] | None = None,
    active_contract_manifest: str | Path | None = None,
    network_recovery_receipt: str | Path | None = None,
    reboot_recovery_receipt: str | Path | None = None,
) -> dict[str, object]:
    if (
        not report.ready
        or not report.aggregate_ran
        or not report.aggregate_passed
        or report.aggregate_failure_reason is not None
        or report.aggregate_failed_modules
    ):
        raise PreviewClosureError("only a ready aggregate run can produce a receipt")
    environment = dict(os.environ if environ is None else environ)
    receipt_anchor = _receipt_anchor_from_environment(environment, required=True)
    assert receipt_anchor is not None
    private_deny_literals = extra_deny_literals_from_env(environment)
    aggregate_private_deny_commitment = _private_deny_review_commitment(
        receipt_anchor,
        private_deny_literals,
    )
    if not secrets.compare_digest(
        str(report.private_deny_review_commitment_sha256),
        aggregate_private_deny_commitment,
    ):
        raise PreviewClosureError("private deny review changed after aggregate")
    source = _validated_source_root(source_root)
    source_head = _source_commit(source)
    current = evaluate_preview_closure(
        source,
        candidate_root,
        run_aggregate=False,
        environ=environment,
        active_contract_manifest=active_contract_manifest,
        network_recovery_receipt=network_recovery_receipt,
        reboot_recovery_receipt=reboot_recovery_receipt,
    )
    if (
        not current.preconditions_ready
        or current.blocking_codes != ("aggregate_not_run",)
        or report.candidate_source_commit != current.candidate_source_commit
        or report.candidate_manifest_digest != current.candidate_manifest_digest
        or report.candidate_files_scanned != current.candidate_files_scanned
        or report.candidate_bytes_scanned != current.candidate_bytes_scanned
        or not current.candidate_stable
        or not current.candidate_finalized_immutable
        or not current.selected_content_matches
        or not current.private_deny_review_supplied
        or report.private_deny_literal_count != current.private_deny_literal_count
    ):
        raise PreviewClosureError("closure state changed before receipt commit")
    runner_sha256, modules, binding_marker = _candidate_receipt_bindings(candidate_root)
    if (
        binding_marker.source_commit != report.candidate_source_commit
        or binding_marker.manifest_digest != report.candidate_manifest_digest
    ):
        raise PreviewClosureError("candidate changed before receipt commit")
    if report.aggregate_module_count != len(modules):
        raise PreviewClosureError("aggregate module inventory changed")
    payload: dict[str, object] = {
        "schema": CLOSURE_RECEIPT_SCHEMA,
        "version": CLOSURE_RECEIPT_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "candidate_source_commit": report.candidate_source_commit,
        "source_head_commit": source_head,
        "candidate_manifest_digest": report.candidate_manifest_digest,
        "candidate_files_scanned": report.candidate_files_scanned,
        "candidate_bytes_scanned": report.candidate_bytes_scanned,
        "candidate_finalized_immutable": report.candidate_finalized_immutable,
        "candidate_stable": report.candidate_stable,
        "source_commit_is_ancestor": report.source_commit_is_ancestor,
        "selected_content_matches": report.selected_content_matches,
        "closure_evidence_sha256": _closure_evidence_digest(source),
        "closure_runner_sha256": runner_sha256,
        "smoke_module_inventory_sha256": _smoke_module_inventory_digest(modules),
        "aggregate_environment_contract_sha256": (
            _aggregate_environment_contract_digest()
        ),
        "aggregate_module_count": report.aggregate_module_count,
        "aggregate_duration_seconds": report.aggregate_duration_seconds,
        "aggregate_passed": True,
        "private_deny_review_supplied": report.private_deny_review_supplied,
        "private_deny_literal_count": report.private_deny_literal_count,
        "private_deny_review_commitment_sha256": (
            aggregate_private_deny_commitment
        ),
        "paths_included": False,
        "private_content_included": False,
        "publishes": False,
        "publication_authorized": False,
        "cutover_authorized": False,
        "starts_services": False,
        "changes_schedules": False,
        "runs_live_actions": False,
    }
    payload["receipt_hmac_sha256"] = _receipt_hmac(payload, receipt_anchor)
    payload["receipt_sha256"] = _receipt_digest(payload)
    return payload


def _path_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _open_receipt_parent(
    output: str | Path,
    *,
    source_root: str | Path,
    candidate_root: str | Path,
) -> tuple[Path, int, str]:
    raw = Path(output)
    if (
        not raw.is_absolute()
        or raw.name in {"", ".", ".."}
        or CLOSURE_RECEIPT_NAME_RE.fullmatch(raw.name) is None
    ):
        raise PreviewClosureError("closure receipt destination is unsafe")
    try:
        parent = raw.parent.resolve(strict=True)
        source = Path(source_root).resolve(strict=True)
        candidate = Path(candidate_root).resolve(strict=True)
    except OSError as exc:
        raise PreviewClosureError("closure receipt destination is unsafe") from exc
    if raw.parent != parent or _path_within(parent, source) or _path_within(parent, candidate):
        raise PreviewClosureError("closure receipt destination is unsafe")
    parent_fd: int | None = None
    try:
        named = os.stat(parent, follow_symlinks=False)
        parent_fd = os.open(parent, _OPEN_DIRECTORY_FLAGS)
        opened = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != CLOSURE_RECEIPT_DIRECTORY_MODE
        ):
            raise PreviewClosureError("closure receipt destination is unsafe")
        result = parent_fd
        parent_fd = None
        return parent, result, raw.name
    except OSError as exc:
        raise PreviewClosureError("closure receipt destination is unsafe") from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("short closure receipt write")
        offset += written


def write_closure_receipt(
    output: str | Path,
    *,
    source_root: str | Path,
    candidate_root: str | Path,
    report: PreviewClosureReport,
    environ: Mapping[str, str] | None = None,
    active_contract_manifest: str | Path | None = None,
    network_recovery_receipt: str | Path | None = None,
    reboot_recovery_receipt: str | Path | None = None,
) -> ClosureReceiptCreation:
    payload = _receipt_payload(
        source_root,
        candidate_root,
        report,
        environ=environ,
        active_contract_manifest=active_contract_manifest,
        network_recovery_receipt=network_recovery_receipt,
        reboot_recovery_receipt=reboot_recovery_receipt,
    )
    content = _canonical_json_bytes(payload)
    if len(content) > CLOSURE_RECEIPT_MAX_BYTES:
        raise PreviewClosureError("closure receipt is too large")
    _parent, parent_fd, name = _open_receipt_parent(
        output,
        source_root=source_root,
        candidate_root=candidate_root,
    )
    temporary = f".{name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    linked = False
    committed_successfully = False
    cleanup_outcome_unknown = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise PreviewClosureError("closure receipt already exists")
        descriptor = os.open(
            temporary,
            flags,
            CLOSURE_RECEIPT_FILE_MODE,
            dir_fd=parent_fd,
        )
        _write_all(descriptor, content)
        os.fsync(descriptor)
        os.link(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary, dir_fd=parent_fd)
        temporary = ""
        committed = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(committed.st_mode)
            or committed.st_uid != os.getuid()
            or stat.S_IMODE(committed.st_mode) != CLOSURE_RECEIPT_FILE_MODE
            or committed.st_nlink != 1
            or (committed.st_dev, committed.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise PreviewClosureError("closure receipt commit is unsafe")
        os.fsync(parent_fd)
        postflight = verify_closure_receipt(
            output,
            source_root=source_root,
            candidate_root=candidate_root,
            environ=environ,
            active_contract_manifest=active_contract_manifest,
            network_recovery_receipt=network_recovery_receipt,
            reboot_recovery_receipt=reboot_recovery_receipt,
        )
        if (
            postflight.receipt_sha256 != payload["receipt_sha256"]
            or not postflight.self_consistent
            or not postflight.aggregate_execution_authenticated
        ):
            raise PreviewClosureError("closure receipt postflight failed")
        committed_successfully = True
        return ClosureReceiptCreation(
            receipt_sha256=str(payload["receipt_sha256"]),
        )
    except PreviewClosureError:
        raise
    except OSError as exc:
        raise PreviewClosureError("closure receipt could not be committed safely") from exc
    finally:
        if temporary:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except OSError:
                pass
        if linked and not committed_successfully and descriptor is not None:
            try:
                opened = os.fstat(descriptor)
            except OSError:
                cleanup_outcome_unknown = True
            else:
                try:
                    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    try:
                        os.fsync(parent_fd)
                    except OSError:
                        cleanup_outcome_unknown = True
                except OSError:
                    cleanup_outcome_unknown = True
                else:
                    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                        cleanup_outcome_unknown = True
                    else:
                        try:
                            os.unlink(name, dir_fd=parent_fd)
                            os.fsync(parent_fd)
                        except OSError:
                            cleanup_outcome_unknown = True
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                if linked and not committed_successfully:
                    cleanup_outcome_unknown = True
        try:
            os.close(parent_fd)
        except OSError:
            if linked and not committed_successfully:
                cleanup_outcome_unknown = True
        if cleanup_outcome_unknown:
            raise ClosureReceiptOutcomeUnknown("closure receipt outcome is unknown")


def _read_closure_receipt(
    receipt: str | Path,
    *,
    source_root: str | Path,
    candidate_root: str | Path,
) -> tuple[dict[str, object], bytes]:
    _parent, parent_fd, name = _open_receipt_parent(
        receipt,
        source_root=source_root,
        candidate_root=candidate_root,
    )
    try:
        try:
            content, file_stat = _read_regular_at(
                parent_fd,
                name,
                max_bytes=CLOSURE_RECEIPT_MAX_BYTES,
                require_single_link=True,
            )
        except (OSError, OverflowError) as exc:
            raise PreviewClosureError("closure receipt is unreadable") from exc
        if (
            file_stat.st_uid != os.getuid()
            or stat.S_IMODE(file_stat.st_mode) != CLOSURE_RECEIPT_FILE_MODE
        ):
            raise PreviewClosureError("closure receipt custody is invalid")
    finally:
        os.close(parent_fd)
    duplicate = False

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        payload = json.loads(content.decode("ascii"), object_pairs_hook=strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreviewClosureError("closure receipt is invalid") from exc
    if (
        duplicate
        or type(payload) is not dict
        or frozenset(payload) != CLOSURE_RECEIPT_KEYS
        or _canonical_json_bytes(payload) != content
    ):
        raise PreviewClosureError("closure receipt is invalid")
    return payload, content


def _validated_receipt_payload(payload: Mapping[str, object]) -> None:
    if payload.get("schema") != CLOSURE_RECEIPT_SCHEMA or payload.get(
        "version"
    ) != CLOSURE_RECEIPT_VERSION:
        raise PreviewClosureError("closure receipt is invalid")
    for key in (
        "candidate_source_commit",
        "source_head_commit",
    ):
        value = payload.get(key)
        if type(value) is not str or SOURCE_COMMIT_RE.fullmatch(value) is None:
            raise PreviewClosureError("closure receipt is invalid")
    for key in (
        "candidate_manifest_digest",
        "closure_evidence_sha256",
        "closure_runner_sha256",
        "smoke_module_inventory_sha256",
        "aggregate_environment_contract_sha256",
        "receipt_hmac_sha256",
        "private_deny_review_commitment_sha256",
        "receipt_sha256",
    ):
        value = payload.get(key)
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise PreviewClosureError("closure receipt is invalid")
    timestamp = payload.get("recorded_at_utc")
    if type(timestamp) is not str or not timestamp.endswith("Z"):
        raise PreviewClosureError("closure receipt is invalid")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise PreviewClosureError("closure receipt is invalid") from exc
    if parsed.tzinfo != timezone.utc or parsed.isoformat().replace("+00:00", "Z") != timestamp:
        raise PreviewClosureError("closure receipt is invalid")
    for key in (
        "candidate_files_scanned",
        "candidate_bytes_scanned",
        "aggregate_module_count",
        "private_deny_literal_count",
    ):
        value = payload.get(key)
        if type(value) is not int or value <= 0:
            raise PreviewClosureError("closure receipt is invalid")
    duration = payload.get("aggregate_duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not isfinite(float(duration))
        or float(duration) < 0
    ):
        raise PreviewClosureError("closure receipt is invalid")
    required_true = (
        "candidate_finalized_immutable",
        "candidate_stable",
        "source_commit_is_ancestor",
        "selected_content_matches",
        "aggregate_passed",
        "private_deny_review_supplied",
    )
    required_false = (
        "paths_included",
        "private_content_included",
        "publishes",
        "publication_authorized",
        "cutover_authorized",
        "starts_services",
        "changes_schedules",
        "runs_live_actions",
    )
    if any(payload.get(key) is not True for key in required_true) or any(
        payload.get(key) is not False for key in required_false
    ):
        raise PreviewClosureError("closure receipt is invalid")
    if not secrets.compare_digest(
        str(payload["receipt_sha256"]), _receipt_digest(payload)
    ):
        raise PreviewClosureError("closure receipt digest is invalid")


def verify_closure_receipt(
    receipt: str | Path,
    *,
    source_root: str | Path,
    candidate_root: str | Path,
    environ: Mapping[str, str] | None = None,
    active_contract_manifest: str | Path | None = None,
    network_recovery_receipt: str | Path | None = None,
    reboot_recovery_receipt: str | Path | None = None,
) -> ClosureReceiptVerification:
    environment = dict(os.environ if environ is None else environ)
    receipt_anchor = _receipt_anchor_from_environment(environment, required=False)
    payload, _content = _read_closure_receipt(
        receipt,
        source_root=source_root,
        candidate_root=candidate_root,
    )
    _validated_receipt_payload(payload)
    source = _validated_source_root(source_root)
    current = evaluate_preview_closure(
        source,
        candidate_root,
        run_aggregate=False,
        environ=environment,
        active_contract_manifest=active_contract_manifest,
        network_recovery_receipt=network_recovery_receipt,
        reboot_recovery_receipt=reboot_recovery_receipt,
    )
    if (
        not current.preconditions_ready
        or current.blocking_codes != ("aggregate_not_run",)
        or payload["candidate_source_commit"] != current.candidate_source_commit
        or payload["source_head_commit"] != _source_commit(source)
        or payload["candidate_manifest_digest"] != current.candidate_manifest_digest
        or payload["candidate_files_scanned"] != current.candidate_files_scanned
        or payload["candidate_bytes_scanned"] != current.candidate_bytes_scanned
        or payload["closure_evidence_sha256"] != _closure_evidence_digest(source)
        or payload["private_deny_review_supplied"]
        != current.private_deny_review_supplied
        or payload["private_deny_literal_count"]
        != current.private_deny_literal_count
    ):
        raise PreviewClosureError("closure receipt no longer matches its evidence")
    runner_sha256, modules, binding_marker = _candidate_receipt_bindings(candidate_root)
    if (
        binding_marker.source_commit != payload["candidate_source_commit"]
        or binding_marker.manifest_digest != payload["candidate_manifest_digest"]
        or payload["closure_runner_sha256"] != runner_sha256
        or payload["smoke_module_inventory_sha256"]
        != _smoke_module_inventory_digest(modules)
        or payload["aggregate_environment_contract_sha256"]
        != _aggregate_environment_contract_digest()
        or payload["aggregate_module_count"] != len(modules)
    ):
        raise PreviewClosureError("closure receipt execution binding is invalid")
    receipt_sha256 = str(payload["receipt_sha256"])
    return ClosureReceiptVerification(
        receipt_sha256=receipt_sha256,
        self_consistent=True,
        aggregate_execution_authenticated=(
            receipt_anchor is not None
            and secrets.compare_digest(
                _receipt_hmac(payload, receipt_anchor),
                str(payload["receipt_hmac_sha256"]),
            )
            and secrets.compare_digest(
                _private_deny_review_commitment(
                    receipt_anchor,
                    extra_deny_literals_from_env(environment),
                ),
                str(payload["private_deny_review_commitment_sha256"]),
            )
        ),
    )


def _preflight_receipt_output(
    output: str | Path,
    *,
    source_root: str | Path,
    candidate_root: str | Path,
) -> None:
    _parent, parent_fd, name = _open_receipt_parent(
        output,
        source_root=source_root,
        candidate_root=candidate_root,
    )
    try:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise PreviewClosureError("closure receipt already exists")
    finally:
        os.close(parent_fd)


def _read_evidence_document(source: Path, name: str) -> str:
    root_fd: int | None = None
    try:
        root_fd = os.open(source, _OPEN_DIRECTORY_FLAGS)
        content, _ = _read_regular_at(
            root_fd,
            name,
            max_bytes=MAX_EVIDENCE_DOCUMENT_BYTES,
            require_single_link=True,
        )
    except (OSError, OverflowError) as exc:
        raise PreviewClosureError("required closure evidence is unreadable") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
    if b"\x00" in content:
        raise PreviewClosureError("required closure evidence is invalid")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreviewClosureError("required closure evidence is invalid") from exc


def _structured_statuses(
    text: str,
    *,
    start: str,
    end: str,
    expected_ids: tuple[str, ...],
) -> tuple[dict[str, str], bool]:
    pattern = re.compile(
        rf"^{re.escape(start)}\n(?P<body>.*?)\n{re.escape(end)}$",
        re.MULTILINE | re.DOTALL,
    )
    matches = tuple(pattern.finditer(text))
    if len(matches) != 1:
        return ({item: "unknown" for item in expected_ids}, False)
    entries: dict[str, str] = {}
    valid = True
    for line in matches[0].group("body").splitlines():
        entry = re.fullmatch(
            r"(?P<identifier>[a-z][a-z0-9_]*)=(?P<status>[a-z]+)",
            line,
        )
        if entry is None:
            valid = False
            continue
        identifier = entry.group("identifier")
        status = entry.group("status")
        if identifier in entries or status not in VALID_DISPOSITIONS:
            valid = False
            continue
        entries[identifier] = status
    if set(entries) != set(expected_ids) or len(entries) != len(expected_ids):
        valid = False
    if not valid:
        return ({item: "unknown" for item in expected_ids}, False)
    return entries, True


def _required_gate_counts(gate: str) -> tuple[int, int, bool]:
    statuses, structurally_valid = _structured_statuses(
        gate,
        start=GATE_STATUS_BLOCK_START,
        end=GATE_STATUS_BLOCK_END,
        expected_ids=REQUIRED_GATE_IDS,
    )
    passed = sum(value == "passed" for value in statuses.values())
    return len(statuses), passed, structurally_valid and passed == len(REQUIRED_GATE_IDS)


def _proof_dispositions(runbook: str) -> tuple[dict[str, str], bool]:
    return _structured_statuses(
        runbook,
        start=PROOF_STATUS_BLOCK_START,
        end=PROOF_STATUS_BLOCK_END,
        expected_ids=PROOF_LANES,
    )


def _connector_attestation_digest(
    lane: str,
    evidence_class: str,
    observed_on: str,
) -> str:
    canonical = "\n".join(
        (
            ATTESTATION_DIGEST_DOMAIN,
            lane,
            "passed",
            evidence_class,
            observed_on,
        )
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _recovery_attestation_digest(
    lane: str,
    evidence_class: str,
    observed_on: str,
    receipt_sha256: str,
    source_commit: str,
    contract_set_sha256: str,
) -> str:
    canonical = "\n".join(
        (
            RECOVERY_ATTESTATION_DIGEST_DOMAIN,
            lane,
            "passed",
            evidence_class,
            observed_on,
            receipt_sha256,
            source_commit,
            contract_set_sha256,
        )
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _attestation_entries(
    runbook: str,
) -> tuple[dict[str, str], frozenset[str], bool]:
    pattern = re.compile(
        rf"^{re.escape(PROOF_ATTESTATION_BLOCK_START)}\n(?P<body>.*?)\n"
        rf"{re.escape(PROOF_ATTESTATION_BLOCK_END)}$",
        re.MULTILINE | re.DOTALL,
    )
    matches = tuple(pattern.finditer(runbook))
    if len(matches) != 1:
        return {}, frozenset(ATTESTATION_LANES), True
    entries: dict[str, str] = {}
    invalid_lanes: set[str] = set()
    global_error = False
    for line in matches[0].group("body").splitlines():
        identifier, separator, value = line.partition("=")
        if not separator or identifier not in ATTESTATION_LANES:
            global_error = True
            continue
        if identifier in entries or not value:
            invalid_lanes.add(identifier)
            continue
        entries[identifier] = value
    invalid_lanes.update(set(ATTESTATION_LANES) - set(entries))
    return entries, frozenset(invalid_lanes), global_error


def _connector_attestations(
    runbook: str,
    dispositions: Mapping[str, str],
    *,
    today: date | None = None,
) -> ConnectorAttestationAudit:
    """Validate content-free connector evidence bound to exact dispositions.

    The receipt intentionally retains no target, payload, account, path, tool-run ID,
    or approval ID.  Its digest is an integrity checksum, not an identity signature:
    it makes a status-only or partial edit fail closed, while the operator remains
    responsible for recording a receipt only after the supervised observation.
    """

    entries, structurally_invalid_lanes, global_error = _attestation_entries(runbook)
    structurally_valid = not global_error and not (
        set(CONNECTOR_PROOF_LANES) & structurally_invalid_lanes
    )
    passed_connectors = {
        lane for lane in CONNECTOR_PROOF_LANES if dispositions.get(lane) == "passed"
    }
    if not entries:
        return ConnectorAttestationAudit(
            structurally_valid=False,
            attested_lanes=(),
            missing_lanes=tuple(sorted(passed_connectors)),
            invalid_lanes=(),
            mismatched_lanes=(),
        )

    attested: set[str] = set()
    missing: set[str] = set()
    invalid: set[str] = set()
    mismatched: set[str] = set()
    current_date = today or datetime.now(timezone.utc).date()
    for lane in CONNECTOR_PROOF_LANES:
        disposition = dispositions.get(lane, "unknown")
        value = entries.get(lane)
        if value is None:
            if disposition == "passed":
                missing.add(lane)
            continue
        if value == "none":
            if disposition == "passed":
                missing.add(lane)
            continue

        fields = value.split("|")
        if len(fields) != 4:
            invalid.add(lane)
            continue
        recorded_disposition, evidence_class, observed_on, recorded_digest = fields
        if recorded_disposition != disposition or recorded_disposition != "passed":
            mismatched.add(lane)
            continue
        if evidence_class not in CONNECTOR_EVIDENCE_CLASSES[lane]:
            invalid.add(lane)
            continue
        try:
            parsed_date = date.fromisoformat(observed_on)
        except ValueError:
            invalid.add(lane)
            continue
        if parsed_date.isoformat() != observed_on or parsed_date > current_date:
            invalid.add(lane)
            continue
        expected_digest = _connector_attestation_digest(
            lane,
            evidence_class,
            observed_on,
        )
        if not re.fullmatch(r"[0-9a-f]{64}", recorded_digest) or not secrets.compare_digest(
            recorded_digest,
            expected_digest,
        ):
            invalid.add(lane)
            continue
        attested.add(lane)

    # A stale attestation on a disabled, pending, failed, or unknown lane is a
    # mismatch rather than historical authority that can silently survive.
    for lane in CONNECTOR_PROOF_LANES:
        if dispositions.get(lane) != "passed" and entries.get(lane) not in {None, "none"}:
            mismatched.add(lane)
            attested.discard(lane)

    return ConnectorAttestationAudit(
        structurally_valid=structurally_valid,
        attested_lanes=tuple(sorted(attested)),
        missing_lanes=tuple(sorted(missing)),
        invalid_lanes=tuple(sorted(invalid)),
        mismatched_lanes=tuple(sorted(mismatched)),
    )


def _recovery_attestation(
    runbook: str,
    dispositions: Mapping[str, str],
    *,
    source: Path | None = None,
    candidate_marker: PublicCandidateMarker | None = None,
    active_contract_manifest: str | Path | None = None,
    network_receipt: str | Path | None = None,
    reboot_receipt: str | Path | None = None,
    today: date | None = None,
) -> RecoveryAttestationAudit:
    """Validate two finalized, narrowly scoped receipts for the optional lane.

    A valid receipt attests only the observer's bounded route, boot-identity,
    process-presence, scheduler-disabled, and source/contract checks.  It is not
    end-to-end service, Internet, message-delivery, or workload-recovery proof.
    """

    entries, structurally_invalid_lanes, global_error = _attestation_entries(runbook)
    structurally_valid = not global_error and not (
        set(RECOVERY_ATTESTATION_LANES) & structurally_invalid_lanes
    )
    disposition = dispositions.get(RECOVERY_PROOF_LANE, "unknown")
    values = {lane: entries.get(lane) for lane in RECOVERY_ATTESTATION_LANES}
    if disposition != "passed":
        stale = any(value not in {None, "none"} for value in values.values())
        return RecoveryAttestationAudit(
            structurally_valid=structurally_valid,
            attested=False,
            missing=False,
            invalid=False,
            mismatched=stale,
        )
    missing = any(value in {None, "none"} for value in values.values()) or (
        active_contract_manifest is None
        or network_receipt is None
        or reboot_receipt is None
    )
    if missing:
        return RecoveryAttestationAudit(
            structurally_valid=structurally_valid,
            attested=False,
            missing=True,
            invalid=False,
            mismatched=False,
        )
    parsed: dict[str, tuple[str, str, str, str]] = {}
    invalid = False
    mismatched = False
    current_date = today or datetime.now(timezone.utc).date()
    for lane in RECOVERY_ATTESTATION_LANES:
        value = values[lane]
        assert value is not None
        fields = value.split("|")
        if len(fields) != 7:
            invalid = True
            continue
        (
            recorded_disposition,
            evidence_class,
            observed_on,
            receipt_sha256,
            source_commit,
            contract_set_sha256,
            recorded_digest,
        ) = fields
        if recorded_disposition != "passed":
            mismatched = True
            continue
        if evidence_class not in RECOVERY_EVIDENCE_CLASSES[lane]:
            invalid = True
            continue
        try:
            parsed_date = date.fromisoformat(observed_on)
        except ValueError:
            parsed_date = None
        if (
            parsed_date is None
            or parsed_date.isoformat() != observed_on
            or parsed_date > current_date
            or not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256)
            or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source_commit)
            or not re.fullmatch(r"[0-9a-f]{64}", contract_set_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", recorded_digest)
        ):
            invalid = True
            continue
        expected_digest = _recovery_attestation_digest(
            lane,
            evidence_class,
            observed_on,
            receipt_sha256,
            source_commit,
            contract_set_sha256,
        )
        if not secrets.compare_digest(recorded_digest, expected_digest):
            invalid = True
            continue
        parsed[lane] = (
            receipt_sha256,
            source_commit,
            contract_set_sha256,
            observed_on,
        )
    if len(parsed) == len(RECOVERY_ATTESTATION_LANES):
        sources = {fields[1] for fields in parsed.values()}
        contracts = {fields[2] for fields in parsed.values()}
        if len(sources) != 1 or len(contracts) != 1:
            mismatched = True
        else:
            source_commit = next(iter(sources))
            if not _recovery_source_matches_candidate(
                source,
                source_commit,
                candidate_marker,
            ):
                mismatched = True
            receipt_paths = {
                "recovery_network": Path(network_receipt),
                "recovery_reboot": Path(reboot_receipt),
            }
            proof_kinds = {
                "recovery_network": "network",
                "recovery_reboot": "reboot",
            }
            for lane, receipt_path in receipt_paths.items():
                (
                    receipt_sha256,
                    recorded_source,
                    contract_set_sha256,
                    observed_on,
                ) = parsed[lane]
                before_snapshot = _stable_recovery_receipt_snapshot(receipt_path)
                try:
                    receipt_summary = v3_recovery_receipt.inspect_receipt(
                        receipt_path,
                        active_contract_manifest=Path(active_contract_manifest),
                    )
                except (OSError, v3_recovery_receipt.RecoveryReceiptError):
                    receipt_summary = None
                after_snapshot = _stable_recovery_receipt_snapshot(receipt_path)
                if (
                    before_snapshot is None
                    or after_snapshot is None
                    or not secrets.compare_digest(
                        before_snapshot.sha256,
                        after_snapshot.sha256,
                    )
                    or before_snapshot.finalized_on != after_snapshot.finalized_on
                    or not secrets.compare_digest(
                        before_snapshot.sha256,
                        receipt_sha256,
                    )
                    or before_snapshot.finalized_on is None
                    or before_snapshot.finalized_on.isoformat() != observed_on
                    or before_snapshot.finalized_on > current_date
                    or receipt_summary is None
                    or receipt_summary.get("proof_kind") != proof_kinds[lane]
                    or receipt_summary.get("proof_state") != "finalized"
                    or receipt_summary.get("verdict") != "proven"
                    or receipt_summary.get("scheduler_disabled") is not True
                    or receipt_summary.get("contract_bound") is not True
                    or receipt_summary.get("active_contract_bound") is not True
                    or receipt_summary.get("source_bound") is not True
                    or receipt_summary.get("source_commit") != recorded_source
                    or receipt_summary.get("contract_set_sha256") != contract_set_sha256
                    or receipt_summary.get("authorizes_execution") is not False
                    or receipt_summary.get("external_side_effects") is not False
                ):
                    invalid = True
    return RecoveryAttestationAudit(
        structurally_valid=structurally_valid,
        attested=structurally_valid and not invalid and not mismatched,
        missing=False,
        invalid=invalid,
        mismatched=mismatched,
    )


def _stable_recovery_receipt_content(path: Path) -> bytes | None:
    if not path.is_absolute():
        return None
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor: int | None = None
    descriptor: int | None = None
    try:
        parent_descriptor = os.open("/", directory_flags)
        for component in path.parent.parts[1:]:
            if component in {"", ".", ".."}:
                return None
            named_parent = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(named_parent.st_mode):
                return None
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            opened_parent = os.fstat(next_descriptor)
            if (
                not stat.S_ISDIR(opened_parent.st_mode)
                or (named_parent.st_dev, named_parent.st_ino)
                != (opened_parent.st_dev, opened_parent.st_ino)
            ):
                os.close(next_descriptor)
                return None
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        final_parent = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(final_parent.st_mode)
            or final_parent.st_uid != os.geteuid()
            or final_parent.st_mode & 0o077
        ):
            return None
        named_before = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(path.name, file_flags, dir_fd=parent_descriptor)
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(named_before.st_mode)
            or not stat.S_ISREG(opened_before.st_mode)
            or named_before.st_dev != opened_before.st_dev
            or named_before.st_ino != opened_before.st_ino
            or opened_before.st_nlink != 1
            or opened_before.st_uid != os.geteuid()
            or opened_before.st_mode & 0o077
            or opened_before.st_size < 1
            or opened_before.st_size > RECOVERY_RECEIPT_MAX_BYTES
        ):
            return None
        remaining = RECOVERY_RECEIPT_MAX_BYTES + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(16 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        opened_after = os.fstat(descriptor)
        named_after = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            len(content) > RECOVERY_RECEIPT_MAX_BYTES
            or (opened_before.st_dev, opened_before.st_ino, opened_before.st_size, opened_before.st_mtime_ns)
            != (opened_after.st_dev, opened_after.st_ino, opened_after.st_size, opened_after.st_mtime_ns)
            or (opened_after.st_dev, opened_after.st_ino)
            != (named_after.st_dev, named_after.st_ino)
        ):
            return None
        return content
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _stable_recovery_receipt_snapshot(path: Path) -> RecoveryReceiptSnapshot | None:
    content = _stable_recovery_receipt_content(path)
    if content is None:
        return None
    try:
        payload = v3_recovery_receipt._validate_payload(
            v3_recovery_receipt._load_canonical_json(content, "state_corrupt")
        )
        finalized_raw = payload.get("finalized_at")
        finalized_on = (
            v3_recovery_receipt._parse_utc_timestamp(finalized_raw).date()
            if finalized_raw is not None
            else None
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        v3_recovery_receipt.RecoveryReceiptError,
    ):
        return None
    return RecoveryReceiptSnapshot(
        sha256=hashlib.sha256(content).hexdigest(),
        finalized_on=finalized_on,
    )


def _stable_recovery_receipt_sha256(path: Path) -> str | None:
    content = _stable_recovery_receipt_content(path)
    return hashlib.sha256(content).hexdigest() if content is not None else None


def _recovery_source_matches_candidate(
    source: Path | None,
    receipt_source_commit: str,
    marker: PublicCandidateMarker | None,
) -> bool:
    if source is None or marker is None:
        return False
    try:
        _run_git(source, "merge-base", "--is-ancestor", receipt_source_commit, marker.source_commit)
        receipt_blobs = _candidate_blobs(source, receipt_source_commit)
        candidate_blobs = _candidate_blobs(source, marker.source_commit)
    except CandidateBuildError:
        return False
    # Recovery receipts are necessarily finalized before their attestations can
    # be written into the proof runbook.  The runbook is now part of the public
    # candidate, so comparing it here would create an impossible self-reference:
    # the receipt would have to name the hash of the commit that embeds the
    # receipt hash.  Bind the receipt to an ancestor with identical executable
    # source while leaving the exact evidence documents to
    # ``_proof_state_bound_to_candidate`` and the candidate manifest.
    def execution_identity(
        blobs: tuple[CandidateBlob, ...],
    ) -> tuple[tuple[str, bool, bytes], ...]:
        return tuple(
            (blob.path, blob.executable, blob.content)
            for blob in blobs
            if blob.path not in EVIDENCE_ONLY_PATHS
        )

    return execution_identity(receipt_blobs) == execution_identity(candidate_blobs)


def _machine_block(document: str, start: str, end: str) -> str | None:
    pattern = re.compile(
        rf"^{re.escape(start)}\n(?P<body>.*?)\n{re.escape(end)}$",
        re.MULTILINE | re.DOTALL,
    )
    matches = tuple(pattern.finditer(document))
    if len(matches) != 1:
        return None
    return f"{start}\n{matches[0].group('body')}\n{end}"


def _proof_state_digests(gate: str, runbook: str) -> tuple[str, str, str] | None:
    """Digest only the three content-free machine closure blocks."""

    gate_block = _machine_block(gate, GATE_STATUS_BLOCK_START, GATE_STATUS_BLOCK_END)
    proof_block = _machine_block(
        runbook,
        PROOF_STATUS_BLOCK_START,
        PROOF_STATUS_BLOCK_END,
    )
    attestation_block = _machine_block(
        runbook,
        PROOF_ATTESTATION_BLOCK_START,
        PROOF_ATTESTATION_BLOCK_END,
    )
    if gate_block is None or proof_block is None or attestation_block is None:
        return None
    gate_digest = hashlib.sha256(
        ("jarvis-v3-preview-required-gate-state-v1\n" + gate_block).encode("utf-8")
    ).hexdigest()
    connector_digest = hashlib.sha256(
        (
            "jarvis-v3-preview-proof-state-v1\n"
            + proof_block
            + "\n"
            + attestation_block
        ).encode("utf-8")
    ).hexdigest()
    canonical = (
        "jarvis-v3-preview-closure-evidence-state-v2\n"
        + gate_block
        + "\n"
        + proof_block
        + "\n"
        + attestation_block
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), gate_digest, connector_digest


def _proof_state_bound_to_candidate(
    source: Path,
    marker: PublicCandidateMarker,
    current_gate: str,
    current_runbook: str,
) -> ClosureEvidenceBinding:
    """Require the candidate source commit to contain the exact closure state."""

    current_digests = _proof_state_digests(current_gate, current_runbook)
    if current_digests is None:
        return ClosureEvidenceBinding(False, False, False, False)
    try:
        historical_gate = _run_git(
            source,
            "show",
            f"{marker.source_commit}:{GATE_DOCUMENT}",
        )
        historical_runbook = _run_git(
            source,
            "show",
            f"{marker.source_commit}:{PROOF_RUNBOOK}",
        )
    except CandidateBuildError:
        return ClosureEvidenceBinding(False, False, False, False)
    if (
        len(historical_gate) > MAX_EVIDENCE_DOCUMENT_BYTES
        or len(historical_runbook) > MAX_EVIDENCE_DOCUMENT_BYTES
        or b"\x00" in historical_gate
        or b"\x00" in historical_runbook
    ):
        return ClosureEvidenceBinding(False, False, False, False)
    try:
        decoded_gate = historical_gate.decode("utf-8")
        decoded_runbook = historical_runbook.decode("utf-8")
    except UnicodeDecodeError:
        return ClosureEvidenceBinding(False, False, False, False)
    historical_digests = _proof_state_digests(decoded_gate, decoded_runbook)
    if historical_digests is None:
        return ClosureEvidenceBinding(False, False, False, False)
    combined_bound = secrets.compare_digest(current_digests[0], historical_digests[0])
    gate_bound = secrets.compare_digest(current_digests[1], historical_digests[1])
    connector_bound = secrets.compare_digest(current_digests[2], historical_digests[2])
    return ClosureEvidenceBinding(
        bound=combined_bound,
        connector_proof_state_bound=connector_bound,
        recovery_proof_state_bound=connector_bound,
        required_gate_state_bound=gate_bound,
    )


def _proof_policy_violations(dispositions: Mapping[str, str]) -> tuple[str, ...]:
    """Return lanes whose structured disposition cannot close this preview."""

    return tuple(
        sorted(
            lane
            for lane in PROOF_LANES
            if dispositions.get(lane, "unknown")
            not in PROOF_DISPOSITION_POLICY[lane]
            and dispositions.get(lane, "unknown")
            not in {"pending", "failed", "unknown"}
        )
    )


def _blob_identity(blobs: tuple[CandidateBlob, ...]) -> tuple[tuple[str, bool, bytes], ...]:
    return tuple((blob.path, blob.executable, blob.content) for blob in blobs)


def _changed_paths(source: Path, candidate_commit: str, head_commit: str) -> frozenset[str]:
    if candidate_commit == head_commit:
        return frozenset()
    raw = _run_git(
        source,
        "diff",
        "--name-only",
        "-z",
        candidate_commit,
        head_commit,
        "--",
    )
    try:
        return frozenset(item.decode("utf-8") for item in raw.split(b"\0") if item)
    except UnicodeDecodeError as exc:
        raise PreviewClosureError("source change set is invalid") from exc


def _candidate_snapshot(
    candidate_root: str | Path,
    *,
    extra_deny_literals: tuple[bytes, ...],
) -> CandidateSnapshot | None:
    opened = _open_candidate_root(candidate_root)
    if opened is None:
        return None
    root, root_fd, root_stat = opened
    try:
        return _candidate_snapshot_fd(
            root,
            root_fd,
            root_stat,
            extra_deny_literals=extra_deny_literals,
        )
    finally:
        os.close(root_fd)


def _candidate_snapshot_fd(
    root: Path,
    root_fd: int,
    root_stat: os.stat_result,
    *,
    extra_deny_literals: tuple[bytes, ...],
) -> CandidateSnapshot | None:
    try:
        before = _candidate_state_fd(root_fd)
        preflight = _scan_release_tree_fd(
            root_fd,
            extra_deny_literals=extra_deny_literals,
        )
        after = _candidate_state_fd(root_fd)
        named = os.stat(root, follow_symlinks=False)
        anchored = (
            named.st_dev == root_stat.st_dev
            and named.st_ino == root_stat.st_ino
            and os.fstat(root_fd).st_dev == root_stat.st_dev
            and os.fstat(root_fd).st_ino == root_stat.st_ino
        )
        if before is None or before != after or not preflight.ok or not anchored:
            return None
        return CandidateSnapshot(
            marker=before.marker,
            files_scanned=preflight.files_scanned,
            bytes_scanned=preflight.bytes_scanned,
            finalized_immutable=_candidate_tree_has_finalized_modes_fd(root_fd),
        )
    except OSError:
        return None


def _source_binding(
    source: Path,
    marker: PublicCandidateMarker,
) -> tuple[bool, bool, bool, bool]:
    source_commit_exists = False
    source_commit_is_ancestor = False
    selected_content_matches = False
    evidence_only_head_drift = False
    try:
        candidate_blobs = _candidate_blobs(source, marker.source_commit)
        source_commit_exists = (
            _manifest_digest(candidate_blobs, marker.source_commit) == marker.manifest_digest
        )
        if not source_commit_exists:
            return False, False, False, False
        head_commit = _source_commit(source)
        _run_git(source, "merge-base", "--is-ancestor", marker.source_commit, head_commit)
        source_commit_is_ancestor = True
        head_blobs = _candidate_blobs(source, head_commit)
        selected_content_matches = _blob_identity(candidate_blobs) == _blob_identity(head_blobs)
        changed = _changed_paths(source, marker.source_commit, head_commit)
        evidence_only_head_drift = changed.issubset(EVIDENCE_ONLY_PATHS)
    except CandidateBuildError:
        pass
    return (
        source_commit_exists,
        source_commit_is_ancestor,
        selected_content_matches,
        evidence_only_head_drift,
    )


def _aggregate_environment(
    candidate_root: str | Path,
    isolated_root: Path,
    environ: Mapping[str, str],
) -> dict[str, str]:
    home = isolated_root / "home"
    temp = isolated_root / "tmp"
    cache = isolated_root / "cache"
    for directory in (home, temp, cache):
        directory.mkdir(mode=0o700)
    empty_env = isolated_root / "runtime.env"
    empty_env.write_text("", encoding="utf-8")
    empty_env.chmod(0o600)
    child = {
        "HOME": os.fspath(home),
        "TMPDIR": os.fspath(temp),
        "XDG_CACHE_HOME": os.fspath(cache),
        "PATH": environ.get("PATH", os.defpath),
        "PYTHONPATH": os.fspath(candidate_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "JARVIS_V3_ENV": os.fspath(empty_env),
        "JARVIS_V3_ENABLE_DAEMONS": "0",
        "JARVIS_V3_ENABLE_SCHEDULER": "0",
        "JARVIS_SMOKE_SUITE_LOCK_PATH": os.fspath(isolated_root / "aggregate.lock"),
    }
    for key in ("LANG", "LC_ALL"):
        value = environ.get(key)
        if value:
            child[key] = value
    return child


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_exit(process: subprocess.Popen[bytes], deadline: float) -> bool:
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process.pid):
            return True
        time.sleep(0.01)
    process.poll()
    return not _process_group_exists(process.pid)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    for requested_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            break
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
        if _wait_process_group_exit(
            process, time.monotonic() + PROCESS_GROUP_CLEANUP_SECONDS
        ):
            break
    try:
        process.wait(timeout=PROCESS_GROUP_CLEANUP_SECONDS)
    except (OSError, subprocess.SubprocessError):
        pass


class _AggregateOutputLimit(RuntimeError):
    def __init__(self, stdout: bytes, stderr: bytes) -> None:
        super().__init__("aggregate output limit exceeded")
        self.stdout = stdout
        self.stderr = stderr


class _AggregateResidualProcessGroup(RuntimeError):
    def __init__(self, stdout: bytes, stderr: bytes) -> None:
        super().__init__("aggregate left descendant processes after exit")
        self.stdout = stdout
        self.stderr = stderr


def _pipe_fds(process: subprocess.Popen[bytes]) -> tuple[int, int] | None:
    if process.stdout is None or process.stderr is None:
        return None
    try:
        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    if type(stdout_fd) is not int or type(stderr_fd) is not int:
        return None
    return stdout_fd, stderr_fd


def _bounded_communicate(
    process: subprocess.Popen[bytes], *, timeout: float
) -> tuple[bytes, bytes]:
    """Drain a real child with a hard combined cap and bounded tail memory.

    Mocked process objects retain the ordinary communicate seam used by the
    offline contract tests; real pipe descriptors always take the bounded path.
    """

    fds = _pipe_fds(process)
    if fds is None:
        return process.communicate(timeout=timeout)
    selector: selectors.BaseSelector | None = None
    streams = {fds[0]: "stdout", fds[1]: "stderr"}
    tails = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    deadline = time.monotonic() + timeout
    try:
        selector = selectors.DefaultSelector()
        for fd in fds:
            selector.register(fd, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    process.args,
                    timeout,
                    output=bytes(tails["stdout"]),
                    stderr=bytes(tails["stderr"]),
                )
            events = selector.select(min(remaining, 0.25))
            if not events:
                continue
            for key, _mask in events:
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                total += len(chunk)
                tail = tails[streams[key.fd]]
                tail.extend(chunk)
                if len(tail) > MAX_AGGREGATE_DIAGNOSTIC_BYTES:
                    del tail[:-MAX_AGGREGATE_DIAGNOSTIC_BYTES]
                if total > MAX_AGGREGATE_CAPTURE_BYTES:
                    raise _AggregateOutputLimit(
                        bytes(tails["stdout"]), bytes(tails["stderr"])
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(
                process.args,
                timeout,
                output=bytes(tails["stdout"]),
                stderr=bytes(tails["stderr"]),
            )
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(
                process.args,
                timeout,
                output=bytes(tails["stdout"]),
                stderr=bytes(tails["stderr"]),
            ) from None
        # A reaped process can leave its now-empty PGID transiently observable
        # while the kernel finishes natural teardown. Give that state the same
        # bounded polling used by cleanup before classifying it as a live
        # residual group; a group that remains present still fails closed.
        group_exit_deadline = min(
            deadline,
            time.monotonic() + PROCESS_GROUP_CLEANUP_SECONDS,
        )
        if not _wait_process_group_exit(process, group_exit_deadline):
            raise _AggregateResidualProcessGroup(
                bytes(tails["stdout"]), bytes(tails["stderr"])
            )
        return bytes(tails["stdout"]), bytes(tails["stderr"])
    finally:
        if selector is not None:
            selector.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def _bounded_aggregate_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
    else:
        encoded = value
    return encoded[-MAX_AGGREGATE_DIAGNOSTIC_BYTES:].decode(
        "utf-8", errors="replace"
    ).replace("\r\n", "\n")


def _aggregate_failure_modules(
    stdout: bytes | str | None,
    expected_modules: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Return only a strictly formed, allowlisted aggregate failure footer."""

    text = _bounded_aggregate_output(stdout).rstrip()
    expected_count = len(expected_modules)
    match = re.search(
        rf"(?:^|\n)FAILED: (?P<count>[1-9][0-9]*) of {expected_count} "
        rf"smoke modules? failed\. Duration: [0-9]+(?:\.[0-9]+)?s\.\n"
        r"(?P<modules>(?:- [A-Za-z0-9_.]+\n?)+)\Z",
        text,
    )
    if match is None:
        return None
    modules = tuple(
        line[2:]
        for line in match.group("modules").splitlines()
        if line.startswith("- ")
    )
    if len(modules) != int(match.group("count")) or len(set(modules)) != len(modules):
        return None
    allowlisted = tuple(module for module in expected_modules if module in set(modules))
    if modules != allowlisted:
        return None
    return modules


def _aggregate_timeout_modules(
    stdout: bytes | str | None,
    expected_modules: tuple[str, ...],
) -> tuple[str, ...]:
    """Identify at most the last allowlisted module from a strict runner heading."""

    text = _bounded_aggregate_output(stdout)
    expected_count = len(expected_modules)
    current: str | None = None
    for match in re.finditer(
        rf"(?m)^== \[(?P<index>[1-9][0-9]*)/{expected_count}\] "
        r"(?P<module>[A-Za-z0-9_.]+) ==$",
        text,
    ):
        index = int(match.group("index"))
        module = match.group("module")
        if index <= expected_count and module == expected_modules[index - 1]:
            current = module
    return (current,) if current is not None else ()


def _run_aggregate(
    candidate_fd: int,
    *,
    environ: Mapping[str, str],
) -> AggregateResult:
    from jarvis_v2.scripts.smoke_test_all import TEST_MODULES

    expected_modules = tuple(TEST_MODULES)
    expected_count = len(expected_modules)
    started = time.monotonic()
    anchored_root = "."
    with TemporaryDirectory(prefix="jarvis-v3-preview-closure-", dir="/private/tmp") as temp:
        isolated_root = Path(temp)
        child_env = _aggregate_environment(anchored_root, isolated_root, environ)
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "jarvis_v2.scripts.smoke_test_all",
                ],
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(candidate_fd,),
                preexec_fn=lambda: os.fchdir(candidate_fd),
                start_new_session=True,
            )
            try:
                stdout, _stderr = _bounded_communicate(
                    process, timeout=AGGREGATE_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(process)
                return AggregateResult(
                    True,
                    False,
                    expected_count,
                    round(time.monotonic() - started, 1),
                    "timed_out",
                    _aggregate_timeout_modules(exc.output, expected_modules),
                )
            except _AggregateOutputLimit:
                _terminate_process_group(process)
                return AggregateResult(
                    True,
                    False,
                    expected_count,
                    round(time.monotonic() - started, 1),
                    "malformed_summary",
                )
            except _AggregateResidualProcessGroup:
                _terminate_process_group(process)
                return AggregateResult(
                    True,
                    False,
                    expected_count,
                    round(time.monotonic() - started, 1),
                    "malformed_summary",
                )
            except BaseException:
                _terminate_process_group(process)
                raise
        except OSError:
            return AggregateResult(
                True,
                False,
                expected_count,
                round(time.monotonic() - started, 1),
                "malformed_summary",
            )
    tail = _bounded_aggregate_output(stdout)
    exact_summary = re.search(
        rf"(?:^|\n)All {expected_count} smoke modules? passed\. Duration: [0-9]+(?:\.[0-9]+)?s\.\s*\Z",
        tail,
    )
    passed = process.returncode == 0 and exact_summary is not None
    failed_modules = (
        _aggregate_failure_modules(stdout, expected_modules)
        if process.returncode != 0
        else None
    )
    return AggregateResult(
        True,
        passed,
        expected_count,
        round(time.monotonic() - started, 1),
        None if passed else ("failed" if failed_modules is not None else "malformed_summary"),
        failed_modules or (),
    )


def evaluate_preview_closure(
    source_root: str | Path,
    candidate_root: str | Path,
    *,
    run_aggregate: bool = False,
    environ: Mapping[str, str] | None = None,
    active_contract_manifest: str | Path | None = None,
    network_recovery_receipt: str | Path | None = None,
    reboot_recovery_receipt: str | Path | None = None,
) -> PreviewClosureReport:
    environment = dict(os.environ if environ is None else environ)
    receipt_anchor = _receipt_anchor_from_environment(environment, required=False)
    source = _validated_source_root(source_root)
    source_head = _source_commit(source)
    gate = _read_evidence_document(source, GATE_DOCUMENT)
    runbook = _read_evidence_document(source, PROOF_RUNBOOK)
    gate_total, gate_checked, gate_valid = _required_gate_counts(gate)
    dispositions, proof_status_valid = _proof_dispositions(runbook)
    connector_attestations = _connector_attestations(runbook, dispositions)
    passed = sum(value == "passed" for value in dispositions.values())
    deferred = sum(value == "deferred" for value in dispositions.values())
    disabled = sum(value == "disabled" for value in dispositions.values())
    pending_lanes = tuple(sorted(key for key, value in dispositions.items() if value == "pending"))
    failed_lanes = tuple(sorted(key for key, value in dispositions.items() if value == "failed"))
    unknown_lanes = tuple(sorted(key for key, value in dispositions.items() if value == "unknown"))
    unaccepted_lanes = _proof_policy_violations(dispositions)

    extra = extra_deny_literals_from_env(environment)
    opened = _open_candidate_root(candidate_root)
    candidate_fd: int | None = None
    before: CandidateSnapshot | None = None
    after: CandidateSnapshot | None = None
    if opened is not None:
        candidate_named_root, candidate_fd, candidate_stat = opened
        before = _candidate_snapshot_fd(
            candidate_named_root,
            candidate_fd,
            candidate_stat,
            extra_deny_literals=extra,
        )
    source_commit_exists = False
    source_commit_is_ancestor = False
    selected_content_matches = False
    evidence_only_head_drift = False
    closure_evidence_binding = ClosureEvidenceBinding(False, False, False, False)
    if before is not None:
        (
            source_commit_exists,
            source_commit_is_ancestor,
            selected_content_matches,
            evidence_only_head_drift,
        ) = _source_binding(source, before.marker)
        closure_evidence_binding = _proof_state_bound_to_candidate(
            source,
            before.marker,
            gate,
            runbook,
        )
    recovery_attestation = _recovery_attestation(
        runbook,
        dispositions,
        source=source,
        candidate_marker=before.marker if before is not None else None,
        active_contract_manifest=active_contract_manifest,
        network_receipt=network_recovery_receipt,
        reboot_receipt=reboot_recovery_receipt,
    )
    recovery_disposition = dispositions.get(RECOVERY_PROOF_LANE, "unknown")
    recovery_inputs_supplied = any(
        value is not None
        for value in (
            active_contract_manifest,
            network_recovery_receipt,
            reboot_recovery_receipt,
        )
    )

    blockers: set[str] = set()
    if not extra:
        blockers.add("private_deny_review_missing")
    if not gate_valid:
        blockers.add("required_gate_incomplete")
    if not proof_status_valid:
        blockers.add("supervised_proof_status_invalid")
    if pending_lanes:
        blockers.add("supervised_proof_pending")
    if failed_lanes:
        blockers.add("supervised_proof_failed")
    if unknown_lanes:
        blockers.add("supervised_proof_disposition_unknown")
    if unaccepted_lanes:
        blockers.add("supervised_proof_policy_violation")
    if not connector_attestations.structurally_valid:
        blockers.add("connector_proof_attestation_invalid")
    if connector_attestations.missing_lanes:
        blockers.add("connector_proof_attestation_missing")
    if connector_attestations.invalid_lanes:
        blockers.add("connector_proof_attestation_invalid")
    if connector_attestations.mismatched_lanes:
        blockers.add("connector_proof_attestation_mismatch")
    if not recovery_attestation.structurally_valid or recovery_attestation.invalid:
        blockers.add("recovery_proof_attestation_invalid")
    if recovery_attestation.missing:
        blockers.add("recovery_proof_attestation_missing")
    if recovery_attestation.mismatched:
        blockers.add("recovery_proof_attestation_mismatch")
    if recovery_disposition != "passed" and recovery_inputs_supplied:
        blockers.add("recovery_proof_inputs_unexpected")
    if not closure_evidence_binding.connector_proof_state_bound:
        blockers.add("connector_proof_state_not_candidate_bound")
    if not closure_evidence_binding.recovery_proof_state_bound:
        blockers.add("recovery_proof_state_not_candidate_bound")
    if not closure_evidence_binding.required_gate_state_bound:
        blockers.add("required_gate_state_not_candidate_bound")
    if not closure_evidence_binding.bound:
        blockers.add("closure_evidence_state_not_candidate_bound")
    if before is None:
        blockers.add("candidate_preflight_or_marker_invalid")
    elif not before.finalized_immutable:
        blockers.add("candidate_not_immutable")
    if not source_commit_exists:
        blockers.add("candidate_source_unbound")
    if not source_commit_is_ancestor:
        blockers.add("candidate_source_not_ancestor")
    if not selected_content_matches:
        blockers.add("selected_content_drift")
    if not evidence_only_head_drift:
        blockers.add("non_evidence_head_drift")

    preconditions_ready_before_aggregate = not blockers
    aggregate = AggregateResult(False, False, 0, None)
    if run_aggregate and preconditions_ready_before_aggregate and candidate_fd is not None:
        try:
            aggregate = _run_aggregate(candidate_fd, environ=environment)
        except BaseException:
            os.close(candidate_fd)
            candidate_fd = None
            raise
        if not aggregate.passed:
            blockers.add("aggregate_failed")
    elif not run_aggregate:
        blockers.add("aggregate_not_run")
    else:
        blockers.add("aggregate_blocked_by_preconditions")

    if opened is not None and candidate_fd is not None:
        after = _candidate_snapshot_fd(
            candidate_named_root,
            candidate_fd,
            candidate_stat,
            extra_deny_literals=extra,
        )
        os.close(candidate_fd)
        candidate_fd = None
    stable = before is not None and before == after
    if after is None:
        blockers.add("candidate_postflight_invalid")
    if not stable:
        blockers.add("candidate_changed_during_closure")

    # Revalidate the clean source after any long aggregate execution. Only the
    # two evidence documents may differ from the candidate commit, and neither
    # may change in the worktree during this command.
    try:
        source_after = _validated_source_root(source)
        if _source_commit(source_after) != source_head:
            blockers.add("source_head_changed_during_closure")
        if _read_evidence_document(source_after, GATE_DOCUMENT) != gate:
            blockers.add("closure_evidence_changed")
        if _read_evidence_document(source_after, PROOF_RUNBOOK) != runbook:
            blockers.add("closure_evidence_changed")
    except (CandidateBuildError, PreviewClosureError):
        blockers.add("source_changed_during_closure")
    finally:
        if candidate_fd is not None:
            os.close(candidate_fd)

    if before is not None:
        binding_after = _source_binding(source, before.marker)
        if binding_after != (
            source_commit_exists,
            source_commit_is_ancestor,
            selected_content_matches,
            evidence_only_head_drift,
        ):
            blockers.add("source_binding_changed_during_closure")
        if (
            _proof_state_bound_to_candidate(source, before.marker, gate, runbook)
            != closure_evidence_binding
        ):
            blockers.add("source_binding_changed_during_closure")

    aggregate_only_blockers = frozenset(
        {"aggregate_not_run", "aggregate_failed", "aggregate_blocked_by_preconditions"}
    )
    preconditions_ready = not (blockers - aggregate_only_blockers)
    ready = not blockers and aggregate.ran and aggregate.passed and stable
    marker = before.marker if before is not None else None
    private_deny_review_commitment = None
    if aggregate.ran and aggregate.passed and receipt_anchor is not None and extra:
        private_deny_review_commitment = _private_deny_review_commitment(
            receipt_anchor,
            extra,
        )
    return PreviewClosureReport(
        ready=ready,
        preconditions_ready=preconditions_ready,
        required_gate_total=gate_total,
        required_gate_checked=gate_checked,
        supervised_proof_total=len(dispositions),
        supervised_proof_passed=passed,
        supervised_proof_deferred=deferred,
        supervised_proof_disabled=disabled,
        supervised_proof_pending=len(pending_lanes),
        supervised_proof_failed=len(failed_lanes),
        supervised_proof_unknown=len(unknown_lanes),
        pending_supervised_proofs=pending_lanes,
        failed_supervised_proofs=failed_lanes,
        unknown_supervised_proofs=unknown_lanes,
        unaccepted_supervised_proofs=unaccepted_lanes,
        connector_proof_attested=len(connector_attestations.attested_lanes),
        unattested_connector_proofs=connector_attestations.missing_lanes,
        invalid_connector_proof_attestations=connector_attestations.invalid_lanes,
        mismatched_connector_proof_attestations=connector_attestations.mismatched_lanes,
        recovery_proof_attested=recovery_attestation.attested,
        recovery_proof_attestation_missing=recovery_attestation.missing,
        recovery_proof_attestation_invalid=(
            not recovery_attestation.structurally_valid or recovery_attestation.invalid
        ),
        recovery_proof_attestation_mismatched=recovery_attestation.mismatched,
        connector_proof_state_candidate_bound=(
            closure_evidence_binding.connector_proof_state_bound
        ),
        recovery_proof_state_candidate_bound=(
            closure_evidence_binding.recovery_proof_state_bound
        ),
        required_gate_state_candidate_bound=(
            closure_evidence_binding.required_gate_state_bound
        ),
        closure_evidence_state_candidate_bound=closure_evidence_binding.bound,
        candidate_source_commit=marker.source_commit if marker is not None else None,
        candidate_manifest_digest=marker.manifest_digest if marker is not None else None,
        candidate_files_scanned=before.files_scanned if before is not None else None,
        candidate_bytes_scanned=before.bytes_scanned if before is not None else None,
        candidate_finalized_immutable=(
            before.finalized_immutable if before is not None else False
        ),
        candidate_preflight_before=before is not None,
        candidate_preflight_after=after is not None,
        candidate_stable=stable,
        source_commit_exists=source_commit_exists,
        source_commit_is_ancestor=source_commit_is_ancestor,
        selected_content_matches=selected_content_matches,
        evidence_only_head_drift=evidence_only_head_drift,
        private_deny_review_supplied=bool(extra),
        private_deny_literal_count=len(extra),
        aggregate_requested=run_aggregate,
        aggregate_ran=aggregate.ran,
        aggregate_passed=aggregate.passed,
        aggregate_module_count=aggregate.module_count,
        aggregate_duration_seconds=aggregate.duration_seconds,
        aggregate_failure_reason=aggregate.failure_reason,
        aggregate_failed_modules=aggregate.failed_modules,
        blocking_codes=tuple(sorted(blockers)),
        private_deny_review_commitment_sha256=private_deny_review_commitment,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the offline, content-free Jarvis V3 preview closure gate."
    )
    parser.add_argument("--source", required=True, help="clean private V3 worktree")
    parser.add_argument("--candidate", required=True, help="immutable public candidate tree")
    parser.add_argument(
        "--run-aggregate",
        action="store_true",
        help="run the isolated aggregate suite inside the candidate before closure",
    )
    parser.add_argument(
        "--receipt-output",
        help=(
            "atomically create an external owner-only receipt after a ready "
            "aggregate run; never writes inside source or candidate"
        ),
    )
    parser.add_argument(
        "--verify-receipt",
        help="read-only verification of an external closure receipt; never reruns aggregate",
    )
    parser.add_argument(
        "--active-contract-manifest",
        help="owner-only active contract manifest; required only when recovery_services passed",
    )
    parser.add_argument(
        "--network-recovery-receipt",
        help="finalized network recovery receipt; required only when recovery_services passed",
    )
    parser.add_argument(
        "--reboot-recovery-receipt",
        help="finalized reboot recovery receipt; required only when recovery_services passed",
    )
    return parser


def _configuration_error_payload() -> dict[str, object]:
    return {
        "schema": "jarvis-v3-preview-closure",
        "version": 1,
        "ready": False,
        "configuration_valid": False,
        "receipt_outcome_unknown": False,
        "blocking_codes": ["configuration_unsafe"],
        "paths_included": False,
        "private_content_included": False,
        "publishes": False,
        "publication_authorized": False,
        "cutover_authorized": False,
        "starts_services": False,
        "changes_schedules": False,
        "runs_live_actions": False,
    }


def _receipt_outcome_unknown_payload() -> dict[str, object]:
    payload = _configuration_error_payload()
    payload["blocking_codes"] = ["closure_receipt_outcome_unknown"]
    payload["receipt_outcome_unknown"] = True
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    environment = dict(os.environ)
    try:
        if args.verify_receipt is not None:
            if (
                args.run_aggregate
                or args.receipt_output is not None
            ):
                raise PreviewClosureError("receipt verification mode is exclusive")
            verification = verify_closure_receipt(
                args.verify_receipt,
                source_root=args.source,
                candidate_root=args.candidate,
                environ=environment,
                active_contract_manifest=args.active_contract_manifest,
                network_recovery_receipt=args.network_recovery_receipt,
                reboot_recovery_receipt=args.reboot_recovery_receipt,
            )
            print(
                json.dumps(
                    {
                        "schema": "jarvis-v3-preview-closure-receipt-verification",
                        "version": 2,
                        "valid": verification.aggregate_execution_authenticated,
                        "self_consistent": verification.self_consistent,
                        "aggregate_execution_authenticated": (
                            verification.aggregate_execution_authenticated
                        ),
                        "receipt_sha256": verification.receipt_sha256,
                        "paths_included": False,
                        "private_content_included": False,
                        "runs_aggregate": False,
                        "publishes": False,
                        "publication_authorized": False,
                        "cutover_authorized": False,
                        "starts_services": False,
                        "changes_schedules": False,
                        "runs_live_actions": False,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0 if verification.aggregate_execution_authenticated else 1
        if args.receipt_output is not None:
            if not args.run_aggregate:
                raise PreviewClosureError("receipt output requires an aggregate run")
            _receipt_anchor_from_environment(environment, required=True)
            _preflight_receipt_output(
                args.receipt_output,
                source_root=args.source,
                candidate_root=args.candidate,
            )
        report = evaluate_preview_closure(
            args.source,
            args.candidate,
            run_aggregate=args.run_aggregate,
            environ=environment,
            active_contract_manifest=args.active_contract_manifest,
            network_recovery_receipt=args.network_recovery_receipt,
            reboot_recovery_receipt=args.reboot_recovery_receipt,
        )
        receipt_creation: ClosureReceiptCreation | None = None
        if args.receipt_output is not None and report.ready:
            receipt_creation = write_closure_receipt(
                args.receipt_output,
                source_root=args.source,
                candidate_root=args.candidate,
                report=report,
                environ=environment,
                active_contract_manifest=args.active_contract_manifest,
                network_recovery_receipt=args.network_recovery_receipt,
                reboot_recovery_receipt=args.reboot_recovery_receipt,
            )
    except ClosureReceiptOutcomeUnknown:
        print(
            json.dumps(
                _receipt_outcome_unknown_payload(),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 3
    except (CandidateBuildError, PreflightConfigError, PreviewClosureError):
        print(json.dumps(_configuration_error_payload(), sort_keys=True, separators=(",", ":")))
        return 2
    payload = report.summary_json_payload()
    if args.receipt_output is not None:
        payload["closure_receipt_written"] = receipt_creation is not None
        payload["closure_receipt_sha256"] = (
            receipt_creation.receipt_sha256 if receipt_creation is not None else None
        )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
