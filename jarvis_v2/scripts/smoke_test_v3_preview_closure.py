"""Focused offline tests for the Jarvis V3 preview closure verifier."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import plistlib
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest import mock

from jarvis_v2.scripts import public_release_candidate as candidate_module
from jarvis_v2.scripts import v3_active_launchagent_contracts as active_contracts
from jarvis_v2.scripts import v3_preview_closure as closure
from jarvis_v2.scripts import v3_recovery_receipt as recovery_receipt
from jarvis_v2.scripts.public_release_candidate import build_public_candidate


PROCESS_CLEANUP_TEST_DEADLINE_SECONDS = 10.0
PROCESS_PROGRESS_OBSERVATION_SECONDS = 0.5


def _closure_environment(*, anchor: str | None = "d" * 64) -> dict[str, str]:
    environment = {
        "PATH": os.defpath,
        closure.EXTRA_DENY_ENV: json.dumps(["synthetic-private-marker-never-present"]),
    }
    if anchor is not None:
        environment[closure.CLOSURE_RECEIPT_ANCHOR_ENV] = anchor
    return environment


def test_runbook_anchor_creation_is_fresh_exclusive_and_no_follow() -> None:
    runbook = (
        Path(__file__).resolve().parents[2] / closure.PROOF_RUNBOOK
    ).read_text(encoding="utf-8")
    required = (
        "mktemp -d /private/tmp/jarvis-v3-closure-evidence.XXXXXXXX",
        "os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC",
        'os.open(os.path.join(root, "anchor.txt"), flags, 0o600)',
        '"$JARVIS_V3_CLOSURE_EVIDENCE_DIR/closure.json"',
    )
    if any(item not in runbook for item in required):
        raise SystemExit("closure runbook lost exclusive no-follow anchor creation")
    forbidden = (
        "install -d -m 700 /private/tmp/jarvis-v3-closure-evidence-REVIEWED-NONCE",
        "openssl rand -hex 32 >",
    )
    if any(item in runbook for item in forbidden):
        raise SystemExit("closure runbook restored clobber-prone anchor creation")


def _progress_snapshot(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii")
    except FileNotFoundError:
        return None


def _assert_process_progress_stopped(
    progress_path: Path,
    *,
    process_group_id: int,
    label: str,
) -> None:
    before = _progress_snapshot(progress_path)
    time.sleep(PROCESS_PROGRESS_OBSERVATION_SECONDS)
    after = _progress_snapshot(progress_path)
    if before == after:
        return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    raise SystemExit(f"{label} continued executing after cleanup")


def _git(root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", os.fspath(root), *args],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode:
        raise SystemExit(f"synthetic closure Git fixture failed: {args!r}")


def _write(root: Path, relative: str, content: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _commit(root: Path, message: str) -> None:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Synthetic Maintainer",
        "-c",
        "user.email=maintainer@example.com",
        "commit",
        "--quiet",
        "-m",
        message,
    )


def _gate(*, checked: bool = True) -> str:
    statuses = {
        identifier: "passed" for identifier in closure.REQUIRED_GATE_IDS
    }
    if not checked:
        statuses[closure.REQUIRED_GATE_IDS[0]] = "pending"
    machine = "\n".join(f"{key}={value}" for key, value in statuses.items())
    return (
        "# Synthetic functional preview gate\n\n"
        "## Required before the preview can be called functional\n\n"
        f"{closure.GATE_STATUS_BLOCK_START}\n{machine}\n"
        f"{closure.GATE_STATUS_BLOCK_END}\n\n"
        "- [x] Human prose is NOT actually satisfied and is ignored by closure.\n\n"
        "## Supervised proofs that do not run unattended\n\n"
        "Proof dispositions are recorded in the private runbook.\n"
    )


def _runbook(*, pending: bool) -> str:
    dispositions = {
        "voice": "passed",
        "ocr": "passed",
        "google_calendar": "passed",
        "apple_reminders": "pending" if pending else "disabled",
        "telegram": "passed",
        "imessage": "passed",
        "instagram": "passed",
        "kakaotalk": "pending" if pending else "disabled",
        "calls": "disabled",
        "recovery_services": "deferred",
    }
    machine = "\n".join(f"{key}={value}" for key, value in dispositions.items())
    evidence_classes = {
        "apple_reminders": "bounded_read_observed",
        "telegram": "approval_bound_sender_success_recipient_confirmed",
        "imessage": "approval_bound_sender_success_recipient_confirmed",
        "instagram": "approval_bound_sender_success_recipient_confirmed",
        "kakaotalk": "approval_bound_target_verified_recipient_confirmed",
    }
    attestation_lines = []
    for lane in closure.CONNECTOR_PROOF_LANES:
        if dispositions[lane] != "passed":
            attestation_lines.append(f"{lane}=none")
            continue
        evidence_class = evidence_classes[lane]
        observed_on = "2026-08-05"
        digest = closure._connector_attestation_digest(
            lane,
            evidence_class,
            observed_on,
        )
        attestation_lines.append(
            f"{lane}=passed|{evidence_class}|{observed_on}|{digest}"
        )
    attestation_lines.extend(("recovery_network=none", "recovery_reboot=none"))
    attestations = "\n".join(attestation_lines)
    return f"""# Synthetic supervised proof runbook

{closure.PROOF_STATUS_BLOCK_START}
{machine}
{closure.PROOF_STATUS_BLOCK_END}

{closure.PROOF_ATTESTATION_BLOCK_START}
{attestations}
{closure.PROOF_ATTESTATION_BLOCK_END}

## Proof 1 — Local push-to-talk and spoken reply

**Status: push-to-talk has not been proved.**

## Proof 2 — On-device OCR

**Status: OCR was bypassed verification.**

## Proof 3 — Read-only personal connectors

**Status: prose is deliberately not authoritative.**

## Proof 4 — Messaging channels

**Status: prose is deliberately not authoritative.**

## Proof 5 — Recovery and optional services

**Status: deferred for the preview.**
"""


def _fixture(root: Path, *, pending: bool = False, checked: bool = True) -> tuple[Path, Path]:
    source = root / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _write(source, "README.md", "Safe synthetic preview source.\n")
    _write(source, "requirements.txt", "\n")
    _write(source, "jarvis_v2/__init__.py", "\n")
    _write(source, "jarvis_v2/module.py", "VALUE = 'offline'\n")
    _write(
        source,
        closure.CLOSURE_RUNNER_PATH,
        "\"\"\"Synthetic closure runner fixture.\"\"\"\nVALUE = 'offline'\n",
    )
    _write(
        source,
        closure.CLOSURE_SMOKE_INVENTORY_PATH,
        "TEST_MODULES = ['synthetic.smoke_one', 'synthetic.smoke_two']\n",
    )
    _write(source, closure.GATE_DOCUMENT, _gate(checked=checked))
    _write(source, closure.PROOF_RUNBOOK, _runbook(pending=pending))
    _commit(source, "synthetic preview source")
    candidate = root / "candidate"
    with mock.patch.object(candidate_module, "PRIVATE_TMP_ROOT", root):
        report = build_public_candidate(source, candidate)
    if not report.ok or not report.candidate_finalized_immutable:
        raise SystemExit(f"synthetic closure candidate failed: {report}")
    return source, candidate


def test_private_deny_review_is_required_before_aggregate() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-closure-deny-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))
        for environment in (
            {"PATH": os.defpath},
            {"PATH": os.defpath, closure.EXTRA_DENY_ENV: "[]"},
        ):
            with mock.patch.object(closure, "_run_aggregate") as aggregate:
                report = closure.evaluate_preview_closure(
                    source,
                    candidate,
                    run_aggregate=True,
                    environ=environment,
                )
            if (
                report.ready
                or report.private_deny_review_supplied
                or report.private_deny_literal_count != 0
                or "private_deny_review_missing" not in report.blocking_codes
                or aggregate.called
            ):
                raise SystemExit(f"empty private deny review did not fail closed: {report}")
        malformed = "not-json-private-value"
        output = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {"PATH": os.defpath, closure.EXTRA_DENY_ENV: malformed},
                clear=True,
            ),
            mock.patch.object(closure, "_run_aggregate") as aggregate,
            contextlib.redirect_stdout(output),
        ):
            code = closure.main(
                [
                    "--source",
                    os.fspath(source),
                    "--candidate",
                    os.fspath(candidate),
                    "--run-aggregate",
                ]
            )
        payload = json.loads(output.getvalue())
        if (
            code != 2
            or payload.get("blocking_codes") != ["configuration_unsafe"]
            or malformed in output.getvalue()
            or aggregate.called
        ):
            raise SystemExit("malformed private deny review did not fail closed")


def test_closure_git_children_do_not_inherit_private_runtime_values() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-closure-git-env-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))
        real_run = subprocess.run
        git_environments: list[dict[str, str] | None] = []

        def capture_run(*args: object, **kwargs: object):
            command = args[0] if args else kwargs.get("args")
            if isinstance(command, list) and command and command[0] == "git":
                git_environments.append(kwargs.get("env"))
            return real_run(*args, **kwargs)

        with mock.patch.object(candidate_module.subprocess, "run", side_effect=capture_run):
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=False,
                environ=_closure_environment(),
            )
        if not report.preconditions_ready or not git_environments:
            raise SystemExit("anchored closure Git environment fixture did not run")
        for child_environment in git_environments:
            if child_environment is None or any(
                key in child_environment
                for key in (
                    closure.CLOSURE_RECEIPT_ANCHOR_ENV,
                    closure.EXTRA_DENY_ENV,
                )
            ):
                raise SystemExit("closure Git child inherited private runtime values")


def test_deferred_or_disabled_recovery_rejects_supplied_inputs() -> None:
    for disposition in ("deferred", "disabled"):
        with TemporaryDirectory(
            prefix=f"jarvis-v3-closure-recovery-{disposition}-",
            dir="/private/tmp",
        ) as temp:
            root = Path(temp)
            source, candidate = _fixture(root)
            if disposition == "disabled":
                runbook = (source / closure.PROOF_RUNBOOK).read_text(encoding="utf-8")
                _write(
                    source,
                    closure.PROOF_RUNBOOK,
                    runbook.replace(
                        "recovery_services=deferred",
                        "recovery_services=disabled",
                        1,
                    ),
                )
                _commit(source, "disable synthetic recovery lane")
                candidate = root / "disabled-candidate"
                with mock.patch.object(candidate_module, "PRIVATE_TMP_ROOT", root):
                    built = build_public_candidate(source, candidate)
                if not built.ok:
                    raise SystemExit("disabled recovery candidate did not build")
            supplied = root / "must-not-be-read"
            for kwargs in (
                {"active_contract_manifest": supplied},
                {"network_recovery_receipt": supplied},
                {"reboot_recovery_receipt": supplied},
                {
                    "active_contract_manifest": supplied,
                    "network_recovery_receipt": supplied,
                    "reboot_recovery_receipt": supplied,
                },
            ):
                with mock.patch.object(closure, "_run_aggregate") as aggregate:
                    report = closure.evaluate_preview_closure(
                        source,
                        candidate,
                        run_aggregate=True,
                        environ=_closure_environment(),
                        **kwargs,
                    )
                if (
                    report.ready
                    or "recovery_proof_inputs_unexpected" not in report.blocking_codes
                    or "recovery_proof_attestation_missing" in report.blocking_codes
                    or aggregate.called
                ):
                    raise SystemExit(
                        f"{disposition} recovery inputs did not fail closed: {report}"
                    )


def _passing_aggregate() -> closure.AggregateResult:
    from jarvis_v2.scripts.smoke_test_all import TEST_MODULES

    return closure.AggregateResult(
        True,
        True,
        len(TEST_MODULES),
        1.2,
    )


def _active_contract_fixture(
    root: Path,
) -> tuple[Path, recovery_receipt.ContractEvidence, mock._patch]:
    env_file = root / "runtime.env"
    env_file.write_text("", encoding="utf-8")
    env_file.chmod(0o600)
    templates = root / "templates"
    templates.mkdir()
    template_root_patch = mock.patch.object(active_contracts, "PROJECT_ROOT", templates)
    template_root_patch.start()
    for contract in active_contracts.SERVICE_CONTRACTS:
        (templates / contract.filename).write_bytes(
            plistlib.dumps(active_contracts._expected_template(contract))
        )
    output = root / "active-contracts"
    active_contracts.build_active_launchagent_contracts(
        env_file=env_file,
        output_dir=output,
        template_root=templates,
    )
    manifest = output / active_contracts.MANIFEST_NAME
    return (
        manifest,
        recovery_receipt.validate_active_contract_manifest(manifest),
        template_root_patch,
    )


def _system_observation(
    evidence: recovery_receipt.ContractEvidence,
    source_commit: str,
    *,
    boot: str,
    network: bool,
) -> recovery_receipt.SystemObservation:
    return recovery_receipt.SystemObservation(
        source_commit=source_commit,
        manifest_digest=evidence.manifest_digest,
        contract_set_digest=evidence.contract_set_digest,
        env_content_digest=evidence.env_content_digest,
        expected_services=evidence.expected_services,
        boot_identity=boot,
        network_reachable=network,
        scheduler_disabled=True,
        service_states={name: True for name in evidence.expected_services},
    )


def test_pending_proofs_block_without_running_aggregate_or_leaking_paths() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-pending-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp), pending=True)
        with mock.patch.object(closure, "_run_aggregate") as aggregate:
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if aggregate.called or report.ready or report.preconditions_ready:
            raise SystemExit(f"pending supervised proofs did not block closure: {report}")
        if report.pending_supervised_proofs != ("apple_reminders", "kakaotalk"):
            raise SystemExit(f"pending proof accounting drifted: {report}")
        if "supervised_proof_pending" not in report.blocking_codes:
            raise SystemExit(f"pending proof blocker was missing: {report}")
        rendered = json.dumps(report.summary_json_payload(), sort_keys=True)
        if os.fspath(source) in rendered or os.fspath(candidate) in rendered:
            raise SystemExit("closure summary leaked a source or candidate path")


def test_passed_or_policy_allowed_proofs_can_close_with_bound_aggregate() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-ready-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))
        with mock.patch.object(closure, "_run_aggregate", return_value=_passing_aggregate()):
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        payload = report.summary_json_payload()
        if not report.ready or not report.preconditions_ready or report.blocking_codes:
            raise SystemExit(f"valid closure fixture was not ready: {report}")
        if payload.get("aggregate_failure_reason") is not None or payload.get(
            "aggregate_failed_modules"
        ) != []:
            raise SystemExit(f"aggregate success diagnostics drifted: {payload}")
        if (
            report.supervised_proof_passed != 6
            or report.supervised_proof_disabled != 3
            or report.supervised_proof_deferred != 1
            or report.connector_proof_attested != 3
            or report.recovery_proof_attested
            or report.recovery_proof_attestation_missing
            or report.recovery_proof_attestation_invalid
            or report.recovery_proof_attestation_mismatched
            or not report.connector_proof_state_candidate_bound
            or not report.recovery_proof_state_candidate_bound
            or not report.required_gate_state_candidate_bound
            or not report.closure_evidence_state_candidate_bound
        ):
            raise SystemExit(f"proof dispositions drifted: {report}")
        for key in ("publication_authorized", "cutover_authorized"):
            if payload.get(key) is not False:
                raise SystemExit(f"closure summary authorized a forbidden boundary: {payload}")
        if payload.get("paths_included") is not False or payload.get("private_content_included") is not False:
            raise SystemExit(f"closure summary lost content-free declaration: {payload}")

        failed_module = "jarvis_v2.scripts.smoke_test_v3_preview_closure"
        aggregate_failure = closure.AggregateResult(
            True,
            False,
            report.aggregate_module_count,
            1.3,
            "failed",
            (failed_module,),
        )
        with mock.patch.object(
            closure, "_run_aggregate", return_value=aggregate_failure
        ):
            failed_report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        failed_payload = failed_report.summary_json_payload()
        if failed_payload.get("aggregate_failure_reason") != "failed" or failed_payload.get(
            "aggregate_failed_modules"
        ) != [failed_module]:
            raise SystemExit(f"aggregate failure diagnostics were not reported: {failed_payload}")


def _recovery_line(
    lane: str,
    *,
    observed_on: str = "2026-08-08",
    receipt_sha256: str = "a" * 64,
    source_commit: str = "b" * 40,
    contract_set_sha256: str = "c" * 64,
) -> str:
    evidence_class = next(iter(closure.RECOVERY_EVIDENCE_CLASSES[lane]))
    digest = closure._recovery_attestation_digest(
        lane,
        evidence_class,
        observed_on,
        receipt_sha256,
        source_commit,
        contract_set_sha256,
    )
    return (
        f"{lane}=passed|{evidence_class}|{observed_on}|{receipt_sha256}|"
        f"{source_commit}|{contract_set_sha256}|{digest}"
    )


def test_recovery_attestation_required_only_for_a_passed_lane() -> None:
    base = _runbook(pending=False)
    dispositions, valid = closure._proof_dispositions(base)
    deferred = closure._recovery_attestation(base, dispositions)
    if not valid or not deferred.structurally_valid or deferred.attested or deferred.missing:
        raise SystemExit(f"deferred recovery unexpectedly required receipts: {deferred}")

    passed = base.replace("recovery_services=deferred", "recovery_services=passed", 1)
    network_line = _recovery_line("recovery_network")
    reboot_line = _recovery_line("recovery_reboot")
    populated = passed.replace("recovery_network=none", network_line, 1).replace(
        "recovery_reboot=none", reboot_line, 1
    )
    dispositions, valid = closure._proof_dispositions(populated)
    missing_paths = closure._recovery_attestation(populated, dispositions)
    if not valid or not missing_paths.missing or missing_paths.attested:
        raise SystemExit(f"passed recovery did not require both receipt paths: {missing_paths}")

    malformed = populated.replace(network_line, "recovery_network=passed|malformed", 1)
    malformed_audit = closure._recovery_attestation(
        malformed,
        dispositions,
        active_contract_manifest="/private/tmp/manifest.json",
        network_receipt="/private/tmp/network.json",
        reboot_receipt="/private/tmp/reboot.json",
    )
    if not malformed_audit.invalid or malformed_audit.attested:
        raise SystemExit(f"malformed recovery attestation was accepted: {malformed_audit}")

    legacy_class = "supervised_network_recovery_observed"
    legacy_digest = closure._recovery_attestation_digest(
        "recovery_network",
        legacy_class,
        "2026-08-08",
        "a" * 64,
        "b" * 40,
        "c" * 64,
    )
    legacy_line = (
        "recovery_network=passed|"
        f"{legacy_class}|2026-08-08|{'a' * 64}|{'b' * 40}|{'c' * 64}|"
        f"{legacy_digest}"
    )
    legacy = populated.replace(network_line, legacy_line, 1)
    legacy_audit = closure._recovery_attestation(
        legacy,
        dispositions,
        active_contract_manifest="/private/tmp/manifest.json",
        network_receipt="/private/tmp/network.json",
        reboot_receipt="/private/tmp/reboot.json",
    )
    if not legacy_audit.invalid or legacy_audit.attested:
        raise SystemExit(f"overbroad legacy recovery evidence was accepted: {legacy_audit}")

    future_line = _recovery_line("recovery_network", observed_on="9999-12-31")
    future = populated.replace(network_line, future_line, 1)
    future_audit = closure._recovery_attestation(
        future,
        dispositions,
        active_contract_manifest="/private/tmp/manifest.json",
        network_receipt="/private/tmp/network.json",
        reboot_receipt="/private/tmp/reboot.json",
        today=closure.date(2026, 8, 8),
    )
    if not future_audit.invalid or future_audit.attested:
        raise SystemExit(f"future recovery attestation was accepted: {future_audit}")

    stale = base.replace("recovery_network=none", network_line, 1)
    stale_dispositions, _ = closure._proof_dispositions(stale)
    stale_audit = closure._recovery_attestation(stale, stale_dispositions)
    if not stale_audit.mismatched or stale_audit.attested:
        raise SystemExit(f"stale deferred recovery attestation was accepted: {stale_audit}")


def test_valid_recovery_pass_requires_candidate_binding() -> None:
    with TemporaryDirectory(
        prefix="jarvis-v3-preview-recovery-binding-",
        dir="/private/tmp",
    ) as temp:
        source, candidate = _fixture(Path(temp))
        base_commit = subprocess.run(
            ["git", "-C", os.fspath(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        network_line = _recovery_line("recovery_network", source_commit=base_commit)
        reboot_line = _recovery_line("recovery_reboot", source_commit=base_commit)
        runbook = _runbook(pending=False).replace(
            "recovery_services=deferred", "recovery_services=passed", 1
        ).replace("recovery_network=none", network_line, 1).replace(
            "recovery_reboot=none", reboot_line, 1
        )
        _write(source, closure.PROOF_RUNBOOK, runbook)
        _commit(source, "record recovery proof after candidate")
        with mock.patch.object(closure, "_run_aggregate") as aggregate:
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if aggregate.called or report.ready or report.preconditions_ready:
            raise SystemExit(f"unbound recovery proof reached closure: {report}")
        if (
            not report.recovery_proof_attestation_missing
            or report.recovery_proof_state_candidate_bound
            or "recovery_proof_state_not_candidate_bound" not in report.blocking_codes
        ):
            raise SystemExit(f"recovery proof missed receipt/candidate binding: {report}")
        if report.connector_proof_attested != 3:
            raise SystemExit(f"recovery proof changed connector count: {report}")


def test_finalized_network_and_reboot_receipts_close_only_when_fully_bound() -> None:
    with TemporaryDirectory(
        prefix="jarvis-v3-preview-recovery-finalized-",
        dir="/private/tmp",
    ) as temp:
        root = Path(temp)
        source, _old_candidate = _fixture(root)
        source_commit = subprocess.run(
            ["git", "-C", os.fspath(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest, evidence, template_root_patch = _active_contract_fixture(root)
        network_receipt = root / "network-receipt.json"
        reboot_receipt = root / "reboot-receipt.json"

        network_observations = iter(
            (
                _system_observation(evidence, source_commit, boot="boot-a", network=True),
                _system_observation(evidence, source_commit, boot="boot-a", network=False),
                _system_observation(evidence, source_commit, boot="boot-a", network=True),
            )
        )
        with mock.patch.object(
            recovery_receipt,
            "observe_system",
            side_effect=lambda _manifest: next(network_observations),
        ):
            recovery_receipt.prepare_receipt(
                network_receipt,
                kind="network",
                active_contract_manifest=manifest,
            )
            recovery_receipt.record_observation(
                network_receipt,
                phase="network-down",
                active_contract_manifest=manifest,
            )
            recovery_receipt.record_observation(
                network_receipt,
                phase="network-up",
                active_contract_manifest=manifest,
            )

        with mock.patch.object(
            recovery_receipt,
            "observe_system",
            return_value=_system_observation(
                evidence, source_commit, boot="boot-before", network=True
            ),
        ):
            recovery_receipt.prepare_receipt(
                reboot_receipt,
                kind="reboot",
                active_contract_manifest=manifest,
            )
        with mock.patch.object(
            recovery_receipt,
            "observe_system",
            return_value=_system_observation(
                evidence, source_commit, boot="boot-after", network=True
            ),
        ):
            recovery_receipt.record_observation(
                reboot_receipt,
                phase="postboot",
                active_contract_manifest=manifest,
            )

        network_snapshot = closure._stable_recovery_receipt_snapshot(network_receipt)
        reboot_snapshot = closure._stable_recovery_receipt_snapshot(reboot_receipt)
        if (
            network_snapshot is None
            or network_snapshot.finalized_on is None
            or reboot_snapshot is None
            or reboot_snapshot.finalized_on is None
        ):
            raise SystemExit("finalized recovery receipts were not validated")
        network_hash = network_snapshot.sha256
        reboot_hash = reboot_snapshot.sha256
        network_observed_on = network_snapshot.finalized_on.isoformat()
        reboot_observed_on = reboot_snapshot.finalized_on.isoformat()
        runbook = _runbook(pending=False).replace(
            "recovery_services=deferred", "recovery_services=passed", 1
        ).replace(
            "recovery_network=none",
            _recovery_line(
                "recovery_network",
                receipt_sha256=network_hash,
                source_commit=source_commit,
                contract_set_sha256=evidence.contract_set_digest,
                observed_on=network_observed_on,
            ),
            1,
        ).replace(
            "recovery_reboot=none",
            _recovery_line(
                "recovery_reboot",
                receipt_sha256=reboot_hash,
                source_commit=source_commit,
                contract_set_sha256=evidence.contract_set_digest,
                observed_on=reboot_observed_on,
            ),
            1,
        )
        _write(source, closure.PROOF_RUNBOOK, runbook)
        _commit(source, "bind finalized recovery receipts")
        candidate = root / "candidate-recovery"
        with mock.patch.object(candidate_module, "PRIVATE_TMP_ROOT", root):
            built = build_public_candidate(source, candidate)
        if not built.ok:
            raise SystemExit("recovery-bound candidate fixture did not build")

        current = _system_observation(
            evidence,
            source_commit,
            boot="boot-current",
            network=True,
        )
        with (
            mock.patch.object(recovery_receipt, "observe_system", return_value=current),
            mock.patch.object(
                closure,
                "_run_aggregate",
                return_value=closure.AggregateResult(
                    True,
                    True,
                    2,
                    1.2,
                ),
            ),
        ):
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
                active_contract_manifest=manifest,
                network_recovery_receipt=network_receipt,
                reboot_recovery_receipt=reboot_receipt,
            )
        if not report.ready or not report.recovery_proof_attested or report.blocking_codes:
            raise SystemExit(f"fully bound finalized recovery receipts did not close: {report}")

        closure_receipt_directory = root / "closure-receipts"
        closure_receipt_directory.mkdir(mode=0o700)
        closure_receipt = closure_receipt_directory / "passed-recovery.json"
        with mock.patch.object(
            recovery_receipt,
            "observe_system",
            return_value=current,
        ):
            closure_receipt_creation = closure.write_closure_receipt(
                closure_receipt,
                source_root=source,
                candidate_root=candidate,
                report=report,
                active_contract_manifest=manifest,
                network_recovery_receipt=network_receipt,
                reboot_recovery_receipt=reboot_receipt,
            )
            authenticated = closure.verify_closure_receipt(
                closure_receipt,
                source_root=source,
                candidate_root=candidate,
                active_contract_manifest=manifest,
                network_recovery_receipt=network_receipt,
                reboot_recovery_receipt=reboot_receipt,
            )
        if not authenticated.aggregate_execution_authenticated:
            raise SystemExit("passed recovery closure receipt was not authenticated")
        closure_receipt_content = closure_receipt.read_text(encoding="ascii")
        if any(
            value in closure_receipt_content
            for value in (
                os.fspath(manifest),
                os.fspath(network_receipt),
                os.fspath(reboot_receipt),
            )
        ):
            raise SystemExit("passed recovery closure receipt retained a private evidence path")

        with mock.patch.object(
            recovery_receipt,
            "observe_system",
            return_value=current,
        ):
            try:
                closure.verify_closure_receipt(
                    closure_receipt,
                    source_root=source,
                    candidate_root=candidate,
                )
            except closure.PreviewClosureError:
                pass
            else:
                raise SystemExit("passed recovery receipt verified without its three inputs")

            try:
                closure.verify_closure_receipt(
                    closure_receipt,
                    source_root=source,
                    candidate_root=candidate,
                    active_contract_manifest=manifest,
                    network_recovery_receipt=reboot_receipt,
                    reboot_recovery_receipt=reboot_receipt,
                )
            except closure.PreviewClosureError:
                pass
            else:
                raise SystemExit("passed recovery receipt verified with wrong recovery evidence")

            missing_receipt = closure_receipt_directory / "missing-evidence.json"
            try:
                closure.write_closure_receipt(
                    missing_receipt,
                    source_root=source,
                    candidate_root=candidate,
                    report=report,
                    active_contract_manifest=manifest,
                    network_recovery_receipt=network_receipt,
                )
            except closure.PreviewClosureError:
                pass
            else:
                raise SystemExit("passed recovery receipt was created with missing evidence")
            if missing_receipt.exists():
                raise SystemExit("missing recovery evidence left a closure receipt")

            cli_output = io.StringIO()
            with contextlib.redirect_stdout(cli_output):
                cli_code = closure.main(
                    [
                        "--source",
                        os.fspath(source),
                        "--candidate",
                        os.fspath(candidate),
                        "--verify-receipt",
                        os.fspath(closure_receipt),
                        "--active-contract-manifest",
                        os.fspath(manifest),
                        "--network-recovery-receipt",
                        os.fspath(network_receipt),
                        "--reboot-recovery-receipt",
                        os.fspath(reboot_receipt),
                    ]
                )
            cli_payload = json.loads(cli_output.getvalue())
            if (
                cli_code != 0
                or cli_payload.get("valid") is not True
                or cli_payload.get("aggregate_execution_authenticated") is not True
            ):
                raise SystemExit(
                    f"passed recovery CLI receipt verification failed: {cli_payload}"
                )
            if any(
                value in cli_output.getvalue()
                for value in (
                    os.fspath(manifest),
                    os.fspath(network_receipt),
                    os.fspath(reboot_receipt),
                )
            ):
                raise SystemExit("passed recovery CLI verification leaked an evidence path")

        marker = candidate_module.PublicCandidateMarker(
            str(report.candidate_source_commit),
            str(report.candidate_manifest_digest),
        )
        network_line = _recovery_line(
            "recovery_network",
            receipt_sha256=network_hash,
            source_commit=source_commit,
            contract_set_sha256=evidence.contract_set_digest,
            observed_on=network_observed_on,
        )
        wrong_finalized_date_line = _recovery_line(
            "recovery_network",
            receipt_sha256=network_hash,
            source_commit=source_commit,
            contract_set_sha256=evidence.contract_set_digest,
            observed_on="2000-01-01",
        )
        wrong_finalized_date = runbook.replace(
            network_line,
            wrong_finalized_date_line,
            1,
        )
        wrong_date_dispositions, _ = closure._proof_dispositions(wrong_finalized_date)
        with mock.patch.object(recovery_receipt, "observe_system", return_value=current):
            wrong_date_audit = closure._recovery_attestation(
                wrong_finalized_date,
                wrong_date_dispositions,
                source=source,
                candidate_marker=marker,
                active_contract_manifest=manifest,
                network_receipt=network_receipt,
                reboot_receipt=reboot_receipt,
            )
        if not wrong_date_audit.invalid or wrong_date_audit.attested:
            raise SystemExit(
                "recovery attestation date was not bound to receipt finalization: "
                f"{wrong_date_audit}"
            )

        wrong_source = runbook.replace(
            _recovery_line(
                "recovery_network",
                receipt_sha256=network_hash,
                source_commit=source_commit,
                contract_set_sha256=evidence.contract_set_digest,
                observed_on=network_observed_on,
            ),
            _recovery_line(
                "recovery_network",
                receipt_sha256=network_hash,
                source_commit="d" * 40,
                contract_set_sha256=evidence.contract_set_digest,
                observed_on=network_observed_on,
            ),
            1,
        ).replace(
            _recovery_line(
                "recovery_reboot",
                receipt_sha256=reboot_hash,
                source_commit=source_commit,
                contract_set_sha256=evidence.contract_set_digest,
                observed_on=reboot_observed_on,
            ),
            _recovery_line(
                "recovery_reboot",
                receipt_sha256=reboot_hash,
                source_commit="d" * 40,
                contract_set_sha256=evidence.contract_set_digest,
                observed_on=reboot_observed_on,
            ),
            1,
        )
        wrong_dispositions, _ = closure._proof_dispositions(wrong_source)
        with mock.patch.object(recovery_receipt, "observe_system", return_value=current):
            wrong_source_audit = closure._recovery_attestation(
                wrong_source,
                wrong_dispositions,
                source=source,
                candidate_marker=marker,
                active_contract_manifest=manifest,
                network_receipt=network_receipt,
                reboot_receipt=reboot_receipt,
            )
        if not wrong_source_audit.mismatched or wrong_source_audit.attested:
            raise SystemExit(f"candidate/source-mismatched receipts were accepted: {wrong_source_audit}")

        mismatched_contract = runbook.replace(
            _recovery_line(
                "recovery_reboot",
                receipt_sha256=reboot_hash,
                source_commit=source_commit,
                contract_set_sha256=evidence.contract_set_digest,
                observed_on=reboot_observed_on,
            ),
            _recovery_line(
                "recovery_reboot",
                receipt_sha256=reboot_hash,
                source_commit=source_commit,
                contract_set_sha256="e" * 64,
                observed_on=reboot_observed_on,
            ),
            1,
        )
        mismatched_dispositions, _ = closure._proof_dispositions(mismatched_contract)
        contract_audit = closure._recovery_attestation(
            mismatched_contract,
            mismatched_dispositions,
            source=source,
            candidate_marker=marker,
            active_contract_manifest=manifest,
            network_receipt=network_receipt,
            reboot_receipt=reboot_receipt,
        )
        if not contract_audit.mismatched or contract_audit.attested:
            raise SystemExit(f"cross-receipt contract mismatch was accepted: {contract_audit}")

        network_receipt.write_bytes(network_receipt.read_bytes() + b" ")
        network_receipt.chmod(0o600)
        with (
            mock.patch.object(recovery_receipt, "observe_system", return_value=current),
            mock.patch.object(closure, "_run_aggregate") as aggregate,
        ):
            tampered = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
                active_contract_manifest=manifest,
                network_recovery_receipt=network_receipt,
                reboot_recovery_receipt=reboot_receipt,
            )
        if (
            aggregate.called
            or tampered.ready
            or tampered.recovery_proof_attested
            or "recovery_proof_attestation_invalid" not in tampered.blocking_codes
        ):
            raise SystemExit(f"tampered receipt-file hash was accepted: {tampered}")
        template_root_patch.stop()


def test_recovery_receipt_hash_rejects_unsafe_permissions_and_symlink_parents() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-receipt-path-", dir="/private/tmp") as temp:
        root = Path(temp)
        real_parent = root / "real"
        real_parent.mkdir()
        real_parent.chmod(0o700)
        receipt = real_parent / "receipt.json"
        receipt.write_text('{"synthetic":true}\n', encoding="utf-8")
        receipt.chmod(0o600)
        expected = closure.hashlib.sha256(receipt.read_bytes()).hexdigest()
        if closure._stable_recovery_receipt_sha256(receipt) != expected:
            raise SystemExit("safe owner-only receipt did not hash stably")

        receipt.chmod(0o644)
        if closure._stable_recovery_receipt_sha256(receipt) is not None:
            raise SystemExit("group/world-readable recovery receipt was accepted")
        receipt.chmod(0o600)
        real_parent.chmod(0o755)
        if closure._stable_recovery_receipt_sha256(receipt) is not None:
            raise SystemExit("non-private recovery receipt parent was accepted")
        real_parent.chmod(0o700)
        with mock.patch.object(closure.os, "geteuid", return_value=os.geteuid() + 1):
            if closure._stable_recovery_receipt_sha256(receipt) is not None:
                raise SystemExit("recovery receipt owned by another uid was accepted")

        parent_alias = root / "parent-alias"
        parent_alias.symlink_to(real_parent, target_is_directory=True)
        if closure._stable_recovery_receipt_sha256(parent_alias / receipt.name) is not None:
            raise SystemExit("symlinked recovery receipt parent was accepted")
        file_alias = real_parent / "receipt-alias.json"
        file_alias.symlink_to(receipt)
        if closure._stable_recovery_receipt_sha256(file_alias) is not None:
            raise SystemExit("symlinked recovery receipt file was accepted")


def test_status_only_connector_pass_cannot_close() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-status-only-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))
        status_only = _runbook(pending=False).replace(
            "kakaotalk=disabled",
            "kakaotalk=passed",
            1,
        )
        _write(source, closure.PROOF_RUNBOOK, status_only)
        _commit(source, "status-only connector edit")
        with mock.patch.object(closure, "_run_aggregate") as aggregate:
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if aggregate.called or report.ready or report.preconditions_ready:
            raise SystemExit(f"status-only connector edit reached closure: {report}")
        if report.unattested_connector_proofs != ("kakaotalk",):
            raise SystemExit(f"status-only edit did not identify missing evidence: {report}")
        if "connector_proof_attestation_missing" not in report.blocking_codes:
            raise SystemExit(f"status-only edit missed evidence blocker: {report}")


def test_connector_attestations_reject_missing_tampered_mismatched_and_unknown() -> None:
    base = _runbook(pending=False)
    telegram_line = next(
        line for line in base.splitlines() if line.startswith("telegram=passed|")
    )
    tampered_line = telegram_line[:-1] + ("0" if telegram_line[-1] != "0" else "1")
    future_date = "9999-12-31"
    future_digest = closure._connector_attestation_digest(
        "telegram",
        "approval_bound_sender_success_recipient_confirmed",
        future_date,
    )
    future_line = (
        "telegram=passed|approval_bound_sender_success_recipient_confirmed|"
        f"{future_date}|{future_digest}"
    )
    cases = {
        "missing": base.replace(f"{telegram_line}\n", ""),
        "tampered": base.replace(telegram_line, tampered_line, 1),
        "future": base.replace(telegram_line, future_line, 1),
        "mismatched": base.replace("telegram=passed|", "telegram=disabled|", 1),
        "unknown": base.replace(
            "telegram=passed|approval_bound_sender_success_recipient_confirmed|",
            "telegram=passed|unknown_evidence_class|",
            1,
        ),
    }
    expected_codes = {
        "missing": "connector_proof_attestation_invalid",
        "tampered": "connector_proof_attestation_invalid",
        "future": "connector_proof_attestation_invalid",
        "mismatched": "connector_proof_attestation_mismatch",
        "unknown": "connector_proof_attestation_invalid",
    }
    for label, runbook in cases.items():
        dispositions, valid = closure._proof_dispositions(runbook)
        if not valid:
            raise SystemExit(f"{label} fixture unexpectedly changed proof dispositions")
        audit = closure._connector_attestations(
            runbook,
            dispositions,
            today=closure.date(2026, 8, 8),
        )
        if (
            audit.structurally_valid
            and not audit.missing_lanes
            and not audit.invalid_lanes
            and not audit.mismatched_lanes
        ):
            raise SystemExit(f"{label} connector evidence was accepted: {audit}")

        with TemporaryDirectory(
            prefix=f"jarvis-v3-preview-attestation-{label}-",
            dir="/private/tmp",
        ) as temp:
            source, candidate = _fixture(Path(temp))
            _write(source, closure.PROOF_RUNBOOK, runbook)
            _commit(source, f"{label} connector attestation")
            with mock.patch.object(closure, "_run_aggregate") as aggregate:
                report = closure.evaluate_preview_closure(
                    source,
                    candidate,
                    run_aggregate=True,
                    environ=_closure_environment(),
                )
            if aggregate.called or report.ready or report.preconditions_ready:
                raise SystemExit(f"{label} connector evidence reached closure: {report}")
            if expected_codes[label] not in report.blocking_codes:
                raise SystemExit(f"{label} connector evidence missed blocker: {report}")


def test_proof_state_change_after_candidate_requires_rebuild() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-evidence-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp), pending=True)
        _write(source, closure.PROOF_RUNBOOK, _runbook(pending=False))
        _commit(source, "record content-free proof dispositions")
        with mock.patch.object(closure, "_run_aggregate") as aggregate:
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if aggregate.called or report.ready or report.preconditions_ready:
            raise SystemExit(f"post-candidate proof-state edit reached closure: {report}")
        if report.selected_content_matches or not report.evidence_only_head_drift:
            raise SystemExit(f"allowlisted proof-state drift was not detected: {report}")
        if (
            report.connector_proof_state_candidate_bound
            or "connector_proof_state_not_candidate_bound" not in report.blocking_codes
            or "selected_content_drift" not in report.blocking_codes
        ):
            raise SystemExit(f"post-candidate proof state was not rejected: {report}")


def test_recovery_source_binding_ignores_only_evidence_documents() -> None:
    with TemporaryDirectory(
        prefix="jarvis-v3-preview-recovery-source-",
        dir="/private/tmp",
    ) as temp:
        source, _candidate = _fixture(Path(temp))
        receipt_commit = subprocess.run(
            ["git", "-C", os.fspath(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        _write(source, closure.PROOF_RUNBOOK, _runbook(pending=True))
        _commit(source, "record later evidence-only state")
        evidence_commit = subprocess.run(
            ["git", "-C", os.fspath(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        evidence_marker = candidate_module.PublicCandidateMarker(
            evidence_commit,
            "a" * 64,
        )
        if not closure._recovery_source_matches_candidate(
            source,
            receipt_commit,
            evidence_marker,
        ):
            raise SystemExit("evidence-only runbook commit broke recovery source binding")

        _write(source, "jarvis_v2/module.py", "VALUE = 'changed-after-receipt'\n")
        _commit(source, "change executable source after recovery observation")
        code_commit = subprocess.run(
            ["git", "-C", os.fspath(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        code_marker = candidate_module.PublicCandidateMarker(code_commit, "b" * 64)
        if closure._recovery_source_matches_candidate(
            source,
            receipt_commit,
            code_marker,
        ):
            raise SystemExit("post-receipt executable source drift was accepted")


def test_matching_status_and_digest_edit_after_candidate_still_requires_rebuild() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-regenerated-digest-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))
        runbook = _runbook(pending=False).replace(
            "kakaotalk=disabled",
            "kakaotalk=passed",
            1,
        )
        evidence_class = "approval_bound_target_verified_recipient_confirmed"
        observed_on = "2026-08-06"
        digest = closure._connector_attestation_digest(
            "kakaotalk",
            evidence_class,
            observed_on,
        )
        runbook = runbook.replace(
            "kakaotalk=none",
            f"kakaotalk=passed|{evidence_class}|{observed_on}|{digest}",
            1,
        )
        dispositions, valid = closure._proof_dispositions(runbook)
        audit = closure._connector_attestations(runbook, dispositions)
        if (
            not valid
            or not audit.structurally_valid
            or audit.missing_lanes
            or audit.invalid_lanes
            or audit.mismatched_lanes
            or "kakaotalk" not in audit.attested_lanes
        ):
            raise SystemExit(f"regenerated-digest fixture itself was invalid: {audit}")
        _write(source, closure.PROOF_RUNBOOK, runbook)
        _commit(source, "status and matching digest after candidate")
        with mock.patch.object(closure, "_run_aggregate") as aggregate:
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if aggregate.called or report.ready or report.preconditions_ready:
            raise SystemExit(f"regenerated proof digest bypassed candidate binding: {report}")
        if (
            report.connector_proof_attested != 4
            or report.connector_proof_state_candidate_bound
            or "connector_proof_state_not_candidate_bound" not in report.blocking_codes
        ):
            raise SystemExit(f"regenerated proof digest missed candidate blocker: {report}")


def test_required_gate_edit_after_candidate_requires_rebuild() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-gate-rebuild-", dir="/private/tmp") as temp:
        fixture_root = Path(temp)
        source, candidate = _fixture(fixture_root, checked=False)
        _write(source, closure.GATE_DOCUMENT, _gate(checked=True))
        _commit(source, "post-candidate required gate edit")

        with mock.patch.object(closure, "_run_aggregate") as aggregate:
            stale_report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if aggregate.called or stale_report.ready or stale_report.preconditions_ready:
            raise SystemExit(f"post-candidate required gate edit reached closure: {stale_report}")
        if stale_report.required_gate_checked != len(closure.REQUIRED_GATE_IDS):
            raise SystemExit(f"required gate edit fixture did not become locally complete: {stale_report}")
        if (
            stale_report.required_gate_state_candidate_bound
            or not stale_report.connector_proof_state_candidate_bound
            or stale_report.closure_evidence_state_candidate_bound
            or "required_gate_state_not_candidate_bound" not in stale_report.blocking_codes
            or "closure_evidence_state_not_candidate_bound" not in stale_report.blocking_codes
        ):
            raise SystemExit(f"post-candidate required gate edit missed binding blocker: {stale_report}")

        rebuilt_candidate = fixture_root / "rebuilt-candidate"
        with mock.patch.object(candidate_module, "PRIVATE_TMP_ROOT", fixture_root):
            rebuilt = build_public_candidate(source, rebuilt_candidate)
        if not rebuilt.ok or not rebuilt.candidate_finalized_immutable:
            raise SystemExit(f"required gate fixture candidate rebuild failed: {rebuilt}")
        with mock.patch.object(closure, "_run_aggregate", return_value=_passing_aggregate()):
            rebuilt_report = closure.evaluate_preview_closure(
                source,
                rebuilt_candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if (
            not rebuilt_report.ready
            or not rebuilt_report.preconditions_ready
            or not rebuilt_report.required_gate_state_candidate_bound
            or not rebuilt_report.connector_proof_state_candidate_bound
            or not rebuilt_report.closure_evidence_state_candidate_bound
            or rebuilt_report.blocking_codes
        ):
            raise SystemExit(f"rebuilt candidate did not bind required gate state: {rebuilt_report}")


def test_selected_source_drift_and_unchecked_gate_block() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-drift-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))
        _write(source, "jarvis_v2/module.py", "VALUE = 'changed'\n")
        _commit(source, "change selected public source")
        report = closure.evaluate_preview_closure(
            source,
            candidate,
            environ=_closure_environment(),
        )
        if report.selected_content_matches or report.evidence_only_head_drift:
            raise SystemExit(f"selected source drift was accepted: {report}")
        if not {"selected_content_drift", "non_evidence_head_drift"}.issubset(report.blocking_codes):
            raise SystemExit(f"selected source drift blockers were incomplete: {report}")

    with TemporaryDirectory(prefix="jarvis-v3-preview-unchecked-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp), checked=False)
        report = closure.evaluate_preview_closure(
            source,
            candidate,
            environ=_closure_environment(),
        )
        if (
            report.required_gate_checked != len(closure.REQUIRED_GATE_IDS) - 1
            or "required_gate_incomplete" not in report.blocking_codes
        ):
            raise SystemExit(f"unchecked gate was accepted: {report}")


def test_unknown_or_failed_disposition_blocks_until_explicitly_disabled() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-unknown-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))
        ambiguous = _runbook(pending=False).replace("kakaotalk=disabled", "kakaotalk=unknown")
        _write(source, closure.PROOF_RUNBOOK, ambiguous)
        _commit(source, "record unknown supervised outcome")
        with mock.patch.object(closure, "_run_aggregate") as aggregate:
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if aggregate.called or report.unknown_supervised_proofs != ("kakaotalk",):
            raise SystemExit(f"unknown supervised outcome did not fail closed: {report}")
        if "supervised_proof_disposition_unknown" not in report.blocking_codes:
            raise SystemExit(f"unknown supervised outcome blocker was missing: {report}")


def test_required_and_connector_lanes_cannot_be_closed_by_deferral() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-policy-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))
        runbook = _runbook(pending=False)
        runbook = runbook.replace("voice=passed", "voice=deferred")
        runbook = runbook.replace("kakaotalk=disabled", "kakaotalk=deferred")
        _write(source, closure.PROOF_RUNBOOK, runbook)
        _commit(source, "record invalid preview deferrals")
        with mock.patch.object(closure, "_run_aggregate") as aggregate:
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if aggregate.called or report.ready or report.preconditions_ready:
            raise SystemExit(f"policy-invalid deferrals reached aggregate closure: {report}")
        if report.unaccepted_supervised_proofs != ("kakaotalk", "voice"):
            raise SystemExit(f"proof policy violations were not exact: {report}")
        if "supervised_proof_policy_violation" not in report.blocking_codes:
            raise SystemExit(f"proof policy blocker was missing: {report}")

        all_deferred = _runbook(pending=False)
        for lane in closure.PROOF_LANES:
            current = next(
                value
                for value in closure.VALID_DISPOSITIONS
                if f"{lane}={value}" in all_deferred
            )
            all_deferred = all_deferred.replace(f"{lane}={current}", f"{lane}=deferred")
        dispositions, valid = closure._proof_dispositions(all_deferred)
        violations = closure._proof_policy_violations(dispositions)
        if not valid or set(violations) != set(closure.PROOF_LANES) - {"recovery_services"}:
            raise SystemExit(f"all-deferred proof ledger was accepted: {violations}")


def test_structured_status_blocks_reject_shape_errors_and_ignore_prose() -> None:
    dispositions, valid = closure._proof_dispositions(_runbook(pending=False))
    if not valid or dispositions["voice"] != "passed" or dispositions["ocr"] != "passed":
        raise SystemExit(f"negated prose influenced structured proof status: {dispositions}")

    malformed_proofs = (
        _runbook(pending=False).replace("voice=passed\n", "voice=passed\nvoice=passed\n"),
        _runbook(pending=False).replace("voice=passed\n", ""),
        _runbook(pending=False).replace("voice=passed\n", "voice=passed\nextra_lane=passed\n"),
        _runbook(pending=False).replace("voice=passed", "voice=bypassed"),
    )
    for malformed in malformed_proofs:
        parsed, parsed_valid = closure._proof_dispositions(malformed)
        if parsed_valid or set(parsed.values()) != {"unknown"}:
            raise SystemExit(f"malformed proof block did not fail closed: {parsed}")

    malformed_gate = _gate().replace(
        "v3_environment_isolation=passed\n",
        "v3_environment_isolation=passed\nv3_environment_isolation=passed\n",
    )
    total, checked, valid = closure._required_gate_counts(malformed_gate)
    if valid or total != len(closure.REQUIRED_GATE_IDS) or checked:
        raise SystemExit("duplicate gate identifier did not fail closed")


def test_candidate_tamper_before_or_during_aggregate_blocks() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-tamper-before-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))
        candidate.chmod(0o700)
        _write(candidate, "unexpected.txt", "unexpected\n")
        candidate.chmod(0o500)
        report = closure.evaluate_preview_closure(
            source,
            candidate,
            environ=_closure_environment(),
        )
        if report.candidate_preflight_before or "candidate_preflight_or_marker_invalid" not in report.blocking_codes:
            raise SystemExit(f"pre-existing candidate tamper was accepted: {report}")

    with TemporaryDirectory(prefix="jarvis-v3-preview-tamper-during-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))

        def tampering_aggregate(*args: object, **kwargs: object) -> closure.AggregateResult:
            candidate.chmod(0o700)
            _write(candidate, "unexpected.txt", "unexpected\n")
            candidate.chmod(0o500)
            return _passing_aggregate()

        with mock.patch.object(closure, "_run_aggregate", side_effect=tampering_aggregate):
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if report.ready or report.candidate_preflight_after or report.candidate_stable:
            raise SystemExit(f"post-aggregate candidate tamper was accepted: {report}")
        if "candidate_changed_during_closure" not in report.blocking_codes:
            raise SystemExit(f"candidate change blocker was missing: {report}")


def test_candidate_path_aba_still_executes_anchored_tree() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-aba-", dir="/private/tmp") as temp:
        fixture_root = Path(temp)
        source, candidate = _fixture(fixture_root)
        original_stat = candidate.stat()

        def aba_aggregate(candidate_fd: int, **kwargs: object) -> closure.AggregateResult:
            parked = fixture_root / "parked-candidate"
            impostor = fixture_root / "candidate"
            candidate.rename(parked)
            impostor.mkdir()
            try:
                anchored = os.fstat(candidate_fd)
                if (anchored.st_dev, anchored.st_ino) != (original_stat.st_dev, original_stat.st_ino):
                    raise SystemExit("aggregate was not bound to the verified candidate descriptor")
                if (impostor.stat().st_dev, impostor.stat().st_ino) == (
                    anchored.st_dev,
                    anchored.st_ino,
                ):
                    raise SystemExit("candidate ABA fixture did not substitute the pathname")
            finally:
                impostor.rmdir()
                parked.rename(candidate)
            return _passing_aggregate()

        with mock.patch.object(closure, "_run_aggregate", side_effect=aba_aggregate):
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if not report.ready or not report.candidate_stable:
            raise SystemExit(f"anchored candidate did not survive pathname ABA: {report}")


def test_source_head_is_rebound_after_aggregate() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-preview-head-drift-", dir="/private/tmp") as temp:
        source, candidate = _fixture(Path(temp))

        def moving_head(*args: object, **kwargs: object) -> closure.AggregateResult:
            _git(
                source,
                "-c",
                "user.name=Synthetic Maintainer",
                "-c",
                "user.email=maintainer@example.com",
                "commit",
                "--allow-empty",
                "--quiet",
                "-m",
                "move closure HEAD",
            )
            return _passing_aggregate()

        with mock.patch.object(closure, "_run_aggregate", side_effect=moving_head):
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if report.ready or "source_head_changed_during_closure" not in report.blocking_codes:
            raise SystemExit(f"source HEAD drift was not rebound after aggregate: {report}")


def test_aggregate_process_gets_only_bounded_noncredential_environment() -> None:
    from jarvis_v2.scripts.smoke_test_all import TEST_MODULES

    expected_count = len(TEST_MODULES)
    process = mock.Mock()
    process.returncode = 0
    process.communicate.return_value = (
        f"All {expected_count} smoke modules passed. Duration: 1.0s.\n".encode(),
        b"",
    )
    supplied = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "OPENAI_API_KEY": "synthetic-secret-value",
        "HTTPS_PROXY": "https://synthetic.invalid",
        "TELEGRAM_BOT_TOKEN": "synthetic-bot-token",
        closure.CLOSURE_RECEIPT_ANCHOR_ENV: "d" * 64,
        closure.EXTRA_DENY_ENV: json.dumps(["synthetic-private-marker"]),
        "JARVIS_V3_ENABLE_DAEMONS": "1",
        "JARVIS_V3_ENABLE_SCHEDULER": "1",
    }
    with TemporaryDirectory(prefix="jarvis-v3-preview-fd-", dir="/private/tmp") as temp:
        candidate_fd = os.open(temp, os.O_RDONLY)
        try:
            with mock.patch.object(closure.subprocess, "Popen", return_value=process) as popen:
                result = closure._run_aggregate(candidate_fd, environ=supplied)
        finally:
            os.close(candidate_fd)
    if not result.passed:
        raise SystemExit(f"synthetic aggregate success was not recognized: {result}")
    child_env = popen.call_args.kwargs["env"]
    for forbidden in (
        "OPENAI_API_KEY",
        "HTTPS_PROXY",
        "TELEGRAM_BOT_TOKEN",
        closure.CLOSURE_RECEIPT_ANCHOR_ENV,
        closure.EXTRA_DENY_ENV,
    ):
        if forbidden in child_env:
            raise SystemExit(f"aggregate inherited forbidden environment key: {forbidden}")
    if child_env.get("JARVIS_V3_ENABLE_DAEMONS") != "0":
        raise SystemExit(f"aggregate daemon gate was not forced closed: {child_env}")
    if child_env.get("JARVIS_V3_ENABLE_SCHEDULER") != "0":
        raise SystemExit(f"aggregate scheduler gate was not forced closed: {child_env}")
    if popen.call_args.kwargs.get("stdout") is not subprocess.PIPE or popen.call_args.kwargs.get("stderr") is not subprocess.PIPE:
        raise SystemExit("aggregate child output was not kept out of the closure summary")
    if popen.call_args.kwargs.get("start_new_session") is not True:
        raise SystemExit("aggregate was not isolated in its own process group")
    if popen.call_args.kwargs.get("pass_fds") != (candidate_fd,):
        raise SystemExit("aggregate did not inherit exactly the anchored candidate descriptor")
    if "cwd" in popen.call_args.kwargs or child_env.get("PYTHONPATH") != ".":
        raise SystemExit("aggregate used the mutable candidate pathname")
    preexec = popen.call_args.kwargs.get("preexec_fn")
    if not callable(preexec):
        raise SystemExit("aggregate did not bind child cwd through the candidate descriptor")


def test_aggregate_failure_diagnostics_are_bounded_and_allowlisted() -> None:
    from jarvis_v2.scripts.smoke_test_all import TEST_MODULES

    expected = tuple(TEST_MODULES)
    selected = (expected[0], expected[-1])
    raw_path = "/private/tmp/synthetic-sensitive-path"
    raw_secret = "synthetic-secret-output"

    def run_with(stdout: bytes, *, returncode: int = 1) -> closure.AggregateResult:
        process = mock.Mock()
        process.returncode = returncode
        process.communicate.return_value = (stdout, raw_secret.encode())
        with TemporaryDirectory(
            prefix="jarvis-v3-preview-diagnostics-", dir="/private/tmp"
        ) as temp:
            candidate_fd = os.open(temp, os.O_RDONLY)
            try:
                with mock.patch.object(
                    closure.subprocess, "Popen", return_value=process
                ):
                    return closure._run_aggregate(
                        candidate_fd, environ=_closure_environment()
                    )
            finally:
                os.close(candidate_fd)

    valid_failure = (
        f"untrusted child text {raw_path} {raw_secret}\n"
        f"FAILED: 2 of {len(expected)} smoke modules failed. Duration: 4.2s.\n"
        f"- {selected[0]}\n- {selected[1]}\n"
    ).encode()
    result = run_with(valid_failure)
    if result.passed or result.failure_reason != "failed" or result.failed_modules != selected:
        raise SystemExit(f"valid bounded aggregate failure was not parsed: {result}")
    rendered = repr(result)
    if raw_path in rendered or raw_secret in rendered:
        raise SystemExit("aggregate failure diagnostics retained raw child output")

    malformed_failure = (
        f"FAILED: 2 of {len(expected)} smoke modules failed. Duration: 4.2s.\n"
        f"- {selected[0]}\n- jarvis_v2.scripts.smoke_test_not_allowlisted\n"
    ).encode()
    malformed = run_with(malformed_failure)
    if (
        malformed.failure_reason != "malformed_summary"
        or malformed.failed_modules
    ):
        raise SystemExit(f"malformed aggregate summary escaped its boundary: {malformed}")

    process = mock.Mock()
    process.communicate.side_effect = subprocess.TimeoutExpired(
        cmd="synthetic aggregate",
        timeout=1,
        output=(
            f"{raw_path}\n== [1/{len(expected)}] {expected[0]} ==\n"
        ).encode(),
        stderr=raw_secret.encode(),
    )
    with TemporaryDirectory(
        prefix="jarvis-v3-preview-timeout-diagnostics-", dir="/private/tmp"
    ) as temp:
        candidate_fd = os.open(temp, os.O_RDONLY)
        try:
            with (
                mock.patch.object(closure.subprocess, "Popen", return_value=process),
                mock.patch.object(closure, "_terminate_process_group"),
            ):
                timed_out = closure._run_aggregate(
                    candidate_fd, environ=_closure_environment()
                )
        finally:
            os.close(candidate_fd)
    if timed_out.failure_reason != "timed_out" or timed_out.failed_modules != (
        expected[0],
    ):
        raise SystemExit(f"aggregate timeout diagnostics were not bounded: {timed_out}")
    if raw_path in repr(timed_out) or raw_secret in repr(timed_out):
        raise SystemExit("aggregate timeout diagnostics retained raw child output")


def test_process_group_cleanup_terminates_descendants() -> None:
    with TemporaryDirectory(
        prefix="jarvis-v3-preview-group-cleanup-", dir="/private/tmp"
    ) as temp:
        progress_path = Path(temp) / "descendant.progress"
        descendant_code = (
            "import time\n"
            "from pathlib import Path\n"
            f"progress = Path({os.fspath(progress_path)!r})\n"
            "for index in range(600):\n"
            "    progress.write_text(str(index), encoding='ascii')\n"
            "    time.sleep(0.05)\n"
        )
        helper = (
            "import signal,subprocess,sys,time\n"
            "from pathlib import Path\n"
            "from jarvis_v2.scripts import smoke_test_all as suite\n"
            f"progress = Path({os.fspath(progress_path)!r})\n"
            f"child=subprocess.Popen([sys.executable,'-c',{descendant_code!r}], "
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
            "stderr=subprocess.DEVNULL,start_new_session=True)\n"
            "suite._ACTIVE_MODULE_PROCESS=child\n"
            "signal.signal(signal.SIGTERM, suite._handle_shutdown_signal)\n"
            "deadline = time.monotonic() + 5\n"
            "while not progress.exists() and time.monotonic() < deadline:\n"
            "    time.sleep(0.01)\n"
            "if not progress.exists():\n"
            "    try:\n"
            "        child.kill()\n"
            "    except ProcessLookupError:\n"
            "        pass\n"
            "    try:\n"
            "        child.wait(timeout=5)\n"
            "    except subprocess.TimeoutExpired:\n"
            "        pass\n"
            "    raise SystemExit('descendant failed to start')\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", helper],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None
        descendant_pid = int(process.stdout.readline().decode("ascii").strip())
        closure._terminate_process_group(process)
        if process.poll() is None:
            raise SystemExit("aggregate parent was not reaped after process-group cleanup")
        _assert_process_progress_stopped(
            progress_path,
            process_group_id=descendant_pid,
            label="aggregate descendant",
        )


def test_aggregate_capture_cap_and_closed_pipe_descendant_cleanup() -> None:
    noisy_processes: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def noisy_popen(*_args, **_kwargs):
        process = real_popen(
            [
                sys.executable,
                "-c",
                "import os,time; os.write(1,b'x'*4096); time.sleep(30)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        noisy_processes.append(process)
        return process

    with TemporaryDirectory(
        prefix="jarvis-v3-preview-output-cap-", dir="/private/tmp"
    ) as temp:
        candidate_fd = os.open(temp, os.O_RDONLY)
        try:
            with (
                mock.patch.object(closure.subprocess, "Popen", side_effect=noisy_popen),
                mock.patch.object(closure, "MAX_AGGREGATE_CAPTURE_BYTES", 1024),
            ):
                result = closure._run_aggregate(
                    candidate_fd, environ=_closure_environment()
                )
        finally:
            os.close(candidate_fd)
    if (
        result.passed
        or result.failure_reason != "malformed_summary"
        or len(noisy_processes) != 1
        or noisy_processes[0].poll() is None
        or noisy_processes[0].stdout is None
        or not noisy_processes[0].stdout.closed
        or noisy_processes[0].stderr is None
        or not noisy_processes[0].stderr.closed
    ):
        raise SystemExit(
            "aggregate output cap did not fail closed, reap the child, and close its pipes"
        )

    from jarvis_v2.scripts.smoke_test_all import TEST_MODULES

    with TemporaryDirectory(
        prefix="jarvis-v3-preview-success-descendant-", dir="/private/tmp"
    ) as temp:
        fixture_root = Path(temp)
        descendant_pid_path = fixture_root / "descendant.pid"
        descendant_progress_path = fixture_root / "descendant.progress"
        descendant_code = (
            "import os, signal, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"Path({os.fspath(descendant_pid_path)!r}).write_text(str(os.getpid()))\n"
            f"progress = Path({os.fspath(descendant_progress_path)!r})\n"
            "for index in range(600):\n"
            "    progress.write_text(str(index), encoding='ascii')\n"
            "    time.sleep(0.05)\n"
        )
        parent_code = (
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            f"p=subprocess.Popen([sys.executable,'-c',{descendant_code!r}],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            f"Path({os.fspath(descendant_pid_path)!r}).write_text(str(p.pid))\n"
            f"progress = Path({os.fspath(descendant_progress_path)!r})\n"
            "deadline = time.monotonic() + 5\n"
            "while not progress.exists() and time.monotonic() < deadline:\n"
            "    time.sleep(0.01)\n"
            "if not progress.exists():\n"
            "    raise SystemExit('descendant failed to start')\n"
            f"print('All {len(TEST_MODULES)} smoke modules passed. Duration: 0.1s.', flush=True)\n"
        )
        residual_processes: list[subprocess.Popen[bytes]] = []

        def residual_popen(*_args, **_kwargs):
            process = real_popen(
                [sys.executable, "-c", parent_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            residual_processes.append(process)
            return process

        candidate_fd = os.open(temp, os.O_RDONLY)
        try:
            with mock.patch.object(
                closure.subprocess, "Popen", side_effect=residual_popen
            ):
                residual_result = closure._run_aggregate(
                    candidate_fd, environ=_closure_environment()
                )
        finally:
            os.close(candidate_fd)
        if (
            residual_result.passed
            or residual_result.failure_reason != "malformed_summary"
            or len(residual_processes) != 1
            or residual_processes[0].poll() is None
            or residual_processes[0].stdout is None
            or not residual_processes[0].stdout.closed
            or residual_processes[0].stderr is None
            or not residual_processes[0].stderr.closed
        ):
            raise SystemExit(
                "valid aggregate footer falsely passed with a residual closed-pipe descendant"
            )
        if not descendant_pid_path.is_file():
            raise SystemExit("aggregate residual descendant did not publish its pid")
        if not descendant_progress_path.is_file():
            raise SystemExit("aggregate residual descendant did not publish progress")
        _assert_process_progress_stopped(
            descendant_progress_path,
            process_group_id=residual_processes[0].pid,
            label="aggregate normal-success descendant",
        )

    clean_processes: list[subprocess.Popen[bytes]] = []

    def clean_popen(*_args, **_kwargs):
        process = real_popen(
            [
                sys.executable,
                "-c",
                f"print('All {len(TEST_MODULES)} smoke modules passed. Duration: 0.1s.')",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        clean_processes.append(process)
        return process

    with TemporaryDirectory(
        prefix="jarvis-v3-preview-clean-success-", dir="/private/tmp"
    ) as temp:
        candidate_fd = os.open(temp, os.O_RDONLY)
        try:
            with mock.patch.object(
                closure.subprocess, "Popen", side_effect=clean_popen
            ):
                clean_result = closure._run_aggregate(
                    candidate_fd, environ=_closure_environment()
                )
        finally:
            os.close(candidate_fd)
    if (
        not clean_result.passed
        or len(clean_processes) != 1
        or clean_processes[0].stdout is None
        or not clean_processes[0].stdout.closed
        or clean_processes[0].stderr is None
        or not clean_processes[0].stderr.closed
    ):
        raise SystemExit("aggregate clean success did not close both owned pipes")

    naturally_reaped_process = real_popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    with mock.patch.object(
        closure,
        "_process_group_exists",
        side_effect=(True, False),
    ) as process_group_exists:
        stdout, stderr = closure._bounded_communicate(
            naturally_reaped_process,
            timeout=PROCESS_CLEANUP_TEST_DEADLINE_SECONDS,
        )
    if (
        stdout
        or stderr
        or naturally_reaped_process.returncode != 0
        or process_group_exists.call_count != 2
    ):
        raise SystemExit("aggregate transient process-group teardown was not rechecked")

    with TemporaryDirectory(
        prefix="jarvis-v3-preview-closed-pipe-progress-", dir="/private/tmp"
    ) as progress_temp:
        progress_path = Path(progress_temp) / "descendant.progress"
        descendant_code = (
            "import signal, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"progress = Path({os.fspath(progress_path)!r})\n"
            "for index in range(600):\n"
            "    progress.write_text(str(index), encoding='ascii')\n"
            "    time.sleep(0.05)\n"
        )
        parent_code = (
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            f"subprocess.Popen([sys.executable,'-c',{descendant_code!r}],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            f"progress = Path({os.fspath(progress_path)!r})\n"
            "deadline = time.monotonic() + 5\n"
            "while not progress.exists() and time.monotonic() < deadline:\n"
            "    time.sleep(0.01)\n"
            "if not progress.exists():\n"
            "    raise SystemExit('descendant failed to start')\n"
            "print('ready', flush=True)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", parent_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None
        if process.stdout.readline().strip() != b"ready":
            raise SystemExit("closed-pipe descendant fixture did not start")
        process.wait(timeout=PROCESS_CLEANUP_TEST_DEADLINE_SECONDS)
        closure._terminate_process_group(process)
        _assert_process_progress_stopped(
            progress_path,
            process_group_id=process.pid,
            label="closed-pipe aggregate descendant",
        )

    deadline_process = real_popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline_process.wait(timeout=PROCESS_CLEANUP_TEST_DEADLINE_SECONDS)
    monotonic_values = iter((100.0, 100.5))

    def expired_monotonic() -> float:
        return next(monotonic_values, 102.0)

    with mock.patch.object(closure.time, "monotonic", side_effect=expired_monotonic):
        try:
            closure._bounded_communicate(deadline_process, timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            if exc.output != b"" or exc.stderr != b"":
                raise SystemExit("aggregate hard-deadline timeout lost bounded empty tails") from exc
        else:
            raise SystemExit("aggregate accepted a child after its hard deadline")
    if (
        deadline_process.stdout is None
        or not deadline_process.stdout.closed
        or deadline_process.stderr is None
        or not deadline_process.stderr.closed
    ):
        raise SystemExit("aggregate hard-deadline timeout left pipe descriptors open")


def test_unexpected_aggregate_communication_error_cleans_group() -> None:
    process = mock.Mock()
    process.pid = 424242
    process.communicate.side_effect = [OSError("synthetic pipe failure"), (b"", b"")]
    with TemporaryDirectory(prefix="jarvis-v3-preview-error-cleanup-", dir="/private/tmp") as temp:
        candidate_fd = os.open(temp, os.O_RDONLY)
        try:
            with (
                mock.patch.object(closure.subprocess, "Popen", return_value=process),
                mock.patch.object(closure.os, "killpg") as killpg,
            ):
                result = closure._run_aggregate(candidate_fd, environ=_closure_environment())
        finally:
            os.close(candidate_fd)
    if (
        result.passed
        or not result.ran
        or result.failure_reason != "malformed_summary"
        or result.failed_modules
        or not killpg.called
    ):
        raise SystemExit("unexpected aggregate communication error left cleanup unproved")

    real_selector_factory = closure.selectors.DefaultSelector

    class ExplodingSelector:
        def __init__(self):
            self.selector = real_selector_factory()

        def register(self, *args, **kwargs):
            return self.selector.register(*args, **kwargs)

        def get_map(self):
            return self.selector.get_map()

        def select(self, _timeout=None):
            raise KeyboardInterrupt("synthetic aggregate selector interruption")

        def close(self) -> None:
            self.selector.close()

    real_process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        with mock.patch.object(
            closure.selectors, "DefaultSelector", ExplodingSelector
        ):
            try:
                closure._bounded_communicate(real_process, timeout=5.0)
            except KeyboardInterrupt as exc:
                if str(exc) != "synthetic aggregate selector interruption":
                    raise
            else:
                raise SystemExit("aggregate selector interruption did not propagate")
    finally:
        closure._terminate_process_group(real_process)
    if (
        real_process.stdout is None
        or not real_process.stdout.closed
        or real_process.stderr is None
        or not real_process.stderr.closed
    ):
        raise SystemExit("unexpected aggregate failure left owned pipe descriptors open")

    constructor_process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        with mock.patch.object(
            closure.selectors,
            "DefaultSelector",
            side_effect=RuntimeError("synthetic selector construction failure"),
        ):
            try:
                closure._bounded_communicate(constructor_process, timeout=5.0)
            except RuntimeError as exc:
                if str(exc) != "synthetic selector construction failure":
                    raise
            else:
                raise SystemExit("aggregate selector construction failure did not propagate")
    finally:
        closure._terminate_process_group(constructor_process)
    if (
        constructor_process.stdout is None
        or not constructor_process.stdout.closed
        or constructor_process.stderr is None
        or not constructor_process.stderr.closed
    ):
        raise SystemExit("selector construction failure left aggregate pipes open")


def test_cli_configuration_failure_is_path_free() -> None:
    hidden_source = "/private/tmp/hidden-source-name"
    hidden_candidate = "/private/tmp/hidden-candidate-name"
    hidden_manifest = "/private/tmp/hidden-manifest-name"
    hidden_network = "/private/tmp/hidden-network-receipt-name"
    hidden_reboot = "/private/tmp/hidden-reboot-receipt-name"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = closure.main(
            [
                "--source",
                hidden_source,
                "--candidate",
                hidden_candidate,
                "--active-contract-manifest",
                hidden_manifest,
                "--network-recovery-receipt",
                hidden_network,
                "--reboot-recovery-receipt",
                hidden_reboot,
            ]
        )
    payload = json.loads(output.getvalue())
    if code != 2 or payload.get("ready") is not False:
        raise SystemExit(f"unsafe closure configuration returned the wrong result: {payload}")
    if any(
        value in output.getvalue()
        for value in (
            hidden_source,
            hidden_candidate,
            hidden_manifest,
            hidden_network,
            hidden_reboot,
        )
    ):
        raise SystemExit("closure configuration error leaked a supplied path")
    if payload.get("publication_authorized") is not False or payload.get("cutover_authorized") is not False:
        raise SystemExit(f"closure configuration error authorized a forbidden action: {payload}")


def test_external_closure_receipt_is_strict_bound_and_read_only_verifiable() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-closure-receipt-", dir="/private/tmp") as temp:
        root = Path(temp)
        root.chmod(0o700)
        source, candidate = _fixture(root)
        receipt_directory = root / "receipts"
        receipt_directory.mkdir(mode=0o700)
        receipt = receipt_directory / "closure.json"
        candidate_marker = (candidate / candidate_module.PUBLIC_CANDIDATE_MARKER_NAME).read_bytes()
        aggregate = closure.AggregateResult(
            True,
            True,
            2,
            1.25,
        )
        with mock.patch.object(closure, "_run_aggregate", return_value=aggregate):
            report = closure.evaluate_preview_closure(
                source,
                candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        if not report.ready:
            raise SystemExit(f"synthetic ready closure did not close: {report}")
        substituted_before_write = receipt_directory / "substituted-deny.json"
        try:
            closure.write_closure_receipt(
                substituted_before_write,
                source_root=source,
                candidate_root=candidate,
                report=report,
                environ={
                    **_closure_environment(),
                    closure.EXTRA_DENY_ENV: json.dumps(
                        ["different-synthetic-private-marker-never-present"]
                    ),
                },
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("receipt accepted a substituted post-aggregate deny review")
        if substituted_before_write.exists():
            raise SystemExit("substituted deny review left a receipt")
        receipt_creation = closure.write_closure_receipt(
            receipt,
            source_root=source,
            candidate_root=candidate,
            report=report,
        )
        verification = closure.verify_closure_receipt(
            receipt,
            source_root=source,
            candidate_root=candidate,
            environ=_closure_environment(anchor=None),
        )
        if (
            verification.receipt_sha256 != receipt_creation.receipt_sha256
            or not verification.self_consistent
            or verification.aggregate_execution_authenticated
        ):
            raise SystemExit("self-consistent receipt overstated aggregate authentication")
        authenticated = closure.verify_closure_receipt(
            receipt,
            source_root=source,
            candidate_root=candidate,
            environ=_closure_environment(),
        )
        if not authenticated.aggregate_execution_authenticated:
            raise SystemExit("closure receipt did not verify against its candidate")
        self_bootstrapped = closure.verify_closure_receipt(
            receipt,
            source_root=source,
            candidate_root=candidate,
            environ=_closure_environment(anchor=receipt_creation.receipt_sha256),
        )
        if self_bootstrapped.aggregate_execution_authenticated:
            raise SystemExit("receipt authenticated using only its embedded digest")
        substituted_deny = closure.verify_closure_receipt(
            receipt,
            source_root=source,
            candidate_root=candidate,
            environ={
                **_closure_environment(),
                closure.EXTRA_DENY_ENV: json.dumps(
                    ["different-synthetic-private-marker-never-present"]
                ),
            },
        )
        if substituted_deny.aggregate_execution_authenticated:
            raise SystemExit("receipt authenticated with a substituted private deny review")
        receipt_stat = receipt.stat()
        if (
            receipt_stat.st_nlink != 1
            or receipt_stat.st_uid != os.getuid()
            or (receipt_stat.st_mode & 0o777) != 0o600
        ):
            raise SystemExit("closure receipt lost exclusive owner-only custody")
        content = receipt.read_text(encoding="ascii")
        if any(
            value in content
            for value in (
                os.fspath(source),
                os.fspath(candidate),
                "private-secret",
                "d" * 64,
            )
        ):
            raise SystemExit("closure receipt retained a path or private value")
        payload = json.loads(content)
        for key in (
            "paths_included",
            "private_content_included",
            "publishes",
            "publication_authorized",
            "cutover_authorized",
            "starts_services",
            "changes_schedules",
            "runs_live_actions",
        ):
            if payload.get(key) is not False:
                raise SystemExit(f"closure receipt authorized a forbidden action: {key}")
        if (candidate / candidate_module.PUBLIC_CANDIDATE_MARKER_NAME).read_bytes() != candidate_marker:
            raise SystemExit("external closure receipt changed the immutable candidate")
        if subprocess.run(
            ["git", "-C", os.fspath(source), "status", "--porcelain=v1"],
            check=False,
            capture_output=True,
        ).stdout:
            raise SystemExit("external closure receipt changed the source worktree")

        try:
            closure.write_closure_receipt(
                receipt,
                source_root=source,
                candidate_root=candidate,
                report=report,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("closure receipt overwrote an existing artifact")

        receipt_symlink = receipt_directory / "receipt-symlink.json"
        receipt_symlink.symlink_to(receipt.name)
        try:
            closure.verify_closure_receipt(
                receipt_symlink,
                source_root=source,
                candidate_root=candidate,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("symlink closure receipt was accepted")

        receipt_hardlink = receipt_directory / "receipt-hardlink.json"
        os.link(receipt, receipt_hardlink)
        try:
            closure.verify_closure_receipt(
                receipt_hardlink,
                source_root=source,
                candidate_root=candidate,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("hard-linked closure receipt was accepted")
        receipt_hardlink.unlink()
        if receipt.stat().st_nlink != 1:
            raise SystemExit("hard-link custody test did not restore the valid receipt")

        loose_parent = root / "loose-receipts"
        loose_parent.mkdir(mode=0o755)
        loose_parent.chmod(0o755)
        loose_receipt = loose_parent / "closure.json"
        loose_receipt.write_bytes(receipt.read_bytes())
        loose_receipt.chmod(0o600)
        try:
            closure.verify_closure_receipt(
                loose_receipt,
                source_root=source,
                candidate_root=candidate,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("receipt under a non-owner-only parent was accepted")

        real_parent = root / "real-receipt-parent"
        real_parent.mkdir(mode=0o700)
        linked_parent = root / "linked-receipt-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        try:
            closure.write_closure_receipt(
                linked_parent / "closure.json",
                source_root=source,
                candidate_root=candidate,
                report=report,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("symlink receipt parent was accepted")
        if (real_parent / "closure.json").exists():
            raise SystemExit("symlink parent rejection still created a receipt")

        for inside in (source / "closure-receipt.json", candidate / "closure-receipt.json"):
            try:
                closure.write_closure_receipt(
                    inside,
                    source_root=source,
                    candidate_root=candidate,
                    report=report,
                )
            except closure.PreviewClosureError:
                pass
            else:
                raise SystemExit("receipt was written inside source or candidate")
            if inside.exists():
                raise SystemExit("rejected in-tree receipt left an artifact")

        postflight_receipt = receipt_directory / "postflight-failed.json"
        with mock.patch.object(
            closure,
            "verify_closure_receipt",
            side_effect=closure.PreviewClosureError("synthetic postflight failure"),
        ):
            try:
                closure.write_closure_receipt(
                    postflight_receipt,
                    source_root=source,
                    candidate_root=candidate,
                    report=report,
                )
            except closure.PreviewClosureError:
                pass
            else:
                raise SystemExit("injected receipt postflight failure was ignored")
        if postflight_receipt.exists() or tuple(
            receipt_directory.glob(".postflight-failed.json.*.tmp")
        ):
            raise SystemExit("failed receipt postflight left an authoritative artifact")

        unknown_receipt = receipt_directory / "postlink-cleanup-unknown.json"
        real_unlink = closure.os.unlink

        def fail_authoritative_unlink(path, *args, **kwargs):
            if path == unknown_receipt.name and kwargs.get("dir_fd") is not None:
                raise OSError("synthetic authoritative unlink failure")
            return real_unlink(path, *args, **kwargs)

        with (
            mock.patch.object(
                closure,
                "verify_closure_receipt",
                side_effect=closure.PreviewClosureError("synthetic postflight failure"),
            ),
            mock.patch.object(closure.os, "unlink", side_effect=fail_authoritative_unlink),
        ):
            try:
                closure.write_closure_receipt(
                    unknown_receipt,
                    source_root=source,
                    candidate_root=candidate,
                    report=report,
                )
            except closure.ClosureReceiptOutcomeUnknown:
                pass
            else:
                raise SystemExit("post-link cleanup failure did not report outcome-unknown")
        if not unknown_receipt.exists():
            raise SystemExit("outcome-unknown fixture did not retain the linked receipt")
        unknown_receipt.unlink()

        source_drift_root = root / "source-drift"
        source_drift_root.mkdir(mode=0o700)
        drift_source, drift_candidate = _fixture(source_drift_root)
        with mock.patch.object(closure, "_run_aggregate", return_value=aggregate):
            drift_report = closure.evaluate_preview_closure(
                drift_source,
                drift_candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        _write(drift_source, "jarvis_v2/module.py", "VALUE = 'changed-after-aggregate'\n")
        _commit(drift_source, "synthetic source drift")
        source_drift_receipt = receipt_directory / "source-drift.json"
        try:
            closure.write_closure_receipt(
                source_drift_receipt,
                source_root=drift_source,
                candidate_root=drift_candidate,
                report=drift_report,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("source drift after aggregate produced a receipt")
        if source_drift_receipt.exists():
            raise SystemExit("source drift rejection left an authoritative receipt")

        candidate_drift_root = root / "candidate-drift"
        candidate_drift_root.mkdir(mode=0o700)
        stable_source, changed_candidate = _fixture(candidate_drift_root)
        with mock.patch.object(closure, "_run_aggregate", return_value=aggregate):
            candidate_drift_report = closure.evaluate_preview_closure(
                stable_source,
                changed_candidate,
                run_aggregate=True,
                environ=_closure_environment(),
            )
        changed_marker = changed_candidate / candidate_module.PUBLIC_CANDIDATE_MARKER_NAME
        changed_candidate.chmod(0o700)
        changed_marker.chmod(0o600)
        changed_marker.write_text("{}\n", encoding="ascii")
        changed_marker.chmod(0o400)
        changed_candidate.chmod(0o500)
        candidate_drift_receipt = receipt_directory / "candidate-drift.json"
        try:
            closure.write_closure_receipt(
                candidate_drift_receipt,
                source_root=stable_source,
                candidate_root=changed_candidate,
                report=candidate_drift_report,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("candidate drift after aggregate produced a receipt")
        if candidate_drift_receipt.exists():
            raise SystemExit("candidate drift rejection left an authoritative receipt")

        tampered = receipt_directory / "tampered.json"
        tampered_payload = dict(payload)
        tampered_payload["closure_runner_sha256"] = "0" * 64
        tampered_payload["receipt_sha256"] = closure._receipt_digest(tampered_payload)
        tampered.write_bytes(closure._canonical_json_bytes(tampered_payload))
        tampered.chmod(0o600)
        try:
            closure.verify_closure_receipt(
                tampered,
                source_root=source,
                candidate_root=candidate,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("rehashed receipt with a false runner binding was accepted")

        public_digest_tamper = receipt_directory / "public-digest-tamper.json"
        public_digest_payload = dict(payload)
        public_digest_payload["aggregate_duration_seconds"] = float(
            public_digest_payload["aggregate_duration_seconds"]
        ) + 1.0
        public_digest_payload["receipt_sha256"] = closure._receipt_digest(
            public_digest_payload
        )
        public_digest_tamper.write_bytes(
            closure._canonical_json_bytes(public_digest_payload)
        )
        public_digest_tamper.chmod(0o600)
        public_digest_verification = closure.verify_closure_receipt(
            public_digest_tamper,
            source_root=source,
            candidate_root=candidate,
            environ=_closure_environment(),
        )
        if public_digest_verification.aggregate_execution_authenticated:
            raise SystemExit("public receipt digest authenticated a changed receipt payload")

        duplicate = receipt_directory / "duplicate.json"
        duplicate.write_text(
            content[:-2] + ',"schema":"jarvis-v3-preview-closure-receipt"}\n',
            encoding="ascii",
        )
        duplicate.chmod(0o600)
        try:
            closure.verify_closure_receipt(
                duplicate,
                source_root=source,
                candidate_root=candidate,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("duplicate-key receipt was accepted")

        failed = closure.AggregateResult(True, False, 2, 0.5, "failed", ("synthetic.smoke_one",))
        failed_report = report.__class__(
            **{
                **report.__dict__,
                "ready": False,
                "aggregate_passed": failed.passed,
                "aggregate_duration_seconds": failed.duration_seconds,
                "aggregate_failure_reason": failed.failure_reason,
                "aggregate_failed_modules": failed.failed_modules,
                "blocking_codes": ("aggregate_failed",),
            }
        )
        try:
            closure.write_closure_receipt(
                receipt_directory / "failed.json",
                source_root=source,
                candidate_root=candidate,
                report=failed_report,
            )
        except closure.PreviewClosureError:
            pass
        else:
            raise SystemExit("failed aggregate produced an authoritative closure receipt")

        cli_output = io.StringIO()
        with (
            mock.patch.object(closure, "_run_aggregate") as aggregate_runner,
            mock.patch.dict(
                os.environ,
                _closure_environment(anchor=None),
                clear=True,
            ),
            contextlib.redirect_stdout(cli_output),
        ):
            code = closure.main(
                [
                    "--source",
                    os.fspath(source),
                    "--candidate",
                    os.fspath(candidate),
                    "--verify-receipt",
                    os.fspath(receipt),
                ]
            )
        cli_payload = json.loads(cli_output.getvalue())
        if (
            code != 1
            or cli_payload.get("valid") is not False
            or cli_payload.get("self_consistent") is not True
            or cli_payload.get("aggregate_execution_authenticated") is not False
            or aggregate_runner.called
        ):
            raise SystemExit(f"unauthenticated receipt verification overstated proof: {cli_payload}")
        if os.fspath(receipt) in cli_output.getvalue():
            raise SystemExit("receipt verification output leaked its supplied path")

        authenticated_output = io.StringIO()
        with (
            mock.patch.object(closure, "_run_aggregate") as aggregate_runner,
            mock.patch.dict(
                os.environ,
                _closure_environment(),
                clear=True,
            ),
            contextlib.redirect_stdout(authenticated_output),
        ):
            authenticated_code = closure.main(
                [
                    "--source",
                    os.fspath(source),
                    "--candidate",
                    os.fspath(candidate),
                    "--verify-receipt",
                    os.fspath(receipt),
                ]
            )
        authenticated_payload = json.loads(authenticated_output.getvalue())
        if (
            authenticated_code != 0
            or authenticated_payload.get("valid") is not True
            or authenticated_payload.get("self_consistent") is not True
            or authenticated_payload.get("aggregate_execution_authenticated") is not True
            or aggregate_runner.called
        ):
            raise SystemExit(
                f"externally anchored receipt did not authenticate: {authenticated_payload}"
            )

        wrong_digest_output = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                _closure_environment(anchor="0" * 64),
                clear=True,
            ),
            contextlib.redirect_stdout(wrong_digest_output),
        ):
            wrong_digest_code = closure.main(
                [
                    "--source",
                    os.fspath(source),
                    "--candidate",
                    os.fspath(candidate),
                    "--verify-receipt",
                    os.fspath(receipt),
                ]
            )
        wrong_digest_payload = json.loads(wrong_digest_output.getvalue())
        if (
            wrong_digest_code != 1
            or wrong_digest_payload.get("valid") is not False
            or wrong_digest_payload.get("self_consistent") is not True
            or wrong_digest_payload.get("aggregate_execution_authenticated") is not False
        ):
            raise SystemExit("wrong external digest did not fail authentication")

        malformed_anchor_output = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                _closure_environment(anchor="é"),
                clear=True,
            ),
            contextlib.redirect_stdout(malformed_anchor_output),
        ):
            malformed_anchor_code = closure.main(
                [
                    "--source",
                    os.fspath(source),
                    "--candidate",
                    os.fspath(candidate),
                    "--verify-receipt",
                    os.fspath(receipt),
                ]
            )
        malformed_anchor_payload = json.loads(malformed_anchor_output.getvalue())
        if (
            malformed_anchor_code != 2
            or malformed_anchor_payload.get("blocking_codes")
            != ["configuration_unsafe"]
            or "é" in malformed_anchor_output.getvalue()
        ):
            raise SystemExit("malformed receipt anchor did not fail content-free")

        outcome_unknown_output = io.StringIO()
        with (
            mock.patch.object(closure, "evaluate_preview_closure", return_value=report),
            mock.patch.object(
                closure,
                "write_closure_receipt",
                side_effect=closure.ClosureReceiptOutcomeUnknown(
                    "synthetic outcome unknown"
                ),
            ),
            contextlib.redirect_stdout(outcome_unknown_output),
        ):
            outcome_unknown_code = closure.main(
                [
                    "--source",
                    os.fspath(source),
                    "--candidate",
                    os.fspath(candidate),
                    "--run-aggregate",
                    "--receipt-output",
                    os.fspath(receipt_directory / "unknown-cli.json"),
                ]
            )
        outcome_unknown_payload = json.loads(outcome_unknown_output.getvalue())
        if (
            outcome_unknown_code != 3
            or outcome_unknown_payload.get("receipt_outcome_unknown") is not True
            or outcome_unknown_payload.get("ready") is not False
        ):
            raise SystemExit("receipt cleanup uncertainty was not explicit at the CLI")

        cli_receipt = receipt_directory / "cli-created.json"
        cli_create_output = io.StringIO()
        with (
            mock.patch.object(closure, "_run_aggregate", return_value=aggregate),
            contextlib.redirect_stdout(cli_create_output),
        ):
            create_code = closure.main(
                [
                    "--source",
                    os.fspath(source),
                    "--candidate",
                    os.fspath(candidate),
                    "--run-aggregate",
                    "--receipt-output",
                    os.fspath(cli_receipt),
                ]
            )
        create_payload = json.loads(cli_create_output.getvalue())
        if (
            create_code != 0
            or create_payload.get("closure_receipt_written") is not True
            or "aggregate_execution_verifier" in create_payload
            or "closure_receipt_anchor" in create_payload
            or not cli_receipt.is_file()
        ):
            raise SystemExit(f"CLI did not create a ready closure receipt: {create_payload}")
        if any(
            hidden in cli_create_output.getvalue()
            for hidden in (os.fspath(source), os.fspath(candidate), os.fspath(cli_receipt))
        ):
            raise SystemExit("closure receipt creation output leaked a supplied path")
        if "d" * 64 in cli_create_output.getvalue() or "d" * 64 in cli_receipt.read_text(
            encoding="ascii"
        ):
            raise SystemExit("closure receipt workflow exposed its authentication anchor")

        invalid_output = io.StringIO()
        with contextlib.redirect_stdout(invalid_output):
            invalid_code = closure.main(
                [
                    "--source",
                    os.fspath(source),
                    "--candidate",
                    os.fspath(candidate),
                    "--receipt-output",
                    os.fspath(receipt_directory / "not-run.json"),
                ]
            )
        invalid_payload = json.loads(invalid_output.getvalue())
        if invalid_code != 2 or invalid_payload.get("ready") is not False:
            raise SystemExit("receipt output without aggregate did not fail closed")


def _run_all_tests() -> None:
    test_runbook_anchor_creation_is_fresh_exclusive_and_no_follow()
    test_private_deny_review_is_required_before_aggregate()
    test_closure_git_children_do_not_inherit_private_runtime_values()
    test_deferred_or_disabled_recovery_rejects_supplied_inputs()
    test_pending_proofs_block_without_running_aggregate_or_leaking_paths()
    test_passed_or_policy_allowed_proofs_can_close_with_bound_aggregate()
    test_recovery_attestation_required_only_for_a_passed_lane()
    test_valid_recovery_pass_requires_candidate_binding()
    # The sanitized candidate intentionally omits the private LaunchAgent
    # templates needed to construct synthetic active contracts. The private
    # source checkout runs the full finalized-receipt integration; the marked
    # candidate retains the remaining closure and deferred-recovery coverage.
    if not candidate_module.valid_public_candidate_marker(
        Path(__file__).resolve().parents[2]
    ):
        test_finalized_network_and_reboot_receipts_close_only_when_fully_bound()
    test_recovery_receipt_hash_rejects_unsafe_permissions_and_symlink_parents()
    test_status_only_connector_pass_cannot_close()
    test_connector_attestations_reject_missing_tampered_mismatched_and_unknown()
    test_proof_state_change_after_candidate_requires_rebuild()
    test_recovery_source_binding_ignores_only_evidence_documents()
    test_matching_status_and_digest_edit_after_candidate_still_requires_rebuild()
    test_required_gate_edit_after_candidate_requires_rebuild()
    test_selected_source_drift_and_unchecked_gate_block()
    test_unknown_or_failed_disposition_blocks_until_explicitly_disabled()
    test_required_and_connector_lanes_cannot_be_closed_by_deferral()
    test_structured_status_blocks_reject_shape_errors_and_ignore_prose()
    test_candidate_tamper_before_or_during_aggregate_blocks()
    test_candidate_path_aba_still_executes_anchored_tree()
    test_source_head_is_rebound_after_aggregate()
    test_aggregate_process_gets_only_bounded_noncredential_environment()
    test_aggregate_failure_diagnostics_are_bounded_and_allowlisted()
    test_process_group_cleanup_terminates_descendants()
    test_aggregate_capture_cap_and_closed_pipe_descendant_cleanup()
    test_unexpected_aggregate_communication_error_cleans_group()
    test_external_closure_receipt_is_strict_bound_and_read_only_verifiable()
    test_cli_configuration_failure_is_path_free()
    print("V3 preview closure smoke passed")


def main() -> None:
    with mock.patch.dict(os.environ, _closure_environment(), clear=True):
        _run_all_tests()


if __name__ == "__main__":
    main()
