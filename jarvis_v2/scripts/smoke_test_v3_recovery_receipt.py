from __future__ import annotations

import io
import hashlib
import json
import os
import plistlib
import shutil
import signal
import stat
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.scripts import v3_active_launchagent_contracts as builder
from jarvis_v2.scripts import v3_recovery_receipt as receipt
from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile


COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
SERVICES = tuple(sorted(contract.name for contract in builder.SERVICE_CONTRACTS))
UP = {name: True for name in SERVICES}
DUMMY_ENV_PATH = "/private/tmp/jarvis-v3-recovery-receipt.env"
IS_PUBLIC_CANDIDATE = structural_public_candidate_profile(builder.PROJECT_ROOT) is not None


def _expect_error(code: str, callback, *args, **kwargs) -> None:
    try:
        callback(*args, **kwargs)
    except receipt.RecoveryReceiptError as exc:
        if exc.code != code:
            raise SystemExit(f"wrong recovery receipt error: {exc.code}; expected {code}")
    else:
        raise SystemExit(f"recovery receipt did not fail closed: {code}")


def _observation(
    evidence: receipt.ContractEvidence,
    *,
    boot: str = "boot-a",
    network: bool = True,
    commit: str = COMMIT_A,
    scheduler: bool = True,
    services: dict[str, bool] | None = None,
    env_digest: str | None = None,
) -> receipt.SystemObservation:
    return receipt.SystemObservation(
        source_commit=commit,
        manifest_digest=evidence.manifest_digest,
        contract_set_digest=evidence.contract_set_digest,
        env_content_digest=env_digest or evidence.env_content_digest,
        expected_services=evidence.expected_services,
        boot_identity=boot,
        network_reachable=network,
        scheduler_disabled=scheduler,
        service_states=dict(services or UP),
    )


def _dummy_evidence() -> receipt.ContractEvidence:
    artifacts: list[tuple[str, bytes]] = []
    for contract in builder.SERVICE_CONTRACTS:
        payload = builder._active_payload(builder._expected_template(contract), DUMMY_ENV_PATH)
        artifacts.append(
            (
                contract.filename,
                plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True),
            )
        )
    return receipt.ContractEvidence(
        manifest_digest="c" * 64,
        contract_set_digest="d" * 64,
        expected_services=SERVICES,
        artifacts=tuple(artifacts),
        env_content_digest="e" * 64,
    )


def _fixture(temp: str) -> tuple[Path, Path, receipt.ContractEvidence]:
    root = Path(temp)
    env_path = root / "private.env"
    env_path.write_text("JARVIS_TEST_SENTINEL=private\n", encoding="utf-8")
    env_path.chmod(0o600)
    output = root / "active"
    builder.build_active_launchagent_contracts(
        env_file=env_path,
        output_dir=output,
    )
    installed = root / "LaunchAgents"
    installed.mkdir(mode=0o700)
    for contract in builder.SERVICE_CONTRACTS:
        shutil.copyfile(output / contract.filename, installed / contract.filename)
        (installed / contract.filename).chmod(0o600)
    evidence = receipt.validate_active_contract_manifest(output / builder.MANIFEST_NAME)
    return output / builder.MANIFEST_NAME, installed, evidence


def test_sanitized_candidate_omits_private_templates_and_fails_closed() -> None:
    with TemporaryDirectory(dir="/private/tmp") as temp:
        root = Path(temp)
        env_path = root / "private.env"
        env_path.write_text("", encoding="utf-8")
        env_path.chmod(0o600)
        templates = root / "templates"
        templates.mkdir()
        for contract in builder.SERVICE_CONTRACTS:
            (templates / contract.filename).write_bytes(
                plistlib.dumps(builder._expected_template(contract))
            )
        output = root / "active"
        builder.build_active_launchagent_contracts(
            env_file=env_path,
            output_dir=output,
            template_root=templates,
        )
        _expect_error(
            "active_contract_provenance_invalid",
            receipt.validate_active_contract_manifest,
            output / builder.MANIFEST_NAME,
        )


def _state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_sealed(path: Path, payload: dict) -> None:
    value = dict(payload)
    value["integrity_digest"] = receipt._integrity_digest(value)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    path.chmod(0o600)


def test_network_and_reboot_transitions_are_observed_not_asserted() -> None:
    with TemporaryDirectory(dir="/private/tmp") as temp:
        manifest = Path(temp) / "unused-manifest.json"
        evidence = _dummy_evidence()
        network_path = Path(temp) / "network.json"
        observations = iter(
            [
                _observation(evidence),
                _observation(evidence, network=False),
                _observation(evidence, network=True),
            ]
        )
        with patch.object(receipt, "observe_system", side_effect=lambda _path: next(observations)):
            prepared = receipt.prepare_receipt(
                network_path, kind="network", active_contract_manifest=manifest
            )
            down = receipt.record_observation(
                network_path, phase="network-down", active_contract_manifest=manifest
            )
            final = receipt.record_observation(
                network_path, phase="network-up", active_contract_manifest=manifest
            )
        if prepared["proof_state"] != "prepared" or down["observation_count"] != 2:
            raise SystemExit("network recovery preparation lost pending state")
        if final["proof_state"] != "finalized" or final["verdict"] != "proven":
            raise SystemExit("observed network recovery did not finalize")
        if final["source_commit"] != COMMIT_A or final["contract_set_sha256"] != evidence.contract_set_digest:
            raise SystemExit("validated summary omitted its source/contract binding")
        if str(network_path) in json.dumps(final):
            raise SystemExit("receipt summary leaked a local path")
        if stat.S_IMODE(network_path.stat().st_mode) != 0o600:
            raise SystemExit("receipt is not owner-only")
        _expect_error(
            "receipt_replay_blocked",
            receipt.record_observation,
            network_path,
            phase="network-up",
            active_contract_manifest=manifest,
        )

        reboot_path = Path(temp) / "reboot.json"
        with patch.object(receipt, "observe_system", return_value=_observation(evidence, boot="before")):
            receipt.prepare_receipt(reboot_path, kind="reboot", active_contract_manifest=manifest)
        with patch.object(receipt, "observe_system", return_value=_observation(evidence, boot="before")):
            _expect_error(
                "wrong_boot_transition",
                receipt.record_observation,
                reboot_path,
                phase="postboot",
                active_contract_manifest=manifest,
            )
        with patch.object(receipt, "observe_system", return_value=_observation(evidence, boot="after")):
            reboot = receipt.record_observation(
                reboot_path, phase="postboot", active_contract_manifest=manifest
            )
        if reboot["verdict"] != "proven":
            raise SystemExit("observed reboot transition did not finalize")


def test_callers_cannot_forge_evidence_or_bindings() -> None:
    with TemporaryDirectory(dir="/private/tmp") as temp:
        manifest = Path(temp) / "unused-manifest.json"
        evidence = _dummy_evidence()
        path = Path(temp) / "forgery.json"
        forged_keywords = {
            "boot_session_identity": "forged-boot",
            "network_reachable": True,
            "scheduler_disabled": True,
            "service_states": UP,
            "source_commit": COMMIT_A,
            "active_contract_manifest_digest": evidence.contract_set_digest,
        }
        try:
            receipt.prepare_receipt(
                path,
                kind="network",
                active_contract_manifest=manifest,
                **forged_keywords,
            )
        except TypeError:
            pass
        else:
            raise SystemExit("Python API still accepts caller-authored proof facts")
        if path.exists():
            raise SystemExit("rejected forged preparation created state")

        output = io.StringIO()
        with redirect_stdout(output):
            code = receipt.main(
                [
                    "prepare", "--state", str(path), "--kind", "network",
                    "--active-contract-manifest", str(manifest),
                    "--source-commit", COMMIT_A,
                ]
            )
        if code == 0 or json.loads(output.getvalue()) != {"error": "invalid_cli_arguments", "ok": False}:
            raise SystemExit("CLI still accepts caller-authored source evidence")

        with patch.object(receipt, "observe_system", return_value=_observation(evidence)):
            receipt.prepare_receipt(path, kind="network", active_contract_manifest=manifest)
        mismatches = [
            (_observation(evidence, commit=COMMIT_B), "source_commit_mismatch"),
            (
                receipt.SystemObservation(
                    source_commit=COMMIT_A,
                    manifest_digest="d" * 64,
                    contract_set_digest=evidence.contract_set_digest,
                    env_content_digest=evidence.env_content_digest,
                    expected_services=evidence.expected_services,
                    boot_identity="boot-a",
                    network_reachable=False,
                    scheduler_disabled=True,
                    service_states=UP,
                ),
                "active_contract_manifest_mismatch",
            ),
            (_observation(evidence, network=False, env_digest="f" * 64), "environment_content_mismatch"),
        ]
        for observed, expected_code in mismatches:
            with patch.object(receipt, "observe_system", return_value=observed):
                _expect_error(
                    expected_code,
                    receipt.record_observation,
                    path,
                    phase="network-down",
                    active_contract_manifest=manifest,
                )


def test_manifest_provenance_installed_custody_and_digests_fail_closed() -> None:
    with TemporaryDirectory(dir="/private/tmp") as temp:
        manifest, installed, evidence = _fixture(temp)
        for contract in builder.SERVICE_CONTRACTS:
            payload = plistlib.loads((manifest.parent / contract.filename).read_bytes())
            if payload["EnvironmentVariables"].get(builder.SCHEDULER_ENABLE_KEY) != "0":
                raise SystemExit("active contract did not explicitly disable scheduler execution")
        with patch.object(receipt, "_installed_launchagent_root", return_value=installed):
            receipt._verify_installed_contracts(evidence)

            target = installed / builder.SERVICE_CONTRACTS[0].filename
            target.chmod(0o644)
            _expect_error("unsafe_installed_contract", receipt._verify_installed_contracts, evidence)
            target.chmod(0o600)

            original = target.read_bytes()
            target.write_bytes(original + b"\n")
            target.chmod(0o600)
            _expect_error("installed_contract_mismatch", receipt._verify_installed_contracts, evidence)
            target.write_bytes(original)
            target.chmod(0o600)

        log_root = Path(temp) / "account-home" / builder.LOG_DIRECTORY_RELATIVE
        log_root.mkdir(mode=builder.LOG_DIRECTORY_MODE, parents=True)
        for contract in builder.SERVICE_CONTRACTS:
            log = log_root / contract.log_filename
            log.write_text("", encoding="utf-8")
            log.chmod(builder.LOG_FILE_MODE)
        with patch.object(builder, "_account_log_directory", return_value=log_root):
            receipt._verify_log_custody()
            first_log = log_root / builder.SERVICE_CONTRACTS[0].log_filename
            first_log.chmod(0o644)
            _expect_error("log_custody_invalid", receipt._verify_log_custody)
            first_log.chmod(builder.LOG_FILE_MODE)
            first_log.unlink()
            outside = Path(temp) / "outside.log"
            outside.write_text("", encoding="utf-8")
            outside.chmod(builder.LOG_FILE_MODE)
            first_log.symlink_to(outside)
            _expect_error("log_custody_invalid", receipt._verify_log_custody)
            first_log.unlink()
            first_log.write_text("", encoding="utf-8")
            first_log.chmod(builder.LOG_FILE_MODE)
            log_root.chmod(0o755)
            _expect_error("log_custody_invalid", receipt._verify_log_custody)
            log_root.chmod(builder.LOG_DIRECTORY_MODE)

        original_manifest_text = manifest.read_text(encoding="utf-8")
        duplicate_manifest = original_manifest_text.replace(
            "{", '{"schema_version":1,', 1
        )
        manifest.write_text(duplicate_manifest, encoding="utf-8")
        manifest.chmod(0o600)
        _expect_error("manifest_invalid", receipt.validate_active_contract_manifest, manifest)
        manifest.write_text(original_manifest_text.rstrip("\n") + " \n", encoding="utf-8")
        manifest.chmod(0o600)
        _expect_error("manifest_invalid", receipt.validate_active_contract_manifest, manifest)
        manifest.write_text(original_manifest_text, encoding="utf-8")
        manifest.chmod(0o600)

        env_path = Path(temp) / "private.env"
        original_env = env_path.read_bytes()
        env_path.write_bytes(original_env + b"JARVIS_CHANGED=1\n")
        env_path.chmod(0o600)
        changed_evidence = receipt.validate_active_contract_manifest(manifest)
        if changed_evidence.env_content_digest == evidence.env_content_digest:
            raise SystemExit("selected environment content change was not bound")
        env_path.write_bytes(original_env)
        env_path.chmod(0o600)

        manifest_data = json.loads(original_manifest_text)
        manifest_data["contract_set_sha256"] = "0" * 64
        manifest.write_text(json.dumps(manifest_data, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        manifest.chmod(0o600)
        _expect_error("contract_set_digest_mismatch", receipt.validate_active_contract_manifest, manifest)

        manifest.write_text(original_manifest_text, encoding="utf-8")
        manifest.chmod(0o600)
        manifest_data = json.loads(original_manifest_text)
        manifest_data["log_contract"]["directory_mode"] = "0755"
        manifest.write_text(
            json.dumps(manifest_data, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        manifest.chmod(0o600)
        _expect_error("manifest_invalid", receipt.validate_active_contract_manifest, manifest)

        manifest.write_text(original_manifest_text, encoding="utf-8")
        manifest.chmod(0o600)
        target_contract = builder.SERVICE_CONTRACTS[0]
        target_path = manifest.parent / target_contract.filename
        original_contract = target_path.read_bytes()
        forged = plistlib.loads(original_contract)
        forged["EnvironmentVariables"][builder.SCHEDULER_ENABLE_KEY] = "1"
        forged_bytes = plistlib.dumps(forged, fmt=plistlib.FMT_XML, sort_keys=True)
        target_path.write_bytes(forged_bytes)
        target_path.chmod(0o600)
        forged_manifest = json.loads(original_manifest_text)
        for item in forged_manifest["artifacts"]:
            if item["name"] == target_contract.filename:
                item["sha256"] = hashlib.sha256(forged_bytes).hexdigest()
        forged_manifest["contract_set_sha256"] = hashlib.sha256(
            json.dumps(
                forged_manifest["artifacts"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        manifest.write_text(
            json.dumps(forged_manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        manifest.chmod(0o600)
        _expect_error("active_contract_invalid", receipt.validate_active_contract_manifest, manifest)


def test_installed_launchagent_root_ignores_home_environment() -> None:
    with TemporaryDirectory(dir="/private/tmp") as temp:
        account_home = Path(temp) / "account-home"
        account_home.mkdir(mode=0o700)
        account_record = type("AccountRecord", (), {"pw_dir": os.fspath(account_home)})()
        with (
            patch.object(receipt.pwd, "getpwuid", return_value=account_record),
            patch.dict(os.environ, {"HOME": "/private/tmp/attacker-home"}),
        ):
            if receipt._installed_launchagent_root() != account_home / "Library" / "LaunchAgents":
                raise SystemExit("installed LaunchAgent root trusted HOME instead of account database")
        relative_record = type("AccountRecord", (), {"pw_dir": "relative-home"})()
        with patch.object(receipt.pwd, "getpwuid", return_value=relative_record):
            _expect_error(
                "installed_contract_root_invalid",
                receipt._installed_launchagent_root,
            )
        linked_home = Path(temp) / "linked-home"
        linked_home.symlink_to(account_home, target_is_directory=True)
        linked_record = type("AccountRecord", (), {"pw_dir": os.fspath(linked_home)})()
        with patch.object(receipt.pwd, "getpwuid", return_value=linked_record):
            _expect_error(
                "installed_contract_root_invalid",
                receipt._installed_launchagent_root,
            )


def test_state_integrity_order_and_unknown_observations_fail_closed() -> None:
    with TemporaryDirectory(dir="/private/tmp") as temp:
        manifest = Path(temp) / "unused-manifest.json"
        evidence = _dummy_evidence()
        path = Path(temp) / "state.json"
        with patch.object(receipt, "observe_system", return_value=_observation(evidence)):
            receipt.prepare_receipt(path, kind="network", active_contract_manifest=manifest)
            _expect_error(
                "observation_out_of_order",
                receipt.record_observation,
                path,
                phase="network-up",
                active_contract_manifest=manifest,
            )
        unhealthy = dict(UP)
        unhealthy[SERVICES[0]] = False
        with patch.object(receipt, "observe_system", return_value=_observation(evidence, network=False, services=unhealthy)):
            _expect_error(
                "expected_service_process_not_present",
                receipt.record_observation,
                path,
                phase="network-down",
                active_contract_manifest=manifest,
            )
        original = _state(path)
        tampered = dict(original)
        tampered["source_commit"] = COMMIT_B
        path.write_bytes(receipt._canonical_json_document(tampered))
        path.chmod(0o600)
        with patch.object(receipt, "observe_system", return_value=_observation(evidence)):
            _expect_error(
                "state_integrity_failed",
                receipt.inspect_receipt,
                path,
                active_contract_manifest=manifest,
            )
        contradiction = json.loads(json.dumps(original))
        contradiction["observations"][0]["network_reachable"] = False
        _rewrite_sealed(path, contradiction)
        with patch.object(receipt, "observe_system", return_value=_observation(evidence)):
            _expect_error(
                "state_corrupt",
                receipt.inspect_receipt,
                path,
                active_contract_manifest=manifest,
            )


def test_canonical_receipts_and_future_timestamps_fail_closed() -> None:
    with TemporaryDirectory(dir="/private/tmp") as temp:
        manifest = Path(temp) / "unused-manifest.json"
        evidence = _dummy_evidence()
        path = Path(temp) / "canonical.json"
        observations = iter(
            [
                _observation(evidence),
                _observation(evidence, network=False),
                _observation(evidence, network=True),
            ]
        )
        with patch.object(receipt, "observe_system", side_effect=lambda _path: next(observations)):
            receipt.prepare_receipt(path, kind="network", active_contract_manifest=manifest)
            receipt.record_observation(path, phase="network-down", active_contract_manifest=manifest)
            receipt.record_observation(path, phase="network-up", active_contract_manifest=manifest)
        canonical = path.read_bytes()
        duplicate = canonical.replace(b"{", b'{"version":3,', 1)
        path.write_bytes(duplicate)
        path.chmod(0o600)
        with patch.object(receipt, "observe_system", return_value=_observation(evidence)):
            _expect_error(
                "state_corrupt",
                receipt.inspect_receipt,
                path,
                active_contract_manifest=manifest,
            )
        path.write_bytes(canonical[:-1] + b" \n")
        path.chmod(0o600)
        with patch.object(receipt, "observe_system", return_value=_observation(evidence)):
            _expect_error(
                "state_corrupt",
                receipt.inspect_receipt,
                path,
                active_contract_manifest=manifest,
            )
        path.write_bytes(canonical)
        future = _state(path)
        future_stamp = "2999-01-01T00:00:00Z"
        future["created_at"] = future_stamp
        future["finalized_at"] = future_stamp
        for observation in future["observations"]:
            observation["observed_at"] = future_stamp
        _rewrite_sealed(path, future)
        with patch.object(receipt, "observe_system", return_value=_observation(evidence)):
            _expect_error(
                "state_timestamp_in_future",
                receipt.inspect_receipt,
                path,
                active_contract_manifest=manifest,
            )


def test_os_observers_are_bounded_read_only_and_fail_closed() -> None:
    completed = subprocess.CompletedProcess
    evidence = _dummy_evidence()

    def rendered_environment(
        contract: builder.ServiceContract,
        *,
        updates: dict[str, str | None] | None = None,
    ) -> tuple[str, ...]:
        payload = builder._active_payload(builder._expected_template(contract), DUMMY_ENV_PATH)
        environment = dict(payload["EnvironmentVariables"])
        for key, value in (updates or {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        return tuple(f"{key} => {value}" for key, value in sorted(environment.items()))

    def launch_state(
        contract: builder.ServiceContract,
        pid: int,
        *,
        arguments: tuple[str, ...] | None = None,
        working_directory: str | None = None,
        environment_lines: tuple[str, ...] | None = None,
        program: str = "/opt/homebrew/bin/python3",
        include_arguments: bool = True,
        include_environment: bool = True,
        include_program: bool = True,
        suffix: str = "",
    ) -> subprocess.CompletedProcess[str]:
        argv = arguments or ("/opt/homebrew/bin/python3", "-m", contract.module)
        env_lines = environment_lines or rendered_environment(contract)
        output = (
            "state = running\n"
            f"pid = {pid}\n"
            + (f"program = {program}\n" if include_program else "")
            + (
                "arguments = {\n"
                + "".join(f"  {argument}\n" for argument in argv)
                + "}\n"
                if include_arguments
                else ""
            )
            + f"working directory = {working_directory or builder.PROJECT_ROOT}\n"
            + (
                "environment = {\n"
                + "".join(f"  {line}\n" for line in env_lines)
                + "}\n"
                if include_environment
                else ""
            )
            + suffix
        )
        return completed([], 0, output, "")

    with patch.object(receipt, "_run_observer", return_value=completed([], 0, "{ sec = 1, usec = 2 }\n", "")) as run:
        if receipt.observe_boot_identity() != "{ sec = 1, usec = 2 }":
            raise SystemExit("boot observer lost stable sysctl identity")
        if run.call_args.args[0] != ["/usr/sbin/sysctl", "-n", "kern.boottime"]:
            raise SystemExit("boot observer command drifted")
    with patch.object(receipt, "_run_observer", return_value=completed([], 0, "not-a-boot-id\n", "")):
        _expect_error("boot_identity_unknown", receipt.observe_boot_identity)
    with patch.object(receipt, "_run_observer", return_value=completed([], 0, "Reachable\n", "")):
        if receipt.observe_network_reachability() is not True:
            raise SystemExit("network observer lost reachable result")
    with patch.object(receipt, "_run_observer", return_value=completed([], 0, "Not Reachable\n", "")):
        if receipt.observe_network_reachability() is not False:
            raise SystemExit("network observer lost unreachable result")
    with patch.object(receipt, "_run_observer", return_value=completed([], 0, "Maybe\n", "")):
        _expect_error("network_outcome_unknown", receipt.observe_network_reachability)
    with patch.object(receipt, "_run_observer", return_value=completed([], 0, "Reachable\nNot Reachable\n", "")):
        _expect_error("network_outcome_unknown", receipt.observe_network_reachability)

    service_results: list[subprocess.CompletedProcess[str]] = []
    for index, contract in enumerate(builder.SERVICE_CONTRACTS, start=101):
        loaded = launch_state(
            contract,
            index,
            environment_lines=(
                *rendered_environment(contract),
                f"XPC_SERVICE_NAME => {contract.label}",
            ),
        )
        process = completed([], 0, f"/opt/homebrew/bin/python3 -m {contract.module}\n", "")
        service_results.extend((loaded, process, loaded, process))
    with patch.object(receipt, "_run_observer", side_effect=service_results):
        if receipt.observe_service_states(evidence) != UP:
            raise SystemExit("service observer did not derive exact running set")
    first_contract = builder.SERVICE_CONTRACTS[0]
    mismatch_results = [
        launch_state(first_contract, 101),
        completed([], 0, "/opt/homebrew/bin/python3 -m wrong.module\n", ""),
    ]
    with patch.object(receipt, "_run_observer", side_effect=mismatch_results):
        _expect_error("service_process_mismatch", receipt.observe_service_states, evidence)
    for bad_argv in (
        "/opt/homebrew/bin/python3evil -m jarvis_v2.scripts.run_telegram_control\n",
        "/opt/homebrew/bin/python3 -m jarvis_v2.scripts.run_telegram_control --extra\n",
    ):
        with patch.object(
            receipt,
            "_run_observer",
            side_effect=[
                launch_state(first_contract, 101),
                completed([], 0, bad_argv, ""),
            ],
        ):
            _expect_error("service_process_mismatch", receipt.observe_service_states, evidence)

    valid_environment = rendered_environment(first_contract)
    contract_mismatches = (
        launch_state(first_contract, 101, include_program=False),
        launch_state(first_contract, 101, program="/opt/homebrew/bin/python3-stale"),
        launch_state(
            first_contract,
            101,
            suffix="program = /opt/homebrew/bin/python3\n",
        ),
        launch_state(first_contract, 101, include_arguments=False),
        launch_state(first_contract, 101, arguments=("/opt/homebrew/bin/python3", "-m")),
        launch_state(first_contract, 101, working_directory="/private/tmp/stale-v3"),
        launch_state(first_contract, 101, include_environment=False),
        launch_state(
            first_contract,
            101,
            environment_lines=rendered_environment(
                first_contract,
                updates={builder.DAEMON_ENABLE_KEY: None},
            ),
        ),
        launch_state(
            first_contract,
            101,
            environment_lines=(
                f"{builder.DAEMON_ENABLE_KEY} => 1",
                *valid_environment,
            ),
        ),
        launch_state(
            first_contract,
            101,
            environment_lines=rendered_environment(
                first_contract,
                updates={builder.DAEMON_ENABLE_KEY: "0"},
            ),
        ),
        launch_state(
            first_contract,
            101,
            environment_lines=rendered_environment(
                first_contract,
                updates={builder.ENV_FILE_KEY: "/private/tmp/stale.env"},
            ),
        ),
        launch_state(
            first_contract,
            101,
            environment_lines=rendered_environment(
                first_contract,
                updates={"PYTHONPATH": "/private/tmp/stale-source"},
            ),
        ),
        launch_state(
            first_contract,
            101,
            environment_lines=(
                *valid_environment,
                "PYTHONHOME => /private/tmp/stale-python-home",
            ),
        ),
        launch_state(
            first_contract,
            101,
            environment_lines=(
                *valid_environment,
                "XPC_SERVICE_NAME => stale.injected.value",
            ),
        ),
        launch_state(
            first_contract,
            101,
            environment_lines=(
                f"{builder.ENV_FILE_KEY} => {DUMMY_ENV_PATH}",
                *valid_environment,
            ),
        ),
        launch_state(
            first_contract,
            101,
            environment_lines=rendered_environment(
                first_contract,
                updates={builder.SCHEDULER_ENABLE_KEY: "1"},
            ),
        ),
        launch_state(
            first_contract,
            101,
            environment_lines=rendered_environment(
                first_contract,
                updates={builder.SCHEDULER_ENABLE_KEY: None},
            ),
        ),
        launch_state(
            first_contract,
            101,
            suffix=(
                "arguments = {\n"
                "  /opt/homebrew/bin/python3\n"
                "  -m\n"
                f"  {first_contract.module}\n"
                "}\n"
            ),
        ),
        launch_state(first_contract, 101, suffix=f"working directory = {builder.PROJECT_ROOT}\n"),
        launch_state(
            first_contract,
            101,
            suffix=(
                "environment = {\n"
                + "".join(f"  {line}\n" for line in valid_environment)
                + "}\n"
            ),
        ),
    )
    for stale_loaded_contract in contract_mismatches:
        with patch.object(receipt, "_run_observer", return_value=stale_loaded_contract):
            _expect_error(
                "service_contract_mismatch",
                receipt.observe_service_states,
                evidence,
            )

    changed_loaded_contract = launch_state(
        first_contract,
        101,
        environment_lines=rendered_environment(
            first_contract,
            updates={builder.ENV_FILE_KEY: "/private/tmp/changed.env"},
        ),
    )
    with patch.object(
        receipt,
        "_run_observer",
        side_effect=[
            launch_state(first_contract, 101),
            completed([], 0, f"/opt/homebrew/bin/python3 -m {first_contract.module}\n", ""),
            changed_loaded_contract,
        ],
    ):
        _expect_error("service_contract_changed", receipt.observe_service_states, evidence)

    injected_environment = (
        *rendered_environment(first_contract),
        f"XPC_SERVICE_NAME => {first_contract.label}",
    )
    changed_injected_environment = (
        *rendered_environment(first_contract),
        "XPC_SERVICE_NAME => stale.injected.value",
    )
    with patch.object(
        receipt,
        "_run_observer",
        side_effect=[
            launch_state(
                first_contract,
                101,
                environment_lines=injected_environment,
            ),
            completed([], 0, f"/opt/homebrew/bin/python3 -m {first_contract.module}\n", ""),
            launch_state(
                first_contract,
                101,
                environment_lines=changed_injected_environment,
            ),
        ],
    ):
        _expect_error("service_contract_changed", receipt.observe_service_states, evidence)

    with patch.object(
        receipt,
        "_run_observer",
        side_effect=[
            launch_state(first_contract, 101),
            completed([], 0, f"/opt/homebrew/bin/python3 -m {first_contract.module}\n", ""),
            launch_state(first_contract, 102),
        ],
    ):
        _expect_error("service_process_changed", receipt.observe_service_states, evidence)

    duplicate_artifact_evidence = receipt.ContractEvidence(
        manifest_digest=evidence.manifest_digest,
        contract_set_digest=evidence.contract_set_digest,
        expected_services=evidence.expected_services,
        artifacts=evidence.artifacts + (evidence.artifacts[0],),
        env_content_digest=evidence.env_content_digest,
    )
    _expect_error(
        "service_contract_evidence_mismatch",
        receipt.observe_service_states,
        duplicate_artifact_evidence,
    )
    missing_artifact_evidence = receipt.ContractEvidence(
        manifest_digest=evidence.manifest_digest,
        contract_set_digest=evidence.contract_set_digest,
        expected_services=evidence.expected_services,
        artifacts=evidence.artifacts[:-1],
        env_content_digest=evidence.env_content_digest,
    )
    _expect_error(
        "service_contract_evidence_mismatch",
        receipt.observe_service_states,
        missing_artifact_evidence,
    )
    stale_artifacts = list(evidence.artifacts)
    stale_payload = plistlib.loads(stale_artifacts[0][1])
    stale_payload["EnvironmentVariables"][builder.ENV_FILE_KEY] = "/private/tmp/stale.env"
    stale_artifacts[0] = (
        stale_artifacts[0][0],
        plistlib.dumps(stale_payload, fmt=plistlib.FMT_XML, sort_keys=True),
    )
    stale_artifact_evidence = receipt.ContractEvidence(
        manifest_digest=evidence.manifest_digest,
        contract_set_digest=evidence.contract_set_digest,
        expected_services=evidence.expected_services,
        artifacts=tuple(stale_artifacts),
        env_content_digest=evidence.env_content_digest,
    )
    with patch.object(receipt, "_run_observer", return_value=launch_state(first_contract, 101)):
        _expect_error(
            "service_contract_mismatch",
            receipt.observe_service_states,
            stale_artifact_evidence,
        )
    with patch.object(
        receipt,
        "_run_observer",
        return_value=completed([], 113, "", "Could not find service"),
    ):
        if receipt.observe_scheduler_disabled() is not True:
            raise SystemExit("scheduler observer did not prove absent service")
    with patch.object(receipt, "_run_observer", return_value=completed([], 1, "", "private error")):
        _expect_error("scheduler_outcome_unknown", receipt.observe_scheduler_disabled)
    with patch.object(
        receipt,
        "_run_observer",
        return_value=completed([], 1, "state = running", "Could not find service"),
    ):
        _expect_error("scheduler_outcome_unknown", receipt.observe_scheduler_disabled)

    bounded = receipt._run_observer(
        ["/usr/bin/printf", "ok"], "observer_failed", max_output_bytes=2
    )
    if bounded.stdout != "ok" or bounded.returncode != 0:
        raise SystemExit("bounded observer lost exact small output")
    _expect_error(
        "observer_failed",
        receipt._run_observer,
        ["/usr/bin/printf", "toolarge"],
        "observer_failed",
        max_output_bytes=3,
    )
    _expect_error(
        "observer_decode_failed",
        receipt._run_observer,
        [sys.executable, "-c", "import os;os.write(1,b'\\xff')"],
        "observer_decode_failed",
        max_output_bytes=8,
    )
    _expect_error(
        "observer_failed",
        receipt._run_observer,
        ["/bin/sleep", "1"],
        "observer_failed",
        timeout=0.01,
        max_output_bytes=32,
    )

    selector_probe = receipt.selectors.DefaultSelector()
    selector_class = type(selector_probe)
    selector_probe.close()
    real_terminate_observer_group = receipt._terminate_observer_group
    with (
        patch.object(
            selector_class,
            "select",
            side_effect=RuntimeError("synthetic selector failure"),
        ),
        patch.object(
            receipt,
            "_terminate_observer_group",
            wraps=real_terminate_observer_group,
        ) as terminate_observer_group,
    ):
        try:
            receipt._run_observer(
                ["/bin/sleep", "30"],
                "unexpected_observer_failure",
                timeout=1.0,
                max_output_bytes=32,
            )
        except RuntimeError as exc:
            if str(exc) != "synthetic selector failure":
                raise
        else:
            raise SystemExit("unexpected observer failure did not propagate")
        if terminate_observer_group.call_count != 1:
            raise SystemExit("unexpected observer failure skipped process-group cleanup")

    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "time.sleep(30)"
    )
    parent_code = (
        "import signal,subprocess,sys,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        "print('ready',flush=True);"
        "time.sleep(30)"
    )
    real_popen = subprocess.Popen
    real_killpg = os.killpg
    spawned: list[subprocess.Popen[bytes]] = []
    group_signals: list[int] = []
    start_new_session_values: list[bool] = []

    def tracking_popen(*args, **kwargs):
        start_new_session_values.append(kwargs.get("start_new_session") is True)
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    def tracking_killpg(process_group_id: int, sent_signal: int) -> None:
        if sent_signal:
            group_signals.append(sent_signal)
        real_killpg(process_group_id, sent_signal)

    started = time.monotonic()
    with (
        patch.object(receipt.subprocess, "Popen", side_effect=tracking_popen),
        patch.object(receipt.os, "killpg", side_effect=tracking_killpg),
    ):
        _expect_error(
            "observer_descendant_timeout",
            receipt._run_observer,
            [sys.executable, "-c", parent_code],
            "observer_descendant_timeout",
            timeout=0.5,
            max_output_bytes=32,
        )
    elapsed = time.monotonic() - started
    if elapsed > 3.0:
        raise SystemExit("observer descendant cleanup exceeded its bounded deadline")
    if (
        len(spawned) != 1
        or start_new_session_values != [True]
        or spawned[0].poll() is None
        or signal.SIGTERM not in group_signals
        or signal.SIGKILL not in group_signals
    ):
        raise SystemExit("observer did not terminate and reap its isolated process group")
    try:
        real_killpg(spawned[0].pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise SystemExit("observer descendant process group remained after cleanup")

    with TemporaryDirectory(
        prefix="jarvis-recovery-success-descendant-", dir="/private/tmp"
    ) as temp:
        descendant_pid_path = Path(temp) / "descendant.pid"
        residual_child_code = (
            "import signal,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "time.sleep(30)"
        )
        residual_parent_code = (
            "import subprocess,sys;from pathlib import Path;"
            f"p=subprocess.Popen([sys.executable,'-c',{residual_child_code!r}],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
            f"Path({os.fspath(descendant_pid_path)!r}).write_text(str(p.pid));"
            "print('ok',flush=True)"
        )
        _expect_error(
            "observer_residual_descendant",
            receipt._run_observer,
            [sys.executable, "-c", residual_parent_code],
            "observer_residual_descendant",
            timeout=5.0,
            max_output_bytes=32,
        )
        if not descendant_pid_path.is_file():
            raise SystemExit("successful observer residual descendant did not publish its pid")
        descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise SystemExit("observer normal-success path left a descendant running")

    deadline_process = real_popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline_process.wait(timeout=2.0)
    monotonic_values = iter((100.0, 100.5))

    def expired_monotonic() -> float:
        return next(monotonic_values, 102.0)

    with (
        patch.object(receipt.subprocess, "Popen", return_value=deadline_process),
        patch.object(receipt.time, "monotonic", side_effect=expired_monotonic),
    ):
        _expect_error(
            "observer_hard_deadline",
            receipt._run_observer,
            ["synthetic-observer"],
            "observer_hard_deadline",
            timeout=1.0,
            max_output_bytes=32,
        )
    if (
        deadline_process.stdout is None
        or not deadline_process.stdout.closed
        or deadline_process.stderr is None
        or not deadline_process.stderr.closed
    ):
        raise SystemExit("observer hard-deadline failure left pipe descriptors open")


def test_clean_head_and_durable_write_contracts() -> None:
    clean = [
        subprocess.CompletedProcess([], 0, COMMIT_A + "\n", ""),
        subprocess.CompletedProcess([], 0, "", ""),
    ]
    with patch.object(receipt, "_run_observer", side_effect=clean):
        if receipt.active_source_commit() != COMMIT_A:
            raise SystemExit("clean HEAD observer drifted")
    dirty = [
        subprocess.CompletedProcess([], 0, COMMIT_A + "\n", ""),
        subprocess.CompletedProcess([], 0, " M private.file\n", ""),
    ]
    with patch.object(receipt, "_run_observer", side_effect=dirty):
        _expect_error("source_worktree_not_clean", receipt.active_source_commit)

    evidence = _dummy_evidence()
    source_state = {"dirty": False, "checks": 0}

    def checked_source() -> str:
        source_state["checks"] += 1
        if source_state["dirty"]:
            raise receipt.RecoveryReceiptError("source_worktree_not_clean")
        return COMMIT_A

    def boot_then_dirty_source() -> str:
        source_state["dirty"] = True
        return "{ sec = 1, usec = 2 }"

    with (
        patch.object(receipt, "active_source_commit", side_effect=checked_source),
        patch.object(receipt, "validate_active_contract_manifest", return_value=evidence),
        patch.object(receipt, "_verify_installed_contracts"),
        patch.object(receipt, "_verify_log_custody"),
        patch.object(receipt, "observe_service_states", return_value=UP),
        patch.object(receipt, "observe_scheduler_disabled", return_value=True),
        patch.object(receipt, "observe_network_reachability", return_value=True),
        patch.object(receipt, "observe_boot_identity", side_effect=boot_then_dirty_source),
    ):
        _expect_error(
            "source_worktree_not_clean",
            receipt.observe_system,
            "/unused-manifest",
        )
    if source_state["checks"] != 2:
        raise SystemExit("source stability was not checked after boot identity observation")

    with TemporaryDirectory(dir="/private/tmp") as temp:
        manifest = Path(temp) / "unused-manifest.json"
        evidence = _dummy_evidence()
        path = Path(temp) / "durable.json"
        real_fsync = os.fsync
        calls: list[int] = []

        def tracking_fsync(fd: int) -> None:
            calls.append(fd)
            real_fsync(fd)

        with patch.object(receipt, "observe_system", return_value=_observation(evidence)), patch.object(
            receipt.os, "fsync", side_effect=tracking_fsync
        ):
            receipt.prepare_receipt(path, kind="network", active_contract_manifest=manifest)
        if len(calls) < 2:
            raise SystemExit("receipt write did not fsync both file and parent directory")


def test_state_path_create_and_timestamp_adversaries_fail_closed() -> None:
    with TemporaryDirectory(dir="/private/tmp") as temp:
        root = Path(temp)
        real_parent = root / "real"
        real_parent.mkdir(mode=0o700)
        linked_parent = root / "linked"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        _expect_error(
            "unsafe_state_parent",
            receipt._validate_state_path,
            linked_parent / "receipt.json",
        )

        manifest = root / "unused-manifest.json"
        evidence = _dummy_evidence()
        path = root / "race.json"
        sentinel = b"racer-won\n"
        def racing_link(source, destination, **kwargs):
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=kwargs["dst_dir_fd"],
            )
            try:
                os.write(descriptor, sentinel)
            finally:
                os.close(descriptor)
            raise FileExistsError("simulated create race")

        observed = _observation(evidence)
        with patch.object(receipt, "observe_system", return_value=observed), patch.object(
            receipt.os, "link", side_effect=racing_link
        ):
            _expect_error(
                "state_already_exists",
                receipt.prepare_receipt,
                path,
                kind="network",
                active_contract_manifest=manifest,
            )
        if path.read_bytes() != sentinel:
            raise SystemExit("atomic create overwrote a racing target")
        path.unlink()

        with patch.object(receipt, "observe_system", return_value=observed):
            receipt.prepare_receipt(path, kind="network", active_contract_manifest=manifest)
        original = _state(path)
        mutations = []
        bad_receipt = dict(original)
        bad_receipt["receipt_id"] = "not-an-id"
        mutations.append(bad_receipt)
        bad_created = dict(original)
        bad_created["created_at"] = "not-a-time"
        mutations.append(bad_created)
        backwards = json.loads(json.dumps(original))
        backwards["observations"][0]["observed_at"] = "2000-01-01T00:00:00Z"
        mutations.append(backwards)
        bad_finalized = dict(original)
        bad_finalized["finalized_at"] = "2026-01-01T00:00:00Z"
        mutations.append(bad_finalized)
        for mutation in mutations:
            _rewrite_sealed(path, mutation)
            with patch.object(receipt, "observe_system", return_value=observed):
                _expect_error(
                    "state_corrupt",
                    receipt.inspect_receipt,
                    path,
                    active_contract_manifest=manifest,
                )
        _rewrite_sealed(path, original)

        replacement = root / "replacement.json"
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o600)

        def replace_state_observer(_manifest):
            os.replace(replacement, path)
            return _observation(evidence, network=False)

        with patch.object(receipt, "observe_system", side_effect=replace_state_observer):
            _expect_error(
                "state_changed",
                receipt.record_observation,
                path,
                phase="network-down",
                active_contract_manifest=manifest,
            )
        if len(_state(path)["observations"]) != 1:
            raise SystemExit("state replacement race was overwritten")

        late_replacement = root / "late-replacement.json"
        late_replacement.write_bytes(path.read_bytes())
        late_replacement.chmod(0o600)
        real_write_state = receipt._write_state

        def replace_after_outer_check(parent_fd, name, payload, **kwargs):
            os.replace(late_replacement, path)
            return real_write_state(parent_fd, name, payload, **kwargs)

        with (
            patch.object(receipt, "observe_system", return_value=_observation(evidence, network=False)),
            patch.object(receipt, "_write_state", side_effect=replace_after_outer_check),
        ):
            _expect_error(
                "state_changed",
                receipt.record_observation,
                path,
                phase="network-down",
                active_contract_manifest=manifest,
            )
        if len(_state(path)["observations"]) != 1:
            raise SystemExit("late state replacement race was overwritten")

        swap_parent = root / "swap"
        swap_parent.mkdir(mode=0o700)
        swap_path = swap_parent / "receipt.json"
        moved_parent = root / "swap-moved"

        def swap_observer(_manifest):
            swap_parent.rename(moved_parent)
            swap_parent.mkdir(mode=0o700)
            return observed

        with patch.object(receipt, "observe_system", side_effect=swap_observer):
            _expect_error(
                "state_parent_changed",
                receipt.prepare_receipt,
                swap_path,
                kind="network",
                active_contract_manifest=manifest,
            )
        if swap_path.exists() or (moved_parent / "receipt.json").exists():
            raise SystemExit("parent-swap refusal wrote a receipt")

        late_parent = root / "late-swap"
        late_parent.mkdir(mode=0o700)
        late_path = late_parent / "receipt.json"
        late_moved_parent = root / "late-swap-moved"

        def swap_after_outer_check(parent_fd, name, payload, **kwargs):
            late_parent.rename(late_moved_parent)
            late_parent.mkdir(mode=0o700)
            return real_write_state(parent_fd, name, payload, **kwargs)

        with (
            patch.object(receipt, "observe_system", return_value=observed),
            patch.object(receipt, "_write_state", side_effect=swap_after_outer_check),
        ):
            _expect_error(
                "state_parent_changed",
                receipt.prepare_receipt,
                late_path,
                kind="network",
                active_contract_manifest=manifest,
            )
        if late_path.exists() or not (late_moved_parent / "receipt.json").exists():
            raise SystemExit("late parent swap returned success or lost the bounded failed write")

        committed_path = root / "committed-race.json"
        committed_replacement = root / "committed-replacement.json"
        committed_replacement.write_bytes(path.read_bytes())
        committed_replacement.chmod(0o600)

        def replace_committed_target(parent_fd, name, payload, **kwargs):
            committed_identity = real_write_state(parent_fd, name, payload, **kwargs)
            os.replace(committed_replacement, committed_path)
            return committed_identity

        with (
            patch.object(receipt, "observe_system", return_value=observed),
            patch.object(receipt, "_write_state", side_effect=replace_committed_target),
        ):
            _expect_error(
                "state_changed",
                receipt.prepare_receipt,
                committed_path,
                kind="network",
                active_contract_manifest=manifest,
            )


def test_cli_success_is_content_path_and_secret_free() -> None:
    with TemporaryDirectory(dir="/private/tmp") as temp:
        manifest = Path(temp) / "unused-manifest.json"
        evidence = _dummy_evidence()
        path = Path(temp) / "private-path-sentinel.json"
        output = io.StringIO()
        with patch.object(receipt, "observe_system", return_value=_observation(evidence)), redirect_stdout(output):
            code = receipt.main(
                [
                    "prepare", "--state", str(path), "--kind", "network",
                    "--active-contract-manifest", str(manifest),
                ]
            )
        if code != 0:
            raise SystemExit(f"observer-only CLI failed: {output.getvalue()!r}")
        rendered = output.getvalue()
        if str(path) in rendered or str(manifest) in rendered or "private.env" in rendered:
            raise SystemExit("observer-only CLI leaked a path or private selector")
        if evidence.env_content_digest in rendered or "env_content" in rendered:
            raise SystemExit("observer-only CLI leaked the private environment binding")
        payload = json.loads(rendered)
        required_bindings = {
            "source_commit": COMMIT_A,
            "active_contract_manifest_sha256": evidence.manifest_digest,
            "contract_set_sha256": evidence.contract_set_digest,
        }
        if any(payload.get(key) != value for key, value in required_bindings.items()):
            raise SystemExit("observer-only CLI omitted validated evidence bindings")


def test_public_contract_denies_live_actions_and_private_assertions() -> None:
    expected = {
        "changes_network_state": False,
        "reboots_computer": False,
        "starts_or_restarts_services": False,
        "sends_messages": False,
        "approves_actions": False,
        "reads_private_payloads": False,
    }
    if receipt.CONTRACT_MANIFEST.get("boundaries") != expected:
        raise SystemExit("recovery receipt live-action boundary drifted")
    parser_text = Path(receipt.__file__).read_text(encoding="utf-8")
    forbidden_flags = (
        'add_argument("--source-commit"',
        'add_argument("--network-reachable"',
        'add_argument("--scheduler-disabled"',
        'add_argument("--service-state"',
        'add_argument("--expected-service"',
        "stdin.buffer",
        "socket.",
        "networksetup",
        "shutdown",
        "kickstart",
        "bootout",
    )
    if any(marker in parser_text for marker in forbidden_flags):
        raise SystemExit("recovery receipt regained an assertion or mutation primitive")


def main() -> None:
    if IS_PUBLIC_CANDIDATE:
        test_sanitized_candidate_omits_private_templates_and_fails_closed()
        test_network_and_reboot_transitions_are_observed_not_asserted()
        test_callers_cannot_forge_evidence_or_bindings()
        test_installed_launchagent_root_ignores_home_environment()
        test_state_integrity_order_and_unknown_observations_fail_closed()
        test_canonical_receipts_and_future_timestamps_fail_closed()
        test_os_observers_are_bounded_read_only_and_fail_closed()
        test_clean_head_and_durable_write_contracts()
        test_state_path_create_and_timestamp_adversaries_fail_closed()
        test_cli_success_is_content_path_and_secret_free()
        test_public_contract_denies_live_actions_and_private_assertions()
        print("v3 recovery receipt smoke test passed")
        return
    test_network_and_reboot_transitions_are_observed_not_asserted()
    test_callers_cannot_forge_evidence_or_bindings()
    test_installed_launchagent_root_ignores_home_environment()
    test_manifest_provenance_installed_custody_and_digests_fail_closed()
    test_state_integrity_order_and_unknown_observations_fail_closed()
    test_canonical_receipts_and_future_timestamps_fail_closed()
    test_os_observers_are_bounded_read_only_and_fail_closed()
    test_clean_head_and_durable_write_contracts()
    test_state_path_create_and_timestamp_adversaries_fail_closed()
    test_cli_success_is_content_path_and_secret_free()
    test_public_contract_denies_live_actions_and_private_assertions()
    print("v3 recovery receipt smoke test passed")


if __name__ == "__main__":
    main()
