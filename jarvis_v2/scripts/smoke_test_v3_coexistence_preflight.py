"""Synthetic adversarial smoke for the content-free coexistence preflight."""

from __future__ import annotations

import io
import json
import os
import plistlib
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.automations import telegram_offset_state
from jarvis_v2.scripts import v3_active_launchagent_contracts as builder
from jarvis_v2.scripts import v3_coexistence_preflight as preflight


TOKEN_A = "12345:" + "A" * 32
TOKEN_B = "67890:" + "B" * 32
OWNER = "4827001"
OFFSET = 918273645


def _write_env(path: Path, values: list[tuple[str, str]]) -> Path:
    path.write_text("".join(f"{key}={value}\n" for key, value in values), encoding="utf-8")
    path.chmod(0o600)
    return path


def _state(path: Path, *, offset: int = OFFSET, raw: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(raw if raw is not None else json.dumps({"offset": offset}), encoding="utf-8")
    path.chmod(0o600)


def _v3_state(
    path: Path,
    *,
    offset: int = OFFSET,
    token: str = TOKEN_B,
    owner: str = OWNER,
    raw: bytes | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = raw
    if payload is None:
        payload = telegram_offset_state._canonical(
            {
                "identity_sha256": telegram_offset_state._identity(token, owner),
                "offset": offset,
                "revision": 1,
                "version": telegram_offset_state.STATE_VERSION,
            }
        )
    path.write_bytes(payload)
    path.chmod(0o600)


def _fixture(
    root: Path,
    *,
    v2_token: str = TOKEN_A,
    v3_token: str = TOKEN_B,
    v2_owner: str = OWNER,
    v3_owner: str = OWNER,
    v2_port: str = "8765",
    v3_port: str = "8766",
    v2_data: Path | None = None,
    v3_data: Path | None = None,
    v2_db: Path | None = None,
    v3_db: Path | None = None,
    v2_state: Path | None = None,
    v3_state: Path | None = None,
    extra_v2: list[tuple[str, str]] | None = None,
    extra_v3: list[tuple[str, str]] | None = None,
    prepare_state_parents: bool = True,
) -> tuple[Path, Path, dict[str, Path]]:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    v2_data = v2_data or root / "v2-data"
    v3_data = v3_data or root / "v3-data"
    v2_db = v2_db or v2_data / "jarvis.sqlite"
    v3_db = v3_db or v3_data / "jarvis.sqlite"
    v2_state = v2_state or v2_data / "telegram.json"
    v3_state = v3_state or v3_data / "telegram.json"
    if prepare_state_parents:
        for state_path in (v2_state, v3_state):
            if state_path.is_absolute():
                state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                state_path.parent.chmod(0o700)
    values_v2 = [
        ("JARVIS_DATA_DIR", os.fspath(v2_data)),
        ("JARVIS_DB_PATH", os.fspath(v2_db)),
        ("JARVIS_TELEGRAM_STATE", os.fspath(v2_state)),
        ("JARVIS_STATUS_PORT", v2_port),
        ("TELEGRAM_BOT_TOKEN", v2_token),
        ("JARVIS_OWNER_TELEGRAM", v2_owner),
    ]
    values_v3 = [
        ("JARVIS_DATA_DIR", os.fspath(v3_data)),
        ("JARVIS_DB_PATH", os.fspath(v3_db)),
        ("JARVIS_TELEGRAM_STATE", os.fspath(v3_state)),
        ("JARVIS_STATUS_PORT", v3_port),
        ("TELEGRAM_BOT_TOKEN", v3_token),
        ("JARVIS_OWNER_TELEGRAM", v3_owner),
    ]
    values_v2.extend(extra_v2 or [])
    values_v3.extend(extra_v3 or [])
    v2_env = _write_env(root / "v2.env", values_v2)
    v3_env = _write_env(root / "v3.env", values_v3)
    template_root = root / "synthetic-templates"
    template_root.mkdir(mode=0o700)
    with patch.object(builder, "PROJECT_ROOT", template_root):
        for contract in builder.SERVICE_CONTRACTS:
            (template_root / contract.filename).write_bytes(
                plistlib.dumps(
                    builder._expected_template(contract),
                    fmt=plistlib.FMT_XML,
                    sort_keys=True,
                )
            )
        output = root / "active-contracts"
        builder.build_active_launchagent_contracts(
            env_file=v3_env,
            output_dir=output,
            template_root=template_root,
        )
    return v2_env, output / builder.MANIFEST_NAME, {
        "v2_data": v2_data,
        "v3_data": v3_data,
        "v2_db": v2_db,
        "v3_db": v3_db,
        "v2_state": v2_state,
        "v3_state": v3_state,
        "v3_env": v3_env,
        "template_root": template_root,
    }


def _inspect(v2_env: Path, manifest: Path, loaded: bool | None = True) -> dict[str, object]:
    template_root = manifest.parent.parent / "synthetic-templates"
    with patch.object(builder, "PROJECT_ROOT", template_root):
        return preflight.inspect_coexistence(
            v2_env,
            manifest,
            launchd_reader=lambda: loaded,
        )


def _require_blocker(report: dict[str, object], reason: str) -> None:
    if reason not in report["blockers"] or report["ready_for_parallel_activation"] is not False:
        raise SystemExit(f"coexistence refusal missing bounded blocker: {reason}")


def _require_warning(report: dict[str, object], warning: str) -> None:
    if warning not in report["warnings"]:
        raise SystemExit(f"coexistence report missing bounded warning: {warning}")


def _snapshot(root: Path) -> dict[str, tuple[bytes, int, int]]:
    return {
        os.fspath(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_fixture_is_candidate_safe_without_checked_in_service_templates() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-candidate-", dir="/private/tmp") as temp:
        root = Path(temp)
        candidate_root = root / "candidate-without-service-plists"
        candidate_root.mkdir(mode=0o700)
        with patch.object(builder, "PROJECT_ROOT", candidate_root):
            v2_env, manifest, paths = _fixture(root / "case")
            report = _inspect(v2_env, manifest, True)
        if report["ready_for_parallel_activation"] is not True:
            raise SystemExit("synthetic coexistence fixture was not candidate-safe")
        if tuple(candidate_root.iterdir()):
            raise SystemExit("coexistence fixture created a checked-in-style service artifact")
        template_root = paths["template_root"]
        if template_root.parent != root / "case" or set(path.name for path in template_root.iterdir()) != {
            contract.filename for contract in builder.SERVICE_CONTRACTS
        }:
            raise SystemExit("coexistence fixture did not bound its synthetic templates to temp")


def test_distinct_configuration_is_parallel_ready_content_free_and_read_only() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-ready-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_env, manifest, paths = _fixture(root)
        _state(paths["v2_state"])
        _v3_state(paths["v3_state"], offset=OFFSET + 1)
        before = _snapshot(root)
        report = _inspect(v2_env, manifest, True)
        after = _snapshot(root)
        if before != after:
            raise SystemExit("coexistence preflight mutated a synthetic runtime artifact")
        if (
            report["ready_for_parallel_activation"] is not True
            or report["blockers"] != []
            or report["warnings"] != []
        ):
            raise SystemExit("fully isolated V2/V3 configuration was not parallel-ready")
        expected_checks = {
            "manifest", "scheduler", "v2_environment", "v3_environment",
            "persistent_paths", "dashboard_port", "telegram_token", "telegram_owner",
            "telegram_state_relation", "v2_telegram_state", "v3_telegram_state",
            "v2_telegram_service",
        }
        if set(report["checks"]) != expected_checks:
            raise SystemExit("coexistence report check schema drifted")
        for field in (
            "values_included", "paths_included", "hashes_included", "mutates_state",
            "launchctl_control_performed",
        ):
            if report[field] is not False:
                raise SystemExit(f"coexistence boundary flag drifted: {field}")
        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (TOKEN_A, TOKEN_B, OWNER, str(root), str(OFFSET), "sha256"):
            if forbidden in rendered:
                raise SystemExit("coexistence report leaked a value, path, offset, or hash")


def test_path_port_database_and_state_collisions_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-collisions-", dir="/private/tmp") as temp:
        root = Path(temp)
        shared = root / "shared"
        v2_env, manifest, _ = _fixture(root / "data", v2_data=shared, v3_data=shared / "v3")
        _require_blocker(_inspect(v2_env, manifest), "persistent_path_overlap")

        same_db = root / "same.sqlite"
        v2_env, manifest, _ = _fixture(root / "db", v2_db=same_db, v3_db=same_db)
        _require_blocker(_inspect(v2_env, manifest), "persistent_path_overlap")

        v2_env, manifest, _ = _fixture(root / "port", v3_port="8765")
        _require_blocker(_inspect(v2_env, manifest), "dashboard_port_collision")

        same_state = root / "same-state.json"
        v2_env, manifest, _ = _fixture(
            root / "state", v2_state=same_state, v3_state=same_state
        )
        _require_blocker(_inspect(v2_env, manifest), "telegram_state_overlap")


def test_alias_and_inode_reuse_are_rejected() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-alias-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_target = root / "v2-target"
        v2_target.mkdir(mode=0o700)
        alias = root / "v3-alias"
        alias.symlink_to(v2_target, target_is_directory=True)
        case = root / "alias-case"
        case.mkdir()
        v2_env, manifest, _ = _fixture(case, v2_data=v2_target, v3_data=alias)
        _require_blocker(_inspect(v2_env, manifest), "persistent_path_invalid")

        hard_case = root / "hardlink-case"
        hard_case.mkdir()
        v2_env, manifest, paths = _fixture(hard_case)
        _state(paths["v2_state"])
        paths["v3_state"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.link(paths["v2_state"], paths["v3_state"])
        report = _inspect(v2_env, manifest)
        if not ({"persistent_path_invalid", "telegram_state_overlap"} & set(report["blockers"])):
            raise SystemExit("existing-file inode reuse did not fail closed")


def test_shared_transport_requires_active_block_or_serial_handoff() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-token-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_env, manifest, _ = _fixture(root, v3_token=TOKEN_A)
        active = _inspect(v2_env, manifest, True)
        _require_blocker(active, "shared_telegram_transport_active")
        if active["serial_handoff_required"] is not False:
            raise SystemExit("active shared transport was mislabeled as a safe handoff")
        stopped = _inspect(v2_env, manifest, False)
        _require_blocker(stopped, "shared_telegram_transport_requires_serial_handoff")
        if stopped["serial_handoff_required"] is not True:
            raise SystemExit("stopped shared transport omitted serial-handoff requirement")
        unknown = _inspect(v2_env, manifest, None)
        _require_blocker(unknown, "v2_telegram_load_state_unknown")


def test_legacy_v2_readable_state_warns_without_masking_shared_transport() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-legacy-state-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_env, manifest, paths = _fixture(root / "distinct")
        _state(paths["v2_state"])
        paths["v2_state"].chmod(0o644)
        distinct = _inspect(v2_env, manifest, True)
        if distinct["ready_for_parallel_activation"] is not True or distinct["blockers"]:
            raise SystemExit("safe legacy-readable V2 state incorrectly blocked parallel readiness")
        if distinct["checks"]["v2_telegram_state"] != "valid_legacy_readable":
            raise SystemExit("safe legacy-readable V2 state lost its generation-specific status")
        if distinct["checks"]["v3_telegram_state"] != "fresh":
            raise SystemExit("fresh strict V3 state lost its generation-specific status")
        if distinct["checks"]["telegram_state_relation"] != "distinct":
            raise SystemExit("distinct Telegram state paths lost their relation status")
        _require_warning(distinct, "v2_telegram_state_legacy_permissions")

        v2_env, manifest, paths = _fixture(root / "shared", v3_token=TOKEN_A)
        _state(paths["v2_state"])
        paths["v2_state"].chmod(0o644)
        shared = _inspect(v2_env, manifest, True)
        _require_warning(shared, "v2_telegram_state_legacy_permissions")
        _require_blocker(shared, "shared_telegram_transport_active")
        if shared["checks"]["v2_telegram_state"] != "valid_legacy_readable":
            raise SystemExit("shared transport masked the legacy-readable state evidence")


def test_unsafe_v2_and_non_owner_only_v3_state_modes_still_block() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-state-modes-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_env, manifest, paths = _fixture(root / "v2")
        _state(paths["v2_state"])
        paths["v2_state"].chmod(0o666)
        report = _inspect(v2_env, manifest)
        _require_blocker(report, "v2_telegram_state_invalid")
        if report["checks"]["v2_telegram_state"] != "invalid" or report["warnings"]:
            raise SystemExit("group/other-writable V2 state was mislabeled as legacy-readable")

        v2_env, manifest, paths = _fixture(root / "v3")
        _v3_state(paths["v3_state"])
        paths["v3_state"].chmod(0o644)
        report = _inspect(v2_env, manifest)
        _require_blocker(report, "v3_telegram_state_invalid")
        if report["checks"]["v3_telegram_state"] != "invalid":
            raise SystemExit("non-owner-only V3 state passed strict custody")


def test_v3_legacy_unbound_and_wrong_identity_states_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-binding-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_env, manifest, paths = _fixture(root / "legacy")
        _state(paths["v3_state"])
        legacy = _inspect(v2_env, manifest)
        _require_blocker(legacy, "v3_telegram_state_invalid")
        if legacy["checks"]["v3_telegram_state"] != "invalid":
            raise SystemExit("unversioned V3 state was accepted")

        v2_env, manifest, paths = _fixture(root / "wrong-identity")
        _v3_state(paths["v3_state"], token=TOKEN_A)
        mismatched = _inspect(v2_env, manifest)
        _require_blocker(mismatched, "v3_telegram_state_invalid")
        if mismatched["checks"]["v3_telegram_state"] != "invalid":
            raise SystemExit("V3 state bound to another bot was accepted")

        rendered = json.dumps(mismatched, sort_keys=True)
        for forbidden in (TOKEN_A, TOKEN_B, OWNER, str(OFFSET), "sha256"):
            if forbidden in rendered:
                raise SystemExit("V3 identity mismatch report leaked private evidence")


def test_missing_or_unsafe_v3_state_parent_is_not_fresh() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-parent-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_env, manifest, paths = _fixture(
            root / "missing",
            prepare_state_parents=False,
        )
        paths["v2_state"].parent.mkdir(parents=True, mode=0o700)
        missing = _inspect(v2_env, manifest)
        _require_blocker(missing, "v3_telegram_state_invalid")
        if missing["checks"]["v3_telegram_state"] != "invalid":
            raise SystemExit("missing V3 state parent was mislabeled as fresh")

        v2_env, manifest, paths = _fixture(root / "unsafe")
        paths["v3_state"].parent.chmod(0o755)
        try:
            unsafe = _inspect(v2_env, manifest)
        finally:
            paths["v3_state"].parent.chmod(0o700)
        _require_blocker(unsafe, "v3_telegram_state_invalid")
        if unsafe["checks"]["v3_telegram_state"] != "invalid":
            raise SystemExit("unsafe V3 state parent was mislabeled as fresh")


def test_duplicate_invalid_identity_port_and_state_inputs_are_bounded() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-invalid-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_env, manifest, _ = _fixture(
            root / "duplicate", extra_v2=[("JARVIS_STATUS_PORT", "9001")]
        )
        _require_blocker(_inspect(v2_env, manifest), "v2_environment_invalid")

        v2_env, manifest, _ = _fixture(root / "identity", v2_token="invalid", v3_port="70000")
        report = _inspect(v2_env, manifest)
        _require_blocker(report, "telegram_token_invalid")
        _require_blocker(report, "dashboard_port_invalid")

        v2_env, manifest, _ = _fixture(
            root / "missing", v2_token="", v3_owner=""
        )
        report = _inspect(v2_env, manifest)
        _require_blocker(report, "telegram_token_missing")
        _require_blocker(report, "telegram_owner_missing")

        v2_env, manifest, _ = _fixture(root / "owner", v2_owner="not-an-owner")
        _require_blocker(_inspect(v2_env, manifest), "telegram_owner_invalid")

        v2_env, manifest, _ = _fixture(root / "path", v2_data=Path("relative-data"))
        report = _inspect(v2_env, manifest)
        _require_blocker(report, "persistent_path_invalid")
        _require_blocker(report, "telegram_state_path_invalid")

        v2_env, manifest, paths = _fixture(root / "state")
        _state(paths["v2_state"], raw='{"offset":1,"offset":2}')
        _require_blocker(_inspect(v2_env, manifest), "v2_telegram_state_invalid")
        rendered = json.dumps(_inspect(v2_env, manifest), sort_keys=True)
        if str(paths["v2_state"]) in rendered or "offset" in rendered:
            raise SystemExit("malformed Telegram state leaked content or location")


def test_manifest_and_scheduler_drift_are_bounded() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-manifest-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_env, manifest, _ = _fixture(root)
        artifact = manifest.parent / builder.SERVICE_CONTRACTS[0].filename
        payload = plistlib.loads(artifact.read_bytes())
        payload["EnvironmentVariables"][builder.SCHEDULER_ENABLE_KEY] = "1"
        artifact.write_bytes(plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True))
        artifact.chmod(0o600)
        report = _inspect(v2_env, manifest)
        _require_blocker(report, "manifest_invalid")
        if report["checks"]["scheduler"] != "unknown":
            raise SystemExit("invalid manifest falsely attested scheduler disablement")


def test_cli_emits_one_strict_json_document_without_launchctl_control() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-coexistence-cli-", dir="/private/tmp") as temp:
        root = Path(temp)
        v2_env, manifest, _ = _fixture(root)
        completed = type("Observed", (), {
            "returncode": 0,
            "stdout": "state = running\n",
            "stderr": "",
        })()
        output = io.StringIO()
        template_root = manifest.parent.parent / "synthetic-templates"
        with patch.object(builder, "PROJECT_ROOT", template_root):
            with patch.object(preflight.recovery, "_run_observer", return_value=completed) as observer:
                with redirect_stdout(output):
                    code = preflight.main(
                        [
                            "--v2-env",
                            os.fspath(v2_env),
                            "--active-contract-manifest",
                            os.fspath(manifest),
                        ]
                    )
        report = json.loads(output.getvalue())
        if code != 0 or report["ready_for_parallel_activation"] is not True:
            raise SystemExit("CLI did not emit a successful strict JSON preflight")
        command = observer.call_args.args[0]
        if command[:2] != ["/bin/launchctl", "print"] or len(command) != 3:
            raise SystemExit("CLI observer used a launchctl control or unbounded command")
        if any(verb in command for verb in ("bootstrap", "bootout", "kickstart", "enable", "disable")):
            raise SystemExit("CLI observer attempted launchctl control")

        secret_argument = os.fspath(root / "PRIVATE-ARGUMENT-DO-NOT-RENDER")
        output = io.StringIO()
        with redirect_stdout(output):
            code = preflight.main(["--unknown", secret_argument])
        if code != 2 or secret_argument in output.getvalue():
            raise SystemExit("CLI argument refusal leaked caller-supplied path material")
        if json.loads(output.getvalue())["blockers"] != ["arguments_invalid"]:
            raise SystemExit("CLI argument refusal was not a strict bounded JSON report")


def main() -> int:
    test_fixture_is_candidate_safe_without_checked_in_service_templates()
    test_distinct_configuration_is_parallel_ready_content_free_and_read_only()
    test_path_port_database_and_state_collisions_fail_closed()
    test_alias_and_inode_reuse_are_rejected()
    test_shared_transport_requires_active_block_or_serial_handoff()
    test_legacy_v2_readable_state_warns_without_masking_shared_transport()
    test_unsafe_v2_and_non_owner_only_v3_state_modes_still_block()
    test_v3_legacy_unbound_and_wrong_identity_states_fail_closed()
    test_missing_or_unsafe_v3_state_parent_is_not_fresh()
    test_duplicate_invalid_identity_port_and_state_inputs_are_bounded()
    test_manifest_and_scheduler_drift_are_bounded()
    test_cli_emits_one_strict_json_document_without_launchctl_control()
    print("v3 coexistence preflight smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
