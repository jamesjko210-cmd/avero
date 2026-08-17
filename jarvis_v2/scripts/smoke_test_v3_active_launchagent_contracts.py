"""Adversarial offline smoke for the V3 active LaunchAgent contract builder."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.scripts import v3_active_launchagent_contracts as builder
from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile
from jarvis_v2.scripts.v3_active_launchagent_contracts import (
    DAEMON_ENABLE_KEY,
    ENV_FILE_KEY,
    FILE_MODE,
    MANIFEST_NAME,
    PROJECT_ROOT,
    SCHEDULER_ENABLE_KEY,
    SERVICE_CONTRACTS,
    ContractBuildError,
    build_active_launchagent_contracts,
)


SECRET = "SYNTHETIC_ACTIVE_CONTRACT_SECRET_7391"
IS_PUBLIC_CANDIDATE = structural_public_candidate_profile(PROJECT_ROOT) is not None


def _expect_reason(reason: str, callback) -> None:
    try:
        callback()
    except ContractBuildError as exc:
        if exc.reason_code != reason or str(exc) != reason:
            raise SystemExit(f"contract refusal was not bounded: {exc!r}")
    else:
        raise SystemExit(f"contract builder accepted invalid input: {reason}")


def _private_env(root: Path, *, content: str | None = None) -> Path:
    path = root / "runtime.env"
    path.write_text(content if content is not None else f"TOKEN={SECRET}\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _template_fixture(root: Path) -> Path:
    templates = root / "templates"
    templates.mkdir(parents=True)
    for contract in SERVICE_CONTRACTS:
        shutil.copyfile(PROJECT_ROOT / contract.filename, templates / contract.filename)
    return templates


def test_sanitized_candidate_omits_templates_and_fails_closed() -> None:
    template_paths = tuple(PROJECT_ROOT / contract.filename for contract in SERVICE_CONTRACTS)
    if any(path.exists() for path in template_paths):
        raise SystemExit("sanitized public candidate included a private service plist")
    with TemporaryDirectory(prefix="jarvis-v3-active-candidate-", dir="/private/tmp") as temp:
        root = Path(temp)
        env_file = _private_env(root)
        output = root / "contracts"
        _expect_reason(
            "template_invalid",
            lambda: build_active_launchagent_contracts(
                env_file=env_file,
                output_dir=output,
            ),
        )
        if os.path.lexists(output):
            raise SystemExit("missing candidate templates created an output directory")


def test_valid_build_is_offline_bounded_and_secret_free() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-active-contract-", dir="/private/tmp") as temp:
        root = Path(temp)
        env_file = _private_env(root)
        output = root / "new-contracts"
        fake_home = root / "home"
        live_agents = fake_home / "Library" / "LaunchAgents"
        live_agents.mkdir(parents=True)
        sentinel = live_agents / "keep.txt"
        sentinel.write_text("unchanged", encoding="utf-8")

        manifest = build_active_launchagent_contracts(env_file=env_file, output_dir=output)
        if stat.S_IMODE(output.stat().st_mode) != 0o700:
            raise SystemExit("active contract output directory is not owner-only")
        if manifest.get("scheduler_activation_present") is not False:
            raise SystemExit("active contract manifest implied scheduler activation")
        if manifest.get("artifact_count") != len(SERVICE_CONTRACTS):
            raise SystemExit("active contract manifest count drifted")
        if manifest.get("schema_version") != 2 or manifest.get(
            "log_contract"
        ) != builder._log_contract_manifest():
            raise SystemExit("active contract manifest lost its owner-only log contract")

        manifest_text = json.dumps(manifest, sort_keys=True)
        for forbidden in (
            SECRET,
            os.fspath(env_file),
            os.fspath(output),
            os.fspath(PROJECT_ROOT),
            "EnvironmentVariables",
            DAEMON_ENABLE_KEY,
            ENV_FILE_KEY,
        ):
            if forbidden in manifest_text:
                raise SystemExit("active contract manifest leaked content, path, or secret material")

        listed = {item["name"]: item["sha256"] for item in manifest["artifacts"]}
        if set(listed) != {contract.filename for contract in SERVICE_CONTRACTS}:
            raise SystemExit("active contract manifest artifact names drifted")
        canonical_receipts = json.dumps(
            manifest["artifacts"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if manifest.get("contract_set_sha256") != hashlib.sha256(canonical_receipts).hexdigest():
            raise SystemExit("active contract set digest drifted")
        for contract in SERVICE_CONTRACTS:
            path = output / contract.filename
            raw = path.read_bytes()
            if stat.S_IMODE(path.stat().st_mode) != FILE_MODE:
                raise SystemExit(f"{contract.name} active contract is not owner-only")
            if SECRET.encode() in raw:
                raise SystemExit(f"{contract.name} active contract copied environment content")
            if hashlib.sha256(raw).hexdigest() != listed[contract.filename]:
                raise SystemExit(f"{contract.name} active contract digest drifted")
            payload = plistlib.loads(raw)
            environment = payload.get("EnvironmentVariables")
            if not isinstance(environment, dict):
                raise SystemExit(f"{contract.name} active contract lost its environment")
            if environment.get(DAEMON_ENABLE_KEY) != "1":
                raise SystemExit(f"{contract.name} active contract was not exactly enabled")
            if environment.get(ENV_FILE_KEY) != os.fspath(env_file):
                raise SystemExit(f"{contract.name} active contract lost selected environment custody")
            if environment.get(SCHEDULER_ENABLE_KEY) != "0":
                raise SystemExit(
                    f"{contract.name} active contract lost its explicit scheduler-disable fence"
                )
            if payload.get("StandardOutPath") != contract.log_path or payload.get(
                "StandardErrorPath"
            ) != contract.log_path:
                raise SystemExit(f"{contract.name} active contract log path drifted")
            log_path = Path(contract.log_path)
            if (
                log_path.parent != builder._account_log_directory()
                or log_path.name != contract.log_filename
                or log_path.is_relative_to(Path("/tmp"))
                or log_path.is_relative_to(Path("/private/tmp"))
            ):
                raise SystemExit(f"{contract.name} active contract retained an unsafe log path")

        disk_manifest = (output / MANIFEST_NAME).read_text(encoding="utf-8")
        if json.loads(disk_manifest) != manifest or SECRET in disk_manifest or "/private/" in disk_manifest:
            raise SystemExit("on-disk active contract manifest leaked or drifted")
        if sentinel.read_text(encoding="utf-8") != "unchanged" or tuple(live_agents.iterdir()) != (sentinel,):
            raise SystemExit("offline contract build touched the live service directory")


def test_environment_custody_rejects_malformed_symlink_and_permissions() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-active-env-", dir="/private/tmp") as temp:
        root = Path(temp)
        env_file = _private_env(root)
        alias = root / "runtime-link.env"
        alias.symlink_to(env_file)
        _expect_reason(
            "env_symlink",
            lambda: build_active_launchagent_contracts(
                env_file=alias, output_dir=root / "symlink-output"
            ),
        )
        env_file.chmod(0o640)
        _expect_reason(
            "env_not_owner_only",
            lambda: build_active_launchagent_contracts(
                env_file=env_file, output_dir=root / "perms-output"
            ),
        )
        env_file.chmod(0o600)
        with patch(
            "jarvis_v2.scripts.v3_active_launchagent_contracts.os.geteuid",
            return_value=os.geteuid() + 1,
        ):
            _expect_reason(
                "env_wrong_owner",
                lambda: build_active_launchagent_contracts(
                    env_file=env_file, output_dir=root / "owner-output"
                ),
            )
        _expect_reason(
            "env_not_absolute",
            lambda: build_active_launchagent_contracts(
                env_file=Path("runtime.env"), output_dir=root / "relative-env-output"
            ),
        )
        directory_env = root / "not-a-file"
        directory_env.mkdir()
        _expect_reason(
            "env_not_regular",
            lambda: build_active_launchagent_contracts(
                env_file=directory_env, output_dir=root / "directory-env-output"
            ),
        )


def test_log_directory_uses_canonical_account_home_without_creating_it() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-log-contract-", dir="/private/tmp") as temp:
        root = Path(temp)
        account_home = root / "account-home"
        account_home.mkdir(mode=0o700)
        account_record = type("AccountRecord", (), {"pw_dir": os.fspath(account_home)})()
        expected = account_home / builder.LOG_DIRECTORY_RELATIVE
        with (
            patch.object(builder.pwd, "getpwuid", return_value=account_record),
            patch.dict(os.environ, {"HOME": os.fspath(root / "attacker-home")}),
        ):
            if builder._account_log_directory() != expected:
                raise SystemExit("log contract trusted HOME instead of the account database")
            for contract in SERVICE_CONTRACTS:
                if Path(contract.log_path) != expected / contract.log_filename:
                    raise SystemExit("service log path escaped the canonical account log directory")
        if expected.exists():
            raise SystemExit("offline log contract lookup created the future log directory")

        relative_record = type("AccountRecord", (), {"pw_dir": "relative-home"})()
        with patch.object(builder.pwd, "getpwuid", return_value=relative_record):
            _expect_reason("account_home_invalid", builder._account_log_directory)

        linked_home = root / "linked-home"
        linked_home.symlink_to(account_home, target_is_directory=True)
        linked_record = type("AccountRecord", (), {"pw_dir": os.fspath(linked_home)})()
        with patch.object(builder.pwd, "getpwuid", return_value=linked_record):
            _expect_reason("account_home_invalid", builder._account_log_directory)


def test_output_must_be_new_and_contained() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-active-output-", dir="/private/tmp") as temp:
        root = Path(temp)
        env_file = _private_env(root)
        existing = root / "existing"
        existing.mkdir()
        _expect_reason(
            "output_exists",
            lambda: build_active_launchagent_contracts(env_file=env_file, output_dir=existing),
        )
        _expect_reason(
            "output_not_absolute",
            lambda: build_active_launchagent_contracts(
                env_file=env_file, output_dir=Path("relative-output")
            ),
        )
        _expect_reason(
            "output_outside_private_tmp",
            lambda: build_active_launchagent_contracts(
                env_file=env_file, output_dir=Path("/var/tmp/jarvis-v3-active-contracts")
            ),
        )
        target = root / "target"
        target.mkdir()
        parent_link = root / "parent-link"
        parent_link.symlink_to(target, target_is_directory=True)
        _expect_reason(
            "output_parent_invalid",
            lambda: build_active_launchagent_contracts(
                env_file=env_file, output_dir=parent_link / "contracts"
            ),
        )


def test_malformed_and_scheduler_templates_fail_before_output() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-active-template-", dir="/private/tmp") as temp:
        root = Path(temp)
        env_file = _private_env(root)
        templates = _template_fixture(root)
        malformed = templates / SERVICE_CONTRACTS[0].filename
        malformed.write_text("not a plist", encoding="utf-8")
        output = root / "malformed-output"
        _expect_reason(
            "template_invalid",
            lambda: build_active_launchagent_contracts(
                env_file=env_file,
                output_dir=output,
                template_root=templates,
            ),
        )
        if os.path.lexists(output):
            raise SystemExit("malformed template created an output directory")

        templates = _template_fixture(root / "second")
        scheduler_template = templates / SERVICE_CONTRACTS[1].filename
        payload = plistlib.loads(scheduler_template.read_bytes())
        payload["EnvironmentVariables"][SCHEDULER_ENABLE_KEY] = "1"
        scheduler_template.write_bytes(plistlib.dumps(payload))
        scheduler_output = root / "scheduler-output"
        _expect_reason(
            "template_invalid",
            lambda: build_active_launchagent_contracts(
                env_file=env_file,
                output_dir=scheduler_output,
                template_root=templates,
            ),
        )
        if os.path.lexists(scheduler_output):
            raise SystemExit("scheduler-bearing template created an output directory")


def test_cli_receipt_is_path_and_secret_free() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-active-cli-", dir="/private/tmp") as temp:
        root = Path(temp)
        env_file = _private_env(root)
        output = root / "cli-output"
        child = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.fspath(PROJECT_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jarvis_v2.scripts.v3_active_launchagent_contracts",
                "--env-file",
                os.fspath(env_file),
                "--output-dir",
                os.fspath(output),
            ],
            cwd=PROJECT_ROOT,
            env=child,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        if result.returncode != 0 or result.stderr:
            raise SystemExit("active contract CLI did not complete cleanly")
        receipt = json.loads(result.stdout)
        if receipt.get("ok") is not True:
            raise SystemExit("active contract CLI receipt did not report success")
        for forbidden in (SECRET, os.fspath(root), os.fspath(PROJECT_ROOT), "/private/"):
            if forbidden in result.stdout:
                raise SystemExit("active contract CLI receipt leaked path or secret material")

        hostile_path = root / f"missing-{SECRET}.env"
        failed = subprocess.run(
            [
                sys.executable,
                "-m",
                "jarvis_v2.scripts.v3_active_launchagent_contracts",
                "--env-file",
                os.fspath(hostile_path),
                "--output-dir",
                os.fspath(root / "failed-output"),
            ],
            cwd=PROJECT_ROOT,
            env=child,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        if failed.returncode != 2 or failed.stderr:
            raise SystemExit("active contract CLI failure was not bounded")
        failure_receipt = json.loads(failed.stdout)
        if failure_receipt != {"ok": False, "reason": "env_missing_or_unreadable"}:
            raise SystemExit("active contract CLI failure receipt drifted")
        if SECRET in failed.stdout or os.fspath(root) in failed.stdout:
            raise SystemExit("active contract CLI failure leaked supplied path material")


def test_builder_has_no_activation_capability() -> None:
    source = (PROJECT_ROOT / "jarvis_v2/scripts/v3_active_launchagent_contracts.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("subprocess", "launch" + "ctl", "Library/" + "LaunchAgents", "run_scheduler")
    found = [marker for marker in forbidden if marker in source]
    if found:
        raise SystemExit(f"offline active contract builder gained activation capability: {found}")


def main() -> None:
    if IS_PUBLIC_CANDIDATE:
        test_sanitized_candidate_omits_templates_and_fails_closed()
        test_builder_has_no_activation_capability()
        print("V3 offline active LaunchAgent contract builder smoke passed")
        return
    test_valid_build_is_offline_bounded_and_secret_free()
    test_environment_custody_rejects_malformed_symlink_and_permissions()
    test_log_directory_uses_canonical_account_home_without_creating_it()
    test_output_must_be_new_and_contained()
    test_malformed_and_scheduler_templates_fail_before_output()
    test_cli_receipt_is_path_and_secret_free()
    test_builder_has_no_activation_capability()
    print("V3 offline active LaunchAgent contract builder smoke passed")


if __name__ == "__main__":
    main()
