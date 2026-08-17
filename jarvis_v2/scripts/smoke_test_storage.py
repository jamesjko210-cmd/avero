"""Smoke tests for Jarvis storage readiness tools (mocked diagnostics, no writes)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools import storage as st


def assert_storage_metadata_bool_is_exact() -> None:
    if st._metadata_bool(True) is not True:
        raise SystemExit("storage exact metadata bool rejected True")
    if st._metadata_bool(False) is not False:
        raise SystemExit("storage exact metadata bool rejected False")
    for value in ("true", "false", "yes", "no", 1, 0, [True], {"ready": True}, None):
        if st._metadata_bool(value) is not False:
            raise SystemExit(f"storage exact metadata bool accepted malformed truthy value: {value!r}")
    if st._metadata_bool("false", default=True) is not True:
        raise SystemExit("storage exact metadata bool did not preserve explicit default")


def _temp_config(tmp: str) -> JarvisConfig:
    root = Path(tmp)
    return JarvisConfig(
        data_dir=root / "data",
        db_path=root / "data" / "jarvis.sqlite",
        obsidian_vault=root / "Vault",
        obsidian_root="Jarvis",
        watched_dirs=(),
    )


def _malformed_ready_diagnostics() -> dict[str, Any]:
    return {
        "available": "true",
        "status": "ready",
        "data_dir": "/tmp/jarvis-storage-data",
        "db_path": "/tmp/jarvis-storage-data/jarvis.sqlite",
        "db_parent": "/tmp/jarvis-storage-data",
        "db_exists": "true",
        "data_dir_exists": "true",
        "db_parent_exists": "true",
        "data_dir_writable": "true",
        "db_parent_writable": "true",
        "db_file_writable": "true",
        "obsidian_vault": "/tmp/jarvis-storage-vault",
        "obsidian_root": "Jarvis",
        "obsidian_root_path": "/tmp/jarvis-storage-vault/Jarvis",
        "obsidian_vault_exists": "true",
        "obsidian_root_exists": "true",
        "obsidian_vault_writable": "true",
        "obsidian_root_writable": "true",
        "workspace_local_notes": "true",
        "metadata_only": "true",
        "issues": ["diagnostic booleans were malformed"],
        "recovery_check_command": st.BOOTSTRAP_CHECK_COMMAND,
        "recovery_check_api": st.STORAGE_RECOVERY_CHECK_API,
        "recovery_command": st.BOOTSTRAP_WRITE_COMMAND,
    }


def _assert_malformed_diagnostic_flags_fail_closed(metadata: dict[str, Any], label: str) -> None:
    for key in [
        "storage_configured_available",
        "storage_db_exists",
        "storage_data_dir_exists",
        "storage_db_parent_exists",
        "storage_data_dir_writable",
        "storage_db_parent_writable",
        "storage_db_file_writable",
        "storage_obsidian_vault_exists",
        "storage_obsidian_root_exists",
        "storage_obsidian_vault_writable",
        "storage_obsidian_root_writable",
        "storage_workspace_local_notes",
        "storage_metadata_only",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} accepted malformed diagnostic boolean for {key}: {metadata}")


def test_storage_status_rejects_malformed_diagnostic_bools() -> None:
    original_storage_diagnostics = st.storage_diagnostics

    def fake_storage_diagnostics(config: JarvisConfig | None) -> dict[str, Any]:
        return _malformed_ready_diagnostics()

    with TemporaryDirectory() as tmp:
        try:
            st.storage_diagnostics = fake_storage_diagnostics  # type: ignore[assignment]
            result = st.make_storage_status_tool(_temp_config(tmp))({})
        finally:
            st.storage_diagnostics = original_storage_diagnostics  # type: ignore[assignment]

    if not result.ok:
        raise SystemExit(f"storage status should return a conservative readiness packet: {result.output}")
    metadata = result.metadata
    _assert_malformed_diagnostic_flags_fail_closed(metadata, "storage status")
    if metadata.get("storage_available") is not False:
        raise SystemExit(f"storage status should not report availability from malformed diagnostics: {metadata}")
    if metadata.get("storage_ready_for_completion_claim") is not False:
        raise SystemExit(f"storage status should not allow completion claim from malformed diagnostics: {metadata}")
    if metadata.get("storage_recovery_required") is not True:
        raise SystemExit(f"storage status should require recovery for malformed diagnostics: {metadata}")
    if metadata.get("storage_readiness_blocks_completion_claim") is not True:
        raise SystemExit(f"storage status should block completion claim for malformed diagnostics: {metadata}")
    for expected in [
        "database parent writable: no",
        "database file writable: no",
        "note vault writable: no",
        "workspace-local notes: no",
        "metadata-only check: no",
    ]:
        if expected not in result.output:
            raise SystemExit(f"storage status output accepted malformed diagnostic boolean for {expected}: {result.output}")
    handoff = metadata.get("storage_status_handoff")
    if not isinstance(handoff, dict) or handoff.get("storage_available") is not False:
        raise SystemExit(f"storage status handoff should mirror conservative availability: {metadata}")
    if "/tmp/" in str(metadata) or "/private/" in str(metadata) or "/\x55sers/" in str(metadata) or "/var/folders/" in str(metadata):
        raise SystemExit(f"storage status should redact local diagnostic paths: {metadata}")


def test_storage_recovery_check_rejects_malformed_diagnostic_bools() -> None:
    original_storage_diagnostics = st.storage_diagnostics

    def fake_storage_diagnostics(config: JarvisConfig | None) -> dict[str, Any]:
        return _malformed_ready_diagnostics()

    with TemporaryDirectory() as tmp:
        try:
            st.storage_diagnostics = fake_storage_diagnostics  # type: ignore[assignment]
            result = st.make_storage_recovery_check_tool(_temp_config(tmp))({})
        finally:
            st.storage_diagnostics = original_storage_diagnostics  # type: ignore[assignment]

    if not result.ok:
        raise SystemExit(f"storage recovery check should return a conservative readiness packet: {result.output}")
    metadata = result.metadata
    _assert_malformed_diagnostic_flags_fail_closed(metadata, "storage recovery check")
    if metadata.get("storage_recovery_check_passed") is not False:
        raise SystemExit(f"storage recovery check should not pass from malformed diagnostics: {metadata}")
    if metadata.get("storage_ready_for_completion_claim") is not False:
        raise SystemExit(f"storage recovery check should not allow completion claim from malformed diagnostics: {metadata}")
    if metadata.get("storage_recovery_required") is not True:
        raise SystemExit(f"storage recovery check should require recovery for malformed diagnostics: {metadata}")
    handoff = metadata.get("storage_recovery_check_handoff")
    if not isinstance(handoff, dict) or handoff.get("storage_recovery_check_passed") is not False:
        raise SystemExit(f"storage recovery handoff should mirror conservative check failure: {metadata}")
    if "/tmp/" in str(metadata) or "/private/" in str(metadata) or "/\x55sers/" in str(metadata) or "/var/folders/" in str(metadata):
        raise SystemExit(f"storage recovery check should redact local diagnostic paths: {metadata}")


def main() -> None:
    assert_storage_metadata_bool_is_exact()
    test_storage_status_rejects_malformed_diagnostic_bools()
    test_storage_recovery_check_rejects_malformed_diagnostic_bools()
    print("Storage readiness smoke passed")


if __name__ == "__main__":
    main()
