from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.store import MemoryRecord, MemoryStore
from jarvis_v2.scripts import bootstrap_memory
from jarvis_v2.scripts.startup import (
    V3_BOOTSTRAP_CHECK_COMMAND,
    V3_BOOTSTRAP_WRITE_COMMAND,
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DIAGNOSE_COMMAND,
    V3_ENV_COMMAND_PREFIX,
    startup_failure_receipt,
)
from jarvis_v2.tools import readiness as readiness_tools
from jarvis_v2.tools.storage import (
    BOOTSTRAP_CHECK_COMMAND,
    BOOTSTRAP_WRITE_COMMAND,
    STORAGE_RECOVERY_CHECK_API,
    STORAGE_RECOVERY_CHECK_COMMAND,
    STORAGE_RECOVERY_PLAN_COMMAND,
    make_storage_recovery_check_tool,
    make_storage_recovery_plan_tool,
    make_storage_status_tool,
    storage_diagnostics,
)


LOCAL_PATH_MARKERS = ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]


class EnvGuard:
    def __init__(self, updates: dict[str, str]):
        self.updates = updates
        self.previous: dict[str, str | None] = {}

    def __enter__(self):
        self.previous = {key: os.environ.get(key) for key in self.updates}
        os.environ.update(self.updates)
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assert_no_local_path(value: object, label: str) -> None:
    text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    for marker in LOCAL_PATH_MARKERS:
        if marker in text:
            raise SystemExit(f"{label} leaked {marker}: {text}")


def assert_storage_metadata_only_boundaries(boundaries: dict[str, object], label: str) -> None:
    expected = {
        "metadata_only": True,
        "reads_database_file": False,
        "reads_db_file_contents": False,
        "reads_vault_files": False,
        "scans_obsidian_vault": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "approves_request": False,
        "approves_requests": False,
        "dismisses_request": False,
        "dismisses_approvals": False,
        "calls_model": False,
        "calls_external_service": False,
        "controls_computer": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} boundary {key} diverged: {boundaries}")


def test_startup_failure_receipt_redacts_config_paths() -> None:
    with EnvGuard(
        {
            "JARVIS_DATA_DIR": "/\x55sers/example/Desktop/Claude code/startup-data",
            "JARVIS_DB_PATH": "/var/folders/zc/jarvis-startup.sqlite",
            "JARVIS_OBSIDIAN_VAULT": "/tmp/jarvis-startup-vault",
        }
    ):
        receipt = startup_failure_receipt(sqlite3.OperationalError("unable to open database"))
        assert_no_local_path(receipt, "startup receipt")
        if receipt.get("db_path") != "<local-path>":
            raise SystemExit(f"startup receipt did not redact db_path: {receipt}")
        if receipt.get("data_dir") != "<local-path>":
            raise SystemExit(f"startup receipt did not redact data_dir: {receipt}")
        if receipt.get("obsidian_vault") != "<local-path>":
            raise SystemExit(f"startup receipt did not redact obsidian_vault: {receipt}")
        message = str(receipt.get("message") or "")
        for expected in [
            "JARVIS_DATA_DIR",
            "JARVIS_DB_PATH",
            V3_BOOTSTRAP_CHECK_COMMAND,
            "retry startup",
        ]:
            if expected not in message:
                raise SystemExit(f"startup SQLite receipt missed actionable guidance {expected}: {receipt}")
        recovery = " ".join(str(item) for item in receipt.get("safe_recovery", []))
        if "without writing" not in recovery or "only after the check reports configured storage ready" not in recovery:
            raise SystemExit(f"startup SQLite receipt missed check-before-write recovery order: {receipt}")
        for expected in (
            V3_BOOTSTRAP_CHECK_COMMAND,
            V3_BOOTSTRAP_WRITE_COMMAND,
            V3_DIAGNOSE_COMMAND,
            V3_DASHBOARD_COMMAND,
            V3_DASHBOARD_INFO_COMMAND,
        ):
            if expected not in recovery:
                raise SystemExit(
                    f"startup recovery dropped the canonical V3 command {expected!r}: {receipt}"
                )
        if recovery.count(V3_ENV_COMMAND_PREFIX) != 5:
            raise SystemExit(f"startup recovery lost V3 environment custody: {receipt}")
        for forbidden in (
            "`python3 -m jarvis_v2",
            "`python3 launch_jarvis_v3",
            "`./launch_jarvis_v3",
        ):
            if forbidden in recovery:
                raise SystemExit(
                    f"startup recovery exposed a bare environment-dropping command {forbidden!r}: {receipt}"
                )

        permission_receipt = startup_failure_receipt(PermissionError("/\x55sers/example/private/startup.sqlite"))
        assert_no_local_path(permission_receipt, "startup permission receipt")
        permission_message = str(permission_receipt.get("message") or "")
        for expected in [
            "JARVIS_DATA_DIR",
            "JARVIS_DB_PATH",
            V3_BOOTSTRAP_CHECK_COMMAND,
            "retry startup",
        ]:
            if expected not in permission_message:
                raise SystemExit(f"startup permission receipt missed actionable guidance {expected}: {permission_receipt}")


def test_bootstrap_import_skips_path_shaped_legacy_records() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-startup-") as temp:
        root = Path(temp)
        old_memory = root / "old-memory.json"
        old_life = root / "old-life.json"
        old_memory.write_text(
            json.dumps(
                {
                    "facts": {
                        "safe fact": "safe body",
                        "/\x55sers/example/Desktop/Claude code/path-title": "unsafe title",
                        "/var/folders/zc/path-title": "unsafe var title",
                    },
                    "notes": [
                        {"title": "Safe note", "content": "ordinary note"},
                        {"title": "Bad note", "content": "/private/tmp/bootstrap-note"},
                        {"title": "Bad tmp note", "content": "/tmp/bootstrap-note"},
                    ],
                    "people": {"Safe Person": {"details": "met at demo"}},
                    "preferences": {"editor": "vim"},
                    "ideas": [
                        {"idea": "/\x55sers/example/Desktop/Claude code/idea"},
                        {"idea": "/var/folders/zc/idea"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        old_life.write_text(
            json.dumps(
                {
                    "identity": {"role": "builder"},
                    "schedule": "weekly review",
                    "people": {"Bad Person": "/private/tmp/person-details"},
                    "projects": ["Safe project"],
                    "preferences": {
                        "/\x55sers/example/Desktop/Claude code/pref": "bad key",
                        "theme": "/tmp/bad-theme",
                    },
                    "context": "ship small readiness slices",
                }
            ),
            encoding="utf-8",
        )

        original_old_memory = bootstrap_memory.OLD_MEMORY
        original_old_life = bootstrap_memory.OLD_LIFE_MODEL
        bootstrap_memory.OLD_MEMORY = old_memory
        bootstrap_memory.OLD_LIFE_MODEL = old_life
        try:
            records, skipped = bootstrap_memory.filter_importable_records(
                bootstrap_memory.records_from_old_memory() + bootstrap_memory.records_from_life_model()
            )
        finally:
            bootstrap_memory.OLD_MEMORY = original_old_memory
            bootstrap_memory.OLD_LIFE_MODEL = original_old_life

        if skipped != 9:
            raise SystemExit(f"bootstrap should skip 9 path-shaped records, skipped {skipped}: {records}")
        if len(records) != 8:
            raise SystemExit(f"bootstrap should keep 8 safe records, kept {len(records)}: {records}")
        assert_no_local_path([record.__dict__ for record in records], "filtered bootstrap records")

        config = JarvisConfig(
            data_dir=root / "data",
            db_path=root / "data" / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        store = MemoryStore(config.db_path)
        vault = ObsidianVault(config.obsidian_vault, config.obsidian_root)
        store.init()
        vault.init()
        items = bootstrap_memory.build_bootstrap_import_items(records)
        for item in items:
            claim = store.claim_bootstrap_memory_occurrence(
                item.record, item.source_key, item.duplicate_ordinal
            )
            outcome = reconcile_memory_projection(
                store,
                vault,
                claim.projection_target.memory_id,
                expected_operation=claim.projection_target.operation,
                expected_revision=claim.projection_target.revision,
                expected_source_digest=claim.projection_target.source_digest,
            )
            if outcome.status != "completed":
                raise SystemExit(f"bootstrap claim did not reconcile its canonical note: {outcome}")

        with store.connect() as conn:
            rows = [dict(row) for row in conn.execute("SELECT category, title, body, source FROM memories")]
        if len(rows) != len(records):
            raise SystemExit(f"bootstrap persisted the wrong number of rows: {rows}")
        assert_no_local_path(rows, "bootstrap sqlite rows")
        note_text = "\n".join(path.read_text(encoding="utf-8") for path in vault.root_path.rglob("*.md"))
        assert_no_local_path(note_text, "bootstrap obsidian notes")


def test_legacy_bootstrap_sources_are_bounded_and_shape_checked() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-source-guard-") as temp:
        root = Path(temp)
        old_memory = root / "private-old-memory.json"
        old_life = root / "private-life-model.json"
        original_old_memory = bootstrap_memory.OLD_MEMORY
        original_old_life = bootstrap_memory.OLD_LIFE_MODEL
        bootstrap_memory.OLD_MEMORY = old_memory
        bootstrap_memory.OLD_LIFE_MODEL = old_life
        try:
            cases = [
                (b'{"facts":', "invalid_json"),
                (b"\xff", "invalid_json"),
                (
                    b"[" * 1_500 + b"0" + b"]" * 1_500,
                    ("invalid_json", "invalid_root"),
                ),
                (json.dumps([]).encode("utf-8"), "invalid_root"),
                (json.dumps({"facts": []}).encode("utf-8"), "invalid_facts"),
                (json.dumps({"notes": ["private-note"]}).encode("utf-8"), "invalid_notes"),
            ]
            for payload, expected_code in cases:
                old_memory.write_bytes(payload)
                try:
                    bootstrap_memory.records_from_old_memory()
                except bootstrap_memory.LegacyBootstrapSourceError as exc:
                    expected_codes = (
                        expected_code
                        if isinstance(expected_code, tuple)
                        else (expected_code,)
                    )
                    if exc.source != "old_memory" or exc.code not in expected_codes:
                        raise SystemExit(
                            f"legacy source guard returned the wrong diagnostic: {exc.source}:{exc.code}"
                        ) from exc
                    if str(old_memory) in str(exc) or "private-note" in str(exc):
                        raise SystemExit(f"legacy source guard leaked source details: {exc}")
                else:
                    raise SystemExit(f"legacy source guard accepted {expected_code} input")

            old_memory.write_bytes(b" " * (bootstrap_memory.MAX_LEGACY_BOOTSTRAP_BYTES + 1))
            try:
                bootstrap_memory.records_from_old_memory()
            except bootstrap_memory.LegacyBootstrapSourceError as exc:
                if exc.source != "old_memory" or exc.code != "oversized":
                    raise SystemExit(f"oversized legacy source returned the wrong diagnostic: {exc}") from exc
            else:
                raise SystemExit("oversized legacy source should be rejected")

            old_memory.write_text("{}", encoding="utf-8")
            with (
                patch.object(bootstrap_memory.os, "open", wraps=os.open) as source_open,
                patch.object(bootstrap_memory.os, "read", wraps=os.read) as source_read,
            ):
                if bootstrap_memory.records_from_old_memory() != []:
                    raise SystemExit("empty legacy memory object should remain valid")
            if source_open.call_count != 1:
                raise SystemExit(f"legacy source should be opened exactly once: {source_open.call_count}")
            if not source_read.called or any(call.args[1] > 64 * 1024 for call in source_read.call_args_list):
                raise SystemExit(f"legacy source read was not chunk-bounded: {source_read.call_args_list}")

            large_body = "x" * (128 * 1024)
            old_memory.write_text(
                json.dumps({"facts": {"bounded-large-record": large_body}}),
                encoding="utf-8",
            )
            large_records = bootstrap_memory.records_from_old_memory()
            if len(large_records) != 1 or large_records[0].body != large_body:
                raise SystemExit("bounded legacy source read stopped before regular-file EOF")

            old_memory.write_text("{}", encoding="utf-8")
            real_read = os.read
            source_grew = False

            def grow_source_then_read(descriptor: int, amount: int) -> bytes:
                nonlocal source_grew
                if not source_grew:
                    with old_memory.open("ab") as stream:
                        stream.write(b" " * (bootstrap_memory.MAX_LEGACY_BOOTSTRAP_BYTES + 1))
                    source_grew = True
                return real_read(descriptor, amount)

            with patch.object(bootstrap_memory.os, "read", side_effect=grow_source_then_read) as growing_read:
                try:
                    bootstrap_memory.records_from_old_memory()
                except bootstrap_memory.LegacyBootstrapSourceError as exc:
                    if exc.source != "old_memory" or exc.code != "oversized":
                        raise SystemExit(f"growing legacy source returned the wrong diagnostic: {exc}") from exc
                else:
                    raise SystemExit("growing legacy source should be rejected at the bounded read limit")
            if not growing_read.called or any(
                call.args[1] > 64 * 1024 for call in growing_read.call_args_list
            ):
                raise SystemExit(f"growing legacy source read was not chunk-bounded: {growing_read.call_args_list}")

            nonregular = root / "private-nonregular-source"
            nonregular.mkdir()
            try:
                bootstrap_memory._legacy_json_object(nonregular, source="old_memory")
            except bootstrap_memory.LegacyBootstrapSourceError as exc:
                if exc.source != "old_memory" or exc.code != "not_regular":
                    raise SystemExit(f"nonregular legacy source returned the wrong diagnostic: {exc}") from exc
            else:
                raise SystemExit("nonregular legacy source should be rejected")

            fifo = root / "private-legacy-source.fifo"
            os.mkfifo(fifo)
            fifo_probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; from pathlib import Path; "
                        "from jarvis_v2.scripts.bootstrap_memory import "
                        "LegacyBootstrapSourceError, _legacy_json_object; "
                        "\ntry: _legacy_json_object(Path(sys.argv[1]), source='old_memory')"
                        "\nexcept LegacyBootstrapSourceError as exc: "
                        "print(f'{exc.source}:{exc.code}')"
                    ),
                    str(fifo),
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if fifo_probe.returncode != 0 or fifo_probe.stdout.strip() != "old_memory:not_regular":
                raise SystemExit(
                    "FIFO legacy source was not rejected promptly with a content-free diagnostic: "
                    f"status={fifo_probe.returncode}; stdout={fifo_probe.stdout!r}; "
                    f"stderr={fifo_probe.stderr!r}"
                )

            old_memory.unlink()
            old_life.write_text(json.dumps({"projects": {"private": "shape"}}), encoding="utf-8")
            try:
                bootstrap_memory.records_from_life_model()
            except bootstrap_memory.LegacyBootstrapSourceError as exc:
                if exc.source != "life_model" or exc.code != "invalid_projects":
                    raise SystemExit(f"life-model shape guard returned the wrong diagnostic: {exc}") from exc
            else:
                raise SystemExit("invalid life-model projects shape should be rejected")
        finally:
            bootstrap_memory.OLD_MEMORY = original_old_memory
            bootstrap_memory.OLD_LIFE_MODEL = original_old_life


def test_legacy_bootstrap_failures_drop_private_exception_context() -> None:
    private_marker = "/\x55sers/private/bootstrap-secret"

    def assert_clean(exc: bootstrap_memory.LegacyBootstrapSourceError, code: str) -> None:
        if exc.source != "old_memory" or exc.code != code:
            raise SystemExit(f"legacy source returned the wrong clean diagnostic: {exc}")
        if exc.__cause__ is not None or exc.__context__ is not None:
            raise SystemExit(
                f"legacy source retained a private cause/context: "
                f"cause={exc.__cause__!r}, context={exc.__context__!r}"
            )
        if private_marker in str(exc) or private_marker in repr(exc):
            raise SystemExit(f"legacy source leaked private exception text: {exc!r}")

    cases = [
        ("open", patch.object(bootstrap_memory.os, "open", side_effect=PermissionError(private_marker))),
        (
            "stat",
            patch.object(bootstrap_memory.os, "fstat", side_effect=OSError(private_marker)),
        ),
        (
            "read",
            patch.object(bootstrap_memory.os, "read", side_effect=OSError(private_marker)),
        ),
    ]
    with TemporaryDirectory(prefix="jarvis-bootstrap-clean-context-") as temp:
        path = Path(temp) / "legacy.json"
        path.write_text("{}", encoding="utf-8")
        for label, failure_patch in cases:
            try:
                with failure_patch:
                    bootstrap_memory._legacy_json_object(path, source="old_memory")
            except bootstrap_memory.LegacyBootstrapSourceError as exc:
                assert_clean(exc, "unreadable")
            else:
                raise SystemExit(f"legacy {label} failure should be sanitized")

        path.write_text('{"facts":', encoding="utf-8")
        try:
            bootstrap_memory._legacy_json_object(path, source="old_memory")
        except bootstrap_memory.LegacyBootstrapSourceError as exc:
            assert_clean(exc, "invalid_json")
        else:
            raise SystemExit("legacy parse failure should be sanitized")


def test_legacy_bootstrap_close_failure_cannot_mask_primary_diagnostic() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-close-failure-") as temp:
        path = Path(temp) / "legacy.json"
        path.write_text('{"facts":', encoding="utf-8")
        with patch.object(
            bootstrap_memory.os,
            "close",
            side_effect=OSError("/private/bootstrap-close-secret"),
        ):
            try:
                bootstrap_memory._legacy_json_object(path, source="old_memory")
            except bootstrap_memory.LegacyBootstrapSourceError as exc:
                if exc.code != "invalid_json":
                    raise SystemExit(f"close failure masked the primary parse diagnostic: {exc}") from exc
                if exc.__cause__ is not None or exc.__context__ is not None:
                    raise SystemExit("primary legacy diagnostic retained close-failure context")
            else:
                raise SystemExit("invalid legacy JSON should fail even when close also fails")

        path.write_text("{}", encoding="utf-8")
        with patch.object(
            bootstrap_memory.os,
            "close",
            side_effect=OSError("/private/bootstrap-close-secret"),
        ):
            try:
                bootstrap_memory._legacy_json_object(path, source="old_memory")
            except bootstrap_memory.LegacyBootstrapSourceError as exc:
                if exc.code != "unreadable" or exc.__cause__ is not None or exc.__context__ is not None:
                    raise SystemExit(f"standalone close failure was not sanitized: {exc!r}") from exc
            else:
                raise SystemExit("standalone legacy close failure should fail closed")


def test_legacy_bootstrap_failure_receipt_is_content_free() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-source-receipt-") as temp:
        root = Path(temp)
        old_memory = root / "private-old-memory.json"
        old_life = root / "missing-life-model.json"
        old_memory.write_text('{"facts": "private-secret-marker"}', encoding="utf-8")
        config = JarvisConfig(
            data_dir=root / "data",
            db_path=root / "data" / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        original_old_memory = bootstrap_memory.OLD_MEMORY
        original_old_life = bootstrap_memory.OLD_LIFE_MODEL
        original_load_config = bootstrap_memory.load_config
        bootstrap_memory.OLD_MEMORY = old_memory
        bootstrap_memory.OLD_LIFE_MODEL = old_life
        bootstrap_memory.load_config = lambda: config
        stdout = StringIO()
        try:
            with redirect_stdout(stdout):
                try:
                    bootstrap_memory.main([])
                except SystemExit as exc:
                    if exc.code != 5:
                        raise SystemExit(f"legacy source failure should exit 5, got {exc.code}") from exc
                else:
                    raise SystemExit("legacy source failure should stop bootstrap")
        finally:
            bootstrap_memory.OLD_MEMORY = original_old_memory
            bootstrap_memory.OLD_LIFE_MODEL = original_old_life
            bootstrap_memory.load_config = original_load_config

        output = stdout.getvalue()
        for expected in ("source=old_memory", "code=invalid_facts", "no source content was displayed"):
            if expected not in output:
                raise SystemExit(f"legacy source failure receipt missed {expected!r}: {output}")
        for forbidden in (str(old_memory), "private-secret-marker"):
            if forbidden in output:
                raise SystemExit(f"legacy source failure receipt leaked {forbidden!r}: {output}")


def _bootstrap_fixture(root: Path) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / "jarvis.sqlite")
    vault = ObsidianVault(root / "Vault", "Jarvis")
    store.init()
    vault.init()
    return store, vault


def _claim_and_reconcile(
    store: MemoryStore,
    vault: ObsidianVault,
    item: bootstrap_memory.BootstrapImportItem,
):
    claim = store.claim_bootstrap_memory_occurrence(
        item.record, item.source_key, item.duplicate_ordinal
    )
    outcome = reconcile_memory_projection(
        store,
        vault,
        claim.projection_target.memory_id,
        expected_operation=claim.projection_target.operation,
        expected_revision=claim.projection_target.revision,
        expected_source_digest=claim.projection_target.source_digest,
    )
    if outcome.status != "completed":
        raise SystemExit(f"bootstrap projection did not complete: {claim}, {outcome}")
    return claim, outcome


def _bootstrap_counts(store: MemoryStore) -> dict[str, int]:
    with store.connect() as conn:
        return {
            "memories": int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]),
            "occurrences": int(
                conn.execute("SELECT COUNT(*) FROM bootstrap_memory_occurrences").fetchone()[0]
            ),
            "jobs": int(conn.execute("SELECT COUNT(*) FROM memory_projection_jobs").fetchone()[0]),
        }


def test_bootstrap_occurrences_are_idempotent_content_addressed_and_private() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-occurrence-") as temp:
        root = Path(temp)
        store, vault = _bootstrap_fixture(root)
        original = MemoryRecord(
            category="preference",
            title="Bootstrap editor",
            body="private-bootstrap-body-alpha",
            source="jarvis-ollama-import",
            confidence=1.0,
        )
        changed = MemoryRecord(
            category=original.category,
            title=original.title,
            body="private-bootstrap-body-beta",
            source=original.source,
            confidence=original.confidence,
        )
        original_item = bootstrap_memory.build_bootstrap_import_items([original])[0]
        changed_item = bootstrap_memory.build_bootstrap_import_items([changed])[0]

        try:
            store.claim_bootstrap_memory_occurrence(
                changed,
                original_item.source_key,
                original_item.duplicate_ordinal,
            )
        except ValueError:
            pass
        else:
            raise SystemExit("bootstrap custody accepted a source key for the wrong record")
        if _bootstrap_counts(store) != {"memories": 0, "occurrences": 0, "jobs": 0}:
            raise SystemExit("bootstrap source-key mismatch crossed the write boundary")

        first, first_outcome = _claim_and_reconcile(store, vault, original_item)
        repeated, repeated_outcome = _claim_and_reconcile(store, vault, original_item)
        if first.status != "created" or repeated.status != "existing":
            raise SystemExit(f"bootstrap exact rerun statuses drifted: {first}, {repeated}")
        if first.memory_id != repeated.memory_id:
            raise SystemExit(f"bootstrap exact rerun changed memory id: {first}, {repeated}")
        if first_outcome.path_display != repeated_outcome.path_display:
            raise SystemExit("bootstrap exact rerun changed its canonical note path")
        if _bootstrap_counts(store) != {"memories": 1, "occurrences": 1, "jobs": 1}:
            raise SystemExit(f"bootstrap exact rerun duplicated custody: {_bootstrap_counts(store)}")
        canonical_notes = list((vault.root_path / "Memory Tree" / "Records").glob("*.md"))
        if len(canonical_notes) != 1:
            raise SystemExit(f"bootstrap exact rerun should create one canonical note: {canonical_notes}")

        changed_claim, _ = _claim_and_reconcile(store, vault, changed_item)
        reverted_claim, _ = _claim_and_reconcile(store, vault, original_item)
        if changed_claim.memory_id == first.memory_id:
            raise SystemExit("changed bootstrap content should create one new occurrence")
        if reverted_claim.status != "existing" or reverted_claim.memory_id != first.memory_id:
            raise SystemExit(f"reverted bootstrap content did not reconnect prior key: {reverted_claim}")
        if _bootstrap_counts(store) != {"memories": 2, "occurrences": 2, "jobs": 2}:
            raise SystemExit(f"changed bootstrap content affected unrelated occurrences: {_bootstrap_counts(store)}")

        summary_key = bootstrap_memory.bootstrap_summary_source_key(
            [original_item, changed_item], 0
        )
        store.reserve_bootstrap_summary_run(summary_key, "2026-07-12")
        with store.connect() as conn:
            ledger = {
                "occurrences": [
                    dict(row)
                    for row in conn.execute("SELECT * FROM bootstrap_memory_occurrences")
                ],
                "jobs": [dict(row) for row in conn.execute("SELECT * FROM memory_projection_jobs")],
                "summaries": [dict(row) for row in conn.execute("SELECT * FROM bootstrap_summary_runs")],
            }
        ledger_text = json.dumps(ledger, sort_keys=True)
        for forbidden in [original.body, changed.body, original.title, str(root)]:
            if forbidden in ledger_text:
                raise SystemExit(f"bootstrap custody ledger leaked source content/path: {forbidden!r}")
        assert_no_local_path(ledger, "bootstrap custody ledgers")


def test_bootstrap_duplicate_ordinals_are_stable_across_reordering() -> None:
    duplicate = MemoryRecord("note", "Repeated", "same", "life-model-import", 1.0)
    other = MemoryRecord("fact", "Other", "different", "life-model-import", 1.0)
    first = bootstrap_memory.build_bootstrap_import_items([duplicate, other, duplicate])
    reordered = bootstrap_memory.build_bootstrap_import_items([duplicate, duplicate, other])
    duplicate_items = [item for item in first if item.record == duplicate]
    if [item.duplicate_ordinal for item in duplicate_items] != [0, 1]:
        raise SystemExit(f"identical bootstrap records lost distinct ordinals: {duplicate_items}")
    if duplicate_items[0].source_key == duplicate_items[1].source_key:
        raise SystemExit("identical bootstrap records received the same occurrence key")
    if {item.source_key for item in first} != {item.source_key for item in reordered}:
        raise SystemExit("bootstrap occurrence key set changed after list reordering")
    if bootstrap_memory.bootstrap_summary_source_key(first, 3) != bootstrap_memory.bootstrap_summary_source_key(
        reordered, 3
    ):
        raise SystemExit("bootstrap summary identity changed after list reordering")


def test_bootstrap_concurrent_claim_and_reservation_rollback() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-concurrent-") as temp:
        root = Path(temp)
        store, _vault = _bootstrap_fixture(root)
        record = MemoryRecord("fact", "Concurrent", "one occurrence", "life-model-import", 1.0)
        item = bootstrap_memory.build_bootstrap_import_items([record])[0]
        barrier = threading.Barrier(3)
        results = []
        failures = []

        def claim() -> None:
            try:
                barrier.wait()
                results.append(
                    store.claim_bootstrap_memory_occurrence(
                        record, item.source_key, item.duplicate_ordinal
                    )
                )
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=claim) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in threads) or failures:
            raise SystemExit(f"concurrent bootstrap claims failed: {failures}")
        if len(results) != 2 or len({result.memory_id for result in results}) != 1:
            raise SystemExit(f"concurrent bootstrap claims diverged: {results}")
        if sorted(result.status for result in results) != ["created", "existing"]:
            raise SystemExit(f"concurrent bootstrap claim statuses drifted: {results}")
        if _bootstrap_counts(store) != {"memories": 1, "occurrences": 1, "jobs": 1}:
            raise SystemExit(f"concurrent bootstrap claims duplicated custody: {_bootstrap_counts(store)}")

        rollback_record = MemoryRecord(
            "fact", "Rollback", "must not commit", "life-model-import", 1.0
        )
        rollback_item = bootstrap_memory.build_bootstrap_import_items([rollback_record])[0]
        with store.connect() as conn:
            conn.execute(
                "CREATE TRIGGER reject_bootstrap_projection BEFORE INSERT ON memory_projection_jobs "
                "WHEN NEW.memory_id > 1 BEGIN SELECT RAISE(ABORT, 'projection rejected'); END"
            )
        try:
            store.claim_bootstrap_memory_occurrence(
                rollback_record,
                rollback_item.source_key,
                rollback_item.duplicate_ordinal,
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise SystemExit("bootstrap reservation failure should surface")
        with store.connect() as conn:
            conn.execute("DROP TRIGGER reject_bootstrap_projection")
        if _bootstrap_counts(store) != {"memories": 1, "occurrences": 1, "jobs": 1}:
            raise SystemExit(f"bootstrap reservation failure did not roll back: {_bootstrap_counts(store)}")


def test_bootstrap_adopts_exact_legacy_rows_one_to_one() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-adopt-") as temp:
        store, _vault = _bootstrap_fixture(Path(temp))
        record = MemoryRecord("fact", "Legacy", "exact", "jarvis-ollama-import", 1.0)
        legacy_ids = [store.add_memory(record) for _ in range(3)]
        items = bootstrap_memory.build_bootstrap_import_items([record, record])
        claims = [
            store.claim_bootstrap_memory_occurrence(
                item.record, item.source_key, item.duplicate_ordinal
            )
            for item in items
        ]
        if [claim.status for claim in claims] != ["adopted", "adopted"]:
            raise SystemExit(f"bootstrap did not adopt exact legacy rows: {claims}")
        if [claim.memory_id for claim in claims] != legacy_ids[:2]:
            raise SystemExit(f"bootstrap legacy adoption was not one-to-one/lowest-id: {claims}")
        with store.connect() as conn:
            unclaimed = [
                int(row[0])
                for row in conn.execute(
                    "SELECT id FROM memories WHERE id NOT IN "
                    "(SELECT memory_id FROM bootstrap_memory_occurrences) ORDER BY id"
                )
            ]
        if unclaimed != [legacy_ids[2]]:
            raise SystemExit(f"bootstrap modified surplus ambiguous legacy rows: {unclaimed}")


def test_bootstrap_occurrence_custody_cascades_on_delete_merge_and_migration() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-cascade-") as temp:
        store, _vault = _bootstrap_fixture(Path(temp))
        delete_record = MemoryRecord(
            "fact", "Delete claimed", "delete body", "life-model-import", 1.0
        )
        delete_item = bootstrap_memory.build_bootstrap_import_items([delete_record])[0]
        delete_claim = store.claim_bootstrap_memory_occurrence(
            delete_record, delete_item.source_key, delete_item.duplicate_ordinal
        )
        delete_target = store.resolve_memory_approval_target(delete_claim.memory_id)
        if delete_target is None:
            raise SystemExit("claimed bootstrap memory lost its exact delete target")
        deleted = store.delete_memory_exact_with_projection(
            delete_target.memory_id,
            delete_target.revision,
            delete_target.binding,
        )
        if deleted.status != "deleted":
            raise SystemExit(f"claimed bootstrap memory could not be deleted: {deleted}")
        with store.connect() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM bootstrap_memory_occurrences WHERE source_key = ?",
                (delete_item.source_key,),
            ).fetchone()[0]
        if remaining != 0:
            raise SystemExit("bootstrap occurrence custody did not cascade after exact delete")

        keep_record = MemoryRecord(
            "fact", "Merge keep", "keep body", "life-model-import", 1.0
        )
        merged_record = MemoryRecord(
            "fact", "Merge delete", "merged body", "life-model-import", 1.0
        )
        keep_item = bootstrap_memory.build_bootstrap_import_items([keep_record])[0]
        merged_item = bootstrap_memory.build_bootstrap_import_items([merged_record])[0]
        keep_claim = store.claim_bootstrap_memory_occurrence(
            keep_record, keep_item.source_key, keep_item.duplicate_ordinal
        )
        merged_claim = store.claim_bootstrap_memory_occurrence(
            merged_record, merged_item.source_key, merged_item.duplicate_ordinal
        )
        merge_targets = store.resolve_memory_merge_approval_targets(
            keep_claim.memory_id, merged_claim.memory_id
        )
        if merge_targets is None:
            raise SystemExit("claimed bootstrap memories lost their exact merge targets")
        keep_target, merged_target = merge_targets
        merged = store.merge_memories_exact_with_projection(
            keep_target.memory_id,
            keep_target.revision,
            keep_target.binding,
            merged_target.memory_id,
            merged_target.revision,
            merged_target.binding,
        )
        if merged.status != "merged":
            raise SystemExit(f"claimed bootstrap memory could not be merged: {merged}")
        with store.connect() as conn:
            occurrence_keys = {
                str(row[0])
                for row in conn.execute(
                    "SELECT source_key FROM bootstrap_memory_occurrences"
                )
            }
        if occurrence_keys != {keep_item.source_key, merged_item.source_key}:
            raise SystemExit(
                f"bootstrap merge custody did not retain both source occurrences: {occurrence_keys}"
            )
        with store.connect() as conn:
            merged_memory_ids = {
                int(row[0])
                for row in conn.execute(
                    "SELECT memory_id FROM bootstrap_memory_occurrences "
                    "WHERE source_key IN (?, ?)",
                    (keep_item.source_key, merged_item.source_key),
                )
            }
        if merged_memory_ids != {keep_claim.memory_id}:
            raise SystemExit(
                f"bootstrap merge custody did not attach both sources to survivor: {merged_memory_ids}"
            )
        rerun_claim = store.claim_bootstrap_memory_occurrence(
            merged_record, merged_item.source_key, merged_item.duplicate_ordinal
        )
        if rerun_claim.status != "existing" or rerun_claim.memory_id != keep_claim.memory_id:
            raise SystemExit(f"bootstrap merged source was recreated on rerun: {rerun_claim}")
        if len(store.list_memories(limit=20)) != 1:
            raise SystemExit("bootstrap merged-source rerun recreated duplicate semantic content")

    with TemporaryDirectory(prefix="jarvis-bootstrap-fk-migration-") as temp:
        store, _vault = _bootstrap_fixture(Path(temp))
        record = MemoryRecord(
            "note", "Migrate custody", "preserve row", "jarvis-ollama-import", 1.0
        )
        item = bootstrap_memory.build_bootstrap_import_items([record])[0]
        claim = store.claim_bootstrap_memory_occurrence(
            record, item.source_key, item.duplicate_ordinal
        )
        with sqlite3.connect(store.db_path) as conn:
            occurrence = conn.execute(
                "SELECT source_key, memory_id, created_at, updated_at "
                "FROM bootstrap_memory_occurrences WHERE source_key = ?",
                (item.source_key,),
            ).fetchone()
            if occurrence is None:
                raise SystemExit("bootstrap migration fixture missed its custody row")
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DROP TABLE bootstrap_memory_occurrences")
            conn.execute(
                """
                CREATE TABLE bootstrap_memory_occurrences (
                    source_key TEXT PRIMARY KEY,
                    memory_id INTEGER NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE RESTRICT
                )
                """
            )
            conn.execute(
                "INSERT INTO bootstrap_memory_occurrences "
                "(source_key, memory_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                occurrence,
            )
        store.init()
        with store.connect() as conn:
            foreign_keys = list(
                conn.execute("PRAGMA foreign_key_list(bootstrap_memory_occurrences)")
            )
            migrated = conn.execute(
                "SELECT source_key, memory_id FROM bootstrap_memory_occurrences"
            ).fetchone()
        if len(foreign_keys) != 1 or not all(
            row["table"] == "memories"
            and row["from"] == "memory_id"
            and row["to"] == "id"
            and str(row["on_delete"]).upper() == "CASCADE"
            for row in foreign_keys
        ):
            raise SystemExit(f"bootstrap custody migration did not install CASCADE: {foreign_keys}")
        with store.connect() as conn:
            unique_memory_indexes = []
            for index in conn.execute("PRAGMA index_list(bootstrap_memory_occurrences)"):
                if int(index["unique"]) != 1:
                    continue
                index_name = str(index["name"]).replace('"', '""')
                columns = [
                    str(row["name"])
                    for row in conn.execute(f'PRAGMA index_info("{index_name}")')
                ]
                if columns == ["memory_id"]:
                    unique_memory_indexes.append(str(index["name"]))
        if unique_memory_indexes:
            raise SystemExit(
                f"bootstrap custody migration kept one-source-per-memory uniqueness: {unique_memory_indexes}"
            )
        if migrated is None or migrated["source_key"] != item.source_key or int(
            migrated["memory_id"]
        ) != claim.memory_id:
            raise SystemExit("bootstrap custody migration did not preserve the claimed occurrence")
        target = store.resolve_memory_approval_target(claim.memory_id)
        if target is None:
            raise SystemExit("migrated bootstrap custody lost its memory target")
        deleted = store.delete_memory_exact_with_projection(
            target.memory_id, target.revision, target.binding
        )
        if deleted.status != "deleted" or _bootstrap_counts(store)["occurrences"] != 0:
            raise SystemExit("migrated bootstrap custody still blocked exact deletion")


def test_bootstrap_projection_failure_and_crash_adoption_repair_same_id() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-projection-repair-") as temp:
        root = Path(temp)
        store, vault = _bootstrap_fixture(root)
        record = MemoryRecord("note", "Repair", "durable", "life-model-import", 1.0)
        item = bootstrap_memory.build_bootstrap_import_items([record])[0]
        claim = store.claim_bootstrap_memory_occurrence(
            record, item.source_key, item.duplicate_ordinal
        )
        original_write = vault.write_memory_projection_with_evidence

        def fail_write(*args, **kwargs):
            raise OSError("simulated projection write failure")

        vault.write_memory_projection_with_evidence = fail_write
        try:
            failed = reconcile_memory_projection(store, vault, claim.memory_id)
        finally:
            vault.write_memory_projection_with_evidence = original_write
        if failed.status != "pending_error" or _bootstrap_counts(store)["memories"] != 1:
            raise SystemExit(f"bootstrap projection failure lost/duplicated occurrence: {failed}")
        repaired_claim, repaired = _claim_and_reconcile(store, vault, item)
        if repaired_claim.memory_id != claim.memory_id or _bootstrap_counts(store)["memories"] != 1:
            raise SystemExit(f"bootstrap projection retry changed committed memory: {repaired_claim}")

        crash_record = MemoryRecord("note", "Crash", "adopt exact bytes", "life-model-import", 1.0)
        crash_item = bootstrap_memory.build_bootstrap_import_items([crash_record])[0]
        crash_claim = store.claim_bootstrap_memory_occurrence(
            crash_record, crash_item.source_key, crash_item.duplicate_ordinal
        )
        job = store.get_memory_projection_job(crash_claim.memory_id)
        snapshot = store.get_current_memory_projection_publish_snapshot(
            crash_claim.memory_id,
            crash_claim.projection_target.revision,
            crash_claim.projection_target.source_digest,
        )
        if job is None or snapshot is None:
            raise SystemExit("bootstrap crash fixture did not retain projection custody")
        crash_path, crash_digest = original_write(
            MemoryRecord(
                snapshot["category"],
                snapshot["title"],
                snapshot["body"],
                snapshot["source"],
                float(snapshot["confidence"]),
            ),
            memory_id=crash_claim.memory_id,
            store_identity=job["store_identity"],
            memory_revision=job["memory_revision"],
            source_digest=job["source_digest"],
            created_at=snapshot["memory_created_at"],
            expected_prior_content_digest=job["prior_content_digest"],
        )
        if store.get_memory_projection_job(crash_claim.memory_id)["state"] != "pending":
            raise SystemExit("bootstrap crash simulation unexpectedly completed the ledger")
        adopted = reconcile_memory_projection(store, vault, crash_claim.memory_id)
        if adopted.status != "completed" or adopted.content_digest != crash_digest:
            raise SystemExit(f"bootstrap did not adopt crash-written exact bytes: {adopted}")
        if adopted.path_display != str(crash_path.relative_to(vault.root_path)):
            raise SystemExit(f"bootstrap crash adoption changed canonical path: {adopted}")
        if _bootstrap_counts(store)["memories"] != 2:
            raise SystemExit("bootstrap crash adoption duplicated the source memory")


def test_bootstrap_summary_is_once_and_retains_reserved_date_after_crash() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-summary-") as temp:
        store, vault = _bootstrap_fixture(Path(temp))
        item = bootstrap_memory.build_bootstrap_import_items(
            [MemoryRecord("fact", "Summary", "body", "life-model-import", 1.0)]
        )[0]
        summary_key = bootstrap_memory.bootstrap_summary_source_key([item], 0)
        first = store.reserve_bootstrap_summary_run(summary_key, "2026-07-11")
        path, appended = vault.append_daily_once_for_date(
            first.target_date, first.source_key, "Jarvis V2 Bootstrap", "summary body"
        )
        if not appended:
            raise SystemExit("bootstrap summary first append did not write")
        retry = store.reserve_bootstrap_summary_run(summary_key, "2026-07-12")
        retry_path, retry_appended = vault.append_daily_once_for_date(
            retry.target_date, retry.source_key, "Jarvis V2 Bootstrap", "summary body"
        )
        if retry.status != "existing" or retry.state != "pending" or retry.target_date != "2026-07-11":
            raise SystemExit(f"bootstrap summary retry lost original target date: {retry}")
        if retry_path != path or retry_appended:
            raise SystemExit("bootstrap summary crash retry duplicated or moved its marker")
        if not store.complete_bootstrap_summary_run(retry.source_key, retry.target_date):
            raise SystemExit("bootstrap summary retry could not complete custody")
        completed = store.reserve_bootstrap_summary_run(summary_key, "2026-07-13")
        if completed.state != "completed" or completed.target_date != "2026-07-11":
            raise SystemExit(f"completed bootstrap summary drifted: {completed}")
        text = path.read_text(encoding="utf-8")
        if text.count(summary_key) != 1:
            raise SystemExit("bootstrap summary marker was not exactly once")
        if (vault.root_path / "Daily" / "2026-07-12.md").exists():
            raise SystemExit("bootstrap summary next-day retry wrote to the wrong date")


def test_bootstrap_check_does_not_construct_storage_or_read_legacy_sources() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-check-strict-") as temp:
        root = Path(temp)
        original_store = bootstrap_memory.MemoryStore
        original_vault = bootstrap_memory.ObsidianVault
        original_old_memory_reader = bootstrap_memory.records_from_old_memory
        original_life_reader = bootstrap_memory.records_from_life_model

        def forbidden(*_args, **_kwargs):
            raise AssertionError("--check crossed the metadata-only bootstrap boundary")

        bootstrap_memory.MemoryStore = forbidden
        bootstrap_memory.ObsidianVault = forbidden
        bootstrap_memory.records_from_old_memory = forbidden
        bootstrap_memory.records_from_life_model = forbidden
        try:
            with EnvGuard(
                {
                    "JARVIS_DATA_DIR": str(root / "data"),
                    "JARVIS_DB_PATH": str(root / "data" / "jarvis.sqlite"),
                    "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                }
            ), redirect_stdout(StringIO()):
                try:
                    bootstrap_memory.main(["--check"])
                except SystemExit as exc:
                    if exc.code not in {0, 2}:
                        raise SystemExit(f"bootstrap --check returned unexpected status: {exc.code}")
                else:
                    raise SystemExit("bootstrap --check should terminate with its readiness status")
        finally:
            bootstrap_memory.MemoryStore = original_store
            bootstrap_memory.ObsidianVault = original_vault
            bootstrap_memory.records_from_old_memory = original_old_memory_reader
            bootstrap_memory.records_from_life_model = original_life_reader
        if any(root.rglob("*")):
            raise SystemExit("bootstrap --check created storage while proving metadata-only readiness")


def test_storage_diagnostics_redacts_display_paths() -> None:
    config = JarvisConfig(
        data_dir=Path("/\x55sers/example/Desktop/Claude code/storage-data"),
        db_path=Path("/var/folders/zc/jarvis-storage.sqlite"),
        obsidian_vault=Path("/tmp/jarvis-storage-vault"),
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    diagnostics = storage_diagnostics(config)
    assert_no_local_path(diagnostics, "storage diagnostics")
    for key in ["data_dir", "db_path", "db_parent", "obsidian_vault", "obsidian_root_path"]:
        if diagnostics.get(key) != "<local-path>":
            raise SystemExit(f"storage diagnostics did not redact {key}: {diagnostics}")


class _FakeRisk:
    name = "HIGH_RISK"


class _FakeRiskyTool:
    risk = _FakeRisk()


def test_readiness_report_uses_exact_storage_diagnostic_bools() -> None:
    with TemporaryDirectory(prefix="jarvis-readiness-storage-exact-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root / "data",
            db_path=root / "data" / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        store = MemoryStore(config.db_path)
        vault = ObsidianVault(config.obsidian_vault, config.obsidian_root)
        store.init()
        vault.init()
        malformed_storage = {
            "status": "ready",
            "available": "true",
            "data_dir": "<local-path>",
            "db_path": "<local-path>",
            "db_parent": "<local-path>",
            "obsidian_vault": "<local-path>",
            "obsidian_root": "Jarvis",
            "obsidian_root_path": "<local-path>",
            "data_dir_exists": "true",
            "db_exists": "true",
            "db_parent_exists": "true",
            "data_dir_writable": "true",
            "db_parent_writable": "true",
            "db_file_writable": "true",
            "obsidian_vault_exists": "true",
            "obsidian_root_exists": "true",
            "obsidian_vault_writable": "true",
            "obsidian_root_writable": "true",
            "workspace_local_notes": "false",
            "metadata_only": "true",
            "issues": [],
        }
        original_diagnostics = readiness_tools._configured_storage_diagnostics
        readiness_tools._configured_storage_diagnostics = lambda _config, _fallback: (malformed_storage, "mock")
        try:
            readiness_report, _prototype = readiness_tools.make_readiness_tools(
                store,
                vault,
                config,
                lambda: [_FakeRiskyTool()],
            )
            result = readiness_report({})
        finally:
            readiness_tools._configured_storage_diagnostics = original_diagnostics

        metadata = result.metadata
        if metadata.get("storage_configured_available") is not False:
            raise SystemExit(f"readiness report trusted non-bool configured availability: {metadata}")
        if metadata.get("storage_available") is not False:
            raise SystemExit(f"readiness report trusted non-bool storage availability: {metadata}")
        if metadata.get("storage_ready_for_completion_claim") is not False:
            raise SystemExit(f"readiness report trusted non-bool completion readiness: {metadata}")
        for key in [
            "storage_db_parent_writable",
            "storage_db_file_writable",
            "storage_obsidian_vault_writable",
            "storage_obsidian_root_writable",
            "storage_metadata_only",
        ]:
            if metadata.get(key) is not False:
                raise SystemExit(f"readiness report leaked non-bool {key}: {metadata}")
        for expected in [
            "- SQLite storage parent writable: needs attention",
            "- SQLite database file writable: needs attention",
            "- Jarvis note vault writable: needs attention",
            "- Jarvis note root writable: needs attention",
        ]:
            if expected not in result.output:
                raise SystemExit(f"readiness report should show malformed storage booleans as attention: {result.output}")


def test_bootstrap_check_reports_storage_readiness_without_writes() -> None:
    with TemporaryDirectory(prefix="jarvis-bootstrap-check-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root / "data",
            db_path=root / "data" / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        with EnvGuard(
            {
                "JARVIS_DATA_DIR": str(config.data_dir),
                "JARVIS_DB_PATH": str(config.db_path),
                "JARVIS_OBSIDIAN_VAULT": str(config.obsidian_vault),
            }
        ):
            lines, code = bootstrap_memory.bootstrap_check_lines()
        output = "\n".join(lines)
        assert_no_local_path(output, "bootstrap check output")
        if code != 2:
            raise SystemExit(f"bootstrap check should report unavailable uncreated storage with exit 2: {code}, {output}")
        if config.data_dir.exists() or config.obsidian_vault.exists() or config.db_path.exists():
            raise SystemExit("bootstrap check should not create storage paths.")
        for expected in [
            "Jarvis bootstrap storage check:",
            "configured storage ready: no",
            "metadata-only check: yes",
            "does not create folders",
            "read legacy memory contents",
        ]:
            if expected not in output:
                raise SystemExit(f"bootstrap check missed expected text {expected}: {output}")

        config.data_dir.mkdir(parents=True)
        config.obsidian_vault.mkdir(parents=True)
        with EnvGuard(
            {
                "JARVIS_DATA_DIR": str(config.data_dir),
                "JARVIS_DB_PATH": str(config.db_path),
                "JARVIS_OBSIDIAN_VAULT": str(config.obsidian_vault),
            }
        ):
            ready_lines, ready_code = bootstrap_memory.bootstrap_check_lines()
        ready_output = "\n".join(ready_lines)
        assert_no_local_path(ready_output, "ready bootstrap check output")
        if ready_code != 0 or "configured storage ready: yes" not in ready_output:
            raise SystemExit(f"bootstrap check should accept writable configured storage: {ready_code}, {ready_output}")
        if config.db_path.exists():
            raise SystemExit("bootstrap check should not initialize SQLite.")


def test_bootstrap_check_fails_closed_on_malformed_storage_bools() -> None:
    original_storage_diagnostics = bootstrap_memory.storage_diagnostics

    def malformed_storage_diagnostics(_config: JarvisConfig) -> dict[str, object]:
        return {
            "available": "true",
            "status": "ready",
            "data_dir": "<local-path>",
            "db_path": "<local-path>",
            "db_parent": "<local-path>",
            "db_parent_writable": "true",
            "db_file_writable": "true",
            "obsidian_vault": "<local-path>",
            "obsidian_root": "Jarvis",
            "obsidian_root_path": "<local-path>",
            "obsidian_vault_writable": "true",
            "metadata_only": "true",
            "issues": [],
            "recovery_check_command": BOOTSTRAP_CHECK_COMMAND,
            "recovery_command": BOOTSTRAP_WRITE_COMMAND,
        }

    bootstrap_memory.storage_diagnostics = malformed_storage_diagnostics
    try:
        lines, code = bootstrap_memory.bootstrap_check_lines()
    finally:
        bootstrap_memory.storage_diagnostics = original_storage_diagnostics

    output = "\n".join(lines)
    assert_no_local_path(output, "malformed bootstrap check output")
    if code != 2:
        raise SystemExit(f"bootstrap check should fail closed on malformed diagnostic booleans: {code}, {output}")
    for expected in [
        "configured storage ready: no",
        "database parent writable: no",
        "database file writable: no",
        "note vault writable: no",
        "metadata-only check: no",
    ]:
        if expected not in output:
            raise SystemExit(f"bootstrap check accepted malformed boolean for {expected!r}: {output}")
    if f"Run `{BOOTSTRAP_WRITE_COMMAND}` only after this check reports configured storage ready." not in output:
        raise SystemExit(f"bootstrap check should keep guarded write instruction visible: {output}")


def test_storage_status_tool_reports_runtime_fallback_without_paths() -> None:
    config = JarvisConfig(
        data_dir=Path("/\x55sers/example/Desktop/Claude code/storage-data"),
        db_path=Path("/var/folders/zc/jarvis-storage.sqlite"),
        obsidian_vault=Path("/tmp/jarvis-storage-vault"),
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    fallback = {
        "db_path": "/private/tmp/jarvis-fallback/jarvis.sqlite",
        "db_path_display": "workspace-local fallback database",
        "vault_path": "/private/tmp/jarvis-fallback/Vault",
        "vault_path_display": "workspace-local fallback notes",
        "reason": "primary_storage_not_writable",
        "exception_type": "OperationalError",
    }
    result = make_storage_status_tool(config, lambda: fallback)({})
    if not result.ok or result.tool_name != "storage_status":
        raise SystemExit(f"storage status tool failed: {result}")
    assert_no_local_path(result.output, "storage status output")
    assert_no_local_path(result.metadata, "storage status metadata")
    metadata = result.metadata
    if metadata.get("storage_runtime_fallback_active") is not True:
        raise SystemExit(f"storage status missed active fallback: {metadata}")
    if metadata.get("storage_status") != "needs attention":
        raise SystemExit(f"storage status should require attention while fallback is active: {metadata}")
    if metadata.get("storage_available") is not True:
        raise SystemExit(f"storage fallback should remain available but degraded: {metadata}")
    if metadata.get("storage_configured_available") is not False:
        raise SystemExit(f"storage status should keep configured storage unavailable in this fixture: {metadata}")
    if metadata.get("storage_ready_for_completion_claim") is not False:
        raise SystemExit(f"storage fallback should block completion readiness: {metadata}")
    if metadata.get("storage_readiness_blocks_completion_claim") is not True:
        raise SystemExit(f"storage status should expose completion-blocking storage readiness: {metadata}")
    if metadata.get("storage_readiness_blocker") != metadata.get("storage_recovery_reason"):
        raise SystemExit(f"storage status readiness blocker should mirror recovery reason: {metadata}")
    if metadata.get("storage_readiness_next_commands") != [
        "storage status",
        STORAGE_RECOVERY_PLAN_COMMAND,
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
        BOOTSTRAP_WRITE_COMMAND,
    ]:
        raise SystemExit(f"storage status missed completion-blocking recovery ladder: {metadata}")
    if metadata.get("storage_readiness_next_command_count") != len(metadata.get("storage_readiness_next_commands", [])):
        raise SystemExit(f"storage status readiness command count drifted: {metadata}")
    if metadata.get("storage_readiness_next_proof_command") != "storage status":
        raise SystemExit(f"storage status missed next storage readiness proof command: {metadata}")
    if metadata.get("storage_recovery_required") is not True:
        raise SystemExit(f"storage fallback should require recovery before completion claim: {metadata}")
    if "runtime is using workspace-local fallback storage" not in metadata.get("storage_recovery_reason", ""):
        raise SystemExit(f"storage fallback should explain recovery reason: {metadata}")
    if metadata.get("storage_recovery_mode") != "repair_configured_storage_then_restart_runtime":
        raise SystemExit(f"storage fallback should distinguish repair-plus-restart recovery mode: {metadata}")
    if metadata.get("storage_recovery_restart_required") is not True:
        raise SystemExit(f"storage fallback should require runtime restart after durable storage repair: {metadata}")
    if "writable durable storage" not in metadata.get("storage_recovery_next_operator_action", ""):
        raise SystemExit(f"storage fallback should expose next operator action: {metadata}")
    if "Recovery mode: repair_configured_storage_then_restart_runtime." not in result.output:
        raise SystemExit(f"storage status output should expose recovery mode: {result.output}")
    if "Next operator action: review the storage recovery plan" not in result.output:
        raise SystemExit(f"storage status output should expose next operator action: {result.output}")
    if metadata.get("storage_runtime_fallback_reason") != "primary_storage_not_writable":
        raise SystemExit(f"storage status missed fallback reason: {metadata}")
    if metadata.get("storage_runtime_fallback_db_path_display") != "workspace-local fallback database":
        raise SystemExit(f"storage status missed safe fallback DB display: {metadata}")
    if metadata.get("storage_runtime_fallback_vault_path_display") != "workspace-local fallback notes":
        raise SystemExit(f"storage status missed safe fallback notes display: {metadata}")
    issues = metadata.get("storage_issues") or []
    if "runtime is using workspace-local fallback storage" not in issues:
        raise SystemExit(f"storage status missed fallback attention issue: {metadata}")
    if metadata.get("storage_issue_count") != len(issues) or metadata.get("storage_issue_count") < 1:
        raise SystemExit(f"storage status issue count mismatch: {metadata}")
    if "- runtime is using workspace-local fallback storage" not in result.output:
        raise SystemExit(f"storage status output should list fallback as an issue: {result.output}")
    if "Recovery required before completion claim: yes" not in result.output:
        raise SystemExit(f"storage status output should expose required recovery gate: {result.output}")
    for key in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "reads_db_file_contents",
        "scans_obsidian_vault",
        "writes_files",
        "writes_database",
        "writes_notes",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"storage status should keep {key}=False: {metadata}")
    if "JARVIS_DB_PATH" not in metadata.get("storage_recovery_envs", []):
        raise SystemExit(f"storage status missed recovery envs: {metadata}")
    if metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
        raise SystemExit(f"storage status missed recovery check command: {metadata}")
    if metadata.get("storage_recovery_check_api") != STORAGE_RECOVERY_CHECK_API:
        raise SystemExit(f"storage status missed recovery check API: {metadata}")
    if metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
        raise SystemExit(f"storage status missed native recovery check command: {metadata}")
    expected_storage_proof_queue = [
        "storage status",
        STORAGE_RECOVERY_PLAN_COMMAND,
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
        BOOTSTRAP_WRITE_COMMAND,
    ]
    if metadata.get("storage_readiness_proof_queue") != expected_storage_proof_queue:
        raise SystemExit(f"storage status missed command-first storage proof queue: {metadata}")
    if metadata.get("storage_readiness_proof_queue_count") != len(expected_storage_proof_queue):
        raise SystemExit(f"storage status storage proof queue count drifted: {metadata}")
    if metadata.get("storage_readiness_proof_queue_preview") != expected_storage_proof_queue[:5]:
        raise SystemExit(f"storage status missed compact storage proof queue preview: {metadata}")
    if metadata.get("storage_readiness_proof_queue_preview_count") != len(expected_storage_proof_queue[:5]):
        raise SystemExit(f"storage status compact storage proof preview count drifted: {metadata}")
    if metadata.get("storage_readiness_proof_queue_remaining_count") != max(len(expected_storage_proof_queue) - len(expected_storage_proof_queue[:5]), 0):
        raise SystemExit(f"storage status compact storage proof remaining count drifted: {metadata}")
    if metadata.get("storage_readiness_first_proof_command") != "storage status":
        raise SystemExit(f"storage status missed first storage proof command: {metadata}")
    handoff = metadata.get("storage_status_handoff") or {}
    handoff_boundaries = handoff.get("boundaries") or {}
    assert_storage_metadata_only_boundaries(handoff_boundaries, "storage status handoff")
    if (
        handoff.get("source") != "storage_status"
        or metadata.get("storage_status_handoff_ready") is not True
        or handoff.get("storage_status_handoff_ready") is not True
        or metadata.get("storage_status_handoff_ready") != handoff.get("handoff_ready")
        or metadata.get("storage_status_ready_for_operator") is not True
        or handoff.get("ready_for_operator") is not True
        or metadata.get("storage_status_state_changed") is not False
        or handoff.get("state_changed") is not False
        or metadata.get("storage_status_changed") != []
        or handoff.get("changed") != []
        or metadata.get("storage_status_content_in_handoff") is not True
        or handoff.get("content_in_handoff") is not True
        or metadata.get("storage_status_authorizes_execution") is not False
        or metadata.get("storage_status_authorizes_completion_claim") is not False
        or metadata.get("storage_status_approval_granted") is not False
        or handoff.get("authorizes_execution") is not False
        or handoff.get("authorizes_completion_claim") is not False
        or handoff.get("approval_granted") is not False
        or metadata.get("storage_status_boundaries") != handoff_boundaries
        or handoff.get("storage_status") != metadata.get("storage_status")
        or handoff.get("storage_available") != metadata.get("storage_available")
        or handoff.get("storage_ready_for_completion_claim") != metadata.get("storage_ready_for_completion_claim")
        or handoff.get("storage_active_route") != metadata.get("storage_active_route")
        or handoff.get("storage_runtime_fallback_active") != metadata.get("storage_runtime_fallback_active")
        or handoff.get("storage_recovery_required") != metadata.get("storage_recovery_required")
        or handoff.get("storage_recovery_reason") != metadata.get("storage_recovery_reason")
        or handoff.get("storage_recovery_mode") != metadata.get("storage_recovery_mode")
        or handoff.get("storage_recovery_next_operator_action") != metadata.get("storage_recovery_next_operator_action")
        or handoff.get("storage_recovery_restart_required") != metadata.get("storage_recovery_restart_required")
        or handoff.get("storage_readiness_blocks_completion_claim") != metadata.get("storage_readiness_blocks_completion_claim")
        or handoff.get("storage_readiness_blocker") != metadata.get("storage_readiness_blocker")
        or handoff.get("storage_readiness_next_commands") != metadata.get("storage_readiness_next_commands")
        or handoff.get("storage_readiness_next_command_count") != metadata.get("storage_readiness_next_command_count")
        or handoff.get("storage_readiness_next_proof_command") != metadata.get("storage_readiness_next_proof_command")
        or handoff.get("storage_readiness_proof_queue") != metadata.get("storage_readiness_proof_queue")
        or handoff.get("storage_readiness_proof_queue_count") != metadata.get("storage_readiness_proof_queue_count")
        or handoff.get("storage_readiness_proof_queue_preview") != metadata.get("storage_readiness_proof_queue_preview")
        or handoff.get("storage_readiness_proof_queue_preview_count") != metadata.get("storage_readiness_proof_queue_preview_count")
        or handoff.get("storage_readiness_proof_queue_remaining_count") != metadata.get("storage_readiness_proof_queue_remaining_count")
        or handoff.get("storage_readiness_first_proof_command") != metadata.get("storage_readiness_first_proof_command")
        or handoff.get("storage_recovery_envs") != metadata.get("storage_recovery_envs")
        or handoff.get("next_commands") != metadata.get("next_commands")
        or handoff.get("issues") != metadata.get("storage_issues")
        or handoff.get("issue_count") != metadata.get("storage_issue_count")
        or handoff_boundaries.get("metadata_only") is not True
        or handoff_boundaries.get("reads_db_file_contents")
        or handoff_boundaries.get("scans_obsidian_vault")
        or handoff_boundaries.get("writes_files")
        or handoff_boundaries.get("writes_database")
        or handoff_boundaries.get("writes_notes")
        or handoff_boundaries.get("queues_approval")
        or handoff_boundaries.get("calls_model")
        or handoff_boundaries.get("calls_external_service")
        or handoff_boundaries.get("controls_computer")
        or handoff_boundaries.get("authorizes_execution")
        or handoff_boundaries.get("authorizes_completion_claim")
        or handoff_boundaries.get("approval_granted")
    ):
        raise SystemExit(f"storage status handoff diverged from flat metadata or safety contract: {metadata}")
    expected_proof_ladder = (
        "Proof ladder: `storage status`, `storage recovery plan`, `storage recovery check`, "
        f"`{BOOTSTRAP_CHECK_COMMAND}`, `{BOOTSTRAP_WRITE_COMMAND}`."
    )
    if expected_proof_ladder not in result.output:
        raise SystemExit(f"storage status should print command-first storage proof ladder: {result.output}")
    if metadata.get("next_commands", [])[:5] != [
        "storage status",
        STORAGE_RECOVERY_PLAN_COMMAND,
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
        BOOTSTRAP_WRITE_COMMAND,
    ]:
        raise SystemExit(f"storage status should run native and shell checks before bootstrap writes: {metadata}")
    if BOOTSTRAP_WRITE_COMMAND not in metadata.get("next_commands", []):
        raise SystemExit(f"storage status missed bootstrap next command: {metadata}")
    if f"Shell equivalent: `{BOOTSTRAP_CHECK_COMMAND}`" not in result.output:
        raise SystemExit(f"storage status should show shell equivalent for native check: {result.output}")


def test_storage_recovery_check_tool_reports_no_write_readiness() -> None:
    config = JarvisConfig(
        data_dir=Path("/\x55sers/example/Desktop/Claude code/storage-data"),
        db_path=Path("/var/folders/zc/jarvis-storage.sqlite"),
        obsidian_vault=Path("/tmp/jarvis-storage-vault"),
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    fallback = {
        "db_path": "/private/tmp/jarvis-fallback/jarvis.sqlite",
        "db_path_display": "workspace-local fallback database",
        "vault_path": "/private/tmp/jarvis-fallback/Vault",
        "vault_path_display": "workspace-local fallback notes",
        "reason": "primary_storage_not_writable",
        "exception_type": "OperationalError",
    }
    result = make_storage_recovery_check_tool(config, lambda: fallback)({})
    if not result.ok or result.tool_name != "storage_recovery_check":
        raise SystemExit(f"storage recovery check tool failed: {result}")
    assert_no_local_path(result.output, "storage recovery check output")
    assert_no_local_path(result.metadata, "storage recovery check metadata")
    metadata = result.metadata
    if metadata.get("storage_recovery_check_passed") is not False:
        raise SystemExit(f"storage recovery check should fail while fallback is active: {metadata}")
    if metadata.get("storage_status") != "needs attention":
        raise SystemExit(f"storage recovery check should expose storage status alias: {metadata}")
    if metadata.get("storage_available") is not True:
        raise SystemExit(f"storage recovery check should expose degraded storage availability: {metadata}")
    if metadata.get("storage_ready_for_completion_claim") is not False:
        raise SystemExit(f"storage recovery check should block completion-readiness alias: {metadata}")
    if metadata.get("storage_active_route") != "workspace-local fallback":
        raise SystemExit(f"storage recovery check should expose active storage route: {metadata}")
    if metadata.get("storage_recovery_required") is not True:
        raise SystemExit(f"storage recovery check should require recovery while fallback is active: {metadata}")
    if metadata.get("storage_recovery_reason") != metadata.get("storage_recovery_blocker"):
        raise SystemExit(f"storage recovery check should mirror blocker as recovery reason: {metadata}")
    if metadata.get("storage_recovery_mode") != "repair_configured_storage_then_restart_runtime":
        raise SystemExit(f"storage recovery check should distinguish repair-plus-restart mode: {metadata}")
    if metadata.get("storage_recovery_restart_required") is not True:
        raise SystemExit(f"storage recovery check should require runtime restart after fallback repair: {metadata}")
    if "writable durable storage" not in metadata.get("storage_recovery_next_operator_action", ""):
        raise SystemExit(f"storage recovery check should expose next operator action: {metadata}")
    if metadata.get("storage_readiness_blocks_completion_claim") is not True:
        raise SystemExit(f"storage recovery check should expose completion-blocking storage readiness: {metadata}")
    if metadata.get("storage_readiness_blocker") != metadata.get("storage_recovery_blocker"):
        raise SystemExit(f"storage recovery check readiness blocker should mirror blocker: {metadata}")
    if metadata.get("storage_readiness_next_commands") != [
        "storage status",
        STORAGE_RECOVERY_PLAN_COMMAND,
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
    ]:
        raise SystemExit(f"storage recovery check missed completion-blocking recovery ladder: {metadata}")
    if metadata.get("storage_readiness_next_command_count") != len(metadata.get("storage_readiness_next_commands", [])):
        raise SystemExit(f"storage recovery check readiness command count drifted: {metadata}")
    if metadata.get("storage_readiness_next_proof_command") != "storage status":
        raise SystemExit(f"storage recovery check missed next storage readiness proof command: {metadata}")
    expected_storage_proof_queue = [
        "storage status",
        STORAGE_RECOVERY_PLAN_COMMAND,
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
    ]
    if metadata.get("storage_readiness_proof_queue") != expected_storage_proof_queue:
        raise SystemExit(f"storage recovery check missed storage readiness proof queue: {metadata}")
    if metadata.get("storage_readiness_proof_queue_count") != len(expected_storage_proof_queue):
        raise SystemExit(f"storage recovery check proof queue count drifted: {metadata}")
    if metadata.get("storage_readiness_proof_queue_preview") != expected_storage_proof_queue[:5]:
        raise SystemExit(f"storage recovery check missed compact storage proof queue preview: {metadata}")
    if metadata.get("storage_readiness_proof_queue_preview_count") != len(expected_storage_proof_queue[:5]):
        raise SystemExit(f"storage recovery check compact storage proof preview count drifted: {metadata}")
    if metadata.get("storage_readiness_proof_queue_remaining_count") != max(len(expected_storage_proof_queue) - len(expected_storage_proof_queue[:5]), 0):
        raise SystemExit(f"storage recovery check compact storage proof remaining count drifted: {metadata}")
    if metadata.get("storage_readiness_first_proof_command") != "storage status":
        raise SystemExit(f"storage recovery check missed first proof command: {metadata}")
    if metadata.get("storage_runtime_fallback_active") is not True:
        raise SystemExit(f"storage recovery check missed active fallback: {metadata}")
    if "workspace-local fallback storage" not in metadata.get("storage_recovery_blocker", ""):
        raise SystemExit(f"storage recovery check missed fallback blocker: {metadata}")
    if metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
        raise SystemExit(f"storage recovery check missed native command metadata: {metadata}")
    if metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
        raise SystemExit(f"storage recovery check missed canonical shell check metadata: {metadata}")
    if metadata.get("storage_recovery_check_api") != STORAGE_RECOVERY_CHECK_API:
        raise SystemExit(f"storage recovery check missed recovery check API metadata: {metadata}")
    if metadata.get("storage_recovery_check_shell_command") != BOOTSTRAP_CHECK_COMMAND:
        raise SystemExit(f"storage recovery check missed shell check metadata: {metadata}")
    if metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
        raise SystemExit(f"storage recovery check missed bootstrap write metadata: {metadata}")
    handoff = metadata.get("storage_recovery_check_handoff") or {}
    handoff_boundaries = handoff.get("boundaries") or {}
    assert_storage_metadata_only_boundaries(handoff_boundaries, "storage recovery check handoff")
    if (
        handoff.get("source") != "storage_recovery_check"
        or metadata.get("storage_recovery_check_handoff_ready") is not True
        or handoff.get("storage_recovery_check_handoff_ready") is not True
        or metadata.get("storage_recovery_check_handoff_ready") != handoff.get("handoff_ready")
        or metadata.get("storage_recovery_check_ready_for_operator") is not True
        or handoff.get("ready_for_operator") is not True
        or metadata.get("storage_recovery_check_state_changed") is not False
        or handoff.get("state_changed") is not False
        or metadata.get("storage_recovery_check_changed") != []
        or handoff.get("changed") != []
        or metadata.get("storage_recovery_check_content_in_handoff") is not True
        or handoff.get("content_in_handoff") is not True
        or metadata.get("storage_recovery_check_authorizes_execution") is not False
        or metadata.get("storage_recovery_check_authorizes_completion_claim") is not False
        or metadata.get("storage_recovery_check_approval_granted") is not False
        or handoff.get("authorizes_execution") is not False
        or handoff.get("authorizes_completion_claim") is not False
        or handoff.get("approval_granted") is not False
        or metadata.get("storage_recovery_check_boundaries") != handoff_boundaries
        or handoff.get("storage_recovery_check_passed") != metadata.get("storage_recovery_check_passed")
        or handoff.get("storage_status") != metadata.get("storage_status")
        or handoff.get("storage_available") != metadata.get("storage_available")
        or handoff.get("storage_ready_for_completion_claim") != metadata.get("storage_ready_for_completion_claim")
        or handoff.get("storage_active_route") != metadata.get("storage_active_route")
        or handoff.get("storage_configured_available") != metadata.get("storage_configured_available")
        or handoff.get("storage_runtime_fallback_active") != metadata.get("storage_runtime_fallback_active")
        or handoff.get("storage_recovery_required") != metadata.get("storage_recovery_required")
        or handoff.get("storage_recovery_reason") != metadata.get("storage_recovery_reason")
        or handoff.get("storage_recovery_blocker") != metadata.get("storage_recovery_blocker")
        or handoff.get("storage_recovery_mode") != metadata.get("storage_recovery_mode")
        or handoff.get("storage_recovery_next_operator_action") != metadata.get("storage_recovery_next_operator_action")
        or handoff.get("storage_recovery_restart_required") != metadata.get("storage_recovery_restart_required")
        or handoff.get("storage_readiness_blocks_completion_claim") != metadata.get("storage_readiness_blocks_completion_claim")
        or handoff.get("storage_readiness_blocker") != metadata.get("storage_readiness_blocker")
        or handoff.get("storage_readiness_next_commands") != metadata.get("storage_readiness_next_commands")
        or handoff.get("storage_readiness_next_command_count") != metadata.get("storage_readiness_next_command_count")
        or handoff.get("storage_readiness_next_proof_command") != metadata.get("storage_readiness_next_proof_command")
        or handoff.get("storage_readiness_proof_queue") != metadata.get("storage_readiness_proof_queue")
        or handoff.get("storage_readiness_proof_queue_count") != metadata.get("storage_readiness_proof_queue_count")
        or handoff.get("storage_readiness_proof_queue_preview") != metadata.get("storage_readiness_proof_queue_preview")
        or handoff.get("storage_readiness_proof_queue_preview_count") != metadata.get("storage_readiness_proof_queue_preview_count")
        or handoff.get("storage_readiness_proof_queue_remaining_count") != metadata.get("storage_readiness_proof_queue_remaining_count")
        or handoff.get("storage_readiness_first_proof_command") != metadata.get("storage_readiness_first_proof_command")
        or handoff.get("storage_recovery_check_tool_command") != metadata.get("storage_recovery_check_tool_command")
        or handoff.get("storage_recovery_check_command") != metadata.get("storage_recovery_check_command")
        or handoff.get("storage_recovery_check_api") != metadata.get("storage_recovery_check_api")
        or handoff.get("storage_recovery_check_shell_command") != metadata.get("storage_recovery_check_shell_command")
        or handoff.get("storage_recovery_command") != metadata.get("storage_recovery_command")
        or handoff.get("next_commands") != metadata.get("next_commands")
        or handoff.get("issues") != metadata.get("storage_issues")
        or handoff.get("issue_count") != metadata.get("storage_issue_count")
        or handoff_boundaries.get("metadata_only") is not True
        or handoff_boundaries.get("reads_db_file_contents")
        or handoff_boundaries.get("scans_obsidian_vault")
        or handoff_boundaries.get("writes_files")
        or handoff_boundaries.get("writes_database")
        or handoff_boundaries.get("writes_notes")
        or handoff_boundaries.get("queues_approval")
        or handoff_boundaries.get("calls_model")
        or handoff_boundaries.get("calls_external_service")
        or handoff_boundaries.get("controls_computer")
        or handoff_boundaries.get("authorizes_execution")
        or handoff_boundaries.get("authorizes_completion_claim")
        or handoff_boundaries.get("approval_granted")
    ):
        raise SystemExit(f"storage recovery check handoff diverged from flat metadata or safety contract: {metadata}")
    if metadata.get("next_commands") != [
        "storage status",
        STORAGE_RECOVERY_PLAN_COMMAND,
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
        BOOTSTRAP_WRITE_COMMAND,
        "setup check",
    ]:
        raise SystemExit(f"storage recovery check should keep the recovery ladder visible after a failed check: {metadata}")
    for first, second in [
        ("storage status", STORAGE_RECOVERY_PLAN_COMMAND),
        (STORAGE_RECOVERY_PLAN_COMMAND, STORAGE_RECOVERY_CHECK_COMMAND),
        (STORAGE_RECOVERY_CHECK_COMMAND, BOOTSTRAP_CHECK_COMMAND),
        (BOOTSTRAP_CHECK_COMMAND, BOOTSTRAP_WRITE_COMMAND),
    ]:
        commands = metadata.get("next_commands") or []
        if commands.index(first) >= commands.index(second):
            raise SystemExit(f"storage recovery check command order drifted: {metadata}")
    for key in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "reads_db_file_contents",
        "scans_obsidian_vault",
        "writes_files",
        "writes_database",
        "writes_notes",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"storage recovery check should keep {key}=False: {metadata}")
    if "Jarvis storage recovery check:" not in result.output:
        raise SystemExit(f"storage recovery check missed title: {result.output}")
    if "recovery mode: repair_configured_storage_then_restart_runtime" not in result.output:
        raise SystemExit(f"storage recovery check should print recovery mode: {result.output}")
    if "next operator action: review the storage recovery plan" not in result.output:
        raise SystemExit(f"storage recovery check should print next operator action: {result.output}")
    if "does not create folders, initialize SQLite" not in result.output:
        raise SystemExit(f"storage recovery check missed no-write boundary: {result.output}")
    expected_proof_ladder = (
        "Proof ladder: `storage status`, `storage recovery plan`, `storage recovery check`, "
        f"`{BOOTSTRAP_CHECK_COMMAND}`."
    )
    if expected_proof_ladder not in result.output:
        raise SystemExit(f"storage recovery check missed proof ladder: {result.output}")
    for expected in [
        f"- `{STORAGE_RECOVERY_PLAN_COMMAND}`",
        f"- `{STORAGE_RECOVERY_CHECK_COMMAND}`",
        f"- `{BOOTSTRAP_CHECK_COMMAND}`",
        f"- `{BOOTSTRAP_WRITE_COMMAND}`",
    ]:
        if expected not in result.output:
            raise SystemExit(f"storage recovery check output missed recovery ladder command {expected}: {result.output}")


def test_storage_recovery_plan_tool_drafts_relative_env_without_writes() -> None:
    config = JarvisConfig(
        data_dir=Path("/\x55sers/example/Desktop/Claude code/storage-data"),
        db_path=Path("/var/folders/zc/jarvis-storage.sqlite"),
        obsidian_vault=Path("/tmp/jarvis-storage-vault"),
        obsidian_root="Jarvis",
        use_model_planner=False,
    )
    fallback = {
        "db_path": "/private/tmp/jarvis-fallback/jarvis.sqlite",
        "db_path_display": "workspace-local fallback database",
        "vault_path": "/private/tmp/jarvis-fallback/Vault",
        "vault_path_display": "workspace-local fallback notes",
        "reason": "primary_storage_not_writable",
        "exception_type": "OperationalError",
    }
    result = make_storage_recovery_plan_tool(config, lambda: fallback, lambda: Path("/private/tmp"))({})
    if not result.ok or result.tool_name != "storage_recovery_plan":
        raise SystemExit(f"storage recovery plan tool failed: {result}")
    assert_no_local_path(result.output, "storage recovery plan output")
    assert_no_local_path(result.metadata, "storage recovery plan metadata")
    metadata = result.metadata
    expected_exports = [
        'export JARVIS_DATA_DIR="$PWD/.jarvis_v3_durable"',
        'export JARVIS_DB_PATH="$PWD/.jarvis_v3_durable/jarvis.sqlite"',
        'export JARVIS_OBSIDIAN_VAULT="$PWD/.jarvis_v3_durable/Vault"',
    ]
    expected_commands = [
        STORAGE_RECOVERY_CHECK_COMMAND,
        BOOTSTRAP_CHECK_COMMAND,
        BOOTSTRAP_WRITE_COMMAND,
        "storage status",
    ]
    if metadata.get("storage_recovery_plan_ready") is not True:
        raise SystemExit(f"storage recovery plan should be ready when cwd is writable: {metadata}")
    if metadata.get("storage_recovery_plan_parent_exists") is not True:
        raise SystemExit(f"storage recovery plan should confirm parent exists: {metadata}")
    if metadata.get("storage_recovery_plan_parent_writable") is not True:
        raise SystemExit(f"storage recovery plan should confirm parent is writable: {metadata}")
    if metadata.get("storage_recovery_plan_data_dir") != ".jarvis_v3_durable":
        raise SystemExit(f"storage recovery plan should use relative durable data dir: {metadata}")
    if metadata.get("storage_recovery_plan_db_path") != ".jarvis_v3_durable/jarvis.sqlite":
        raise SystemExit(f"storage recovery plan should use relative durable DB path: {metadata}")
    if metadata.get("storage_recovery_plan_obsidian_vault") != ".jarvis_v3_durable/Vault":
        raise SystemExit(f"storage recovery plan should use relative durable notes path: {metadata}")
    if metadata.get("storage_recovery_plan_exports") != expected_exports:
        raise SystemExit(f"storage recovery plan export lines drifted: {metadata}")
    if metadata.get("storage_recovery_plan_next_commands") != expected_commands:
        raise SystemExit(f"storage recovery plan proof commands drifted: {metadata}")
    if metadata.get("storage_recovery_plan_next_command_count") != len(expected_commands):
        raise SystemExit(f"storage recovery plan command count drifted: {metadata}")
    if metadata.get("storage_recovery_plan_next_required_command") != STORAGE_RECOVERY_CHECK_COMMAND:
        raise SystemExit(f"storage recovery plan should require native recovery check first: {metadata}")
    if metadata.get("storage_recovery_plan_next_proof_command") != STORAGE_RECOVERY_CHECK_COMMAND:
        raise SystemExit(f"storage recovery plan should expose native recovery check as next proof: {metadata}")
    if metadata.get("storage_runtime_fallback_active") is not True:
        raise SystemExit(f"storage recovery plan should keep fallback state visible: {metadata}")
    if metadata.get("storage_configured_available") is not False:
        raise SystemExit(f"storage recovery plan should not treat broken configured storage as ready: {metadata}")
    if metadata.get("storage_recovery_required") is not True:
        raise SystemExit(f"storage recovery plan should show recovery required: {metadata}")
    if metadata.get("storage_recovery_check_tool_command") != STORAGE_RECOVERY_CHECK_COMMAND:
        raise SystemExit(f"storage recovery plan missed recovery check tool command: {metadata}")
    if metadata.get("storage_recovery_check_command") != BOOTSTRAP_CHECK_COMMAND:
        raise SystemExit(f"storage recovery plan missed bootstrap check command: {metadata}")
    if metadata.get("storage_recovery_command") != BOOTSTRAP_WRITE_COMMAND:
        raise SystemExit(f"storage recovery plan missed bootstrap write command: {metadata}")
    if metadata.get("next_commands") != expected_commands:
        raise SystemExit(f"storage recovery plan top-level next commands drifted: {metadata}")
    for key in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "reads_db_file_contents",
        "scans_obsidian_vault",
        "writes_files",
        "writes_database",
        "writes_notes",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"storage recovery plan should keep {key}=False: {metadata}")
    handoff = metadata.get("storage_recovery_plan_handoff") or {}
    handoff_boundaries = handoff.get("boundaries") or {}
    assert_storage_metadata_only_boundaries(handoff_boundaries, "storage recovery plan handoff")
    if (
        handoff.get("source") != "storage_recovery_plan"
        or metadata.get("storage_recovery_plan_handoff_ready") is not True
        or handoff.get("storage_recovery_plan_handoff_ready") is not True
        or metadata.get("storage_recovery_plan_handoff_ready") != handoff.get("handoff_ready")
        or metadata.get("storage_recovery_plan_ready_for_operator") is not True
        or handoff.get("ready_for_operator") is not True
        or metadata.get("storage_recovery_plan_state_changed") is not False
        or handoff.get("state_changed") is not False
        or metadata.get("storage_recovery_plan_changed") != []
        or handoff.get("changed") != []
        or metadata.get("storage_recovery_plan_content_in_handoff") is not True
        or handoff.get("content_in_handoff") is not True
        or metadata.get("storage_recovery_plan_authorizes_execution") is not False
        or metadata.get("storage_recovery_plan_authorizes_completion_claim") is not False
        or metadata.get("storage_recovery_plan_approval_granted") is not False
        or handoff.get("authorizes_execution") is not False
        or handoff.get("authorizes_completion_claim") is not False
        or handoff.get("approval_granted") is not False
        or metadata.get("storage_recovery_plan_boundaries") != handoff_boundaries
        or handoff.get("storage_recovery_plan_ready") != metadata.get("storage_recovery_plan_ready")
        or handoff.get("storage_recovery_plan_exports") != metadata.get("storage_recovery_plan_exports")
        or handoff.get("storage_recovery_plan_next_commands") != metadata.get("storage_recovery_plan_next_commands")
        or handoff.get("storage_recovery_plan_next_command_count") != metadata.get("storage_recovery_plan_next_command_count")
        or handoff.get("storage_recovery_plan_next_required_command") != metadata.get("storage_recovery_plan_next_required_command")
        or handoff.get("storage_recovery_plan_reason") != metadata.get("storage_recovery_plan_reason")
        or handoff.get("storage_runtime_fallback_active") != metadata.get("storage_runtime_fallback_active")
        or handoff.get("storage_configured_available") != metadata.get("storage_configured_available")
        or handoff.get("storage_configured_diagnostics_source") != metadata.get("storage_configured_diagnostics_source")
    ):
        raise SystemExit(f"storage recovery plan handoff diverged from flat metadata or safety contract: {metadata}")
    for expected in [
        "Jarvis storage recovery plan:",
        "$PWD/.jarvis_v3_durable",
        "plan ready to bootstrap: yes",
        "This is read-only",
        "This does not create the directory, initialize SQLite",
        f"- `{STORAGE_RECOVERY_CHECK_COMMAND}`",
        f"- `{BOOTSTRAP_CHECK_COMMAND}`",
        f"- `{BOOTSTRAP_WRITE_COMMAND}`",
    ]:
        if expected not in result.output:
            raise SystemExit(f"storage recovery plan output missed {expected!r}: {result.output}")


def test_storage_recovery_plan_planner_aliases_route_to_read_only_tool() -> None:
    planner = RuleBasedPlanner()
    for prompt in [
        STORAGE_RECOVERY_PLAN_COMMAND,
        "jarvis storage recovery plan",
        "storage repair plan",
        "memory storage repair plan",
        "durable storage plan",
        "durable storage recovery plan",
        "fix storage plan",
    ]:
        plan = planner.plan(prompt)
        if len(plan.actions) != 1 or plan.actions[0].tool_name != "storage_recovery_plan":
            raise SystemExit(f"storage recovery plan prompt did not route to read-only plan tool: {prompt!r} -> {plan}")


def test_storage_recovery_check_distinguishes_ready_configured_storage_with_runtime_fallback() -> None:
    with TemporaryDirectory(prefix="jarvis-storage-ready-fallback-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root / "data",
            db_path=root / "data" / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        config.data_dir.mkdir(parents=True)
        config.obsidian_vault.mkdir(parents=True)
        fallback = {
            "db_path": str(root / "fallback" / "jarvis.sqlite"),
            "db_path_display": "workspace-local fallback database",
            "vault_path": str(root / "fallback" / "Vault"),
            "vault_path_display": "workspace-local fallback notes",
            "reason": "primary_storage_not_writable",
            "exception_type": "OperationalError",
        }

        status_result = make_storage_status_tool(config, lambda: fallback)({})
        check_result = make_storage_recovery_check_tool(config, lambda: fallback)({})
        assert_no_local_path(status_result.output, "ready configured fallback storage status output")
        assert_no_local_path(status_result.metadata, "ready configured fallback storage status metadata")
        assert_no_local_path(check_result.output, "ready configured fallback recovery check output")
        assert_no_local_path(check_result.metadata, "ready configured fallback recovery check metadata")

        for label, result in [
            ("storage status", status_result),
            ("storage recovery check", check_result),
        ]:
            metadata = result.metadata
            if metadata.get("storage_status") != "needs attention":
                raise SystemExit(f"{label} should expose degraded storage status: {metadata}")
            if metadata.get("storage_available") is not True:
                raise SystemExit(f"{label} should expose available-but-degraded storage: {metadata}")
            if metadata.get("storage_ready_for_completion_claim") is not False:
                raise SystemExit(f"{label} should block completion readiness while fallback is active: {metadata}")
            if metadata.get("storage_active_route") != "workspace-local fallback":
                raise SystemExit(f"{label} should expose fallback as active route: {metadata}")
            if metadata.get("storage_configured_available") is not True:
                raise SystemExit(f"{label} should see configured storage ready: {metadata}")
            if metadata.get("storage_runtime_fallback_active") is not True:
                raise SystemExit(f"{label} should still report active runtime fallback: {metadata}")
            if metadata.get("storage_recovery_mode") != "restart_runtime_to_configured_storage":
                raise SystemExit(f"{label} missed restart-only recovery mode: {metadata}")
            if metadata.get("storage_recovery_restart_required") is not True:
                raise SystemExit(f"{label} should require restart/reload onto configured storage: {metadata}")
            next_action = metadata.get("storage_recovery_next_operator_action", "")
            if "restart or reload Jarvis" not in next_action or "`storage status`" not in next_action:
                raise SystemExit(f"{label} missed restart/reload next operator action: {metadata}")
            if "runtime is using workspace-local fallback storage" not in metadata.get("storage_recovery_reason", ""):
                raise SystemExit(f"{label} should keep fallback reason visible: {metadata}")
            if "restart_runtime_to_configured_storage" not in result.output:
                raise SystemExit(f"{label} should print restart recovery mode: {result.output}")
            if "restart or reload Jarvis" not in result.output:
                raise SystemExit(f"{label} should print restart next action: {result.output}")


def test_storage_recovery_check_uses_primary_diagnostics_during_runtime_fallback() -> None:
    with TemporaryDirectory(prefix="jarvis-storage-primary-diag-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root / "fallback-data",
            db_path=root / "fallback-data" / "jarvis.sqlite",
            obsidian_vault=root / "fallback-vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        config.data_dir.mkdir(parents=True)
        config.obsidian_vault.mkdir(parents=True)
        primary_diagnostics = {
            "available": False,
            "status": "needs attention",
            "data_dir": "<local-path>",
            "db_path": "<local-path>",
            "db_parent": "<local-path>",
            "db_exists": True,
            "data_dir_exists": True,
            "db_parent_exists": True,
            "data_dir_writable": False,
            "db_parent_writable": False,
            "db_file_writable": False,
            "obsidian_vault": "<local-path>",
            "obsidian_root": "Jarvis",
            "obsidian_root_path": "<local-path>",
            "obsidian_vault_exists": True,
            "obsidian_root_exists": True,
            "obsidian_vault_writable": False,
            "obsidian_root_writable": False,
            "workspace_local_notes": False,
            "metadata_only": True,
            "issues": [
                "database parent is not writable",
                "database file is not writable",
                "Obsidian vault is not writable",
            ],
            "recovery_check_command": BOOTSTRAP_CHECK_COMMAND,
            "recovery_command": BOOTSTRAP_WRITE_COMMAND,
        }
        fallback = {
            "db_path": str(root / "fallback" / "jarvis.sqlite"),
            "db_path_display": "workspace-local fallback database",
            "vault_path": str(root / "fallback" / "Vault"),
            "vault_path_display": "workspace-local fallback notes",
            "reason": "primary_storage_not_writable",
            "exception_type": "OperationalError",
            "primary_storage_diagnostics": primary_diagnostics,
        }

        status_result = make_storage_status_tool(config, lambda: fallback)({})
        check_result = make_storage_recovery_check_tool(config, lambda: fallback)({})
        for label, result in [
            ("storage status", status_result),
            ("storage recovery check", check_result),
        ]:
            assert_no_local_path(result.output, f"{label} primary diagnostics output")
            assert_no_local_path(result.metadata, f"{label} primary diagnostics metadata")
            metadata = result.metadata
            if metadata.get("storage_status") != "needs attention":
                raise SystemExit(f"{label} should expose primary storage status alias: {metadata}")
            if metadata.get("storage_available") is not True:
                raise SystemExit(f"{label} should expose fallback availability while primary is broken: {metadata}")
            if metadata.get("storage_ready_for_completion_claim") is not False:
                raise SystemExit(f"{label} should block completion readiness while fallback is active: {metadata}")
            if metadata.get("storage_active_route") != "workspace-local fallback":
                raise SystemExit(f"{label} should expose fallback route while primary diagnostics are used: {metadata}")
            if metadata.get("storage_configured_available") is not False:
                raise SystemExit(f"{label} should not treat fallback paths as configured storage: {metadata}")
            if metadata.get("storage_configured_status") != "needs attention":
                raise SystemExit(f"{label} should surface primary configured storage status: {metadata}")
            if metadata.get("storage_configured_diagnostics_source") != "primary_storage_before_fallback":
                raise SystemExit(f"{label} missed primary diagnostic source: {metadata}")
            if metadata.get("storage_recovery_mode") != "repair_configured_storage_then_restart_runtime":
                raise SystemExit(f"{label} should require repair before restart when primary storage is not writable: {metadata}")
            if BOOTSTRAP_CHECK_COMMAND not in metadata.get("storage_readiness_next_commands", []):
                raise SystemExit(f"{label} should keep shell parity check in the proof ladder: {metadata}")
            if BOOTSTRAP_WRITE_COMMAND not in metadata.get("next_commands", []):
                raise SystemExit(f"{label} should keep guarded bootstrap write after the check: {metadata}")


def main() -> None:
    test_startup_failure_receipt_redacts_config_paths()
    test_bootstrap_import_skips_path_shaped_legacy_records()
    test_legacy_bootstrap_sources_are_bounded_and_shape_checked()
    test_legacy_bootstrap_failures_drop_private_exception_context()
    test_legacy_bootstrap_close_failure_cannot_mask_primary_diagnostic()
    test_legacy_bootstrap_failure_receipt_is_content_free()
    test_bootstrap_occurrences_are_idempotent_content_addressed_and_private()
    test_bootstrap_duplicate_ordinals_are_stable_across_reordering()
    test_bootstrap_concurrent_claim_and_reservation_rollback()
    test_bootstrap_adopts_exact_legacy_rows_one_to_one()
    test_bootstrap_occurrence_custody_cascades_on_delete_merge_and_migration()
    test_bootstrap_projection_failure_and_crash_adoption_repair_same_id()
    test_bootstrap_summary_is_once_and_retains_reserved_date_after_crash()
    test_bootstrap_check_does_not_construct_storage_or_read_legacy_sources()
    test_storage_diagnostics_redacts_display_paths()
    test_readiness_report_uses_exact_storage_diagnostic_bools()
    test_bootstrap_check_reports_storage_readiness_without_writes()
    test_bootstrap_check_fails_closed_on_malformed_storage_bools()
    test_storage_status_tool_reports_runtime_fallback_without_paths()
    test_storage_recovery_check_tool_reports_no_write_readiness()
    test_storage_recovery_plan_tool_drafts_relative_env_without_writes()
    test_storage_recovery_plan_planner_aliases_route_to_read_only_tool()
    test_storage_recovery_check_distinguishes_ready_configured_storage_with_runtime_fallback()
    test_storage_recovery_check_uses_primary_diagnostics_during_runtime_fallback()
    print("bootstrap startup smoke passed")


if __name__ == "__main__":
    main()
