from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jarvis_v2.config import load_config
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import (
    MemoryRecord,
    MemoryStore,
    bootstrap_memory_occurrence_source_key,
)
from jarvis_v2.scripts.startup import is_startup_storage_error, print_startup_failure
from jarvis_v2.tools.storage import _metadata_bool, storage_diagnostics
from jarvis_v2.v3_commands import V3_BOOTSTRAP_CHECK_COMMAND, V3_BOOTSTRAP_WRITE_COMMAND


OLD_MEMORY = Path("~/.jarvis_ollama_memory.json").expanduser()
OLD_LIFE_MODEL = Path("~/.jarvis_life_model.json").expanduser()
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
BOOTSTRAP_CHECK_COMMAND = V3_BOOTSTRAP_CHECK_COMMAND
BOOTSTRAP_WRITE_COMMAND = V3_BOOTSTRAP_WRITE_COMMAND
BOOTSTRAP_MEMORY_SOURCE_KEY_PREFIX = "bootstrap-memory:v1:"
BOOTSTRAP_SUMMARY_SOURCE_KEY_PREFIX = "bootstrap-summary:v1:"
MAX_LEGACY_BOOTSTRAP_BYTES = 2_000_000


class LegacyBootstrapSourceError(RuntimeError):
    def __init__(self, source: str, code: str):
        self.source = source
        self.code = code
        super().__init__(f"{source}:{code}")


@dataclass(frozen=True)
class BootstrapImportItem:
    record: MemoryRecord
    duplicate_ordinal: int
    source_key: str


def _canonical_bootstrap_record_payload(
    record: MemoryRecord,
    *,
    duplicate_ordinal: int | None,
) -> bytes:
    if not isinstance(record, MemoryRecord):
        raise TypeError("bootstrap import record is invalid")
    if type(duplicate_ordinal) is not int and duplicate_ordinal is not None:
        raise TypeError("bootstrap duplicate ordinal must be an integer")
    if duplicate_ordinal is not None and duplicate_ordinal < 0:
        raise ValueError("bootstrap duplicate ordinal must not be negative")
    payload: dict[str, object] = {
        "body": record.body,
        "category": record.category,
        "confidence": record.confidence,
        "source": record.source,
        "title": record.title,
    }
    if duplicate_ordinal is not None:
        payload["duplicate_ordinal"] = duplicate_ordinal
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def bootstrap_memory_source_key(record: MemoryRecord, duplicate_ordinal: int) -> str:
    return bootstrap_memory_occurrence_source_key(record, duplicate_ordinal)


def build_bootstrap_import_items(records: list[MemoryRecord]) -> list[BootstrapImportItem]:
    duplicate_counts: dict[bytes, int] = {}
    items: list[BootstrapImportItem] = []
    for record in records:
        identity = _canonical_bootstrap_record_payload(record, duplicate_ordinal=None)
        ordinal = duplicate_counts.get(identity, 0)
        duplicate_counts[identity] = ordinal + 1
        items.append(
            BootstrapImportItem(
                record=record,
                duplicate_ordinal=ordinal,
                source_key=bootstrap_memory_source_key(record, ordinal),
            )
        )
    return items


def bootstrap_summary_source_key(items: list[BootstrapImportItem], skipped: int) -> str:
    if type(skipped) is not int or skipped < 0:
        raise ValueError("bootstrap skipped count must not be negative")
    payload = json.dumps(
        {
            "occurrence_keys": sorted(item.source_key for item in items),
            "skipped": skipped,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return BOOTSTRAP_SUMMARY_SOURCE_KEY_PREFIX + hashlib.sha256(payload).hexdigest()


def safe_setup_text(value: object) -> str:
    return LOCAL_PATH_RE.sub("<local-path>", str(value or ""))


def record_has_local_path(record: MemoryRecord) -> bool:
    return any(
        LOCAL_PATH_RE.search(str(value or ""))
        for value in (record.category, record.title, record.body, record.source)
    )


def filter_importable_records(records: list[MemoryRecord]) -> tuple[list[MemoryRecord], int]:
    importable = [record for record in records if not record_has_local_path(record)]
    return importable, len(records) - len(importable)


def _legacy_json_object(path: Path, *, source: str) -> dict[str, object] | None:
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    missing = False
    open_failed = False
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        missing = True
    except OSError:
        open_failed = True
    if missing:
        return None
    if open_failed or descriptor is None:
        raise LegacyBootstrapSourceError(source, "unreadable") from None

    try:
        source_stat: os.stat_result | None = None
        stat_failed = False
        try:
            source_stat = os.fstat(descriptor)
        except OSError:
            stat_failed = True
        if stat_failed or source_stat is None:
            raise LegacyBootstrapSourceError(source, "unreadable") from None
        if not stat.S_ISREG(source_stat.st_mode):
            raise LegacyBootstrapSourceError(source, "not_regular")
        if source_stat.st_size < 0 or source_stat.st_size > MAX_LEGACY_BOOTSTRAP_BYTES:
            raise LegacyBootstrapSourceError(source, "oversized")
        chunks: list[bytes] = []
        remaining = MAX_LEGACY_BOOTSTRAP_BYTES + 1
        read_failed = False
        try:
            while remaining > 0:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        except OSError:
            read_failed = True
        if read_failed:
            raise LegacyBootstrapSourceError(source, "unreadable") from None

        raw = b"".join(chunks)
        if len(raw) > MAX_LEGACY_BOOTSTRAP_BYTES:
            raise LegacyBootstrapSourceError(source, "oversized")
        parse_failed = False
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            parse_failed = True
            data = None
        if parse_failed:
            raise LegacyBootstrapSourceError(source, "invalid_json") from None
        if not isinstance(data, dict):
            raise LegacyBootstrapSourceError(source, "invalid_root")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise

    close_failed = False
    try:
        os.close(descriptor)
    except OSError:
        close_failed = True
    if close_failed:
        raise LegacyBootstrapSourceError(source, "unreadable") from None
    return data


def _legacy_mapping(
    data: dict[str, object], key: str, *, source: str
) -> dict[object, object]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise LegacyBootstrapSourceError(source, f"invalid_{key}")
    return value


def _legacy_list(data: dict[str, object], key: str, *, source: str) -> list[object]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise LegacyBootstrapSourceError(source, f"invalid_{key}")
    return value


def bootstrap_check_lines() -> tuple[list[str], int]:
    config = load_config()
    diagnostics = storage_diagnostics(config)
    issues = list(diagnostics.get("issues") or [])
    ready = _metadata_bool(diagnostics.get("available"))
    lines = [
        "Jarvis bootstrap storage check:",
        f"- configured storage ready: {'yes' if ready else 'no'}",
        f"- database display: `{diagnostics.get('db_path', '')}`",
        f"- database parent writable: {'yes' if _metadata_bool(diagnostics.get('db_parent_writable')) else 'no'}",
        f"- database file writable: {'yes' if _metadata_bool(diagnostics.get('db_file_writable')) else 'no'}",
        f"- note vault display: `{diagnostics.get('obsidian_vault', '')}`",
        f"- note vault writable: {'yes' if _metadata_bool(diagnostics.get('obsidian_vault_writable')) else 'no'}",
        f"- note root display: `{diagnostics.get('obsidian_root_path', '')}`",
        f"- metadata-only check: {'yes' if _metadata_bool(diagnostics.get('metadata_only')) else 'no'}",
        "",
        "Issues:",
    ]
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Next:",
            f"- Run `{BOOTSTRAP_WRITE_COMMAND}` only after this check reports configured storage ready.",
            "",
            "Boundary:",
            "- This check is read-only and metadata-only. It does not create folders, write files, initialize SQLite, scan note contents, read legacy memory contents, import memories, queue approvals, call models, call external services, or control the computer.",
        ]
    )
    return lines, 0 if ready else 2


def records_from_old_memory() -> list[MemoryRecord]:
    data = _legacy_json_object(OLD_MEMORY, source="old_memory")
    if data is None:
        return []
    records: list[MemoryRecord] = []

    for key, value in _legacy_mapping(data, "facts", source="old_memory").items():
        body = value.get("value", value) if isinstance(value, dict) else value
        records.append(MemoryRecord("facts", str(key), str(body), "jarvis-ollama-import"))

    for note in _legacy_list(data, "notes", source="old_memory"):
        if not isinstance(note, dict):
            raise LegacyBootstrapSourceError("old_memory", "invalid_notes")
        records.append(
            MemoryRecord(
                "notes",
                str(note.get("title", "Imported note")),
                str(note.get("content", "")),
                "jarvis-ollama-import",
            )
        )

    for name, person in _legacy_mapping(data, "people", source="old_memory").items():
        details = person.get("details", person) if isinstance(person, dict) else person
        records.append(MemoryRecord("people", str(name), str(details), "jarvis-ollama-import"))

    for category, value in _legacy_mapping(
        data, "preferences", source="old_memory"
    ).items():
        records.append(MemoryRecord("preferences", str(category), str(value), "jarvis-ollama-import"))

    for idea in _legacy_list(data, "ideas", source="old_memory"):
        title = str(idea.get("idea", "Imported idea"))[:60] if isinstance(idea, dict) else str(idea)[:60]
        body = str(idea.get("idea", idea)) if isinstance(idea, dict) else str(idea)
        records.append(MemoryRecord("ideas", title, body, "jarvis-ollama-import"))

    return records


def records_from_life_model() -> list[MemoryRecord]:
    data = _legacy_json_object(OLD_LIFE_MODEL, source="life_model")
    if data is None:
        return []
    records: list[MemoryRecord] = []

    for key, value in _legacy_mapping(data, "identity", source="life_model").items():
        records.append(MemoryRecord("identity", str(key), str(value), "life-model-import"))

    if data.get("schedule"):
        records.append(MemoryRecord("routine", "Schedule", str(data["schedule"]), "life-model-import"))

    for name, details in _legacy_mapping(data, "people", source="life_model").items():
        records.append(MemoryRecord("people", str(name), str(details), "life-model-import"))

    for project in _legacy_list(data, "projects", source="life_model"):
        records.append(MemoryRecord("projects", str(project)[:80], str(project), "life-model-import"))

    for key, value in _legacy_mapping(
        data, "preferences", source="life_model"
    ).items():
        records.append(MemoryRecord("preferences", str(key), str(value), "life-model-import"))

    if data.get("context"):
        records.append(MemoryRecord("current context", "Current Context", str(data["context"]), "life-model-import"))

    return records


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Initialize or check Jarvis V2 memory storage.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configured storage readiness without creating folders, initializing SQLite, or importing legacy memory.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.check:
        lines, code = bootstrap_check_lines()
        print("\n".join(lines))
        raise SystemExit(code)

    config = load_config()
    store = MemoryStore(config.db_path)
    vault = ObsidianVault(config.obsidian_vault, config.obsidian_root)

    try:
        store.init()
        vault.init()
    except Exception as exc:
        if not is_startup_storage_error(exc):
            raise
        print_startup_failure(exc, program="Jarvis bootstrap")
        raise SystemExit(3) from exc

    try:
        records, skipped = filter_importable_records(
            records_from_old_memory() + records_from_life_model()
        )
    except LegacyBootstrapSourceError as exc:
        print(
            "Jarvis bootstrap could not safely read a legacy memory source.\n"
            f"Diagnostic: source={exc.source}; code={exc.code}.\n"
            "The source was not imported and no source content was displayed."
        )
        raise SystemExit(5) from exc
    items = build_bootstrap_import_items(records)
    claim_counts = {"created": 0, "adopted": 0, "existing": 0}
    for item in items:
        claim = store.claim_bootstrap_memory_occurrence(
            item.record, item.source_key, item.duplicate_ordinal
        )
        if claim.status not in claim_counts:
            raise RuntimeError("Jarvis bootstrap received an invalid occurrence claim status.")
        claim_counts[claim.status] += 1
        projection_target = claim.projection_target
        projection = reconcile_memory_projection(
            store,
            vault,
            projection_target.memory_id,
            expected_operation=projection_target.operation,
            expected_revision=projection_target.revision,
            expected_source_digest=projection_target.source_digest,
        )
        if projection.status != "completed":
            print(
                "Jarvis bootstrap saved a memory but could not finish its note projection. "
                "Run this bootstrap again or use `repair memory projections`."
            )
            raise SystemExit(4)

    summary_key = bootstrap_summary_source_key(items, skipped)
    summary = store.reserve_bootstrap_summary_run(
        summary_key,
        datetime.now().strftime("%Y-%m-%d"),
    )
    if summary.state == "pending":
        vault.append_daily_once_for_date(
            summary.target_date,
            summary.source_key,
            "Jarvis V2 Bootstrap",
            (
                "Bootstrap occurrence import converged.\n\n"
                f"Memory occurrences: {len(items)}.\n\n"
                f"Skipped local-path-shaped memories: {skipped}."
            ),
        )
        if not store.complete_bootstrap_summary_run(summary.source_key, summary.target_date):
            print(
                "Jarvis bootstrap wrote its daily summary but could not finalize custody. "
                "Run this bootstrap again."
            )
            raise SystemExit(5)
    print(f"Initialized Jarvis V2 at {safe_setup_text(config.db_path)}")
    print(f"Obsidian root: {safe_setup_text(vault.root_path)}")
    print(f"Imported memories: {len(items)}")
    print(f"Created memories: {claim_counts['created']}")
    print(f"Adopted legacy memories: {claim_counts['adopted']}")
    print(f"Existing memories: {claim_counts['existing']}")
    print(f"Skipped local-path-shaped memories: {skipped}")


if __name__ == "__main__":
    main()
