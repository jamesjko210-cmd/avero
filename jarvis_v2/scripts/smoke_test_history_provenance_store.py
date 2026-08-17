from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from jarvis_v2.memory.store import (
    HISTORY_DISCLOSURE_STALE_AFTER_SECONDS,
    MemoryRecord,
    MemoryStore,
    SkillRecord,
    history_message_content_digest,
    history_source_digest,
)


FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64
LINEAGE_A = "1" * 64
LINEAGE_B = "2" * 64
CONTENT_A = history_message_content_digest("user", "source-a")
CONTENT_B = history_message_content_digest("assistant", "source-b")
SOURCE_DIGEST = history_source_digest(
    [(1, LINEAGE_A, CONTENT_A), (2, LINEAGE_B, CONTENT_B)]
)


def _bound_digest(store: MemoryStore, sources: list[tuple[int, str]]) -> str:
    evidence: list[tuple[int, str, str]] = []
    for message_id, token in sources:
        provenance = store.read_message_provenance(message_id)
        if provenance is None or type(provenance.get("content_digest")) is not str:
            raise SystemExit("content-bound message evidence is unavailable")
        evidence.append((message_id, token, str(provenance["content_digest"])))
    return history_source_digest(evidence)


def _expect_error(error: type[BaseException], action: Callable[[], object], label: str) -> None:
    try:
        action()
    except error:
        return
    except BaseException as exc:
        raise SystemExit(f"{label} raised {type(exc).__name__}, expected {error.__name__}") from exc
    raise SystemExit(f"{label} did not fail closed")


def _activate_remote_epoch(store: MemoryStore, fingerprint: str = FINGERPRINT_A) -> str:
    return store.ensure_active_history_policy_epoch(
        policy_fingerprint=fingerprint,
        provider="openai",
        model_identifier="gpt-5.6-terra",
        destination_class="openai-api",
        session_generation=1,
        explicit_consent_satisfied=True,
    )


def _create_legacy_database(path: Path) -> None:
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.executescript(
                """
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                INSERT INTO messages(session_id, role, content, created_at)
                VALUES ('legacy-session', 'user', 'LEGACY PRIVATE CONTENT', '2026-01-01T00:00:00Z');

                CREATE TABLE conversation_compaction_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_key TEXT NOT NULL UNIQUE,
                    first_message_id INTEGER NOT NULL CHECK(first_message_id > 0),
                    last_message_id INTEGER NOT NULL CHECK(last_message_id >= first_message_id),
                    outcome TEXT NOT NULL,
                    memory_id INTEGER,
                    mirror_state TEXT NOT NULL DEFAULT 'not_needed',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(first_message_id, last_message_id)
                );
                INSERT INTO conversation_compaction_batches(
                    batch_key, first_message_id, last_message_id, outcome,
                    memory_id, mirror_state, created_at, updated_at
                ) VALUES (
                    'legacy:1', 1, 1, 'nothing_durable', NULL, 'not_needed',
                    '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'
                );
                """
            )


def _create_just_added_provenance_database(path: Path) -> None:
    epoch_id = "e" * 64
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.executescript(
                f"""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{{}}',
                    created_at TEXT NOT NULL
                );
                INSERT INTO messages(session_id, role, content, metadata, created_at)
                VALUES
                    ('migration-safe', 'user', 'OLD USER PRIVATE CONTENT', '{{}}', '2026-01-01T00:00:00Z'),
                    ('migration-safe', 'assistant', 'OLD ASSISTANT PRIVATE CONTENT', '{{}}', '2026-01-01T00:00:01Z');

                CREATE TABLE history_policy_epochs (
                    epoch_id TEXT PRIMARY KEY,
                    policy_fingerprint TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_identifier TEXT NOT NULL,
                    destination_class TEXT NOT NULL,
                    session_generation INTEGER NOT NULL,
                    explicit_consent_satisfied INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                INSERT INTO history_policy_epochs(
                    epoch_id, policy_fingerprint, provider, model_identifier,
                    destination_class, session_generation,
                    explicit_consent_satisfied, created_at
                ) VALUES (
                    '{epoch_id}', '{FINGERPRINT_A}', 'openai', 'gpt-5.6-terra',
                    'openai-api', 1, 1, '2026-01-01T00:00:00Z'
                );

                CREATE TABLE active_history_policy_epoch (
                    singleton_id INTEGER PRIMARY KEY,
                    active_epoch_id TEXT NOT NULL UNIQUE,
                    activated_at TEXT NOT NULL
                );
                INSERT INTO active_history_policy_epoch(singleton_id, active_epoch_id, activated_at)
                VALUES (1, '{epoch_id}', '2026-01-01T00:00:00Z');

                CREATE TABLE message_provenance (
                    message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    policy_epoch_id TEXT NOT NULL REFERENCES history_policy_epochs(epoch_id),
                    lineage_token TEXT NOT NULL UNIQUE,
                    destination_class TEXT NOT NULL,
                    remote_eligible INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                INSERT INTO message_provenance(
                    message_id, role, policy_epoch_id, lineage_token,
                    destination_class, remote_eligible, created_at
                ) VALUES
                    (1, 'user', '{epoch_id}', '{"3" * 64}', 'openai-api', 1, '2026-01-01T00:00:00Z'),
                    (2, 'assistant', '{epoch_id}', '{"4" * 64}', 'openai-api', 1, '2026-01-01T00:00:01Z');

                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO memories(category, title, body, source, confidence, created_at, updated_at)
                VALUES ('history', 'OLD MEMORY TITLE', 'OLD MEMORY PRIVATE BODY', 'legacy', 1.0,
                        '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');

                CREATE TABLE memory_provenance (
                    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                    lineage_state TEXT NOT NULL,
                    source_count INTEGER NOT NULL,
                    source_digest TEXT NOT NULL,
                    policy_epoch_id TEXT NOT NULL REFERENCES history_policy_epochs(epoch_id),
                    destination_class TEXT NOT NULL,
                    remote_eligible INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                INSERT INTO memory_provenance(
                    memory_id, lineage_state, source_count, source_digest,
                    policy_epoch_id, destination_class, remote_eligible, created_at
                ) VALUES (1, 'complete', 1, '{"5" * 64}', '{epoch_id}',
                          'openai-api', 1, '2026-01-01T00:00:00Z');
                """
            )


def test_clean_database_and_policy_rotation() -> None:
    with TemporaryDirectory(prefix="jarvis-history-provenance-clean-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        bootstrap = store.read_active_history_epoch()
        if re.fullmatch(r"[0-9a-f]{64}", bootstrap["epoch_id"]) is None:
            raise SystemExit(f"bootstrap epoch id is not opaque: {bootstrap}")
        expected_columns = {
            "policy_fingerprint",
            "provider",
            "model_identifier",
            "destination_class",
            "session_generation",
            "explicit_consent_satisfied",
        }
        with store.connect() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(history_policy_epochs)")}
        if not expected_columns.issubset(columns):
            raise SystemExit(f"history policy epoch columns are incomplete: {columns}")

        epoch_a = _activate_remote_epoch(store)
        if _activate_remote_epoch(store) != epoch_a:
            raise SystemExit("an unchanged active policy tuple minted an unnecessary epoch")
        epoch_b = store.ensure_active_history_policy_epoch(
            policy_fingerprint=FINGERPRINT_B,
            provider="openai",
            model_identifier="gpt-5.6-terra",
            destination_class="openai-api",
            session_generation=1,
            explicit_consent_satisfied=True,
        )
        epoch_a_again = store.ensure_active_history_policy_epoch(
            policy_fingerprint=FINGERPRINT_A,
            provider="openai",
            model_identifier="gpt-5.6-terra",
            destination_class="openai-api",
            session_generation=1,
            explicit_consent_satisfied=True,
        )
        if len({epoch_a, epoch_b, epoch_a_again}) != 3:
            raise SystemExit("returning to an old policy fingerprint reactivated an old epoch")
        active = store.read_active_history_epoch()
        if active["epoch_id"] != epoch_a_again or active["session_generation"] != 1:
            raise SystemExit(f"active history policy pointer drifted: {active}")

        with store.connect() as conn:
            _expect_error(
                sqlite3.IntegrityError,
                lambda: conn.execute(
                    "UPDATE history_policy_epochs SET provider = 'local' WHERE epoch_id = ?",
                    (epoch_a,),
                ),
                "immutable epoch update",
            )
            _expect_error(
                sqlite3.IntegrityError,
                lambda: conn.execute("DELETE FROM history_policy_epochs WHERE epoch_id = ?", (epoch_a,)),
                "immutable epoch delete",
            )


def test_legacy_migration_is_local_only() -> None:
    with TemporaryDirectory(prefix="jarvis-history-provenance-legacy-") as temp:
        path = Path(temp) / "legacy.sqlite"
        _create_legacy_database(path)
        store = MemoryStore(path)
        store.init()
        message = store.get_message(1)
        if message is None or message["content"] != "LEGACY PRIVATE CONTENT" or message["metadata"] != "{}":
            raise SystemExit(f"legacy message behavior changed during migration: {message}")
        provenance = store.read_message_provenance(1)
        if provenance is None or provenance["provenance_state"] != "legacy_unknown":
            raise SystemExit(f"legacy message was not classified explicitly: {provenance}")
        if provenance["destination_class"] != "local-only" or provenance["remote_eligible"]:
            raise SystemExit(f"legacy message became remotely eligible: {provenance}")
        with store.connect() as conn:
            batch = conn.execute("SELECT * FROM conversation_compaction_batches WHERE id = 1").fetchone()
            foreign_keys = list(conn.execute("PRAGMA foreign_key_list(message_provenance)"))
        if (
            batch is None
            or batch["eligible_source_count"] != 0
            or batch["withheld_source_count"] != 0
            or batch["source_provenance_digest"] is not None
            or batch["policy_epoch_id"] is not None
        ):
            raise SystemExit(f"legacy compaction migration did not use conservative defaults: {dict(batch or {})}")
        if {row["table"] for row in foreign_keys} != {"messages", "history_policy_epochs"}:
            raise SystemExit(f"message provenance foreign keys are incomplete: {foreign_keys}")


def test_just_added_provenance_schema_migrates_transitive_lineage() -> None:
    with TemporaryDirectory(prefix="jarvis-history-provenance-transitive-migration-") as temp:
        path = Path(temp) / "just-added.sqlite"
        _create_just_added_provenance_database(path)
        store = MemoryStore(path)
        store.init()
        user = store.read_message_provenance(1)
        assistant = store.read_message_provenance(2)
        memory = store.read_memory_provenance(1)
        if (
            user is None
            or user["lineage_state"] != "direct_current"
            or user["source_count"] != 0
            or user["source_digest"] is not None
            or not user["lineage_complete"]
            or user["content_matches"]
            or user["remote_eligible"]
        ):
            raise SystemExit(f"old user provenance migration drifted: {user}")
        if (
            assistant is None
            or assistant["lineage_state"] != "mixed_restricted"
            or assistant["source_count"] != 0
            or assistant["source_digest"] is not None
            or assistant["lineage_complete"]
            or assistant["recorded_remote_eligible"]
            or assistant["remote_eligible"]
        ):
            raise SystemExit(f"old assistant provenance migration did not fail closed: {assistant}")
        if (
            memory is None
            or memory["lineage_state"] != "complete"
            or memory["content_matches"]
            or memory["recorded_remote_eligible"]
            or memory["remote_eligible"]
            or memory["invalidated_at"] is None
        ):
            raise SystemExit(f"old memory provenance migration was not local-only: {memory}")
        with store.connect() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(message_provenance)")}
            rows = [dict(row) for row in conn.execute("SELECT * FROM message_provenance ORDER BY message_id")]
        if not {"lineage_state", "source_count", "source_digest", "content_digest"}.issubset(columns):
            raise SystemExit(f"transitive message lineage columns were not migrated: {columns}")
        serialized = json.dumps(rows, sort_keys=True)
        if "OLD USER PRIVATE CONTENT" in serialized or "OLD ASSISTANT PRIVATE CONTENT" in serialized:
            raise SystemExit("transitive lineage migration retained message content")


def test_strict_validation_and_atomic_message_logging() -> None:
    with TemporaryDirectory(prefix="jarvis-history-provenance-validation-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        _expect_error(
            ValueError,
            lambda: store.start_history_policy_epoch(
                policy_fingerprint="not-a-digest",
                provider="openai",
                model_identifier="gpt-5",
                destination_class="openai-api",
                session_generation=1,
                explicit_consent_satisfied=True,
            ),
            "malformed policy fingerprint",
        )
        for bad_policy in (
            {"provider": "open*", "model_identifier": "gpt-5", "session_generation": 1, "explicit_consent_satisfied": True},
            {"provider": "openai", "model_identifier": "model with spaces", "session_generation": 1, "explicit_consent_satisfied": True},
            {"provider": "openai", "model_identifier": "m" * 129, "session_generation": 1, "explicit_consent_satisfied": True},
            {"provider": "openai", "model_identifier": "gpt-5", "session_generation": True, "explicit_consent_satisfied": True},
            {"provider": "openai", "model_identifier": "gpt-5", "session_generation": 1, "explicit_consent_satisfied": 1},
        ):
            _expect_error(
                (TypeError if bad_policy["explicit_consent_satisfied"] == 1 and type(bad_policy["explicit_consent_satisfied"]) is int else ValueError),
                lambda values=bad_policy: store.start_history_policy_epoch(
                    policy_fingerprint=FINGERPRINT_A,
                    destination_class="openai-api",
                    **values,
                ),
                "malformed policy tuple",
            )
        _expect_error(
            ValueError,
            lambda: history_source_digest([(True, LINEAGE_A, CONTENT_A)]),
            "boolean source id",
        )
        _expect_error(
            ValueError,
            lambda: history_source_digest([(1, "private text", CONTENT_A)]),
            "content lineage token",
        )
        _expect_error(
            ValueError,
            lambda: history_source_digest(
                [(1, LINEAGE_A, CONTENT_A), (1, LINEAGE_B, CONTENT_B)]
            ),
            "duplicate source id",
        )
        if (
            history_source_digest(
                [2, 1], [LINEAGE_B, LINEAGE_A], [CONTENT_B, CONTENT_A]
            )
            != SOURCE_DIGEST
        ):
            raise SystemExit("source provenance digest is not canonical")

        epoch = _activate_remote_epoch(store)
        _expect_error(
            ValueError,
            lambda: store.finalize_history_disclosure_receipt("f" * 64, "approved"),
            "unknown disclosure receipt state",
        )
        _expect_error(
            ValueError,
            lambda: store.prepare_history_disclosure_receipt(
                session_id="session-safe",
                provider="openai",
                destination_class="openai-api",
                source_count=True,
                source_digest=SOURCE_DIGEST,
                policy_epoch_id=epoch,
            ),
            "boolean disclosure source count",
        )
        _expect_error(
            ValueError,
            lambda: store.prepare_history_disclosure_receipt(
                session_id="session-safe",
                provider="openai",
                destination_class="openai-api",
                source_count=2,
                source_digest="malformed",
                policy_epoch_id=epoch,
            ),
            "malformed disclosure source digest",
        )
        secret = "MESSAGE CONTENT MUST NEVER ENTER PROVENANCE"
        invalid_cases = (
            {"role": "system", "destination_class": "openai-api", "remote_eligible": True},
            {"role": "user", "destination_class": "*", "remote_eligible": True},
            {"role": "user", "destination_class": "openai-api", "remote_eligible": 1},
            {"role": "user", "destination_class": "openai-api", "remote_eligible": True, "unknown": "x"},
            {"role": "user", "destination_class": "openai-api", "remote_eligible": True, "policy_epoch_id": "bad"},
            {"role": "user", "destination_class": "openai-api", "remote_eligible": True, "lineage_state": "unknown"},
            {
                "role": "user",
                "destination_class": "openai-api",
                "remote_eligible": True,
                "lineage_state": "direct_current",
                "source_count": 1,
                "source_digest": SOURCE_DIGEST,
            },
            {
                "role": "assistant",
                "destination_class": "openai-api",
                "remote_eligible": True,
                "lineage_state": "personal_derived",
                "source_count": 0,
            },
            {
                "role": "assistant",
                "destination_class": "openai-api",
                "remote_eligible": True,
                "lineage_state": "tool_derived",
                "source_count": 1,
                "source_digest": "malformed",
            },
        )
        for provenance in invalid_cases:
            _expect_error(
                (TypeError if provenance.get("remote_eligible") == 1 and type(provenance.get("remote_eligible")) is int else ValueError),
                lambda value=provenance: store.log_message(
                    "session-safe", "user", secret, {"kept": True}, provenance=value
                ),
                "invalid atomic message provenance",
            )
        with store.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        if count != 0:
            raise SystemExit("invalid provenance left a partially logged message")
        if store.read_active_history_epoch()["epoch_id"] != epoch:
            raise SystemExit("validation failures changed the active policy epoch")


def test_message_memory_receipts_and_foreign_keys() -> None:
    with TemporaryDirectory(prefix="jarvis-history-provenance-flow-") as temp:
        path = Path(temp) / "store.sqlite"
        store = MemoryStore(path)
        store.init()
        epoch = _activate_remote_epoch(store)
        secret = "PRIVATE BODY 8d915a SHOULD NOT APPEAR"
        first_id = store.log_message(
            "session-safe",
            "user",
            secret,
            {"preserved": secret},
            provenance={
                "role": "user",
                "policy_epoch_id": epoch,
                "destination_class": "openai-api",
                "remote_eligible": True,
            },
        )
        first = store.read_message_provenance(first_id)
        if (
            first is None
            or not first["remote_eligible"]
            or first["provenance_state"] != "recorded"
            or first["lineage_state"] != "direct_current"
            or first["source_count"] != 0
            or first["source_digest"] is not None
            or not first["lineage_complete"]
        ):
            raise SystemExit(f"atomic message provenance was not recorded: {first}")
        message = store.get_message(first_id)
        if message is None or json.loads(message["metadata"])["preserved"] != secret:
            raise SystemExit("message content or metadata behavior changed")

        second_id = store.log_message("session-safe", "assistant", "ordinary reply")
        first_source_digest = _bound_digest(
            store, [(first_id, str(first["lineage_token"]))]
        )
        attached_token = store.attach_message_provenance(
            second_id,
            role="assistant",
            policy_epoch_id=epoch,
            lineage_state="personal_derived",
            source_count=1,
            source_digest=first_source_digest,
            destination_class="openai-api",
            remote_eligible=True,
        )
        second = store.read_message_provenance(second_id)
        if (
            second is None
            or second["lineage_state"] != "personal_derived"
            or second["source_count"] != 1
            or second["source_digest"] != first_source_digest
            or not second["lineage_complete"]
            or not second["remote_eligible"]
        ):
            raise SystemExit(f"personal-derived assistant provenance drifted: {second}")
        _expect_error(
            RuntimeError,
            lambda: store.attach_message_provenance(
                second_id,
                role="assistant",
                policy_epoch_id=epoch,
                destination_class="openai-api",
                remote_eligible=True,
            ),
            "second message provenance attachment",
        )
        _expect_error(
            ValueError,
            lambda: store.attach_message_provenance(
                second_id,
                role="user",
                policy_epoch_id=epoch,
                destination_class="openai-api",
                remote_eligible=True,
            ),
            "message provenance role mismatch",
        )

        restricted_id = store.log_message(
            "session-safe",
            "assistant",
            "conservative default reply",
            provenance={
                "policy_epoch_id": epoch,
                "destination_class": "openai-api",
                "remote_eligible": True,
            },
        )
        restricted = store.read_message_provenance(restricted_id)
        if (
            restricted is None
            or restricted["lineage_state"] != "mixed_restricted"
            or restricted["source_count"] != 0
            or restricted["source_digest"] is not None
            or restricted["lineage_complete"]
            or restricted["recorded_remote_eligible"]
            or restricted["remote_eligible"]
        ):
            raise SystemExit(f"assistant default lineage did not fail closed: {restricted}")

        tool_id = store.log_message(
            "session-safe",
            "assistant",
            "tool-derived reply",
            provenance={
                "policy_epoch_id": epoch,
                "lineage_state": "tool_derived",
                "source_count": 1,
                "source_digest": first_source_digest,
                "destination_class": "openai-api",
                "remote_eligible": True,
            },
        )
        tool = store.read_message_provenance(tool_id)
        if (
            tool is None
            or tool["lineage_state"] != "tool_derived"
            or not tool["lineage_complete"]
            or tool["recorded_remote_eligible"]
            or tool["remote_eligible"]
        ):
            raise SystemExit(f"tool-derived lineage did not fail closed: {tool}")
        source_digest = _bound_digest(
            store,
            [(first_id, str(first["lineage_token"])), (second_id, attached_token)]
        )

        memory_id = store.add_memory(MemoryRecord("history", "private", secret, "smoke", 1.0))
        legacy_memory = store.read_memory_provenance(memory_id)
        if legacy_memory is None or legacy_memory["lineage_state"] != "legacy_unknown" or legacy_memory["remote_eligible"]:
            raise SystemExit(f"missing memory provenance was not local-only: {legacy_memory}")
        _expect_error(
            ValueError,
            lambda: store.record_memory_provenance(
                memory_id,
                lineage_state="unknown",
                source_count=2,
                source_digest=source_digest,
                policy_epoch_id=epoch,
                destination_class="openai-api",
                remote_eligible=False,
            ),
            "unknown memory lineage state",
        )
        recorded_memory = store.record_memory_provenance(
            memory_id,
            lineage_state="complete",
            source_count=2,
            source_digest=source_digest,
            policy_epoch_id=epoch,
            destination_class="openai-api",
            remote_eligible=True,
        )
        if not recorded_memory["remote_eligible"]:
            raise SystemExit(f"complete current memory provenance lost eligibility: {recorded_memory}")
        _expect_error(
            RuntimeError,
            lambda: store.record_memory_provenance(
                memory_id,
                lineage_state="complete",
                source_count=2,
                source_digest=source_digest,
                policy_epoch_id=epoch,
                destination_class="openai-api",
                remote_eligible=True,
            ),
            "second memory provenance record",
        )

        receipt_id = store.prepare_history_disclosure_receipt(
            session_id="session-safe",
            provider="openai",
            destination_class="openai-api",
            source_count=2,
            source_digest=source_digest,
            policy_epoch_id=epoch,
        )
        confirmed = store.finalize_history_disclosure_receipt(receipt_id, "confirmed")
        if confirmed["state"] != "confirmed" or not confirmed["effective_confirmation"]:
            raise SystemExit(f"current consented receipt did not confirm: {confirmed}")
        _expect_error(
            RuntimeError,
            lambda: store.finalize_history_disclosure_receipt(receipt_id, "confirmed"),
            "second receipt finalization",
        )

        stale_epoch_receipt = store.prepare_history_disclosure_receipt(
            session_id="session-safe",
            provider="openai",
            destination_class="openai-api",
            source_count=2,
            source_digest=source_digest,
            policy_epoch_id=epoch,
        )
        _activate_remote_epoch(store, FINGERPRINT_B)
        blocked = store.finalize_history_disclosure_receipt(stale_epoch_receipt, "confirmed")
        if blocked["state"] != "blocked" or blocked["effective_confirmation"]:
            raise SystemExit(f"stale policy epoch retained disclosure authority: {blocked}")
        if store.read_message_provenance(first_id)["remote_eligible"]:
            raise SystemExit("old message provenance retained eligibility after epoch rotation")
        if store.read_memory_provenance(memory_id)["remote_eligible"]:
            raise SystemExit("old memory provenance retained eligibility after epoch rotation")

        current = store.read_active_history_epoch()
        recovery_receipt = store.prepare_history_disclosure_receipt(
            session_id="session-safe",
            provider=current["provider"],
            destination_class=current["destination_class"],
            source_count=2,
            source_digest=source_digest,
            policy_epoch_id=current["epoch_id"],
        )
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        if store.recover_stale_history_disclosure_receipts(future) != 1:
            raise SystemExit("explicit stale receipt recovery count drifted")
        if store.get_history_disclosure_receipt(recovery_receipt)["state"] != "uncertain":
            raise SystemExit("explicit stale receipt recovery did not fail closed")

        init_receipt = store.prepare_history_disclosure_receipt(
            session_id="session-safe",
            provider=current["provider"],
            destination_class=current["destination_class"],
            source_count=2,
            source_digest=source_digest,
            policy_epoch_id=current["epoch_id"],
        )
        stale_receipt = "8" * 64
        stale_at = (
            datetime.now(timezone.utc)
            - timedelta(seconds=HISTORY_DISCLOSURE_STALE_AFTER_SECONDS + 1)
        ).replace(tzinfo=None, microsecond=0).isoformat() + "Z"
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO history_disclosure_receipts("
                "receipt_id, state, session_id, policy_epoch_id, provider, "
                "destination_class, source_count, source_digest, prepared_at, "
                "updated_at, finalized_at) "
                "VALUES (?, 'prepared', 'session-safe', ?, ?, ?, 2, ?, ?, ?, NULL)",
                (
                    stale_receipt,
                    current["epoch_id"],
                    current["provider"],
                    current["destination_class"],
                    source_digest,
                    stale_at,
                    stale_at,
                ),
            )
        store.init()
        if store.get_history_disclosure_receipt(init_receipt)["state"] != "prepared":
            raise SystemExit("repeat init incorrectly recovered a fresh in-flight receipt")
        if store.get_history_disclosure_receipt(stale_receipt)["state"] != "uncertain":
            raise SystemExit("repeat init did not recover a stale prepared receipt")
        if store.read_active_history_epoch()["epoch_id"] != current["epoch_id"]:
            raise SystemExit("repeat init unexpectedly rotated the active policy epoch")

        invalid_lineage_message_id = store.log_message(
            "session-safe", "user", "database lineage constraint target"
        )
        with store.connect() as conn:
            forbidden_columns = {"content", "metadata", "request", "args", "output", "body"}
            for table in (
                "history_policy_epochs",
                "active_history_policy_epoch",
                "message_provenance",
                "history_disclosure_receipts",
                "memory_provenance",
            ):
                columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                if columns & forbidden_columns:
                    raise SystemExit(f"{table} contains content-bearing columns: {columns}")
                rows = [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
                if secret in json.dumps(rows, sort_keys=True):
                    raise SystemExit(f"{table} retained message or memory content")

            _expect_error(
                sqlite3.IntegrityError,
                lambda: conn.execute(
                    "INSERT INTO history_disclosure_receipts("
                    "receipt_id, state, session_id, policy_epoch_id, provider, destination_class, "
                    "source_count, source_digest, prepared_at, updated_at, finalized_at) "
                    "VALUES (?, 'confirmed', 'session-safe', ?, ?, ?, 2, ?, ?, ?, ?)",
                    (
                        "d" * 64,
                        current["epoch_id"],
                        current["provider"],
                        current["destination_class"],
                        source_digest,
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:00:00Z",
                    ),
                ),
                "terminal disclosure receipt insertion",
            )
            _expect_error(
                sqlite3.IntegrityError,
                lambda: conn.execute(
                    "INSERT INTO message_provenance("
                    "message_id, role, lineage_state, source_count, source_digest, "
                    "policy_epoch_id, lineage_token, destination_class, "
                    "remote_eligible, created_at) "
                    "VALUES (?, 'user', 'direct_current', 1, ?, ?, ?, ?, 0, ?)",
                    (
                        invalid_lineage_message_id,
                        source_digest,
                        current["epoch_id"],
                        "5" * 64,
                        current["destination_class"],
                        "2026-01-01T00:00:00Z",
                    ),
                ),
                "database-level inconsistent message lineage",
            )
            _expect_error(
                sqlite3.IntegrityError,
                lambda: conn.execute(
                    "INSERT INTO message_provenance("
                    "message_id, role, lineage_state, source_count, source_digest, "
                    "policy_epoch_id, lineage_token, destination_class, "
                    "remote_eligible, created_at) "
                    "VALUES (999999, 'user', 'direct_current', 0, NULL, ?, ?, ?, 0, ?)",
                    (current["epoch_id"], "f" * 64, current["destination_class"], "2026-01-01T00:00:00Z"),
                ),
                "orphan message provenance",
            )

        with store.connect() as conn:
            conn.execute("DELETE FROM messages WHERE id = ?", (second_id,))
            if conn.execute(
                "SELECT 1 FROM message_provenance WHERE message_id = ?", (second_id,)
            ).fetchone() is not None:
                raise SystemExit("message provenance did not cascade with its message")
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if conn.execute(
                "SELECT 1 FROM memory_provenance WHERE memory_id = ?", (memory_id,)
            ).fetchone() is not None:
                raise SystemExit("memory provenance did not cascade with its memory")


def test_compaction_provenance_summary() -> None:
    with TemporaryDirectory(prefix="jarvis-history-provenance-compaction-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        epoch = _activate_remote_epoch(store)
        message_id = store.log_message(
            "session-safe",
            "user",
            "source body",
            provenance={
                "destination_class": "openai-api",
                "remote_eligible": True,
                "policy_epoch_id": epoch,
                "lineage_token": LINEAGE_A,
            },
        )
        digest = _bound_digest(store, [(message_id, LINEAGE_A)])
        result = store.commit_conversation_compaction(
            expected_after_id=0,
            first_message_id=message_id,
            last_message_id=message_id,
            record=None,
            eligible_source_count=1,
            withheld_source_count=0,
            source_provenance_digest=digest,
            policy_epoch_id=epoch,
        )
        if result.status != "committed":
            raise SystemExit(f"provenance-aware compaction commit failed: {result}")
        with store.connect() as conn:
            row = conn.execute(
                "SELECT eligible_source_count, withheld_source_count, "
                "source_provenance_digest, policy_epoch_id "
                "FROM conversation_compaction_batches WHERE id = ?",
                (result.batch_id,),
            ).fetchone()
        if row is None or dict(row) != {
            "eligible_source_count": 1,
            "withheld_source_count": 0,
            "source_provenance_digest": digest,
            "policy_epoch_id": epoch,
        }:
            raise SystemExit(f"compaction provenance summary drifted: {dict(row or {})}")
        _expect_error(
            ValueError,
            lambda: store.commit_conversation_compaction(
                expected_after_id=message_id,
                first_message_id=message_id + 1,
                last_message_id=message_id + 1,
                record=None,
                eligible_source_count=True,
            ),
            "boolean compaction provenance count",
        )
        with store.connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] != 0:
                raise SystemExit("nothing_durable compaction unexpectedly created a memory")
            if conn.execute("SELECT COUNT(*) FROM memory_provenance").fetchone()[0] != 0:
                raise SystemExit("nothing_durable compaction unexpectedly created memory provenance")

    with TemporaryDirectory(prefix="jarvis-history-provenance-compaction-memory-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        epoch = _activate_remote_epoch(store)
        message_id = store.log_message(
            "session-safe",
            "user",
            "digest source",
            provenance={
                "destination_class": "openai-api",
                "remote_eligible": True,
                "policy_epoch_id": epoch,
                "lineage_token": LINEAGE_A,
            },
        )
        digest = _bound_digest(store, [(message_id, LINEAGE_A)])
        result = store.commit_conversation_compaction(
            expected_after_id=0,
            first_message_id=message_id,
            last_message_id=message_id,
            record=MemoryRecord(
                "conversation-digest",
                "Conversation digest atomic provenance",
                "- content remains only in memories",
                "conversation_compaction",
                0.8,
            ),
            eligible_source_count=1,
            withheld_source_count=0,
            source_provenance_digest=digest,
            policy_epoch_id=epoch,
        )
        provenance = store.read_memory_provenance(int(result.memory_id or 0))
        if (
            result.status != "committed"
            or not result.memory_created
            or provenance is None
            or provenance["lineage_state"] != "complete"
            or provenance["source_count"] != 1
            or provenance["source_digest"] != digest
            or provenance["policy_epoch_id"] != epoch
            or provenance["destination_class"] != "openai-api"
            or not provenance["remote_eligible"]
        ):
            raise SystemExit(f"digest memory provenance was not inherited atomically: {result} {provenance}")

    with TemporaryDirectory(prefix="jarvis-history-provenance-compaction-withheld-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        epoch = _activate_remote_epoch(store)
        first_id = store.log_message(
            "session-safe",
            "user",
            "eligible source",
            provenance={
                "destination_class": "openai-api",
                "remote_eligible": True,
                "policy_epoch_id": epoch,
                "lineage_token": LINEAGE_A,
            },
        )
        last_id = store.log_message(
            "session-safe",
            "assistant",
            "withheld source",
            provenance={
                "destination_class": "openai-api",
                "remote_eligible": True,
                "policy_epoch_id": epoch,
                "lineage_token": LINEAGE_B,
            },
        )
        digest = _bound_digest(
            store, [(first_id, LINEAGE_A), (last_id, LINEAGE_B)]
        )
        result = store.commit_conversation_compaction(
            expected_after_id=0,
            first_message_id=first_id,
            last_message_id=last_id,
            record=MemoryRecord(
                "conversation-digest",
                "Conversation digest partial provenance",
                "- partial summary",
                "conversation_compaction",
                0.8,
            ),
            eligible_source_count=1,
            withheld_source_count=1,
            source_provenance_digest=digest,
            policy_epoch_id=epoch,
        )
        provenance = store.read_memory_provenance(int(result.memory_id or 0))
        if (
            provenance is None
            or provenance["lineage_state"] != "partial"
            or provenance["source_count"] != 2
            or provenance["remote_eligible"]
        ):
            raise SystemExit(f"withheld compaction sources did not restrict the digest: {provenance}")

    with TemporaryDirectory(prefix="jarvis-history-provenance-compaction-legacy-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        epoch = _activate_remote_epoch(store)
        message_id = store.log_message(
            "session-safe",
            "user",
            "legacy source",
            provenance={
                "destination_class": "openai-api",
                "remote_eligible": True,
                "policy_epoch_id": epoch,
                "lineage_token": LINEAGE_A,
            },
        )
        legacy_title = "Conversation digest 2026-01-01 \u2192 2026-01-01 (messages #1\u2013#1)"
        legacy_body = "- pre-existing legacy digest"
        legacy_memory_id = store.add_memory(
            MemoryRecord(
                "conversation-digest",
                legacy_title,
                legacy_body,
                "conversation_compaction",
                0.8,
            )
        )
        digest = _bound_digest(store, [(message_id, LINEAGE_A)])
        result = store.commit_conversation_compaction(
            expected_after_id=0,
            first_message_id=message_id,
            last_message_id=message_id,
            record=None,
            legacy_memory_id=legacy_memory_id,
            legacy_expected_title=legacy_title,
            legacy_expected_body=legacy_body,
            eligible_source_count=1,
            withheld_source_count=0,
            source_provenance_digest=digest,
            policy_epoch_id=epoch,
        )
        legacy_provenance = store.read_memory_provenance(legacy_memory_id)
        if (
            result.status != "committed"
            or result.memory_created
            or legacy_provenance is None
            or legacy_provenance["lineage_state"] != "legacy_unknown"
            or legacy_provenance["remote_eligible"]
        ):
            raise SystemExit(f"legacy digest adoption incorrectly blessed provenance: {result} {legacy_provenance}")

    with TemporaryDirectory(prefix="jarvis-history-provenance-compaction-rollback-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        epoch = _activate_remote_epoch(store)
        message_id = store.log_message(
            "session-safe",
            "user",
            "rollback source",
            provenance={
                "destination_class": "openai-api",
                "remote_eligible": True,
                "policy_epoch_id": epoch,
                "lineage_token": LINEAGE_A,
            },
        )
        digest = _bound_digest(store, [(message_id, LINEAGE_A)])
        _expect_error(
            ValueError,
            lambda: store.commit_conversation_compaction(
                expected_after_id=0,
                first_message_id=message_id,
                last_message_id=message_id,
                record=MemoryRecord(
                    "conversation-digest",
                    "Conversation digest forged provenance",
                    "- must not commit",
                    "conversation_compaction",
                    0.8,
                ),
                eligible_source_count=1,
                withheld_source_count=0,
                source_provenance_digest="f" * 64,
                policy_epoch_id=epoch,
            ),
            "forged compaction source digest",
        )
        if store.conversation_compaction_watermark() != 0 or store.recent_memories(limit=5):
            raise SystemExit("forged compaction provenance left partial state")
        with store.connect() as conn:
            conn.execute(
                "CREATE TRIGGER reject_compaction_memory_provenance "
                "BEFORE INSERT ON memory_provenance BEGIN "
                "SELECT RAISE(ABORT, 'forced provenance failure'); END"
            )
        _expect_error(
            sqlite3.IntegrityError,
            lambda: store.commit_conversation_compaction(
                expected_after_id=0,
                first_message_id=message_id,
                last_message_id=message_id,
                record=MemoryRecord(
                    "conversation-digest",
                    "Conversation digest rollback",
                    "- must roll back",
                    "conversation_compaction",
                    0.8,
                ),
                eligible_source_count=1,
                withheld_source_count=0,
                source_provenance_digest=digest,
                policy_epoch_id=epoch,
            ),
            "compaction memory provenance insertion failure",
        )
        with store.connect() as conn:
            counts = {
                "memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
                "memory_provenance": conn.execute(
                    "SELECT COUNT(*) FROM memory_provenance"
                ).fetchone()[0],
                "batches": conn.execute(
                    "SELECT COUNT(*) FROM conversation_compaction_batches"
                ).fetchone()[0],
            }
        if counts != {"memories": 0, "memory_provenance": 0, "batches": 0}:
            raise SystemExit(f"failed provenance insert left partial compaction state: {counts}")
        if store.conversation_compaction_watermark() != 0:
            raise SystemExit("failed provenance insert advanced the compaction watermark")


def test_content_binding_mutation_invalidation_and_skill_origin() -> None:
    with TemporaryDirectory(prefix="jarvis-history-content-binding-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        epoch = _activate_remote_epoch(store)
        message_id = store.log_message(
            "session-safe",
            "user",
            "ORIGINAL PRIVATE MESSAGE",
            provenance={
                "destination_class": "openai-api",
                "remote_eligible": True,
                "policy_epoch_id": epoch,
                "lineage_token": LINEAGE_A,
            },
        )
        original = store.read_message_provenance(message_id)
        recent = store.recent_messages(limit=1, session_id="session-safe")
        if (
            original is None
            or not original["content_matches"]
            or not original["remote_eligible"]
            or len(recent) != 1
            or recent[0]["content_digest"] != original["content_digest"]
        ):
            raise SystemExit("chat-facing message reads lost bounded content-hash evidence")
        with store.connect() as conn:
            conn.execute(
                "UPDATE messages SET content = 'MUTATED PRIVATE MESSAGE' WHERE id = ?",
                (message_id,),
            )
        mutated = store.read_message_provenance(message_id)
        if mutated is None or mutated["content_matches"] or mutated["remote_eligible"]:
            raise SystemExit(f"post-provenance message mutation remained eligible: {mutated}")
        with store.connect() as conn:
            conn.execute("UPDATE messages SET role = 'system' WHERE id = ?", (message_id,))
        role_mutated = store.read_message_provenance(message_id)
        if role_mutated is None or role_mutated["content_matches"] or role_mutated["remote_eligible"]:
            raise SystemExit(f"post-provenance role mutation did not fail closed: {role_mutated}")

    with TemporaryDirectory(prefix="jarvis-memory-content-binding-") as temp:
        store = MemoryStore(Path(temp) / "store.sqlite")
        store.init()
        epoch = _activate_remote_epoch(store)

        def compact_one(content: str, token: str, title: str, body: str) -> int:
            message_id = store.log_message(
                "session-safe",
                "user",
                content,
                provenance={
                    "destination_class": "openai-api",
                    "remote_eligible": True,
                    "policy_epoch_id": epoch,
                    "lineage_token": token,
                },
            )
            result = store.commit_conversation_compaction(
                expected_after_id=store.conversation_compaction_watermark(),
                first_message_id=message_id,
                last_message_id=message_id,
                record=MemoryRecord(
                    "conversation-digest", title, body, "conversation_compaction", 0.8
                ),
                eligible_source_count=1,
                source_provenance_digest=_bound_digest(
                    store, [(message_id, token)]
                ),
                policy_epoch_id=epoch,
            )
            if result.status != "committed" or result.memory_id is None:
                raise SystemExit(f"content-binding memory fixture failed: {result}")
            return int(result.memory_id)

        update_id = compact_one(
            "update source", LINEAGE_A, "Bound update digest", "- original digest body"
        )
        if not store.update_memory(
            update_id,
            "conversation-digest",
            "Bound update digest",
            "- rewritten digest body",
            0.8,
        ):
            raise SystemExit("post-compaction memory update fixture did not update")
        updated = store.read_memory_provenance(update_id)
        if (
            updated is None
            or updated["content_matches"]
            or updated["recorded_remote_eligible"]
            or updated["remote_eligible"]
            or updated["invalidated_at"] is None
        ):
            raise SystemExit(f"updated memory retained remotely eligible provenance: {updated}")

        restricted_id = store.add_memory(
            MemoryRecord(
                "conversation-digest",
                "Restricted merge digest",
                "RESTRICTED COMPACTION BODY",
                "conversation_compaction",
                0.8,
            )
        )
        store.record_memory_provenance(
            restricted_id,
            lineage_state="partial",
            source_count=1,
            source_digest="7" * 64,
            destination_class="openai-api",
            remote_eligible=False,
            policy_epoch_id=epoch,
        )
        ordinary_id = store.add_memory(
            MemoryRecord("history", "Ordinary merge memory", "ORDINARY BODY", "manual", 1.0)
        )
        protected_ids = (restricted_id, ordinary_id)
        before_rows = {
            memory_id: dict(store.get_memory(memory_id) or {})
            for memory_id in protected_ids
        }
        before_provenance = store.read_memory_provenance(restricted_id)
        targets = store.resolve_memory_merge_approval_targets(restricted_id, ordinary_id)
        if targets is None:
            raise SystemExit("provenance merge targets were unavailable")
        restricted_target, ordinary_target = targets

        if store.merge_memories(ordinary_id, restricted_id):
            raise SystemExit("ordinary survivor absorbed a restricted compaction digest")
        if store.merge_memories(restricted_id, ordinary_id):
            raise SystemExit("restricted digest survivor absorbed an ordinary memory")
        exact_forward = store.merge_memories_exact(
            ordinary_id,
            ordinary_target.revision,
            ordinary_target.binding,
            restricted_id,
            restricted_target.revision,
            restricted_target.binding,
        )
        exact_reverse = store.merge_memories_exact(
            restricted_id,
            restricted_target.revision,
            restricted_target.binding,
            ordinary_id,
            ordinary_target.revision,
            ordinary_target.binding,
        )
        projected_forward = store.merge_memories_exact_with_projection(
            ordinary_id,
            ordinary_target.revision,
            ordinary_target.binding,
            restricted_id,
            restricted_target.revision,
            restricted_target.binding,
        )
        projected_reverse = store.merge_memories_exact_with_projection(
            restricted_id,
            restricted_target.revision,
            restricted_target.binding,
            ordinary_id,
            ordinary_target.revision,
            ordinary_target.binding,
        )
        refused = (
            exact_forward,
            exact_reverse,
            projected_forward,
            projected_reverse,
        )
        if any(result.status != "invalid" or result.projection_targets for result in refused):
            raise SystemExit(f"exact provenance merge did not fail closed: {refused}")
        after_rows = {
            memory_id: dict(store.get_memory(memory_id) or {})
            for memory_id in protected_ids
        }
        with store.connect() as conn:
            projection_count = conn.execute(
                "SELECT COUNT(*) FROM memory_projection_jobs WHERE memory_id IN (?, ?)",
                protected_ids,
            ).fetchone()[0]
        if (
            after_rows != before_rows
            or store.read_memory_provenance(restricted_id) != before_provenance
            or projection_count != 0
        ):
            raise SystemExit("rejected provenance merge changed rows, lineage, or projections")

        legacy_digest_id = store.add_memory(
            MemoryRecord(
                "conversation-digest",
                "Legacy unknown digest",
                "LEGACY UNKNOWN DIGEST BODY",
                "conversation_compaction",
                0.8,
            )
        )
        legacy_ordinary_id = store.add_memory(
            MemoryRecord("history", "Legacy peer", "LEGACY ORDINARY BODY", "manual", 1.0)
        )
        legacy_ids = (legacy_digest_id, legacy_ordinary_id)
        legacy_before = {
            memory_id: dict(store.get_memory(memory_id) or {}) for memory_id in legacy_ids
        }
        if store.merge_memories(legacy_ordinary_id, legacy_digest_id):
            raise SystemExit("ordinary survivor absorbed a legacy-unknown digest")
        if store.merge_memories(legacy_digest_id, legacy_ordinary_id):
            raise SystemExit("legacy-unknown digest survivor absorbed an ordinary memory")
        legacy_after = {
            memory_id: dict(store.get_memory(memory_id) or {}) for memory_id in legacy_ids
        }
        if legacy_after != legacy_before:
            raise SystemExit("rejected legacy-unknown digest merge changed either row")

        ordinary_keep_id = store.add_memory(
            MemoryRecord("history", "Ordinary keep", "KEEP BODY", "manual", 0.5)
        )
        ordinary_delete_id = store.add_memory(
            MemoryRecord("history", "Ordinary delete", "DELETE BODY", "manual", 0.9)
        )
        if not store.merge_memories(ordinary_keep_id, ordinary_delete_id):
            raise SystemExit("same-origin ordinary memory merge was rejected")
        ordinary_keep = store.get_memory(ordinary_keep_id)
        if (
            ordinary_keep is None
            or "DELETE BODY" not in str(ordinary_keep["body"])
            or store.get_memory(ordinary_delete_id) is not None
        ):
            raise SystemExit("same-origin ordinary memory merge did not preserve legacy behavior")

        restricted = SkillRecord(
            "Session-derived durable skill",
            "when reviewing history",
            "[session-derived]\n## Source Session Signals\n- private source omitted",
            "session-derived,draft,human-review-required",
        )
        target = store.save_skill_draft_with_projection_target(restricted)
        created = store.get_skill_by_id(target.skill_id)
        if created is None or created["origin"] != "session_history":
            raise SystemExit(f"session-derived skill origin was not recorded: {created}")
        store.save_skill_with_projection_target(
            SkillRecord(
                restricted.name,
                "reviewed trigger",
                "Markers intentionally removed after human rewrite.",
                "reviewed",
            )
        )
        rewritten = store.get_skill_by_id(target.skill_id)
        if (
            rewritten is None
            or rewritten["review_status"] != "active"
            or rewritten["origin"] != "session_history"
        ):
            raise SystemExit(f"skill rewrite laundered restricted origin: {rewritten}")
        manual_id = store.save_skill(
            SkillRecord("Manual durable skill", "manual trigger", "Manual body", "reviewed")
        )
        manual = store.get_skill_by_id(manual_id)
        if manual is None or manual["origin"] != "user_authored":
            raise SystemExit(f"new user-authored skill origin drifted: {manual}")
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skills(name, trigger, body, tags, created_at, updated_at) "
                "VALUES ('Legacy unknown skill', '', 'legacy body', '', ?, ?)",
                (utc := datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z", utc),
            )
        legacy = store.get_skill("Legacy unknown skill")
        if legacy is None or legacy["origin"] != "legacy_unknown":
            raise SystemExit(f"legacy skill origin was not conservative: {legacy}")


def test_egress_authority_rejects_every_stale_recovery_remainder() -> None:
    with TemporaryDirectory(prefix="jarvis-history-egress-stale-") as temp:
        store = MemoryStore(Path(temp) / "history.sqlite")
        store.init()
        _activate_remote_epoch(store)
        current = store.read_active_history_epoch()
        fresh_receipt = store.prepare_history_disclosure_receipt(
            session_id="session-egress",
            provider=current["provider"],
            destination_class=current["destination_class"],
            source_count=2,
            source_digest=SOURCE_DIGEST,
            policy_epoch_id=current["epoch_id"],
        )
        stale_at = (
            datetime.now(timezone.utc)
            - timedelta(seconds=HISTORY_DISCLOSURE_STALE_AFTER_SECONDS + 5)
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        stale_receipts = [f"{0x1000 + index:064x}" for index in range(102)]
        malformed_receipt = "c" * 63 + "1"
        future_receipt = "c" * 63 + "2"
        future_at = (
            datetime.now(timezone.utc) + timedelta(days=1)
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        with store.connect() as conn:
            conn.executemany(
                "INSERT INTO history_disclosure_receipts("
                "receipt_id, state, session_id, policy_epoch_id, provider, "
                "destination_class, source_count, source_digest, prepared_at, "
                "updated_at, finalized_at) "
                "VALUES (?, 'prepared', 'session-egress', ?, ?, ?, 2, ?, ?, ?, NULL)",
                [
                    (
                        receipt_id,
                        current["epoch_id"],
                        current["provider"],
                        current["destination_class"],
                        SOURCE_DIGEST,
                        stale_at,
                        stale_at,
                    )
                    for receipt_id in stale_receipts
                ]
                + [
                    (
                        malformed_receipt,
                        current["epoch_id"],
                        current["provider"],
                        current["destination_class"],
                        SOURCE_DIGEST,
                        "not-a-timestamp",
                        stale_at,
                    ),
                    (
                        future_receipt,
                        current["epoch_id"],
                        current["provider"],
                        current["destination_class"],
                        SOURCE_DIGEST,
                        future_at,
                        future_at,
                    ),
                ],
            )

        store.init()
        stale_states = {
            receipt_id: store.get_history_disclosure_receipt(receipt_id)["state"]
            for receipt_id in stale_receipts
        }
        stale_remainders = [
            receipt_id for receipt_id, state in stale_states.items() if state == "prepared"
        ]
        if len(stale_remainders) != 2 or sum(
            state == "uncertain" for state in stale_states.values()
        ) != 100:
            raise SystemExit(
                f"bounded startup recovery did not leave exactly two stale remainders: {stale_states}"
            )

        def exercise_egress(receipt_id: str) -> None:
            with store.history_disclosure_egress_fence(
                receipt_id=receipt_id,
                policy_epoch_id=current["epoch_id"],
                provider=current["provider"],
                model_identifier=current["model_identifier"],
                destination_class=current["destination_class"],
                source_count=2,
                source_digest=SOURCE_DIGEST,
            ):
                pass

        for receipt_id in stale_receipts:
            _expect_error(
                PermissionError,
                lambda value=receipt_id: exercise_egress(value),
                "stale disclosure receipt egress",
            )
        for receipt_id, label in (
            (malformed_receipt, "malformed disclosure timestamp egress"),
            (future_receipt, "future disclosure timestamp egress"),
        ):
            _expect_error(
                PermissionError,
                lambda value=receipt_id: exercise_egress(value),
                label,
            )
        if store.get_history_disclosure_receipt(fresh_receipt)["state"] != "prepared":
            raise SystemExit("repeat init did not preserve the fresh prepared receipt")
        exercise_egress(fresh_receipt)


def main() -> None:
    test_clean_database_and_policy_rotation()
    test_legacy_migration_is_local_only()
    test_just_added_provenance_schema_migrates_transitive_lineage()
    test_strict_validation_and_atomic_message_logging()
    test_message_memory_receipts_and_foreign_keys()
    test_compaction_provenance_summary()
    test_content_binding_mutation_invalidation_and_skill_origin()
    test_egress_authority_rejects_every_stale_recovery_remainder()
    print("History provenance store smoke passed")


if __name__ == "__main__":
    main()
