from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.execution_outcome import classify_approved_execution_outcome
from jarvis_v2.agent.types import ApprovalArgumentResolution, RiskLevel
from jarvis_v2.memory.memory_projection import (
    reconcile_memory_projection,
    reconcile_pending_memory_projections,
)
from jarvis_v2.memory.obsidian import MemoryProjectionDeleteResult, ObsidianVault
from jarvis_v2.memory.preference_projection import (
    reconcile_pending_preference_projections,
    reconcile_preference_projection,
)
from jarvis_v2.memory.store import (
    MemoryPreferencePromotionResult,
    MemoryProjectionTarget,
    MemoryRecord,
    MemoryStore,
    PreferenceProjectionTarget,
    PreferenceRecord,
    preference_identity_part,
    preference_projection_source_digest,
)
from jarvis_v2.tools.knowledge_promotion import make_knowledge_promotion_tools


GRAPH_TABLES = (
    "preferences",
    "preference_identity_owners",
    "preference_memory_links",
    "preference_projection_state",
    "preference_projection_jobs",
    "memory_projection_jobs",
)
MAX_SQLITE_INT = 9223372036854775807


def _setup(root: Path) -> tuple[MemoryStore, ObsidianVault]:
    store = MemoryStore(root / "store.sqlite")
    store.init()
    vault = ObsidianVault(root / "vault")
    vault.init()
    return store, vault


def _seed(
    store: MemoryStore,
    title: str = "Generic preference candidate",
    body: str = "A reviewed generic memory that may become a preference.",
    *,
    category: str = "preferences",
    source: str = "preference-promotion-smoke-private-source",
) -> int:
    return store.add_memory(
        MemoryRecord(
            category=category,
            title=title,
            body=body,
            source=source,
            confidence=0.71,
        )
    )


def _target(store: MemoryStore, memory_id: int) -> Any:
    target = store.resolve_knowledge_promotion_approval_target(memory_id)
    if target is None:
        raise SystemExit(f"memory #{memory_id} did not resolve as a promotion candidate")
    memory_target = getattr(target, "memory_target", target)
    if (
        getattr(target, "target_kind", None) != "preference"
        or getattr(memory_target, "memory_id", None) != memory_id
        or type(getattr(memory_target, "revision", None)) is not int
        or re.fullmatch(r"[0-9a-f]{64}", getattr(memory_target, "binding", "")) is None
    ):
        raise SystemExit(f"preference candidate target is malformed: {target!r}")
    return target


def _memory_target(target: Any) -> Any:
    return getattr(target, "memory_target", target)


def _promote_exact(
    store: MemoryStore,
    memory_id: int,
    target: Any,
    *,
    category: str = "communication",
    key: str = "response tone",
    value: str = "warm and direct",
) -> MemoryPreferencePromotionResult:
    bound = _memory_target(target)
    return store.promote_memory_to_preference_exact(
        memory_id,
        bound.revision,
        bound.binding,
        PreferenceRecord(key=key, value=value, category=category),
    )


def _status(result: Any) -> str:
    value = getattr(result, "status", "")
    return value if type(value) is str else ""


def _success_fields(
    result: Any, *, expected_memory_id: int
) -> tuple[int, int, PreferenceProjectionTarget, MemoryProjectionTarget]:
    if type(result) is not MemoryPreferencePromotionResult or result.status != "promoted":
        raise SystemExit(f"exact preference promotion did not succeed: {result!r}")
    preference_id = result.preference_id
    memory_id = result.memory_id
    preference_target = result.preference_projection_target
    memory_target = result.memory_projection_target
    if (
        type(preference_id) is not int
        or preference_id < 1
        or memory_id != expected_memory_id
        or type(preference_target) is not PreferenceProjectionTarget
        or type(memory_target) is not MemoryProjectionTarget
        or type(preference_target.generation) is not int
        or preference_target.generation < 1
        or re.fullmatch(r"[0-9a-f]{64}", preference_target.source_digest) is None
        or memory_target.memory_id != expected_memory_id
        or type(memory_target.revision) is not int
        or memory_target.revision < 2
        or memory_target.operation != "publish"
        or re.fullmatch(r"[0-9a-f]{64}", memory_target.source_digest) is None
        or re.fullmatch(r"[0-9a-f]{64}", result.source_integrity_binding or "")
        is None
    ):
        raise SystemExit(f"successful promotion evidence is malformed: {result!r}")
    return preference_id, memory_id, preference_target, memory_target


def _assert_empty_refusal(result: Any, label: str) -> None:
    if type(result) is not MemoryPreferencePromotionResult:
        raise SystemExit(f"{label} returned the wrong result type: {result!r}")
    if result.status not in {
        "not_found",
        "stale",
        "not_candidate",
        "wrong_target",
        "identity_exists",
        "identity_conflict",
        "already_exists",
    }:
        raise SystemExit(f"{label} returned an unrecognized refusal: {result!r}")
    if any(
        value is not None
        for value in (
            result.preference_id,
            result.memory_id,
            result.preference_projection_target,
            result.memory_projection_target,
            result.source_integrity_binding,
        )
    ):
        raise SystemExit(f"{label} returned contradictory mutation evidence: {result!r}")


def _rows(store: MemoryStore, table: str) -> list[tuple[Any, ...]]:
    with store.connect() as conn:
        return [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]


def _database_snapshot(store: MemoryStore) -> dict[str, list[tuple[Any, ...]]]:
    return {
        table: _rows(store, table)
        for table in ("memories",) + GRAPH_TABLES
    }


def _vault_snapshot(vault: ObsidianVault) -> dict[str, bytes]:
    return {
        str(path.relative_to(vault.root_path)): path.read_bytes()
        for path in sorted(vault.root_path.rglob("*"))
        if path.is_file()
    }


def _assert_zero_effects(
    store: MemoryStore,
    vault: ObsidianVault,
    before_database: dict[str, list[tuple[Any, ...]]],
    before_vault: dict[str, bytes],
    label: str,
) -> None:
    after_database = _database_snapshot(store)
    after_vault = _vault_snapshot(vault)
    if after_database != before_database or after_vault != before_vault:
        raise SystemExit(
            f"{label} changed durable state: database={after_database != before_database}, "
            f"vault={after_vault != before_vault}"
        )


def _factory_handlers(
    store: MemoryStore,
    vault: ObsidianVault,
) -> tuple[
    Callable[[dict[str, Any]], Any],
    Callable[[dict[str, Any]], Any],
    Callable[[dict[str, Any]], Any],
]:
    packet = None
    promote = None
    resolver = None
    made = make_knowledge_promotion_tools(store, vault)
    items = list(made.items()) if isinstance(made, dict) else [("", item) for item in made]
    for supplied_name, item in items:
        handler = getattr(item, "handler", item)
        name = str(
            supplied_name
            or getattr(item, "name", "")
            or getattr(handler, "__name__", "")
        )
        if name == "knowledge_promotion_packet" and callable(handler):
            packet = handler
        elif name == "promote_memory_to_preference" and callable(handler):
            promote = handler
            candidate_resolver = getattr(item, "approval_argument_resolver", None)
            if callable(candidate_resolver):
                resolver = candidate_resolver
        elif name == "resolve_promote_memory_to_preference_approval" and callable(handler):
            resolver = handler
    if not callable(packet) or not callable(promote) or not callable(resolver):
        raise SystemExit("preference promotion packet, handler, and resolver are not all exposed")
    return packet, promote, resolver


def _raw_args(
    memory_id: int,
    target: Any,
    *,
    category: str = "communication",
    key: str = "response tone",
    value: str = "warm and direct",
) -> dict[str, Any]:
    bound = _memory_target(target)
    return {
        "memory_id": memory_id,
        "reviewed_revision": bound.revision,
        "review_token": bound.binding,
        "category": category,
        "key": key,
        "value": value,
    }


def _bound_args(
    resolver: Callable[[dict[str, Any]], Any],
    raw_args: dict[str, Any],
) -> dict[str, Any]:
    resolution = resolver(raw_args)
    if type(resolution) is not ApprovalArgumentResolution:
        raise SystemExit(f"preference promotion approval did not resolve: {resolution!r}")
    expected_keys = (set(raw_args) - {"review_token"}) | {
        "review_binding",
        "target_revision",
        "target_binding",
    }
    if set(resolution.args) != expected_keys:
        raise SystemExit(f"approval resolver changed the bound argument contract: {resolution!r}")
    return resolution.args


def _tool_promote(
    promote: Callable[[dict[str, Any]], Any],
    resolver: Callable[[dict[str, Any]], Any],
    memory_id: int,
    target: Any,
    *,
    category: str = "communication",
    key: str = "response tone",
    value: str = "warm and direct",
) -> Any:
    return promote(
        _bound_args(
            resolver,
            _raw_args(
                memory_id,
                target,
                category=category,
                key=key,
                value=value,
            ),
        )
    )


def _assert_private(
    values: Iterable[Any],
    forbidden: Iterable[str],
    label: str,
) -> None:
    rendered = "\n".join(
        json.dumps(value, sort_keys=True, default=str) if isinstance(value, dict) else repr(value)
        for value in values
    )
    leaked = [marker for marker in forbidden if marker and marker in rendered]
    if leaked:
        raise SystemExit(f"{label} leaked private promotion material: {leaked}")


def test_exact_packet_approval_mapping_reuses_source_and_is_active() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-exact-") as temp:
        store, vault = _setup(Path(temp))
        store.set_preference_with_projections(
            PreferenceRecord("response tone", "formal", "writing")
        )
        memory_id = _seed(store)
        target = _target(store, memory_id)
        bound = _memory_target(target)
        packet, promote, resolver = _factory_handlers(store, vault)

        before_packet = _database_snapshot(store)
        packet_result = packet({"memory_id": memory_id})
        if (
            not packet_result.ok
            or str(memory_id) not in packet_result.output
            or str(bound.revision) not in packet_result.output
            or bound.binding not in packet_result.output
            or "preference" not in packet_result.output.casefold()
            or packet_result.metadata.get("state_changed") is not False
        ):
            raise SystemExit(f"packet did not expose the exact read-only binding: {packet_result}")
        if _database_snapshot(store) != before_packet:
            raise SystemExit("preference promotion packet mutated the store")

        category = "Communication"
        key = "Response Tone"
        value = "Warm, concise, and direct"
        raw = _raw_args(
            memory_id,
            target,
            category=category,
            key=key,
            value=value,
        )
        resolved = _bound_args(resolver, raw)
        expected_resolved = {
            key: value for key, value in raw.items() if key != "review_token"
        } | {
            "review_binding": bound.binding,
            "target_revision": bound.revision,
            "target_binding": bound.binding,
        }
        if resolved != expected_resolved:
            raise SystemExit(
                "approval did not bind the exact packet and explicit mapping: "
                f"{resolved}"
            )

        before_tamper = _database_snapshot(store)
        stale_raw = dict(raw)
        stale_raw["review_token"] = "0" * 64
        stale_resolution = resolver(stale_raw)
        if type(stale_resolution) is ApprovalArgumentResolution:
            raise SystemExit("a token not issued by the exact packet crossed approval resolution")
        for field in ("review_binding", "target_binding"):
            tampered_bound = dict(resolved)
            tampered_bound[field] = "0" * 64
            tampered_result = promote(tampered_bound)
            if tampered_result.ok or tampered_result.metadata.get("state_changed") is not False:
                raise SystemExit(f"tampered {field} crossed the handler gate")
        if _database_snapshot(store) != before_tamper:
            raise SystemExit("approval-binding tampering changed durable state")

        result = promote(resolved)
        if not result.ok or result.metadata.get("preference_id") is None:
            raise SystemExit(f"tool preference promotion failed: {result}")

        preference_id = result.metadata["preference_id"]
        with store.connect() as conn:
            preferences = list(conn.execute("SELECT * FROM preferences ORDER BY id"))
            preference = conn.execute(
                "SELECT * FROM preferences WHERE id = ?", (preference_id,)
            ).fetchone()
            owner = conn.execute(
                "SELECT * FROM preference_identity_owners WHERE preference_id = ?",
                (preference_id,),
            ).fetchone()
            link = conn.execute(
                "SELECT * FROM preference_memory_links WHERE preference_id = ?",
                (preference_id,),
            ).fetchone()
            source = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            source_count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        if (
            preference is None
            or preference["category"] != category
            or preference["key"] != key
            or preference["value"] != value
            or preference["status"] != "active"
            or owner is None
            or owner["category_fold"] != preference_identity_part(category)
            or owner["key_fold"] != preference_identity_part(key)
            or link is None
            or link["memory_id"] != memory_id
            or source is None
            or source["category"] != "preferences"
            or source["title"] != key
            or source["source"] != "preferences"
            or source["body"]
            != f"Category: {category}\nKey: {key}\nValue: {value}\nStatus: active"
            or source_count != 2
            or len(preferences) != 2
        ):
            raise SystemExit(
                "promotion did not create the exact canonical mapping on the source memory"
            )
        same_key_rows = [
            row
            for row in preferences
            if preference_identity_part(row["key"]) == "response tone"
        ]
        if len(same_key_rows) != 2 or {row["category"] for row in same_key_rows} != {
            "writing",
            category,
        }:
            raise SystemExit("same key in a different category was not allowed")


def test_normalized_identity_and_ownerless_shadow_refuse_with_zero_effects() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-collision-") as temp:
        root = Path(temp)
        for mode in ("canonical", "ownerless_shadow"):
            store, vault = _setup(root / mode)
            memory_id = _seed(store)
            target = _target(store, memory_id)
            if mode == "canonical":
                store.set_preference_with_projections(
                    PreferenceRecord("Cafe\u0301 Tone", "existing", "COMMUNICATION")
                )
            else:
                with store.connect() as conn:
                    conn.execute(
                        "INSERT INTO preferences(category, key, value, status, revision, "
                        "created_at, updated_at) VALUES (?, ?, ?, 'active', 1, ?, ?)",
                        (
                            "ＣＯＭＭＵＮＩＣＡＴＩＯＮ",
                            "ＣＡＦÉ ＴＯＮＥ",
                            "legacy shadow",
                            "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z",
                        ),
                    )
            before_database = _database_snapshot(store)
            before_vault = _vault_snapshot(vault)
            result = _promote_exact(
                store,
                memory_id,
                target,
                category="communication",
                key="CAFÉ TONE",
                value="new value",
            )
            _assert_empty_refusal(result, mode)
            _assert_zero_effects(store, vault, before_database, before_vault, mode)
            _packet, promote, resolver = _factory_handlers(store, vault)
            tool_result = _tool_promote(
                promote,
                resolver,
                memory_id,
                target,
                category="communication",
                key="CAFÉ TONE",
                value="new value",
            )
            if (
                tool_result.ok
                or tool_result.metadata.get("state_changed") is not False
                or tool_result.metadata.get("promotion_outcome") == "unknown"
            ):
                raise SystemExit(
                    f"{mode} collision was not a known zero-effect refusal: {tool_result}"
                )
            _assert_zero_effects(
                store, vault, before_database, before_vault, f"{mode} tool refusal"
            )


def test_owner_only_normalized_shadow_refuses_with_zero_effects() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-owner-shadow-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(store)
        target = _target(store, memory_id)
        with store.connect() as conn:
            inserted = conn.execute(
                "INSERT INTO preferences(category, key, value, status, revision, "
                "created_at, updated_at) VALUES (?, ?, ?, 'active', 1, ?, ?)",
                (
                    "unrelated category",
                    "unrelated key",
                    "legacy owner row",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ),
            )
            shadow_id = int(inserted.lastrowid)
            conn.execute(
                "INSERT INTO preference_identity_owners(category_fold, key_fold, "
                "preference_id) VALUES (?, ?, ?)",
                (
                    preference_identity_part("ＣＯＭＭＵＮＩＣＡＴＩＯＮ"),
                    preference_identity_part("ＲＥＳＰＯＮＳＥ ＴＯＮＥ"),
                    shadow_id,
                ),
            )
        before_database = _database_snapshot(store)
        before_vault = _vault_snapshot(vault)
        result = _promote_exact(
            store,
            memory_id,
            target,
            category="communication",
            key="response tone",
            value="new value",
        )
        if result.status not in {"identity_exists", "identity_conflict"}:
            raise SystemExit(f"owner-only normalized shadow was not refused: {result!r}")
        _assert_empty_refusal(result, "owner-only normalized shadow")
        _assert_zero_effects(
            store,
            vault,
            before_database,
            before_vault,
            "owner-only normalized shadow",
        )

        _packet, promote, resolver = _factory_handlers(store, vault)
        tool_result = _tool_promote(
            promote,
            resolver,
            memory_id,
            target,
            category="communication",
            key="response tone",
            value="new value",
        )
        if (
            tool_result.ok
            or tool_result.metadata.get("state_changed") is not False
            or tool_result.metadata.get("promotion_outcome") == "unknown"
        ):
            raise SystemExit(
                f"owner-only shadow was not a known tool refusal: {tool_result}"
            )
        _assert_zero_effects(
            store,
            vault,
            before_database,
            before_vault,
            "owner-only normalized shadow tool refusal",
        )


def test_stale_revision_content_drift_and_wrong_target_refuse_exactly() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-stale-") as temp:
        root = Path(temp)
        for mode in ("revision", "content_drift", "wrong_target"):
            category = "decisions" if mode == "wrong_target" else "preferences"
            store, vault = _setup(root / mode)
            memory_id = _seed(store, category=category)
            target = store.resolve_knowledge_promotion_approval_target(memory_id)
            if target is None:
                raise SystemExit(f"{mode} fixture did not resolve")
            bound = _memory_target(target)
            if mode == "revision":
                with store.connect() as conn:
                    conn.execute(
                        "UPDATE memories SET revision = revision + 1, updated_at = ? WHERE id = ?",
                        ("2026-02-02T00:00:00Z", memory_id),
                    )
            elif mode == "content_drift":
                with store.connect() as conn:
                    conn.execute(
                        "UPDATE memories SET body = ? WHERE id = ?",
                        ("same revision private drift", memory_id),
                    )
            before_database = _database_snapshot(store)
            before_vault = _vault_snapshot(vault)
            result = store.promote_memory_to_preference_exact(
                memory_id,
                bound.revision,
                bound.binding,
                PreferenceRecord("response tone", "warm", "communication"),
            )
            _assert_empty_refusal(result, mode)
            if mode == "wrong_target" and result.status != "wrong_target":
                raise SystemExit(f"wrong target was not distinguished: {result!r}")
            if mode != "wrong_target" and result.status != "stale":
                raise SystemExit(f"{mode} did not report stale exact binding: {result!r}")
            _assert_zero_effects(store, vault, before_database, before_vault, mode)


def test_malformed_oversized_control_reserved_and_path_fields_refuse_preapproval() -> None:
    cases: dict[str, tuple[str, Any]] = {
        "missing_category": ("category", None),
        "boolean_key": ("key", True),
        "oversized_category": ("category", "c" * 65),
        "oversized_key": ("key", "k" * 121),
        "oversized_value": ("value", "v" * 2001),
        "control_category": ("category", "comm\x00unication"),
        "control_key": ("key", "response\u202etone"),
        "control_value": ("value", "warm\n\x07direct"),
        "reserved_category": ("category", "jarvis_projection"),
        "reserved_key": ("key", "<!-- private -->"),
        "reserved_value": ("value", "source_digest: " + "a" * 64),
        "path_category": ("category", "/\x55sers/private/category"),
        "path_key": ("key", "../private/key"),
        "path_value": ("value", "C:\\private\\value.txt"),
        "placeholder_value": ("value", "<explicit value>"),
    }
    with TemporaryDirectory(prefix="jarvis-preference-promotion-validation-") as temp:
        root = Path(temp)
        for label, (field, invalid) in cases.items():
            store, vault = _setup(root / label)
            memory_id = _seed(store)
            target = _target(store, memory_id)
            _packet, _promote, resolver = _factory_handlers(store, vault)
            args = _raw_args(memory_id, target)
            if label == "missing_category":
                del args[field]
            else:
                args[field] = invalid
            before_database = _database_snapshot(store)
            before_vault = _vault_snapshot(vault)
            result = resolver(args)
            if type(result) is ApprovalArgumentResolution or getattr(result, "ok", False):
                raise SystemExit(f"{label} crossed the preapproval validation gate: {result!r}")
            if getattr(result, "metadata", {}).get("state_changed") is not False:
                raise SystemExit(f"{label} did not explicitly report zero state change: {result!r}")
            _assert_zero_effects(store, vault, before_database, before_vault, label)


def test_direct_store_validator_rejects_invalid_records_and_oversized_targets() -> None:
    valid = {
        "category": "communication",
        "key": "response tone",
        "value": "warm and direct",
        "status": "active",
    }
    record_cases: dict[str, dict[str, str]] = {
        "non_active_status": {**valid, "status": "inactive"},
        "untrimmed_status": {**valid, "status": " active "},
        "untrimmed_category": {**valid, "category": " communication"},
        "untrimmed_key": {**valid, "key": "response tone "},
        "untrimmed_value": {**valid, "value": " warm and direct"},
        "blank_category": {**valid, "category": " "},
        "blank_key": {**valid, "key": "\t"},
        "blank_value": {**valid, "value": "\n"},
        "control_category": {**valid, "category": "comm\x00unication"},
        "format_control_key": {**valid, "key": "response\u202etone"},
        "surrogate_value": {**valid, "value": "warm\ud800direct"},
        "noncharacter_category": {**valid, "category": "comm\ufdd0unication"},
        "noncharacter_key": {**valid, "key": "tone\ufffe"},
        "oversized_category": {**valid, "category": "c" * 65},
        "oversized_key": {**valid, "key": "k" * 121},
        "oversized_value": {**valid, "value": "v" * 2001},
    }
    with TemporaryDirectory(prefix="jarvis-preference-promotion-store-validation-") as temp:
        root = Path(temp)
        cases: list[tuple[str, int | None, int | None, dict[str, str]]] = [
            (label, None, None, fields) for label, fields in record_cases.items()
        ]
        cases.extend(
            (
                ("oversized_memory_id", MAX_SQLITE_INT + 1, None, valid),
                ("oversized_revision", None, MAX_SQLITE_INT + 1, valid),
            )
        )
        for label, supplied_memory_id, supplied_revision, fields in cases:
            store, vault = _setup(root / label)
            memory_id = _seed(store)
            target = _target(store, memory_id)
            bound = _memory_target(target)
            before_database = _database_snapshot(store)
            before_vault = _vault_snapshot(vault)
            record = PreferenceRecord(
                category=fields["category"],
                key=fields["key"],
                value=fields["value"],
                status=fields["status"],
            )
            try:
                store.promote_memory_to_preference_exact(
                    supplied_memory_id or memory_id,
                    supplied_revision or bound.revision,
                    bound.binding,
                    record,
                )
            except ValueError:
                pass
            else:
                raise SystemExit(f"direct store validator accepted {label}")
            _assert_zero_effects(
                store,
                vault,
                before_database,
                before_vault,
                f"direct validator {label}",
            )


def test_legitimate_url_value_is_allowed() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-url-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(store)
        target = _target(store, memory_id)
        value = "Use https://docs.example.com/preferences?q=reviewed#tone as the reference"
        _packet, promote, resolver = _factory_handlers(store, vault)
        result = _tool_promote(
            promote,
            resolver,
            memory_id,
            target,
            category="references",
            key="tone guide",
            value=value,
        )
        if not result.ok:
            raise SystemExit(f"legitimate HTTP URL value did not pass the tool boundary: {result}")
        preference_id = result.metadata.get("preference_id")
        with store.connect() as conn:
            row = conn.execute(
                "SELECT value, status FROM preferences WHERE id = ?", (preference_id,)
            ).fetchone()
            link = conn.execute(
                "SELECT memory_id FROM preference_memory_links WHERE preference_id = ?",
                (preference_id,),
            ).fetchone()
        if link is None or link[0] != memory_id or row is None or tuple(row) != (value, "active"):
            raise SystemExit("legitimate HTTP URL value was rejected or altered")


def test_http_url_rules_and_active_render_payloads() -> None:
    valid_values = (
        "Use https://docs.example.com:8443/preferences?q=reviewed#tone.",
        "Reference http://127.0.0.1:8080/preferences for this local service.",
        "Reference https://[2001:db8::1]:443/preferences for IPv6.",
        "Reference https://münich.example/preferences for the localized guide.",
    )
    invalid_cases = {
        "missing_host": ("value", "Reference https:///preferences"),
        "userinfo": ("value", "Reference https://user:pass@example.com/preferences"),
        "invalid_port": ("value", "Reference https://example.com:70000/preferences"),
        "invalid_numeric_host": ("value", "Reference https://999.999.1.1/preferences"),
        "invalid_ipv6": ("value", "Reference https://[2001:db8::zz]/preferences"),
        "url_in_category": ("category", "https://example.com/category"),
        "url_in_key": ("key", "https://example.com/key"),
        "markdown_image": ("value", "![tone](https://example.com/tone.png)"),
        "html_image": ("value", '<img src="https://example.com/tone.png">'),
        "script": ("value", "<script>reviewed()</script>"),
        "nfkc_svg": ("value", "<ｓｖｇ><image href=x></ｓｖｇ>"),
        "generic_html": ("value", '<div class="tone">reviewed preference text'),
        "inline_css": ("value", '<span style="color: red">reviewed'),
        "inline_css_url": (
            "value",
            '<div style="background-image: url(https://example.com/tone.png)">tone',
        ),
        "css_url_resource": (
            "value",
            "Use background-image: url(https://example.com/tone.png)",
        ),
    }
    with TemporaryDirectory(prefix="jarvis-preference-promotion-url-rules-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(store)
        target = _target(store, memory_id)
        _packet, _promote, resolver = _factory_handlers(store, vault)
        before_database = _database_snapshot(store)
        before_vault = _vault_snapshot(vault)

        for value in valid_values:
            resolution = resolver(_raw_args(memory_id, target, value=value))
            if type(resolution) is not ApprovalArgumentResolution:
                raise SystemExit(f"valid HTTP URL did not resolve for approval: {resolution}")
            if resolution.args.get("value") != value:
                raise SystemExit("valid HTTP URL value was altered during approval resolution")
            _assert_zero_effects(
                store,
                vault,
                before_database,
                before_vault,
                "valid HTTP URL approval resolution",
            )

        unsafe_failures: list[str] = []
        for label, (field, payload) in invalid_cases.items():
            args = _raw_args(memory_id, target)
            args[field] = payload
            refusal = resolver(args)
            if type(refusal) is ApprovalArgumentResolution or getattr(refusal, "ok", False):
                unsafe_failures.append(f"{label}: reached approval")
            elif label in {
                "markdown_image",
                "html_image",
                "script",
                "nfkc_svg",
                "generic_html",
                "inline_css",
                "inline_css_url",
                "css_url_resource",
            }:
                reason = str(getattr(refusal, "metadata", {}).get("reason", ""))
                if "active_render_content" not in reason:
                    unsafe_failures.append(f"{label}: reason={reason!r}")
            _assert_zero_effects(
                store,
                vault,
                before_database,
                before_vault,
                f"URL/render refusal {label}",
            )
        if unsafe_failures:
            raise SystemExit(
                f"unsafe URL or active render validation failed: {unsafe_failures!r}"
            )


def test_replay_and_concurrent_promotions_have_one_winner() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-replay-") as temp:
        store, vault = _setup(Path(temp))
        memory_id = _seed(store)
        target = _target(store, memory_id)
        first = _promote_exact(store, memory_id, target)
        _success_fields(first, expected_memory_id=memory_id)
        before_database = _database_snapshot(store)
        before_vault = _vault_snapshot(vault)
        replay = _promote_exact(store, memory_id, target)
        _assert_empty_refusal(replay, "replay")
        _assert_zero_effects(store, vault, before_database, before_vault, "replay")

    with TemporaryDirectory(prefix="jarvis-preference-promotion-concurrent-") as temp:
        store, _vault = _setup(Path(temp))
        memory_id = _seed(store)
        target = _target(store, memory_id)
        barrier = threading.Barrier(2)

        def compete(value: str) -> MemoryPreferencePromotionResult:
            peer = MemoryStore(store.db_path)
            peer.init()
            barrier.wait(timeout=10)
            return _promote_exact(peer, memory_id, target, value=value)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [
                future.result(timeout=15)
                for future in (
                    pool.submit(compete, "warm"),
                    pool.submit(compete, "direct"),
                )
            ]
        if sum(_status(result) == "promoted" for result in results) != 1:
            raise SystemExit(f"concurrent promotion did not have exactly one winner: {results!r}")
        for result in results:
            if result.status != "promoted":
                _assert_empty_refusal(result, "concurrent loser")
        with store.connect() as conn:
            counts = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "preferences",
                    "preference_identity_owners",
                    "preference_memory_links",
                )
            }
            link = conn.execute("SELECT memory_id FROM preference_memory_links").fetchone()
        if counts != {table: 1 for table in counts} or link is None or link[0] != memory_id:
            raise SystemExit(f"concurrent promotion split canonical custody: {counts}, {link}")


def test_ordinary_set_preference_race_preserves_one_canonical_owner() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-set-race-") as temp:
        store, _vault = _setup(Path(temp))
        memory_id = _seed(store)
        target = _target(store, memory_id)
        barrier = threading.Barrier(2)

        def promote() -> MemoryPreferencePromotionResult:
            peer = MemoryStore(store.db_path)
            peer.init()
            barrier.wait(timeout=10)
            return _promote_exact(
                peer,
                memory_id,
                target,
                category="Communication",
                key="Response Tone",
                value="promoted value",
            )

        def ordinary_set() -> int:
            peer = MemoryStore(store.db_path)
            peer.init()
            barrier.wait(timeout=10)
            return peer.set_preference(
                PreferenceRecord("response tone", "ordinary value", "communication")
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            promotion_future = pool.submit(promote)
            set_future = pool.submit(ordinary_set)
            promotion = promotion_future.result(timeout=15)
            ordinary_id = set_future.result(timeout=15)
        with store.connect() as conn:
            preferences = list(conn.execute("SELECT * FROM preferences"))
            owners = list(conn.execute("SELECT * FROM preference_identity_owners"))
            links = list(conn.execute("SELECT * FROM preference_memory_links"))
            source = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if (
            len(preferences) != 1
            or len(owners) != 1
            or ordinary_id != preferences[0]["id"]
            or owners[0]["preference_id"] != ordinary_id
        ):
            raise SystemExit("ordinary setter race split the normalized preference identity")
        if promotion.status == "promoted":
            _success_fields(promotion, expected_memory_id=memory_id)
            if (
                len(links) != 1
                or links[0]["preference_id"] != ordinary_id
                or links[0]["memory_id"] != memory_id
                or source["source"] != "preferences"
            ):
                raise SystemExit("winning promotion lost custody during the ordinary setter race")
            if not store.preference_mutation_source_integrity(
                int(promotion.preference_id),
                memory_id,
                promotion.preference_projection_target,
                promotion.memory_projection_target,
                promotion.source_integrity_binding,
            ):
                raise SystemExit(
                    "ordinary setter race left promoted preference custody inconsistent"
                )
        else:
            _assert_empty_refusal(promotion, "ordinary setter race loser")
            if (
                any(link["memory_id"] == memory_id for link in links)
                or source["source"] == "preferences"
            ):
                raise SystemExit("losing promotion partially claimed source custody")


def test_trigger_and_generation_max_failures_roll_back_everything() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-rollback-") as temp:
        root = Path(temp)
        for mode in ("trigger", "generation_max"):
            store, vault = _setup(root / mode)
            memory_id = _seed(store)
            target = _target(store, memory_id)
            with store.connect() as conn:
                if mode == "trigger":
                    conn.execute(
                        "CREATE TRIGGER reject_promoted_preference_link BEFORE INSERT ON "
                        "preference_memory_links BEGIN SELECT RAISE(ABORT, "
                        "'private injected promotion failure'); END"
                    )
                else:
                    conn.execute(
                        "UPDATE preference_projection_state SET generation = ? "
                        "WHERE singleton_id = 1",
                        (MAX_SQLITE_INT,),
                    )
            before_database = _database_snapshot(store)
            before_vault = _vault_snapshot(vault)
            try:
                _promote_exact(store, memory_id, target)
            except (RuntimeError, sqlite3.DatabaseError):
                pass
            else:
                raise SystemExit(f"{mode} failure did not abort promotion")
            _assert_zero_effects(store, vault, before_database, before_vault, mode)


def test_pending_promotion_graphs_repair_once_and_cleanup_legacy_generic_note() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-repair-") as temp:
        store, vault = _setup(Path(temp))
        original_record = MemoryRecord(
            category="preferences",
            title="Generic preference candidate",
            body="A reviewed generic memory that may become a preference.",
            source="preference-promotion-smoke-private-source",
            confidence=0.71,
        )
        memory_id = store.add_memory(original_record)
        target = _target(store, memory_id)
        with store.connect() as conn:
            store_identity = str(
                conn.execute(
                    "SELECT identity FROM store_instance_state WHERE singleton_id = 1"
                ).fetchone()[0]
            )
        legacy_path = vault.write_memory(
            original_record,
            memory_id,
            store_identity=store_identity,
        )
        result = _promote_exact(store, memory_id, target)
        preference_id, _, preference_target, memory_target = _success_fields(
            result, expected_memory_id=memory_id
        )
        with store.connect() as conn:
            counts = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "memories",
                    "preferences",
                    "preference_identity_owners",
                    "preference_memory_links",
                    "preference_projection_state",
                    "preference_projection_jobs",
                    "memory_projection_jobs",
                )
            }
            preference_job = conn.execute(
                "SELECT * FROM preference_projection_jobs WHERE singleton_id = 1"
            ).fetchone()
            memory_job = conn.execute(
                "SELECT * FROM memory_projection_jobs WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if counts != {table: 1 for table in counts}:
            raise SystemExit(f"promotion did not reserve one exact pending graph: {counts}")
        if (
            preference_job is None
            or preference_job["state"] != "pending"
            or memory_job is None
            or memory_job["state"] != "pending"
            or memory_job["legacy_category"] != original_record.category
            or memory_job["legacy_title"] != original_record.title
        ):
            raise SystemExit("promotion pending graph lost its projection or legacy identity")

        cleanup_calls: list[dict[str, Any]] = []
        original_cleanup = vault.delete_legacy_memory_projection

        def cleanup_spy(**kwargs: Any) -> MemoryProjectionDeleteResult:
            cleanup_calls.append(dict(kwargs))
            if legacy_path.exists():
                legacy_path.unlink()
            return MemoryProjectionDeleteResult("deleted")

        vault.delete_legacy_memory_projection = cleanup_spy  # type: ignore[method-assign]
        try:
            preference_summary = reconcile_pending_preference_projections(store, vault)
            memory_summary = reconcile_pending_memory_projections(store, vault)
            second_preference = reconcile_pending_preference_projections(store, vault)
            second_memory = reconcile_pending_memory_projections(store, vault)
        finally:
            vault.delete_legacy_memory_projection = original_cleanup  # type: ignore[method-assign]

        if (
            preference_summary.attempted != 1
            or preference_summary.completed != 1
            or preference_summary.pending != 0
            or memory_summary.attempted != 1
            or memory_summary.completed != 1
            or memory_summary.pending != 0
            or second_preference.completed != 1
            or second_preference.pending != 0
            or second_memory.attempted != 0
            or second_memory.pending != 0
        ):
            raise SystemExit(
                "promotion projection repairs did not converge idempotently: "
                f"{preference_summary}, {memory_summary}, "
                f"{second_preference}, {second_memory}"
            )
        if len(cleanup_calls) != 1:
            raise SystemExit(f"legacy cleanup did not run exactly once: {cleanup_calls}")
        cleanup = cleanup_calls[0]
        if (
            cleanup.get("memory_id") != memory_id
            or cleanup.get("store_identity") != store_identity
            or cleanup.get("legacy_category") != original_record.category
            or cleanup.get("legacy_title") != original_record.title
            or legacy_path.exists()
        ):
            raise SystemExit(f"legacy generic-note cleanup was not exactly bound: {cleanup}")
        if store.preference_mutation_projection_completion(
            preference_id,
            memory_id,
            preference_target,
            memory_target,
        ) != (True, True):
            raise SystemExit("repaired promotion graph was not certified complete")

        with store.connect() as conn:
            final_counts = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "memories",
                    "preferences",
                    "preference_identity_owners",
                    "preference_memory_links",
                    "preference_projection_jobs",
                    "memory_projection_jobs",
                )
            }
            final_preference_job = conn.execute(
                "SELECT * FROM preference_projection_jobs WHERE singleton_id = 1"
            ).fetchone()
            final_memory_job = conn.execute(
                "SELECT * FROM memory_projection_jobs WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        canonical_notes = sorted(
            (vault.root_path / "Memory Tree" / "Records").glob("*.md")
        )
        preference_note = vault.root_path / "Memory Tree" / "Preferences.md"
        expected_canonical = vault.root_path / str(
            final_memory_job["canonical_path_display"]
        )
        if (
            final_counts != {table: 1 for table in final_counts}
            or final_preference_job is None
            or final_preference_job["state"] != "completed"
            or final_memory_job is None
            or final_memory_job["state"] != "completed"
            or canonical_notes != [expected_canonical]
            or not preference_note.is_file()
            or legacy_path.exists()
        ):
            raise SystemExit(
                "projection repair duplicated rows or canonical notes: "
                f"counts={final_counts}, canonical_notes={canonical_notes}"
            )


def test_projection_completion_pending_and_supersession_are_truthful() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-projection-") as temp:
        root = Path(temp)
        store, vault = _setup(root / "direct")
        memory_id = _seed(store)
        target = _target(store, memory_id)
        result = _promote_exact(store, memory_id, target)
        preference_id, _, preference_target, memory_target = _success_fields(
            result, expected_memory_id=memory_id
        )
        if store.preference_mutation_projection_completion(
            preference_id, memory_id, preference_target, memory_target
        ) != (False, False):
            raise SystemExit("pending projection jobs were reported completed")
        preference_outcome = reconcile_preference_projection(store, vault, preference_target)
        memory_outcome = reconcile_memory_projection(
            store,
            vault,
            memory_id,
            expected_operation=memory_target.operation,
            expected_revision=memory_target.revision,
            expected_source_digest=memory_target.source_digest,
        )
        if (
            preference_outcome.status != "completed"
            or memory_outcome.status != "completed"
            or store.preference_mutation_projection_completion(
                preference_id, memory_id, preference_target, memory_target
            )
            != (True, True)
        ):
            raise SystemExit("completed preference promotion projections were not certified")
        store.set_preference_with_projections(
            PreferenceRecord("other key", "other value", "other category")
        )
        superseded = store.preference_mutation_projection_completion(
            preference_id, memory_id, preference_target, memory_target
        )
        if superseded[0] is not False or type(superseded[1]) is not bool:
            raise SystemExit(
                f"superseded preference generation was certified current: {superseded}"
            )

        pending_store, pending_vault = _setup(root / "handler-pending")
        pending_id = _seed(pending_store)
        pending_target = _target(pending_store, pending_id)
        _packet, promote, resolver = _factory_handlers(pending_store, pending_vault)
        original = pending_vault.write_preferences_with_evidence

        def fail_projection(*_args: Any, **_kwargs: Any):
            raise OSError("private path /\x55sers/the operator/Secret/Preferences.md")

        pending_vault.write_preferences_with_evidence = (  # type: ignore[method-assign]
            fail_projection
        )
        try:
            pending_result = _tool_promote(promote, resolver, pending_id, pending_target)
        finally:
            pending_vault.write_preferences_with_evidence = original  # type: ignore[method-assign]
        if (
            pending_result.ok
            or pending_result.metadata.get("promotion_committed") is not True
            or pending_result.metadata.get("projection_pending") is not True
            or "do not promote" not in pending_result.output.casefold()
        ):
            raise SystemExit(f"pending projections were reported as success: {pending_result}")
        _assert_private(
            (pending_result.output, pending_result.metadata),
            (str(root), "Preferences.md", "private path"),
            "pending projection result",
        )

        malformed_store, malformed_vault = _setup(root / "malformed-completion")
        malformed_id = _seed(malformed_store)
        malformed_target = _target(malformed_store, malformed_id)
        _packet, malformed_promote, malformed_resolver = _factory_handlers(
            malformed_store, malformed_vault
        )
        original_completion = malformed_store.preference_mutation_projection_completion
        malformed_store.preference_mutation_projection_completion = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: ("completed", 1)
        )
        try:
            malformed_result = _tool_promote(
                malformed_promote,
                malformed_resolver,
                malformed_id,
                malformed_target,
            )
        finally:
            setattr(
                malformed_store,
                "preference_mutation_projection_completion",
                original_completion,
            )
        if (
            malformed_result.ok
            or malformed_result.metadata.get("promotion_committed") is not True
            or malformed_result.metadata.get("projection_completion_contract_valid") is not False
            or "contract was malformed" not in malformed_result.output
            or "do not promote" not in malformed_result.output.casefold()
        ):
            raise SystemExit(
                f"truthy malformed projection completion certified durability: {malformed_result}"
            )


def test_identical_set_preference_preserves_completed_linked_memory_projection() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-identical-projection-") as temp:
        store, vault = _setup(Path(temp))
        record = PreferenceRecord(
            category="communication",
            key="identical response tone",
            value="warm and direct",
        )
        initial = store.set_preference_with_projections(record)
        memory_target = initial.memory_projection_target
        outcome = reconcile_memory_projection(
            store,
            vault,
            initial.memory_id,
            expected_operation=memory_target.operation,
            expected_revision=memory_target.revision,
            expected_source_digest=memory_target.source_digest,
        )
        before_job_row = store.get_memory_projection_job(initial.memory_id)
        if outcome.status != "completed" or before_job_row is None:
            raise SystemExit(f"linked memory projection did not complete: {outcome!r}")
        before_job = dict(before_job_row)
        before_vault = _vault_snapshot(vault)

        repeated = store.set_preference_with_projections(record)
        after_job_row = store.get_memory_projection_job(initial.memory_id)
        after_job = None if after_job_row is None else dict(after_job_row)
        evidence_fields = (
            "memory_id",
            "store_identity",
            "operation",
            "state",
            "memory_revision",
            "source_digest",
            "canonical_path_display",
            "content_digest",
            "prior_content_digest",
            "completion_status",
            "last_error_code",
            "created_at",
            "completed_at",
        )
        changed_evidence = {
            field: (
                before_job.get(field),
                None if after_job is None else after_job.get(field),
            )
            for field in evidence_fields
            if after_job is None or after_job.get(field) != before_job.get(field)
        }
        if (
            repeated.changed is not False
            or repeated.preference_id != initial.preference_id
            or repeated.memory_id != initial.memory_id
            or repeated.preference_projection_target
            != initial.preference_projection_target
            or repeated.memory_projection_target != initial.memory_projection_target
            or before_job.get("state") != "completed"
            or changed_evidence
            or store.count_pending_memory_projection_jobs() != 0
            or _vault_snapshot(vault) != before_vault
        ):
            raise SystemExit(
                "identical set_preference reopened or altered the completed linked-memory "
                f"projection: repeated={repeated!r}, changed_evidence={changed_evidence!r}"
            )


def test_real_source_integrity_detects_custody_content_and_job_corruption() -> None:
    modes = (
        "deleted_link",
        "repointed_link",
        "normalized_duplicate",
        "tampered_memory",
        "preference_job_identity",
        "memory_job_identity",
    )
    with TemporaryDirectory(prefix="jarvis-preference-promotion-real-integrity-") as temp:
        root = Path(temp)
        for mode in modes:
            store, _vault = _setup(root / mode)
            memory_id = _seed(store)
            target = _target(store, memory_id)
            result = _promote_exact(store, memory_id, target)
            preference_id, _, preference_target, memory_target = _success_fields(
                result, expected_memory_id=memory_id
            )
            baseline = store.preference_mutation_source_integrity(
                preference_id,
                memory_id,
                preference_target,
                memory_target,
                result.source_integrity_binding,
            )
            if baseline is not True:
                raise SystemExit(f"uncorrupted source integrity was not true for {mode}")

            checked_preference_target = preference_target
            with store.connect() as conn:
                if mode == "deleted_link":
                    conn.execute(
                        "DELETE FROM preference_memory_links WHERE preference_id = ?",
                        (preference_id,),
                    )
                elif mode == "repointed_link":
                    inserted = conn.execute(
                        "INSERT INTO memories(category, title, body, source, confidence, "
                        "revision, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                        (
                            "preferences",
                            "alternate memory",
                            "alternate body",
                            "integrity-smoke",
                            0.5,
                            "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z",
                        ),
                    )
                    alternate_id = int(inserted.lastrowid)
                    conn.execute(
                        "UPDATE preference_memory_links SET memory_id = ? "
                        "WHERE preference_id = ?",
                        (alternate_id, preference_id),
                    )
                elif mode == "normalized_duplicate":
                    conn.execute(
                        "INSERT INTO preferences(category, key, value, status, revision, "
                        "created_at, updated_at) VALUES (?, ?, ?, 'active', 1, ?, ?)",
                        (
                            "COMMUNICATION",
                            "RESPONSE TONE",
                            "normalized duplicate",
                            "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:00Z",
                        ),
                    )
                    rows = tuple(
                        conn.execute(
                            "SELECT * FROM preferences ORDER BY category COLLATE NOCASE, "
                            "key COLLATE NOCASE, id"
                        )
                    )
                    digest = preference_projection_source_digest(
                        preference_target.generation,
                        rows,
                    )
                    conn.execute(
                        "UPDATE preference_projection_jobs SET source_digest = ? "
                        "WHERE singleton_id = 1",
                        (digest,),
                    )
                    checked_preference_target = PreferenceProjectionTarget(
                        preference_target.generation,
                        digest,
                    )
                elif mode == "tampered_memory":
                    conn.execute(
                        "UPDATE memories SET body = ? WHERE id = ?",
                        ("tampered canonical preference memory", memory_id),
                    )
                elif mode == "preference_job_identity":
                    conn.execute(
                        "UPDATE preference_projection_jobs SET source_digest = ? "
                        "WHERE singleton_id = 1",
                        ("f" * 64,),
                    )
                else:
                    conn.execute(
                        "UPDATE memory_projection_jobs SET source_digest = ? "
                        "WHERE memory_id = ?",
                        ("f" * 64, memory_id),
                    )
            corrupted = store.preference_mutation_source_integrity(
                preference_id,
                memory_id,
                checked_preference_target,
                memory_target,
                result.source_integrity_binding,
            )
            if corrupted is not False:
                raise SystemExit(
                    f"real source integrity accepted {mode}: {corrupted!r}"
                )


def test_source_integrity_mismatch_and_unavailable_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-integrity-") as temp:
        root = Path(temp)
        for mode in ("mismatch", "unavailable"):
            store, vault = _setup(root / mode)
            memory_id = _seed(store)
            target = _target(store, memory_id)
            _packet, promote, resolver = _factory_handlers(store, vault)
            original = store.preference_mutation_source_integrity
            if mode == "mismatch":
                replacement = lambda *_args, **_kwargs: False
            else:
                def replacement(*_args: Any, **_kwargs: Any) -> bool:
                    raise sqlite3.DatabaseError("private source verification outage")
            store.preference_mutation_source_integrity = replacement  # type: ignore[method-assign]
            try:
                result = _tool_promote(promote, resolver, memory_id, target)
            finally:
                store.preference_mutation_source_integrity = original  # type: ignore[method-assign]
            expected = False if mode == "mismatch" else None
            if (
                result.ok
                or result.metadata.get("promotion_committed") is not True
                or result.metadata.get("source_integrity_verified") is not expected
                or result.metadata.get("source_integrity_status") != mode
                or "do not promote" not in result.output.casefold()
            ):
                raise SystemExit(f"{mode} source integrity was not fail-closed: {result}")
            if mode == "mismatch" and "projection repair alone cannot resolve" not in result.output:
                raise SystemExit(f"integrity mismatch looked projection-repairable: {result}")
            _assert_private(
                (result.output, result.metadata),
                (str(root), "private source verification outage"),
                f"{mode} source integrity",
            )


def test_malformed_and_contradictory_results_are_unknown_without_replay() -> None:
    modes = (
        "exception",
        "shaped_refusal",
        "contradictory_refusal",
        "malformed_success",
        "wrong_memory_success",
        "unknown_status",
    )
    with TemporaryDirectory(prefix="jarvis-preference-promotion-unknown-") as temp:
        root = Path(temp)
        for mode in modes:
            store, vault = _setup(root / mode)
            memory_id = _seed(store)
            target = _target(store, memory_id)
            _packet, promote, resolver = _factory_handlers(store, vault)
            original = store.promote_memory_to_preference_exact
            calls = 0

            def replacement(*_args: Any, **_kwargs: Any) -> Any:
                nonlocal calls
                calls += 1
                if mode == "exception":
                    raise sqlite3.DatabaseError("private mutation path /tmp/private.sqlite")
                if mode == "shaped_refusal":
                    return type("ShapedRefusal", (), {"status": "stale"})()
                if mode == "contradictory_refusal":
                    return MemoryPreferencePromotionResult(
                        "stale",
                        1,
                        memory_id,
                        PreferenceProjectionTarget(1, "a" * 64),
                        MemoryProjectionTarget(memory_id, 2, "publish", "b" * 64),
                    )
                if mode == "malformed_success":
                    return MemoryPreferencePromotionResult(
                        "promoted", 1, memory_id, object(), object()  # type: ignore[arg-type]
                    )
                if mode == "wrong_memory_success":
                    return MemoryPreferencePromotionResult(
                        "promoted",
                        1,
                        memory_id + 1,
                        PreferenceProjectionTarget(1, "a" * 64),
                        MemoryProjectionTarget(memory_id + 1, 2, "publish", "b" * 64),
                    )
                return MemoryPreferencePromotionResult("maybe")

            store.promote_memory_to_preference_exact = replacement  # type: ignore[method-assign]
            try:
                result = _tool_promote(promote, resolver, memory_id, target)
            finally:
                store.promote_memory_to_preference_exact = original  # type: ignore[method-assign]
            outcome = classify_approved_execution_outcome(
                ok=result.ok,
                metadata=result.metadata,
                risk=RiskLevel.HIGH_RISK,
            )
            if (
                calls != 1
                or result.ok
                or result.metadata.get("promotion_outcome") != "unknown"
                or result.metadata.get("projection_state") != "unknown"
                or result.metadata.get("projection_pending") is not None
                or result.metadata.get("retry_safe") is not False
                or outcome.outcome_unknown is not True
                or "do not promote" not in result.output.casefold()
            ):
                raise SystemExit(f"{mode} did not produce an unknown no-replay outcome: {result}")
            _assert_private(
                (result.output, result.metadata),
                (str(root), "/tmp/private.sqlite", "private mutation path"),
                f"{mode} unknown result",
            )


def test_metadata_token_path_and_value_privacy() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-promotion-privacy-") as temp:
        root = Path(temp)
        store, vault = _setup(root)
        title = "PRIVATE-PREFERENCE-PROMOTION-TITLE"
        body = "PRIVATE-PREFERENCE-PROMOTION-BODY /\x55sers/the operator/Secret/source.txt"
        source = "PRIVATE-PREFERENCE-PROMOTION-SOURCE"
        value = "PRIVATE-PREFERENCE-PROMOTION-VALUE"
        memory_id = _seed(store, title, body, source=source)
        target = _target(store, memory_id)
        token = _memory_target(target).binding
        packet, promote, resolver = _factory_handlers(store, vault)
        packet_result = packet({"memory_id": memory_id})
        if not packet_result.ok or token not in packet_result.output:
            raise SystemExit("private packet did not expose its review token only in output")
        _assert_private(
            (packet_result.metadata,),
            (token, str(root), str(vault.root_path), title, body, source, "source.txt"),
            "packet metadata",
        )
        result = _tool_promote(
            promote,
            resolver,
            memory_id,
            target,
            category="private category marker",
            key="private key marker",
            value=value,
        )
        if not result.ok:
            raise SystemExit(f"privacy fixture promotion failed: {result}")
        _assert_private(
            (result.output, result.metadata),
            (
                token,
                str(root),
                str(vault.root_path),
                title,
                body,
                source,
                value,
                "source.txt",
            ),
            "successful promotion result",
        )


def main() -> None:
    test_exact_packet_approval_mapping_reuses_source_and_is_active()
    test_normalized_identity_and_ownerless_shadow_refuse_with_zero_effects()
    test_owner_only_normalized_shadow_refuses_with_zero_effects()
    test_stale_revision_content_drift_and_wrong_target_refuse_exactly()
    test_malformed_oversized_control_reserved_and_path_fields_refuse_preapproval()
    test_direct_store_validator_rejects_invalid_records_and_oversized_targets()
    test_legitimate_url_value_is_allowed()
    test_http_url_rules_and_active_render_payloads()
    test_replay_and_concurrent_promotions_have_one_winner()
    test_ordinary_set_preference_race_preserves_one_canonical_owner()
    test_trigger_and_generation_max_failures_roll_back_everything()
    test_pending_promotion_graphs_repair_once_and_cleanup_legacy_generic_note()
    test_projection_completion_pending_and_supersession_are_truthful()
    test_identical_set_preference_preserves_completed_linked_memory_projection()
    test_real_source_integrity_detects_custody_content_and_job_corruption()
    test_source_integrity_mismatch_and_unavailable_fail_closed()
    test_malformed_and_contradictory_results_are_unknown_without_replay()
    test_metadata_token_path_and_value_privacy()
    print("Knowledge preference promotion smoke passed")


if __name__ == "__main__":
    main()
