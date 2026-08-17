from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore, PersonRecord
from jarvis_v2.tools.people import make_people_tools


ALPHA_NAME = "Selector Alpha"
BETA_NAME = "Selector Beta"
ALPHA_NOTE = "PRIVATE-ALPHA-NOTE"
BETA_NOTE = "PRIVATE-BETA-NOTE"
ALPHA_INTERACTION = "PRIVATE-ALPHA-INTERACTION"


class _HostileSelector:
    def __str__(self) -> str:
        raise RuntimeError("PRIVATE-HOSTILE-SELECTOR-DETAIL")


class _ReadTrackingStore(MemoryStore):
    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self.private_note_sql_reads = 0

    def connect(self) -> sqlite3.Connection:
        conn = super().connect()

        def authorize(
            action_code: int,
            table_name: str | None,
            column_name: str | None,
            _database_name: str | None,
            _trigger_name: str | None,
        ) -> int:
            if (
                action_code == sqlite3.SQLITE_READ
                and table_name == "people"
                and column_name == "notes"
            ):
                self.private_note_sql_reads += 1
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorize)
        return conn


class _StoreReadProbe:
    def __init__(self, store: _ReadTrackingStore) -> None:
        self._store = store
        self.interaction_queries = 0
        self.interaction_row_reads = 0
        self.name_resolution_error: Exception | None = None
        self.hide_name_resolution = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def reset_counts(self) -> None:
        self._store.private_note_sql_reads = 0
        self.interaction_queries = 0
        self.interaction_row_reads = 0

    def get_person(
        self,
        person_id: int | None = None,
        name: str | None = None,
    ) -> Any:
        if name is not None:
            if self.name_resolution_error is not None:
                raise self.name_resolution_error
            if self.hide_name_resolution:
                return None
        return self._store.get_person(person_id=person_id, name=name)

    def get_person_if_identity_matches(
        self,
        person_id: int,
        identity_key: str,
    ) -> tuple[str, Any]:
        if self.name_resolution_error is not None:
            raise self.name_resolution_error
        if self.hide_name_resolution:
            return "stale_person_identity", None
        return self._store.get_person_if_identity_matches(person_id, identity_key)

    def list_person_interactions(self, person_id: int, limit: int = 10) -> list[Any]:
        self.interaction_queries += 1
        rows = self._store.list_person_interactions(person_id, limit=limit)
        self.interaction_row_reads += len(rows)
        return rows


def _fixture(root: Path):
    store = _ReadTrackingStore(root / "memory.sqlite")
    store.init()
    alpha_id = store.upsert_person(
        PersonRecord(name=ALPHA_NAME, relation="synthetic", notes=ALPHA_NOTE)
    )
    beta_id = store.upsert_person(
        PersonRecord(name=BETA_NAME, relation="synthetic", notes=BETA_NOTE)
    )
    store.log_person_interaction(
        alpha_id,
        ALPHA_INTERACTION,
        "2026-07-14T00:00:00Z",
    )
    probe = _StoreReadProbe(store)
    _, _, get_person, _ = make_people_tools(
        probe,  # type: ignore[arg-type]
        ObsidianVault(root / "vault"),
    )
    return store, probe, get_person, alpha_id, beta_id


def _assert_no_private_reads(probe: _StoreReadProbe, *, label: str) -> None:
    if (
        probe._store.private_note_sql_reads != 0
        or probe.interaction_queries != 0
        or probe.interaction_row_reads != 0
    ):
        raise SystemExit(
            f"{label} crossed the private-read gate: "
            f"notes={probe._store.private_note_sql_reads}, "
            f"interaction_queries={probe.interaction_queries}, "
            f"interaction_rows={probe.interaction_row_reads}"
        )


def _assert_content_free(result: Any, *, label: str, forbidden: tuple[str, ...]) -> None:
    rendered = result.output + json.dumps(
        result.metadata,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    leaked = [value for value in forbidden if value and value in rendered]
    if leaked:
        raise SystemExit(f"{label} leaked selector or private content: {leaked}")
    if result.metadata.get("reads_private_data") is not False:
        raise SystemExit(f"{label} did not retain content-free refusal metadata")


def test_conflict_is_content_free_and_match_opens_private_read_gate() -> None:
    with TemporaryDirectory(prefix="jarvis-get-person-selectors-") as temp:
        root = Path(temp)
        _, probe, get_person, alpha_id, _ = _fixture(root)

        probe.reset_counts()
        conflict = get_person({"person_id": alpha_id, "name": BETA_NAME})
        if conflict.ok or conflict.metadata.get("reason") != "person_target_mismatch":
            raise SystemExit(f"conflicting selectors did not fail closed: {conflict}")
        _assert_no_private_reads(probe, label="conflicting selectors")
        _assert_content_free(
            conflict,
            label="conflicting selectors",
            forbidden=(
                ALPHA_NAME,
                BETA_NAME,
                ALPHA_NOTE,
                BETA_NOTE,
                ALPHA_INTERACTION,
                str(root),
            ),
        )

        probe.reset_counts()
        matching = get_person(
            {"person_id": alpha_id, "name": "  selector   alpha  "}
        )
        if not matching.ok:
            raise SystemExit(f"matching selectors should succeed: {matching}")
        if ALPHA_NOTE not in matching.output or ALPHA_INTERACTION not in matching.output:
            raise SystemExit("matching selectors did not return the selected private profile")
        if (
            probe._store.private_note_sql_reads != 1
            or probe.interaction_queries != 1
            or probe.interaction_row_reads != 1
        ):
            raise SystemExit(
                "matching selectors did not open one exact private-read path: "
                f"notes={probe._store.private_note_sql_reads}, "
                f"interaction_queries={probe.interaction_queries}, "
                f"interaction_rows={probe.interaction_row_reads}"
            )


def test_single_selectors_and_direct_planner_aliases_are_preserved() -> None:
    with TemporaryDirectory(prefix="jarvis-get-person-single-") as temp:
        _, probe, get_person, alpha_id, _ = _fixture(Path(temp))
        for args in ({"person_id": alpha_id}, {"name": "selector alpha"}):
            probe.reset_counts()
            result = get_person(dict(args))
            if not result.ok or ALPHA_NOTE not in result.output:
                raise SystemExit(f"ordinary single-selector read regressed: {args} -> {result}")
            if (
                probe._store.private_note_sql_reads != 1
                or probe.interaction_queries != 1
            ):
                raise SystemExit(f"single-selector read missed its ordinary private path: {args}")

        aliases = (
            (f"person #{alpha_id} please", {"person_id": alpha_id}),
            (f"person {ALPHA_NAME}", {"name": ALPHA_NAME}),
        )
        for command, expected_args in aliases:
            actions = RuleBasedPlanner().plan(command).actions
            if (
                len(actions) != 1
                or actions[0].tool_name != "get_person"
                or actions[0].args != expected_args
            ):
                raise SystemExit(f"direct get_person planner alias regressed: {command!r}")


def test_malformed_and_missing_selectors_remain_fail_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-get-person-invalid-") as temp:
        root = Path(temp)
        _, probe, get_person, alpha_id, _ = _fixture(root)
        cases = (
            ({}, "missing_person"),
            ({"person_id": alpha_id, "name": None}, "invalid_person_selector"),
            ({"person_id": alpha_id, "name": ""}, "invalid_person_selector"),
            ({"person_id": alpha_id, "name": [ALPHA_NAME]}, "invalid_person_selector"),
            (
                {"person_id": alpha_id, "name": _HostileSelector()},
                "invalid_person_selector",
            ),
            (
                {"person_id": alpha_id, "name": "/private/tmp/selector-name"},
                "invalid_person_selector",
            ),
            ({"person_id": alpha_id, "name": "\ud800"}, "invalid_person_selector"),
            ({"person_id": alpha_id, "name": "X" * 121}, "invalid_person_selector"),
            ({"person_id": None, "name": ALPHA_NAME}, "invalid_person_selector"),
            ({"person_id": "not-an-id", "name": ALPHA_NAME}, "bad_person_id"),
            ({"person_id": alpha_id + 999, "name": ALPHA_NAME}, "not_found"),
        )
        for args, expected_reason in cases:
            probe.reset_counts()
            result = get_person(dict(args))
            if result.ok or result.metadata.get("reason") != expected_reason:
                raise SystemExit(
                    f"malformed/missing selector path did not fail closed: "
                    f"{args!r} -> {result}"
                )
            _assert_no_private_reads(probe, label=f"invalid selector {expected_reason}")
            _assert_content_free(
                result,
                label=f"invalid selector {expected_reason}",
                forbidden=(
                    ALPHA_NAME,
                    ALPHA_NOTE,
                    ALPHA_INTERACTION,
                    "/private/tmp/selector-name",
                    "PRIVATE-HOSTILE-SELECTOR-DETAIL",
                    "Missing Synthetic Person",
                    str(root),
                ),
            )

        for args in (
            {"person_id": alpha_id + 999},
            {"name": "Missing Synthetic Person"},
        ):
            probe.reset_counts()
            result = get_person(dict(args))
            if result.ok or result.metadata.get("reason") != "not_found":
                raise SystemExit(f"ordinary missing-person read regressed: {args!r}")
            if probe.interaction_queries != 0 or probe.interaction_row_reads != 0:
                raise SystemExit(f"missing-person read queried interactions: {args!r}")


def test_stale_and_ambiguous_identity_paths_are_content_free() -> None:
    with TemporaryDirectory(prefix="jarvis-get-person-stale-") as temp:
        root = Path(temp)
        store, probe, get_person, alpha_id, _ = _fixture(root)
        now = "2026-07-14T00:00:01Z"
        with store.connect() as conn:
            stale_id = int(
                conn.execute(
                    """
                    INSERT INTO people(name, relation, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (" selector   alpha ", "stale", BETA_NOTE, now, now),
                ).lastrowid
            )

        probe.reset_counts()
        stale = get_person({"person_id": stale_id, "name": ALPHA_NAME})
        if stale.ok or stale.metadata.get("reason") != "noncanonical_person_id":
            raise SystemExit(f"stale identity did not fail closed: {stale}")
        _assert_no_private_reads(probe, label="stale identity")
        _assert_content_free(
            stale,
            label="stale identity",
            forbidden=(ALPHA_NAME, ALPHA_NOTE, BETA_NOTE, ALPHA_INTERACTION, str(root)),
        )

        probe.reset_counts()
        probe.hide_name_resolution = True
        missing_owner = get_person({"person_id": alpha_id, "name": ALPHA_NAME})
        probe.hide_name_resolution = False
        if missing_owner.ok or missing_owner.metadata.get("reason") != "stale_person_identity":
            raise SystemExit(f"missing canonical identity did not fail closed: {missing_owner}")
        _assert_no_private_reads(probe, label="missing canonical identity")

        probe.reset_counts()
        probe.name_resolution_error = ValueError(
            "PRIVATE-AMBIGUOUS-IDENTITY-DETAIL"
        )
        ambiguous = get_person({"person_id": alpha_id, "name": ALPHA_NAME})
        if ambiguous.ok or ambiguous.metadata.get("reason") != "ambiguous_person_identity":
            raise SystemExit(f"ambiguous identity did not fail closed: {ambiguous}")
        _assert_no_private_reads(probe, label="ambiguous identity")
        _assert_content_free(
            ambiguous,
            label="ambiguous identity",
            forbidden=(
                ALPHA_NAME,
                ALPHA_NOTE,
                ALPHA_INTERACTION,
                "PRIVATE-AMBIGUOUS-IDENTITY-DETAIL",
                str(root),
            ),
        )


def main() -> None:
    test_conflict_is_content_free_and_match_opens_private_read_gate()
    test_single_selectors_and_direct_planner_aliases_are_preserved()
    test_malformed_and_missing_selectors_remain_fail_closed()
    test_stale_and_ambiguous_identity_paths_are_content_free()
    print("get_person selector privacy smoke passed")


if __name__ == "__main__":
    main()
