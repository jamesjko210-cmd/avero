from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.memory.store import PersonIdentityUnavailable, PersonRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.people import _metadata_bool, _people_handoff_metadata


class HostileRow:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-row {self.marker}>"


def assert_route(command: str, tool_name: str, args: dict | None = None) -> None:
    plan = RuleBasedPlanner().plan(command)
    actions = plan.actions
    if len(actions) != 1 or actions[0].tool_name != tool_name:
        raise SystemExit(f"{command!r} should route to {tool_name}: {[(a.tool_name, a.args) for a in actions]}")
    expected_args = args or {}
    if actions[0].args != expected_args:
        raise SystemExit(f"{command!r} should pass {expected_args}: {actions[0].args}")


def assert_unclaimed(command: str) -> None:
    plan = RuleBasedPlanner().plan(command)
    if plan.actions:
        raise SystemExit(f"{command!r} should stay unclaimed: {[(a.tool_name, a.args) for a in plan.actions]}")


def test_people_read_alias_routes() -> None:
    for command in (
        "people please",
        "person list please",
        "people list please",
        "show latest people",
        "show saved people please",
        "who do you know",
        "who are my people",
        "people I know please",
        # Real gap found live 2026-07-09: "show my people" fell through to
        # chat while "show saved people" and "who are my people" both worked.
        "show my people",
        # Real gap found live 2026-07-10, same class: "show my people" worked
        # but bare "my people"/"list my people" and any "recent" qualifier
        # fell through to chat.
        "my people",
        "list my people",
        "recent people",
        "show my recent people",
        "list my recent people",
        "my recent people",
        # Real gap found live 2026-07-10 (round 43): trailing "please" broke
        # this whole exact-match set, same class as the notes fix -- see
        # smoke_test_notes.py's matching comment for the full root cause.
        # Fixed with a local trailing-only strip.
        "show my recent people please",
    ):
        assert_route(command, "list_people")
    assert_route("person the operator", "get_person", {"name": "the operator"})
    assert_route("person #1 please", "get_person", {"person_id": 1})
    for command in ("contacts please", "show contacts"):
        assert_unclaimed(command)
    # Real gap found live 2026-07-10, same privacy-relevant misroute class as
    # the round-21/26/27/28 notes/tasks/files/memory/goal word-order fixes:
    # "search for the/a person/people named X" leaked to a public web_lookup
    # search. "person" is ambiguous between saved People profiles and macOS
    # Contacts, so rather than guess a target tool, the fix is a targeted
    # exclusion -- this should stay unclaimed (fall through to chat) instead
    # of leaking to the web.
    for command in (
        "search for the person named john",
        "search for a person named john",
        "search for the people named john",
    ):
        for action in RuleBasedPlanner().plan(command).actions:
            if action.tool_name == "web_lookup":
                raise SystemExit(f"person query should not leak to a public web search: {command!r} -> {action}")


def assert_runtime_routes_people_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-people-count-alias-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for text in (
            "people count",
            "person count",
            "count people",
            "how many people do i have",
            "사람 몇 명",
            "인물 몇 개",
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [action.get("tool_name") for action in planned] != ["list_people"]:
                raise SystemExit(f"runtime missed people-count alias {text!r}: {planned} / {result.response!r}")
            if len(result.tool_results) != 1 or result.tool_results[0].tool_name != "list_people":
                raise SystemExit(f"people-count alias should execute one list_people tool for {text!r}: {result.tool_results}")
            metadata = result.tool_results[0].metadata
            if "Count: 0" not in result.response or metadata.get("total_people") != 0:
                raise SystemExit(f"empty people-count alias should answer with zero count for {text!r}: {result.response!r} / {metadata}")
            for key in (
                "writes_files",
                "writes_memory",
                "writes_notes",
                "queues_approval",
                "controls_computer",
                "reads_private_data",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
            ):
                if metadata.get(key):
                    raise SystemExit(f"people-count alias unexpectedly set {key} for {text!r}: {metadata}")
        if runtime.store.list_people(limit=100):
            raise SystemExit("people count aliases must not create or mutate people.")
        if runtime.store.list_memories(limit=100):
            raise SystemExit("people count aliases must not create memory rows.")


def find_id_bound_person_projection(root: Path, person_id: int, *, label: str) -> Path:
    matches = [
        path
        for path in (root / "People").glob("*.md")
        if path.name.endswith(f" [{person_id}].md")
    ]
    if len(matches) != 1:
        raise SystemExit(f"{label} expected one ID-bound person mirror for #{person_id}: {matches}")
    return matches[0]


def assert_write_receipt(result, *, root: Path, label: str) -> None:
    metadata = result.metadata
    path_display = metadata.get("path_display")
    if not isinstance(path_display, str) or not path_display.startswith("People/"):
        raise SystemExit(f"{label} missed safe People path_display: {metadata}")
    person_id = metadata.get("person_id")
    if type(person_id) is not int or path_display != f"People/person-{person_id}.md":
        raise SystemExit(f"{label} metadata path_display was not opaque and ID-bound: {metadata}")
    if "path" in metadata:
        raise SystemExit(f"{label} exposed absolute path metadata: {metadata}")
    if result.tool_name in {"add_person", "log_interaction"}:
        if "Person note updated." not in result.output:
            raise SystemExit(f"{label} missed person-note update receipt: {result.output}")
        matching = [
            path
            for path in (root / "People").glob("*.md")
            if path.name.endswith(f" [{person_id}].md")
        ]
        if len(matching) != 1:
            raise SystemExit(f"{label} did not publish one ID-bound person mirror: {matching}")
        exposed = result.output + json.dumps(metadata, sort_keys=True, default=str)
        if any(fragment in exposed for fragment in (str(root), "/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
            raise SystemExit(f"{label} leaked a local path in output or metadata: {exposed}")
        return
    marker = "Saved note: "
    if marker not in result.output:
        raise SystemExit(f"{label} missed safe saved-note receipt: {result.output}")
    output_display = result.output.rsplit(marker, 1)[1].strip()
    output_path = Path(output_display)
    resolved_output_path = root / output_path
    if (
        output_path.is_absolute()
        or ".." in output_path.parts
        or not output_display.startswith("People/")
        or not resolved_output_path.exists()
    ):
        raise SystemExit(f"{label} output note display was not a safe existing mirror: {result.output}")
    canonical_path = find_id_bound_person_projection(root, person_id, label=label)
    if resolved_output_path != canonical_path:
        raise SystemExit(
            f"{label} user-facing note display did not identify the ID-bound mirror: "
            f"{resolved_output_path} / {canonical_path}"
        )
    exposed = result.output + json.dumps(metadata, sort_keys=True, default=str)
    if any(fragment in exposed for fragment in (str(root), "/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} leaked a local path in output or metadata: {exposed}")


def assert_people_contract(
    metadata: dict,
    handoff: dict,
    label: str,
    *,
    prefix: str,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
) -> None:
    for key, expected in (
        ("handoff_ready", True),
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if key == "handoff_ready":
            if handoff.get(key) != expected or metadata.get(f"{prefix}_handoff_ready") != expected:
                raise SystemExit(f"{label} missed people contract {key}={expected}: metadata={metadata} handoff={handoff}")
            continue
        if metadata.get(key) != expected or handoff.get(key) != expected:
            raise SystemExit(f"{label} missed people contract {key}={expected}: metadata={metadata} handoff={handoff}")
        prefixed_key = f"{prefix}_{key}"
        if metadata.get(prefixed_key) != expected:
            raise SystemExit(f"{label} missed prefixed people contract {prefixed_key}={expected}: metadata={metadata}")
    next_commands = list(handoff.get("next_commands") or [])
    expected_next = next_commands[0] if next_commands else ""
    if (
        handoff.get("next_safe_command") != expected_next
        or metadata.get("next_safe_command") != expected_next
        or metadata.get(f"{prefix}_next_safe_command") != expected_next
    ):
        raise SystemExit(f"{label} missed next_safe_command parity: metadata={metadata} handoff={handoff}")
    if (
        handoff.get("next_safe_commands") != next_commands
        or metadata.get("next_safe_commands") != next_commands
        or metadata.get(f"{prefix}_next_safe_commands") != next_commands
    ):
        raise SystemExit(f"{label} missed next_safe_commands parity: metadata={metadata} handoff={handoff}")
    if (
        handoff.get("next_safe_command_count") != len(next_commands)
        or metadata.get("next_safe_command_count") != len(next_commands)
        or metadata.get(f"{prefix}_next_safe_command_count") != len(next_commands)
    ):
        raise SystemExit(f"{label} missed next_safe_command_count parity: metadata={metadata} handoff={handoff}")


def assert_people_mutation_handoff(
    metadata: dict,
    label: str,
    *,
    source: str,
    mutation: str,
    changed: list[str],
) -> None:
    handoff = metadata.get("people_mutation_handoff")
    if metadata.get("people_mutation_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed people_mutation_handoff readiness: {metadata}")
    if handoff.get("source") != source or handoff.get("mutation") != mutation:
        raise SystemExit(f"{label} mutation handoff missed source/mutation parity: {handoff}")
    if handoff.get("person_id") != metadata.get("person_id"):
        raise SystemExit(f"{label} mutation handoff missed person id parity: {metadata}")
    assert_people_contract(
        metadata,
        handoff,
        label,
        prefix="people_mutation",
        state_changed=True,
        changed=changed,
        content_in_handoff=False,
    )
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries.get("read_only") is not False:
        raise SystemExit(f"{label} mutation handoff missed write boundary: {handoff}")
    for key in ("writes_files", "writes_memory", "writes_notes"):
        if metadata.get(key) is not True or boundaries.get(key) is not True:
            raise SystemExit(f"{label} mutation handoff should keep {key}=True: metadata={metadata} handoff={handoff}")
    for key in (
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if metadata.get(key) is not False or boundaries.get(key) is not False:
            raise SystemExit(f"{label} mutation handoff should keep {key}=False: metadata={metadata} handoff={handoff}")
    if any(fragment in str(handoff) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} mutation handoff leaked a local path: {handoff}")


def assert_people_list_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("people_list_handoff")
    if metadata.get("people_list_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed people_list_handoff readiness: {metadata}")
    assert_people_contract(
        metadata,
        handoff,
        label,
        prefix="people_list",
        state_changed=False,
        changed=[],
        content_in_handoff=False,
    )


def assert_people_read_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("people_read_handoff")
    if metadata.get("people_read_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed people_read_handoff readiness: {metadata}")
    if handoff.get("source") != "get_person" or handoff.get("person_id") != metadata.get("person_id"):
        raise SystemExit(f"{label} read handoff missed source/person parity: {handoff}")
    if handoff.get("interaction_count") != metadata.get("interactions"):
        raise SystemExit(f"{label} read handoff missed interaction parity: {metadata}")
    assert_people_contract(
        metadata,
        handoff,
        label,
        prefix="people_read",
        state_changed=False,
        changed=[],
        content_in_handoff=False,
    )
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} read handoff missed read-only boundary: {handoff}")
    for key in (
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if boundaries.get(key):
            raise SystemExit(f"{label} read handoff should keep {key}=False: {handoff}")


def assert_people_refusal_handoff(
    result,
    *,
    source: str,
    mutation: str,
    reason: str,
    forbidden_values: tuple[str, ...] = (),
    person_id: int | None = None,
) -> None:
    metadata = result.metadata
    handoff = metadata.get("people_refusal_handoff")
    if result.ok or not isinstance(handoff, dict):
        raise SystemExit(f"{source} {reason} missed people_refusal_handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("mutation") != mutation:
        raise SystemExit(f"{source} {reason} handoff missed source/mutation parity: {handoff}")
    if metadata.get("reason") != reason or handoff.get("reason") != reason:
        raise SystemExit(f"{source} {reason} handoff missed reason parity: {metadata}")
    if handoff.get("ready_for_operator") is not True or handoff.get("refused") is not True:
        raise SystemExit(f"{source} {reason} handoff missed refusal readiness: {handoff}")
    if handoff.get("person_id") != person_id or metadata.get("person_id") != person_id:
        raise SystemExit(
            f"{source} {reason} refusal person-id mismatch: expected={person_id} metadata={metadata}"
        )
    if handoff.get("changed") != []:
        raise SystemExit(f"{source} {reason} refusal should report no changed fields: {handoff}")
    assert_people_contract(
        metadata,
        handoff,
        f"{source} {reason}",
        prefix="people_refusal",
        state_changed=False,
        changed=[],
        content_in_handoff=False,
    )
    commands = handoff.get("next_commands")
    if not isinstance(commands, list) or "people" not in commands or not isinstance(handoff.get("retry_command"), str):
        raise SystemExit(f"{source} {reason} handoff missed recovery commands: {handoff}")
    for raw_key in ("raw_name", "raw_person_id"):
        if raw_key in metadata or raw_key in handoff:
            raise SystemExit(f"{source} {reason} leaked {raw_key} into refusal metadata: {metadata}")
    exposed = json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str)
    for forbidden in forbidden_values:
        if forbidden and forbidden in exposed:
            raise SystemExit(
                f"{source} {reason} leaked private refusal input {forbidden!r}: {metadata}"
            )
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
        raise SystemExit(f"{source} {reason} handoff missed read-only boundary: {handoff}")
    for key in (
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "calls_model",
        "executes_tools",
        "reads_private_data",
        "creates_person",
        "creates_interaction",
        "authorizes_retry",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if boundaries.get(key) is not False:
            raise SystemExit(f"{source} {reason} handoff should keep {key}=False: {handoff}")
    for key in (
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "authorizes_retry",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if metadata.get(key) is not False:
            raise SystemExit(f"{source} {reason} flat metadata should keep {key}=False: {metadata}")
    if any(fragment in str(handoff) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{source} {reason} handoff leaked a local path: {handoff}")


def assert_people_exact_metadata_bool() -> None:
    if _metadata_bool(True) is not True:
        raise SystemExit("people exact bool helper should preserve True.")
    if _metadata_bool(False, default=True) is not False:
        raise SystemExit("people exact bool helper should preserve False.")
    for value in ("true", "false", "yes", "0", 1, 0, [], ["content"], None):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"people exact bool helper should reject malformed handoff flags: {value!r}")
    if _metadata_bool("fallback", default=True) is not True:
        raise SystemExit("people exact bool helper should honor explicit malformed-value default.")


def assert_people_malformed_handoff_flags() -> None:
    handoff = {
        "source": "list_people",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["people"],
        "boundaries": {"read_only": True},
    }
    metadata = _people_handoff_metadata("people_list_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("people_list_state_changed") is not False:
        raise SystemExit(f"malformed people state_changed should not become truthy: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("people_list_content_in_handoff") is not False:
        raise SystemExit(f"malformed people content_in_handoff should not become truthy: {metadata}")


def assert_person_store_retry_safety() -> None:
    with TemporaryDirectory(prefix="jarvis-people-store-retry-") as temp:
        runtime = make_temp_runtime(Path(temp))
        store = runtime.store
        person_id = store.upsert_person(
            PersonRecord(name="Direct Retry Person", relation="friend", notes="same note block")
        )
        repeated_id = store.upsert_person(
            PersonRecord(name=" direct   retry person ", relation="colleague", notes="same note block")
        )
        if repeated_id != person_id:
            raise SystemExit("direct person retry did not resolve to the original identity owner")
        normalized_lookup = store.get_person(name="  DIRECT   RETRY PERSON  ")
        if normalized_lookup is None or int(normalized_lookup["id"]) != person_id:
            raise SystemExit(f"normalized person lookup missed the deterministic owner: {normalized_lookup}")
        row = store.get_person(person_id=person_id)
        if row is None or row["notes"] != "same note block" or row["relation"] != "colleague":
            raise SystemExit(f"direct person retry duplicated its trailing note or missed its update: {row}")
        store.upsert_person(
            PersonRecord(name="DIRECT RETRY PERSON", notes="genuinely different note")
        )
        row = store.get_person(person_id=person_id)
        if row is None or row["notes"] != "same note block\ngenuinely different note":
            raise SystemExit(f"direct person update did not append a distinct note: {row}")
        store.upsert_person(
            PersonRecord(name="Direct Retry Person", notes="genuinely different note")
        )
        row = store.get_person(person_id=person_id)
        if row is None or row["notes"] != "same note block\ngenuinely different note":
            raise SystemExit(f"direct person retry duplicated a trailing note block: {row}")

        now = "2026-07-11T00:00:00+00:00"
        with store.connect() as conn:
            first_shadow = conn.execute(
                """
                INSERT INTO people(name, relation, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (" Legacy Person ", "owner", "owner note", now, now),
            ).lastrowid
            second_shadow = conn.execute(
                """
                INSERT INTO people(name, relation, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("legacy   person", "shadow", "shadow note", now, now),
            ).lastrowid
        store = type(store)(store.db_path)
        store.init()
        owner_id = store.upsert_person(
            PersonRecord(name="ＬＥＧＡＣＹ PERSON", relation="updated", notes="owner note")
        )
        if owner_id != first_shadow:
            raise SystemExit(f"legacy person shadows did not select the lowest-ID owner: {owner_id}")
        with store.connect() as conn:
            owner = conn.execute("SELECT * FROM people WHERE id = ?", (first_shadow,)).fetchone()
            shadow = conn.execute("SELECT * FROM people WHERE id = ?", (second_shadow,)).fetchone()
        if (
            owner is None
            or owner["relation"] != "updated"
            or owner["notes"] != "owner note"
            or shadow is None
            or shadow["relation"] != "shadow"
            or shadow["notes"] != "shadow note"
        ):
            raise SystemExit(f"legacy person shadow ownership was not preserved: {owner} / {shadow}")


def assert_legacy_path_shaped_person_refusal() -> None:
    with TemporaryDirectory(prefix="jarvis-people-legacy-path-name-") as temp:
        runtime = make_temp_runtime(Path(temp))
        stored_name = "/\x55sers/private/Legacy Person"
        now = "2026-07-12T00:00:00+00:00"
        with runtime.store.connect() as conn:
            person_id = int(
                conn.execute(
                    """
                    INSERT INTO people(name, relation, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (stored_name, "legacy", "legacy", now, now),
                ).lastrowid
            )
            before = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "person_interactions",
                    "memories",
                    "interaction_memory_links",
                    "person_projection_jobs",
                    "memory_projection_jobs",
                )
            }

        result = runtime.registry.get("log_interaction").handler(
            {"person_id": person_id, "summary": "must not be stored"}
        )
        if result.ok or result.metadata.get("reason") != "invalid_name":
            raise SystemExit(f"legacy path-shaped stored name was not refused: {result}")
        assert_people_refusal_handoff(
            result,
            source="log_interaction",
            mutation="interaction_create",
            reason="invalid_name",
            forbidden_values=(stored_name,),
            person_id=person_id,
        )
        exposed = result.output + json.dumps(result.metadata, sort_keys=True, ensure_ascii=False)
        if any(fragment in exposed for fragment in (stored_name, "/\x55sers/", "/private/")):
            raise SystemExit(f"legacy path-shaped stored name leaked through refusal: {exposed}")

        with runtime.store.connect() as conn:
            after = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in before
            }
        if after != before:
            raise SystemExit(f"legacy path-shaped stored name refusal changed state: {before} -> {after}")


def assert_log_interaction_identity_refusals() -> None:
    with TemporaryDirectory(prefix="jarvis-people-interaction-identity-") as temp:
        runtime = make_temp_runtime(Path(temp))
        store = runtime.store
        owner_name = "Legacy Shadow Owner"
        shadow_name = " legacy   shadow owner "
        private_summary = "PRIVATE-SHADOW-INTERACTION"
        owner_id = store.upsert_person(PersonRecord(name=owner_name))
        now = "2026-07-12T00:00:00+00:00"
        with store.connect() as conn:
            shadow_id = int(
                conn.execute(
                    """
                    INSERT INTO people(name, relation, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (shadow_name, "legacy", "shadow", now, now),
                ).lastrowid
            )

        exact_shadow = store.get_person(person_id=shadow_id)
        if exact_shadow is None or int(exact_shadow["id"]) != shadow_id:
            raise SystemExit("exact-ID person reads lost legacy shadow compatibility")
        exact_shadow_read = runtime.registry.get("get_person").handler(
            {"person_id": shadow_id}
        )
        if not exact_shadow_read.ok or exact_shadow_read.metadata.get("person_id") != shadow_id:
            raise SystemExit("get_person lost exact-ID legacy shadow read compatibility")
        canonical = store.get_person(name=owner_name)
        if canonical is None or int(canonical["id"]) != owner_id:
            raise SystemExit("legacy shadow fixture did not retain its canonical owner")

        contract = runtime.registry.get("log_interaction").auto_mutation_contract
        if (
            contract is None
            or contract.semantic_preflight is None
            or contract.semantic_preflight_result_builder is None
        ):
            raise SystemExit("log_interaction identity refusal lost semantic preflight")

        tracked_tables = (
            "people",
            "person_interactions",
            "memories",
            "interaction_memory_links",
            "person_projection_jobs",
            "memory_projection_jobs",
            "auto_mutation_receipts",
        )

        def state() -> dict[str, int]:
            with store.connect() as conn:
                return {
                    table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in tracked_tables
                }

        def assert_refusal(result, reason: str, *, label: str, person_id: int | None) -> None:
            assert_people_refusal_handoff(
                result,
                source="log_interaction",
                mutation="interaction_create",
                reason=reason,
                forbidden_values=(owner_name, shadow_name, private_summary, str(runtime.vault.root_path)),
                person_id=person_id,
            )
            if reason.replace("_", " ") in result.output.lower():
                raise SystemExit(f"{label} exposed an internal refusal code: {result.output}")
            exposed = result.output + json.dumps(
                result.metadata, sort_keys=True, ensure_ascii=False, default=str
            )
            if len(result.output) > 200 or any(
                value in exposed
                for value in (
                    owner_name,
                    shadow_name,
                    private_summary,
                    str(runtime.vault.root_path),
                    "PRIVATE identity reconciliation backlog path",
                )
            ):
                raise SystemExit(f"{label} refusal was unbounded or leaked private content")
            for key in (
                "authorizes_retry",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
                "queues_approval",
                "requires_approval",
                "writes_files",
                "writes_database",
                "writes_memory",
                "writes_notes",
            ):
                if result.metadata.get(key) is not False:
                    raise SystemExit(f"{label} should keep {key}=False: {result.metadata}")

        shadow_args = {"person_id": shadow_id, "summary": private_summary}
        before_shadow = state()
        shadow_reason = contract.semantic_preflight(dict(shadow_args))
        if shadow_reason != "noncanonical_person_id":
            raise SystemExit(f"legacy shadow preflight returned {shadow_reason!r}")
        shadow_preflight = contract.semantic_preflight_result_builder(
            dict(shadow_args), shadow_reason
        )
        assert_refusal(
            shadow_preflight,
            shadow_reason,
            label="legacy shadow preflight",
            person_id=shadow_id,
        )
        if shadow_preflight.metadata.get("handler_invoked") is not False:
            raise SystemExit("legacy shadow preflight claimed handler execution")
        shadow_direct = runtime.registry.get("log_interaction").handler(dict(shadow_args))
        assert_refusal(
            shadow_direct,
            shadow_reason,
            label="legacy shadow direct handler",
            person_id=shadow_id,
        )
        if state() != before_shadow:
            raise SystemExit("legacy shadow preflight/direct refusal performed writes")

        original_get_person = store.get_person

        def unavailable_name_resolution(*, person_id=None, name=None):
            if name is not None:
                raise PersonIdentityUnavailable(
                    "PRIVATE identity reconciliation backlog path"
                )
            return original_get_person(person_id=person_id)

        store.get_person = unavailable_name_resolution  # type: ignore[method-assign]
        unavailable_args = {"person_id": owner_id, "summary": private_summary}
        before_unavailable = state()
        unavailable_reason = contract.semantic_preflight(dict(unavailable_args))
        if unavailable_reason != "identity_index_unavailable":
            raise SystemExit(f"identity backlog preflight returned {unavailable_reason!r}")
        unavailable_preflight = contract.semantic_preflight_result_builder(
            dict(unavailable_args), unavailable_reason
        )
        assert_refusal(
            unavailable_preflight,
            unavailable_reason,
            label="identity backlog preflight",
            person_id=owner_id,
        )
        unavailable_direct = runtime.registry.get("log_interaction").handler(
            dict(unavailable_args)
        )
        assert_refusal(
            unavailable_direct,
            unavailable_reason,
            label="identity backlog direct handler",
            person_id=owner_id,
        )
        if state() != before_unavailable:
            raise SystemExit("identity backlog preflight/direct refusal performed writes")

        unavailable_get = runtime.registry.get("get_person").handler(
            {"name": owner_name}
        )
        assert_people_refusal_handoff(
            unavailable_get,
            source="get_person",
            mutation="person_read",
            reason="identity_index_unavailable",
            forbidden_values=(owner_name, "PRIVATE identity reconciliation backlog path"),
        )

        store.get_person = original_get_person  # type: ignore[method-assign]
        original_record_person = store.record_person_with_projections
        original_record_interaction = store.record_person_interaction_with_projections

        def unavailable_person_write(_record):
            raise PersonIdentityUnavailable("PRIVATE add-person backlog path")

        def unavailable_interaction_write(**_kwargs):
            raise PersonIdentityUnavailable("PRIVATE interaction backlog path")

        store.record_person_with_projections = unavailable_person_write  # type: ignore[method-assign]
        unavailable_add = runtime.registry.get("add_person").handler(
            {"name": owner_name, "notes": private_summary}
        )
        assert_people_refusal_handoff(
            unavailable_add,
            source="add_person",
            mutation="person_create",
            reason="identity_index_unavailable",
            forbidden_values=(
                owner_name,
                private_summary,
                "PRIVATE add-person backlog path",
            ),
        )

        store.record_person_with_projections = original_record_person  # type: ignore[method-assign]
        store.record_person_interaction_with_projections = unavailable_interaction_write  # type: ignore[method-assign]
        unavailable_race = runtime.registry.get("log_interaction").handler(
            {"person_id": owner_id, "summary": private_summary}
        )
        assert_people_refusal_handoff(
            unavailable_race,
            source="log_interaction",
            mutation="interaction_create",
            reason="identity_index_unavailable",
            forbidden_values=(
                owner_name,
                private_summary,
                "PRIVATE interaction backlog path",
            ),
        )
        store.record_person_interaction_with_projections = original_record_interaction  # type: ignore[method-assign]
        if state() != before_unavailable:
            raise SystemExit("identity backlog handler refusals performed writes")


def assert_person_projection_identity_safety() -> None:
    with TemporaryDirectory(prefix="jarvis-people-projection-") as temp:
        runtime = make_temp_runtime(Path(temp))
        root = runtime.vault.root_path
        shared_prefix = "P" * 90
        first_name = shared_prefix + ("A" * 30)
        second_name = shared_prefix + ("B" * 30)
        first = runtime.registry.get("add_person").handler(
            {"name": first_name, "relation": "friend", "notes": "first independent note"}
        )
        second = runtime.registry.get("add_person").handler(
            {"name": second_name, "relation": "colleague", "notes": "second independent note"}
        )
        if not first.ok or not second.ok:
            raise SystemExit(f"colliding-prefix people were not created: {first.output} / {second.output}")
        first_display = first.metadata.get("path_display")
        second_display = second.metadata.get("path_display")
        if not isinstance(first_display, str) or not isinstance(second_display, str):
            raise SystemExit("colliding-prefix people missed safe path_display metadata")
        first_id = int(first.metadata["person_id"])
        second_id = int(second.metadata["person_id"])
        person_notes = list((root / "People").glob("*.md"))
        first_paths = [path for path in person_notes if path.name.endswith(f" [{first_id}].md")]
        second_paths = [path for path in person_notes if path.name.endswith(f" [{second_id}].md")]
        if len(first_paths) != 1 or len(second_paths) != 1:
            raise SystemExit(
                f"colliding-prefix people missed their ID-bound mirrors: {person_notes}"
            )
        first_path = first_paths[0]
        second_path = second_paths[0]
        if "path" in first.metadata or "path" in second.metadata:
            raise SystemExit("colliding-prefix people exposed absolute path metadata")
        if first_path == second_path or not first_path.exists() or not second_path.exists():
            raise SystemExit(f"colliding-prefix people did not receive distinct mirrors: {first_path} / {second_path}")
        if f"[{first_id}]" not in first_path.name or f"[{second_id}]" not in second_path.name:
            raise SystemExit(f"person mirrors missed immutable ID-bearing names: {first_path.name} / {second_path.name}")

        first_interaction = runtime.registry.get("log_interaction").handler(
            {"person_id": first_id, "summary": "first interaction only"}
        )
        second_interaction = runtime.registry.get("log_interaction").handler(
            {"person_id": second_id, "summary": "second interaction only"}
        )
        if not first_interaction.ok or not second_interaction.ok:
            raise SystemExit("colliding-prefix interaction updates failed")
        if first_interaction.metadata.get("path_display") != first_display:
            raise SystemExit("first person interaction changed the stable mirror path")
        if second_interaction.metadata.get("path_display") != second_display:
            raise SystemExit("second person interaction changed the stable mirror path")
        if "path" in first_interaction.metadata or "path" in second_interaction.metadata:
            raise SystemExit("person interaction exposed absolute path metadata")
        first_text = first_path.read_text(encoding="utf-8")
        second_text = second_path.read_text(encoding="utf-8")
        if "first independent note" not in first_text or "first interaction only" not in first_text:
            raise SystemExit("first person mirror lost its independent note or interaction")
        if "second independent note" in first_text or "second interaction only" in first_text:
            raise SystemExit("first person mirror absorbed second-person content")
        if "second independent note" not in second_text or "second interaction only" not in second_text:
            raise SystemExit("second person mirror lost its independent note or interaction")
        if "first independent note" in second_text or "first interaction only" in second_text:
            raise SystemExit("second person mirror absorbed first-person content")

    with TemporaryDirectory(prefix="jarvis-people-legacy-projection-") as temp:
        runtime = make_temp_runtime(Path(temp))
        root = runtime.vault.root_path
        people_root = runtime.vault.root_path / "People"
        store_identity = runtime.store.get_store_identity()

        def person(
            person_id: int,
            name: str,
            notes: str = "note",
            updated_at: str = "2026-07-12T00:00:00+00:00",
        ) -> dict[str, object]:
            return {
                "id": person_id,
                "name": name,
                "relation": "friend",
                "notes": notes,
                "last_contact_at": None,
                "revision": 0,
                "created_at": "2026-07-12T00:00:00+00:00",
                "updated_at": updated_at,
            }

        matching_legacy = people_root / "Legacy Match.md"
        matching_legacy_content = "---\nid: 41\n---\n\n# hand-written same-id note\n"
        matching_legacy.write_text(matching_legacy_content, encoding="utf-8")
        matching_path = runtime.vault.write_person(
            person(41, "Legacy Match"), [], store_identity=store_identity
        )
        if matching_path == matching_legacy or matching_legacy.read_text(encoding="utf-8") != matching_legacy_content:
            raise SystemExit("same-ID unowned legacy person note was overwritten")
        if runtime.vault.write_person(
            person(41, "Legacy Match", "updated note"), [], store_identity=store_identity
        ) != matching_path:
            raise SystemExit("owned canonical person mirror path was not stable on re-export")

        unowned_legacy = people_root / "Legacy Unowned.md"
        unowned_content = "# hand-written person note\n"
        unowned_legacy.write_text(unowned_content, encoding="utf-8")
        unowned_path = runtime.vault.write_person(
            person(42, "Legacy Unowned"), [], store_identity=store_identity
        )
        if unowned_path == unowned_legacy or unowned_legacy.read_text(encoding="utf-8") != unowned_content:
            raise SystemExit("unowned legacy person note was overwritten")

        foreign_legacy = people_root / "Legacy Foreign.md"
        foreign_content = "---\nid: 999\n---\n\n# foreign person\n"
        foreign_legacy.write_text(foreign_content, encoding="utf-8")
        foreign_path = runtime.vault.write_person(
            person(43, "Legacy Foreign"), [], store_identity=store_identity
        )
        if foreign_path == foreign_legacy or foreign_legacy.read_text(encoding="utf-8") != foreign_content:
            raise SystemExit("foreign-ID legacy person note was overwritten")

        conflict = people_root / "Canonical Conflict [44].md"
        conflict_content = "---\nid: 999\n---\n\n# conflicting canonical owner\n"
        conflict.write_text(conflict_content, encoding="utf-8")
        try:
            runtime.vault.write_person(
                person(44, "Canonical Conflict"), [], store_identity=store_identity
            )
        except FileExistsError:
            pass
        else:
            raise SystemExit("conflicting ID-bearing person destination did not fail closed")
        if conflict.read_text(encoding="utf-8") != conflict_content:
            raise SystemExit("conflicting ID-bearing person destination changed on refusal")

        same_id_conflict = people_root / "Canonical Same Id [46].md"
        same_id_content = "---\nid: 46\n---\n\n# hand-written same-id canonical note\n"
        same_id_conflict.write_text(same_id_content, encoding="utf-8")
        try:
            runtime.vault.write_person(
                person(46, "Canonical Same Id"), [], store_identity=store_identity
            )
        except FileExistsError:
            pass
        else:
            raise SystemExit("same-ID unowned canonical person destination did not fail closed")
        if same_id_conflict.read_text(encoding="utf-8") != same_id_content:
            raise SystemExit("same-ID unowned canonical person destination was overwritten")

        stable = runtime.vault.write_person(
            person(45, "Stable Export"), [], store_identity=store_identity
        )
        if runtime.vault.write_person(
            person(45, "Stable Export", "later note"), [], store_identity=store_identity
        ) != stable:
            raise SystemExit("canonical person mirror path changed on re-export")

        crlf_path = runtime.vault.write_person(
            person(47, "CRLF Person"), [], store_identity=store_identity
        )
        crlf_path.write_bytes(crlf_path.read_bytes().replace(b"\n", b"\r\n"))
        if runtime.vault.write_person(
            person(47, "CRLF Person", "updated after CRLF", "2026-07-12T00:00:01+00:00"),
            [],
            store_identity=store_identity,
        ) != crlf_path:
            raise SystemExit("owned CRLF person projection changed its stable path")
        if "updated after CRLF" not in crlf_path.read_text(encoding="utf-8"):
            raise SystemExit("owned CRLF person projection was not updated")

        stale_person_id = runtime.store.upsert_person(
            PersonRecord(name="Stale Snapshot", relation="friend", notes="older")
        )
        stale_person = runtime.store.get_person(person_id=stale_person_id)
        runtime.store.upsert_person(
            PersonRecord(name="Stale Snapshot", relation="colleague", notes="newer")
        )
        fresh_person = runtime.store.get_person(person_id=stale_person_id)
        stale_path = runtime.vault.write_person(
            fresh_person, [], store_identity=store_identity
        )
        try:
            runtime.vault.write_person(stale_person, [], store_identity=store_identity)
        except RuntimeError:
            pass
        else:
            raise SystemExit("stale person snapshot was allowed to replace a newer mirror")
        if "colleague" not in stale_path.read_text(encoding="utf-8"):
            raise SystemExit("stale person refusal did not preserve the newer mirror")

        unicode_result = runtime.registry.get("add_person").handler(
            {"name": "가상연락처이" * 20, "relation": "friend", "notes": "unicode path"}
        )
        unicode_display = unicode_result.metadata.get("path_display")
        if not isinstance(unicode_display, str):
            raise SystemExit(f"long Unicode person name missed path_display: {unicode_result}")
        unicode_id = int(unicode_result.metadata["person_id"])
        unicode_paths = [
            path
            for path in (root / "People").glob("*.md")
            if path.name.endswith(f" [{unicode_id}].md")
        ]
        if not unicode_result.ok or len(unicode_paths) != 1:
            raise SystemExit(f"long Unicode person name did not receive a mirror: {unicode_result}")
        unicode_path = unicode_paths[0]
        if "path" in unicode_result.metadata:
            raise SystemExit("long Unicode person result exposed absolute path metadata")
        if len(unicode_path.name.encode("utf-8")) > 255:
            raise SystemExit("person mirror filename exceeded the filesystem byte limit")



def main() -> None:
    test_people_read_alias_routes()
    assert_runtime_routes_people_count_aliases()
    assert_people_exact_metadata_bool()
    assert_people_malformed_handoff_flags()
    assert_person_store_retry_safety()
    assert_legacy_path_shaped_person_refusal()
    assert_log_interaction_identity_refusals()
    assert_person_projection_identity_safety()
    with TemporaryDirectory(prefix="jarvis-people-") as temp:
        runtime = make_temp_runtime(Path(temp))
        root = runtime.vault.root_path
        cases = [
            "add person Maya relation collaborator notes likes direct project updates",
            "log interaction with Maya: discussed Jarvis memory and next assistant functions",
            "people",
            "show person Maya",
            "search memory for project updates",
            "export state",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1600])
            print()
            if case == "add person Maya relation collaborator notes likes direct project updates":
                metadata = result.tool_results[0].metadata
                if metadata.get("person_id") != 1 or metadata.get("writes_files") is not True:
                    raise SystemExit("add_person missed durable write metadata.")
                if metadata.get("writes_memory") is not True or metadata.get("writes_notes") is not True:
                    raise SystemExit("add_person missed memory/note write metadata.")
                if metadata.get("reads_private_data") is not False or metadata.get("controls_computer") is not False:
                    raise SystemExit("add_person missed safety metadata.")
                if metadata.get("requires_approval") or metadata.get("external_side_effect"):
                    raise SystemExit("add_person should stay local-safe without external side effects.")
                assert_people_mutation_handoff(metadata, "runtime add_person", source="add_person", mutation="person_create", changed=["person"])
                assert_write_receipt(result.tool_results[0], root=root, label="runtime add_person")
            if case == "log interaction with Maya: discussed Jarvis memory and next assistant functions":
                metadata = result.tool_results[0].metadata
                if metadata.get("person_id") != 1 or metadata.get("interaction_id") != 1:
                    raise SystemExit("log_interaction missed person/interaction metadata.")
                assert_people_mutation_handoff(metadata, "runtime log_interaction", source="log_interaction", mutation="interaction_create", changed=["interaction"])
                assert_write_receipt(result.tool_results[0], root=root, label="runtime log_interaction")
            if case == "people":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 25 or metadata.get("writes_files") is not False:
                    raise SystemExit("list_people missed limit/read-only metadata.")
                if "Count:" not in result.response or metadata.get("total_people") != metadata.get("count"):
                    raise SystemExit(f"list_people missed visible/metadata count parity: {result.response!r} / {metadata}")
                handoff = metadata.get("people_list_handoff")
                if not isinstance(handoff, dict) or handoff.get("source") != "list_people":
                    raise SystemExit(f"list_people missed structured handoff metadata: {metadata}")
                if handoff.get("ready_for_operator") is not True or handoff.get("count") != metadata.get("count"):
                    raise SystemExit(f"list_people handoff missed readiness/count parity: {handoff}")
                assert_people_list_handoff(metadata, "runtime list_people")
                if handoff.get("limit") != 25 or handoff.get("person_ids") != [1]:
                    raise SystemExit(f"list_people handoff missed limit/id parity: {handoff}")
                rows = handoff.get("rows")
                if not isinstance(rows, list) or rows[0].get("show_command") != "show person 1":
                    raise SystemExit(f"list_people handoff missed ID-only show command: {handoff}")
                if rows[0].get("log_interaction_command") != "log interaction with 1: <summary>":
                    raise SystemExit(f"list_people handoff missed ID-only interaction command: {handoff}")
                if handoff.get("first_show_command") != "show person 1" or "people" not in handoff.get("next_commands", []):
                    raise SystemExit(f"list_people handoff missed next commands: {handoff}")
                if "Maya" in json.dumps(metadata, sort_keys=True, ensure_ascii=False):
                    raise SystemExit(f"list_people metadata leaked a raw person name: {metadata}")
                boundaries = handoff.get("boundaries")
                if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
                    raise SystemExit(f"list_people handoff missed read-only boundary: {handoff}")
                for key in (
                    "writes_files",
                    "writes_memory",
                    "writes_notes",
                    "queues_approval",
                    "controls_computer",
                    "external_side_effect",
                    "authorizes_execution",
                    "authorizes_completion_claim",
                    "approval_granted",
                ):
                    if boundaries.get(key):
                        raise SystemExit(f"list_people handoff should keep {key}=False: {handoff}")
            if case == "show person Maya":
                metadata = result.tool_results[0].metadata
                if metadata.get("person_id") != 1 or metadata.get("interactions") != 1:
                    raise SystemExit("get_person missed person interaction metadata.")
                assert_people_read_handoff(metadata, "runtime show person")
                if "Maya" in json.dumps(metadata, sort_keys=True, ensure_ascii=False):
                    raise SystemExit(f"get_person metadata leaked a raw person name: {metadata}")

        direct_list = runtime.registry.get("list_people").handler({"limit": "bad"})
        if not direct_list.ok or direct_list.metadata.get("limit") != 25 or direct_list.metadata.get("raw_limit") != "bad":
            raise SystemExit("list_people did not sanitize a bad limit.")
        if direct_list.metadata.get("writes_files") is not False or direct_list.metadata.get("reads_private_data") is not False:
            raise SystemExit("list_people missed safety metadata.")
        if direct_list.metadata.get("people_list_handoff", {}).get("limit") != 25:
            raise SystemExit(f"list_people handoff should use sanitized limit: {direct_list.metadata}")
        assert_people_list_handoff(direct_list.metadata, "direct list bad limit")
        direct_list_bool = runtime.registry.get("list_people").handler({"limit": False})
        if not direct_list_bool.ok or direct_list_bool.metadata.get("limit") != 25 or direct_list_bool.metadata.get("raw_limit") != "False":
            raise SystemExit(f"list_people should treat boolean limits as malformed defaults: {direct_list_bool.metadata}")
        if direct_list_bool.metadata.get("writes_files") or direct_list_bool.metadata.get("writes_memory"):
            raise SystemExit(f"list_people boolean limit should stay read-only: {direct_list_bool.metadata}")
        direct_list_long_limit = runtime.registry.get("list_people").handler({"limit": "l" * 200})
        if direct_list_long_limit.metadata.get("raw_limit") != ("l" * 77 + "..."):
            raise SystemExit(f"list_people did not bound raw bad limit metadata: {direct_list_long_limit.metadata}")
        for path_limit in (
            "/\x55sers/example/private/people-limit",
            "/var/folders/zc/jarvis/people-limit",
            "/tmp/jarvis/people-limit",
        ):
            path_bad_list = runtime.registry.get("list_people").handler({"limit": path_limit})
            if path_bad_list.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"list_people leaked local path in raw limit metadata: {path_bad_list.metadata}")

        direct_person = runtime.registry.get("get_person").handler({"name": "Maya", "limit": "bad"})
        if not direct_person.ok or direct_person.metadata.get("limit") != 10 or direct_person.metadata.get("raw_limit") != "bad":
            raise SystemExit("get_person did not sanitize a bad interaction limit.")
        assert_people_read_handoff(direct_person.metadata, "direct get bad limit")
        direct_person_bool_limit = runtime.registry.get("get_person").handler({"name": "Maya", "limit": True})
        if not direct_person_bool_limit.ok or direct_person_bool_limit.metadata.get("limit") != 10 or direct_person_bool_limit.metadata.get("raw_limit") != "True":
            raise SystemExit(f"get_person should treat boolean limits as malformed defaults: {direct_person_bool_limit.metadata}")
        direct_person_large = runtime.registry.get("get_person").handler({"name": "Maya", "limit": 999999})
        if not direct_person_large.ok or direct_person_large.metadata.get("limit") != 200:
            raise SystemExit("get_person did not clamp a large limit.")
        for path_limit in (
            "/private/tmp/jarvis-person-limit",
            "/var/folders/zc/jarvis/person-limit",
            "/tmp/jarvis/person-limit",
        ):
            path_bad_person_limit = runtime.registry.get("get_person").handler({"name": "Maya", "limit": path_limit})
            if path_bad_person_limit.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"get_person leaked local path in raw limit metadata: {path_bad_person_limit.metadata}")

        direct_empty = runtime.registry.get("list_people").handler({"limit": -10})
        if not direct_empty.ok or direct_empty.metadata.get("limit") != 1:
            raise SystemExit("list_people did not clamp a low limit.")
        if direct_empty.metadata.get("people_list_handoff", {}).get("limit") != 1:
            raise SystemExit(f"list_people handoff should use clamped low limit: {direct_empty.metadata}")

        marker = "PEOPLE_HOSTILE_ROW_SHOULD_NOT_LEAK /\x55sers/example/private/people.sqlite"
        original_list_people = runtime.store.list_people
        runtime.store.list_people = lambda limit=25: [
            HostileRow(marker),
            {
                "id": 44,
                "name": "Readable Person",
                "relation": "/private/tmp/person-relation",
                "last_contact_at": "/var/folders/zc/person-last-contact",
            },
        ]
        try:
            hostile_list = runtime.registry.get("list_people").handler({})
        finally:
            runtime.store.list_people = original_list_people
        if not hostile_list.ok:
            raise SystemExit(f"list_people should tolerate malformed people rows: {hostile_list.output}")
        for expected in [
            "hidden malformed people rows: 1",
            "#44 Readable Person | <local-path> | last contact <local-path>",
        ]:
            if expected not in hostile_list.output:
                raise SystemExit(f"list_people malformed-row output missed {expected!r}: {hostile_list.output}")
        leak_text = hostile_list.output + json.dumps(hostile_list.metadata, sort_keys=True)
        for leaked in [
            "PEOPLE_HOSTILE_ROW_SHOULD_NOT_LEAK",
            "/\x55sers/example/private",
            "people.sqlite",
            "/private/tmp/person-relation",
            "/var/folders/zc/person-last-contact",
        ]:
            if leaked in leak_text:
                raise SystemExit(f"list_people leaked hostile row detail {leaked!r}: {leak_text}")
        if (
            hostile_list.metadata.get("count") != 1
            or hostile_list.metadata.get("readable_people_rows") != 1
            or hostile_list.metadata.get("unreadable_people_rows") != 1
        ):
            raise SystemExit(f"list_people missed readable/unreadable people counters: {hostile_list.metadata}")
        hostile_handoff = hostile_list.metadata.get("people_list_handoff") or {}
        if hostile_handoff.get("count") != 1 or hostile_handoff.get("person_ids") != [44]:
            raise SystemExit(f"list_people hostile-row handoff should preserve readable people only: {hostile_handoff}")
        rows = hostile_handoff.get("rows") or []
        if (
            not rows
            or rows[0].get("id") != 44
            or rows[0].get("has_relation") is not True
            or rows[0].get("has_last_contact") is not True
            or rows[0].get("show_command") != "show person 44"
            or rows[0].get("log_interaction_command")
            != "log interaction with 44: <summary>"
        ):
            raise SystemExit(
                f"list_people hostile-row handoff should expose only ID-bound state: {hostile_handoff}"
            )
        metadata_text = json.dumps(hostile_list.metadata, sort_keys=True, ensure_ascii=False)
        if "Readable Person" in metadata_text or "<local-path>" in metadata_text:
            raise SystemExit(f"list_people handoff retained private row content: {hostile_list.metadata}")
        assert_people_list_handoff(hostile_list.metadata, "hostile list people")

        interaction_marker = "PEOPLE_INTERACTION_HOSTILE_ROW_SHOULD_NOT_LEAK /\x55sers/example/private/interactions.sqlite"
        original_get_person = runtime.store.get_person
        original_list_interactions = runtime.store.list_person_interactions
        runtime.store.get_person = lambda person_id=None, name=None: {
            "id": 45,
            "name": "Readable Detail",
            "relation": "/private/tmp/person-detail-relation",
            "notes": "/\x55sers/example/private/person-detail-notes",
            "last_contact_at": "/var/folders/zc/person-detail-last-contact",
        }
        runtime.store.list_person_interactions = lambda person_id, limit=10: [
            HostileRow(interaction_marker),
            {
                "id": 46,
                "happened_at": "/private/tmp/person-interaction-time",
                "summary": "/\x55sers/example/private/person-interaction-summary",
            },
        ]
        try:
            hostile_person = runtime.registry.get("get_person").handler({"name": "Readable Detail"})
        finally:
            runtime.store.get_person = original_get_person
            runtime.store.list_person_interactions = original_list_interactions
        if not hostile_person.ok:
            raise SystemExit(f"get_person should tolerate malformed interaction rows: {hostile_person.output}")
        for expected in [
            "#45 Readable Detail",
            "Relation: <local-path>",
            "Notes: <local-path>",
            "Last contact: <local-path>",
            "hidden malformed interaction rows: 1",
            "- <local-path>: <local-path>",
        ]:
            if expected not in hostile_person.output:
                raise SystemExit(f"get_person malformed-row output missed {expected!r}: {hostile_person.output}")
        leak_text = hostile_person.output + json.dumps(hostile_person.metadata, sort_keys=True)
        for leaked in [
            "PEOPLE_INTERACTION_HOSTILE_ROW_SHOULD_NOT_LEAK",
            "/\x55sers/example/private",
            "interactions.sqlite",
            "/private/tmp/person-detail-relation",
            "/var/folders/zc/person-detail-last-contact",
            "/private/tmp/person-interaction-time",
        ]:
            if leaked in leak_text:
                raise SystemExit(f"get_person leaked hostile interaction detail {leaked!r}: {leak_text}")
        if (
            hostile_person.metadata.get("person_id") != 45
            or hostile_person.metadata.get("interactions") != 1
            or hostile_person.metadata.get("readable_person_interaction_rows") != 1
            or hostile_person.metadata.get("unreadable_person_interaction_rows") != 1
        ):
            raise SystemExit(f"get_person missed readable/unreadable interaction counters: {hostile_person.metadata}")
        read_handoff = hostile_person.metadata.get("people_read_handoff") or {}
        if read_handoff.get("person_id") != 45 or read_handoff.get("interaction_count") != 1:
            raise SystemExit(f"get_person hostile-row handoff should preserve readable data only: {read_handoff}")
        if (
            read_handoff.get("interaction_ids") != [46]
            or read_handoff.get("has_relation") is not True
            or read_handoff.get("has_notes") is not True
            or read_handoff.get("has_last_contact") is not True
            or read_handoff.get("next_commands", [None])[0]
            != "log interaction with 45: <summary>"
        ):
            raise SystemExit(
                f"get_person hostile-row handoff should expose only IDs and presence flags: {read_handoff}"
            )
        metadata_text = json.dumps(hostile_person.metadata, sort_keys=True, ensure_ascii=False)
        if "Readable Detail" in metadata_text or "<local-path>" in metadata_text:
            raise SystemExit(f"get_person handoff retained private record content: {hostile_person.metadata}")
        assert_people_read_handoff(hostile_person.metadata, "hostile get person")

        empty_runtime = make_temp_runtime(Path(temp) / "empty")
        empty_list = empty_runtime.registry.get("list_people").handler({})
        empty_handoff = empty_list.metadata.get("people_list_handoff")
        if not empty_list.ok or not isinstance(empty_handoff, dict):
            raise SystemExit(f"empty list_people missed handoff metadata: {empty_list.metadata}")
        if empty_handoff.get("count") != 0 or empty_handoff.get("person_ids") != [] or empty_handoff.get("first_show_command") is not None:
            raise SystemExit(f"empty list_people handoff should preserve empty state: {empty_handoff}")
        if "Count: 0" not in empty_list.output or empty_list.metadata.get("total_people") != 0:
            raise SystemExit(f"empty list_people should expose zero-count output/metadata: {empty_list.output!r} / {empty_list.metadata}")
        if empty_handoff.get("boundaries", {}).get("read_only") is not True or empty_handoff.get("boundaries", {}).get("writes_files"):
            raise SystemExit(f"empty list_people handoff should stay read-only: {empty_handoff}")
        assert_people_list_handoff(empty_list.metadata, "empty list people")

        missing_name = runtime.registry.get("add_person").handler({"name": "", "relation": "friend"})
        if missing_name.ok or missing_name.metadata.get("reason") != "missing_name":
            raise SystemExit("add_person missing name should include safe refusal metadata.")
        assert_people_refusal_handoff(
            missing_name, source="add_person", mutation="person_create", reason="missing_name"
        )
        for path_name in (
            "/\x55sers/example/private/person-name",
            "/var/folders/zc/jarvis/person-name",
            "/tmp/jarvis/person-name",
        ):
            path_bad_name = runtime.registry.get("add_person").handler({"name": path_name, "relation": "friend"})
            if path_bad_name.ok or path_bad_name.metadata.get("reason") != "invalid_name":
                raise SystemExit("add_person path-shaped name should include safe refusal metadata.")
            if path_bad_name.metadata.get("writes_files") or path_bad_name.metadata.get("writes_memory") or path_bad_name.metadata.get("writes_notes"):
                raise SystemExit(f"add_person path-shaped name should not write: {path_bad_name.metadata}")
            if any(fragment in path_bad_name.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"add_person leaked local path in refusal output: {path_bad_name.output}")
            assert_people_refusal_handoff(
                path_bad_name,
                source="add_person",
                mutation="person_create",
                reason="invalid_name",
                forbidden_values=(path_name,),
            )

        missing_person = runtime.registry.get("get_person").handler({"name": "Nobody"})
        if missing_person.ok or missing_person.metadata.get("reason") != "not_found":
            raise SystemExit("get_person missing path should include stable refusal metadata.")
        assert_people_refusal_handoff(
            missing_person,
            source="get_person",
            mutation="person_read",
            reason="not_found",
            forbidden_values=("Nobody",),
        )

        bad_person_id = runtime.registry.get("get_person").handler({"person_id": "bad"})
        if bad_person_id.ok or bad_person_id.metadata.get("reason") != "bad_person_id":
            raise SystemExit("get_person bad id should include safe refusal metadata.")
        if bad_person_id.metadata.get("writes_files") or bad_person_id.metadata.get("queues_approval"):
            raise SystemExit("get_person bad id should stay read-only and approval-free.")
        assert_people_refusal_handoff(
            bad_person_id,
            source="get_person",
            mutation="person_read",
            reason="bad_person_id",
        )
        bool_person_id = runtime.registry.get("get_person").handler({"person_id": True})
        if bool_person_id.ok or bool_person_id.metadata.get("reason") != "bad_person_id":
            raise SystemExit("get_person boolean id should include safe refusal metadata.")
        if bool_person_id.metadata.get("writes_files") or bool_person_id.metadata.get("writes_memory") or bool_person_id.metadata.get("writes_notes"):
            raise SystemExit(f"get_person boolean id should not write: {bool_person_id.metadata}")
        assert_people_refusal_handoff(
            bool_person_id,
            source="get_person",
            mutation="person_read",
            reason="bad_person_id",
            forbidden_values=("True",),
        )

        fractional_person_id = runtime.registry.get("get_person").handler({"person_id": 1.9})
        if fractional_person_id.ok or fractional_person_id.metadata.get("reason") != "bad_person_id":
            raise SystemExit("get_person fractional id should be refused instead of truncated")
        fractional_interaction = runtime.registry.get("log_interaction").handler(
            {"person_id": 1.9, "summary": "must not reach person one"}
        )
        if fractional_interaction.ok or fractional_interaction.metadata.get("reason") != "bad_person_id":
            raise SystemExit("log_interaction fractional id should be refused instead of truncated")

        long_bad_person_id = runtime.registry.get("get_person").handler({"person_id": "p" * 200})
        assert_people_refusal_handoff(
            long_bad_person_id,
            source="get_person",
            mutation="person_read",
            reason="bad_person_id",
            forbidden_values=("p" * 80,),
        )
        for path_id in (
            "/\x55sers/example/private/person-id",
            "/var/folders/zc/jarvis/person-id",
            "/tmp/jarvis/person-id",
        ):
            path_bad_person_id = runtime.registry.get("get_person").handler({"person_id": path_id})
            assert_people_refusal_handoff(
                path_bad_person_id,
                source="get_person",
                mutation="person_read",
                reason="bad_person_id",
                forbidden_values=(path_id,),
            )

        for bad_numeric_id in (0, -1):
            bad_numeric_person = runtime.registry.get("get_person").handler({"person_id": bad_numeric_id})
            if bad_numeric_person.ok or "positive number" not in bad_numeric_person.output:
                raise SystemExit("get_person should reject non-positive person ids before lookup.")
            if bad_numeric_person.metadata.get("reason") != "bad_person_id" or bad_numeric_person.metadata.get("person_id") is not None:
                raise SystemExit(f"get_person non-positive id should include bad-id metadata: {bad_numeric_person.metadata}")
            if bad_numeric_person.metadata.get("writes_files") or bad_numeric_person.metadata.get("writes_memory") or bad_numeric_person.metadata.get("writes_notes"):
                raise SystemExit(f"get_person non-positive id should not write: {bad_numeric_person.metadata}")
            assert_people_refusal_handoff(
                bad_numeric_person,
                source="get_person",
                mutation="person_read",
                reason="bad_person_id",
            )

        missing_summary = runtime.registry.get("log_interaction").handler({"name": "Maya", "summary": ""})
        if missing_summary.ok or missing_summary.metadata.get("reason") != "missing_summary":
            raise SystemExit("log_interaction missing summary should include safe refusal metadata.")
        assert_people_refusal_handoff(
            missing_summary,
            source="log_interaction",
            mutation="interaction_create",
            reason="missing_summary",
            forbidden_values=("Maya",),
        )
        for path_name in (
            "/private/tmp/jarvis-person-name",
            "/var/folders/zc/jarvis/person-name",
            "/tmp/jarvis/person-name",
        ):
            path_bad_interaction_name = runtime.registry.get("log_interaction").handler({"name": path_name, "summary": "hello"})
            if path_bad_interaction_name.ok or path_bad_interaction_name.metadata.get("reason") != "invalid_name":
                raise SystemExit("log_interaction path-shaped name should include safe refusal metadata.")
            if path_bad_interaction_name.metadata.get("writes_files") or path_bad_interaction_name.metadata.get("writes_memory") or path_bad_interaction_name.metadata.get("writes_notes"):
                raise SystemExit(f"log_interaction path-shaped name should not write: {path_bad_interaction_name.metadata}")
            if any(fragment in path_bad_interaction_name.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"log_interaction leaked local path in refusal output: {path_bad_interaction_name.output}")
            assert_people_refusal_handoff(
                path_bad_interaction_name,
                source="log_interaction",
                mutation="interaction_create",
                reason="invalid_name",
                forbidden_values=(path_name,),
            )

        bad_interaction_id = runtime.registry.get("log_interaction").handler({"person_id": "bad", "summary": "hello"})
        if bad_interaction_id.ok or bad_interaction_id.metadata.get("reason") != "bad_person_id":
            raise SystemExit("log_interaction bad person id should include safe refusal metadata.")
        if bad_interaction_id.metadata.get("writes_files") or bad_interaction_id.metadata.get("queues_approval"):
            raise SystemExit("log_interaction bad person id should stay read-only and approval-free.")
        assert_people_refusal_handoff(
            bad_interaction_id,
            source="log_interaction",
            mutation="interaction_create",
            reason="bad_person_id",
        )
        for path_id in (
            "/private/tmp/jarvis-interaction-id",
            "/var/folders/zc/jarvis/interaction-id",
            "/tmp/jarvis/interaction-id",
        ):
            path_bad_interaction_id = runtime.registry.get("log_interaction").handler({"person_id": path_id, "summary": "hello"})
            assert_people_refusal_handoff(
                path_bad_interaction_id,
                source="log_interaction",
                mutation="interaction_create",
                reason="bad_person_id",
                forbidden_values=(path_id,),
            )
        bool_interaction_id = runtime.registry.get("log_interaction").handler({"person_id": True, "summary": "hello"})
        if bool_interaction_id.ok or bool_interaction_id.metadata.get("reason") != "bad_person_id":
            raise SystemExit("log_interaction boolean id should include safe refusal metadata.")
        if bool_interaction_id.metadata.get("writes_files") or bool_interaction_id.metadata.get("writes_memory") or bool_interaction_id.metadata.get("writes_notes"):
            raise SystemExit(f"log_interaction boolean id should not write: {bool_interaction_id.metadata}")
        assert_people_refusal_handoff(
            bool_interaction_id,
            source="log_interaction",
            mutation="interaction_create",
            reason="bad_person_id",
            forbidden_values=("True",),
        )

        for bad_numeric_id in (0, -1):
            bad_numeric_interaction_id = runtime.registry.get("log_interaction").handler({"person_id": bad_numeric_id, "summary": "hello"})
            if bad_numeric_interaction_id.ok or "positive number" not in bad_numeric_interaction_id.output:
                raise SystemExit("log_interaction should reject non-positive person ids before mutation.")
            if bad_numeric_interaction_id.metadata.get("reason") != "bad_person_id" or bad_numeric_interaction_id.metadata.get("person_id") is not None:
                raise SystemExit(f"log_interaction non-positive id should include bad-id metadata: {bad_numeric_interaction_id.metadata}")
            if bad_numeric_interaction_id.metadata.get("writes_files") or bad_numeric_interaction_id.metadata.get("writes_memory") or bad_numeric_interaction_id.metadata.get("writes_notes"):
                raise SystemExit(f"log_interaction non-positive id should not write: {bad_numeric_interaction_id.metadata}")
            assert_people_refusal_handoff(
                bad_numeric_interaction_id,
                source="log_interaction",
                mutation="interaction_create",
                reason="bad_person_id",
            )

        bounded = runtime.registry.get("add_person").handler({"name": "n" * 500, "relation": "r" * 500, "notes": "x" * 5000})
        if not bounded.ok:
            raise SystemExit("add_person should accept bounded long person fields.")
        if bounded.metadata.get("name_chars") != 120 or bounded.metadata.get("notes_chars") != 4000:
            raise SystemExit("add_person did not bound long name/notes fields.")
        assert_people_mutation_handoff(bounded.metadata, "bounded add_person", source="add_person", mutation="person_create", changed=["person"])
        assert_write_receipt(bounded, root=root, label="bounded add_person")


if __name__ == "__main__":
    main()
