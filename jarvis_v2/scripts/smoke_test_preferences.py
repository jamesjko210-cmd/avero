from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.preferences import _metadata_bool, _preference_handoff_metadata, _preference_refusal_metadata


class HostileRow:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def keys(self):
        raise RuntimeError(self.marker)

    def __getitem__(self, key: str):
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


def test_preference_read_alias_routes() -> None:
    for command in (
        "preferences please",
        "preference list please",
        "show latest preferences",
        "show preferences please",
        "what preferences do you know",
        # Real gap found live 2026-07-09: "show my preferences" fell through to
        # chat even though "my preferences" and "show preferences" both worked.
        "show my preferences",
        "show my preferences please",
        # Real gap found live 2026-07-10, same class, surfaced via a WS4
        # mixed-conversation remeasurement: "list preferences" and "show my
        # preferences" both worked, but "list my preferences" (the natural
        # combination) and any "recent" qualifier fell through to chat.
        "list my preferences",
        "list my preferences please",
        "recent preferences",
        "show my recent preferences",
        "list my recent preferences",
        "my recent preferences",
        # Real gap found live 2026-07-10 (round 43): "show my preferences
        # please" already worked because it was a pre-existing literal set
        # entry, but the newer "recent" qualifier combination above did NOT --
        # trailing "please" on any phrase not itself a literal "X please"
        # entry fell through, since list_preferences isn't in
        # POLITE_COMMAND_RETRY_TOOLS. Fixed with a local trailing-only strip
        # (see smoke_test_notes.py's matching comment for the full root cause).
        "show my recent preferences please",
    ):
        assert_route(command, "list_preferences")
    for command in (
        "all preferences please",
        "show all preferences please",
    ):
        assert_route(command, "list_preferences", {"status": "all"})


def test_preference_search_query_does_not_leak_to_web() -> None:
    # Real gap found live 2026-07-10, same privacy-relevant misroute class as
    # the round-21/26/27/28 notes/tasks/files/memory/goal word-order fixes:
    # "search for the/a preference about X" leaked to a public web_lookup
    # search. No dedicated keyword-search-by-content tool exists for
    # preferences (only list_preferences/set_preference), so the fix is a
    # targeted exclusion rather than a new route -- this should fall through
    # to chat instead of leaking to the web.
    for command in ("search for the preference about theme", "search for a preference about theme"):
        plan = RuleBasedPlanner().plan(command)
        if [a.tool_name for a in plan.actions] == ["web_lookup"]:
            raise SystemExit(f"preference query should not leak to a public web search: {command!r} -> {plan.actions}")


def assert_runtime_routes_preference_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-preference-count-alias-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for text in (
            "preference count",
            "preferences count",
            "count preferences",
            "how many preferences do i have",
            "선호 몇 개",
            "선호사항 개수",
            "설정 몇 개",
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [action.get("tool_name") for action in planned] != ["list_preferences"]:
                raise SystemExit(f"runtime missed preference-count alias {text!r}: {planned} / {result.response!r}")
            if len(result.tool_results) != 1 or result.tool_results[0].tool_name != "list_preferences":
                raise SystemExit(f"preference-count alias should execute one list_preferences tool for {text!r}: {result.tool_results}")
            metadata = result.tool_results[0].metadata
            if "Count: 0" not in result.response or metadata.get("total_preferences") != 0:
                raise SystemExit(f"empty preference-count alias should answer with zero count for {text!r}: {result.response!r} / {metadata}")
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
                    raise SystemExit(f"preference-count alias unexpectedly set {key} for {text!r}: {metadata}")
        if runtime.store.list_preferences(status=None, limit=100):
            raise SystemExit("preference count aliases must not create or mutate preferences.")
        if runtime.store.list_memories(limit=100):
            raise SystemExit("preference count aliases must not create memory rows.")


def assert_write_receipt(result, *, root: Path, label: str) -> None:
    metadata = result.metadata
    path_display = metadata.get("path_display")
    if "path" in metadata:
        raise SystemExit(f"{label} exposed forbidden absolute path metadata: {metadata}")
    if path_display != "Memory Tree/Preferences.md":
        raise SystemExit(f"{label} missed the opaque preference path_display: {metadata}")
    if not (root / "Memory Tree" / "Preferences.md").exists():
        raise SystemExit(f"{label} did not persist the preference projection.")
    if f"Saved note: {path_display}" not in result.output:
        raise SystemExit(f"{label} missed safe saved-note receipt: {result.output}")
    assert_no_local_paths(metadata, label)
    if str(root) in result.output or "/var/folders/" in result.output or "/private/" in result.output or "/\x55sers/" in result.output:
        raise SystemExit(f"{label} leaked a local path in output: {result.output}")


def assert_no_local_paths(value, label: str) -> None:
    text = repr(value)
    if "/private/" in text or "/\x55sers/" in text or "/var/folders/" in text or "/tmp/" in text:
        raise SystemExit(f"{label} handoff should not expose raw local paths: {text[:1000]}")


def assert_preference_contract(
    metadata: dict,
    handoff: dict,
    label: str,
    *,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
) -> None:
    for key, value in [
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ]:
        if metadata.get(key) != value or handoff.get(key) != value:
            raise SystemExit(f"{label} preference handoff {key} parity failed: {handoff} / {metadata}")


def assert_preference_safe_commands(metadata: dict, handoff: dict, handoff_key: str, label: str) -> None:
    prefix = handoff_key.removesuffix("_handoff")
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        expected = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        expected = [str(value) for value in raw_next if str(value or "").strip()]
    first = expected[0] if expected else ""
    checks = {
        "handoff_ready": True,
        "next_safe_command": first,
        "next_safe_commands": expected,
        "next_safe_command_count": len(expected),
    }
    for key, value in checks.items():
        if handoff.get(key) != value:
            raise SystemExit(f"{label} handoff {key} mismatch: {handoff}")
    if metadata.get(f"{handoff_key}_ready") is not True or metadata.get(f"{prefix}_handoff_ready") is not True:
        raise SystemExit(f"{label} missed handoff ready aliases: {metadata}")
    for key, value in [
        ("ready_for_operator", True),
        ("state_changed", handoff.get("state_changed")),
        ("changed", handoff.get("changed")),
        ("content_in_handoff", handoff.get("content_in_handoff")),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
        ("next_safe_command", first),
        ("next_safe_commands", expected),
        ("next_safe_command_count", len(expected)),
    ]:
        if metadata.get(key) != value:
            raise SystemExit(f"{label} metadata {key} mismatch: {metadata}")
        prefixed_key = f"{prefix}_{key}"
        if metadata.get(prefixed_key) != value:
            raise SystemExit(f"{label} metadata {prefixed_key} mismatch: {metadata}")


def assert_preference_mutation_handoff(metadata: dict, label: str, *, source: str, mutation: str, changed: list[str]) -> None:
    handoff = metadata.get("preference_mutation_handoff")
    if metadata.get("preference_mutation_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready preference_mutation_handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("mutation") != mutation:
        raise SystemExit(f"{label} mutation source/type mismatch: {handoff}")
    if handoff.get("preference_id") != metadata.get("preference_id"):
        raise SystemExit(f"{label} mutation preference id parity failed: {handoff} / {metadata}")
    assert_preference_contract(metadata, handoff, label, state_changed=True, changed=changed, content_in_handoff=False)
    assert_preference_safe_commands(metadata, handoff, "preference_mutation_handoff", label)
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") is not False or boundaries.get("writes_files") is not True or boundaries.get("writes_notes") is not True:
        raise SystemExit(f"{label} mutation handoff missed durable write boundaries: {handoff}")
    for key in (
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if boundaries.get(key):
            raise SystemExit(f"{label} mutation handoff should keep {key}=False: {handoff}")
    assert_no_local_paths(handoff, label)


def assert_preference_refusal_handoff(
    metadata: dict,
    label: str,
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
) -> None:
    handoff = metadata.get("preference_refusal_handoff")
    if metadata.get("preference_refusal_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready preference_refusal_handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} has wrong preference refusal source/readiness: {handoff}")
    if handoff.get("mutation") != mutation or handoff.get("reason") != reason or handoff.get("refused") is not True:
        raise SystemExit(f"{label} has wrong preference refusal mutation/reason: {handoff}")
    if handoff.get("changed") != []:
        raise SystemExit(f"{label} should not report changed preference rows: {handoff}")
    assert_preference_contract(
        metadata,
        handoff,
        label,
        state_changed=False,
        changed=[],
        content_in_handoff=any(key in handoff for key in ("raw_key", "raw_value", "raw_status", "raw_preference_id")),
    )
    assert_preference_safe_commands(metadata, handoff, "preference_refusal_handoff", label)
    if not isinstance(handoff.get("next_commands"), dict) or "preferences" not in handoff["next_commands"].values():
        raise SystemExit(f"{label} should include preference recovery commands: {handoff}")
    for key in ("preference_id", "category", "status", "raw_key", "raw_value", "raw_status", "raw_preference_id"):
        if key in handoff and key in metadata and handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "read_only": read_only,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_database": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "completes_tasks": False,
        "creates_preference": False,
        "updates_preference": False,
        "reads_private_data": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} refusal boundary {key} should be {value}: {boundaries}")
    assert_no_local_paths(handoff, label)


def assert_preference_exact_metadata_bool() -> None:
    if _metadata_bool(True) is not True:
        raise SystemExit("preference exact bool helper should preserve True.")
    if _metadata_bool(False, default=True) is not False:
        raise SystemExit("preference exact bool helper should preserve False.")
    for value in ("true", "false", "yes", "0", 1, 0, [], ["content"], None):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"preference exact bool helper should reject malformed handoff flags: {value!r}")
    if _metadata_bool("fallback", default=True) is not True:
        raise SystemExit("preference exact bool helper should honor explicit malformed-value default.")


def assert_preference_malformed_handoff_flags() -> None:
    handoff = {
        "source": "list_preferences",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["preferences"],
        "boundaries": {"read_only": True},
    }
    metadata = _preference_handoff_metadata("preference_list_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("preference_list_state_changed") is not False:
        raise SystemExit(f"malformed preference state_changed should not become truthy: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("preference_list_content_in_handoff") is not False:
        raise SystemExit(f"malformed preference content_in_handoff should not become truthy: {metadata}")

    refusal_metadata = _preference_refusal_metadata(
        source="list_preferences",
        mutation="preference_list",
        reason="bad_status",
        read_only=True,
        raw_status="bad",
    )
    refusal_metadata["preference_refusal_handoff"]["content_in_handoff"] = "true"
    repaired = _preference_handoff_metadata(
        "preference_refusal_handoff",
        refusal_metadata["preference_refusal_handoff"],
    )
    if repaired.get("content_in_handoff") is not False or repaired.get("preference_refusal_content_in_handoff") is not False:
        raise SystemExit(f"malformed preference refusal content flag should not become truthy: {repaired}")


def main() -> None:
    test_preference_read_alias_routes()
    test_preference_search_query_does_not_leak_to_web()
    assert_runtime_routes_preference_count_aliases()
    assert_preference_exact_metadata_bool()
    assert_preference_malformed_handoff_flags()
    with TemporaryDirectory(prefix="jarvis-preferences-") as temp:
        runtime = make_temp_runtime(Path(temp))
        root = runtime.vault.root_path
        cases = [
            "remember preference voice tone as conversational category communication",
            "set preference response style to direct and warm category communication",
            "set preference planning depth to concise first category work",
            "preferences",
            "my preferences",
            "what preferences do you remember",
            "show preference voice tone",
            "show preference response style",
            "preference 1 retired",
            "all preferences",
            "search memory for direct warm",
            "export state",
            "save preference tone direct",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1600])
            print()
            if case == "remember preference voice tone as conversational category communication":
                metadata = result.tool_results[0].metadata
                if metadata.get("preference_id") != 1 or metadata.get("writes_files") is not True:
                    raise SystemExit("set_preference missed durable write metadata.")
                if metadata.get("writes_memory") is not True or metadata.get("writes_notes") is not True:
                    raise SystemExit("set_preference missed memory/note write metadata.")
                if metadata.get("requires_approval") or metadata.get("external_side_effect"):
                    raise SystemExit("set_preference should stay local-safe without external side effects.")
                assert_preference_mutation_handoff(
                    metadata,
                    "set_preference",
                    source="set_preference",
                    mutation="preference_create",
                    changed=["preference"],
                )
                assert_write_receipt(result.tool_results[0], root=root, label="set_preference")
            if case in {"preferences", "my preferences", "what preferences do you remember"}:
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 100 or metadata.get("writes_files") is not False:
                    raise SystemExit("list_preferences missed sanitized limit/read-only metadata.")
                if "Count:" not in result.response or metadata.get("total_preferences") != metadata.get("count"):
                    raise SystemExit(f"list_preferences missed visible/metadata count parity: {result.response!r} / {metadata}")
                handoff = metadata.get("preference_list_handoff")
                if not isinstance(handoff, dict) or handoff.get("source") != "list_preferences":
                    raise SystemExit(f"list_preferences missed structured handoff metadata: {metadata}")
                if metadata.get("preference_list_handoff_ready") is not True:
                    raise SystemExit(f"list_preferences missed handoff ready flag: {metadata}")
                if handoff.get("ready_for_operator") is not True or handoff.get("count") != metadata.get("count"):
                    raise SystemExit(f"list_preferences handoff missed readiness/count parity: {handoff}")
                assert_preference_contract(metadata, handoff, "list_preferences", state_changed=False, changed=[], content_in_handoff=True)
                assert_preference_safe_commands(metadata, handoff, "preference_list_handoff", "list_preferences")
                if handoff.get("status") != "active" or handoff.get("limit") != 100 or handoff.get("category") is not None:
                    raise SystemExit(f"list_preferences handoff missed normalized filter metadata: {handoff}")
                if handoff.get("preference_ids") != [2, 1, 3]:
                    raise SystemExit(f"list_preferences handoff missed preference id parity: {handoff}")
                rows = handoff.get("rows")
                if not isinstance(rows, list) or rows[0].get("show_command") != "show preference response style":
                    raise SystemExit(f"list_preferences handoff missed show command: {handoff}")
                if rows[0].get("retire_command") != "preference 2 retired" or rows[0].get("reactivate_command") != "preference 2 active":
                    raise SystemExit(f"list_preferences handoff missed status commands: {handoff}")
                if handoff.get("first_show_command") != "show preference response style" or "all preferences" not in handoff.get("next_commands", []):
                    raise SystemExit(f"list_preferences handoff missed next commands: {handoff}")
                boundaries = handoff.get("boundaries")
                if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
                    raise SystemExit(f"list_preferences handoff missed read-only boundary: {handoff}")
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
                        raise SystemExit(f"list_preferences handoff should keep {key}=False: {handoff}")
            if case == "show preference voice tone":
                metadata = result.tool_results[0].metadata
                if metadata.get("preference_id") != 1 or metadata.get("reads_private_data") is not False:
                    raise SystemExit("get_preference missed read-only metadata.")
                handoff = metadata.get("preference_read_handoff")
                if metadata.get("preference_read_handoff_ready") is not True or not isinstance(handoff, dict):
                    raise SystemExit(f"get_preference missed read handoff: {metadata}")
                assert_preference_contract(metadata, handoff, "get_preference", state_changed=False, changed=[], content_in_handoff=True)
                assert_preference_safe_commands(metadata, handoff, "preference_read_handoff", "get_preference")
                boundaries = handoff.get("boundaries") or {}
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
                        raise SystemExit(f"get_preference read handoff should keep {key}=False: {handoff}")
            if case == "preference 1 retired":
                metadata = result.tool_results[0].metadata
                if metadata.get("preference_id") != 1 or metadata.get("status") != "retired":
                    raise SystemExit("set_preference_status missed preference metadata.")
                assert_preference_mutation_handoff(
                    metadata,
                    "set_preference_status",
                    source="set_preference_status",
                    mutation="preference_status_update",
                    changed=["preference_status"],
                )
                assert_write_receipt(result.tool_results[0], root=root, label="set_preference_status")
            if case == "save preference tone direct":
                metadata = result.tool_results[0].metadata
                if "tone = direct" not in result.response or metadata.get("key_chars") != 4 or metadata.get("value_chars") != 6:
                    raise SystemExit(f"simple preference alias missed saved key/value evidence: {result.response} / {metadata}")
                assert_preference_mutation_handoff(
                    metadata,
                    "simple set_preference alias",
                    source="set_preference",
                    mutation="preference_create",
                    changed=["preference"],
                )
                assert_write_receipt(result.tool_results[0], root=root, label="simple set_preference alias")

        direct_list = runtime.registry.get("list_preferences").handler({"limit": "bad"})
        if not direct_list.ok or direct_list.metadata.get("limit") != 100 or direct_list.metadata.get("raw_limit") != "bad":
            raise SystemExit("list_preferences did not sanitize a bad limit.")
        if direct_list.metadata.get("writes_files") is not False or direct_list.metadata.get("reads_private_data") is not False:
            raise SystemExit("list_preferences missed safety metadata.")
        if direct_list.metadata.get("preference_list_handoff", {}).get("limit") != 100:
            raise SystemExit(f"list_preferences handoff should use sanitized limit: {direct_list.metadata}")
        assert_preference_contract(
            direct_list.metadata,
            direct_list.metadata.get("preference_list_handoff", {}),
            "direct list_preferences",
            state_changed=False,
            changed=[],
            content_in_handoff=True,
        )
        assert_preference_safe_commands(
            direct_list.metadata,
            direct_list.metadata.get("preference_list_handoff", {}),
            "preference_list_handoff",
            "direct list_preferences",
        )
        direct_list_bool = runtime.registry.get("list_preferences").handler({"limit": False})
        if not direct_list_bool.ok or direct_list_bool.metadata.get("limit") != 100 or direct_list_bool.metadata.get("raw_limit") != "False":
            raise SystemExit(f"list_preferences should treat boolean limits as malformed defaults: {direct_list_bool.metadata}")
        if direct_list_bool.metadata.get("writes_files") or direct_list_bool.metadata.get("writes_memory"):
            raise SystemExit(f"list_preferences boolean limit should stay read-only: {direct_list_bool.metadata}")
        direct_list_long_limit = runtime.registry.get("list_preferences").handler({"limit": "l" * 200})
        if direct_list_long_limit.metadata.get("raw_limit") != ("l" * 77 + "..."):
            raise SystemExit(f"list_preferences did not bound raw bad limit metadata: {direct_list_long_limit.metadata}")
        for path_limit in (
            "/\x55sers/example/private/preference-limit",
            "/var/folders/zc/jarvis/preference-limit",
            "/tmp/jarvis/preference-limit",
        ):
            path_bad_list = runtime.registry.get("list_preferences").handler({"limit": path_limit})
            if path_bad_list.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"list_preferences leaked local path in raw limit metadata: {path_bad_list.metadata}")

        direct_list_large = runtime.registry.get("list_preferences").handler({"limit": 999999, "status": "all"})
        if not direct_list_large.ok or direct_list_large.metadata.get("limit") != 200:
            raise SystemExit("list_preferences did not clamp a large limit.")
        large_handoff = direct_list_large.metadata.get("preference_list_handoff", {})
        if large_handoff.get("status") != "all" or large_handoff.get("limit") != 200:
            raise SystemExit(f"list_preferences handoff missed all-status/large-limit mode: {direct_list_large.metadata}")

        direct_category = runtime.registry.get("list_preferences").handler({"category": "communication", "status": "active", "limit": 10})
        category_handoff = direct_category.metadata.get("preference_list_handoff", {})
        if not direct_category.ok or category_handoff.get("category") != "communication":
            raise SystemExit(f"list_preferences handoff missed category filter: {direct_category.metadata}")
        if category_handoff.get("preference_ids") != [2]:
            raise SystemExit(f"list_preferences category handoff missed filtered ids: {category_handoff}")

        bad_list_status = runtime.registry.get("list_preferences").handler({"status": "archived"})
        if bad_list_status.ok or bad_list_status.metadata.get("reason") != "bad_status":
            raise SystemExit("list_preferences bad status did not include refusal metadata.")
        if bad_list_status.metadata.get("raw_status") != "archived":
            raise SystemExit(f"list_preferences bad status should preserve bounded raw status: {bad_list_status.metadata}")
        if bad_list_status.metadata.get("writes_files") or bad_list_status.metadata.get("queues_approval"):
            raise SystemExit("list_preferences bad status should stay read-only and approval-free.")
        assert_preference_refusal_handoff(
            bad_list_status.metadata,
            "list_preferences bad status",
            source="list_preferences",
            mutation="preference_list",
            reason="bad_status",
            read_only=True,
        )
        for path_status in (
            "/private/tmp/jarvis-preference-status",
            "/var/folders/zc/jarvis/preference-status",
            "/tmp/jarvis/preference-status",
        ):
            path_bad_list_status = runtime.registry.get("list_preferences").handler({"status": path_status})
            if path_bad_list_status.metadata.get("raw_status") != "<local-path>":
                raise SystemExit(f"list_preferences leaked local path in raw status metadata: {path_bad_list_status.metadata}")
            assert_preference_refusal_handoff(
                path_bad_list_status.metadata,
                "list_preferences path bad status",
                source="list_preferences",
                mutation="preference_list",
                reason="bad_status",
                read_only=True,
            )

        missing = runtime.registry.get("get_preference").handler({"key": "missing"})
        if not missing.ok or missing.metadata.get("preference_id") is not None:
            raise SystemExit("get_preference missing path did not return stable metadata.")
        missing_handoff = missing.metadata.get("preference_read_handoff")
        if missing.metadata.get("preference_read_handoff_ready") is not True or not isinstance(missing_handoff, dict):
            raise SystemExit(f"get_preference missing path missed read handoff: {missing.metadata}")
        assert_preference_contract(missing.metadata, missing_handoff, "missing get_preference", state_changed=False, changed=[], content_in_handoff=False)
        assert_preference_safe_commands(missing.metadata, missing_handoff, "preference_read_handoff", "missing get_preference")
        if missing.metadata.get("recovery_commands") != [
            "preferences",
            "set preference <key> to <value> category <category>",
        ]:
            raise SystemExit(f"get_preference missing recovery order drifted: {missing.metadata}")
        for token in ("Run `preferences`", "review saved keys", "set preference <key> to <value> category <category>", "normal local-safe policy"):
            if token not in missing.output:
                raise SystemExit(f"get_preference missing output missed {token!r}: {missing.output}")
        if missing.metadata.get("retry_requires_preference_refresh") is not True:
            raise SystemExit(f"get_preference missing recovery missed refresh requirement: {missing.metadata}")
        if missing.metadata.get("authorizes_retry") or missing.metadata.get("authorizes_preference_mutation"):
            raise SystemExit(f"get_preference missing recovery granted authority: {missing.metadata}")

        empty_runtime = make_temp_runtime(Path(temp) / "empty")
        empty_list = empty_runtime.registry.get("list_preferences").handler({"status": "active"})
        empty_handoff = empty_list.metadata.get("preference_list_handoff")
        if not empty_list.ok or not isinstance(empty_handoff, dict):
            raise SystemExit(f"empty list_preferences missed handoff metadata: {empty_list.metadata}")
        if empty_handoff.get("count") != 0 or empty_handoff.get("preference_ids") != [] or empty_handoff.get("first_show_command") is not None:
            raise SystemExit(f"empty list_preferences handoff should preserve empty state: {empty_handoff}")
        if "Count: 0" not in empty_list.output or empty_list.metadata.get("total_preferences") != 0:
            raise SystemExit(f"empty list_preferences should expose zero-count output/metadata: {empty_list.output!r} / {empty_list.metadata}")
        assert_preference_contract(empty_list.metadata, empty_handoff, "empty list_preferences", state_changed=False, changed=[], content_in_handoff=False)
        assert_preference_safe_commands(empty_list.metadata, empty_handoff, "preference_list_handoff", "empty list_preferences")
        if empty_handoff.get("boundaries", {}).get("read_only") is not True or empty_handoff.get("boundaries", {}).get("writes_files"):
            raise SystemExit(f"empty list_preferences handoff should stay read-only: {empty_handoff}")

        malformed_runtime = make_temp_runtime(Path(temp) / "malformed")
        leak_markers = ["PREFERENCE_LIST_ROW_SECRET", "PREFERENCE_READ_ROW_SECRET"]
        preference_rows = [
            HostileRow(leak_markers[0]),
            {
                "id": 44,
                "category": "communication",
                "key": "reply style",
                "value": "warm and concise",
                "status": "active",
            },
        ]
        malformed_runtime.store.list_preferences = lambda category=None, status="active", limit=100: preference_rows[:limit]
        malformed_list = malformed_runtime.registry.get("list_preferences").handler({"limit": 10})
        if not malformed_list.ok:
            raise SystemExit(f"list_preferences should tolerate malformed local rows: {malformed_list.output}")
        malformed_handoff = malformed_list.metadata.get("preference_list_handoff")
        if not isinstance(malformed_handoff, dict):
            raise SystemExit(f"malformed list_preferences missed handoff: {malformed_list.metadata}")
        combined_list = malformed_list.output + repr(malformed_list.metadata)
        if any(marker in combined_list for marker in leak_markers):
            raise SystemExit(f"list_preferences leaked hostile row marker: {combined_list}")
        if malformed_list.metadata.get("count") != 2 or malformed_list.metadata.get("readable_preference_rows") != 1:
            raise SystemExit(f"list_preferences malformed-row counts diverged: {malformed_list.metadata}")
        if malformed_list.metadata.get("unreadable_preference_rows") != 1:
            raise SystemExit(f"list_preferences missed unreadable row count: {malformed_list.metadata}")
        if malformed_handoff.get("preference_ids") != [44] or malformed_handoff.get("first_show_command") != "show preference reply style":
            raise SystemExit(f"list_preferences should preserve readable preference rows behind malformed rows: {malformed_handoff}")
        if "could not be read safely" not in malformed_list.output:
            raise SystemExit(f"list_preferences should mention unreadable rows safely: {malformed_list.output}")
        assert_preference_contract(malformed_list.metadata, malformed_handoff, "malformed list_preferences", state_changed=False, changed=[], content_in_handoff=True)
        assert_preference_safe_commands(malformed_list.metadata, malformed_handoff, "preference_list_handoff", "malformed list_preferences")

        malformed_runtime.store.get_preference = lambda key, category=None: HostileRow(leak_markers[1])
        malformed_get = malformed_runtime.registry.get("get_preference").handler({"key": "reply style"})
        if not malformed_get.ok:
            raise SystemExit(f"get_preference should tolerate malformed local rows: {malformed_get.output}")
        malformed_get_handoff = malformed_get.metadata.get("preference_read_handoff")
        if not isinstance(malformed_get_handoff, dict):
            raise SystemExit(f"malformed get_preference missed handoff: {malformed_get.metadata}")
        combined_get = malformed_get.output + repr(malformed_get.metadata)
        if any(marker in combined_get for marker in leak_markers):
            raise SystemExit(f"get_preference leaked hostile row marker: {combined_get}")
        if malformed_get_handoff.get("status") != "unreadable" or malformed_get.metadata.get("preference_id") is not None:
            raise SystemExit(f"get_preference should fail closed on unreadable row: {malformed_get.metadata}")
        if "could not be read safely" not in malformed_get.output:
            raise SystemExit(f"get_preference should explain unreadable row safely: {malformed_get.output}")
        assert_preference_contract(
            malformed_get.metadata,
            malformed_get_handoff,
            "malformed get_preference",
            state_changed=False,
            changed=[],
            content_in_handoff=False,
        )
        assert_preference_safe_commands(malformed_get.metadata, malformed_get_handoff, "preference_read_handoff", "malformed get_preference")

        missing_key = runtime.registry.get("get_preference").handler({"key": ""})
        if missing_key.ok or missing_key.metadata.get("reason") != "missing_key":
            raise SystemExit("get_preference missing key did not include safe refusal metadata.")
        if missing_key.metadata.get("raw_key") != "":
            raise SystemExit(f"get_preference missing key should preserve bounded raw key metadata: {missing_key.metadata}")
        assert_preference_refusal_handoff(
            missing_key.metadata,
            "get_preference missing key",
            source="get_preference",
            mutation="preference_read",
            reason="missing_key",
            read_only=True,
        )
        for path_key in (
            "/\x55sers/example/private/preference-key",
            "/var/folders/zc/jarvis/preference-key",
            "/tmp/jarvis/preference-key",
        ):
            path_bad_get_key = runtime.registry.get("get_preference").handler({"key": path_key})
            if path_bad_get_key.ok or path_bad_get_key.metadata.get("reason") != "invalid_key":
                raise SystemExit("get_preference path-shaped key should include safe refusal metadata.")
            if path_bad_get_key.metadata.get("raw_key") != "<local-path>":
                raise SystemExit(f"get_preference leaked local path in raw key metadata: {path_bad_get_key.metadata}")
            if path_bad_get_key.metadata.get("writes_files") or path_bad_get_key.metadata.get("writes_memory") or path_bad_get_key.metadata.get("writes_notes"):
                raise SystemExit(f"get_preference path-shaped key should stay read-only: {path_bad_get_key.metadata}")
            if any(fragment in path_bad_get_key.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"get_preference leaked local path in refusal output: {path_bad_get_key.output}")
            assert_preference_refusal_handoff(
                path_bad_get_key.metadata,
                "get_preference path-shaped key",
                source="get_preference",
                mutation="preference_read",
                reason="invalid_key",
                read_only=True,
            )

        bad_status = runtime.registry.get("set_preference_status").handler({"preference_id": 1, "status": "archived"})
        if bad_status.ok or bad_status.metadata.get("reason") != "bad_status":
            raise SystemExit("set_preference_status bad status did not include refusal metadata.")
        if bad_status.metadata.get("raw_status") != "archived":
            raise SystemExit(f"set_preference_status bad status should preserve bounded raw status: {bad_status.metadata}")
        assert_preference_refusal_handoff(
            bad_status.metadata,
            "set_preference_status bad status",
            source="set_preference_status",
            mutation="preference_status_update",
            reason="bad_status",
            read_only=False,
        )
        for path_status in (
            "/private/tmp/jarvis-set-preference-status",
            "/var/folders/zc/jarvis/set-preference-status",
            "/tmp/jarvis/set-preference-status",
        ):
            path_bad_status = runtime.registry.get("set_preference_status").handler({"preference_id": 1, "status": path_status})
            if path_bad_status.metadata.get("raw_status") != "<local-path>":
                raise SystemExit(f"set_preference_status leaked local path in raw status metadata: {path_bad_status.metadata}")
            assert_preference_refusal_handoff(
                path_bad_status.metadata,
                "set_preference_status path bad status",
                source="set_preference_status",
                mutation="preference_status_update",
                reason="bad_status",
                read_only=False,
            )

        bad_preference_id = runtime.registry.get("set_preference_status").handler({"preference_id": "pref-abc", "status": "retired"})
        if bad_preference_id.ok or bad_preference_id.metadata.get("reason") != "bad_preference_id":
            raise SystemExit("set_preference_status bad id did not include refusal metadata.")
        if bad_preference_id.metadata.get("raw_preference_id") != "pref-abc":
            raise SystemExit(f"set_preference_status should preserve bounded raw preference id: {bad_preference_id.metadata}")
        if bad_preference_id.metadata.get("writes_files") or bad_preference_id.metadata.get("writes_memory") or bad_preference_id.metadata.get("writes_notes"):
            raise SystemExit(f"set_preference_status bad id should not write: {bad_preference_id.metadata}")
        assert_preference_refusal_handoff(
            bad_preference_id.metadata,
            "set_preference_status bad id",
            source="set_preference_status",
            mutation="preference_status_update",
            reason="bad_preference_id",
            read_only=False,
        )
        bool_preference_id = runtime.registry.get("set_preference_status").handler({"preference_id": True, "status": "retired"})
        if bool_preference_id.ok or bool_preference_id.metadata.get("reason") != "bad_preference_id":
            raise SystemExit("set_preference_status boolean id did not include refusal metadata.")
        if bool_preference_id.metadata.get("raw_preference_id") != "True":
            raise SystemExit(f"set_preference_status should preserve boolean raw preference id: {bool_preference_id.metadata}")
        if bool_preference_id.metadata.get("writes_files") or bool_preference_id.metadata.get("writes_memory") or bool_preference_id.metadata.get("writes_notes"):
            raise SystemExit(f"set_preference_status boolean id should not write: {bool_preference_id.metadata}")
        for path_id in (
            "/\x55sers/example/private/preference-id",
            "/var/folders/zc/jarvis/preference-id",
            "/tmp/jarvis/preference-id",
        ):
            path_bad_preference_id = runtime.registry.get("set_preference_status").handler({"preference_id": path_id, "status": "retired"})
            if path_bad_preference_id.metadata.get("raw_preference_id") != "<local-path>":
                raise SystemExit(f"set_preference_status leaked local path in raw id metadata: {path_bad_preference_id.metadata}")
            assert_preference_refusal_handoff(
                path_bad_preference_id.metadata,
                "set_preference_status path bad id",
                source="set_preference_status",
                mutation="preference_status_update",
                reason="bad_preference_id",
                read_only=False,
            )

        for bad_numeric_id in (0, -1):
            bad_numeric_preference_id = runtime.registry.get("set_preference_status").handler({"preference_id": bad_numeric_id, "status": "retired"})
            if bad_numeric_preference_id.ok or "positive number" not in bad_numeric_preference_id.output:
                raise SystemExit("set_preference_status should reject non-positive ids before mutation.")
            if bad_numeric_preference_id.metadata.get("reason") != "bad_preference_id" or bad_numeric_preference_id.metadata.get("preference_id") is not None:
                raise SystemExit(f"set_preference_status non-positive id should include bad-id metadata: {bad_numeric_preference_id.metadata}")
            if bad_numeric_preference_id.metadata.get("raw_preference_id") != str(bad_numeric_id):
                raise SystemExit(f"set_preference_status should preserve non-positive raw id metadata: {bad_numeric_preference_id.metadata}")
            if bad_numeric_preference_id.metadata.get("writes_files") or bad_numeric_preference_id.metadata.get("writes_memory") or bad_numeric_preference_id.metadata.get("writes_notes"):
                raise SystemExit(f"set_preference_status non-positive id should not write: {bad_numeric_preference_id.metadata}")
            assert_preference_refusal_handoff(
                bad_numeric_preference_id.metadata,
                "set_preference_status non-positive id",
                source="set_preference_status",
                mutation="preference_status_update",
                reason="bad_preference_id",
                read_only=False,
            )

        missing_status_target = runtime.registry.get("set_preference_status").handler({"preference_id": 999999, "status": "retired"})
        if missing_status_target.ok or missing_status_target.metadata.get("reason") != "not_found":
            raise SystemExit(f"set_preference_status missing target should return not-found metadata: {missing_status_target.metadata}")
        if missing_status_target.metadata.get("writes_files") or missing_status_target.metadata.get("writes_memory") or missing_status_target.metadata.get("writes_notes"):
            raise SystemExit(f"set_preference_status missing target should not write: {missing_status_target.metadata}")
        expected_preference_recovery = [
            "preferences",
            "show preference <correct preference key>",
            "preference <correct preference id> <active|retired>",
        ]
        if missing_status_target.metadata.get("recovery_commands") != expected_preference_recovery:
            raise SystemExit(f"set_preference_status missing target recovery order drifted: {missing_status_target.metadata}")
        for token in ("Run `preferences`", "refresh preference IDs", "show preference <correct preference key>", "normal local-safe policy"):
            if token not in missing_status_target.output:
                raise SystemExit(f"set_preference_status missing target output missed {token!r}: {missing_status_target.output}")
        for key in ("retry_requires_preference_refresh", "retry_requires_corrected_id", "recovery_commands_require_normal_policy"):
            if missing_status_target.metadata.get(key) is not True:
                raise SystemExit(f"set_preference_status missing target missed {key}: {missing_status_target.metadata}")
        for key in ("retry_requires_fresh_approval", "authorizes_retry", "authorizes_preference_mutation"):
            if missing_status_target.metadata.get(key):
                raise SystemExit(f"set_preference_status missing target unexpectedly set {key}: {missing_status_target.metadata}")
        assert_preference_refusal_handoff(
            missing_status_target.metadata,
            "set_preference_status missing target",
            source="set_preference_status",
            mutation="preference_status_update",
            reason="not_found",
            read_only=False,
        )

        missing_set_key = runtime.registry.get("set_preference").handler({"key": "", "value": "warm"})
        if missing_set_key.ok or missing_set_key.metadata.get("raw_key") != "":
            raise SystemExit(f"set_preference missing key should preserve bounded raw key metadata: {missing_set_key.metadata}")
        assert_preference_refusal_handoff(
            missing_set_key.metadata,
            "set_preference missing key",
            source="set_preference",
            mutation="preference_create",
                reason="missing_key",
                read_only=False,
            )
        missing_set_value = runtime.registry.get("set_preference").handler({"key": "voice", "value": ""})
        if missing_set_value.ok or missing_set_value.metadata.get("reason") != "missing_value":
            raise SystemExit(f"set_preference missing value should include refusal metadata: {missing_set_value.metadata}")
        if missing_set_value.metadata.get("writes_files") or missing_set_value.metadata.get("writes_memory") or missing_set_value.metadata.get("writes_notes"):
            raise SystemExit(f"set_preference missing value should not write: {missing_set_value.metadata}")
        assert_preference_refusal_handoff(
            missing_set_value.metadata,
            "set_preference missing value",
            source="set_preference",
            mutation="preference_create",
            reason="missing_value",
            read_only=False,
        )
        for path_key in (
            "/private/tmp/jarvis-preference-key",
            "/var/folders/zc/jarvis/preference-key",
            "/tmp/jarvis/preference-key",
        ):
            path_bad_set_key = runtime.registry.get("set_preference").handler({"key": path_key, "value": "warm"})
            if path_bad_set_key.ok or path_bad_set_key.metadata.get("reason") != "invalid_key":
                raise SystemExit("set_preference path-shaped key should include safe refusal metadata.")
            if path_bad_set_key.metadata.get("raw_key") != "<local-path>":
                raise SystemExit(f"set_preference leaked local path in raw key metadata: {path_bad_set_key.metadata}")
            if path_bad_set_key.metadata.get("writes_files") or path_bad_set_key.metadata.get("writes_memory") or path_bad_set_key.metadata.get("writes_notes"):
                raise SystemExit(f"set_preference path-shaped key should not write: {path_bad_set_key.metadata}")
            if any(fragment in path_bad_set_key.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"set_preference leaked local path in refusal output: {path_bad_set_key.output}")
            assert_preference_refusal_handoff(
                path_bad_set_key.metadata,
                "set_preference path-shaped key",
                source="set_preference",
                mutation="preference_create",
                reason="invalid_key",
                read_only=False,
            )

        long_preference = runtime.registry.get("set_preference").handler({"key": "k" * 500, "value": "v" * 5000, "category": "c" * 200})
        if not long_preference.ok:
            raise SystemExit("set_preference should accept bounded long text.")
        if long_preference.metadata.get("key_chars") != 120 or long_preference.metadata.get("value_chars") != 2000:
            raise SystemExit("set_preference did not bound long key/value text.")
        assert_preference_mutation_handoff(
            long_preference.metadata,
            "long set_preference",
            source="set_preference",
            mutation="preference_create",
            changed=["preference"],
        )
        assert_write_receipt(long_preference, root=root, label="long set_preference")


if __name__ == "__main__":
    main()
