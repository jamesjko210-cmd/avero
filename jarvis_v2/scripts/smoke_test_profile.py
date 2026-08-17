from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.profile import _metadata_bool, _profile_handoff_metadata, make_profile_tools


def assert_route(command: str, tool_name: str, args: dict | None = None) -> None:
    plan = RuleBasedPlanner().plan(command)
    actions = plan.actions
    if len(actions) != 1 or actions[0].tool_name != tool_name:
        raise SystemExit(f"{command!r} should route to {tool_name}: {[(a.tool_name, a.args) for a in actions]}")
    expected_args = args or {}
    if actions[0].args != expected_args:
        raise SystemExit(f"{command!r} should pass {expected_args}: {actions[0].args}")


def test_profile_read_alias_routes() -> None:
    for command in (
        "profile please",
        "profile notes please",
        "read profile please",
        "show latest profile",
        "my profile please",
        "profile me please",
        "what do you know about the operator",
        # Real bug found live 2026-07-09: "read my profile" was silently
        # misrouted to `read_text_file` (path="my profile") instead of
        # `read_profile`, because a downstream generic file-read guard used a
        # naive `"file" in low` substring check -- and "profile" contains
        # "file" as a substring ("pro" + "file"). Fixed the guard to use a
        # \bfile\b word-boundary check, and added these natural phrasings to
        # the exact-match set so they resolve directly.
        "read my profile",
        "show my profile",
        "what's in my profile",
    ):
        assert_route(command, "read_profile")
    # Non-regression: the word-boundary fix to the "file" guard must still let
    # genuine file-read requests (the literal word "file") through.
    assert_route("read file notes", "read_text_file", {"path": "notes"})
    assert_route(
        "add profile note Family: Lives in Seoul",
        "add_profile_note",
        {"heading": "Family", "body": "Lives in Seoul"},
    )
    assert_route(
        "add profile note https://example.com is a reference",
        "add_profile_note",
        {
            "heading": "https://example.com is a reference",
            "body": "https://example.com is a reference",
        },
    )


def assert_write_receipt(result, *, root: Path, label: str) -> None:
    metadata = result.metadata
    path_display = metadata.get("path_display")
    if "path" in metadata:
        raise SystemExit(f"{label} exposed a raw local path in metadata: {metadata}")
    if not (root / "Profile.md").exists():
        raise SystemExit(f"{label} did not preserve the saved profile note.")
    if not isinstance(path_display, str) or path_display != "Profile.md":
        raise SystemExit(f"{label} missed safe Profile.md path_display: {metadata}")
    if f"Saved note: {path_display}" not in result.output:
        raise SystemExit(f"{label} missed safe saved-note receipt: {result.output}")
    if any(fragment in result.output for fragment in (str(root), "/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} leaked a local path in output: {result.output}")


def assert_profile_boundaries(handoff: dict, *, writes: bool, label: str) -> None:
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("read_only") is not (not writes):
        raise SystemExit(f"{label} read_only boundary mismatch: {handoff}")
    if boundaries.get("reads_profile") is not (not writes):
        raise SystemExit(f"{label} reads_profile boundary mismatch: {handoff}")
    for key in ["writes_files", "writes_memory", "writes_notes", "writes_database"]:
        if boundaries.get(key) is not writes:
            raise SystemExit(f"{label} {key} boundary mismatch: {handoff}")
    for key in [
        "queues_approval",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ]:
        if boundaries.get(key):
            raise SystemExit(f"{label} should keep {key}=False: {handoff}")


def assert_profile_handoff_contract(
    metadata: dict,
    handoff: dict,
    handoff_key: str,
    label: str,
    *,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
) -> None:
    prefix = handoff_key.removesuffix("_handoff")
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        expected_next = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        expected_next = [str(value) for value in raw_next if str(value or "").strip()]
    expected_first = expected_next[0] if expected_next else ""
    expectations = {
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": state_changed,
        "changed": changed,
        "content_in_handoff": content_in_handoff,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_safe_command": expected_first,
        "next_safe_commands": expected_next,
        "next_safe_command_count": len(expected_next),
    }
    for key, expected in expectations.items():
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} handoff {key} mismatch: {handoff}")
        if key != "handoff_ready" and metadata.get(key) != expected:
            raise SystemExit(f"{label} metadata {key} mismatch: {metadata}")
    if metadata.get(f"{handoff_key}_ready") is not True or metadata.get(f"{prefix}_handoff_ready") is not True:
        raise SystemExit(f"{label} missed handoff ready aliases: {metadata}")
    for key in (
        "ready_for_operator",
        "state_changed",
        "changed",
        "content_in_handoff",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "next_safe_command",
        "next_safe_commands",
        "next_safe_command_count",
    ):
        prefixed_key = f"{prefix}_{key}"
        if metadata.get(prefixed_key) != expectations[key]:
            raise SystemExit(f"{label} metadata {prefixed_key} mismatch: {metadata}")


def assert_profile_write_handoff(result, *, label: str) -> None:
    metadata = result.metadata
    handoff = metadata.get("profile_write_handoff")
    if not metadata.get("profile_write_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing profile_write_handoff: {metadata}")
    if handoff.get("source") != "add_profile_note" or handoff.get("memory_id") != metadata.get("memory_id"):
        raise SystemExit(f"{label} write handoff source/memory mismatch: {handoff}")
    if handoff.get("path_display") != metadata.get("path_display") or handoff.get("category") != metadata.get("category"):
        raise SystemExit(f"{label} write handoff path/category mismatch: {handoff}")
    if handoff.get("body_chars") != metadata.get("body_chars") or handoff.get("heading") not in result.output:
        raise SystemExit(f"{label} write handoff body/heading mismatch: {handoff}")
    assert_profile_handoff_contract(
        metadata,
        handoff,
        "profile_write_handoff",
        label,
        state_changed=True,
        changed=["profile_note", "memory"],
        content_in_handoff=True,
    )
    for expected in ["read profile", "export state"]:
        if expected not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} write handoff missed next command {expected!r}: {handoff}")
    if any(fragment in str(handoff) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} write handoff leaked local path: {handoff}")
    assert_profile_boundaries(handoff, writes=True, label=label)


def assert_profile_refusal_handoff(
    result,
    *,
    label: str,
    reason: str,
    raw_heading: str | None = None,
    raw_category: str | None = None,
    raw_body: str | None = None,
    body_chars: int | None = None,
    limit: int | None = None,
) -> None:
    metadata = result.metadata
    handoff = metadata.get("profile_refusal_handoff")
    if result.ok or not metadata.get("profile_refusal_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed profile_refusal_handoff: {metadata}")
    if "profile_write_handoff" in metadata:
        raise SystemExit(f"{label} refusal should not emit profile_write_handoff: {metadata}")
    if handoff.get("source") != "add_profile_note" or handoff.get("mutation") != "profile_note_create":
        raise SystemExit(f"{label} refusal handoff missed source/mutation: {handoff}")
    if metadata.get("reason") != reason or handoff.get("reason") != reason:
        raise SystemExit(f"{label} refusal handoff missed reason parity: {metadata}")
    if handoff.get("ready_for_operator") is not True or handoff.get("refused") is not True:
        raise SystemExit(f"{label} refusal handoff missed readiness/refused state: {handoff}")
    if handoff.get("changed") != []:
        raise SystemExit(f"{label} refusal handoff should report no changed fields: {handoff}")
    assert_profile_handoff_contract(
        metadata,
        handoff,
        "profile_refusal_handoff",
        label,
        state_changed=False,
        changed=[],
        content_in_handoff=any(
            key in handoff for key in ("raw_heading", "raw_category", "raw_body", "body_chars", "limit")
        ),
    )
    if "read profile" not in handoff.get("next_commands", []) or not isinstance(handoff.get("retry_command"), str):
        raise SystemExit(f"{label} refusal handoff missed recovery commands: {handoff}")
    if raw_heading is not None and (metadata.get("raw_heading") != raw_heading or handoff.get("raw_heading") != raw_heading):
        raise SystemExit(f"{label} refusal handoff missed raw heading parity: {metadata}")
    if raw_category is not None and (metadata.get("raw_category") != raw_category or handoff.get("raw_category") != raw_category):
        raise SystemExit(f"{label} refusal handoff missed raw category parity: {metadata}")
    if raw_body is not None and (metadata.get("raw_body") != raw_body or handoff.get("raw_body") != raw_body):
        raise SystemExit(f"{label} refusal handoff missed raw body parity: {metadata}")
    if body_chars is not None and (metadata.get("body_chars") != body_chars or handoff.get("body_chars") != body_chars):
        raise SystemExit(f"{label} refusal handoff missed body_chars parity: {metadata}")
    if limit is not None and (metadata.get("limit") != limit or handoff.get("limit") != limit):
        raise SystemExit(f"{label} refusal handoff missed limit parity: {metadata}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} refusal handoff missed read-only boundary: {handoff}")
    for key in (
        "reads_profile",
        "writes_files",
        "writes_memory",
        "writes_notes",
        "writes_database",
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "calls_model",
        "executes_tools",
        "creates_profile_note",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} refusal handoff should keep {key}=False: {handoff}")
    for key in (
        "writes_files",
        "writes_memory",
        "writes_notes",
        "writes_database",
        "queues_approval",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} flat refusal metadata should keep {key}=False: {metadata}")
    if any(fragment in str(handoff) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} refusal handoff leaked local path: {handoff}")


def assert_profile_read_handoff(result, *, label: str, empty: bool = False) -> None:
    metadata = result.metadata
    handoff = metadata.get("profile_read_handoff")
    if not metadata.get("profile_read_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing profile_read_handoff: {metadata}")
    if handoff.get("source") != "read_profile" or handoff.get("path_display") != "Profile.md":
        raise SystemExit(f"{label} read handoff source/path mismatch: {handoff}")
    if handoff.get("chars") != metadata.get("chars") or handoff.get("max_chars") != metadata.get("max_chars"):
        raise SystemExit(f"{label} read handoff metadata parity failed: {metadata}")
    if handoff.get("empty") is not empty:
        raise SystemExit(f"{label} read handoff empty mismatch: {handoff}")
    assert_profile_handoff_contract(
        metadata,
        handoff,
        "profile_read_handoff",
        label,
        state_changed=False,
        changed=[],
        content_in_handoff=not empty,
    )
    for expected in ["add profile note <heading>: <body>", "export state"]:
        if expected not in handoff.get("next_commands", []):
            raise SystemExit(f"{label} read handoff missed next command {expected!r}: {handoff}")
    if any(fragment in str(handoff) for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} read handoff leaked local path: {handoff}")
    assert_profile_boundaries(handoff, writes=False, label=label)


def assert_profile_exact_metadata_bool() -> None:
    if _metadata_bool(True) is not True:
        raise SystemExit("profile exact bool helper should preserve True.")
    if _metadata_bool(False, default=True) is not False:
        raise SystemExit("profile exact bool helper should preserve False.")
    for value in ("true", "false", "yes", "0", 1, 0, [], ["content"], None):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"profile exact bool helper should reject malformed handoff flags: {value!r}")
    if _metadata_bool("fallback", default=True) is not True:
        raise SystemExit("profile exact bool helper should honor explicit malformed-value default.")


def assert_profile_malformed_handoff_flags() -> None:
    handoff = {
        "source": "read_profile",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["read profile"],
        "boundaries": {"read_only": True},
    }
    metadata = _profile_handoff_metadata("profile_read_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("profile_read_state_changed") is not False:
        raise SystemExit(f"malformed profile state_changed should not become truthy: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("profile_read_content_in_handoff") is not False:
        raise SystemExit(f"malformed profile content_in_handoff should not become truthy: {metadata}")


def main() -> None:
    test_profile_read_alias_routes()
    assert_profile_exact_metadata_bool()
    assert_profile_malformed_handoff_flags()
    with TemporaryDirectory(prefix="jarvis-profile-") as temp:
        runtime = make_temp_runtime(Path(temp))
        add_profile_note, read_profile = make_profile_tools(runtime.store, runtime.vault)
        root = runtime.vault.root_path
        initial_profile = read_profile({})
        if not initial_profile.ok or initial_profile.metadata.get("chars") != 0:
            raise SystemExit(f"heading-only read_profile should report no curated content: {initial_profile.metadata}")
        if initial_profile.metadata.get("empty_reason") != "no_curated_profile_notes":
            raise SystemExit(f"empty read_profile should name the empty-profile reason: {initial_profile.metadata}")
        for expected in (
            "I don't have curated profile notes yet.",
            "add profile note <heading>: <body>",
            "search memory for profile",
        ):
            if expected not in initial_profile.output:
                raise SystemExit(f"empty read_profile missed {expected!r}: {initial_profile.output}")
        if "# Profile" in initial_profile.output:
            raise SystemExit(f"empty read_profile should not return a bare markdown heading: {initial_profile.output}")
        assert_profile_read_handoff(initial_profile, label="initial read_profile", empty=True)

        empty_runtime = runtime.handle("read profile")
        if not empty_runtime.verified or not empty_runtime.tool_results:
            raise SystemExit(f"empty profile runtime route should verify through read_profile: {empty_runtime.response}")
        if empty_runtime.tool_results[0].tool_name != "read_profile":
            raise SystemExit(f"empty profile runtime route should use read_profile: {[r.tool_name for r in empty_runtime.tool_results]}")
        if "I don't have curated profile notes yet." not in empty_runtime.response or "# Profile" in empty_runtime.response:
            raise SystemExit(f"empty profile runtime response should be friendly, not bare markdown: {empty_runtime.response}")
        assert_profile_read_handoff(empty_runtime.tool_results[0], label="empty runtime read_profile", empty=True)
        cases = [
            "add profile note the operator wants Jarvis to feel direct, warm, and capable.",
            "profile note the operator prefers short status updates.",
            "read profile",
            "tell me about me",
            "who am I",
            "what is my name",
            "search memory for direct warm capable",
            "list tools profile",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1200])
            print()
            if case == "add profile note the operator wants Jarvis to feel direct, warm, and capable.":
                assert_write_receipt(result.tool_results[0], root=root, label="runtime add_profile_note")
                assert_profile_write_handoff(result.tool_results[0], label="runtime add_profile_note")
            if case == "profile note the operator prefers short status updates.":
                assert_write_receipt(result.tool_results[0], root=root, label="runtime profile note alias")
                assert_profile_write_handoff(result.tool_results[0], label="runtime profile note alias")
            if case in {"read profile", "tell me about me", "who am I", "what is my name"}:
                assert_profile_read_handoff(result.tool_results[0], label="runtime read_profile")

        read_result = read_profile({"max_chars": "not-a-number"})
        if read_result.metadata.get("max_chars") != 4000 or read_result.metadata.get("writes_files"):
            raise SystemExit("read_profile should sanitize bad max_chars and remain read-only.")
        if read_result.metadata.get("raw_max_chars") != "not-a-number":
            raise SystemExit(f"read_profile should preserve bounded raw max_chars metadata: {read_result.metadata}")
        if read_result.metadata.get("writes_memory") or read_result.metadata.get("writes_notes"):
            raise SystemExit("read_profile should not mark memory or note writes.")
        if read_result.metadata.get("writes_database") is not False:
            raise SystemExit("read_profile should explicitly report no database writes.")
        assert_profile_read_handoff(read_result, label="bad max_chars read_profile")

        bool_result = read_profile({"max_chars": True})
        if bool_result.metadata.get("max_chars") != 4000 or bool_result.metadata.get("raw_max_chars") != "True":
            raise SystemExit(f"read_profile should treat boolean max_chars as malformed and preserve raw metadata: {bool_result.metadata}")
        if bool_result.metadata.get("writes_files") or bool_result.metadata.get("writes_database"):
            raise SystemExit("read_profile boolean max_chars should stay read-only.")
        assert_profile_read_handoff(bool_result, label="boolean max_chars read_profile")

        long_bad_result = read_profile({"max_chars": "m" * 120})
        if long_bad_result.metadata.get("max_chars") != 4000:
            raise SystemExit("read_profile should sanitize long bad max_chars.")
        if long_bad_result.metadata.get("raw_max_chars") != ("m" * 77 + "..."):
            raise SystemExit(f"read_profile should bound long raw max_chars metadata: {long_bad_result.metadata}")

        for path_max in (
            "/\x55sers/example/private/profile-max",
            "/var/folders/zc/jarvis/profile-max",
            "/tmp/jarvis/profile-max",
        ):
            path_bad_result = read_profile({"max_chars": path_max})
            if path_bad_result.metadata.get("max_chars") != 4000:
                raise SystemExit("read_profile should sanitize path-shaped bad max_chars.")
            if path_bad_result.metadata.get("raw_max_chars") != "<local-path>":
                raise SystemExit(f"read_profile should redact path-shaped raw max_chars metadata: {path_bad_result.metadata}")

        clipped_result = read_profile({"max_chars": 999999})
        if clipped_result.metadata.get("max_chars") != 20000:
            raise SystemExit("read_profile should clamp oversized max_chars.")
        assert_profile_read_handoff(clipped_result, label="clipped read_profile")

        tiny_result = read_profile({"max_chars": -5})
        if tiny_result.metadata.get("max_chars") != 1:
            raise SystemExit("read_profile should clamp low max_chars.")
        if tiny_result.metadata.get("empty_reason") != "read_limit_before_profile_notes":
            raise SystemExit(f"tiny read_profile should explain that the read limit hid profile notes: {tiny_result.metadata}")
        if tiny_result.metadata.get("profile_has_curated_notes") is not True:
            raise SystemExit(f"tiny read_profile should not claim curated notes are absent: {tiny_result.metadata}")
        if "read limit is too small" not in tiny_result.output or "read profile max_chars 4000" not in tiny_result.output:
            raise SystemExit(f"tiny read_profile missed recovery guidance: {tiny_result.output}")
        assert_profile_read_handoff(tiny_result, label="tiny read_profile", empty=True)

        oversized_result = add_profile_note({"heading": "Too Large", "body": "x" * 50001})
        if oversized_result.ok or "too large" not in oversized_result.output:
            raise SystemExit("add_profile_note should refuse oversized profile bodies.")
        if oversized_result.metadata.get("body_chars") != 50001 or oversized_result.metadata.get("writes_files"):
            raise SystemExit("add_profile_note oversized refusal should include safe non-write metadata.")
        if "profile_write_handoff" in oversized_result.metadata:
            raise SystemExit(f"add_profile_note oversized refusal should not emit handoff: {oversized_result.metadata}")
        assert_profile_refusal_handoff(oversized_result, label="oversized add_profile_note", reason="oversized_body", raw_heading="Too Large", raw_category="identity", body_chars=50001, limit=50000)

        add_result = add_profile_note({"heading": "Safe Metadata", "body": "Jarvis records profile notes safely."})
        if not add_result.metadata.get("writes_files") or not add_result.metadata.get("writes_database"):
            raise SystemExit("add_profile_note should mark local file and database writes.")
        if not add_result.metadata.get("writes_memory") or not add_result.metadata.get("writes_notes"):
            raise SystemExit("add_profile_note should mark memory and note writes.")
        if add_result.metadata.get("queues_approval") or add_result.metadata.get("controls_computer"):
            raise SystemExit("add_profile_note should not queue approvals or control the computer.")
        assert_write_receipt(add_result, root=root, label="direct add_profile_note")
        assert_profile_write_handoff(add_result, label="direct add_profile_note")

        missing_body = add_profile_note({"heading": "Missing Body", "body": ""})
        if missing_body.ok or missing_body.metadata.get("reason") != "missing_body":
            raise SystemExit("add_profile_note missing body should include safe refusal metadata.")
        if missing_body.metadata.get("writes_database") is not False:
            raise SystemExit("add_profile_note missing body should explicitly report no database writes.")
        if "profile_write_handoff" in missing_body.metadata:
            raise SystemExit(f"add_profile_note missing body should not emit handoff: {missing_body.metadata}")
        assert_profile_refusal_handoff(missing_body, label="missing body add_profile_note", reason="missing_body", raw_heading="Missing Body", raw_category="identity", raw_body="")
        for path_heading in (
            "/\x55sers/example/private/profile-heading",
            "/var/folders/zc/jarvis/profile-heading",
            "/tmp/jarvis/profile-heading",
        ):
            path_bad_heading = add_profile_note({"heading": path_heading, "body": "safe body"})
            if path_bad_heading.ok or path_bad_heading.metadata.get("reason") != "invalid_heading":
                raise SystemExit("add_profile_note path-shaped heading should include safe refusal metadata.")
            if path_bad_heading.metadata.get("raw_heading") != "<local-path>":
                raise SystemExit(f"add_profile_note leaked local path in raw heading metadata: {path_bad_heading.metadata}")
            if path_bad_heading.metadata.get("writes_files") or path_bad_heading.metadata.get("writes_memory") or path_bad_heading.metadata.get("writes_notes") or path_bad_heading.metadata.get("writes_database"):
                raise SystemExit(f"add_profile_note path-shaped heading should not write: {path_bad_heading.metadata}")
            if any(fragment in path_bad_heading.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"add_profile_note leaked local path in heading refusal output: {path_bad_heading.output}")
            assert_profile_refusal_handoff(path_bad_heading, label="path heading add_profile_note", reason="invalid_heading", raw_heading="<local-path>")

        for path_category in (
            "/private/tmp/profile-category",
            "/var/folders/zc/jarvis/profile-category",
            "/tmp/jarvis/profile-category",
        ):
            path_bad_category = add_profile_note({"heading": "Safe Heading", "body": "safe body", "category": path_category})
            if path_bad_category.ok or path_bad_category.metadata.get("reason") != "invalid_category":
                raise SystemExit("add_profile_note path-shaped category should include safe refusal metadata.")
            if path_bad_category.metadata.get("raw_category") != "<local-path>":
                raise SystemExit(f"add_profile_note leaked local path in raw category metadata: {path_bad_category.metadata}")
            if path_bad_category.metadata.get("writes_files") or path_bad_category.metadata.get("writes_memory") or path_bad_category.metadata.get("writes_notes") or path_bad_category.metadata.get("writes_database"):
                raise SystemExit(f"add_profile_note path-shaped category should not write: {path_bad_category.metadata}")
            if any(fragment in path_bad_category.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"add_profile_note leaked local path in category refusal output: {path_bad_category.output}")
            assert_profile_refusal_handoff(path_bad_category, label="path category add_profile_note", reason="invalid_category", raw_category="<local-path>")

        for path_body in (
            "/\x55sers/example/private/profile-body",
            "/var/folders/zc/jarvis/profile-body",
            "/tmp/jarvis/profile-body",
        ):
            path_bad_body = add_profile_note({"heading": "Safe Heading", "body": f"profile body {path_body}"})
            if path_bad_body.ok or path_bad_body.metadata.get("reason") != "invalid_body":
                raise SystemExit("add_profile_note path-shaped body should include safe refusal metadata.")
            if path_bad_body.metadata.get("raw_body") != "profile body <local-path>":
                raise SystemExit(f"add_profile_note leaked local path in raw body metadata: {path_bad_body.metadata}")
            if path_bad_body.metadata.get("writes_files") or path_bad_body.metadata.get("writes_memory") or path_bad_body.metadata.get("writes_notes") or path_bad_body.metadata.get("writes_database"):
                raise SystemExit(f"add_profile_note path-shaped body should not write: {path_bad_body.metadata}")
            if any(fragment in path_bad_body.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"add_profile_note leaked local path in body refusal output: {path_bad_body.output}")
            assert_profile_refusal_handoff(path_bad_body, label="path body add_profile_note", reason="invalid_body", raw_body="profile body <local-path>")

        legacy_profile = runtime.vault.root_path / "Profile.md"
        legacy_profile.write_text(
            "# Profile\n\nLegacy /private/tmp/profile-read and /var/folders/zc/jarvis/profile-read and /tmp/jarvis/profile-read\n",
            encoding="utf-8",
        )
        legacy_read = read_profile({"max_chars": 4000})
        if any(fragment in legacy_read.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")) or "<local-path>" not in legacy_read.output:
            raise SystemExit(f"read_profile should redact legacy local paths: {legacy_read.output}")
        assert_profile_read_handoff(legacy_read, label="legacy read_profile")

        bounded = add_profile_note({"heading": "h" * 500, "body": "short body", "category": "c" * 500})
        if not bounded.ok:
            raise SystemExit("add_profile_note should accept bounded long heading/category.")
        if bounded.metadata.get("heading_chars") != 120 or len(bounded.metadata.get("category", "")) != 64:
            raise SystemExit("add_profile_note did not bound long heading/category.")
        assert_write_receipt(bounded, root=root, label="bounded add_profile_note")
        assert_profile_write_handoff(bounded, label="bounded add_profile_note")


if __name__ == "__main__":
    main()
