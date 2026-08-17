"""Smoke tests for the contacts resolver (mocked Contacts lookup, no address book)."""

from __future__ import annotations

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools import contacts_connector as cc


def _tools():
    return {t.name: t for t in cc.make_contacts_tools(load_config())}


def _mock_query(mapping):
    """Return a fake _run_contacts_query backed by a {query_substr: raw} map."""

    def _runner(query: str) -> str:
        low = query.lower()
        for key, raw in mapping.items():
            if key in low:
                return raw
        return ""

    return _runner


def _reset(mapping):
    cc.clear_contact_cache()
    cc._run_contacts_query = _mock_query(mapping)  # type: ignore


def _assert_contact_handoff(metadata: dict, label: str, *, status: str, reason: str = "", reads_personal_data: bool = True) -> dict:
    handoff = metadata.get("contact_lookup_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed contact lookup handoff: {metadata}")
    if metadata.get("contact_lookup_handoff_ready") is not True:
        raise SystemExit(f"{label} missed contact handoff readiness: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} missed nested contact handoff readiness: {handoff}")
    if metadata.get("ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should be ready for operator clients: {metadata}")
    if metadata.get("contact_lookup_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} ready alias parity failed: {metadata} vs {handoff}")
    if metadata.get("state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report no state change: {metadata}")
    if metadata.get("changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should expose empty changed lists: {metadata}")
    if metadata.get("contact_lookup_state_changed") != handoff.get("state_changed") or metadata.get("contact_lookup_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} state alias parity failed: {metadata} vs {handoff}")
    if metadata.get("content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep contact content out of handoff prose fields: {metadata}")
    if metadata.get("contact_lookup_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} content alias parity failed: {metadata} vs {handoff}")
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False in flat and nested metadata: {metadata} / {handoff}")
        alias = f"contact_lookup_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if handoff.get("source") != "find_contact" or handoff.get("status") != status:
        raise SystemExit(f"{label} contact handoff source/status wrong: {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} contact handoff reason wrong: {handoff}")
    if handoff.get("match_count") != metadata.get("match_count"):
        raise SystemExit(f"{label} contact handoff count parity failed: {handoff} vs {metadata}")
    rows = handoff.get("matches")
    if not isinstance(rows, list) or len(rows) != handoff.get("match_count"):
        raise SystemExit(f"{label} contact handoff rows wrong: {handoff}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} contact handoff should not include contact prose content: {handoff}")
    if metadata.get("contact_lookup_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} content metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("exception_type", "") != metadata.get("exception_type", ""):
        raise SystemExit(f"{label} exception metadata parity failed: {metadata} vs {handoff}")
    if metadata.get("contact_lookup_exception_type", "") != handoff.get("exception_type", ""):
        raise SystemExit(f"{label} exception alias parity failed: {metadata} vs {handoff}")
    if handoff.get("retry_safe") is not (status != "ok"):
        raise SystemExit(f"{label} contact handoff retry state wrong: {handoff}")
    if handoff.get("next_safe_command") != "find contact <name>":
        raise SystemExit(f"{label} contact next safe command wrong: {handoff}")
    expected_commands = [handoff["next_safe_command"]]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} contact safe-command list wrong: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} contact safe-command count wrong: {handoff}")
    if metadata.get("contact_lookup_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("contact_lookup_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("contact_lookup_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} next-command count alias parity failed: {metadata} vs {handoff}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "calls_model": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_personal_data": reads_personal_data,
        "reads_private_data": False,
        "executes_side_effect": False,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "controls_computer": False,
    }
    if boundaries != expected:
        raise SystemExit(f"{label} contact boundaries wrong: {boundaries}")
    if metadata.get("contact_lookup_boundaries") != boundaries:
        raise SystemExit(f"{label} contact boundary alias parity failed: {metadata} vs {handoff}")
    for key, value in expected.items():
        if metadata.get(key, value) is not value:
            raise SystemExit(f"{label} flat contact boundary {key} wrong: {metadata}")
    return handoff


def test_find_contact_is_local_safe() -> None:
    if _tools()["find_contact"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("find_contact should be LOCAL_SAFE")


def test_single_match_resolves() -> None:
    _reset({"fixture": "Fixture Example:::+1 415 555 0100:::fixture@example.com\n"})
    matches = cc.resolve_contact("fixture")
    if len(matches) != 1:
        raise SystemExit(f"expected one match, got {matches}")
    m = matches[0]
    if m.name != "Fixture Example" or m.phone != "+1 415 555 0100" or m.handle != "+1 415 555 0100":
        raise SystemExit(f"single match parsed wrong: {m}")


def test_multiple_matches_returned_for_disambiguation() -> None:
    _reset({"fixture": "Fixture Example:::+1 415 555 0100:::\nFixture Kim:::+1 415 555 0200:::\n"})
    matches = cc.resolve_contact("fixture")
    if len(matches) != 2:
        raise SystemExit(f"expected two matches for disambiguation, got {matches}")
    names = {m.name for m in matches}
    if names != {"Fixture Example", "Fixture Kim"}:
        raise SystemExit(f"wrong names: {names}")


def test_no_match_returns_empty() -> None:
    _reset({"fixture": "Fixture Example:::+1 415 555 0100:::\n"})
    if cc.resolve_contact("zelophehad") != []:
        raise SystemExit("unknown name should resolve to no matches")


def test_email_only_contact_uses_email_handle() -> None:
    _reset({"sam": "Fixture Sam::::::sam@example.com\n"})
    matches = cc.resolve_contact("sam")
    if len(matches) != 1 or matches[0].handle != "sam@example.com":
        raise SystemExit(f"email-only handle wrong: {matches}")


def test_multi_handle_contact_surfaces_all_and_prefers_mobile() -> None:
    # name:::label^value|label^value:::email|email   (labels in macOS internal form)
    _reset({"jane": "Jane Doe:::_$!<Work>!$_^+1 415 555 0001|_$!<Mobile>!$_^+1 415 555 0002:::jane@example.com|jane@example.com\n"})
    matches = cc.resolve_contact("jane")
    if len(matches) != 1:
        raise SystemExit(f"expected one Jane, got {matches}")
    m = matches[0]
    if m.all_phones() != ["+1 415 555 0001", "+1 415 555 0002"]:
        raise SystemExit(f"all phones wrong: {m.all_phones()}")
    if m.all_emails() != ["jane@example.com", "jane@example.com"]:
        raise SystemExit(f"all emails wrong: {m.all_emails()}")
    # handle must prefer the MOBILE number even though it's listed second
    if m.handle != "+1 415 555 0002":
        raise SystemExit(f"handle should prefer mobile, got {m.handle}")
    # labels should be cleaned from the macOS internal form
    labels = [label for label, _ in m.phones]
    if labels != ["Work", "Mobile"]:
        raise SystemExit(f"labels not cleaned: {labels}")


def test_find_contact_lists_all_handles() -> None:
    _reset({"jane": "Jane Doe:::_$!<Mobile>!$_^+15551112222:::jane@example.com|jane@example.com\n"})
    out = _tools()["find_contact"].handler({"query": "jane"})
    if "+15551112222" not in out.output or "jane@example.com" not in out.output or "jane@example.com" not in out.output:
        raise SystemExit(f"find_contact should list all handles: {out.output}")
    if "(Mobile)" not in out.output:
        raise SystemExit(f"find_contact should show phone label: {out.output}")


def test_matches_ranked_by_relevance() -> None:
    # "David" should rank a prefix match above a mere substring match, regardless
    # of the order Contacts returns them in.
    _reset({"david": "Han David:::^010111:::\nDavid Mom:::^010222:::\nDavid:::^010333:::\n"})
    names = [m.name for m in cc.resolve_contact("David")]
    if names != ["David", "David Mom", "Han David"]:
        raise SystemExit(f"ranking wrong: {names}")


def test_exact_duplicate_contacts_deduped() -> None:
    # Same name + same handle twice → one entry; different handle → kept.
    _reset({"sam": "Sam Lee:::^0101:::\nSam Lee:::^0101:::\nSam Lee:::^0102:::\n"})
    matches = cc.resolve_contact("sam")
    if len(matches) != 2:
        raise SystemExit(f"expected 2 after dedupe, got {[(m.name, m.handle) for m in matches]}")
    if sorted(m.handle for m in matches) != ["0101", "0102"]:
        raise SystemExit(f"dedupe dropped the wrong entries: {[m.handle for m in matches]}")


def test_phone_and_email_bypass_resolution() -> None:
    if not cc.looks_like_handle("+1 415 555 0100"):
        raise SystemExit("phone should be detected as a handle")
    if not cc.looks_like_handle("fixture@example.com"):
        raise SystemExit("email should be detected as a handle")
    if cc.looks_like_handle("fixture"):
        raise SystemExit("a bare name must NOT look like a handle")


def test_lookup_failure_is_silent_empty() -> None:
    cc.clear_contact_cache()

    def boom(query: str) -> str:
        raise RuntimeError("Contacts permission denied")

    cc._run_contacts_query = boom  # type: ignore
    if cc.resolve_contact("fixture") != []:
        raise SystemExit("missing Contacts permission should yield no matches, not raise")


def test_contacts_store_status_is_content_free() -> None:
    original = cc._run_contacts_count
    try:
        cc._run_contacts_count = lambda: 0  # type: ignore
        if cc.contacts_store_status() != "empty":
            raise SystemExit("zero saved contacts should report an empty Contacts store")
        cc._run_contacts_count = lambda: 3  # type: ignore
        if cc.contacts_store_status() != "available":
            raise SystemExit("non-empty Contacts store should report available")

        def unavailable() -> int:
            raise RuntimeError("Contacts permission denied")

        cc._run_contacts_count = unavailable  # type: ignore
        if cc.contacts_store_status() != "unavailable":
            raise SystemExit("unreadable Contacts store should report unavailable")
    finally:
        cc._run_contacts_count = original  # type: ignore


def test_find_contact_lookup_failure_names_recovery() -> None:
    cc.clear_contact_cache()

    def boom(query: str) -> str:
        raise RuntimeError("Contacts permission denied near /\x55sers/example/private")

    cc._run_contacts_query = boom  # type: ignore
    out = _tools()["find_contact"].handler({"query": "fixture"})
    lower = out.output.lower()
    if out.ok:
        raise SystemExit(f"Contacts lookup failure should fail the direct tool: {out.output} / {out.metadata}")
    for expected in [
        "couldn't check macos contacts",
        "open contacts once",
        "system settings",
        "privacy & security",
        "contacts",
        "retry `find contact <name>`",
    ]:
        if expected not in lower:
            raise SystemExit(f"Contacts lookup failure should name recovery step {expected!r}: {out.output}")
    for forbidden in ["permission denied", "/users/", "/private/", "traceback"]:
        if forbidden in lower:
            raise SystemExit(f"Contacts lookup failure leaked raw text {forbidden!r}: {out.output}")
    if out.metadata.get("reason") != "contacts_unavailable" or out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"Contacts lookup failure should expose bounded diagnostic metadata: {out.metadata}")
    action = cc.PERSONAL_READ_RECOVERY_ACTION
    if action not in out.output:
        raise SystemExit(f"Contacts lookup failure hid the canonical recovery action: {out.output}")
    if out.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"Contacts lookup recovery declaration drifted: {out.metadata}")
    expected_recovery = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected_recovery.items():
        if out.metadata.get(key) != value:
            raise SystemExit(
                f"Contacts lookup recovery field {key} drifted: {out.metadata}"
            )
    _assert_contact_handoff(
        out.metadata,
        "find_contact Contacts unavailable",
        status="unavailable",
        reason="contacts_unavailable",
        reads_personal_data=True,
    )
    if cc.resolve_contact("fixture") != []:
        raise SystemExit("direct tool failure must not change silent resolver behavior")


def test_local_path_queries_never_read_contacts() -> None:
    calls = []
    cc.clear_contact_cache()

    def fail_query(query: str) -> str:
        calls.append(query)
        raise AssertionError("path-shaped contact lookup should not query Contacts")

    cc._run_contacts_query = fail_query  # type: ignore
    if cc.resolve_contact("/\x55sers/example/private/contact-name") != []:
        raise SystemExit("path-shaped resolver input should return no matches")
    out = _tools()["find_contact"].handler({"query": "/private/tmp/jarvis-contact-name"})
    if out.ok or "not a local file path" not in out.output:
        raise SystemExit(f"path-shaped find_contact query should be refused locally: {out.output}")
    if out.metadata.get("reason") != "invalid_query" or out.metadata.get("local_path_query") is not True:
        raise SystemExit(f"path-shaped find_contact query should expose refusal metadata: {out.metadata}")
    handoff = _assert_contact_handoff(
        out.metadata,
        "find_contact path-shaped query",
        status="refused",
        reason="invalid_query",
        reads_personal_data=False,
    )
    if handoff.get("query") != "<local-path>":
        raise SystemExit(f"path-shaped contact handoff should redact query: {handoff}")
    if calls:
        raise SystemExit(f"path-shaped contact lookup unexpectedly queried Contacts: {calls}")


def test_find_contact_tool_outputs() -> None:
    _reset({"fixture": "Fixture Example:::+1 415 555 0100:::fixture@example.com\n"})
    out = _tools()["find_contact"].handler({"query": "fixture"})
    if not out.ok or "Fixture Example" not in out.output or out.metadata.get("match_count") != 1:
        raise SystemExit(f"find_contact single output wrong: {out.output} / {out.metadata}")
    handoff = _assert_contact_handoff(out.metadata, "find_contact single", status="ok")
    if handoff.get("query") != "fixture" or handoff.get("matches", [{}])[0].get("handle_type") != "phone":
        raise SystemExit(f"find_contact single handoff should expose safe row metadata: {handoff}")

    _reset({})
    out = _tools()["find_contact"].handler({"query": "ghost"})
    if not out.ok or "couldn't find" not in out.output or out.metadata.get("match_count") != 0:
        raise SystemExit(f"find_contact no-match output wrong: {out.output} / {out.metadata}")
    _assert_contact_handoff(out.metadata, "find_contact no match", status="empty", reason="no_match")

    out = _tools()["find_contact"].handler({})
    if out.ok or "Who should I look up" not in out.output:
        raise SystemExit(f"find_contact missing-query output wrong: {out.output}")
    _assert_contact_handoff(
        out.metadata,
        "find_contact missing query",
        status="refused",
        reason="missing_query",
        reads_personal_data=False,
    )


def test_planner_routes_find_contact() -> None:
    p = RuleBasedPlanner()
    for q in [
        "find contact fixture",
        "find contact fixture please",
        "find the contact fixture please",
        "look up contact fixture please",
        "lookup contact fixture please",
        "search contact fixture please",
        "who is contact fixture please",
        "what's fixture's number",
        "what's fixture's email please",
        "look up fixture in my contacts",
        "look up fixture in my contacts please",
        # Real gaps found live 2026-07-09: "search contacts for X" (plural
        # "contacts", not "contact") fell through to chat, and "look up X's
        # phone number" (no "contact" word, no "what's" lead-in) silently
        # misrouted to a public web_lookup for a person's phone number instead
        # of the local, LOCAL_SAFE find_contact tool -- a privacy-relevant
        # misroute, not just a refusal. Note: "look up X's email" (as opposed
        # to "phone number") is intentionally NOT included here -- "email" is
        # a RISKY_NATURAL_ORDER_HINTS keyword, so "look up ..." (an
        # ACTION_ORDER_STARTS verb) combined with "email" correctly routes
        # through the safety dispatcher instead of auto-resolving; that is
        # deliberate design, not a bug, and matches the "what's X's email"
        # question-form phrasing (which bypasses the risky check entirely)
        # already covered above.
        "search contacts for fixture",
        "search contacts for fixture please",
        "look up fixture's phone number",
        "look up fixture's phone number please",
    ]:
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["find_contact"]:
            raise SystemExit(f"find_contact route missed: {q!r} -> {[a.tool_name for a in actions]}")
        if actions[0].args != {"query": "fixture"}:
            raise SystemExit(f"find_contact route should clean polite query: {q!r} -> {actions[0].args}")
    if p.plan("contact fixture please").actions:
        raise SystemExit("bare contact phrase should stay unclaimed because it may imply an action")
    # Real gap found live 2026-07-10: "find contact fixture and then call him"
    # (a compound sentence) swallowed the whole second clause into the
    # contact search query ("fixture and then call him"), which would fail to
    # match any saved contact instead of finding "fixture" -- and silently
    # dropped the "call him" intent.
    compound_actions = p.plan("find contact fixture and then call him").actions
    if [a.tool_name for a in compound_actions] != ["find_contact"] or compound_actions[0].args != {"query": "fixture"}:
        raise SystemExit(f"planner should stop contact query at a compound-sentence boundary: {compound_actions}")


def main() -> None:
    test_find_contact_is_local_safe()
    test_single_match_resolves()
    test_multiple_matches_returned_for_disambiguation()
    test_no_match_returns_empty()
    test_email_only_contact_uses_email_handle()
    test_multi_handle_contact_surfaces_all_and_prefers_mobile()
    test_find_contact_lists_all_handles()
    test_matches_ranked_by_relevance()
    test_exact_duplicate_contacts_deduped()
    test_phone_and_email_bypass_resolution()
    test_lookup_failure_is_silent_empty()
    test_contacts_store_status_is_content_free()
    test_find_contact_lookup_failure_names_recovery()
    test_local_path_queries_never_read_contacts()
    test_find_contact_tool_outputs()
    test_planner_routes_find_contact()
    print("Contacts connector smoke passed")


if __name__ == "__main__":
    main()
