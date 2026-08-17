"""Smoke tests for fuzzy contact suggestions (mocked address book — no Contacts app)."""

from __future__ import annotations

from jarvis_v2.tools import contacts_fuzzy as cf


_BOOK = "Fixture Example|||Fixturette Kim|||Fixture Lee|||Fixture Secondary|||Mom|||가상연락처삼|||"


def _mock(raw: str = _BOOK):
    cf._run_all_contacts_query = lambda: raw  # type: ignore


def _reset():
    cf._run_all_contacts_query = None  # type: ignore


def test_first_name_matches_full_contacts() -> None:
    _mock()
    try:
        names = cf.suggest_contacts("fixture")
        if "Fixture Example" not in names or "Fixture Lee" not in names:
            raise SystemExit(f"first name should surface the Fixtures: {names}")
        if len(names) > cf._MAX_SUGGESTIONS:
            raise SystemExit(f"should cap at {cf._MAX_SUGGESTIONS}: {names}")
    finally:
        _reset()


def test_typo_and_prefix_match() -> None:
    _mock()
    try:
        if "Fixture Example" not in cf.suggest_contacts("fixtur"):  # typo
            raise SystemExit("typo 'fixtur' should still suggest Fixture")
        if "Fixture Example" not in cf.suggest_contacts("fix"):  # prefix
            raise SystemExit("prefix 'fix' should still suggest Fixture")
    finally:
        _reset()


def test_korean_name_match() -> None:
    _mock()
    try:
        if cf.suggest_contacts("가상연락") != ["가상연락처삼"]:
            raise SystemExit(f"Korean partial should match: {cf.suggest_contacts('가상연락')}")
    finally:
        _reset()


def test_unrelated_query_returns_nothing() -> None:
    _mock()
    try:
        if cf.suggest_contacts("xyzzyqwerty") != []:
            raise SystemExit("unrelated query must not surface noise candidates")
    finally:
        _reset()


def test_local_path_and_empty_are_refused() -> None:
    _mock()
    try:
        if cf.suggest_contacts("") != [] or cf.suggest_contacts("/\x55sers/example/x") != []:
            raise SystemExit("empty / path-shaped queries must return nothing")
    finally:
        _reset()


def test_missing_contacts_is_silent_empty() -> None:
    def boom():
        raise RuntimeError("Contacts permission denied")

    cf._run_all_contacts_query = boom  # type: ignore
    try:
        if cf.suggest_contacts("fixture") != []:
            raise SystemExit("missing Contacts permission should yield no suggestions, not raise")
    finally:
        _reset()


def test_suggestion_clause_phrasing() -> None:
    _mock("Fixture Example|||Synthetic Teston|||")
    try:
        one = cf.suggestion_clause("testy")
        if "Did you mean Synthetic Teston?" not in one or "full name" not in one:
            raise SystemExit(f"single-candidate clause wrong: {one!r}")
        _mock()  # back to multi-Fixture book
        many = cf.suggestion_clause("fixture")
        if "Did you mean" not in many or " or " not in many:
            raise SystemExit(f"multi-candidate clause wrong: {many!r}")
        if cf.suggestion_clause("xyzzyqwerty") != "":
            raise SystemExit("no-candidate clause should be empty")
    finally:
        _reset()


def test_send_recipient_resolution_matches_channel_type() -> None:
    # Handle-based vs name-based channels resolve an unknown name differently:
    #   iMessage needs a phone/email HANDLE, so a fuzzy near-miss must NOT become a
    #   send target — it refuses (handle=None) and only *suggests* the closest names.
    #   Kakao searches its own friend list by NAME, so an unknown name falls through
    #   with the raw name (the approval gate is the safety net) — no refusal.
    from jarvis_v2.tools import contacts_connector, kakao_connector, imessage_connector

    _mock("Fixture Example|||Fixturette Kim|||")
    original_resolve = contacts_connector.resolve_contact
    try:
        contacts_connector.resolve_contact = lambda _query: []  # type: ignore

        # iMessage (handle-based): refuse + suggest, never resolve a handle.
        handle, meta, message = imessage_connector._resolve_send_recipient("fixture")
        if handle is not None:
            raise SystemExit(f"imessage: fuzzy must NOT resolve a handle: {handle!r}")
        if meta.get("contact_resolution_status") != "not_found":
            raise SystemExit(f"imessage: status should stay not_found: {meta}")
        if "Did you mean" not in message:
            raise SystemExit(f"imessage: refusal should include a suggestion: {message!r}")

        # Kakao (name-based): pass the raw name through, no refusal.
        k_handle, k_meta, k_message = kakao_connector._resolve_send_recipient("fixture")
        if k_handle != "fixture":
            raise SystemExit(f"kakao: name-based channel should pass the raw name through: {k_handle!r}")
        if k_meta.get("contact_resolution_status") != "unverified_name":
            raise SystemExit(f"kakao: status should be unverified_name: {k_meta}")
        if k_message:
            raise SystemExit(f"kakao: name fall-through should not refuse: {k_message!r}")
    finally:
        contacts_connector.resolve_contact = original_resolve
        _reset()


def main() -> None:
    test_first_name_matches_full_contacts()
    test_typo_and_prefix_match()
    test_korean_name_match()
    test_unrelated_query_returns_nothing()
    test_local_path_and_empty_are_refused()
    test_missing_contacts_is_silent_empty()
    test_suggestion_clause_phrasing()
    test_send_recipient_resolution_matches_channel_type()
    print("Contacts fuzzy smoke passed")


if __name__ == "__main__":
    main()
