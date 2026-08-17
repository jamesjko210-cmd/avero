"""Shared macOS Contacts resolver for Jarvis V2.

`resolve_contact(query)` looks a partial name up against the macOS Contacts /
address book (via `osascript`) and returns structured matches so the messaging
send tools can turn "fixture" into the right "Fixture Example" handle instead of
matching literally.

Behaviour:
1. exactly one match  -> a single-element list,
2. multiple matches   -> the full list (caller asks "Fixture Example or Fixture Kim?"),
3. no match           -> an empty list.

The lookup is read-only and results are cached in-process. The first call
triggers the macOS Contacts permission prompt for the Jarvis Python process; if
permission is missing the lookup returns no matches rather than raising.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    PERSONAL_READ_RECOVERY_ACTION,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig


MAX_QUERY_CHARS = 80
MAX_MATCHES = 25
MAX_HANDLES_PER_KIND = 8
_FIELD_SEP = ":::"
_ENTRY_SEP = "|"
_LABEL_SEP = "^"
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^[+]?[\d\s().\-]{7,}$")

# In-process cache keyed by the normalized query.
_CACHE: dict[str, list["ContactMatch"]] = {}


@dataclass(frozen=True)
class ContactMatch:
    name: str
    phone: str = ""
    email: str = ""
    # Full handle lists from live Contacts. `phones` holds (label, value) pairs so a
    # mobile number can be preferred over home/work. Empty when a match is built by
    # hand (e.g. tests) — `phone`/`email` then act as the single best handle.
    phones: tuple[tuple[str, str], ...] = ()
    emails: tuple[str, ...] = ()

    def all_phones(self) -> list[str]:
        if self.phones:
            return [value for _label, value in self.phones if value]
        return [self.phone] if self.phone else []

    def all_emails(self) -> list[str]:
        if self.emails:
            return [value for value in self.emails if value]
        return [self.email] if self.email else []

    @property
    def handle(self) -> str:
        """Best handle for messaging: a mobile phone first, then any phone, then email."""
        for label, value in self.phones:
            low = label.lower()
            if value and ("mobile" in low or "iphone" in low or "cell" in low):
                return value
        if self.phone:
            return self.phone
        phones = self.all_phones()
        if phones:
            return phones[0]
        if self.email:
            return self.email
        emails = self.all_emails()
        return emails[0] if emails else ""


def clear_contact_cache() -> None:
    _CACHE.clear()


def looks_like_handle(value: Any) -> bool:
    """True when the value is already a phone number or email (skip resolution)."""
    return looks_like_email_handle(value) or looks_like_phone_handle(value)


def looks_like_email_handle(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and _EMAIL_RE.match(text))


def looks_like_phone_handle(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(
        text
        and _PHONE_RE.match(text)
        and sum(ch.isdigit() for ch in text) >= 7
    )


def _normalize_query(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:MAX_QUERY_CHARS]


def _display(value: Any, limit: int = MAX_QUERY_CHARS) -> str:
    text = " ".join(str(value or "").strip().split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    return text[:limit] if len(text) > limit else text


def _looks_like_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _contacts_script(query: str) -> str:
    safe = query.replace("\\", "\\\\").replace('"', '\\"')
    return f'''tell application "Contacts"
    set theMatches to (people whose name contains "{safe}")
    set theOutput to ""
    repeat with aPerson in theMatches
        set theName to ""
        try
            set theName to name of aPerson
        end try
        set thePhones to ""
        try
            repeat with aPhone in phones of aPerson
                set thisLabel to ""
                try
                    set thisLabel to (label of aPhone) as text
                end try
                set thisValue to ""
                try
                    set thisValue to (value of aPhone) as text
                end try
                if thisValue is not "" then
                    if thePhones is not "" then set thePhones to thePhones & "{_ENTRY_SEP}"
                    set thePhones to thePhones & thisLabel & "{_LABEL_SEP}" & thisValue
                end if
            end repeat
        end try
        set theEmails to ""
        try
            repeat with anEmail in emails of aPerson
                set thisEmail to ""
                try
                    set thisEmail to (value of anEmail) as text
                end try
                if thisEmail is not "" then
                    if theEmails is not "" then set theEmails to theEmails & "{_ENTRY_SEP}"
                    set theEmails to theEmails & thisEmail
                end if
            end repeat
        end try
        set theOutput to theOutput & theName & "{_FIELD_SEP}" & thePhones & "{_FIELD_SEP}" & theEmails & linefeed
    end repeat
    return theOutput
end tell'''


def _contacts_count_script() -> str:
    """Return a content-free Contacts availability probe."""
    return 'tell application "Contacts" to return count of people'


def _osascript_contacts(query: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["osascript", "-e", _contacts_script(query)],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _run_contacts_query(query: str) -> str:
    """Run the Contacts lookup and return raw stdout. Mocked in smoke tests.

    Contacts.app must be running for AppleScript to query it; a query against a
    closed app fails with error -600 ("Application isn't running"). If that
    happens we background-launch Contacts (no focus steal) and retry once.
    """
    result = _osascript_contacts(query)
    if result.returncode != 0 and "-600" in (result.stderr or ""):
        try:
            subprocess.run(["open", "-ga", "Contacts"], capture_output=True, timeout=10)
            time.sleep(1.0)
        except Exception:
            pass
        result = _osascript_contacts(query)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "osascript failed")
    return result.stdout


def _run_contacts_count() -> int:
    """Return the saved-contact count without reading contact details."""
    result = subprocess.run(
        ["osascript", "-e", _contacts_count_script()],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "osascript failed")
    try:
        count = int(result.stdout.strip())
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Contacts count was invalid") from exc
    if count < 0:
        raise RuntimeError("Contacts count was invalid")
    return count


def contacts_store_status() -> str:
    """Return a content-free Contacts-store state for opt-in diagnostics."""
    try:
        return "empty" if _run_contacts_count() == 0 else "available"
    except Exception:
        return "unavailable"


def _clean_label(label: str) -> str:
    """Turn a macOS internal phone label (e.g. _$!<Mobile>!$_) into a clean word."""
    text = (label or "").strip()
    match = re.search(r"<([^>]+)>", text)
    if match:
        text = match.group(1)
    text = text.strip("_$!<>").strip()
    return text


def _parse_phones(raw: str) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()
    phones: list[tuple[str, str]] = []
    for entry in raw.split(_ENTRY_SEP):
        entry = entry.strip()
        if not entry:
            continue
        if _LABEL_SEP in entry:
            label, _, value = entry.partition(_LABEL_SEP)
        else:
            label, value = "", entry
        value = value.strip()
        if value:
            phones.append((_clean_label(label), value))
        if len(phones) >= MAX_HANDLES_PER_KIND:
            break
    return tuple(phones)


def _parse_emails(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    emails: list[str] = []
    for entry in raw.split(_ENTRY_SEP):
        entry = entry.strip()
        if entry:
            emails.append(entry)
        if len(emails) >= MAX_HANDLES_PER_KIND:
            break
    return tuple(emails)


def _parse_matches(raw: str, query: str) -> list[ContactMatch]:
    matches: list[ContactMatch] = []
    low_query = query.lower()
    for line in (raw or "").splitlines():
        if not line.strip():
            continue
        parts = line.split(_FIELD_SEP)
        name = parts[0].strip() if len(parts) > 0 else ""
        phones = _parse_phones(parts[1].strip() if len(parts) > 1 else "")
        emails = _parse_emails(parts[2].strip() if len(parts) > 2 else "")
        if not name:
            continue
        # Defensive: only keep rows whose name actually contains the query
        # (the AppleScript filter already does this, but guards mocked input).
        if low_query and low_query not in name.lower():
            continue
        # Pin the single best phone/email so .phone/.email stay populated for
        # callers and metadata that read them directly. Prefer a mobile phone.
        best_phone = ""
        for label, value in phones:
            low = label.lower()
            if "mobile" in low or "iphone" in low or "cell" in low:
                best_phone = value
                break
        if not best_phone and phones:
            best_phone = phones[0][1]
        match = ContactMatch(
            name=name,
            phone=best_phone,
            email=emails[0] if emails else "",
            phones=phones,
            emails=emails,
        )
        matches.append(match)
        if len(matches) >= MAX_MATCHES:
            break
    return matches


def _match_rank(name: str, low_query: str) -> int:
    """Lower is a better match: exact name < starts-with < whole-word < substring."""
    low_name = name.lower()
    if low_name == low_query:
        return 0
    if low_name.startswith(low_query):
        return 1
    if re.search(r"(?:^|\s)" + re.escape(low_query), low_name):
        return 2
    return 3


def _rank_and_dedupe(matches: list[ContactMatch], query: str) -> list[ContactMatch]:
    """Order matches by relevance and drop exact-duplicate contacts.

    Sorting is stable, so contacts with the same rank keep their Contacts order.
    Duplicates are dropped only when name + all phones + all emails are identical
    (genuinely different numbers under the same name are kept for disambiguation).
    """
    low_query = query.lower()
    ranked = sorted(matches, key=lambda m: _match_rank(m.name, low_query))
    deduped: list[ContactMatch] = []
    seen: set[tuple] = set()
    for match in ranked:
        signature = (
            match.name.lower(),
            tuple(match.all_phones()),
            tuple(e.lower() for e in match.all_emails()),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(match)
    return deduped


def _contacts_recovery_message(query: str) -> str:
    suffix = f' for "{_display(query)}"' if query else ""
    return (
        f"I couldn't check macOS Contacts{suffix}. Open Contacts once, allow Contacts access for Jarvis/Python "
        "in macOS System Settings > Privacy & Security > Contacts, then retry `find contact <name>`. "
        f"{PERSONAL_READ_RECOVERY_ACTION}"
    )


def _resolve_contact_with_diagnostic(query: Any) -> tuple[list[ContactMatch], str]:
    """Resolve Contacts and return (matches, bounded exception_type).

    The public resolver intentionally hides failures as [] so send tools fail
    closed as "not found"; the direct find_contact tool uses the diagnostic to
    show the operator a recovery path when Contacts is unavailable.
    """
    normalized = _normalize_query(query)
    if not normalized or _looks_like_local_path(normalized):
        return [], ""
    key = normalized.lower()
    if key in _CACHE:
        return _CACHE[key], ""
    try:
        raw = _run_contacts_query(normalized)
    except Exception as e:
        # Missing permission / Contacts unavailable -> treat as no match.
        # Do not cache failures so a later permitted call can succeed.
        return [], type(e).__name__
    matches = _rank_and_dedupe(_parse_matches(raw, normalized), normalized)
    _CACHE[key] = matches
    return matches, ""


def resolve_contact(query: Any) -> list[ContactMatch]:
    """Resolve a partial name to structured Contacts matches (cached, read-only).

    Returns [] on no match or when Contacts is unavailable / not permitted.
    """
    matches, _exception_type = _resolve_contact_with_diagnostic(query)
    return matches


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": False,
        "calls_external_services": False,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_personal_data": True,
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
    base.update(extra)
    return base


def _contact_boundaries(*, reads_personal_data: bool = True) -> dict[str, bool]:
    return {
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


def _contact_rows(matches: list[ContactMatch]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in matches:
        phones = match.all_phones()
        emails = match.all_emails()
        rows.append(
            {
                "name": _display(match.name),
                "has_phone": bool(phones),
                "has_email": bool(emails),
                "phone_count": len(phones),
                "email_count": len(emails),
                "handle_type": "phone" if phones else ("email" if emails else "none"),
            }
        )
    return rows


def _contact_lookup_handoff(
    *,
    query: str,
    status: str,
    reason: str = "",
    matches: list[ContactMatch] | None = None,
    reads_personal_data: bool = True,
    exception_type: str = "",
) -> dict[str, Any]:
    rows = _contact_rows(matches or [])
    boundaries = _contact_boundaries(reads_personal_data=reads_personal_data)
    next_safe_command = "find contact <name>"
    handoff = {
        "source": "find_contact",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "content_in_handoff": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "query": _display(query),
        "match_count": len(rows),
        "matches": rows,
        "content_in_metadata": False,
        "exception_type": exception_type,
        "retry_safe": status != "ok",
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "contact_lookup_handoff_ready": True,
        "contact_lookup_ready_for_operator": handoff["ready_for_operator"],
        "contact_lookup_state_changed": handoff["state_changed"],
        "contact_lookup_changed": handoff["changed"],
        "contact_lookup_content_in_handoff": handoff["content_in_handoff"],
        "contact_lookup_content_in_metadata": handoff["content_in_metadata"],
        "contact_lookup_next_safe_command": handoff["next_safe_command"],
        "contact_lookup_next_safe_commands": handoff["next_safe_commands"],
        "contact_lookup_next_safe_command_count": handoff["next_safe_command_count"],
        "contact_lookup_authorizes_execution": handoff["authorizes_execution"],
        "contact_lookup_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "contact_lookup_approval_granted": handoff["approval_granted"],
        "contact_lookup_exception_type": handoff["exception_type"],
        "contact_lookup_boundaries": boundaries,
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "content_in_handoff": handoff["content_in_handoff"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        "contact_lookup_handoff": handoff,
    }


def make_contacts_tools(config: JarvisConfig):
    def find_contact(args: dict[str, Any]) -> ToolResult:
        raw_query = args.get("query") or args.get("name") or args.get("contact")
        query = _normalize_query(raw_query)
        if not query:
            return ToolResult(
                "find_contact",
                False,
                "Who should I look up? Give me a name.",
                _safe_metadata(
                    reads_personal_data=False,
                    reason="missing_query",
                    query="",
                    match_count=0,
                    **_contact_lookup_handoff(
                        query="",
                        status="refused",
                        reason="missing_query",
                        reads_personal_data=False,
                    ),
                ),
            )
        if _looks_like_local_path(query):
            return ToolResult(
                "find_contact",
                False,
                "Please give me a contact name, not a local file path.",
                _safe_metadata(
                    reads_personal_data=False,
                    reason="invalid_query",
                    local_path_query=True,
                    query=_display(query),
                    match_count=0,
                    **_contact_lookup_handoff(
                        query=query,
                        status="refused",
                        reason="invalid_query",
                        reads_personal_data=False,
                    ),
                ),
            )
        matches, exception_type = _resolve_contact_with_diagnostic(query)
        if exception_type:
            failure_output = _contacts_recovery_message(query)
            return ToolResult(
                "find_contact",
                False,
                failure_output,
                declare_retryable_personal_read_failure(
                    _safe_metadata(
                        query=_display(query),
                        match_count=0,
                        reason="contacts_unavailable",
                        exception_type=exception_type,
                        **_contact_lookup_handoff(
                            query=query,
                            status="unavailable",
                            reason="contacts_unavailable",
                            exception_type=exception_type,
                        ),
                    ),
                    output=failure_output,
                    action=PERSONAL_READ_RECOVERY_ACTION,
                    commands=("setup check",),
                ),
            )
        if not matches:
            return ToolResult(
                "find_contact",
                True,
                f"I couldn't find a contact matching \"{_display(query)}\".",
                _safe_metadata(
                    query=_display(query),
                    match_count=0,
                    **_contact_lookup_handoff(query=query, status="empty", reason="no_match"),
                ),
            )
        lines = [f"Found {len(matches)} contact{'s' if len(matches) != 1 else ''} for \"{_display(query)}\":"]
        for match in matches:
            details = []
            for label, value in match.phones:
                tag = f" ({label})" if label else ""
                details.append(f"phone{tag} {_display(value)}")
            if not match.phones and match.phone:
                details.append(f"phone {_display(match.phone)}")
            for value in match.all_emails():
                details.append(f"email {_display(value)}")
            suffix = f" — {', '.join(details)}" if details else " — no phone or email on file"
            lines.append(f"• {_display(match.name)}{suffix}")
        return ToolResult(
            "find_contact",
            True,
            "\n".join(lines),
            _safe_metadata(
                query=_display(query),
                match_count=len(matches),
                **_contact_lookup_handoff(query=query, status="ok", matches=matches),
            ),
        )

    from jarvis_v2.tools.registry import Tool

    return [
        Tool(
            "find_contact",
            "Look up a person in macOS Contacts by partial name and show their phone/email. "
            "Args: query. Read-only; needs Contacts permission.",
            RiskLevel.LOCAL_SAFE,
            find_contact,
            "personal",
        )
    ]
