"""Fuzzy "did you mean?" contact suggestions for near-miss recipient names.

When an exact/substring contact lookup finds nothing (e.g. "fixture" when the
address book has "Fixture Example"), this module suggests the closest real contacts so
Jarvis can ask "Did you mean Fixture Example?" instead of a dead-end refusal.

Critical safety boundary: this is a DISCOVERY aid, never an AUTHORIZATION one. It
only returns *names to show the user* — it never resolves a handle, never sends,
and never feeds a fuzzy guess into the approval gate. The user must re-issue the
command with a real name, which then goes through the normal exact resolver
(`contacts_connector.resolve_contact`) and approval flow. Fuzzy improves what we
suggest; it must never change what we send.

Built as a standalone seam (it does NOT modify `resolve_contact`) so it composes
with the resolve-before-approval work without touching that code path.
`_run_all_contacts_query` is the mockable seam used by the smoke tests.
"""

from __future__ import annotations

import difflib
import re
import subprocess
import time
from typing import Any

# A suggestion must clear this similarity bar (0..1). Generous — these are only
# candidates the user confirms, not anything that gets sent — but high enough that
# unrelated names don't show up as noise.
_MIN_SCORE = 0.6
_MAX_SUGGESTIONS = 3
_MAX_QUERY_CHARS = 80
_RECORD_SEP = "|||"
_TOKEN_RE = re.compile(r"[^\s]+")
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)

_ALL_CONTACTS_SCRIPT = (
    'tell application "Contacts"\n'
    "    set theNames to \"\"\n"
    "    repeat with aPerson in people\n"
    "        set thisName to \"\"\n"
    "        try\n"
    "            set thisName to (name of aPerson) as text\n"
    "        end try\n"
    f'        if thisName is not "" then set theNames to theNames & thisName & "{_RECORD_SEP}"\n'
    "    end repeat\n"
    "    return theNames\n"
    "end tell"
)

# Test seam: set to a callable()->str to bypass real AppleScript in smoke tests.
_run_all_contacts_query = None  # type: ignore[assignment]


def _osascript_all_contacts() -> subprocess.CompletedProcess:
    return subprocess.run(
        ["osascript", "-e", _ALL_CONTACTS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=20,
    )


def _real_all_contacts_query() -> str:
    """Return raw newline/sep-joined contact names. Background-launch Contacts on -600."""
    result = _osascript_all_contacts()
    if result.returncode != 0 and "-600" in (result.stderr or ""):
        try:
            subprocess.run(["open", "-ga", "Contacts"], capture_output=True, timeout=10)
            time.sleep(1.0)
        except Exception:
            pass
        result = _osascript_all_contacts()
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "osascript failed")
    return result.stdout


def _all_contact_names() -> list[str]:
    runner = _run_all_contacts_query or _real_all_contacts_query
    try:
        raw = runner()
    except Exception:
        # Missing permission / Contacts unavailable -> no suggestions (silent).
        return []
    names: list[str] = []
    for chunk in (raw or "").split(_RECORD_SEP):
        name = chunk.strip()
        if name:
            names.append(name)
    return names


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())[:_MAX_QUERY_CHARS]


def _looks_like_local_path(value: str) -> bool:
    return bool(LOCAL_PATH_RE.search(value)) or value.startswith("/") or "\\" in value


def _score(query_norm: str, name: str) -> float:
    """Best similarity of the query against the full name or any of its tokens.

    Token scoring lets a single first name ("fixture") match a full contact
    ("Fixture Example") at ratio 1.0 rather than being penalised by the surname.
    """
    name_norm = name.lower()
    best = difflib.SequenceMatcher(None, query_norm, name_norm).ratio()
    for token in _TOKEN_RE.findall(name_norm):
        best = max(best, difflib.SequenceMatcher(None, query_norm, token).ratio())
        # A query that is a clean prefix of a name token ("fix" -> "fixture") is a
        # strong signal that ratio alone underweights.
        if token.startswith(query_norm) and len(query_norm) >= 3:
            best = max(best, 0.9)
    return best


def suggest_contacts(query: Any, *, limit: int = _MAX_SUGGESTIONS) -> list[str]:
    """Return up to `limit` real contact names closest to `query` (names only).

    Empty when the query is unusable, Contacts is unavailable, or nothing clears
    the similarity bar. Never returns handles and never triggers a send.
    """
    query_norm = _normalize(query)
    if not query_norm or _looks_like_local_path(query_norm):
        return []
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for name in _all_contact_names():
        if name in seen:
            continue
        seen.add(name)
        score = _score(query_norm, name)
        if score >= _MIN_SCORE:
            scored.append((score, name))
    scored.sort(key=lambda pair: (-pair[0], pair[1].lower()))
    return [name for _, name in scored[: max(1, limit)]]


def suggestion_clause(query: Any, *, limit: int = _MAX_SUGGESTIONS) -> str:
    """A ready-to-append 'Did you mean ...?' clause, or '' if there are no candidates."""
    names = suggest_contacts(query, limit=limit)
    if not names:
        return ""
    if len(names) == 1:
        return f" Did you mean {names[0]}? Reply with the full name."
    listed = ", ".join(names[:-1]) + f", or {names[-1]}"
    return f" Did you mean {listed}? Reply with the full name."
