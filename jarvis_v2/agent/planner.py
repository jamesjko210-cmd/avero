from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

from jarvis_v2.agent.types import Plan, PlannedAction


# Regex-cache latency fix (Fable checkpoint plan item C1, 2026-07-10). This file
# issues ~590-710 DISTINCT inline regex patterns (`re.search(r"...", low)` etc.)
# per fall-through `plan()` call. CPython's `re` module caches compiled patterns
# in a dict capped at `re._MAXCACHE` (default 512). Because the planner's working
# set exceeds that cap, every chat-routed turn -- the ones that fall through the
# whole matcher chain -- evicts and RECOMPILES patterns, measured at 60-123ms per
# call versus ~1ms once the whole set stays resident (a 100x+ planning-latency
# tax paid on top of model generation on every conversational turn, directly
# relevant to the open WS4/DoD-A latency item). Raising the process-wide cache
# ceiling so the entire planner pattern set coexists is the full fix with zero
# call-site edits and zero routing-behavior change (a larger compiled-pattern
# cache only ever changes speed/memory, never match results) -- notably it also
# speeds up the frozen send/call block's patterns without editing one line of it.
# `re._MAXCACHE` is a CPython internal; the guard degrades to a harmless no-op on
# any future runtime that renames or removes it, reverting to prior behavior
# rather than crashing. Kept generously above the observed ~714 working set so
# headroom remains as patterns are added.
try:  # pragma: no cover - trivial, exercised indirectly by the latency guard test
    if getattr(re, "_MAXCACHE", 0) < 2048:
        re._MAXCACHE = 2048
except Exception:
    pass


TIME_DIFFERENCE_LOCATIONS = (
    "san francisco",
    "south korea",
    "new york",
    "los angeles",
    "hong kong",
    "singapore",
    "melbourne",
    "london",
    "seoul",
    "korea",
    "tokyo",
    "japan",
    "nyc",
    "la",
    "paris",
    "berlin",
    "sydney",
    "utc",
)
WEEKDAY_WORDS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
MONTH_WORD_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
TASK_DUE_QUERY_PATTERN = (
    rf"(?:today|tomorrow|this week|next week|this weekend|next weekend|"
    rf"(?:(?:this|next)\s+)?(?:{WEEKDAY_WORDS})|"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}[-/]\d{1,2}(?:[-/]\d{4})?|"
    rf"(?:{MONTH_WORD_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTH_WORD_PATTERN})(?:,?\s+\d{{4}})?)"
)
RELATIVE_DATE_NUMBER_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty"
RELATIVE_DATE_QUANTITY_PATTERN = rf"(?:(?:in\s+)?(?:\d{{1,3}}|{RELATIVE_DATE_NUMBER_WORDS})\s+(?:days?|weeks?)(?:\s+from\s+now)?|(?:\d{{1,3}}|{RELATIVE_DATE_NUMBER_WORDS})\s+(?:days?|weeks?)\s+ago|next week|last week|a week from now|a week ago|one week from now|one week ago)"
RELATIVE_DATE_TARGET_PATTERN = rf"(?:the\s+)?(?:day after tomorrow|day before yesterday|tomorrow|yesterday|(?:next|this|last)\s+(?:{WEEKDAY_WORDS})|{RELATIVE_DATE_QUANTITY_PATTERN})"
SPOKEN_INTEGER_TOKEN_PATTERN = "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|and|a"
SPOKEN_INTEGER_PATTERN = rf"(?:-?\d{{1,7}}|(?:(?:minus|negative)\s+)?(?:{SPOKEN_INTEGER_TOKEN_PATTERN})(?:[-\s]+(?:{SPOKEN_INTEGER_TOKEN_PATTERN}))*)"
LOCAL_TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".css",
    ".html",
    ".csv",
}
DEFAULT_HARNESS_PACKET_REQUEST = "what should Jarvis do next"
HARNESS_PACKET_DISPLAY_PREFIX = r"(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?)?"
POLITE_COMMAND_RETRY_TOOLS = {
    "after_action_learning_packet",
    "agi_gate_report",
    "agi_next_build_move",
    "architecture_map",
    "approval_chain_proof",
    "approval_execution_packet",
    "approval_history",
    "approval_queue_summary",
    "approval_readiness_packet",
    "brain_loop_report",
    "build_delta_report",
    "build_progress_report",
    "build_target_packet",
    "capability_map",
    "chat_continuity_brief",
    "chat_context",
    "chat_prompt_preview",
    "chat_response_health",
    "chat_safety_report",
    "completion_audit_packet",
    "completion_claim_gate",
    "completion_next_proof_packet",
    "completion_proof_refresh_packet",
    "command_cockpit_packet",
    "continuation_packet",
    "evidence_ledger",
    "execution_audit_gate",
    "execution_health_report",
    "execution_learning_closure_packet",
    "execution_recovery_packet",
    "execution_governor_packet",
    "focus_brief",
    "handoff_brief",
    "harness_completion_assessment",
    "harness_doctrine",
    "harness_operations_brief",
    "harness_readiness_digest",
    "harness_status",
    "inspect_pending_approval",
    "jarvis_doctor",
    "jarvis_help",
    "jarvis_status",
    "list_pending_approvals",
    "list_sessions",
    "model_routing_status",
    "next_action_packet",
    "privacy_report",
    "priority_stack",
    "priority_goal",
    "prototype_readiness_checklist",
    "recent_conversation",
    "recent_tool_runs",
    "recovery_closure_checklist",
    "readiness_report",
    "return_brief",
    "roadmap_report",
    "review_pending_approvals",
    "risk_matrix",
    "risk_preflight",
    "runtime_trace_receipt",
    "safe_next_actions",
    "learning_review",
    "queue_learning_tasks",
    "save_learning_review",
    "safety_status",
    "session_closeout",
    "session_learning_preview",
    "setup_check",
    "storage_status",
    "verification_receipt",
    "work_block_checkpoint",
    "work_session_packet",
    "work_queue",
}
LEADING_POLITE_WRAPPER_RE = re.compile(
    r"^(?:please show me|please show|can you please|could you please|would you please|can you kindly|could you kindly|would you kindly|can you just|could you just|would you just|can you|could you|would you|show me|please|pls|kindly|just|show)\s+",
    re.IGNORECASE,
)
NATURAL_CALC_RE = re.compile(
    r"^(?P<expression>(?:divide|split|multiply)\s+[-+*/().,\d\s%a-zA-Z_]+\s+by\s+[-+*/().,\d\s%a-zA-Z_]+|(?:add|multiply)\s+[-+*/().,\d\s%a-zA-Z_]+\s+and\s+[-+*/().,\d\s%a-zA-Z_]+|[-+*/().,\d\s%a-zA-Z_]+\s+(?:divided by|over|times|plus|minus|multiplied by)\s+[-+*/().,\d\s%a-zA-Z_]+|[-+*/().,\d\s%a-zA-Z_]+\s+(?:percent|per cent)\s+of\s+[-+*/().,\d\s%a-zA-Z_]+|subtract\s+[-+*/().,\d\s%a-zA-Z_]+\s+from\s+[-+*/().,\d\s%a-zA-Z_]+)$",
    re.IGNORECASE,
)


TRAILING_POLITE_WRAPPER_RE = re.compile(
    r"(?:[\s,.;!?]{0,20}(?:please|pls|thanks|thank you)[\s,.;!?]{0,20}){1,10}$",
    re.IGNORECASE,
)


# Fable-authorized entry-point ReDoS guard (2026-07-10 checkpoint). Rounds 43-44
# found a severe, live catastrophic-backtracking vulnerability in the frozen
# send/call routing block (the kakao/telegram/imessage "trailing-service"
# patterns: an unbounded lazy `(?P<to>.+?)` followed by a required literal that,
# when absent, forces the engine to try every split against a long whitespace/
# space-separated run). Confirmed by fresh timing at this checkpoint: the raw
# kakao regex takes 0.12s at 100 internal spaces and TIMES OUT (>4s) at 500;
# the full pipeline still hangs at ~1000 spaces. Sonnet correctly declined to
# edit the frozen block and flagged it for authorization. This is that fix,
# and it is deliberately placed OUTSIDE the frozen block: normalizing the input
# ONCE at plan()'s entry (before any of the ~13,000 lines of matching run) caps
# the worst case for EVERY pattern in the file -- the frozen send/call patterns
# AND the never-individually-audited call/facetime ones -- without touching a
# single line inside the freeze.
#
# Two-part guard, both linear-time and both verified content-safe:
#   1. Collapse any run of 5+ whitespace to a single space. Normal spacing
#      (1-4 chars, incl. typical double/triple spaces and single newlines
#      between lines) is untouched; only genuinely-unusual runs -- the exact
#      thing that triggers the `.+?` vs `\s+` overlap blowup -- are neutralized.
#      This alone fixes the realistic ACCIDENTAL trigger (a paste with a long
#      whitespace run: 50,000 spaces -> 0.05s after this).
#   2. Cap the matching length at 4000 chars. The multi-run variant (a pasted
#      table / column-aligned text with many separate 5+ runs) is reachable by
#      legitimate input, not only a crafted attack, and its cost scales with
#      total length, not just per-run length; the cap bounds the absolute worst
#      case to <1s. This is safe because (a) content-bearing commands match at
#      the FRONT (their verb is in the first ~20 chars), so capping the tail
#      never changes WHICH tool routes -- only the captured content of an inline
#      note/task/memory body beyond 4000 chars is truncated, a rare edge; and
#      (b) crucially, the runtime sends the model the ORIGINAL user message, not
#      this normalized copy (runtime.py handle() -> chat.respond(user_input)),
#      so chat-routed replies keep full fidelity regardless of this cap.
# The guard is idempotent (its own output has no 5+ runs and is <=4000 chars),
# so the recursive politeness/vocative retries re-run it harmlessly.
_PLANNER_INPUT_MAX_MATCH_LEN = 4000
_PATHOLOGICAL_WHITESPACE_RUN_RE = re.compile(r"\s{5,}")
_KNOWLEDGE_PROMOTION_COMMAND_MAX_CHARS = 9000
_KNOWLEDGE_PROMOTION_COMMAND_RE = re.compile(
    r"promote\s+memory\s+#?(?P<memory_id>\d+)\s+revision\s+#?(?P<reviewed_revision>\d+)"
    r"\s+token\s+(?P<review_token>[0-9a-f]{64})\s+to\s+(?:a\s+)?decision\s*:\s*"
    r"(?P<title>[^|\n]+?)\s*\|\s*(?P<rationale>[^|\n]+?)\s*\|\s*"
    r"(?P<impact>[^|\n]+?)\s*",
    re.IGNORECASE,
)
_KNOWLEDGE_PROMOTION_PREFERENCE_COMMAND_RE = re.compile(
    r"promote\s+memory\s+#?(?P<memory_id>\d+)\s+revision\s+#?(?P<reviewed_revision>\d+)"
    r"\s+token\s+(?P<review_token>[0-9a-f]{64})\s+to\s+(?:a\s+)?preference\s*:\s*"
    r"(?P<category>[^|\n]+?)\s*\|\s*(?P<key>[^|\n]+?)\s*\|\s*"
    r"(?P<value>[^|\n]+?)\s*",
    re.IGNORECASE,
)
_KNOWLEDGE_PROMOTION_PROFILE_COMMAND_RE = re.compile(
    r"promote\s+memory\s+#?(?P<memory_id>\d+)\s+revision\s+#?(?P<reviewed_revision>\d+)"
    r"\s+token\s+(?P<review_token>[0-9a-f]{64})\s+to\s+(?:a\s+)?profile\s*:\s*"
    r"(?P<heading>[^|\n]+?)\s*\|\s*(?P<category>[^|\n]+?)\s*\|\s*"
    r"(?P<body>[^|\n]+?)\s*",
    re.IGNORECASE,
)
_KNOWLEDGE_PROMOTION_PREFERENCE_CATEGORY_MAX_CHARS = 64
_KNOWLEDGE_PROMOTION_PREFERENCE_KEY_MAX_CHARS = 120
_KNOWLEDGE_PROMOTION_PREFERENCE_VALUE_MAX_CHARS = 2000
_KNOWLEDGE_PROMOTION_PROFILE_HEADING_MAX_CHARS = 120
_KNOWLEDGE_PROMOTION_PROFILE_CATEGORY_MAX_CHARS = 64
_KNOWLEDGE_PROMOTION_PROFILE_BODY_MAX_CHARS = 50_000
_KNOWLEDGE_PROMOTION_PLACEHOLDER_RE = re.compile(
    r"<\s*(?:explicit\s+)?(?:title|rationale|impact)\s*>", re.IGNORECASE
)
_KNOWLEDGE_PROMOTION_PREFERENCE_PLACEHOLDER_RE = re.compile(
    r"<\s*(?:explicit\s+)?(?:category|key|value)\s*>", re.IGNORECASE
)
_KNOWLEDGE_PROMOTION_PROFILE_PLACEHOLDER_RE = re.compile(
    r"<\s*(?:explicit\s+)?(?:heading|category|body)\s*>", re.IGNORECASE
)
_DECISION_OUTCOME_COMMAND_RE = re.compile(
    r"decision\s+outcome\s+#?(?P<decision_id>\d+)\s*:\s*(?P<summary>.*)",
    re.IGNORECASE | re.DOTALL,
)
_DECISION_OUTCOME_SUMMARY_MAX_CHARS = 4000
_TASK_COMPLETION_COMMAND_MAX_CHARS = 1200
NEGATED_REQUEST_PLAN_NOTE = "Negated/cancelled request routed to chat instead of guessing which action to skip."


def _normalize_planner_input(user_input: str) -> str:
    normalized = _PATHOLOGICAL_WHITESPACE_RUN_RE.sub(" ", str(user_input or ""))
    if len(normalized) > _PLANNER_INPUT_MAX_MATCH_LEN:
        normalized = normalized[:_PLANNER_INPUT_MAX_MATCH_LEN]
    return normalized


def _raw_first_and_then_clause(user_input: str) -> str:
    """Return the first compound clause without rewriting its content.

    Matching still uses the normalized, bounded planner input. This helper is
    only for free-text tool arguments after their route is already known, so a
    pasted table or aligned text is stored with its original whitespace.
    """
    raw = str(user_input or "").strip()
    size = len(raw)
    index = 0
    while index < size:
        if not raw[index].isspace():
            index += 1
            continue
        separator_start = index
        while index < size and raw[index].isspace():
            index += 1
        if raw[index : index + 3].lower() != "and":
            continue
        after_and = index + 3
        if after_and >= size or not raw[after_and].isspace():
            continue
        index = after_and
        while index < size and raw[index].isspace():
            index += 1
        if raw[index : index + 4].lower() != "then":
            continue
        after_then = index + 4
        if after_then >= size or not raw[after_then].isspace():
            continue
        return raw[:separator_start].rstrip()
    return raw


def _raw_free_text_after_prefix(raw_text: str, prefix_pattern: str) -> str:
    """Extract a free-text tail from a bounded command header."""
    match = re.match(prefix_pattern, raw_text[:256], re.IGNORECASE)
    return raw_text[match.end() :].strip() if match is not None else ""


def _bounded_positive_sqlite_id_digits(value: str) -> int | None:
    if type(value) is not str or not (0 < len(value) <= 19):
        return None
    parsed = int(value)
    return parsed if 0 < parsed <= 9223372036854775807 else None


def _task_completion_command_plan(user_input: str) -> Plan | None:
    raw = str(user_input or "").lstrip()
    for _wrapper in range(2):
        vocative = re.match(
            r"(?:(?:hey|hi)[ \t]{1,10})?jarvis[ \t]*[,;:]?[ \t\r\n]*",
            raw,
            re.IGNORECASE,
        )
        if vocative is not None:
            raw = raw[vocative.end() :]
            continue
        polite = LEADING_POLITE_WRAPPER_RE.match(raw)
        if polite is not None:
            raw = raw[polite.end() :]
            continue
        break
    recognized = re.match(
        r"(?:task[ \t]{1,10}completion[ \t]{1,10}packet|"
        r"completion[ \t]{1,10}packet[ \t]{1,10}for[ \t]{1,10}task|"
        r"can[ \t]{1,10}(?:complete|finish)[ \t]{1,10}task|"
        r"(?:complete|finish|done)[ \t]{1,10}task[ \t]{1,10}"
        r"#?\d+[ \t]{1,10}with)\b",
        raw[:160],
        re.IGNORECASE,
    )
    if recognized is None:
        return None
    if len(raw) > _TASK_COMPLETION_COMMAND_MAX_CHARS:
        return Plan(
            goal="Reject an invalid task completion command.",
            actions=[],
            needs_model=False,
            notes="Task completion evidence commands must stay within the bounded input contract.",
        )

    packet_match = re.match(
        r"^(?:task[ \t]{1,10}completion[ \t]{1,10}packet|"
        r"completion[ \t]{1,10}packet[ \t]{1,10}for[ \t]{1,10}task|"
        r"can[ \t]{1,10}(?:complete|finish)[ \t]{1,10}task)"
        r"[ \t]{1,10}#?(?P<task_id>\d+)(?P<suffix>.*)\Z",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if packet_match is not None:
        task_id = _bounded_positive_sqlite_id_digits(packet_match.group("task_id"))
        suffix = packet_match.group("suffix")
        if not suffix or suffix.isspace():
            evidence = ""
        elif suffix.startswith(":"):
            evidence = suffix[1:]
        elif suffix[0].isspace():
            evidence = suffix.lstrip(" \t\r\n")
        else:
            task_id = None
            evidence = ""
        if task_id is not None:
            return Plan(
                goal="Check task completion evidence.",
                actions=[
                    PlannedAction(
                        "task_completion_packet",
                        {"task_id": task_id, "evidence": evidence},
                        "The user asked Jarvis to inspect completion evidence before closing a tracked task.",
                    )
                ],
            )

    completion_match = re.match(
        r"^(?:complete|finish|done)[ \t]{1,10}task[ \t]{1,10}"
        r"#?(?P<task_id>\d+)[ \t]{1,10}with[ \t]{1,10}"
        r"(?P<payload>.*)\Z",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if completion_match is not None:
        task_id = _bounded_positive_sqlite_id_digits(completion_match.group("task_id"))
        payload = completion_match.group("payload")
        verification_run_id = ""
        run_match = re.match(
            r"^(?:verification[ \t]{1,10}run|run)[ \t]{1,10}"
            r"#?(?P<run_id>\d+)(?P<rest>.*)\Z",
            payload,
            re.IGNORECASE | re.DOTALL,
        )
        if run_match is not None:
            verification_run_id = run_match.group("run_id")
            rest = run_match.group("rest")
            if rest.startswith(":"):
                payload = rest[1:]
            elif not rest or rest[0].isspace():
                payload = rest.lstrip(" \t\r\n")
            else:
                task_id = None
        labelled = re.match(
            r"^(?:evidence|proof|receipt)[ \t]*:(?P<evidence>.*)\Z",
            payload,
            re.IGNORECASE | re.DOTALL,
        )
        if labelled is not None:
            evidence = labelled.group("evidence")
        else:
            unlabelled = re.match(
                r"^(?:evidence|proof|receipt)[ \t]{1,10}(?P<evidence>.*)\Z",
                payload,
                re.IGNORECASE | re.DOTALL,
            )
            evidence = unlabelled.group("evidence") if unlabelled else payload
        if task_id is not None:
            return Plan(
                goal="Complete task with evidence.",
                actions=[
                    PlannedAction(
                        "complete_task_with_evidence",
                        {
                            "task_id": task_id,
                            "verification_run_id": verification_run_id,
                            "evidence": evidence,
                        },
                        "The user asked Jarvis to close a task with explicit evidence preserved in the audit log.",
                    )
                ],
            )

    return Plan(
        goal="Reject an invalid task completion command.",
        actions=[],
        needs_model=False,
        notes=(
            "Use `task completion packet #<id>: <evidence>` or "
            "`complete task #<id> with evidence: <proof>`."
        ),
    )


def _decision_outcome_command_plan(user_input: str) -> Plan | None:
    raw = str(user_input or "").strip()
    if re.match(r"^decision\s+outcome\b", raw, re.IGNORECASE) is None:
        return None
    # Parse this command before the planner-wide 4,000-character match cap so
    # an overlong personal-history write cannot be silently truncated into a
    # valid mutation. Preserve the global command-chain behavior by keeping
    # only the first explicit "and then" clause.
    first_clause = re.split(
        r"\s{1,10}and\s{1,10}then\s{1,10}",
        raw,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if len(first_clause) > _DECISION_OUTCOME_SUMMARY_MAX_CHARS + 128:
        match = None
    else:
        match = _DECISION_OUTCOME_COMMAND_RE.fullmatch(first_clause)
    if match is not None:
        decision_id = _bounded_positive_sqlite_id_digits(match.group("decision_id"))
        summary = unicodedata.normalize("NFKC", match.group("summary"))
        summary = summary.replace("\r\n", "\n").replace("\r", "\n").strip()
        if (
            decision_id is not None
            and summary
            and len(summary) <= _DECISION_OUTCOME_SUMMARY_MAX_CHARS
        ):
            return Plan(
                goal="Record a user-reported decision outcome.",
                actions=[
                    PlannedAction(
                        "record_decision_outcome",
                        {"decision_id": decision_id, "summary": summary},
                        "The user explicitly reported an outcome for one durable decision.",
                    )
                ],
            )
    return Plan(
        goal="Reject an invalid decision outcome command.",
        actions=[],
        needs_model=False,
        notes=(
            "Decision outcomes require `decision outcome #<id>: <summary>` with a positive "
            "bounded id and a nonempty summary of at most 4000 characters."
        ),
    )


def _knowledge_promotion_contains_placeholder(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    compact = re.sub(r"\s+", "", normalized)
    return _KNOWLEDGE_PROMOTION_PLACEHOLDER_RE.search(normalized) is not None or re.search(
        r"<(?:explicit)?(?:title|rationale|impact)>", compact
    ) is not None


def _knowledge_promotion_preference_contains_placeholder(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    compact = re.sub(r"\s+", "", normalized)
    return (
        _KNOWLEDGE_PROMOTION_PREFERENCE_PLACEHOLDER_RE.search(normalized) is not None
        or re.search(r"<(?:explicit)?(?:category|key|value)>", compact) is not None
    )


def _knowledge_promotion_profile_contains_placeholder(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    compact = re.sub(r"\s+", "", normalized)
    return (
        _KNOWLEDGE_PROMOTION_PROFILE_PLACEHOLDER_RE.search(normalized) is not None
        or re.search(r"<(?:explicit)?(?:heading|category|body)>", compact) is not None
    )


def _knowledge_promotion_command_plan(user_input: str) -> Plan | None:
    raw = str(user_input or "").strip()
    if re.match(r"^promote\s+memory\b", raw, re.IGNORECASE) is None:
        return None
    memory_match = re.match(
        r"^promote\s+memory\s+#?(?P<memory_id>\d+)", raw, re.IGNORECASE
    )
    memory_digits = memory_match.group("memory_id") if memory_match else ""
    memory_id = _bounded_positive_sqlite_id_digits(memory_digits)
    match = (
        _KNOWLEDGE_PROMOTION_COMMAND_RE.fullmatch(raw)
        if len(raw) <= _KNOWLEDGE_PROMOTION_COMMAND_MAX_CHARS
        else None
    )
    if match is not None:
        reviewed_revision_digits = match.group("reviewed_revision")
        reviewed_revision = _bounded_positive_sqlite_id_digits(reviewed_revision_digits)
        fields = {
            "title": match.group("title").strip(),
            "rationale": match.group("rationale").strip(),
            "impact": match.group("impact").strip(),
        }
        if (
            all(fields.values())
            and len(fields["title"]) <= 240
            and len(fields["rationale"]) <= 4000
            and len(fields["impact"]) <= 4000
            and memory_id is not None
            and memory_id > 0
            and reviewed_revision is not None
            and reviewed_revision > 0
            and not any(_knowledge_promotion_contains_placeholder(value) for value in fields.values())
        ):
            return Plan(
                goal="Promote one reviewed memory to a decision.",
                actions=[
                    PlannedAction(
                        "promote_memory_to_decision",
                        {
                            "memory_id": memory_id,
                            "reviewed_revision": reviewed_revision,
                            "review_token": match.group("review_token"),
                            **fields,
                        },
                        "The user supplied every decision field explicitly and asked for an approval-gated ownership transfer.",
                    )
                ],
            )
    preference_match = (
        _KNOWLEDGE_PROMOTION_PREFERENCE_COMMAND_RE.fullmatch(raw)
        if len(raw) <= _KNOWLEDGE_PROMOTION_COMMAND_MAX_CHARS
        else None
    )
    if preference_match is not None:
        reviewed_revision_digits = preference_match.group("reviewed_revision")
        reviewed_revision = _bounded_positive_sqlite_id_digits(reviewed_revision_digits)
        fields = {
            "category": preference_match.group("category").strip(),
            "key": preference_match.group("key").strip(),
            "value": preference_match.group("value").strip(),
        }
        if (
            all(fields.values())
            and len(fields["category"])
            <= _KNOWLEDGE_PROMOTION_PREFERENCE_CATEGORY_MAX_CHARS
            and len(fields["key"]) <= _KNOWLEDGE_PROMOTION_PREFERENCE_KEY_MAX_CHARS
            and len(fields["value"]) <= _KNOWLEDGE_PROMOTION_PREFERENCE_VALUE_MAX_CHARS
            and memory_id is not None
            and memory_id > 0
            and reviewed_revision is not None
            and reviewed_revision > 0
            and not any(
                _knowledge_promotion_preference_contains_placeholder(value)
                for value in fields.values()
            )
        ):
            return Plan(
                goal="Promote one reviewed memory to a preference.",
                actions=[
                    PlannedAction(
                        "promote_memory_to_preference",
                        {
                            "memory_id": memory_id,
                            "reviewed_revision": reviewed_revision,
                            "review_token": preference_match.group("review_token"),
                            **fields,
                        },
                        "The user supplied every preference field explicitly and asked for an approval-gated ownership transfer.",
                    )
                ],
            )
    profile_match = (
        _KNOWLEDGE_PROMOTION_PROFILE_COMMAND_RE.fullmatch(raw)
        if len(raw) <= _KNOWLEDGE_PROMOTION_COMMAND_MAX_CHARS
        else None
    )
    if profile_match is not None:
        reviewed_revision_digits = profile_match.group("reviewed_revision")
        reviewed_revision = _bounded_positive_sqlite_id_digits(reviewed_revision_digits)
        fields = {
            "heading": profile_match.group("heading").strip(),
            "category": profile_match.group("category").strip(),
            "body": profile_match.group("body").strip(),
        }
        if (
            all(fields.values())
            and len(fields["heading"])
            <= _KNOWLEDGE_PROMOTION_PROFILE_HEADING_MAX_CHARS
            and len(fields["category"])
            <= _KNOWLEDGE_PROMOTION_PROFILE_CATEGORY_MAX_CHARS
            and len(fields["body"]) <= _KNOWLEDGE_PROMOTION_PROFILE_BODY_MAX_CHARS
            and memory_id is not None
            and memory_id > 0
            and reviewed_revision is not None
            and reviewed_revision > 0
            and not any(
                _knowledge_promotion_profile_contains_placeholder(value)
                for value in fields.values()
            )
        ):
            return Plan(
                goal="Promote one reviewed memory to an owned profile note.",
                actions=[
                    PlannedAction(
                        "promote_memory_to_profile",
                        {
                            "memory_id": memory_id,
                            "reviewed_revision": reviewed_revision,
                            "review_token": profile_match.group("review_token"),
                            **fields,
                        },
                        "The user supplied every profile field explicitly and asked for an approval-gated ownership transfer.",
                    )
                ],
            )
    if memory_id is not None and memory_id > 0:
        return Plan(
            goal="Refresh an invalid knowledge promotion command.",
            actions=[
                PlannedAction(
                    "knowledge_promotion_packet",
                    {"memory_id": memory_id},
                    "The promotion command was malformed or exceeded a field limit, so Jarvis will refresh the read-only packet instead of truncating or guessing.",
                )
            ],
        )
    return Plan(
        goal="Reject an invalid knowledge promotion command.",
        actions=[],
        needs_model=False,
        notes="Knowledge promotion requires a positive memory id and the exact packet grammar.",
    )


def _strip_trailing_politeness_suffix_only(value: str) -> str:
    return TRAILING_POLITE_WRAPPER_RE.sub("", str(value or "").strip()).strip()


def _strip_trailing_politeness(value: str) -> str:
    text = str(value or "").strip()
    for _ in range(4):
        stripped = LEADING_POLITE_WRAPPER_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    return _strip_trailing_politeness_suffix_only(text)


def _normalize_due_task_command(value: str) -> str:
    command = (
        unicodedata.normalize("NFKC", str(value or ""))
        .replace("’", "'")
        .replace("‘", "'")
        .strip()
    )
    command = re.sub(r"[\s?!.]+$", "", command).strip()
    command = _strip_trailing_politeness_suffix_only(command)
    return re.sub(r"[\s?!.]+$", "", command).strip()


_REMINDER_CANCELLATION_COMMAND_RE = re.compile(
    r"^(?P<verb>cancel|clear|delete|remove|stop)\b"
    r"(?P<middle>(?:\s+(?:all|my|the|a|an|this|that|every|pending|any|some|one|first|second|last|latest|oldest|no|none|not|never|but|except|other|than|of|out)){0,6})"
    r"\s+(?P<kind>reminders?|timers?|alarms?)\b(?P<tail>.*)\Z",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_reminder_cancellation_command(value: str) -> str:
    command = str(value or "").strip()
    for _ in range(4):
        previous = command
        command = LEADING_POLITE_WRAPPER_RE.sub("", command, count=1).strip()
        command = re.sub(
            r"^(?:(?:hey|hi)\s*[,;:!?]?\s+)?jarvis\s*[,;:!?]?\s*",
            "",
            command,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        if command == previous:
            break
    return _strip_trailing_politeness_suffix_only(command).strip().rstrip(" ?!.")


def _match_reminder_cancellation_command(value: str) -> re.Match[str] | None:
    return _REMINDER_CANCELLATION_COMMAND_RE.fullmatch(
        _normalize_reminder_cancellation_command(value)
    )


def _reminder_cancellation_targets_file(match: re.Match[str]) -> bool:
    tail = match.group("tail").strip().lower()
    if re.search(r"\.[a-z0-9][a-z0-9_-]{0,15}(?:\s|\Z)", tail):
        return True
    if "/" in tail or "\\" in tail:
        return True
    return re.search(
        r"\b(?:file|files|folder|folders|directory|directories|document|documents)\b",
        tail,
    ) is not None


def _is_direct_reminder_cancellation_command(value: str) -> bool:
    match = _match_reminder_cancellation_command(value)
    return match is not None and not _reminder_cancellation_targets_file(match)


def _plan_reminder_cancellation(value: str) -> Plan | None:
    command = _normalize_reminder_cancellation_command(value)
    command_match = _REMINDER_CANCELLATION_COMMAND_RE.fullmatch(command)
    if command_match is not None and _reminder_cancellation_targets_file(command_match):
        return None
    cancellation_mention = re.search(
        r"\b(?:cancel|clear|delete|remove|stop)\b.{0,40}\b(?:reminders?|timers?|alarms?)\b"
        r"|\b(?:reminders?|timers?|alarms?)\b.{0,40}\b(?:cancelled|canceled|cleared|deleted|removed|stopped)\b",
        command,
        re.IGNORECASE | re.DOTALL,
    )
    direct_imperative = re.match(
        r"^(?:cancel|clear|delete|remove|stop)\b",
        command,
        re.IGNORECASE,
    )
    if cancellation_mention is not None and direct_imperative is None:
        return Plan(
            goal="Explain reminder cancellation without changing reminders.",
            actions=[
                PlannedAction(
                    "respond",
                    {
                        "text": (
                            "I didn't change any reminders. Run `list reminders` to see stable ids, then use "
                            "`cancel reminder <id>` for one or `cancel all reminders` for all cancellable reminders."
                        )
                    },
                    "Non-imperative reminder cancellation language must remain read-only.",
                )
            ],
        )
    if re.search(
        r"\b(?:tell\s+me\s+how\s+to|how\s+(?:do|can|should)\s+i|what\s+if\s+i|"
        r"(?:can|could|should|would)\s+i|is\s+it\s+(?:possible|safe|okay|ok)\s+to)\b"
        r".{0,40}\b(?:cancel|clear|delete|remove|stop)\b.{0,20}\b(?:reminders?|timers?|alarms?)\b",
        command,
        re.IGNORECASE,
    ):
        return Plan(
            goal="Explain reminder cancellation without changing reminders.",
            actions=[
                PlannedAction(
                    "respond",
                    {
                        "text": (
                            "Run `list reminders` to see stable ids. Use `cancel reminder <id>` for one, "
                            "or `cancel all reminders` for all reminders that are not already being sent."
                        )
                    },
                    "Instructional and hypothetical language must remain read-only.",
                )
            ],
        )
    match = command_match
    if match is None:
        return None

    middle = match.group("middle").strip().lower()
    kind = match.group("kind").lower()
    tail = match.group("tail").strip().rstrip(" ?!.")
    tail = re.sub(r"\s+(?:right\s+)?now\Z", "", tail, flags=re.IGNORECASE).strip()
    if re.search(r"\b(?:no|none|not|never)\b", middle):
        return Plan(
            goal="Keep reminders unchanged.",
            actions=[],
            needs_model=False,
            notes="The reminder cancellation request explicitly selected no reminders.",
        )
    exception_language = re.search(r"\b(?:but|except|other\s+than)\b", f"{middle} {tail}") is not None
    ambiguous_middle = re.search(r"\b(?:any|some|one|first|second|last|latest|oldest)\b", middle) is not None
    if exception_language or ambiguous_middle:
        return Plan(
            goal="Ask for exact reminder ids before cancelling a subset.",
            actions=[
                PlannedAction(
                    "respond",
                    {
                        "text": (
                            "I won't guess a reminder subset. Run `list reminders`, then say "
                            "`cancel reminder <id>` for one item, or `cancel all reminders`."
                        )
                    },
                    "Exception, ordinal, and subset language must not broaden into cancel-all.",
                )
            ],
        )

    broad_middle = {
        "",
        "my",
        "the",
        "all",
        "all my",
        "all the",
        "all of my",
        "every",
        "pending",
        "my pending",
        "all pending",
        "all my pending",
        "out",
    }
    selective_middle = {"", "a", "an", "my", "the", "this", "that"}
    plural = kind.endswith("s")
    explicit_all = re.search(r"\ball\b", middle) is not None or tail.lower() == "all"
    explicit_every = re.search(r"\bevery\b", middle) is not None
    if (plural or explicit_all or explicit_every) and middle in broad_middle and tail.lower() in {"", "all"}:
        return Plan(
            goal="Cancel reminders.",
            actions=[PlannedAction("cancel_reminders", {}, "User explicitly asked to cancel all reminders.")],
        )

    selector_match = re.fullmatch(
        r"(?:id\s+)?#?(?P<reminder_id>[0-9a-f][0-9a-f-]{7,35})",
        tail,
        re.IGNORECASE,
    )
    if selector_match is not None and middle in selective_middle:
        return Plan(
            goal="Cancel one reminder.",
            actions=[
                PlannedAction(
                    "cancel_reminders",
                    {"reminder_id": selector_match.group("reminder_id")},
                    "User supplied the stable reminder id shown by list reminders.",
                )
            ],
        )

    return Plan(
        goal="Ask for an exact reminder id before cancelling one reminder.",
        actions=[
            PlannedAction(
                "respond",
                {
                    "text": (
                        "I need the reminder id shown by `list reminders`. Say `cancel reminder <id>`, "
                        "or `cancel all reminders`."
                    )
                },
                "A singular or qualified reminder cancellation must not guess which reminder to remove.",
            )
        ],
    )


def _clean_harness_packet_request(value: str | None) -> str:
    request = _strip_trailing_politeness(str(value or "").strip())
    return request or DEFAULT_HARNESS_PACKET_REQUEST


def _clean_completion_objective(value: str | None) -> str:
    objective = _strip_trailing_politeness(str(value or "").strip())
    objective = objective.strip(" \t\r\n:")
    if re.fullmatch(r"[\s,.;:!?]*", objective):
        return ""
    return re.sub(r"[\s.?!]+$", "", objective).strip()


def _clean_execution_prep_request(value: str | None) -> str:
    request = _strip_trailing_politeness(str(value or "").strip())
    request = request.strip(" \t\r\n:")
    request = re.sub(r"^(?:for|about|around|regarding)\s+", "", request, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"[\s,.;:!?]*", request):
        return ""
    return re.sub(r"[\s.?!]+$", "", request).strip()


def _clean_specialist_packet_request(value: str | None) -> str:
    request = str(value or "").strip()
    request = request.strip(" \t\r\n:")
    request = re.sub(r"^(?:for|about|around|regarding)\s+", "", request, flags=re.IGNORECASE).strip()
    politeness_residual = re.sub(r"[\s,.;:!?]+", "", request)
    politeness_residual = re.sub(r"please|pls|thanks|thankyou", "", politeness_residual, flags=re.IGNORECASE)
    if not request or not politeness_residual:
        return DEFAULT_HARNESS_PACKET_REQUEST
    if re.fullmatch(r"[\s,.;:!?]*", request):
        return DEFAULT_HARNESS_PACKET_REQUEST
    return re.sub(r"[\s.?!]+$", "", request).strip() or DEFAULT_HARNESS_PACKET_REQUEST


def _clean_specialist_natural_request(value: str | None) -> str:
    request = _strip_trailing_politeness(str(value or "").strip())
    request = request.strip(" \t\r\n:")
    request = re.sub(r"^(?:for|about|around|regarding)\s+", "", request, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"[\s,.;:!?]*", request):
        return DEFAULT_HARNESS_PACKET_REQUEST
    return re.sub(r"[\s.?!]+$", "", request).strip() or DEFAULT_HARNESS_PACKET_REQUEST


def _clean_execution_case_id(value: str | None) -> str:
    case_id = _strip_trailing_politeness(str(value or "").strip())
    case_id = case_id.strip(" \t\r\n:#")
    case_id = re.sub(r"^(?:the\s+)?case\s+", "", case_id, flags=re.IGNORECASE).strip(" #")
    if case_id.lower() in {"last", "newest"}:
        return "latest"
    if not case_id or case_id.lower() == "latest":
        return "latest"
    return case_id


def _clean_capability_focus(value: str | None) -> str:
    focus = _strip_trailing_politeness(str(value or "").strip())
    focus = re.sub(r"^(?:with|about|for|on|around|regarding|use|using)\s+", "", focus, flags=re.IGNORECASE).strip()
    if focus.lower() in {"latest", "last", "newest", "current", "me", "myself", "the operator"}:
        return ""
    return focus


def _clean_tool_detail_name(value: str | None) -> str:
    name = _strip_trailing_politeness(str(value or "").strip())
    name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"\s+(?:tool|function)$", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"[\s-]+", "_", name.strip(" .?!"))
    return name.lower()


def _looks_like_operator_action_request(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:run command|shell|script|send|email|imessage|kakao|message|text|write|type|paste|click|open|delete|move|copy|file|files|approve|dismiss|calendar|event|remind|reminder|contact|computer|screen|browser|telegram)\b",
            text,
        )
    )


def _clean_operator_action_request(value: str | None) -> str:
    request = _clean_harness_packet_request(value)
    replacements = [
        (r"^running\s+command\b", "run command"),
        (r"^sending\s+(email|imessage|kakao|message|text)\b", r"send \1"),
        (r"^texting\s+", "text "),
        (r"^messaging\s+", "message "),
        (r"^deleting\b", "delete"),
        (r"^moving\b", "move"),
        (r"^copying\b", "copy"),
        (r"^opening\b", "open"),
        (r"^clicking\b", "click"),
        (r"^typing\b", "type"),
        (r"^pasting\b", "paste"),
        (r"^writing\b", "write"),
    ]
    for pattern, replacement in replacements:
        request = re.sub(pattern, replacement, request, count=1, flags=re.IGNORECASE)
    return request.strip()


def _looks_like_tool_detail_reference(value: str | None, *, explicit_tool_word: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if not text or "/" in text or "\\" in text:
        return False
    if _clean_tool_detail_name(text) in {
        "activity",
        "activities",
        "audit",
        "history",
        "log",
        "logs",
        "receipt",
        "receipts",
        "run",
        "runs",
    }:
        return False
    if "_" in text:
        return True
    if explicit_tool_word:
        return True
    return text in {
        "run shell command",
        "send email",
        "send imessage",
        "send kakao",
        "tool search",
        "risk matrix",
        "capability map",
    }


def _looks_like_local_file_reference(value: str) -> bool:
    text = str(value or "").strip()
    if not text or re.search(r"(?:https?://|www\.)", text, re.IGNORECASE):
        return False
    text = text.strip("'\" ")
    if "/" in text or text.startswith(("./", "../", "~")):
        return True
    suffix_match = re.search(r"(\.[A-Za-z0-9]{1,8})(?:$|[?#])", text)
    if not suffix_match:
        return False
    return suffix_match.group(1).lower() in LOCAL_TEXT_FILE_EXTENSIONS


def _clean_file_search_pattern(value: str) -> str:
    pattern = re.sub(r"\s+", " ", str(value or "").strip(" .?!"))
    if re.fullmatch(r"(?:please|pls|thanks|thank you)", pattern, re.IGNORECASE):
        return ""
    pattern = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", pattern, flags=re.IGNORECASE).strip()
    pattern = re.sub(r"^(?:for\s+)?(?:my|the|a|an)\s+", "", pattern, flags=re.IGNORECASE).strip()
    pattern = re.sub(r"^(?:named|called|matching|containing|with|for)\s+", "", pattern, flags=re.IGNORECASE).strip()
    pattern = re.sub(r"\s+(?:file|files|document|documents)$", "", pattern, flags=re.IGNORECASE).strip()
    pattern = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", pattern, flags=re.IGNORECASE).strip()
    return pattern


def _clean_file_arg(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .?!"))
    if re.fullmatch(r"(?:please|pls|thanks|thank you)", text, re.IGNORECASE):
        return ""
    text = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", text, flags=re.IGNORECASE).strip()
    return text


def _parse_small_count(value: str) -> int:
    text = str(value or "").strip().lower()
    if text.isdigit():
        return int(text)
    return {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }.get(text, 0)


def _parse_spoken_integer(value: str) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if re.fullmatch(r"-?\d{1,7}", text):
        return int(text)
    text = re.sub(r"[-\s]+", " ", text).strip()
    sign = 1
    if text.startswith(("minus ", "negative ")):
        sign = -1
        text = re.sub(r"^(?:minus|negative)\s+", "", text, count=1).strip()
    if text == "a hundred":
        return sign * 100
    ones = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
    }
    tens = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    tokens = [token for token in text.split() if token != "and"]
    if not tokens or tokens == ["a"]:
        return None
    total = 0
    current = 0
    used_number = False
    for token in tokens:
        if token == "a":
            value_part = 1
        elif token in ones:
            value_part = ones[token]
        elif token in tens:
            value_part = tens[token]
        elif token == "hundred":
            current = max(current, 1) * 100
            used_number = True
            continue
        else:
            return None
        current += value_part
        used_number = True
    total += current
    if not used_number:
        return None
    return sign * total


def _split_compact_time_difference_locations(value: str) -> tuple[str, str] | None:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = text.strip("?.! ")
    if not text:
        return None
    for left in TIME_DIFFERENCE_LOCATIONS:
        prefix = f"{left} "
        if not text.startswith(prefix):
            continue
        right = text[len(prefix) :].strip()
        if right in TIME_DIFFERENCE_LOCATIONS:
            return left, right
    return None


def _parse_time_difference_request(low_command: str) -> tuple[str, str] | None:
    text = re.sub(r"\s+", " ", str(low_command or "").strip().lower())
    text = text.strip("?.! ")
    patterns = (
        r"^(?:what(?:'s| is)?\s+)?(?:the\s+)?time difference between (?P<source>.+?) and (?P<target>.+)$",
        r"^time difference (?P<source>.+?) (?:and|to|from) (?P<target>.+)$",
        r"^(?P<source>.+?) to (?P<target>.+?) time difference$",
        r"^how many hours (?P<direction>ahead|behind) is (?P<source>.+?) (?:from|of|than) (?P<target>.+)$",
        r"^is (?P<source>.+?) (?P<direction>ahead|behind) of (?P<target>.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group("source").strip(), match.group("target").strip()
    compact = re.search(r"^time difference (?P<locations>.+)$", text)
    if compact:
        return _split_compact_time_difference_locations(compact.group("locations"))
    return None


def _parse_relative_date_request(low_command: str) -> str | None:
    text = re.sub(r"\s+", " ", str(low_command or "").strip().lower())
    text = text.strip("?.! ")
    patterns = (
        rf"^(?:what(?:'s| is)?|tell me)\s+(?:the\s+)?(?:date|day|weekday|day of week|day of the week)\s+(?:is|was|for|on|in)\s+(?P<target>{RELATIVE_DATE_TARGET_PATTERN})$",
        rf"^(?:what(?:'s| is)?|tell me)\s+(?:the\s+)?(?P<target>{RELATIVE_DATE_TARGET_PATTERN})'?s\s+(?:date|day|weekday|day of week|day of the week)$",
        rf"^(?:date|day|weekday|day of week|day of the week)\s+(?:for\s+|on\s+)?(?P<target>{RELATIVE_DATE_TARGET_PATTERN})$",
        rf"^(?P<target>{RELATIVE_DATE_TARGET_PATTERN})\s+(?:date|day|weekday)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group("target").strip()
    return None


def _parse_execution_case_save_spec(spec: str) -> dict[str, str]:
    parts = [part.strip() for part in re.split(r"\s*;\s*", spec) if part.strip()]
    fields = {
        "request": spec.strip(),
        "initial_evidence": "",
        "initial_event_type": "",
        "receipt_kind": "",
        "receipt_id": "",
    }
    request_parts: list[str] = []
    for part in parts:
        match = re.match(
            r"^(request|order|objective|initial evidence|initial_evidence|evidence|verification|verify|tests|test|event type|event_type|receipt kind|receipt_kind|receipt|receipt id|receipt_id|run id|run_id|approval id|approval_id)\s*:?\s*(?P<value>.+)$",
            part,
            re.IGNORECASE,
        )
        if not match:
            request_parts.append(part)
            continue
        key = match.group(1).lower().replace(" ", "_")
        value = match.group("value").strip()
        if key in {"request", "order", "objective"}:
            request_parts.append(value)
        elif key in {"initial_evidence", "evidence", "verification", "verify", "tests", "test"}:
            fields["initial_evidence"] = value
        elif key == "event_type":
            fields["initial_event_type"] = value
        elif key in {"receipt_kind", "receipt"}:
            fields["receipt_kind"] = value
        elif key in {"receipt_id", "run_id", "approval_id"}:
            fields["receipt_id"] = value
    if request_parts:
        fields["request"] = "; ".join(request_parts)
    return {key: value for key, value in fields.items() if value}


def _extract_news_query(low_command: str) -> str:
    command = re.sub(r"\s+", " ", low_command.strip(" ?.!"))
    command = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", command).strip()
    command = re.sub(
        r"^(?:show|list|display|give me|tell me)\s+(?:the\s+)?(?:(?:latest|top|today'?s?)\s+)?",
        "",
        command,
    ).strip()
    command = re.sub(
        r"^what(?:'?s| is| are)?\s+(?:in\s+)?(?:the\s+)?(?:(?:latest|top|today'?s?)\s+)?(?=(?:news|headlines)\b)",
        "",
        command,
    ).strip()
    stop_topics = {
        "current",
        "latest",
        "the latest",
        "top",
        "today",
        "todays",
        "today's",
        "world",
        "the world",
        "please",
        "pls",
        "show",
        "what",
        "what are the",
        "what is the",
    }
    patterns = (
        r"\bnews (?:about|on|regarding|for)\s+(?P<topic>.{2,80}?)(?:[\?\.!]|$)",
        r"\bheadlines (?:about|on|regarding|for)\s+(?P<topic>.{2,80}?)(?:[\?\.!]|$)",
        r"^(?:news|headlines)\s+(?P<topic>[a-z0-9][a-z0-9&+.\-'\s]{1,60}?)(?:[\?\.!]|$)",
        r"^(?:what(?:'?s| is| are)?\s+)?(?:happening|going on)\s+(?:in|with|about)\s+(?P<topic>.{2,80}?)(?:[\?\.!]|$)",
        r"^(?P<topic>[a-z0-9][a-z0-9&+.\-'\s]{1,60}?)\s+(?:latest\s+|top\s+)?(?:news|headlines)(?:\s+(?:today|updates?|latest))?[\?\.!]*$",
    )
    for pattern in patterns:
        match = re.search(pattern, command)
        if not match:
            continue
        topic = re.sub(r"\s+", " ", (match.group("topic") or "").strip(" ?.!"))
        topic = re.sub(r"^(?:the|latest|top|today'?s?)\s+", "", topic).strip()
        topic = re.sub(r"\s+(?:news|headlines|updates?|today|latest|please|pls|thanks|thank you)$", "", topic).strip()
        if topic and topic not in stop_topics:
            return topic
    return ""


def _clean_lookup_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip("?.! ")
    text = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+mean$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(?:the\s+)?word\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(?:for|of)\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", text, flags=re.IGNORECASE).strip()
    return text


def _clean_freeform_query(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip("?.! ")
    text = re.sub(r"\s+(?:please|pls|thanks|thank you)$", "", text, flags=re.IGNORECASE).strip()
    return text


def _parse_scope_packet_spec(spec: str) -> dict[str, str]:
    parts = [part.strip() for part in re.split(r"\s*;\s*", spec) if part.strip()]
    fields = {"action": spec.strip(), "target": "", "time_range": "", "data_level": "", "verification": "", "rollback": "", "tests": "", "audit": "", "acceptance": "", "approval": "", "status": "", "expected_scope_hash": ""}
    action_parts: list[str] = []
    for part in parts:
        match = re.match(r"^(target|source|time|time range|when|date|data|data level|expected_scope_hash|expected scope hash|integration_scope_hash|integration scope hash|scope_hash|scope hash|scope|verification|verify|post-run|post run|rollback|cancel|undo|recovery|tests|test|test plan|smoke tests|audit|audit plan|audit trail|acceptance|acceptance gate|evidence|proof|status|status api|dashboard|api|approval|approval id|approval receipt)\s*:?\s*(?P<value>.+)$", part, re.IGNORECASE)
        if not match:
            action_parts.append(part)
            continue
        key = match.group(1).lower()
        value = match.group("value").strip()
        if key in {"target", "source"}:
            fields["target"] = value
        elif key in {"time", "time range", "when", "date"}:
            fields["time_range"] = value
        elif key in {"verification", "verify", "post-run", "post run"}:
            fields["verification"] = value
        elif key in {"rollback", "cancel", "undo", "recovery"}:
            fields["rollback"] = value
        elif key in {"tests", "test", "test plan", "smoke tests"}:
            fields["tests"] = value
        elif key in {"audit", "audit plan", "audit trail"}:
            fields["audit"] = value
        elif key in {"acceptance", "acceptance gate", "evidence", "proof"}:
            fields["acceptance"] = value
        elif key in {"approval", "approval id", "approval receipt"}:
            fields["approval"] = value
        elif key in {"status", "status api", "dashboard", "api"}:
            fields["status"] = value
        elif key in {"expected_scope_hash", "expected scope hash", "integration_scope_hash", "integration scope hash", "scope_hash", "scope hash"}:
            fields["expected_scope_hash"] = value
        else:
            fields["data_level"] = value
    if action_parts:
        fields["action"] = "; ".join(action_parts)
    return fields


def _parse_voice_packet_spec(spec: str) -> dict[str, str]:
    fields = {
        "transcript": "",
        "confirmed": "",
        "mode": "",
        "privacy_receipt_id": "",
        "receipt_id": "",
        "receipt_nonce": "",
        "verification_receipt_sha256": "",
        "execution_health_sha256": "",
        "execution_audit_sha256": "",
        "after_action_learning_sha256": "",
        "evidence": "",
    }
    parts = [part.strip() for part in re.split(r"\s*;\s*", spec) if part.strip()]
    transcript = parts[0] if parts else spec.strip()
    evidence_parts: list[str] = []
    for part in parts[1:]:
        match = re.match(r"^(confirmed|reviewed|mode|privacy_receipt_id|privacy receipt id|capture_receipt_id|capture receipt id|receipt_id|receipt id|confirmation_receipt_id|confirmation receipt id|receipt_nonce|receipt nonce|freshness_nonce|freshness nonce|verification_receipt_sha256|verification receipt sha256|verification_sha256|verification sha256|execution_health_sha256|execution health sha256|post_health_sha256|post health sha256|health_sha256|health sha256|execution_audit_sha256|execution audit sha256|post_audit_sha256|post audit sha256|audit_sha256|audit sha256|after_action_learning_sha256|after action learning sha256|learning_sha256|learning sha256)\s*[=:]?\s*(?P<value>.+)$", part, re.IGNORECASE)
        if not match:
            evidence_parts.append(part)
            continue
        key = match.group(1).lower().replace(" ", "_")
        value = match.group("value").strip()
        if key in {"confirmed", "reviewed"}:
            fields["confirmed"] = value
        elif key == "mode":
            fields["mode"] = value
        elif key in {"privacy_receipt_id", "capture_receipt_id"}:
            fields["privacy_receipt_id"] = value
        elif key in {"receipt_id", "confirmation_receipt_id"}:
            fields["receipt_id"] = value
        elif key in {"receipt_nonce", "freshness_nonce"}:
            fields["receipt_nonce"] = value
        elif key in {"verification_receipt_sha256", "verification_sha256"}:
            fields["verification_receipt_sha256"] = value
        elif key in {"execution_health_sha256", "post_health_sha256", "health_sha256"}:
            fields["execution_health_sha256"] = value
        elif key in {"execution_audit_sha256", "post_audit_sha256", "audit_sha256"}:
            fields["execution_audit_sha256"] = value
        elif key in {"after_action_learning_sha256", "learning_sha256"}:
            fields["after_action_learning_sha256"] = value

    suffix_patterns = [
        ("after_action_learning_sha256", r"\s+(?:after_action_learning_sha256|after action learning sha256|learning_sha256|learning sha256)\s*[=:]\s*(?P<value>[0-9a-fA-F]{64})\s*$"),
        ("execution_audit_sha256", r"\s+(?:execution_audit_sha256|execution audit sha256|post_audit_sha256|post audit sha256|audit_sha256|audit sha256)\s*[=:]\s*(?P<value>[0-9a-fA-F]{64})\s*$"),
        ("execution_health_sha256", r"\s+(?:execution_health_sha256|execution health sha256|post_health_sha256|post health sha256|health_sha256|health sha256)\s*[=:]\s*(?P<value>[0-9a-fA-F]{64})\s*$"),
        ("verification_receipt_sha256", r"\s+(?:verification_receipt_sha256|verification receipt sha256|verification_sha256|verification sha256)\s*[=:]\s*(?P<value>[0-9a-fA-F]{64})\s*$"),
        ("receipt_nonce", r"\s+(?:receipt_nonce|receipt nonce|freshness_nonce|freshness nonce)\s*[=:]\s*(?P<value>\S+)\s*$"),
        ("receipt_id", r"\s+(?:receipt_id|receipt id|confirmation_receipt_id|confirmation receipt id)\s*[=:]\s*(?P<value>\S+)\s*$"),
        ("privacy_receipt_id", r"\s+(?:privacy_receipt_id|privacy receipt id|capture_receipt_id|capture receipt id)\s*[=:]\s*(?P<value>\S+)\s*$"),
        ("mode", r"\s+mode\s*[=:]\s*(?P<value>\S+)\s*$"),
        ("confirmed", r"\s+(?:confirmed|reviewed)\s*[=:]\s*(?P<value>true|false|yes|no|1|0|confirmed|reviewed)\s*$"),
    ]
    changed = True
    while changed:
        changed = False
        for key, pattern in suffix_patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                fields[key] = fields[key] or match.group("value").strip()
                transcript = transcript[: match.start()].rstrip()
                changed = True

    fields["transcript"] = transcript.strip()
    fields["evidence"] = "; ".join(evidence_parts)
    return fields


def _approval_id_arg(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized in {"latest", "last", "newest", "current"}:
        return "latest"
    return int(normalized)


RISKY_NATURAL_ORDER_HINTS = {
    "computer",
    "screen",
    "click",
    "mouse",
    "keyboard",
    "terminal",
    "shell",
    "python",
    "script",
    "install",
    "write",
    "edit",
    "delete",
    "move",
    "rename",
    "clipboard",
    "email",
    "calendar",
    "text",
    "imessage",
    "message",
    "dm",
    "kakao",
    "kakaotalk",
    "카카오",
    "카카오톡",
    "send",
    "post",
    "purchase",
    "book",
    "pay",
    "share",
}

ACTION_ORDER_STARTS = (
    "use ",
    "run ",
    "execute ",
    "open ",
    "inspect ",
    "check ",
    "look ",
    "click ",
    "type ",
    "write ",
    "edit ",
    "delete ",
    "move ",
    "rename ",
    "send ",
    "text ",
    "imessage ",
    "message ",
    "dm ",
    "kakao ",
    "kakaotalk ",
    "카카오 ",
    "카카오톡 ",
    "email ",
    "book ",
    "purchase ",
    "pay ",
    "install ",
    "make ",
    "create ",
    "organize ",
    "summarize ",
    "find ",
    "search ",
)

QUESTION_STARTS = (
    "can ",
    "could ",
    "would ",
    "should ",
    "what ",
    "why ",
    "how ",
    "when ",
    "where ",
    "who ",
    "is ",
    "are ",
    "do ",
    "does ",
)


def _looks_like_risky_natural_order(low_command: str) -> bool:
    if not low_command or low_command.startswith(QUESTION_STARTS):
        return False
    if not any(hint in low_command for hint in RISKY_NATURAL_ORDER_HINTS):
        return False
    if low_command.startswith(ACTION_ORDER_STARTS):
        return True
    return any(
        phrase in low_command
        for phrase in (
            " and send ",
            " and email ",
            " then send ",
            " then email ",
            " and write ",
            " then write ",
            " my computer to ",
            " the screen ",
        )
    )


_WRITER_COMMAND_RE = re.compile(
    r"^(?:compose(?: and write)?|draft\b.+\blike a human\b|write a paragraph\b|"
    r"autowrite|human[- ]?write|human[- ]?type|write (?:this )?like a human|"
    r"type (?:this |it )?(?:out )?like a human|write humanly|paste|type out|type this|"
    r"insert text|drop text|write block|put text|write text)\s*:?\s*\S",
    re.IGNORECASE | re.DOTALL,
)


def _is_writer_command(text: str) -> bool:
    """Explicit writer commands ('write text: ...', 'human write: ...') own their
    HIGH_RISK writer tools and must not be intercepted by the risky-order dispatch."""
    return bool(_WRITER_COMMAND_RE.match(text.strip()))


def _is_personal_read_query(low_command: str) -> bool:
    """Read-only personal queries (check email, what's on my calendar, read
    messages) must reach their local-safe fast-paths instead of being treated as
    risky natural orders. Kept tight so action/compound orders ("send an email
    and delete the event") still go through the risky dispatch."""
    if re.search(r"\b(search|find)\s+(?:my\s+|recent\s+|new\s+|for\s+(?:recent\s+|new\s+)?)?(e-?mails?|mail|inbox|gmail)\b", low_command):
        return True
    if re.search(r"\b(e-?mails?|mail)\b.{0,20}\b(from|about|containing|with subject|subject)\b", low_command):
        return True
    if re.search(r"\b(read|show|open)\b.{0,30}\b(e-?mail|mail|gmail)\b.{0,20}\b(body|content|message|from|about|subject)\b", low_command):
        return True
    if re.search(r"\b(any|new|unread|check|read|show|latest|got|do i have)\b.{0,20}\b(e-?mails?|mail|inbox|gmail)\b", low_command):
        return True
    if re.search(r"^(?:e-?mails?|mail|inbox|gmail)(?:\s+(?:please|pls|thanks|thank you))?$", low_command):
        return True
    if re.search(r"\bon my calendar\b|\b(events?|agenda|appointments?)\b|\bmy schedule\b", low_command) and not re.search(
        r"\b(create|add|delete|cancel|move|set up|schedule a|book)\b", low_command
    ):
        return True
    if re.search(r"\b(?:check|show|open|see|view)\b.{0,20}\bmy calendar\b", low_command) and not re.search(
        r"\b(create|add|delete|cancel|move|set up|schedule a|book)\b", low_command
    ):
        return True
    if re.search(
        r"\b(?:check|show|open|see|view)\s+calendar\s+(?:today|tomorrow|tonight|this\s+(?:morning|afternoon|evening|night|weekend|week|month|year)|next\s+(?:weekend|week|month|year)|weekend|week|month|year|(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))(?:\s+(?:morning|afternoon|evening|night))?\b",
        low_command,
    ) and not re.search(r"\b(create|add|delete|cancel|move|set up|schedule a|book)\b", low_command):
        return True
    if re.search(
        r"\b(?:check|show|list|open|see|view|what(?:'s| is| are)?|anything|which|pick|prioritize)\b.{0,28}\b(?:tasks?|todos?|to do|to-do|todo list|task list|my list|my plate)\b",
        low_command,
    ) and not re.search(r"\b(create|add|delete|cancel|drop|complete|finish|mark|move|set up|schedule a|book)\b", low_command):
        return True
    if re.search(r"\b(read|show|check|recent|last|any|new)\b.{0,20}\b(imessages?|messages?|texts?|sms)\b", low_command):
        return True
    return False


_REMINDER_DUE_HINT_RE = re.compile(
    r"\bin\s+\d+\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)\b"
    r"|\bin\s+(?:an?|half(?:\s+an?)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)\b"
    r"|\b(?:recurring|repeat(?:ing)?|daily|weekly|monthly|yearly|annually)\b"
    r"|\bevery\s+(?:\d+\s*)?(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weekdays?|weeks?|months?|years?)\b"
    r"|\beach\s+(day|weekday|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b"
    r"|\b(?:at\s+)?(?:noon|midnight)\b"
    r"|\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b\d+\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)\s+timer\b",
    re.IGNORECASE,
)


def _has_reminder_due_hint(low_command: str) -> bool:
    return bool(_REMINDER_DUE_HINT_RE.search(low_command))


def _clean_email_sender_hint(value: str) -> str:
    sender = value.strip(" .")
    sender = re.sub(r"\s+\b(?:today|yesterday|tomorrow|this week|last week)\b\s*$", "", sender, flags=re.IGNORECASE)
    sender = re.sub(r"\s+\b(?:sent|send)\s+me\b\s*$", "", sender, flags=re.IGNORECASE)
    return sender.strip(" .")


def _is_calendar_create_intent(low_command: str) -> bool:
    return bool(re.search(
        r"\b(create|add|set up|book|make|new|put)\b.{0,25}\b(event|meeting|appointment|lunch|dinner|call)\b"
        r"|\bschedule\b.{0,40}\b(at|on|tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d)"
        r"|\b(create|add|set up|book|make|new|put|schedule)\b.{0,50}\b(all[-\s]?day|for the day|whole day)\b",
        low_command,
    ))


def _plan_calendar_delete(text: str, low_command: str) -> Plan | None:
    command = text.strip()
    exact_match = re.fullmatch(
        r"(?:delete|cancel|remove)\s+(?:calendar\s+)?event(?:\s+id)?\s+#?"
        r"(?P<event_id>[a-z0-9][a-z0-9_.@:-]{0,1023})",
        command,
        re.IGNORECASE,
    )
    if exact_match:
        return Plan(
            goal="Delete a calendar event.",
            actions=[
                PlannedAction(
                    "delete_event",
                    {"event_id": exact_match.group("event_id")},
                    "User asked to delete an explicit calendar event id.",
                )
            ],
        )
    if not re.search(r"\b(delete|cancel|remove)\b", low_command) or not re.search(
        r"\b(events?|meetings?|appointments?)\b", low_command
    ):
        return None
    return Plan(
        goal="Ask for exact calendar event id before deleting.",
        actions=[
            PlannedAction(
                "respond",
                {
                    "text": (
                        "I can delete a calendar event once you provide its exact event id. "
                        "Try listing the event, then say something like: 'delete event evt123'."
                    )
                },
                "Fuzzy calendar deletes must not guess which event to remove.",
            )
        ],
    )


def _plan_explicit_calendar_update(text: str, low_command: str) -> Plan | None:
    match = re.search(r"\b(move|reschedule|update|rename|change)\b.{0,30}\bevent(?:\s+id)?\s+#?([a-z0-9_.@:-]+)\b", low_command)
    if not match:
        return None
    event_id = match.group(2)
    args: dict[str, object] = {"event_id": event_id}
    if re.search(r"\b(rename|call|title)\b", low_command):
        title_m = re.search(r"\b(?:to|as|called|titled)\s+(.{2,120})$", text, re.IGNORECASE)
        if title_m:
            args["title"] = title_m.group(1).strip(" .?!")
    else:
        from jarvis_v2.tools.nl_datetime import parse_event
        parsed = parse_event(text)
        if parsed and parsed[0] != "__need_time__":
            _, start, end = parsed
            args["start"] = start
            args["end"] = end
    if len(args) == 1:
        return Plan(
            goal="Ask for the event update details.",
            actions=[PlannedAction(
                "respond",
                {"text": "Which title, date, or time should I apply to that event id?"},
                "Explicit event update did not include a parseable change.",
            )],
        )
    return Plan(
        goal="Update a calendar event.",
        actions=[PlannedAction("update_event", args, "User asked to update an explicit calendar event id.")],
    )


def _plan_fuzzy_calendar_update(low_command: str) -> Plan | None:
    if not re.search(r"\b(move|reschedule|update|rename|change)\b", low_command):
        return None
    if not re.search(
        r"\b(event|meeting|appointment|calendar|schedule)\b|\bmy\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
        low_command,
    ):
        return None
    return Plan(
        goal="Ask for exact calendar event id before editing.",
        actions=[PlannedAction(
            "respond",
            {"text": "I can help move or edit a calendar event, but I need the exact event id first. Try listing the event, then say something like: 'move event evt123 to June 20 at 2pm'."},
            "Fuzzy calendar edits must not guess which event to change.",
        )],
    )


class RuleBasedPlanner:
    """
    Small deterministic planner for V2's first core.

    A model planner can be added behind this later; the important thing is that
    runtime state, permissions, and tool execution are already separate.
    """

    def plan(
        self,
        user_input: str,
        *,
        allow_risky_natural_dispatch: bool = True,
        _vocative_stripped: bool = False,
        _polite_stripped: bool = False,
    ) -> Plan:
        original_user_input = str(user_input or "")
        # Real gap found live 2026-07-10 (rounds 33-35): individually patching
        # every free-text capture group and bare exact-match trigger against
        # compound "X and then Y" sentences does not scale -- this file has
        # hundreds of such patterns, and per-tool fixes this session already
        # found the bug independently in add_task, create_reminder,
        # find_contact, note/task/file search, create_goal, decision
        # recording, and export_state_snapshot, plus untested cases in
        # translate/research (raw text bled into the tool arg) and roadmap/
        # readiness/safety/list-goals/preferences/tasks (bare exact-match
        # triggers that require no trailing text at all, so the WHOLE match
        # failed and a later, unrelated pattern silently claimed the second
        # clause instead -- worse than clause-bleed, since the first,
        # explicitly-requested action never ran at all). Truncating at the
        # first top-level " and then " here, before any downstream matching,
        # closes this bug class once for every existing pattern in the file
        # AND any future one, instead of requiring a fix at every call site.
        # Deliberately scoped to " and then " only, NOT bare " and " (no
        # "then"): bare "and" is legitimate content in many working commands
        # today -- arithmetic chains ("add 5 and 10 and 15"), enumerations
        # ("search for apples and bananas"), compound conversion targets
        # ("convert X to Y and Z", already separately truncated in round 21).
        # Verified against every existing "and then" test case added in
        # rounds 32-34 (all already expect this exact truncation point) and
        # confirmed no working command anywhere in this session's fixtures
        # relies on "and then" being preserved as literal content.
        # Real gap found live 2026-07-10 (round 37, self-audit of round 35):
        # unbounded `\s+` quantifiers chained back-to-back (`\s{1,10}and\s{1,10}then\s{1,10}`)
        # cause catastrophic O(n^2) backtracking on pathological whitespace-
        # heavy input with no "and then" present -- "don't" + 50,000 spaces +
        # "remind me" took 6.5s on this regex alone (500,000 spaces: still
        # only 0.03s once bounded). Since this runs on EVERY command at the
        # planner's entry point, an unbounded version here is a real
        # algorithmic-complexity (ReDoS-shaped) risk from a single
        # pathological input, not a contrived edge case. Bounding each \s+ to
        # a small, generous constant (legitimate spacing between words is
        # essentially never more than a few characters) caps the worst-case
        # backtracking multiplier to a small constant independent of input
        # length, restoring linear-time behavior, with zero effect on any
        # correctly-spaced real input.
        # Fable-authorized entry-point ReDoS guard (2026-07-10 checkpoint):
        # normalize pathological whitespace runs and cap matching length BEFORE
        # any pattern runs, so the frozen send/call block's catastrophic-
        # backtracking patterns (and every other pattern in the file) are bounded
        # without editing a single frozen line. See _normalize_planner_input.
        task_completion_plan = _task_completion_command_plan(user_input)
        if task_completion_plan is not None:
            return task_completion_plan
        knowledge_promotion_plan = _knowledge_promotion_command_plan(user_input)
        if knowledge_promotion_plan is not None:
            return knowledge_promotion_plan
        decision_outcome_plan = _decision_outcome_command_plan(user_input)
        if decision_outcome_plan is not None:
            return decision_outcome_plan
        normalized_for_reminder_guard = _PATHOLOGICAL_WHITESPACE_RUN_RE.sub(
            " ", str(user_input or "")
        )
        if (
            len(normalized_for_reminder_guard) > _PLANNER_INPUT_MAX_MATCH_LEN
            and _is_direct_reminder_cancellation_command(normalized_for_reminder_guard[:512])
        ):
            return Plan(
                goal="Refuse a truncated reminder cancellation request.",
                actions=[
                    PlannedAction(
                        "respond",
                        {
                            "text": (
                                "I didn't change any reminders because that cancellation command was too long "
                                "to verify without truncation. Run `list reminders`, then send one short "
                                "`cancel reminder <id>` command."
                            )
                        },
                        "Reminder cancellation must never execute from a truncated command.",
                    )
                ],
                needs_model=False,
            )
        user_input = _normalize_planner_input(original_user_input)
        compound_cancel = re.search(
            r"\b(?:cancel|clear|delete|remove|stop)\b.{0,40}\b(?:reminders?|timers?|alarms?)\b"
            r".{0,80}\band\s{1,10}then\s{1,10}"
            r"(?:don['’]?t|do\s+not|never|wait|hold\s+on|cancel\s+that|actually\b.{0,20}(?:don['’]?t|do\s+not))\b",
            user_input,
            re.IGNORECASE | re.DOTALL,
        )
        if compound_cancel is not None:
            return Plan(
                goal="Keep reminders unchanged.",
                actions=[
                    PlannedAction(
                        "respond",
                        {"text": "Okay, I won't cancel those reminders."},
                        "A later clause explicitly withdrew the reminder cancellation request.",
                    )
                ],
                needs_model=False,
            )
        and_then_split = re.split(r"\s{1,10}and\s{1,10}then\s{1,10}", user_input.strip(), maxsplit=1, flags=re.IGNORECASE)
        text = and_then_split[0].strip() if len(and_then_split) > 1 else user_input.strip()
        raw_text = _raw_first_and_then_clause(original_user_input)
        low = text.lower()
        low_command = low.rstrip(" ?!.")
        # Real gap found live 2026-07-10 (round 36): none of this file's
        # ~13,000 lines of action-matching patterns account for an explicit
        # negation/cancellation lead-in, so a negated request would silently
        # execute the OPPOSITE of what was asked -- "don't remind me to call
        # mom" created a reminder anyway (set_reminder matched the "remind me
        # to call mom" tail, ignoring "don't"), "never mind the weather" and
        # "no need to check the weather" still fetched the weather (with a
        # garbled location swallowing the negation words themselves), and
        # "don't tell me the weather" ran get_weather with empty args. This
        # is more severe than the round-33/34/35 clause-bleed bugs: those
        # produced wrong DATA; this produces the WRONG ACTION, including a
        # LOCAL_SAFE write (an unwanted persisted reminder) the user
        # explicitly declined. Deterministic regex matching cannot reliably
        # understand negation scope, so rather than guess, an unambiguous
        # negation/cancellation lead-in routes to chat instead, where the
        # model can actually parse "don't do X" and respond appropriately
        # (acknowledge, ask for clarification, etc.) instead of blindly
        # executing the positive-sounding tail. Scoped tightly to the START
        # of the command (optionally preceded by a small, explicit filler set
        # like "cancel that,"/"actually,") so it does NOT catch "don't"
        # appearing later as legitimate content, e.g. "add task buy milk,
        # don't forget the eggs" (verified this stays on add_task, unaffected).
        if re.match(
            r"^(?:don'?t|do\s+not|never\s*mind|no\s+need\s+(?:to|for))\b"
            r"|^(?:cancel that|actually|wait|hold on),?\s+(?:don'?t|do\s+not)\b",
            low_command,
        ):
            return Plan(
                goal="Respond conversationally.",
                actions=[],
                needs_model=True,
                notes=NEGATED_REQUEST_PLAN_NOTE,
            )
        if not _polite_stripped:
            stripped_polite = _strip_trailing_politeness(text)
            if stripped_polite and stripped_polite.lower() != low:
                retry = self.plan(
                    stripped_polite,
                    allow_risky_natural_dispatch=allow_risky_natural_dispatch,
                    _vocative_stripped=_vocative_stripped,
                    _polite_stripped=True,
                )
                if retry.actions and all(action.tool_name in POLITE_COMMAND_RETRY_TOOLS for action in retry.actions):
                    return retry
        translation_language = r"(korean|english|japanese|chinese|spanish|french|german|한국어|영어|일본어|중국어)"
        bare_translation_m = re.search(rf"^(.{{1,120}}?)\s+in\s+{translation_language}$", low_command)
        bare_translation_like = False
        if bare_translation_m:
            bare_phrase = bare_translation_m.group(1).strip()
            bare_translation_like = bool(
                re.search(r"[a-z가-힣]", bare_phrase)
                and not re.match(
                    r"^(?:what|when|where|who|why|how|is|are|do|does|did|will|should|can|could|would|tell me|look ?up|search|research|find|open|show|list|weather|forecast|news|market|markets|stock|crypto|calendar|schedule|agenda|reminder|timer|alarm|email|message)\b",
                    bare_phrase,
                )
            )

        if bare_translation_like or re.search(rf"\bhow to say\b.{{1,60}}\bin {translation_language}\b|\bwhat does\b.{{1,60}}\bmean in {translation_language}\b|^say\b.{{1,60}}\bin {translation_language}\b", low_command):
            return Plan(
                goal="Translate text.",
                actions=[PlannedAction("translate", {"request": raw_text or text}, "User asked for a translation.")],
            )

        help_command = _strip_trailing_politeness(text).lower()
        if help_command in {
            "help me",
            "how do i use jarvis",
            "how do i use you",
            "how should i use jarvis",
            "what should i ask jarvis",
            "what should i ask you",
            "what commands can i use",
            "what commands do you understand",
            "show me commands",
            "show commands",
            "command list",
            "commands",
            "what can you help with",
        }:
            return Plan(
                goal="Show Jarvis help.",
                actions=[
                    PlannedAction(
                        "jarvis_help",
                        {"topic": ""},
                        "The user asked how to use Jarvis.",
                    )
                ],
            )

        help_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?)?(?:help|jarvis help)(?:\s+(?P<topic>[a-z][a-z _-]{0,80}))?$",
            text,
            re.IGNORECASE,
        )
        if help_match:
            topic = " ".join((help_match.group("topic") or "").strip().lower().replace("_", " ").replace("-", " ").split())
            allowed_multiword_help_topics = {
                "approval queue",
                "capability cockpit",
                "channel health",
                "channel status",
                "control plane",
                "daily brief",
                "daily briefing",
                "frozen redos",
                "frozen routing",
                "frozen routing risk",
                "input length guard",
                "internal orchestration",
                "live proof",
                "live proofs",
                "live test",
                "live tests",
                "long command safety",
                "long input safety",
                "long message redos",
                "morning brief",
                "morning briefing",
                "pending approval",
                "pending approvals",
                "planner input guard",
                "regex dos",
                "regex safety",
                "risk matrix",
                "send call routing risk",
            }
            if " " in topic and topic not in allowed_multiword_help_topics:
                help_match = None
        if help_match:
            return Plan(
                goal="Show Jarvis help.",
                actions=[
                    PlannedAction(
                        "jarvis_help",
                        {"topic": (help_match.group("topic") or "").strip()},
                        "The user asked for command help.",
                    )
                ],
            )

        if low in {
            "architecture",
            "architecture map",
            "show architecture",
            "show architecture map",
            "show latest architecture",
            "show latest architecture map",
            "jarvis architecture",
            "jarvis v2 architecture",
            "brain architecture",
            "assistant architecture",
            "six layer architecture",
            "six-layer architecture",
        }:
            return Plan(
                goal="Show Jarvis architecture map.",
                actions=[
                    PlannedAction(
                        "architecture_map",
                        {},
                        "The user asked how Jarvis maps to the core assistant architecture.",
                    )
                ],
            )

        if low in {
            "priority goal",
            "jarvis priority goal",
            "remember the goal",
            "what is the goal",
            "what is our goal",
            "what is jarvis goal",
            "what is jarvis's goal",
            "what should be our priority",
            "what is the priority",
            "jarvis priority",
        }:
            return Plan(
                goal="Show the top Jarvis build priority.",
                actions=[
                    PlannedAction(
                        "priority_goal",
                        {},
                        "The user asked for the goal that should steer Jarvis work until completion.",
                    )
                ],
            )

        if low in {
            "harness status",
            "agent harness",
            "agent harness status",
            "jarvis harness",
            "jarvis harness status",
            "agi harness",
            "agi direction",
            "jarvis agi direction",
        }:
            return Plan(
                goal="Show Jarvis agent harness status.",
                actions=[
                    PlannedAction(
                        "harness_status",
                        {},
                        "The user asked how Jarvis is becoming an AI agent harness toward AGI-like assistant behavior.",
                    )
                ],
            )

        if low in {
            "harness doctrine",
            "agent harness doctrine",
            "jarvis harness doctrine",
            "harness principles",
            "agent harness principles",
            "agent harness meaning",
            "what is an agent harness",
            "what does agent harness mean",
        }:
            return Plan(
                goal="Show Jarvis agent harness doctrine.",
                actions=[
                    PlannedAction(
                        "harness_doctrine",
                        {},
                        "The user asked what agent harness means and how that should steer Jarvis completion.",
                    )
                ],
            )

        coding_discipline_match = re.match(
            r"^(?:coding discipline|karpathy discipline|pre-code discipline|coding discipline packet|karpathy packet|what coding discipline applies|what coding discipline applies to|how should (?:codex|jarvis|we|i) code this safely|how should (?:codex|jarvis|we|i) implement this safely|safe coding discipline)(?:\s+(?:for|about|around|regarding|to))?\s*:?\s*(?P<objective>.+)?$",
            text,
            re.I,
        )
        if coding_discipline_match:
            objective = _clean_execution_prep_request(coding_discipline_match.group("objective")) or "continue Jarvis harness work"
            return Plan(
                goal="Build a read-only coding discipline packet before implementation.",
                actions=[
                    PlannedAction(
                        "coding_discipline_packet",
                        {"objective": objective},
                        "The user asked for a Karpathy-style coding discipline packet before implementation.",
                    )
                ],
            )

        if low in {
            "agi gates",
            "agi gate report",
            "agi direction gates",
            "harness gates",
            "agent harness gates",
            "jarvis agi gates",
            "jarvis gate report",
        }:
            return Plan(
                goal="Show Jarvis AGI-direction harness gates.",
                actions=[
                    PlannedAction(
                        "agi_gate_report",
                        {},
                        "The user asked for the measurable AGI-direction gates of the Jarvis agent harness.",
                    )
                ],
            )

        natural_completion_closeness_match = re.search(
            r"^how close (?:is|are) (?:(?:jarvis|the jarvis harness|the harness|this harness|the agent harness)(?: to (?:done|complete|finished|completion))?|we to (?:finishing|completing) (?:jarvis|the jarvis harness|the harness|this harness|the agent harness))\??$",
            text,
            re.IGNORECASE,
        )
        if natural_completion_closeness_match:
            return Plan(
                goal="Estimate Jarvis harness prototype completion.",
                actions=[
                    PlannedAction(
                        "harness_completion_assessment",
                        {},
                        "The user asked how close the Jarvis agent-harness prototype is to completion.",
                    )
                ],
            )

        agi_next_match = re.search(
            r"^(?:agi next build move|next agi build move|next agi move|agi build target|agent harness build target|next harness gate|next harness improvement|what is the next agi build move|what(?:'s| is) the next build move|what should (?:we|i|you|jarvis|codex) build next for agi|what should (?:jarvis|codex) work on next|what should (?:jarvis|codex) improve next|what should (?:we|you|jarvis|codex) improve next (?:for|in|on) (?:jarvis|the jarvis harness|the harness|this harness|the agent harness|agi)|what should (?:jarvis|codex) fix next|which agi gate should (?:we|i|you|jarvis) improve next|what agi gate should (?:we|i|you|jarvis) improve next|what gap should (?:we|i|you|jarvis|codex) close next)(?::?\s*(?P<gate>.*))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if agi_next_match:
            gate = _clean_execution_prep_request(agi_next_match.group("gate"))
            return Plan(
                goal="Select the next AGI-direction build target.",
                actions=[
                    PlannedAction(
                        "agi_next_build_move",
                        {"gate": gate},
                        "The user asked which AGI-direction real-execution gate should be improved next.",
                    )
                ],
            )

        if low in {
            "harness completion",
            "jarvis completion",
            "jarvis progress percent",
            "completion assessment",
            "how complete is jarvis",
            "is jarvis 40% done",
            "is jarvis forty percent done",
            "how far along is jarvis",
            "what percent done is jarvis",
            "what percent complete is jarvis",
        }:
            return Plan(
                goal="Estimate Jarvis harness prototype completion.",
                actions=[
                    PlannedAction(
                        "harness_completion_assessment",
                        {},
                        "The user asked for a completion estimate for the Jarvis agent-harness prototype.",
                    )
                ],
            )

        natural_readiness_gap_match = re.search(
            r"^(?:why (?:(?:isn'?t|is not) (?:jarvis|the jarvis harness|the harness|this harness|the agent harness)|(?:is|are) (?:jarvis|the jarvis harness|the harness|this harness|the agent harness) not) ready|what would make (?:jarvis|the jarvis harness|the harness|this harness|the agent harness) ready)\??$",
            text,
            re.IGNORECASE,
        )
        if natural_readiness_gap_match:
            return Plan(
                goal="Show the compact read-only Jarvis completion readiness digest.",
                actions=[
                    PlannedAction(
                        "harness_readiness_digest",
                        {},
                        "The user asked why Jarvis is not ready yet or what would make the harness ready.",
                    )
                ],
            )

        natural_completion_proof_gap_match = re.search(
            r"^(?:what (?:proof|evidence) (?:is )?missing for (?:jarvis|the jarvis harness|the harness|this harness|the agent harness)|what (?:proof|evidence) do (?:we|i|you|jarvis) need (?:for (?:jarvis|the jarvis harness|the harness|this harness|the agent harness)|before (?:jarvis|the jarvis harness|the harness|this harness|the agent harness) (?:is )?(?:ready|done|complete|finished)))\??$",
            text,
            re.IGNORECASE,
        )
        if natural_completion_proof_gap_match:
            return Plan(
                goal="Audit completion evidence before claiming the Jarvis objective is done.",
                actions=[
                    PlannedAction(
                        "completion_audit_packet",
                        {},
                        "The user asked what proof or evidence is still missing before Jarvis can be trusted as ready or complete.",
                    )
                ],
            )

        natural_completion_audit_match = re.search(
            r"^(?:why (?:isn'?t|is not) (?:jarvis|this|the goal) (?:done|complete|finished)|what(?:'s| is) missing before (?:jarvis|this|the goal) (?:is )?(?:done|complete|finished)|what(?:'s| is) left (?:to (?:finish|complete) (?:jarvis|the jarvis harness|the harness|this harness|the agent harness)|before (?:jarvis|this|the goal) (?:is )?(?:done|complete|finished))|what remains before (?:jarvis|this|the goal) (?:is )?(?:done|complete|finished)|what do (?:we|i|you|jarvis) need (?:to (?:finish|complete) (?:jarvis|the jarvis harness|the harness|this harness|the agent harness)|before (?:jarvis|this|the goal) (?:is )?(?:done|complete|finished))|audit whether (?:jarvis|this|the goal) (?:is )?(?:done|complete|finished))\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_completion_audit_match:
            objective = _clean_completion_objective(natural_completion_audit_match.group("objective"))
            return Plan(
                goal="Audit completion evidence before claiming the Jarvis objective is done.",
                actions=[
                    PlannedAction(
                        "completion_audit_packet",
                        {"objective": objective} if objective else {},
                        "The user asked in natural language what evidence is missing before Jarvis can be called complete.",
                    )
                ],
            )

        natural_prove_completion_match = re.search(
            r"^(?:prove (?:that )?(?:jarvis|this|the goal|it) (?:is )?(?:done|complete|finished)|prove (?:we|you|jarvis) (?:finished|completed) (?:jarvis|this|the goal|it))\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_prove_completion_match:
            objective = _clean_completion_objective(natural_prove_completion_match.group("objective"))
            return Plan(
                goal="Audit completion evidence before claiming the Jarvis objective is done.",
                actions=[
                    PlannedAction(
                        "completion_audit_packet",
                        {"objective": objective} if objective else {},
                        "The user asked for proof before accepting a completion claim.",
                    )
                ],
            )

        completion_proof_refresh_match = re.search(
            r"^(?:completion proof refresh|refresh completion proof|refresh proof lanes|completion lane refresh|proof lane refresh|refresh completion evidence|completion evidence refresh)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if completion_proof_refresh_match:
            objective = _clean_completion_objective(completion_proof_refresh_match.group("objective"))
            return Plan(
                goal="Refresh the read-only completion proof lane map.",
                actions=[
                    PlannedAction(
                        "completion_proof_refresh_packet",
                        {"objective": objective} if objective else {},
                        "The user asked for a lane-by-lane proof refresh before trusting completion.",
                    )
                ],
            )

        completion_audit_match = re.search(
            r"^(?:completion audit|audit completion|completion proof|completion packet|goal audit|prove completion|is this complete)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if completion_audit_match:
            objective = _clean_completion_objective(completion_audit_match.group("objective"))
            return Plan(
                goal="Audit completion evidence before claiming the Jarvis objective is done.",
                actions=[
                    PlannedAction(
                        "completion_audit_packet",
                        {"objective": objective} if objective else {},
                        "The user asked for a requirement-by-requirement completion proof packet.",
                    )
                ],
            )

        natural_evidence_ledger_match = re.search(
            r"^(?:what (?:proof|evidence) do (?:we|i|you|jarvis) have(?: for)?|show (?:me )?(?:the )?(?:proof|evidence)(?: for)?|where is (?:the )?(?:proof|evidence)(?: for)?)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_evidence_ledger_match:
            objective = _clean_completion_objective(natural_evidence_ledger_match.group("objective"))
            return Plan(
                goal="Show the read-only Jarvis harness evidence ledger.",
                actions=[
                    PlannedAction(
                        "evidence_ledger",
                        {"objective": objective} if objective else {},
                        "The user asked in natural language for the proof ledger before trusting a completion claim.",
                    )
                ],
            )

        evidence_ledger_match = re.search(
            r"^(?:evidence ledger|proof ledger|harness ledger|jarvis evidence|show evidence|completion evidence)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if evidence_ledger_match:
            objective = _clean_completion_objective(evidence_ledger_match.group("objective"))
            return Plan(
                goal="Show the read-only Jarvis harness evidence ledger.",
                actions=[
                    PlannedAction(
                        "evidence_ledger",
                        {"objective": objective} if objective else {},
                        "The user asked for a proof ledger across harness lanes, audit evidence, tasks, and approvals.",
                    )
                ],
            )

        natural_completion_claim_match = re.search(
            r"^(?:are (?:we|you|jarvis) (?:done|complete|finished)(?: with (?:jarvis|this|the goal))?|is (?:jarvis|this|the goal) (?:done|complete|finished)|did (?:we|you|jarvis) finish (?:jarvis|this|the goal)|can (?:you|jarvis|we|i) (?:say|claim|mark) (?:jarvis|this|the goal)?\s*(?:is )?(?:done|complete|finished)|should (?:we|you|jarvis) (?:mark|call|claim) (?:jarvis|this|the goal|it)?\s*(?:done|complete|finished)|can (?:this|it|the goal|jarvis) be (?:marked|called|claimed) (?:done|complete|finished)|mark (?:jarvis|this|the goal|it) (?:done|complete|finished)|claim (?:jarvis|this|the goal|it)?\s*(?:done|complete|finished))\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_completion_claim_match:
            objective = _clean_completion_objective(natural_completion_claim_match.group("objective"))
            return Plan(
                goal="Run the read-only completion claim gate before any completion claim.",
                actions=[
                    PlannedAction(
                        "completion_claim_gate",
                        {"objective": objective} if objective else {},
                        "The user asked in natural language whether Jarvis can safely claim completion.",
                    )
                ],
            )

        completion_claim_gate_match = re.search(
            r"^(?:completion claim gate|completion claim|claim gate|completion gate|can jarvis claim done|can jarvis claim complete|can i claim complete|can we claim complete|can this be claimed done|claim done|claim complete|claim completion)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if completion_claim_gate_match:
            objective = _clean_completion_objective(completion_claim_gate_match.group("objective"))
            return Plan(
                goal="Run the read-only completion claim gate before any completion claim.",
                actions=[
                    PlannedAction(
                        "completion_claim_gate",
                        {"objective": objective} if objective else {},
                        "The user asked whether Jarvis can safely claim an objective is complete.",
                    )
                ],
            )

        completion_next_proof_match = re.search(
            r"^(?:completion next proof|next completion proof|next proof|next proof command|completion proof next|proof next|what next proof|what proof next|what is the next proof|what proof should (?:we|i|you|jarvis) (?:do|run|gather) next|what should (?:we|i|you|jarvis) prove next|what evidence next|what evidence should (?:we|i|you|jarvis) (?:do|run|gather) next|what verification next|what verification should (?:we|i|you|jarvis) (?:do|run|gather) next|next completion evidence|next completion check|what should (?:we|i|you|jarvis) verify next for completion|what(?:'s| is) the next step to (?:finish|complete) (?:jarvis|the jarvis harness|the harness|this harness|the agent harness)|next step to (?:finish|complete) (?:jarvis|the jarvis harness|the harness|this harness|the agent harness))\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if completion_next_proof_match:
            objective = _clean_completion_objective(completion_next_proof_match.group("objective"))
            return Plan(
                goal="Choose the next read-only completion proof command.",
                actions=[
                    PlannedAction(
                        "completion_next_proof_packet",
                        {"objective": objective} if objective else {},
                        "The user asked for the next single proof command before trusting completion.",
                    )
                ],
            )

        operator_handoff_match = re.search(
            r"^(?:operator handoff|next operator handoff|handoff next proof|proof handoff|operator next command|what should the operator do next|what should the operator do next)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if operator_handoff_match:
            objective = _clean_completion_objective(operator_handoff_match.group("objective"))
            return Plan(
                goal="Show the next operator-reviewable proof handoff.",
                actions=[
                    PlannedAction(
                        "operator_handoff_packet",
                        {"objective": objective} if objective else {},
                        "The user asked for the next operator-safe command and why it comes first.",
                    )
                ],
            )

        readiness_digest_match = re.search(
            r"^(?:harness readiness digest|readiness digest|jarvis readiness digest|completion readiness digest|harness readiness|jarvis readiness|completion readiness|what is the readiness digest|what is harness readiness|what is jarvis readiness|what is completion readiness|show readiness digest|show harness readiness|readiness summary|status digest|harness triage|jarvis triage|readiness triage|completion triage|how done is jarvis now|what blocks jarvis now|what is blocking completion|what is blocking jarvis|what blockers remain|completion blockers|harness blockers|what(?:'s| is) (?:jarvis|the jarvis harness|the harness|this harness|the agent harness) missing|what(?:'s| is) missing for (?:jarvis|the jarvis harness|the harness|this harness|the agent harness)|what(?:'s| is) left for (?:jarvis|the jarvis harness|the harness|this harness|the agent harness)|what remains for (?:jarvis|the jarvis harness|the harness|this harness|the agent harness)|what (?:are )?(?:the )?remaining gaps|remaining gaps|next missing capability)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if readiness_digest_match:
            objective = _clean_completion_objective(readiness_digest_match.group("objective"))
            return Plan(
                goal="Show the compact read-only Jarvis completion readiness digest.",
                actions=[
                    PlannedAction(
                        "harness_readiness_digest",
                        {"objective": objective} if objective else {},
                        "The user asked for the current Jarvis readiness state, top blockers, and next proof command.",
                    )
                ],
            )

        proof_bundle_match = re.search(
            r"^(?:execution proof bundle|proof bundle|action proof bundle|trust bundle|trust packet|pretrust packet|pre-trust packet)(?:\s+(?:for|about|around|regarding))?\s*:?\s*(?P<request>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if proof_bundle_match:
            request = _clean_execution_prep_request(proof_bundle_match.group("request"))
            fields = _parse_scope_packet_spec(request)
            return Plan(
                goal="Build the read-only proof bundle before trusting execution.",
                actions=[
                    PlannedAction(
                        "execution_proof_bundle",
                        {
                            "request": fields["action"],
                            "verification": fields["verification"],
                            "tests": fields["tests"],
                            "evidence": fields["acceptance"],
                            "recovery": fields["rollback"],
                            "approval": fields["approval"],
                        },
                        "The user asked for route, risk, approval, verification, audit, and recovery proof before trusting a real action.",
                    )
                ],
            )

        natural_proof_bundle_match = re.search(
            r"^(?:can (?:we|i|jarvis) trust|is\s+(?P<trustworthy_request>.+?)\s+trustworthy|what proof (?:is needed|do we need|does jarvis need) before (?:running|executing|trusting)|what (?:do we|does jarvis) need before (?:running|executing|trusting)|before (?:running|executing|trusting)\s+(?P<before_request>.+?)\s+what (?:proof|do we|does jarvis) (?:is needed|do we need|need)|show (?:the )?pre[- ]?trust proof(?: for)?|build (?:the )?pre[- ]?trust proof(?: for)?)\s*:?\s*(?P<request>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_proof_bundle_match:
            request = _clean_execution_prep_request(
                natural_proof_bundle_match.group("request")
                or natural_proof_bundle_match.group("trustworthy_request")
                or natural_proof_bundle_match.group("before_request")
            )
            fields = _parse_scope_packet_spec(request)
            return Plan(
                goal="Build the read-only proof bundle before trusting execution.",
                actions=[
                    PlannedAction(
                        "execution_proof_bundle",
                        {
                            "request": fields["action"],
                            "verification": fields["verification"],
                            "tests": fields["tests"],
                            "evidence": fields["acceptance"],
                            "recovery": fields["rollback"],
                            "approval": fields["approval"],
                        },
                        "The user asked in natural language what proof is needed before trusting or running an action.",
                    )
                ],
            )

        mission_control_match = re.search(
            r"^(?:execution mission control|mission control packet|execution mission packet|mission rehearsal|operation mission control|operational mission control|set up mission control|setup mission control|plan the mission)(?:\s+(?:for|about|around|regarding))?\s*:?\s*(?P<request>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if mission_control_match:
            request = _clean_execution_prep_request(mission_control_match.group("request"))
            return Plan(
                goal="Build the read-only execution mission-control rehearsal before real execution.",
                actions=[
                    PlannedAction(
                        "execution_mission_control",
                        {"request": request} if request else {},
                        "The user asked for a single mission-control packet across route, gate, act, verify, recover, and learn without executing the order.",
                    )
                ],
            )

        natural_mission_control_match = re.search(
            r"^mission control\s+(?:for|about|around|regarding)\s+(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_mission_control_match:
            request = _clean_execution_prep_request(natural_mission_control_match.group("request"))
            return Plan(
                goal="Build the read-only execution mission-control rehearsal before real execution.",
                actions=[
                    PlannedAction(
                        "execution_mission_control",
                        {"request": request} if request else {},
                        "The user asked for a single mission-control packet across route, gate, act, verify, recover, and learn without executing the order.",
                    )
                ],
            )

        natural_save_execution_case_match = re.search(
            r"^(?:make|create|save|open|prepare)\s+(?:a\s+)?(?:durable\s+)?(?:execution\s+)?case(?:\s+file)?\s+(?:for|about|around|regarding)\s+(?P<request>.+)$"
            r"|^case\s+file\s+(?:for|about|around|regarding)\s+(?P<case_file_request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_save_execution_case_match:
            request = _clean_execution_prep_request(
                natural_save_execution_case_match.group("request")
                or natural_save_execution_case_match.group("case_file_request")
            )
            args = _parse_execution_case_save_spec(request) if request else {}
            return Plan(
                goal="Save a local Jarvis execution case file before real execution.",
                actions=[
                    PlannedAction(
                        "save_execution_case",
                        args,
                        "The user asked to create a durable Jarvis-owned case file for a future order without executing it.",
                    )
                ],
            )

        execution_case_handoff_match = re.search(
            r"^(?:execution case handoff|case handoff packet|execution handoff packet|cockpit case handoff|cockpit to case|cockpit-to-case|save case handoff|case save handoff)\s*:?\s*(?P<request>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_case_handoff_match:
            request = (execution_case_handoff_match.group("request") or "").strip()
            return Plan(
                goal="Build the read-only handoff from command cockpit to durable execution case.",
                actions=[
                    PlannedAction(
                        "execution_case_handoff_packet",
                        {"request": request} if request else {},
                        "The user asked for the cockpit-to-case handoff path before saving or running a durable execution case.",
                    )
                ],
            )

        save_execution_case_match = re.search(
            r"^(?:save execution case|create execution case|open execution case|execution case file|save mission case)\s*:?\s*(?P<request>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if save_execution_case_match:
            request = (save_execution_case_match.group("request") or "").strip()
            args = _parse_execution_case_save_spec(request) if request else {}
            return Plan(
                goal="Save a local Jarvis execution case file before real execution.",
                actions=[
                    PlannedAction(
                        "save_execution_case",
                        args,
                        "The user asked to create a durable Jarvis-owned case file for a future order without executing it.",
                    )
                ],
            )

        natural_execution_case_evidence_packet_match = re.search(
            r"^(?:preview|validate|check|inspect)\s+(?:case\s+)?(?:evidence|proof|receipt)\s+(?:for|against)\s+(?:the\s+)?(?P<case_id>latest|last|newest|case\s+\d+|\d+)\s+case\s*:?\s*(?P<summary>.+)$"
            r"|^(?:preview|validate|check|inspect)\s+(?:case\s+)?(?:evidence|proof|receipt)\s+(?:for|against)\s+case\s+(?P<case_id_after>\d+|latest|last|newest)\s*:?\s*(?P<summary_after>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_execution_case_evidence_packet_match:
            case_id = _clean_execution_case_id(
                natural_execution_case_evidence_packet_match.group("case_id")
                or natural_execution_case_evidence_packet_match.group("case_id_after")
            )
            summary = _strip_trailing_politeness(
                natural_execution_case_evidence_packet_match.group("summary")
                or natural_execution_case_evidence_packet_match.group("summary_after")
                or ""
            )
            return Plan(
                goal="Preview execution case evidence before saving it.",
                actions=[
                    PlannedAction(
                        "execution_case_evidence_packet",
                        {"case_id": case_id, "summary": summary, "event_type": "evidence"},
                        "The user asked to validate execution-case evidence and receipt metadata before appending it locally.",
                    )
                ],
            )

        execution_case_evidence_packet_match = re.search(
            r"^(?:execution case evidence packet|case evidence packet|case evidence preview|preview case evidence|evidence intake|case evidence intake)\s*(?P<case_id>latest|last|newest|\d+)?\s*:?\s*(?P<summary>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_case_evidence_packet_match:
            case_id = (execution_case_evidence_packet_match.group("case_id") or "latest").strip()
            summary = (execution_case_evidence_packet_match.group("summary") or "").strip()
            return Plan(
                goal="Preview execution case evidence before saving it.",
                actions=[
                    PlannedAction(
                        "execution_case_evidence_packet",
                        {"case_id": case_id, "summary": summary, "event_type": "evidence"},
                        "The user asked to validate execution-case evidence and receipt metadata before appending it locally.",
                    )
                ],
            )

        natural_append_execution_case_match = re.search(
            r"^(?:attach|add|append)\s+(?:case\s+)?(?:evidence|proof|receipt)\s+to\s+(?:the\s+)?(?P<case_id>latest|last|newest|case\s+\d+|\d+)\s+case\s*:?\s*(?P<summary>.+)$"
            r"|^(?:attach|add|append)\s+(?:case\s+)?(?:evidence|proof|receipt)\s+to\s+case\s+(?P<case_id_after>\d+|latest|last|newest)\s*:?\s*(?P<summary_after>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_append_execution_case_match:
            case_id = _clean_execution_case_id(
                natural_append_execution_case_match.group("case_id")
                or natural_append_execution_case_match.group("case_id_after")
            )
            summary = _strip_trailing_politeness(
                natural_append_execution_case_match.group("summary")
                or natural_append_execution_case_match.group("summary_after")
                or ""
            )
            return Plan(
                goal="Append a local evidence event to a saved execution case.",
                actions=[
                    PlannedAction(
                        "append_execution_case_evidence",
                        {"case_id": case_id, "summary": summary, "event_type": "evidence"},
                        "The user asked to attach proof or outcome evidence to a saved execution case without running it.",
                    )
                ],
            )

        append_execution_case_match = re.search(
            r"^(?:append execution case evidence|add execution case evidence|execution case evidence|case evidence|case receipt|attach case receipt)\s*(?P<case_id>latest|last|newest|\d+)?\s*:?\s*(?P<summary>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if append_execution_case_match:
            case_id = (append_execution_case_match.group("case_id") or "latest").strip()
            summary = (append_execution_case_match.group("summary") or "").strip()
            return Plan(
                goal="Append a local evidence event to a saved execution case.",
                actions=[
                    PlannedAction(
                        "append_execution_case_evidence",
                        {"case_id": case_id, "summary": summary, "event_type": "evidence"},
                        "The user asked to attach proof or outcome evidence to a saved execution case without running it.",
                    )
                ],
            )

        natural_inspect_execution_case_match = re.search(
            r"^(?:inspect|show|open|view)\s+(?:the\s+)?(?P<case_id>latest|last|newest|case\s+\d+|\d+)\s+case$"
            r"|^(?:inspect|show|open|view)\s+case\s+(?P<case_id_after>\d+|latest|last|newest)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_inspect_execution_case_match:
            case_id = _clean_execution_case_id(
                natural_inspect_execution_case_match.group("case_id")
                or natural_inspect_execution_case_match.group("case_id_after")
            )
            return Plan(
                goal="Inspect a saved Jarvis execution case file.",
                actions=[
                    PlannedAction(
                        "inspect_execution_case",
                        {"case_id": case_id},
                        "The user asked to inspect a saved execution case without running it.",
                    )
                ],
            )

        inspect_execution_case_match = re.search(
            r"^(?:inspect execution case|execution case|show execution case|case file)\s*:?\s*(?P<case_id>latest|last|newest|\d+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if inspect_execution_case_match:
            case_id = (inspect_execution_case_match.group("case_id") or "latest").strip()
            return Plan(
                goal="Inspect a saved Jarvis execution case file.",
                actions=[
                    PlannedAction(
                        "inspect_execution_case",
                        {"case_id": case_id},
                        "The user asked to inspect a saved execution case without running it.",
                    )
                ],
            )

        natural_execution_case_gate_match = re.search(
            r"^(?:is|check|show|review)\s+(?:the\s+)?(?P<case_id>latest|last|newest|case\s+\d+|\d+)\s+case\s+(?:ready|readiness)$"
            r"|^(?:is|check|show|review)\s+case\s+(?P<case_id_after>\d+|latest|last|newest)\s+(?:ready|readiness)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_execution_case_gate_match:
            case_id = _clean_execution_case_id(
                natural_execution_case_gate_match.group("case_id")
                or natural_execution_case_gate_match.group("case_id_after")
            )
            return Plan(
                goal="Gate a saved execution case against evidence, approvals, and recovery before review.",
                actions=[
                    PlannedAction(
                        "execution_case_gate",
                        {"case_id": case_id},
                        "The user asked for a read-only readiness gate over a saved execution case without running it.",
                    )
                ],
            )

        execution_case_gate_match = re.search(
            r"^(?:execution case gate|case gate|case readiness|execution case readiness|case completion gate)\s*:?\s*(?P<case_id>latest|last|newest|\d+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_case_gate_match:
            case_id = (execution_case_gate_match.group("case_id") or "latest").strip()
            return Plan(
                goal="Gate a saved execution case against evidence, approvals, and recovery before review.",
                actions=[
                    PlannedAction(
                        "execution_case_gate",
                        {"case_id": case_id},
                        "The user asked for a read-only readiness gate over a saved execution case without running it.",
                    )
                ],
            )

        natural_execution_case_review_match = re.search(
            r"^(?:review|human review|case review|execution case review)\s+(?:the\s+)?(?P<case_id>latest|last|newest|case\s+\d+|\d+)\s+case$"
            r"|^(?:review|human review)\s+case\s+(?P<case_id_after>\d+|latest|last|newest)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_execution_case_review_match:
            case_id = _clean_execution_case_id(
                natural_execution_case_review_match.group("case_id")
                or natural_execution_case_review_match.group("case_id_after")
            )
            return Plan(
                goal="Build a read-only human-review packet for a saved execution case.",
                actions=[
                    PlannedAction(
                        "execution_case_review_packet",
                        {"case_id": case_id},
                        "The user asked for a human-review packet over a saved execution case before trusting completion.",
                    )
                ],
            )

        execution_case_review_match = re.search(
            r"^(?:execution case review|case review|execution case review packet|case review packet|execution case human review|case human review)\s*:?\s*(?P<case_id>latest|last|newest|\d+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_case_review_match:
            case_id = (execution_case_review_match.group("case_id") or "latest").strip()
            return Plan(
                goal="Build a read-only human-review packet for a saved execution case.",
                actions=[
                    PlannedAction(
                        "execution_case_review_packet",
                        {"case_id": case_id},
                        "The user asked for a human-review packet over a saved execution case before trusting completion.",
                    )
                ],
            )

        natural_execution_case_closure_match = re.search(
            r"^(?:close out|closeout|close|closure for|proof closure for)\s+(?:the\s+)?(?P<case_id>latest|last|newest|case\s+\d+|\d+)\s+case$"
            r"|^(?:close out|closeout|close|closure for|proof closure for)\s+case\s+(?P<case_id_after>\d+|latest|last|newest)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_execution_case_closure_match:
            case_id = _clean_execution_case_id(
                natural_execution_case_closure_match.group("case_id")
                or natural_execution_case_closure_match.group("case_id_after")
            )
            return Plan(
                goal="Build a read-only closure packet for a saved execution case.",
                actions=[
                    PlannedAction(
                        "execution_case_closure_packet",
                        {"case_id": case_id},
                        "The user asked whether a saved execution case has closed proof debt before trusting completion.",
                    )
                ],
            )

        execution_case_closure_match = re.search(
            r"^(?:execution case closure|case closure|case closure packet|execution case closure packet|case proof closure|execution case proof closure|case closeout|execution case closeout)\s*:?\s*(?P<case_id>latest|last|newest|\d+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_case_closure_match:
            case_id = (execution_case_closure_match.group("case_id") or "latest").strip()
            return Plan(
                goal="Build a read-only closure packet for a saved execution case.",
                actions=[
                    PlannedAction(
                        "execution_case_closure_packet",
                        {"case_id": case_id},
                        "The user asked whether a saved execution case has closed proof debt before trusting completion.",
                    )
                ],
            )

        natural_execution_case_timeline_match = re.search(
            r"^(?:show|view|inspect|open)\s+(?:the\s+)?(?P<case_id>latest|last|newest|case\s+\d+|\d+)\s+case\s+(?:timeline|history)$"
            r"|^(?:show|view|inspect|open)\s+case\s+(?P<case_id_after>\d+|latest|last|newest)\s+(?:timeline|history)$"
            r"|^(?:show|view|inspect|open)\s+(?:the\s+)?case\s+(?:timeline|history)\s+(?P<case_id_tail>\d+|latest|last|newest)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_execution_case_timeline_match:
            case_id = _clean_execution_case_id(
                natural_execution_case_timeline_match.group("case_id")
                or natural_execution_case_timeline_match.group("case_id_after")
                or natural_execution_case_timeline_match.group("case_id_tail")
            )
            return Plan(
                goal="Show a chronological read-only timeline for a saved execution case.",
                actions=[
                    PlannedAction(
                        "execution_case_timeline",
                        {"case_id": case_id},
                        "The user asked for a saved execution case history/timeline without running it.",
                    )
                ],
            )

        execution_case_timeline_match = re.search(
            r"^(?:execution case timeline|case timeline|execution timeline|case history|execution case history)\s*:?\s*(?P<case_id>latest|last|newest|\d+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_case_timeline_match:
            case_id = (execution_case_timeline_match.group("case_id") or "latest").strip()
            return Plan(
                goal="Show a chronological read-only timeline for a saved execution case.",
                actions=[
                    PlannedAction(
                        "execution_case_timeline",
                        {"case_id": case_id},
                        "The user asked for a saved execution case history/timeline without running it.",
                    )
                ],
            )

        execution_runbook_match = re.search(
            r"^(?:execution runbook|action runbook|runbook|operation runbook|operational runbook|real execution runbook|give me a runbook|what is the runbook|steps before (?:running|executing)|operational plan)(?:\s+(?:for|about|around|regarding))?\s*:?\s*(?P<request>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_runbook_match:
            request = _clean_execution_prep_request(execution_runbook_match.group("request"))
            return Plan(
                goal="Build the read-only operational runbook before real execution.",
                actions=[
                    PlannedAction(
                        "execution_runbook",
                        {"request": request} if request else {},
                        "The user asked for the before/during/after harness runbook that governs a real order without executing it.",
                    )
                ],
            )

        if low in {
            "roadmap",
            "roadmap report",
            "jarvis roadmap",
            "jarvis v2 roadmap",
            "assistant roadmap",
            "build roadmap",
            "show roadmap",
            "show me the roadmap",
            "what should we build next",
            "what should jarvis build next",
        }:
            return Plan(
                goal="Show Jarvis build roadmap.",
                actions=[
                    PlannedAction(
                        "roadmap_report",
                        {},
                        "The user asked for the safe phased roadmap for building Jarvis.",
                    )
                ],
            )

        model_status_command = _strip_trailing_politeness(text)
        model_status_command = re.sub(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?",
            "",
            model_status_command,
            flags=re.IGNORECASE,
        ).strip().lower()
        model_status_command = re.sub(r"^(?:latest|last|newest|current)\s+", "", model_status_command).strip()
        if model_status_command in {
            "model status",
            "model routing",
            "model routing status",
            "brain model status",
            "ollama status",
            "local model status",
            "model health",
            "local model health",
            "ai model status",
            "llm status",
            "llm health",
            "local llm status",
            "chat model status",
            "chat model",
            "planner model status",
            "planner model",
        }:
            return Plan(
                goal="Show model routing status.",
                actions=[
                    PlannedAction(
                        "model_routing_status",
                        {},
                        "The user asked whether Jarvis chat/planner model routing is ready.",
                    )
                ],
            )

        if model_status_command in {
            "channel status",
            "channel health",
            "channels status",
            "channels health",
            "messaging status",
            "messaging health",
            "message channels status",
            "channel report",
            "channel health report",
            "check channel health",
        }:
            return Plan(
                goal="Report messaging/calling channel health from audit metadata.",
                actions=[
                    PlannedAction(
                        "channel_health",
                        {},
                        "The user asked whether the messaging and calling channels are healthy.",
                    )
                ],
            )

        specialist_handoff_quality_display_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:latest|last|newest|current)?\s*)?"
            r"(?P<kind>specialist handoff quality gate|specialist quality gate|handoff quality gate|brain handoff quality gate|model handoff quality gate|multi brain handoff quality gate|multi-brain handoff quality gate)"
            r"(?::?\s*(?P<request>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_handoff_quality_display_match:
            request = _clean_specialist_packet_request(specialist_handoff_quality_display_match.group("request"))
            return Plan(
                goal="Bind specialist route quality to handoff receipt quality before proposal review.",
                actions=[
                    PlannedAction(
                        "specialist_handoff_quality_gate",
                        {"request": request},
                        "The user asked to prove measured specialist handoff quality before any specialist output can approach tool proposal review.",
                    )
                ],
            )

        specialist_preflight_display_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:latest|last|newest|current)?\s*)?"
            r"(?P<kind>"
            r"specialist router contract|specialist routing contract|brain router contract|multi brain router contract|multi-brain router contract|route specialist|"
            r"specialist route quality|route quality|brain route quality|multi brain route quality|multi-brain route quality|specialist quality|"
            r"specialist execution readiness|specialist readiness|brain execution readiness|brain readiness|model execution readiness|multi brain readiness|multi-brain readiness|"
            r"specialist handoff receipt|handoff receipt|brain handoff receipt|model handoff receipt|route handoff receipt|"
            r"specialist proposal gate|specialist tool proposal gate|brain proposal gate|model proposal gate|multi brain proposal gate|multi-brain proposal gate"
            r")(?::?\s*(?P<request>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_preflight_display_match:
            kind = specialist_preflight_display_match.group("kind").lower()
            request = _clean_specialist_packet_request(specialist_preflight_display_match.group("request"))
            if "quality" in kind:
                tool_name = "specialist_route_quality"
                goal = "Measure specialist route quality before model execution."
                reason = "The user asked for a read-only specialist route confidence and verifier coverage packet."
            elif "readiness" in kind:
                tool_name = "specialist_execution_readiness"
                goal = "Gate specialist model execution readiness."
                reason = "The user asked whether the request is ready for specialist model drafting or should hold for route, approval, or model setup review."
            elif "handoff" in kind:
                tool_name = "specialist_handoff_receipt"
                goal = "Build a read-only specialist handoff receipt for the order."
                reason = "The user asked Jarvis to package an order for the selected specialist brain with input, output, safety, and verification contracts."
            elif "proposal" in kind:
                tool_name = "specialist_proposal_gate"
                goal = "Gate specialist draft-to-tool proposal readiness."
                reason = "The user asked to prove route, readiness, handoff, verifier, model, and approval boundaries before specialist output can become tool proposals."
            else:
                tool_name = "specialist_router_contract"
                goal = "Preview specialist brain routing contract."
                reason = "The user asked for a read-only specialist brain route and handoff contract."
            return Plan(
                goal=goal,
                actions=[
                    PlannedAction(
                        tool_name,
                        {"request": request},
                        reason,
                    )
                ],
            )

        def _specialist_structured_proof_args(body: str) -> dict[str, str]:
            body = body.strip()
            key_names = [
                "verification_receipt_sha256",
                "runtime_trace_receipt",
                "runtime_trace_sha256",
                "execution_recovery_sha256",
                "execution_audit_sha256",
                "after_action_learning_sha256",
                "completion_claim_sha256",
                "verification_receipt",
                "execution_recovery",
                "execution_audit",
                "after_action_learning",
                "completion_claim",
                "recovery_sha256",
                "audit_sha256",
                "learning_sha256",
                "claim_sha256",
                "receipt_sha256",
                "runtime_trace",
                "tool_name",
                "arguments",
                "verification",
                "expected",
                "recovery",
                "learning",
                "blockers",
                "receipt",
                "audit",
                "claim",
                "trace",
                "tool",
                "args",
            ]
            keys = "|".join(re.escape(key) for key in key_names)
            field_pattern = rf"(?:^|[;\n])\s*(?P<key>{keys})(?:\s*:|\s+)(?P<value>.*?)(?=\s*(?:[;\n]\s*(?:{keys})(?:\s*:|\s+))|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            request_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                request_parts.append(body[cursor:start].strip(" ;\n"))
                cursor = end
            request_parts.append(body[cursor:].strip(" ;\n"))
            request = "; ".join(part for part in request_parts if part).strip() or body
            if not request or re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", request, re.IGNORECASE):
                request = DEFAULT_HARNESS_PACKET_REQUEST
            return {
                "request": request,
                "tool": fields.get("tool") or fields.get("tool_name") or "",
                "arguments": fields.get("arguments") or fields.get("args") or "",
                "verification": fields.get("verification") or fields.get("expected") or "",
                "runtime_trace": fields.get("runtime_trace") or fields.get("runtime_trace_receipt") or fields.get("trace") or "",
                "verification_receipt": fields.get("verification_receipt") or fields.get("receipt") or "",
                "execution_audit": fields.get("execution_audit") or fields.get("audit") or "",
                "execution_recovery": fields.get("execution_recovery") or fields.get("recovery") or "",
                "after_action_learning": fields.get("after_action_learning") or fields.get("learning") or "",
                "completion_claim": fields.get("completion_claim") or fields.get("claim") or "",
                "runtime_trace_sha256": fields.get("runtime_trace_sha256") or "",
                "verification_receipt_sha256": fields.get("verification_receipt_sha256") or fields.get("receipt_sha256") or "",
                "execution_audit_sha256": fields.get("execution_audit_sha256") or fields.get("audit_sha256") or "",
                "execution_recovery_sha256": fields.get("execution_recovery_sha256") or fields.get("recovery_sha256") or "",
                "after_action_learning_sha256": fields.get("after_action_learning_sha256") or fields.get("learning_sha256") or "",
                "completion_claim_sha256": fields.get("completion_claim_sha256") or fields.get("claim_sha256") or "",
                "blockers": fields.get("blockers") or "",
            }

        natural_specialist_router_match = re.search(
            r"^(?:which|what)\s+(?:specialist|brain)\s+(?:should\s+)?(?:handle|handles|take|own|work on)\s+(?P<request>.+)$"
            r"|^(?:route|send)\s+(?:this|that|it)?\s*(?:to|through)\s+(?:the\s+)?(?:right\s+|best\s+)?(?:specialist|brain)\s*:?\s*(?P<route_request>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_specialist_router_match:
            request = _clean_specialist_natural_request(
                natural_specialist_router_match.group("request")
                or natural_specialist_router_match.group("route_request")
            )
            return Plan(
                goal="Preview specialist brain routing contract.",
                actions=[
                    PlannedAction(
                        "specialist_router_contract",
                        {"request": request},
                        "The user asked which specialist lane should handle the request without executing it.",
                    )
                ],
            )

        natural_specialist_readiness_match = re.search(
            r"^(?:is|are)\s+(?:the\s+)?(?:specialist|brain|multi[- ]brain)\s+(?:ready|ready\s+to\s+(?:draft|handle|work on))\s+(?P<request>.+)$"
            r"|^(?:can|could|should)\s+(?:a\s+)?(?:specialist|brain|multi[- ]brain)\s+(?:draft|handle|work on)\s+(?P<can_request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_specialist_readiness_match:
            request = _clean_specialist_natural_request(
                natural_specialist_readiness_match.group("request")
                or natural_specialist_readiness_match.group("can_request")
            )
            return Plan(
                goal="Gate specialist model execution readiness.",
                actions=[
                    PlannedAction(
                        "specialist_execution_readiness",
                        {"request": request},
                        "The user asked whether a request is ready for specialist drafting without running the draft.",
                    )
                ],
            )

        natural_specialist_handoff_match = re.search(
            r"^(?:make|create|prepare|build)\s+(?:a\s+)?(?:specialist|brain|multi[- ]brain)\s+handoff\s+(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_specialist_handoff_match:
            request = _clean_specialist_natural_request(natural_specialist_handoff_match.group("request"))
            return Plan(
                goal="Build a read-only specialist handoff receipt for the order.",
                actions=[
                    PlannedAction(
                        "specialist_handoff_receipt",
                        {"request": request},
                        "The user asked Jarvis to package an order for the selected specialist lane without executing it.",
                    )
                ],
            )

        natural_specialist_orchestration_match = re.search(
            r"^(?:(?:multi[- ]brain|specialist|brain)\s+(?:plan|orchestration plan|harness plan))\s+(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_specialist_orchestration_match:
            request = _clean_specialist_natural_request(natural_specialist_orchestration_match.group("request"))
            return Plan(
                goal="Build a read-only multi-brain specialist orchestration packet.",
                actions=[
                    PlannedAction(
                        "specialist_orchestration_packet",
                        {"request": request},
                        "The user asked to plan a specialist/multi-brain harness flow without model calls or execution.",
                    )
                ],
            )

        natural_specialist_dry_run_match = re.search(
            r"^(?:dry\s+run|rehearse|preview|review)\s+(?:the\s+)?(?:specialist|brain|multi[- ]brain)\s+(?:tool\s+)?(?:proposal|tool proposal|action proposal)(?:\s+(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_specialist_dry_run_match:
            request = _clean_specialist_natural_request(natural_specialist_dry_run_match.group("request"))
            return Plan(
                goal="Review specialist tool proposal dry-run readiness.",
                actions=[
                    PlannedAction(
                        "specialist_tool_dry_run_packet",
                        {"request": request},
                        "The user asked to dry-run review a specialist tool proposal without executing it.",
                    )
                ],
            )

        natural_specialist_closure_match = re.search(
            r"^(?:close|close out|finish)\s+(?:the\s+)?(?:specialist|brain|multi[- ]brain)\s+(?:run|handoff|cycle)\s+(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_specialist_closure_match:
            request = _clean_specialist_natural_request(natural_specialist_closure_match.group("request"))
            return Plan(
                goal="Close specialist runtime handoff proof after normal runtime review.",
                actions=[
                    PlannedAction(
                        "specialist_post_run_closure_packet",
                        _specialist_structured_proof_args(request),
                        "The user asked to close the specialist post-run proof loop without executing or bypassing runtime review.",
                    )
                ],
            )

        natural_specialist_cycle_match = re.search(
            r"^(?:(?:specialist|brain|multi[- ]brain)\s+cycle)\s+(?P<request>(?:for|about|around|regarding)\s+.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_specialist_cycle_match:
            request = _clean_specialist_natural_request(natural_specialist_cycle_match.group("request"))
            return Plan(
                goal="Ledger the full specialist proposal cycle before fresh specialist review.",
                actions=[
                    PlannedAction(
                        "specialist_cycle_ledger",
                        _specialist_structured_proof_args(request),
                        "The user asked to bind a complete specialist route/proposal/runtime/closure proof loop before starting another specialist review.",
                    )
                ],
            )

        specialist_downstream_proof_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:latest|last|newest|current)?\s*)?"
            r"(?P<kind>"
            r"specialist orchestration packet|specialist orchestration|multi brain orchestration|multi-brain orchestration|multi brain harness|multi-brain harness|brain orchestration packet|brain orchestration|"
            r"specialist action proposal contract|specialist tool action contract|specialist tool proposal contract|brain action proposal contract|model action proposal contract|multi brain action proposal contract|multi-brain action proposal contract|"
            r"specialist tool dry run packet|specialist tool dry-run packet|specialist tool dry run|specialist tool dry-run|specialist dry run packet|brain tool dry run|model tool dry run|multi brain tool dry run|multi-brain tool dry run|"
            r"specialist proposal completion gate|specialist tool completion gate|specialist action completion gate|brain proposal completion gate|model proposal completion gate|multi brain proposal completion gate|multi-brain proposal completion gate|"
            r"specialist execution handoff packet|specialist execution handoff|specialist tool execution handoff|specialist action execution handoff|brain execution handoff|model execution handoff|multi brain execution handoff|multi-brain execution handoff|"
            r"specialist post-run closure packet|specialist post run closure packet|specialist post-run closure|specialist post run closure|specialist execution closure|specialist handoff closure|brain post-run closure|model post-run closure|multi brain post-run closure|multi-brain post-run closure|"
            r"specialist cycle ledger|specialist proposal cycle ledger|specialist route cycle ledger|specialist proof ledger|multi brain cycle ledger|multi-brain cycle ledger"
            r")(?::?\s*(?P<request>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_downstream_proof_match:
            kind = specialist_downstream_proof_match.group("kind").lower()
            request = (specialist_downstream_proof_match.group("request") or "").strip()
            if not request or re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", request, re.IGNORECASE):
                request = DEFAULT_HARNESS_PACKET_REQUEST
            if "orchestration" in kind or "harness" in kind:
                tool_name = "specialist_orchestration_packet"
                goal = "Build a read-only multi-brain specialist orchestration packet."
                reason = "The user asked to route a request through a multi-brain harness flow without model calls or execution."
                args = {"request": request}
            elif "action proposal" in kind or "tool action" in kind or "tool proposal" in kind:
                tool_name = "specialist_action_proposal_contract"
                goal = "Inspect specialist draft-to-tool proposal readiness."
                reason = "The user asked to prove a specialist draft cannot become an executable tool action without ToolRegistry, exact arguments, PermissionPolicy, approval, and verification evidence."
                args = {"request": request}
            elif "dry" in kind:
                tool_name = "specialist_tool_dry_run_packet"
                goal = "Review specialist tool proposal dry-run readiness."
                reason = "The user asked to dry-run review a specialist tool proposal without executing it."
                args = {"request": request}
            elif "completion" in kind:
                tool_name = "specialist_proposal_completion_gate"
                goal = "Final-gate specialist draft-to-tool proposal completion readiness."
                reason = "The user asked to prove a specialist tool proposal is coherent enough for operator review without executing it."
                args = {"request": request}
            elif "execution handoff" in kind:
                tool_name = "specialist_execution_handoff_packet"
                goal = "Package specialist proposal for normal runtime review."
                reason = "The user asked to hand off a completed specialist proposal to the normal runtime review path without executing it."
                args = {"request": request}
            elif "closure" in kind:
                tool_name = "specialist_post_run_closure_packet"
                goal = "Close specialist runtime handoff proof after normal runtime review."
                reason = "The user asked to close the specialist post-run proof loop without executing or bypassing runtime review."
                args = _specialist_structured_proof_args(request)
            else:
                tool_name = "specialist_cycle_ledger"
                goal = "Ledger the full specialist proposal cycle before fresh specialist review."
                reason = "The user asked to bind a complete specialist route/proposal/runtime/closure proof loop before starting another specialist review."
                args = _specialist_structured_proof_args(request)
            return Plan(
                goal=goal,
                actions=[
                    PlannedAction(
                        tool_name,
                        args,
                        reason,
                    )
                ],
            )

        specialist_router_match = re.search(
            r"^(?:specialist router contract|specialist routing contract|brain router contract|multi brain router contract|multi-brain router contract|route specialist)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_router_match:
            return Plan(
                goal="Preview specialist brain routing contract.",
                actions=[
                    PlannedAction(
                        "specialist_router_contract",
                        {"request": specialist_router_match.group("request").strip()},
                        "The user asked for a read-only specialist brain route and handoff contract.",
                    )
                ],
            )

        specialist_orchestration_match = re.search(
            r"^(?:specialist orchestration packet|specialist orchestration|multi brain orchestration|multi-brain orchestration|multi brain harness|multi-brain harness|brain orchestration packet|brain orchestration)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_orchestration_match:
            return Plan(
                goal="Build a read-only multi-brain specialist orchestration packet.",
                actions=[
                    PlannedAction(
                        "specialist_orchestration_packet",
                        {"request": specialist_orchestration_match.group("request").strip()},
                        "The user asked to route a request through a multi-brain harness flow without model calls or execution.",
                    )
                ],
            )

        specialist_quality_match = re.search(
            r"^(?:specialist route quality|route quality|brain route quality|multi brain route quality|multi-brain route quality|specialist quality)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_quality_match:
            return Plan(
                goal="Measure specialist route quality before model execution.",
                actions=[
                    PlannedAction(
                        "specialist_route_quality",
                        {"request": specialist_quality_match.group("request").strip()},
                        "The user asked for a read-only specialist route confidence and verifier coverage packet.",
                    )
                ],
            )

        specialist_readiness_match = re.search(
            r"^(?:specialist execution readiness|specialist readiness|brain execution readiness|brain readiness|model execution readiness|multi brain readiness|multi-brain readiness)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_readiness_match:
            return Plan(
                goal="Gate specialist model execution readiness.",
                actions=[
                    PlannedAction(
                        "specialist_execution_readiness",
                        {"request": specialist_readiness_match.group("request").strip()},
                        "The user asked whether the request is ready for specialist model drafting or should hold for route, approval, or model setup review.",
                    )
                ],
            )

        specialist_handoff_quality_match = re.search(
            r"^(?:specialist handoff quality gate|specialist quality gate|handoff quality gate|brain handoff quality gate|model handoff quality gate|multi brain handoff quality gate|multi-brain handoff quality gate)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_handoff_quality_match:
            return Plan(
                goal="Bind specialist route quality to handoff receipt quality before proposal review.",
                actions=[
                    PlannedAction(
                        "specialist_handoff_quality_gate",
                        {"request": specialist_handoff_quality_match.group("request").strip()},
                        "The user asked to prove measured specialist handoff quality before any specialist output can approach tool proposal review.",
                    )
                ],
            )

        specialist_proposal_gate_match = re.search(
            r"^(?:specialist proposal gate|specialist tool proposal gate|brain proposal gate|model proposal gate|multi brain proposal gate|multi-brain proposal gate)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_proposal_gate_match:
            return Plan(
                goal="Gate specialist draft-to-tool proposal readiness.",
                actions=[
                    PlannedAction(
                        "specialist_proposal_gate",
                        {"request": specialist_proposal_gate_match.group("request").strip()},
                        "The user asked to prove route, readiness, handoff, verifier, model, and approval boundaries before specialist output can become tool proposals.",
                    )
                ],
            )

        specialist_model_draft_match = re.search(
            r"^(?:specialist model draft|specialist draft|brain model draft|model draft|multi brain draft|multi-brain draft)(?::?\s*(?P<request>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_model_draft_match:
            request = (specialist_model_draft_match.group("request") or "").strip()
            if not request or re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", request, re.IGNORECASE):
                request = DEFAULT_HARNESS_PACKET_REQUEST
            return Plan(
                goal="Run or safely fall back from a bounded specialist model draft.",
                actions=[
                    PlannedAction(
                        "specialist_model_draft",
                        {"request": request},
                        "The user asked for a specialist draft through the bounded model lane with route, readiness, and handoff proof gates.",
                    )
                ],
            )

        specialist_action_proposal_match = re.search(
            r"^(?:specialist action proposal contract|specialist tool action contract|specialist tool proposal contract|brain action proposal contract|model action proposal contract|multi brain action proposal contract|multi-brain action proposal contract)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_action_proposal_match:
            return Plan(
                goal="Inspect specialist draft-to-tool proposal readiness.",
                actions=[
                    PlannedAction(
                        "specialist_action_proposal_contract",
                        {"request": specialist_action_proposal_match.group("request").strip()},
                        "The user asked to prove a specialist draft cannot become an executable tool action without ToolRegistry, exact arguments, PermissionPolicy, approval, and verification evidence.",
                    )
                ],
            )

        specialist_tool_dry_run_match = re.search(
            r"^(?:specialist tool dry run|specialist dry run packet|specialist tool dry-run|brain tool dry run|model tool dry run|multi brain tool dry run|multi-brain tool dry run)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_tool_dry_run_match:
            return Plan(
                goal="Review specialist tool proposal dry-run readiness.",
                actions=[
                    PlannedAction(
                        "specialist_tool_dry_run_packet",
                        {"request": specialist_tool_dry_run_match.group("request").strip()},
                        "The user asked to dry-run review a specialist tool proposal without executing it.",
                    )
                ],
            )

        specialist_proposal_completion_match = re.search(
            r"^(?:specialist proposal completion gate|specialist tool completion gate|specialist action completion gate|brain proposal completion gate|model proposal completion gate|multi brain proposal completion gate|multi-brain proposal completion gate)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_proposal_completion_match:
            return Plan(
                goal="Final-gate specialist draft-to-tool proposal completion readiness.",
                actions=[
                    PlannedAction(
                        "specialist_proposal_completion_gate",
                        {"request": specialist_proposal_completion_match.group("request").strip()},
                        "The user asked to prove a specialist tool proposal is coherent enough for operator review without executing it.",
                    )
                ],
            )

        specialist_execution_handoff_match = re.search(
            r"^(?:specialist execution handoff|specialist execution handoff packet|specialist tool execution handoff|specialist action execution handoff|brain execution handoff|model execution handoff|multi brain execution handoff|multi-brain execution handoff)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_execution_handoff_match:
            return Plan(
                goal="Package specialist proposal for normal runtime review.",
                actions=[
                    PlannedAction(
                        "specialist_execution_handoff_packet",
                        {"request": specialist_execution_handoff_match.group("request").strip()},
                        "The user asked to hand off a completed specialist proposal to the normal runtime review path without executing it.",
                    )
                ],
            )

        specialist_cycle_ledger_match = re.search(
            r"^(?:specialist cycle ledger|specialist proposal cycle ledger|specialist route cycle ledger|specialist proof ledger|multi brain cycle ledger|multi-brain cycle ledger)(?::?\s*(?P<body>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_cycle_ledger_match:
            body = specialist_cycle_ledger_match.group("body").strip()
            key_names = [
                "verification_receipt_sha256",
                "runtime_trace_receipt",
                "runtime_trace_sha256",
                "execution_recovery_sha256",
                "execution_audit_sha256",
                "after_action_learning_sha256",
                "completion_claim_sha256",
                "verification_receipt",
                "execution_recovery",
                "execution_audit",
                "after_action_learning",
                "completion_claim",
                "recovery_sha256",
                "audit_sha256",
                "learning_sha256",
                "claim_sha256",
                "receipt_sha256",
                "runtime_trace",
                "tool_name",
                "arguments",
                "verification",
                "expected",
                "recovery",
                "learning",
                "blockers",
                "receipt",
                "audit",
                "claim",
                "trace",
                "tool",
                "args",
            ]
            keys = "|".join(re.escape(key) for key in key_names)
            field_pattern = rf"(?:^|[;\n])\s*(?P<key>{keys})(?:\s*:|\s+)(?P<value>.*?)(?=\s*(?:[;\n]\s*(?:{keys})(?:\s*:|\s+))|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            request_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                request_parts.append(body[cursor:start].strip(" ;\n"))
                cursor = end
            request_parts.append(body[cursor:].strip(" ;\n"))
            request = "; ".join(part for part in request_parts if part).strip() or body
            return Plan(
                goal="Ledger the full specialist proposal cycle before fresh specialist review.",
                actions=[
                    PlannedAction(
                        "specialist_cycle_ledger",
                        {
                            "request": request,
                            "tool": fields.get("tool") or fields.get("tool_name") or "",
                            "arguments": fields.get("arguments") or fields.get("args") or "",
                            "verification": fields.get("verification") or fields.get("expected") or "",
                            "runtime_trace": fields.get("runtime_trace") or fields.get("runtime_trace_receipt") or fields.get("trace") or "",
                            "verification_receipt": fields.get("verification_receipt") or fields.get("receipt") or "",
                            "execution_audit": fields.get("execution_audit") or fields.get("audit") or "",
                            "execution_recovery": fields.get("execution_recovery") or fields.get("recovery") or "",
                            "after_action_learning": fields.get("after_action_learning") or fields.get("learning") or "",
                            "completion_claim": fields.get("completion_claim") or fields.get("claim") or "",
                            "runtime_trace_sha256": fields.get("runtime_trace_sha256") or "",
                            "verification_receipt_sha256": fields.get("verification_receipt_sha256") or fields.get("receipt_sha256") or "",
                            "execution_audit_sha256": fields.get("execution_audit_sha256") or fields.get("audit_sha256") or "",
                            "execution_recovery_sha256": fields.get("execution_recovery_sha256") or fields.get("recovery_sha256") or "",
                            "after_action_learning_sha256": fields.get("after_action_learning_sha256") or fields.get("learning_sha256") or "",
                            "completion_claim_sha256": fields.get("completion_claim_sha256") or fields.get("claim_sha256") or "",
                            "blockers": fields.get("blockers") or "",
                        },
                        "The user asked to bind a complete specialist route/proposal/runtime/closure proof loop before starting another specialist review.",
                    )
                ],
            )

        specialist_post_run_closure_match = re.search(
            r"^(?:specialist post-run closure|specialist post run closure|specialist execution closure|specialist handoff closure|brain post-run closure|model post-run closure|multi brain post-run closure|multi-brain post-run closure)(?::?\s*(?P<body>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_post_run_closure_match:
            body = specialist_post_run_closure_match.group("body").strip()
            key_names = [
                "verification_receipt_sha256",
                "runtime_trace_receipt",
                "runtime_trace_sha256",
                "execution_recovery_sha256",
                "execution_audit_sha256",
                "after_action_learning_sha256",
                "completion_claim_sha256",
                "verification_receipt",
                "execution_recovery",
                "execution_audit",
                "after_action_learning",
                "completion_claim",
                "recovery_sha256",
                "audit_sha256",
                "learning_sha256",
                "claim_sha256",
                "receipt_sha256",
                "runtime_trace",
                "tool_name",
                "arguments",
                "verification",
                "expected",
                "recovery",
                "learning",
                "blockers",
                "receipt",
                "audit",
                "claim",
                "trace",
                "tool",
                "args",
            ]
            keys = "|".join(re.escape(key) for key in key_names)
            field_pattern = rf"(?:^|[;\n])\s*(?P<key>{keys})(?:\s*:|\s+)(?P<value>.*?)(?=\s*(?:[;\n]\s*(?:{keys})(?:\s*:|\s+))|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            request_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                request_parts.append(body[cursor:start].strip(" ;\n"))
                cursor = end
            request_parts.append(body[cursor:].strip(" ;\n"))
            request = "; ".join(part for part in request_parts if part).strip() or body
            return Plan(
                goal="Close specialist runtime handoff proof after normal runtime review.",
                actions=[
                    PlannedAction(
                        "specialist_post_run_closure_packet",
                        {
                            "request": request,
                            "tool": fields.get("tool") or fields.get("tool_name") or "",
                            "arguments": fields.get("arguments") or fields.get("args") or "",
                            "verification": fields.get("verification") or fields.get("expected") or "",
                            "runtime_trace": fields.get("runtime_trace") or fields.get("runtime_trace_receipt") or fields.get("trace") or "",
                            "verification_receipt": fields.get("verification_receipt") or fields.get("receipt") or "",
                            "execution_audit": fields.get("execution_audit") or fields.get("audit") or "",
                            "execution_recovery": fields.get("execution_recovery") or fields.get("recovery") or "",
                            "after_action_learning": fields.get("after_action_learning") or fields.get("learning") or "",
                            "completion_claim": fields.get("completion_claim") or fields.get("claim") or "",
                            "runtime_trace_sha256": fields.get("runtime_trace_sha256") or "",
                            "verification_receipt_sha256": fields.get("verification_receipt_sha256") or fields.get("receipt_sha256") or "",
                            "execution_audit_sha256": fields.get("execution_audit_sha256") or fields.get("audit_sha256") or "",
                            "execution_recovery_sha256": fields.get("execution_recovery_sha256") or fields.get("recovery_sha256") or "",
                            "after_action_learning_sha256": fields.get("after_action_learning_sha256") or fields.get("learning_sha256") or "",
                            "completion_claim_sha256": fields.get("completion_claim_sha256") or fields.get("claim_sha256") or "",
                            "blockers": fields.get("blockers") or "",
                        },
                        "The user asked to close the specialist post-run proof loop without executing or bypassing runtime review.",
                    )
                ],
            )

        natural_harness_lifecycle_match = re.search(
            r"^(?:what is the lifecycle|where is this in the lifecycle|where does this sit in the lifecycle|show lifecycle|show the lifecycle|lifecycle for)(?:\s+(?:for|about|around|regarding))?\s*:?\s*(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_harness_lifecycle_match:
            request = _clean_execution_prep_request(natural_harness_lifecycle_match.group("request"))
            return Plan(
                goal="Map the order onto Jarvis's read-only harness lifecycle state.",
                actions=[
                    PlannedAction(
                        "harness_lifecycle_state",
                        {"request": request},
                        "The user asked where an order sits in the perceive-ground-route-plan-gate-act-verify-learn harness lifecycle.",
                    )
                ],
            )

        natural_harness_control_match = re.search(
            r"^(?:how should (?:jarvis|we|i|you) control this action|show (?:the )?control surface|control surface for|control surface about|what control surface applies to|what controls apply to)\s*:?\s*(?P<request>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_harness_control_match:
            request = _clean_execution_prep_request(natural_harness_control_match.group("request"))
            return Plan(
                goal="Inspect Jarvis's read-only harness control surface for an order.",
                actions=[
                    PlannedAction(
                        "harness_control_surface",
                        {"request": request} if request else {"request": DEFAULT_HARNESS_PACKET_REQUEST},
                        "The user asked to see the engine, steering, pedals, brakes, dashboard, and proof controls before Jarvis acts.",
                    )
                ],
            )

        natural_harness_operations_match = re.search(
            r"^(?:how should (?:jarvis|we|i|you) operate on|what is the operations brief|show (?:the )?operations brief|operations brief for|operations brief about|operator brief for|operator brief about)\s*:?\s*(?P<objective>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_harness_operations_match:
            objective = _clean_execution_prep_request(natural_harness_operations_match.group("objective"))
            return Plan(
                goal="Prepare the read-only Jarvis harness operations brief.",
                actions=[
                    PlannedAction(
                        "harness_operations_brief",
                        {"objective": objective},
                        "The user asked for the next safe harness build move based on approvals, tasks, goals, and recent failures.",
                    )
                ],
            )

        harness_lifecycle_match = re.search(
            r"^(?:harness lifecycle|lifecycle state|agent lifecycle|jarvis lifecycle|harness lifecycle state)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if harness_lifecycle_match:
            request = _clean_execution_prep_request(harness_lifecycle_match.group("request"))
            return Plan(
                goal="Map the order onto Jarvis's read-only harness lifecycle state.",
                actions=[
                    PlannedAction(
                        "harness_lifecycle_state",
                        {"request": request},
                        "The user asked where an order sits in the perceive-ground-route-plan-gate-act-verify-learn harness lifecycle.",
                    )
                ],
            )

        harness_control_match = re.search(
            r"^(?:harness control|control surface|agent harness control|jarvis control surface|harness dashboard|steering packet)(?::?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if harness_control_match:
            request = _clean_execution_prep_request(harness_control_match.group("request"))
            return Plan(
                goal="Inspect Jarvis's read-only harness control surface for an order.",
                actions=[
                    PlannedAction(
                        "harness_control_surface",
                        {"request": request},
                        "The user asked to see the engine, steering, pedals, brakes, dashboard, and proof controls before Jarvis acts.",
                    )
                ],
            )

        harness_operations_match = re.search(
            r"^(?:harness operations|operations brief|operator brief|jarvis operations|harness operator brief)(?::?\s*(?P<objective>.*))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if harness_operations_match:
            objective = _clean_execution_prep_request(harness_operations_match.group("objective"))
            return Plan(
                goal="Prepare the read-only Jarvis harness operations brief.",
                actions=[
                    PlannedAction(
                        "harness_operations_brief",
                        {"objective": objective},
                        "The user asked for the next safe harness build move based on approvals, tasks, goals, and recent failures.",
                    )
                ],
            )

        model_prompt_display_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:latest|last|newest|current)?\s*(?:model planner prompt preview|planner prompt preview|model planning prompt preview|model planner prompt|planner prompt)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if model_prompt_display_match:
            return Plan(
                goal="Preview the model planner prompt.",
                actions=[
                    PlannedAction(
                        "model_planner_prompt_preview",
                        {"request": DEFAULT_HARNESS_PACKET_REQUEST},
                        "The user asked to inspect the model-planner prompt packet without calling a model.",
                    )
                ],
            )

        model_prompt_match = re.search(
            r"^(?:model planner prompt preview|planner prompt preview|model planning prompt preview|preview model planner prompt|preview planner prompt)(?::?\s*(?P<request>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if model_prompt_match:
            request = (model_prompt_match.group("request") or "").strip()
            if not request or re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", request, re.IGNORECASE):
                request = DEFAULT_HARNESS_PACKET_REQUEST
            return Plan(
                goal="Preview the model planner prompt.",
                actions=[
                    PlannedAction(
                        "model_planner_prompt_preview",
                        {"request": request},
                        "The user asked to inspect the model-planner prompt packet without calling a model.",
                    )
                ],
            )

        chat_loop_match = re.search(
            r"^(?:chat loop preview|conversation loop preview|jarvis loop preview|talk loop preview)(?::?\s*(?P<prompt>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if chat_loop_match:
            return Plan(
                goal="Preview the conversational loop.",
                actions=[
                    PlannedAction(
                        "chat_loop_preview",
                        {"prompt": chat_loop_match.group("prompt").strip()},
                        "The user asked to inspect how Jarvis routes a conversational turn.",
                    )
                ],
            )

        global_capability_command = _strip_trailing_politeness(text).lower()
        global_capability_command = re.sub(
            r"^(?:show|view|review|inspect|preview|list)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?",
            "",
            global_capability_command,
        ).strip()
        if global_capability_command in {
            "tools",
            "tool list",
            "available tools",
            "jarvis tools",
            "jarvis tool list",
            "your tools",
            "your tool list",
            "what are your tools",
            "what are jarvis tools",
            "what are jarvis's tools",
            "what tools does jarvis have",
            "what tools do you have",
        }:
            return Plan(
                goal="List available tools.",
                actions=[
                    PlannedAction(
                        "list_tools",
                        {},
                        "The user asked what functions are available.",
                    )
                ],
            )

        if global_capability_command in {
            "abilities",
            "ability list",
            "jarvis abilities",
            "jarvis ability list",
            "your abilities",
            "your ability list",
            "what are your abilities",
            "what are jarvis abilities",
            "what are jarvis's abilities",
            "what can you help me with",
            "what can jarvis help me with",
            "how can you help me",
            "how can jarvis help me",
        }:
            return Plan(
                goal="Show Jarvis capability map.",
                actions=[
                    PlannedAction(
                        "capability_map",
                        {"focus": ""},
                        "The user asked what Jarvis can do.",
                    )
                ],
            )

        topic_tool_discovery_match = re.search(
            r"^(?:show|list|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?(?:jarvis\s+)?tools\s+(?:for|about|on|around|regarding|with)\s+(?P<focus>.+)$"
            r"|^(?:jarvis\s+)?tools?\s+(?:for|about|on|around|regarding|with)\s+(?P<focus5>.+)$"
            r"|^(?P<focus2>(?!(?:show|list|view|review|inspect|preview|available|jarvis|your)\b)[a-z][a-z\s-]{1,40}?)\s+tools$"
            r"|^(?:what|which)\s+(?P<focus7>[a-z][a-z\s-]{1,40}?)\s+tools?\s+(?:do\s+you\s+have|does\s+jarvis\s+have|can\s+(?:you|jarvis)\s+use)$"
            r"|^(?:what|which)\s+tools?\s+(?:can\s+(?:you|jarvis)\s+use\s+)?(?:do\s+you\s+have\s+)?(?:for|about|on|around|regarding|with|help(?:s)?\s+with|handle(?:s)?)\s+(?P<focus3>.+)$"
            r"|^(?:what|which)\s+tools?\s+(?:do|does|handle|handles)\s+(?P<focus6>[a-z][a-z\s-]{1,40}?)$"
            r"|^(?:what|which)\s+tools?\s+(?:can\s+(?:you|jarvis)\s+use\s+)?(?:do\s+you\s+have\s+)?(?:to|for)\s+(?P<focus4>[a-z][a-z\s-]{1,40}?)$",
            text,
            re.IGNORECASE,
        )
        if topic_tool_discovery_match:
            focus = _clean_capability_focus(
                topic_tool_discovery_match.group("focus")
                or topic_tool_discovery_match.group("focus2")
                or topic_tool_discovery_match.group("focus3")
                or topic_tool_discovery_match.group("focus4")
                or topic_tool_discovery_match.group("focus5")
                or topic_tool_discovery_match.group("focus6")
                or topic_tool_discovery_match.group("focus7")
                or ""
            )
            return Plan(
                goal="Show focused Jarvis tools.",
                actions=[
                    PlannedAction(
                        "capability_map",
                        {"focus": focus},
                        "The user asked which Jarvis tools apply in a specific area.",
                    )
                ],
            )

        focused_capability_match = re.search(
            r"^(?:show|list)\s+(?:jarvis\s+)?(?:capabilities|abilities|tools)\s+(?:for|about|on)\s+(?P<focus>.+)$"
            r"|^list\s+(?P<focus4>[a-z][a-z\s-]{1,40}?)\s+tools$"
            r"|^(?:what(?:'s| is| are)?|show|tell me)\s+(?:your|jarvis(?:'s)?)?\s*(?P<focus2>[a-z][a-z\s-]{1,40}?)\s+(?:capabilities|abilities|tools)\??$"
            r"|^(?:help(?:\s+me)?|jarvis help)\s+(?:with|for|on)\s+(?P<focus3>.+)$",
            text,
            re.IGNORECASE,
        )
        if not focused_capability_match:
            focused_capability_match = re.search(
                r"^(?:what\s+can\s+(?:you|jarvis)\s+help(?:\s+me)?\s+with|how\s+can\s+(?:you|jarvis)\s+help\s+with|help\s+me\s+use)\s+(?P<focus>.+)$",
                text,
                re.IGNORECASE,
            )
        if focused_capability_match:
            focus = _clean_capability_focus(
                focused_capability_match.group("focus")
                or focused_capability_match.group("focus2")
                or focused_capability_match.group("focus3")
                or focused_capability_match.group("focus4")
                or ""
            )
            return Plan(
                goal="Show focused Jarvis capabilities.",
                actions=[
                    PlannedAction(
                        "capability_map",
                        {"focus": focus},
                        "The user asked what Jarvis can do in a specific area.",
                    )
                ],
            )

        capability_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?)?(?:capability map|capabilities|capability list|show capabilities|show jarvis capabilities|what can jarvis do|what can you do|jarvis capabilities)(?:\s+(?P<focus>.+))?$",
            text,
            re.IGNORECASE,
        )
        if capability_match:
            focus = _clean_capability_focus(capability_match.group("focus"))
            return Plan(
                goal="Show Jarvis capability map.",
                actions=[
                    PlannedAction(
                        "capability_map",
                        {"focus": focus},
                        "The user asked what Jarvis can do.",
                    )
                ],
            )

        tool_search_match = re.search(
            r"^(?:tool search|search tools|find tools|find jarvis tools|jarvis tool search)(?::?\s*(?P<query>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if tool_search_match:
            query = _clean_capability_focus(tool_search_match.group("query").strip())
            return Plan(
                goal="Search Jarvis tools.",
                actions=[
                    PlannedAction(
                        "tool_search",
                        {"query": query},
                        "The user asked to find Jarvis capabilities without running them.",
                    )
                ],
            )

        if re.fullmatch(
            r"(?:show|view|review|inspect|preview)?\s*(?:the\s+)?tool\s+health(?:\s+(?:report|please|pls|thanks|thank you))?",
            low,
        ):
            return Plan(
                goal="Summarize recent execution health before continuing autonomy.",
                actions=[
                    PlannedAction(
                        "execution_health_report",
                        {},
                        "The user asked for aggregate tool health, which belongs to the execution health diagnostic rather than one tool detail.",
                    )
                ],
            )

        natural_tool_detail_match = re.search(
            r"^(?:what\s+does|what\s+is|tell\s+me\s+about|explain)\s+(?:the\s+)?(?P<name>[a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+){0,5}?)(?:\s+(?P<tool_word>tool|function))?(?:\s+do)?\??$"
            r"|^(?:does|do)\s+(?:the\s+)?(?P<name2>[a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+){0,5}?)(?:\s+(?P<tool_word2>tool|function))?\s+(?:need|require)\s+approval\??$"
            r"|^is\s+(?:the\s+)?(?P<name3>[a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+){0,5}?)(?:\s+(?P<tool_word3>tool|function))?\s+(?:safe|risky|high\s+risk|approval\s+gated)\??$"
            r"|^(?:risk\s+(?:of|for)|approval\s+(?:of|for)|approval\s+required\s+for)\s+(?:the\s+)?(?P<name4>[a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+){0,5}?)(?:\s+(?P<tool_word4>tool|function))?\??$"
            r"|^(?:show|tell\s+me)\s+(?:the\s+)?(?:risk|approval)\s+(?:for|of)\s+(?:the\s+)?(?P<name5>[a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+){0,5}?)(?:\s+(?P<tool_word5>tool|function))?\??$"
            r"|^tool\s+(?P<name6>[a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+){0,5}?)$"
            r"|^(?P<name7>[a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+){0,5}?)\s+(?:tool|function|command)$",
            text,
            re.IGNORECASE,
        )
        if natural_tool_detail_match:
            raw_name = (
                natural_tool_detail_match.group("name")
                or natural_tool_detail_match.group("name2")
                or natural_tool_detail_match.group("name3")
                or natural_tool_detail_match.group("name4")
                or natural_tool_detail_match.group("name5")
                or natural_tool_detail_match.group("name6")
                or natural_tool_detail_match.group("name7")
                or ""
            )
            explicit_tool_word = any(
                natural_tool_detail_match.group(name)
                for name in ("tool_word", "tool_word2", "tool_word3", "tool_word4", "tool_word5", "name6", "name7")
            )
            if _looks_like_tool_detail_reference(raw_name, explicit_tool_word=explicit_tool_word):
                return Plan(
                    goal="Inspect one Jarvis tool.",
                    actions=[
                        PlannedAction(
                            "tool_detail",
                            {"name": _clean_tool_detail_name(raw_name)},
                            "The user asked about one Jarvis tool's behavior, risk, or approval boundary without running it.",
                        )
                    ],
                )

        tool_detail_match = re.search(
            r"^(?:tool detail|show tool|inspect tool|jarvis tool detail)(?::?\s*(?P<name>[a-zA-Z0-9_\-]+))$",
            text,
            re.IGNORECASE,
        )
        if tool_detail_match:
            return Plan(
                goal="Inspect one Jarvis tool.",
                actions=[
                    PlannedAction(
                        "tool_detail",
                        {"name": tool_detail_match.group("name").strip()},
                        "The user asked to inspect one Jarvis tool without running it.",
                    )
                ],
            )

        if low_command in {
            "risk",
            "risk matrix",
            "risk report",
            "risk status",
            "show risk",
            "show risk matrix",
            "show latest risk",
            "show latest risk matrix",
            "tool risk matrix",
            "jarvis risk matrix",
            "capability risk matrix",
            "approval risk matrix",
            "permissions",
            "permission status",
            "permission report",
            "jarvis permissions",
            "show permissions",
            "show my permissions",
            "show jarvis permissions",
            "show permission status",
            "what are my permissions",
            "what permissions do i have",
            "what permissions do you have",
            "what permissions does jarvis have",
        }:
            return Plan(
                goal="Show Jarvis tool risk matrix.",
                actions=[
                    PlannedAction(
                        "risk_matrix",
                        {},
                        "The user asked to inspect tool risk boundaries without running tools.",
                    )
                ],
            )

        if low in {
            "safety",
            "safety status",
            "show safety",
            "show safety status",
            "show latest safety",
            "show latest safety status",
            "jarvis safety",
            "safety check",
            "safety report",
            "is jarvis safe",
            "what's my safety status",
            "what is my safety status",
        }:
            return Plan(
                goal="Show Jarvis safety status.",
                actions=[
                    PlannedAction(
                        "safety_status",
                        {},
                        "The user asked for Jarvis safety boundaries and current risk state.",
                    )
                ],
            )

        if low in {
            "privacy",
            "privacy report",
            "privacy status",
            "show privacy",
            "show privacy report",
            "show latest privacy",
            "show latest privacy report",
            "privacy boundaries",
            "privacy boundary",
            "data boundaries",
            "what data can jarvis access",
            "what can jarvis access",
            "what data do you use",
            "what data does jarvis use",
        }:
            return Plan(
                goal="Show Jarvis privacy boundaries.",
                actions=[
                    PlannedAction(
                        "privacy_report",
                        {},
                        "The user asked what data Jarvis may access and what requires approval.",
                    )
                ],
            )

        if low in {
            "readiness",
            "readiness report",
            "readiness status",
            "show readiness",
            "show readiness report",
            "show latest readiness",
            "show latest readiness report",
            "jarvis readiness",
            "assistant readiness",
            "is jarvis ready",
            "is jarvis ready?",
            "is jarvis ready yet",
            "is jarvis ready yet?",
            "is jarvis production ready",
            "is jarvis production ready?",
            "is the harness ready",
            "is the harness ready?",
            "is the jarvis harness ready",
            "is the jarvis harness ready?",
            "is the agent harness ready",
            "is the agent harness ready?",
            "ready check",
            "readiness check",
            "what's my readiness",
            "what is my readiness",
        }:
            return Plan(
                goal="Show Jarvis readiness report.",
                actions=[
                    PlannedAction(
                        "readiness_report",
                        {},
                        "The user asked whether Jarvis is ready for safe use.",
                    )
                ],
            )

        if low_command in {
            "startup recovery report",
            "startup recovery status",
            "startup projection recovery report",
            "startup projection recovery status",
            "show startup recovery",
            "show startup recovery report",
            "did startup recovery run",
            "what did jarvis repair at startup",
        }:
            return Plan(
                goal="Show the durable startup projection recovery audit.",
                actions=[
                    PlannedAction(
                        "startup_recovery_report",
                        {},
                        "The user asked what projection repair Jarvis performed before accepting commands.",
                    )
                ],
            )

        if low in {
            "storage recovery plan",
            "jarvis storage recovery plan",
            "storage repair plan",
            "memory storage repair plan",
            "durable storage plan",
            "durable storage recovery plan",
            "fix storage plan",
        }:
            return Plan(
                goal="Draft a read-only durable-storage recovery plan.",
                actions=[
                    PlannedAction(
                        "storage_recovery_plan",
                        {},
                        "The user asked for a concrete storage recovery plan without writing files or changing runtime state.",
                    )
                ],
            )

        if low in {
            "storage recovery check",
            "jarvis storage recovery check",
            "storage bootstrap check",
            "memory recovery check",
            "durable storage check",
            "durable storage recovery check",
        }:
            return Plan(
                goal="Run Jarvis native read-only storage recovery readiness check.",
                actions=[
                    PlannedAction(
                        "storage_recovery_check",
                        {},
                        "The user asked to verify durable storage recovery readiness without writing storage.",
                    )
                ],
            )

        if low in {
            "storage status",
            "jarvis storage",
            "jarvis storage status",
            "memory storage status",
            "storage fallback status",
            "storage diagnostics",
            "memory diagnostics",
        }:
            return Plan(
                goal="Show Jarvis storage routing and fallback diagnostics.",
                actions=[
                    PlannedAction(
                        "storage_status",
                        {},
                        "The user asked about Jarvis storage, memory persistence, or fallback routing.",
                    )
                ],
            )

        if low in {
            "prototype readiness",
            "prototype checklist",
            "prototype readiness checklist",
            "show prototype readiness",
            "show prototype checklist",
            "show latest prototype readiness",
            "show latest prototype checklist",
            "jarvis prototype readiness",
            "jarvis prototype checklist",
            "can i test jarvis",
            "can i test jarvis?",
            "can i try jarvis",
            "can i try jarvis?",
            "can i use jarvis now",
            "can i use jarvis now?",
            "what can i test now",
            "what can i try now",
        }:
            return Plan(
                goal="Show Jarvis prototype readiness checklist.",
                actions=[
                    PlannedAction(
                        "prototype_readiness_checklist",
                        {},
                        "The user asked what parts of Jarvis are safe to try as a prototype now.",
                    )
                ],
            )

        if low in {
            "return brief",
            "returning brief",
            "i'm back",
            "im back",
            "i am back",
            "what did i miss",
            "where are we",
            "where are we?",
            "where were we",
            "where were we?",
            "get me up to speed",
            "bring me up to speed",
            "get me up to speed please",
            "bring me up to speed please",
        }:
            return Plan(
                goal="Show a return brief.",
                actions=[
                    PlannedAction(
                        "return_brief",
                        {},
                        "The user asked for a consolidated readiness, activity, and next-action catch-up.",
                    )
                ],
            )

        if low_command in {
            "stop",
            "stop jarvis",
            "jarvis stop",
            "stop talking",
            "stop speaking",
            "be quiet",
            "cancel jarvis",
            "cancel request",
        }:
            return Plan(
                goal="Acknowledge non-destructive stop request.",
                actions=[
                    PlannedAction(
                        "stop_jarvis",
                        {},
                        "The user asked Jarvis to stop without removing messages or changing approvals.",
                    )
                ],
            )

        work_session_packet_match = re.search(
            r"^(?:work session packet|session start packet|start packet|safe work session packet)(?:\s*:?\s*(?P<objective>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if work_session_packet_match:
            return Plan(
                goal="Build a work-session start packet.",
                actions=[
                    PlannedAction(
                        "work_session_packet",
                        {"objective": (work_session_packet_match.group("objective") or "").strip()},
                        "The user asked for a safe work-session start packet before acting.",
                    )
                ],
            )

        focus_match = re.search(
            r"^(?:(?:give me |show me |show my |i need )?(?:a\s+)?)?(?:focus brief|focus plan|focus me|work session|start work|start work session|session plan|what should i focus on(?:\s+now)?)(?:\s*:?\s*(?P<objective>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if focus_match:
            objective = _strip_trailing_politeness((focus_match.group("objective") or "").strip())
            return Plan(
                goal="Build a focus brief.",
                actions=[
                    PlannedAction(
                        "focus_brief",
                        {"objective": objective},
                        "The user asked for a safe work-session plan from current Jarvis context.",
                    )
                ],
            )

        next_session_match = re.search(
            r"^(?:next session plan|resume plan|plan next session|plan the next jarvis session|next jarvis session|next session|session resume plan|what should i do next when i return|what should we do next session|how should i restart jarvis work|how should i resume jarvis)(?:\s*:?\s*(?P<objective>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if next_session_match:
            objective = _strip_trailing_politeness((next_session_match.group("objective") or "").strip())
            return Plan(
                goal="Build a next-session resume plan.",
                actions=[
                    PlannedAction(
                        "next_session_plan",
                        {"objective": objective},
                        "The user asked for a safe plan for the next Jarvis work session.",
                    )
                ],
            )

        if low in {
            "save return brief",
            "export return brief",
            "write return brief",
            "save catch up brief",
            "save catch-up brief",
        }:
            return Plan(
                goal="Save return brief to Obsidian.",
                actions=[
                    PlannedAction(
                        "save_return_brief",
                        {},
                        "The user asked to save the consolidated return brief.",
                    )
                ],
            )

        if low in {
            "handoff",
            "handoff brief",
            "jarvis handoff",
            "handoff report",
            "resume brief",
            "handoff summary",
            "give me a handoff",
        }:
            return Plan(
                goal="Show Jarvis handoff brief.",
                actions=[
                    PlannedAction(
                        "handoff_brief",
                        {},
                        "The user asked for a consolidated safe handoff before resuming work.",
                    )
                ],
            )

        if low in {
            "save handoff",
            "save handoff brief",
            "write handoff",
            "write handoff brief",
            "export handoff",
            "export handoff brief",
            "save jarvis handoff",
        }:
            return Plan(
                goal="Save Jarvis handoff brief.",
                actions=[
                    PlannedAction(
                        "save_handoff_brief",
                        {},
                        "The user asked to save the consolidated safe handoff brief.",
                    )
                ],
            )

        if low in {
            "session closeout",
            "closeout",
            "closeout brief",
            "end session",
            "end of session",
            "wrap up session",
        }:
            return Plan(
                goal="Show session closeout.",
                actions=[
                    PlannedAction(
                        "session_closeout",
                        {},
                        "The user asked for a safe end-of-session closeout.",
                    )
                ],
            )

        if low in {
            "save session closeout",
            "write session closeout",
            "export session closeout",
            "save closeout",
            "write closeout",
            "export closeout",
        }:
            return Plan(
                goal="Save session closeout.",
                actions=[
                    PlannedAction(
                        "save_session_closeout",
                        {},
                        "The user asked to save the safe end-of-session closeout.",
                    )
                ],
            )

        autonomy_plan_match = re.search(
            r"^(?:show\s+(?:me\s+)?(?:the\s+)?)?(?:autonomy plan|safe autonomy plan|agent plan|plan autonomy|make a safe plan)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if autonomy_plan_match:
            request = _clean_harness_packet_request(autonomy_plan_match.group("request"))
            return Plan(
                goal="Draft a safe autonomy plan.",
                actions=[
                    PlannedAction(
                        "autonomy_plan",
                        {"request": request},
                        "The user asked for a safe plan before autonomous work.",
                    )
                ],
            )

        risk_preflight_match = re.search(
            r"^(?:(?:risk|safety)\s+(?:preflight|check|preview)|preflight(?:\s+check|\s+request)?)(?:\s*:?\s*(?P<request>.+))?$"
            r"|^(?:check\s+before(?:\s+you)?|before\s+you\s+check)\s+(?P<request2>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if risk_preflight_match:
            raw_request = risk_preflight_match.group("request") or risk_preflight_match.group("request2")
            request = _clean_operator_action_request(raw_request)
            return Plan(
                goal="Preflight a request for likely risk areas.",
                actions=[
                    PlannedAction(
                        "risk_preflight",
                        {"request": request},
                        "The user asked to classify likely risk before Jarvis acts.",
                    )
                ],
            )

        natural_approval_rehearsal_match = re.search(
            r"^(?:would|will)\s+(?:this|that|the\s+(?:request|command|action|order))\s+(?:need|require)\s+approval\s*:?\s*(?P<request>.+)$"
            r"|^(?:would|will|does|do)\s+(?P<request2>.+?)\s+(?:need|require)\s+approval\??$"
            r"|^(?:check|preview|show)\s+(?:the\s+)?approval(?:\s+(?:need|requirement|requirements|risk|gate|gating))?\s+(?:for|of)\s+(?P<request3>.+)$"
            r"|^approval\s+(?:preview|check|preflight)\s+(?:for\s+)?(?P<request9>.+)$"
            r"|^(?:can|could)\s+(?:you|jarvis|we|i)\s+safely\s+(?P<request4>.+)$"
            r"|^is\s+it\s+safe\s+to\s+(?P<request5>.+)$"
            r"|^(?:do|does)\s+(?:i|you|jarvis|we|the operator)\s+(?:need|require)\s+(?:(?:my|the operator's?s|owner)\s+)?approval\s+(?:to|before|for)\s+(?P<request6>.+)$"
            r"|^(?:would|will)\s+(?:i|you|jarvis|we|the operator)\s+need\s+to\s+(?:approve|review|authorize)\s+(?P<request7>.+)$"
            r"|^(?:would|will)\s+(?:this|that|the\s+(?:request|command|action|order))\s+(?:need|require)\s+(?:(?:my|the operator's?s|owner)\s+)?approval\s+(?:to|before)\s+(?P<request8>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_approval_rehearsal_match:
            raw_request = (
                natural_approval_rehearsal_match.group("request")
                or natural_approval_rehearsal_match.group("request2")
                or natural_approval_rehearsal_match.group("request3")
                or natural_approval_rehearsal_match.group("request9")
                or natural_approval_rehearsal_match.group("request4")
                or natural_approval_rehearsal_match.group("request5")
                or natural_approval_rehearsal_match.group("request6")
                or natural_approval_rehearsal_match.group("request7")
                or natural_approval_rehearsal_match.group("request8")
                or ""
            )
            guarded_question = bool(
                natural_approval_rehearsal_match.group("request4")
                or natural_approval_rehearsal_match.group("request5")
                or natural_approval_rehearsal_match.group("request6")
                or natural_approval_rehearsal_match.group("request7")
                or natural_approval_rehearsal_match.group("request8")
            )
            request = _clean_operator_action_request(raw_request)
            existing_approval_readiness_request = re.fullmatch(
                r"(?:approve|run|rerun)\s+(?:approval\s*)?#?(?:\d+|latest|last|newest|current)",
                request,
                re.IGNORECASE,
            )
            if _looks_like_tool_detail_reference(request):
                return Plan(
                    goal="Inspect one Jarvis tool.",
                    actions=[
                        PlannedAction(
                            "tool_detail",
                            {"name": _clean_tool_detail_name(request)},
                            "The user asked about one Jarvis tool's approval boundary without running it.",
                        )
                    ],
                )
            if (not guarded_question or _looks_like_operator_action_request(request)) and not existing_approval_readiness_request:
                return Plan(
                    goal="Preview a request without executing it.",
                    actions=[
                        PlannedAction(
                            "action_rehearsal",
                            {"request": request},
                            "The user asked whether a request would be safe or approval-gated, so Jarvis should preview it without executing tools.",
                        )
                    ],
                )

        action_readiness_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:action readiness|readiness packet|readiness check|go no go|go/no-go|go or no go|can jarvis safely do this|is this a go|ready to)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if action_readiness_match:
            request = _clean_operator_action_request(action_readiness_match.group("request"))
            if text.lower().startswith("ready to ") and not _looks_like_operator_action_request(request):
                pass
            else:
                return Plan(
                    goal="Check proposed action readiness.",
                    actions=[
                        PlannedAction(
                            "action_readiness_packet",
                            {"request": request},
                            "The user asked whether a proposed action is ready, needs preflight, or should stop.",
                        )
                    ],
                )

        natural_ready_to_conversation = re.search(
            r"^ready to\s+(?:talk|chat|discuss|think|reflect)\b",
            text,
            re.IGNORECASE,
        )
        if natural_ready_to_conversation:
            return Plan(goal="No deterministic tool route.", actions=[])

        execution_contract_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:execution contract|action contract|harness contract|jarvis execution contract|order contract)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_contract_match:
            request = _clean_harness_packet_request(execution_contract_match.group("request"))
            return Plan(
                goal="Build a read-only execution contract for the order.",
                actions=[
                    PlannedAction(
                        "execution_contract",
                        {"request": request},
                        "The user asked Jarvis to contract route, safety gates, verification, recovery, and learning before execution.",
                    )
                ],
            )

        argument_contract_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:argument contract|args contract|tool argument contract|tool args contract|argument packet|args packet|tool argument packet|tool args packet)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if argument_contract_match:
            request = _clean_harness_packet_request(argument_contract_match.group("request"))
            return Plan(
                goal="Build a read-only tool-argument contract packet.",
                actions=[
                    PlannedAction(
                        "argument_contract_packet",
                        {"request": request},
                        "The user asked Jarvis to prove exact planned tool arguments before execution or approval.",
                    )
                ],
            )

        verification_packet_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:verification packet|verify packet|verification plan|proof packet|proof plan|evidence packet|evidence plan|how will jarvis verify)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if verification_packet_match:
            request = _clean_harness_packet_request(verification_packet_match.group("request"))
            return Plan(
                goal="Build a read-only verification packet for the order.",
                actions=[
                    PlannedAction(
                        "verification_packet",
                        {"request": request},
                        "The user asked Jarvis to define evidence, failure signals, and recovery before or after execution.",
                    )
                ],
            )

        natural_verification_packet_match = re.search(
            r"^(?:how (?:will|would|should|do) (?:jarvis|we|i) verify(?: that)?|what evidence (?:will|would|should)?\s*prove(?:s)?|what proof (?:will|would|should)?\s*prove(?:s)?|what would count as proof for|how do we prove)\s*:?\s*(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_verification_packet_match:
            return Plan(
                goal="Build a read-only verification packet for the order.",
                actions=[
                    PlannedAction(
                        "verification_packet",
                        {"request": natural_verification_packet_match.group("request").strip()},
                        "The user asked in natural language what evidence would prove the work.",
                    )
                ],
            )

        acceptance_gate_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:acceptance gate|execution acceptance|done gate|completion acceptance|acceptance check|can jarvis call this done)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if acceptance_gate_match:
            raw_request = _clean_harness_packet_request(acceptance_gate_match.group("request"))
            request_parts = [part.strip() for part in raw_request.split(";") if part.strip()]
            request_text = request_parts[0] if request_parts else raw_request
            args = {"request": request_text}
            for part in request_parts[1:]:
                key, _, value = part.partition(" ")
                key = key.strip().lower().rstrip(":")
                value = value.strip()
                if key in {"evidence", "proof", "receipt"}:
                    args["evidence"] = value
                elif key in {"tests", "test", "verification"}:
                    args["tests"] = value
                elif key in {"rollback", "recovery"}:
                    args["rollback"] = value
            return Plan(
                goal="Gate whether a Jarvis behavior is accepted as done.",
                actions=[
                    PlannedAction(
                        "execution_acceptance_gate",
                        args,
                        "The user asked Jarvis to check route, approval, evidence, tests, audit, and recovery before treating work as complete.",
                    )
                ],
            )

        natural_acceptance_inline_request = ""
        for natural_acceptance_inline_pattern in (
            r"^(?:can|should) (?:we|i|jarvis) call (?P<request>.+?) done\s*[?.!]*$",
            r"^(?:can|should) (?:we|i|jarvis) mark (?P<request>.+?) (?:as )?done\s*[?.!]*$",
            r"^is (?P<request>.+?) accepted as done\s*[?.!]*$",
            r"^is (?P<request>.+?) ready to accept\s*[?.!]*$",
        ):
            natural_acceptance_inline_match = re.search(
                natural_acceptance_inline_pattern,
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if natural_acceptance_inline_match:
                natural_acceptance_inline_request = _clean_operator_action_request(
                    natural_acceptance_inline_match.group("request")
                )
                break
        if natural_acceptance_inline_request and _looks_like_operator_action_request(natural_acceptance_inline_request):
            return Plan(
                goal="Gate whether a Jarvis behavior is accepted as done.",
                actions=[
                    PlannedAction(
                        "execution_acceptance_gate",
                        {"request": natural_acceptance_inline_request},
                        "The user asked in natural language whether an action has enough evidence to be accepted as done.",
                    )
                ],
            )

        natural_acceptance_gate_match = re.search(
            r"^(?:can (?:we|i|jarvis) call this done|is this accepted as done|is this ready to accept|should (?:we|i|jarvis) accept this as done|can (?:we|i|jarvis) mark this action done)\s*:?\s*(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_acceptance_gate_match:
            raw_request = natural_acceptance_gate_match.group("request").strip()
            request_parts = [part.strip() for part in raw_request.split(";") if part.strip()]
            request_text = request_parts[0] if request_parts else raw_request
            args = {"request": request_text}
            for part in request_parts[1:]:
                key, _, value = part.partition(" ")
                key = key.strip().lower().rstrip(":")
                value = value.strip()
                if key in {"evidence", "proof", "receipt"}:
                    args["evidence"] = value
                elif key in {"tests", "test", "verification"}:
                    args["tests"] = value
                elif key in {"rollback", "recovery"}:
                    args["rollback"] = value
            return Plan(
                goal="Gate whether a Jarvis behavior is accepted as done.",
                actions=[
                    PlannedAction(
                        "execution_acceptance_gate",
                        args,
                        "The user asked in natural language whether an action has enough evidence to be accepted as done.",
                    )
                ],
            )

        execution_readiness_matrix_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:execution readiness matrix|readiness matrix|go no go matrix|go/no-go matrix|harness readiness matrix|action matrix|execution matrix)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_readiness_matrix_match:
            request = _clean_harness_packet_request(execution_readiness_matrix_match.group("request"))
            return Plan(
                goal="Build a read-only execution readiness matrix.",
                actions=[
                    PlannedAction(
                        "execution_readiness_matrix",
                        {"request": request},
                        "The user asked for a harness matrix covering route, risk, approval, proof, recovery, and go/no-go state.",
                    )
                ],
            )

        dispatch_decision_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:dispatch decision packet|dispatch decision|dispatch packet|command dispatch|order dispatch|go no go dispatch|go/no-go dispatch|dispatch)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if dispatch_decision_match:
            request = _clean_harness_packet_request(dispatch_decision_match.group("request"))
            return Plan(
                goal="Build a read-only dispatch decision packet.",
                actions=[
                    PlannedAction(
                        "dispatch_decision_packet",
                        {"request": request},
                        "The user asked Jarvis to choose whether an order should become chat, auto-safe tools, approval-gated work, a hold, or a clarification request.",
                    )
                ],
            )

        execution_governor_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:execution governor packet|execution governor|governor check|governor packet|go no go governor|go/no-go governor|jarvis governor|command governor|harness governor|governor|govern)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if execution_governor_match:
            request = _clean_operator_action_request(execution_governor_match.group("request"))
            request = re.sub(r"^check\s+", "", request, count=1, flags=re.IGNORECASE).strip()
            return Plan(
                goal="Build a read-only execution governor packet.",
                actions=[
                    PlannedAction(
                        "execution_governor_packet",
                        {"request": request},
                        "The user asked Jarvis for a single go/no-go verdict across intake, dispatch, proof, recovery, and approval gates.",
                    )
                ],
            )

        command_cockpit_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:command cockpit|cockpit packet|command cockpit packet|jarvis cockpit|harness cockpit|order cockpit|command dashboard|order dashboard)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if command_cockpit_match:
            request = _clean_harness_packet_request(command_cockpit_match.group("request"))
            return Plan(
                goal="Build a read-only command cockpit packet.",
                actions=[
                    PlannedAction(
                        "command_cockpit_packet",
                        {"request": request},
                        "The user asked Jarvis for one command-first dashboard across intake, governor, dispatch, readiness, verification, recovery, and learning state.",
                    )
                ],
            )

        command_intake_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:command intake|order intake|intake packet|command intake packet|jarvis intake|speech intake|text intake)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if command_intake_match:
            request = _clean_harness_packet_request(command_intake_match.group("request"))
            return Plan(
                goal="Build a read-only command intake packet.",
                actions=[
                    PlannedAction(
                        "command_intake_packet",
                        {"request": request},
                        "The user asked Jarvis to turn a natural order into route, risk, approval, next-command, and proof metadata before execution.",
                    )
                ],
            )

        planner_gap_match = re.search(
            rf"^{HARNESS_PACKET_DISPLAY_PREFIX}(?:planner gap|route gap|planning gap|planner gap packet|route gap packet|gap check|gap packet)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if planner_gap_match:
            request = _clean_harness_packet_request(planner_gap_match.group("request"))
            return Plan(
                goal="Build a read-only planner gap packet.",
                actions=[
                    PlannedAction(
                        "planner_gap_packet",
                        {"request": request},
                        "The user asked Jarvis to detect whether the steering layer has an exact safe route or needs a planner pattern, tool contract, or smoke test.",
                    )
                ],
            )

        natural_run_verification_match = re.search(
            r"^(?:receipt|verification|proof)\s+for\s+(?:tool\s+)?run\s*#?(?P<run_id_a>\d+|latest|last|newest|current)\??$"
            r"|^(?:prove|verify|check)\s+(?:tool\s+)?run\s*#?(?P<run_id_b>\d+|latest|last|newest|current)\??$"
            r"|^(?:what happened (?:in|with|during)|show(?: me)? (?:the )?(?:proof|receipt|verification) for|trace)\s+(?:tool\s+)?run\s*#?(?P<run_id_c>\d+|latest|last|newest|current)\??$",
            text,
            re.IGNORECASE,
        )
        if natural_run_verification_match:
            run_id = (
                natural_run_verification_match.group("run_id_a")
                or natural_run_verification_match.group("run_id_b")
                or natural_run_verification_match.group("run_id_c")
                or "latest"
            ).strip()
            return Plan(
                goal="Build a read-only after-action verification receipt.",
                actions=[
                    PlannedAction(
                        "verification_receipt",
                        {"run_id": run_id, "expectation": ""},
                        "The user asked in natural language for proof of what happened in one logged tool run.",
                    )
                ],
            )

        natural_message_trace_match = re.search(
            r"^(?:runtime trace|runtime trace receipt|trace receipt|trace)\s+(?:for\s+)?(?:message|assistant message|turn)\s*#?(?P<message_id>\d+)\??$",
            text,
            re.IGNORECASE,
        )
        if natural_message_trace_match:
            message_id = natural_message_trace_match.group("message_id").strip()
            return Plan(
                goal="Show a stored Jarvis runtime trace receipt.",
                actions=[
                    PlannedAction(
                        "runtime_trace_receipt",
                        {"message_id": message_id},
                        "The user asked to inspect a stored runtime trace for one assistant message.",
                    )
                ],
            )

        natural_run_audit_match = re.search(
            r"^(?:audit|execution audit|audit gate|run integrity|run integrity report|tool run integrity)\s+(?:for\s+)?(?:tool\s+)?run\s*#?(?P<run_id>\d+|latest|last|newest|current)\??$",
            text,
            re.IGNORECASE,
        )
        if natural_run_audit_match:
            return Plan(
                goal="Scan recent tool runs for execution-integrity blockers.",
                actions=[
                    PlannedAction(
                        "execution_audit_gate",
                        {},
                        "The user asked in natural language for an execution-integrity audit around a logged run.",
                    )
                ],
            )

        verification_receipt_display_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:latest|last|newest|current)\s+)?(?:verification|verification receipts?|verify receipts?|tool run receipts?|after[- ]action receipts?)(?:\s+(?:please|pls|thanks|thank you))?$",
            text,
            re.IGNORECASE,
        )
        if verification_receipt_display_match:
            return Plan(
                goal="Build a read-only after-action verification receipt.",
                actions=[
                    PlannedAction(
                        "verification_receipt",
                        {"run_id": "latest", "expectation": ""},
                        "The user asked to inspect the latest verification receipt without adding an expectation.",
                    )
                ],
            )

        natural_verification_receipt_match = re.search(
            r"^(?:verification receipt|verify receipt|tool run receipt|after[- ]action receipt)\s+(?:for\s+)?(?:run\s+)?#?(?P<run_id_a>\d+|latest|last|newest|current)(?:\s*:?\s*(?P<expectation_a>.*))?$"
            r"|^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:verification\s+)?(?:receipt|verification receipt|tool run receipt)\s+(?:for\s+)?(?:run\s+)?#?(?P<run_id_b>\d+|latest|last|newest|current)(?:\s*:?\s*(?P<expectation_b>.*))?$"
            r"|^(?:did|does|was|is)\s+(?:the\s+)?(?:run\s+)?#?(?P<run_id_c>\d+|latest|last|newest|current)\s+(?:pass\s+verification|verified|verification\s+passed)\??$"
            r"|^(?:verify|check\s+verification\s+for|show\s+verification\s+for)\s+(?:run\s+)?#?(?P<run_id_d>\d+|latest|last|newest|current)\??$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_verification_receipt_match:
            run_id = (
                natural_verification_receipt_match.group("run_id_a")
                or natural_verification_receipt_match.group("run_id_b")
                or natural_verification_receipt_match.group("run_id_c")
                or natural_verification_receipt_match.group("run_id_d")
                or "latest"
            ).strip()
            expectation = _strip_trailing_politeness(
                (
                    natural_verification_receipt_match.group("expectation_a")
                    or natural_verification_receipt_match.group("expectation_b")
                    or ""
                ).strip()
            ).strip()
            return Plan(
                goal="Build a read-only after-action verification receipt.",
                actions=[
                    PlannedAction(
                        "verification_receipt",
                        {"run_id": run_id, "expectation": expectation},
                        "The user asked in natural language to inspect whether one logged tool run has verification evidence.",
                    )
                ],
            )

        verification_receipt_match = re.search(
            r"^(?:verification receipt|verify receipt|tool run receipt|after action receipt|after-action receipt)\s*#?(?P<run_id>\d+|latest|last|newest|current)?\s*:?\s*(?P<expectation>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if verification_receipt_match:
            run_id = (verification_receipt_match.group("run_id") or "latest").strip()
            expectation = _strip_trailing_politeness((verification_receipt_match.group("expectation") or "").strip()).strip()
            return Plan(
                goal="Build a read-only after-action verification receipt.",
                actions=[
                    PlannedAction(
                        "verification_receipt",
                        {"run_id": run_id, "expectation": expectation},
                        "The user asked Jarvis to verify one logged tool run with audit evidence before claiming completion.",
                    )
                ],
            )

        runtime_trace_match = re.search(
            r"^(?:runtime trace|runtime trace receipt|trace receipt|latest runtime trace|latest trace receipt|harness trace receipt)\s*#?(?P<message_id>\d+)?$",
            text,
            re.IGNORECASE,
        )
        if runtime_trace_match:
            message_id = (runtime_trace_match.group("message_id") or "").strip()
            return Plan(
                goal="Show a stored Jarvis runtime trace receipt.",
                actions=[
                    PlannedAction(
                        "runtime_trace_receipt",
                        {"message_id": message_id} if message_id else {},
                        "The user asked to inspect a stored runtime route, lifecycle stages, approvals, and tool-result evidence.",
                    )
                ],
            )

        execution_audit_match = re.search(
            r"^(?:execution audit gate|execution audit|run integrity|run integrity report|tool run integrity|audit gate)(?:\s*:?\s*(?P<limit>\d+))?$",
            text,
            re.IGNORECASE,
        )
        if execution_audit_match:
            limit = (execution_audit_match.group("limit") or "").strip()
            args = {"limit": limit} if limit else {}
            return Plan(
                goal="Scan recent tool runs for execution-integrity blockers.",
                actions=[
                    PlannedAction(
                        "execution_audit_gate",
                        args,
                        "The user asked to inspect recent tool runs for failed execution, missing approval evidence, and recovery needs.",
                    )
                ],
            )

        if low in {
            "recovery",
            "recovery please",
            "show recovery",
            "show recovery please",
            "show latest recovery",
            "show latest recovery please",
        }:
            return Plan(
                goal="Build a read-only recovery packet for the latest problem run.",
                actions=[
                    PlannedAction(
                        "execution_recovery_packet",
                        {},
                        "The user asked for a safe recovery route after a failed, blocked, or weakly evidenced execution.",
                    )
                ],
            )

        execution_recovery_match = re.search(
            r"^(?:execution recovery packet|execution recovery|recovery packet|recovery plan|recover execution|recover run|run recovery)\s*#?(?P<run_id>\d+|latest|last|newest|current)?\s*:?\s*(?P<limit>\d+)?$",
            text,
            re.IGNORECASE,
        )
        if execution_recovery_match:
            run_id = (execution_recovery_match.group("run_id") or "").strip()
            limit = (execution_recovery_match.group("limit") or "").strip()
            args = {}
            if run_id and run_id.lower() not in {"latest", "last", "newest", "current"}:
                args["run_id"] = run_id
            if limit:
                args["limit"] = limit
            return Plan(
                goal="Build a read-only recovery packet for the latest problem run.",
                actions=[
                    PlannedAction(
                        "execution_recovery_packet",
                        args,
                        "The user asked for a safe recovery route after a failed, blocked, or weakly evidenced execution.",
                    )
                ],
            )

        natural_run_recovery_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:execution\s+)?(?:recovery|recovery packet|recovery plan)\s+for\s+(?:run\s*)?#?(?P<run_id_a>\d+|latest|last|newest|current)$"
            r"|^(?:recovery|recovery packet|recovery plan|execution recovery|execution recovery packet)\s+for\s+(?:run\s*)?#?(?P<run_id_f>\d+|latest|last|newest|current)$"
            r"|^how\s+(?:do|should)\s+(?:jarvis|we|i)\s+recover(?:\s+from)?\s+(?:the\s+)?(?:failed|blocked|stuck)?\s*(?:run\s*)?#?(?P<run_id_b>\d+|latest|last|newest|current)\??$"
            r"|^what should (?:jarvis|we|i) do after (?:run\s*)?#?(?P<run_id_c>\d+|latest|last|newest|current)\s+(?:failed|blocked|got stuck)\??$"
            r"|^what should (?:jarvis|we|i) do after (?:the\s+)?(?:failed|blocked|stuck)\s+(?:run\s*)?#?(?P<run_id_d>\d+|latest|last|newest|current)\??$"
            r"|^(?:what failed in|why did)\s+(?:run\s*)?#?(?P<run_id_e>\d+|latest|last|newest|current)(?:\s+fail)?\??$",
            text,
            re.IGNORECASE,
        )
        if natural_run_recovery_match:
            run_id = (
                natural_run_recovery_match.group("run_id_a")
                or natural_run_recovery_match.group("run_id_f")
                or natural_run_recovery_match.group("run_id_b")
                or natural_run_recovery_match.group("run_id_c")
                or natural_run_recovery_match.group("run_id_d")
                or natural_run_recovery_match.group("run_id_e")
                or ""
            ).strip()
            args = {}
            if run_id and run_id.lower() not in {"latest", "last", "newest", "current"}:
                args["run_id"] = run_id
            return Plan(
                goal="Build a read-only recovery packet for one logged run.",
                actions=[
                    PlannedAction(
                        "execution_recovery_packet",
                        args,
                        "The user asked in natural language how to inspect or recover one logged failed, blocked, or stuck run.",
                    )
                ],
            )

        natural_recovery_match = re.search(
            r"^(?:what should (?:jarvis|we|i) do after (?:the )?(?:failed|blocked|stuck) (?:run|execution)|what(?:'s| is) (?:the )?(?:next )?(?:safe )?recovery (?:step|route|command|packet|queue)(?: for (?:run\s*)?#?(?P<run_id_a>\d+|latest|last|newest|current))?|how should (?:jarvis|we|i) recover(?: from (?:the )?(?:failed|blocked|stuck) (?:run|execution|tool run))?|show (?:the )?recovery queue)(?:\s*:?\s*(?P<limit>\d+))?$",
            text,
            re.IGNORECASE,
        )
        if natural_recovery_match:
            run_id = (natural_recovery_match.group("run_id_a") or "").strip()
            limit = (natural_recovery_match.group("limit") or "").strip()
            args = {}
            if run_id and run_id.lower() not in {"latest", "last", "newest", "current"}:
                args["run_id"] = run_id
            if limit:
                args["limit"] = limit
            return Plan(
                goal="Build a read-only recovery packet for the latest problem run.",
                actions=[
                    PlannedAction(
                        "execution_recovery_packet",
                        args,
                        "The user asked in natural language what safe recovery step should follow a failed, blocked, or stuck execution.",
                    )
                ],
            )

        natural_show_learning_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?)?(?:after[- ]action\s+)?learning(?: packet)?\s+for\s+(?:run\s*)?#?(?P<run_id>\d+|latest|last|newest|current)$"
            r"|^(?:after[- ]action|after[- ]action learning(?: packet)?)\s+for\s+(?:run\s*)?#?(?P<run_id_b>\d+|latest|last|newest|current)$",
            text,
            re.IGNORECASE,
        )
        if natural_show_learning_match:
            run_id = (natural_show_learning_match.group("run_id") or natural_show_learning_match.group("run_id_b") or "").strip()
            args = {}
            if run_id and run_id.lower() not in {"latest", "last", "newest", "current"}:
                args["run_id"] = run_id
            return Plan(
                goal="Build a read-only after-action learning packet from execution audit evidence.",
                actions=[
                    PlannedAction(
                        "after_action_learning_packet",
                        args,
                        "The user asked to inspect after-action learning for one logged run.",
                    )
                ],
            )

        after_action_learning_match = re.search(
            r"^(?:after action learning packet|after-action learning packet|after action learning|after-action learning|execution learning packet|run learning packet|learn from run)\s*#?(?P<run_id>\d+|latest|last|newest|current)?\s*:?\s*(?P<limit>\d+)?$",
            text,
            re.IGNORECASE,
        )
        if after_action_learning_match:
            run_id = (after_action_learning_match.group("run_id") or "").strip()
            limit = (after_action_learning_match.group("limit") or "").strip()
            args = {}
            if run_id and run_id.lower() not in {"latest", "last", "newest", "current"}:
                args["run_id"] = run_id
            if limit:
                args["limit"] = limit
            return Plan(
                goal="Build a read-only after-action learning packet from execution audit evidence.",
                actions=[
                    PlannedAction(
                        "after_action_learning_packet",
                        args,
                        "The user asked Jarvis to convert a logged execution into reviewable learning candidates.",
                    )
                ],
            )

        natural_after_action_learning_match = re.search(
            r"^(?:what (?:can|should|did) (?:jarvis|we|i)?\s*learn from (?:run\s*)?#?(?P<run_id_a>\d+|latest|last|newest|current)?|what should become (?:a )?(?:test|task|skill|memory) from (?:run\s*)?#?(?P<run_id_b>\d+|latest|last|newest|current)?|promote (?:the )?(?:failed|blocked|latest|last)?\s*(?:run\s*)?#?(?P<run_id_c>\d+|latest|last|newest|current)? to learning review)(?:\s*:?\s*(?P<limit>\d+))?$",
            text,
            re.IGNORECASE,
        )
        if natural_after_action_learning_match:
            run_id = (
                natural_after_action_learning_match.group("run_id_a")
                or natural_after_action_learning_match.group("run_id_b")
                or natural_after_action_learning_match.group("run_id_c")
                or ""
            ).strip()
            limit = (natural_after_action_learning_match.group("limit") or "").strip()
            args = {}
            if run_id and run_id.lower() not in {"latest", "last", "newest", "current"}:
                args["run_id"] = run_id
            if limit:
                args["limit"] = limit
            return Plan(
                goal="Build a read-only after-action learning packet from execution audit evidence.",
                actions=[
                    PlannedAction(
                        "after_action_learning_packet",
                        args,
                        "The user asked in natural language what Jarvis can learn from a logged execution before promoting it.",
                    )
                ],
            )

        natural_learning_closure_match = re.search(
            r"^(?:close|show|view|review|inspect|preview)\s+(?:the\s+)?(?:execution\s+)?learning(?:\s+debt|\s+closure)?(?:\s+packet)?\s+for\s+(?:run\s*)?#?(?P<run_id>\d+|latest|last|newest|current)$",
            text,
            re.IGNORECASE,
        )
        if natural_learning_closure_match:
            run_id = (natural_learning_closure_match.group("run_id") or "").strip()
            args = {}
            if run_id and run_id.lower() not in {"latest", "last", "newest", "current"}:
                args["run_id"] = run_id
            return Plan(
                goal="Check whether execution learning debt is closed.",
                actions=[
                    PlannedAction(
                        "execution_learning_closure_packet",
                        args,
                        "The user asked to inspect learning-closure status for one logged run.",
                    )
                ],
            )

        natural_learning_closed_question_match = re.search(
            r"^(?:is|was)\s+(?:run\s*)?#?(?P<run_id_a>\d+|latest|last|newest|current)\s+(?:execution\s+)?learning\s+(?:closed|done|complete)\??$"
            r"|^(?:is|was)\s+(?:the\s+)?(?:execution\s+)?learning\s+(?:closed|done|complete)\s+for\s+(?:run\s*)?#?(?P<run_id_b>\d+|latest|last|newest|current)\??$"
            r"|^(?:execution\s+)?learning\s+closure\s+for\s+(?:run\s*)?#?(?P<run_id_c>\d+|latest|last|newest|current)\??$",
            text,
            re.IGNORECASE,
        )
        if natural_learning_closed_question_match:
            run_id = (
                natural_learning_closed_question_match.group("run_id_a")
                or natural_learning_closed_question_match.group("run_id_b")
                or natural_learning_closed_question_match.group("run_id_c")
                or ""
            ).strip()
            args = {}
            if run_id and run_id.lower() not in {"latest", "last", "newest", "current"}:
                args["run_id"] = run_id
            return Plan(
                goal="Check whether execution learning debt is closed.",
                actions=[
                    PlannedAction(
                        "execution_learning_closure_packet",
                        args,
                        "The user asked in natural language whether learning debt is closed for one logged run.",
                    )
                ],
            )

        execution_learning_closure_match = re.search(
            r"^(?:execution learning closure|execution learning closure packet|learning closure packet|learning closure|close learning debt|close execution learning|execution learning debt closure)\s*#?(?P<run_id>\d+|latest|last|newest|current)?\s*:?\s*(?P<limit>\d+)?$",
            text,
            re.IGNORECASE,
        )
        if execution_learning_closure_match:
            run_id = (execution_learning_closure_match.group("run_id") or "").strip()
            limit = (execution_learning_closure_match.group("limit") or "").strip()
            args = {}
            if run_id and run_id.lower() not in {"latest", "last", "newest", "current"}:
                args["run_id"] = run_id
            if limit:
                args["limit"] = limit
            return Plan(
                goal="Check whether execution learning debt is closed.",
                actions=[
                    PlannedAction(
                        "execution_learning_closure_packet",
                        args,
                        "The user asked for a read-only closure gate over verification, recovery, after-action learning, and repeated-failure promotion proof.",
                    )
                ],
            )

        reversed_latest_execution_proof_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?P<run_id>latest|last|newest|current)\s+(?P<kind>verification receipt|verify receipt|tool run receipt|execution recovery|recovery packet|recovery plan|after[- ]action learning(?: packet)?|execution learning closure(?: packet)?|learning closure(?: packet)?)$",
            text,
            re.IGNORECASE,
        )
        if reversed_latest_execution_proof_match:
            run_id = reversed_latest_execution_proof_match.group("run_id").lower()
            kind = reversed_latest_execution_proof_match.group("kind").lower().replace("-", " ")
            if kind in {"verification receipt", "verify receipt", "tool run receipt"}:
                return Plan(
                    goal="Build a read-only after-action verification receipt.",
                    actions=[
                        PlannedAction(
                            "verification_receipt",
                            {"run_id": _approval_id_arg(run_id), "expectation": ""},
                            "The user asked to inspect the latest verification receipt in reversed natural wording.",
                        )
                    ],
                )
            if kind in {"execution recovery", "recovery packet", "recovery plan"}:
                return Plan(
                    goal="Build a read-only recovery packet for the latest problem run.",
                    actions=[
                        PlannedAction(
                            "execution_recovery_packet",
                            {},
                            "The user asked to inspect the latest recovery packet in reversed natural wording.",
                        )
                    ],
                )
            if kind in {"after action learning", "after action learning packet"}:
                return Plan(
                    goal="Build a read-only after-action learning packet from execution audit evidence.",
                    actions=[
                        PlannedAction(
                            "after_action_learning_packet",
                            {},
                            "The user asked to inspect the latest after-action learning packet in reversed natural wording.",
                        )
                    ],
                )
            return Plan(
                goal="Check whether execution learning debt is closed.",
                actions=[
                    PlannedAction(
                        "execution_learning_closure_packet",
                        {},
                        "The user asked to inspect the latest execution learning closure in reversed natural wording.",
                    )
                ],
            )

        reversed_latest_diagnostic_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:latest|last|newest|current)\s+(?P<kind>runtime trace(?: receipt)?|trace receipt|execution audit(?: gate)?|audit gate|run integrity(?: report)?|execution health(?: report)?|runtime health|run health|tool health|recovery closure(?: checklist)?|execution recovery closure|execution closure checklist)$",
            text,
            re.IGNORECASE,
        )
        if reversed_latest_diagnostic_match:
            kind = reversed_latest_diagnostic_match.group("kind").lower()
            if kind in {"runtime trace", "runtime trace receipt", "trace receipt"}:
                return Plan(
                    goal="Show a stored Jarvis runtime trace receipt.",
                    actions=[
                        PlannedAction(
                            "runtime_trace_receipt",
                            {},
                            "The user asked to inspect the latest runtime trace in reversed natural wording.",
                        )
                    ],
                )
            if kind in {"execution audit", "execution audit gate", "audit gate", "run integrity", "run integrity report"}:
                return Plan(
                    goal="Scan recent tool runs for execution-integrity blockers.",
                    actions=[
                        PlannedAction(
                            "execution_audit_gate",
                            {},
                            "The user asked to inspect the latest execution audit in reversed natural wording.",
                        )
                    ],
                )
            if kind in {"recovery closure", "recovery closure checklist", "execution recovery closure", "execution closure checklist"}:
                return Plan(
                    goal="Show the read-only recovery closure checklist before retry or completion.",
                    actions=[
                        PlannedAction(
                            "recovery_closure_checklist",
                            {},
                            "The user asked to inspect the latest recovery closure checklist in reversed natural wording.",
                        )
                    ],
                )
            return Plan(
                goal="Summarize recent execution health before continuing autonomy.",
                actions=[
                    PlannedAction(
                        "execution_health_report",
                        {},
                        "The user asked to inspect the latest execution health diagnostic in reversed natural wording.",
                    )
                ],
            )

        reversed_latest_tool_runs_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:latest|last|newest|current)\s+(?:recent\s+)?(?P<kind>tool runs|tool activity|tool audit(?: log)?|audit log|recent audit)$",
            text,
            re.IGNORECASE,
        )
        if reversed_latest_tool_runs_match:
            return Plan(
                goal="Show recent tool audit log.",
                actions=[
                    PlannedAction(
                        "recent_tool_runs",
                        {},
                        "The user asked to inspect recent tool activity in reversed natural wording.",
                    )
                ],
            )

        execution_health_match = re.search(
            r"^(?:execution health report|execution health|runtime health|run health|tool health|action health|execution health check)\s*:?\s*(?P<limit>\d+)?$",
            text,
            re.IGNORECASE,
        )
        if execution_health_match:
            limit = (execution_health_match.group("limit") or "").strip()
            args = {"limit": limit} if limit else {}
            return Plan(
                goal="Summarize recent execution health before continuing autonomy.",
                actions=[
                    PlannedAction(
                        "execution_health_report",
                        args,
                        "The user asked for aggregate runtime health across action runs, repeated failures, approvals, verification, and next safe audit command.",
                    )
                ],
            )

        natural_execution_health_match = re.search(
            r"^(?:is execution healthy|is runtime healthy|can jarvis continue(?: safely)?|what failed(?: in execution| recently)?|what (?:runs|tool runs) failed recently|what(?:'s| is| broke| broke recently| is broken| went wrong) (?:in )?execution|why is execution broken|show (?:the )?(?:failed|blocked) runs|recent (?:failed|blocked) runs|recent tool failures|recent failed tool runs|recent tool runs failed|show (?:the )?(?:execution|runtime|tool) health queue)(?:\s*:?\s*(?P<limit>\d+))?$",
            text,
            re.IGNORECASE,
        )
        if natural_execution_health_match:
            limit = (natural_execution_health_match.group("limit") or "").strip()
            args = {"limit": limit} if limit else {}
            return Plan(
                goal="Summarize recent execution health before continuing autonomy.",
                actions=[
                    PlannedAction(
                        "execution_health_report",
                        args,
                        "The user asked in natural language whether recent execution is healthy or what failed before continuing.",
                    )
                ],
            )

        recovery_closure_match = re.search(
            r"^(?:recovery closure checklist|execution recovery closure|execution closure checklist|closure checklist|recovery closure|close recovery proof|close execution health)(?:\s*:?\s*(?P<limit>\d+))?$",
            text,
            re.IGNORECASE,
        )
        if recovery_closure_match:
            limit = (recovery_closure_match.group("limit") or "").strip()
            args = {"limit": limit} if limit else {}
            return Plan(
                goal="Show the read-only recovery closure checklist before retry or completion.",
                actions=[
                    PlannedAction(
                        "recovery_closure_checklist",
                        args,
                        "The user asked for the exact proof checklist needed to close execution-health recovery debt.",
                    )
                ],
            )

        specialist_handoff_match = re.search(
            r"^(?:specialist handoff receipt|specialist handoff|handoff receipt|brain handoff receipt|model handoff receipt|route handoff receipt)\s*:?\s*(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if specialist_handoff_match:
            return Plan(
                goal="Build a read-only specialist handoff receipt for the order.",
                actions=[
                    PlannedAction(
                        "specialist_handoff_receipt",
                        {"request": specialist_handoff_match.group("request").strip()},
                        "The user asked Jarvis to package an order for the selected specialist brain with input, output, safety, and verification contracts.",
                    )
                ],
            )

        harness_cycle_match = re.search(
            r"^(?:harness cycle|agent harness cycle|jarvis harness cycle|harness preview|agent harness preview|agi harness cycle)\s*:?\s*(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if harness_cycle_match:
            return Plan(
                goal="Preview the full Jarvis harness cycle for an order.",
                actions=[
                    PlannedAction(
                        "harness_cycle_preview",
                        {"request": harness_cycle_match.group("request").strip()},
                        "The user asked to preview how the agent harness would route, gate, act, verify, and learn before execution.",
                    )
                ],
            )

        natural_command_diagnosis_match = re.search(
            r"^(?:how would (?:jarvis|you) route|why (?:did|would) (?:(?:jarvis|you) route(?:\s+(?:this|that|the request|the command))?|(?:this|that|the request|the command) route)(?: to [a-z0-9_ -]+)?|diagnose route for|diagnose routing for|what route would (?:jarvis|you) use for)\s*:?\s*(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_command_diagnosis_match:
            return Plan(
                goal="Diagnose command routing.",
                actions=[
                    PlannedAction(
                        "command_diagnosis",
                        {"request": natural_command_diagnosis_match.group("request").strip()},
                        "The user asked in natural language to diagnose how Jarvis would route a command without executing it.",
                    )
                ],
            )

        command_diagnosis_match = re.search(
            r"^(?:command diagnosis|diagnose command|diagnose request|route diagnosis|routing diagnosis)\s*:?\s*(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if command_diagnosis_match:
            return Plan(
                goal="Diagnose command routing.",
                actions=[
                    PlannedAction(
                        "command_diagnosis",
                        {"request": command_diagnosis_match.group("request").strip()},
                        "The user asked to diagnose how Jarvis would route a command without executing it.",
                    )
                ],
            )

        risky_lifecycle_match = re.search(
            r"^(?:risky request lifecycle|risk lifecycle|approval lifecycle|safety lifecycle|request safety lifecycle)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if risky_lifecycle_match:
            request = (risky_lifecycle_match.group("request") or DEFAULT_HARNESS_PACKET_REQUEST).strip()
            return Plan(
                goal="Map risky request lifecycle.",
                actions=[
                    PlannedAction(
                        "risky_request_lifecycle",
                        {"request": request},
                        "The user asked to map how a risky request moves through preflight, approval, last-look, rerun, and audit.",
                    )
                ],
            )

        agent_loop_packet_match = re.search(
            r"^(?:agent loop packet|second loop packet|second chat loop packet|jarvis second loop packet|task loop packet|work loop packet)\s*:?\s*(?P<goal>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if agent_loop_packet_match:
            return Plan(
                goal="Prepare Jarvis's second task/action loop packet.",
                actions=[
                    PlannedAction(
                        "agent_loop_packet",
                        {"goal": agent_loop_packet_match.group("goal").strip()},
                        "The user asked for an inspectable second-loop task packet before work begins.",
                    )
                ],
            )

        agent_loop_match = re.search(
            r"^(?:agent loop preview|second loop|second chat loop|jarvis second loop|work loop preview|task loop preview)\s*:?\s*(?P<goal>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if agent_loop_match:
            return Plan(
                goal="Preview Jarvis's second task/action loop.",
                actions=[
                    PlannedAction(
                        "agent_loop_preview",
                        {"goal": agent_loop_match.group("goal").strip()},
                        "The user asked to inspect the second loop Jarvis would use after chat becomes work.",
                    )
                ],
            )

        assistant_turn_rehearsal_early_match = re.search(
            r"^(?:"
            r"(?:show|view|review|inspect|preview)(?:\s+me)?(?:\s+the)?(?:\s+(?:latest|last|newest|current))?\s+(?:assistant\s+(?:turn\s+)?(?:rehearsal|preview)|assistant\s+turn|turn\s+rehearsal|chat\s+(?:turn\s+)?(?:rehearsal|preview)|chat\s+turn|message\s+rehearsal)|"
            r"(?:assistant\s+(?:turn\s+)?(?:rehearsal|preview)|turn\s+rehearsal|chat\s+(?:turn\s+)?(?:rehearsal|preview)|message\s+rehearsal|rehearse\s+assistant\s+turn|dry\s+run\s+(?:assistant|chat)\s+turn|chat\s+dry\s+run)"
            r")(?:\s*:?\s*(?P<message>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if assistant_turn_rehearsal_early_match:
            message = (assistant_turn_rehearsal_early_match.group("message") or "").strip()
            if re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", message, re.IGNORECASE):
                message = ""
            return Plan(
                goal="Preview a full assistant turn.",
                actions=[
                    PlannedAction(
                        "assistant_turn_rehearsal",
                        {"message": message or DEFAULT_HARNESS_PACKET_REQUEST},
                        "The user asked to preview whether a message becomes chat or tool execution.",
                    )
                ],
            )

        natural_rehearsal_with_match = re.search(
            r"^what would (?:jarvis|you) do with\s+(?P<request>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_rehearsal_with_match:
            return Plan(
                goal="Preview a request without executing it.",
                actions=[
                    PlannedAction(
                        "action_rehearsal",
                        {"request": natural_rehearsal_with_match.group("request").strip()},
                        "The user asked what Jarvis would do with a request, so Jarvis should rehearse it without executing tools.",
                    )
                ],
            )

        natural_rehearsal_wrapper_match = re.search(
            r"^(?:(?P<prefix>dry[- ]run|simulate|preview|walk through|plan only)\s+(?P<request>.+)|(?:show me\s+)?what (?:would happen|happens) if (?:i|we|you|jarvis)\s+(?P<request2>.+)|before you do it\s+(?P<request3>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_rehearsal_wrapper_match:
            request = (
                natural_rehearsal_wrapper_match.group("request")
                or natural_rehearsal_wrapper_match.group("request2")
                or natural_rehearsal_wrapper_match.group("request3")
                or ""
            ).strip()
            prefix = (natural_rehearsal_wrapper_match.group("prefix") or "").strip().lower()
            if prefix.startswith("dry") or _looks_like_risky_natural_order(request.lower()):
                return Plan(
                    goal="Preview a request without executing it.",
                    actions=[
                        PlannedAction(
                            "action_rehearsal",
                            {"request": request},
                            "The user asked to simulate or preview an action-looking request without executing tools.",
                        )
                    ],
                )

        rehearsal_match = re.search(
            r"^(?:rehearse|dry[- ]run|action rehearsal|preview action|preview request|what would jarvis do)(?:\s*:?\s*(?P<request>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if rehearsal_match:
            request = (rehearsal_match.group("request") or DEFAULT_HARNESS_PACKET_REQUEST).strip()
            return Plan(
                goal="Preview a request without executing it.",
                actions=[
                    PlannedAction(
                        "action_rehearsal",
                        {"request": request},
                        "The user asked Jarvis to rehearse a request without executing tools.",
                    )
                ],
            )

        quick_note_capture_match = re.search(
            r"^(?:"
            r"take\s+(?:a\s+)?note(?:\s+of)?|"
            r"jot\s+(?:this\s+)?down|"
            r"write\s+down|"
            r"write\s+this\s+down|"
            r"save\s+a\s+note(?:\s+that)?|"
            r"save\s+this\s+note|"
            r"save\s+note\s*:|"
            r"note\s+to\s+self|"
            r"make\s+(?:a\s+)?note(?:\s+of)?"
            r")\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if quick_note_capture_match:
            body = re.sub(r"^(?:that|of)\s+", "", quick_note_capture_match.group("body").strip(), flags=re.IGNORECASE)
            if body:
                return Plan(
                    goal="Capture Jarvis Obsidian note.",
                    actions=[
                        PlannedAction(
                            "write_jarvis_note",
                            {
                                "path": "Inbox/Captured Notes.md",
                                "body": body,
                                "mode": "append",
                            },
                            "The user asked to capture a local Jarvis-owned Obsidian note.",
                        )
                    ],
                )

        remember_match = re.search(
            r"^(?:remember|note|save)\s+(?:that\s+)?(?P<body>.+)$",
            text,
            flags=re.IGNORECASE,
        )
        structured_save_prefixes = (
            "save skill",
            "save latest page",
            "save recent page",
            "save last page",
            "save page",
            "save build progress",
            "save progress report",
            "save handoff",
            "save handoff brief",
            "save jarvis handoff",
            "save jarvis note",
            "save to jarvis note",
            "save to jarvis notes",
            "save to note",
            "save to notes",
            "save note",
            "save decision",
            "save a decision",
            "save chat context",
            "save brain context",
            "save session closeout",
            "save closeout",
            "save morning startup",
            "save startup brief",
            "save work queue",
            "save build target packet",
            "save jarvis build target",
            "save next build target",
            "save work block checkpoint",
            "save work-block checkpoint",
            "save checkpoint packet",
            "save build delta",
            "save checkpoint delta",
            "save approval review",
            "save pending approval review",
            "save feedback report",
            "save feedback actions",
            "save feedback review",
            "save feedback improvements",
            "save learning review",
            "queue learning tasks",
            "save preference",
            "remember preference",
            "save person",
            "remember person",
            "save decision",
            "remember decision",
            "save state",
        )
        early_proactive_command = _strip_trailing_politeness(text).lower()
        early_proactive_command = re.sub(r"^(?:latest|current)\s+", "", early_proactive_command).strip()
        if early_proactive_command in {
            "daily plan",
            "plan today",
            "today plan",
            "today's plan",
            "plan my day",
            "what is my plan today",
            "what's my plan today",
            "save daily plan",
            "write daily plan",
            "save today plan",
            "write today plan",
            "save my daily plan",
            "save plan for today",
            "my daily plan",
        }:
            return Plan(
                goal="Generate tactical daily plan.",
                actions=[
                    PlannedAction(
                        "daily_plan",
                        {"target_date": datetime.now().strftime("%Y-%m-%d")},
                        "The user asked for a tactical daily plan.",
                    )
                ],
            )
        if early_proactive_command in {
            "morning startup",
            "startup brief",
            "start my day",
            "start day",
            "morning plan",
            "today startup",
        }:
            return Plan(
                goal="Show morning startup.",
                actions=[PlannedAction("morning_startup", {}, "The user asked for a safe startup brief for the day.")],
            )
        if early_proactive_command in {
            "save morning startup",
            "write morning startup",
            "save startup brief",
            "write startup brief",
            "save start my day",
            "write start my day",
            "save morning plan",
            "write morning plan",
        }:
            return Plan(
                goal="Save morning startup.",
                actions=[PlannedAction("save_morning_startup", {}, "The user asked to save the morning startup brief.")],
            )
        ephemeral_conversation_memory = bool(
            remember_match
            and re.search(
                r"\b(?:within|for)\s+this\s+conversation\b",
                low,
            )
            and re.search(
                r"\b(?:without\s+saving|do\s+not\s+save|don't\s+save)\b",
                low,
            )
        )
        if (
            remember_match
            and not low.startswith(structured_save_prefixes)
            and not ephemeral_conversation_memory
        ):
            body = _raw_free_text_after_prefix(
                raw_text,
                r"^(?:remember|note|save)\s+(?:that\s+)?",
            ) or remember_match.group("body").strip()
            title = body[:60].rstrip(".")
            return Plan(
                goal="Store a memory.",
                actions=[
                    PlannedAction(
                        "remember",
                        {"category": "facts", "title": title, "body": body},
                        "The user asked Jarvis to remember something.",
                    )
                ],
            )

        early_conversation_search_match = re.search(
            r"^(?:(?:search|find)\s+(?:my\s+|the\s+)?(?:conversation|conversations|chat|chats)\s+(?:(?:for|about)\s+)?|(?:show|review)\s+(?:my\s+|the\s+)?(?:conversation|conversations|chat|chats)\s+(?:for|about)\s+|(?:conversation|conversations|chat|chats)\s+search\s+)(?P<query>.+)$",
            text,
            re.IGNORECASE,
        )
        if early_conversation_search_match:
            query = _strip_trailing_politeness(early_conversation_search_match.group("query")).strip()
            if query:
                return Plan(
                    goal="Search conversation history.",
                    actions=[
                        PlannedAction(
                            "search_conversations",
                            {"query": query},
                            "The user asked to search conversation history.",
                        )
                    ],
                )

        search_match = re.search(
            r"^(?:(?:search|find|recall)\s+(?:memory|memories)\s*(?:for|about)?|(?:memory|memories)\s+(?:for|about))\s+(?P<query>.+)$"
            # Real gap found live 2026-07-10, same privacy-relevant misroute
            # class as the round-21/26/27 notes/tasks/files word-order fixes:
            # "search for the/a memory about X" / "find the/a memory about X"
            # (article BEFORE the noun) fell through to the generic public
            # web-search fallback instead of searching Jarvis's own memory.
            r"|^(?:search for|find)\s+(?:the\s+|a\s+|my\s+)?(?:memory|memories)\s+(?:for|about)\s+(?P<query2>.+)$",
            text,
            re.IGNORECASE,
        )
        if search_match:
            query = _strip_trailing_politeness(search_match.group("query") or search_match.group("query2")).strip()
            return Plan(
                goal=f"Search memory for {query}.",
                actions=[PlannedAction("search_memory", {"query": query}, "The user asked for memory recall.")],
            )

        brain_search_match = re.search(
            r"^(?:(?:show|open|preview)(?:\s+(?:me|latest|current|newest))?\s+)?(?:gbrain|brain)\s+search(?:\s*:?\s*(?P<query>.*))?$"
            r"|^search\s+(?:my\s+|the\s+)?(?:gbrain|brain)\s+(?:for\s+)?(?P<query2>.+)$",
            text,
            re.IGNORECASE,
        )
        if brain_search_match:
            query = _strip_trailing_politeness(brain_search_match.group("query") or brain_search_match.group("query2") or "").strip()
            return Plan(
                goal=f"Search Jarvis brain for {query}.",
                actions=[PlannedAction("brain_search", {"query": query, "limit": 8}, "The user asked for GBrain-style memory search.")],
            )

        brain_think_match = re.search(r"^(?:gbrain|brain)\s+think\s*:\s*(?P<question>.+)$", text, re.IGNORECASE)
        if brain_think_match:
            question = brain_think_match.group("question").strip()
            return Plan(
                goal="Think from Jarvis memory.",
                actions=[PlannedAction("brain_think", {"question": question, "limit": 8}, "The user asked for GBrain-style cited synthesis.")],
            )

        brain_think_default_match = re.search(
            r"^(?:(?:show|open|preview)(?:\s+(?:me|latest|current|newest))?\s+)?(?:(?:gbrain|brain)\s+think|think\s+with\s+(?:gbrain|brain))(?:\s+(?:please|pls|thanks|thank you))?$",
            text,
            re.IGNORECASE,
        )
        if brain_think_default_match:
            return Plan(
                goal="Think from Jarvis memory.",
                actions=[
                    PlannedAction(
                        "brain_think",
                        {"question": DEFAULT_HARNESS_PACKET_REQUEST, "limit": 8},
                        "The user asked for GBrain-style cited synthesis.",
                    )
                ],
            )

        brain_graph_display_match = re.search(
            r"^(?:(?:show|open|preview)(?:\s+(?:me|latest|current|newest))?\s+)?(?:(?:gbrain|brain|knowledge)\s+graph|(?:gbrain|brain)\s+graph\s+preview)(?:\s+(?:please|pls|thanks|thank you))?$",
            text,
            re.IGNORECASE,
        )
        if brain_graph_display_match:
            return Plan(
                goal="Preview Jarvis knowledge graph.",
                actions=[PlannedAction("brain_graph", {"limit": 12}, "The user asked for a GBrain-style graph preview.")],
            )

        brain_neighbors_match = re.search(
            r"^(?:(?:show|open|preview)(?:\s+(?:me|latest|current|newest))?\s+)?(?:(?:gbrain|brain)\s+neighbors?|(?:gbrain|brain)\s+related|related\s+(?:gbrain|brain)\s+memory)\s+#?(?P<memory_id>\d+)(?:\s+(?:please|pls|thanks|thank you))?$",
            text,
            re.IGNORECASE,
        )
        if brain_neighbors_match:
            return Plan(
                goal="Find related Jarvis memories.",
                actions=[
                    PlannedAction(
                        "brain_neighbors",
                        {"memory_id": int(brain_neighbors_match.group("memory_id")), "limit": 6},
                        "The user asked for GBrain-style related memories.",
                    )
                ],
            )

        remember_about_match = re.search(
            r"^what\s+do\s+you\s+remember\s+about\s+(?P<query>.+)$",
            text,
            re.IGNORECASE,
        )
        if remember_about_match:
            query = _strip_trailing_politeness(remember_about_match.group("query")).strip()
            return Plan(
                goal=f"Search memory for {query}.",
                actions=[PlannedAction("search_memory", {"query": query}, "The user asked what Jarvis remembers about a topic.")],
            )

        if low in {
            "memories",
            "memories please",
            "memory list",
            "memory list please",
            "list memories",
            "list memories please",
            "show memories",
            "show memories please",
            "show latest memories",
            "show latest memories please",
            "recent memories",
            "recent memories please",
            "facts",
            "facts please",
            "recent facts",
            "recent facts please",
            "what facts do you know",
            "what facts do you know please",
            "remembered facts",
            "remembered facts please",
        } or "recent memor" in low or "what do you remember" in low:
            return Plan(
                goal="Show recent memories.",
                actions=[PlannedAction("recent_memories", {"limit": 10}, "The user asked what Jarvis remembers.")],
            )

        feedback_command = _strip_trailing_politeness(text).lower()
        if feedback_command in {
            "feedback",
            "feedback report",
            "jarvis feedback",
            "jarvis feedback report",
            "show feedback",
            "show latest feedback",
            "latest feedback",
            "review feedback",
            "what feedback do you have",
        }:
            return Plan(
                goal="Show Jarvis feedback report.",
                actions=[PlannedAction("feedback_report", {}, "The user asked to review captured Jarvis feedback.")],
            )

        if feedback_command in {"save feedback report", "write feedback report", "export feedback report", "save feedback review", "write feedback review"}:
            return Plan(
                goal="Save Jarvis feedback report.",
                actions=[PlannedAction("save_feedback_report", {}, "The user asked to save the Jarvis feedback report.")],
            )

        if feedback_command in {"feedback actions", "jarvis feedback actions", "feedback next actions", "feedback improvements", "improve from feedback"}:
            return Plan(
                goal="Suggest feedback-driven improvements.",
                actions=[PlannedAction("feedback_actions", {}, "The user asked for reviewable next actions from Jarvis feedback.")],
            )

        if low in {
            "failure clusters",
            "repeated failure clusters",
            "failure cluster report",
            "miss clusters",
            "repeated misses",
            "regression clusters",
            "learning clusters",
        }:
            return Plan(
                goal="Review repeated Jarvis misses.",
                actions=[PlannedAction("repeated_failure_clusters", {}, "The user asked to group repeated Jarvis failures into safe learning candidates.")],
            )

        failure_learning_closure_match = re.search(
            r"^(?:failure\s+learning\s+closure\s+ledger|learning\s+closure\s+ledger|failure\s+closure\s+ledger|regression\s+learning\s+closure|learning\s+closure)\s*:?\s*(?P<summary>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_learning_closure_match:
            summary = failure_learning_closure_match.group("summary").strip()
            return Plan(
                goal="Bind repeated-failure learning closure evidence into one read-only ledger.",
                actions=[
                    PlannedAction(
                        "failure_learning_closure_ledger",
                        {"summary": summary} if summary else {},
                        "The user asked for a read-only closure ledger over repeated-failure cockpit, receipt, completion, handoff, closeout, and durable learning-record evidence.",
                    )
                ],
            )

        failure_learning_cockpit_match = re.search(
            r"^(?:failure\s+learning\s+cockpit|learning\s+cockpit|failure\s+cockpit|regression\s+cockpit)\s*:?\s*(?P<cluster>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_learning_cockpit_match:
            cluster = failure_learning_cockpit_match.group("cluster").strip()
            return Plan(
                goal="Consolidate repeated-failure learning readiness before patch review.",
                actions=[
                    PlannedAction(
                        "failure_learning_cockpit",
                        {"cluster": cluster} if cluster else {},
                        "The user asked for a read-only learning cockpit before turning repeated failures into tests.",
                    )
                ],
            )

        failure_patch_completion_gate_match = re.search(
            r"^(?:failure\s+patch\s+completion\s+gate|patch\s+completion\s+gate|failure\s+completion\s+gate|regression\s+completion\s+gate)\s*:?\s*(?P<summary>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_patch_completion_gate_match:
            summary = failure_patch_completion_gate_match.group("summary").strip()
            return Plan(
                goal="Gate repeated-failure patch completion evidence.",
                actions=[
                    PlannedAction(
                        "failure_patch_completion_gate",
                        {"summary": summary} if summary else {},
                        "The user asked for a read-only completion gate before claiming a repeated-failure patch is fixed.",
                    )
                ],
            )

        failure_patch_closeout_match = re.search(
            r"^(?:failure\s+patch\s+closeout|patch\s+closeout|regression\s+patch\s+closeout|failure\s+closeout)\s*:?\s*(?P<summary>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_patch_closeout_match:
            summary = failure_patch_closeout_match.group("summary").strip()
            return Plan(
                goal="Close repeated-failure patch review after completion proof.",
                actions=[
                    PlannedAction(
                        "failure_patch_closeout_packet",
                        {"summary": summary} if summary else {},
                        "The user asked for a read-only closeout after the repeated-failure patch handoff and completion-review evidence.",
                    )
                ],
            )

        failure_learning_record_match = re.search(
            r"^(?:failure\s+learning\s+record|learning\s+record\s+packet|failure\s+learning\s+record\s+packet|regression\s+learning\s+record)\s*:?\s*(?P<summary>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_learning_record_match:
            summary = failure_learning_record_match.group("summary").strip()
            return Plan(
                goal="Verify durable learning record evidence after repeated-failure patch closeout.",
                actions=[
                    PlannedAction(
                        "failure_learning_record_packet",
                        {"summary": summary} if summary else {},
                        "The user asked for a read-only learning-record gate after repeated-failure patch closeout.",
                    )
                ],
            )

        failure_patch_handoff_match = re.search(
            r"^(?:failure\s+patch\s+handoff|patch\s+handoff|regression\s+patch\s+handoff|failure\s+handoff)\s*:?\s*(?P<summary>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_patch_handoff_match:
            summary = failure_patch_handoff_match.group("summary").strip()
            return Plan(
                goal="Package repeated-failure patch proof for completion review.",
                actions=[
                    PlannedAction(
                        "failure_patch_handoff_packet",
                        {"summary": summary} if summary else {},
                        "The user asked for a read-only handoff after the repeated-failure patch completion gate.",
                    )
                ],
            )

        failure_patch_application_bridge_match = re.search(
            r"^(?:failure\s+patch\s+application\s+bridge|patch\s+application\s+bridge|applied\s+patch\s+bridge|failure\s+apply\s+bridge|regression\s+application\s+bridge)\s*:?\s*(?P<summary>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_patch_application_bridge_match:
            summary = failure_patch_application_bridge_match.group("summary").strip()
            return Plan(
                goal="Bind an applied repeated-failure patch to its reviewed apply contract.",
                actions=[
                    PlannedAction(
                        "failure_patch_application_bridge",
                        {"summary": summary} if summary else {},
                        "The user asked for a read-only bridge between an applied patch, its apply contract, receipt, verification, compile pass, and rollback.",
                    )
                ],
            )

        failure_patch_receipt_match = re.search(
            r"^(?:failure\s+patch\s+receipt|patch\s+receipt|failure\s+receipt|regression\s+patch\s+receipt)\s*:?\s*(?P<summary>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_patch_receipt_match:
            summary = failure_patch_receipt_match.group("summary").strip()
            return Plan(
                goal="Review repeated-failure patch receipt evidence.",
                actions=[
                    PlannedAction(
                        "failure_patch_receipt_packet",
                        {"summary": summary} if summary else {},
                        "The user asked for a read-only post-patch receipt before claiming a repeated failure is fixed.",
                    )
                ],
            )

        failure_promotion_match = re.search(
            r"^(?:failure\s+promotion\s+packet|promote\s+failure\s+cluster|cluster\s+promotion\s+packet|regression\s+promotion\s+packet)\s*:?\s*(?P<cluster>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_promotion_match:
            cluster = failure_promotion_match.group("cluster").strip()
            return Plan(
                goal="Draft failure-cluster promotion packet.",
                actions=[
                    PlannedAction(
                        "failure_promotion_packet",
                        {"cluster": cluster} if cluster else {},
                        "The user asked for a read-only promotion plan from repeated failure clusters.",
                    )
                ],
            )

        failure_implementation_match = re.search(
            r"^(?:failure\s+implementation\s+packet|implement\s+failure\s+cluster|regression\s+implementation\s+packet|failure\s+work\s+packet|miss\s+implementation\s+packet)\s*:?\s*(?P<cluster>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_implementation_match:
            cluster = failure_implementation_match.group("cluster").strip()
            return Plan(
                goal="Draft failure-cluster implementation packet.",
                actions=[
                    PlannedAction(
                        "failure_implementation_packet",
                        {"cluster": cluster} if cluster else {},
                        "The user asked for an implementation-ready read-only work packet from repeated failure clusters.",
                    )
                ],
            )

        failure_apply_match = re.search(
            r"^(?:failure\s+apply\s+contract|apply\s+failure\s+contract|failure\s+patch\s+contract|apply\s+failure\s+cluster|regression\s+apply\s+contract)\s*:?\s*(?P<cluster>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_apply_match:
            cluster = failure_apply_match.group("cluster").strip()
            return Plan(
                goal="Draft failure-cluster apply contract.",
                actions=[
                    PlannedAction(
                        "failure_apply_contract",
                        {"cluster": cluster} if cluster else {},
                        "The user asked for a read-only last-look contract before applying a repeated-failure patch.",
                    )
                ],
            )

        if feedback_command in {"save feedback actions", "write feedback actions", "export feedback actions", "save feedback improvements", "write feedback improvements"}:
            return Plan(
                goal="Save feedback-driven improvements.",
                actions=[PlannedAction("save_feedback_actions", {}, "The user asked to save reviewable feedback-driven next actions.")],
            )

        failure_to_test_match = re.search(
            r"^(?:failure\s+to\s+test|test\s+preview|regression\s+preview|turn\s+failure\s+into\s+test|turn\s+this\s+miss\s+into\s+a\s+test)\s*:?\s*(?P<failure>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if failure_to_test_match:
            return Plan(
                goal="Preview a regression test from a Jarvis miss.",
                actions=[
                    PlannedAction(
                        "failure_to_test_preview",
                        {"failure": failure_to_test_match.group("failure").strip()},
                        "The user asked to turn a Jarvis failure into a safe test preview.",
                    )
                ],
            )

        feedback_match = re.search(
            r"^(?:jarvis\s+feedback|feedback|assistant\s+feedback)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if feedback_match:
            return Plan(
                goal="Record Jarvis feedback.",
                actions=[
                    PlannedAction(
                        "record_feedback",
                        {"body": feedback_match.group("body").strip()},
                        "The user gave reviewable feedback about Jarvis behavior.",
                    )
                ],
            )

        if low_command in {
            "build delta",
            "jarvis build delta",
            "checkpoint delta",
            "latest build delta",
            "what changed since checkpoint",
            "what changed in the last run",
            "what changed this round",
        }:
            return Plan(
                goal="Summarize latest Jarvis build delta.",
                actions=[
                    PlannedAction(
                        "build_delta_report",
                        {},
                        "The user asked what changed in the latest Jarvis build checkpoint.",
                    )
                ],
            )

        work_block_checkpoint_match = re.search(
            r"^(?:work block checkpoint|work-block checkpoint|checkpoint packet|resumable checkpoint|work checkpoint)\s*:?\s*(?P<objective>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if work_block_checkpoint_match:
            objective = work_block_checkpoint_match.group("objective").strip()
            return Plan(
                goal="Prepare a resumable Jarvis work-block checkpoint.",
                actions=[
                    PlannedAction(
                        "work_block_checkpoint",
                        {"objective": objective} if objective else {},
                        "The user asked for a read-only resumable work-block checkpoint.",
                    )
                ],
            )

        save_work_block_checkpoint_match = re.search(
            r"^(?:save work block checkpoint|save work-block checkpoint|write work block checkpoint|export work block checkpoint|save checkpoint packet)\s*:?\s*(?P<objective>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if save_work_block_checkpoint_match:
            objective = save_work_block_checkpoint_match.group("objective").strip()
            return Plan(
                goal="Save a resumable Jarvis work-block checkpoint.",
                actions=[
                    PlannedAction(
                        "save_work_block_checkpoint",
                        {"objective": objective} if objective else {},
                        "The user asked to save a resumable work-block checkpoint.",
                    )
                ],
            )

        supersession_match = re.search(
            r"^(?:operator instruction supersession|instruction supersession|newest instruction gate|autonomy supersession gate)\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if supersession_match:
            body = supersession_match.group("body").strip()
            field_pattern = r"\b(?P<key>previous_instruction|previous|latest_instruction|latest|stop_at|stop|until|current_time|now|timezone)\s*=\s*(?P<value>.*?)(?=\s+\b(?:previous_instruction|previous|latest_instruction|latest|stop_at|stop|until|current_time|now|timezone)\s*=|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            objective_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                objective_parts.append(body[cursor:start].strip())
                cursor = end
            objective_parts.append(body[cursor:].strip())
            objective = " ".join(part for part in objective_parts if part).strip()
            args = {
                "objective": objective or "continue Jarvis work",
                "previous_instruction": fields.get("previous_instruction") or fields.get("previous") or "",
                "latest_instruction": fields.get("latest_instruction") or fields.get("latest") or "",
                "stop_at": fields.get("stop_at") or fields.get("stop") or fields.get("until") or "",
                "current_time": fields.get("current_time") or fields.get("now") or "",
                "timezone": fields.get("timezone") or "",
            }
            return Plan(
                goal="Prove the newest operator instruction governs autonomy.",
                actions=[
                    PlannedAction(
                        "operator_instruction_supersession_packet",
                        args,
                        "The user asked for a read-only supersession gate before autonomous continuation.",
                    )
                ],
            )

        operator_timebox_match = re.search(
            r"^(?:operator timebox|timebox contract|work window contract|stop time contract|continuation timebox)\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if operator_timebox_match:
            body = operator_timebox_match.group("body").strip()
            field_pattern = r"\b(?P<key>stop_at|stop|until|current_time|now|timezone)\s*=\s*(?P<value>.*?)(?=\s+\b(?:stop_at|stop|until|current_time|now|timezone)\s*=|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            if not (fields.get("stop_at") or fields.get("stop") or fields.get("until")):
                until_match = re.search(
                    r"\buntil\s+(?P<value>\d{4}-\d{2}-\d{2}T[^\s,;]+)",
                    body,
                    re.IGNORECASE,
                )
                if until_match:
                    fields["until"] = until_match.group("value").strip()
                    spans.append(until_match.span())
                    spans.sort()
            objective_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                objective_parts.append(body[cursor:start].strip())
                cursor = end
            objective_parts.append(body[cursor:].strip())
            objective = " ".join(part for part in objective_parts if part).strip()
            args = {
                "objective": objective or "continue Jarvis work",
                "stop_at": fields.get("stop_at") or fields.get("stop") or fields.get("until") or "",
                "current_time": fields.get("current_time") or fields.get("now") or "",
                "timezone": fields.get("timezone") or "",
            }
            return Plan(
                goal="Check whether Jarvis is still inside the operator's explicit work window.",
                actions=[
                    PlannedAction(
                        "operator_timebox_contract",
                        args,
                        "The user asked for a read-only stop-window contract before continuing Jarvis work.",
                    )
                ],
            )

        checkpoint_cockpit_match = re.search(
            r"^(?:checkpoint recovery cockpit|recovery cockpit|continuation cockpit|long-running autonomy cockpit)\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if checkpoint_cockpit_match:
            body = checkpoint_cockpit_match.group("body").strip()
            field_pattern = r"\b(?P<key>stop_at|stop|until|current_time|now|timezone)\s*=\s*(?P<value>.*?)(?=\s+\b(?:stop_at|stop|until|current_time|now|timezone)\s*=|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            objective_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                objective_parts.append(body[cursor:start].strip())
                cursor = end
            objective_parts.append(body[cursor:].strip())
            objective = " ".join(part for part in objective_parts if part).strip()
            return Plan(
                goal="Consolidate checkpoint recovery readiness before continuing local-safe Jarvis work.",
                actions=[
                    PlannedAction(
                        "checkpoint_recovery_cockpit",
                        {
                            "objective": objective or "Continue Jarvis V2 safely.",
                            "stop_at": fields.get("stop_at") or fields.get("stop") or fields.get("until") or "",
                            "current_time": fields.get("current_time") or fields.get("now") or "",
                            "timezone": fields.get("timezone") or "",
                        },
                        "The user asked for a checkpoint recovery cockpit before resuming long-running work.",
                    )
                ],
            )

        autonomy_resume_gate_match = re.search(
            r"^(?:autonomy resume gate|resume autonomy gate|normal follow-through gate|normal followthrough gate|autonomous continuation gate|long-running autonomy resume gate)\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if autonomy_resume_gate_match:
            body = autonomy_resume_gate_match.group("body").strip()
            keys = "stop_at|until|current_time|now|timezone|step|reviewed_step|approved_step|verification|tests|receipt|receipt_path|receipt_sha256|receipt_hash|checkpoint|checkpoint_path|checkpoint_sha256|checkpoint_hash|stop_condition|stop_rule|blockers|approval|approval_reference"
            field_pattern = rf"\b(?P<key>{keys})\s*=\s*(?P<value>.*?)(?=\s+\b(?:{keys})\s*=|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            objective_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                objective_parts.append(body[cursor:start].strip())
                cursor = end
            objective_parts.append(body[cursor:].strip())
            objective = " ".join(part for part in objective_parts if part).strip()
            return Plan(
                goal="Gate whether normal autonomous follow-through may resume after checkpoint recovery.",
                actions=[
                    PlannedAction(
                        "autonomy_resume_gate",
                        {
                            "objective": objective or "Continue Jarvis V2 safely.",
                            "stop_at": fields.get("stop_at") or fields.get("until") or "",
                            "current_time": fields.get("current_time") or fields.get("now") or "",
                            "timezone": fields.get("timezone") or "",
                            "reviewed_step": fields.get("reviewed_step") or fields.get("approved_step") or fields.get("step") or "",
                            "verification": fields.get("verification") or fields.get("tests") or "",
                            "receipt_path": fields.get("receipt_path") or fields.get("receipt") or "",
                            "receipt_sha256": fields.get("receipt_sha256") or fields.get("receipt_hash") or "",
                            "checkpoint_path": fields.get("checkpoint_path") or fields.get("checkpoint") or "",
                            "checkpoint_sha256": fields.get("checkpoint_sha256") or fields.get("checkpoint_hash") or "",
                            "stop_condition": fields.get("stop_condition") or fields.get("stop_rule") or "",
                            "blockers": fields.get("blockers") or "",
                            "approval_reference": fields.get("approval_reference") or fields.get("approval") or "",
                        },
                        "The user asked for a read-only autonomy resume gate before normal follow-through continues.",
                    )
                ],
            )

        autonomy_continuation_execution_match = re.search(
            r"^(?:autonomy continuation execution|autonomous continuation execution|continuation execution packet|normal continuation execution packet|local-safe continuation packet)\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if autonomy_continuation_execution_match:
            body = autonomy_continuation_execution_match.group("body").strip()
            keys = "stop_at|until|current_time|now|timezone|step|reviewed_step|approved_step|verification|tests|receipt|receipt_path|receipt_sha256|receipt_hash|checkpoint|checkpoint_path|checkpoint_sha256|checkpoint_hash|stop_condition|stop_rule|blockers|approval|approval_reference|next_step|next|next_verification|post_verification"
            field_pattern = rf"\b(?P<key>{keys})\s*=\s*(?P<value>.*?)(?=\s+\b(?:{keys})\s*=|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            objective_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                objective_parts.append(body[cursor:start].strip())
                cursor = end
            objective_parts.append(body[cursor:].strip())
            objective = " ".join(part for part in objective_parts if part).strip()
            return Plan(
                goal="Gate one normal local-safe continuation step after autonomy resume proof.",
                actions=[
                    PlannedAction(
                        "autonomy_continuation_execution_packet",
                        {
                            "objective": objective or "Continue Jarvis V2 safely.",
                            "stop_at": fields.get("stop_at") or fields.get("until") or "",
                            "current_time": fields.get("current_time") or fields.get("now") or "",
                            "timezone": fields.get("timezone") or "",
                            "reviewed_step": fields.get("reviewed_step") or fields.get("approved_step") or fields.get("step") or "",
                            "verification": fields.get("verification") or fields.get("tests") or "",
                            "receipt_path": fields.get("receipt_path") or fields.get("receipt") or "",
                            "receipt_sha256": fields.get("receipt_sha256") or fields.get("receipt_hash") or "",
                            "checkpoint_path": fields.get("checkpoint_path") or fields.get("checkpoint") or "",
                            "checkpoint_sha256": fields.get("checkpoint_sha256") or fields.get("checkpoint_hash") or "",
                            "stop_condition": fields.get("stop_condition") or fields.get("stop_rule") or "",
                            "blockers": fields.get("blockers") or "",
                            "approval_reference": fields.get("approval_reference") or fields.get("approval") or "",
                            "next_step": fields.get("next_step") or fields.get("next") or "",
                            "next_verification": fields.get("next_verification") or fields.get("post_verification") or "",
                        },
                        "The user asked for the final read-only gate before one local-safe autonomous continuation step.",
                    )
                ],
            )

        autonomy_step_closure_match = re.search(
            r"^(?:autonomy step closure|autonomous step closure|continuation step closure|local-safe step closure|post-step autonomy closure|post step autonomy closure)\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if autonomy_step_closure_match:
            body = autonomy_step_closure_match.group("body").strip()
            keys = (
                "stop_at|until|current_time|now|timezone|reviewed_step|recovery_step|approved_step|recovery_verification|"
                "verification|tests|recovery_receipt_path|receipt_path|receipt|recovery_receipt_sha256|receipt_sha256|receipt_hash|recovery_checkpoint_path|checkpoint_path|checkpoint|recovery_checkpoint_sha256|checkpoint_sha256|checkpoint_hash|"
                "stop_condition|stop_rule|blockers|approval|approval_reference|next_step|next|next_verification|"
                "post_verification_target|completed_step|actual_step|step|post_step_verification|post_verification|verification_evidence|"
                "closure_verification|post_step_receipt_path|post_receipt|verification_receipt|post_step_receipt_sha256|post_receipt_sha256|verification_receipt_sha256|post_step_receipt_hash|post_step_checkpoint_path|"
                "post_checkpoint|fresh_checkpoint|post_step_checkpoint_sha256|post_checkpoint_sha256|fresh_checkpoint_sha256|post_step_checkpoint_hash|execution_health|health|execution_audit|audit|after_action_learning|learning"
            )
            field_pattern = rf"\b(?P<key>{keys})\s*=\s*(?P<value>.*?)(?=\s+\b(?:{keys})\s*=|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            objective_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                objective_parts.append(body[cursor:start].strip())
                cursor = end
            objective_parts.append(body[cursor:].strip())
            objective = " ".join(part for part in objective_parts if part).strip()
            return Plan(
                goal="Close one local-safe autonomous continuation step before another continuation review.",
                actions=[
                    PlannedAction(
                        "autonomy_step_closure_packet",
                        {
                            "objective": objective or "Continue Jarvis V2 safely.",
                            "stop_at": fields.get("stop_at") or fields.get("until") or "",
                            "current_time": fields.get("current_time") or fields.get("now") or "",
                            "timezone": fields.get("timezone") or "",
                            "reviewed_step": fields.get("reviewed_step") or fields.get("recovery_step") or fields.get("approved_step") or "",
                            "recovery_verification": fields.get("recovery_verification") or fields.get("verification") or fields.get("tests") or "",
                            "recovery_receipt_path": fields.get("recovery_receipt_path") or fields.get("receipt_path") or fields.get("receipt") or "",
                            "recovery_receipt_sha256": fields.get("recovery_receipt_sha256") or fields.get("receipt_sha256") or fields.get("receipt_hash") or "",
                            "recovery_checkpoint_path": fields.get("recovery_checkpoint_path") or fields.get("checkpoint_path") or fields.get("checkpoint") or "",
                            "recovery_checkpoint_sha256": fields.get("recovery_checkpoint_sha256") or fields.get("checkpoint_sha256") or fields.get("checkpoint_hash") or "",
                            "stop_condition": fields.get("stop_condition") or fields.get("stop_rule") or "",
                            "blockers": fields.get("blockers") or "",
                            "approval_reference": fields.get("approval_reference") or fields.get("approval") or "",
                            "next_step": fields.get("next_step") or fields.get("next") or "",
                            "next_verification": fields.get("next_verification") or fields.get("post_verification_target") or "",
                            "completed_step": fields.get("completed_step") or fields.get("actual_step") or fields.get("step") or "",
                            "post_step_verification": fields.get("post_step_verification") or fields.get("verification_evidence") or fields.get("post_verification") or fields.get("closure_verification") or "",
                            "post_step_receipt_path": fields.get("post_step_receipt_path") or fields.get("post_receipt") or fields.get("verification_receipt") or "",
                            "post_step_receipt_sha256": fields.get("post_step_receipt_sha256") or fields.get("post_receipt_sha256") or fields.get("verification_receipt_sha256") or fields.get("post_step_receipt_hash") or "",
                            "post_step_checkpoint_path": fields.get("post_step_checkpoint_path") or fields.get("post_checkpoint") or fields.get("fresh_checkpoint") or "",
                            "post_step_checkpoint_sha256": fields.get("post_step_checkpoint_sha256") or fields.get("post_checkpoint_sha256") or fields.get("fresh_checkpoint_sha256") or fields.get("post_step_checkpoint_hash") or "",
                            "execution_health": fields.get("execution_health") or fields.get("health") or "",
                            "execution_audit": fields.get("execution_audit") or fields.get("audit") or "",
                            "after_action_learning": fields.get("after_action_learning") or fields.get("learning") or "",
                        },
                        "The user asked for a read-only post-step closure packet before another autonomous continuation review.",
                    )
                ],
            )

        autonomy_cycle_ledger_match = re.search(
            r"^(?:autonomy cycle ledger|autonomous cycle ledger|continuation cycle ledger|local-safe cycle ledger|autonomy proof ledger)\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if autonomy_cycle_ledger_match:
            body = autonomy_cycle_ledger_match.group("body").strip()
            keys = (
                "stop_at|until|current_time|now|timezone|reviewed_step|recovery_step|approved_step|recovery_verification|"
                "verification|tests|recovery_receipt_path|receipt_path|receipt|recovery_receipt_sha256|receipt_sha256|receipt_hash|recovery_checkpoint_path|checkpoint_path|checkpoint|recovery_checkpoint_sha256|checkpoint_sha256|checkpoint_hash|"
                "stop_condition|stop_rule|blockers|approval|approval_reference|next_step|next|next_verification|"
                "post_verification_target|completed_step|actual_step|step|post_step_verification|post_verification|verification_evidence|"
                "closure_verification|post_step_receipt_path|post_receipt|verification_receipt|post_step_receipt_sha256|post_receipt_sha256|verification_receipt_sha256|post_step_receipt_hash|post_step_checkpoint_path|"
                "post_checkpoint|fresh_checkpoint|post_step_checkpoint_sha256|post_checkpoint_sha256|fresh_checkpoint_sha256|post_step_checkpoint_hash|execution_health|health|execution_audit|audit|after_action_learning|learning"
            )
            field_pattern = rf"\b(?P<key>{keys})\s*=\s*(?P<value>.*?)(?=\s+\b(?:{keys})\s*=|$)"
            fields: dict[str, str] = {}
            spans: list[tuple[int, int]] = []
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
                spans.append(match.span())
            objective_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                objective_parts.append(body[cursor:start].strip())
                cursor = end
            objective_parts.append(body[cursor:].strip())
            objective = " ".join(part for part in objective_parts if part).strip()
            return Plan(
                goal="Bind a complete autonomy continuation cycle before another review.",
                actions=[
                    PlannedAction(
                        "autonomy_cycle_ledger",
                        {
                            "objective": objective or "Continue Jarvis V2 safely.",
                            "stop_at": fields.get("stop_at") or fields.get("until") or "",
                            "current_time": fields.get("current_time") or fields.get("now") or "",
                            "timezone": fields.get("timezone") or "",
                            "reviewed_step": fields.get("reviewed_step") or fields.get("recovery_step") or fields.get("approved_step") or "",
                            "recovery_verification": fields.get("recovery_verification") or fields.get("verification") or fields.get("tests") or "",
                            "recovery_receipt_path": fields.get("recovery_receipt_path") or fields.get("receipt_path") or fields.get("receipt") or "",
                            "recovery_receipt_sha256": fields.get("recovery_receipt_sha256") or fields.get("receipt_sha256") or fields.get("receipt_hash") or "",
                            "recovery_checkpoint_path": fields.get("recovery_checkpoint_path") or fields.get("checkpoint_path") or fields.get("checkpoint") or "",
                            "recovery_checkpoint_sha256": fields.get("recovery_checkpoint_sha256") or fields.get("checkpoint_sha256") or fields.get("checkpoint_hash") or "",
                            "stop_condition": fields.get("stop_condition") or fields.get("stop_rule") or "",
                            "blockers": fields.get("blockers") or "",
                            "approval_reference": fields.get("approval_reference") or fields.get("approval") or "",
                            "next_step": fields.get("next_step") or fields.get("next") or "",
                            "next_verification": fields.get("next_verification") or fields.get("post_verification_target") or "",
                            "completed_step": fields.get("completed_step") or fields.get("actual_step") or fields.get("step") or "",
                            "post_step_verification": fields.get("post_step_verification") or fields.get("verification_evidence") or fields.get("post_verification") or fields.get("closure_verification") or "",
                            "post_step_receipt_path": fields.get("post_step_receipt_path") or fields.get("post_receipt") or fields.get("verification_receipt") or "",
                            "post_step_receipt_sha256": fields.get("post_step_receipt_sha256") or fields.get("post_receipt_sha256") or fields.get("verification_receipt_sha256") or fields.get("post_step_receipt_hash") or "",
                            "post_step_checkpoint_path": fields.get("post_step_checkpoint_path") or fields.get("post_checkpoint") or fields.get("fresh_checkpoint") or "",
                            "post_step_checkpoint_sha256": fields.get("post_step_checkpoint_sha256") or fields.get("post_checkpoint_sha256") or fields.get("fresh_checkpoint_sha256") or fields.get("post_step_checkpoint_hash") or "",
                            "execution_health": fields.get("execution_health") or fields.get("health") or "",
                            "execution_audit": fields.get("execution_audit") or fields.get("audit") or "",
                            "after_action_learning": fields.get("after_action_learning") or fields.get("learning") or "",
                        },
                        "The user asked for a read-only autonomy cycle ledger before another continuation review.",
                    )
                ],
            )

        checkpoint_followthrough_match = re.search(
            r"^(?:checkpoint recovery followthrough|checkpoint recovery follow-through|recovery followthrough packet|recovery follow-through packet|checkpoint followthrough packet|checkpoint follow-through packet)\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if checkpoint_followthrough_match:
            body = checkpoint_followthrough_match.group("body").strip()
            keys = "step|reviewed_step|approved_step|verification|tests|receipt|receipt_path|receipt_sha256|receipt_hash|checkpoint|checkpoint_path|checkpoint_sha256|checkpoint_hash|stop|stop_condition|blockers|approval|approval_reference"
            field_pattern = rf"\b(?P<key>{keys})\s*=\s*(?P<value>.*?)(?=\s+\b(?:{keys})\s*=|$)"
            fields: dict[str, str] = {}
            for match in re.finditer(field_pattern, body, re.IGNORECASE | re.DOTALL):
                fields[match.group("key").lower()] = match.group("value").strip()
            return Plan(
                goal="Prove checkpoint recovery closure before normal follow-through resumes.",
                actions=[
                    PlannedAction(
                        "checkpoint_recovery_followthrough_packet",
                        {
                            "reviewed_step": fields.get("reviewed_step") or fields.get("approved_step") or fields.get("step") or "",
                            "verification": fields.get("verification") or fields.get("tests") or "",
                            "receipt_path": fields.get("receipt_path") or fields.get("receipt") or "",
                            "receipt_sha256": fields.get("receipt_sha256") or fields.get("receipt_hash") or "",
                            "checkpoint_path": fields.get("checkpoint_path") or fields.get("checkpoint") or "",
                            "checkpoint_sha256": fields.get("checkpoint_sha256") or fields.get("checkpoint_hash") or "",
                            "stop_condition": fields.get("stop_condition") or fields.get("stop") or "",
                            "blockers": fields.get("blockers") or "",
                            "approval_reference": fields.get("approval_reference") or fields.get("approval") or "",
                        },
                        "The user asked for a read-only checkpoint recovery follow-through packet.",
                    )
                ],
            )

        checkpoint_execute_match = re.search(
            r"^(?:checkpoint recovery execute|execute checkpoint recovery|apply reviewed checkpoint recovery|reviewed recovery execute)\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if checkpoint_execute_match:
            body = checkpoint_execute_match.group("body").strip()
            reviewed = bool(re.search(r"\b(?:reviewed|approved|confirmed)\s*=\s*(?:true|yes|1)\b", body, re.IGNORECASE))
            step_match = re.search(r"(?:step|approved_step)\s*=\s*(?P<value>.*?)(?=\s+(?:verification|tests|files|outcome|blockers|approval|reviewed)\s*=|$)", body, re.IGNORECASE | re.DOTALL)
            verification_match = re.search(r"(?:verification|tests)\s*=\s*(?P<value>.*?)(?=\s+(?:step|approved_step|files|outcome|blockers|approval|reviewed)\s*=|$)", body, re.IGNORECASE | re.DOTALL)
            files_match = re.search(r"(?:files|file)\s*=\s*(?P<value>.*?)(?=\s+(?:step|approved_step|verification|tests|outcome|blockers|approval|reviewed)\s*=|$)", body, re.IGNORECASE | re.DOTALL)
            outcome_match = re.search(r"outcome\s*=\s*(?P<value>.*?)(?=\s+(?:step|approved_step|verification|tests|files|blockers|approval|reviewed)\s*=|$)", body, re.IGNORECASE | re.DOTALL)
            blockers_match = re.search(r"blockers\s*=\s*(?P<value>.*?)(?=\s+(?:step|approved_step|verification|tests|files|outcome|approval|reviewed)\s*=|$)", body, re.IGNORECASE | re.DOTALL)
            approval_match = re.search(r"(?:approval|approval_reference)\s*=\s*(?P<value>.*?)(?=\s+(?:step|approved_step|verification|tests|files|outcome|blockers|reviewed)\s*=|$)", body, re.IGNORECASE | re.DOTALL)
            args = {
                "objective": "Resume one checkpointed Jarvis task.",
                "reviewed": reviewed,
                "approved_step": step_match.group("value").strip() if step_match else "",
                "verification": verification_match.group("value").strip() if verification_match else "",
                "files": files_match.group("value").strip() if files_match else "",
                "outcome": outcome_match.group("value").strip() if outcome_match else "",
                "blockers": blockers_match.group("value").strip() if blockers_match else "",
                "approval_reference": approval_match.group("value").strip() if approval_match else "",
            }
            return Plan(
                goal="Record a reviewed local-safe checkpoint recovery execution.",
                actions=[
                    PlannedAction(
                        "checkpoint_recovery_execute",
                        args,
                        "The user asked to execute a reviewed local-safe checkpoint recovery step.",
                    )
                ],
            )

        checkpoint_apply_match = re.search(
            r"^(?:checkpoint recovery apply|apply checkpoint recovery|recovery apply packet|checkpoint apply packet)\s*:?\s*(?P<objective>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if checkpoint_apply_match:
            objective = checkpoint_apply_match.group("objective").strip()
            return Plan(
                goal="Prepare an approval-gated checkpoint recovery apply packet.",
                actions=[
                    PlannedAction(
                        "checkpoint_recovery_apply_packet",
                        {"objective": objective} if objective else {},
                        "The user asked for a safe packet before applying checkpoint recovery.",
                    )
                ],
            )

        checkpoint_receipt_match = re.search(
            r"^(?:checkpoint recovery receipt|recovery receipt|save recovery receipt|checkpoint receipt)\s*:?\s*(?P<objective>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if checkpoint_receipt_match:
            objective = checkpoint_receipt_match.group("objective").strip()
            return Plan(
                goal="Save a checkpoint recovery receipt.",
                actions=[
                    PlannedAction(
                        "checkpoint_recovery_receipt",
                        {"objective": objective} if objective else {},
                        "The user asked to record a reviewed checkpoint recovery step.",
                    )
                ],
            )

        checkpoint_recovery_match = re.search(
            r"^(?:checkpoint recovery|recover checkpoint|resume from checkpoint|recovery preview|work block recovery)\s*:?\s*(?P<objective>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if checkpoint_recovery_match:
            objective = checkpoint_recovery_match.group("objective").strip()
            return Plan(
                goal="Preview recovery from the latest Jarvis work-block checkpoint.",
                actions=[
                    PlannedAction(
                        "checkpoint_recovery_preview",
                        {"objective": objective} if objective else {},
                        "The user asked how Jarvis should resume safely from the latest checkpoint.",
                    )
                ],
            )

        if low_command in {
            "save build delta",
            "write build delta",
            "export build delta",
            "save checkpoint delta",
            "write checkpoint delta",
            "export checkpoint delta",
        }:
            return Plan(
                goal="Save latest Jarvis build delta.",
                actions=[
                    PlannedAction(
                        "save_build_delta",
                        {},
                        "The user asked to save the latest Jarvis build checkpoint delta.",
                    )
                ],
            )

        # Real gap found live 2026-07-10 (round 43): _strip_trailing_politeness's
        # leading-strip side effect means verb-only entries like "show jarvis
        # notes" have no bare fallback in this set, so applying the full
        # leading+trailing strip upstream would break them -- see the
        # already-documented project_jarvis_strip_trailing_politeness_leading_strip
        # memory. Using the TRAILING-only strip here is safe (a no-op when there's
        # no trailing please/pls/thanks, so it can't break any existing entry) and
        # fixes "show my recent notes please"-style phrasing, which previously fell
        # through this exact-match set, then got discarded by the global politeness-
        # retry's POLITE_COMMAND_RETRY_TOOLS allowlist gate (list_jarvis_notes isn't
        # in it), landing in chat instead.
        note_list_command = _strip_trailing_politeness_suffix_only(low_command)
        if note_list_command in {
            "list jarvis notes",
            "list jarvis notes please",
            "show jarvis notes",
            "show latest jarvis notes",
            "list notes",
            "list notes please",
            "show notes",
            "show latest notes",
            "show my notes",
            "list my notes",
            "my notes",
            "notes",
            "notes please",
            "what notes do i have",
            # Real gap found live 2026-07-10, same class as the round-39 "list
            # my open tasks" bug: a natural qualifier word ("recent"/"latest")
            # inserted before "notes" broke every my-prefixed combination even
            # though the bare/show-latest forms above already worked.
            "recent notes",
            "latest notes",
            "show my recent notes",
            "list my recent notes",
            "my recent notes",
            "show my latest notes",
            "list my latest notes",
            "my latest notes",
            # Bare note-SEARCH phrases (no query given) belong here too: routing
            # them to a listing is the useful answer, and critically they must
            # never fall through to the generic web-search pattern -- observed in
            # review: bare "search my notes" hit web_lookup with query "my notes"
            # (public web) because the note-search regex requires a non-empty
            # query and nothing else claimed the bare phrase.
            "search notes",
            "search my notes",
            "find notes",
            "find my notes",
            "search my jarvis notes",
            "all jarvis notes",
            # Korean parity (Fable checkpoint plan item B2, 2026-07-10): all
            # probed Korean note-list phrasings fell to chat, which fabricated
            # "I don't have direct access to your notes". Exact-match only.
            "내 노트 보여줘",
            "내 메모 보여줘",
            "최근 메모 보여줘",
            "최근 노트 보여줘",
            "메모 목록",
            "노트 목록",
            "내 메모",
            "내 노트",
            "list obsidian notes",
            "show obsidian notes",
        }:
            return Plan(
                goal="List Jarvis Obsidian notes.",
                actions=[
                    PlannedAction(
                        "list_jarvis_notes",
                        {},
                        "The user asked for a metadata-only list of Jarvis notes.",
                    )
                ],
            )

        if low_command in {
            "recent saved notes",
            "recent jarvis notes",
            "saved notes",
            "saved jarvis notes",
            "what notes were saved",
            "what did jarvis save",
            "show saved notes",
        }:
            return Plan(
                goal="List recent Jarvis saved notes.",
                actions=[
                    PlannedAction(
                        "recent_saved_notes",
                        {},
                        "The user asked for a metadata-only list of recently saved Jarvis notes.",
                    )
                ],
            )

        if low in {
            "activity digest",
            "show activity digest",
            "recent activity",
            "show recent activity",
            "catch me up",
            "catch me up please",
            "catch up",
            "catch up please",
            "what changed",
            "what changed recently",
            "what happened",
        } or (
            ("what changed" in low or "catch me up" in low or "what happened" in low)
            and any(phrase in low for phrase in ("gone", "away", "since", "while"))
        ):
            return Plan(
                goal="Summarize recent Jarvis activity.",
                actions=[
                    PlannedAction(
                        "activity_digest",
                        {},
                        "The user asked for a continuity summary of recent Jarvis activity.",
                    )
                ],
            )

        if low_command in {
            "build progress",
            "jarvis build progress",
            "progress report",
            "build report",
            "what changed in jarvis",
            "what did you build",
        }:
            return Plan(
                goal="Summarize Jarvis build progress.",
                actions=[
                    PlannedAction(
                        "build_progress_report",
                        {},
                        "The user asked what has changed in the Jarvis build.",
                    )
                ],
            )

        if low_command in {
            "save build progress",
            "write build progress",
            "export build progress",
            "save progress report",
            "write progress report",
            "export progress report",
        }:
            return Plan(
                goal="Save Jarvis build progress.",
                actions=[
                    PlannedAction(
                        "save_build_progress",
                        {},
                        "The user asked to save the Jarvis build progress report.",
                    )
                ],
            )

        if low_command in {
            "safe next actions",
            "safe next steps",
            "what next",
            "what should we do next",
            "what should jarvis do next",
            "what should i work on next",
            "what should we work on next",
            "what should be continued",
            "continue what",
        }:
            return Plan(
                goal="Suggest safe next assistant actions.",
                actions=[
                    PlannedAction(
                        "safe_next_actions",
                        {},
                        "The user asked what Jarvis should safely continue next.",
                    )
                ],
            )

        natural_safe_next_match = re.search(
            r"^(?:what should (?:i|we|jarvis) work on (?:next|today|now)|what should (?:i|we|jarvis) do (?:next|today|now|right now)|what(?:'s| is) the next thing (?:i|we|jarvis) should do|what is the next safe move|what(?:'s| is) the next safe action)(?:\s+(?:for|on|in)\s+(?P<objective>.*\b(?:jarvis|jarvis\s+v2|agent\s+harness|ai\s+agent\s+harness|harness)\b.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_safe_next_match:
            return Plan(
                goal="Choose one safe next action packet.",
                actions=[
                    PlannedAction(
                        "next_action_packet",
                        {},
                        "The user asked in natural language for the next safe Jarvis move, so Jarvis should choose one inspectable action packet before acting.",
                    )
                ],
            )

        if low_command in {
            "next action packet",
            "safe action packet",
            "next move packet",
            "plan next action",
            "choose next action",
            "pick next action",
            "one next action",
            "next move",
            "one next move",
            "my next move",
            "what is my next move",
            "what's my next move",
            "next step",
            "one next step",
            "my next step",
            "what is my next step",
            "what's my next step",
            "what should i do now",
            "what should i do right now",
            "what should i do today",
            "what should i work on today",
            "what's the next thing i should do",
            "what is the next thing i should do",
            "send my next move",
            "send me my next move",
            "priority card",
            "my priority card",
            "send priority card",
            "send me a priority card",
            "send me my priority card",
            "show priority card",
            "show me my priority card",
        }:
            return Plan(
                goal="Choose one safe next action packet.",
                actions=[
                    PlannedAction(
                        "next_action_packet",
                        {},
                        "The user asked for one proposed next move with risk and verification before acting.",
                    )
                ],
            )

        if low_command in {
            "priority stack",
            "jarvis priority stack",
            "priority order",
            "rank next work",
            "rank next actions",
            "why this next",
            "what should come first",
        }:
            return Plan(
                goal="Rank Jarvis work priorities.",
                actions=[
                    PlannedAction(
                        "priority_stack",
                        {},
                        "The user asked Jarvis to rank approvals, tasks, goals, and setup before acting.",
                    )
                ],
            )

        natural_continuation_match = re.search(
            r"^(?:resume|continue|restart|start)\s+(?:the\s+)?(?:jarvis|jarvis\s+v2|agent\s+harness|ai\s+agent\s+harness|harness)\s+(?:work\s+)?(?:safely|safety-first|without risky actions|with guardrails)(?:\s*:?\s*(?P<objective>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_continuation_match:
            objective = (natural_continuation_match.group("objective") or "continue Jarvis V2 safely").strip()
            return Plan(
                goal="Prepare a safe Jarvis continuation packet.",
                actions=[
                    PlannedAction(
                        "continuation_packet",
                        {"objective": objective},
                        "The user asked in natural language to resume Jarvis work safely before acting.",
                    )
                ],
            )

        continuation_match = re.search(
            r"^(?:continuation packet|continue packet|safe continuation packet|build continuation packet|jarvis continuation packet)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE,
        )
        if continuation_match:
            return Plan(
                goal="Prepare a safe Jarvis continuation packet.",
                actions=[
                    PlannedAction(
                        "continuation_packet",
                        {"objective": (continuation_match.group("objective") or "").strip()},
                        "The user asked for a safe read-only continuation loop before more Jarvis build work.",
                    )
                ],
            )

        natural_build_target_match = re.search(
            r"^(?:what(?:'s| is) (?:the )?next (?:build )?target(?: for)?|what should (?:i|we|jarvis) build next(?: for| in| on)?|pick (?:the )?next build target(?: for)?|choose (?:the )?next build target(?: for)?)\s*:?\s*(?P<objective>.*\b(?:jarvis|jarvis\s+v2|agent\s+harness|ai\s+agent\s+harness|harness)\b.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_build_target_match:
            return Plan(
                goal="Select one safe Jarvis build target.",
                actions=[
                    PlannedAction(
                        "build_target_packet",
                        {"objective": natural_build_target_match.group("objective").strip()},
                        "The user asked in natural language for the next safe Jarvis build target with files, tests, and safety boundaries.",
                    )
                ],
            )

        build_target_match = re.search(
            r"^(?:build target packet|target packet|jarvis build target|safe build target|next build target)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE,
        )
        if build_target_match:
            return Plan(
                goal="Select one safe Jarvis build target.",
                actions=[
                    PlannedAction(
                        "build_target_packet",
                        {"objective": (build_target_match.group("objective") or "").strip()},
                        "The user asked for one scoped Jarvis build target with files, tests, and safety boundaries.",
                    )
                ],
            )

        natural_harness_slice_match = re.search(
            r"^(?:pick|choose|select|plan)\s+(?:the\s+)?next\s+(?:harness|agent harness|jarvis)\s+(?:slice|build slice|implementation slice)(?:\s+(?:for|on)\s+(?P<objective>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_harness_slice_match:
            objective = (natural_harness_slice_match.group("objective") or "continue building Jarvis V2 as an agent harness").strip()
            return Plan(
                goal="Select one autonomous Jarvis harness build slice.",
                actions=[
                    PlannedAction(
                        "harness_build_slice",
                        {"objective": objective},
                        "The user asked in natural language for the next Jarvis harness implementation slice.",
                    )
                ],
            )

        harness_build_slice_match = re.search(
            r"^(?:harness build slice|agent harness build slice|autonomous build slice|jarvis build slice|next harness slice)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE,
        )
        if harness_build_slice_match:
            return Plan(
                goal="Select one autonomous Jarvis harness build slice.",
                actions=[
                    PlannedAction(
                        "harness_build_slice",
                        {"objective": (harness_build_slice_match.group("objective") or "").strip()},
                        "The user asked for one implementation slice for safe autonomous Jarvis harness work.",
                    )
                ],
            )

        natural_jarvis_build_match = re.search(
            r"^(?:ok\s+)?(?:continue|keep|resume|work(?:\s+on)?|build|make|finish|improve|develop)\s+(?:building\s+|making\s+|working\s+on\s+)?(?P<objective>.*\b(?:jarvis|jarvis\s+v2|agent\s+harness|ai\s+agent\s+harness|harness)\b.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_jarvis_build_match:
            return Plan(
                goal="Select one autonomous Jarvis harness build slice.",
                actions=[
                    PlannedAction(
                        "harness_build_slice",
                        {"objective": natural_jarvis_build_match.group("objective").strip() or "continue building Jarvis V2 as an agent harness"},
                        "The user gave a natural Jarvis build continuation order, so Jarvis should select the next safe harness implementation slice instead of treating it as ordinary chat.",
                    )
                ],
            )

        save_build_target_match = re.search(
            r"^(?:save build target packet|write build target packet|export build target packet|save jarvis build target|save next build target)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE,
        )
        if save_build_target_match:
            return Plan(
                goal="Save one safe Jarvis build target.",
                actions=[
                    PlannedAction(
                        "save_build_target_packet",
                        {"objective": (save_build_target_match.group("objective") or "").strip()},
                        "The user asked to save a scoped Jarvis build target packet to Obsidian.",
                    )
                ],
            )

        if low_command in {
            "work queue",
            "jarvis work queue",
            "safe work queue",
            "action queue",
            "priority queue",
            "what is the queue",
        }:
            return Plan(
                goal="Show Jarvis work queue.",
                actions=[
                    PlannedAction(
                        "work_queue",
                        {},
                        "The user asked for an ordered safe work queue.",
                    )
                ],
            )

        if low_command in {
            "save work queue",
            "write work queue",
            "export work queue",
            "save jarvis work queue",
        }:
            return Plan(
                goal="Save Jarvis work queue.",
                actions=[
                    PlannedAction(
                        "save_work_queue",
                        {},
                        "The user asked to save the ordered safe work queue.",
                    )
                ],
            )

        if low_command in {
            "mission control",
            "update mission control",
            "export mission control",
            "write mission control",
            "refresh mission control",
        }:
            return Plan(
                goal="Write Jarvis Mission Control note.",
                actions=[
                    PlannedAction(
                        "export_mission_control",
                        {},
                        "The user asked to refresh the Obsidian mission-control note.",
                    )
                ],
            )

        organize_match = re.search(
            r"^(?:organize|process|capture)\s+(?:brain\s+dump|dump|note|notes)\s*:?\s*(?P<text>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if organize_match:
            return Plan(
                goal="Organize a brain dump into Jarvis records.",
                actions=[
                    PlannedAction(
                        "organize_note",
                        {"text": organize_match.group("text").strip()},
                        "The user asked Jarvis to organize a note or brain dump.",
                    )
                ],
            )

        profile_match = re.search(
            r"^(?:(?:add|save)\s+(?:profile\s+)?(?:note|context)|profile\s+(?:note|context))\s+(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if profile_match:
            raw_body = profile_match.group("body").strip()
            explicit_heading = re.fullmatch(
                r"(?P<heading>[^:\n]{1,120}):\s+(?P<body>.+)",
                raw_body,
                re.DOTALL,
            )
            if explicit_heading is not None:
                heading = explicit_heading.group("heading").strip()
                body = explicit_heading.group("body").strip()
            else:
                heading = raw_body[:60].rstrip(".")
                body = raw_body
            return Plan(
                goal="Add profile note.",
                actions=[
                    PlannedAction(
                        "add_profile_note",
                        {"heading": heading, "body": body},
                        "The user asked to save curated profile context.",
                    )
                ],
            )

        natural_self_knowledge = re.sub(r"\s+", " ", text.lower()).strip("?.! ")
        natural_self_knowledge = re.sub(
            r"\s+(?:please|pls|thanks|thank you)$",
            "",
            natural_self_knowledge,
            flags=re.IGNORECASE,
        ).strip()
        if natural_self_knowledge in {
            "what do you know about me",
            "tell me about myself",
            "what are my current goals",
            "show my open commitments",
            "what are my open commitments",
            "나에 대해 뭘 알아",
            "나에 대해 무엇을 알아",
            "나에 대해 알려줘",
            "나에 대해 알려주세요",
            "저에 대해 뭘 알아",
            "저에 대해 알려주세요",
            "자비스가 나에 대해 뭘 기억해",
            "자비스가 나에 대해 무엇을 기억해",
        }:
            return Plan(
                goal="Review bounded personal context.",
                actions=[
                    PlannedAction(
                        "chat_context",
                        {"prompt": text},
                        "The user asked for their bounded local personal state.",
                    )
                ],
            )

        if low in {
            "profile",
            "profile please",
            "profile notes",
            "profile notes please",
            "read profile",
            "read profile please",
            "show profile",
            "show profile please",
            "show latest profile",
            "show latest profile please",
            "jarvis profile",
            "my profile",
            "my profile please",
            "read my profile",
            "read my profile please",
            "show my profile",
            "show my profile please",
            "what's in my profile",
            "what is in my profile",
            "about me",
            "tell me about me",
            "what is my name",
            "who am i",
            "what do you know about the operator",
            "profile me",
            "profile me please",
        }:
            return Plan(
                goal="Read profile.",
                actions=[PlannedAction("read_profile", {}, "The user asked to read the profile.")],
            )

        knowledge_about_match = re.search(
            r"^what\s+do\s+you\s+know\s+about\s+(?P<query>.+)$",
            text,
            re.IGNORECASE,
        )
        if knowledge_about_match:
            query = _strip_trailing_politeness(knowledge_about_match.group("query")).strip()
            if query:
                return Plan(
                    goal=f"Search memory for {query}.",
                    actions=[PlannedAction("search_memory", {"query": query}, "The user asked what Jarvis knows about a topic.")],
                )

        # Real gap found live 2026-07-10 (round 44): unbounded lazy captures
        # (`.+?`) here cause quadratic-time backtracking on long non-matching
        # input (a preference command with no "to"/"as"/"=" anywhere) -- 20,000
        # whitespace chars took 2.25s, growing quadratically with length.
        # Bounded to generous limits no real preference key/value would ever
        # need (100/1000 chars) -- confirmed fast at 20,000+ chars after.
        preference_match = re.search(
            r"^(?:set|save|remember)\s+preference\s+(?P<key>.{1,100}?)\s+(?:to|as|=)\s+(?P<value>.{1,1000}?)(?:\s+category\s+(?P<category>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if preference_match:
            return Plan(
                goal="Save preference.",
                actions=[
                    PlannedAction(
                        "set_preference",
                        {
                            "key": preference_match.group("key").strip(),
                            "value": preference_match.group("value").strip(),
                            "category": (preference_match.group("category") or "general").strip(),
                        },
                        "The user asked Jarvis to save a structured preference.",
                    )
                ],
            )

        simple_preference_match = re.search(
            r"^(?:set|save|remember)\s+preference\s+(?P<key>[a-zA-Z][\w-]{0,50})\s+(?P<value>.+?)(?:\s+category\s+(?P<category>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if simple_preference_match:
            return Plan(
                goal="Save preference.",
                actions=[
                    PlannedAction(
                        "set_preference",
                        {
                            "key": simple_preference_match.group("key").strip(),
                            "value": simple_preference_match.group("value").strip(),
                            "category": (simple_preference_match.group("category") or "general").strip(),
                        },
                        "The user asked Jarvis to save a structured preference.",
                    )
                ],
            )

        # Real gap found live 2026-07-10 (round 43), same class as the notes
        # fix above: a trailing-only politeness strip is safe here (no-op when
        # nothing to strip) and fixes "show my recent preferences please"-style
        # phrasing that previously fell through to chat.
        preference_list_command = _strip_trailing_politeness_suffix_only(low)
        if preference_list_command in {
            "preferences",
            "preferences please",
            "preference list",
            "preference list please",
            "list preferences",
            "list preferences please",
            "show preferences",
            "show preferences please",
            "show my preferences",
            "show my preferences please",
            "show latest preferences",
            "show latest preferences please",
            "my preferences",
            "my preferences please",
            # Real gap found live 2026-07-10, same class as the round-39 "list
            # my open tasks" bug: "list preferences" and "show my preferences"
            # both worked, but "list my preferences" (the natural combination
            # of the two) and any "recent" qualifier fell through to chat.
            "list my preferences",
            "list my preferences please",
            "recent preferences",
            "show my recent preferences",
            "list my recent preferences",
            "my recent preferences",
            "what are my preferences",
            "what preferences do you remember",
            "what preferences do you have for me",
            "what preferences do you know",
            "what preferences do you know please",
            # Korean parity (Fable checkpoint plan item B2, 2026-07-10).
            "내 선호 보여줘",
            "선호 목록",
            "내 선호",
        }:
            return Plan(
                goal="List preferences.",
                actions=[PlannedAction("list_preferences", {}, "The user asked to list preferences.")],
            )

        if low in {
            "all preferences",
            "all preferences please",
            "list all preferences",
            "list all preferences please",
            "show all preferences",
            "show all preferences please",
        }:
            return Plan(
                goal="List all preferences.",
                actions=[PlannedAction("list_preferences", {"status": "all"}, "The user asked to list all preferences.")],
            )

        get_preference_match = re.search(r"^(?:get|show|inspect)\s+preference\s+(?P<key>.+)$", text, re.IGNORECASE)
        if get_preference_match:
            return Plan(
                goal="Inspect preference.",
                actions=[
                    PlannedAction(
                        "get_preference",
                        {"key": get_preference_match.group("key").strip()},
                        "The user asked to inspect one preference.",
                    )
                ],
            )

        preference_status_match = re.search(
            r"^(?:set\s+)?preference\s+#?(?P<preference_id>\d+)\s+(?:to\s+)?(?P<status>active|retired)$",
            text,
            re.IGNORECASE,
        )
        if preference_status_match:
            return Plan(
                goal="Set preference status.",
                actions=[
                    PlannedAction(
                        "set_preference_status",
                        {
                            "preference_id": int(preference_status_match.group("preference_id")),
                            "status": preference_status_match.group("status").lower(),
                        },
                        "The user asked to update a preference status.",
                    )
                ],
            )

        add_person_match = re.search(
            r"^(?:add|save|remember)\s+person\s+(?P<name>.+?)(?:\s+relation\s+(?P<relation>.+?))?(?:\s+notes?\s+(?P<notes>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if add_person_match:
            return Plan(
                goal="Save person profile.",
                actions=[
                    PlannedAction(
                        "add_person",
                        {
                            "name": add_person_match.group("name").strip(),
                            "relation": (add_person_match.group("relation") or "").strip(),
                            "notes": (add_person_match.group("notes") or "").strip(),
                        },
                        "The user asked Jarvis to save a person profile.",
                    )
                ],
            )

        # Real gap found live 2026-07-10 (round 43), same class as the notes
        # fix above: a trailing-only politeness strip is safe here (no-op when
        # nothing to strip) and fixes "show my recent people please"-style
        # phrasing that previously fell through to chat.
        people_list_command = _strip_trailing_politeness_suffix_only(low)
        if people_list_command in {
            "people",
            "people please",
            "person list",
            "person list please",
            "people list",
            "people list please",
            "list people",
            "list people please",
            "show people",
            "show people please",
            "show latest people",
            "show latest people please",
            "saved people",
            "saved people please",
            "show saved people",
            "show saved people please",
            "who do you know",
            "who do you know please",
            "show my people",
            "show my people please",
            "who are my people",
            "who are my people please",
            "people i know",
            "people i know please",
            # Korean parity (Fable checkpoint plan item B2, 2026-07-10).
            "사람 목록",
            "인물 목록",
            "아는 사람 보여줘",
            # Real gap found live 2026-07-10, same class as the round-39/40
            # list-command qualifier-word bug: "show my people" worked but
            # "my people"/"list my people" (bare or "list"-prefixed) fell
            # through to chat, and no "recent" qualifier was recognized.
            "my people",
            "list my people",
            "list my people please",
            "recent people",
            "show my recent people",
            "list my recent people",
            "my recent people",
        }:
            return Plan(
                goal="List people.",
                actions=[PlannedAction("list_people", {}, "The user asked to list saved people.")],
            )

        get_person_match = re.search(
            r"^(?:(?:show|get|inspect)\s+)?person\s+(?P<name_or_id>.+?)(?:\s+please)?$",
            text,
            re.IGNORECASE,
        )
        if get_person_match:
            name_or_id = get_person_match.group("name_or_id").strip()
            if name_or_id.lower() == "please":
                return Plan()
            args = {"person_id": int(name_or_id.lstrip("#"))} if name_or_id.lstrip("#").isdigit() else {"name": name_or_id}
            return Plan(
                goal="Inspect person.",
                actions=[PlannedAction("get_person", args, "The user asked to inspect a saved person.")],
            )

        interaction_match = re.search(
            r"^(?:log|record)\s+(?:interaction|contact)\s+(?:with\s+)?(?P<name>.+?)\s*:?\s+(?P<summary>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if interaction_match:
            return Plan(
                goal="Log person interaction.",
                actions=[
                    PlannedAction(
                        "log_interaction",
                        {
                            "name": interaction_match.group("name").strip(),
                            "summary": interaction_match.group("summary").strip(),
                        },
                        "The user asked to log an interaction with a person.",
                    )
                ],
            )

        decision_match = re.search(
            # Real gap found live 2026-07-10, same compound-sentence
            # clause-bleed class as this session's round-33 fixes: "record
            # decision use postgres and then show my goals" would have
            # written a decision literally titled "use postgres and then
            # show my goals" instead of "use postgres", silently dropping
            # the second intent.
            r"^(?:record|save|log|add)\s{1,10}(?:a\s{1,10}|an\s{1,10})?decision(?:\s{1,10}to)?\s{1,10}(?P<title>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?(?:\s{1,10}because\s{1,10}(?P<rationale>.+?))?(?:\s{1,10}impact\s{1,10}(?P<impact>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if decision_match:
            return Plan(
                goal="Record decision.",
                actions=[
                    PlannedAction(
                        "record_decision",
                        {
                            "title": decision_match.group("title").strip(),
                            "rationale": (decision_match.group("rationale") or "").strip(),
                            "impact": (decision_match.group("impact") or "").strip(),
                        },
                        "The user asked Jarvis to record a durable decision.",
                    )
                ],
            )

        decision_list_alias_match = re.search(
            r"^(?:(?:show|list|open|preview)\s+)?(?:my\s+)?(?:(?:me|latest|current|newest|recent)\s+)?(?:decisions?|decision\s+list|decision\s+log)(?:\s+(?:please|pls|thanks|thank you))?$"
            r"|^what\s+decisions\s+did\s+we\s+make(?:\s+(?:please|pls|thanks|thank you))?$"
            r"|^what\s+decisions\s+(?:have\s+)?(?:i|we)\s+made(?:\s+(?:please|pls|thanks|thank you))?$",
            text,
            re.IGNORECASE,
        )
        if decision_list_alias_match:
            return Plan(
                goal="List decisions.",
                actions=[PlannedAction("list_decisions", {"status": "active"}, "The user asked to list durable decisions.")],
            )

        if re.search(r"^(?:(?:show|list|open|preview)\s+)?all\s+decisions(?:\s+(?:please|pls|thanks|thank you))?$", text, re.IGNORECASE):
            return Plan(
                goal="List all decisions.",
                actions=[PlannedAction("list_decisions", {"status": "all"}, "The user asked to list all decisions.")],
            )

        get_decision_match = re.search(r"^(?:get|show|inspect)\s+decision\s+#?(?P<decision_id>\d+)$", text, re.IGNORECASE)
        if get_decision_match:
            return Plan(
                goal="Inspect decision.",
                actions=[
                    PlannedAction(
                        "get_decision",
                        {"decision_id": int(get_decision_match.group("decision_id"))},
                        "The user asked to inspect one decision.",
                    )
                ],
            )

        decision_status_match = re.search(
            r"^(?:set\s+)?decision\s+#?(?P<decision_id>\d+)\s+(?:to\s+)?(?P<status>active|superseded|retired)$",
            text,
            re.IGNORECASE,
        )
        if decision_status_match:
            return Plan(
                goal="Set decision status.",
                actions=[
                    PlannedAction(
                        "set_decision_status",
                        {
                            "decision_id": int(decision_status_match.group("decision_id")),
                            "status": decision_status_match.group("status").lower(),
                        },
                        "The user asked to update a decision status.",
                    )
                ],
            )

        note_list_match = re.search(
            r"^(?:list|show)\s+(?:jarvis\s+)?(?:notes|obsidian)\s+(?:in|under)\s+(?P<folder>.+)$",
            text,
            re.IGNORECASE,
        )
        if note_list_match:
            return Plan(
                goal="List Jarvis Obsidian notes.",
                actions=[
                    PlannedAction(
                        "list_jarvis_notes",
                        {"folder": note_list_match.group("folder").strip()},
                        "The user asked for a metadata-only list of Jarvis notes in a folder.",
                    )
                ],
            )

        note_search_match = re.search(
            # Real gap found live 2026-07-10: "search my notes for the budget
            # and then show my tasks" (a compound sentence) swallowed the
            # whole second clause into the search query, silently searching
            # for "the budget and then show my tasks" instead of "the
            # budget" -- and silently dropping the "show my tasks" intent.
            # Applied to all three alternatives below.
            r"^(?:search|find)\s{1,10}(?:my\s{1,10})?(?:jarvis\s{1,10})?(?:notes?|obsidian)\s{1,10}(?:for|about)?\s*(?P<query>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?$"
            r"|^(?:show|list)\s{1,10}(?:my\s{1,10})?(?:jarvis\s{1,10})?(?:notes?|obsidian)\s{1,10}(?:for|about)\s*(?P<display_query>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?$"
            # Real gap found live 2026-07-10, same privacy-relevant misroute
            # class as the round-21 "find my note about X" fix but a different
            # word order: "search for the/a note about X" and "find the/a note
            # about X" (verb + preposition/article BEFORE the noun, rather
            # than verb + optional-possessive + noun) fell through this regex
            # entirely and hit the generic public web-search fallback instead.
            r"|^(?:search for|find)\s{1,10}(?:the\s{1,10}|a\s{1,10}|my\s{1,10})?(?:jarvis\s{1,10})?notes?\s{1,10}(?:for|about)\s{1,10}(?P<query2>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?$",
            text,
            re.IGNORECASE,
        )
        if note_search_match:
            return Plan(
                goal="Search Jarvis Obsidian notes.",
                actions=[
                    PlannedAction(
                        "search_jarvis_notes",
                        {
                            "query": _strip_trailing_politeness(
                                note_search_match.group("query")
                                or note_search_match.group("display_query")
                                or note_search_match.group("query2")
                            ).strip()
                        },
                        "The user asked to search Jarvis notes.",
                    )
                ],
            )

        outline_note_match = re.search(
            r"^(?:outline|summarize|inspect)\s+(?:jarvis\s+)?note\s+(?P<path>.+)$",
            text,
            re.IGNORECASE,
        )
        if outline_note_match:
            return Plan(
                goal="Outline Jarvis Obsidian note.",
                actions=[
                    PlannedAction(
                        "outline_jarvis_note",
                        {"path": outline_note_match.group("path").strip()},
                        "The user asked for a read-only outline of a Jarvis note.",
                    )
                ],
            )

        read_note_match = re.search(
            r"^(?:read|open|show)\s+(?:(?:jarvis|my)\s+){0,2}notes?(?:\s*[:,]\s*|\s+)(?P<path>.+)$",
            text,
            re.IGNORECASE,
        )
        if read_note_match:
            return Plan(
                goal="Read Jarvis Obsidian note.",
                actions=[
                    PlannedAction(
                        "read_jarvis_note",
                        {
                            "path": _strip_trailing_politeness(
                                read_note_match.group("path")
                            ).rstrip(" ?!.")
                        },
                        "The user asked to read a Jarvis note.",
                    )
                ],
            )

        write_note_match = re.search(
            r"^(?P<verb>write|create|append|save)\s+(?:to\s+)?(?:jarvis\s+)?notes?\s+(?P<path>[^:]+):\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if write_note_match:
            verb = write_note_match.group("verb").lower()
            return Plan(
                goal="Write Jarvis Obsidian note.",
                actions=[
                    PlannedAction(
                        "write_jarvis_note",
                        {
                            "path": write_note_match.group("path").strip(),
                            "body": write_note_match.group("body").strip(),
                            "mode": "create" if verb == "create" else "append",
                        },
                        "The user asked to write a local Jarvis-owned Obsidian note.",
                    )
                ],
            )

        capture_note_match = re.search(
            r"^(?:capture\s+(?:jarvis\s+)?note|save\s+to\s+(?:jarvis\s+)?notes?|write\s+this\s+down)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if capture_note_match:
            return Plan(
                goal="Capture Jarvis Obsidian note.",
                actions=[
                    PlannedAction(
                        "write_jarvis_note",
                        {
                            "path": "Inbox/Captured Notes.md",
                            "body": capture_note_match.group("body").strip(),
                            "mode": "append",
                        },
                        "The user asked to capture a local Jarvis-owned Obsidian note.",
                    )
                ],
            )

        conversation_search_match = re.search(
            r"^(?:(?:search|find)\s+(?:conversation|conversations|chat|chats)\s+(?:(?:for|about)\s+)?|(?:show|review)\s+(?:conversation|conversations|chat|chats)\s+(?:for|about)\s+|(?:conversation|conversations|chat|chats)\s+search\s+)(?P<query>.+)$",
            text,
            re.IGNORECASE,
        )
        if conversation_search_match:
            query = _strip_trailing_politeness(conversation_search_match.group("query")).strip()
            if query:
                return Plan(
                    goal="Search conversation history.",
                    actions=[
                        PlannedAction(
                            "search_conversations",
                            {"query": query},
                            "The user asked to search conversation history.",
                        )
                    ],
                )

        recent_conversation_display_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current|recent)\s+)?(?P<kind>conversation(?:s)?|chat(?:s)?|conversation history|chat history)$",
            text,
            re.IGNORECASE,
        )
        recent_conversation_question_match = re.search(
            r"^(?:what did we(?: just)? (?:talk about|discuss)|what were we(?: just)? talking about|what was (?:our|the) (?:last|latest|recent) (?:conversation|chat))$",
            text,
            re.IGNORECASE,
        )
        if (
            low
            in {
                "conversation",
                "recent conversation",
                "recent conversations",
                "recent chat",
                "recent chats",
                "show recent chat",
                "conversation history",
                "chat history",
            }
            or recent_conversation_display_match
            or recent_conversation_question_match
        ):
            return Plan(
                goal="Show recent conversation.",
                actions=[PlannedAction("recent_conversation", {}, "The user asked to see recent conversation history.")],
            )

        if low in {
            "chat response health",
            "chat health",
            "conversation health",
            "chat source health",
            "chat fallback report",
            "chat model fallback report",
        }:
            return Plan(
                goal="Show chat response health.",
                actions=[
                    PlannedAction(
                        "chat_response_health",
                        {},
                        "The user asked to inspect recent chat response sources and fallback state.",
                    )
                ],
            )

        chat_diagnostic_display_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:latest|last|newest|current)\s+(?P<kind>chat response health|chat health|conversation health|chat source health|chat fallback report|chat model fallback report|chat safety(?: report)?|conversation safety(?: report)?|chat continuity brief|conversation continuity brief|conversation catch-?up|chat catch-?up)$",
            text,
            re.IGNORECASE,
        )
        if chat_diagnostic_display_match:
            kind = chat_diagnostic_display_match.group("kind").lower().replace("-", "")
            if "safety" in kind:
                return Plan(
                    goal="Show chat safety report.",
                    actions=[
                        PlannedAction(
                            "chat_safety_report",
                            {},
                            "The user asked to inspect latest chat safety boundaries in natural wording.",
                        )
                    ],
                )
            if "continuity" in kind or "catchup" in kind:
                return Plan(
                    goal="Show chat continuity brief.",
                    actions=[
                        PlannedAction(
                            "chat_continuity_brief",
                            {},
                            "The user asked to inspect latest chat continuity in natural wording.",
                        )
                    ],
                )
            return Plan(
                goal="Show chat response health.",
                actions=[
                    PlannedAction(
                        "chat_response_health",
                        {},
                        "The user asked to inspect latest chat response health in natural wording.",
                    )
                ],
            )

        if low in {
            "chat continuity brief",
            "conversation continuity brief",
            "conversation catchup",
            "conversation catch-up",
            "chat catchup",
            "chat catch-up",
            "where did we leave off",
            "continue from last time",
            "continue where we left off",
            "what were we doing",
            "what were we working on",
            "what were we building",
            "what was i doing with jarvis",
        }:
            return Plan(
                goal="Show chat continuity brief.",
                actions=[
                    PlannedAction(
                        "chat_continuity_brief",
                        {},
                        "The user asked for a read-only catch-up on the current conversation thread.",
                    )
                ],
            )

        if low in {
            "chat safety",
            "chat safety report",
            "conversation safety",
            "conversation safety report",
            "is chat grounded",
            "is jarvis chat grounded",
            "how is chat safe",
            "how is jarvis chat safe",
        }:
            return Plan(
                goal="Show chat safety report.",
                actions=[
                    PlannedAction(
                        "chat_safety_report",
                        {},
                        "The user asked how conversational answers are grounded and kept safe.",
                    )
                ],
            )

        session_list_match = re.search(
            r"^(?:show|view|review|inspect|preview|list)\s+(?:me\s+|my\s+)?(?:the\s+)?(?:(?:latest|last|newest|current|recent)\s+)?(?P<kind>sessions|conversation sessions|chat sessions|session list|conversation session list|chat session list)$",
            text,
            re.IGNORECASE,
        )
        if low in {
            "sessions",
            "list sessions",
            "conversation sessions",
            "chat sessions",
            "session list",
            "conversation session list",
            "chat session list",
        } or session_list_match:
            return Plan(
                goal="List conversation sessions.",
                actions=[PlannedAction("list_sessions", {}, "The user asked to list conversation sessions.")],
            )

        if low in {"export session", "export this session", "save session", "save this session"}:
            return Plan(
                goal="Export current session.",
                actions=[PlannedAction("export_session", {}, "The user asked to export the current session.")],
            )

        if low in {"summarize session", "summarize this session", "reflect session", "session summary"}:
            return Plan(
                goal="Summarize current session.",
                actions=[PlannedAction("summarize_session", {}, "The user asked to summarize the current session.")],
            )

        chat_context_display_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:latest|last|newest|current)?\s*(?P<kind>chat context|conversation context|brain context|chat prompt preview|conversation prompt preview|chat prompt|conversation prompt|session learning preview|session learning)$",
            text,
            re.IGNORECASE,
        )
        if chat_context_display_match:
            kind = chat_context_display_match.group("kind").lower()
            if "session learning" in kind:
                return Plan(
                    goal="Preview possible learning from this session.",
                    actions=[
                        PlannedAction(
                            "session_learning_preview",
                            {},
                            "The user asked to inspect the latest session-learning preview in natural wording.",
                        )
                    ],
                )
            if "prompt" in kind:
                return Plan(
                    goal="Preview chat prompt packet.",
                    actions=[
                        PlannedAction(
                            "chat_prompt_preview",
                            {"prompt": ""},
                            "The user asked to preview the conversational prompt packet in natural wording.",
                        )
                    ],
                )
            return Plan(
                goal="Preview chat context.",
                actions=[
                    PlannedAction(
                        "chat_context",
                        {"prompt": ""},
                        "The user asked to preview the conversational brain context in natural wording.",
                    )
                ],
            )

        chat_context_match = re.search(
            r"^(?:chat context|conversation context|brain context|preview chat context)(?::?\s*(?P<prompt>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if chat_context_match:
            return Plan(
                goal="Preview chat context.",
                actions=[
                    PlannedAction(
                        "chat_context",
                        {"prompt": (chat_context_match.group("prompt") or "").strip()},
                        "The user asked to preview the conversational brain context.",
                    )
                ],
            )

        chat_prompt_preview_match = re.search(
            r"^(?:chat prompt preview|conversation prompt preview|preview chat prompt|preview conversation prompt)(?::?\s*(?P<prompt>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if chat_prompt_preview_match:
            return Plan(
                goal="Preview chat prompt packet.",
                actions=[
                    PlannedAction(
                        "chat_prompt_preview",
                        {"prompt": (chat_prompt_preview_match.group("prompt") or "").strip()},
                        "The user asked to preview the conversational prompt packet and grounding boundaries.",
                    )
                ],
            )

        assistant_turn_match = re.search(
            r"^(?:assistant turn rehearsal|turn rehearsal|chat rehearsal|message rehearsal|rehearse assistant turn)(?::?\s*(?P<message>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if assistant_turn_match:
            return Plan(
                goal="Preview a full assistant turn.",
                actions=[
                    PlannedAction(
                        "assistant_turn_rehearsal",
                        {"message": assistant_turn_match.group("message").strip()},
                        "The user asked to preview whether a message becomes chat or tool execution.",
                    )
                ],
            )

        save_chat_context_match = re.search(
            r"^(?:save chat context|write chat context|export chat context|save brain context)(?::?\s*(?P<prompt>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if save_chat_context_match:
            return Plan(
                goal="Save chat context preview.",
                actions=[
                    PlannedAction(
                        "save_chat_context",
                        {"prompt": (save_chat_context_match.group("prompt") or "").strip()},
                        "The user asked to save the conversational brain context preview.",
                    )
                ],
            )

        if low in {
            "session learning preview",
            "latest session learning preview",
            "last session learning preview",
            "current session learning preview",
            "learning preview",
            "preview session learning",
            "what should jarvis learn from this session",
            "what should you learn from this session",
        }:
            return Plan(
                goal="Preview possible learning from this session.",
                actions=[
                    PlannedAction(
                        "session_learning_preview",
                        {},
                        "The user asked to review possible memories, preferences, tasks, or skills from the current session without saving them.",
                    )
                ],
            )

        draft_skill_match = re.search(
            r"^(?:draft|propose|create)\s+(?:a\s+)?skill\s+from\s+(?:this\s+)?session(?:\s+called\s+(?P<name>.+))?$",
            text,
            re.IGNORECASE,
        )
        if draft_skill_match:
            return Plan(
                goal="Draft skill from session.",
                actions=[
                    PlannedAction(
                        "draft_skill_from_session",
                        {"name": (draft_skill_match.group("name") or "").strip()},
                        "The user asked Jarvis to draft a reusable skill from the current session.",
                    )
                ],
            )

        goal_list_alias_match = re.search(
            r"^(?:(?:show|list|find|open|preview)(?:\s+(?:me|my|latest|current|newest))?\s+)?(?:my\s+)?(?:goals?|goal\s+list|projects?|project\s+list)(?:\s+(?:please|pls|thanks|thank you))?$",
            text,
            re.IGNORECASE,
        )
        if goal_list_alias_match:
            return Plan(
                goal="List goals.",
                actions=[PlannedAction("list_goals", {"status": "active"}, "The user asked to list tracked goals.")],
            )

        next_actions_alias_match = re.search(
            r"^(?:(?:show|list|find|open|preview)(?:\s+(?:me|my|latest|current|newest))?\s+)?(?:my\s+)?next\s+actions?(?:\s+(?:please|pls|thanks|thank you))?$",
            text,
            re.IGNORECASE,
        )
        if next_actions_alias_match:
            return Plan(
                goal="List next goal actions.",
                actions=[PlannedAction("next_actions", {}, "The user asked what to work on next.")],
            )

        goal_proactive_command = text.strip()
        for _ in range(4):
            stripped_goal_command = re.sub(
                r"^(?:(?:hey|ok|okay|hi|hello)\W+)?jarvis\W+",
                "",
                goal_proactive_command,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            stripped_goal_command = LEADING_POLITE_WRAPPER_RE.sub(
                "",
                stripped_goal_command,
                count=1,
            ).strip()
            if stripped_goal_command == goal_proactive_command:
                break
            goal_proactive_command = stripped_goal_command
        goal_proactive_command = _strip_trailing_politeness_suffix_only(
            goal_proactive_command
        ).lower().rstrip(" ?!.")
        if re.search(r"\b(?:goal nudges?|stale goals?)\b", goal_proactive_command):
            if re.match(
                r"^(?:don'?t|do\s+not|not(?:\s+to)?|never\s*mind|no\s+need\s+(?:to|for))\b"
                r"|^(?:cancel that|actually|wait|hold on),?\s+(?:don'?t|do\s+not)\b",
                goal_proactive_command,
            ):
                return Plan(
                    goal="Respond conversationally.",
                    actions=[],
                    needs_model=True,
                    notes="Negated goal-nudge request routed to chat instead of executing an action.",
                )
            if re.fullmatch(
                r"schedule\s+(?:(?:a|the)\s+)?(?:daily\s+)?(?:goal nudges?|stale goals?)(?:\s+(?:daily|every day))?",
                goal_proactive_command,
            ):
                return Plan(
                    goal="Schedule goal nudge.",
                    actions=[
                        PlannedAction(
                            "schedule_goal_nudge",
                            {},
                            "The user asked to schedule daily goal nudges.",
                        )
                    ],
                )
            if re.match(r"^schedule\b", goal_proactive_command):
                return Plan(
                    goal="Clarify the goal nudge schedule.",
                    actions=[
                        PlannedAction(
                            "respond",
                            {
                                "text": (
                                    "Goal nudges currently support the default daily schedule. "
                                    "Say 'schedule goal nudge' or 'schedule goal nudge daily'."
                                )
                            },
                            "Unsupported timing details must not silently become a daily schedule.",
                        )
                    ],
                )
            run_goal_nudge_match = re.fullmatch(
                r"(?:run|start|trigger)\s+(?:the\s+)?(?:job\s+)?(?:goal nudges?|stale goals?)(?:\s+job)?\s+now",
                goal_proactive_command,
            )
            if run_goal_nudge_match:
                return Plan(
                    goal="Run scheduled job now.",
                    actions=[
                        PlannedAction(
                            "run_job_now",
                            {"name": "Goal Nudge"},
                            "The user asked to run the Goal Nudge job immediately.",
                        )
                    ],
                )
            return Plan(
                goal="Generate goal nudge.",
                actions=[PlannedAction("goal_nudge", {}, "The user asked for stale-goal nudges.")],
            )

        create_goal_match = re.search(
            # Real gap found live 2026-07-10, same compound-sentence
            # clause-bleed class as this session's round-33 add_task/
            # create_reminder/find_contact/search fixes: "create goal learn
            # spanish and then show my tasks" would have written a goal
            # literally titled "learn spanish and then show my tasks"
            # instead of "learn spanish", silently dropping the second
            # intent.
            r"^(?:(?:create|add|start|new|set)\s{1,10}(?:a\s{1,10}|an\s{1,10})?(?:goal|project)\s{1,10}(?:to\s{1,10})?|(?:goal|project)\s{1,10}(?!#?\d+\b))(?P<title>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?(?:\s{1,10}(?:because|purpose|so that)\s{1,10}(?P<purpose>.+?))?(?:\s{1,10}(?:by|horizon)\s{1,10}(?P<horizon>.+))?$",
            text,
            re.IGNORECASE,
        )
        if create_goal_match:
            return Plan(
                goal="Create a durable goal.",
                actions=[
                    PlannedAction(
                        "create_goal",
                        {
                            "title": create_goal_match.group("title").strip(),
                            "purpose": (create_goal_match.group("purpose") or "").strip(),
                            "horizon": (create_goal_match.group("horizon") or "").strip(),
                        },
                        "The user asked Jarvis to track a goal.",
                    )
                ],
            )

        # Real gap found live 2026-07-10 (round 43), same class as the notes
        # fix above: a trailing-only politeness strip is safe here (no-op when
        # nothing to strip) and fixes "show my active goals please"-style
        # phrasing that previously fell through to chat.
        goal_list_command = _strip_trailing_politeness_suffix_only(low)
        if goal_list_command in {
            "goals",
            "list goals",
            "show goals",
            "show my goals",
            "my goals",
            "what are my goals",
            "what goals do i have",
            "active goals",
            "show active goals",
            "list active goals",
            "my active goals",
            # Real gap found live 2026-07-10, same class as the round-39 "list
            # my open tasks" bug: "my active goals" alone worked, but every
            # show/list-prefixed combination of the same qualifier didn't.
            "show my active goals",
            "list my active goals",
            "current goals",
            "show current goals",
            "list current goals",
            "my current goals",
            "show my current goals",
            "list my current goals",
            "projects",
            "list projects",
            "projects list",
            "show projects",
            "show my projects",
            "my projects",
            "what projects do i have",
            # Korean parity (Fable checkpoint plan item B2, 2026-07-10): "내
            # 목표 보여줘" fell to chat, which then answered from stale memory
            # context instead of the live goal store. Unspaced variants are
            # included because Korean chat commonly omits spaces, and these two
            # were previously (only) caught by the runtime's compact-matching
            # pre-planner suggestion set, from which they were removed in favor
            # of this direct route.
            "내 목표 보여줘",
            "목표 보여줘",
            "목표보여줘",
            "목표 목록",
            "목표목록",
            "내 목표",
            "내목표",
        }:
            return Plan(
                goal="List goals.",
                actions=[PlannedAction("list_goals", {"status": "active"}, "The user asked to list tracked goals.")],
            )

        if low in {
            "next actions",
            "next action",
            "my next actions",
            "what are my next actions",
            "what are the next actions",
            "what should i do next",
            "what should i do next today",
            "show next actions",
            "show my next actions",
        }:
            return Plan(
                goal="List next goal actions.",
                actions=[PlannedAction("next_actions", {}, "The user asked what to work on next.")],
            )

        task_command = _strip_trailing_politeness(text)
        task_low = task_command.lower().rstrip(" ?!.")

        add_task_match = re.search(
            # Real gap found live 2026-07-10: "add task buy milk and then show
            # my goals" (a compound sentence) swallowed the whole second
            # clause into the task body, silently writing a confusing task
            # titled "buy milk and then show my goals" instead of just "buy
            # milk" -- and silently dropping the "show my goals" intent
            # entirely (a data-quality bug, not just a routing one).
            r"^(?:add|create|capture|new)\s{1,10}(?:a\s{1,10}|an\s{1,10})?task(?:\s{1,10}to)?\s{1,10}(?P<body>.+?)(?:\s{1,10}due\s{1,10}(?P<due>.+?))?(?:\s{1,10}priority\s{1,10}(?P<priority>low|normal|high))?(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?$",
            task_command,
            re.IGNORECASE | re.DOTALL,
        )
        if add_task_match:
            body = add_task_match.group("body").strip()
            if not add_task_match.group("due") and not add_task_match.group("priority"):
                body = _raw_free_text_after_prefix(
                    raw_text,
                    r"^(?:add|create|capture|new)\s+(?:a\s+|an\s+)?task(?:\s+to)?\s+",
                ) or body
                body = _strip_trailing_politeness_suffix_only(body)
            return Plan(
                goal="Capture a task.",
                actions=[
                    PlannedAction(
                        "add_task",
                        {
                            "body": body,
                            "due": (add_task_match.group("due") or "").strip(),
                            "priority": (add_task_match.group("priority") or "normal").lower(),
                        },
                        "The user asked Jarvis to capture a lightweight task.",
                    )
                ],
            )

        if task_low in {
            "todo overview",
            "todos overview",
            "todo summary",
            "todos summary",
            "todo report",
            "todos report",
            "task priorities",
            "todo priorities",
            "prioritize my tasks",
            "prioritize my todo list",
            "what's due",
            "what is due",
            "anything due",
        }:
            return Plan(
                goal="Summarize tasks.",
                actions=[PlannedAction("task_overview", {}, "The user asked for a read-only task overview.")],
            )

        if task_low in {"todo board", "todos board", "todo kanban", "todos kanban", "todo dashboard", "todos dashboard"}:
            return Plan(
                goal="Show task board.",
                actions=[PlannedAction("task_board", {}, "The user asked for tasks grouped by status without changing them.")],
            )

        priority_task_match = re.search(
            r"^(?:(?:show|list|find)\s+)?(?:my\s+)?(?P<priority_first>urgent|important|priority|high priority|priority high|low priority|priority low|normal priority|priority normal|medium priority|priority medium)\s+(?:tasks?|todos?|task list|todo list)$"
            r"|^(?:(?:show|list|find)\s+)?(?:my\s+)?(?:tasks?|todos?)\s+priority\s+(?P<priority_after>high|low|normal|medium)$",
            text,
            re.IGNORECASE,
        )
        if priority_task_match:
            priority_word = (priority_task_match.group("priority_first") or priority_task_match.group("priority_after") or "").strip().lower()
            if priority_word in {"low priority", "priority low", "low"}:
                priority_query = "low"
            elif priority_word in {"normal priority", "priority normal", "normal", "medium priority", "priority medium", "medium"}:
                priority_query = "normal"
            else:
                priority_query = "high"
            return Plan(
                goal=f"Search {priority_query}-priority tasks.",
                actions=[
                    PlannedAction(
                        "search_tasks",
                        {"query": priority_query, "status": "open"},
                        "The user asked for open tasks by priority.",
                    )
                ],
            )

        add_todo_match = re.search(
            r"^(?:add\s+todo|add\s+to\s+do|new\s+todo|new\s+to\s+do|todo|to\s+do)\s+(?!(?:list|board|dashboard|kanban|overview|summary|report)$)(?P<body>.+?)(?:\s+due\s+(?P<due>.+?))?(?:\s+priority\s+(?P<priority>low|normal|high))?$",
            task_command,
            re.IGNORECASE | re.DOTALL,
        )
        if add_todo_match:
            return Plan(
                goal="Capture a task.",
                actions=[
                    PlannedAction(
                        "add_task",
                        {
                            "body": add_todo_match.group("body").strip(),
                            "due": (add_todo_match.group("due") or "").strip(),
                            "priority": (add_todo_match.group("priority") or "normal").lower(),
                        },
                        "The user asked Jarvis to capture a lightweight todo.",
                    )
                ],
            )

        korean_dated_task_match = re.fullmatch(
            r"(?P<day>오늘|내일)\s*(?:내\s+)?(?:할\s*일|작업)(?:을|를|은|는)?"
            r"(?:\s*(?:뭐야|무엇이야|알려줘|알려주세요|보여줘|보여주세요))?",
            task_low.rstrip(" ?!."),
        )
        if korean_dated_task_match:
            due_query = "tomorrow" if korean_dated_task_match.group("day") == "내일" else "today"
            return Plan(
                goal="Search due tasks.",
                actions=[
                    PlannedAction(
                        "search_tasks",
                        {"query": due_query, "status": "open"},
                        "The user asked in Korean for open tasks matching a due day.",
                    )
                ],
            )

        korean_task_list_command = task_low.rstrip(" ?!.")
        if korean_task_list_command in {
            "내 작업 보여줘",
            "내 열린 작업 보여줘",
            "작업 보여줘",
            "작업 목록",
            "내 작업",
            "할 일",
            "할 일 목록",
            "할 일 보여줘",
            "내 할 일",
            "내 할 일 보여줘",
            "내 할 일 뭐야",
            "내 할 일 알려줘",
            "내 할 일 알려주세요",
            "할 일 뭐야",
            "할 일 알려줘",
            "내 작업 뭐야",
            "내 작업 알려줘",
            "작업 뭐야",
        }:
            return Plan(
                goal="List open tasks.",
                actions=[PlannedAction("list_tasks", {"status": "open"}, "The user asked in Korean to list tasks.")],
            )

        if task_low in {
            "tasks",
            "list tasks",
            "open tasks",
            "list open tasks",
            "show open tasks",
            "latest tasks",
            "current tasks",
            "newest tasks",
            "show tasks",
            "show latest tasks",
            "show current tasks",
            "show newest tasks",
            "show my tasks",
            "list my tasks",
            "list my open tasks",
            "show my open tasks",
            "my tasks",
            "my open tasks",
            "what tasks do i have",
            "what are my tasks",
            "what are my open tasks",
            "what open tasks do i have",
            "what do i need to do",
            "what do i have to do",
            "anything i need to do",
            "what's on my list",
            "what is on my list",
            "what's on my plate",
            "what is on my plate",
            "tasks please",
            "task list",
            "task list please",
            "todo",
            "todos",
            "todos please",
            "my todo",
            "show my todo",
            "show todos",
            "show my todos",
            "what are my todos",
            "todo list",
            "todo list please",
            "to do list",
            "to-do list",
            "show todo list",
            "show to do list",
            "show my todo list",
            "my todo list",
            "my to do list",
            "what is on my todo list",
            "what's on my todo list",
            # Korean parity (Fable checkpoint plan item B2, 2026-07-10): every
            # probed Korean list phrasing fell to chat, where the model then
            # FABRICATED a capability denial ("Sorry, I'm not capable of
            # showing your tasks here") -- the same hallucination class as the
            # English round-39 "list my open tasks" bug. Exact-match aliases
            # only, planner-level (runtime.py is Codex's active lane).
            "내 작업 보여줘",
            "내 열린 작업 보여줘",
            "작업 보여줘",
            "작업 목록",
            "내 작업",
            "할 일",
            "할 일 목록",
            "할 일 보여줘",
            "내 할 일",
            "내 할 일 보여줘",
            "내 할 일 뭐야",
            "내 할 일 알려줘",
            "내 할 일 알려주세요",
            "할 일 뭐야",
            "할 일 알려줘",
            "내 작업 뭐야",
            "내 작업 알려줘",
            "작업 뭐야",
        }:
            return Plan(
                goal="List open tasks.",
                actions=[PlannedAction("list_tasks", {"status": "open"}, "The user asked to list tasks.")],
            )

        status_task_match = re.search(
            r"^(?:(?:show|list|find)\s+)?(?:my\s+)?(?P<status_word>done|completed|finished|paused|dropped)\s+(?:tasks?|todos?)$"
            r"|^(?:(?:show|list|find)\s+)?(?:my\s+)?(?P<kind>tasks?|todos?)\s+(?P<status_after>done|completed|finished|paused|dropped)$"
            r"|^(?:(?:show|list|find)\s+)?(?:my\s+)?(?P<status_list>done|completed|finished|paused|dropped)\s+(?:todo\s+list|to\s+do\s+list|to-do\s+list)$",
            text,
            re.IGNORECASE,
        )
        if status_task_match:
            status_word = (
                status_task_match.group("status_word")
                or status_task_match.group("status_after")
                or status_task_match.group("status_list")
                or ""
            ).strip().lower()
            status = "done" if status_word in {"completed", "finished"} else status_word
            return Plan(
                goal=f"List {status} tasks.",
                actions=[
                    PlannedAction(
                        "list_tasks",
                        {"status": status},
                        "The user asked to list tasks by status.",
                    )
                ],
            )

        if re.search(
            r"^(?:(?:show|list|find)\s+)?(?:my\s+)?(?:overdue|past due)\s+(?:tasks?|todos?)$"
            r"|^(?:(?:show|list|find)\s+)?(?:my\s+)?(?:tasks?|todos?)\s+(?:overdue|past due)$"
            r"|^(?:what\s+(?:tasks?|todos?)\s+are\s+(?:overdue|past due)|anything\s+(?:overdue|past due))$",
            low,
        ):
            return Plan(
                goal="List overdue tasks.",
                actions=[
                    PlannedAction(
                        "overdue_tasks",
                        {},
                        "The user asked for open tasks with parseable due labels before today.",
                    )
                ],
            )

        due_task_command = _normalize_due_task_command(text)
        due_task_full_command = _normalize_due_task_command(user_input)
        due_calendar_compound_match = re.fullmatch(
            rf"(?:(?:show|list|find)\s+)?(?:my\s+)?tasks?\s+due\s+"
            rf"(?P<due>{TASK_DUE_QUERY_PATTERN})(?:\s*[,;]\s*|\s+(?:and\s+then|then)\s+)"
            r"(?:show|list|open)\s+(?:my\s+)?calendar",
            due_task_full_command,
            re.IGNORECASE,
        )
        if due_calendar_compound_match:
            due_query = due_calendar_compound_match.group("due").strip().casefold()
            return Plan(
                goal="Search due tasks and show the matching calendar window.",
                actions=[
                    PlannedAction(
                        "search_tasks",
                        {"query": due_query, "status": "open"},
                        "The user first asked for open tasks matching a due label.",
                    ),
                    PlannedAction(
                        "list_events",
                        {"range": due_query},
                        "The user also asked to see the calendar for that window.",
                    ),
                ],
            )
        due_tasks_match = re.fullmatch(
            rf"(?:(?:show|list|find)\s+)?(?:my\s+)?tasks?\s+due\s+(?P<due>{TASK_DUE_QUERY_PATTERN})"
            rf"|(?:what|which)\s+tasks?\s+(?:are\s+)?due\s+(?P<due_question>{TASK_DUE_QUERY_PATTERN})"
            rf"|(?:(?:what(?:'s| is)|anything|show|list|find)\s+)?due\s+(?P<due_shorthand>{TASK_DUE_QUERY_PATTERN})"
            rf"|(?:anything|what(?:'s| is)\s+anything)\s+due\s+(?P<due_anything>{TASK_DUE_QUERY_PATTERN})"
            rf"|do\s+i\s+have\s+(?:anything|any\s+(?:tasks?|todos?))\s+due\s+(?P<due_have>{TASK_DUE_QUERY_PATTERN})",
            due_task_command,
            re.IGNORECASE,
        )
        if due_tasks_match:
            due_query = (
                due_tasks_match.group("due")
                or due_tasks_match.group("due_question")
                or due_tasks_match.group("due_shorthand")
                or due_tasks_match.group("due_anything")
                or due_tasks_match.group("due_have")
                or ""
            ).strip().casefold()
            return Plan(
                goal="Search due tasks.",
                actions=[
                    PlannedAction(
                        "search_tasks",
                        {"query": due_query, "status": "open"},
                        "The user asked for open tasks matching a due label.",
                    )
                ],
            )

        dated_task_match = re.search(
            r"^(?P<date_first>today|tomorrow|this week|next week)(?:'s|’s)?\s+(?:tasks?|todos?)$"
            r"|^(?:show|list|find)\s+(?P<date_after_verb>today|tomorrow|this week|next week)(?:'s|’s)?\s+(?:tasks?|todos?)$"
            r"|^(?:(?:show|list|find)\s+)?(?:my\s+)?(?:tasks?|todos?)\s+(?P<date_last>today|tomorrow|this week|next week)$"
            r"|^what\s+do\s+i\s+need\s+to\s+do\s+(?P<need_date>today|tomorrow|this week|next week)$"
            r"|^what\s+do\s+i\s+have\s+to\s+do\s+(?P<have_date>today|tomorrow|this week|next week)$"
            r"|^what(?:'s| is)\s+on\s+my\s+(?:plate|list)\s+(?P<plate_date>today|tomorrow|this week|next week)$",
            text,
            re.IGNORECASE,
        )
        if dated_task_match:
            due_query = (
                dated_task_match.group("date_first")
                or dated_task_match.group("date_after_verb")
                or dated_task_match.group("date_last")
                or dated_task_match.group("need_date")
                or dated_task_match.group("have_date")
                or dated_task_match.group("plate_date")
                or ""
            ).strip()
            return Plan(
                goal="Search due tasks.",
                actions=[
                    PlannedAction(
                        "search_tasks",
                        {"query": due_query, "status": "open"},
                        "The user asked for open tasks matching a due label.",
                    )
                ],
            )

        if re.search(r"^(?:(?:show|list|find|what(?: are|'?s| is)?)\s+)?(?:my\s+)?(?:high priority|priority high)\s+(?:tasks?|todos?)$", low):
            return Plan(
                goal="Search high-priority tasks.",
                actions=[
                    PlannedAction(
                        "search_tasks",
                        {"query": "high", "status": "open"},
                        "The user asked for open high-priority tasks.",
                    )
                ],
            )

        if low in {"task overview", "tasks overview", "task summary", "tasks summary", "task report", "tasks report", "show task overview", "show tasks overview", "show task summary", "show tasks summary", "show task report", "show tasks report"}:
            return Plan(
                goal="Summarize tasks.",
                actions=[PlannedAction("task_overview", {}, "The user asked for a read-only task overview.")],
            )

        if low in {"task board", "tasks board", "task kanban", "tasks kanban", "task dashboard", "tasks dashboard", "show task board", "show tasks board", "show task kanban", "show tasks kanban", "show task dashboard", "show tasks dashboard"}:
            return Plan(
                goal="Show task board.",
                actions=[PlannedAction("task_board", {}, "The user asked for tasks grouped by status without changing them.")],
            )

        if low in {
            "next task",
            "next todo",
            "next thing to do",
            "show next task",
            "show next todo",
            "show my next task",
            "show my next todo",
            "what task next",
            "what todo next",
            "what is next task",
            "what's next task",
            "what is next todo",
            "what's next todo",
            "what is my next task",
            "what's my next task",
            "what is my next todo",
            "what's my next todo",
            "what task should i do",
            "what todo should i do",
            "what should i do first",
            "what should i tackle first",
            "what should i work on first",
            "what task should i do next",
            "what todo should i do next",
            "which task next",
            "which todo next",
            "which task should i do",
            "which todo should i do",
            "which task should i do first",
            "which todo should i do first",
            "pick next task",
            "pick next todo",
            "pick my next task",
            "pick my next todo",
        }:
            return Plan(
                goal="Pick next task.",
                actions=[PlannedAction("next_task", {}, "The user asked Jarvis to pick the next open task without changing anything.")],
            )

        preview_tasks_match = re.search(
            r"^(?:preview|show|scan)\s+tasks?\s+from\s+(?:jarvis\s+)?note\s+(?P<path>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if preview_tasks_match:
            return Plan(
                goal="Preview tasks in a Jarvis note.",
                actions=[
                    PlannedAction(
                        "preview_tasks_from_note",
                        {"path": preview_tasks_match.group("path").strip()},
                        "The user asked to preview markdown checkboxes before importing them.",
                    )
                ],
            )

        import_tasks_match = re.search(
            r"^(?:import|capture|add)\s+tasks?\s+from\s+(?:jarvis\s+)?note\s+(?P<path>.+?)(?:\s+priority\s+(?P<priority>low|normal|high))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if import_tasks_match:
            return Plan(
                goal="Import tasks from a Jarvis note.",
                actions=[
                    PlannedAction(
                        "import_tasks_from_note",
                        {
                            "path": import_tasks_match.group("path").strip(),
                            "priority": (import_tasks_match.group("priority") or "normal").lower(),
                        },
                        "The user asked Jarvis to convert open markdown checkboxes into tracked tasks.",
                    )
                ],
            )

        if low in {"all tasks", "list all tasks", "show all tasks"}:
            return Plan(
                goal="List all tasks.",
                actions=[PlannedAction("list_tasks", {"status": "all"}, "The user asked to list all tasks.")],
            )

        search_tasks_match = re.search(
            # Real gap found live 2026-07-10: "search tasks for groceries and
            # then show my calendar" (a compound sentence) swallowed the
            # whole second clause into the search query instead of stopping
            # at "groceries" -- and silently dropped the "show my calendar"
            # intent. Applied to both alternatives below.
            r"^(?:search|find)\s{1,10}(?:my\s{1,10})?tasks?\s{1,10}(?:(?:for|about)\s{1,10})?(?P<query>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?(?:\s{1,10}status\s{1,10}(?P<status>open|done|paused|dropped|all))?$"
            # Real gap found live 2026-07-10, same privacy-relevant misroute
            # class as the round-21/26 notes-search word-order fixes: "search
            # for the/a task about X" / "find the/a task about X" (article
            # BEFORE the noun) fell through to the generic public web-search
            # fallback instead of searching Jarvis's own tracked tasks.
            r"|^(?:search for|find)\s{1,10}(?:the\s{1,10}|a\s{1,10}|my\s{1,10})?tasks?\s{1,10}(?:for|about)\s{1,10}(?P<query2>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?(?:\s{1,10}status\s{1,10}(?P<status2>open|done|paused|dropped|all))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if search_tasks_match:
            return Plan(
                goal="Search tasks.",
                actions=[
                    PlannedAction(
                        "search_tasks",
                        {
                            "query": _strip_trailing_politeness(
                                search_tasks_match.group("query") or search_tasks_match.group("query2")
                            ).rstrip(" ?!."),
                            "status": (search_tasks_match.group("status") or search_tasks_match.group("status2") or "all").lower(),
                        },
                        "The user asked to search tracked tasks without changing them.",
                    )
                ],
            )

        inspect_task_match = re.search(
            r"^(?:show|inspect|view)?\s*task\s+#?(?P<task_id>\d+)(?:\s+(?:status|detail|details))?$",
            text,
            re.IGNORECASE,
        )
        if inspect_task_match:
            return Plan(
                goal="Inspect task.",
                actions=[
                    PlannedAction(
                        "inspect_task",
                        {"task_id": int(inspect_task_match.group("task_id"))},
                        "The user asked to inspect a task without changing it.",
                    )
                ],
            )

        task_completion_packet_match = re.search(
            r"^(?:task completion packet|completion packet for task|can complete task|can finish task)\s+#?(?P<task_id>\d+)(?:\s*:?\s*(?P<evidence>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if task_completion_packet_match:
            return Plan(
                goal="Check task completion evidence.",
                actions=[
                    PlannedAction(
                        "task_completion_packet",
                        {
                            "task_id": int(task_completion_packet_match.group("task_id")),
                            "evidence": (task_completion_packet_match.group("evidence") or "").strip(),
                        },
                        "The user asked Jarvis to inspect completion evidence before closing a tracked task.",
                    )
                ],
            )

        exact_task_command = (
            user_input.lstrip() if len(and_then_split) == 1 else task_command
        )
        complete_task_with_evidence_match = re.search(
            r"^(?:complete|finish|done)\s+task\s+#?(?P<task_id>\d+)\s+with\s+"
            r"(?P<payload>.+)$",
            exact_task_command,
            re.IGNORECASE | re.DOTALL,
        )
        if complete_task_with_evidence_match:
            payload = complete_task_with_evidence_match.group("payload")
            verification_run_id = ""
            run_match = re.match(
                r"^(?:verification\s+run|run)\s+#?(?P<run_id>\d+)(?P<rest>.*)$",
                payload,
                re.IGNORECASE | re.DOTALL,
            )
            if run_match:
                verification_run_id = run_match.group("run_id")
                rest = run_match.group("rest")
                payload = rest[1:] if rest.startswith(":") else rest.lstrip()
            labelled = re.match(
                r"^(?:evidence|proof|receipt)[ \t]*:(?P<evidence>.*)$",
                payload,
                re.IGNORECASE | re.DOTALL,
            )
            if labelled:
                evidence = labelled.group("evidence")
            else:
                unlabelled = re.match(
                    r"^(?:evidence|proof|receipt)\s+(?P<evidence>.+)$",
                    payload,
                    re.IGNORECASE | re.DOTALL,
                )
                evidence = unlabelled.group("evidence") if unlabelled else payload
            return Plan(
                goal="Complete task with evidence.",
                actions=[
                    PlannedAction(
                        "complete_task_with_evidence",
                        {
                            "task_id": int(complete_task_with_evidence_match.group("task_id")),
                            "verification_run_id": verification_run_id,
                            "evidence": evidence,
                        },
                        "The user asked Jarvis to close a task with explicit evidence preserved in the audit log.",
                    )
                ],
            )

        complete_task_match = re.search(
            r"^(?:complete|finish|done)\s+task\s+#?(?P<task_id>\d+)$"
            r"|^mark\s+task\s+#?(?P<marked_task_id>\d+)\s+(?:done|complete|completed|finished)$",
            task_command,
            re.IGNORECASE,
        )
        if complete_task_match:
            return Plan(
                goal="Complete task.",
                actions=[
                    PlannedAction(
                        "complete_task",
                        {"task_id": int(complete_task_match.group("task_id") or complete_task_match.group("marked_task_id"))},
                        "The user asked to complete a task.",
                    )
                ],
            )

        update_task_match = re.search(
            r"^(?P<status>pause|paused|reopen|resume|open|drop|dropped)\s+task\s+#?(?P<task_id>\d+)$",
            text,
            re.IGNORECASE,
        )
        if update_task_match:
            raw_status = update_task_match.group("status").lower()
            status = {
                "pause": "paused",
                "paused": "paused",
                "reopen": "open",
                "resume": "open",
                "open": "open",
                "drop": "dropped",
                "dropped": "dropped",
            }[raw_status]
            return Plan(
                goal="Update task status.",
                actions=[
                    PlannedAction(
                        "update_task_status",
                        {"task_id": int(update_task_match.group("task_id")), "status": status},
                        "The user asked to update a task status.",
                    )
                ],
            )

        task_priority_match = re.search(
            r"^set\s+task\s+#?(?P<task_id>\d+)\s+priority\s+(?P<priority>low|normal|high)$",
            text,
            re.IGNORECASE,
        )
        if task_priority_match:
            return Plan(
                goal="Update task priority.",
                actions=[
                    PlannedAction(
                        "update_task_details",
                        {"task_id": int(task_priority_match.group("task_id")), "priority": task_priority_match.group("priority").lower()},
                        "The user asked to update a task priority.",
                    )
                ],
            )

        task_due_match = re.search(
            r"^set\s+task\s+#?(?P<task_id>\d+)\s+due\s+(?P<due>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if task_due_match:
            return Plan(
                goal="Update task due label.",
                actions=[
                    PlannedAction(
                        "update_task_details",
                        {"task_id": int(task_due_match.group("task_id")), "due": task_due_match.group("due").strip()},
                        "The user asked to update a task due label.",
                    )
                ],
            )

        task_body_match = re.search(
            r"^(?:rename|edit)\s+task\s+#?(?P<task_id>\d+)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if task_body_match:
            return Plan(
                goal="Update task body.",
                actions=[
                    PlannedAction(
                        "update_task_details",
                        {"task_id": int(task_body_match.group("task_id")), "body": task_body_match.group("body").strip()},
                        "The user asked to update a task body.",
                    )
                ],
            )

        set_task_status_match = re.search(
            r"^set\s+task\s+#?(?P<task_id>\d+)\s+(?:to\s+)?(?P<status>open|done|paused|dropped)$",
            text,
            re.IGNORECASE,
        )
        if set_task_status_match:
            return Plan(
                goal="Set task status.",
                actions=[
                    PlannedAction(
                        "update_task_status",
                        {"task_id": int(set_task_status_match.group("task_id")), "status": set_task_status_match.group("status").lower()},
                        "The user asked to set a task status.",
                    )
                ],
            )

        if low in {"export tasks", "export open tasks", "save tasks to obsidian"}:
            return Plan(
                goal="Export open tasks.",
                actions=[PlannedAction("export_tasks", {}, "The user asked to export tasks to Obsidian.")],
            )

        goal_status_match = re.search(r"^(?:goal|project)\s+#?(?P<goal_id>\d+)(?:\s+status)?$", text, re.IGNORECASE)
        if goal_status_match:
            return Plan(
                goal="Show goal status.",
                actions=[
                    PlannedAction(
                        "goal_status",
                        {"goal_id": int(goal_status_match.group("goal_id"))},
                        "The user asked to inspect a goal.",
                    )
                ],
            )

        add_goal_step_match = re.search(
            r"^(?:add\s+step|add\s+task)\s+(?:to\s+)?(?:goal|project)\s+#?(?P<goal_id>\d+)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if add_goal_step_match:
            return Plan(
                goal="Add goal step.",
                actions=[
                    PlannedAction(
                        "add_goal_step",
                        {
                            "goal_id": int(add_goal_step_match.group("goal_id")),
                            "body": add_goal_step_match.group("body").strip(),
                        },
                        "The user asked to add a step to a goal.",
                    )
                ],
            )

        complete_step_match = re.search(
            r"^(?:complete|finish|done)\s+(?:goal\s+)?step\s+#?(?P<step_id>\d+)$",
            text,
            re.IGNORECASE,
        )
        if complete_step_match:
            return Plan(
                goal="Complete goal step.",
                actions=[
                    PlannedAction(
                        "complete_goal_step",
                        {"step_id": int(complete_step_match.group("step_id"))},
                        "The user asked to mark a goal step done.",
                    )
                ],
            )

        set_goal_status_match = re.search(
            r"^(?:set\s+)?(?:goal|project)\s+#?(?P<goal_id>\d+)\s+(?:to\s+)?(?P<status>active|paused|done|dropped)$",
            text,
            re.IGNORECASE,
        )
        if set_goal_status_match:
            return Plan(
                goal="Set goal status.",
                actions=[
                    PlannedAction(
                        "set_goal_status",
                        {
                            "goal_id": int(set_goal_status_match.group("goal_id")),
                            "status": set_goal_status_match.group("status").lower(),
                        },
                        "The user asked to change a goal status.",
                    )
                ],
            )

        export_goal_match = re.search(
            r"^(?:export|save)\s+(?:goal|project)\s+#?(?P<goal_id>\d+)(?:\s+to\s+obsidian)?$",
            text,
            re.IGNORECASE,
        )
        if export_goal_match:
            return Plan(
                goal="Export goal.",
                actions=[
                    PlannedAction(
                        "export_goal",
                        {"goal_id": int(export_goal_match.group("goal_id"))},
                        "The user asked to export a goal to Obsidian.",
                    )
                ],
            )

        if "weak memor" in low or "bad memor" in low or "memory cleanup" in low:
            return Plan(
                goal="Find weak memories.",
                actions=[PlannedAction("list_weak_memories", {}, "The user asked to inspect weak memories.")],
            )

        if low_command in {"learning review", "learning report", "jarvis learning review", "my learning review", "review learning", "self improvement review", "show learning review", "show my learning review"}:
            return Plan(
                goal="Review safe learning opportunities.",
                actions=[PlannedAction("learning_review", {}, "The user asked for a safe learning review.")],
            )

        knowledge_promotion_packet_match = re.fullmatch(
            r"(?:knowledge promotion packet|review knowledge memory)\s+#?(?P<memory_id>\d+)",
            text,
            re.IGNORECASE,
        )
        if knowledge_promotion_packet_match:
            memory_id = _bounded_positive_sqlite_id_digits(
                knowledge_promotion_packet_match.group("memory_id")
            )
            if memory_id is None:
                return Plan(
                    goal="Reject an invalid knowledge promotion packet request.",
                    actions=[],
                    needs_model=False,
                    notes="Knowledge promotion packet review requires a positive bounded memory id.",
                )
            return Plan(
                goal="Review one knowledge promotion candidate.",
                actions=[
                    PlannedAction(
                        "knowledge_promotion_packet",
                        {"memory_id": memory_id},
                        "The user asked to inspect one exact generic-memory candidate before any ownership transfer.",
                    )
                ],
            )

        if low_command in {"save learning review", "write learning review", "export learning review", "save learning report"}:
            return Plan(
                goal="Save safe learning review.",
                actions=[PlannedAction("save_learning_review", {}, "The user asked to save the safe learning review.")],
            )

        if low_command in {"queue learning tasks", "create learning tasks", "add learning tasks", "task learning review", "turn learning review into tasks"}:
            return Plan(
                goal="Queue safe learning tasks.",
                actions=[PlannedAction("queue_learning_tasks", {}, "The user asked to queue local tasks from the safe learning review.")],
            )

        if low in {"memory stats", "memory statistics", "memory summary stats", "show memory stats", "show my memory stats"}:
            return Plan(
                goal="Show memory statistics.",
                actions=[PlannedAction("memory_stats", {}, "The user asked for memory statistics.")],
            )

        if "duplicate memor" in low or "duplicate memories" in low:
            return Plan(
                goal="Find duplicate memories.",
                actions=[PlannedAction("list_duplicate_memories", {}, "The user asked to inspect duplicate memories.")],
            )

        merge_memory_match = re.search(r"^merge\s+memory\s+#?(?P<delete_id>\d+)\s+(?:into|to)\s+#?(?P<keep_id>\d+)$", text, re.IGNORECASE)
        if merge_memory_match:
            return Plan(
                goal="Merge memories.",
                actions=[
                    PlannedAction(
                        "merge_memories",
                        {
                            "keep_id": int(merge_memory_match.group("keep_id")),
                            "delete_id": int(merge_memory_match.group("delete_id")),
                        },
                        "The user asked to merge duplicate memories.",
                    )
                ],
            )

        delete_memory_match = re.search(r"^(?:delete|remove|forget)\s+memory\s+#?(?P<memory_id>\d+)$", text, re.IGNORECASE)
        if delete_memory_match:
            return Plan(
                goal="Delete a memory.",
                actions=[
                    PlannedAction(
                        "delete_memory",
                        {"memory_id": int(delete_memory_match.group("memory_id"))},
                        "The user asked to delete a specific memory.",
                    )
                ],
            )

        get_memory_match = re.search(r"^(?:get|show|inspect|review)\s+memory\s+#?(?P<memory_id>\d+)$", text, re.IGNORECASE)
        if get_memory_match:
            return Plan(
                goal="Inspect a memory.",
                actions=[
                    PlannedAction(
                        "get_memory",
                        {"memory_id": int(get_memory_match.group("memory_id"))},
                        "The user asked to inspect one memory.",
                    )
                ],
            )

        if "memory tree" in low and ("summarize" in low or "update" in low or "snapshot" in low):
            return Plan(
                goal="Update Memory Tree summary.",
                actions=[PlannedAction("memory_tree_summary", {}, "The user asked to update the Memory Tree.")],
            )

        if low in {"ingest inbox", "ingest my inbox", "ingest obsidian inbox", "import inbox", "process inbox", "index inbox"}:
            return Plan(
                goal="Ingest Obsidian Inbox.",
                actions=[PlannedAction("ingest_obsidian_inbox", {}, "The user asked to ingest the Obsidian Inbox.")],
            )

        if low in {"clear inbox", "clear my inbox", "clear obsidian inbox", "reset inbox", "empty inbox"}:
            return Plan(
                goal="Clear Obsidian Inbox.",
                actions=[PlannedAction("clear_obsidian_inbox", {}, "The user asked to clear the Obsidian Inbox.")],
            )

        if low in {
            "recent file digest",
            "file digest",
            "recent files digest",
            "watch files",
            "show recent file digest",
            "what files changed recently",
            "what files have changed recently",
        }:
            return Plan(
                goal="Write recent file digest.",
                actions=[PlannedAction("recent_file_digest", {}, "The user asked for a digest of watched files.")],
            )

        morning_brief_schedule = re.search(
            r"\bschedule\s+(?:a\s+)?(?:morning brief|daily briefing)(?:\s+(?:at|for)\s+(?P<time>\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?",
            low,
        )
        if morning_brief_schedule:
            time_text = (morning_brief_schedule.group("time") or "").strip()
            args = {"time": time_text} if time_text else {}
            return Plan(
                goal="Schedule Telegram morning brief.",
                actions=[
                    PlannedAction(
                        "schedule_morning_brief",
                        args,
                        "The user asked to schedule a Telegram morning brief.",
                    )
                ],
            )

        if "schedule daily brief" in low:
            return Plan(
                goal="Schedule daily brief.",
                actions=[PlannedAction("schedule_daily_brief", {}, "The user asked to schedule a daily brief.")],
            )

        if "schedule weekly review" in low:
            return Plan(
                goal="Schedule weekly review.",
                actions=[PlannedAction("schedule_weekly_review", {}, "The user asked to schedule a weekly review.")],
            )

        if low in {"schedule assistant basics", "schedule jarvis basics", "schedule default jobs", "schedule default automations"}:
            return Plan(
                goal="Schedule default assistant jobs.",
                actions=[PlannedAction("schedule_assistant_basics", {}, "The user asked to schedule default assistant automations.")],
            )

        if low in {
            "scheduler context refresh",
            "scheduler context refresh packet",
            "context refresh packet",
            "background continuity packet",
            "scheduled context refresh",
            "mission control refresh packet",
        }:
            return Plan(
                goal="Check scheduled context refresh readiness.",
                actions=[
                    PlannedAction(
                        "scheduler_context_refresh_packet",
                        {},
                        "The user asked for a read-only scheduler continuity readiness packet.",
                    )
                ],
            )

        if "schedule inbox ingest" in low or "schedule ingest inbox" in low:
            return Plan(
                goal="Schedule Inbox ingest.",
                actions=[PlannedAction("schedule_inbox_ingest", {}, "The user asked to schedule Inbox ingestion.")],
            )

        if "schedule file digest" in low or "schedule recent file" in low:
            return Plan(
                goal="Schedule file digest.",
                actions=[PlannedAction("schedule_file_digest", {}, "The user asked to schedule watched-file digests.")],
            )

        scheduler_command = _strip_trailing_politeness(text).lower()
        scheduler_job_aliases = {
            "daily brief": "Daily Brief",
            "morning brief": "Morning Brief",
            "inbox ingest": "Inbox Ingest",
            "recent file digest": "Recent File Digest",
            "file digest": "Recent File Digest",
            "goal nudge": "Goal Nudge",
            "weekly review": "Weekly Review",
            "state snapshot": "State Snapshot",
            "conversation compaction": "Conversation Compaction",
            "compaction": "Conversation Compaction",
        }

        if re.search(r"^(?:run|start|trigger)\s+(?:all\s+)?(?:due\s+)?(?:scheduled\s+)?jobs(?:\s+now)?$", scheduler_command):
            return Plan(
                goal="Run due scheduled jobs.",
                actions=[PlannedAction("run_due_jobs", {}, "The user asked to run due scheduled jobs.")],
            )

        def _normalize_job_alias_name(name: str) -> str:
            # Real gap found live 2026-07-09: "pause the morning brief job" /
            # "run the morning brief job now" fell through to the generic
            # \bmorning brief\b substring catch-all (which ignores the
            # pause/run verb entirely) because the alias lookup only stripped
            # a leading "job " prefix, not a leading "the " article or a
            # trailing " job" suffix -- both natural in "the <job> job"
            # phrasing.
            normalized = name.strip()
            if normalized.startswith("the "):
                normalized = normalized[4:].strip()
            if normalized.startswith("job "):
                normalized = normalized[4:].strip()
            if normalized.endswith(" job"):
                normalized = normalized[: -len(" job")].strip()
            return normalized

        run_named_job_match = re.search(r"^(?:run|start|trigger)\s+(?P<name>.+?)\s+now$", scheduler_command)
        if run_named_job_match:
            job_name = _normalize_job_alias_name(run_named_job_match.group("name"))
            if job_name in scheduler_job_aliases:
                return Plan(
                    goal="Run scheduled job now.",
                    actions=[
                        PlannedAction(
                            "run_job_now",
                            {"name": scheduler_job_aliases[job_name]},
                            "The user asked to run a scheduled job immediately.",
                        )
                    ],
                )

        manage_named_job_match = re.search(r"^(?P<verb>pause|resume|delete)\s+(?P<name>.+)$", scheduler_command)
        if manage_named_job_match:
            verb = manage_named_job_match.group("verb").lower()
            job_name = _normalize_job_alias_name(manage_named_job_match.group("name"))
            if job_name in scheduler_job_aliases:
                tool = {"pause": "pause_job", "resume": "resume_job", "delete": "delete_job"}[verb]
                return Plan(
                    goal=f"{verb.title()} scheduled job.",
                    actions=[
                        PlannedAction(
                            tool,
                            {"name": scheduler_job_aliases[job_name]},
                            "The user asked to manage a scheduled job.",
                        )
                    ],
                )

        if (
            re.search(r"^(?:(?:show|list|open|view)\s+(?:my\s+)?(?:latest\s+|current\s+|newest\s+)?|(?:latest|current|newest)\s+)?(?:scheduled\s+)?(?:jobs|automations)(?:\s+please)?$", low)
            or re.search(r"^what\s+jobs\s+are\s+scheduled(?:\s+please)?$", low)
        ):
            return Plan(
                goal="List scheduled jobs.",
                actions=[PlannedAction("list_scheduled_jobs", {}, "The user asked to list scheduled jobs.")],
            )

        if re.search(
            r"^(?:(?:run|send)\s+(?:the\s+)?(?:morning brief|daily briefing)\s+now|(?:send|push)\s+(?:me\s+)?(?:my\s+)?(?:(?:today'?s|today|daily|morning)\s+)?brief(?:ing)?(?:\s+(?:to|on)\s+(?:my\s+)?(?:phone|telegram))?(?:\s+now)?)$",
            text,
            re.IGNORECASE,
        ):
            return Plan(
                goal="Run Telegram morning brief now.",
                actions=[
                    PlannedAction(
                        "run_job_now",
                        {"name": "Morning Brief"},
                        "The user asked to send the scheduled Morning Brief immediately.",
                    )
                ],
            )

        run_job_now_match = re.search(r"^run\s+job\s+(?P<name>.+)\s+now$", text, re.IGNORECASE)
        if run_job_now_match:
            job_name = _strip_trailing_politeness(run_job_now_match.group("name")).strip()
            job_name = scheduler_job_aliases.get(job_name.lower(), job_name)
            return Plan(
                goal="Run scheduled job now.",
                actions=[
                    PlannedAction(
                        "run_job_now",
                        {"name": job_name},
                        "The user asked to run a job immediately.",
                    )
                ],
            )

        manage_job_match = re.search(r"^(?P<verb>pause|resume|delete)\s+job\s+(?P<name>.+)$", text, re.IGNORECASE)
        if manage_job_match:
            verb = manage_job_match.group("verb").lower()
            job_name = _strip_trailing_politeness(manage_job_match.group("name")).strip()
            job_name = scheduler_job_aliases.get(job_name.lower(), job_name)
            tool = {"pause": "pause_job", "resume": "resume_job", "delete": "delete_job"}[verb]
            return Plan(
                goal=f"{verb.title()} scheduled job.",
                actions=[
                    PlannedAction(
                        tool,
                        {"name": job_name},
                        "The user asked to manage a scheduled job.",
                    )
                ],
            )

        # "morning brief" / "daily briefing" -> the live brief (weather + today's
        # calendar + headlines). Checked before "daily brief" because "daily
        # briefing" contains that substring. "daily brief" / "proactive brief"
        # keep the existing context brief (tasks/approvals/decisions).
        live_brief_command = _strip_trailing_politeness(text).lower()
        if (
            re.search(r"\bmorning brief(?:ing)?\b|\bdaily briefing\b", live_brief_command)
            or live_brief_command in {
                "brief",
                "brief me",
                "today brief",
                "today's brief",
                "what's my brief today",
                "what is my brief today",
                "phone brief",
                "telegram brief",
            }
        ):
            return Plan(
                goal="Give the daily brief.",
                actions=[PlannedAction("daily_briefing", {}, "The user asked for a brief.")],
            )
        if "daily brief" in low or "proactive brief" in low:
            return Plan(
                goal="Generate proactive daily brief.",
                actions=[PlannedAction("daily_brief", {}, "The user asked for a proactive brief.")],
            )

        proactive_command = _strip_trailing_politeness(text).lower()
        proactive_command = re.sub(r"^(?:latest|current)\s+", "", proactive_command).strip()
        if proactive_command in {
            "daily plan",
            "plan today",
            "today plan",
            "today's plan",
            "plan my day",
            "what is my plan today",
            "what's my plan today",
        }:
            return Plan(
                goal="Generate tactical daily plan.",
                actions=[
                    PlannedAction(
                        "daily_plan",
                        {"target_date": datetime.now().strftime("%Y-%m-%d")},
                        "The user asked for a tactical daily plan.",
                    )
                ],
            )

        if proactive_command in {
            "morning startup",
            "startup brief",
            "start my day",
            "start day",
            "morning plan",
            "today startup",
        }:
            return Plan(
                goal="Show morning startup.",
                actions=[PlannedAction("morning_startup", {}, "The user asked for a safe startup brief for the day.")],
            )

        if low in {"save morning startup", "write morning startup", "save startup brief", "write startup brief"}:
            return Plan(
                goal="Save morning startup.",
                actions=[PlannedAction("save_morning_startup", {}, "The user asked to save the morning startup brief.")],
            )

        if "goal nudge" in low or "stale goal" in low:
            return Plan(
                goal="Generate goal nudge.",
                actions=[PlannedAction("goal_nudge", {}, "The user asked for stale-goal nudges.")],
            )

        weekly_review_command = _strip_trailing_politeness(text).lower()
        if weekly_review_command in {
            "weekly review context",
            "weekly review model context",
            "model weekly review context",
            "preview weekly review context",
        }:
            return Plan(
                goal="Preview weekly review model context.",
                actions=[PlannedAction("weekly_review_context", {}, "The user asked to inspect the read-only weekly review model-assist packet.")],
            )

        if weekly_review_command in {
            "weekly review prompt",
            "weekly review prompt preview",
            "preview weekly review prompt",
            "weekly model prompt",
            "weekly review model prompt",
            "weekly review model prompt preview",
        }:
            return Plan(
                goal="Preview weekly review model prompt.",
                actions=[
                    PlannedAction(
                        "weekly_review_prompt_preview",
                        {},
                        "The user asked to preview a bounded weekly-review prompt packet without calling a model.",
                    )
                ],
            )

        if "weekly review" in low:
            return Plan(
                goal="Generate weekly review.",
                actions=[PlannedAction("weekly_review", {}, "The user asked for a weekly review.")],
            )

        computer_readiness_match = re.search(
            r"^(?:computer readiness|computer control readiness|computer readiness check|computer preflight|desktop control readiness|desktop preflight)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if computer_readiness_match:
            return Plan(
                goal="Check computer-control readiness.",
                actions=[
                    PlannedAction(
                        "computer_control_readiness",
                        {"objective": (computer_readiness_match.group("objective") or "").strip()},
                        "The user asked for a read-only desktop-control readiness preflight.",
                    )
                ],
            )

        computer_plan_match = re.search(
            r"^(?:computer task plan|desktop task plan|screen task plan|observe act verify plan|oav plan|computer control plan)\s*:?\s*(?P<objective>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if computer_plan_match:
            return Plan(
                goal="Plan a safe computer-control task.",
                actions=[
                    PlannedAction(
                        "computer_task_plan",
                        {"objective": (computer_plan_match.group("objective") or "").strip()},
                        "The user asked for a safe observe-act-verify plan before desktop control.",
                    )
                ],
            )

        computer_action_packet_match = re.search(
            r"^(?:computer action packet|desktop action packet|screen action packet|oav action packet|observe act verify action packet)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if computer_action_packet_match:
            return Plan(
                goal="Prepare one computer-control action packet.",
                actions=[
                    PlannedAction(
                        "computer_action_packet",
                        {"spec": (computer_action_packet_match.group("spec") or "").strip()},
                        "The user asked to prepare one read-only observe-act-verify action packet before approval.",
                    )
                ],
            )

        approved_screen_receipt_match = re.search(
            r"^(?:approved screen observation receipt|screen observation receipt|screenshot observation receipt|approved screenshot receipt|oav observation receipt)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if approved_screen_receipt_match:
            spec = (approved_screen_receipt_match.group("spec") or "").strip()
            args: dict[str, object] = {"spec": spec}
            for arg_key, aliases in {
                "expectation": ("expectation", "expected", "expect"),
                "observation": ("observation", "observed", "evidence", "screen"),
                "source": ("observation source", "source"),
                "action": ("action", "step"),
                "observation_id": ("observation id", "receipt id", "screen receipt"),
                "screenshot_path": ("screenshot path", "path"),
                "screenshot_sha256": ("screenshot sha256", "sha256", "hash"),
                "captured_at": ("captured at", "timestamp", "time"),
                "redaction_review": ("redaction review", "redaction", "privacy"),
            }.items():
                for alias in aliases:
                    match = re.search(rf"(?:^|;)\s*{re.escape(alias)}\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
                    if match:
                        args[arg_key] = match.group("value").strip()
                        break
            approval_match = re.search(r"\bapproval(?:\s+id|\s+#)?\s+(?P<value>\d+)\b", spec, re.IGNORECASE)
            if approval_match:
                args["approval_id"] = approval_match.group("value")
            return Plan(
                goal="Validate an approved screen-observation receipt before OAV proof review.",
                actions=[
                    PlannedAction(
                        "approved_screen_observation_receipt",
                        args,
                        "The user asked to validate supplied screenshot-observation receipt metadata without observing or controlling the computer.",
                    )
                ],
            )

        screen_freshness_match = re.search(
            r"^(?:screen observation freshness|screen freshness packet|screen observation freshness packet|approved screen freshness|oav freshness packet)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if screen_freshness_match:
            spec = (screen_freshness_match.group("spec") or "").strip()
            args: dict[str, object] = {"spec": spec}
            for arg_key, aliases in {
                "expectation": ("expectation", "expected", "expect"),
                "observation": ("observation", "observed", "evidence", "screen"),
                "source": ("observation source", "source"),
                "action": ("action", "step"),
                "observation_id": ("observation id", "receipt id", "screen receipt"),
                "screenshot_path": ("screenshot path", "path"),
                "screenshot_sha256": ("screenshot sha256", "sha256", "hash"),
                "captured_at": ("captured at", "timestamp", "time"),
                "current_time": ("current time", "reviewed at", "now"),
                "redaction_review": ("redaction review", "redaction", "privacy"),
            }.items():
                for alias in sorted(aliases, key=len, reverse=True):
                    match = re.search(rf"(?:^|;)\s*{re.escape(alias)}\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
                    if match:
                        args[arg_key] = match.group("value").strip()
                        break
            approval_match = re.search(r"\bapproval(?:\s+id|\s+#)?\s+(?P<value>\d+)\b", spec, re.IGNORECASE)
            if approval_match:
                args["approval_id"] = approval_match.group("value")
            max_age_match = re.search(r"\b(?:max age|max_age|max age seconds|max_age_seconds|ttl)\s*:?\s*(?P<value>\d+)\b", spec, re.IGNORECASE)
            if max_age_match:
                args["max_age_seconds"] = max_age_match.group("value")
            return Plan(
                goal="Check approved screen-observation receipt freshness before OAV proof review.",
                actions=[
                    PlannedAction(
                        "screen_observation_freshness_packet",
                        args,
                        "The user asked for a read-only freshness packet for supplied screenshot-observation receipt metadata.",
                    )
                ],
            )

        screen_confidence_match = re.search(
            r"^(?:screen observation confidence|screen confidence packet|screen observation confidence packet|visual confidence packet|screenshot confidence packet)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if screen_confidence_match:
            spec = (screen_confidence_match.group("spec") or "").strip()
            args: dict[str, object] = {"expectation": spec, "observation": "", "action": "", "source": ""}
            expectation_match = re.search(r"(?:^|;)\s*(?:expectation|expected|expect)\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
            observation_match = re.search(r"(?:^|;)\s*(?:observation|observed|evidence|screen)\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
            source_match = re.search(r"(?:^|;)\s*(?:source|observation source)\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
            action_match = re.search(r"(?:^|;)\s*(?:action|step)\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
            if expectation_match:
                args["expectation"] = expectation_match.group("value").strip()
            if observation_match:
                args["observation"] = observation_match.group("value").strip()
            if source_match:
                args["source"] = source_match.group("value").strip()
            if action_match:
                args["action"] = action_match.group("value").strip()
            return Plan(
                goal="Score screen observation confidence before computer-control review.",
                actions=[
                    PlannedAction(
                        "screen_observation_confidence_packet",
                        args,
                        "The user asked for a read-only screen confidence packet before computer control approval.",
                    )
                ],
            )

        screen_vision_prompt_match = re.search(
            r"^(?:screen vision prompt preview|screen vision prompt|vision prompt preview|visual verification prompt|visual model prompt|oav vision prompt preview|oav vision prompt)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if screen_vision_prompt_match:
            spec = (screen_vision_prompt_match.group("spec") or "").strip()
            args: dict[str, object] = {"spec": spec}
            for arg_key, aliases in {
                "expectation": ("expectation", "expected", "expect"),
                "observation": ("observation", "observed", "evidence", "screen"),
                "source": ("observation source", "source"),
                "action": ("action", "step"),
                "observation_id": ("observation id", "receipt id", "screen receipt"),
                "screenshot_path": ("screenshot path", "path"),
                "captured_at": ("captured at", "timestamp", "time"),
                "redaction_review": ("redaction review", "redaction", "privacy"),
            }.items():
                for alias in sorted(aliases, key=len, reverse=True):
                    match = re.search(rf"(?:^|;)\s*{re.escape(alias)}\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
                    if match:
                        args[arg_key] = match.group("value").strip()
                        break
            approval_match = re.search(r"\bapproval(?:\s+id|\s+#)?\s+(?P<value>\d+)\b", spec, re.IGNORECASE)
            if approval_match:
                args["approval_id"] = approval_match.group("value")
            return Plan(
                goal="Preview a bounded vision-model prompt for supplied approved screen observation evidence.",
                actions=[
                    PlannedAction(
                        "screen_vision_prompt_preview",
                        args,
                        "The user asked for a read-only vision prompt preview without observing, reading images, calling a model, or controlling the computer.",
                    )
                ],
            )

        screen_vision_review_match = re.search(
            r"^(?:screen vision model review preview|screen vision model review|screen vision review preview|screen vision review|oav vision model review|oav vision review|local vision review)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if screen_vision_review_match:
            spec = (screen_vision_review_match.group("spec") or "").strip()
            args: dict[str, object] = {"spec": spec}
            for arg_key, aliases in {
                "expectation": ("expectation", "expected", "expect"),
                "observation": ("observation", "observed", "evidence", "screen"),
                "source": ("observation source", "source"),
                "action": ("action", "step"),
                "observation_id": ("observation id", "receipt id", "screen receipt"),
                "screenshot_path": ("screenshot path", "path"),
                "captured_at": ("captured at", "timestamp", "time"),
                "redaction_review": ("redaction review", "redaction", "privacy"),
                "consent": ("consent", "reviewed", "confirmed"),
            }.items():
                for alias in sorted(aliases, key=len, reverse=True):
                    match = re.search(rf"(?:^|;)\s*{re.escape(alias)}\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
                    if match:
                        args[arg_key] = match.group("value").strip()
                        break
            approval_match = re.search(r"\bapproval(?:\s+id|\s+#)?\s+(?P<value>\d+)\b", spec, re.IGNORECASE)
            if approval_match:
                args["approval_id"] = approval_match.group("value")
            return Plan(
                goal="Queue an approval-gated local vision review for one supplied approved screenshot artifact.",
                actions=[
                    PlannedAction(
                        "screen_vision_model_review_preview",
                        args,
                        "The user asked to analyze a screenshot artifact with a local vision reviewer, which requires approval before reading image contents.",
                    )
                ],
            )

        screen_verification_contract_match = re.search(
            r"^(?:screen verification contract|screen verifier contract|visual verification contract|oav verification contract|observe act verify verifier)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if screen_verification_contract_match:
            spec = (screen_verification_contract_match.group("spec") or "").strip()
            args: dict[str, object] = {"expectation": spec, "observation": "", "action": ""}
            expectation_match = re.search(r"(?:^|;)\s*(?:expectation|expected|expect)\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
            observation_match = re.search(r"(?:^|;)\s*(?:observation|observed|evidence|screen)\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
            action_match = re.search(r"(?:^|;)\s*(?:action|step)\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
            if expectation_match:
                args["expectation"] = expectation_match.group("value").strip()
            if observation_match:
                args["observation"] = observation_match.group("value").strip()
            if action_match:
                args["action"] = action_match.group("value").strip()
            return Plan(
                goal="Compare supplied screen observation evidence with an expected state.",
                actions=[
                    PlannedAction(
                        "screen_verification_contract",
                        args,
                        "The user asked for a read-only screen verification contract without observing or controlling the computer.",
                    )
                ],
            )

        oav_proof_packet_match = re.search(
            r"^(?:observe act verify proof|observe act verify proof packet|oav proof|oav proof packet|computer control proof packet)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_proof_packet_match:
            spec = (oav_proof_packet_match.group("spec") or "").strip()
            args: dict[str, object] = {"spec": spec}
            for arg_key, aliases in {
                "expectation": ("expectation", "expected", "expect"),
                "observation": ("observation", "observed", "evidence", "screen"),
                "after_observation": ("after", "after observation", "verification", "verified"),
                "source": ("source", "observation source"),
                "action": ("action", "step"),
            }.items():
                for alias in sorted(aliases, key=len, reverse=True):
                    match = re.search(rf"(?:^|;)\s*{re.escape(alias)}\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
                    if match:
                        args[arg_key] = match.group("value").strip()
                        break
            x_match = re.search(r"\bx\b\s*:?\s*(-?\d+)", spec, re.IGNORECASE)
            y_match = re.search(r"\by\b\s*:?\s*(-?\d+)", spec, re.IGNORECASE)
            text_match = re.search(r"\btext\b\s*:?\s*(?P<text>[^;]+)", spec, re.IGNORECASE | re.DOTALL)
            if x_match:
                args["x"] = int(x_match.group(1))
            if y_match:
                args["y"] = int(y_match.group(1))
            if text_match:
                args["text"] = text_match.group("text").strip()
            return Plan(
                goal="Build read-only observe-act-verify proof before computer-control approval review.",
                actions=[
                    PlannedAction(
                        "observe_act_verify_proof_packet",
                        args,
                        "The user asked for an integrated read-only proof packet before any computer-control primitive moves toward approval.",
                    )
                ],
            )

        oav_route_lock_match = re.search(
            r"^(?:observe act verify route lock|oav route lock|computer route lock|computer control route lock|desktop route lock|screen control route lock)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_route_lock_match:
            spec = (oav_route_lock_match.group("spec") or "").strip()
            args: dict[str, object] = {"spec": spec}
            for arg_key, aliases in {
                "expectation": ("expectation", "expected", "expect"),
                "observation": ("observation", "observed", "evidence", "screen"),
                "after_observation": ("after", "after observation", "verification", "verified"),
                "source": ("source", "observation source"),
                "action": ("action", "step"),
            }.items():
                for alias in sorted(aliases, key=len, reverse=True):
                    match = re.search(rf"(?:^|;)\s*{re.escape(alias)}\s*:?\s*(?P<value>[^;]+)", spec, re.IGNORECASE)
                    if match:
                        args[arg_key] = match.group("value").strip()
                        break
            x_match = re.search(r"\bx\b\s*:?\s*(-?\d+)", spec, re.IGNORECASE)
            y_match = re.search(r"\by\b\s*:?\s*(-?\d+)", spec, re.IGNORECASE)
            text_match = re.search(r"\btext\b\s*:?\s*(?P<text>[^;]+)", spec, re.IGNORECASE | re.DOTALL)
            if x_match:
                args["x"] = int(x_match.group(1))
            if y_match:
                args["y"] = int(y_match.group(1))
            if text_match:
                args["text"] = text_match.group("text").strip()
            return Plan(
                goal="Keep computer-control routing locked until OAV proof and approval-chain evidence are present.",
                actions=[
                    PlannedAction(
                        "observe_act_verify_route_lock",
                        args,
                        "The user asked to gate whether a computer-control primitive may leave proof-only mode.",
                    )
                ],
            )

        oav_approval_bridge_match = re.search(
            r"^(?:observe act verify approval bridge|oav approval bridge|computer approval bridge|computer control approval bridge|desktop approval bridge|screen control approval bridge)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_approval_bridge_match:
            spec = (oav_approval_bridge_match.group("spec") or "").strip()
            return Plan(
                goal="Bind OAV proof, route lock, approval id, approved rerun, and verification receipt before final computer-control review.",
                actions=[
                    PlannedAction(
                        "observe_act_verify_approval_bridge",
                        {"spec": spec},
                        "The user asked to bridge observe-act-verify proof artifacts before any computer-control route can be treated as ready for final review.",
                    )
                ],
            )

        oav_cockpit_match = re.search(
            r"^(?:observe act verify cockpit|oav cockpit|computer cockpit|computer control cockpit|desktop cockpit|screen control cockpit|vision cockpit)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_cockpit_match:
            spec = (oav_cockpit_match.group("spec") or "").strip()
            return Plan(
                goal="Consolidate OAV proof, route lock, approval bridge, receipt binding, and blockers before final computer-control review.",
                actions=[
                    PlannedAction(
                        "observe_act_verify_cockpit",
                        {"spec": spec},
                        "The user asked for the observe-act-verify cockpit before any computer-control route can be treated as ready for final review.",
                    )
                ],
            )

        oav_final_review_match = re.search(
            r"^(?:observe act verify final review|oav final review|computer final review|computer control final review|desktop final review|screen control final review|vision final review)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_final_review_match:
            spec = (oav_final_review_match.group("spec") or "").strip()
            return Plan(
                goal="Check final OAV evidence before a human decision on one computer-control primitive.",
                actions=[
                    PlannedAction(
                        "observe_act_verify_final_review",
                        {"spec": spec},
                        "The user asked for the final observe-act-verify review packet before any computer-control primitive can be considered by the operator.",
                    )
                ],
            )

        oav_action_audit_match = re.search(
            r"^(?:observe act verify action audit|oav action audit|computer action audit|computer control action audit|desktop action audit|screen control action audit|vision action audit)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_action_audit_match:
            spec = (oav_action_audit_match.group("spec") or "").strip()
            return Plan(
                goal="Audit the exact OAV primitive before an approval decision.",
                actions=[
                    PlannedAction(
                        "observe_act_verify_action_audit",
                        {"spec": spec},
                        "The user asked for the final OAV action audit before one computer-control primitive can be considered for approval.",
                    )
                ],
            )

        oav_execution_handoff_match = re.search(
            r"^(?:observe act verify execution handoff|oav execution handoff|computer execution handoff|computer control execution handoff|desktop execution handoff|screen control execution handoff|vision execution handoff)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_execution_handoff_match:
            spec = (oav_execution_handoff_match.group("spec") or "").strip()
            return Plan(
                goal="Package the final OAV approval handoff before one computer-control primitive.",
                actions=[
                    PlannedAction(
                        "observe_act_verify_execution_handoff",
                        {"spec": spec},
                        "The user asked for the final OAV execution handoff before deciding on one computer-control primitive.",
                    )
                ],
            )

        oav_post_run_closure_match = re.search(
            r"^(?:observe act verify post run closure|observe act verify post-run closure|oav post run closure|oav post-run closure|computer post run closure|computer post-run closure|computer control post run closure|computer control post-run closure|desktop post run closure|screen control post run closure|vision post run closure)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_post_run_closure_match:
            spec = (oav_post_run_closure_match.group("spec") or "").strip()
            return Plan(
                goal="Close post-run OAV proof before any next computer-control primitive.",
                actions=[
                    PlannedAction(
                        "observe_act_verify_post_run_closure",
                        {"spec": spec},
                        "The user asked to close the post-run observe-act-verify proof loop before any next computer-control primitive can be reviewed.",
                    )
                ],
            )

        oav_cycle_ledger_match = re.search(
            r"^(?:observe act verify cycle ledger|observe-act-verify cycle ledger|oav cycle ledger|computer control cycle ledger|desktop control cycle ledger|screen control cycle ledger|vision cycle ledger|computer primitive cycle ledger)\s*:?\s*(?P<spec>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_cycle_ledger_match:
            spec = (oav_cycle_ledger_match.group("spec") or "").strip()
            return Plan(
                goal="Bind the full OAV primitive lifecycle into one read-only proof ledger.",
                actions=[
                    PlannedAction(
                        "observe_act_verify_cycle_ledger",
                        {"spec": spec},
                        "The user asked for a full observe-act-verify cycle ledger before any next computer-control primitive can be reviewed.",
                    )
                ],
            )

        local_read_match = re.search(
            r"^(?:read|open)\s+(?!(?:text\s+from\s+)?(?:this\s+|a\s+|the\s+)?(?:document photo|receipt photo|screenshot|picture|photo|image|receipt)\b)(?:file\s+)?(?P<path>.+)$",
            text,
            re.IGNORECASE,
        )
        if local_read_match and _looks_like_local_file_reference(local_read_match.group("path")):
            path = _clean_file_arg(local_read_match.group("path"))
            return Plan(
                goal=f"Read file {path}.",
                actions=[PlannedAction("read_text_file", {"path": path}, "The user asked to read a local text file.")],
            )

        fetch_match = re.search(r"^(?:fetch|read|summarize)\s+(?:page\s+)?(?P<url>https?://\S+|\S+\.\S+)$", text, re.IGNORECASE)
        if fetch_match:
            return Plan(
                goal="Fetch web page.",
                actions=[
                    PlannedAction(
                        "fetch_page",
                        {"url": fetch_match.group("url").strip()},
                        "The user asked to read a web page.",
                    )
                ],
            )

        links_match = re.search(r"^(?:extract|list|show)\s+links\s+(?:from|on)\s+(?P<url>https?://\S+|\S+\.\S+)$", text, re.IGNORECASE)
        if links_match:
            return Plan(
                goal="Extract web page links.",
                actions=[
                    PlannedAction(
                        "extract_links",
                        {"url": links_match.group("url").strip()},
                        "The user asked to extract links from a web page.",
                    )
                ],
            )

        open_url_match = re.search(
            r"^(?:open|launch)\s+(?:(?:the\s+)?(?:url|website|site|link)\s+(?P<url2>\S+)|(?P<url>https?://\S+|www\.\S+))$",
            text,
            re.IGNORECASE,
        )
        if open_url_match:
            url = (open_url_match.group("url") or open_url_match.group("url2") or "").strip()
            return Plan(
                goal="Open URL.",
                actions=[
                    PlannedAction(
                        "open_url",
                        {"url": url},
                        "The user asked to open a URL.",
                    )
                ],
            )

        web_search_match = re.search(r"^(?:web search|search web|search online|google)\s+(?P<query>.+)$", text, re.IGNORECASE)
        if web_search_match:
            return Plan(
                goal="Search the web.",
                actions=[
                    PlannedAction(
                        "web_search",
                        {"query": _clean_freeform_query(web_search_match.group("query"))},
                        "The user asked to search the web.",
                    )
                ],
            )

        if "recent browser" in low or "browser history" in low or "recent pages" in low:
            return Plan(
                goal="List recent browser pages.",
                actions=[PlannedAction("recent_browser_pages", {}, "The user asked for fetched browser page history.")],
            )

        summarize_page_match = re.search(r"^(?:summarize|summary of)\s+(?:page\s+)?#?(?P<page_id>\d+)$", text, re.IGNORECASE)
        if summarize_page_match:
            return Plan(
                goal="Summarize browser page.",
                actions=[
                    PlannedAction(
                        "summarize_page",
                        {"page_id": int(summarize_page_match.group("page_id"))},
                        "The user asked to summarize a fetched page.",
                    )
                ],
            )

        if low in {"summarize latest page", "summarize recent page", "summarize last page"}:
            return Plan(
                goal="Summarize latest browser page.",
                actions=[PlannedAction("summarize_page", {}, "The user asked to summarize the latest fetched page.")],
            )

        save_page_match = re.search(r"^(?:save|write)\s+(?:page\s+)?#?(?P<page_id>\d+)\s+(?:to\s+)?(?:obsidian|notes)$", text, re.IGNORECASE)
        if save_page_match:
            return Plan(
                goal="Save browser page note.",
                actions=[
                    PlannedAction(
                        "save_page_note",
                        {"page_id": int(save_page_match.group("page_id"))},
                        "The user asked to save a browser page note.",
                    )
                ],
            )

        if low in {"save latest page to obsidian", "save recent page to obsidian", "save last page to obsidian"}:
            return Plan(
                goal="Save latest browser page note.",
                actions=[PlannedAction("save_page_note", {}, "The user asked to save the latest browser page note.")],
            )

        # Bounded Korean clock/weather aliases avoid a model fallback for common
        # read-only voice transcripts. Keep the two intents distinct: the
        # transcribed words, rather than an inferred original utterance, govern
        # routing.
        korean_timezone_time = re.fullmatch(
            r"(?P<city>서울|도쿄|뉴욕|런던|UTC)\s*(?:은|는|에)?\s*"
            r"(?:(?:지금|현재)\s*)?(?:(?:몇|며)\s*시(?:야|아|예요|에요|니|인가요|인지\s*(?:알려\s*줘(?:요)?|알려\s*주세요))?|"
            r"시간(?:은|이)?(?:\s*(?:뭐야|뭐예요|무엇(?:이야|인가요)?|알려\s*줘(?:요)?|"
            r"알려\s*주세요|보여\s*줘(?:요)?|보여\s*주세요|확인해\s*줘(?:요)?|확인해\s*주세요))?)\s*[?!.]*",
            text,
            re.IGNORECASE,
        )
        if korean_timezone_time:
            location = {
                "서울": "Seoul",
                "도쿄": "Tokyo",
                "뉴욕": "New York",
                "런던": "London",
                "utc": "UTC",
            }[korean_timezone_time.group("city").casefold()]
            return Plan(
                goal="Answer with the requested current time.",
                actions=[
                    PlannedAction(
                        "current_time",
                        {"location": location, "locale": "ko"},
                        "User asked in Korean for a supported location's current time.",
                    )
                ],
            )
        korean_current_time = re.fullmatch(
            r"(?:지금|현재)?\s*(?:(?:몇|며)\s*시(?:야|아|예요|에요|니|인가요|인지\s*(?:알려\s*줘(?:요)?|알려\s*주세요))?|"
            r"시간(?:은|이)?(?:\s*(?:뭐야|뭐예요|무엇(?:이야|인가요)?|알려\s*줘(?:요)?|"
            r"알려\s*주세요|보여\s*줘(?:요)?|보여\s*주세요|확인해\s*줘(?:요)?|확인해\s*주세요))?)\s*[?!.]*",
            text,
        )
        korean_current_time_suffix = re.search(
            r"(?:^|[\s,])(?:지금|현재)?\s*(?:몇|며)\s*시(?:야|아|예요|에요|니|인가요)\s*[?!.]*$",
            text,
        )
        if korean_current_time or korean_current_time_suffix:
            return Plan(
                goal="Answer with the local current time.",
                actions=[PlannedAction("current_time", {"locale": "ko"}, "User asked in Korean for the current time.")],
            )
        korean_weather = re.fullmatch(
            r"(?:지금|현재|오늘)?\s*날씨(?:\s*(?:야|어때(?:요)?|알려\s*줘(?:요)?|"
            r"알려\s*주세요|보여\s*줘(?:요)?|보여\s*주세요|확인해\s*줘(?:요)?|"
            r"확인해\s*주세요))?\s*[?!.]*",
            text,
        )
        if korean_weather:
            return Plan(
                goal="Report the weather.",
                actions=[PlannedAction("get_weather", {}, "User asked in Korean for the weather.")],
            )

        # Allow a trailing "now", "right now", or "today" plus filler so phrasings
        # like "what time is it now" route to the tool instead of the slow model.
        time_command = re.sub(r"\s+(right\s+now|now|today|currently|please|for me)$", "", low_command).strip()
        if re.search(r"\b(?:daylight savings?|dst)\b", low_command):
            return Plan(goal="No deterministic daylight-saving route.", actions=[])
        sun_time_like = bool(re.search(r"\bsun\s*(?:rise|set)\b|\bsunrise\b|\bsunset\b", low_command))
        time_until_m = re.fullmatch(
            r"(?:how much (?:time|longer)(?:\s+(?:left|remaining))?|time(?:\s+(?:left|remaining))?)\s+(?:until|til|till|to|before)\s+(.{2,60}?)[\?\.!]*",
            text,
            re.IGNORECASE,
        )
        if time_until_m:
            return Plan(
                goal="Count days until a date.",
                actions=[PlannedAction("days_until", {"target": time_until_m.group(1).strip()}, "User asked for a countdown.")],
            )
        local_timezone_phrases = {
            "timezone",
            "time zone",
            "current timezone",
            "current time zone",
            "local timezone",
            "local time zone",
            "my timezone",
            "my time zone",
            "what timezone am i in",
            "what time zone am i in",
            "what's my timezone",
            "whats my timezone",
            "what is my timezone",
            "what is my time zone",
            "what timezone is it",
            "what time zone is it",
        }
        if time_command in local_timezone_phrases:
            return Plan(
                goal="Answer with the local timezone.",
                actions=[PlannedAction("current_time", {"show_timezone": True}, "The user asked for the local timezone.")],
            )
        timezone_location_match = re.search(
            r"^(?:what\s+)?(?:time\s+zone|timezone)\s+(?:is|in|for)\s+(?P<location>[a-z][a-z\s]{1,40}|utc)$",
            low_command,
        ) or re.search(
            r"^(?P<location>[a-z][a-z\s]{1,40}?|utc)\s+(?:time\s+zone|timezone)$",
            low_command,
        )
        if timezone_location_match:
            return Plan(
                goal="Answer with the timezone for a location.",
                actions=[
                    PlannedAction(
                        "current_time",
                        {"location": timezone_location_match.group("location").strip(), "show_timezone": True},
                        "The user asked for a location timezone.",
                    )
                ],
            )
        time_difference_locations = _parse_time_difference_request(low_command)
        if time_difference_locations:
            source, target = time_difference_locations
            return Plan(
                goal="Compare timezone offsets between two locations.",
                actions=[
                    PlannedAction(
                        "time_difference",
                        {"source": source, "target": target},
                        "The user asked for the time difference between two locations.",
                    )
                ],
            )
        relative_date_target = _parse_relative_date_request(low_command)
        if relative_date_target:
            return Plan(
                goal="Answer a local relative-date question.",
                actions=[
                    PlannedAction(
                        "relative_date",
                        {"target": relative_date_target},
                        "The user asked for simple local date math.",
                    )
                ],
            )
        world_time_match = re.search(
            r"^(?:what(?:'s| is)?\s+)?(?:the\s+)?(?:current\s+)?time(?:\s+is\s+it)?\s+(?:in|at|for)\s+(?P<location>[a-z][a-z\s]{1,40}|utc)$",
            low_command,
        )
        terse_world_time_match = re.search(
            r"^(?:(?:current|local)\s+)?time\s+(?:in\s+)?(?P<location>[a-z][a-z\s]{1,40}|utc)$",
            low_command,
        ) or re.search(
            r"^(?P<location>[a-z][a-z\s]{1,40}?|utc)\s+(?:(?:current|local)\s+)?time$",
            low_command,
        )
        question_world_time_match = re.search(
            r"^what\s+time\s+(?:(?:is\s+it|is)\s+)?(?:in\s+)?(?P<location>[a-z][a-z\s]{1,40}|utc)$",
            low_command,
        )
        if question_world_time_match:
            question_location = question_world_time_match.group("location").strip()
            if question_location in {"it", "is it"} or question_location.startswith("is "):
                question_world_time_match = None
        world_time_match = world_time_match or terse_world_time_match or question_world_time_match
        if world_time_match and not sun_time_like:
            return Plan(
                goal="Answer with the current time for a location.",
                actions=[
                    PlannedAction(
                        "current_time",
                        {"location": world_time_match.group("location").strip()},
                        "The user asked for the current time in a location.",
                    )
                ],
            )
        if time_command in {
            "time",
            "date",
            "month",
            "year",
            "current time",
            "current date",
            "what time is it",
            "whats the time",
            "what's the time",
            "what is the time",
            "what time is it now",
            "do you have the time",
            "got the time",
            "what date is it",
            "what date is",
            "what date is today",
            "whats the date",
            "what's the date",
            "what is the date",
            "whats todays date",
            "what's today's date",
            "what is todays date",
            "what is today's date",
            "tell me todays date",
            "tell me today's date",
            "what is today",
            "what day is it",
            "what day is",
            "what day is today",
            "what's the day",
            "whats the day",
            "what day of week is it",
            "what day of the week is it",
            "what's the day of the week",
            "whats the day of the week",
            "day of week",
            "day of the week",
            "current day of week",
            "current day of the week",
            "what month is it",
            "what year is it",
            "current month",
            "current year",
            "month today",
            "year today",
            "today's date",
            "todays date",
            "today date",
        } or re.fullmatch(r"(what(')?s |what is |tell me )?the (time|date|day)", time_command):
            return Plan(
                goal="Answer with the current time.",
                actions=[PlannedAction("current_time", {}, "The user asked for time or date.")],
            )

        if low.startswith("daily note ") or low.startswith("write daily note "):
            body = re.sub(r"^(write\s+)?daily note\s+", "", text, flags=re.IGNORECASE).strip()
            return Plan(
                goal="Write to the Obsidian daily note.",
                actions=[
                    PlannedAction(
                        "write_daily_note",
                        {"heading": "Quick Capture", "body": body},
                        "The user asked to write a daily note.",
                    )
                ],
            )

        list_tools_match = re.search(
            r"^(?:(?:list|show)\s+tools|tool\s+list)(?:\s+(?:(?:for|in)\s+)?(?P<toolset>[a-z_ -]+?))?(?:\s+(?:please|pls|thanks|thank you))?$",
            text,
            re.IGNORECASE,
        )
        if list_tools_match:
            toolset = _strip_trailing_politeness((list_tools_match.group("toolset") or "").strip()).strip().replace(" ", "_")
            return Plan(
                goal="List available tools.",
                actions=[
                    PlannedAction(
                        "list_tools",
                        {"toolset": toolset} if toolset else {},
                        "The user asked what functions are available.",
                    )
                ],
            )

        if low in {
            "tools",
            "tools please",
            "available tools",
            "available tools please",
            "jarvis tools",
            "jarvis tools please",
            "jarvis tool list",
            "jarvis tool list please",
            "your tools",
            "your tools please",
            "your tool list",
            "your tool list please",
            "what are your tools",
            "what are your tools please",
            "what are jarvis tools",
            "what are jarvis tools please",
            "what are jarvis's tools",
            "what are jarvis's tools please",
            "what tools does jarvis have",
            "what tools does jarvis have please",
            "what tools do you have",
            "what tools do you have please",
            "what functions do you have",
            "what functions do you have please",
        }:
            return Plan(
                goal="List available tools.",
                actions=[PlannedAction("list_tools", {}, "The user asked what functions are available.")],
            )

        if low in {
            "jarvis status",
            "assistant status",
            "brain status",
            "status",
            "show status",
            "show latest status",
            "show jarvis status",
            "show latest jarvis status",
        }:
            return Plan(
                goal="Show Jarvis status.",
                actions=[PlannedAction("jarvis_status", {}, "The user asked for Jarvis status.")],
            )

        if low in {
            "brain loop",
            "brain loop report",
            "show brain loop",
            "show brain loop report",
            "show latest brain loop",
            "show latest brain loop report",
            "jarvis brain loop",
            "agent loop",
            "assistant loop",
            "how is jarvis thinking",
            "how is jarvis oriented",
        }:
            return Plan(
                goal="Show Jarvis brain loop.",
                actions=[PlannedAction("brain_loop_report", {}, "The user asked for Jarvis's current perception-memory-planning-action loop.")],
            )

        if low in {
            "recent tool runs",
            "recent tool runs please",
            "tool runs",
            "tool runs please",
            "audit",
            "audit please",
            "audit log",
            "audit log please",
            "recent audit",
            "recent audit please",
            "show audit",
            "show audit please",
            "show latest audit",
            "show latest audit please",
            "what did you run",
            "what did you run please",
            "what you ran",
            "what you ran please",
            "what tools ran recently",
            "what tools have run recently",
        }:
            return Plan(
                goal="Show recent tool audit log.",
                actions=[PlannedAction("recent_tool_runs", {}, "The user asked for recent tool activity.")],
            )

        if low in {
            "pending approvals",
            "pending approvals please",
            "pending approval",
            "pending approval please",
            "list approvals",
            "list approvals please",
            "approval list",
            "approval list please",
            "approvals",
            "approvals please",
            "approval queue",
            "approval queue please",
            "show approvals",
            "show approvals please",
            "show latest approvals",
            "show latest approvals please",
            "what needs approval",
            "what needs approval please",
            "what needs my approval",
            "what needs my approval please",
            "what is waiting for approval",
            "what is waiting for approval please",
            "what approvals are pending",
            "what approvals are pending please",
            "which approvals are pending",
            "which approvals are pending please",
            "what is pending approval",
            "what is pending approval please",
            "what's pending approval",
            "what's pending approval please",
            "anything waiting for approval",
            "anything waiting for approval please",
            "anything waiting on approval",
            "anything waiting on approval please",
            "anything i need to approve",
            "anything i need to approve please",
            "do i need to approve anything",
            "do i need to approve anything please",
            "do you need approval from me",
            "do you need approval from me please",
            "what do you need me to approve",
            "what do you need me to approve please",
            "pending approval queue",
            "pending approval queue please",
            "show waiting approvals",
            "show waiting approvals please",
            "show approval queue",
            "show approval queue please",
            "show latest approval queue",
            "show latest approval queue please",
            "do i have pending approvals",
            "do i have pending approvals please",
            "are there pending approvals",
            "are there pending approvals please",
        }:
            return Plan(
                goal="Show pending approval queue.",
                actions=[PlannedAction("list_pending_approvals", {}, "The user asked for blocked requests waiting on approval.")],
            )

        if low in {
            "approval summary",
            "approval summary please",
            "approval queue summary",
            "approval queue summary please",
            "pending approval summary",
            "pending approval summary please",
            "summarize approvals",
            "summarize approvals please",
            "summarize pending approvals",
            "summarize pending approvals please",
            "approval glance",
            "approval glance please",
            "approval status",
            "approval status please",
            "approval report",
            "approval report please",
            "approval queue status",
            "approval queue status please",
            "show approval status",
            "show approval status please",
            "show approval report",
            "show approval report please",
            "show approval queue status",
            "show approval queue status please",
            "what approval is blocking jarvis",
            "what approval is blocking jarvis please",
            "what approval is blocking completion",
            "what approval is blocking completion please",
            "what approval blocks jarvis",
            "what approval blocks jarvis please",
            "what approval blocks completion",
            "what approval blocks completion please",
        }:
            return Plan(
                goal="Summarize pending approvals.",
                actions=[PlannedAction("approval_queue_summary", {}, "The user asked for a compact read-only summary of pending approvals.")],
            )

        if low in {"approval readiness", "approval readiness packet", "approval go no go", "approval go/no-go", "latest approval readiness"}:
            return Plan(
                goal="Check approval readiness before last-look approval.",
                actions=[
                    PlannedAction(
                        "approval_readiness_packet",
                        {"approval_id": "latest"},
                        "The user asked for a read-only approval readiness check before any risky rerun.",
                    )
                ],
            )

        if low in {
            "approval review",
            "review approvals",
            "review pending approvals",
            "should i approve",
            "should i approve these",
            "approval risk review",
        }:
            return Plan(
                goal="Review pending approval risks.",
                actions=[PlannedAction("review_pending_approvals", {}, "The user asked to review pending approval risks before approving.")],
            )

        if low in {
            "approval history",
            "approval history please",
            "approval audit",
            "approval log",
            "latest approval history",
            "latest approval history please",
            "show approval history",
            "show approval history please",
            "show latest approval history",
            "show latest approval history please",
            "recent approvals",
            "recent approval history",
            "what approvals happened",
            "what approval decisions happened",
        }:
            return Plan(
                goal="Show approval history.",
                actions=[PlannedAction("approval_history", {}, "The user asked for an audit-only view of recent approval decisions.")],
            )

        if low in {"show latest approval", "show last approval", "review latest approval", "review last approval", "inspect latest approval", "inspect last approval"}:
            return Plan(
                goal="Inspect pending approval.",
                actions=[
                    PlannedAction(
                        "inspect_pending_approval",
                        {"approval_id": "latest"},
                        "The user asked to inspect the latest queued approval before deciding.",
                    )
                ],
            )

        if low in {"preview latest approval", "preview last approval", "last look", "approval last look"}:
            return Plan(
                goal="Preview approval execution packet.",
                actions=[
                    PlannedAction(
                        "approval_execution_packet",
                        {"approval_id": "latest"},
                        "The user asked for a read-only last-look packet before approving the latest request.",
                    )
                ],
            )

        inspect_approval_match = re.search(
            r"^(?:(?:inspect|show|view|review)\s+(?:pending\s+)?approval|(?:inspect|show|view|review)\s+(?:latest|last|newest|current)\s+approval|approval\s+details?|approval\s+detail)\s+#?(?P<approval_id>\d+|latest|last|newest|current)?$",
            text,
            re.IGNORECASE,
        )
        if inspect_approval_match:
            approval_id = inspect_approval_match.group("approval_id") or "latest"
            return Plan(
                goal="Inspect pending approval.",
                actions=[
                    PlannedAction(
                        "inspect_pending_approval",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked to inspect one queued approval before deciding.",
                    )
                ],
            )

        approval_packet_match = re.search(
            r"^(?:approval execution packet|approval packet|approval last look|last look approval|pre-approval packet|preview approval|preview latest approval|last look)\s+#?(?P<approval_id>\d+|latest|last|newest|current)?$",
            text,
            re.IGNORECASE,
        )
        if approval_packet_match:
            approval_id = approval_packet_match.group("approval_id") or "latest"
            return Plan(
                goal="Preview approval execution packet.",
                actions=[
                    PlannedAction(
                        "approval_execution_packet",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked for a read-only last-look packet before approving one request.",
                    )
                ],
            )

        natural_approval_last_look_match = re.search(
            r"^(?:show (?:me )?(?:the )?(?:last[- ]look|approval packet|execution packet)(?: for)?(?: approval)?|what would (?:approval|approving approval)|preview what happens if (?:i|we|jarvis) approve(?: approval)?|what happens if (?:i|we|jarvis) approve(?: approval)?)\s*#?(?P<approval_id>\d+|latest|last|newest|current)?(?:\s+do)?$",
            text,
            re.IGNORECASE,
        )
        if natural_approval_last_look_match:
            approval_id = natural_approval_last_look_match.group("approval_id") or "latest"
            return Plan(
                goal="Preview approval execution packet.",
                actions=[
                    PlannedAction(
                        "approval_execution_packet",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked in natural language for the read-only last-look packet before approving one request.",
                    )
                ],
            )

        approval_readiness_match = re.search(
            r"^(?:approval readiness packet|approval readiness|approval go/no-go|approval go no go|approval preflight|approval queue position)\s+#?(?P<approval_id>\d+|latest|last|newest|current)?$",
            text,
            re.IGNORECASE,
        )
        if approval_readiness_match:
            approval_id = approval_readiness_match.group("approval_id") or "latest"
            return Plan(
                goal="Check approval readiness before last-look approval.",
                actions=[
                    PlannedAction(
                        "approval_readiness_packet",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked to inspect queue position, staleness, exact arguments, and next safe approval command.",
                    )
                ],
            )

        approval_resume_match = re.search(
            r"^(?:approval resume packet|approval resume|resume approval|resume queued approval|approval rerun packet|rerun packet)\s+#?(?P<approval_id>\d+|latest|last|newest|current)?$",
            text,
            re.IGNORECASE,
        )
        if approval_resume_match:
            approval_id = approval_resume_match.group("approval_id") or "latest"
            return Plan(
                goal="Preview one-shot approval resume contract.",
                actions=[
                    PlannedAction(
                        "approval_resume_packet",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked to preview the safe one-shot resume contract for one queued approval before approving it.",
                    )
                ],
            )

        natural_approval_readiness_match = re.search(
            r"^(?:is (?:approval\s*)?#?(?P<approval_id_a>\d+|latest|last|newest|current)?\s*(?:ready|safe) (?:to )?(?:approve|run|rerun)"
            r"|is (?:approval\s*)?#?(?P<approval_id_f>\d+|latest|last|newest|current)\s+(?:safe|ready)\??"
            r"|(?:approval\s*)?#?(?P<approval_id_h>\d+|latest|last|newest|current)\s+(?:safe|ready)\??"
            r"|is it (?:safe|okay|ok|alright) to (?:approve|run|rerun) (?:approval\s*)?#?(?P<approval_id_e>\d+|latest|last|newest|current)?"
            r"|can (?:i|we|jarvis) (?:approve|run|rerun) (?:approval\s*)?#?(?P<approval_id_b>\d+|latest|last|newest|current)?(?: now| safely)?"
            r"|should (?:i|we|jarvis) approve (?:(?:the|this)\s+)?(?:approval\s*)?#?(?P<approval_id_c>\d+|latest|last|newest|current)?(?:\s+approval)?"
            r"|what should (?:i|we|jarvis) check before approving (?:approval\s*)?#?(?P<approval_id_d>\d+|latest|last|newest|current)?"
            r"|(?:check|review|inspect)\s+(?:approval\s*)?#?(?P<approval_id_g>\d+|latest|last|newest|current)\s+before\s+(?:approving|running|rerunning))$",
            text,
            re.IGNORECASE,
        )
        if natural_approval_readiness_match:
            approval_id = (
                natural_approval_readiness_match.group("approval_id_a")
                or natural_approval_readiness_match.group("approval_id_f")
                or natural_approval_readiness_match.group("approval_id_h")
                or natural_approval_readiness_match.group("approval_id_e")
                or natural_approval_readiness_match.group("approval_id_b")
                or natural_approval_readiness_match.group("approval_id_c")
                or natural_approval_readiness_match.group("approval_id_d")
                or natural_approval_readiness_match.group("approval_id_g")
                or "latest"
            )
            return Plan(
                goal="Check approval readiness before last-look approval.",
                actions=[
                    PlannedAction(
                        "approval_readiness_packet",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked in natural language whether a queued approval is safe or ready before any risky rerun.",
                    )
                ],
            )

        natural_approval_readiness_object_match = re.search(
            r"^(?:can|could|should)\s+(?:i|we|jarvis)\s+(?:approve|run|rerun)\s+(?:(?:the|this|that)\s+)?(?:approval\s*)?#?(?:approval\s+)?(?P<approval_id_can>\d+|latest|last|newest|current)?(?:\s+approval)?(?:\s+(?:now|safely|today))?$"
            r"|^what should (?:i|we|jarvis) check before approving\s+(?:(?:the|this|that)\s+)?(?:approval\s*)?#?(?:approval\s+)?(?P<approval_id_check>\d+|latest|last|newest|current)?(?:\s+approval)?$"
            rf"|^(?:what is|what's|why is)\s+(?:approval\s*)?#?(?P<approval_id_wait>\d+|latest|last|newest|current)\s+(?:waiting for|blocked|held|not ready)\??$"
            rf"|^(?:approval\s*)?#?(?P<approval_id_next>\d+|latest|last|newest|current)\s+(?:next step|next command)\??$"
            rf"|^approval\s*#?(?P<approval_id_status>\d+|latest|last|newest|current)\s+(?:status|readiness)\??$",
            text,
            re.IGNORECASE,
        )
        if natural_approval_readiness_object_match:
            approval_id = (
                natural_approval_readiness_object_match.group("approval_id_can")
                or natural_approval_readiness_object_match.group("approval_id_check")
                or natural_approval_readiness_object_match.group("approval_id_wait")
                or natural_approval_readiness_object_match.group("approval_id_next")
                or natural_approval_readiness_object_match.group("approval_id_status")
                or "latest"
            )
            return Plan(
                goal="Check approval readiness before last-look approval.",
                actions=[
                    PlannedAction(
                        "approval_readiness_packet",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked whether a queued approval is ready, blocked, or safe before any risky rerun.",
                    )
                ],
            )

        if re.search(
            r"^(?:next approval step|next approval command|approval next step|approval next command|what is the next approval command|what's the next approval command|what should i do next for approvals|what should we do next for approvals)$",
            text,
            re.IGNORECASE,
        ):
            return Plan(
                goal="Summarize pending approvals.",
                actions=[
                    PlannedAction(
                        "approval_queue_summary",
                        {},
                        "The user asked for the next safe approval command at the queue level.",
                    )
                ],
            )

        natural_approval_last_look_object_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:last[- ]look|approval packet|execution packet)\s+(?:for\s+)?(?:(?:the|this|that)\s+)?(?:approval\s*)?#?(?:approval\s+)?(?P<approval_id_packet>\d+|latest|last|newest|current)?(?:\s+approval)?$"
            r"|^(?:what happens if|what would happen if|preview what happens if)\s+(?:i|we|jarvis)?\s*(?:approve|run|rerun)\s+(?:(?:the|this|that)\s+)?(?:approval\s*)?#?(?:approval\s+)?(?P<approval_id_preview>\d+|latest|last|newest|current)?(?:\s+approval)?$",
            text,
            re.IGNORECASE,
        )
        if natural_approval_last_look_object_match:
            approval_id = (
                natural_approval_last_look_object_match.group("approval_id_packet")
                or natural_approval_last_look_object_match.group("approval_id_preview")
                or "latest"
            )
            return Plan(
                goal="Preview approval execution packet.",
                actions=[
                    PlannedAction(
                        "approval_execution_packet",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked in natural language for the read-only last-look packet before approving one request.",
                    )
                ],
            )

        natural_approval_resume_object_match = re.search(
            r"^(?:resume|rerun|run)\s+(?:(?:the|this|that)\s+)?(?:approval\s*)?#?(?:approval\s+)?(?P<approval_id>\d+|latest|last|newest|current)?(?:\s+approval)?\s+(?:safely|resume packet|rerun packet|packet)$",
            text,
            re.IGNORECASE,
        )
        if natural_approval_resume_object_match:
            approval_id = natural_approval_resume_object_match.group("approval_id") or "latest"
            return Plan(
                goal="Preview one-shot approval resume contract.",
                actions=[
                    PlannedAction(
                        "approval_resume_packet",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked to preview the safe one-shot resume contract for one queued approval before approving it.",
                    )
                ],
            )

        natural_approval_chain_object_match = re.search(
            r"^(?:prove|show proof|show evidence|approval proof|approval evidence)\s+(?:that\s+)?(?:(?:the|this|that)\s+)?(?:approval\s*)?#?(?:approval\s+)?(?P<approval_id_proof>\d+|latest|last|newest|current)?(?:\s+approval)?(?:\s+(?:ran|reran|executed|completed))?$"
            r"|^(?:approval proof|approval evidence)\s+(?:for\s+)(?:(?:the|this|that)\s+)?(?:approval\s*)?#?(?:approval\s+)?(?P<approval_id_evidence>\d+|latest|last|newest|current)?(?:\s+approval)?$",
            text,
            re.IGNORECASE,
        )
        if natural_approval_chain_object_match:
            approval_id = (
                natural_approval_chain_object_match.group("approval_id_proof")
                or natural_approval_chain_object_match.group("approval_id_evidence")
                or "latest"
            )
            return Plan(
                goal="Verify approval-chain execution proof.",
                actions=[
                    PlannedAction(
                        "approval_chain_proof",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked to verify whether an approval has linked execution evidence.",
                    )
                ],
            )

        approval_chain_match = re.search(
            r"^(?:approval chain proof|approval proof|approval evidence|prove approval|approval execution proof|approval audit proof)\s+#?(?P<approval_id>\d+|latest|last|newest|current)?$",
            text,
            re.IGNORECASE,
        )
        if approval_chain_match:
            approval_id = approval_chain_match.group("approval_id") or "latest"
            return Plan(
                goal="Verify approval-chain execution proof.",
                actions=[
                    PlannedAction(
                        "approval_chain_proof",
                        {"approval_id": _approval_id_arg(approval_id)},
                        "The user asked to verify whether an approval has linked execution evidence.",
                    )
                ],
            )

        reversed_latest_approval_match = re.search(
            r"^(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?P<approval_id>latest|last|newest|current)\s+approval\s+(?P<kind>readiness|details?|packet|execution packet|last[- ]look|chain proof|proof|evidence)$",
            text,
            re.IGNORECASE,
        )
        if reversed_latest_approval_match:
            approval_id = _approval_id_arg(reversed_latest_approval_match.group("approval_id"))
            kind = reversed_latest_approval_match.group("kind").lower().replace("-", " ")
            if kind == "readiness":
                return Plan(
                    goal="Check approval readiness before last-look approval.",
                    actions=[
                        PlannedAction(
                            "approval_readiness_packet",
                            {"approval_id": approval_id},
                            "The user asked to inspect latest approval readiness in reversed natural wording.",
                        )
                    ],
                )
            if kind in {"detail", "details"}:
                return Plan(
                    goal="Inspect pending approval.",
                    actions=[
                        PlannedAction(
                            "inspect_pending_approval",
                            {"approval_id": approval_id},
                            "The user asked to inspect the latest queued approval in reversed natural wording.",
                        )
                    ],
                )
            if kind in {"chain proof", "proof", "evidence"}:
                return Plan(
                    goal="Verify approval-chain execution proof.",
                    actions=[
                        PlannedAction(
                            "approval_chain_proof",
                            {"approval_id": approval_id},
                            "The user asked to verify latest approval proof in reversed natural wording.",
                        )
                    ],
                )
            return Plan(
                goal="Preview approval execution packet.",
                actions=[
                    PlannedAction(
                        "approval_execution_packet",
                        {"approval_id": approval_id},
                        "The user asked for the latest read-only approval packet in reversed natural wording.",
                    )
                ],
            )

        if low in {
            "save approval review",
            "write approval review",
            "export approval review",
            "save pending approval review",
            "write pending approval review",
        }:
            return Plan(
                goal="Save pending approval review.",
                actions=[PlannedAction("save_approval_review", {}, "The user asked to save the pending approval risk review.")],
            )

        dismiss_approval_match = re.search(
            r"^(?:dismiss|clear|remove)\s+(?:(?:pending\s+)?approval\s*)?#?(?P<approval_id>\d+|latest|last|newest|current)$",
            text,
            re.IGNORECASE,
        )
        if dismiss_approval_match:
            return Plan(
                goal="Dismiss pending approval.",
                actions=[
                    PlannedAction(
                        "dismiss_pending_approval",
                        {"approval_id": _approval_id_arg(dismiss_approval_match.group("approval_id"))},
                        "The user asked to dismiss a pending approval.",
                    )
                ],
            )

        approve_approval_match = re.search(
            r"^(?:approve|run|rerun)\s+(?:(?:pending\s+)?approval\s*)?#?(?P<approval_id>\d+|latest|last|newest|current)$",
            text,
            re.IGNORECASE,
        )
        if approve_approval_match:
            return Plan(
                goal="Approve and rerun pending request.",
                actions=[
                    PlannedAction(
                        "approve_pending_approval",
                        {"approval_id": _approval_id_arg(approve_approval_match.group("approval_id"))},
                        "The user explicitly approved a pending high-risk request.",
                    )
                ],
            )

        if re.search(
            # Real gap found live 2026-07-10: unlike round 33/34's partial-
            # match clause-bleed bugs, this trigger requires an EXACT match
            # with no trailing text at all, so "save state and then show my
            # calendar" failed the whole match here and fell through to a
            # LATER regex that caught "show my calendar" instead -- silently
            # dropping the "save state" request entirely rather than writing
            # wrong data. Added an "and then ..." tail as an accepted,
            # ignored suffix so the primary intent is no longer silently lost.
            r"^(?:export\s{1,10}state|save\s{1,10}state|(?:show\s{1,10})?state\s{1,10}snapshot|current\s{1,10}context\s{1,10}snapshot|update\s{1,10}current\s{1,10}context)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?(?:\s{1,10}(?:please|pls|thanks|thank you))?$",
            text,
            re.IGNORECASE,
        ):
            return Plan(
                goal="Export Jarvis state snapshot.",
                actions=[PlannedAction("export_state_snapshot", {}, "The user asked to export current Jarvis state.")],
            )

        if low in {"status dashboard", "jarvis dashboard", "dashboard", "open dashboard"}:
            return Plan(
                goal="Show dashboard launch info.",
                actions=[PlannedAction("status_dashboard", {}, "The user asked about the Jarvis dashboard.")],
            )

        if low in {
            "jarvis doctor",
            "assistant doctor",
            "doctor",
            "diagnose jarvis",
            "setup doctor",
            "diagnostics",
            "diagnostic report",
            "health",
            "health check",
            "jarvis health",
            "jarvis health check",
            # Real bug found live 2026-07-09: "run doctor" / "run diagnostics"
            # fell through every specific handler down to the generic
            # `run <shell command>` catch-all, which queued a HIGH_RISK
            # approval to literally shell-exec a command named "doctor" /
            # "diagnostics" instead of showing Jarvis's own READ_ONLY
            # diagnostic report -- the wrong tool entirely, at the wrong risk
            # tier, for an unambiguous read-only request.
            "run doctor",
            "run diagnostics",
            "run diagnostic",
            "run diagnostic report",
            "run jarvis doctor",
            "run health check",
        }:
            return Plan(
                goal="Diagnose Jarvis setup.",
                actions=[PlannedAction("jarvis_doctor", {}, "The user asked to diagnose Jarvis setup.")],
            )

        # Real gap found live 2026-07-10 (round 44), same class as the
        # preference_match fix above: unbounded lazy captures cause quadratic
        # backtracking on long non-matching input. Bounded to generous limits
        # (name/trigger) no real skill name/trigger description would need.
        save_skill_match = re.search(
            r"^save\s+skill\s+(?P<name>.{1,100}?)\s+(?:when|trigger)\s+(?P<trigger>.{1,300}?)\s+(?:do|procedure|:)\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if save_skill_match:
            return Plan(
                goal=f"Save skill {save_skill_match.group('name').strip()}.",
                actions=[
                    PlannedAction(
                        "save_skill",
                        {
                            "name": save_skill_match.group("name").strip(),
                            "trigger": save_skill_match.group("trigger").strip(),
                            "body": save_skill_match.group("body").strip(),
                        },
                        "The user asked to save a reusable skill.",
                    )
                ],
            )

        linked_skill_extract_match = re.search(
            r"^(?:extract|preview|propose)\s+(?:linked\s+)?skills?(?:\s+from)?\s*(?P<document>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if linked_skill_extract_match and (
            "http" in linked_skill_extract_match.group("document").lower()
            or "hermes" in linked_skill_extract_match.group("document").lower()
            or "openhuman" in linked_skill_extract_match.group("document").lower()
            or "document" in low
            or "links" in low
        ):
            return Plan(
                goal="Extract link-inspired skill candidates.",
                actions=[
                    PlannedAction(
                        "extract_linked_skills",
                        {"document": linked_skill_extract_match.group("document").strip()},
                        "The user asked to extract reusable skills from linked documents.",
                    )
                ],
            )

        linked_skill_install_match = re.search(
            r"^(?:install|implement|save)\s+(?:the\s+)?(?:linked\s+)?skills?(?:\s+from)?\s*(?P<document>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if linked_skill_install_match and (
            "http" in linked_skill_install_match.group("document").lower()
            or "hermes" in linked_skill_install_match.group("document").lower()
            or "openhuman" in linked_skill_install_match.group("document").lower()
            or "document" in low
            or "links" in low
        ):
            return Plan(
                goal="Install link-inspired skills.",
                actions=[
                    PlannedAction(
                        "install_linked_skills",
                        {"document": linked_skill_install_match.group("document").strip()},
                        "The user asked to save reusable skills extracted from linked documents.",
                    )
                ],
            )

        skill_match_preview_match = re.search(
            r"^(?:skill match preview|match skills?|which skill(?:s)?(?: should Jarvis use)?|preview skill match)(?:\s*:?\s*(?P<request>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if skill_match_preview_match:
            return Plan(
                goal="Preview matching saved skills.",
                actions=[
                    PlannedAction(
                        "skill_match_preview",
                        {"request": skill_match_preview_match.group("request").strip()},
                        "The user asked to preview which saved skills apply before acting.",
                    )
                ],
            )

        # Real gap found live 2026-07-10 (round 43), same class as the notes
        # fix above: a trailing-only politeness strip is safe here (no-op when
        # nothing to strip) and fixes "show my recent skills please"-style
        # phrasing that previously fell through to chat.
        skill_list_command = _strip_trailing_politeness_suffix_only(low)
        if skill_list_command in {
            "skills",
            "skills please",
            "skill list",
            "skill list please",
            "skills list",
            "skills list please",
            "list skills",
            "list skills please",
            "show skills",
            "show skills please",
            "show my skills",
            "show my skills please",
            "show latest skills",
            "show latest skills please",
            # Real gap found live 2026-07-10, same class as the round-39/40
            # list-command qualifier-word bug: "show my skills" worked but
            # "my skills"/"list my skills" (bare or "list"-prefixed) fell
            # through to chat, and no "recent" qualifier was recognized.
            "my skills",
            "list my skills",
            "list my skills please",
            "recent skills",
            "show my recent skills",
            "list my recent skills",
            "my recent skills",
            "saved skills",
            "saved skills please",
            "show saved skills",
            "show saved skills please",
            # Korean parity (Fable checkpoint plan item B2, 2026-07-10).
            "내 스킬 보여줘",
            "스킬 보여줘",
            "스킬 목록",
            "내 스킬",
        }:
            return Plan(
                goal="List saved skills.",
                actions=[PlannedAction("list_skills", {}, "The user asked to list skills.")],
            )

        skill_search_match = re.search(r"^(?:search|find)\s+skills?\s+(?:for|about)?\s*(?P<query>.+)$", text, re.IGNORECASE)
        if skill_search_match:
            query = _strip_trailing_politeness(skill_search_match.group("query")).strip()
            return Plan(
                goal="Search saved skills.",
                actions=[
                    PlannedAction(
                        "search_skills",
                        {"query": query},
                        "The user asked to search skills.",
                    )
                ],
            )

        get_skill_match = re.search(r"^(?:get|read|show)\s+skill\s+(?P<name>.+)$", text, re.IGNORECASE)
        if get_skill_match:
            return Plan(
                goal="Read a saved skill.",
                actions=[
                    PlannedAction(
                        "get_skill",
                        {"name": get_skill_match.group("name").strip()},
                        "The user asked to read a saved skill.",
                    )
                ],
            )

        delete_skill_match = re.search(r"^(?:delete|remove)\s+skill\s+(?P<name>.+)$", text, re.IGNORECASE)
        if delete_skill_match:
            return Plan(
                goal="Delete a saved skill.",
                actions=[
                    PlannedAction(
                        "delete_skill",
                        {"name": delete_skill_match.group("name").strip()},
                        "The user asked to delete a saved skill.",
                    )
                ],
            )

        list_match = re.search(r"^(?:(?:list|show)\s+)?(?:me\s+|my\s+)?(?:files|folder|directory)(?:\s+in)?\s*(?P<directory>.*)$", text, re.IGNORECASE)
        if list_match:
            directory = _clean_file_arg(list_match.group("directory")) or "."
            return Plan(
                goal=f"List files in {directory}.",
                actions=[PlannedAction("list_files", {"directory": directory}, "The user asked to list files.")],
            )

        canonical_image_ocr_match = re.fullmatch(
            r"ocr_image[ \t]+path=(?P<path>\S+)",
            text,
            re.IGNORECASE,
        )
        if canonical_image_ocr_match:
            return Plan(
                goal="Read text from an explicit local image with on-device OCR.",
                actions=[
                    PlannedAction(
                        "ocr_image",
                        {"path": canonical_image_ocr_match.group("path")},
                        "The user supplied the canonical OCR command with one explicitly bound path.",
                    )
                ],
            )

        image_ocr_match = re.search(
            r"^(?:ocr (?:this\s+)?(?:document photo|receipt photo|screenshot|picture|photo|image|receipt)|read text from (?:this\s+)?(?:document photo|receipt photo|screenshot|picture|photo|image|receipt)|extract text from (?:this\s+)?(?:document photo|receipt photo|screenshot|picture|photo|image|receipt))\s*:?\s*(?P<body>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if image_ocr_match:
            body = image_ocr_match.group("body").strip()
            if re.search(r"\.(?:png|jpe?g|gif|bmp|tiff?|heic|webp)\b", body, re.IGNORECASE):
                return Plan(
                    goal="Read text from an explicit local image with on-device OCR.",
                    actions=[
                        PlannedAction(
                            "ocr_image",
                            {"path": body},
                            "The user supplied an explicit image path for local OCR.",
                        )
                    ],
                )
            return Plan(
                goal="Plan safe photo/document intake before reading image contents.",
                actions=[
                    PlannedAction(
                        "photo_document_intake_plan",
                        {"request": body},
                        "The user asked about image text without an explicit image path, so Jarvis should produce a safe intake plan first.",
                    )
                ],
            )

        read_match = re.search(
            r"^(?:read|open)\s+(?!(?:text\s+from\s+)?(?:this\s+|a\s+|the\s+)?(?:document photo|receipt photo|screenshot|picture|photo|image|receipt)\b)(?:file\s+)?(?P<path>.+)$",
            text,
            re.IGNORECASE,
        )
        if read_match and (re.search(r"\bfile\b", low) or "/" in text or "." in read_match.group("path")):
            path = _clean_file_arg(read_match.group("path"))
            return Plan(
                goal=f"Read file {path}.",
                actions=[PlannedAction("read_text_file", {"path": path}, "The user asked to read a text file.")],
            )

        find_match = None
        for file_search_pattern in (
            # Real gap found live 2026-07-10: "find file budget and then send
            # it to john" (a compound sentence) swallowed the whole second
            # clause into the search pattern instead of stopping at "budget"
            # -- and silently dropped the "send it to john" intent. Applied
            # to all four alternatives below.
            r"^(?:find|search)\s{1,10}(?:file|files)\s{1,10}(?P<pattern>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?(?:\s{1,10}in\s{1,10}(?P<root>.+?))?(?:\s{1,10}(?:please|pls|thanks|thank you))?$",
            r"^(?:find|search)\s{1,10}(?:for\s{1,10})?(?P<pattern>.+?)\s{1,10}(?:file|files|documents?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?(?:\s{1,10}in\s{1,10}(?P<root>.+?))?(?:\s{1,10}(?:please|pls|thanks|thank you))?$",
            r"^(?:find|search)\s{1,10}(?:for\s{1,10})?(?:file|files|documents?)\s{1,10}(?:named|called|matching|containing|with|for)\s{1,10}(?P<pattern>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?(?:\s{1,10}in\s{1,10}(?P<root>.+?))?(?:\s{1,10}(?:please|pls|thanks|thank you))?$",
            # Real gap found live 2026-07-10, same privacy-relevant misroute
            # class as the round-21/26 notes-search and this round's
            # task-search word-order fixes: "search for the/a file about X" /
            # "find the/a file about X" (article BEFORE the noun) fell
            # through to the generic public web-search fallback instead of
            # searching local files. Deliberately do not accept "my files"
            # here: that implies a broad personal scope and should not become
            # an automatic local file search.
            r"^(?:find|search)\s{1,10}(?:for\s{1,10})?(?:the\s{1,10}|a\s{1,10})?(?:file|files|documents?)\s{1,10}(?:about|named|called|matching|containing|with|for)\s{1,10}(?P<pattern>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?(?:\s{1,10}in\s{1,10}(?P<root>.+?))?(?:\s{1,10}(?:please|pls|thanks|thank you))?$",
        ):
            find_match = re.search(file_search_pattern, text, re.IGNORECASE)
            if find_match:
                break
        if find_match:
            pattern = _clean_file_search_pattern(find_match.group("pattern"))
            if not pattern:
                return Plan(goal="", actions=[])
            return Plan(
                goal="Find matching files.",
                actions=[
                    PlannedAction(
                        "find_files",
                        {
                            "pattern": pattern,
                            "root": _clean_file_arg(find_match.group("root") or ".") or ".",
                        },
                        "The user asked to find files.",
                    )
                ],
            )

        # Real gap found live 2026-07-10 (round 44), same class as the
        # preference_match fix above: unbounded lazy capture causes quadratic
        # backtracking on long non-matching input. Bounded to 300 chars --
        # generous even for a deeply nested real filesystem path.
        write_match = re.search(
            r"^(?:write|create|make)\s+(?:a\s+|an\s+)?(?:file\s+)?(?:called\s+)?(?P<path>.{1,300}?)\s+(?:with|as|:)\s*(?P<content>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if write_match:
            return Plan(
                goal=f"Write file {write_match.group('path').strip()}.",
                actions=[
                    PlannedAction(
                        "write_text_file",
                        {
                            "path": write_match.group("path").strip(),
                            "content": write_match.group("content").strip(),
                            "overwrite": False,
                        },
                        "The user asked to write a file.",
                    )
                ],
            )

        command_match = re.search(
            r"^(?:run|execute)\s+(?:shell\s+)?(?:command\s+)?(?P<command>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if command_match:
            return Plan(
                goal="Run shell command.",
                actions=[
                    PlannedAction(
                        "run_shell_command",
                        {"command": command_match.group("command").strip()},
                        "The user asked to run a local command.",
                    )
                ],
            )

        voice_setup_command = _strip_trailing_politeness(low_command)
        if voice_setup_command in {
            "voice setup",
            "voice setup check",
            "voice input setup",
            "voice input setup check",
            "voice input readiness",
            "voice input check",
            "voice check",
            "microphone setup",
            "microphone setup check",
            "microphone readiness",
            "microphone check",
            "mic check",
            "asr setup check",
            "asr check",
            "speech setup check",
            "speech check",
            "speech input readiness",
            "is voice input ready",
            "is microphone input ready",
            "is speech input ready",
            "is asr ready",
            "can jarvis listen yet",
            "can jarvis use the microphone",
            "can jarvis use microphone",
        }:
            return Plan(
                goal="Check voice input readiness.",
                actions=[
                    PlannedAction(
                        "voice_setup_check",
                        {},
                        "The user asked to inspect voice and ASR setup without starting capture.",
                    )
                ],
            )

        capture_privacy_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?)?"
            r"(?:voice capture privacy packet|voice privacy packet|microphone privacy packet|voice capture privacy|voice privacy|microphone privacy)(?:\s*:?\s*(?P<mode>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if capture_privacy_match:
            mode = (capture_privacy_match.group("mode") or "").strip()
            if re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", mode, re.IGNORECASE):
                mode = ""
            return Plan(
                goal="Preview visible push-to-talk microphone privacy boundary.",
                actions=[
                    PlannedAction(
                        "voice_capture_privacy_packet",
                        {"mode": mode or "browser-push-to-talk"},
                        "The user asked to inspect the microphone privacy boundary before browser capture.",
                    )
                ],
            )

        natural_voice_command = voice_setup_command.rstrip("?")
        natural_voice_plan_phrases = {
            "can i talk to jarvis",
            "can i speak to jarvis",
            "can i use voice with jarvis",
            "can i use the microphone with jarvis",
            "how do i talk to jarvis",
            "how do i speak to jarvis",
            "how do i use voice with jarvis",
            "start voice input",
            "begin voice input",
            "start listening",
            "listen to me",
            "start recording voice",
            "begin recording voice",
            "turn on microphone",
            "turn on the microphone",
            "open microphone",
            "open the microphone",
        }
        natural_voice_plan_match = re.search(
            r"^(?:how does (?:voice input|microphone input|spoken command|spoken input) work|how can (?:jarvis|i) (?:listen|use (?:the )?microphone|take voice input)|what(?:'s| is) the (?:voice input|microphone input|spoken command) plan)$",
            natural_voice_command,
            re.IGNORECASE,
        )
        if natural_voice_plan_match or natural_voice_command in natural_voice_plan_phrases:
            return Plan(
                goal="Plan safe voice input.",
                actions=[
                    PlannedAction(
                        "voice_input_plan",
                        {"mode": ""},
                        "The user asked how voice input or microphone command intake works without starting capture.",
                    )
                ],
            )

        native_mic_gate_match = re.search(
            r"^(?:voice native microphone gate|native microphone gate|native mic gate|voice native mic gate)(?:\s*:?\s*(?P<spec>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if native_mic_gate_match:
            spec = (native_mic_gate_match.group("spec") or "").strip()
            args: dict[str, Any] = {}
            for match in re.finditer(r"(?P<key>mode|permission_receipt_id|receipt_id|visible_state|max_duration_seconds)\s*=\s*(?P<value>[^\s;]+)", spec, re.IGNORECASE):
                args[match.group("key").lower()] = match.group("value").strip()
            if spec and "mode" not in args and "=" not in spec:
                args["mode"] = spec
            return Plan(
                goal="Gate future native microphone capture behind visible permission proof.",
                actions=[
                    PlannedAction(
                        "voice_native_microphone_gate_packet",
                        args,
                        "The user asked to inspect the native microphone capture gate without requesting microphone access or recording audio.",
                    )
                ],
            )

        photo_intake_match = re.search(
            r"^(?:photo/document intake plan|photo document intake plan|photo intake plan|image intake plan|telegram photo intake plan|document photo intake plan|document intake plan|receipt photo plan|screenshot intake plan|photo receipt plan|review photo plan)(?:\s*:?\s*(?P<path>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if photo_intake_match:
            return Plan(
                goal="Plan safe phone photo/document intake.",
                actions=[
                    PlannedAction(
                        "photo_document_intake_plan",
                        {"path": (photo_intake_match.group("path") or "").strip()},
                        "The user asked for a safe photo/document intake plan before OCR or durable writes.",
                    )
                ],
            )

        natural_photo_intake_match = re.search(
            r"^(?:(?:review|summarize|read|scan|process|describe|extract facts from)\s+(?:this\s+|a\s+|the\s+)?(?:document photo|receipt photo|screenshot|picture|photo|image|receipt)|what does\s+(?:this\s+|a\s+|the\s+)?(?:document photo|receipt photo|screenshot|picture|photo|image|receipt)\s+say|what(?:'s| is)\s+in\s+(?:this\s+|a\s+|the\s+)?(?:document photo|receipt photo|screenshot|picture|photo|image|receipt))(?:\s*:?\s*(?P<path>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if natural_photo_intake_match:
            return Plan(
                goal="Plan safe photo/document intake before reading or saving content.",
                actions=[
                    PlannedAction(
                        "photo_document_intake_plan",
                        {"path": (natural_photo_intake_match.group("path") or "").strip()},
                        "The user used a natural photo/document intake phrase; Jarvis should keep it read-only until an explicit image read and write confirmation happen.",
                    )
                ],
            )

        file_transcription_match = re.search(
            r"^(?:voice file transcription plan|audio file transcription plan|audio transcription plan|file transcription plan|voice note transcription plan|voice note plan|telegram voice note plan|voice memo transcription plan|voice memo plan)(?:\s*:?\s*(?P<path>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if file_transcription_match:
            return Plan(
                goal="Plan safe user-supplied audio-file transcription.",
                actions=[
                    PlannedAction(
                        "voice_file_transcription_plan",
                        {"path": (file_transcription_match.group("path") or "").strip()},
                        "The user asked for a safe file-import transcription plan without recording or decoding audio.",
                    )
                ],
            )

        audio_file_gate_match = re.search(
            r"^(?:voice audio file gate|audio file gate|voice transcription gate|audio transcription gate|voice note gate|telegram voice note gate|voice memo gate)\s*:?\s*(?P<path>.+?)(?:\s+consent=(?P<consent>true|false|yes|no|1|0))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if audio_file_gate_match:
            return Plan(
                goal="Gate a user-supplied audio file before any transcription.",
                actions=[
                    PlannedAction(
                        "voice_audio_file_gate_packet",
                        {
                            "path": audio_file_gate_match.group("path").strip(),
                            "consent": (audio_file_gate_match.group("consent") or "").strip(),
                        },
                        "The user asked to check an audio file before any future transcription can read it.",
                    )
                ],
            )

        audio_file_transcription_match = re.search(
            r"^(?:voice audio file transcribe|voice audio transcription preview|voice file transcribe|transcribe voice audio file|transcribe audio file|voice note transcribe|telegram voice note transcribe|voice memo transcribe|transcribe voice note|transcribe voice memo)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if audio_file_transcription_match:
            body = audio_file_transcription_match.group("body").strip()
            consent_match = re.search(r"\bconsent=(?P<consent>true|false|yes|no|1|0|approved|confirmed|reviewed)\b", body, re.IGNORECASE)
            receipt_match = re.search(r"\breceipt_id=(?P<receipt_id>\S+)", body, re.IGNORECASE)
            path = re.sub(r"\s+\b(?:consent|receipt_id)=\S+", "", body, flags=re.IGNORECASE).strip()
            return Plan(
                goal="Transcribe a user-supplied audio file into a temporary preview after approval.",
                actions=[
                    PlannedAction(
                        "voice_audio_file_transcription_preview",
                        {
                            "path": path,
                            "consent": (consent_match.group("consent") if consent_match else "").strip(),
                            "receipt_id": (receipt_match.group("receipt_id") if receipt_match else "").strip(),
                        },
                        "The user asked to transcribe an explicit audio file; this reads personal audio and must be approval-gated.",
                    )
                ],
            )

        send_confirmed_transcript_match = re.search(
            r"^(?:send confirmed transcript|route confirmed transcript|route voice transcript)\s*:?\s*(?P<transcript>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if send_confirmed_transcript_match:
            return Plan(
                goal="Gate a confirmed voice transcript before normal routing.",
                actions=[
                    PlannedAction(
                        "voice_route_gate_packet",
                        {
                            "transcript": send_confirmed_transcript_match.group("transcript").strip(),
                            "confirmed": "true",
                        },
                        "The user asked to route a confirmed transcript; Jarvis must gate it before any action.",
                    )
                ],
            )

        transcript_review_match = re.search(
            r"^(?:voice transcript review|transcript review|review transcript|spoken command review)\s*:?\s*(?P<transcript>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if transcript_review_match:
            return Plan(
                goal="Review a supplied voice transcript.",
                actions=[
                    PlannedAction(
                        "voice_transcript_review",
                        {"transcript": transcript_review_match.group("transcript").strip()},
                        "The user asked to preview a voice transcript before routing it into actions.",
                    )
                ],
            )

        transcript_receipt_match = re.search(
            r"^(?:voice confirmation receipt|voice transcript receipt|speech confirmation receipt|spoken command receipt)\s*:?\s*(?P<transcript>.+?)(?:\s+confirmed=(?P<confirmed>true|false|yes|no|1|0))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if transcript_receipt_match:
            return Plan(
                goal="Build an auditable voice transcript confirmation receipt.",
                actions=[
                    PlannedAction(
                        "voice_confirmation_receipt",
                        {
                            "transcript": transcript_receipt_match.group("transcript").strip(),
                            "confirmed": (transcript_receipt_match.group("confirmed") or "").strip(),
                        },
                        "The user asked for an auditable confirmation receipt before routing a supplied voice transcript.",
                    )
                ],
            )

        voice_confirmation_audit_match = re.search(
            r"^(?:voice confirmation audit ledger|voice transcript audit ledger|speech confirmation audit ledger|spoken confirmation audit ledger|voice receipt audit ledger)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_confirmation_audit_match:
            voice_fields = _parse_voice_packet_spec(voice_confirmation_audit_match.group("body").strip())
            return Plan(
                goal="Audit supplied voice confirmation receipt proof before command intake.",
                actions=[
                    PlannedAction(
                        "voice_confirmation_audit_ledger",
                        {
                            "transcript": voice_fields["transcript"],
                            "confirmed": voice_fields["confirmed"],
                            "mode": voice_fields["mode"],
                            "privacy_receipt_id": voice_fields["privacy_receipt_id"],
                            "receipt_id": voice_fields["receipt_id"],
                            "receipt_nonce": voice_fields["receipt_nonce"],
                        },
                        "The user asked to require externally supplied voice receipt proof before a spoken transcript can count as command-intake ready.",
                    )
                ],
            )

        voice_route_gate_match = re.search(
            r"^(?:voice route gate|voice routing gate|speech route gate|spoken route gate)\s*:?\s*(?P<transcript>.+?)(?:\s+confirmed=(?P<confirmed>true|false|yes|no|1|0))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_route_gate_match:
            return Plan(
                goal="Gate a supplied voice transcript before normal routing.",
                actions=[
                    PlannedAction(
                        "voice_route_gate_packet",
                        {
                            "transcript": voice_route_gate_match.group("transcript").strip(),
                            "confirmed": (voice_route_gate_match.group("confirmed") or "").strip(),
                        },
                        "The user asked to check whether a supplied voice transcript may enter normal routing.",
                    )
                ],
            )

        voice_route_proof_match = re.search(
            r"^(?:voice route proof bundle|voice proof bundle|speech route proof bundle|spoken route proof bundle)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_route_proof_match:
            voice_fields = _parse_voice_packet_spec(voice_route_proof_match.group("body").strip())
            return Plan(
                goal="Bundle voice route proof before normal routing.",
                actions=[
                    PlannedAction(
                        "voice_route_proof_bundle",
                        {
                            "transcript": voice_fields["transcript"],
                            "confirmed": voice_fields["confirmed"],
                            "mode": voice_fields["mode"],
                            "privacy_receipt_id": voice_fields["privacy_receipt_id"],
                            "receipt_id": voice_fields["receipt_id"],
                            "receipt_nonce": voice_fields["receipt_nonce"],
                        },
                        "The user asked to prove the spoken transcript chain before it can enter normal routing.",
                    )
                ],
            )

        voice_runtime_bridge_match = re.search(
            r"^(?:voice runtime bridge|voice command bridge|voice command intake bridge|spoken runtime bridge|speech runtime bridge)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_runtime_bridge_match:
            voice_fields = _parse_voice_packet_spec(voice_runtime_bridge_match.group("body").strip())
            return Plan(
                goal="Bridge a confirmed voice transcript into command intake proof.",
                actions=[
                    PlannedAction(
                        "voice_runtime_bridge_packet",
                        {
                            "transcript": voice_fields["transcript"],
                            "confirmed": voice_fields["confirmed"],
                            "mode": voice_fields["mode"],
                            "privacy_receipt_id": voice_fields["privacy_receipt_id"],
                            "receipt_id": voice_fields["receipt_id"],
                            "receipt_nonce": voice_fields["receipt_nonce"],
                        },
                        "The user asked to bridge a confirmed spoken transcript into command intake without executing it.",
                    )
                ],
            )

        voice_command_cockpit_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?)?"
            r"(?:voice command cockpit|voice cockpit|speech command cockpit|spoken command cockpit)(?:\s*:?\s*(?P<body>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_command_cockpit_match:
            body = (voice_command_cockpit_match.group("body") or "").strip()
            if re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", body, re.IGNORECASE):
                body = ""
            voice_fields = _parse_voice_packet_spec(body)
            return Plan(
                goal="Consolidate voice command proof before command intake.",
                actions=[
                    PlannedAction(
                        "voice_command_cockpit",
                        {
                            "transcript": voice_fields["transcript"],
                            "confirmed": voice_fields["confirmed"],
                            "mode": voice_fields["mode"],
                            "privacy_receipt_id": voice_fields["privacy_receipt_id"],
                            "receipt_id": voice_fields["receipt_id"],
                            "receipt_nonce": voice_fields["receipt_nonce"],
                        },
                        "The user asked for the voice command cockpit before a spoken transcript enters command intake.",
                    )
                ],
            )

        voice_action_audit_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?)?"
            r"(?:voice action audit|voice command audit|voice pre-action audit|speech action audit|spoken action audit|voice action audit packet)(?:\s*:?\s*(?P<body>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_action_audit_match:
            body = (voice_action_audit_match.group("body") or "").strip()
            if re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", body, re.IGNORECASE):
                body = ""
            voice_fields = _parse_voice_packet_spec(body)
            return Plan(
                goal="Audit a confirmed voice transcript before command intake.",
                actions=[
                    PlannedAction(
                        "voice_action_audit_packet",
                        {
                            "transcript": voice_fields["transcript"],
                            "confirmed": voice_fields["confirmed"],
                            "mode": voice_fields["mode"],
                            "privacy_receipt_id": voice_fields["privacy_receipt_id"],
                            "receipt_id": voice_fields["receipt_id"],
                            "receipt_nonce": voice_fields["receipt_nonce"],
                        },
                        "The user asked for the final spoken-action audit before a confirmed transcript can enter command intake.",
                    )
                ],
            )

        voice_execution_handoff_match = re.search(
            r"^(?:voice execution handoff|voice handoff packet|speech execution handoff|spoken execution handoff|voice command handoff)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_execution_handoff_match:
            voice_fields = _parse_voice_packet_spec(voice_execution_handoff_match.group("body").strip())
            return Plan(
                goal="Package a confirmed voice transcript for command-intake handoff.",
                actions=[
                    PlannedAction(
                        "voice_execution_handoff_packet",
                        {
                            "transcript": voice_fields["transcript"],
                            "confirmed": voice_fields["confirmed"],
                            "mode": voice_fields["mode"],
                            "privacy_receipt_id": voice_fields["privacy_receipt_id"],
                            "receipt_id": voice_fields["receipt_id"],
                            "receipt_nonce": voice_fields["receipt_nonce"],
                        },
                        "The user asked for the final spoken-command handoff after voice action audit before command intake.",
                    )
                ],
            )

        voice_post_run_closure_match = re.search(
            r"^(?:voice post-run closure|voice post run closure|speech post-run closure|spoken post-run closure|voice closure packet)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_post_run_closure_match:
            body = voice_post_run_closure_match.group("body").strip()
            voice_fields = _parse_voice_packet_spec(body)
            return Plan(
                goal="Close post-run proof for a confirmed voice command.",
                actions=[
                    PlannedAction(
                        "voice_post_run_closure_packet",
                        {
                            "transcript": voice_fields["transcript"],
                            "confirmed": voice_fields["confirmed"],
                            "mode": voice_fields["mode"],
                            "privacy_receipt_id": voice_fields["privacy_receipt_id"],
                            "receipt_id": voice_fields["receipt_id"],
                            "receipt_nonce": voice_fields["receipt_nonce"],
                            "evidence": voice_fields["evidence"],
                            "verification_receipt_sha256": voice_fields["verification_receipt_sha256"],
                            "execution_health_sha256": voice_fields["execution_health_sha256"],
                            "execution_audit_sha256": voice_fields["execution_audit_sha256"],
                            "after_action_learning_sha256": voice_fields["after_action_learning_sha256"],
                        },
                        "The user asked to close verification, audit, health, and learning proof after a spoken command.",
                    )
                ],
            )

        voice_cycle_ledger_match = re.search(
            r"^(?:voice cycle ledger|speech cycle ledger|spoken command cycle ledger|voice proof ledger|spoken proof ledger)\s*:?\s*(?P<body>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_cycle_ledger_match:
            body = voice_cycle_ledger_match.group("body").strip()
            voice_fields = _parse_voice_packet_spec(body)
            return Plan(
                goal="Bind the full confirmed-speech cycle into a prior-command proof ledger.",
                actions=[
                    PlannedAction(
                        "voice_cycle_ledger",
                        {
                            "transcript": voice_fields["transcript"],
                            "confirmed": voice_fields["confirmed"],
                            "mode": voice_fields["mode"],
                            "privacy_receipt_id": voice_fields["privacy_receipt_id"],
                            "receipt_id": voice_fields["receipt_id"],
                            "receipt_nonce": voice_fields["receipt_nonce"],
                            "evidence": voice_fields["evidence"],
                            "verification_receipt_sha256": voice_fields["verification_receipt_sha256"],
                            "execution_health_sha256": voice_fields["execution_health_sha256"],
                            "execution_audit_sha256": voice_fields["execution_audit_sha256"],
                            "after_action_learning_sha256": voice_fields["after_action_learning_sha256"],
                        },
                        "The user asked for the full voice lifecycle ledger before a new spoken command review starts.",
                    )
                ],
            )

        voice_stop_intent_match = re.search(
            r"^(?:voice stop intent|voice stop packet|speech stop packet|spoken stop packet|voice cancel intent|voice rerecord intent|voice stop|stop voice input|cancel voice input|rerecord voice input|stop listening|cancel listening|rerecord this command)\s*:?\s*(?P<phrase>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_stop_intent_match:
            phrase = _strip_trailing_politeness((voice_stop_intent_match.group("phrase") or "").strip())
            if re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", phrase, re.IGNORECASE):
                phrase = ""
            return Plan(
                goal="Review a non-destructive voice stop intent.",
                actions=[
                    PlannedAction(
                        "voice_stop_intent_packet",
                        {"phrase": phrase},
                        "The user asked for a read-only packet proving stop/cancel/rerecord speech does not become command execution.",
                    )
                ],
            )

        transcript_confirmation_match = re.search(
            r"^(?:voice confirmation|voice confirmation packet|confirm voice transcript|spoken command confirmation)\s*:?\s*(?P<transcript>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if transcript_confirmation_match:
            return Plan(
                goal="Build a voice transcript confirmation packet.",
                actions=[
                    PlannedAction(
                        "voice_confirmation_packet",
                        {"transcript": transcript_confirmation_match.group("transcript").strip()},
                        "The user asked for a safe confirmation packet before acting on a voice transcript.",
                    )
                ],
            )

        voice_lifecycle_match = re.search(
            r"^(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?)?"
            r"(?:voice command lifecycle|voice lifecycle|speech command lifecycle|spoken command lifecycle|voice proof chain|spoken proof chain|voice command proof chain|spoken command proof chain|what(?:'s| is) the voice proof chain|what(?:'s| is) the spoken proof chain|what happens after i speak to jarvis)(?:\s*:?\s*(?P<transcript>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_lifecycle_match:
            transcript = (voice_lifecycle_match.group("transcript") or "").strip()
            if re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", transcript, re.IGNORECASE):
                transcript = ""
            return Plan(
                goal="Show the safe voice command lifecycle.",
                actions=[
                    PlannedAction(
                        "voice_command_lifecycle",
                        {"transcript": transcript},
                        "The user asked for the safe end-to-end path from spoken command to approval-gated action.",
                    )
                ],
            )

        if "system info" in low or "computer info" in low or "system status" in low:
            return Plan(
                goal="Read system information.",
                actions=[PlannedAction("system_info", {}, "The user asked for system information.")],
            )

        if low in {"setup", "setup status", "show setup", "show setup status"} or "setup check" in low or "dependency check" in low:
            return Plan(
                goal="Check Jarvis setup.",
                actions=[PlannedAction("setup_check", {}, "The user asked to check setup/dependencies.")],
            )

        if (
            "integration status" in low
            or "personal integrations" in low
            or "personal integration status" in low
            or "what personal integrations are active" in low
        ):
            return Plan(
                goal="Show integration status.",
                actions=[PlannedAction("integration_status", {}, "The user asked about personal integrations.")],
            )

        integration_readiness_match = re.search(
            r"^(?:integration readiness|personal integration readiness|connector readiness|integration readiness report|which integration should jarvis build next)\s*:?\s*(?P<connector>.+)?$",
            text,
            re.IGNORECASE,
        )
        if integration_readiness_match:
            connector = (integration_readiness_match.group("connector") or "").strip()
            return Plan(
                goal="Rank personal integration readiness.",
                actions=[
                    PlannedAction(
                        "integration_readiness_report",
                        {"connector": connector},
                        "The user asked which personal connector is safest to migrate next.",
                    )
                ],
            )

        integration_matrix_match = re.search(
            r"^(?:integration execution matrix|connector execution matrix|personal integration execution matrix|personal connector execution matrix|connector rollout matrix|personal integration rollout matrix)\s*:?\s*(?P<connector>.+)?$",
            text,
            re.IGNORECASE,
        )
        if integration_matrix_match:
            connector = (integration_matrix_match.group("connector") or "").strip()
            return Plan(
                goal="Show personal connector execution lanes and proof gates.",
                actions=[
                    PlannedAction(
                        "integration_execution_matrix",
                        {"connector": connector},
                        "The user asked for the connector execution matrix before enabling personal integrations.",
                    )
                ],
            )

        legacy_connector_audit_match = re.search(
            r"^(?:legacy connector migration audit|old jarvis connector migration audit|old jarvis migration audit|legacy integration audit|old jarvis migration packet|legacy connector packet)\s*:?\s*(?P<scope>.+)?$",
            text,
            re.IGNORECASE,
        )
        if legacy_connector_audit_match:
            scope = (legacy_connector_audit_match.group("scope") or "browser calendar email").strip()
            return Plan(
                goal="Audit old Jarvis connector migration readiness.",
                actions=[
                    PlannedAction(
                        "legacy_connector_migration_audit",
                        {"scope": scope},
                        "The user asked to continue old Jarvis browser/calendar/email migration safely.",
                    )
                ],
            )

        integration_adapter_manifest_match = re.search(
            r"^(?:integration adapter manifest|connector adapter manifest|personal integration adapter manifest|personal connector adapter manifest|adapter rollout manifest|connector stub manifest)\s*:?\s*(?P<connector>.+)?$",
            text,
            re.IGNORECASE,
        )
        if integration_adapter_manifest_match:
            connector = (integration_adapter_manifest_match.group("connector") or "").strip()
            return Plan(
                goal="Show disabled personal connector adapter stubs and enablement gates.",
                actions=[
                    PlannedAction(
                        "integration_adapter_manifest",
                        {"connector": connector},
                        "The user asked for the disabled adapter manifest before implementing personal connector adapters.",
                    )
                ],
            )

        integration_plan_match = re.search(
            r"^(?:integration migration plan|personal integration plan|connector migration plan|migrate integration|plan integration)\s*:?\s*(?P<connector>.+)?$",
            text,
            re.IGNORECASE,
        )
        if integration_plan_match:
            connector = (integration_plan_match.group("connector") or "calendar").strip()
            return Plan(
                goal="Plan a safe personal integration migration.",
                actions=[
                    PlannedAction(
                        "integration_migration_plan",
                        {"connector": connector},
                        "The user asked for a safe migration plan for a personal connector.",
                    )
                ],
            )

        integration_action_match = re.search(
            r"^(?:integration action preview|connector action preview|personal action preview)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<action>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_action_match:
            return Plan(
                goal="Preview personal connector action risk.",
                actions=[
                    PlannedAction(
                        "integration_action_preview",
                        {
                            "connector": integration_action_match.group("connector").strip(),
                            "action": integration_action_match.group("action").strip(),
                        },
                        "The user asked to classify a future personal connector action before any connector reads data or causes side effects.",
                    )
                ],
            )

        integration_scope_match = re.search(
            r"^(?:integration scope packet|connector scope packet|personal integration scope|personal connector scope)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_scope_match:
            scope = _parse_scope_packet_spec(integration_scope_match.group("spec").strip())
            return Plan(
                goal="Prepare scoped personal connector packet.",
                actions=[
                    PlannedAction(
                        "integration_scope_packet",
                        {
                            "connector": integration_scope_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                        },
                        "The user asked to scope a future personal connector request before any connector reads data or causes side effects.",
                    )
                ],
            )

        integration_dry_run_match = re.search(
            r"^(?:integration dry run contract|integration dry-run contract|connector dry run contract|connector dry-run contract|personal connector dry run|personal integration dry run)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_dry_run_match:
            scope = _parse_scope_packet_spec(integration_dry_run_match.group("spec").strip())
            verification_match = re.search(r"(?:^|;)\s*(?:verification|verify|post-run|post run)\s*:?\s*(?P<value>[^;]+)", integration_dry_run_match.group("spec"), re.IGNORECASE)
            return Plan(
                goal="Prepare personal connector dry-run contract.",
                actions=[
                    PlannedAction(
                        "integration_dry_run_contract",
                        {
                            "connector": integration_dry_run_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"] or (verification_match.group("value").strip() if verification_match else ""),
                        },
                        "The user asked to prepare a dry-run harness contract with metadata row contract proof before any personal connector reads data or causes side effects.",
                    )
                ],
            )

        integration_runbook_match = re.search(
            r"^(?:integration runbook|connector runbook|personal integration runbook|personal connector runbook|connector execution runbook)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_runbook_match:
            scope = _parse_scope_packet_spec(integration_runbook_match.group("spec").strip())
            return Plan(
                goal="Prepare personal connector execution runbook.",
                actions=[
                    PlannedAction(
                        "integration_runbook",
                        {
                            "connector": integration_runbook_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "rollback": scope["rollback"],
                        },
                        "The user asked to prepare a connector execution runbook with preflight, metadata row contract, approval, execution, verification, rollback, and audit boundaries.",
                    )
                ],
            )

        integration_promotion_match = re.search(
            r"^(?:integration promotion gate|connector promotion gate|personal integration promotion gate|connector implementation gate|personal connector implementation gate)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_promotion_match:
            scope = _parse_scope_packet_spec(integration_promotion_match.group("spec").strip())
            return Plan(
                goal="Check whether a personal connector can be promoted toward implementation.",
                actions=[
                    PlannedAction(
                        "integration_promotion_gate",
                        {
                            "connector": integration_promotion_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "rollback": scope["rollback"],
                            "tests": scope["tests"],
                            "audit": scope["audit"],
                        },
                        "The user asked to check whether a future personal connector is ready to move from design/runbook into implementation with metadata row contract proof.",
                    )
                ],
            )

        integration_implementation_match = re.search(
            r"^(?:integration implementation spec|connector implementation spec|personal integration implementation spec|personal connector implementation spec|connector tool spec|personal connector tool spec)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_implementation_match:
            scope = _parse_scope_packet_spec(integration_implementation_match.group("spec").strip())
            return Plan(
                goal="Draft a personal connector implementation spec.",
                actions=[
                    PlannedAction(
                        "integration_implementation_spec",
                        {
                            "connector": integration_implementation_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "rollback": scope["rollback"],
                            "tests": scope["tests"],
                            "audit": scope["audit"],
                        },
                        "The user asked to draft the exact ToolRegistry, planner, API, test, metadata row contract, audit, verification, and approval contract for a future personal connector.",
                    )
                ],
            )

        integration_preflight_match = re.search(
            r"^(?:integration preflight contract|connector preflight contract|personal integration preflight|personal connector preflight|connector enablement contract|integration enablement contract)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_preflight_match:
            scope = _parse_scope_packet_spec(integration_preflight_match.group("spec").strip())
            return Plan(
                goal="Draft a personal connector preflight contract.",
                actions=[
                    PlannedAction(
                        "integration_preflight_contract",
                        {
                            "connector": integration_preflight_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "rollback": scope["rollback"],
                            "tests": scope["tests"],
                            "audit": scope["audit"],
                        },
                        "The user asked to draft the final connector preflight contract with exact future arguments, metadata row contract proof, enablement gates, proof, and stop conditions.",
                    )
                ],
            )

        integration_enablement_match = re.search(
            r"^(?:integration enablement gate|connector enablement gate|personal integration enablement gate|personal connector enablement gate|integration final gate|connector final gate|personal connector final gate)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_enablement_match:
            scope = _parse_scope_packet_spec(integration_enablement_match.group("spec").strip())
            return Plan(
                goal="Check the final personal connector enablement gate.",
                actions=[
                    PlannedAction(
                        "integration_enablement_gate",
                        {
                            "connector": integration_enablement_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "rollback": scope["rollback"],
                            "tests": scope["tests"],
                            "audit": scope["audit"],
                            "acceptance": scope["acceptance"],
                        },
                        "The user asked to check the final harness enablement gate before a personal connector can move toward real use.",
                    )
                ],
            )

        integration_rehearsal_match = re.search(
            r"^(?:integration rehearsal receipt|connector rehearsal receipt|personal integration rehearsal|personal connector rehearsal|integration execution rehearsal|connector execution rehearsal)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_rehearsal_match:
            scope = _parse_scope_packet_spec(integration_rehearsal_match.group("spec").strip())
            return Plan(
                goal="Rehearse a personal connector execution envelope without enabling the connector.",
                actions=[
                    PlannedAction(
                        "integration_rehearsal_receipt",
                        {
                            "connector": integration_rehearsal_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "rollback": scope["rollback"],
                            "tests": scope["tests"],
                            "audit": scope["audit"],
                            "acceptance": scope["acceptance"],
                        },
                        "The user asked to rehearse a future personal connector execution envelope while keeping the connector disabled.",
                    )
                ],
            )

        integration_proof_bundle_match = re.search(
            r"^(?:integration proof bundle|connector proof bundle|personal integration proof bundle|personal connector proof bundle|integration implementation proof|connector implementation proof)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_proof_bundle_match:
            scope = _parse_scope_packet_spec(integration_proof_bundle_match.group("spec").strip())
            return Plan(
                goal="Bundle personal connector proof before implementation review.",
                actions=[
                    PlannedAction(
                        "integration_proof_bundle",
                        {
                            "connector": integration_proof_bundle_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "rollback": scope["rollback"],
                            "tests": scope["tests"],
                            "audit": scope["audit"],
                            "acceptance": scope["acceptance"],
                        },
                        "The user asked to bundle metadata preview, disabled adapter acceptance, metadata row contract, enablement, rehearsal, audit, verification, rollback, and stop-condition proof before connector implementation review.",
                    )
                ],
            )

        integration_implementation_review_match = re.search(
            r"^(?:integration implementation review|connector implementation review|personal integration implementation review|personal connector implementation review|integration code review gate|connector code review gate)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_implementation_review_match:
            scope = _parse_scope_packet_spec(integration_implementation_review_match.group("spec").strip())
            return Plan(
                goal="Gate personal connector implementation review readiness.",
                actions=[
                    PlannedAction(
                        "integration_implementation_review",
                        {
                            "connector": integration_implementation_review_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "rollback": scope["rollback"],
                            "tests": scope["tests"],
                            "audit": scope["audit"],
                            "acceptance": scope["acceptance"],
                            "status": scope["status"],
                        },
                        "The user asked to gate connector implementation review readiness across proof bundle, metadata row contract, preflight row-contract proof, implementation spec, status/API evidence, smoke tests, audit, verification, rollback, and stop conditions.",
                    )
                ],
            )

        integration_route_lock_match = re.search(
            r"^(?:integration route lock|connector route lock|personal integration route lock|personal connector route lock|connector routing lock|integration routing lock|connector nl route lock|integration nl route lock)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_route_lock_match:
            scope = _parse_scope_packet_spec(integration_route_lock_match.group("spec").strip())
            return Plan(
                goal="Gate personal connector route unlock readiness.",
                actions=[
                    PlannedAction(
                        "integration_route_lock",
                        {
                            "connector": integration_route_lock_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "rollback": scope["rollback"],
                            "tests": scope["tests"],
                            "audit": scope["audit"],
                            "acceptance": scope["acceptance"],
                            "status": scope["status"],
                            "expected_scope_hash": scope["expected_scope_hash"],
                        },
                        "The user asked to prove that a personal connector route remains locked until implementation review, status evidence, and exact-scope proof are present.",
                    )
                ],
            )

        integration_metadata_preview_match = re.search(
            r"^(?:integration metadata preview|connector metadata preview|personal integration metadata preview|personal connector metadata preview|metadata adapter preview|connector adapter preview)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_metadata_preview_match:
            scope = _parse_scope_packet_spec(integration_metadata_preview_match.group("spec").strip())
            return Plan(
                goal="Preview a personal connector metadata adapter without enabling the connector.",
                actions=[
                    PlannedAction(
                        "integration_metadata_preview",
                        {
                            "connector": integration_metadata_preview_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                            "tests": scope["tests"],
                            "audit": scope["audit"],
                            "acceptance": scope["acceptance"],
                        },
                        "The user asked to preview a future personal connector metadata adapter while keeping account access and real connector calls disabled.",
                    )
                ],
            )

        integration_adapter_acceptance_match = re.search(
            r"^(?:integration adapter acceptance|connector adapter acceptance|personal integration adapter acceptance|personal connector adapter acceptance|disabled adapter acceptance|connector acceptance)\s*:?\s*(?P<connector>[a-zA-Z -]+?)(?:\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_adapter_acceptance_match:
            spec = (integration_adapter_acceptance_match.group("spec") or "").strip()
            scope = _parse_scope_packet_spec(spec) if spec else {"target": "", "time_range": ""}
            return Plan(
                goal="Run disabled personal connector adapter acceptance cases without enabling the connector.",
                actions=[
                    PlannedAction(
                        "integration_adapter_acceptance",
                        {
                            "connector": integration_adapter_acceptance_match.group("connector").strip(),
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                        },
                        "The user asked to prove metadata-only adapter acceptance and blocked private/side-effect paths before connector enablement.",
                    )
                ],
            )

        integration_adapter_probe_match = re.search(
            r"^(?:integration adapter probe|connector adapter probe|personal integration adapter probe|personal connector adapter probe|disabled adapter probe|connector probe)\s*:?\s*(?P<connector>[a-zA-Z -]+?)\s*(?:->|:|,|\bfor\b)\s*(?P<spec>.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if integration_adapter_probe_match:
            scope = _parse_scope_packet_spec(integration_adapter_probe_match.group("spec").strip())
            return Plan(
                goal="Probe a disabled personal connector adapter boundary without enabling the connector.",
                actions=[
                    PlannedAction(
                        "integration_adapter_probe",
                        {
                            "connector": integration_adapter_probe_match.group("connector").strip(),
                            "action": scope["action"],
                            "target": scope["target"],
                            "time_range": scope["time_range"],
                            "data_level": scope["data_level"],
                            "verification": scope["verification"],
                        },
                        "The user asked to exercise a disabled connector adapter boundary with fake metadata and blocked private/side-effect paths.",
                    )
                ],
            )

        integration_contract_match = re.search(
            r"^(?:integration boundary contract|integration contract|connector boundary contract|connector contract|personal integration contract)\s*:?\s*(?P<connector>.+)?$",
            text,
            re.IGNORECASE,
        )
        if integration_contract_match:
            connector = (integration_contract_match.group("connector") or "calendar").strip()
            return Plan(
                goal="Show personal integration boundary contract.",
                actions=[
                    PlannedAction(
                        "integration_boundary_contract",
                        {"connector": connector},
                        "The user asked for the privacy and approval contract for a personal connector.",
                    )
                ],
            )

        if "open jarvis vault" in low or "open obsidian jarvis" in low:
            return Plan(
                goal="Open Jarvis Obsidian vault.",
                actions=[PlannedAction("open_jarvis_vault", {}, "The user asked to open Jarvis in Obsidian.")],
            )

        if (
            re.search(r"\b(?:remind me|set\s+(?:a\s+)?reminder|create\s+(?:a\s+)?reminder|add\s+(?:a\s+)?reminder)\b", low_command)
            and re.search(
                r"\b(?:when|once)\s+i\s+(?:get\s+home|arrive\s+home|come\s+home|get\s+to|arrive\s+at|arrive\s+to|reach|leave|am\s+leaving)\b",
                low_command,
            )
        ):
            return Plan(
                goal="Draft a location-triggered reminder.",
                actions=[
                    PlannedAction(
                        "location_reminder_draft",
                        {"text": raw_text or text},
                        "The user asked for a location-triggered reminder draft.",
                    )
                ],
            )

        reminder_match = re.search(
            # Real gap found live 2026-07-10: "remind me to call mom and then
            # set a timer for 10 minutes" (a compound sentence) swallowed the
            # whole second clause into the reminder title, writing a
            # confusing reminder titled "call mom and then set a timer for 10
            # minutes" instead of just "call mom" -- and silently dropping
            # the timer intent entirely. `set_reminder` (the timed-reminder
            # path just above) is unaffected since it always uses the raw
            # `text`, not this `title` group.
            r"^(?:remind me to|(?:set|create|add)\s{1,10}(?:a\s{1,10})?reminder(?:\s{1,10}to)?)\s{1,10}(?P<title>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?$",
            text,
            re.IGNORECASE,
        )
        if reminder_match and _has_reminder_due_hint(low):
            return Plan(
                goal="Set a reminder/timer.",
                actions=[
                    PlannedAction(
                        "set_reminder",
                        {"text": raw_text or text},
                        "The user asked for a timed reminder.",
                    )
                ],
            )
        if reminder_match:
            title = _raw_free_text_after_prefix(
                raw_text,
                r"^(?:remind\s+me\s+to|(?:set|create|add)\s+(?:a\s+)?reminder(?:\s+to)?)\s+",
            ) or reminder_match.group("title").strip()
            return Plan(
                goal="Create reminder.",
                actions=[
                    PlannedAction(
                        "create_reminder",
                        {"title": title},
                        "The user asked to create a reminder.",
                    )
                ],
            )

        if "running apps" in low or "open apps" in low:
            return Plan(
                goal="List running apps.",
                actions=[PlannedAction("list_running_apps", {}, "The user asked for running applications.")],
            )

        if "frontmost app" in low or "current app" in low or "focused app" in low or "app is focused" in low:
            return Plan(
                goal="Get current focused app.",
                actions=[PlannedAction("frontmost_app", {}, "The user asked which app is focused.")],
            )

        clipboard_command = _strip_trailing_politeness_suffix_only(low_command)
        if clipboard_command in {"clipboard", "read clipboard", "get clipboard", "what is on my clipboard"} or re.fullmatch(
            r"what(?:'s|’s| is)\s+(?:on|in)\s+(?:my\s+|the\s+)?clipboard",
            clipboard_command,
        ):
            return Plan(
                goal="Read clipboard.",
                actions=[PlannedAction("get_clipboard", {}, "The user asked to read the clipboard.")],
            )

        if low in {"voices", "list voices", "show voices", "speech voices"}:
            return Plan(
                goal="List speech voices.",
                actions=[PlannedAction("list_voices", {}, "The user asked for available text-to-speech voices.")],
            )

        voice_input_match = re.search(
            r"^(?:voice input plan|microphone plan|asr plan|wake word plan|wake-word plan|listening plan)(?:\s*:?\s*(?P<mode>.+))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_input_match:
            return Plan(
                goal="Plan safe voice input.",
                actions=[
                    PlannedAction(
                        "voice_input_plan",
                        {"mode": (voice_input_match.group("mode") or "").strip()},
                        "The user asked for microphone, ASR, or wake-word safety planning.",
                    )
                ],
            )

        voice_reply_preview_match = re.search(
            r"^(?:voice reply preview|speech reply preview|spoken reply preview|preview voice reply)(?:\s*:?\s*(?P<text>.+))$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if voice_reply_preview_match:
            return Plan(
                goal="Preview spoken response.",
                actions=[
                    PlannedAction(
                        "voice_reply_preview",
                        {"text": voice_reply_preview_match.group("text").strip()},
                        "The user asked to preview a spoken response without actually speaking.",
                    )
                ],
            )

        spoken_turn_rehearsal_match = re.search(
            r"^(?:"
            r"(?:(?:show|view|review|inspect|preview)\s+(?:me\s+)?(?:the\s+)?(?:(?:latest|last|newest|current)\s+)?)?(?:spoken turn rehearsal|voice turn rehearsal|speech turn rehearsal|spoken rehearsal|voice rehearsal|spoken turn preview|voice turn preview)|"
            r"(?:preview|dry run)\s+(?:spoken|voice)\s+turn|"
            r"rehearse\s+spoken\s+turn"
            r")(?:\s*:?\s*(?P<message>.*))?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if spoken_turn_rehearsal_match:
            message = (spoken_turn_rehearsal_match.group("message") or "").strip()
            if re.fullmatch(r"(?:please|pls|thanks|thank you)[\s,.;!?]*", message, re.IGNORECASE):
                message = ""
            return Plan(
                goal="Rehearse spoken assistant turn.",
                actions=[
                    PlannedAction(
                        "spoken_turn_rehearsal",
                        {"message": message or DEFAULT_HARNESS_PACKET_REQUEST},
                        "The user asked to preview chat-vs-tool routing plus a spoken response without speaking or acting.",
                    )
                ],
            )

        speak_match = re.search(r"^(?:say|speak|read aloud|read this aloud)\s+(?P<text>.+)$", text, re.IGNORECASE | re.DOTALL)
        if speak_match:
            return Plan(
                goal="Speak text aloud.",
                actions=[
                    PlannedAction(
                        "speak",
                        {"text": speak_match.group("text").strip()},
                        "The user asked Jarvis to speak text aloud.",
                    )
                ],
            )

        if low in {"enable computer control", "enable computer use", "computer control on"}:
            return Plan(
                goal="Enable computer control.",
                actions=[PlannedAction("enable_computer_control", {}, "The user asked to enable computer control.")],
            )

        if low in {"disable computer control", "disable computer use", "computer control off"}:
            return Plan(
                goal="Disable computer control.",
                actions=[PlannedAction("disable_computer_control", {}, "The user asked to disable computer control.")],
            )

        if "computer control status" in low:
            return Plan(
                goal="Check computer control status.",
                actions=[PlannedAction("computer_control_status", {}, "The user asked for computer control status.")],
            )

        if "screen size" in low:
            return Plan(
                goal="Read screen size.",
                actions=[PlannedAction("screen_size", {}, "The user asked for screen dimensions.")],
            )

        if "mouse position" in low:
            return Plan(
                goal="Read mouse position.",
                actions=[PlannedAction("mouse_position", {}, "The user asked for mouse position.")],
            )

        if low.startswith("screenshot") or re.search(r"\btake\s+(?:a\s+)?screenshot\b", low):
            return Plan(
                goal="Take screenshot.",
                actions=[PlannedAction("screenshot", {}, "The user asked for a screenshot.")],
            )

        verify_screen_match = re.search(
            r"^(?:verify screen|screen verification|verify current screen)(?:\s+expectation)?\s*:?\s*(?P<expectation>.+)?$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if verify_screen_match and (verify_screen_match.group("expectation") or "").strip():
            return Plan(
                goal="Verify screen.",
                actions=[
                    PlannedAction(
                        "verify_screen",
                        {"expectation": verify_screen_match.group("expectation").strip()},
                        "The user asked Jarvis to capture and verify the screen against an expectation.",
                    )
                ],
            )

        oav_match = re.search(
            r"^(?:observe act verify|oav)\s+action\s+(?P<action>click|move_mouse|move|type_text|type)\b(?P<rest>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if oav_match:
            action = oav_match.group("action").lower()
            if action == "move":
                action = "move_mouse"
            if action == "type":
                action = "type_text"
            rest = oav_match.group("rest") or ""
            args: dict[str, object] = {"action": action}
            expectation_match = re.search(r"\bexpectation\b\s*:?\s*(?P<expectation>.+)$", rest, re.IGNORECASE | re.DOTALL)
            if expectation_match:
                args["expectation"] = expectation_match.group("expectation").strip()
                rest = rest[: expectation_match.start()].strip()
            if action in {"click", "move_mouse"}:
                x_match = re.search(r"\bx\b\s*:?\s*(-?\d+)", rest, re.IGNORECASE)
                y_match = re.search(r"\by\b\s*:?\s*(-?\d+)", rest, re.IGNORECASE)
                if x_match:
                    args["x"] = int(x_match.group(1))
                if y_match:
                    args["y"] = int(y_match.group(1))
            if action == "type_text":
                text_match = re.search(r"\btext\b\s*:?\s*(?P<text>.+)$", rest, re.IGNORECASE | re.DOTALL)
                if text_match:
                    args["text"] = text_match.group("text").strip()
            return Plan(
                goal="Run one observe-act-verify computer-control step.",
                actions=[
                    PlannedAction(
                        "observe_act_verify",
                        args,
                        "The user asked for one observe-act-verify desktop-control step.",
                    )
                ],
            )

        if re.search(r"\bobserve\s+(?:the\s+)?screen\b", low) or "what is on screen" in low:
            return Plan(
                goal="Observe screen.",
                actions=[PlannedAction("observe_screen", {}, "The user asked Jarvis to observe the screen.")],
            )

        clipboard_match = re.search(r"^(?:copy|set clipboard(?: to)?)\s+(?P<text>.+)$", text, re.IGNORECASE | re.DOTALL)
        if clipboard_match:
            return Plan(
                goal="Set clipboard.",
                actions=[
                    PlannedAction(
                        "set_clipboard",
                        {"text": clipboard_match.group("text").strip()},
                        "The user asked to copy text to the clipboard.",
                    )
                ],
            )

        open_app_match = re.search(r"^(?:open|launch)\s+(?P<name>[A-Za-z0-9 ._-]+)$", text, re.IGNORECASE)
        if open_app_match and not low.startswith(("open file", "open http")):
            name = open_app_match.group("name").strip()
            weekday_window_name = re.fullmatch(
                r"(?:calendar\s+)?(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?",
                name.lower(),
            )
            if low.startswith("open ") and (
                re.search(r"^(?:e-?mail|mail)\s+(?:from|about|subject|with subject|containing)\b", name.lower())
                or re.search(r"^at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?$", name.lower())
                or name.lower() in {
                "today",
                "tomorrow",
                "tonight",
                "this week",
                "this morning",
                "this afternoon",
                "this evening",
                "calendar this morning",
                "calendar this afternoon",
                "calendar this evening",
                "calendar this weekend",
                "calendar next weekend",
                "calendar weekend",
                "calendar this month",
                "calendar next month",
                "calendar month",
                "calendar this year",
                "calendar next year",
                "calendar year",
                "tomorrow morning",
                "tomorrow afternoon",
                "tomorrow evening",
                "tomorrow night",
                "calendar today",
                "calendar tomorrow",
                "calendar tonight",
                "calendar tomorrow night",
                "calendar this week",
                "calendar next week",
                "calendar week",
                "next week",
                "this weekend",
                "next weekend",
                "weekend",
                "this month",
                "next month",
                "this year",
                "next year",
                } or weekday_window_name
            ):
                pass
            else:
                return Plan(
                    goal=f"Open {name}.",
                    actions=[PlannedAction("open_application", {"name": name}, "The user asked to open an app.")],
                )

        volume_match = re.search(r"\b(?:set\s+)?volume\s*(?:to)?\s*(?P<level>\d{1,3})?\s*%?", low)
        if volume_match and "volume" in low:
            args = {}
            if volume_match.group("level"):
                args["level"] = int(volume_match.group("level"))
            return Plan(
                goal="Read or set volume.",
                actions=[PlannedAction("volume", args, "The user asked about system volume.")],
            )

        bmi_weight_units = r"(?:kg|kilograms?|lbs?|pounds?)"
        bmi_height_units = r"(?:cm|centimeters?|m|meters?|inches?|inch|in|ft|feet|foot)"
        bmi_metric_match = re.search(
            rf"^(?:calculate\s+)?bmi(?:\s+for)?\s+(?P<weight>\d+(?:\.\d+)?)\s*(?P<weight_unit>{bmi_weight_units})"
            rf"(?:\s+(?:and|at|with))?\s+(?P<height>\d+(?:\.\d+)?)\s*(?P<height_unit>{bmi_height_units})[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if bmi_metric_match:
            return Plan(
                goal="Calculate BMI.",
                actions=[
                    PlannedAction(
                        "calculate_bmi",
                        {
                            "weight": float(bmi_metric_match.group("weight")),
                            "weight_unit": bmi_metric_match.group("weight_unit").lower(),
                            "height": float(bmi_metric_match.group("height")),
                            "height_unit": bmi_metric_match.group("height_unit").lower(),
                        },
                        "The user asked for a read-only BMI calculation.",
                    )
                ],
            )
        bmi_symbolic_height_match = re.search(
            rf"^(?:calculate\s+)?bmi(?:\s+for)?\s+(?P<weight>\d+(?:\.\d+)?)\s*(?P<weight_unit>{bmi_weight_units})"
            r"(?:\s+(?:and|at|with))?\s+(?P<feet>\d+(?:\.\d+)?)\s*(?:'|ft|feet|foot)\s*"
            r"(?P<inches>\d+(?:\.\d+)?)?\s*(?:\"|inches|inch|in)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if bmi_symbolic_height_match:
            feet = float(bmi_symbolic_height_match.group("feet"))
            inches = float(bmi_symbolic_height_match.group("inches") or 0)
            return Plan(
                goal="Calculate BMI.",
                actions=[
                    PlannedAction(
                        "calculate_bmi",
                        {
                            "weight": float(bmi_symbolic_height_match.group("weight")),
                            "weight_unit": bmi_symbolic_height_match.group("weight_unit").lower(),
                            "height": (feet * 12.0) + inches,
                            "height_unit": "inches",
                        },
                        "The user asked for a read-only BMI calculation with feet-and-inches height.",
                    )
                ],
            )

        convert_match = re.search(
            r"^(?:(?:what is|what's|how much is|how cold is|how hot is)\s+)?(?:convert\s+)?(?P<value>-?\d+(?:\.\d+)?)\s*(?P<from>[a-zA-Z/ ]+?)\s+(?:to|in|into)\s+(?P<to>[a-zA-Z/ ]+?)(?:\s+and\s+.+)?$",
            text,
            re.IGNORECASE,
        )
        if convert_match:
            from jarvis_v2.tools.currency_connector import _code as _currency_code
            if _currency_code(convert_match.group("from")) and _currency_code(convert_match.group("to")):
                return Plan(
                    goal="Convert currency.",
                    actions=[PlannedAction("convert_currency", {"text": text}, "User asked for a currency conversion.")],
                )
            return Plan(
                goal="Convert units.",
                actions=[
                    PlannedAction(
                        "convert_units",
                        {
                            "value": float(convert_match.group("value")),
                            "from_unit": convert_match.group("from").strip(),
                            "to_unit": convert_match.group("to").strip(),
                        },
                        "The user asked for unit conversion.",
                    )
                ],
            )
        currency_token_pattern = (
            r"(?:usd|krw|eur|jpy|gbp|cny|aud|cad|chf|hkd|sgd|inr|won|yen|"
            r"euros?|dollars?|bucks?|pounds?|yuan)"
        )
        if re.search(
            rf"^(?:(?:currency|exchange rate)\s+)?(?:\d[\d,.]*(?:\.\d+)?\s+)?{currency_token_pattern}"
            rf"(?:\s+(?:to|in|into))?\s+{currency_token_pattern}$",
            low_command,
        ):
            return Plan(
                goal="Convert currency.",
                actions=[PlannedAction("convert_currency", {"text": text}, "User asked for a terse currency conversion.")],
            )
        simple_unit_tokens = {
            "c",
            "celsius",
            "f",
            "fahrenheit",
            "k",
            "kelvin",
            "m",
            "meter",
            "meters",
            "km",
            "kilometer",
            "kilometers",
            "cm",
            "centimeter",
            "centimeters",
            "mm",
            "ft",
            "feet",
            "foot",
            "in",
            "inch",
            "inches",
            "mi",
            "mile",
            "miles",
            "kg",
            "kilogram",
            "kilograms",
            "g",
            "gram",
            "grams",
            "lb",
            "lbs",
            "pound",
            "pounds",
            "oz",
            "ounce",
            "ounces",
            "l",
            "liter",
            "liters",
            "ml",
            "milliliter",
            "milliliters",
            "gal",
            "gallon",
            "gallons",
            "cup",
            "cups",
        }
        unit_number_words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        unit_number_word_pattern = (
            r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
            r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
            r"eighty|ninety)(?:\s+(?:one|two|three|four|five|six|seven|eight|nine))?"
        )
        spoken_unit_value_pattern = (
            rf"(?:-?\d+(?:\.\d+)?|(?:minus|negative)\s+(?:\d+(?:\.\d+)?|{unit_number_word_pattern})|"
            rf"{unit_number_word_pattern}\s+and\s+a\s+half|{unit_number_word_pattern}\s+point\s+"
            rf"(?:zero|one|two|three|four|five|six|seven|eight|nine)(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine))*|"
            rf"(?:one|two|three)\s+quarters?|half(?:\s+(?:a|an))?|(?:a|one)\s+half|quarter(?:\s+(?:a|an))?|(?:a|one)\s+quarter|"
            rf"{unit_number_word_pattern}|a|an)"
        )

        def _parse_unit_number_words(raw_number: str) -> float:
            words = raw_number.split()
            if len(words) == 1:
                return float(unit_number_words[words[0]])
            return float(unit_number_words[words[0]] + unit_number_words[words[1]])

        def _parse_spoken_unit_value(raw_value: str) -> float:
            normalized = re.sub(r"\s+", " ", raw_value.strip().lower())
            if normalized in {"one", "a", "an"}:
                return 1.0
            if normalized in {"half", "half a", "half an", "a half", "one half"}:
                return 0.5
            if normalized in {"quarter", "quarter a", "quarter an", "a quarter", "one quarter"}:
                return 0.25
            if normalized.startswith(("minus ", "negative ")):
                return -_parse_spoken_unit_value(normalized.split(" ", 1)[1])
            quarter_match = re.fullmatch(r"(?P<count>one|two|three)\s+quarters?", normalized)
            if quarter_match:
                return _parse_unit_number_words(quarter_match.group("count")) * 0.25
            half_match = re.fullmatch(rf"(?P<number>{unit_number_word_pattern})\s+and\s+a\s+half", normalized)
            if half_match:
                return _parse_unit_number_words(half_match.group("number")) + 0.5
            point_match = re.fullmatch(rf"(?P<number>{unit_number_word_pattern})\s+point\s+(?P<digits>(?:zero|one|two|three|four|five|six|seven|eight|nine)(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine))*)", normalized)
            if point_match:
                digits = "".join(str(int(unit_number_words[word])) for word in point_match.group("digits").split())
                return float(f"{int(_parse_unit_number_words(point_match.group('number')))}.{digits}")
            if re.fullmatch(unit_number_word_pattern, normalized):
                return _parse_unit_number_words(normalized)
            return float(normalized)

        mixed_height_target_units = {
            "m",
            "meter",
            "meters",
            "cm",
            "centimeter",
            "centimeters",
            "mm",
            "ft",
            "feet",
            "foot",
            "in",
            "inch",
            "inches",
        }
        mixed_height_match = re.search(
            rf"^(?P<feet>{spoken_unit_value_pattern})\s+(?:ft|feet|foot)\s+"
            rf"(?P<inches>{spoken_unit_value_pattern})(?:\s+(?:inches|inch|in))?\s+"
            rf"(?:(?:to|in|into)\s+)?(?P<to>[a-zA-Z]+)$",
            text,
            re.IGNORECASE,
        )
        if mixed_height_match:
            to_unit = mixed_height_match.group("to").lower()
            if to_unit in mixed_height_target_units:
                if to_unit in {"in", "inch", "inches"} and not re.search(r"\s(?:to|into)\s+(?:in|inch|inches)$", text, re.IGNORECASE):
                    mixed_height_match = None
                else:
                    feet = _parse_spoken_unit_value(mixed_height_match.group("feet"))
                    inches = _parse_spoken_unit_value(mixed_height_match.group("inches"))
                    return Plan(
                        goal="Convert mixed feet and inches.",
                        actions=[
                            PlannedAction(
                                "convert_units",
                                {"value": (feet * 12.0) + inches, "from_unit": "inches", "to_unit": to_unit},
                                "The user asked for a mixed feet-and-inches conversion.",
                            )
                        ],
                    )
        symbolic_height_match = re.search(
            r"^(?:(?:convert|height)\s+)?(?P<feet>\d+(?:\.\d+)?)\s*(?:'|ft|feet|foot)\s*"
            r"(?P<inches>\d+(?:\.\d+)?)?\s*(?:\"|inches|inch|in)?\s+"
            r"(?:to|in|into)\s+(?P<to>[a-zA-Z]+)$",
            text,
            re.IGNORECASE,
        )
        if symbolic_height_match:
            to_unit = symbolic_height_match.group("to").lower()
            if to_unit in mixed_height_target_units:
                feet = float(symbolic_height_match.group("feet"))
                inches = float(symbolic_height_match.group("inches") or 0)
                return Plan(
                    goal="Convert symbolic feet and inches.",
                    actions=[
                        PlannedAction(
                            "convert_units",
                            {"value": (feet * 12.0) + inches, "from_unit": "inches", "to_unit": to_unit},
                            "The user asked for a symbolic feet-and-inches conversion.",
                        )
                    ],
                )
        terse_unit_match = re.search(
            rf"^(?P<value>{spoken_unit_value_pattern})\s+(?P<from>[a-zA-Z]+)\s+(?:(?:to|in|into)\s+)?(?P<to>[a-zA-Z]+)$",
            text,
            re.IGNORECASE,
        )
        if terse_unit_match:
            from_unit = terse_unit_match.group("from").lower()
            to_unit = terse_unit_match.group("to").lower()
            if from_unit in simple_unit_tokens and to_unit in simple_unit_tokens:
                value = _parse_spoken_unit_value(terse_unit_match.group("value"))
                return Plan(
                    goal="Convert units.",
                    actions=[
                        PlannedAction(
                            "convert_units",
                            {"value": value, "from_unit": from_unit, "to_unit": to_unit},
                            "The user asked for a terse unit conversion.",
                        )
                    ],
                )
        how_many_value_units_match = re.search(
            rf"^how many\s+(?P<to>[a-zA-Z/ ]+?)\s+(?:is|are)\s+(?P<value>{spoken_unit_value_pattern})\s+(?P<from>[a-zA-Z/ ]+)$",
            text,
            re.IGNORECASE,
        )
        if how_many_value_units_match:
            from jarvis_v2.tools.currency_connector import _code as _currency_code

            how_many_from = how_many_value_units_match.group("from").strip()
            how_many_to = how_many_value_units_match.group("to").strip()
            if _currency_code(how_many_from) and _currency_code(how_many_to):
                return Plan(
                    goal="Convert currency.",
                    actions=[PlannedAction("convert_currency", {"text": text}, "User asked for a currency conversion.")],
                )
            return Plan(
                goal="Convert units.",
                actions=[
                    PlannedAction(
                        "convert_units",
                        {
                            "value": _parse_spoken_unit_value(how_many_value_units_match.group("value")),
                            "from_unit": how_many_from,
                            "to_unit": how_many_to,
                        },
                        "The user asked for unit conversion.",
                    )
                ],
            )
        how_many_units_match = re.search(
            # Real gap found live 2026-07-10: "how many minutes ARE THERE in
            # 2.5 hours" swallowed "are there" into the unit-name capture
            # (producing to_unit="minutes are there", an unknown-unit
            # refusal) because only "are " was an optional connector, not
            # "are there ". Added "there" as an additional optional word.
            r"^how many\s+(?P<to>[a-zA-Z/ ]+?)\s+(?:are\s+)?(?:there\s+)?in\s+(?:(?P<value>-?\d+(?:\.\d+)?)|a|an|one|1)\s+(?P<from>[a-zA-Z/ ]+)$",
            text,
            re.IGNORECASE,
        )
        if how_many_units_match:
            from jarvis_v2.tools.currency_connector import _code as _currency_code

            value_text = how_many_units_match.group("value")
            how_many_from2 = how_many_units_match.group("from").strip()
            how_many_to2 = how_many_units_match.group("to").strip()
            if _currency_code(how_many_from2) and _currency_code(how_many_to2):
                return Plan(
                    goal="Convert currency.",
                    actions=[PlannedAction("convert_currency", {"text": text}, "User asked for a currency conversion.")],
                )
            return Plan(
                goal="Convert units.",
                actions=[
                    PlannedAction(
                        "convert_units",
                        {
                            "value": float(value_text) if value_text else 1.0,
                            "from_unit": how_many_from2,
                            "to_unit": how_many_to2,
                        },
                        "The user asked for unit conversion.",
                    )
                ],
            )

        spell_match = re.search(
            r"^(?:spell(?!\s+check\b)(?: out)?|how do you spell)\s+(?P<word>[a-zA-Z0-9][a-zA-Z0-9' -]{0,78}?)(?: please)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if spell_match:
            return Plan(
                goal="Spell a word.",
                actions=[PlannedAction("spell_word", {"word": spell_match.group("word").strip()}, "The user asked Jarvis to spell a word.")],
            )

        count_text_match = re.search(
            r"^(?:(?:word|character|char|letter)\s+count|count\s+(?:the\s+)?(?:words?|characters?|chars?|letters?)(?:\s+in)?)\s+(?P<text>.+?)(?: please)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if not count_text_match:
            count_text_match = re.search(
                r"^how\s+many\s+(?:words?|characters?|chars?|letters?)\s+(?:are\s+)?(?:in|is in)\s+(?P<text>.+?)(?: please)?[\?\.!]*$",
                text,
                re.IGNORECASE,
            )
        if count_text_match:
            count_text = count_text_match.group("text").strip()
            if re.search(r"^(?:(?:word|character|char|letter)\s+count|count\s+(?:the\s+)?(?:words?|characters?|chars?|letters?))\s+(?:for|of)\s+", text, re.IGNORECASE):
                count_text = re.sub(r"^(?:for|of)\s+", "", count_text, flags=re.IGNORECASE).strip()
            return Plan(
                goal="Count words and characters in text.",
                actions=[PlannedAction("count_text", {"text": count_text}, "The user asked Jarvis to count words or characters.")],
            )

        initials_text_match = re.search(
            r"^(?:what\s+are\s+the\s+)?(?:initials|first\s+letters)\s+(?:of|for)\s+(?P<text>.+?)(?: please)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if not initials_text_match:
            initials_text_match = re.search(
                r"^(?:get|make)\s+(?:the\s+)?(?:initials|first\s+letters)\s+(?:(?:of|for)\s+)?(?P<text>.+?)(?: please)?[\?\.!]*$",
                text,
                re.IGNORECASE,
            )
        if not initials_text_match:
            initials_text_match = re.search(
                r"^(?:make\s+)?(?:an?\s+)?acronym\s+(?:(?:of|for)\s+)?(?P<text>.+?)(?: please)?[\?\.!]*$",
                text,
                re.IGNORECASE,
            )
        if initials_text_match:
            initials_text = initials_text_match.group("text").strip()
            return Plan(
                goal="Transform text locally.",
                actions=[PlannedAction("transform_text", {"text": initials_text, "mode": "initials"}, "The user asked Jarvis to extract initials locally.")],
            )

        transform_text_match = re.search(
            r"^(?P<mode>uppercase|lowercase|title\s?case|titlecase|sentence case|swap case|swapcase|camel\s?case|pascal\s?case|snake\s?case|kebab\s?case|initials|acronym|first letters|capitalize|slugify|slug|remove extra spaces|normalize spaces|normalise spaces|collapse spaces|clean spaces|remove spaces|no spaces|trim spaces|trim whitespace|strip spaces|strip whitespace|trim|sort words|alphabetize words?|alphabetize|reverse words|reverse text|reverse|unique words|dedupe words|remove punctuation|strip punctuation)\s+(?P<text>.+?)(?: please)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if not transform_text_match:
            # Real gap found live 2026-07-10 (round 44): unbounded lazy `text`
            # capture here must find a required trailing mode word -- 20,000
            # whitespace chars with no such word took 12.2s (the worst of the
            # 5 transform_text_match patterns). Bounded to 2000 chars, well
            # beyond any realistic text-to-transform length.
            transform_text_match = re.search(
                r"^make\s+(?P<text>.{1,2000}?)\s+(?P<mode>uppercase|lowercase|title\s?case|titlecase|sentence case|capitalized|slugified|slug|no spaces|camel\s?case|pascal\s?case|snake\s?case|kebab\s?case)(?: please)?[\?\.!]*$",
                text,
                re.IGNORECASE,
            )
        if not transform_text_match:
            transform_text_match = re.search(
                r"^make\s+(?P<mode>initials|an acronym|a acronym)\s+(?P<text>.+?)(?: please)?[\?\.!]*$",
                text,
                re.IGNORECASE,
            )
        if not transform_text_match:
            # Real gap found live 2026-07-10 (round 44), same class as the
            # "make X mode" pattern above: unbounded lazy `text` capture
            # caused quadratic backtracking (2.87s at 20,000 non-matching
            # whitespace chars). Bounded to 2000 chars.
            transform_text_match = re.search(
                r"^(?:convert|turn)\s+(?P<text>.{1,2000}?)\s+(?:to|into)\s+(?P<mode>uppercase|lowercase|title\s?case|titlecase|sentence case|capitalized|camel\s?case|pascal\s?case|snake\s?case|kebab\s?case|slugified|slug|no spaces|remove spaces|trim spaces|remove punctuation|strip punctuation)(?: please)?[\?\.!]*$",
                text,
                re.IGNORECASE,
            )
        if not transform_text_match:
            transform_text_match = re.search(
                r"^(?P<mode>sort|alphabetize|dedupe|unique)\s+the\s+words\s+(?P<text>.+?)(?: please)?[\?\.!]*$",
                text,
                re.IGNORECASE,
            )
        if transform_text_match:
            mode = transform_text_match.group("mode").lower()
            if mode == "capitalized":
                mode = "capitalize"
            elif mode in {"slugified", "slug"}:
                mode = "slugify"
            elif mode in {"alphabetize", "alphabetize words", "sort"}:
                mode = "sort words"
            elif mode in {"dedupe", "unique"}:
                mode = "unique words"
            elif mode in {"titlecase", "title case"}:
                mode = "title case"
            elif mode == "camelcase":
                mode = "camel case"
            elif mode == "pascalcase":
                mode = "pascal case"
            elif mode == "snakecase":
                mode = "snake case"
            elif mode == "kebabcase":
                mode = "kebab case"
            elif mode in {"trim", "trim whitespace", "strip spaces", "strip whitespace"}:
                mode = "trim spaces"
            elif mode in {"normalize spaces", "normalise spaces", "collapse spaces", "clean spaces", "remove extra spaces"}:
                mode = "normalize spaces"
            elif mode in {"an acronym", "a acronym", "acronym", "first letters"}:
                mode = "initials"
            transform_text = transform_text_match.group("text").strip()
            if mode == "initials":
                transform_text = re.sub(r"^(?:of|for)\s+", "", transform_text, flags=re.IGNORECASE).strip()
            if mode in {"normalize spaces", "remove spaces", "no spaces", "trim spaces", "trim whitespace", "remove punctuation", "strip punctuation"}:
                transform_text = re.sub(r"^from\s+", "", transform_text, flags=re.IGNORECASE).strip()
            if mode in {"sort words", "unique words", "dedupe words"} and re.match(r"^the\s+words\s+", transform_text, re.IGNORECASE):
                transform_text = re.sub(r"^the\s+words\s+", "", transform_text, flags=re.IGNORECASE).strip()
            if mode == "reverse" and re.match(r"^the\s+words\s+", transform_text, re.IGNORECASE):
                mode = "reverse words"
                transform_text = re.sub(r"^the\s+words\s+", "", transform_text, flags=re.IGNORECASE).strip()
            elif mode == "reverse" and re.match(r"^the\s+text\s+", transform_text, re.IGNORECASE):
                mode = "reverse text"
                transform_text = re.sub(r"^the\s+text\s+", "", transform_text, flags=re.IGNORECASE).strip()
            if mode in {
                "uppercase",
                "lowercase",
                "title case",
                "sentence case",
                "swap case",
                "capitalize",
                "camel case",
                "pascal case",
                "snake case",
                "kebab case",
                "slugify",
                "normalize spaces",
                "trim spaces",
            }:
                transform_text = re.sub(r"^the\s+(?:phrase|text|words)\s+", "", transform_text, flags=re.IGNORECASE).strip()
                word_label_m = re.fullmatch(r"the\s+word\s+(?P<word>[A-Za-z0-9][A-Za-z0-9'’-]*)", transform_text, re.IGNORECASE)
                if word_label_m:
                    transform_text = word_label_m.group("word")
            if re.match(r"^(?:and|then)\s+", transform_text, re.IGNORECASE):
                transform_text_match = None
        if transform_text_match:
            return Plan(
                goal="Transform text locally.",
                actions=[PlannedAction("transform_text", {"text": transform_text, "mode": mode}, "The user asked Jarvis to transform text locally.")],
            )

        repeat_text_match = re.search(
            r"^repeat\s+(?P<text>.+?)\s+(?:x\s*)?(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s+times)?(?: please)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if repeat_text_match:
            count_text = repeat_text_match.group("count").lower()
            count_words = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
            }
            count = int(count_text) if count_text.isdigit() else count_words.get(count_text, 0)
            return Plan(
                goal="Repeat text locally.",
                actions=[PlannedAction("transform_text", {"text": repeat_text_match.group("text").strip(), "mode": "repeat", "count": count}, "The user asked Jarvis to repeat text locally.")],
            )

        explicit_tip_match = re.search(
            r"^(?:calculate|calc|what(?:'s| is))?\s*(?:an?\s+|the\s+)?(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\s+tip\s+(?:on|for)\s+\$?(?P<amount>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?(?:\s+bill)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if explicit_tip_match:
            expression = f"({explicit_tip_match.group('pct')}/100) * {explicit_tip_match.group('amount')}"
            return Plan(
                goal="Calculate tip.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for tip math.")],
            )

        reverse_tip_match = re.search(
            r"^(?:calculate|calc)?\s*tip\s+(?:on|for)\s+\$?(?P<amount>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?\s+(?:at|with|for)\s+(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent|per cent)[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if reverse_tip_match:
            expression = f"({reverse_tip_match.group('pct')}/100) * {reverse_tip_match.group('amount')}"
            return Plan(
                goal="Calculate tip.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for tip math.")],
            )

        tip_total_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:the\s+)?total\s+with\s+(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\s+tip\s+(?:on|for)\s+\$?(?P<amount>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?(?:\s+bill)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if tip_total_match:
            expression = f"{tip_total_match.group('amount')} * (1 + {tip_total_match.group('pct')}/100)"
            return Plan(
                goal="Calculate total with tip.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for bill total with tip.")],
            )

        split_bill_match = re.search(
            r"^split\s+(?:a\s+)?(?:(?:the\s+)?bill\s+)?(?:of\s+|for\s+)?\$?(?P<amount>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?(?:\s+bill)?\s+(?:between|among|by|for)?\s*(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s*(?:people|persons|ways)?)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if split_bill_match:
            count = _parse_small_count(split_bill_match.group("count"))
            if count < 1:
                return Plan(goal="", actions=[])
            expression = f"{split_bill_match.group('amount')} divided by {count}"
            return Plan(
                goal="Split a bill.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked to split a bill.")],
            )

        each_split_match = re.search(
            r"^(?:how\s+much\s+each|each|per\s+person)\s+(?:for\s+)?\$?(?P<amount>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?\s+(?:split\s+)?(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:ways?|people|persons)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if each_split_match:
            count = _parse_small_count(each_split_match.group("count"))
            if count < 1:
                return Plan(goal="", actions=[])
            expression = f"{each_split_match.group('amount')} divided by {count}"
            return Plan(
                goal="Split a bill.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked to split a bill.")],
            )

        percent_of_total_match = re.search(
            r"^(?:(?:what\s+)?(?:percent|percentage)\s+is\s+\$?(?P<part_a>\d+(?:\.\d+)?)\s+of\s+\$?(?P<whole_a>\d+(?:\.\d+)?)|\$?(?P<part_b>\d+(?:\.\d+)?)\s+is\s+what\s+(?:percent|percentage)\s+of\s+\$?(?P<whole_b>\d+(?:\.\d+)?))[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if percent_of_total_match:
            part = percent_of_total_match.group("part_a") or percent_of_total_match.group("part_b")
            whole = percent_of_total_match.group("whole_a") or percent_of_total_match.group("whole_b")
            expression = f"({part} / {whole}) * 100"
            return Plan(
                goal="Calculate what percent one value is of another.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for percent-of-total math.")],
            )

        percent_change_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:the\s+)?(?:(?P<kind>percent|percentage)\s+change|(?:(?:percent|percentage)\s+)?(?P<direction>increase|decrease))\s+from\s+\$?(?P<start>-?\d+(?:\.\d+)?)\s+to\s+\$?(?P<end>-?\d+(?:\.\d+)?)[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if percent_change_match:
            start = percent_change_match.group("start")
            end = percent_change_match.group("end")
            direction = (percent_change_match.group("direction") or "").lower()
            if direction == "decrease":
                expression = f"(({start} - {end}) / {start}) * 100"
            else:
                expression = f"(({end} - {start}) / {start}) * 100"
            return Plan(
                goal="Calculate percent change.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for percent-change math.")],
            )

        tax_amount_match = re.search(
            r"^(?:(?:sales\s+)?tax\s+(?:on|for)\s+\$?(?P<amount>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?\s+(?:at|with|for)\s+(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent|per cent))[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if tax_amount_match:
            expression = f"({tax_amount_match.group('pct')}/100) * {tax_amount_match.group('amount')}"
            return Plan(
                goal="Calculate tax.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for tax math.")],
            )

        tax_total_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:the\s+)?total\s+(?:with|after)\s+(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\s+(?:sales\s+)?tax\s+(?:on|for)\s+\$?(?P<amount>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if tax_total_match:
            expression = f"{tax_total_match.group('amount')} * (1 + {tax_total_match.group('pct')}/100)"
            return Plan(
                goal="Calculate total with tax.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for total with tax.")],
            )

        quick_multiplier_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?P<op>half|double|triple)\s+(?:of\s+)?\$?(?P<value>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if quick_multiplier_match:
            op = quick_multiplier_match.group("op").lower()
            value = quick_multiplier_match.group("value")
            operators = {"half": "/ 2", "double": "* 2", "triple": "* 3"}
            return Plan(
                goal="Calculate quick math.",
                actions=[PlannedAction("calculate", {"expression": f"{value} {operators[op]}"}, "The user asked for quick math.")],
            )

        square_root_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:the\s+)?(?:square\s+root\s+of|sqrt)\s+\$?(?P<value>\d+(?:\.\d+)?)[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if square_root_match:
            return Plan(
                goal="Calculate square root.",
                actions=[PlannedAction("calculate", {"expression": f"sqrt({square_root_match.group('value')})"}, "The user asked for a square-root calculation.")],
            )

        power_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:(?P<value_a>\d+(?:\.\d+)?)\s+(?P<suffix>squared|cubed)|(?P<verb>square|cube)\s+(?P<value_b>\d+(?:\.\d+)?))[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if power_match:
            value = power_match.group("value_a") or power_match.group("value_b")
            exponent = "2" if (power_match.group("suffix") or power_match.group("verb") or "").lower() in {"squared", "square"} else "3"
            return Plan(
                goal="Calculate power.",
                actions=[PlannedAction("calculate", {"expression": f"{value} ** {exponent}"}, "The user asked for power math.")],
            )

        power_of_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?P<base>\d+(?:\.\d+)?)\s+to\s+the\s+(?:power\s+of\s+)?(?P<exp>\d+(?:\.\d+)?)(?:th|st|nd|rd)?(?:\s+power)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if power_of_match:
            base, exponent = power_of_match.group("base"), power_of_match.group("exp")
            return Plan(
                goal="Calculate power.",
                actions=[PlannedAction("calculate", {"expression": f"{base} ** {exponent}"}, "The user asked for power math.")],
            )

        factorial_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:(?P<value_a>\d+)\s+factorial|factorial\s+(?:of\s+)?(?P<value_b>\d+))[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if factorial_match:
            value = factorial_match.group("value_a") or factorial_match.group("value_b")
            return Plan(
                goal="Calculate factorial.",
                actions=[PlannedAction("calculate", {"expression": f"factorial({value})"}, "The user asked for factorial math.")],
            )

        average_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:the\s+)?(?:average|mean)\s+(?:of\s+)?(?P<values>-?\d+(?:\.\d+)?(?:\s*,?\s*(?:and\s+)?-?\d+(?:\.\d+)?){1,9})[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if average_match:
            values = re.findall(r"-?\d+(?:\.\d+)?", average_match.group("values"))
            if len(values) >= 2:
                return Plan(
                    goal="Calculate average.",
                    actions=[PlannedAction("calculate", {"expression": f"({' + '.join(values)}) / {len(values)}"}, "The user asked for average math.")],
                )

        # Feature gaps found live 2026-07-09 (flagged in that session's log,
        # implemented here): "sum of A, B, C", "max/min of a list", and
        # "difference between A and B" all fell through to chat -- the model
        # answered correctly by reasoning, but that's not a deterministic
        # guarantee for larger/harder inputs, unlike the calculate() tool.
        # sum/max/min reuse average_match's exact value-list grammar (same
        # "X, Y, and Z" / "X Y Z" tolerance); sum/max/min are already allowed
        # builtins in calculate()'s eval namespace, so no new tool code needed.
        sum_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:the\s+)?sum\s+of\s+(?P<values>-?\d+(?:\.\d+)?(?:\s*,?\s*(?:and\s+)?-?\d+(?:\.\d+)?){1,9})[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if sum_match:
            values = re.findall(r"-?\d+(?:\.\d+)?", sum_match.group("values"))
            if len(values) >= 2:
                return Plan(
                    goal="Calculate sum.",
                    actions=[PlannedAction("calculate", {"expression": f"sum([{', '.join(values)}])"}, "The user asked for a sum.")],
                )

        max_min_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:the\s+)?(?P<fn>max(?:imum)?|min(?:imum)?)\s+of\s+(?P<values>-?\d+(?:\.\d+)?(?:\s*,?\s*(?:and\s+)?-?\d+(?:\.\d+)?){1,9})[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if max_min_match:
            values = re.findall(r"-?\d+(?:\.\d+)?", max_min_match.group("values"))
            if len(values) >= 2:
                fn = "max" if max_min_match.group("fn").lower().startswith("max") else "min"
                return Plan(
                    goal=f"Calculate {fn}imum.",
                    actions=[PlannedAction("calculate", {"expression": f"{fn}({', '.join(values)})"}, f"The user asked for the {fn}imum of a list.")],
                )

        difference_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:the\s+)?difference\s+between\s+(?P<value_a>-?\d+(?:\.\d+)?)\s+and\s+(?P<value_b>-?\d+(?:\.\d+)?)[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if difference_match:
            value_a, value_b = difference_match.group("value_a"), difference_match.group("value_b")
            return Plan(
                goal="Calculate difference.",
                actions=[PlannedAction("calculate", {"expression": f"abs({value_a} - {value_b})"}, "The user asked for the difference between two numbers.")],
            )

        round_match = re.search(
            r"^(?:what(?:'s| is)\s+)?round\s+(?P<value>-?\d+(?:\.\d+)?|pi|tau|e)(?:\s+to\s+(?P<places>\d+|one|two|three|four|five|six)\s+(?:decimal\s+places|decimals?|places))?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if round_match:
            places_text = (round_match.group("places") or "").lower()
            place_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
            places = int(places_text) if places_text.isdigit() else place_words.get(places_text)
            expression = f"round({round_match.group('value').lower()}{', ' + str(places) if places is not None else ''})"
            return Plan(
                goal="Calculate rounded number.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for rounding math.")],
            )

        abs_match = re.search(
            r"^(?:what(?:'s| is)\s+)?(?:the\s+)?(?:absolute\s+value\s+of|abs)\s+(?P<value>-?\d+(?:\.\d+)?)[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if abs_match:
            return Plan(
                goal="Calculate absolute value.",
                actions=[PlannedAction("calculate", {"expression": f"abs({abs_match.group('value')})"}, "The user asked for absolute-value math.")],
            )

        discount_match = re.search(
            r"^(?:(?:what(?:'s| is)\s+)?|price\s+after\s+)?(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\s+(?:off|discount)\s+(?:(?:on|of|for)\s+)?\$?(?P<amount>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?(?:\s+(?:bill|price))?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if discount_match:
            expression = f"{discount_match.group('amount')} * (1 - {discount_match.group('pct')}/100)"
            return Plan(
                goal="Calculate discounted price.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for discount math.")],
            )

        percent_adjust_match = re.search(
            r"^(?P<op>increase|raise|decrease|reduce|lower)\s+\$?(?P<amount>\d+(?:\.\d+)?)(?:\s*(?:dollars?|usd|krw|won))?\s+by\s+(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent|per cent)[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if percent_adjust_match:
            direction = "+" if percent_adjust_match.group("op").lower() in {"increase", "raise"} else "-"
            expression = f"{percent_adjust_match.group('amount')} * (1 {direction} {percent_adjust_match.group('pct')}/100)"
            return Plan(
                goal="Calculate percent adjustment.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for percent adjustment math.")],
            )

        calc_match = re.search(r"^(?:calculate|calc|what is|what's)\s+(?P<expression>[-+*/().,\d\s%a-zA-Z_]+)$", text, re.IGNORECASE)
        # Real gap found live 2026-07-10: this expression class is any letters/
        # digits/math-symbols after "what's"/"what is", so a non-math phrase
        # like "what's the weather like in 3 days" (all-alnum, contains a
        # digit) was silently treated as a calculator expression -- it fails
        # safely (a clean "could not calculate" refusal, not a wrong number),
        # but it's still the wrong tool for an unambiguous weather question.
        # Excluding weather/forecast wording here is narrower and lower-risk
        # than restructuring this broad, general-purpose expression matcher.
        if (
            calc_match
            and any(ch.isdigit() for ch in calc_match.group("expression"))
            and not re.search(r"\b(?:weather|forecast|rain|snow|temperature|temp)\b", calc_match.group("expression"), re.IGNORECASE)
        ):
            expression = calc_match.group("expression").replace("%", "/100")
            return Plan(
                goal="Calculate expression.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for a calculation.")],
            )
        symbol_calc_match = re.fullmatch(r"[-+*/().,\d\s%]+", text.strip())
        if symbol_calc_match and any(ch.isdigit() for ch in text) and any(op in text for op in "+-*/%"):
            return Plan(
                goal="Calculate expression.",
                actions=[PlannedAction("calculate", {"expression": text.strip().replace("%", "/100")}, "The user asked for a calculation.")],
            )
        natural_calc_match = NATURAL_CALC_RE.search(text)
        if natural_calc_match and any(ch.isdigit() for ch in natural_calc_match.group("expression")):
            expression = natural_calc_match.group("expression").strip()
            subtract_match = re.fullmatch(r"subtract\s+(.+?)\s+from\s+(.+)", expression, re.IGNORECASE)
            if subtract_match:
                expression = f"{subtract_match.group(2).strip()} - {subtract_match.group(1).strip()}"
            expression = re.sub(r"^(?:divide|split)\s+(.+?)\s+by\s+(.+)$", r"\1 divided by \2", expression, flags=re.IGNORECASE)
            expression = re.sub(r"^multiply\s+(.+?)\s+by\s+(.+)$", r"\1 multiplied by \2", expression, flags=re.IGNORECASE)
            # Real bug found live 2026-07-09: "add A and B and C" (3+ terms) was
            # substituted via a single lazy-first-group pattern that only
            # converted the FIRST "and" to "plus", leaving later "and"s as the
            # literal word -- which `eval()` then silently interpreted as
            # Python's boolean `and` operator, producing a wrong-but-plausible
            # result with no error ("add 5 and 10 and 15" -> "5 plus 10 and 15"
            # -> eval "5 + 10 and 15" -> 15, not the correct sum 30). Fixed by
            # replacing EVERY "and" (not just the first) once the string is
            # confirmed to start with "add"/"multiply ... and", since for a pure
            # addition/multiplication chain every conjunction means the same
            # operator throughout.
            add_and_match = re.match(r"^add\s+(.+)$", expression, re.IGNORECASE)
            if add_and_match:
                expression = re.sub(r"\band\b", "plus", add_and_match.group(1), flags=re.IGNORECASE)
            else:
                multiply_and_match = re.match(r"^multiply\s+(.+)$", expression, re.IGNORECASE)
                if multiply_and_match and re.search(r"\band\b", multiply_and_match.group(1), re.IGNORECASE):
                    expression = re.sub(r"\band\b", "multiplied by", multiply_and_match.group(1), flags=re.IGNORECASE)
            return Plan(
                goal="Calculate expression.",
                actions=[PlannedAction("calculate", {"expression": expression}, "The user asked for a calculation.")],
            )

        password_match = re.search(
            r"(?:generate|make|create|give me|i need)\s+(?:a\s+)?(?:new\s+)?"
            r"(?:(?:secure|random|strong)\s+)*(?:alphanumeric\s+)?password"
            r"(?:\s+(?P<length>\d+)(?:\s*(?:chars?|characters?|letters?|length))?)?",
            low,
        )
        if password_match and "alphanumeric" in password_match.group(0) and not password_match.group("length"):
            trailing_length = re.search(r"\balphanumeric\s+password\s+(?P<length>\d+)\b", low)
            if trailing_length:
                password_match = trailing_length
        if not password_match:
            password_match = re.search(
                r"(?:generate|make)\s+(?:a\s+)?(?:secure\s+)?(?P<length>\d+)\s*(?:chars?|characters?|letters?|length)\s+(?:alphanumeric\s+)?password\b",
                low,
            )
        if not password_match:
            password_match = re.search(
                r"^(?:random|secure|random\s+secure|secure\s+random)\s+password"
                r"(?:\s+(?P<length>\d+)(?:\s*(?:chars?|characters?|letters?|length))?)?"
                r"(?:\s+(?:no|without)\s+(?:special\s+)?(?:symbols?|characters?))?$",
                low,
            )
        if not password_match:
            password_match = re.search(
                r"(?:^|\b)password\s+(?P<length>\d+)\s*(?:chars?|characters?|letters?|length)\b",
                low,
            )
        if password_match:
            args = {}
            if password_match.group("length"):
                args["length"] = int(password_match.group("length"))
            if re.search(r"\b(?:no|without)\s+(?:special\s+)?(?:symbols?|characters?)\b", low) or re.search(
                r"\b(?:alphanumeric|letters?\s+and\s+numbers?)\b",
                low,
            ):
                args["include_symbols"] = False
            return Plan(
                goal="Generate a secure password.",
                actions=[PlannedAction("generate_password", args, "The user asked for a generated password.")],
            )

        if re.fullmatch(
            r"(?:uuid|guid|(?:generate|make|new|create|random)\s+(?:a\s+)?(?:uuid|guid)|(?:generate|make|new|create|random)\s+(?:a\s+)?(?:uuid|guid)\s+v?4)",
            low,
        ):
            return Plan(
                goal="Generate a UUID.",
                actions=[PlannedAction("generate_uuid", {}, "The user asked for a local UUID.")],
            )

        email_send_match = (
            re.match(
                r"^(?:send\s+)?(?:an?\s+)?e-?mail\s+(?:to\s+)?(?P<to>.+?)\s+"
                r"(?:saying|that says|message|:)\s*(?P<body>.+)$",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            or re.match(
                r"^e-?mail\s+(?P<to>[^:]{1,254}):\s*(?P<body>.+)$",
                text,
                re.IGNORECASE | re.DOTALL,
            )
        )
        if email_send_match:
            return Plan(
                goal="Send an email.",
                actions=[
                    PlannedAction(
                        "send_email",
                        {
                            "to": email_send_match.group("to").strip(" ."),
                            "body": email_send_match.group("body").strip(),
                        },
                        "User asked to send an email; this remains approval-gated.",
                    )
                ],
            )

        kakao_preopened_match = re.match(
            r"^(?:kakao|kakaotalk)\s+preopened_exact_chat\s+(?P<to>[^:]{1,120}):\s*(?P<message>.+)$",
            text,
            re.IGNORECASE,
        )
        if kakao_preopened_match:
            return Plan(
                goal="Send a KakaoTalk message only in an operator-preopened exact chat.",
                actions=[
                    PlannedAction(
                        "send_kakao",
                        {
                            "to": kakao_preopened_match.group("to"),
                            "message": kakao_preopened_match.group("message"),
                            "target_mode": "preopened_exact_chat",
                        },
                        (
                            "User asked to send only in an already-open exact KakaoTalk chat; "
                            "this remains approval-gated."
                        ),
                    )
                ],
            )

        kakao_match = (
            re.match(r"^(?:kakao|kakaotalk)\s+(?P<to>[^:]{1,120}):\s*(?P<message>.+)$", text, re.IGNORECASE)
            # Recipient-first: "send <name> a kakao[talk] [message] saying|: <msg>".
            # This is the most natural phrasing ("send fixture a kakao saying hi") and
            # must be checked before the service-first form below.
            or re.search(
                r"\bsend\s+(?P<to>.+?)\s+(?:a|an)\s+(?:kakao|kakaotalk)(?:\s+(?:message|dm))?\s+(?:saying|that says|message|:)\s*(?P<message>.+)$",
                text,
                re.IGNORECASE,
            )
            or re.search(
                r"\b(?:send\s+)?(?:a\s+)?(?:kakao|kakaotalk)(?:\s+message)?\s+(?:to\s+)?(?P<to>.+?)\s+(?:saying|that says|message|:)\s*(?P<message>.+)$",
                text,
                re.IGNORECASE,
            )
            # Trailing-service phrasing: "text/message <name> to|on|via kakao[talk]
            # saying|: <msg>". Here the recipient comes first and the service is
            # named after it, so the iMessage pattern would otherwise swallow
            # "<name> to kakao" as the recipient and mis-route to iMessage.
            or re.search(
                r"\b(?:send\s+)?(?:a\s+)?(?:text|imessage|message|dm|msg)?\s*(?P<to>.+?)\s+(?:to|on|via|through|in|over)\s+(?:kakao|kakaotalk|카카오톡|카카오)\b(?:\s+message)?\s*(?:saying|that says|message|:)\s*(?P<message>.+)$",
                text,
                re.IGNORECASE,
            )
            or re.search(
                r"(?:카카오톡|카카오)\s+(?P<to>.{1,120}?)에게\s+(?P<message>.+)$",
                text,
                re.IGNORECASE,
            )
        )
        if kakao_match:
            return Plan(
                goal="Send a KakaoTalk message.",
                actions=[
                    PlannedAction(
                        "send_kakao",
                        {"to": kakao_match.group("to").strip(" ."), "message": kakao_match.group("message").strip()},
                        "User asked to send a KakaoTalk message.",
                    )
                ],
            )

        instagram_text = text
        instagram_allow_new_recipient = False
        instagram_text, new_prefix_count = re.subn(
            r"\bsend\s+(?:a\s+)?new\s+(instagram|ig)(\s+(?:dm|message))?",
            r"send an \1\2",
            instagram_text,
            count=1,
            flags=re.IGNORECASE,
        )
        instagram_allow_new_recipient = new_prefix_count == 1
        new_suffix = re.search(
            r"\s*[,;]\s*(?:(?:they|this person)\s+(?:are|is)\s+not\s+in\s+my\s+(?:friends|contacts)|"
            r"i\s+(?:do\s+not|don't)\s+follow\s+(?:them|this person)|new\s+(?:person|recipient))\s*[.!]?\s*$",
            instagram_text,
            re.IGNORECASE,
        )
        if new_suffix:
            instagram_allow_new_recipient = True
            instagram_text = instagram_text[: new_suffix.start()].rstrip()

        instagram_match = (
            re.match(r"^(?:instagram|ig)\s+dm\s+(?P<to>\S{1,120})\s+(?P<message>.+)$", instagram_text, re.IGNORECASE)
            # Recipient-first: "send <name> an instagram dm saying|: <msg>".
            # <to> allows multi-word display names ("Fixture Example"), lazily up to
            # the "a|an instagram" that follows.
            or re.search(
                r"\bsend\s+(?P<to>.+?)\s+(?:a|an)\s+(?:instagram|ig)(?:\s+(?:dm|message))?\s+(?:saying|that says|message|:)\s*(?P<message>.+)$",
                instagram_text,
                re.IGNORECASE,
            )
            or re.search(
                r"\b(?:send\s+)?(?:an?\s+)?(?:instagram|ig)(?:\s+dm|\s+message)?\s+(?:to\s+)?(?P<to>\S{1,120})\s+(?:saying|that says|message|:)\s*(?P<message>.+)$",
                instagram_text,
                re.IGNORECASE,
            )
            or re.search(
                r"\bsend\s+(?:an?\s+)?dm\s+(?:to\s+)?(?P<to>@?\w[\w.]{0,119})\s+on\s+(?:instagram|ig)\s+(?:saying\s+)?(?P<message>.+)$",
                instagram_text,
                re.IGNORECASE,
            )
            or re.search(
                r"\bdm\s+(?P<to>@?\w[\w.]{0,119})\s+on\s+(?:instagram|ig)\s+(?:saying\s+)?(?P<message>.+)$",
                instagram_text,
                re.IGNORECASE,
            )
        )
        if instagram_match:
            instagram_args = {
                "to": instagram_match.group("to").strip(" ."),
                "message": instagram_match.group("message").strip(),
            }
            if instagram_allow_new_recipient:
                instagram_args["allow_new_recipient"] = True
            return Plan(
                goal="Send an Instagram DM.",
                actions=[
                    PlannedAction(
                        "send_instagram_dm",
                        instagram_args,
                        "User asked to send an Instagram DM.",
                    )
                ],
            )

        # Telegram MUST be matched before iMessage: "send a telegram message to X"
        # contains the word "message", so the generic iMessage pattern would
        # swallow it and route to the wrong channel (hit live with BotFather).
        telegram_send_m = (
            re.search(
                r"\bsend\s+(?P<to>.+?)\s+(?:a|an)\s+telegram(?:\s+(?:message|dm))?\s+(?:saying|that says|message|:)\s*(?P<message>.+)$",
                text, re.IGNORECASE,
            )
            or re.search(
                r"\b(?:send\s+)?(?:a\s+)?telegram(?:\s+message)?\s+(?:to\s+)?(?P<to>.+?)\s+(?:saying|that says|message|:)\s*(?P<message>.+)$",
                text, re.IGNORECASE,
            )
            # Trailing-service phrasing: "text/message <name> on|via telegram saying <msg>".
            or re.search(
                r"\b(?:send\s+)?(?:a\s+)?(?:text|imessage|message|dm|msg)?\s*(?P<to>.+?)\s+(?:to|on|via|through|in|over)\s+telegram\b(?:\s+message)?\s*(?:saying|that says|message|:)\s*(?P<message>.+)$",
                text, re.IGNORECASE,
            )
            or re.match(r"^telegram\s+(?P<to>[^:]{1,120}):\s*(?P<message>.+)$", text, re.IGNORECASE)
        )
        if telegram_send_m:
            return Plan(
                goal="Send a Telegram message.",
                actions=[PlannedAction("send_telegram", {"to": telegram_send_m.group("to").strip(" ."), "message": telegram_send_m.group("message").strip()}, "User asked to send a Telegram message.")],
            )

        imessage_match = (
            re.match(r"^(?:text|imessage|message|dm)\s+(?P<to>[^:]{1,120}):\s*(?P<message>.+)$", text, re.IGNORECASE)
            # Recipient-first: "send <name> a text|imessage|message saying|: <msg>".
            # iMessage is the default channel, so this also catches plain
            # "send <name> a message saying <msg>" with no named service.
            or re.search(
                r"\bsend\s+(?P<to>.+?)\s+(?:a|an)\s+(?:text|imessage|message|dm)\s+(?:saying|that says|message|:)\s*(?P<message>.+)$",
                text,
                re.IGNORECASE,
            )
            or re.search(
                r"\b(?:send\s+)?(?:an?\s+)?(?:text|imessage|message|dm)(?:\s+message)?\s+(?:to\s+)?(?P<to>.+?)\s+(?:saying|that says|message|:)\s*(?P<message>.+)$",
                text,
                re.IGNORECASE,
            )
            or re.match(
                r"^(?:text|imessage|message|dm)\s+(?P<to>.+?)\s+(?:saying|that says|message)\s*(?P<message>.+)$",
                text,
                re.IGNORECASE,
            )
        )
        if imessage_match:
            return Plan(
                goal="Send an iMessage.",
                actions=[
                    PlannedAction(
                        "send_imessage",
                        {"to": imessage_match.group("to").strip(" ."), "message": imessage_match.group("message").strip()},
                        "User asked to send an iMessage; this remains approval-gated.",
                    )
                ],
            )

        # --- Calling (FaceTime/phone, KakaoTalk, Instagram, Telegram) ---
        # Channel-specific calls are matched first so "call X on kakao" routes to
        # KakaoTalk instead of being swallowed by the generic FaceTime pattern.
        _call_wants_video = bool(re.search(r"\bvideo\b", low_command)) or (
            "facetime" in low_command and "facetime audio" not in low_command and "audio" not in low_command
        )
        _call_wants_facetime_audio = "facetime" in low_command and not _call_wants_video

        def _contact_call_mode() -> str:
            # A normal "call X" is a PHONE call via the contact's number (through
            # the iPhone). FaceTime is used only when the operator explicitly says
            # facetime / video call.
            if _call_wants_video:
                return "video"
            if _call_wants_facetime_audio:
                return "audio"
            return "phone"

        kakao_call_m = re.search(
            r"\b(?:make\s+)?(?:a\s+)?(?:voice|video|audio)?\s*call\s+(?P<to>.+?)\s+(?:on|via|through|in|over)\s+(?:kakao|kakaotalk|카카오톡|카카오)\b",
            text, re.IGNORECASE,
        ) or re.search(
            r"\b(?:kakao|kakaotalk|카카오톡|카카오)\s+(?:voice|video|audio)?\s*call\s+(?P<to>.+)$",
            text, re.IGNORECASE,
        )
        if kakao_call_m:
            return Plan(
                goal="Call on KakaoTalk.",
                actions=[PlannedAction("call_kakao", {"to": kakao_call_m.group("to").strip(" ."), "mode": "video" if _call_wants_video else "audio"}, "User asked to call someone on KakaoTalk.")],
            )

        instagram_call_m = re.search(
            r"\b(?:make\s+)?(?:a\s+)?(?:voice|video|audio)?\s*call\s+(?P<to>.+?)\s+(?:on|via|through|over)\s+(?:instagram|ig)\b",
            text, re.IGNORECASE,
        ) or re.search(
            r"\b(?:instagram|ig)\s+(?:voice|video|audio)?\s*call\s+(?P<to>.+)$",
            text, re.IGNORECASE,
        )
        if instagram_call_m:
            return Plan(
                goal="Call on Instagram.",
                actions=[PlannedAction("call_instagram", {"to": instagram_call_m.group("to").strip(" ."), "mode": "video" if _call_wants_video else "audio"}, "User asked to call someone on Instagram.")],
            )

        telegram_call_m = re.search(
            r"\b(?:make\s+)?(?:a\s+)?(?:voice|video|audio)?\s*call\s+(?P<to>.+?)\s+(?:on|via|through|over)\s+telegram\b",
            text, re.IGNORECASE,
        ) or re.search(
            r"\btelegram\s+(?:voice|video|audio)?\s*call\s+(?P<to>.+)$",
            text, re.IGNORECASE,
        )
        if telegram_call_m:
            return Plan(
                goal="Call on Telegram.",
                actions=[PlannedAction("call_telegram", {"to": telegram_call_m.group("to").strip(" ."), "mode": "video" if _call_wants_video else "audio"}, "User asked to call someone on Telegram.")],
            )

        # Generic native call (FaceTime video/audio, or phone via Continuity).
        contact_call_m = (
            re.search(r"^\s*facetime\s+(?:audio\s+)?(?P<to>.+)$", text, re.IGNORECASE)
            or re.search(r"^\s*(?:video\s*call|videocall)\s+(?P<to>.+)$", text, re.IGNORECASE)
            or re.search(r"^\s*(?:call|phone|ring|dial|give\s+(?P<to2>.+?)\s+a\s+call)\b\s*(?P<to>.*)$", text, re.IGNORECASE)
        )
        if contact_call_m:
            groups = contact_call_m.groupdict()
            target = (groups.get("to") or "").strip(" .") or (groups.get("to2") or "").strip(" .")
            # Drop trailing "on his/her/their/the phone" so the name resolves cleanly.
            target = re.sub(
                r"\s+on\s+(?:his|her|their|the|my)?\s*(?:phone|cell|cellphone|mobile)\s*$",
                "",
                target,
                flags=re.IGNORECASE,
            ).strip(" .")
            if target:
                return Plan(
                    goal="Place a call.",
                    actions=[PlannedAction("call_contact", {"to": target, "mode": _contact_call_mode()}, "User asked to call a contact via FaceTime/phone.")],
                )

        calendar_delete = _plan_calendar_delete(text, low_command)
        if calendar_delete:
            return calendar_delete
        explicit_calendar_update = _plan_explicit_calendar_update(text, low_command)
        if explicit_calendar_update:
            return explicit_calendar_update
        fuzzy_calendar_update = _plan_fuzzy_calendar_update(low_command)
        if fuzzy_calendar_update:
            return fuzzy_calendar_update
        if (
            allow_risky_natural_dispatch
            and _looks_like_risky_natural_order(low_command)
            and not _is_personal_read_query(low_command)
            and not _is_writer_command(text)
            and not _is_calendar_create_intent(low_command)
            and not _is_direct_reminder_cancellation_command(text)
        ):
            return Plan(
                goal="Route risky natural order through read-only dispatch.",
                actions=[
                    PlannedAction(
                        "dispatch_decision_packet",
                        {"request": text},
                        "The user gave an action-like natural order with risky signals, so Jarvis should choose chat, auto-safe work, approval-gated work, hold, or clarification before execution.",
                    )
                ],
            )

        # --- Personal connector fast-paths ---
        paragraph_type_m = re.match(r"^write a paragraph about\s+(.+?)\s+and type it\.?$", text, re.IGNORECASE | re.DOTALL)
        if paragraph_type_m:
            topic = paragraph_type_m.group(1).strip()
            return Plan(
                goal="Compose text and type it.",
                actions=[
                    PlannedAction(
                        "compose_and_write",
                        {"prompt": f"Write a paragraph about {topic}.", "mode": "human"},
                        "User asked Jarvis to generate text and type it.",
                    )
                ],
            )
        draft_human_m = re.match(r"^draft an email about\s+(.+?)\s+like a human\.?$", text, re.IGNORECASE | re.DOTALL)
        if draft_human_m:
            topic = draft_human_m.group(1).strip()
            return Plan(
                goal="Draft and human-type an email.",
                actions=[
                    PlannedAction(
                        "compose_and_write",
                        {"prompt": f"Draft an email about {topic}.", "mode": "human", "speed": "normal"},
                        "User asked Jarvis to draft text and type it like a human.",
                    )
                ],
            )
        compose_docs_m = re.match(r"^compose\s+(.+?)\s+in\s+google docs\.?$", text, re.IGNORECASE | re.DOTALL)
        if compose_docs_m:
            prompt = compose_docs_m.group(1).strip()
            return Plan(
                goal="Compose text into Google Docs.",
                actions=[
                    PlannedAction(
                        "compose_and_write",
                        {"prompt": prompt, "mode": "human"},
                        "User asked Jarvis to compose text in Google Docs.",
                    )
                ],
            )
        compose_m = re.match(r"^compose(?: and write)?\s*:?\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if compose_m:
            return Plan(
                goal="Compose and write text.",
                actions=[
                    PlannedAction(
                        "compose_and_write",
                        {"prompt": compose_m.group(1).strip(), "mode": "human"},
                        "User asked Jarvis to generate and write text.",
                    )
                ],
            )
        hw_m = re.match(r"^(?:autowrite|human[- ]?write|human[- ]?type|write (?:this )?like a human|type (?:this |it )?(?:out )?like a human|write humanly)\s*:?\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if hw_m:
            body = hw_m.group(1).strip()
            sp = re.match(r"(slow|relaxed|normal|fast|blazing)\s*:?\s*(.*)$", body, re.IGNORECASE | re.DOTALL)
            hw_args: dict[str, Any] = {}
            if sp and sp.group(2).strip():
                hw_args["speed"] = sp.group(1).lower()
                hw_args["text"] = sp.group(2).strip()
            else:
                hw_args["text"] = body
            return Plan(
                goal="Type text like a human.",
                actions=[PlannedAction("human_write", hw_args, "User asked to write like a human.")],
            )
        pt_m = re.match(r"^(?:paste|type\s+(?:this|it)\s+out|type\s+out|type\s+this|insert text|drop text|write block|put text|write text)\s*:?\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if pt_m:
            return Plan(
                goal="Write a block of text.",
                actions=[PlannedAction("paste_text", {"text": pt_m.group(1).strip()}, "User asked to drop a block of text.")],
            )
        if re.search(r"\b(sunrise|sunset)\b|\bwhat time (is|does)\b.{0,12}\b(sun (rise|set)|sunrise|sunset)\b|\bwhen (is|does)\b.{0,12}\b(sunrise|sunset|sun (rise|set))\b", low_command):
            return Plan(
                goal="Report sunrise/sunset.",
                actions=[PlannedAction("get_sun_times", {"text": text}, "User asked about sunrise/sunset.")],
            )
        cd_reverse_left_m = re.search(
            r"^(.{2,60}?)\s+(?:days|weeks|months)\s+left(?:[\?\.!])?$",
            text,
            re.IGNORECASE,
        )
        cd_m = cd_reverse_left_m or re.search(
            r"\b(?:how many (?:days|weeks|months|sleeps)|days|weeks|months|how long|how much longer|count\s+down|countdown)\s+(?:(?:until|til|till|to|before|for)\s+)?(.{2,60}?)[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if cd_m and re.fullmatch(r"left\s+(?:until|til|till|to)\s+.{2,60}", cd_m.group(1).strip(), re.IGNORECASE):
            cd_m = re.search(
                r"\b(?:how many\s+)?(?:days|weeks|months)\s+left\s+(?:until|til|till|to)\s+(.{2,60}?)[\?\.!]*$",
                text,
                re.IGNORECASE,
            )
        if not cd_m:
            cd_m = re.search(r"^(.{2,60}?)\s+countdown(?: please)?[\?\.!]*$", text, re.IGNORECASE)
        if not cd_m:
            cd_m = re.search(
                r"^when is\s+((?:christmas|xmas|christmas eve|new years?(?: day| eve)?|new year's(?: day| eve)?|halloween|easter(?: sunday)?|mlk day|martin luther king(?: jr)? day|presidents day|president's day|washington'?s birthday|juneteenth|veteran'?s day|groundhog day|columbus day|indigenous people'?s day|thanksgiving(?: day)?|black friday|mothers? day|mother's day|fathers? day|father's day|memorial day|labou?r day|valentine'?s(?: day)?|valentines(?: day)?|\d{4}-\d{1,2}-\d{1,2}|[a-z]{3,9}\s+\d{1,2}))(?:[\?\.!])?$",
                text,
                re.IGNORECASE,
            )
        if cd_m:
            target = cd_m.group(1).strip()
            if target.lower() in {"until", "til", "till", "to", "before", "for", "left", "remaining"}:
                return Plan(goal="No deterministic countdown route.", actions=[])
            # Real gap found live 2026-07-09: "how many WEEKS/MONTHS until X"
            # always answered in days regardless of the unit actually asked for
            # -- the unit word was consumed while extracting `target` and never
            # passed to the tool. Detect it separately and pass it through.
            countdown_args: dict[str, Any] = {"target": target}
            if re.search(r"\bweeks?\b", text, re.IGNORECASE) and not re.search(r"\bdays?\b", text, re.IGNORECASE):
                countdown_args["unit"] = "weeks"
            elif re.search(r"\bmonths?\b", text, re.IGNORECASE) and not re.search(r"\bdays?\b", text, re.IGNORECASE):
                countdown_args["unit"] = "months"
            return Plan(
                goal="Count days until a date.",
                actions=[PlannedAction("days_until", countdown_args, "User asked for a countdown.")],
            )
        if re.search(r"^(?:joke|tell joke|tell me something funny|funny joke)(?: please)?[\?\.!]*$|\btell me a joke\b|\b(another|a) joke\b|\bmake me laugh\b|\bdad joke\b|\bgot any jokes\b", low_command):
            return Plan(
                goal="Tell a joke.",
                actions=[PlannedAction("tell_joke", {}, "User asked for a joke.")],
            )
        if re.fullmatch(r"(?:flip (?:a )?coin|coin flip|coin toss|heads or tails|(?:choose|flip|pick|random|toss) heads or tails|toss (?:a )?coin)(?: please)?[\?\.!]*", low_command):
            return Plan(
                goal="Flip a coin.",
                actions=[PlannedAction("flip_coin", {}, "User asked Jarvis to flip a coin.")],
            )
        random_number_m = re.fullmatch(
            r"(?:(?:pick|choose|give me)\s+(?:a\s+)?(?:random\s+)?number|random(?:\s+number)?)"
            rf"(?:\s+(?:(?:between|from)\s+)?(?P<min>{SPOKEN_INTEGER_PATTERN})\s*(?:and|to|-|through)\s*(?P<max>{SPOKEN_INTEGER_PATTERN}))?"
            r"(?: please)?[\?\.!]*",
            low_command,
        )
        if random_number_m:
            args: dict[str, Any] = {}
            if random_number_m.group("min") and random_number_m.group("max"):
                low_bound = _parse_spoken_integer(random_number_m.group("min"))
                high_bound = _parse_spoken_integer(random_number_m.group("max"))
                if low_bound is None or high_bound is None:
                    return Plan(goal="Unclear random number range.", actions=[])
                args["min"] = low_bound
                args["max"] = high_bound
            return Plan(
                goal="Pick a random number.",
                actions=[PlannedAction("random_number", args, "User asked Jarvis to pick a random number.")],
            )
        choose_option_m = re.fullmatch(
            r"(?:(?:help me\s+)?(?:choose|pick|decide)|(?:should i|can you|could you)\s+(?:choose|pick))(?:\s+one)?(?:\s*:\s*|\s+)(?:(?:between|from|of)(?:\s*:\s*|\s+))?(?P<options>.+?)(?: please)?[\?\.!]*",
            text,
            re.IGNORECASE,
        )
        if choose_option_m:
            raw_options = choose_option_m.group("options").strip()
            if re.search(r"\b(?:random\s+)?number\b", raw_options, re.IGNORECASE):
                raw_options = ""
            options = [
                option.strip(" \t\r\n,;:.!?")
                for option in re.split(r"\s*(?:,|;|\bor\b|\band\b)\s*", raw_options, flags=re.IGNORECASE)
            ]
            options = [option for option in options if option]
            deduped_options: list[str] = []
            seen_options: set[str] = set()
            for option in options:
                key = option.casefold()
                if key in seen_options:
                    continue
                seen_options.add(key)
                deduped_options.append(option[:80])
            if len(deduped_options) < 2:
                choose_option_m = None
        if choose_option_m:
            return Plan(
                goal="Choose between options.",
                actions=[
                    PlannedAction(
                        "choose_option",
                        {"options": deduped_options[:12]},
                        "User asked Jarvis to choose between options.",
                    )
                ],
            )
        dice_m = re.fullmatch(
            r"(?:roll (?:(?P<count>\d{1,2})\s*)?d(?P<sides>\d{1,3})"
            r"|roll (?:(?P<alt_count>\d{1,2}|one|two|three|four|five|six)\s+)?(?:a |the )?d(?P<alt_sides>\d{1,3})"
            r"|roll (?:(?P<sided_count>\d{1,2}|one|two|three|four|five|six)\s+)?(?:a |the )?(?P<sided_sides>\d{1,3}|four|six|eight|ten|twelve|twenty|hundred)[ -]?sided (?:die|dice)"
            r"|roll (?P<pair>a pair|pair) of dice"
            r"|roll (?:(?P<dice_count>\d{1,2}|one|two|three|four|five|six)\s+)?(?:a |the )?(?:die|dice))(?: please)?[\?\.!]*",
            low_command,
        )
        if dice_m:
            args: dict[str, Any] = {}
            word_counts = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
            word_sides = {"four": 4, "six": 6, "eight": 8, "ten": 10, "twelve": 12, "twenty": 20, "hundred": 100}
            if dice_m.group("sides"):
                args["sides"] = int(dice_m.group("sides"))
            if dice_m.group("alt_sides"):
                args["sides"] = int(dice_m.group("alt_sides"))
            sided_sides = dice_m.group("sided_sides")
            if sided_sides:
                args["sides"] = word_sides.get(sided_sides, int(sided_sides) if sided_sides.isdigit() else 6)
            if dice_m.group("count"):
                args["count"] = int(dice_m.group("count"))
            alt_count = dice_m.group("alt_count")
            if alt_count:
                args["count"] = word_counts.get(alt_count, int(alt_count) if alt_count.isdigit() else 1)
            sided_count = dice_m.group("sided_count")
            if sided_count:
                args["count"] = word_counts.get(sided_count, int(sided_count) if sided_count.isdigit() else 1)
            dice_count = dice_m.group("dice_count")
            if dice_count:
                args["count"] = word_counts.get(dice_count, int(dice_count) if dice_count.isdigit() else 1)
            if dice_m.group("pair"):
                args["count"] = 2
            return Plan(
                goal="Roll dice.",
                actions=[PlannedAction("roll_dice", args, "User asked Jarvis to roll dice.")],
            )
        fc_m = re.search(
            # Real gap found live 2026-07-10: "find contact john and then
            # call him" (a compound sentence) swallowed the whole second
            # clause into the contact search query ("john and then call
            # him"), which would fail to match any saved contact instead of
            # finding "john" -- and silently dropped the "call him" intent.
            r"\b(?:find|look ?up|search(?: for)?|who is)\s{1,10}(?:the\s{1,10})?contacts?\s{1,10}(?:for\s{1,10}|of\s{1,10})?(?P<a>.+?)(?:\s{1,10}and\s{1,10}then\s{1,10}.+)?$"
            r"|\b(?:find|look ?up)\s+(?P<b>.+?)\s+in (?:my )?contacts\b"
            r"|\b(?:what(?:'?s| is)|look ?up)\s+(?P<c>.+?)(?:'s|s')?\s+(?:phone )?(?:number|phone|email)\b",
            low_command,
        )
        if fc_m:
            who = _clean_freeform_query(fc_m.group("a") or fc_m.group("b") or fc_m.group("c") or "")
            if who:
                return Plan(
                    goal="Look up a contact in macOS Contacts.",
                    actions=[PlannedAction("find_contact", {"query": who}, "User asked to find a contact.")],
                )
        if re.search(r"\b(next |upcoming |any )?(public )?holidays?\b|\bis (today|it) a holiday\b|\bwhen is the next holiday\b|\b(공휴일|휴일)\b", low_command):
            return Plan(
                goal="List upcoming public holidays.",
                actions=[PlannedAction("next_holidays", {"text": text}, "User asked about holidays.")],
            )
        from jarvis_v2.tools.air_connector import CITY_COORDS as _air_city_coords
        air_city_pattern = "|".join(re.escape(city) for city in sorted(_air_city_coords, key=len, reverse=True))
        short_air_city = re.search(
            rf"^(?:air\s+(?:{air_city_pattern})|(?:{air_city_pattern})\s+air|how(?:'s| is)\s+(?:{air_city_pattern})\s+air)(?:\s+(?:today|tomorrow|now|right now))?$",
            low_command,
        )
        if short_air_city or re.search(r"\bair quality\b|\bfine dust\b|\b(미세먼지|공기질)\b|\bis the air (bad|clean|good|safe)\b|\bhow('?s| is) the air\b|\bhow (bad|clean|good|safe) is the air\b|\baqi\b|\bpollution\b", low_command):
            return Plan(
                goal="Report air quality.",
                actions=[PlannedAction("get_air_quality", {"text": text}, "User asked about air quality.")],
            )
        if re.search(
            r"\bon this day\b|\b(?:this day|today) in history\b|\bhistory today\b|\bwhat happened (?:on this day|today in history)\b|\bhistorical events today\b|\bhistoric events today\b|\bhistory facts? today\b|\bhistorical facts? today\b"
            r"|\bwhat happened (?:in history )?on (?:[a-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}[/-]\d{1,2})(?: in history)?\b"
            r"|^(?:[a-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}[/-]\d{1,2})(?: in)? history\b"
            r"|^history (?:on )?(?:[a-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}[/-]\d{1,2})\b",
            low_command,
        ):
            return Plan(
                goal="On this day in history.",
                actions=[PlannedAction("on_this_day", {"text": text}, "User asked about historical events.")],
            )
        day_summary_m = re.search(
            r"\b(?:what(?:'s| is| does)|how(?:'s| is| does))\s+my\s+(?:day|schedule|calendar|agenda|week)\s+(?:look(?:ing)?(?:\s+like)?|hold)\b"
            r"|\b(?:what(?:'s| is| does)|how(?:'s| is| does))\s+tomorrow\s+look(?:\s+like)?\b"
            r"|\b(?:what(?:'s| is| does)|how(?:'s| is| does))\s+my\s+week\s+look(?:\s+like)?\b"
            r"|\b(?:what(?:'s| is)|how(?:'s| is))\s+my\s+(?:schedule|calendar|agenda)(?:\s+(?:today|tomorrow|tonight|this\s+(?:morning|afternoon|evening|night|weekend|week|month|year)|next\s+(?:weekend|week|month|year)|weekend|week|month|year|(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?))?\b"
            r"|^my\s+(?:schedule|calendar|agenda)\s+(?:today|tomorrow|tonight|this\s+(?:morning|afternoon|evening|night|weekend|week|month|year)|next\s+(?:weekend|week|month|year)|weekend|week|month|year|(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?)\b",
            low_command,
        )
        if day_summary_m:
            weekday_m = re.search(r"\b(?P<next>next\s+)?(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low_command)
            daypart_m = re.search(r"\b(?P<part>morning|afternoon|evening|night)\b", low_command)
            daypart = daypart_m.group("part") if daypart_m else ""
            if daypart == "night":
                daypart = "evening"
            if re.search(r"\btomorrow\b", low_command):
                range_name = "tomorrow"
                if daypart:
                    range_name = f"{range_name}_{daypart}"
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": range_name}, "User asked what tomorrow looks like.")],
                )
            if re.search(r"\bnext\s+week\b", low_command):
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": "next_week"}, "User asked what next week looks like.")],
                )
            if re.search(r"\bnext\s+weekend\b", low_command):
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": "next_weekend"}, "User asked what next weekend looks like.")],
                )
            if re.search(r"\bnext\s+month\b", low_command):
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": "next_month"}, "User asked what next month looks like.")],
                )
            if re.search(r"\bthis\s+month\b", low_command):
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": "month"}, "User asked what this month looks like.")],
                )
            if re.search(r"\bnext\s+year\b", low_command):
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": "next_year"}, "User asked what next year looks like.")],
                )
            if re.search(r"\bthis\s+year\b", low_command):
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": "year"}, "User asked what this year looks like.")],
                )
            if re.search(r"\b(?:this\s+)?weekend\b", low_command):
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": "weekend"}, "User asked what this weekend looks like.")],
                )
            if re.search(r"\bthis\s+(?:morning|afternoon|evening|night)\b", low_command) and daypart:
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": f"today_{daypart}"}, "User asked what this part of today looks like.")],
                )
            if weekday_m:
                day = weekday_m.group("day")
                range_name = f"next_{day}" if weekday_m.group("next") else day
                if daypart:
                    range_name = f"{range_name}_{daypart}"
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": range_name}, "User asked what that weekday looks like.")],
                )
            if re.search(r"\b(this week|week)\b", low_command):
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": "week"}, "User asked what their week looks like.")],
                )
            if re.search(r"\b(today|tonight)\b", low_command):
                range_name = "today_evening" if re.search(r"\btonight\b", low_command) else "today"
                if daypart and range_name == "today":
                    range_name = f"{range_name}_{daypart}"
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {"range": range_name}, "User asked what today looks like.")],
                )
            if re.search(r"\bmy\s+(?:schedule|calendar|agenda)\b", low_command):
                return Plan(
                    goal="List calendar events.",
                    actions=[PlannedAction("list_events", {}, "User asked what their schedule looks like.")],
                )
            return Plan(
                goal="Give the daily brief.",
                actions=[PlannedAction("daily_briefing", {}, "User asked for a daily brief.")],
            )
        dictionary_lookup_m = re.search(
            r"^(?:dictionary\s+(?:look\s*up|lookup)\s+|dictionary\s+(?:meaning|definition)\s+(?:of|for)|(?:look\s*up|lookup|search(?:\s+for)?|find)\s+(?:the\s+)?(?:definition|meaning)\s+(?:of|for)|(?:look\s*up|lookup)\s+)(?P<word>.{1,60}?)(?:\s+in\s+(?:the\s+)?dictionary)?(?: please)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        define_m = re.search(
            r"^(?:define|dictionary|definition(?: of)?|what(?:'s| is) the definition of|what does|what(?:'s| is) the meaning of|meaning(?: of)?)\s+(.{1,60}?)(?:\s+mean)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        stand_for_m = re.search(
            r"^what\s+does\s+(?:the\s+)?(?P<word>[a-z][a-z'-]{0,58})\s+stand\s+for(?: please)?[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        suffix_define_m = re.search(
            r"^(?:what(?:'s| is)\s+)?([a-z][a-z'-]{0,58})\s+(?:meaning|definition)[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if dictionary_lookup_m and re.search(r"\bdictionary\b|\bdefinition\b|\bmeaning\b", text, re.IGNORECASE):
            word = _clean_lookup_text(dictionary_lookup_m.group("word"))
            return Plan(
                goal="Define a word.",
                actions=[PlannedAction("define", {"word": word}, "User asked for a definition.")],
            )
        if stand_for_m:
            word = _clean_lookup_text(stand_for_m.group("word"))
            return Plan(
                goal="Define a word.",
                actions=[PlannedAction("define", {"word": word}, "User asked what a term stands for.")],
            )
        if define_m:
            if re.search(r"\bstand\s+for\b", define_m.group(1), re.IGNORECASE):
                return Plan(goal="Answer a stand-for question.", actions=[])
            word = _clean_lookup_text(define_m.group(1))
            return Plan(
                goal="Define a word.",
                actions=[PlannedAction("define", {"word": word}, "User asked for a definition.")],
            )
        if suffix_define_m:
            word = _clean_lookup_text(suffix_define_m.group(1))
            return Plan(
                goal="Define a word.",
                actions=[PlannedAction("define", {"word": word}, "User asked for a definition.")],
            )
        if re.search(r"\b(news|headlines)\b|\bwhat(?:'?s| is| are)? (?:happening|going on)\b", low_command):
            args = {}
            news_query = _extract_news_query(low_command)
            if news_query:
                args["query"] = news_query
            return Plan(
                goal="Report news headlines.",
                actions=[PlannedAction("get_news", args, "User asked for news.")],
            )
        explicit_wiki_m = re.search(
            r"^(?:wiki(?:pedia)?\s+(?:about|for|summary(?:\s+of)?)|(?:summarize|summary\s+of)\s+wiki(?:pedia)?|search\s+wiki(?:pedia)?\s+for)\s+(?P<query>.{2,120}?)[\?\.!]*$"
            r"|^look ?up\s+(?P<lookup_query>.{2,120}?)\s+(?:on|in)?\s*(?:wiki|wikipedia)[\?\.!]*$",
            text,
            re.IGNORECASE,
        )
        if explicit_wiki_m:
            wiki_query = _clean_lookup_text(explicit_wiki_m.group("query") or explicit_wiki_m.group("lookup_query") or "")
            return Plan(
                goal="Summarize from Wikipedia.",
                actions=[PlannedAction("wiki_summary", {"query": wiki_query}, "User asked about a topic.")],
            )
        suffix_wiki_m = re.search(r"^(.{2,120}?)\s+(?:wiki|wikipedia)(?:\s+(?:please|pls))?[\?\.!]*$", text, re.IGNORECASE)
        if suffix_wiki_m:
            wiki_query = _clean_lookup_text(suffix_wiki_m.group(1))
            return Plan(
                goal="Summarize from Wikipedia.",
                actions=[PlannedAction("wiki_summary", {"query": wiki_query}, "User asked about a topic.")],
            )
        wiki_m = re.search(r"^(?:who (?:is|was|are)|tell me about|wiki(?:pedia)?|what (?:is|are))\s+(.{2,120}?)[\?\.!]*$", text, re.IGNORECASE)
        if wiki_m:
            wiki_query = _clean_lookup_text(wiki_m.group(1))
            query_low = wiki_query.lower()
            blocked_wiki_prefixes = (
                "my ",
                "the weather",
                "weather",
                "on my calendar",
                "on my schedule",
                "on my agenda",
                "the time",
                "time",
                "the date",
                "date",
                "today",
                "the news",
                "news",
                "the market",
                "market",
                "markets",
                "bitcoin",
                "btc",
                "ethereum",
                "eth",
                "stock",
                "aapl stock",
            )
            translation_like = bool(re.search(rf"\bwhat(?:'?s| is)\b.{{1,60}}\bin {translation_language}\b", low_command))
            news_like = bool(re.search(r"^(?:what(?:'?s| is| are)?\s+)?(?:happening|going on)\b", query_low))
            weather_like = bool(
                query_low.endswith((" weather", " forecast", " rain forecast", " snow forecast"))
                or query_low in {"the weather", "weather", "the forecast", "forecast"}
            )
            stock_quote_like = bool(
                re.search(
                    r"\b(?:aapl|msft|nvda|tsla|googl?|meta|amzn)\b.{0,18}\b(?:stock|share|quote|price|trading|at|worth)\b"
                    r"|\b(?:apple|microsoft|nvidia|tesla|google|alphabet|meta|facebook|amazon)\b.{0,18}\b(?:stock|share|quote|price|trading|at|worth)\b",
                    query_low,
                )
            )
            conversational_possessive_like = bool(
                re.search(
                    r"(?:['’]s)\s+(?:finish\s+target|target|deadline|due\s+date|birthday|schedule|status|priority|next\s+step)\b",
                    query_low,
                )
            )
            if (
                query_low.startswith(blocked_wiki_prefixes)
                or translation_like
                or news_like
                or weather_like
                or stock_quote_like
                or conversational_possessive_like
            ):
                pass
            else:
                return Plan(
                    goal="Summarize from Wikipedia.",
                    actions=[PlannedAction("wiki_summary", {"query": wiki_query}, "User asked about a topic.")],
                )
        research_m = re.search(r"\b(research|deep dive|look into|read up(?:\s+on)?)\b\s+(?:about\s+|on\s+|into\s+)?(.{2,180})$", text, re.IGNORECASE)
        conversation_summary_like = bool(
            re.search(
                r"^(?:summarize|recap|repeat)\b.{0,100}\b(?:conversation|chat|session)\b",
                low_command,
            )
        )
        if research_m and not conversation_summary_like:
            return Plan(
                goal="Research a topic with sources.",
                actions=[PlannedAction("research", {"query": _clean_freeform_query(research_m.group(2))}, "User asked to research a topic.")],
            )
        lookup_m = re.search(
            r"^google\s+(?P<query_leading>.{2,180})$"
            # "search for (the/a/my) goal(s)/skill(s)/preference(s)/person(s)/
            # people about X" is excluded here (same privacy-relevant misroute
            # class as the round-21/26/27 notes/tasks/files/memory word-order
            # fixes) -- no dedicated keyword-search-by-content tool exists for
            # any of these (goals/skills/preferences only support list/exact-
            # name lookup, and "person" is ambiguous between saved people
            # profiles and macOS Contacts), so rather than guess, this just
            # keeps a private-sounding query from leaking to a public web
            # search; it falls through to chat instead. Real gap found live
            # 2026-07-10.
            r"|\b(?:look up|lookup(?!\s+(?:contacts?|files?|notes?|calendar|emails?|gmail|mail)\b)|search (?:the web|online|google) for|web search|search for(?!\s+(?:the\s+|a\s+|my\s+)?(?:goals?|skills?|preferences?|persons?|people)\b|\s+(?:recent\s+|new\s+)?emails?\b|\s+mail\b|\s+gmail\b)|find (?:info|information) (?:about|on))\s+(?P<query_verb>.{2,180})$",
            text,
            re.IGNORECASE,
        )
        if lookup_m:
            # "google" is only treated as a search verb when it LEADS the
            # command (e.g. "google python tutorials") -- as an unanchored
            # \b match it used to fire on "google" appearing anywhere, so
            # "compare apple and google stock" silently became a web search
            # for just "stock", discarding "compare apple and google" and
            # pre-empting the correct get_stock_price route below. Real gap
            # found live 2026-07-10.
            query_text = lookup_m.group("query_leading") or lookup_m.group("query_verb")
            return Plan(
                goal="Look something up on the web.",
                actions=[PlannedAction("web_lookup", {"query": _clean_freeform_query(query_text)}, "User asked to search the web.")],
            )
        casual_search_m = re.search(
            r"^search\s+(?!my\s+(?:files|tasks?)\b|files\b|tasks?\b|notes?\b|contacts?\b|calendar\b|my\s+emails?\b|for\s+emails?\b|emails?\b|gmail\b|mail\b|for\s+(?:the\s+|a\s+|my\s+)?(?:goals?|skills?|preferences?|persons?|people)\b)(?:the\s+web\s+)?(.{2,180})$",
            text,
            re.IGNORECASE,
        )
        if casual_search_m:
            return Plan(
                goal="Look something up on the web.",
                actions=[
                    PlannedAction(
                        "web_lookup",
                        {"query": _clean_freeform_query(casual_search_m.group(1))},
                        "User asked to search the web.",
                    )
                ],
            )
        # NOTE: the "how are/how's" form must mention markets/stocks/crypto in the
        # OTHER group — a previous version matched "how's <anything> today" and
        # swallowed "how's the weather today?" into a markets snapshot (live bug).
        if re.search(
            r"\b(?:how are|how'?s)\b.{0,20}\b(?:markets?|stocks?|crypto)\b"
            r"|\b(?:markets?)\b.{0,20}\b(?:today|doing|overview|snapshot|update)\b"
            r"|^(?:(?:show|list|display|give|tell)(?:\s+me)?\s+)?(?:the\s+)?markets?(?:\s+(?:please|pls|thanks|thank you))?$"
            r"|^(?:(?:show|list|display|give|tell)(?:\s+me)?\s+)?(?:the\s+)?(?:stock market|stocks?)(?:\s+(?:please|pls|thanks|thank you))?$"
            r"|^(?:(?:show|list|display|give|tell)(?:\s+me)?\s+)?(?:the\s+)?crypto(?:\s+(?:please|pls|thanks|thank you))?$"
            r"|\bmarket overview\b|\bmarket update\b|\bmarkets today\b|\bcrypto prices\b|\bcrypto market(?:s)?(?:\s+(?:today|overview|snapshot|update))?\b"
            r"|\b(?:stocks?|stock market)\s+(?:today|doing|overview|snapshot|update)\b|\bhow are stocks\b",
            low_command,
        ):
            crypto_only = bool(
                re.search(
                    r"\bcrypto prices\b|\bcrypto market|^(?:(?:show|list|display|give me|tell me)\s+)?(?:the\s+)?crypto(?:\s+(?:please|pls|thanks|thank you))?$",
                    low_command,
                )
            )
            return Plan(
                goal="Report a compact markets overview.",
                actions=[PlannedAction("get_markets_overview", {"text": text, "crypto_only": crypto_only}, "User asked for a market snapshot.")],
            )
        if re.search(r"\b(bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge|cardano|ripple|xrp|crypto)\b.{0,18}\b(price|worth|value|cost|trading|how much|at)\b|\b(price|worth|value|how much is|how much's|what'?s)\b.{0,18}\b(bitcoin|btc|ethereum|eth|solana|sol|dogecoin|cardano|ripple|crypto)\b|\bcrypto price\b", low_command):
            return Plan(
                goal="Report a crypto price.",
                actions=[PlannedAction("get_crypto_price", {"text": text}, "User asked for a crypto price.")],
            )
        if re.search(
            r"\bstock price\b|\bshare price\b|\b(stock|ticker|shares?)\b.{0,8}\b(price|of|for|quote)\b"
            r"|\bprice (of|for)\b.{0,12}\bstock\b|\b[a-z]{1,5}\s+stock\b|\bstock\s+[a-z]{1,5}\b"
            r"|\b(?:aapl|msft|nvda|tsla|googl?|meta|amzn)\s+(?:price|quote|trading|at)\b"
            r"|\b(?:quote|price)\s+(?:aapl|msft|nvda|tsla|googl?|meta|amzn)\b"
            r"|\b(?:what'?s|what is|how much is)\s+(?:aapl|msft|nvda|tsla|googl?|meta|amzn)\s+(?:trading\s+)?(?:at|worth)\b"
            r"|\b(?:apple|microsoft|nvidia|tesla|google|alphabet|meta|facebook|amazon)\b.{0,18}\b(?:stock|share|quote|price|trading|at)\b",
            low_command,
        ) or re.search(
            r"\b(?:[A-Z]{1,5}\s+(?:price|quote|trading|at)|(?:quote|price)\s+[A-Z]{1,5}|(?:what'?s|what is|how much is)\s+[A-Z]{1,5}\s+(?:trading\s+)?(?:at|worth))\b",
            text,
        ):
            return Plan(
                goal="Report a stock price.",
                actions=[PlannedAction("get_stock_price", {"text": text}, "User asked for a stock price.")],
            )
        if re.search(
            rf"\btranslate\b|\bhow do you say\b|\bwhat(?:'?s| is)\b.{{1,60}}\bin {translation_language}\b"
            rf"|^(?:{translation_language})\s+for\s+.+",
            low_command,
        ):
            return Plan(
                goal="Translate text.",
                actions=[PlannedAction("translate", {"request": raw_text or text}, "User asked for a translation.")],
            )
        if re.search(r"\bconvert\b.{0,30}\b(usd|krw|eur|jpy|gbp|cny|won|yen|euros?|dollars?|pounds?|yuan)\b|\b\d[\d,.]*\s*(usd|krw|eur|jpy|gbp|won|yen|euros?|dollars?|pounds?)\b.{0,15}\b(to|in|into)\b|\bexchange rate\b|\bhow much is\b.{0,30}\b(in|to)\b.{0,15}\b(usd|krw|eur|jpy|gbp|won|yen|euros?|dollars?|pounds?)\b", low_command):
            return Plan(
                goal="Convert currency.",
                actions=[PlannedAction("convert_currency", {"text": text}, "User asked for a currency conversion.")],
            )
        if re.search(r"\bbrief me\b|^good morning( jarvis)?$|\bwhat('?s| does) my day (look like|hold)\b|\bhow('?s| is) my day look(?:ing)?\b", low_command):
            return Plan(
                goal="Give the daily brief.",
                actions=[PlannedAction("daily_briefing", {}, "User asked for a daily brief.")],
            )
        reminder_cancellation = _plan_reminder_cancellation(text)
        if reminder_cancellation is not None:
            return reminder_cancellation
        apple_reminders_exact_list_match = re.fullmatch(
            r"(?:show\s+)?(?:my\s+)?(?:apple|mac|macos|iphone)\s+reminders?\s+"
            r"(?:from|in)\s+(?:the\s+)?list\s*:\s*(?P<list_name>.+?)"
            r"(?:\s*;\s*limit\s*:\s*(?P<limit>[^;\r\n]+))?",
            raw_text or text,
            re.IGNORECASE | re.DOTALL,
        )
        if apple_reminders_exact_list_match is None:
            apple_reminders_exact_list_match = re.fullmatch(
                r"(?:내\s+)?(?:애플|맥|아이폰)\s*리마인더\s*(?:목록|리스트)\s*:\s*"
                r"(?P<list_name>.+)",
                raw_text or text,
                re.IGNORECASE | re.DOTALL,
            )
        if apple_reminders_exact_list_match is not None:
            exact_list_name = apple_reminders_exact_list_match.group("list_name").strip()
            raw_exact_limit = (apple_reminders_exact_list_match.groupdict().get("limit") or "").strip()
            exact_limit: Any = int(raw_exact_limit) if raw_exact_limit.isdigit() else raw_exact_limit or 5
            if (
                len(exact_list_name) >= 2
                and exact_list_name[0] == exact_list_name[-1]
                and exact_list_name[0] in {'"', "'"}
            ):
                exact_list_name = exact_list_name[1:-1].strip()
            return Plan(
                goal="List a bounded exact Apple Reminders list.",
                actions=[
                    PlannedAction(
                        "apple_reminders",
                        {"list_name": exact_list_name, "limit": exact_limit},
                        "User asked for a bounded read from one exact macOS Reminders list.",
                    )
                ],
            )
        if low_command in {
            "리마인더",
            "리마인더 목록",
            "리마인더 보여줘",
            "리마인더 보여주세요",
            "리마인더 있어",
            "내 리마인더",
            "내 리마인더 목록",
            "내 리마인더 보여줘",
            "내 리마인더 뭐야",
            "알림 목록",
            "알림 보여줘",
            "알림 보여주세요",
            "알림 있어",
            "내 알림",
            "내 알림 목록",
            "내 알림 보여줘",
            "내 알림 뭐야",
            "타이머 목록",
            "타이머 보여줘",
            "알람 목록",
            "알람 보여줘",
            "설정된 리마인더",
            "설정된 알림",
        }:
            return Plan(
                goal="List reminders.",
                actions=[PlannedAction("list_reminders", {}, "User asked in Korean to see their reminders.")],
            )
        if re.search(
            r"\b(apple|mac|macos|ios|iphone)\b.{0,12}\b(reminders?)\b"
            r"|\breminders?\b.{0,6}\bapp\b",
            low_command,
        ):
            return Plan(
                goal="List Apple reminders.",
                actions=[PlannedAction("apple_reminders", {}, "User asked for their macOS Reminders app to-dos.")],
            )
        if re.search(
            r"^(?:reminders?|timers?|alarms?)(?:\s+(?:please|pls|thanks|thank you))?$"
            r"|^(?:reminder|timer|alarm)\s+list(?:\s+(?:please|pls|thanks|thank you))?$"
            r"|\b(list|show|what|any|my)\b.{0,16}\b(reminders?|timers?|alarms?)\b"
            r"|\bis there (?:a|an|any) (?:reminder|timer|alarm)s?(?:\s+(?:set|running))?\b"
            r"|\bdo i have(?: any)? (?:reminders?|timers?|alarms?)\b",
            low_command,
        ):
            return Plan(
                goal="List reminders.",
                actions=[PlannedAction("list_reminders", {}, "User asked to see their reminders.")],
            )
        if re.search(
            r"\bremind me\b|\bset (a )?timer\b|\btimer for\b|\b\d+\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|w)\s+timer\b"
            r"|\btimer\s+\d+\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|w)\b"
            r"|\b(?:ping|notify) me\b|\bwake me(?: up)?\b|\bset (?:an?\s+)?alarm\b|\balarm (?:in|for)\b"
            r"|\balarm\s+(?:at\s+|for\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s+alarm\b"
            r"|\b(?:recurring|repeat(?:ing)?)\b.{0,30}\b(reminder|timer)\b",
            low_command,
        ):
            return Plan(
                goal="Set a reminder/timer.",
                actions=[PlannedAction("set_reminder", {"text": raw_text or text}, "User asked for a reminder or timer.")],
            )
        if re.search(
            r"\bweather\b|\bforecast\b|\bis it (going to |gonna )?(rain|snow|sunny|hot|cold|raining)\b"
            r"|\bwill it (rain|snow)\b|\bwill it be (hot|cold|warm)\b|\bhow (hot|cold|warm) (?:is it|will it be)\b"
            r"|\b(?:what(?:'?s| is)|current)\s+the\s+(?:temperature|temp)(?:\s+(?:outside|today|tomorrow|tonight|now|right now))?\b"
            r"|\bcurrent\s+(?:temperature|temp)\b|\b(?:temperature|temp)\s+(?:outside|today|tomorrow|tonight|now|right now)\b"
            r"|\b(?:temperature|temp)\s+(?:in|for|at)\s+[a-z][a-z\s]{1,40}\b"
            r"|\b(?:do i need|should i bring) (?:an?\s+)?umbrella\b",
            low_command,
        ) or re.search(
            r"^(?:rain|snow|umbrella|temperature|temp)\s+(?:today|tomorrow|tonight|later|now|right now|outside|this morning|this afternoon|this evening|this weekend|weekend|[a-z][a-z\s]{1,40})(?:[\?\.!]|$)"
            r"|^[a-z][a-z\s]{1,40}\s+(?:temperature|temp)(?:\s+(?:today|tomorrow|tonight|later|now|right now|outside|this morning|this afternoon|this evening|this weekend|weekend)|[\?\.!]|$)",
            low_command,
        ):
            weather_temporal = r"(?:today|tomorrow|tonight|later|now|right now|outside|this morning|this afternoon|this evening|this weekend|weekend)"
            # Real gap found live 2026-07-10: in a compound sentence ("check
            # the weather in seoul and then add a task to bring an
            # umbrella"), the lazy location group couldn't find its
            # temporal/punctuation/end-of-string terminator within 40 chars
            # starting at "in seoul" (the rest of the sentence has neither),
            # so the whole match failed there and `re.search` fell back to
            # matching the LATER, unrelated "to bring an umbrella" clause
            # instead -- silently asking for weather in a nonsense location.
            # Added "and"/"then" as explicit terminators so the location
            # capture can't bleed across a clause boundary.
            loc_m = re.search(rf"\b(?:in|for|at|to)\s+([a-z][a-z\s]{{1,40}}?)(?:\s+{weather_temporal}|\s+and\b|\s+then\b|[\?\.!]|$)", low_command)
            terse_loc_m = re.search(
                rf"^(?:weather|forecast|rain forecast|snow forecast|rain|snow|umbrella|temperature|temp)\s+([a-z][a-z\s]{{1,40}}?)(?:\s+{weather_temporal}|[\?\.!]|$)",
                low_command,
            )
            temporal_prefix_loc_m = re.search(
                rf"^{weather_temporal}\s+(?:weather|forecast|rain forecast|snow forecast|temperature|temp)\s+([a-z][a-z\s]{{1,40}}?)(?:[\?\.!]|$)",
                low_command,
            )
            reverse_loc_m = re.search(
                rf"^([a-z][a-z\s]{{1,40}}?)\s+(?:weather|forecast|rain forecast|snow forecast|temperature|temp)(?:\s+{weather_temporal}|[\?\.!]|$)",
                low_command,
            )
            args = {}
            if loc_m or terse_loc_m or temporal_prefix_loc_m or reverse_loc_m:
                loc = (loc_m or terse_loc_m or temporal_prefix_loc_m or reverse_loc_m).group(1).strip()
                loc = re.sub(r"^(?:tell me about|tell me|what(?:'?s| is)|how(?:'?s| is))\s+", "", loc).strip()
                loc = re.sub(rf"^{weather_temporal}\s+", "", loc).strip()
                loc = re.sub(rf"\s+{weather_temporal}$", "", loc).strip()
                loc = re.sub(rf"^{weather_temporal}$", "", loc).strip()
                if loc and loc not in {
                    "here",
                    "my area",
                    "this",
                    "current",
                    "later",
                    "now",
                    "right now",
                    "outside",
                    "tonight",
                    "this morning",
                    "this afternoon",
                    "this evening",
                    "this weekend",
                    "weekend",
                    "the",
                    "the weather",
                    "the forecast",
                    "the morning",
                    "the afternoon",
                    "tell me",
                    "tell me about",
                    "about",
                    "what is",
                    "what is the",
                    "what's",
                    "what's the",
                    "how is",
                    "how is the",
                    "how's",
                    "how's the",
                }:
                    args["location"] = loc
            return Plan(
                goal="Report the weather.",
                actions=[PlannedAction("get_weather", args, "User asked about the weather.")],
            )
        # Canonicalize only bounded Korean availability questions that the
        # existing English availability parser can represent exactly. Korean
        # clock times/ranges and compound write requests intentionally fall
        # through instead of silently checking the wrong window.
        korean_availability = re.fullmatch(
            r"(?P<day>오늘|내일)\s*"
            r"(?:(?P<part>아침|오전|오후|저녁|밤)(?:에)?)?\s*"
            r"(?P<predicate>시간\s*(?:있어|있어요|있니|있나요|돼|돼요|되니|되나요|될까)|"
            r"바빠|바빠요|바쁘니|바쁜가요)",
            low_command,
        )
        if korean_availability:
            day = "tomorrow" if korean_availability.group("day") == "내일" else "today"
            part = {
                "아침": "morning",
                "오전": "morning",
                "오후": "afternoon",
                "저녁": "evening",
                "밤": "evening",
            }.get(korean_availability.group("part") or "", "")
            predicate = korean_availability.group("predicate")
            availability = "busy" if predicate.startswith("바") else "free"
            canonical_text = " ".join(value for value in ("am I", availability, day, part) if value)
            return Plan(
                goal="Check calendar availability.",
                actions=[
                    PlannedAction(
                        "check_availability",
                        {"text": canonical_text},
                        "User asked in Korean whether they are free or busy.",
                    )
                ],
            )

        # Bilingual command-first calendar reads. Keep this deliberately narrow
        # and anchored so Korean create/update requests cannot be mistaken for
        # a read and bypass their normal parsing/approval path.
        korean_calendar_read = re.fullmatch(
            r"(?P<day>오늘|내일)\s*(?:(?P<part>아침|오전|오후|저녁|밤)(?:에)?)?\s*"
            r"(?:내\s+)?(?:일정|스케줄|캘린더)(?:을|를|은|는)?"
            r"(?:\s*(?:좀|한번))?"
            r"(?:\s*(?:알려|보여|확인해?)\s*(?:줘(?:요)?|주세요))?",
            low_command,
        )
        if korean_calendar_read:
            range_name = "tomorrow" if korean_calendar_read.group("day") == "내일" else "today"
            daypart = {
                "아침": "morning",
                "오전": "morning",
                "오후": "afternoon",
                "저녁": "evening",
                "밤": "evening",
            }.get(korean_calendar_read.group("part") or "")
            if daypart:
                range_name = f"{range_name}_{daypart}"
            return Plan(
                goal="List calendar events.",
                actions=[
                    PlannedAction(
                        "list_events",
                        {"range": range_name},
                        "User asked in Korean to read their calendar.",
                    )
                ],
            )
        korean_calendar_period_read = re.fullmatch(
            r"(?P<period>이번\s*주|다음\s*주|이번\s*달|다음\s*달|이번\s*년|다음\s*년|이번\s*주말|다음\s*주말)\s*"
            r"(?:내\s+)?(?:일정|스케줄|캘린더)(?:을|를|은|는)?"
            r"(?:\s*(?:좀|한번))?"
            r"(?:\s*(?:알려|보여|확인해?)\s*(?:줘(?:요)?|주세요))?",
            low_command,
        )
        if korean_calendar_period_read:
            range_name = {
                "이번주": "week",
                "다음주": "next_week",
                "이번달": "month",
                "다음달": "next_month",
                "이번년": "year",
                "다음년": "next_year",
                "이번주말": "weekend",
                "다음주말": "next_weekend",
            }[re.sub(r"\s+", "", korean_calendar_period_read.group("period"))]
            return Plan(
                goal="List calendar events.",
                actions=[
                    PlannedAction(
                        "list_events",
                        {"range": range_name},
                        "User asked in Korean to read their calendar.",
                    )
                ],
            )

        # Availability ("am I free at 3pm", "am I busy tomorrow afternoon").
        if re.search(
            r"\b(am i|are you|is it|am i gonna be|will i be)\b.{0,15}\b(free|busy|available|open|booked)\b"
            r"|\b(?:free|busy|available|open|booked)\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b"
            r"|\b(?:free|busy|available|open|booked)\s+(?:today|tomorrow|tonight|this\s+weekend|next\s+weekend|weekend|this\s+month|next\s+month|this\s+year|next\s+year|this\s+week|next\s+week|week|this\s+morning|this\s+afternoon|this\s+evening|this\s+night|tomorrow\s+morning|tomorrow\s+afternoon|tomorrow\s+evening|tomorrow\s+night)\b"
            r"|\b(?:free|busy|available|open|booked)\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?\b"
            r"|^(?:check\s+(?:my\s+)?|what(?:'s| is)\s+my\s+)?(?:availability|free time)(?:\s+(?:today|tomorrow|tonight|this\s+weekend|next\s+weekend|weekend|this\s+month|next\s+month|this\s+year|next\s+year|this\s+week|next\s+week|week|this\s+morning|this\s+afternoon|this\s+evening|this\s+night|tomorrow\s+morning|tomorrow\s+afternoon|tomorrow\s+evening|tomorrow\s+night))?(?:\s+(?:please|pls|thanks|thank you))?$"
            r"|\b(?:any|do i have(?: any| an?| a)?|is there(?: any| an?| a)?)\s+(?:free\s+time|openings?|slots?)\b"
            r"|\bcan i\s+(?:meet|schedule\s+(?:something|anything|a meeting)|book\s+(?:something|anything|a meeting))\b.{0,30}\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b"
            r"|\bdo i have (anything|plans|time|something)\b|\bany (plans|meetings?)\b",
            low_command,
        ):
            return Plan(
                goal="Check calendar availability.",
                actions=[PlannedAction("check_availability", {"text": text}, "User asked whether they are free.")],
            )
        # Listing the calendars themselves (their names) — needs the plural
        # "calendars" or an explicit "which/what calendars" so it doesn't steal
        # "what's on my calendar today" (that's an events query, handled below).
        if re.search(r"\b(list|show|which|what)\b.{0,30}\bcalendars\b", low_command):
            return Plan(
                goal="List Google Calendars.",
                actions=[PlannedAction("list_calendars", {}, "User asked to list calendars.")],
            )
        # Events / schedule / agenda queries, including natural date ranges.
        # Create-intent ("create/add/schedule an event ...") is NOT a list query;
        # it needs date/time parsing and is handled by create_event (see CODEX_TASKS).
        _cal_list_phrase = bool(
            re.search(r"\bon my (?:calendar|schedule|agenda)\b", low_command)
            or re.search(r"^(?:schedule|agenda)\s+(?:today|tomorrow|tonight|tomorrow\s+night|this\s+(?:morning|afternoon|evening|night|weekend|week|month|year)|next\s+(?:weekend|week|month|year)|weekend|week|month|year)\b", low_command)
            or re.search(r"^(?:calendar|schedule|agenda)\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?\b", low_command)
            or re.search(r"^(?:calendar|schedule|agenda|events)\s+(?:this\s+weekend|next\s+weekend|weekend)\b", low_command)
            or re.search(r"^(?:calendar|schedule|agenda|events)\s+(?:this\s+month|next\s+month|month)\b", low_command)
            or re.search(r"^(?:calendar|schedule|agenda|events)\s+(?:this\s+year|next\s+year|year)\b", low_command)
        )
        _cal_create = _is_calendar_create_intent(low_command) and not _cal_list_phrase
        _cal_event = re.search(
            r"\b(event|events|schedule|agenda|appointments?|meetings?)\b|\bon my calendar\b|\bwhat do i have\b|\bwhat'?s? (on|happening)\b"
            r"|\bwhat am i (?:doing|up to)\b"
            r"|\bwhat'?s? my day\b|\bhow does my day look\b"
            r"|^(?:calendar|calendar brief|show calendar|show latest calendar)(?:\s+(?:please|pls|thanks|thank you))?$"
            r"|\b(?:show|open|check|see|view)\b.{0,20}\bmy calendar\b|\bcalendar\b.{0,20}\b(today|tomorrow|tonight|this (?:morning|afternoon|evening|night|weekend|week|month|year)|next (?:weekend|week|month|year)|weekend|week|month|year|(?:next )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?)\b",
            low_command,
        )
        explicit_calendar_update = _plan_explicit_calendar_update(text, low_command)
        if explicit_calendar_update:
            return explicit_calendar_update
        fuzzy_calendar_update = _plan_fuzzy_calendar_update(low_command)
        if fuzzy_calendar_update:
            return fuzzy_calendar_update
        if _cal_create:
            from jarvis_v2.tools.nl_datetime import parse_event
            parsed = parse_event(text)
            if parsed and parsed[0] != "__need_time__":
                title, start, end = parsed
                return Plan(
                    goal="Create a calendar event.",
                    actions=[PlannedAction(
                        "create_event",
                        {"title": title, "start": start, "end": end},
                        "User asked to schedule an event with a parsed date/time.",
                    )],
                )
            if parsed and parsed[0] == "__need_time__":
                return Plan(
                    goal="Ask for the event time.",
                    actions=[PlannedAction(
                        "respond",
                        {"text": "What day and time should I set that for? (e.g. 'tomorrow at 3pm')"},
                        "Create-intent without a parseable time; ask rather than invent one.",
                    )],
                )
            # parse returned None: not actually an event, fall through to chat.
        elif _cal_event:
            args: dict[str, Any] = {}
            weekday_m = re.search(r"\b(?P<next>next\s+)?(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low_command)
            daypart_m = re.search(r"\b(?P<part>morning|afternoon|evening|night)\b", low_command)
            daypart = daypart_m.group("part") if daypart_m else ""
            if daypart == "night":
                daypart = "evening"
            if re.search(r"\btomorrow\b", low_command):
                args["range"] = "tomorrow"
            elif daypart and re.search(r"\bthis\s+(?:morning|afternoon|evening|night)\b", low_command):
                args["range"] = "today"
            elif re.search(r"\b(today|tonight)\b", low_command):
                args["range"] = "today"
            elif re.search(r"\bnext\s+week\b", low_command):
                args["range"] = "next_week"
            elif re.search(r"\bnext\s+weekend\b", low_command):
                args["range"] = "next_weekend"
            elif re.search(r"\bnext\s+month\b", low_command):
                args["range"] = "next_month"
            elif re.search(r"\bthis\s+month\b", low_command):
                args["range"] = "month"
            elif re.search(r"\bmonth\b", low_command):
                args["range"] = "month"
            elif re.search(r"\bnext\s+year\b", low_command):
                args["range"] = "next_year"
            elif re.search(r"\bthis\s+year\b", low_command):
                args["range"] = "year"
            elif re.search(r"\byear\b", low_command):
                args["range"] = "year"
            elif re.search(r"\b(?:this\s+)?weekend\b", low_command):
                args["range"] = "weekend"
            elif weekday_m:
                day = weekday_m.group("day")
                args["range"] = f"next_{day}" if weekday_m.group("next") else day
            elif re.search(r"\b(this week|week|coming days|next few days)\b", low_command):
                args["range"] = "week"
            limit_m = re.search(r"\b(?!(?:19|20)\d{2}\b)(\d+)\b", low_command)
            if limit_m:
                args["max_results"] = int(limit_m.group(1))
            if daypart and args.get("range") in {"today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "next_monday", "next_tuesday", "next_wednesday", "next_thursday", "next_friday", "next_saturday", "next_sunday"}:
                args["range"] = f"{args['range']}_{daypart}"
            elif re.search(r"\btonight\b", low_command) and args.get("range") == "today":
                args["range"] = "today_evening"
            return Plan(
                goal="List calendar events.",
                actions=[PlannedAction("list_events", args, "User asked about their schedule/events.")],
            )
        korean_email_term = r"(?:이메일|메일|받은\s*편지함)"
        korean_email_search = re.search(
            rf"{korean_email_term}(?:에서)?\s+(.{{1,120}}?)\s*(?:을|를)?\s*(?:찾아|검색)(?:해줘|해|줘)?\s*$",
            text,
        ) or re.search(
            rf"^(.{{1,120}}?)\s+{korean_email_term}(?:을|를)?\s*(?:찾아|검색)(?:해줘|해|줘)?\s*$",
            text,
        )
        if korean_email_search:
            query = korean_email_search.group(1).strip(" .")
            return Plan(
                goal="Search emails.",
                actions=[PlannedAction("search_emails", {"query": query}, "User asked to search email.")],
            )
        if re.search(
            rf"(?:최근\s*|새\s*|안\s*읽은\s*|읽지\s*않은\s*)?{korean_email_term}(?:을|를)?\s*(?:좀\s*)?(?:보여|확인|읽어|체크)",
            text,
        ):
            args = {}
            if re.search(r"(?:안\s*읽은|읽지\s*않은|새\s*(?:이메일|메일))", text):
                args["unread"] = True
            return Plan(
                goal="Read recent emails.",
                actions=[PlannedAction("read_emails", args, "User asked to check email.")],
            )
        if (
            re.search(
                r"\b(search|find)\s+(?:my\s+|recent\s+|new\s+|for\s+(?:recent\s+|new\s+)?)?(e-?mails?|mail|inbox|gmail)\b"
                r"|\b(e-?mails?|mail|inbox|gmail)\b.{0,20}\b(from|about|containing|with subject|subject)\b"
                r"|^what\s+e-?mails?\s+did\s+.{2,80}?\s+(?:send|sent)(?:\s+me)?\??$",
                low_command,
            )
            and not re.search(r"\b(read|show|open)\b.{0,30}\b(e-?mail|mail|gmail)\b", low_command)
        ):
            args: dict[str, Any] = {}
            sender_m = re.search(r"\bfrom\s+(.{2,80}?)(?=\s+(?:about|subject|with subject|containing)\b|$)", text, re.IGNORECASE)
            sent_by_m = re.search(r"^what\s+e-?mails?\s+did\s+(.{2,80}?)\s+(?:send|sent)(?:\s+me)?\??$", text, re.IGNORECASE)
            subject_m = re.search(r"\b(?:subject|with subject)\s+(.{2,120})", text, re.IGNORECASE)
            about_m = re.search(r"\b(?:about|containing|for)\s+(.{2,120})", text, re.IGNORECASE)
            if sender_m:
                args["sender"] = _clean_email_sender_hint(sender_m.group(1))
            elif sent_by_m:
                args["sender"] = _clean_email_sender_hint(sent_by_m.group(1))
            if subject_m:
                args["subject"] = subject_m.group(1).strip(" .")
            elif about_m:
                args["query"] = about_m.group(1).strip(" .")
            limit_m = re.search(r"\b(?!(?:19|20)\d{2}\b)(\d+)\b", low_command)
            if limit_m:
                args["limit"] = int(limit_m.group(1))
            return Plan(
                goal="Search emails.",
                actions=[PlannedAction("search_emails", args, "User asked to search email.")],
            )
        if re.search(r"\b(read|show|open)\b.{0,30}\b(e-?mail|mail|gmail)\b.{0,20}\b(body|content|message|from|about|subject)\b|\bread the (?:e-?mail|mail)\b", low_command):
            args = {}
            sender_m = re.search(r"\bfrom\s+(.{2,80}?)(?=\s+(?:about|subject|with subject|containing)\b|$)", text, re.IGNORECASE)
            subject_m = re.search(r"\b(?:subject|with subject)\s+(.{2,120})", text, re.IGNORECASE)
            about_m = re.search(r"\b(?:about|containing)\s+(.{2,120})", text, re.IGNORECASE)
            index_m = re.search(r"\b(?:email|message)\s+(\d+)\b|\b#(\d+)\b", low_command)
            if sender_m:
                args["sender"] = sender_m.group(1).strip(" .")
            if subject_m:
                args["subject"] = subject_m.group(1).strip(" .")
            elif about_m:
                args["query"] = about_m.group(1).strip(" .")
            if index_m:
                args["index"] = int(index_m.group(1) or index_m.group(2))
            return Plan(
                goal="Read an email body.",
                actions=[PlannedAction("read_email_body", args, "User asked to read an email body.")],
            )
        if re.search(
            r"\b(any |new |unread |check |read |show |latest )\b.{0,20}\b(e-?mails?|mail|inbox|gmail)\b"
            r"|\b(e-?mails?|mail|inbox)\b.{0,15}\b(today|new|unread)\b"
            r"|^(?:e-?mails?|mail|inbox|gmail)(?:\s+(?:please|pls|thanks|thank you))?$",
            low_command,
        ):
            args = {}
            if re.search(r"\b(unread|new)\b", low_command):
                args["unread"] = True
            limit_m = re.search(r"\b(?!(?:19|20)\d{2}\b)(\d+)\b", low_command)
            if limit_m:
                args["limit"] = int(limit_m.group(1))
            return Plan(
                goal="Read recent emails.",
                actions=[PlannedAction("read_emails", args, "User asked to check email.")],
            )
        if re.search(r"\b(read|show|check|recent|last)\b.{0,30}\b(imessages?|messages?|texts?|sms)\b", low_command):
            limit_m = re.search(r"\b(?!(?:19|20)\d{2}\b)(\d+)\b", low_command)
            args = {"limit": int(limit_m.group(1))} if limit_m else {}
            return Plan(
                goal="Read recent iMessages.",
                actions=[PlannedAction("read_recent_imessages", args, "User asked to read recent iMessages.")],
            )

        # Fallback: if the message would go to the model and it starts with a
        # "Jarvis" vocative ("Jarvis what time is it"), retry once with the name
        # stripped so the habit doesn't bypass deterministic tool routing (and
        # let the model fabricate). Specific "jarvis <cmd>" rules already matched
        # above, so this only affects messages that were headed to the model.
        if not _vocative_stripped:
            stripped = re.sub(
                r"^(?:hey\s+|ok\s+|okay\s+|hi\s+|hello\s+)?jarvis[\s,:]+",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
            if stripped and stripped.lower() != low:
                retry = self.plan(
                    stripped,
                    allow_risky_natural_dispatch=allow_risky_natural_dispatch,
                    _vocative_stripped=True,
                    _polite_stripped=_polite_stripped,
                )
                if retry.actions:
                    return retry

        return Plan(
            goal="Respond conversationally.",
            actions=[],
            needs_model=True,
            notes="This should route to a model planner once configured.",
        )
