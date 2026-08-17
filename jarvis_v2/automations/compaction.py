"""Conversation compaction — the "Memory Trees" pattern.

The messages table grows forever (700+ rows) but nothing ever distills it:
recent_messages(limit=30) means anything older than a few days is effectively
forgotten. This job compresses old conversation rows into durable summary
memories (the same memories the Daily/Morning Brief already surfaces), so
long-term context compounds instead of scrolling away.

Conversation history reaches Ollama only when its destination is allowed and
both OLLAMA_NO_CLOUD=1 and
JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=1 permit personal context for a
non-cloud-alias model. These settings express intent and explicit consent; they
do not prove on-device execution. Other approved model routes are blocked
unless JARVIS_ALLOW_REMOTE_COMPACTION=1 separately permits old-history
processing. The job performs no user-facing sends or approval actions. Progress
is tracked in the scheduled job's metadata (last_compacted_id), so each run
picks up where the previous one stopped and nothing is ever summarized twice.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from jarvis_v2.memory.store import (
    ConversationCompactionCommit,
    MemoryRecord,
    MemoryStore,
    history_message_content_digest,
    history_source_digest,
)

COMPACTION_JOB_NAME = "Conversation Compaction"
COMPACTION_JOB_TYPE = "conversation_compaction"
COMPACTION_INTERVAL_MINUTES = 1440

# Leave the freshest conversation alone — it is still live context.
KEEP_RECENT_HOURS = 48
# Don't bother the model for a trickle; wait until there's a real batch.
MIN_BATCH_MESSAGES = 20
# Upper bound per run keeps a single tick's model work predictable.
MAX_MESSAGES_PER_RUN = 240
CHUNK_MESSAGES = 40
# Summarization needs far more headroom than the 8s interactive chat timeout.
MODEL_TIMEOUT_SECONDS = 180
MAX_DIGEST_CHARS = 4000

NOTHING_TO_COMPACT = "Nothing to compact"

SUMMARY_SYSTEM_PROMPT = """You compress an assistant's conversation log into durable notes.
From the transcript below, extract ONLY information worth remembering weeks later:
- stable facts about the user or their projects
- preferences and standing instructions
- decisions made and their reasons
- open threads or commitments that were not finished

Write terse markdown bullets. Ignore greetings, chit-chat, transient status
chatter (volume, uptime, load, battery), and anything already obvious.
NEVER include passwords, passcodes, API keys, tokens, one-time codes, or any
other credential value — if one appears in the transcript, note the event
without the value (e.g. "generated a password for X"). Never invent details
that are not in the transcript. If the transcript contains nothing durable,
reply exactly: NOTHING_DURABLE"""

# Belt-and-braces on top of the prompt rule: any digest line that mentions a
# credential keyword gets its value-looking tokens masked before the digest
# is persisted anywhere. A live run proved the model will happily copy
# generated passwords into the digest without this.
_CREDENTIAL_KEYWORD_RE = re.compile(
    r"password|passcode|passphrase|api[\s_-]?key|token|secret|credential|one[\s-]?time code|otp",
    re.IGNORECASE,
)
_VALUE_TOKEN_RE = re.compile(r"`[^`]{6,}`|\S{8,}")
_OPAQUE_TOKEN_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CompactionSummarizer:
    """A summarizer plus its exact, auditable history destination contract."""

    callback: Callable[[str], str]
    provider: str
    model_identifier: str
    destination_class: str
    explicit_consent_satisfied: bool
    remote_history_egress: bool

    def __call__(self, transcript: str) -> str:
        return self.callback(transcript)


@dataclass(frozen=True)
class _CompactionRoute:
    provider: str
    model_identifier: str
    destination_class: str
    explicit_consent_satisfied: bool
    remote_history_egress: bool
    allowed: bool


@dataclass(frozen=True)
class _CompactionFrontier:
    rows: tuple[sqlite3.Row, ...]
    withheld_rows: tuple[sqlite3.Row, ...]


def _route_from_declared_summarizer(summarizer: CompactionSummarizer) -> _CompactionRoute:
    if not callable(summarizer.callback):
        raise TypeError("compaction summarizer callback must be callable")
    values = (summarizer.provider, summarizer.model_identifier, summarizer.destination_class)
    if any(
        type(value) is not str
        or len(value) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value) is None
        for value in values
    ) or any(marker in summarizer.destination_class for marker in ("*", "?", "[", "]")):
        raise ValueError("compaction summarizer destination contract is malformed")
    if type(summarizer.explicit_consent_satisfied) is not bool:
        raise TypeError("compaction summarizer consent must be an exact boolean")
    if type(summarizer.remote_history_egress) is not bool:
        raise TypeError("compaction summarizer egress class must be an exact boolean")
    return _CompactionRoute(
        provider=summarizer.provider,
        model_identifier=summarizer.model_identifier,
        destination_class=summarizer.destination_class,
        explicit_consent_satisfied=summarizer.explicit_consent_satisfied,
        remote_history_egress=summarizer.remote_history_egress,
        allowed=summarizer.explicit_consent_satisfied,
    )


def _default_route(config) -> _CompactionRoute:
    from jarvis_v2.agent.model_provider import (
        normalized_model_provider,
        ollama_local_only_policy,
        resolve_ollama_destination,
    )

    provider = normalized_model_provider(getattr(config, "model_provider", "ollama"))
    model = str(getattr(config, "chat_model", "") or "")
    if provider == "openai":
        consent = bool(getattr(config, "allow_remote_conversation_compaction", False))
        return _CompactionRoute(
            provider=provider,
            model_identifier=model,
            destination_class="external_provider",
            explicit_consent_satisfied=consent,
            remote_history_egress=True,
            allowed=consent,
        )
    if provider == "ollama":
        destination = resolve_ollama_destination()
        policy = ollama_local_only_policy(model)
        consent = bool(policy.unverified_context_consent_allowed)
        destination_class = (
            "loopback_daemon_unverified" if destination.allowed else "blocked_destination"
        )
        return _CompactionRoute(
            provider=provider,
            model_identifier=model,
            destination_class=destination_class,
            explicit_consent_satisfied=consent,
            remote_history_egress=True,
            allowed=bool(destination.allowed and policy.personal_context_allowed),
        )
    return _CompactionRoute(
        provider=str(provider or "invalid"),
        model_identifier=model or "invalid",
        destination_class="invalid_provider",
        explicit_consent_satisfied=False,
        remote_history_egress=True,
        allowed=False,
    )


def _route_matches_epoch(
    route: _CompactionRoute,
    epoch: dict[str, object],
    *,
    expected_epoch_id: str | None = None,
) -> bool:
    return bool(
        (expected_epoch_id is None or str(epoch.get("epoch_id") or "") == expected_epoch_id)
        and str(epoch.get("provider") or "") == route.provider
        and str(epoch.get("model_identifier") or "") == route.model_identifier
        and str(epoch.get("destination_class") or "") == route.destination_class
        and epoch.get("explicit_consent_satisfied")
        is route.explicit_consent_satisfied
    )


def _lineage_is_complete(row: sqlite3.Row) -> bool:
    state = row["lineage_state"]
    source_count = row["source_count"]
    source_digest = row["source_digest"]
    if type(state) is not str or type(source_count) is not int or source_count < 0:
        return False
    digest_valid = type(source_digest) is str and _OPAQUE_TOKEN_RE.fullmatch(source_digest) is not None
    if state == "direct_current":
        return source_count == 0 and source_digest is None
    if state in {"personal_derived", "tool_derived", "mixed_restricted"}:
        return source_count > 0 and digest_valid
    return False


def _row_is_eligible(
    row: sqlite3.Row,
    *,
    route: _CompactionRoute,
    epoch_id: str,
) -> bool:
    role = row["role"]
    lineage_token = row["lineage_token"]
    if type(role) is not str or role not in {"user", "assistant"}:
        return False
    if row["provenance_role"] != role or not _lineage_is_complete(row):
        return False
    content_digest = row["content_digest"]
    if (
        type(content_digest) is not str
        or _OPAQUE_TOKEN_RE.fullmatch(content_digest) is None
        or history_message_content_digest(role, str(row["content"])) != content_digest
    ):
        return False
    if row["lineage_state"] not in {"direct_current", "personal_derived"}:
        return False
    if type(lineage_token) is not str or _OPAQUE_TOKEN_RE.fullmatch(lineage_token) is None:
        return False
    if row["policy_epoch_id"] != epoch_id or row["active_epoch_id"] != epoch_id:
        return False
    if row["destination_class"] != route.destination_class:
        return False
    if bool(row["epoch_consent"]) is not route.explicit_consent_satisfied:
        return False
    if type(row["remote_eligible"]) is not int or row["remote_eligible"] != 1:
        return False
    return True


def _redact_secrets(digest: str) -> str:
    lines = []
    for line in digest.splitlines():
        if _CREDENTIAL_KEYWORD_RE.search(line):
            def _mask(match: re.Match) -> str:
                token = match.group(0)
                # keep ordinary words/paths readable; mask value-like tokens
                # (anything with digits or punctuation beyond - _ / .)
                bare = token.strip("`'\",;:")
                if re.fullmatch(r"[A-Za-z][A-Za-z\-_/.]*", bare):
                    return token
                return "[redacted]"

            line = _VALUE_TOKEN_RE.sub(_mask, line)
        lines.append(line)
    return "\n".join(lines)


def fetch_compactable_messages(
    store: MemoryStore,
    after_id: int,
    *,
    now: datetime | None = None,
    limit: int = MAX_MESSAGES_PER_RUN,
    route: _CompactionRoute | None = None,
    policy_epoch_id: str | None = None,
) -> list[sqlite3.Row]:
    """Return the maximal oldest-first prefix with effective provenance."""
    return list(
        _fetch_compaction_frontier(
            store,
            after_id,
            now=now,
            limit=limit,
            route=route,
            policy_epoch_id=policy_epoch_id,
        ).rows
    )


def _fetch_compaction_frontier(
    store: MemoryStore,
    after_id: int,
    *,
    now: datetime | None = None,
    limit: int = MAX_MESSAGES_PER_RUN,
    route: _CompactionRoute | None = None,
    policy_epoch_id: str | None = None,
) -> _CompactionFrontier:
    cutoff = (now or datetime.now(timezone.utc).replace(tzinfo=None)) - timedelta(hours=KEEP_RECENT_HOURS)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.astimezone(timezone.utc).replace(tzinfo=None)
    epoch = store.read_active_history_epoch()
    effective_route = route or _CompactionRoute(
        provider=str(epoch["provider"]),
        model_identifier=str(epoch["model_identifier"]),
        destination_class=str(epoch["destination_class"]),
        explicit_consent_satisfied=bool(epoch["explicit_consent_satisfied"]),
        remote_history_egress=bool(epoch["explicit_consent_satisfied"]),
        allowed=True,
    )
    epoch_id = str(policy_epoch_id or epoch["epoch_id"])
    if epoch_id != str(epoch["epoch_id"]) or not _route_matches_epoch(effective_route, epoch):
        return _CompactionFrontier((), ())
    with store.connect() as conn:
        candidates = conn.execute(
            """
            SELECT message.id, message.session_id, message.role, message.content,
                   message.created_at, provenance.role AS provenance_role,
                   provenance.lineage_state, provenance.source_count,
                   provenance.source_digest, provenance.policy_epoch_id,
                   provenance.lineage_token, provenance.content_digest,
                   provenance.destination_class,
                   provenance.remote_eligible,
                   active.active_epoch_id,
                   epoch.explicit_consent_satisfied AS epoch_consent
            FROM messages AS message
            LEFT JOIN message_provenance AS provenance
              ON provenance.message_id = message.id
            LEFT JOIN active_history_policy_epoch AS active ON active.singleton_id = 1
            LEFT JOIN history_policy_epochs AS epoch
              ON epoch.epoch_id = provenance.policy_epoch_id
            WHERE message.id > ?
            ORDER BY message.id ASC LIMIT ?
            """,
            (int(after_id), int(limit)),
        ).fetchall()
    rows: list[sqlite3.Row] = []
    withheld_rows: list[sqlite3.Row] = []
    mode: str | None = None
    for row in candidates:
        try:
            created = datetime.fromisoformat(str(row["created_at"] or "").replace("Z", "+00:00"))
            if created.tzinfo is not None:
                created = created.astimezone(timezone.utc).replace(tzinfo=None)
        except (TypeError, ValueError):
            break
        if created >= cutoff:
            break
        eligible = _row_is_eligible(row, route=effective_route, epoch_id=epoch_id)
        row_mode = "eligible" if eligible else "withheld"
        if mode is None:
            mode = row_mode
        if row_mode != mode:
            break
        if eligible:
            rows.append(row)
        else:
            withheld_rows.append(row)
    return _CompactionFrontier(tuple(rows), tuple(withheld_rows))


def _transcript(rows: list[sqlite3.Row]) -> str:
    lines = []
    for row in rows:
        content = " ".join(str(row["content"] or "").split())
        if not content:
            continue
        lines.append(f"{row['role']}: {content[:600]}")
    return "\n".join(lines)


def _default_summarize(
    config,
    route: _CompactionRoute | None = None,
) -> CompactionSummarizer:
    route = route or _default_route(config)
    def summarize(transcript: str) -> str:
        from jarvis_v2.agent.model_provider import generate_model_text, provider_output_token_limit

        return generate_model_text(
            provider=route.provider,
            model=route.model_identifier,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            timeout_seconds=MODEL_TIMEOUT_SECONDS,
            max_output_tokens=provider_output_token_limit(
                config.model_provider,
                local_output_tokens=512,
                openai_output_tokens=config.openai_max_output_tokens,
            ),
            reasoning_effort=config.chat_reasoning_effort,
        )

    return CompactionSummarizer(
        callback=summarize,
        provider=route.provider,
        model_identifier=route.model_identifier,
        destination_class=route.destination_class,
        explicit_consent_satisfied=route.explicit_consent_satisfied,
        remote_history_egress=route.remote_history_egress,
    )


def _default_compaction_allowed(config) -> bool:
    return _default_route(config).allowed


def _mirror_marker(first_id: int, last_id: int) -> str:
    return f"jarvis-conversation-compaction:v1:{first_id}:{last_id}"


def _repair_pending_mirrors(store: MemoryStore, vault) -> None:
    for row in store.pending_conversation_compaction_mirrors(limit=20):
        marker = _mirror_marker(int(row["first_message_id"]), int(row["last_message_id"]))
        try:
            vault.append_daily_once(
                marker,
                "Jarvis Conversation Digest",
                f"## {row['title']}\n\n{row['body']}",
            )
        except Exception:
            continue
        try:
            store.mark_conversation_compaction_mirror_completed(int(row["batch_id"]))
        except Exception:
            continue


def _commit_result_or_raise(result: ConversationCompactionCommit) -> int:
    if result.status == "committed":
        return result.watermark
    if result.status == "stale_watermark":
        return result.watermark
    if result.status == "lease_lost":
        raise RuntimeError("conversation compaction lease was lost before commit")
    if result.status == "frontier_mismatch":
        raise RuntimeError("conversation compaction message frontier changed before commit")
    if result.status == "legacy_memory_changed":
        raise RuntimeError("conversation compaction legacy memory changed before commit")
    raise RuntimeError("conversation compaction commit failed")


def build_conversation_compaction(
    store: MemoryStore,
    vault,
    config,
    *,
    last_compacted_id: int = 0,
    now: datetime | None = None,
    summarize: CompactionSummarizer | None = None,
    job_id: int | None = None,
    lease_token: str = "",
) -> tuple[str, int | None]:
    """Compact one batch of old messages into a summary memory.

    Injected summarizers must use CompactionSummarizer so their destination
    and consent contract is explicit before any history is released.
    Returns (report, new_last_compacted_id). new_last_compacted_id is None
    when nothing was compacted (caller keeps the old watermark).
    """
    try:
        requested_watermark = max(0, int(last_compacted_id))
    except (TypeError, ValueError):
        requested_watermark = 0
    authoritative_watermark = store.conversation_compaction_watermark()
    recovered_watermark = authoritative_watermark if authoritative_watermark != requested_watermark else None
    _repair_pending_mirrors(store, vault)

    if summarize is None:
        route = _default_route(config)
        declared_summarizer = _default_summarize(config, route)
    elif isinstance(summarize, CompactionSummarizer):
        try:
            route = _route_from_declared_summarizer(summarize)
        except (TypeError, ValueError):
            return (
                "Compaction blocked: the summarizer destination contract is malformed; "
                "no history was disclosed and the watermark was not advanced.",
                recovered_watermark,
            )
        declared_summarizer = summarize
    else:
        return (
            "Compaction blocked: injected summarizers must declare an exact destination contract; "
            "no history was disclosed and the watermark was not advanced.",
            recovered_watermark,
        )

    try:
        epoch = store.read_active_history_epoch()
    except Exception:
        return (
            "Compaction blocked: the active history policy epoch is unavailable; "
            "no history was disclosed and the watermark was not advanced.",
            recovered_watermark,
        )
    if not route.allowed or not _route_matches_epoch(route, epoch):
        return (
            "Compaction skipped: old conversation history is not approved for this model route or "
            "the active history policy epoch does not exactly match its provider, model, destination, "
            "and consent. No conversation text was sent and the compaction watermark was not advanced. "
            "Validated loopback Ollama processing requires both OLLAMA_NO_CLOUD=1 and "
            "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=1; other approved routes require "
            "JARVIS_ALLOW_REMOTE_COMPACTION=1. An invalid provider, stale epoch, disallowed destination, "
            "and cloud-model aliases remain blocked.",
            recovered_watermark,
        )

    epoch_id = str(epoch["epoch_id"])
    frontier = _fetch_compaction_frontier(
        store,
        authoritative_watermark,
        now=now,
        route=route,
        policy_epoch_id=epoch_id,
    )
    withheld_rows = list(frontier.withheld_rows)
    if withheld_rows:
        first_withheld = int(withheld_rows[0]["id"])
        last_withheld = int(withheld_rows[-1]["id"])
        skipped = store.commit_conversation_compaction(
            expected_after_id=authoritative_watermark,
            first_message_id=first_withheld,
            last_message_id=last_withheld,
            record=None,
            job_id=job_id,
            lease_token=lease_token,
            eligible_source_count=0,
            withheld_source_count=len(withheld_rows),
            source_provenance_digest=None,
            policy_epoch_id=None,
        )
        watermark = _commit_result_or_raise(skipped)
        return (
            f"Compaction advanced over {len(withheld_rows)} local-only messages "
            f"#{first_withheld}–#{last_withheld} without disclosure.",
            watermark,
        )
    rows = list(frontier.rows)
    if len(rows) < MIN_BATCH_MESSAGES:
        return (
            f"{NOTHING_TO_COMPACT}: {len(rows)} uncompacted messages "
            f"(waiting for {MIN_BATCH_MESSAGES}).",
            recovered_watermark,
        )

    first_id, last_id = int(rows[0]["id"]), int(rows[-1]["id"])
    source_digest = history_source_digest(
        [
            (
                int(row["id"]),
                str(row["lineage_token"]),
                str(row["content_digest"]),
            )
            for row in rows
        ]
    )
    span = f"{str(rows[0]['created_at'])[:10]} → {str(rows[-1]['created_at'])[:10]}"
    title = f"Conversation digest {span} (messages #{first_id}–#{last_id})"

    legacy_rows = store.legacy_conversation_digest_rows(title)
    if legacy_rows:
        bodies = {str(row["body"] or "") for row in legacy_rows}
        if len(bodies) > 1:
            return (
                f"Compaction blocked: conflicting legacy digests exist for messages #{first_id}–#{last_id}; "
                "the watermark was not advanced.",
                recovered_watermark,
            )
        adopted = store.commit_conversation_compaction(
            expected_after_id=authoritative_watermark,
            first_message_id=first_id,
            last_message_id=last_id,
            record=None,
            job_id=job_id,
            lease_token=lease_token,
            legacy_memory_id=int(legacy_rows[0]["id"]),
            legacy_expected_title=title,
            legacy_expected_body=str(legacy_rows[0]["body"] or ""),
            eligible_source_count=len(rows),
            withheld_source_count=0,
            source_provenance_digest=source_digest,
            policy_epoch_id=epoch_id,
        )
        watermark = _commit_result_or_raise(adopted)
        return (
            f"Recovered legacy conversation digest coverage for messages #{first_id}–#{last_id} ({span}).",
            watermark,
        )

    transcript_chunks = [
        transcript
        for start in range(0, len(rows), CHUNK_MESSAGES)
        if (transcript := _transcript(rows[start : start + CHUNK_MESSAGES]))
    ]
    if not transcript_chunks:
        committed = store.commit_conversation_compaction(
            expected_after_id=authoritative_watermark,
            first_message_id=first_id,
            last_message_id=last_id,
            record=None,
            job_id=job_id,
            lease_token=lease_token,
            eligible_source_count=len(rows),
            withheld_source_count=0,
            source_provenance_digest=source_digest,
            policy_epoch_id=epoch_id,
        )
        watermark = _commit_result_or_raise(committed)
        return (
            f"Compacted messages #{first_id}–#{last_id} ({span}): nothing durable found.",
            watermark,
        )

    try:
        receipt_id = store.prepare_history_disclosure_receipt(
            session_id="conversation-compaction",
            provider=route.provider,
            destination_class=route.destination_class,
            source_count=len(rows),
            source_digest=source_digest,
            policy_epoch_id=epoch_id,
        )
    except Exception:
        return (
            "Compaction blocked: a durable history disclosure receipt could not be prepared; "
            "no history was disclosed and the watermark was not advanced.",
            recovered_watermark,
        )

    def finalize_receipt(state: str) -> None:
        store.finalize_history_disclosure_receipt(receipt_id, state)

    bullet_blocks: list[str] = []
    attempted = False
    fence_entered = False
    try:
        with store.history_disclosure_egress_fence(
            receipt_id=receipt_id,
            policy_epoch_id=epoch_id,
            provider=route.provider,
            model_identifier=route.model_identifier,
            destination_class=route.destination_class,
            source_count=len(rows),
            source_digest=source_digest,
        ):
            fence_entered = True
            current_watermark = store.conversation_compaction_watermark()
            if current_watermark != authoritative_watermark:
                finalize_receipt("blocked")
                return (
                    "Compaction blocked: another worker advanced the watermark before dispatch; "
                    "no history was disclosed.",
                    current_watermark,
                )
            if not store.validate_conversation_compaction_source(
                first_message_id=first_id,
                last_message_id=last_id,
                source_count=len(rows),
                source_digest=source_digest,
                policy_epoch_id=epoch_id,
            ):
                finalize_receipt("blocked")
                return (
                    "Compaction blocked: message content changed before dispatch; "
                    "the watermark was not advanced.",
                    recovered_watermark,
                )

            for transcript in transcript_chunks:
                try:
                    attempted = True
                    summary = declared_summarizer.callback(transcript)
                    summary = (summary or "").strip()
                except Exception as exc:
                    try:
                        finalize_receipt("uncertain" if attempted else "blocked")
                    except Exception:
                        pass
                    return (
                        f"Compaction skipped: model unavailable ({type(exc).__name__}); "
                        "the disclosure receipt is uncertain and the watermark was not advanced.",
                        recovered_watermark,
                    )
                if summary and "NOTHING_DURABLE" not in summary:
                    bullet_blocks.append(summary)

            record = None
            if bullet_blocks:
                digest = _redact_secrets("\n".join(bullet_blocks))[:MAX_DIGEST_CHARS].strip()
                record = MemoryRecord(
                    category="conversation-digest",
                    title=title,
                    body=digest,
                    source="conversation_compaction",
                    confidence=0.8,
                )
            try:
                committed = store.commit_conversation_compaction(
                    expected_after_id=authoritative_watermark,
                    first_message_id=first_id,
                    last_message_id=last_id,
                    record=record,
                    job_id=job_id,
                    lease_token=lease_token,
                    eligible_source_count=len(rows),
                    withheld_source_count=0,
                    source_provenance_digest=source_digest,
                    policy_epoch_id=epoch_id,
                )
                watermark = _commit_result_or_raise(committed)
            except Exception:
                try:
                    finalize_receipt("uncertain")
                except Exception:
                    pass
                raise
            finalize_receipt("confirmed")
    except Exception:
        if fence_entered:
            raise
        try:
            finalize_receipt("blocked")
        except Exception:
            pass
        return (
            "Compaction blocked: the history authority fence could not be acquired or validated; "
            "no history was disclosed and the watermark was not advanced.",
            recovered_watermark,
        )
    if record is None:
        return (
            f"Compacted messages #{first_id}–#{last_id} ({span}): nothing durable found.",
            watermark,
        )
    _repair_pending_mirrors(store, vault)

    return (
        f"Compacted {len(rows)} messages #{first_id}–#{last_id} ({span}) into memory: {title}",
        watermark,
    )
