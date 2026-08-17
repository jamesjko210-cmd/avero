from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import unicodedata

from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.agent.model_provider import (
    ModelProviderError,
    OLLAMA_UNVERIFIED_PERSONAL_CONTEXT_ENV,
    SUPPORTED_MODEL_PROVIDERS,
    generate_model_text,
    normalized_model_provider,
    ollama_local_only_policy,
    provider_output_token_limit,
    resolve_ollama_destination,
    safe_model_usage_receipt,
)


SYSTEM_PROMPT = """You are J.A.R.V.I.S., the operator's personal AI.

You are conversational like ChatGPT or Claude: thoughtful, direct, warm, and able to talk naturally.
You are not only a command runner. You can discuss ideas, ask clarifying questions, reason through problems,
and remember useful context.
You are the operator's Jarvis. Do not call the operator Tony Stark, do not roleplay Marvel scenes, and do not invent a shared
history from fiction. The name J.A.R.V.I.S. is an inspiration for tone, not a fictional identity to imitate.

HARD RULES (never break these):
1. You are in conversation mode and CANNOT perform actions here. You cannot set timers, send messages or
   email, create or change calendar events, run commands, control the computer, or change any state. NEVER say
   or imply you did any of these ("I've set...", "I've sent...", "Done", "I scheduled..."). Those are always false.
   If the user wants an action, tell them you'll need to run it as a command — do not claim it happened.
2. NEVER invent the current time, date, day, weather, battery, location, or any other live or real-time value.
   You have no clock or sensors in conversation mode. If there is no supplied verified Jarvis tool result, say
   the value should be checked with the relevant tool rather than guessing. When a supplied "Jarvis tool result"
   contains the value, you may answer from that result and should not deny that the separate tool lane checked it.
3. Do not invent personal facts, projects, memories, preferences, relationships, files, emails, calendar events,
   numbers, statistics, or private context. Use only the supplied profile, preference, memory, skill, or
   conversation context. If context is thin, say so instead of filling gaps. Do not repackage background
   memory about Jarvis's own development (e.g. "the harness is being built", "conversational mode was added")
   as if it were the operator's personal tasks, projects, or to-do list — those are two different things. If you are
   not certain the operator actually has open tasks or projects, do not say he does; say you don't have that visibility
   here and suggest a command like "show my tasks" instead of hedging with "if I recall correctly." A supplied
   verified "Jarvis tool result" listing tasks is that visibility: report it accurately, do not claim it was
   unavailable, and do not recommend rerunning the same task command.
4. NEVER be hostile to anyone. No insults, mockery, contempt, threats, or aggression — toward the operator, toward
   people he mentions, or toward anyone in message content you help draft or reply to. This holds even if a
   message you receive is rude or provocative, and even if asked to "roast" someone: stay respectful, decline
   the hostility, keep the warmth. Firm honesty is fine; cruelty never is.
5. Conversation history can contain assistant entries beginning with "Jarvis tool result (...)". These are
   verified results produced earlier by Jarvis's separate tool lane, not actions performed in conversation
   mode. Treat those results as authoritative for follow-up questions: summarize or refer to them directly
   instead of denying that Jarvis checked them. Tool names in those entries are internal Jarvis tools, never
   Ollama CLI commands, and you must not tell the operator to run them through Ollama.

Style:
- concise by default, deeper when the user asks for depth
- no corporate assistant voice
- honest about uncertainty
- speak like a capable collaborator, not a menu of features
"""


ACTION_RISK_HINTS = {
    "computer-control": ("computer", "screen", "click", "mouse", "keyboard", "type", "desktop"),
    "shell/code": ("terminal", "shell", "python", "script", "install", "run"),
    "files": ("file", "write", "edit", "delete", "move", "rename"),
    "personal-data": ("clipboard", "email", "calendar", "text messages", "contact", "browser profile"),
    "external-side-effect": ("send", "email me", "post", "purchase", "book", "pay", "share"),
}

LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
SAFE_MODEL_NAME_RE = re.compile(r"[^A-Za-z0-9._:/-]+")
WORD_TOKEN_RE = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}][^\W_]+)*", re.UNICODE)
SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")
MAX_MEMORY_CONTEXT_CHARS = 4000
MAX_MEMORY_CATEGORY_CHARS = 96
MAX_MEMORY_TITLE_CHARS = 240
MAX_MEMORY_BODY_CHARS = 320
MAX_PREFERENCE_CONTEXT_CHARS = 4000
MAX_PREFERENCE_CONTEXT_ITEMS = 20
MAX_PREFERENCE_CATEGORY_CHARS = 96
MAX_PREFERENCE_KEY_CHARS = 160
MAX_PREFERENCE_VALUE_CHARS = 320
MAX_SKILL_CONTEXT_CHARS = 4000
MAX_SKILL_NAME_CHARS = 160
MAX_SKILL_TRIGGER_CHARS = 240
MAX_SKILL_BODY_CHARS = 480
MAX_SKILL_SEARCH_FIELD_CHARS = 1000
MAX_HISTORY_GROUNDING_MESSAGE_CHARS = 4000
MAX_STORED_GROUNDING_CHARS = 12_000
HISTORY_PROVENANCE_KEY = "_jarvis_history_provenance"
HISTORY_PROVENANCE_VERSION = 1
_PROCESS_GENERATION = secrets.token_hex(16)


def _word_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(WORD_TOKEN_RE.findall(normalized))


def _contains_token_phrase(text: str, phrase: str) -> bool:
    text_tokens = _word_tokens(text)
    phrase_tokens = _word_tokens(phrase)
    phrase_length = len(phrase_tokens)
    if not phrase_length or phrase_length > len(text_tokens):
        return False
    return any(
        text_tokens[index : index + phrase_length] == phrase_tokens
        for index in range(len(text_tokens) - phrase_length + 1)
    )


def _contains_any_token_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_token_phrase(text, phrase) for phrase in phrases)


def _safe_preview_text(value: object) -> str:
    if value is None:
        text = ""
    else:
        try:
            text = str(value)
        except Exception:
            text = "<unreadable>"
    return LOCAL_PATH_RE.sub("<local-path>", text.strip())


def _safe_model_name(value: object) -> str:
    text = _safe_preview_text(value)
    text = SAFE_MODEL_NAME_RE.sub("", text).strip(".:/-")
    return text[:80] if text else "configured-model"


def _ollama_recovery_hint(exc: Exception, model: str) -> str:
    """Return content-free fallback guidance for an unexpected adapter error."""
    if isinstance(exc, ModelProviderError):
        return exc.recovery_hint
    safe_model = _safe_model_name(model)
    return (
        f"{safe_model} is unavailable for an unrecognized reason ({type(exc).__name__}). "
        f"Start Ollama and run `model status` in Jarvis; if the model isn't pulled, run `ollama pull {safe_model}` outside Jarvis."
    )


def _model_recovery_hint(exc: Exception | None, model: str, provider: str) -> str:
    selected = normalized_model_provider(provider)
    if selected not in SUPPORTED_MODEL_PROVIDERS:
        if isinstance(exc, ModelProviderError):
            return exc.recovery_hint
        return "Set JARVIS_MODEL_PROVIDER to ollama or openai, then retry `model routing status`."
    if selected == "openai":
        if isinstance(exc, ModelProviderError):
            return exc.recovery_hint
        return "Run `model routing status` to check OpenAI configuration before retrying."
    if isinstance(exc, ModelProviderError):
        return exc.recovery_hint
    if exc is not None:
        return _ollama_recovery_hint(exc, model)
    return (
        f"Start Ollama and run `model status` in Jarvis; if the model is missing, run "
        f"`ollama pull {model}` outside Jarvis."
    )


def _invalid_provider_error() -> ModelProviderError:
    return ModelProviderError(
        "model_provider_invalid",
        "Set JARVIS_MODEL_PROVIDER to ollama or openai, then retry `model routing status`.",
    )


def _ollama_context_policy(
    *,
    destination_allowed: bool,
    cloud_model_alias: bool,
    no_cloud_requested: bool,
    consent_allowed: bool,
) -> str:
    if not destination_allowed:
        return "blocked_destination"
    if cloud_model_alias:
        return "blocked_cloud_model"
    if not no_cloud_requested:
        return "local_provider_stateless"
    if not consent_allowed:
        return "local_provider_unverified_context_consent_required"
    return "local_provider_unverified_context_consent_granted"


def _safe_row_value(row: object, key: str, default: str = "") -> str:
    try:
        if isinstance(row, dict):
            value = row.get(key, default)
        else:
            value = row[key]  # type: ignore[index]
    except Exception:
        return default
    text = _safe_preview_text(value)
    return text if text else default


_REMOTE_ELIGIBLE_SKILL_ORIGINS = frozenset({"user_authored"})


def _skill_has_durable_remote_origin(row: object) -> bool:
    """Only explicit, bounded durable origins may enter a provider request."""
    origin = _safe_row_value(row, "origin", "")
    return origin in _REMOTE_ELIGIBLE_SKILL_ORIGINS


def _disclosure_item_digest(kind: str, material: object) -> str:
    payload = json.dumps(
        {
            "kind": kind,
            "material": material,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _HistoryDisclosureCallbackBundle:
    prepare: Callable[[dict[str, object]], object]
    validate: Callable[[object], bool]
    finalize: Callable[[object, str], str | None]
    fence: Callable[[object], AbstractContextManager[None]]


@dataclass(frozen=True)
class _HistoryPolicyDecision:
    signature: str
    epoch: int
    epoch_id: str
    fingerprint: str
    provider: str
    model: str
    destination_binding: str
    destination_class: str
    provider_valid: bool
    destination_allowed: bool
    remote_context_allowed: bool
    stored_context_consent: bool
    stored_context_allowed: bool
    cloud_model_alias: bool
    ollama_destination: object | None
    ollama_local_only: object
    provider_environ: dict[str, str]


@dataclass(frozen=True)
class _HistoryProvenance:
    version: int
    process_generation: str
    session_generation: str
    turn_generation: int
    role: str
    policy_epoch: int
    policy_epoch_id: str
    policy_fingerprint: str
    provider: str
    model: str
    destination_binding: str
    destination_class: str
    future_history_allowed: bool
    personal_derived: bool
    source_count: int
    source_digest: str | None
    lineage_token: str
    seal: str


@dataclass(frozen=True)
class _HistoryEligibility:
    messages: tuple[dict[str, str], ...]
    retained: int
    withheld: int
    reason_counts: dict[str, int]
    lineage_digest: str
    disclosure_digests: tuple[str, ...]


@dataclass(frozen=True)
class _ChatContextSnapshot:
    profile: str
    preferences: str
    memories: str
    skills: str
    profile_state: str
    preference_state: str
    memory_state: str
    skill_state: str
    recent_history: tuple[dict[str, str], ...]
    history_eligibility: _HistoryEligibility
    profile_source_digest: str
    memory_source_digests: tuple[str, ...]
    profile_disclosure_digests: tuple[str, ...]
    preference_disclosure_digests: tuple[str, ...]
    memory_disclosure_digests: tuple[str, ...]
    skill_disclosure_digests: tuple[str, ...]
    memory_ids: tuple[int, ...]

    def source_states(self) -> dict[str, str]:
        return {
            "profile": self.profile_state,
            "preferences": self.preference_state,
            "memory": self.memory_state,
            "skills": self.skill_state,
        }


@dataclass
class ChatBrain:
    model: str
    store: MemoryStore
    vault: ObsidianVault | None = None
    history: list[dict[str, object]] = field(default_factory=list)
    max_history_messages: int = 16
    model_timeout_seconds: float = 2.5
    max_reply_tokens: int = 300
    provider: str = "ollama"
    reasoning_effort: str = "medium"
    openai_max_output_tokens: int = 25_000
    allow_remote_personal_context: bool = False
    last_turn_metadata: dict[str, object] = field(default_factory=dict)
    _history_disclosure_callbacks: _HistoryDisclosureCallbackBundle | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _session_generation: str = field(default_factory=lambda: secrets.token_hex(16), init=False, repr=False)
    _seal_key: bytes = field(default_factory=lambda: secrets.token_bytes(32), init=False, repr=False)
    _policy_epoch: int = field(default=0, init=False, repr=False)
    _policy_signature: str | None = field(default=None, init=False, repr=False)
    _policy_mutation_generation: int = field(default=0, init=False, repr=False)
    _provider_environment_generation: int = field(default=0, init=False, repr=False)
    _provider_environment_identity: str | None = field(default=None, init=False, repr=False)
    _turn_generation: int = field(default=0, init=False, repr=False)
    _history_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _policy_state_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    def _get_model(self) -> str:
        lock = getattr(self, "_policy_state_lock", None)
        if lock is None:
            return self._policy_model
        with lock:
            return self._policy_model

    def _set_model(self, value: str) -> None:
        lock = getattr(self, "_policy_state_lock", None)
        if lock is None:
            self._policy_model = value
            return
        with lock:
            if self._policy_model != value:
                self._policy_mutation_generation += 1
            self._policy_model = value

    def _get_provider(self) -> str:
        lock = getattr(self, "_policy_state_lock", None)
        if lock is None:
            return self._policy_provider
        with lock:
            return self._policy_provider

    def _set_provider(self, value: str) -> None:
        lock = getattr(self, "_policy_state_lock", None)
        if lock is None:
            self._policy_provider = value
            return
        with lock:
            if self._policy_provider != value:
                self._policy_mutation_generation += 1
            self._policy_provider = value

    def _get_allow_remote_personal_context(self) -> bool:
        lock = getattr(self, "_policy_state_lock", None)
        if lock is None:
            return self._policy_allow_remote_personal_context
        with lock:
            return self._policy_allow_remote_personal_context

    def _set_allow_remote_personal_context(self, value: bool) -> None:
        lock = getattr(self, "_policy_state_lock", None)
        if lock is None:
            self._policy_allow_remote_personal_context = value
            return
        with lock:
            if self._policy_allow_remote_personal_context != value:
                self._policy_mutation_generation += 1
            self._policy_allow_remote_personal_context = value

    def install_history_disclosure_callbacks(
        self,
        *,
        prepare: Callable[[dict[str, object]], object],
        validate: Callable[[object], bool],
        finalize: Callable[[object, str], str | None],
        fence: Callable[[object], AbstractContextManager[None]],
    ) -> None:
        if not all(callable(callback) for callback in (prepare, validate, finalize, fence)):
            raise TypeError("history disclosure callback bundle must be complete")
        bundle = _HistoryDisclosureCallbackBundle(
            prepare=prepare,
            validate=validate,
            finalize=finalize,
            fence=fence,
        )
        self._history_disclosure_callbacks = bundle

    def clear_history_disclosure_callbacks(self) -> None:
        self._history_disclosure_callbacks = None

    def _recent_history(self) -> list[dict[str, object]]:
        limit = max(0, int(self.max_history_messages))
        return self.history[-limit:] if limit else []

    @staticmethod
    def _provider_environment_snapshot() -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for key in (
            "OPENAI_API_KEY",
            "OLLAMA_HOST",
            "OLLAMA_NO_CLOUD",
            OLLAMA_UNVERIFIED_PERSONAL_CONTEXT_ENV,
        ):
            try:
                value = os.environ.get(key)
            except (KeyError, OSError):
                continue
            if value is not None:
                snapshot[key] = value
        return snapshot

    def _observe_provider_environment(self) -> tuple[dict[str, str], int]:
        with self._policy_state_lock:
            snapshot = self._provider_environment_snapshot()
            identity = hashlib.sha256(
                json.dumps(
                    snapshot,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
            if self._provider_environment_identity is None:
                self._provider_environment_identity = identity
            elif identity != self._provider_environment_identity:
                self._provider_environment_generation += 1
                self._provider_environment_identity = identity
            return snapshot, self._provider_environment_generation

    def _policy_material(self) -> tuple[str, dict[str, object]]:
        with self._policy_state_lock:
            provider = self.provider
            model = self.model
            allow_remote_personal_context = self.allow_remote_personal_context
            policy_mutation_generation = self._policy_mutation_generation
            (
                provider_environ,
                provider_environment_generation,
            ) = self._observe_provider_environment()
        selected_provider = normalized_model_provider(provider)
        ollama_destination = (
            resolve_ollama_destination(provider_environ)
            if selected_provider == "ollama"
            else None
        )
        ollama_local_only = ollama_local_only_policy(model, provider_environ)
        if selected_provider == "openai":
            destination_identity = "openai:api.openai.com:v1:responses"
            destination_class = "external_provider"
            destination_allowed = True
        elif selected_provider == "ollama" and ollama_destination is not None:
            destination_identity = "|".join(
                (
                    ollama_destination.base_url,
                    ollama_destination.source,
                    ollama_destination.scheme,
                    ollama_destination.address_family,
                    str(ollama_destination.port),
                    ollama_destination.diagnostic,
                )
            )
            destination_allowed = bool(ollama_destination.allowed)
            destination_class = (
                "loopback_daemon_unverified"
                if destination_allowed
                else "blocked_destination"
            )
        else:
            destination_identity = "invalid-provider"
            destination_class = "invalid_provider"
            destination_allowed = False
        destination_binding = hashlib.sha256(
            destination_identity.encode("utf-8")
        ).hexdigest()
        remote_context_allowed = bool(
            selected_provider == "openai"
            and allow_remote_personal_context is True
        )
        stored_context_consent = bool(
            allow_remote_personal_context is True
            if selected_provider == "openai"
            else (
                selected_provider == "ollama"
                and ollama_local_only.unverified_context_consent_allowed
            )
        )
        stored_context_allowed = bool(
            remote_context_allowed
            if selected_provider == "openai"
            else (
                selected_provider == "ollama"
                and destination_allowed
                and ollama_local_only.personal_context_allowed
            )
        )
        material = {
            "version": HISTORY_PROVENANCE_VERSION,
            "policy_mutation_generation": policy_mutation_generation,
            "provider_environment_generation": provider_environment_generation,
            "provider": selected_provider,
            "model": str(model),
            "destination_binding": destination_binding,
            "destination_class": destination_class,
            "destination_allowed": destination_allowed,
            "openai_stored_context_consent": allow_remote_personal_context is True,
            "ollama_no_cloud_configured": ollama_local_only.configured,
            "ollama_no_cloud_valid": ollama_local_only.valid,
            "ollama_no_cloud_requested": ollama_local_only.requested,
            "ollama_context_consent_configured": (
                ollama_local_only.unverified_context_consent_configured
            ),
            "ollama_context_consent_valid": (
                ollama_local_only.unverified_context_consent_valid
            ),
            "ollama_context_consent_allowed": (
                ollama_local_only.unverified_context_consent_allowed
            ),
            "ollama_cloud_model_alias": ollama_local_only.cloud_model_alias,
        }
        signature = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return signature, {
            "provider": selected_provider,
            "model": str(model),
            "destination_binding": destination_binding,
            "destination_class": destination_class,
            "destination_allowed": destination_allowed,
            "provider_valid": selected_provider in SUPPORTED_MODEL_PROVIDERS,
            "remote_context_allowed": remote_context_allowed,
            "stored_context_consent": stored_context_consent,
            "stored_context_allowed": stored_context_allowed,
            "cloud_model_alias": bool(ollama_local_only.cloud_model_alias),
            "ollama_destination": ollama_destination,
            "ollama_local_only": ollama_local_only,
            "provider_environ": provider_environ,
        }

    def _history_policy_decision(self, *, commit: bool) -> _HistoryPolicyDecision:
        with self._policy_state_lock:
            signature, values = self._policy_material()
            changed = signature != self._policy_signature
            epoch = self._policy_epoch + (1 if changed else 0)
            if commit:
                self._policy_epoch = epoch
                self._policy_signature = signature
            fingerprint = hashlib.sha256(signature.encode("utf-8")).hexdigest()
            epoch_id = hmac.new(
                self._seal_key,
                f"{self._session_generation}|{epoch}|{fingerprint}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return _HistoryPolicyDecision(
                signature=signature,
                epoch=epoch,
                epoch_id=epoch_id,
                fingerprint=fingerprint,
                **values,
            )

    def _revalidate_history_policy(self, decision: _HistoryPolicyDecision) -> None:
        signature, _ = self._policy_material()
        if signature != decision.signature:
            raise ModelProviderError(
                "chat_policy_changed_before_dispatch",
                "Retry the chat turn after the provider privacy settings stop changing.",
            )

    def _provenance_payload(
        self,
        role: str,
        content: str,
        provenance: _HistoryProvenance,
    ) -> bytes:
        payload = {
            "version": provenance.version,
            "process_generation": provenance.process_generation,
            "session_generation": provenance.session_generation,
            "turn_generation": provenance.turn_generation,
            "role": role,
            "content": content,
            "policy_epoch": provenance.policy_epoch,
            "policy_epoch_id": provenance.policy_epoch_id,
            "policy_fingerprint": provenance.policy_fingerprint,
            "provider": provenance.provider,
            "model": provenance.model,
            "destination_binding": provenance.destination_binding,
            "destination_class": provenance.destination_class,
            "future_history_allowed": provenance.future_history_allowed,
            "personal_derived": provenance.personal_derived,
            "source_count": provenance.source_count,
            "source_digest": provenance.source_digest,
            "lineage_token": provenance.lineage_token,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def _make_history_entry(
        self,
        *,
        role: str,
        content: str,
        decision: _HistoryPolicyDecision,
        turn_generation: int,
        personal_derived: bool,
        source_count: int,
        source_digest: str | None,
    ) -> dict[str, object]:
        unsealed = _HistoryProvenance(
            version=HISTORY_PROVENANCE_VERSION,
            process_generation=_PROCESS_GENERATION,
            session_generation=self._session_generation,
            turn_generation=turn_generation,
            role=role,
            policy_epoch=decision.epoch,
            policy_epoch_id=decision.epoch_id,
            policy_fingerprint=decision.fingerprint,
            provider=decision.provider,
            model=decision.model,
            destination_binding=decision.destination_binding,
            destination_class=decision.destination_class,
            future_history_allowed=decision.stored_context_allowed,
            personal_derived=personal_derived,
            source_count=max(0, int(source_count)),
            source_digest=source_digest,
            lineage_token=secrets.token_hex(24),
            seal="",
        )
        seal = hmac.new(
            self._seal_key,
            self._provenance_payload(role, content, unsealed),
            hashlib.sha256,
        ).hexdigest()
        provenance = _HistoryProvenance(
            **{
                **unsealed.__dict__,
                "seal": seal,
            }
        )
        return {
            "role": role,
            "content": content,
            HISTORY_PROVENANCE_KEY: provenance,
        }

    def _history_entry_reason(
        self,
        entry: object,
        decision: _HistoryPolicyDecision,
    ) -> str:
        if type(entry) is not dict:
            return "malformed_entry"
        role = entry.get("role")
        content = entry.get("content")
        if role not in {"user", "assistant"} or type(role) is not str:
            return "invalid_role"
        if type(content) is not str:
            return "malformed_content"
        if HISTORY_PROVENANCE_KEY not in entry:
            return "legacy_unmarked"
        provenance = entry.get(HISTORY_PROVENANCE_KEY)
        if type(provenance) is not _HistoryProvenance:
            return "malformed_provenance"
        if provenance.version != HISTORY_PROVENANCE_VERSION or provenance.role != role:
            return "malformed_provenance"
        if (
            provenance.process_generation != _PROCESS_GENERATION
            or provenance.session_generation != self._session_generation
        ):
            return "generation_mismatch"
        try:
            sealed_payload = self._provenance_payload(role, content, provenance)
        except Exception:
            return "malformed_provenance"
        expected_seal = hmac.new(
            self._seal_key,
            sealed_payload,
            hashlib.sha256,
        ).hexdigest()
        if type(provenance.seal) is not str or not hmac.compare_digest(
            provenance.seal, expected_seal
        ):
            return "invalid_seal"
        if provenance.policy_epoch != decision.epoch or provenance.policy_epoch_id != decision.epoch_id:
            return "policy_epoch_mismatch"
        if (
            provenance.policy_fingerprint != decision.fingerprint
            or provenance.provider != decision.provider
            or provenance.model != decision.model
            or provenance.destination_binding != decision.destination_binding
            or provenance.destination_class != decision.destination_class
        ):
            return "policy_fingerprint_mismatch"
        if provenance.future_history_allowed is not True:
            return "future_history_not_granted"
        if provenance.personal_derived is True:
            if (
                type(provenance.source_count) is not int
                or provenance.source_count <= 0
                or type(provenance.source_digest) is not str
                or SHA256_HEX_RE.fullmatch(provenance.source_digest) is None
            ):
                return "malformed_lineage"
        elif (
            provenance.personal_derived is not False
            or provenance.source_count != 0
            or provenance.source_digest is not None
        ):
            return "malformed_lineage"
        return "eligible"

    def _eligible_history(
        self,
        decision: _HistoryPolicyDecision,
    ) -> _HistoryEligibility:
        retained = self._recent_history()
        eligible: list[dict[str, str]] = []
        lineage_sources: list[dict[str, str | None]] = []
        disclosure_digests: list[str] = []
        reason_counts: dict[str, int] = {}
        for entry in retained:
            reason = self._history_entry_reason(entry, decision)
            if reason != "eligible":
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                continue
            provenance = entry[HISTORY_PROVENANCE_KEY]
            if type(provenance) is not _HistoryProvenance:
                reason_counts["malformed_provenance"] = (
                    reason_counts.get("malformed_provenance", 0) + 1
                )
                continue
            content = str(entry["content"])
            if len(content) > MAX_HISTORY_GROUNDING_MESSAGE_CHARS:
                reason_counts["content_too_large"] = (
                    reason_counts.get("content_too_large", 0) + 1
                )
                continue
            eligible.append(
                {"role": str(entry["role"]), "content": content}
            )
            lineage_sources.append(
                {
                    "lineage_token": provenance.lineage_token,
                    "source_digest": provenance.source_digest,
                }
            )
            disclosure_digests.append(
                _disclosure_item_digest(
                    "eligible_history_item_v1",
                    {
                        "role": str(entry["role"]),
                        "content": content,
                        "lineage_token": provenance.lineage_token,
                        "source_digest": provenance.source_digest,
                    },
                )
            )
        lineage_digest = hashlib.sha256(
            json.dumps(
                {
                    "kind": "eligible_history_sources_v1",
                    "policy_epoch_id": decision.epoch_id,
                    "policy_fingerprint": decision.fingerprint,
                    "lineage_sources": lineage_sources,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return _HistoryEligibility(
            messages=tuple(eligible),
            retained=len(retained),
            withheld=len(retained) - len(eligible),
            reason_counts=reason_counts,
            lineage_digest=lineage_digest,
            disclosure_digests=tuple(disclosure_digests),
        )

    def _derived_source_digest(
        self,
        *,
        decision: _HistoryPolicyDecision,
        source_kinds: list[str],
        history_lineage_digest: str | None,
        durable_source_digests: tuple[str, ...] = (),
    ) -> str | None:
        normalized_kinds = sorted(set(source_kinds))
        if history_lineage_digest is not None:
            normalized_kinds.append("eligible_history")
        normalized_durable_digests = sorted(set(durable_source_digests))
        if not normalized_kinds and not normalized_durable_digests:
            return None
        material = {
            "kind": "chat_personal_source_lineage_v1",
            "policy_epoch_id": decision.epoch_id,
            "policy_fingerprint": decision.fingerprint,
            "source_kinds": normalized_kinds,
            "eligible_history_source_digest": history_lineage_digest,
            "durable_source_digests": normalized_durable_digests,
        }
        return hashlib.sha256(
            json.dumps(
                material,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _history_message_provenance_metadata(
        self,
        entry: dict[str, object],
    ) -> dict[str, object]:
        provenance = entry.get(HISTORY_PROVENANCE_KEY)
        if type(provenance) is not _HistoryProvenance:
            raise RuntimeError("new chat history entry lost its provenance")
        return {
            "role": provenance.role,
            "lineage_state": (
                "personal_derived" if provenance.personal_derived else "direct"
            ),
            "policy_epoch_id": provenance.policy_epoch_id,
            "policy_fingerprint": provenance.policy_fingerprint,
            "provider": provenance.provider,
            "destination_class": provenance.destination_class,
            "future_history_allowed": provenance.future_history_allowed,
            "source_count": provenance.source_count,
            "source_digest": provenance.source_digest,
            "lineage_token_digest": hashlib.sha256(
                provenance.lineage_token.encode("ascii")
            ).hexdigest(),
        }

    def _current_message_only_messages(self, user_input: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "History disclosure receipt policy for this turn: prior conversation history and "
                    "stored profile, preferences, memory, and saved skills were kept local. Answer only "
                    "from the current user message and general knowledge; do not invent personal context."
                ),
            },
            {"role": "user", "content": user_input},
        ]

    def _stored_context_disclosure_material(
        self,
        *,
        decision: _HistoryPolicyDecision,
        context: _ChatContextSnapshot,
        context_sources: list[str],
        history_message_count: int,
        availability_instruction: str,
    ) -> tuple[int, str]:
        source_items = {
            "profile": context.profile_disclosure_digests,
            "preferences": context.preference_disclosure_digests,
            "memory": context.memory_disclosure_digests,
            "skills": context.skill_disclosure_digests,
        }
        if not availability_instruction:
            raise RuntimeError("stored context source health disclosure is missing")
        item_digests = [
            _disclosure_item_digest(
                "source_health_availability_instruction_v1",
                {"content": availability_instruction},
            )
        ]
        for source in context_sources:
            source_digests = source_items.get(source, ())
            if not source_digests:
                raise RuntimeError("stored context disclosure source lost item custody")
            item_digests.extend(source_digests)
        if not 0 <= history_message_count <= len(
            context.history_eligibility.disclosure_digests
        ):
            raise RuntimeError("stored context disclosure history count is invalid")
        item_digests.extend(
            context.history_eligibility.disclosure_digests[:history_message_count]
        )
        if not item_digests or any(
            SHA256_HEX_RE.fullmatch(digest) is None for digest in item_digests
        ):
            raise RuntimeError("stored context disclosure material is incomplete")
        source_digest = hashlib.sha256(
            json.dumps(
                {
                    "kind": "chat_stored_context_disclosure_v2",
                    "policy_epoch_id": decision.epoch_id,
                    "policy_fingerprint": decision.fingerprint,
                    "item_digests": item_digests,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return len(item_digests), source_digest

    def _bounded_stored_grounding(
        self,
        context: _ChatContextSnapshot,
        availability_instruction: str,
    ) -> tuple[list[dict[str, str]], list[str], int]:
        remaining = MAX_STORED_GROUNDING_CHARS - len(availability_instruction)
        if remaining < 0:
            raise RuntimeError("stored context source health exceeded grounding budget")
        messages: list[dict[str, str]] = []
        included_sources: list[str] = []
        for source, value in (
            ("profile", context.profile),
            ("preferences", context.preferences),
            ("memory", context.memories),
            ("skills", context.skills),
        ):
            if not value or len(value) > remaining:
                continue
            messages.append({"role": "system", "content": value})
            included_sources.append(source)
            remaining -= len(value)
        history_count = 0
        for message in context.recent_history:
            content = message["content"]
            if len(content) > remaining:
                break
            messages.append(dict(message))
            history_count += 1
            remaining -= len(content)
        return messages, included_sources, history_count

    @staticmethod
    def _source_health_availability_instruction(
        unavailable_sources: list[str],
    ) -> str:
        if unavailable_sources:
            return (
                "Personal context availability for this turn: these local sources could not be fully checked: "
                + ", ".join(unavailable_sources)
                + ". Do not infer that their contents are empty, and do not invent missing personal context. "
                "This current-turn status supersedes any older availability notice in conversation history."
            )
        return (
            "Personal context availability for this turn: all configured local sources were readable. "
            "Any older availability notice in conversation history is historical and does not describe this turn."
        )

    def _memory_profile_ownership_is_current(
        self,
        context: _ChatContextSnapshot,
    ) -> bool:
        if not context.memory_ids:
            return True
        try:
            profile_owned = self.store.profile_owned_memory_ids(context.memory_ids)
        except Exception:
            return False
        return type(profile_owned) is frozenset and not profile_owned

    @staticmethod
    def _history_disclosure_handle_is_active(
        handle: object,
        validator: Callable[[object], bool],
    ) -> bool:
        try:
            return validator(handle) is True
        except Exception:
            return False

    def _trim_history(self) -> None:
        limit = max(0, int(self.max_history_messages))
        if not limit:
            self.history.clear()
        elif len(self.history) > limit:
            del self.history[:-limit]

    def respond(self, user_input: str) -> str:
        with self._history_lock:
            return self._respond_locked(user_input)

    def record_tool_turn(
        self,
        user_input: str,
        assistant_response: str,
        tool_names: tuple[str, ...],
    ) -> bool:
        """Retain a verified read-only tool exchange for consented conversational follow-ups."""
        clean_tool_names = tuple(
            sorted(
                {
                    str(name).strip()
                    for name in tool_names
                    if isinstance(name, str) and str(name).strip()
                }
            )
        )
        user_content = str(user_input)
        tool_content = (
            f"Jarvis tool result ({', '.join(clean_tool_names)}):\n{assistant_response}"
        )
        if (
            not clean_tool_names
            or not user_content
            or len(user_content) > MAX_HISTORY_GROUNDING_MESSAGE_CHARS
            or len(tool_content) > MAX_HISTORY_GROUNDING_MESSAGE_CHARS
        ):
            return False
        with self._history_lock:
            decision = self._history_policy_decision(commit=True)
            if not decision.stored_context_allowed:
                return False
            source_digest = hashlib.sha256(
                json.dumps(
                    {
                        "kind": "verified_read_only_tool_turn_v1",
                        "policy_epoch_id": decision.epoch_id,
                        "policy_fingerprint": decision.fingerprint,
                        "tool_names": clean_tool_names,
                        "response_digest": hashlib.sha256(
                            assistant_response.encode("utf-8")
                        ).hexdigest(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self._turn_generation += 1
            user_entry = self._make_history_entry(
                role="user",
                content=user_content,
                decision=decision,
                turn_generation=self._turn_generation,
                personal_derived=False,
                source_count=0,
                source_digest=None,
            )
            assistant_entry = self._make_history_entry(
                role="assistant",
                content=tool_content,
                decision=decision,
                turn_generation=self._turn_generation,
                personal_derived=True,
                source_count=len(clean_tool_names),
                source_digest=source_digest,
            )
            self.history.extend((user_entry, assistant_entry))
            self._trim_history()
            return True

    def _respond_locked(self, user_input: str) -> str:
        decision = self._history_policy_decision(commit=True)
        context = self._capture_context_snapshot(user_input, decision=decision)
        loop = self._classify_loop(user_input, context=context, decision=decision)
        profile_context = context.profile
        preference_context = context.preferences
        memory_context = context.memories
        skill_context = context.skills
        selected_provider = decision.provider
        share_stored_context = decision.stored_context_allowed
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        actual_context_sources: list[str] = []
        actual_history_messages = 0
        unavailable_sources = [
            source
            for source, state in context.source_states().items()
            if state == "unavailable"
        ]
        if share_stored_context:
            availability_instruction = self._source_health_availability_instruction(
                unavailable_sources
            )
            (
                stored_messages,
                actual_context_sources,
                actual_history_messages,
            ) = self._bounded_stored_grounding(context, availability_instruction)
            messages.extend(stored_messages)
        else:
            availability_instruction = (
                "Remote personal-context policy for this turn: stored profile, preferences, memory, saved skills, "
                "and prior conversation history were deliberately kept local and were not included in this model "
                "request. Answer only from the current user message and general knowledge; do not invent personal context."
            )
        messages.append({"role": "system", "content": availability_instruction})
        messages.append({"role": "user", "content": user_input})
        loop["history_messages_in_request"] = actual_history_messages
        loop["personal_context_sources_in_request"] = actual_context_sources
        disclosure_callbacks = self._history_disclosure_callbacks
        loop["history_disclosure_prepare_status"] = (
            "not_configured" if disclosure_callbacks is None else "not_required"
        )
        loop["history_disclosure_finalize_status"] = "not_required"
        receipt_handle: object | None = None

        if loop["reply_path"] == "grounded_memory":
            answer = self._grounded_memory_response(
                user_input,
                profile_context=profile_context,
                preference_context=preference_context,
                memory_context=memory_context,
                skill_context=skill_context,
                context_states=context.source_states(),
            )
            route_metadata = self._turn_metadata(
                loop, "grounded_memory", model_error="", decision=decision
            )
        elif loop["reply_path"] == "safety_preflight_guidance":
            answer = self._safety_preflight_response(user_input, loop["risk_signals"])
            route_metadata = self._turn_metadata(
                loop, "safety_preflight_guidance", model_error="", decision=decision
            )
        else:
            model_usage: dict[str, object] = {}
            model_call_attempted = False
            try:
                if selected_provider not in SUPPORTED_MODEL_PROVIDERS:
                    raise _invalid_provider_error()

                def invoke_model(request_messages: list[dict[str, str]]) -> str:
                    nonlocal model_call_attempted
                    with self._policy_state_lock:
                        dispatch_decision = self._history_policy_decision(commit=False)
                        if dispatch_decision.signature != decision.signature:
                            raise ModelProviderError(
                                "chat_policy_changed_before_dispatch",
                                "Retry the chat turn after the provider privacy settings stop changing.",
                            )
                        with self.store.profile_memory_ownership_egress_fence(
                            context.memory_ids
                        ) as profile_owned:
                            with self._profile_snapshot_evidence_fence(
                                context,
                                required="profile" in actual_context_sources,
                            ) as profile_is_current:
                                memory_is_disclosed = "memory" in actual_context_sources
                                ownership_is_current = True
                                if memory_is_disclosed:
                                    ownership_is_current = bool(
                                        not profile_owned
                                        if type(profile_owned) is frozenset
                                        else self._memory_profile_ownership_is_current(context)
                                    )
                                if memory_is_disclosed and not ownership_is_current:
                                    raise ModelProviderError(
                                        "chat_memory_profile_ownership_changed_before_dispatch",
                                        "Retry the chat turn after profile memory ownership stops changing.",
                                    )
                                if not profile_is_current:
                                    raise ModelProviderError(
                                        "chat_profile_changed_before_dispatch",
                                        "Retry the chat turn after profile updates stop changing.",
                                    )
                                model_output_tokens = provider_output_token_limit(
                                    dispatch_decision.provider,
                                    local_output_tokens=self.max_reply_tokens,
                                    openai_output_tokens=self.openai_max_output_tokens,
                                )
                                self._revalidate_history_policy(dispatch_decision)
                                if (
                                    "memory" in actual_context_sources
                                    and not self._memory_profile_ownership_is_current(context)
                                ):
                                    raise ModelProviderError(
                                        "chat_memory_profile_ownership_changed_before_dispatch",
                                        "Retry the chat turn after profile memory ownership stops changing.",
                                    )
                                model_call_attempted = True
                                return generate_model_text(
                                    provider=dispatch_decision.provider,
                                    model=dispatch_decision.model,
                                    messages=request_messages,
                                    timeout_seconds=self.model_timeout_seconds,
                                    temperature=0.6,
                                    max_output_tokens=model_output_tokens,
                                    reasoning_effort=self.reasoning_effort,
                                    keep_alive="30m",
                                    environ=dispatch_decision.provider_environ,
                                    usage_metadata=model_usage,
                                )

                self._revalidate_history_policy(decision)
                disclosure_required = share_stored_context
                if disclosure_required:
                    if disclosure_callbacks is not None:
                        source_count, source_digest = self._stored_context_disclosure_material(
                            decision=decision,
                            context=context,
                            context_sources=actual_context_sources,
                            history_message_count=actual_history_messages,
                            availability_instruction=availability_instruction,
                        )
                        prepare_payload = {
                            "session_generation": self._session_generation,
                            "policy_epoch_id": decision.epoch_id,
                            "policy_fingerprint": decision.fingerprint,
                            "provider": decision.provider,
                            "destination_class": decision.destination_class,
                            "source_count": source_count,
                            "source_digest": source_digest,
                        }
                        try:
                            receipt_handle = disclosure_callbacks.prepare(prepare_payload)
                        except Exception:
                            receipt_handle = None
                    if (
                        receipt_handle is None
                        or disclosure_callbacks is None
                        or not self._history_disclosure_handle_is_active(
                            receipt_handle, disclosure_callbacks.validate
                        )
                    ):
                        messages = self._current_message_only_messages(user_input)
                        actual_context_sources = []
                        actual_history_messages = 0
                        loop["history_messages_in_request"] = 0
                        loop["personal_context_sources_in_request"] = []
                        loop["history_disclosure_prepare_status"] = "failed_stripped"
                    else:
                        loop["history_disclosure_prepare_status"] = "prepared"
                self._revalidate_history_policy(decision)
                if (
                    receipt_handle is not None
                    and disclosure_callbacks is not None
                    and not self._history_disclosure_handle_is_active(
                        receipt_handle, disclosure_callbacks.validate
                    )
                ):
                    raise ModelProviderError(
                        "chat_policy_changed_before_dispatch",
                        "Retry the chat turn after the provider privacy settings stop changing.",
                    )
                if receipt_handle is None:
                    answer = invoke_model(messages)
                else:
                    try:
                        assert disclosure_callbacks is not None
                        with disclosure_callbacks.fence(receipt_handle):
                            self._revalidate_history_policy(decision)
                            if not self._history_disclosure_handle_is_active(
                                receipt_handle, disclosure_callbacks.validate
                            ):
                                raise ModelProviderError(
                                    "chat_policy_changed_before_dispatch",
                                    "Retry the chat turn after the provider privacy settings stop changing.",
                                )
                            try:
                                answer = invoke_model(messages)
                            except Exception:
                                disclosure_outcome = (
                                    "uncertain" if model_call_attempted else "blocked"
                                )
                                finalized = disclosure_callbacks.finalize(
                                    receipt_handle, disclosure_outcome
                                )
                                loop["history_disclosure_finalize_status"] = (
                                    finalized
                                    if finalized in {"blocked", "uncertain"}
                                    else disclosure_outcome
                                )
                                receipt_handle = None
                                raise
                            finalized = disclosure_callbacks.finalize(
                                receipt_handle, "confirmed"
                            )
                            loop["history_disclosure_finalize_status"] = (
                                finalized
                                if finalized in {"confirmed", "blocked", "uncertain"}
                                else "uncertain"
                            )
                            receipt_handle = None
                    except Exception:
                        if model_call_attempted:
                            raise
                        if receipt_handle is not None and disclosure_callbacks is not None:
                            try:
                                finalized = disclosure_callbacks.finalize(
                                    receipt_handle, "blocked"
                                )
                                loop["history_disclosure_finalize_status"] = (
                                    finalized
                                    if finalized in {"blocked", "uncertain"}
                                    else "blocked"
                                )
                            except Exception:
                                loop["history_disclosure_finalize_status"] = (
                                    "blocked_finalize_failed"
                                )
                        receipt_handle = None
                        messages = self._current_message_only_messages(user_input)
                        actual_context_sources = []
                        actual_history_messages = 0
                        loop["history_messages_in_request"] = 0
                        loop["personal_context_sources_in_request"] = []
                        loop["history_disclosure_prepare_status"] = "failed_stripped"
                        answer = invoke_model(messages)
                route_metadata = self._turn_metadata(
                    loop,
                    "model",
                    model_error="",
                    model_usage=model_usage,
                    model_call_attempted=model_call_attempted,
                    decision=decision,
                )
            except Exception as exc:
                if receipt_handle is not None and disclosure_callbacks is not None:
                    disclosure_outcome = "uncertain" if model_call_attempted else "blocked"
                    try:
                        finalized = disclosure_callbacks.finalize(
                            receipt_handle, disclosure_outcome
                        )
                        loop["history_disclosure_finalize_status"] = (
                            finalized
                            if finalized in {"blocked", "uncertain"}
                            else disclosure_outcome
                        )
                    except Exception:
                        loop["history_disclosure_finalize_status"] = disclosure_outcome
                answer = self._fallback_response(
                    user_input,
                    profile_context=profile_context,
                    preference_context=preference_context,
                    memory_context=memory_context,
                    skill_context=skill_context,
                    exc=exc,
                    model=decision.model,
                    provider=decision.provider,
                    model_call_attempted=model_call_attempted,
                )
                route_metadata = self._turn_metadata(
                    loop,
                    "fallback",
                    model_error=(
                        exc.diagnostic
                        if isinstance(exc, ModelProviderError)
                        else "model_unavailable"
                    ),
                    model_exception_type=type(exc).__name__,
                    model_usage=model_usage,
                    model_call_attempted=model_call_attempted,
                    decision=decision,
                )
            if unavailable_sources:
                answer += self._ordinary_context_availability_note(unavailable_sources)

        local_source_kinds = [
            source_kind
            for source_kind, value in (
                ("local_profile", profile_context),
                ("local_preferences", preference_context),
                ("local_memory", memory_context),
                ("local_ordinary_skill", skill_context),
            )
            if value
        ]
        request_source_kinds = [
            {
                "profile": "local_profile",
                "preferences": "local_preferences",
                "memory": "local_memory",
                "skills": "local_ordinary_skill",
            }[source]
            for source in actual_context_sources
            if source in {"profile", "preferences", "memory", "skills"}
        ]
        history_was_consumed = False
        if route_metadata["source"] == "model":
            assistant_source_kinds = request_source_kinds
            history_was_consumed = bool(actual_history_messages)
        elif route_metadata["source"] == "fallback":
            assistant_source_kinds = local_source_kinds
            history_was_consumed = bool(
                route_metadata.get("model_call_attempted")
                and actual_history_messages
            )
        elif route_metadata["source"] == "grounded_memory":
            assistant_source_kinds = local_source_kinds
        else:
            assistant_source_kinds = []
        assistant_source_count = len(set(assistant_source_kinds)) + int(
            history_was_consumed
        )
        assistant_durable_source_digests = list(
            context.memory_source_digests
            if "local_memory" in assistant_source_kinds
            else ()
        )
        if "local_profile" in assistant_source_kinds:
            assistant_durable_source_digests.append(context.profile_source_digest)
        assistant_source_digest = self._derived_source_digest(
            decision=decision,
            source_kinds=assistant_source_kinds,
            history_lineage_digest=(
                context.history_eligibility.lineage_digest
                if history_was_consumed
                else None
            ),
            durable_source_digests=tuple(assistant_durable_source_digests),
        )
        self._turn_generation += 1
        user_entry = self._make_history_entry(
            role="user",
            content=user_input,
            decision=decision,
            turn_generation=self._turn_generation,
            personal_derived=False,
            source_count=0,
            source_digest=None,
        )
        assistant_entry = self._make_history_entry(
            role="assistant",
            content=answer,
            decision=decision,
            turn_generation=self._turn_generation,
            personal_derived=bool(assistant_source_count),
            source_count=int(assistant_source_count),
            source_digest=assistant_source_digest,
        )
        self.history.append(user_entry)
        self.history.append(assistant_entry)
        self._trim_history()
        route_metadata["history_message_provenance"] = {
            "user": self._history_message_provenance_metadata(user_entry),
            "assistant": self._history_message_provenance_metadata(assistant_entry),
        }
        self.last_turn_metadata = route_metadata
        return answer

    def _turn_metadata(
        self,
        loop: dict[str, object],
        source: str,
        model_error: str,
        model_exception_type: str = "",
        model_usage: dict[str, object] | None = None,
        model_call_attempted: bool | None = None,
        decision: _HistoryPolicyDecision | None = None,
    ) -> dict[str, object]:
        decision = decision or self._history_policy_decision(commit=False)
        selected_provider = decision.provider
        model_output_tokens = provider_output_token_limit(
            selected_provider,
            local_output_tokens=self.max_reply_tokens,
            openai_output_tokens=self.openai_max_output_tokens,
        )
        if model_call_attempted is None:
            model_call_attempted = source in {"model", "fallback"}
        provider_valid = decision.provider_valid
        remote_model_attempted = bool(model_call_attempted and selected_provider == "openai")
        model_response_received = bool(model_call_attempted and source == "model")
        remote_model_response_received = bool(
            model_response_received and selected_provider == "openai"
        )
        ollama_destination = decision.ollama_destination
        ollama_local_only = decision.ollama_local_only
        destination_allowed = decision.destination_allowed
        destination_blocked = bool(
            model_call_attempted and selected_provider == "ollama" and not destination_allowed
        )
        cloud_model_blocked = bool(
            model_call_attempted
            and ollama_local_only is not None
            and ollama_local_only.cloud_model_alias
        )
        remote_context_allowed = decision.remote_context_allowed
        stored_context_consent = decision.stored_context_consent
        stored_context_allowed = decision.stored_context_allowed
        context_sources_in_request = (
            list(loop.get("personal_context_sources_in_request") or [])
            if model_call_attempted and stored_context_allowed
            else []
        )
        history_in_request = bool(
            model_call_attempted
            and stored_context_allowed
            and int(loop.get("history_messages_in_request") or 0) > 0
        )
        shared_context_sources = (
            list(context_sources_in_request)
            if remote_model_attempted and remote_context_allowed
            else []
        )
        shared_history = bool(
            remote_model_attempted
            and remote_context_allowed
            and int(loop.get("history_messages_in_request") or 0) > 0
        )
        request_blocked = bool(destination_blocked or cloud_model_blocked or not provider_valid)
        if model_response_received:
            model_execution_status = "response_received"
            model_execution_occurred: bool | None = True
        elif model_call_attempted and not request_blocked:
            model_execution_status = "outcome_unknown"
            model_execution_occurred = None
        else:
            model_execution_status = "not_executed"
            model_execution_occurred = False
        if remote_model_response_received:
            external_processing_status = "confirmed"
            external_processing_occurred: bool | None = True
        elif model_call_attempted and not request_blocked:
            external_processing_status = "unknown"
            external_processing_occurred = None
        else:
            external_processing_status = "not_executed"
            external_processing_occurred = False
        usage_receipt = safe_model_usage_receipt(model_usage, selected_provider)
        metadata = {
            "source": source,
            "reply_path": loop["reply_path"],
            "turn_type": loop["turn_type"],
            "risk_signals": loop["risk_signals"],
            "used_model": source == "model",
            "used_fallback": source == "fallback",
            "model": decision.model,
            "model_provider": selected_provider,
            "model_provider_valid": provider_valid,
            "model_request_blocked_by_invalid_provider": bool(
                not provider_valid and source == "fallback"
            ),
            "model_call_attempted": model_call_attempted,
            "calls_model": model_call_attempted,
            "calls_external_service": remote_model_attempted,
            "external_model_request_attempted": remote_model_attempted,
            "external_model_response_received": remote_model_response_received,
            "external_model_request_outcome": (
                "response_received"
                if remote_model_response_received
                else ("outcome_unknown" if remote_model_attempted else "not_attempted")
            ),
            "model_execution_status": model_execution_status,
            "model_execution_occurred": model_execution_occurred,
            "external_processing_status": external_processing_status,
            "external_processing_occurred": external_processing_occurred,
            "current_user_message_processed_externally": external_processing_occurred,
            "stored_personal_context_processed_externally": (
                external_processing_occurred if context_sources_in_request else False
            ),
            "history_processed_externally": (
                external_processing_occurred if history_in_request else False
            ),
            "model_destination_allowed": destination_allowed,
            "model_request_blocked_by_destination_policy": destination_blocked,
            "model_request_blocked_by_cloud_policy": cloud_model_blocked,
            "model_destination_policy": (
                "openai_external"
                if selected_provider == "openai"
                else (
                    "ollama_loopback"
                    if selected_provider == "ollama" and destination_allowed
                    else (
                        "ollama_blocked_nonlocal"
                        if selected_provider == "ollama"
                        else "invalid_provider"
                    )
                )
            ),
            "model_execution_location_policy": (
                "external_provider"
                if selected_provider == "openai"
                else (
                    "loopback_daemon_execution_unverified"
                    if selected_provider == "ollama"
                    else "not_applicable_invalid_provider"
                )
            ),
            "model_execution_locality_verified": False,
            "current_message_on_device_verified": bool(
                not model_call_attempted or destination_blocked or cloud_model_blocked
            ),
            "ollama_destination_diagnostic": (
                ollama_destination.diagnostic if ollama_destination is not None else "not_applicable"
            ),
            "ollama_destination_value_exposed": False,
            "ollama_redirects_allowed": False,
            "ollama_proxy_environment_allowed": False,
            **(
                ollama_local_only.receipt()
                if ollama_local_only is not None
                else {
                    "ollama_no_cloud_configured": False,
                    "ollama_no_cloud_valid": True,
                    "ollama_no_cloud_requested": False,
                    "ollama_cloud_model_alias": False,
                    "ollama_personal_context_allowed": False,
                    "ollama_unverified_context_consent_configured": False,
                    "ollama_unverified_context_consent_valid": True,
                    "ollama_unverified_context_consent_allowed": False,
                    "ollama_unverified_context_consent_env": "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT",
                    "ollama_execution_locality_verified": False,
                    "ollama_cloud_policy_diagnostic": "not_applicable",
                    "ollama_cloud_policy_value_exposed": False,
                }
            ),
            "shares_conversation_with_external_model": remote_model_attempted,
            "current_user_message_shared_with_external_model": remote_model_attempted,
            "current_user_message_in_external_model_request": remote_model_attempted,
            "shares_stored_personal_context_with_external_model": bool(
                shared_context_sources
            ),
            "shares_history_with_external_model": shared_history,
            "stored_personal_context_in_external_model_request": bool(
                shared_context_sources
            ),
            "history_in_external_model_request": shared_history,
            "stored_personal_context_consent_required": selected_provider in {"openai", "ollama"},
            "stored_personal_context_explicit_consent": stored_context_consent,
            "stored_personal_context_policy_satisfied": stored_context_allowed,
            "stored_personal_context_in_model_request": bool(context_sources_in_request),
            "personal_context_sources_in_model_request": context_sources_in_request,
            "history_in_model_request": history_in_request,
            "remote_personal_context_allowed": remote_context_allowed,
            "remote_personal_context_policy": (
                ("enabled" if remote_context_allowed else "disabled")
                if selected_provider == "openai"
                else _ollama_context_policy(
                    destination_allowed=destination_allowed,
                    cloud_model_alias=bool(
                        ollama_local_only is not None and ollama_local_only.cloud_model_alias
                    ),
                    no_cloud_requested=bool(
                        ollama_local_only is not None and ollama_local_only.requested
                    ),
                    consent_allowed=bool(
                        ollama_local_only is not None
                        and ollama_local_only.unverified_context_consent_allowed
                    ),
                )
                if selected_provider == "ollama"
                else "invalid_provider"
            ),
            "remote_personal_context_sources_shared": shared_context_sources,
            "remote_personal_context_source_health_shared": bool(
                remote_model_attempted and remote_context_allowed
            ),
            "remote_personal_context_content_in_metadata": False,
            "personal_context_omitted_from_remote_model": bool(
                remote_model_attempted and not shared_context_sources
            ),
            "personal_context_omitted_by_destination_policy": destination_blocked,
            "personal_context_omitted_by_cloud_policy": bool(
                model_call_attempted
                and selected_provider == "ollama"
                and not (
                    destination_allowed
                    and ollama_local_only is not None
                    and ollama_local_only.personal_context_allowed
                )
            ),
            "external_side_effect": False,
            "model_request_content_in_metadata": False,
            "model_response_content_in_metadata": False,
            "reasoning_effort": self.reasoning_effort,
            "model_timeout_seconds": self.model_timeout_seconds,
            "max_history_messages": self.max_history_messages,
            "max_reply_tokens": self.max_reply_tokens,
            "model_max_output_tokens": model_output_tokens,
            "openai_total_output_token_ceiling": (
                model_output_tokens if selected_provider == "openai" else 0
            ),
            "model_error": model_error[:220],
            "model_exception_type": model_exception_type[:80],
            "history_retained_messages": int(
                loop.get("history_retained_messages") or 0
            ),
            "history_eligible_messages": int(
                loop.get("history_eligible_messages") or 0
            ),
            "history_withheld_messages": int(
                loop.get("history_withheld_messages") or 0
            ),
            "history_withheld_reason_counts": dict(
                loop.get("history_withheld_reason_counts") or {}
            ),
            "history_policy_epoch": decision.epoch,
            "history_policy_epoch_id": decision.epoch_id,
            "history_policy_fingerprint": decision.fingerprint,
            "history_policy_content_in_fingerprint": False,
            "history_disclosure_prepare_status": str(
                loop.get("history_disclosure_prepare_status") or "not_required"
            ),
            "history_disclosure_finalize_status": str(
                loop.get("history_disclosure_finalize_status") or "not_required"
            ),
            "history_disclosure_receipt_prepared": bool(
                loop.get("history_disclosure_prepare_status") == "prepared"
            ),
            "history_disclosure_callback_content_in_metadata": False,
            **usage_receipt,
        }
        for key in (
            "next_command",
            "recommended_next_commands",
            "context_source_states",
            "context_unavailable_sources",
            "context_fully_available",
        ):
            if key in loop:
                metadata[key] = loop[key]
        return metadata

    def _ordinary_context_availability_note(self, unavailable_sources: list[str]) -> str:
        return (
            "\n\nPersonal context note: I couldn't fully check "
            + ", ".join(unavailable_sources)
            + " for this answer, so it may be less personalized. I did not assume the missing context was empty."
        )

    def _capture_context_snapshot(
        self,
        user_input: str,
        *,
        decision: _HistoryPolicyDecision | None = None,
    ) -> _ChatContextSnapshot:
        decision = decision or self._history_policy_decision(commit=False)
        history_eligibility = self._eligible_history(decision)
        profile, profile_state = self._read_profile_context()
        preferences, preference_state = self._read_preference_context()
        (
            memories,
            memory_state,
            memory_source_digests,
            memory_disclosure_digests,
            memory_ids,
        ) = self._read_memory_context_with_custody(
            user_input,
            decision=decision,
        )
        skills, skill_state = self._read_skill_context(user_input)
        profile_source_digest = self._profile_source_digest(profile, profile_state)
        return _ChatContextSnapshot(
            profile=profile,
            preferences=preferences,
            memories=memories,
            skills=skills,
            profile_state=profile_state,
            preference_state=preference_state,
            memory_state=memory_state,
            skill_state=skill_state,
            recent_history=history_eligibility.messages,
            history_eligibility=history_eligibility,
            profile_source_digest=profile_source_digest,
            memory_source_digests=memory_source_digests,
            profile_disclosure_digests=self._context_disclosure_digests(
                "verified_profile_content_v1",
                profile,
                custody_digest=profile_source_digest,
            ),
            preference_disclosure_digests=self._context_disclosure_digests(
                "preference_item_v1",
                preferences,
            ),
            memory_disclosure_digests=memory_disclosure_digests,
            skill_disclosure_digests=self._context_disclosure_digests(
                "skill_item_v1",
                skills,
            ),
            memory_ids=memory_ids,
        )

    def _classify_loop(
        self,
        user_input: str,
        *,
        context: _ChatContextSnapshot | None = None,
        decision: _HistoryPolicyDecision | None = None,
    ) -> dict[str, object]:
        decision = decision or self._history_policy_decision(commit=False)
        display_input = _safe_preview_text(user_input)
        risk_signals = self._risk_signals(user_input)
        reply_path = self._reply_path(user_input, risk_signals)
        selected_provider = decision.provider
        provider_valid = decision.provider_valid
        would_call_model = reply_path == "model_or_fallback_chat"
        would_call_external = bool(would_call_model and selected_provider == "openai")
        ollama_destination = decision.ollama_destination
        ollama_local_only = decision.ollama_local_only
        destination_allowed = decision.destination_allowed
        cloud_model_alias = decision.cloud_model_alias
        would_make_model_network_request = bool(
            would_call_model
            and provider_valid
            and destination_allowed
            and not cloud_model_alias
        )
        remote_context_allowed = decision.remote_context_allowed
        stored_context_consent = decision.stored_context_consent
        stored_context_allowed = decision.stored_context_allowed
        if not would_make_model_network_request:
            external_processing_if_executed = "not_applicable"
        elif selected_provider == "openai":
            external_processing_if_executed = "confirmed_external_provider"
        else:
            external_processing_if_executed = "unknown_unverified_ollama_daemon"
        loop = {
            "turn_type": self._turn_type(user_input, risk_signals),
            "reply_path": reply_path,
            "risk_signals": risk_signals,
            "max_history_messages": self.max_history_messages,
            "history_policy_epoch": decision.epoch,
            "history_policy_epoch_id": decision.epoch_id,
            "history_policy_fingerprint": decision.fingerprint,
            "history_policy_content_in_fingerprint": False,
            "calls_model": False,
            "would_call_model": would_call_model,
            "model_provider": selected_provider,
            "model_provider_valid": provider_valid,
            "model_request_would_be_blocked_by_invalid_provider": bool(
                would_call_model and not provider_valid
            ),
            "would_call_external_service": would_call_external,
            "would_make_model_network_request": would_make_model_network_request,
            "model_destination_allowed": destination_allowed,
            "model_request_would_be_blocked_by_destination_policy": bool(
                would_call_model
                and selected_provider == "ollama"
                and not destination_allowed
            ),
            "model_request_would_be_blocked_by_cloud_policy": bool(
                would_call_model and cloud_model_alias
            ),
            "model_destination_policy": (
                "openai_external"
                if selected_provider == "openai"
                else (
                    "ollama_loopback"
                    if selected_provider == "ollama" and destination_allowed
                    else (
                        "ollama_blocked_nonlocal"
                        if selected_provider == "ollama"
                        else "invalid_provider"
                    )
                )
            ),
            "model_execution_location_policy": (
                "external_provider"
                if selected_provider == "openai"
                else (
                    "loopback_daemon_execution_unverified"
                    if selected_provider == "ollama"
                    else "not_applicable_invalid_provider"
                )
            ),
            "model_execution_locality_verified": False,
            "model_execution_status": "not_executed",
            "model_execution_occurred": False,
            "external_processing_status": "not_executed",
            "external_processing_occurred": False,
            "external_processing_if_executed": external_processing_if_executed,
            "current_user_message_processed_externally": False,
            "stored_personal_context_processed_externally": False,
            "history_processed_externally": False,
            "current_message_on_device_verified": bool(
                not would_make_model_network_request
            ),
            "ollama_destination_diagnostic": (
                ollama_destination.diagnostic if ollama_destination is not None else "not_applicable"
            ),
            "ollama_destination_value_exposed": False,
            **(
                ollama_local_only.receipt()
                if ollama_local_only is not None
                else {
                    "ollama_no_cloud_configured": False,
                    "ollama_no_cloud_valid": True,
                    "ollama_no_cloud_requested": False,
                    "ollama_cloud_model_alias": False,
                    "ollama_personal_context_allowed": False,
                    "ollama_unverified_context_consent_configured": False,
                    "ollama_unverified_context_consent_valid": True,
                    "ollama_unverified_context_consent_allowed": False,
                    "ollama_unverified_context_consent_env": "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT",
                    "ollama_execution_locality_verified": False,
                    "ollama_cloud_policy_diagnostic": "not_applicable",
                    "ollama_cloud_policy_value_exposed": False,
                }
            ),
            "would_share_current_message_with_external_model": would_call_external,
            "would_share_stored_personal_context_with_external_model": bool(
                would_call_external and remote_context_allowed
            ),
            "stored_personal_context_consent_required": selected_provider in {"openai", "ollama"},
            "stored_personal_context_explicit_consent": stored_context_consent,
            "stored_personal_context_policy_satisfied": stored_context_allowed,
            "would_include_stored_personal_context_in_model_request": bool(
                would_call_model and stored_context_allowed
            ),
            "remote_personal_context_allowed": remote_context_allowed,
            "remote_personal_context_policy": (
                ("enabled" if remote_context_allowed else "disabled")
                if selected_provider == "openai"
                else _ollama_context_policy(
                    destination_allowed=destination_allowed,
                    cloud_model_alias=cloud_model_alias,
                    no_cloud_requested=bool(
                        ollama_local_only is not None and ollama_local_only.requested
                    ),
                    consent_allowed=bool(
                        ollama_local_only is not None
                        and ollama_local_only.unverified_context_consent_allowed
                    ),
                )
                if selected_provider == "ollama"
                else "invalid_provider"
            ),
            "executes_tools": False,
            "queues_approval": False,
            "writes_memory": False,
            "controls_computer": False,
        }
        if risk_signals:
            commands = self._risk_route_commands(display_input)
            loop["next_command"] = commands[0]
            loop["recommended_next_commands"] = commands
        if context is not None:
            context_states = context.source_states()
            unavailable_sources = [
                source for source, state in context_states.items() if state == "unavailable"
            ]
            loop.update(
                {
                    "has_profile": bool(context.profile),
                    "preferences": len(self._compact_context_lines(context.preferences, limit=20)),
                    "memories": len(self._compact_context_lines(context.memories, limit=20)),
                    "skills": len(self._compact_context_lines(context.skills, limit=20)),
                    "recent_history_messages": context.history_eligibility.retained,
                    "history_retained_messages": context.history_eligibility.retained,
                    "history_eligible_messages": len(context.recent_history),
                    "history_withheld_messages": context.history_eligibility.withheld,
                    "history_withheld_reason_counts": dict(
                        context.history_eligibility.reason_counts
                    ),
                    "context_source_states": context_states,
                    "context_unavailable_sources": unavailable_sources,
                    "context_fully_available": not unavailable_sources,
                    "reads_private_data": True,
                    "reads_personal_data": True,
                }
            )
            sources_to_share = []
            if context.profile:
                sources_to_share.append("profile")
            if context.preferences:
                sources_to_share.append("preferences")
            if context.memories:
                sources_to_share.append("memory")
            if context.skills:
                sources_to_share.append("skills")
            loop["remote_personal_context_sources_to_share"] = (
                sources_to_share
                if would_call_external and remote_context_allowed
                else []
            )
            loop["stored_personal_context_sources_to_include"] = (
                sources_to_share
                if would_call_model and stored_context_allowed
                else []
            )
            loop["would_share_history_with_external_model"] = bool(
                would_call_external and remote_context_allowed and context.recent_history
            )
            loop["would_include_history_in_model_request"] = bool(
                would_call_model and stored_context_allowed and context.recent_history
            )
        return loop

    def preview_loop(self, user_input: str) -> dict[str, object]:
        with self._history_lock:
            decision = self._history_policy_decision(commit=False)
            context = self._capture_context_snapshot(user_input, decision=decision)
            return self._classify_loop(
                user_input, context=context, decision=decision
            )

    def preview_model_boundary(
        self,
        user_input: str,
        *,
        has_profile: bool = False,
        preferences: int = 0,
        memories: int = 0,
        skills: int = 0,
    ) -> dict[str, object]:
        """Forecast provider disclosure without reading personal context again."""
        with self._history_lock:
            decision = self._history_policy_decision(commit=False)
            history_eligibility = self._eligible_history(decision)
            loop = self._classify_loop(user_input, decision=decision)
        recent_history_messages = history_eligibility.retained
        eligible_history_messages = len(history_eligibility.messages)
        sources_to_share = []
        if has_profile:
            sources_to_share.append("profile")
        if preferences > 0:
            sources_to_share.append("preferences")
        if memories > 0:
            sources_to_share.append("memory")
        if skills > 0:
            sources_to_share.append("skills")
        sharing_enabled = bool(
            loop["would_call_external_service"] and decision.remote_context_allowed
        )
        inclusion_enabled = bool(
            loop["would_call_model"]
            and loop["stored_personal_context_policy_satisfied"]
        )
        loop.update(
            {
                "recent_history_messages": recent_history_messages,
                "history_retained_messages": recent_history_messages,
                "history_eligible_messages": eligible_history_messages,
                "history_withheld_messages": history_eligibility.withheld,
                "history_withheld_reason_counts": dict(
                    history_eligibility.reason_counts
                ),
                "remote_personal_context_sources_to_share": (
                    sources_to_share if sharing_enabled else []
                ),
                "would_share_history_with_external_model": bool(
                    sharing_enabled and eligible_history_messages
                ),
                "stored_personal_context_sources_to_include": (
                    sources_to_share if inclusion_enabled else []
                ),
                "would_include_history_in_model_request": bool(
                    inclusion_enabled and eligible_history_messages
                ),
            }
        )
        return loop

    def preview_loop_text(self, user_input: str) -> tuple[str, dict[str, object]]:
        with self._history_lock:
            decision = self._history_policy_decision(commit=False)
            context = self._capture_context_snapshot(user_input, decision=decision)
            loop = self._classify_loop(
                user_input, context=context, decision=decision
            )
        display_input = _safe_preview_text(user_input)
        risk_signals = loop["risk_signals"]
        lines = [
            "Jarvis chat loop preview:",
            "This is read-only. It shows how the conversational brain would route a message before any model call or tool execution.",
            "",
            "Message:",
            f"- {display_input or '(no message supplied)'}",
            "",
            "Loop:",
            "- 1. Perceive text input.",
            "- 2. Gather profile, preferences, memories, saved skills, and recent conversation.",
            "- 3. Classify the turn as memory, safety/action, or normal conversation.",
            "- 4. Choose deterministic grounded reply, safety preflight guidance, or model/fallback chat.",
            "- 5. Keep actions behind planner, ToolRegistry, PermissionPolicy, approval queue, and audit log.",
            "",
            "Classification:",
            f"- turn type: {loop['turn_type']}",
            f"- reply path: {loop['reply_path']}",
            f"- risk signals: {', '.join(risk_signals) if risk_signals else 'none detected'}",
            "",
            "Grounding counts:",
            f"- profile context: {'present' if loop['has_profile'] else 'empty'}",
            f"- active preferences: {loop['preferences']}",
            f"- relevant memories: {loop['memories']}",
            f"- relevant skills: {loop['skills']}",
            f"- recent history messages: {loop['recent_history_messages']} / {loop['max_history_messages']} configured",
            "",
            "Context source health:",
            *(
                f"- {source}: {state}"
                for source, state in dict(loop["context_source_states"]).items()
            ),
            "",
            "Model boundary:",
            f"- configured provider: {loop['model_provider']}",
            f"- model destination policy: {loop['model_destination_policy']}",
            f"- model destination allowed: {'yes' if loop['model_destination_allowed'] else 'no'}",
            f"- would make a model network request: {'yes' if loop['would_make_model_network_request'] else 'no'}",
            f"- would call an external model: {'yes' if loop['would_call_external_service'] else 'no'}",
            f"- model execution location policy: {loop['model_execution_location_policy']}",
            f"- preview execution status: {loop['model_execution_status']}",
            f"- external processing if the model runs: {loop['external_processing_if_executed']}",
            f"- current message verified to remain on-device: {'yes' if loop['current_message_on_device_verified'] else 'no'}",
            f"- would share this current message externally: {'yes' if loop['would_share_current_message_with_external_model'] else 'no'}",
            f"- would share stored profile/preferences/memory/skills externally: {'yes' if loop['would_share_stored_personal_context_with_external_model'] else 'no'}",
            f"- would share prior chat history externally: {'yes' if loop.get('would_share_history_with_external_model') else 'no'}",
            f"- stored-context explicit consent: {'yes' if loop['stored_personal_context_explicit_consent'] else 'no'}",
            f"- would include stored profile/preferences/memory/skills in the model request: {'yes' if loop['would_include_stored_personal_context_in_model_request'] else 'no'}",
            f"- would include prior chat history in the model request: {'yes' if loop.get('would_include_history_in_model_request') else 'no'}",
            f"- remote stored-context opt-in: {loop['remote_personal_context_policy']}",
            "",
            "Boundary:",
            "- This preview reads local profile, preferences, memories, saved skills, and recent in-memory conversation to report the grounding counts above.",
            "- It does not call the model.",
            "- It does not execute tools, write memory, control the computer, or queue approvals.",
        ]
        if risk_signals:
            lines.extend(
                [
                    "",
                    "Next route:",
                    f"- `{loop['next_command']}`",
                ]
            )
            lines.append("- Risky action language should enter through `execution governor`, then use risk preflight, action rehearsal, and autonomy planning under that route before execution.")
        return "\n".join(lines), loop

    def _risk_route_commands(self, user_input: str) -> list[str]:
        request = _safe_preview_text(user_input)
        return [
            f"execution governor: {request}",
            f"risk preflight: {request}",
            f"action rehearsal: {request}",
            f"autonomy plan: {request}",
        ]

    def _needs_grounded_memory_reply(self, user_input: str) -> bool:
        if _contains_any_token_phrase(
            user_input,
            (
                "within this conversation",
                "for this conversation",
                "without saving",
                "do not save",
                "don't save",
            ),
        ):
            return False
        memory_phrases = (
            "memory",
            "remember",
            "what do you know about me",
            "what do you know about the operator",
            "what have you learned",
            "show my profile",
            "what is in my profile",
            "what are my preferences",
        )
        return _contains_any_token_phrase(user_input, memory_phrases)

    def _risk_signals(self, user_input: str) -> list[str]:
        return [
            label
            for label, hints in ACTION_RISK_HINTS.items()
            if _contains_any_token_phrase(user_input, hints)
        ]

    def _turn_type(self, user_input: str, risk_signals: list[str]) -> str:
        if self._needs_grounded_memory_reply(user_input):
            return "memory_grounded"
        if risk_signals or _contains_any_token_phrase(
            user_input, ("do this", "act", "execute", "control", "open", "send")
        ):
            return "action_or_safety"
        return "conversation"

    def _reply_path(self, user_input: str, risk_signals: list[str]) -> str:
        if self._needs_grounded_memory_reply(user_input):
            return "grounded_memory"
        if risk_signals:
            return "safety_preflight_guidance"
        return "model_or_fallback_chat"

    def _grounded_memory_response(
        self,
        user_input: str,
        *,
        profile_context: str,
        preference_context: str,
        memory_context: str,
        skill_context: str,
        context_states: dict[str, str],
    ) -> str:
        low = user_input.lower()
        profile_lines = self._compact_profile_lines(profile_context, limit=3)
        preference_lines = self._compact_context_lines(preference_context, limit=4)
        memory_lines = self._compact_context_lines(memory_context, limit=5)
        skill_lines = self._compact_context_lines(skill_context, limit=3)

        if any(phrase in low for phrase in ("do you remember", "remember anything", "what do you remember")):
            opener = "Here is what I can actually see in memory right now:"
        elif "profile" in low:
            opener = "I can talk from the profile context I can actually read:"
        elif "preference" in low:
            opener = "I can apply these saved preferences I can actually see:"
        else:
            opener = (
                "I can talk about memory, but I should stay grounded: I only know what is in Jarvis memory, "
                "profile notes, preferences, saved skills, and recent conversation context."
            )

        sections: list[str] = [opener]
        if profile_lines:
            sections.append("\nProfile context:\n" + "\n".join(f"- {line}" for line in profile_lines))
        if preference_lines:
            sections.append("\nActive preferences:\n" + "\n".join(f"- {line}" for line in preference_lines))
        if memory_lines:
            sections.append("\nRelevant memory:\n" + "\n".join(f"- {line}" for line in memory_lines))
        if skill_lines:
            sections.append("\nRelevant saved skills:\n" + "\n".join(f"- {line}" for line in skill_lines))
        unavailable_sources = [
            source for source, state in context_states.items() if state == "unavailable"
        ]
        if len(sections) == 1 and unavailable_sources:
            sections.append(
                "\nI couldn't verify saved personal context right now because these sources are unavailable: "
                + ", ".join(unavailable_sources)
                + ". I will not claim that nothing is saved. Please retry after the local context stores recover."
            )
        elif len(sections) == 1:
            sections.append("\nI do not see matching saved context yet. You can tell me what to remember, and I will store it through the normal memory tools.")
        elif unavailable_sources:
            sections.append(
                "\nContext availability note: I could not fully check "
                + ", ".join(unavailable_sources)
                + ". The context above is only the portion I could verify."
            )

        sections.append(
            "\nSafety boundary: memory can inform chat, but tool execution still goes through the planner, registry, permission policy, and approval queue."
        )
        return "\n".join(sections)

    def _safety_preflight_response(self, user_input: str, risk_signals: object) -> str:
        signals = [str(signal) for signal in risk_signals] if isinstance(risk_signals, list) else []
        lines = [
            "That sounds like it may involve action, so I should not treat it as plain chat.",
            "",
            "Before acting, I would route it through the command-first harness loop:",
            "- `execution governor: " + user_input + "`",
            "- `risk preflight: " + user_input + "`",
            "- `action rehearsal: " + user_input + "`",
            "- `autonomy plan: " + user_input + "`",
        ]
        if signals:
            lines.extend(["", "Likely risk areas:"])
            lines.extend(f"- {signal}" for signal in signals)
        if "computer-control" in signals:
            lines.extend(
                [
                    "",
                    "Computer-control shape: observe, plan, approve, act, then verify.",
                ]
            )
        lines.extend(
            [
                "",
                "Safety boundary: I can talk through the plan now, but actual shell/code, file changes, personal-data access, external side effects, and computer control still go through the planner, registry, permission policy, approval queue, and audit log.",
            ]
        )
        return "\n".join(lines)

    def _fallback_response(
        self,
        user_input: str,
        *,
        profile_context: str,
        preference_context: str,
        memory_context: str,
        skill_context: str,
        exc: Exception | None = None,
        model: str | None = None,
        provider: str | None = None,
        model_call_attempted: bool | None = None,
    ) -> str:
        low = user_input.lower()
        context_lines = self._compact_context_lines(memory_context, limit=3)
        preference_lines = self._compact_context_lines(preference_context, limit=3)
        skill_lines = self._compact_context_lines(skill_context, limit=2)

        if any(phrase in low for phrase in ("talk normally", "just talk", "can we talk", "chat normally")):
            answer = (
                "Yes. I can stay in conversation mode instead of only acting like a command runner. "
                "The configured chat model did not produce a usable response, so this is my built-in fallback voice: useful, honest, "
                "and careful about the context available to me."
            )
        elif self._mentions_any(low, ("what can you do", "capabilities", "capability", "able to do", "features")):
            answer = (
                "I can talk through ideas, remember durable context, search Jarvis memory and Obsidian notes, manage "
                "tasks/goals/decisions/preferences, draft skills from sessions, produce return briefs, plan safe autonomy, "
                "and use local tools through the runtime. The important boundary is that computer control, clipboard reads, "
                "shell/code execution, destructive file changes, reminders, and outside-world actions need explicit approval. "
                "For the full live list, ask `capability map`."
            )
        elif self._mentions_any(low, ("computer control", "control my computer", "click", "screen", "mouse", "keyboard", "desktop")):
            answer = (
                "Yes, but it should be handled as observe, plan, approve, act, verify. Jarvis can inspect computer-control "
                "status and prepare an autonomy plan without approval, but screenshots, clicks, typing, clipboard reads, "
                "and full observe-act-verify steps are gated because they can expose private data or change your machine. "
                "A good starting command is `autonomy plan: <the desktop task>`."
            )
        elif self._mentions_any(low, ("safe", "safety", "harm", "approval", "dangerous")):
            answer = (
                "The safe shape is: read-only and local-safe actions can run, while personal data, external effects, shell/code, "
                "destructive changes, and computer-control actions stop for approval. Blocked actions create a safety receipt "
                "and stay visible in `pending approvals`, so nothing risky quietly disappears or runs behind your back."
            )
        elif "brain" in low and any(word in low for word in ("project", "jarvis", "assistant", "agent")):
            answer = (
                "Jarvis's brain should be a layered loop: conversation first, then memory, then planning, then tools, "
                "then verification. The important part is not pretending to be all-powerful; it should know what it knows, "
                "ask when a step is risky, and write useful state back into memory so each session compounds."
            )
        elif self._mentions_any(low, ("obsidian", "memory", "remember", "notes")) and not self._mentions_any(low, ("do you remember", "remember anything", "what do you remember")):
            answer = (
                "Memory should stay inspectable. Jarvis uses SQLite for fast lookup and Obsidian notes for human-readable "
                "state: facts, profile notes, preferences, people, decisions, goals, tasks, skills, Current Context, and "
                "Mission Control. That means you can talk to it naturally, but still audit what it thinks it knows."
            )
        elif self._mentions_any(low, ("next", "continue", "build", "work on")) and "jarvis" in low:
            answer = (
                "The next valuable direction is making Jarvis more useful between sessions: better return briefs, better "
                "Mission Control, safer autonomy plans, and stronger chat fallback when the local model is offline. Then we "
                "can migrate older calendar/email/browser tools into the same approval system instead of bolting them on raw."
            )
        elif any(phrase in low for phrase in ("do you remember", "remember anything", "what do you remember")):
            if context_lines:
                answer = "I found relevant memory. The useful bits are:\n" + "\n".join(f"- {line}" for line in context_lines)
            else:
                answer = "I do not see a matching memory yet, but I can save one if you tell me what should persist."
        elif low.rstrip("?").startswith(("what", "why", "how", "should", "could", "can")):
            answer = (
                "My best current read: keep this practical and incremental. We should turn repeated assistant behaviors "
                "into explicit tools, keep risky actions behind approvals, and make the chat layer feel natural on top of "
                "that system instead of separate from it."
            )
        else:
            answer = (
                "I am here. The configured chat model did not produce a usable response, but I can still keep the conversation "
                "moving and use any Jarvis context that is available. Tell me the next thing you want to think through or have me do."
            )

        if preference_lines:
            answer += "\n\nI am also applying these active preferences:\n" + "\n".join(f"- {line}" for line in preference_lines)
        if skill_lines:
            answer += "\n\nRelevant saved skills I noticed:\n" + "\n".join(f"- {line}" for line in skill_lines)
        if context_lines and "I found relevant memory" not in answer:
            answer += "\n\nRelevant memory I can see:\n" + "\n".join(f"- {line}" for line in context_lines)
        if profile_context and "profile" in low:
            answer += "\n\nI can also read the curated profile context when you ask for it."
        selected_model = model if isinstance(model, str) and model.strip() else self.model
        selected_provider = provider if isinstance(provider, str) and provider.strip() else self.provider
        recovery_hint = _model_recovery_hint(exc, selected_model, selected_provider)
        diagnostic = exc.diagnostic if isinstance(exc, ModelProviderError) else "model_unavailable"
        normalized_provider = normalized_model_provider(selected_provider)
        if model_call_attempted is False and normalized_provider in SUPPORTED_MODEL_PROVIDERS:
            provider_label = "OpenAI" if normalized_provider == "openai" else "Local model"
            answer += (
                f"\n\n{provider_label} routing note: {selected_model} did not receive a model request, "
                f"so I used fallback chat. {recovery_hint} Diagnostic: {diagnostic}."
            )
        elif normalized_provider == "openai":
            answer += (
                f"\n\nOpenAI model note: {selected_model} did not produce a usable complete response, "
                f"so I used fallback chat. {recovery_hint} Diagnostic: {diagnostic}."
            )
        elif normalized_provider not in SUPPORTED_MODEL_PROVIDERS:
            answer += (
                "\n\nModel routing note: the configured provider is invalid, so no model request was made "
                "and I used fallback chat. "
                f"{recovery_hint} Diagnostic: {diagnostic}."
            )
        else:
            answer += (
                f"\n\nLocal model note: {selected_model} is unavailable, so I used fallback chat. "
                f"{recovery_hint} Diagnostic: {diagnostic}."
            )
        return answer

    def _mentions_any(self, text: str, phrases: tuple[str, ...]) -> bool:
        return _contains_any_token_phrase(text, phrases)

    def _compact_context_lines(self, context: str, limit: int) -> list[str]:
        lines: list[str] = []
        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            line = re.sub(r"\s+", " ", line[2:]).strip()
            if len(line) > 220:
                line = line[:217].rstrip() + "..."
            lines.append(line)
            if len(lines) >= limit:
                break
        return lines

    def _compact_profile_lines(self, context: str, limit: int) -> list[str]:
        lines: list[str] = []
        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("Curated profile context"):
                continue
            line = line.strip("# ").strip()
            if not line or line == "Profile":
                continue
            line = re.sub(r"\s+", " ", line)
            if len(line) > 220:
                line = line[:217].rstrip() + "..."
            lines.append(line)
            if len(lines) >= limit:
                break
        return lines

    @contextmanager
    def _hold_profile_context_evidence(self):
        if self.vault is None:
            yield "", "unavailable"
            return
        evidence_stack = ExitStack()
        try:
            snapshot = evidence_stack.enter_context(
                self.store.hold_profile_knowledge_snapshot()
            )
            evidence_stack.enter_context(self.vault.profile_grounding_evidence_lock())
            custody_unavailable = bool(snapshot.invalid_count or snapshot.truncated)
            current_memory_ids: set[int] = set()
            for note in sorted(snapshot.notes, key=lambda item: item.memory_id):
                verified = evidence_stack.enter_context(
                    self.vault.canonical_memory_projection_evidence_lock(
                        memory_id=note.memory_id,
                        store_identity=snapshot.store_identity,
                        expected_relative_path=note.canonical_path_display,
                        expected_content_digest=note.content_digest,
                    )
                )
                if verified is True:
                    current_memory_ids.add(note.memory_id)
                else:
                    custody_unavailable = True
            verified_notes = tuple(
                note for note in snapshot.notes if note.memory_id in current_memory_ids
            )
            view = self.vault.read_profile_grounding(verified_notes, max_chars=1800)
            revalidated_view = self.vault.read_profile_grounding(
                verified_notes,
                max_chars=1800,
            )
            if revalidated_view != view:
                raise RuntimeError("profile grounding changed during custody read")
            custody_unavailable = bool(
                custody_unavailable
                or view.invalid
                or view.truncated
                or len(view.verified_source_keys) != len(verified_notes)
            )
            text = view.text.strip()
            if not text or text == "# Profile":
                held_context = ""
                held_state = "unavailable" if custody_unavailable else "empty"
            else:
                held_context = (
                    "Curated profile context. Treat unmarked user-authored text and store-verified "
                    "Jarvis profile notes as high-priority context:\n" + text
                )
                held_state = "unavailable" if custody_unavailable else "ok"
        except Exception:
            try:
                evidence_stack.close()
            except Exception:
                pass
            fallback_stack = ExitStack()
            try:
                fallback_stack.enter_context(self.vault.profile_grounding_evidence_lock())
                view = self.vault.read_profile_grounding((), max_chars=1800)
            except Exception:
                try:
                    fallback_stack.close()
                except Exception:
                    pass
                yield "", "unavailable"
                return
            text = view.text.strip()
            context = (
                "Curated profile context. Only unmarked user-authored text was available; "
                "Jarvis-generated profile notes could not be verified:\n" + text
                if text and text != "# Profile"
                else ""
            )
            try:
                yield context, "unavailable"
            except BaseException as exc:
                if not fallback_stack.__exit__(type(exc), exc, exc.__traceback__):
                    raise
            else:
                fallback_stack.close()
            return
        try:
            yield held_context, held_state
        except BaseException as exc:
            if not evidence_stack.__exit__(type(exc), exc, exc.__traceback__):
                raise
        else:
            evidence_stack.close()

    def _read_profile_context(self) -> tuple[str, str]:
        try:
            with self._hold_profile_context_evidence() as held:
                return held
        except Exception:
            if self.vault is None:
                return "", "unavailable"
            try:
                view = self.vault.read_profile_grounding((), max_chars=1800)
            except Exception:
                return "", "unavailable"
            text = view.text.strip()
            context = (
                "Curated profile context. Only unmarked user-authored text was available; "
                "Jarvis-generated profile notes could not be verified:\n" + text
                if text and text != "# Profile"
                else ""
            )
            return context, "unavailable"

    @contextmanager
    def _profile_snapshot_evidence_fence(
        self,
        context: _ChatContextSnapshot,
        *,
        required: bool,
    ):
        if not required:
            yield True
            return
        with self._hold_profile_context_evidence() as (current_text, current_state):
            yield hmac.compare_digest(
                context.profile_source_digest,
                self._profile_source_digest(current_text, current_state),
            )

    def _profile_context(self) -> str:
        return self._read_profile_context()[0]

    @staticmethod
    def _profile_source_digest(context: str, state: str) -> str:
        material = json.dumps(
            {
                "kind": "verified_profile_snapshot_v1",
                "state": str(state),
                "context": str(context),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _context_disclosure_digests(
        kind: str,
        context: str,
        *,
        custody_digest: str | None = None,
    ) -> tuple[str, ...]:
        if not context:
            return ()
        lines = context.splitlines()[1:]
        disclosed = [(index, line) for index, line in enumerate(lines) if line.strip()]
        if not disclosed:
            return ()
        context_digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        return tuple(
            _disclosure_item_digest(
                kind,
                {
                    "index": index,
                    "line": line,
                    "context_digest": context_digest,
                    "custody_digest": custody_digest,
                },
            )
            for index, line in disclosed
        )

    def _profile_snapshot_is_current(self, context: _ChatContextSnapshot) -> bool:
        with self._profile_snapshot_evidence_fence(context, required=True) as current:
            return current

    def _read_preference_context(self) -> tuple[str, str]:
        try:
            rows = self.store.list_preferences(
                status="active",
                limit=MAX_PREFERENCE_CONTEXT_ITEMS + 1,
            )
        except Exception:
            return "", "unavailable"
        if not rows:
            return "", "empty"
        lines = ["Structured user preferences. Follow these unless the user says otherwise:"]
        clipped = len(rows) > MAX_PREFERENCE_CONTEXT_ITEMS
        for row in rows[:MAX_PREFERENCE_CONTEXT_ITEMS]:
            source_category = _safe_row_value(row, "category", "uncategorized")
            source_key = _safe_row_value(row, "key", "preference")
            source_value = _safe_row_value(row, "value", "<unreadable>")
            raw_category = re.sub(r"[\r\n]+", " ", source_category)
            raw_key = re.sub(r"[\r\n]+", " ", source_key)
            raw_value = re.sub(r"[\r\n]+", " ", source_value)
            category = raw_category[:MAX_PREFERENCE_CATEGORY_CHARS]
            key = raw_key[:MAX_PREFERENCE_KEY_CHARS]
            value = raw_value[:MAX_PREFERENCE_VALUE_CHARS]
            clipped = clipped or any(
                source != raw
                for source, raw in (
                    (source_category, raw_category),
                    (source_key, raw_key),
                    (source_value, raw_value),
                )
            )
            clipped = clipped or any(
                len(raw) > len(bounded)
                for raw, bounded in (
                    (raw_category, category),
                    (raw_key, key),
                    (raw_value, value),
                )
            )
            line = f"- [{category}] {key}: {value}"
            if len("\n".join((*lines, line))) > MAX_PREFERENCE_CONTEXT_CHARS:
                clipped = True
                break
            lines.append(line)
        if len(lines) == 1:
            return "", "unavailable" if clipped else "empty"
        return "\n".join(lines), "unavailable" if clipped else "ok"

    def _preference_context(self) -> str:
        return self._read_preference_context()[0]

    def _read_skill_context(self, user_input: str) -> tuple[str, str]:
        terms = [word.strip(".,?!:;()[]{}").lower()[:64] for word in user_input.split()]
        terms = [word for word in terms if len(word) >= 4]
        try:
            candidates = self.store.list_active_skills(limit=20)
        except Exception:
            return "", "unavailable"
        if not terms:
            return "", "empty"
        rows = []
        for row in candidates:
            if not _skill_has_durable_remote_origin(row):
                continue
            searchable = " ".join(
                _safe_row_value(row, key, "")[:MAX_SKILL_SEARCH_FIELD_CHARS]
                for key in ("name", "trigger", "body", "tags")
            ).casefold()
            if any(term in searchable for term in terms[:6]):
                rows.append(row)
            if len(rows) >= 3:
                break
        if not rows:
            return "", "empty"
        lines = ["Relevant saved skills. Use as procedural guidance if helpful:"]
        clipped = False
        for row in rows:
            source_name = _safe_row_value(row, "name", "<unreadable>")
            source_trigger = _safe_row_value(row, "trigger", "<unreadable>")
            source_body = _safe_row_value(row, "body", "<unreadable>")
            raw_name = re.sub(r"[\r\n]+", " ", source_name)
            raw_trigger = re.sub(r"[\r\n]+", " ", source_trigger)
            raw_body = re.sub(r"[\r\n]+", " ", source_body)
            name = raw_name[:MAX_SKILL_NAME_CHARS]
            trigger = raw_trigger[:MAX_SKILL_TRIGGER_CHARS]
            body = raw_body[:MAX_SKILL_BODY_CHARS]
            clipped = clipped or any(
                source != raw
                for source, raw in (
                    (source_name, raw_name),
                    (source_trigger, raw_trigger),
                    (source_body, raw_body),
                )
            )
            clipped = clipped or any(
                len(raw) > len(bounded)
                for raw, bounded in (
                    (raw_name, name),
                    (raw_trigger, trigger),
                    (raw_body, body),
                )
            )
            line = f"- {name} | trigger: {trigger} | procedure: {body}"
            if len("\n".join((*lines, line))) > MAX_SKILL_CONTEXT_CHARS:
                clipped = True
                break
            lines.append(line)
        if len(lines) == 1:
            return "", "unavailable" if clipped else "empty"
        return "\n".join(lines), "unavailable" if clipped else "ok"

    def _skill_context(self, user_input: str) -> str:
        return self._read_skill_context(user_input)[0]

    def _durable_memory_destination_class(
        self,
        decision: _HistoryPolicyDecision,
    ) -> str:
        return decision.destination_class

    def _derived_memory_source_digest(
        self,
        row: object,
        decision: _HistoryPolicyDecision,
    ) -> str | None:
        try:
            memory_id = row.get("id") if type(row) is dict else row["id"]  # type: ignore[index]
        except Exception:
            return None
        if type(memory_id) is not int or memory_id <= 0:
            return None
        try:
            provenance = self.store.read_memory_provenance(memory_id)
        except Exception:
            return None
        if type(provenance) is not dict:
            return None
        source_digest = provenance.get("source_digest")
        policy_epoch_id = provenance.get("policy_epoch_id")
        source_count = provenance.get("source_count")
        if (
            provenance.get("memory_id") != memory_id
            or provenance.get("lineage_state") != "complete"
            or provenance.get("recorded_remote_eligible") is not True
            or provenance.get("remote_eligible") is not True
            or provenance.get("epoch_active") is not True
            or provenance.get("destination_class")
            != self._durable_memory_destination_class(decision)
            or type(source_count) is not int
            or source_count <= 0
            or type(source_digest) is not str
            or SHA256_HEX_RE.fullmatch(source_digest) is None
            or type(policy_epoch_id) is not str
            or SHA256_HEX_RE.fullmatch(policy_epoch_id) is None
            or type(provenance.get("created_at")) is not str
            or not provenance.get("created_at")
        ):
            return None
        return source_digest

    def _read_memory_context_with_custody(
        self,
        user_input: str,
        *,
        decision: _HistoryPolicyDecision | None = None,
    ) -> tuple[
        str,
        str,
        tuple[str, ...],
        tuple[str, ...],
        tuple[int, ...],
    ]:
        terms = [word.strip(".,?!:;()[]{}").lower()[:64] for word in user_input.split()]
        terms = [word for word in terms if len(word) >= 4]
        query = " OR ".join(terms[:6])
        rows = []
        read_failed = False
        if query:
            try:
                rows = self.store.search_memories(query, limit=10)
            except Exception:
                read_failed = True
        if not rows:
            try:
                rows = self.store.recent_memories(limit=10)
            except Exception:
                read_failed = True
        candidates: list[tuple[int, object]] = []
        malformed_candidate = False
        for row in rows[:10]:
            try:
                memory_id = row.get("id") if type(row) is dict else row["id"]  # type: ignore[index]
            except Exception:
                memory_id = None
            if type(memory_id) is not int or memory_id < 1:
                malformed_candidate = True
                continue
            source = unicodedata.normalize(
                "NFKC", _safe_row_value(row, "source", "")
            ).casefold()
            if source == "profile":
                continue
            candidates.append((memory_id, row))
        try:
            profile_owned = self.store.profile_owned_memory_ids(
                memory_id for memory_id, _row in candidates
            )
        except Exception:
            return "", "unavailable", (), (), ()
        if type(profile_owned) is not frozenset:
            return "", "unavailable", (), (), ()
        selected_rows = [
            (memory_id, row)
            for memory_id, row in candidates
            if memory_id not in profile_owned
        ][:5]
        read_failed = read_failed or malformed_candidate
        if not selected_rows:
            return "", "unavailable" if read_failed else "empty", (), (), ()

        eligible_rows: list[tuple[int, object, str | None, bool]] = []
        if decision is not None and decision.stored_context_allowed:
            for memory_id, row in selected_rows:
                category = unicodedata.normalize(
                    "NFKC", _safe_row_value(row, "category", "")
                ).casefold()
                source = unicodedata.normalize(
                    "NFKC", _safe_row_value(row, "source", "")
                ).casefold()
                if (
                    source == "conversation_compaction"
                    or category == "conversation-digest"
                ):
                    source_digest = self._derived_memory_source_digest(row, decision)
                    if source_digest is None:
                        continue
                    eligible_rows.append((memory_id, row, source_digest, True))
                else:
                    eligible_rows.append((memory_id, row, None, False))
        else:
            eligible_rows = [
                (memory_id, row, None, False)
                for memory_id, row in selected_rows
            ]
        if not eligible_rows:
            return "", "unavailable" if read_failed else "empty", (), (), ()

        lines = ["Relevant memory context. Use only if actually helpful:"]
        disclosed_items: list[tuple[int, str, str | None, bool]] = []
        for memory_id, row, source_digest, is_compaction in eligible_rows:
            raw_category = _safe_row_value(row, "category", "memory")
            raw_title = _safe_row_value(row, "title", "<unreadable>")
            raw_body = _safe_row_value(row, "body", "<unreadable>")
            category = raw_category[:MAX_MEMORY_CATEGORY_CHARS]
            title = raw_title[:MAX_MEMORY_TITLE_CHARS]
            body = raw_body[:MAX_MEMORY_BODY_CHARS]
            if (
                len(raw_category) > len(category)
                or len(raw_title) > len(title)
                or len(raw_body) > len(body)
            ):
                read_failed = True
            line = f"- [{category}] {title}: {body}"
            if len("\n".join((*lines, line))) > MAX_MEMORY_CONTEXT_CHARS:
                read_failed = True
                break
            lines.append(line)
            disclosed_items.append(
                (memory_id, line, source_digest, is_compaction)
            )
        if not disclosed_items:
            return "", "unavailable", (), (), ()
        context_text = "\n".join(lines)
        context_digest = hashlib.sha256(context_text.encode("utf-8")).hexdigest()
        disclosure_digests = tuple(
            _disclosure_item_digest(
                (
                    "compaction_memory_item_v1"
                    if is_compaction
                    else "memory_item_v1"
                ),
                {
                    "memory_id": memory_id,
                    "line": line,
                    "context_digest": context_digest,
                    "durable_source_digest": source_digest,
                },
            )
            for memory_id, line, source_digest, is_compaction in disclosed_items
        )
        return (
            context_text,
            "unavailable" if read_failed else "ok",
            tuple(
                source_digest
                for _memory_id, _line, source_digest, is_compaction in disclosed_items
                if is_compaction and source_digest is not None
            ),
            disclosure_digests,
            tuple(memory_id for memory_id, _line, _digest, _compaction in disclosed_items),
        )

    def _read_memory_context(
        self,
        user_input: str,
        *,
        decision: _HistoryPolicyDecision | None = None,
    ) -> tuple[str, str, tuple[str, ...]]:
        context, state, source_digests, _disclosure_digests, _memory_ids = (
            self._read_memory_context_with_custody(
                user_input,
                decision=decision,
            )
        )
        return context, state, source_digests

    def _memory_context(self, user_input: str) -> str:
        return self._read_memory_context(user_input)[0]


ChatBrain.model = property(  # type: ignore[assignment]
    ChatBrain._get_model,
    ChatBrain._set_model,
)
ChatBrain.provider = property(  # type: ignore[assignment]
    ChatBrain._get_provider,
    ChatBrain._set_provider,
)
ChatBrain.allow_remote_personal_context = property(  # type: ignore[assignment]
    ChatBrain._get_allow_remote_personal_context,
    ChatBrain._set_allow_remote_personal_context,
)
