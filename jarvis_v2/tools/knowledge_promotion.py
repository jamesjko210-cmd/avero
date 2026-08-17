from __future__ import annotations

import ipaddress
import math
import re
import secrets
import unicodedata
from typing import Any
from urllib.parse import urlsplit

from jarvis_v2.agent.failure_guidance import declare_failure_guidance
from jarvis_v2.agent.types import ApprovalArgumentResolution, ToolResult
from jarvis_v2.memory.decision_projection import reconcile_decision_projection
from jarvis_v2.memory.memory_projection import reconcile_memory_projection
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.preference_projection import reconcile_preference_projection
from jarvis_v2.memory.profile_projection import (
    MAX_PROFILE_CATEGORY_CHARS,
    MAX_PROFILE_HEADING_CHARS,
    MAX_PROFILE_WRITE_CHARS,
    reconcile_profile_projection_candidate,
)
from jarvis_v2.memory.store import (
    DecisionProjectionTarget,
    DecisionRecord,
    MemoryDecisionPromotionResult,
    MemoryProfilePromotionResult,
    MemoryPreferencePromotionResult,
    MemoryProjectionTarget,
    MemoryRecord,
    MemoryStore,
    PreferenceProjectionTarget,
    PreferenceRecord,
    profile_note_source_key,
)


MAX_TITLE_CHARS = 240
MAX_RATIONALE_CHARS = 4000
MAX_IMPACT_CHARS = 4000
MAX_PACKET_CATEGORY_CHARS = 1000
MAX_PACKET_TITLE_CHARS = 4096
MAX_PACKET_BODY_CHARS = 1200
MAX_PREFERENCE_CATEGORY_CHARS = 64
MAX_PREFERENCE_KEY_CHARS = 120
MAX_PREFERENCE_VALUE_CHARS = 2000
MAX_CANDIDATE_CATEGORY_CHARS = MAX_PACKET_CATEGORY_CHARS
MAX_CANDIDATE_TITLE_CHARS = MAX_PACKET_TITLE_CHARS
MAX_CANDIDATE_BODY_CHARS = MAX_PACKET_BODY_CHARS
MAX_CANDIDATE_SOURCE_CHARS = 256
MAX_CANDIDATE_TIMESTAMP_CHARS = 256

LOCAL_PATH_RE = re.compile(
    r"(?:file://[^\n\r]*|"
    r"~[A-Z0-9._-]*[/\\][^\n\r]*|"
    r"\.\.?[/\\][^\n\r]*|"
    r"[A-Z]:[/\\][^\n\r]*|"
    r"[A-Z]:(?![/\\])[^ \n\r]*\.[A-Z0-9]{1,16}\b|"
    r"[A-Z]:(?![/\\])[^ \n\r]*[/\\][^\n\r]*|"
    r"\\\\[^\\/\s]+[/\\][^\n\r]*|"
    r"(?<![A-Z0-9])\\(?!\\)[^\n\r]*|"
    r"//[^/\s]+/[^\n\r]*|"
    r"(?<![A-Z0-9:/])/(?!/)[^\n\r]*)",
    re.IGNORECASE,
)
RELATIVE_FILE_PATH_RE = re.compile(
    r"(?:^|(?<=[\s('`\"]))(?:[A-Z0-9._-]+[/\\])+[A-Z0-9._-]+\.[A-Z0-9]{1,16}"
    r"(?=$|[\s)'`,;:\"])",
    re.IGNORECASE,
)
HTTP_URL_CANDIDATE_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
HTTP_DNS_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(
    r"<\s*(?:explicit\s+)?(?:title|rationale|impact)\s*>", re.IGNORECASE
)
RESERVED_CONTENT_RE = re.compile(
    r"(?:"
    r"<!--|-->"
    r"|jarvis[-_](?:projection|profile-note(?:-start)?|conversation-compaction|custody|ownership)\b"
    r"|(?:store_identity|source_digest|source_revision)\s*:"
    r")",
    re.IGNORECASE,
)
PREFERENCE_PLACEHOLDER_RE = re.compile(
    r"<\s*(?:explicit\s+)?(?:category|key|value)\s*>", re.IGNORECASE
)
PROFILE_PLACEHOLDER_RE = re.compile(
    r"<\s*(?:explicit\s+)?(?:heading|category|body)\s*>", re.IGNORECASE
)
ACTIVE_RENDER_CONTENT_RE = re.compile(
    r"(?:"
    r"!\s*\["
    r"|<\s*/?\s*(?:img|iframe|video|audio|source|object|embed|script|link|track|"
    r"image|use|feimage|input|portal|style|svg|meta)\b"
    r"|<\s*/?\s*[a-z][^>\r\n]*>"
    r"|(?:url\s*\(|@import\b)"
    r")",
    re.IGNORECASE,
)

_PACKET_ARGS = frozenset({"memory_id"})
_RAW_PROMOTION_ARGS = frozenset(
    {
        "memory_id",
        "reviewed_revision",
        "review_token",
        "title",
        "rationale",
        "impact",
    }
)
_BOUND_PROMOTION_ARGS = frozenset(
    {
        "memory_id",
        "reviewed_revision",
        "review_binding",
        "title",
        "rationale",
        "impact",
        "target_revision",
        "target_binding",
    }
)
_RAW_PREFERENCE_PROMOTION_ARGS = frozenset(
    {
        "memory_id",
        "reviewed_revision",
        "review_token",
        "category",
        "key",
        "value",
    }
)
_BOUND_PREFERENCE_PROMOTION_ARGS = frozenset(
    {
        "memory_id",
        "reviewed_revision",
        "review_binding",
        "category",
        "key",
        "value",
        "target_revision",
        "target_binding",
    }
)
_RAW_PROFILE_PROMOTION_ARGS = frozenset(
    {
        "memory_id",
        "reviewed_revision",
        "review_token",
        "heading",
        "category",
        "body",
    }
)
_BOUND_PROFILE_PROMOTION_ARGS = frozenset(
    {
        "memory_id",
        "reviewed_revision",
        "review_binding",
        "heading",
        "category",
        "body",
        "target_revision",
        "target_binding",
    }
)
_MISSING = object()
TARGET_KIND_BY_CATEGORY = {
    "identity": "profile",
    "profile": "profile",
    "preference": "preference",
    "preferences": "preference",
    "decision": "decision",
    "decisions": "decision",
    "people": "person",
    "person": "person",
    "relationships": "person",
    "goal": "goal",
    "goals": "goal",
    "projects": "goal",
}
PROMOTION_REVIEW_RECOVERY_ACTION = (
    "Inspect the current source and target records, correct the reported issue, "
    "then request a fresh review before any new promotion."
)
PROMOTION_UNKNOWN_RECOVERY_ACTION = (
    "Verify the source record, target record, custody link, and pending projection "
    "state before deciding whether repair is needed; do not repeat the promotion."
)
COMMITTED_PROMOTION_RECOVERY_ACTION = (
    "Do not repeat the promotion; inspect the source record, target record, custody "
    "link, and pending projections before deciding whether repair is needed."
)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "reads_private_data": False,
        "reads_personal_data": False,
        "writes_files": False,
        "writes_notes": False,
        "writes_memory": False,
        "writes_database": False,
        "controls_computer": False,
        "queues_approval": False,
        "external_side_effect": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def _positive_id(value: Any) -> int | None:
    return (
        value
        if type(value) is int and 0 < value <= 9223372036854775807
        else None
    )


def _row_value(row: Any, key: str, default: Any = _MISSING) -> Any:
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        if key not in row.keys():
            return default
        return row[key]
    except Exception:
        return default


def _bounded_packet_text(value: str, limit: int) -> str:
    bounded = value if len(value) <= limit else value[: max(0, limit - 3)] + "..."
    rendered: list[str] = []
    for character in bounded:
        if character == "\n":
            rendered.append("\\n")
        elif character == "\r":
            rendered.append("\\r")
        elif character == "\t":
            rendered.append("\\t")
        elif unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}:
            rendered.append(f"\\u{ord(character):04x}")
        else:
            rendered.append(character)
    safe = "".join(rendered)
    safe = (
        safe.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("!", "\\!")
    )
    return safe


def _is_valid_http_url_candidate(candidate: str) -> bool:
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() or character == "\\" for character in parsed.netloc)
    ):
        return False
    authority = parsed.netloc
    if authority.startswith("["):
        close = authority.find("]")
        if close < 2:
            return False
        suffix = authority[close + 1 :]
        if suffix and re.fullmatch(r":[0-9]{1,5}", suffix) is None:
            return False
        try:
            ipaddress.IPv6Address(authority[1:close])
        except ValueError:
            return False
    else:
        if authority.count(":") > 1:
            return False
        raw_host = authority
        if ":" in authority:
            raw_host, raw_port = authority.rsplit(":", 1)
            if not raw_port.isdigit():
                return False
        try:
            ascii_host = raw_host.encode("idna").decode("ascii")
        except UnicodeError:
            return False
        labels = ascii_host.split(".")
        if re.fullmatch(r"[0-9.]+", ascii_host):
            try:
                ipaddress.IPv4Address(ascii_host)
            except ValueError:
                return False
        if (
            not ascii_host
            or len(ascii_host) > 253
            or any(HTTP_DNS_LABEL_RE.fullmatch(label) is None for label in labels)
        ):
            return False
    return port is None or 0 < port <= 65535


def _contains_reserved_placeholder(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    compact = re.sub(r"\s+", "", normalized)
    return PLACEHOLDER_RE.search(normalized) is not None or re.search(
        r"<(?:explicit)?(?:title|rationale|impact)>", compact
    ) is not None


def _unsafe_text_reason(
    value: str, *, allow_relative_separators: bool = False
) -> str | None:
    if ACTIVE_RENDER_CONTENT_RE.search(unicodedata.normalize("NFKC", value)):
        return "active_render_content"
    path_scan_parts: list[str] = []
    cursor = 0
    malformed_http_url = False
    for match in HTTP_URL_CANDIDATE_RE.finditer(value):
        raw_candidate = match.group(0)
        candidate = raw_candidate.rstrip(".,;:!?)]}")
        if not _is_valid_http_url_candidate(candidate):
            malformed_http_url = True
            continue
        path_scan_parts.append(value[cursor : match.start()])
        path_scan_parts.append(raw_candidate[len(candidate) :])
        cursor = match.end()
    path_scan_parts.append(value[cursor:])
    path_scan_text = "".join(path_scan_parts)
    if (
        malformed_http_url
        or LOCAL_PATH_RE.search(path_scan_text)
        or (
            allow_relative_separators
            and RELATIVE_FILE_PATH_RE.search(path_scan_text)
        )
        or (
            not allow_relative_separators
            and ("/" in path_scan_text or "\\" in path_scan_text)
        )
    ):
        return "local_path"
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in value
    ):
        return "control_character"
    if RESERVED_CONTENT_RE.search(value):
        return "reserved_content"
    return None


def _validated_decision_fields(
    args: dict[str, Any],
) -> tuple[dict[str, str] | None, str | None, str | None]:
    values: dict[str, str] = {}
    limits = {
        "title": MAX_TITLE_CHARS,
        "rationale": MAX_RATIONALE_CHARS,
        "impact": MAX_IMPACT_CHARS,
    }
    for field, limit in limits.items():
        raw = args.get(field, _MISSING)
        if type(raw) is not str:
            return None, f"invalid_{field}", f"Explicit {field} text is required."
        value = raw.strip()
        if not value:
            return None, f"missing_{field}", f"Explicit {field} text is required."
        if _contains_reserved_placeholder(value):
            return None, f"placeholder_{field}", f"Replace the {field} placeholder with reviewed text."
        if len(value) > limit:
            return None, f"oversized_{field}", f"{field.capitalize()} exceeds the {limit}-character limit."
        unsafe_reason = _unsafe_text_reason(value)
        if unsafe_reason is not None:
            return (
                None,
                f"unsafe_{field}_{unsafe_reason}",
                f"{field.capitalize()} contains unsafe content and cannot be promoted.",
            )
        values[field] = value
    return values, None, None


def _contains_preference_placeholder(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    compact = re.sub(r"\s+", "", normalized)
    return PREFERENCE_PLACEHOLDER_RE.search(normalized) is not None or re.search(
        r"<(?:explicit)?(?:category|key|value)>", compact
    ) is not None


def _unsafe_preference_text_reason(value: str, *, allow_http_url: bool) -> str | None:
    if not allow_http_url and HTTP_URL_CANDIDATE_RE.search(value):
        return "local_path"
    return _unsafe_text_reason(
        value,
        allow_relative_separators=allow_http_url,
    )


def _validated_preference_fields(
    args: dict[str, Any],
) -> tuple[dict[str, str] | None, str | None, str | None]:
    values: dict[str, str] = {}
    limits = {
        "category": MAX_PREFERENCE_CATEGORY_CHARS,
        "key": MAX_PREFERENCE_KEY_CHARS,
        "value": MAX_PREFERENCE_VALUE_CHARS,
    }
    for field, limit in limits.items():
        raw = args.get(field, _MISSING)
        if type(raw) is not str:
            return None, f"invalid_{field}", f"Explicit {field} text is required."
        value = raw.strip()
        if not value:
            return None, f"missing_{field}", f"Explicit {field} text is required."
        if _contains_preference_placeholder(value):
            return (
                None,
                f"placeholder_{field}",
                f"Replace the {field} placeholder with reviewed text.",
            )
        if len(value) > limit:
            return (
                None,
                f"oversized_{field}",
                f"{field.capitalize()} exceeds the {limit}-character limit.",
            )
        unsafe_reason = _unsafe_preference_text_reason(
            value,
            allow_http_url=field == "value",
        )
        if unsafe_reason is not None:
            return (
                None,
                f"unsafe_{field}_{unsafe_reason}",
                f"{field.capitalize()} contains unsafe content and cannot be promoted.",
            )
        values[field] = value
    return values, None, None


def _validated_profile_fields(
    args: dict[str, Any],
) -> tuple[dict[str, str] | None, str | None, str | None]:
    values: dict[str, str] = {}
    limits = {
        "heading": MAX_PROFILE_HEADING_CHARS,
        "category": MAX_PROFILE_CATEGORY_CHARS,
        "body": MAX_PROFILE_WRITE_CHARS,
    }
    for field, limit in limits.items():
        raw = args.get(field, _MISSING)
        if type(raw) is not str:
            return None, f"invalid_{field}", f"Explicit profile {field} text is required."
        if field == "heading":
            value = " ".join(unicodedata.normalize("NFKC", raw).split())
        elif field == "category":
            value = " ".join(unicodedata.normalize("NFKC", raw).casefold().split())
        else:
            value = (
                unicodedata.normalize("NFKC", raw)
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .strip()
            )
        if not value:
            return None, f"missing_{field}", f"Explicit profile {field} text is required."
        if value != raw:
            return (
                None,
                f"noncanonical_{field}",
                f"Profile {field} must already use its canonical reviewed form.",
            )
        normalized = unicodedata.normalize("NFKC", value).casefold()
        compact = re.sub(r"\s+", "", normalized)
        if (
            PROFILE_PLACEHOLDER_RE.search(normalized) is not None
            or re.search(r"<(?:explicit)?(?:heading|category|body)>", compact) is not None
        ):
            return (
                None,
                f"placeholder_{field}",
                f"Replace the profile {field} placeholder with reviewed text.",
            )
        if len(value) > limit:
            return (
                None,
                f"oversized_{field}",
                f"Profile {field} exceeds the {limit}-character limit.",
            )
        unsafe_reason = _unsafe_text_reason(value)
        if unsafe_reason is not None:
            return (
                None,
                f"unsafe_{field}_{unsafe_reason}",
                f"Profile {field} contains unsafe content and cannot be promoted.",
            )
        values[field] = value
    return values, None, None


def _refusal(
    tool_name: str,
    output: str,
    reason: str,
    *,
    memory_id: int | None = None,
    resolver: bool = False,
) -> ToolResult:
    public_output = (
        output
        if PROMOTION_REVIEW_RECOVERY_ACTION in output
        else f"{output.rstrip()} {PROMOTION_REVIEW_RECOVERY_ACTION}"
    )
    mutation = {
        "promote_memory_to_decision": "memory_to_decision_promotion",
        "promote_memory_to_preference": "memory_to_preference_promotion",
        "promote_memory_to_profile": "memory_to_profile_promotion",
    }.get(tool_name, "knowledge_promotion_review")
    metadata = _safe_metadata(
        read_only=tool_name == "knowledge_promotion_packet",
        requires_approval=tool_name
        in {
            "promote_memory_to_decision",
            "promote_memory_to_preference",
            "promote_memory_to_profile",
        },
        risk_level=("read_only" if tool_name == "knowledge_promotion_packet" else "high_risk"),
        reason=reason,
        mutation=mutation,
        state_changed=False,
        promotion_committed=False,
        handler_invoked=not resolver,
        executed_handler=not resolver,
    )
    if memory_id is not None:
        metadata["memory_id"] = memory_id
    if resolver:
        metadata["approval_argument_resolution_status"] = reason
        metadata["requires_confirmation"] = False
    return ToolResult(
        tool_name,
        False,
        public_output,
        declare_failure_guidance(
            metadata,
            output=public_output,
            action=PROMOTION_REVIEW_RECOVERY_ACTION,
        ),
    )


def _unknown_failure_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata.update(
        {
            "outcome_known": False,
            "outcome_unknown": True,
            "execution_outcome_unknown": True,
            "side_effect_possible": True,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return metadata


def _committed_promotion_failure(
    tool_name: str,
    output: str,
    metadata: dict[str, Any],
) -> ToolResult:
    public_output = (
        output
        if COMMITTED_PROMOTION_RECOVERY_ACTION in output
        else f"{output.rstrip()} {COMMITTED_PROMOTION_RECOVERY_ACTION}"
    )
    committed_metadata = dict(metadata)
    committed_metadata.update(
        {
            "outcome_known": True,
            "outcome_unknown": False,
            "execution_outcome_unknown": False,
            "side_effect_possible": True,
            "retry_safe": False,
            "automatic_retry_allowed": False,
            "authorizes_retry": False,
        }
    )
    return ToolResult(
        tool_name,
        False,
        public_output,
        declare_failure_guidance(
            committed_metadata,
            output=public_output,
            action=COMMITTED_PROMOTION_RECOVERY_ACTION,
        ),
    )


def _unknown_promotion_outcome(memory_id: int, output: str) -> ToolResult:
    public_output = (
        output
        if PROMOTION_UNKNOWN_RECOVERY_ACTION in output
        else f"{output.rstrip()} {PROMOTION_UNKNOWN_RECOVERY_ACTION}"
    )
    return ToolResult(
        "promote_memory_to_decision",
        False,
        public_output,
        declare_failure_guidance(
            _unknown_failure_metadata(_safe_metadata(
                read_only=False,
                risk_level="high_risk",
                requires_approval=True,
                memory_id=memory_id,
                mutation="memory_to_decision_promotion",
                promotion_outcome="unknown",
                projection_state="unknown",
                execution_outcome_unknown=True,
                outcome_known=False,
                side_effect_possible=True,
                durability_uncertain=True,
                retry_safe=False,
                writes_files=True,
                writes_notes=True,
                writes_memory=True,
                writes_database=True,
            )),
            output=public_output,
            action=PROMOTION_UNKNOWN_RECOVERY_ACTION,
        ),
    )


def _unknown_preference_promotion_outcome(memory_id: int, output: str) -> ToolResult:
    public_output = (
        output
        if PROMOTION_UNKNOWN_RECOVERY_ACTION in output
        else f"{output.rstrip()} {PROMOTION_UNKNOWN_RECOVERY_ACTION}"
    )
    return ToolResult(
        "promote_memory_to_preference",
        False,
        public_output,
        declare_failure_guidance(
            _unknown_failure_metadata(_safe_metadata(
                read_only=False,
                risk_level="high_risk",
                requires_approval=True,
                memory_id=memory_id,
                mutation="memory_to_preference_promotion",
                promotion_outcome="unknown",
                projection_state="unknown",
                execution_outcome_unknown=True,
                outcome_known=False,
                side_effect_possible=True,
                durability_uncertain=True,
                retry_safe=False,
                writes_files=True,
                writes_notes=True,
                writes_memory=True,
                writes_database=True,
            )),
            output=public_output,
            action=PROMOTION_UNKNOWN_RECOVERY_ACTION,
        ),
    )


def _unknown_profile_promotion_outcome(memory_id: int, output: str) -> ToolResult:
    public_output = (
        output
        if PROMOTION_UNKNOWN_RECOVERY_ACTION in output
        else f"{output.rstrip()} {PROMOTION_UNKNOWN_RECOVERY_ACTION}"
    )
    return ToolResult(
        "promote_memory_to_profile",
        False,
        public_output,
        declare_failure_guidance(
            _unknown_failure_metadata(_safe_metadata(
                read_only=False,
                risk_level="high_risk",
                requires_approval=True,
                memory_id=memory_id,
                mutation="memory_to_profile_promotion",
                promotion_outcome="unknown",
                projection_state="unknown",
                execution_outcome_unknown=True,
                outcome_known=False,
                side_effect_possible=True,
                durability_uncertain=True,
                retry_safe=False,
                writes_files=True,
                writes_notes=True,
                writes_memory=True,
                writes_database=True,
            )),
            output=public_output,
            action=PROMOTION_UNKNOWN_RECOVERY_ACTION,
        ),
    )


def _validated_target(target: Any, memory_id: int) -> tuple[int, str, str] | None:
    try:
        target_kind = target.target_kind
        memory_target = getattr(target, "memory_target", target)
        target_memory_id = memory_target.memory_id
        revision = memory_target.revision
        binding = memory_target.binding
    except Exception:
        return None
    if (
        _positive_id(target_memory_id) is None
        or _positive_id(revision) is None
        or type(binding) is not str
        or re.fullmatch(r"[0-9a-f]{64}", binding) is None
        or type(target_kind) is not str
        or target_kind not in {"profile", "preference", "decision", "person", "goal"}
        or target_memory_id != memory_id
    ):
        return None
    return revision, binding, target_kind


def _validated_candidate_snapshot(
    target: Any, *, target_kind: str
) -> tuple[str, str, str, str, float, str, str] | None:
    try:
        category = target.category
        title = target.title
        body = target.body
        source = target.source
        confidence = target.confidence
        created_at = target.created_at
        updated_at = target.updated_at
    except Exception:
        return None
    if (
        type(category) is not str
        or type(title) is not str
        or type(body) is not str
        or len(category) > MAX_CANDIDATE_CATEGORY_CHARS
        or len(title) > MAX_CANDIDATE_TITLE_CHARS
        or len(body) > MAX_CANDIDATE_BODY_CHARS
        or type(source) is not str
        or len(source) > MAX_CANDIDATE_SOURCE_CHARS
        or isinstance(confidence, bool)
        or type(confidence) not in {int, float}
        or not math.isfinite(float(confidence))
        or type(created_at) is not str
        or type(updated_at) is not str
        or len(created_at) > MAX_CANDIDATE_TIMESTAMP_CHARS
        or len(updated_at) > MAX_CANDIDATE_TIMESTAMP_CHARS
        or re.match(r"^\d{4}-\d{2}-\d{2}", created_at) is None
        or re.match(r"^\d{4}-\d{2}-\d{2}", updated_at) is None
        or any(separator in created_at or separator in updated_at for separator in ("\r", "\n"))
        or TARGET_KIND_BY_CATEGORY.get(category.strip().casefold()) != target_kind
    ):
        return None
    return (
        category,
        title,
        body,
        source,
        float(confidence),
        created_at,
        updated_at,
    )


def _resolve_candidate_target(store: MemoryStore, memory_id: int) -> Any:
    resolver = getattr(store, "resolve_knowledge_promotion_approval_target", None)
    if not callable(resolver):
        resolver = getattr(store, "resolve_knowledge_promotion_candidate", None)
    if not callable(resolver):
        raise RuntimeError("knowledge promotion resolver is unavailable")
    return resolver(memory_id)


def _mutation_fields(
    mutation: Any, *, expected_memory_id: int
) -> tuple[int, int, DecisionProjectionTarget, MemoryProjectionTarget] | None:
    try:
        decision_id = mutation.decision_id
        memory_id = mutation.memory_id
        decision_target = mutation.decision_projection_target
        memory_target = mutation.memory_projection_target
        decision_target_id = decision_target.decision_id
        decision_target_revision = decision_target.revision
        decision_target_digest = decision_target.source_digest
        memory_target_id = memory_target.memory_id
        memory_target_revision = memory_target.revision
        memory_target_operation = memory_target.operation
        memory_target_digest = memory_target.source_digest
    except Exception:
        return None
    if (
        _positive_id(decision_id) is None
        or _positive_id(memory_id) is None
        or memory_id != expected_memory_id
        or type(decision_target) is not DecisionProjectionTarget
        or type(memory_target) is not MemoryProjectionTarget
    ):
        return None
    if (
        _positive_id(decision_target_id) is None
        or decision_target_id != decision_id
        or _positive_id(decision_target_revision) is None
        or type(decision_target_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", decision_target_digest) is None
        or _positive_id(memory_target_id) is None
        or memory_target_id != memory_id
        or _positive_id(memory_target_revision) is None
        or type(memory_target_operation) is not str
        or memory_target_operation != "publish"
        or type(memory_target_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", memory_target_digest) is None
    ):
        return None
    return decision_id, memory_id, decision_target, memory_target


def _preference_mutation_fields(
    mutation: Any, *, expected_memory_id: int
) -> tuple[int, int, PreferenceProjectionTarget, MemoryProjectionTarget, str] | None:
    try:
        preference_id = mutation.preference_id
        memory_id = mutation.memory_id
        preference_target = mutation.preference_projection_target
        memory_target = mutation.memory_projection_target
        preference_generation = preference_target.generation
        preference_digest = preference_target.source_digest
        memory_target_id = memory_target.memory_id
        memory_target_revision = memory_target.revision
        memory_target_operation = memory_target.operation
        memory_target_digest = memory_target.source_digest
        source_integrity_binding = mutation.source_integrity_binding
    except Exception:
        return None
    if (
        _positive_id(preference_id) is None
        or _positive_id(memory_id) is None
        or memory_id != expected_memory_id
        or type(preference_target) is not PreferenceProjectionTarget
        or type(memory_target) is not MemoryProjectionTarget
    ):
        return None
    if (
        type(preference_generation) is not int
        or preference_generation < 0
        or preference_generation > 9223372036854775807
        or type(preference_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", preference_digest) is None
        or _positive_id(memory_target_id) is None
        or memory_target_id != memory_id
        or _positive_id(memory_target_revision) is None
        or type(memory_target_operation) is not str
        or memory_target_operation != "publish"
        or type(memory_target_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", memory_target_digest) is None
        or type(source_integrity_binding) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_integrity_binding) is None
    ):
        return None
    return (
        preference_id,
        memory_id,
        preference_target,
        memory_target,
        source_integrity_binding,
    )


def make_knowledge_promotion_tools(store: MemoryStore, vault: ObsidianVault):
    def knowledge_promotion_packet(args: dict[str, Any]) -> ToolResult:
        if type(args) is not dict or set(args) != _PACKET_ARGS:
            return _refusal(
                "knowledge_promotion_packet",
                "Provide exactly one positive memory_id to inspect a promotion candidate.",
                "invalid_arguments",
            )
        memory_id = _positive_id(args.get("memory_id"))
        if memory_id is None:
            return _refusal(
                "knowledge_promotion_packet",
                "memory_id must be a positive integer.",
                "invalid_memory_id",
            )
        try:
            target = _resolve_candidate_target(store, memory_id)
        except Exception:
            return _refusal(
                "knowledge_promotion_packet",
                "The exact promotion candidate could not be inspected safely.",
                "candidate_unavailable",
                memory_id=memory_id,
            )
        if target is None:
            return _refusal(
                "knowledge_promotion_packet",
                f"Memory #{memory_id} is not an available knowledge-promotion candidate.",
                "not_found_or_not_candidate",
                memory_id=memory_id,
            )
        validated_target = _validated_target(target, memory_id)
        if validated_target is None:
            return _refusal(
                "knowledge_promotion_packet",
                "The candidate identity was malformed and was not shown.",
                "malformed_candidate_target",
                memory_id=memory_id,
            )
        revision, _binding, target_kind = validated_target
        snapshot = _validated_candidate_snapshot(target, target_kind=target_kind)
        if snapshot is None:
            return _refusal(
                "knowledge_promotion_packet",
                "The candidate snapshot was malformed and was not shown.",
                "malformed_candidate_snapshot",
                memory_id=memory_id,
            )
        category, title, body, source, confidence, created_at, updated_at = snapshot

        category_display = _bounded_packet_text(category, MAX_PACKET_CATEGORY_CHARS)
        title_display = _bounded_packet_text(title, MAX_PACKET_TITLE_CHARS)
        body_display = _bounded_packet_text(body, MAX_PACKET_BODY_CHARS)
        source_display = _bounded_packet_text(source, MAX_CANDIDATE_SOURCE_CHARS)
        lines = [
            f"Knowledge promotion review for memory #{memory_id}, revision {revision}:",
            f"Suggested target from category label: {target_kind}",
            f"Category label: {category_display or '<empty>'}",
            f"Memory title: {title_display or '<empty>'}",
            f"Memory body: {body_display or '<empty>'}",
            f"Source: {source_display or '<empty>'}",
            f"Confidence: {confidence:g}",
            f"Created: {created_at}",
            f"Updated: {updated_at}",
            "",
            "The category is a label only, not semantic proof of the suggested target.",
        ]
        if target_kind == "decision":
            command = (
                f"promote memory {memory_id} revision {revision} token {_binding} to decision: "
                "<explicit title> | <explicit rationale> | <explicit impact>"
            )
            lines.extend(
                [
                    "Review the memory content and supply every decision field explicitly; nothing is inferred from it.",
                    "Strict command template:",
                    f"`{command}`",
                    "Promotion is high risk and will require approval bound to this exact candidate version.",
                ]
            )
        elif target_kind == "preference":
            command = (
                f"promote memory {memory_id} revision {revision} token {_binding} to preference: "
                "<explicit category> | <explicit key> | <explicit value>"
            )
            lines.extend(
                [
                    "Review the memory content and supply every preference field explicitly; nothing is inferred from it.",
                    "The created preference status is fixed to active.",
                    "Strict command template:",
                    f"`{command}`",
                    "Promotion is high risk and will require approval bound to this exact candidate version.",
                ]
            )
        elif target_kind == "profile":
            command = (
                f"promote memory {memory_id} revision {revision} token {_binding} to profile: "
                "<explicit heading> | <explicit category> | <explicit body>"
            )
            lines.extend(
                [
                    "Review the memory content and supply every profile field explicitly; nothing is inferred from it.",
                    "Profile fields must already use their canonical reviewed form.",
                    "Strict command template:",
                    f"`{command}`",
                    "Promotion is high risk and will require approval bound to this exact candidate version.",
                ]
            )
        else:
            lines.append(
                f"Ownership transfer to {target_kind} is not implemented yet; keep this candidate review-only."
            )
        output = "\n".join(lines)
        return ToolResult(
            "knowledge_promotion_packet",
            True,
            output,
            _safe_metadata(
                read_only=True,
                risk_level="read_only",
                reads_private_data=True,
                reads_personal_data=True,
                memory_id=memory_id,
                target_revision=revision,
                target_kind=target_kind,
                classification_basis="category_label_only",
                semantic_proof=False,
                authorizes_promotion=False,
                content_in_output=True,
                state_changed=False,
            ),
        )

    def resolve_promote_memory_to_decision_approval(
        args: dict[str, Any],
    ) -> ApprovalArgumentResolution | ToolResult:
        if type(args) is not dict or set(args) != _RAW_PROMOTION_ARGS:
            return _refusal(
                "promote_memory_to_decision",
                "Provide exactly memory_id, reviewed_revision, review_token, title, rationale, and impact before approval.",
                "invalid_arguments",
                resolver=True,
            )
        memory_id = _positive_id(args.get("memory_id"))
        if memory_id is None:
            return _refusal(
                "promote_memory_to_decision",
                "memory_id must be a positive integer.",
                "invalid_memory_id",
                resolver=True,
            )
        reviewed_revision = _positive_id(args.get("reviewed_revision"))
        if reviewed_revision is None:
            return _refusal(
                "promote_memory_to_decision",
                "reviewed_revision must be the positive revision shown in the promotion packet.",
                "invalid_reviewed_revision",
                memory_id=memory_id,
                resolver=True,
            )
        review_token = args.get("review_token")
        if type(review_token) is not str or re.fullmatch(r"[0-9a-f]{64}", review_token) is None:
            return _refusal(
                "promote_memory_to_decision",
                "review_token must be the opaque token shown in the promotion packet.",
                "invalid_review_token",
                memory_id=memory_id,
                resolver=True,
            )
        fields, reason, output = _validated_decision_fields(args)
        if fields is None:
            return _refusal(
                "promote_memory_to_decision",
                output or "Explicit decision fields are invalid.",
                reason or "invalid_decision_fields",
                memory_id=memory_id,
                resolver=True,
            )
        try:
            target = _resolve_candidate_target(store, memory_id)
        except Exception:
            return _refusal(
                "promote_memory_to_decision",
                "The promotion candidate could not be resolved safely, so no approval was queued.",
                "candidate_unavailable",
                memory_id=memory_id,
                resolver=True,
            )
        if target is None:
            return _refusal(
                "promote_memory_to_decision",
                f"Memory #{memory_id} is not an available promotion candidate, so no approval was queued.",
                "not_found_or_not_candidate",
                memory_id=memory_id,
                resolver=True,
            )
        validated_target = _validated_target(target, memory_id)
        if validated_target is None:
            return _refusal(
                "promote_memory_to_decision",
                "The promotion candidate identity was malformed, so no approval was queued.",
                "malformed_candidate_target",
                memory_id=memory_id,
                resolver=True,
            )
        revision, binding, target_kind = validated_target
        if _validated_candidate_snapshot(target, target_kind=target_kind) is None:
            return _refusal(
                "promote_memory_to_decision",
                "The promotion candidate snapshot was malformed, so no approval was queued.",
                "malformed_candidate_snapshot",
                memory_id=memory_id,
                resolver=True,
            )
        if revision != reviewed_revision:
            return _refusal(
                "promote_memory_to_decision",
                "The candidate changed after review, so no approval was queued. Request a fresh promotion packet.",
                "reviewed_revision_stale",
                memory_id=memory_id,
                resolver=True,
            )
        if not secrets.compare_digest(binding, review_token):
            return _refusal(
                "promote_memory_to_decision",
                "The candidate content changed after review, so no approval was queued. Request a fresh promotion packet.",
                "review_token_stale",
                memory_id=memory_id,
                resolver=True,
            )
        if target_kind != "decision":
            return _refusal(
                "promote_memory_to_decision",
                f"Memory #{memory_id} is not a decision-promotion candidate, so no approval was queued.",
                "wrong_target_kind",
                memory_id=memory_id,
                resolver=True,
            )
        return ApprovalArgumentResolution(
            {
                "memory_id": memory_id,
                "reviewed_revision": reviewed_revision,
                "review_binding": binding,
                "title": fields["title"],
                "rationale": fields["rationale"],
                "impact": fields["impact"],
                "target_revision": revision,
                "target_binding": binding,
            },
            {
                "approval_argument_resolution_status": "resolved",
                "memory_id": memory_id,
                "target_revision": revision,
                "target_kind": "decision",
                "knowledge_promotion_target_bound": True,
            },
        )

    def promote_memory_to_decision(args: dict[str, Any]) -> ToolResult:
        if type(args) is not dict or set(args) != _BOUND_PROMOTION_ARGS:
            return _refusal(
                "promote_memory_to_decision",
                "Promotion requires the exact resolver-bound candidate and all explicit decision fields.",
                "invalid_or_unbound_arguments",
            )
        memory_id = _positive_id(args.get("memory_id"))
        if memory_id is None:
            return _refusal(
                "promote_memory_to_decision",
                "memory_id must be a positive integer.",
                "invalid_memory_id",
            )
        fields, reason, output = _validated_decision_fields(args)
        if fields is None:
            return _refusal(
                "promote_memory_to_decision",
                output or "Explicit decision fields are invalid.",
                reason or "invalid_decision_fields",
                memory_id=memory_id,
            )
        target_revision = args.get("target_revision")
        reviewed_revision = args.get("reviewed_revision")
        review_binding = args.get("review_binding")
        target_binding = args.get("target_binding")
        if (
            _positive_id(target_revision) is None
            or target_revision > 9223372036854775806
            or _positive_id(reviewed_revision) is None
            or reviewed_revision > 9223372036854775806
            or reviewed_revision != target_revision
            or type(review_binding) is not str
            or re.fullmatch(r"[0-9a-f]{64}", review_binding) is None
            or type(target_binding) is not str
            or re.fullmatch(r"[0-9a-f]{64}", target_binding) is None
            or not secrets.compare_digest(review_binding, target_binding)
        ):
            return _refusal(
                "promote_memory_to_decision",
                "The approval-bound promotion target is missing or invalid. Nothing was changed.",
                "unbound_target",
                memory_id=memory_id,
            )

        try:
            result = store.promote_memory_to_decision_exact(
                memory_id,
                target_revision,
                target_binding,
                DecisionRecord(
                    title=fields["title"],
                    rationale=fields["rationale"],
                    impact=fields["impact"],
                ),
            )
        except Exception:
            return _unknown_promotion_outcome(
                memory_id,
                "The promotion outcome is unknown after an internal storage failure. "
                "Do not promote the memory again. Inspect the memory and decisions first; use "
                "projection repair only if the committed source records are intact and their "
                "projection jobs are pending.",
            )
        if type(result) is not MemoryDecisionPromotionResult:
            return _unknown_promotion_outcome(
                memory_id,
                "The promotion returned a malformed result. Do not promote the memory again. "
                "Inspect the memory and decisions first; use projection repair only if the "
                "committed source records are intact and their projection jobs are pending.",
            )
        status = result.status
        if type(status) is not str:
            return _unknown_promotion_outcome(
                memory_id,
                "The promotion returned a malformed result. Do not promote the memory again. "
                "Inspect the memory and decisions first; use projection repair only if the "
                "committed source records are intact and their projection jobs are pending.",
            )
        if status in {"not_found", "stale", "not_candidate", "wrong_target"}:
            if any(
                value is not None
                for value in (
                    result.decision_id,
                    result.memory_id,
                    result.decision_projection_target,
                    result.memory_projection_target,
                )
            ):
                return _unknown_promotion_outcome(
                    memory_id,
                    "The promotion returned contradictory refusal evidence. Do not promote the "
                    "memory again; inspect the memory and decisions first.",
                )
            return _refusal(
                "promote_memory_to_decision",
                "The approved memory candidate changed, disappeared, or is no longer eligible. Nothing was changed.",
                "exact_promotion_refused",
                memory_id=memory_id,
            )
        if status != "promoted":
            return _unknown_promotion_outcome(
                memory_id,
                "The promotion returned an unrecognized outcome. Do not promote the memory again. "
                "Inspect the memory and decisions first; use projection repair only if the "
                "committed source records are intact and their projection jobs are pending.",
            )

        mutation_fields = _mutation_fields(result, expected_memory_id=memory_id)
        if mutation_fields is None:
            return _unknown_promotion_outcome(
                memory_id,
                "The promotion returned incomplete success evidence. Do not promote the memory "
                "again; inspect the memory, decisions, custody link, and projection jobs first.",
            )

        decision_id, mutation_memory_id, decision_target, memory_target = mutation_fields
        decision_status = "error"
        memory_status = "error"
        try:
            decision_outcome = reconcile_decision_projection(store, vault, decision_target)
            decision_status = decision_outcome.status
        except Exception:
            pass
        try:
            memory_outcome = reconcile_memory_projection(
                store,
                vault,
                mutation_memory_id,
                expected_operation=memory_target.operation,
                expected_revision=memory_target.revision,
                expected_source_digest=memory_target.source_digest,
            )
            memory_status = memory_outcome.status
        except Exception:
            pass

        decision_complete = False
        memory_complete = False
        completion_contract_valid = False
        try:
            completion = store.decision_mutation_projection_completion(
                decision_id,
                mutation_memory_id,
                decision_target,
                memory_target,
            )
            if (
                type(completion) is tuple
                and len(completion) == 2
                and all(type(value) is bool for value in completion)
            ):
                decision_complete, memory_complete = completion
                completion_contract_valid = True
        except Exception:
            pass
        decision_row_matches = False
        try:
            decision = store.get_decision(decision_id)
            decision_row_matches = bool(
                decision is not None
                and _row_value(decision, "title") == fields["title"]
                and _row_value(decision, "rationale") == fields["rationale"]
                and _row_value(decision, "impact") == fields["impact"]
            )
        except Exception:
            pass

        source_integrity_verified: bool | None = None
        try:
            source_integrity = store.decision_mutation_source_integrity(
                decision_id,
                mutation_memory_id,
                decision_target,
                memory_target,
            )
            if type(source_integrity) is bool:
                source_integrity_verified = source_integrity
        except Exception:
            pass

        complete = bool(
            mutation_memory_id == memory_id
            and source_integrity_verified
            and completion_contract_valid
            and decision_status == "completed"
            and memory_status == "completed"
            and decision_complete
            and memory_complete
            and decision_row_matches
        )
        metadata = _safe_metadata(
            read_only=False,
            risk_level="high_risk",
            requires_approval=True,
            memory_id=memory_id,
            decision_id=decision_id,
            mutation="memory_to_decision_promotion",
            promotion_committed=True,
            projection_pending=not complete,
            decision_projection_status=decision_status,
            memory_projection_status=memory_status,
            decision_projection_complete=bool(decision_complete),
            memory_projection_complete=bool(memory_complete),
            projection_completion_contract_valid=completion_contract_valid,
            decision_row_verified=decision_row_matches,
            source_integrity_verified=source_integrity_verified,
            source_integrity_status=(
                "verified"
                if source_integrity_verified is True
                else "mismatch"
                if source_integrity_verified is False
                else "unavailable"
            ),
            writes_files=True,
            writes_notes=True,
            writes_memory=True,
            writes_database=True,
            state_changed=True,
        )
        if not complete:
            if source_integrity_verified is None:
                return _committed_promotion_failure(
                    "promote_memory_to_decision",
                    f"Promoted memory #{memory_id} to decision #{decision_id}, but source "
                    "integrity verification was unavailable. Do not promote the memory again. "
                    "Inspect both records, their custody link, and projection jobs before deciding "
                    "whether projection repair is appropriate.",
                    metadata,
                )
            if not (
                mutation_memory_id == memory_id
                and decision_row_matches
                and source_integrity_verified
            ):
                return _committed_promotion_failure(
                    "promote_memory_to_decision",
                    f"Promoted memory #{memory_id} to decision #{decision_id}, but source "
                    "integrity verification failed. Do not promote the memory again. "
                    f"Inspect `show memory {memory_id}` and `show decision {decision_id}`; "
                    "projection repair alone cannot resolve a source-integrity mismatch.",
                    metadata,
                )
            if not completion_contract_valid:
                return _committed_promotion_failure(
                    "promote_memory_to_decision",
                    f"Promoted memory #{memory_id} to decision #{decision_id}, but the "
                    "durability verification contract was malformed. Do not promote the "
                    "memory again. Inspect both records and their projection jobs before repair.",
                    metadata,
                )
            repairs = []
            if decision_status != "completed" or not decision_complete or not decision_row_matches:
                repairs.append("`repair decision projections`")
            if memory_status != "completed" or not memory_complete or mutation_memory_id != memory_id:
                repairs.append("`repair memory projections`")
            repair_text = " and ".join(repairs) or "the projection repair commands"
            return _committed_promotion_failure(
                "promote_memory_to_decision",
                f"Promoted memory #{memory_id} to decision #{decision_id}, but durable "
                f"projection verification remains pending. Run {repair_text}; "
                "do not promote the memory again.",
                metadata,
            )
        return ToolResult(
            "promote_memory_to_decision",
            True,
            f"Promoted memory #{memory_id} to decision #{decision_id}; both durable projections were verified.",
            metadata,
        )

    def resolve_promote_memory_to_preference_approval(
        args: dict[str, Any],
    ) -> ApprovalArgumentResolution | ToolResult:
        if type(args) is not dict or set(args) != _RAW_PREFERENCE_PROMOTION_ARGS:
            return _refusal(
                "promote_memory_to_preference",
                "Provide exactly memory_id, reviewed_revision, review_token, category, key, and value before approval.",
                "invalid_arguments",
                resolver=True,
            )
        memory_id = _positive_id(args.get("memory_id"))
        if memory_id is None:
            return _refusal(
                "promote_memory_to_preference",
                "memory_id must be a positive integer.",
                "invalid_memory_id",
                resolver=True,
            )
        reviewed_revision = _positive_id(args.get("reviewed_revision"))
        if reviewed_revision is None:
            return _refusal(
                "promote_memory_to_preference",
                "reviewed_revision must be the positive revision shown in the promotion packet.",
                "invalid_reviewed_revision",
                memory_id=memory_id,
                resolver=True,
            )
        review_token = args.get("review_token")
        if type(review_token) is not str or re.fullmatch(r"[0-9a-f]{64}", review_token) is None:
            return _refusal(
                "promote_memory_to_preference",
                "review_token must be the opaque token shown in the promotion packet.",
                "invalid_review_token",
                memory_id=memory_id,
                resolver=True,
            )
        fields, reason, output = _validated_preference_fields(args)
        if fields is None:
            return _refusal(
                "promote_memory_to_preference",
                output or "Explicit preference fields are invalid.",
                reason or "invalid_preference_fields",
                memory_id=memory_id,
                resolver=True,
            )
        try:
            target = _resolve_candidate_target(store, memory_id)
        except Exception:
            return _refusal(
                "promote_memory_to_preference",
                "The promotion candidate could not be resolved safely, so no approval was queued.",
                "candidate_unavailable",
                memory_id=memory_id,
                resolver=True,
            )
        if target is None:
            return _refusal(
                "promote_memory_to_preference",
                f"Memory #{memory_id} is not an available promotion candidate, so no approval was queued.",
                "not_found_or_not_candidate",
                memory_id=memory_id,
                resolver=True,
            )
        validated_target = _validated_target(target, memory_id)
        if validated_target is None:
            return _refusal(
                "promote_memory_to_preference",
                "The promotion candidate identity was malformed, so no approval was queued.",
                "malformed_candidate_target",
                memory_id=memory_id,
                resolver=True,
            )
        revision, binding, target_kind = validated_target
        if _validated_candidate_snapshot(target, target_kind=target_kind) is None:
            return _refusal(
                "promote_memory_to_preference",
                "The promotion candidate snapshot was malformed, so no approval was queued.",
                "malformed_candidate_snapshot",
                memory_id=memory_id,
                resolver=True,
            )
        if revision != reviewed_revision:
            return _refusal(
                "promote_memory_to_preference",
                "The candidate changed after review, so no approval was queued. Request a fresh promotion packet.",
                "reviewed_revision_stale",
                memory_id=memory_id,
                resolver=True,
            )
        if not secrets.compare_digest(binding, review_token):
            return _refusal(
                "promote_memory_to_preference",
                "The candidate content changed after review, so no approval was queued. Request a fresh promotion packet.",
                "review_token_stale",
                memory_id=memory_id,
                resolver=True,
            )
        if target_kind != "preference":
            return _refusal(
                "promote_memory_to_preference",
                f"Memory #{memory_id} is not a preference-promotion candidate, so no approval was queued.",
                "wrong_target_kind",
                memory_id=memory_id,
                resolver=True,
            )
        return ApprovalArgumentResolution(
            {
                "memory_id": memory_id,
                "reviewed_revision": reviewed_revision,
                "review_binding": binding,
                "category": fields["category"],
                "key": fields["key"],
                "value": fields["value"],
                "target_revision": revision,
                "target_binding": binding,
            },
            {
                "approval_argument_resolution_status": "resolved",
                "memory_id": memory_id,
                "target_revision": revision,
                "target_kind": "preference",
                "knowledge_promotion_target_bound": True,
            },
        )

    def promote_memory_to_preference(args: dict[str, Any]) -> ToolResult:
        if type(args) is not dict or set(args) != _BOUND_PREFERENCE_PROMOTION_ARGS:
            return _refusal(
                "promote_memory_to_preference",
                "Promotion requires the exact resolver-bound candidate and all explicit preference fields.",
                "invalid_or_unbound_arguments",
            )
        memory_id = _positive_id(args.get("memory_id"))
        if memory_id is None:
            return _refusal(
                "promote_memory_to_preference",
                "memory_id must be a positive integer.",
                "invalid_memory_id",
            )
        fields, reason, output = _validated_preference_fields(args)
        if fields is None:
            return _refusal(
                "promote_memory_to_preference",
                output or "Explicit preference fields are invalid.",
                reason or "invalid_preference_fields",
                memory_id=memory_id,
            )
        target_revision = args.get("target_revision")
        reviewed_revision = args.get("reviewed_revision")
        review_binding = args.get("review_binding")
        target_binding = args.get("target_binding")
        if (
            _positive_id(target_revision) is None
            or target_revision > 9223372036854775806
            or _positive_id(reviewed_revision) is None
            or reviewed_revision > 9223372036854775806
            or reviewed_revision != target_revision
            or type(review_binding) is not str
            or re.fullmatch(r"[0-9a-f]{64}", review_binding) is None
            or type(target_binding) is not str
            or re.fullmatch(r"[0-9a-f]{64}", target_binding) is None
            or not secrets.compare_digest(review_binding, target_binding)
        ):
            return _refusal(
                "promote_memory_to_preference",
                "The approval-bound promotion target is missing or invalid. Nothing was changed.",
                "unbound_target",
                memory_id=memory_id,
            )

        try:
            result = store.promote_memory_to_preference_exact(
                memory_id,
                target_revision,
                target_binding,
                PreferenceRecord(
                    category=fields["category"],
                    key=fields["key"],
                    value=fields["value"],
                    status="active",
                ),
            )
        except Exception:
            return _unknown_preference_promotion_outcome(
                memory_id,
                "The promotion outcome is unknown after an internal storage failure. "
                "Do not promote the memory again. Inspect the memory and preferences first; use "
                "projection repair only if the committed source records are intact and their "
                "projection jobs are pending.",
            )
        if type(result) is not MemoryPreferencePromotionResult:
            return _unknown_preference_promotion_outcome(
                memory_id,
                "The promotion returned a malformed result. Do not promote the memory again. "
                "Inspect the memory and preferences first; use projection repair only if the "
                "committed source records are intact and their projection jobs are pending.",
            )
        status = result.status
        if type(status) is not str:
            return _unknown_preference_promotion_outcome(
                memory_id,
                "The promotion returned a malformed result. Do not promote the memory again. "
                "Inspect the memory and preferences first; use projection repair only if the "
                "committed source records are intact and their projection jobs are pending.",
            )
        known_refusals = {
            "not_found",
            "stale",
            "not_candidate",
            "wrong_target",
            "identity_exists",
            "identity_conflict",
        }
        if status in known_refusals:
            if any(
                value is not None
                for value in (
                    result.preference_id,
                    result.memory_id,
                    result.preference_projection_target,
                    result.memory_projection_target,
                    result.source_integrity_binding,
                )
            ):
                return _unknown_preference_promotion_outcome(
                    memory_id,
                    "The promotion returned contradictory refusal evidence. Do not promote the "
                    "memory again; inspect the memory and preferences first.",
                )
            return _refusal(
                "promote_memory_to_preference",
                "The approved memory candidate changed, disappeared, is no longer eligible, or the exact preference already exists. Nothing was changed.",
                "exact_promotion_refused",
                memory_id=memory_id,
            )
        if status != "promoted":
            return _unknown_preference_promotion_outcome(
                memory_id,
                "The promotion returned an unrecognized outcome. Do not promote the memory again. "
                "Inspect the memory and preferences first; use projection repair only if the "
                "committed source records are intact and their projection jobs are pending.",
            )

        mutation_fields = _preference_mutation_fields(result, expected_memory_id=memory_id)
        if mutation_fields is None:
            return _unknown_preference_promotion_outcome(
                memory_id,
                "The promotion returned incomplete success evidence. Do not promote the memory "
                "again; inspect the memory, preference, custody link, and projection jobs first.",
            )

        (
            preference_id,
            mutation_memory_id,
            preference_target,
            memory_target,
            source_integrity_binding,
        ) = mutation_fields
        preference_status = "error"
        memory_status = "error"
        try:
            preference_outcome = reconcile_preference_projection(
                store,
                vault,
                preference_target,
            )
            preference_status = preference_outcome.status
        except Exception:
            pass
        try:
            memory_outcome = reconcile_memory_projection(
                store,
                vault,
                mutation_memory_id,
                expected_operation=memory_target.operation,
                expected_revision=memory_target.revision,
                expected_source_digest=memory_target.source_digest,
            )
            memory_status = memory_outcome.status
        except Exception:
            pass

        preference_complete = False
        memory_complete = False
        completion_contract_valid = False
        try:
            completion = store.preference_mutation_projection_completion(
                preference_id,
                mutation_memory_id,
                preference_target,
                memory_target,
            )
            if (
                type(completion) is tuple
                and len(completion) == 2
                and all(type(value) is bool for value in completion)
            ):
                preference_complete, memory_complete = completion
                completion_contract_valid = True
        except Exception:
            pass

        preference_row_matches = False
        try:
            preference = store.get_preference_by_id(preference_id)
            preference_row_matches = bool(
                preference is not None
                and _row_value(preference, "category") == fields["category"]
                and _row_value(preference, "key") == fields["key"]
                and _row_value(preference, "value") == fields["value"]
                and _row_value(preference, "status") == "active"
            )
        except Exception:
            pass

        source_integrity_verified: bool | None = None
        try:
            source_integrity = store.preference_mutation_source_integrity(
                preference_id,
                mutation_memory_id,
                preference_target,
                memory_target,
                source_integrity_binding,
            )
            if type(source_integrity) is bool:
                source_integrity_verified = source_integrity
        except Exception:
            pass

        complete = bool(
            mutation_memory_id == memory_id
            and source_integrity_verified
            and completion_contract_valid
            and preference_status == "completed"
            and memory_status == "completed"
            and preference_complete
            and memory_complete
            and preference_row_matches
        )
        metadata = _safe_metadata(
            read_only=False,
            risk_level="high_risk",
            requires_approval=True,
            memory_id=memory_id,
            preference_id=preference_id,
            target_kind="preference",
            mutation="memory_to_preference_promotion",
            promotion_committed=True,
            projection_pending=not complete,
            preference_projection_status=preference_status,
            memory_projection_status=memory_status,
            preference_projection_complete=bool(preference_complete),
            memory_projection_complete=bool(memory_complete),
            projection_completion_contract_valid=completion_contract_valid,
            preference_row_verified=preference_row_matches,
            source_integrity_verified=source_integrity_verified,
            source_integrity_status=(
                "verified"
                if source_integrity_verified is True
                else "mismatch"
                if source_integrity_verified is False
                else "unavailable"
            ),
            retry_safe=False,
            writes_files=True,
            writes_notes=True,
            writes_memory=True,
            writes_database=True,
            state_changed=True,
        )
        if not complete:
            if source_integrity_verified is None:
                return _committed_promotion_failure(
                    "promote_memory_to_preference",
                    f"Promoted memory #{memory_id} to preference #{preference_id}, but source "
                    "integrity verification was unavailable. Do not promote the memory again. "
                    "Inspect both records, their custody link, and projection jobs before deciding "
                    "whether projection repair is appropriate.",
                    metadata,
                )
            if not (
                mutation_memory_id == memory_id
                and preference_row_matches
                and source_integrity_verified
            ):
                return _committed_promotion_failure(
                    "promote_memory_to_preference",
                    f"Promoted memory #{memory_id} to preference #{preference_id}, but source "
                    "integrity verification failed. Do not promote the memory again. Inspect the "
                    "memory and preference source records; projection repair alone cannot resolve "
                    "a source-integrity mismatch.",
                    metadata,
                )
            if not completion_contract_valid:
                return _committed_promotion_failure(
                    "promote_memory_to_preference",
                    f"Promoted memory #{memory_id} to preference #{preference_id}, but the "
                    "durability verification contract was malformed. Do not promote the memory "
                    "again. Inspect both records and their projection jobs before repair.",
                    metadata,
                )
            repairs = []
            if preference_status != "completed" or not preference_complete:
                repairs.append("`repair preference projections`")
            if memory_status != "completed" or not memory_complete:
                repairs.append("`repair memory projections`")
            repair_text = " and ".join(repairs) or "the projection repair commands"
            return _committed_promotion_failure(
                "promote_memory_to_preference",
                f"Promoted memory #{memory_id} to preference #{preference_id}, but durable "
                f"projection verification remains pending. Run {repair_text}; do not promote "
                "the memory again.",
                metadata,
            )
        return ToolResult(
            "promote_memory_to_preference",
            True,
            f"Promoted memory #{memory_id} to preference #{preference_id}; both durable projections were verified.",
            metadata,
        )

    return (
        knowledge_promotion_packet,
        promote_memory_to_decision,
        resolve_promote_memory_to_decision_approval,
        promote_memory_to_preference,
        resolve_promote_memory_to_preference_approval,
    )


def make_profile_promotion_tools(store: MemoryStore, vault: ObsidianVault):
    """Expose the profile branch without widening the existing promotion factory."""

    def resolve_promote_memory_to_profile_approval(
        args: dict[str, Any],
    ) -> ApprovalArgumentResolution | ToolResult:
        if type(args) is not dict or set(args) != _RAW_PROFILE_PROMOTION_ARGS:
            return _refusal(
                "promote_memory_to_profile",
                "Provide exactly memory_id, reviewed_revision, review_token, heading, category, and body before approval.",
                "invalid_arguments",
                resolver=True,
            )
        memory_id = _positive_id(args.get("memory_id"))
        reviewed_revision = _positive_id(args.get("reviewed_revision"))
        review_token = args.get("review_token")
        if memory_id is None:
            return _refusal(
                "promote_memory_to_profile",
                "memory_id must be a positive integer.",
                "invalid_memory_id",
                resolver=True,
            )
        if reviewed_revision is None:
            return _refusal(
                "promote_memory_to_profile",
                "reviewed_revision must be the positive revision shown in the promotion packet.",
                "invalid_reviewed_revision",
                memory_id=memory_id,
                resolver=True,
            )
        if type(review_token) is not str or re.fullmatch(r"[0-9a-f]{64}", review_token) is None:
            return _refusal(
                "promote_memory_to_profile",
                "review_token must be the opaque token shown in the promotion packet.",
                "invalid_review_token",
                memory_id=memory_id,
                resolver=True,
            )
        fields, reason, output = _validated_profile_fields(args)
        if fields is None:
            return _refusal(
                "promote_memory_to_profile",
                output or "Explicit profile fields are invalid.",
                reason or "invalid_profile_fields",
                memory_id=memory_id,
                resolver=True,
            )
        try:
            target = _resolve_candidate_target(store, memory_id)
        except Exception:
            target = None
        if target is None:
            return _refusal(
                "promote_memory_to_profile",
                f"Memory #{memory_id} is not an available profile-promotion candidate, so no approval was queued.",
                "not_found_or_not_candidate",
                memory_id=memory_id,
                resolver=True,
            )
        validated_target = _validated_target(target, memory_id)
        if (
            validated_target is None
            or _validated_candidate_snapshot(target, target_kind=validated_target[2]) is None
        ):
            return _refusal(
                "promote_memory_to_profile",
                "The promotion candidate was malformed, so no approval was queued.",
                "malformed_candidate_target",
                memory_id=memory_id,
                resolver=True,
            )
        revision, binding, target_kind = validated_target
        if revision != reviewed_revision:
            return _refusal(
                "promote_memory_to_profile",
                "The candidate changed after review, so no approval was queued. Request a fresh promotion packet.",
                "reviewed_revision_stale",
                memory_id=memory_id,
                resolver=True,
            )
        if not secrets.compare_digest(binding, review_token):
            return _refusal(
                "promote_memory_to_profile",
                "The candidate content changed after review, so no approval was queued. Request a fresh promotion packet.",
                "review_token_stale",
                memory_id=memory_id,
                resolver=True,
            )
        if target_kind != "profile":
            return _refusal(
                "promote_memory_to_profile",
                f"Memory #{memory_id} is not a profile-promotion candidate, so no approval was queued.",
                "wrong_target_kind",
                memory_id=memory_id,
                resolver=True,
            )
        return ApprovalArgumentResolution(
            {
                "memory_id": memory_id,
                "reviewed_revision": reviewed_revision,
                "review_binding": binding,
                **fields,
                "target_revision": revision,
                "target_binding": binding,
            },
            {
                "approval_argument_resolution_status": "resolved",
                "memory_id": memory_id,
                "target_revision": revision,
                "target_kind": "profile",
                "knowledge_promotion_target_bound": True,
            },
        )

    def promote_memory_to_profile(args: dict[str, Any]) -> ToolResult:
        if type(args) is not dict or set(args) != _BOUND_PROFILE_PROMOTION_ARGS:
            return _refusal(
                "promote_memory_to_profile",
                "Promotion requires the exact resolver-bound candidate and all explicit profile fields.",
                "invalid_or_unbound_arguments",
            )
        memory_id = _positive_id(args.get("memory_id"))
        fields, reason, output = _validated_profile_fields(args)
        if memory_id is None:
            return _refusal(
                "promote_memory_to_profile",
                "memory_id must be a positive integer.",
                "invalid_memory_id",
            )
        if fields is None:
            return _refusal(
                "promote_memory_to_profile",
                output or "Explicit profile fields are invalid.",
                reason or "invalid_profile_fields",
                memory_id=memory_id,
            )
        target_revision = args.get("target_revision")
        reviewed_revision = args.get("reviewed_revision")
        review_binding = args.get("review_binding")
        target_binding = args.get("target_binding")
        if (
            _positive_id(target_revision) is None
            or target_revision > 9223372036854775806
            or _positive_id(reviewed_revision) is None
            or reviewed_revision > 9223372036854775806
            or reviewed_revision != target_revision
            or type(review_binding) is not str
            or re.fullmatch(r"[0-9a-f]{64}", review_binding) is None
            or type(target_binding) is not str
            or re.fullmatch(r"[0-9a-f]{64}", target_binding) is None
            or not secrets.compare_digest(review_binding, target_binding)
        ):
            return _refusal(
                "promote_memory_to_profile",
                "The approval-bound promotion target is missing or invalid. Nothing was changed.",
                "unbound_target",
                memory_id=memory_id,
            )
        source_key = profile_note_source_key(
            fields["heading"], fields["body"], fields["category"]
        )
        try:
            result = store.promote_memory_to_profile_note_exact(
                memory_id,
                target_revision,
                target_binding,
                MemoryRecord(
                    category=fields["category"],
                    title=fields["heading"],
                    body=fields["body"],
                    source="profile",
                    confidence=1.0,
                ),
                source_key,
            )
        except Exception:
            return _unknown_profile_promotion_outcome(
                memory_id,
                "The profile promotion outcome is unknown after an internal storage failure. Do not promote the memory again; inspect its custody and pending profile projections first.",
            )
        if type(result) is not MemoryProfilePromotionResult or type(result.status) is not str:
            return _unknown_profile_promotion_outcome(
                memory_id,
                "The profile promotion returned a malformed result. Do not promote the memory again; inspect its custody and pending profile projections first.",
            )
        if result.status in {
            "not_found",
            "stale",
            "not_candidate",
            "wrong_target",
            "identity_exists",
            "capacity_reached",
        }:
            if any(
                value is not None
                for value in (
                    result.memory_id,
                    result.source_key,
                    result.memory_projection_target,
                )
            ):
                return _unknown_profile_promotion_outcome(
                    memory_id,
                    "The profile promotion returned contradictory refusal evidence. Do not promote the memory again; inspect its custody first.",
                )
            return _refusal(
                "promote_memory_to_profile",
                "The approved memory candidate changed, disappeared, is no longer eligible, or the exact profile note already exists. Nothing was changed.",
                "exact_promotion_refused",
                memory_id=memory_id,
            )
        if (
            result.status != "promoted"
            or result.memory_id != memory_id
            or result.source_key != source_key
            or result.memory_projection_target is None
        ):
            return _unknown_profile_promotion_outcome(
                memory_id,
                "The profile promotion returned incomplete success evidence. Do not promote the memory again; inspect its custody and pending profile projections first.",
            )
        projection_status = "pending_error"
        try:
            source_row = store.get_profile_projection_source(source_key, memory_id)
            if source_row is not None:
                projection_status = reconcile_profile_projection_candidate(
                    store, vault, source_row
                ).status
        except Exception:
            pass
        profile_integrity_verified: bool | None = None
        try:
            with store.memory_read_custody(memory_id) as custody:
                note = custody.profile_note
                profile_integrity_verified = bool(
                    custody.profile_owned
                    and note is not None
                    and note.source_key == source_key
                    and note.category == fields["category"]
                    and note.heading == fields["heading"]
                    and note.body == fields["body"]
                )
        except Exception:
            pass
        complete = projection_status == "completed" and profile_integrity_verified is True
        metadata = _safe_metadata(
            read_only=False,
            risk_level="high_risk",
            requires_approval=True,
            memory_id=memory_id,
            target_kind="profile",
            mutation="memory_to_profile_promotion",
            promotion_committed=True,
            projection_pending=not complete,
            profile_projection_status=projection_status,
            profile_integrity_verified=profile_integrity_verified,
            retry_safe=False,
            writes_files=True,
            writes_notes=True,
            writes_memory=True,
            writes_database=True,
            state_changed=True,
        )
        if not complete:
            return _committed_promotion_failure(
                "promote_memory_to_profile",
                f"Promoted memory #{memory_id} to an owned profile note, but durable projection verification remains pending. Do not promote the memory again; inspect pending profile projections before repair.",
                metadata,
            )
        return ToolResult(
            "promote_memory_to_profile",
            True,
            f"Promoted memory #{memory_id} to an owned profile note; durable projection custody was verified.",
            metadata,
        )

    return promote_memory_to_profile, resolve_promote_memory_to_profile_approval


__all__ = ["make_knowledge_promotion_tools", "make_profile_promotion_tools"]
