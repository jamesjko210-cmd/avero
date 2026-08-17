"""Generate text with the configured model, then write it into the focused app.

`compose_and_write` is HIGH_RISK because the final step controls the keyboard
or clipboard paste target. The Jarvis runtime must approval-gate it before any
real typing/pasting happens.
"""

from __future__ import annotations

import re
from typing import Any

from jarvis_v2.agent.model_provider import (
    ModelProviderError,
    generate_model_text,
    normalized_model_provider,
    provider_output_token_limit,
)
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools import writer_connector


MAX_PROMPT_CHARS = 1200
MAX_GENERATED_CHARS = writer_connector.MAX_TEXT_CHARS
MAX_GENERATED_TOKENS = 2000
VALID_MODES = {"human", "paste"}
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": True,
        "calls_external_service": False,
        "executes_tools": True,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": True,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": True,
        "controls_computer": True,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def _short(value: Any, *, limit: int = MAX_PROMPT_CHARS) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_raw(value: Any, *, limit: int = 80) -> str:
    text = "" if value is None else str(value).strip()
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_model_name(value: Any) -> str:
    model = _short_raw(value, limit=80)
    if not model or model == "<local-path>":
        return "the configured chat model"
    return model


def _compose_model_recovery_hint(config: JarvisConfig, exc: Exception | None = None) -> str:
    model = _safe_model_name(getattr(config, "chat_model", ""))
    if normalized_model_provider(getattr(config, "model_provider", "ollama")) == "openai":
        if isinstance(exc, ModelProviderError):
            return f"{exc.recovery_hint} Diagnostic: compose_model_unavailable."
        return (
            "Run `model routing status` to check OpenAI configuration before retrying. "
            "Diagnostic: compose_model_unavailable."
        )
    return (
        "Run `model routing status`, start Ollama, and if the configured chat model is missing "
        f"run `ollama pull {model}`. Diagnostic: compose_model_unavailable."
    )


def _mode(value: Any) -> tuple[str, str | None]:
    raw = str(value or "human").strip().lower()
    if raw in VALID_MODES:
        return raw, None
    return "human", _short_raw(raw) or None


def _looks_like_local_path(value: str) -> bool:
    return bool(LOCAL_PATH_RE.search(value))


def _compose_boundaries(
    *,
    calls_model: bool = False,
    calls_external_service: bool = False,
    executes_tools: bool = False,
    controls_computer: bool = False,
) -> dict[str, bool]:
    return {
        "calls_model": calls_model,
        "calls_external_service": calls_external_service,
        "executes_tools": executes_tools,
        "reads_personal_data": False,
        "reads_private_data": False,
        "executes_side_effect": controls_computer,
        "external_side_effect": False,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": controls_computer,
        "controls_computer": controls_computer,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _compose_write_handoff(
    *,
    prompt: str,
    status: str,
    reason: str = "",
    mode: str = "human",
    raw_mode: str | None = None,
    generated_chars: int = 0,
    generated_truncated: bool = False,
    writer_tool: str = "",
    writer_ok: bool | None = None,
    writer_metadata: dict[str, Any] | None = None,
    calls_model: bool = False,
    calls_external_service: bool = False,
    executes_tools: bool = False,
    controls_computer: bool = False,
    model_provider: str = "",
) -> dict[str, Any]:
    return {
        "compose_write_handoff": {
            "source": "compose_and_write",
            "status": status,
            "reason": reason,
            "prompt_preview": _short_raw(prompt, limit=160),
            "prompt_chars": len(prompt),
            "local_path_prompt": bool(LOCAL_PATH_RE.search(prompt or "")),
            "mode": mode,
            "raw_mode": raw_mode,
            "generated_chars": generated_chars,
            "generated_truncated": generated_truncated,
            "generated_content_in_metadata": False,
            "model_provider": model_provider,
            "writer_tool": writer_tool,
            "writer_ok": writer_ok,
            "writer_metadata": writer_metadata or {},
            "approval_required_before_execution": controls_computer,
            "manual_review_required": controls_computer,
            "authorizes_execution": False,
            "authorizes_completion_claim": False,
            "approval_granted": False,
            "next_safe_command": "review pending approvals" if controls_computer else "compose and write <prompt>",
            "boundaries": _compose_boundaries(
                calls_model=calls_model,
                calls_external_service=calls_external_service,
                executes_tools=executes_tools,
                controls_computer=controls_computer,
            ),
        }
    }


def _generate_text(config: JarvisConfig, prompt: str) -> str:
    return generate_model_text(
        provider=config.model_provider,
        model=config.chat_model,
        messages=[
            {
                "role": "system",
                "content": "Draft clean, useful text for the user to write. Return only the final text.",
            },
            {"role": "user", "content": prompt},
        ],
        timeout_seconds=config.chat_timeout_seconds,
        max_output_tokens=provider_output_token_limit(
            config.model_provider,
            local_output_tokens=MAX_GENERATED_TOKENS,
            openai_output_tokens=config.openai_max_output_tokens,
        ),
        temperature=0.4,
        reasoning_effort=config.chat_reasoning_effort,
        keep_alive="30m",
    )


def make_compose_tools(config: JarvisConfig):
    def compose_and_write(args: dict[str, Any]) -> ToolResult:
        model_provider = normalized_model_provider(getattr(config, "model_provider", "ollama"))
        calls_external_service = model_provider == "openai"
        prompt = _short(args.get("prompt") or args.get("text"))
        if not prompt:
            return ToolResult(
                "compose_and_write",
                False,
                "What should I compose?",
                _safe_metadata(
                    calls_model=False,
                    executes_tools=False,
                    executes_side_effect=False,
                    requires_approval=False,
                    controls_computer=False,
                    reason="missing_prompt",
                    **_compose_write_handoff(prompt=prompt, status="refused", reason="missing_prompt", calls_model=False),
                ),
            )
        if _looks_like_local_path(prompt):
            return ToolResult(
                "compose_and_write",
                False,
                "Please give me text to compose, not a local file path.",
                _safe_metadata(
                    calls_model=False,
                    executes_tools=False,
                    executes_side_effect=False,
                    requires_approval=False,
                    controls_computer=False,
                    reason="invalid_prompt",
                    local_path_prompt=True,
                    prompt_chars=len(prompt),
                    **_compose_write_handoff(prompt=prompt, status="refused", reason="invalid_prompt"),
                ),
            )

        mode, raw_mode = _mode(args.get("mode"))
        try:
            generated = _generate_text(config, prompt)[:MAX_GENERATED_CHARS]
        except Exception as exc:
            recovery_hint = _compose_model_recovery_hint(config, exc)
            return ToolResult(
                "compose_and_write",
                False,
                (
                    "I couldn't compose that text because the configured model is unavailable right now. "
                    f"{recovery_hint} I did not type or paste anything."
                ),
                _safe_metadata(
                    executes_tools=False,
                    executes_side_effect=False,
                    requires_approval=False,
                    controls_computer=False,
                    calls_external_service=calls_external_service,
                    model_provider=model_provider,
                    prompt_chars=len(prompt),
                    mode=mode,
                    raw_mode=raw_mode,
                    reason="model_error",
                    exception_type=type(exc).__name__,
                    model_recovery_hint=recovery_hint,
                    diagnostic="compose_model_unavailable",
                    **_compose_write_handoff(
                        prompt=prompt,
                        status="failed",
                        reason="model_error",
                        mode=mode,
                        raw_mode=raw_mode,
                        calls_model=True,
                        calls_external_service=calls_external_service,
                        model_provider=model_provider,
                    ),
                ),
            )
        if not generated.strip():
            return ToolResult(
                "compose_and_write",
                False,
                "The model did not return any text, so I did not type anything.",
                _safe_metadata(
                    executes_tools=False,
                    executes_side_effect=False,
                    requires_approval=False,
                    controls_computer=False,
                    calls_external_service=calls_external_service,
                    model_provider=model_provider,
                    prompt_chars=len(prompt),
                    mode=mode,
                    raw_mode=raw_mode,
                    reason="empty_generation",
                    **_compose_write_handoff(
                        prompt=prompt,
                        status="failed",
                        reason="empty_generation",
                        mode=mode,
                        raw_mode=raw_mode,
                        calls_model=True,
                        calls_external_service=calls_external_service,
                        model_provider=model_provider,
                    ),
                ),
            )

        writer_tools = {tool.name: tool for tool in writer_connector.make_writer_tools(config)}
        writer_name = "paste_text" if mode == "paste" else "human_write"
        writer_args: dict[str, Any] = {"text": generated}
        if "countdown" in args:
            writer_args["countdown"] = args.get("countdown")
        if writer_name == "human_write":
            writer_args["speed"] = args.get("speed") or "normal"
            if "typos" in args:
                writer_args["typos"] = args.get("typos")
        writer_result = writer_tools[writer_name].handler(writer_args)
        return ToolResult(
            "compose_and_write",
            writer_result.ok,
            f"Composed {len(generated)} characters, then {writer_result.output}",
            _safe_metadata(
                calls_external_service=calls_external_service,
                model_provider=model_provider,
                prompt_chars=len(prompt),
                generated_chars=len(generated),
                generated_truncated=len(generated) >= MAX_GENERATED_CHARS,
                mode=mode,
                raw_mode=raw_mode,
                writer_tool=writer_name,
                writer_ok=writer_result.ok,
                writer_metadata=writer_result.metadata,
                **_compose_write_handoff(
                    prompt=prompt,
                    status="writer_ok" if writer_result.ok else "writer_failed",
                    reason="" if writer_result.ok else "writer_error",
                    mode=mode,
                    raw_mode=raw_mode,
                    generated_chars=len(generated),
                    generated_truncated=len(generated) >= MAX_GENERATED_CHARS,
                    writer_tool=writer_name,
                    writer_ok=writer_result.ok,
                    writer_metadata=writer_result.metadata,
                    calls_model=True,
                    calls_external_service=calls_external_service,
                    executes_tools=True,
                    controls_computer=True,
                    model_provider=model_provider,
                ),
            ),
        )

    from jarvis_v2.tools.registry import Tool

    return [
        Tool(
            "compose_and_write",
            "Generate text with the configured model, then type or paste it into the focused window. Args: prompt, mode (human|paste), speed.",
            RiskLevel.HIGH_RISK,
            compose_and_write,
            "personal",
        )
    ]
