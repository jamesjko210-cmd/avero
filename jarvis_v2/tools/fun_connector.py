"""Light fun tools for Jarvis V2 (dad jokes and local playful prompts)."""

from __future__ import annotations

import random
import re
from typing import Any

from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools._http import friendly_http_error, http_get_json


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "calls_model": False,
        "calls_external_service": True,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_personal_data": False,
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


def _fun_boundaries(*, calls_external_service: bool = True) -> dict[str, bool]:
    return {
        "calls_model": False,
        "calls_external_service": calls_external_service,
        "executes_tools": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "reads_personal_data": False,
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


def _joke_handoff(
    *,
    status: str,
    reason: str = "",
    joke_returned: bool = False,
    joke_chars: int = 0,
    exception_type: str = "",
) -> dict[str, Any]:
    next_safe_command = "tell me a joke"
    boundaries = _fun_boundaries(calls_external_service=True)
    handoff = {
        "source": "tell_joke",
        "status": status,
        "reason": reason,
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "joke_returned": joke_returned,
        "joke_chars": joke_chars,
        "content_in_handoff": False,
        "content_in_metadata": False,
        "retry_safe": status in {"empty", "unavailable"},
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "exception_type": exception_type,
        "boundaries": boundaries,
    }
    return {
        "joke_handoff_ready": True,
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        "content_in_handoff": handoff["content_in_handoff"],
        "joke_ready_for_operator": handoff["ready_for_operator"],
        "joke_state_changed": handoff["state_changed"],
        "joke_changed": handoff["changed"],
        "joke_content_in_handoff": handoff["content_in_handoff"],
        "joke_content_in_metadata": handoff["content_in_metadata"],
        "joke_next_safe_command": handoff["next_safe_command"],
        "joke_next_safe_commands": handoff["next_safe_commands"],
        "joke_next_safe_command_count": handoff["next_safe_command_count"],
        "joke_authorizes_execution": handoff["authorizes_execution"],
        "joke_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "joke_approval_granted": handoff["approval_granted"],
        "joke_boundaries": boundaries,
        "joke_handoff": handoff,
    }


def _local_fun_handoff(*, source: str, result: str, next_safe_command: str) -> dict[str, Any]:
    boundaries = _fun_boundaries(calls_external_service=False)
    handoff = {
        "source": source,
        "status": "ok",
        "handoff_ready": True,
        "ready_for_operator": True,
        "state_changed": False,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "result": result,
        "content_in_handoff": False,
        "content_in_metadata": False,
        "retry_safe": False,
        "next_safe_command": next_safe_command,
        "next_safe_commands": [next_safe_command],
        "next_safe_command_count": 1,
        "boundaries": boundaries,
    }
    return {
        "ready_for_operator": handoff["ready_for_operator"],
        "state_changed": handoff["state_changed"],
        "changed": handoff["changed"],
        "content_in_handoff": handoff["content_in_handoff"],
        "authorizes_execution": handoff["authorizes_execution"],
        "authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "approval_granted": handoff["approval_granted"],
        "fun_handoff_ready": True,
        "fun_ready_for_operator": handoff["ready_for_operator"],
        "fun_state_changed": handoff["state_changed"],
        "fun_changed": handoff["changed"],
        "fun_content_in_handoff": handoff["content_in_handoff"],
        "fun_content_in_metadata": handoff["content_in_metadata"],
        "fun_next_safe_command": handoff["next_safe_command"],
        "fun_next_safe_commands": handoff["next_safe_commands"],
        "fun_next_safe_command_count": handoff["next_safe_command_count"],
        "fun_authorizes_execution": handoff["authorizes_execution"],
        "fun_authorizes_completion_claim": handoff["authorizes_completion_claim"],
        "fun_approval_granted": handoff["approval_granted"],
        "fun_boundaries": boundaries,
        "fun_handoff": handoff,
    }


def _fetch_joke() -> str:
    data = http_get_json(
        "https://icanhazdadjoke.com/",
        headers={"Accept": "application/json", "User-Agent": "JarvisV2 (personal assistant)"},
        timeout=12,
    )
    return (data.get("joke") or "").strip()


def _joke_error_message(exc: Exception) -> str:
    base = friendly_http_error(exc, subject="a joke", service="the joke service")
    return (
        f"{base} Check network access to icanhazdadjoke.com, run `setup check`, "
        "then retry `tell me a joke`."
    )


def _clean_choice_option(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip(" \t\r\n,;:.!?")
    return text[:80]


def _parse_choice_options(args: dict[str, Any]) -> list[str]:
    raw_options = args.get("options")
    options: list[str] = []
    if isinstance(raw_options, list):
        options = [_clean_choice_option(option) for option in raw_options]
    else:
        text = str(args.get("text") or "")
        pieces = re.split(r"\s*(?:,|;|\bor\b|\band\b)\s*", text, flags=re.IGNORECASE)
        options = [_clean_choice_option(piece) for piece in pieces]
    deduped: list[str] = []
    seen: set[str] = set()
    for option in options:
        key = option.lower()
        if not option or key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    return deduped[:12]


def make_fun_tools(config: JarvisConfig):
    def tell_joke(args: dict[str, Any]) -> ToolResult:
        try:
            joke = _fetch_joke()
            if not joke:
                return ToolResult(
                    "tell_joke",
                    True,
                    "I'm fresh out of jokes right now.",
                    _safe_metadata(
                        joke_returned=False,
                        joke_chars=0,
                        **_joke_handoff(status="empty", reason="empty_response"),
                    ),
                )
            return ToolResult(
                "tell_joke",
                True,
                joke,
                _safe_metadata(
                    joke_returned=True,
                    joke_chars=len(joke),
                    **_joke_handoff(status="ok", joke_returned=True, joke_chars=len(joke)),
                ),
            )
        except Exception as e:
            return ToolResult(
                "tell_joke",
                False,
                _joke_error_message(e),
                _safe_metadata(
                    joke_returned=False,
                    joke_chars=0,
                    exception_type=type(e).__name__,
                    error_type=type(e).__name__,
                    **_joke_handoff(
                        status="unavailable",
                        reason="fetch_error",
                        exception_type=type(e).__name__,
                    ),
                ),
            )

    def flip_coin(args: dict[str, Any]) -> ToolResult:
        side = random.choice(["heads", "tails"])
        return ToolResult(
            "flip_coin",
            True,
            f"Coin flip: {side}.",
            _safe_metadata(
                calls_external_service=False,
                result=side,
                choices=["heads", "tails"],
                **_local_fun_handoff(source="flip_coin", result=side, next_safe_command="flip a coin"),
            ),
        )

    def roll_dice(args: dict[str, Any]) -> ToolResult:
        raw_sides = args.get("sides", 6)
        raw_count = args.get("count", 1)
        try:
            sides = int(raw_sides)
        except (TypeError, ValueError):
            sides = 6
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 1
        sides = max(2, min(sides, 100))
        count = max(1, min(count, 12))
        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls)
        label = f"{count}d{sides}" if count > 1 else f"d{sides}"
        output = f"Rolled {label}: {', '.join(str(roll) for roll in rolls)}"
        if count > 1:
            output += f" (total {total})"
        output += "."
        return ToolResult(
            "roll_dice",
            True,
            output,
            _safe_metadata(
                calls_external_service=False,
                result=total,
                rolls=rolls,
                sides=sides,
                count=count,
                **_local_fun_handoff(source="roll_dice", result=str(total), next_safe_command="roll a die"),
            ),
        )

    def random_number(args: dict[str, Any]) -> ToolResult:
        raw_min = args.get("min", 1)
        raw_max = args.get("max", 100)
        try:
            low = int(raw_min)
        except (TypeError, ValueError):
            low = 1
        try:
            high = int(raw_max)
        except (TypeError, ValueError):
            high = 100
        low = max(-1_000_000, min(low, 1_000_000))
        high = max(-1_000_000, min(high, 1_000_000))
        if low > high:
            low, high = high, low
        result = random.randint(low, high)
        return ToolResult(
            "random_number",
            True,
            f"Random number ({low}-{high}): {result}.",
            _safe_metadata(
                calls_external_service=False,
                result=result,
                min=low,
                max=high,
                **_local_fun_handoff(source="random_number", result=str(result), next_safe_command="pick a random number"),
            ),
        )

    def choose_option(args: dict[str, Any]) -> ToolResult:
        options = _parse_choice_options(args)
        if len(options) < 2:
            return ToolResult(
                "choose_option",
                False,
                "Give me at least two options, e.g. 'choose between pizza and sushi'.",
                _safe_metadata(
                    calls_external_service=False,
                    options=options,
                    option_count=len(options),
                    reason="missing_options",
                    **_local_fun_handoff(source="choose_option", result="", next_safe_command="choose between pizza and sushi"),
                ),
            )
        choice = random.choice(options)
        return ToolResult(
            "choose_option",
            True,
            f"I pick: {choice}.",
            _safe_metadata(
                calls_external_service=False,
                result=choice,
                choice=choice,
                options=options,
                option_count=len(options),
                **_local_fun_handoff(source="choose_option", result=choice, next_safe_command="choose between pizza and sushi"),
            ),
        )

    from jarvis_v2.tools.registry import Tool
    return [
        Tool(
            "tell_joke",
            "Tell a (dad) joke. No args.",
            RiskLevel.LOCAL_SAFE,
            tell_joke,
            "personal",
        ),
        Tool(
            "flip_coin",
            "Flip a coin locally. No args.",
            RiskLevel.READ_ONLY,
            flip_coin,
            "personal",
        ),
        Tool(
            "roll_dice",
            "Roll one or more dice locally. Args: count, sides.",
            RiskLevel.READ_ONLY,
            roll_dice,
            "personal",
        ),
        Tool(
            "random_number",
            "Pick a random number locally. Args: min, max.",
            RiskLevel.READ_ONLY,
            random_number,
            "personal",
        ),
        Tool(
            "choose_option",
            "Choose one option locally. Args: options or text.",
            RiskLevel.READ_ONLY,
            choose_option,
            "personal",
        ),
    ]
