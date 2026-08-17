from __future__ import annotations

import re
from typing import Any

from jarvis_v2.agent.types import Plan, ToolResult


MAX_VERIFICATION_FAILURE_CHARS = 260
MAX_VERIFICATION_TOOL_NAME_CHARS = 80
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")


def _safe_text(value: Any) -> str:
    try:
        if value is None:
            return ""
        return str(value)
    except Exception:
        return ""


def _failure_preview(value: Any, *, limit: int = MAX_VERIFICATION_FAILURE_CHARS) -> str:
    text = " ".join(_safe_text(value).split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


class Verifier:
    def verify(self, plan: Plan, results: list[ToolResult]) -> tuple[bool, str]:
        if not plan.actions:
            return False, "Plan had no actions."
        if not results:
            return False, "No actions executed."

        malformed = [result for result in results if result.ok is not True and result.ok is not False]
        if malformed:
            tool_name = _failure_preview(
                malformed[0].tool_name,
                limit=MAX_VERIFICATION_TOOL_NAME_CHARS,
            ) or "<unknown tool>"
            return False, f"Execution receipt from {tool_name!r} had a malformed success flag."

        failed = [result for result in results if result.ok is False]
        if failed:
            joined = "; ".join(
                f"{_failure_preview(result.tool_name, limit=MAX_VERIFICATION_TOOL_NAME_CHARS) or '<unknown tool>'}: {_failure_preview(result.output)}"
                for result in failed
            )
            return False, joined

        if len(results) != len(plan.actions):
            return (
                False,
                f"Execution receipt count mismatch: planned {len(plan.actions)}, received {len(results)}.",
            )

        for index, (action, result) in enumerate(zip(plan.actions, results), start=1):
            expected_tool = action.tool_name
            received_tool = result.tool_name
            if (
                not isinstance(expected_tool, str)
                or not isinstance(received_tool, str)
                or expected_tool != received_tool
            ):
                expected = _failure_preview(expected_tool, limit=MAX_VERIFICATION_TOOL_NAME_CHARS) or "<unknown tool>"
                received = _failure_preview(received_tool, limit=MAX_VERIFICATION_TOOL_NAME_CHARS) or "<unknown tool>"
                return (
                    False,
                    f"Execution receipt {index} did not match planned tool {expected!r} (received {received!r}).",
                )

        return True, "All planned actions completed."
