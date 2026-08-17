from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class RiskLevel(IntEnum):
    READ_ONLY = 0
    LOCAL_SAFE = 1
    PERSONAL_DATA = 2
    EXTERNAL_SIDE_EFFECT = 3
    HIGH_RISK = 4


@dataclass(frozen=True)
class PlannedAction:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class Plan:
    goal: str
    actions: list[PlannedAction]
    needs_model: bool = False
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalArgumentResolution:
    args: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeResult:
    user_input: str
    plan: Plan
    tool_results: list[ToolResult]
    verified: bool
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)
