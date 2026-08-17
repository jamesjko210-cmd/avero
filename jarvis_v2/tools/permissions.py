from __future__ import annotations

from dataclasses import dataclass

from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.tools.registry import Tool


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str
    requires_confirmation: bool = False


class PermissionPolicy:
    def __init__(self, max_auto_risk: RiskLevel = RiskLevel.LOCAL_SAFE):
        if type(max_auto_risk) is not RiskLevel:
            raise ValueError("max_auto_risk must be an exact RiskLevel")
        self.max_auto_risk = max_auto_risk

    def check(self, tool: Tool, approved: bool = False) -> PermissionDecision:
        if type(tool.risk) is not RiskLevel:
            return PermissionDecision(False, "Tool risk classification is invalid.")
        if tool.risk <= self.max_auto_risk:
            return PermissionDecision(True, "Allowed by auto policy.")
        if approved:
            return PermissionDecision(True, "Allowed by explicit approval.")
        return PermissionDecision(
            False,
            f"'{tool.name}' is risk level {tool.risk.name}; explicit approval required.",
            requires_confirmation=True,
        )
