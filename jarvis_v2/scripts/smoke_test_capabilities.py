from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import capabilities as capability_tools
from jarvis_v2.tools.registry import _command_diagnosis_handoff, _continuity_brief_handoff, _metadata_bool
from jarvis_v2.tools.registry import Tool

_LEAK_MARKER = "/\x55sers/hostile/secret-capability-registry"


class _ExplodingToolName:
    @property
    def name(self):
        raise RuntimeError(f"bad tool name {_LEAK_MARKER}")


class _SortingRegistry:
    def __init__(self, tools):
        self._tools = {f"fixture_{index}": tool for index, tool in enumerate(tools)}

    def list(self):
        return sorted(self._tools.values(), key=lambda tool: tool.name)


def _noop_tool(args: dict) -> ToolResult:
    return ToolResult("noop", True, "ok")


def assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def assert_registry_metadata_bool_is_exact() -> None:
    if _metadata_bool(True) is not True:
        raise SystemExit("registry exact metadata bool rejected True")
    if _metadata_bool(False) is not False:
        raise SystemExit("registry exact metadata bool rejected False")
    for value in ("true", "false", "yes", "no", 1, 0, [True], {"ready": True}, None):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"registry exact metadata bool accepted malformed truthy value: {value!r}")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("registry exact metadata bool did not preserve explicit default")


def assert_registry_continuity_handoff_flags_are_exact() -> None:
    handoff = _continuity_brief_handoff(
        source="handoff_brief",
        limit=5,
        next_metadata={
            "execution_health_review_required": "true",
            "doctor_completion_claim_ready": "false",
            "safe_next_actions_handoff": {"next_commands": ["safe next actions"]},
        },
        agi_metadata={},
    )
    if handoff.get("execution_health_review_required") is not False:
        raise SystemExit(f"registry handoff accepted malformed execution health review flag: {handoff}")
    if handoff.get("doctor_completion_claim_ready") is not False:
        raise SystemExit(f"registry handoff accepted malformed completion readiness flag: {handoff}")
    exact = _continuity_brief_handoff(
        source="handoff_brief",
        limit=5,
        next_metadata={
            "execution_health_review_required": True,
            "doctor_completion_claim_ready": True,
        },
        agi_metadata={},
    )
    if exact.get("execution_health_review_required") is not True or exact.get("doctor_completion_claim_ready") is not True:
        raise SystemExit(f"registry handoff rejected exact readiness flags: {exact}")


def assert_registry_command_diagnosis_handoff_flags_are_exact() -> None:
    base = {
        "display_request": "run command python3 --version",
        "route": "hold",
        "recommendation": "RECOVERY_CLOSURE_REQUIRED",
        "next_commands": ["execution recovery packet 1"],
        "plan_goal": "Run shell command.",
        "planned_actions": [],
        "approval_required": False,
        "pending_approvals": 0,
        "approval_queue_forecast": [],
        "forecast_new_approvals": 0,
        "forecast_reused_approval_ids": [],
    }
    malformed = _command_diagnosis_handoff(
        **base,
        recovery_closure={
            "state": "blocked",
            "blocks_auto_execution": "true",
            "required_commands": ["execution recovery packet 1"],
            "next_required_command": "execution recovery packet 1",
        },
        execution_learning_debt={
            "state": "LEARNING_DEBT_AFTER_FAILURE",
            "blocks_completion_claim": "true",
            "required_commands": ["after-action learning packet 1"],
            "next_required_command": "after-action learning packet 1",
        },
    )
    if malformed.get("recovery_closure_blocks_auto_execution") is not False:
        raise SystemExit(f"registry command diagnosis handoff accepted malformed recovery blocker: {malformed}")
    if malformed.get("execution_learning_blocks_completion_claim") is not False:
        raise SystemExit(f"registry command diagnosis handoff accepted malformed learning blocker: {malformed}")
    exact = _command_diagnosis_handoff(
        **base,
        recovery_closure={
            "state": "blocked",
            "blocks_auto_execution": True,
            "required_commands": ["execution recovery packet 1"],
            "next_required_command": "execution recovery packet 1",
        },
        execution_learning_debt={
            "state": "LEARNING_DEBT_AFTER_FAILURE",
            "blocks_completion_claim": True,
            "required_commands": ["after-action learning packet 1"],
            "next_required_command": "after-action learning packet 1",
        },
    )
    if exact.get("recovery_closure_blocks_auto_execution") is not True:
        raise SystemExit(f"registry command diagnosis handoff rejected exact recovery blocker: {exact}")
    if exact.get("execution_learning_blocks_completion_claim") is not True:
        raise SystemExit(f"registry command diagnosis handoff rejected exact learning blocker: {exact}")


def assert_command_diagnosis_routes_approval_held_runs_to_review() -> None:
    with TemporaryDirectory(prefix="jarvis-command-diagnosis-held-") as temp:
        runtime = make_temp_runtime(Path(temp))
        held_run_id = runtime.store.log_tool_run(
            runtime.session_id,
            "send_telegram",
            "HIGH_RISK",
            False,
            False,
            "raw held marker should stay private; queued as approval #7",
            approval_id=7,
            metadata={"failure_kind": "approval-gate", "requires_confirmation": True},
        )
        result = runtime.registry.get("command_diagnosis").handler({"request": "run command echo hello"})
        if not result.ok:
            raise SystemExit(f"command diagnosis should render approval-held recovery review: {result.output}")
        if "Approval-held execution review:" not in result.output:
            raise SystemExit(f"command diagnosis missed approval-held review section: {result.output}")
        if "- real failed/blocked action runs: 0" not in result.output:
            raise SystemExit(f"command diagnosis should not count approval holds as failures: {result.output}")
        if "- approval-held action runs: 1" not in result.output:
            raise SystemExit(f"command diagnosis missed approval-held count: {result.output}")
        for command in ("approval readiness 7", "approval packet 7", "approval chain proof 7"):
            if command not in result.output:
                raise SystemExit(f"command diagnosis missed approval command {command!r}: {result.output}")
        if f"execution recovery packet {held_run_id}" in result.output:
            raise SystemExit(f"command diagnosis should not recover approval-held run: {result.output}")
        if "raw held marker should stay private" in result.output:
            raise SystemExit(f"command diagnosis leaked raw held output: {result.output}")

        metadata = result.metadata
        if metadata.get("route") != "recovery_closure":
            raise SystemExit(f"command diagnosis should hold risky command behind review: {metadata}")
        if metadata.get("recovery_closure_state") != "approval_held_review_required":
            raise SystemExit(f"command diagnosis metadata missed approval-held state: {metadata}")
        if metadata.get("recovery_closure_failed_or_blocked_action_runs") != 0:
            raise SystemExit(f"command diagnosis metadata counted approval hold as failure: {metadata}")
        if metadata.get("recovery_closure_approval_held_action_runs") != 1:
            raise SystemExit(f"command diagnosis metadata missed approval-held count: {metadata}")
        if metadata.get("recovery_closure_approval_held_target_run_id") != held_run_id:
            raise SystemExit(f"command diagnosis metadata missed approval-held run id: {metadata}")
        if f"execution recovery packet {held_run_id}" in metadata.get("recovery_closure_required_commands", []):
            raise SystemExit(f"command diagnosis metadata should not request held-run recovery: {metadata}")
        for command in ("approval readiness 7", "approval packet 7", "approval chain proof 7"):
            if command not in metadata.get("recovery_closure_required_commands", []):
                raise SystemExit(f"command diagnosis metadata missed approval command {command!r}: {metadata}")


def assert_safety_aliases_route_to_read_only_surfaces() -> None:
    planner = RuleBasedPlanner()
    expected_tools = {
        "privacy please": "privacy_report",
        "privacy report please": "privacy_report",
        "show privacy": "privacy_report",
        "show latest privacy": "privacy_report",
        "safety please": "safety_status",
        "safety status please": "safety_status",
        "show safety": "safety_status",
        "show latest safety": "safety_status",
        "risk please": "risk_matrix",
        "risk matrix please": "risk_matrix",
        "show risk matrix": "risk_matrix",
        "permissions please": "risk_matrix",
        "permission status please": "risk_matrix",
        "jarvis permissions": "risk_matrix",
        "show my permissions": "risk_matrix",
        "show jarvis permissions": "risk_matrix",
        "what are my permissions": "risk_matrix",
        "what permissions do I have": "risk_matrix",
        "what permissions do you have": "risk_matrix",
        "what permissions does Jarvis have": "risk_matrix",
    }
    for phrase, expected_tool in expected_tools.items():
        plan = planner.plan(phrase)
        action_tools = [action.tool_name for action in plan.actions]
        if action_tools != [expected_tool]:
            raise SystemExit(f"safety alias did not route to {expected_tool}: {phrase!r} -> {action_tools!r}")


def assert_capability_aliases_route_with_clean_focus() -> None:
    planner = RuleBasedPlanner()
    expected = {
        "show latest capabilities": ("capability_map", {"focus": ""}),
        "show latest capability map": ("capability_map", {"focus": ""}),
        "abilities please": ("capability_map", {"focus": ""}),
        "show abilities please": ("capability_map", {"focus": ""}),
        "ability list please": ("capability_map", {"focus": ""}),
        "show ability list": ("capability_map", {"focus": ""}),
        "jarvis abilities please": ("capability_map", {"focus": ""}),
        "what are your abilities": ("capability_map", {"focus": ""}),
        "what are your abilities please": ("capability_map", {"focus": ""}),
        "what are jarvis abilities": ("capability_map", {"focus": ""}),
        "list jarvis abilities": ("capability_map", {"focus": ""}),
        "what can you help me with": ("capability_map", {"focus": ""}),
        "what can jarvis help me with please": ("capability_map", {"focus": ""}),
        "how can you help me": ("capability_map", {"focus": ""}),
        "what can you do for me": ("capability_map", {"focus": ""}),
        "what can you do with approvals please": ("capability_map", {"focus": "approvals"}),
        "what can you do about approvals": ("capability_map", {"focus": "approvals"}),
        "what can you do for calendar": ("capability_map", {"focus": "calendar"}),
        "what can you do with conversation": ("capability_map", {"focus": "conversation"}),
        "what can you do with messages": ("capability_map", {"focus": "messages"}),
        "what can you do with voice": ("capability_map", {"focus": "voice"}),
        "what can you do with cockpit": ("capability_map", {"focus": "cockpit"}),
        "what can you do with status": ("capability_map", {"focus": "status"}),
        "what can you do with control plane": ("capability_map", {"focus": "control plane"}),
        "what can you do with orchestration": ("capability_map", {"focus": "orchestration"}),
        "what can you do with agent status": ("capability_map", {"focus": "agent status"}),
        "what can you do with research": ("capability_map", {"focus": "research"}),
        "what can you do with web lookup": ("capability_map", {"focus": "web lookup"}),
        "what can you do with web search": ("capability_map", {"focus": "web search"}),
        "help with research": ("capability_map", {"focus": "research"}),
        "help with web": ("capability_map", {"focus": "web"}),
        "help with messages": ("capability_map", {"focus": "messages"}),
        "help with voice": ("capability_map", {"focus": "voice"}),
        "help with cockpit": ("capability_map", {"focus": "cockpit"}),
        "what can you do with telegram": ("capability_map", {"focus": "telegram"}),
        "what can you do with kakao": ("capability_map", {"focus": "kakao"}),
        "what can you help with reminders": ("capability_map", {"focus": "reminders"}),
        "help me use reminders please": ("capability_map", {"focus": "reminders"}),
        "how can you help with reminders": ("capability_map", {"focus": "reminders"}),
        "how can jarvis help with calendar please": ("capability_map", {"focus": "calendar"}),
        "what can you do around email": ("capability_map", {"focus": "email"}),
        "what can you do regarding email": ("capability_map", {"focus": "email"}),
        "what are your reminder abilities": ("capability_map", {"focus": "reminder"}),
        "what tools can you use for reminders": ("capability_map", {"focus": "reminders"}),
        "what tools do you have for reminders": ("capability_map", {"focus": "reminders"}),
        "what tools can jarvis use for calendar": ("capability_map", {"focus": "calendar"}),
        "what tools help with reminders": ("capability_map", {"focus": "reminders"}),
        "what email tools do you have": ("capability_map", {"focus": "email"}),
        "what tool handles reminders": ("capability_map", {"focus": "reminders"}),
        "which tool handles reminders": ("capability_map", {"focus": "reminders"}),
        "which tools do reminders": ("capability_map", {"focus": "reminders"}),
        "tools for reminders": ("capability_map", {"focus": "reminders"}),
        "reminder tools please": ("capability_map", {"focus": "reminder"}),
        "show reminder tools please": ("capability_map", {"focus": "reminder"}),
        "show me tools for email": ("capability_map", {"focus": "email"}),
        "show tools for reminders please": ("capability_map", {"focus": "reminders"}),
        "capability map latest please": ("capability_map", {"focus": ""}),
    }
    for phrase, expected_action in expected.items():
        plan = planner.plan(phrase)
        actions = [(action.tool_name, action.args) for action in plan.actions]
        if actions != [expected_action]:
            raise SystemExit(f"Capability alias misplanned {phrase!r}: {actions!r}")
    expected_tool_search = {
        "find tools for reminders": {"query": "reminders"},
        "search tools for reminders": {"query": "reminders"},
        "tool search reminders": {"query": "reminders"},
        "tool search: reminders": {"query": "reminders"},
    }
    for phrase, expected_args in expected_tool_search.items():
        plan = planner.plan(phrase)
        actions = [(action.tool_name, action.args) for action in plan.actions]
        if actions != [("tool_search", expected_args)]:
            raise SystemExit(f"Tool-search alias misplanned {phrase!r}: {actions!r}")
    expected_tool_detail = {
        "what does run_shell_command do": {"name": "run_shell_command"},
        "what does the run_shell_command tool do": {"name": "run_shell_command"},
        "what is run_shell_command": {"name": "run_shell_command"},
        "does run_shell_command need approval": {"name": "run_shell_command"},
        "does run_shell_command require approval": {"name": "run_shell_command"},
        "is run_shell_command safe": {"name": "run_shell_command"},
        "is send_email high risk": {"name": "send_email"},
        "is send_email risky": {"name": "send_email"},
        "risk of send_email": {"name": "send_email"},
        "approval required for send_email": {"name": "send_email"},
        "approval for send_email": {"name": "send_email"},
        "show risk for send_email": {"name": "send_email"},
        "tell me about run_shell_command tool": {"name": "run_shell_command"},
        "explain run_shell_command tool": {"name": "run_shell_command"},
        "tool run_shell_command": {"name": "run_shell_command"},
        "run_shell_command tool": {"name": "run_shell_command"},
        "what does run shell command tool do": {"name": "run_shell_command"},
        "what does run shell command do": {"name": "run_shell_command"},
        "approval required for send email tool": {"name": "send_email"},
    }
    for phrase, expected_args in expected_tool_detail.items():
        plan = planner.plan(phrase)
        actions = [(action.tool_name, action.args) for action in plan.actions]
        if actions != [("tool_detail", expected_args)]:
            raise SystemExit(f"Tool-detail alias misplanned {phrase!r}: {actions!r}")
    preserved_routes = {
        "what does api stand for": ("define", {"word": "api"}),
        "what is pending approval": ("list_pending_approvals", {}),
        "what does weather mean": ("define", {"word": "weather"}),
        "what is calendar": ("wiki_summary", {"query": "calendar"}),
    }
    for phrase, expected_action in preserved_routes.items():
        plan = planner.plan(phrase)
        actions = [(action.tool_name, action.args) for action in plan.actions]
        if actions != [expected_action]:
            raise SystemExit(f"Tool-detail alias hijacked adjacent route {phrase!r}: {actions!r}")


def assert_capabilities_doc_matches_live_proof_truth() -> None:
    doc = Path("CAPABILITIES.md").read_text(encoding="utf-8")
    normalized_doc = " ".join(doc.split())
    forbidden_fragments = [
        "Korean and English both work end-to-end",
        "Hangul are preserved through the approval queue and delivery",
    ]
    for fragment in forbidden_fragments:
        if fragment in normalized_doc:
            raise SystemExit(f"CAPABILITIES.md overclaims live delivery: {fragment!r}")
    for expected in [
        "Telegram, Instagram, iMessage, and KakaoTalk sends have recipient-confirmed evidence.",
        "KakaoTalk's GUI execution remained outcome-unknown until the operator verified the exact chat and the recipient confirmed arrival.",
        "Every new send still requires a fresh approval.",
        "Calls are explicitly disabled for the August 15 functional preview; any later re-enable and live proof requires a separate, operator-present approval gate.",
        "**Messaging (KR/EN)** — Telegram, KakaoTalk, Instagram, iMessage",
        "Updated to mirror the live cockpit on 2026-08-07",
    ]:
        if expected not in normalized_doc:
            raise SystemExit(f"CAPABILITIES.md missed current live-proof truth: {expected!r}")


def assert_readme_matches_live_proof_freeze() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    stale_claims = [
        "Opt-in end-to-end health check",
        "End-to-end feature check",
        "Two opt-in diagnostics confirm what actually works on this Mac.",
    ]
    for stale_claim in stale_claims:
        if stale_claim in normalized_readme:
            raise SystemExit(
                f"README overclaims live health checks as end-to-end during the live-proof freeze: {stale_claim!r}"
            )
    for expected in [
        "Two opt-in diagnostics inspect service reachability and feature readiness on this Mac.",
        "Feature readiness check",
        "voice transcription and warmup",
        "shared dashboard/local/Telegram voice",
        "personal integration proof rows",
        "scheduled morning-brief freshness and 7-day job proof",
        "acceptance coverage/gaps/next-proof rows",
        "aggregate-smoke proof",
        "operator-workflow evaluation proof",
        "channel health",
        "guardrail, proof-ledger, completion, and learning control-plane rows",
        "phone approvals",
        "phone control",
        "Telegram, Instagram, iMessage, and KakaoTalk have recipient-confirmed V3 send proofs; calls are disabled for the August 15 preview, and any later re-enable and live proof requires a separate, operator-present approval gate. KakaoTalk required operator verification after an outcome-unknown GUI attempt.",
        "Telegram, Instagram, iMessage, and KakaoTalk have separate recipient-confirmed V3 evidence, while calls are disabled for the August 15 preview; any later re-enable and live proof requires a separate, operator-present approval gate. KakaoTalk required operator verification after an outcome-unknown GUI attempt.",
        "This is not live proof of sends, calls, approvals, microphone capture, or reboot survival.",
        "live-proof caveats",
        "Telegram, Instagram, iMessage, and KakaoTalk have recipient-confirmed V3 evidence; KakaoTalk required operator chat verification after an outcome-unknown GUI attempt. Calls are disabled for the August 15 preview; any later re-enable and live proof requires a separate, operator-present approval gate. Examples remain approval-gated capability routes.",
    ]:
        if expected not in normalized_readme:
            raise SystemExit(f"README missed live-proof caveat: {expected!r}")


def assert_capability_inspection_counts_hidden_registry_tools() -> None:
    good_tool = Tool(
        "good_fixture",
        "Good approval fixture for registry recovery.",
        RiskLevel.READ_ONLY,
        _noop_tool,
        "core",
    )
    registry = _SortingRegistry([_ExplodingToolName(), good_tool])
    cases = [
        (
            "capability_map",
            capability_tools.make_capability_tools(registry.list),
            {},
            "good_fixture",
        ),
        (
            "tool_search",
            capability_tools.make_tool_search_tool(registry.list),
            {"query": "good"},
            "good_fixture",
        ),
        (
            "tool_detail",
            capability_tools.make_tool_detail_tool(registry.list),
            {"name": "good_fixture"},
            "Jarvis tool detail: good_fixture",
        ),
        (
            "risk_matrix",
            capability_tools.make_risk_matrix_tool(registry.list),
            {},
            "READ_ONLY",
        ),
    ]
    for label, handler, args, expected_text in cases:
        result = handler(args)
        if not result.ok:
            raise SystemExit(f"{label} should survive one malformed registry tool: {result}")
        if expected_text not in result.output:
            raise SystemExit(f"{label} lost the readable fixture while hiding malformed registry tools: {result.output}")
        if result.metadata.get("unreadable_registry_tools") != 1:
            raise SystemExit(f"{label} missed hidden registry count: {result.metadata}")
        handoff_key = f"{label}_handoff"
        if result.metadata.get(handoff_key, {}).get("unreadable_registry_tools") != 1:
            raise SystemExit(f"{label} handoff missed hidden registry count: {result.metadata}")
        blob = result.output + str(result.metadata)
        if _LEAK_MARKER in blob:
            raise SystemExit(f"{label} leaked malformed registry path marker: {blob}")


def assert_list_tools_counts_hidden_registry_tools() -> None:
    with TemporaryDirectory(prefix="jarvis-list-tools-hidden-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.registry._tools["bad_list_fixture"] = _ExplodingToolName()
        for label, args, expected in [
            ("list_tools all", {}, "list_tools"),
            ("list_tools core", {"toolset": "core"}, "capability_map"),
        ]:
            result = runtime.registry.get("list_tools").handler(args)
            if not result.ok:
                raise SystemExit(f"{label} should survive one malformed registry tool: {result}")
            if expected not in result.output:
                raise SystemExit(f"{label} lost readable registry tools while hiding a malformed one: {result.output[:1000]}")
            if result.metadata.get("unreadable_registry_tools") != 1:
                raise SystemExit(f"{label} missed hidden registry count: {result.metadata}")
            for flag in ("calls_model", "executes_tools", "queues_approval", "requires_approval", "external_side_effect", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
                if result.metadata.get(flag):
                    raise SystemExit(f"{label} should remain read-only and non-authorizing: {result.metadata}")
            if "1 unreadable registry tool(s) hidden for safety" not in result.output:
                raise SystemExit(f"{label} missed visible hidden-registry note: {result.output[-500:]}")
            blob = result.output + str(result.metadata)
            if _LEAK_MARKER in blob:
                raise SystemExit(f"{label} leaked malformed registry path marker: {blob}")
            assert_no_local_path(blob, label)


def assert_capability_map_handoff(metadata: dict, label: str, *, status: str = "ok") -> None:
    if metadata.get("capability_map_handoff_ready") is not True:
        raise SystemExit(f"{label} missed capability map handoff readiness: {metadata}")
    handoff = metadata.get("capability_map_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed structured capability map handoff: {metadata}")
    for key, expected in {
        "source": "capability_map",
        "status": status,
        "focus": metadata.get("focus"),
        "toolsets": metadata.get("toolsets"),
        "tools": metadata.get("tools"),
        "content_in_handoff": False,
        "state_changed": False,
        "ready_for_operator": True,
    }.items():
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} handoff {key} mismatch: {handoff}")
    if handoff.get("changed") != []:
        raise SystemExit(f"{label} handoff should report no changed state: {handoff}")
    next_commands = handoff.get("next_commands")
    if not isinstance(next_commands, list) or not {"capability map", "tool search: approvals", "tool detail: run_shell_command", "risk matrix"}.issubset(set(next_commands)):
        raise SystemExit(f"{label} handoff missed recovery commands: {handoff}")
    boundary = handoff.get("boundary")
    if not isinstance(boundary, dict) or boundary.get("read_only") is not True:
        raise SystemExit(f"{label} handoff missed read-only boundary: {handoff}")
    for key in [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_private_data",
        "writes_files",
        "writes_notes",
        "writes_memory",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "controls_computer",
        "external_side_effect",
        "requires_approval",
    ]:
        if boundary.get(key) is not False:
            raise SystemExit(f"{label} handoff unsafe boundary {key}: {handoff}")
    section_rows = handoff.get("section_rows")
    if status == "ok":
        if not isinstance(section_rows, list) or len(section_rows) != metadata.get("toolsets"):
            raise SystemExit(f"{label} handoff section row count mismatch: {handoff}")
        if sum(row.get("tool_count", 0) for row in section_rows) != metadata.get("tools"):
            raise SystemExit(f"{label} handoff tool count mismatch: {handoff}")
        for row in section_rows:
            if not {"toolset", "label", "tool_count", "risk_counts", "example_tools"}.issubset(row):
                raise SystemExit(f"{label} handoff row missed fields: {row}")
    elif section_rows != []:
        raise SystemExit(f"{label} empty handoff should not carry section rows: {handoff}")
    assert_no_local_path(handoff, f"{label} capability map handoff")


def assert_runtime_routes_inventory_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-capability-count-aliases-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            "tool count",
            "tools count",
            "count tools",
            "how many tools",
            "how many tools do you have",
            "capability count",
            "count capabilities",
            "how many capabilities",
            "ability count",
            "count abilities",
            "integrations count",
            "how many integrations",
        ]
        for text in cases:
            result = runtime.handle(text)
            tool_names = [tool_result.tool_name for tool_result in result.tool_results]
            if tool_names != ["capability_map"]:
                raise SystemExit(f"inventory count alias {text!r} routed to {tool_names}, expected ['capability_map']")
            if not result.verified:
                raise SystemExit(f"inventory count alias {text!r} should be verified: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("focus") not in {"", None}:
                raise SystemExit(f"inventory count alias {text!r} should show the full map, not focus {metadata.get('focus')!r}: {metadata}")
            if "Focus: count" in result.response or "No Jarvis capabilities matched" in result.response:
                raise SystemExit(f"inventory count alias {text!r} should render the full capability map: {result.response[:500]}")
            if metadata.get("tools", 0) <= 0 or metadata.get("toolsets", 0) <= 0:
                raise SystemExit(f"inventory count alias {text!r} missed tool inventory counts: {metadata}")
            assert_capability_map_handoff(metadata, f"inventory count alias {text!r}")
            for flag in (
                "queues_approval",
                "requires_approval",
                "authorizes_execution",
                "approval_granted",
                "writes_memory",
                "writes_notes",
                "writes_files",
                "writes_database",
                "external_side_effect",
                "controls_computer",
            ):
                if metadata.get(flag) is True:
                    raise SystemExit(f"inventory count alias {text!r} should remain read-only, but {flag}=True: {metadata}")


def assert_runtime_routes_risk_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-risk-count-aliases-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            "risk count",
            "risk counts",
            "tool risk count",
            "tool risk counts",
            "approval gated count",
            "approval-gated count",
            "approval gated tools count",
            "how many approval gated tools",
            "how many approval-gated tools",
            "high risk count",
            "high-risk count",
            "how many high risk tools",
            "how many high-risk tools",
            "read only tool count",
            "read-only tool count",
            "how many read only tools",
            "how many read-only tools",
            "local safe tool count",
            "personal data tool count",
        ]
        for text in cases:
            result = runtime.handle(text)
            tool_names = [tool_result.tool_name for tool_result in result.tool_results]
            if tool_names != ["risk_matrix"]:
                raise SystemExit(f"risk count alias {text!r} routed to {tool_names}, expected ['risk_matrix']")
            if not result.verified:
                raise SystemExit(f"risk count alias {text!r} should be verified: {result.response}")
            if "Jarvis risk matrix" not in result.response or "Risk totals:" not in result.response:
                raise SystemExit(f"risk count alias {text!r} missed risk matrix output: {result.response[:500]}")
            if "No Jarvis capabilities matched" in result.response or "No Jarvis tool named" in result.response:
                raise SystemExit(f"risk count alias {text!r} drifted to an inspection miss: {result.response[:500]}")
            metadata = result.tool_results[0].metadata
            risk_totals = metadata.get("risk_totals")
            if not isinstance(risk_totals, dict):
                raise SystemExit(f"risk count alias {text!r} missed structured risk totals: {metadata}")
            for risk in ("READ_ONLY", "LOCAL_SAFE", "PERSONAL_DATA", "EXTERNAL_SIDE_EFFECT", "HIGH_RISK"):
                if risk_totals.get(risk, 0) <= 0:
                    raise SystemExit(f"risk count alias {text!r} missed {risk} count: {metadata}")
            if metadata.get("tools", 0) != sum(risk_totals.values()):
                raise SystemExit(f"risk count alias {text!r} tool count diverged from risk totals: {metadata}")
            if metadata.get("approval_gated") != (
                risk_totals.get("PERSONAL_DATA", 0)
                + risk_totals.get("EXTERNAL_SIDE_EFFECT", 0)
                + risk_totals.get("HIGH_RISK", 0)
            ):
                raise SystemExit(f"risk count alias {text!r} approval-gated count diverged: {metadata}")
            assert_inspection_handoff(
                metadata,
                f"risk count alias {text!r}",
                "risk_matrix",
                status="ok",
                required_commands={"tool search: approval", "tool detail: run_shell_command", "capability map"},
            )
            for flag in (
                "queues_approval",
                "requires_approval",
                "authorizes_execution",
                "approval_granted",
                "writes_memory",
                "writes_notes",
                "writes_files",
                "external_side_effect",
                "controls_computer",
            ):
                if metadata.get(flag) is True:
                    raise SystemExit(f"risk count alias {text!r} should remain read-only, but {flag}=True: {metadata}")


def assert_runtime_routes_korean_inventory_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-korean-count-aliases-") as temp:
        runtime = make_temp_runtime(Path(temp))
        capability_cases = [
            "도구 몇 개",
            "도구 개수",
            "자비스 도구 몇 개",
            "기능 몇 개",
            "능력 몇 개",
            "통합 몇 개",
        ]
        for text in capability_cases:
            result = runtime.handle(text)
            tool_names = [tool_result.tool_name for tool_result in result.tool_results]
            if tool_names != ["capability_map"]:
                raise SystemExit(f"Korean inventory count alias {text!r} routed to {tool_names}, expected ['capability_map']")
            if not result.verified:
                raise SystemExit(f"Korean inventory count alias {text!r} should be verified: {result.response}")
            metadata = result.tool_results[0].metadata
            if metadata.get("focus") not in {"", None}:
                raise SystemExit(f"Korean inventory count alias {text!r} should show the full map, not focus {metadata.get('focus')!r}: {metadata}")
            if "Focus: count" in result.response or "No Jarvis capabilities matched" in result.response:
                raise SystemExit(f"Korean inventory count alias {text!r} should render the full capability map: {result.response[:500]}")
            if metadata.get("tools", 0) <= 0 or metadata.get("toolsets", 0) <= 0:
                raise SystemExit(f"Korean inventory count alias {text!r} missed tool inventory counts: {metadata}")
            assert_capability_map_handoff(metadata, f"Korean inventory count alias {text!r}")

        risk_cases = [
            "리스크 몇 개",
            "위험 몇 개",
            "위험 도구 몇 개",
            "고위험 도구 몇 개",
            "고위험 몇 개",
            "승인 필요한 도구 몇 개",
            "승인 필요 몇 개",
            "읽기 전용 도구 몇 개",
            "로컬 안전 도구 몇 개",
            "개인정보 도구 몇 개",
        ]
        for text in risk_cases:
            result = runtime.handle(text)
            tool_names = [tool_result.tool_name for tool_result in result.tool_results]
            if tool_names != ["risk_matrix"]:
                raise SystemExit(f"Korean risk count alias {text!r} routed to {tool_names}, expected ['risk_matrix']")
            if not result.verified:
                raise SystemExit(f"Korean risk count alias {text!r} should be verified: {result.response}")
            if "Jarvis risk matrix" not in result.response or "Risk totals:" not in result.response:
                raise SystemExit(f"Korean risk count alias {text!r} missed risk matrix output: {result.response[:500]}")
            metadata = result.tool_results[0].metadata
            risk_totals = metadata.get("risk_totals")
            if not isinstance(risk_totals, dict):
                raise SystemExit(f"Korean risk count alias {text!r} missed structured risk totals: {metadata}")
            if metadata.get("tools", 0) != sum(risk_totals.values()):
                raise SystemExit(f"Korean risk count alias {text!r} tool count diverged from risk totals: {metadata}")
            if metadata.get("approval_gated") != (
                risk_totals.get("PERSONAL_DATA", 0)
                + risk_totals.get("EXTERNAL_SIDE_EFFECT", 0)
                + risk_totals.get("HIGH_RISK", 0)
            ):
                raise SystemExit(f"Korean risk count alias {text!r} approval-gated count diverged: {metadata}")
            assert_inspection_handoff(
                metadata,
                f"Korean risk count alias {text!r}",
                "risk_matrix",
                status="ok",
                required_commands={"tool search: approval", "tool detail: run_shell_command", "capability map"},
            )

        for text in capability_cases + risk_cases:
            result = runtime.handle(text)
            metadata = result.tool_results[0].metadata
            for flag in (
                "queues_approval",
                "requires_approval",
                "authorizes_execution",
                "approval_granted",
                "writes_memory",
                "writes_notes",
                "writes_files",
                "external_side_effect",
                "controls_computer",
            ):
                if metadata.get(flag) is True:
                    raise SystemExit(f"Korean count alias {text!r} should remain read-only, but {flag}=True: {metadata}")


def assert_inspection_handoff(metadata: dict, label: str, key: str, *, status: str, required_commands: set[str] | None = None) -> None:
    ready_key = f"{key}_handoff_ready"
    handoff_key = f"{key}_handoff"
    if metadata.get(ready_key) is not True:
        raise SystemExit(f"{label} missed {key} handoff readiness: {metadata}")
    handoff = metadata.get(handoff_key)
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed structured {key} handoff: {metadata}")
    for handoff_field, expected in {
        "source": key,
        "status": status,
        "content_in_handoff": False,
        "state_changed": False,
        "ready_for_operator": True,
    }.items():
        if handoff.get(handoff_field) != expected:
            raise SystemExit(f"{label} {key} handoff {handoff_field} mismatch: {handoff}")
    if handoff.get("changed") != []:
        raise SystemExit(f"{label} {key} handoff should report no changed state: {handoff}")
    next_commands = handoff.get("next_commands")
    if not isinstance(next_commands, list) or not next_commands:
        raise SystemExit(f"{label} {key} handoff missed next commands: {handoff}")
    if required_commands and not required_commands.issubset(set(next_commands)):
        raise SystemExit(f"{label} {key} handoff missed required commands {required_commands}: {handoff}")
    boundary = handoff.get("boundary")
    if not isinstance(boundary, dict) or boundary.get("read_only") is not True:
        raise SystemExit(f"{label} {key} handoff missed read-only boundary: {handoff}")
    for boundary_key in [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "reads_private_data",
        "writes_files",
        "writes_notes",
        "writes_memory",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "approves_request",
        "dismisses_request",
        "controls_computer",
        "external_side_effect",
        "requires_approval",
    ]:
        if boundary.get(boundary_key) is not False:
            raise SystemExit(f"{label} {key} handoff unsafe boundary {boundary_key}: {handoff}")
    assert_no_local_path(handoff, f"{label} {key} handoff")


def main() -> None:
    assert_registry_metadata_bool_is_exact()
    assert_registry_continuity_handoff_flags_are_exact()
    assert_registry_command_diagnosis_handoff_flags_are_exact()
    assert_command_diagnosis_routes_approval_held_runs_to_review()
    assert_safety_aliases_route_to_read_only_surfaces()
    assert_capability_aliases_route_with_clean_focus()
    assert_capabilities_doc_matches_live_proof_truth()
    assert_readme_matches_live_proof_freeze()
    assert_capability_inspection_counts_hidden_registry_tools()
    assert_list_tools_counts_hidden_registry_tools()
    assert_runtime_routes_inventory_count_aliases()
    assert_runtime_routes_risk_count_aliases()
    assert_runtime_routes_korean_inventory_count_aliases()
    readme = Path("README.md").read_text(encoding="utf-8")
    if (
        "schedule default assistant automations with `schedule assistant basics`, resume a paused State Snapshot with `resume job State Snapshot`, or resume paused Memory Trees compaction with `resume job Conversation Compaction`"
        not in readme
    ):
        raise SystemExit("README automation guidance should mention resuming paused State Snapshot and Conversation Compaction jobs.")

    with TemporaryDirectory(prefix="jarvis-capabilities-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            (
                "capability map",
                [
                    "Jarvis capability map",
                    "Everyday abilities:",
                    "Info: weather, news, air quality, sunrise/sunset, history, holidays, Wikipedia, definitions. Example: `weather in Tokyo`",
                    "Productivity: calendar, availability, email, tasks, reminders, timers. Example: `am I free tomorrow afternoon?`",
                    "Messages & calls: send messages on KakaoTalk, Instagram, iMessage, or Telegram with approval; call routes exist but are disabled for the August 15 preview. Example: `send 가상연락처일 a kakao saying on my way`",
                    "Live proof: Telegram, Instagram, iMessage, and KakaoTalk sends have recipient-confirmed V3 evidence.",
                    "KakaoTalk required operator chat verification plus recipient confirmation after an outcome-unknown GUI attempt.",
                    "Every new send remains approval-gated",
                    "Calls are disabled for the August 15 functional preview; any later re-enable and live proof requires a separate, operator-present approval gate",
                    "Utilities: translate, currency, unit conversion, time zones/time differences, relative dates, countdowns, calculate/BMI, spelling, password/UUID generation, text transforms/counts. Example: `convert 100 USD to KRW`",
                    "Markets: crypto and stock price lookups. Example: `stock price of AAPL`",
                    "Research & Web: web lookup, web search, page fetch/summarize, and research synthesis. Example: `research the best setup for local AI notes`",
                    "Writing: paste_text, human_write, compose_and_write. Example: `write a paragraph about Jarvis and type it`",
                    "Fun: jokes and lightweight playful prompts. Example: `tell me a quick joke`",
                    "Voice & images: talk by voice (the always-on-top Jarvis HUD window with WhisperFlow dictation, push-to-talk, or a Telegram voice note), read text from a photo/document (on-device OCR), look up a contact's number. Example: `what does this receipt say`",
                    "Control plane: capability cockpit, earned-trust checklist, channel health, Jarvis status, readiness report, safety/risk review, completion gate, and internal worker readiness. Example: `cockpit summary`",
                    "Safety model",
                    "Capability areas",
                    "Good starting commands",
                    "tool_search",
                    "tool_detail",
                    "risk_matrix",
                    "harness_build_slice",
                    "status dashboard",
                    "integration_preflight_contract",
                    "recent saved notes",
                    "computer control",
                    "requires approval",
                ],
            ),
            (
                "capability list please",
                [
                    "Jarvis capability map",
                    "Everyday abilities:",
                    "Safety model",
                    "Good starting commands",
                ],
            ),
            (
                "what can Jarvis do continuity",
                [
                    "Focus: continuity",
                    "continuity",
                    "harness_build_slice",
                    "harness build slice: personal connector readiness",
                    "continuation_packet",
                    "build_target_packet",
                ],
            ),
            (
                "what can Jarvis do memory",
                [
                    "Focus: memory",
                    "memory curation",
                    "personal context status",
                    "knowledge promotion packet <id>",
                    "does not run tools",
                ],
            ),
            (
                "what can Jarvis do personal",
                [
                    "Focus: personal",
                    "Everyday abilities:",
                    "Info: weather, news, air quality, sunrise/sunset, history, holidays, Wikipedia, definitions",
                    "Productivity: calendar, availability, email, tasks, reminders, timers",
                    "Utilities: translate, currency, unit conversion, time zones/time differences, relative dates, countdowns, calculate/BMI, spelling, password/UUID generation, text transforms/counts",
                    "Markets: crypto and stock price lookups",
                    "Research & Web: web lookup, web search, page fetch/summarize, and research synthesis",
                    "Writing: paste_text, human_write, compose_and_write",
                    "Fun: jokes and lightweight playful prompts",
                    "personal integrations",
                    "legacy_connector_migration_audit",
                    "integration_proof_bundle",
                    "integration_implementation_review",
                    "integration execution matrix",
                    "integration adapter manifest",
                    "integration adapter probe: email -> search mailbox metadata",
                    "integration adapter acceptance: email",
                    "integration route lock",
                    "legacy connector migration audit",
                    "integration proof bundle: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed",
                    "integration implementation review: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed",
                    "integration route lock: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only; rollback no connector call; tests adapter acceptance passed; audit adapter acceptance receipt; acceptance gate passed; status status api smoke passed",
                    "integration dry run contract: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only; verification fake rows only",
                    "integration_preflight_contract",
                    "integration preflight contract",
                    "integration implementation spec",
                    "what's on my calendar today",
                    "what is on my schedule today",
                    "what am I doing today",
                    "schedule today",
                    "free today",
                    "schedule PTO all day tomorrow",
                    "move event evt123 to June 20 at 2pm",
                    "calendar edits need an exact event id",
                    "translate hello to Korean",
                    "how do you say thank you in Korean",
                    "convert 100 USD to KRW",
                    "weather in Tokyo",
                    "news about Korea",
                    "what is bitcoin worth",
                    "stock price of AAPL",
                ],
            ),
            (
                "what can Jarvis do info",
                [
                    "Focus: info",
                    "Everyday ability: Info",
                    "weather, news, air quality, sunrise/sunset, history, holidays, Wikipedia, definitions",
                    "Example: `weather in Tokyo`",
                    "get_weather",
                    "get_news",
                    "get_air_quality",
                    "get_sun_times",
                    "on_this_day",
                    "next_holidays",
                    "wiki_summary",
                    "define",
                    "today in history",
                    "tell me about Ada Lovelace",
                    "does not run tools",
                ],
            ),
            (
                "capability map 날씨",
                [
                    "Focus: info",
                    "Everyday ability: Info",
                    "get_weather",
                    "news about Korea",
                ],
            ),
            (
                "what can Jarvis do productivity",
                [
                    "Focus: productivity",
                    "Everyday ability: Productivity",
                    "calendar, availability, email, tasks, reminders, timers",
                    "list_events",
                    "what is on my schedule today",
                    "what am I doing today",
                    "free today",
                    "check_availability",
                    "read_emails",
                    "search_emails",
                    "location_reminder_draft",
                    "set_reminder",
                    "show alarms",
                    "remind me to call Sam when I get home",
                    "schedule lunch tomorrow at noon",
                    "requires approval",
                ],
            ),
            (
                "capability map 이메일",
                [
                    "Focus: productivity",
                    "Everyday ability: Productivity",
                    "read_emails",
                    "search_emails",
                ],
            ),
            (
                "what can you do with reminders",
                [
                    "Focus: productivity",
                    "Everyday ability: Productivity",
                    "calendar, availability, email, tasks, reminders, timers",
                    "set_reminder",
                    "list_reminders",
                    "does not run tools",
                ],
            ),
            (
                "what can you do with calendar",
                [
                    "Focus: productivity",
                    "Everyday ability: Productivity",
                    "list_events",
                    "create_event",
                    "requires approval",
                ],
            ),
            (
                "what can you do with messages",
                [
                    "Focus: messages",
                    "Everyday ability: Messages & calls",
                    "KakaoTalk, Instagram, iMessage, Telegram, or email",
                    "send messages on KakaoTalk, Instagram, iMessage, Telegram, or email with approval; call routes exist but are disabled for the August 15 preview",
                    "Live proof: Telegram, Instagram, iMessage, and KakaoTalk sends have recipient-confirmed V3 evidence.",
                    "KakaoTalk required operator chat verification plus recipient confirmation after an outcome-unknown GUI attempt.",
                    "Every new send remains approval-gated",
                    "Calls are disabled for the August 15 functional preview; any later re-enable and live proof requires a separate, operator-present approval gate",
                    "Trust the Capability Cockpit and `channel health` for the live delivery state.",
                    "send_kakao",
                    "send_instagram_dm",
                    "send_imessage",
                    "send_telegram",
                    "send_email",
                    "find_contact",
                    "call_contact",
                    "call_kakao",
                    "call_telegram",
                    "call_instagram",
                    "send 가상연락처이 a telegram saying 안녕하세요",
                    "channel health",
                    "requires approval",
                    "does not run tools",
                ],
            ),
            (
                "what can you do with telegram",
                [
                    "Focus: messages",
                    "Everyday ability: Messages & calls",
                    "send_telegram",
                    "call_telegram",
                    "send 가상연락처일 a telegram saying hi",
                    "call routes exist but are disabled for the August 15 preview",
                    "Every new send remains approval-gated",
                    "Calls are disabled for the August 15 functional preview; any later re-enable and live proof requires a separate, operator-present approval gate",
                ],
            ),
            (
                "what can you do with voice",
                [
                    "Focus: voice",
                    "Everyday ability: Voice & images",
                    "push-to-talk or Telegram voice-note transcripts",
                    "confirmation receipts",
                    "voice_setup_check",
                    "voice_command_cockpit",
                    "voice_confirmation_receipt",
                    "voice_route_gate_packet",
                    "voice_audio_file_gate_packet",
                    "photo_document_intake_plan",
                    "ocr_image",
                    "voice stop intent: stop listening",
                    "does not run tools",
                ],
            ),
            (
                "what can you do with cockpit",
                [
                    "Focus: control_plane",
                    "Everyday ability: Control plane",
                    "read-only cockpit/status views",
                    "earned-trust checklist",
                    "capability_map",
                    "capability_cockpit",
                    "channel_health",
                    "jarvis_status",
                    "status_dashboard",
                    "readiness_report",
                    "safety_status",
                    "risk_matrix",
                    "completion_claim_gate",
                    "subagent_fleet_status",
                    "cockpit summary",
                    "trust checklist",
                    "earned trust checklist",
                    "why should I trust Jarvis",
                    "channel health",
                    "subagent fleet status",
                    "internal orchestration",
                    "does not run tools",
                ],
            ),
            (
                "what can you do with status",
                [
                    "Focus: control_plane",
                    "Everyday ability: Control plane",
                    "jarvis_status",
                    "status_dashboard",
                    "readiness_report",
                    "safety_status",
                    "completion_claim_gate",
                ],
            ),
            (
                "what can you do with agent status",
                [
                    "Focus: control_plane",
                    "Everyday ability: Control plane",
                    "subagent_fleet_status",
                    "internal orchestration",
                    "does not run tools",
                ],
            ),
            (
                "capability map 상태",
                [
                    "Focus: control_plane",
                    "Everyday ability: Control plane",
                    "capability_cockpit",
                    "channel_health",
                    "jarvis_status",
                ],
            ),
            (
                "show capabilities for reminders",
                [
                    "Focus: productivity",
                    "Everyday ability: Productivity",
                    "set_reminder",
                    "list_reminders",
                    "does not run tools",
                ],
            ),
            (
                "what are your reminder capabilities",
                [
                    "Focus: productivity",
                    "Everyday ability: Productivity",
                    "set_reminder",
                    "list_reminders",
                    "does not run tools",
                ],
            ),
            (
                "list reminder tools",
                [
                    "Focus: productivity",
                    "Everyday ability: Productivity",
                    "set_reminder",
                    "list_reminders",
                    "does not run tools",
                ],
            ),
            (
                "help with timers",
                [
                    "Focus: productivity",
                    "Everyday ability: Productivity",
                    "set_reminder",
                    "does not run tools",
                ],
            ),
            (
                "list task tools",
                [
                    "Focus: tasks",
                    "tasks:",
                    "list_tasks",
                    "does not run tools",
                ],
            ),
            (
                "capability map 할일",
                [
                    "Focus: tasks",
                    "tasks:",
                    "list_tasks",
                ],
            ),
            (
                "help with notes",
                [
                    "Focus: notes",
                    "Obsidian notes",
                    "list_jarvis_notes",
                    "take a note: ...",
                    "does not run tools",
                ],
            ),
            (
                "help with files",
                [
                    "Focus: files",
                    "find_files",
                    "read_text_file",
                    "requires approval",
                ],
            ),
            (
                "list approval tools",
                [
                    "Focus: approvals",
                    "approval queue",
                    "approval_history",
                    "does not run tools",
                ],
            ),
            (
                "help with scheduled jobs",
                [
                    "Focus: scheduler",
                    "scheduled jobs",
                    "list_scheduled_jobs",
                    "does not run tools",
                ],
            ),
            (
                "help with automation",
                [
                    "Focus: scheduler",
                    "scheduled jobs",
                    "list_scheduled_jobs",
                    "does not run tools",
                ],
            ),
            (
                "list automation tools",
                [
                    "Focus: scheduler",
                    "scheduled jobs",
                    "list_scheduled_jobs",
                    "does not run tools",
                ],
            ),
            (
                "list goal tools",
                [
                    "Focus: goals",
                    "goals and projects",
                    "list_goals",
                    "does not run tools",
                ],
            ),
            (
                "list decision tools",
                [
                    "Focus: decisions",
                    "decisions",
                    "list_decisions",
                    "does not run tools",
                ],
            ),
            (
                "list preference tools",
                [
                    "Focus: preferences",
                    "preferences",
                    "list_preferences",
                    "does not run tools",
                ],
            ),
            (
                "list skill tools",
                [
                    "Focus: skills",
                    "skills",
                    "list_skills",
                    "does not run tools",
                ],
            ),
            (
                "help with profiles",
                [
                    "Focus: profile",
                    "profile",
                    "read_profile",
                    "does not run tools",
                ],
            ),
            (
                "list person tools",
                [
                    "Focus: people",
                    "people memory",
                    "list_people",
                    "does not run tools",
                ],
            ),
            (
                "help with conversations",
                [
                    "Focus: conversation",
                    "conversation memory",
                    "chat_context",
                    "does not run tools",
                ],
            ),
            (
                "list chat tools",
                [
                    "Focus: conversation",
                    "conversation memory",
                    "chat_context",
                    "does not run tools",
                ],
            ),
            (
                "what can Jarvis do utilities",
                [
                    "Focus: utilities",
                    "Everyday ability: Utilities",
                    "translate, currency, unit conversion, time zones/time differences, relative dates, countdowns, calculate/BMI, spelling, password/UUID generation, text transforms/counts",
                    "translate",
                    "current_time",
                    "time_difference",
                    "relative_date",
                    "convert_currency",
                    "convert_units",
                    "days_until",
                    "calculate",
                    "calculate_bmi",
                    "generate_password",
                    "generate_uuid",
                    "spell_word",
                    "count_text",
                    "transform_text",
                    "convert 10 km to miles",
                    "what timezone am I in",
                    "time difference between Seoul and London",
                    "what date is tomorrow",
                    "what date is in two days",
                    "what day is next Friday",
                    "what is 10 pounds in kg",
                    "how many cups in a liter",
                    "how many weeks until Christmas",
                    "days until Thanksgiving",
                    "countdown Halloween",
                    "spell restaurant",
                    "word count hello world",
                    "letter count restaurant",
                    "uppercase hello world",
                    "uppercase the phrase hello world",
                    "convert hello world to snake case",
                    "camel case hello world",
                    "initials the operator",
                    "what are the initials of the operator",
                    "slugify hello world",
                    "bmi 70 kg 180 cm",
                    "generate a 16 character password",
                    "generate a 16 character password without symbols",
                    "generate uuid",
                    "repeat hello 3 times",
                    "when is Christmas",
                    "when is MLK Day",
                    "when is Thanksgiving",
                    "percentage change from 50 to 60",
                    "total with 8% tax on 100",
                    "add 15 and 20",
                    "does not run tools",
                ],
            ),
            (
                "capability map 번역",
                [
                    "Focus: utilities",
                    "Everyday ability: Utilities",
                    "convert 100 USD to KRW",
                ],
            ),
            (
                "what can Jarvis do markets",
                [
                    "Focus: markets",
                    "Everyday ability: Markets",
                    "crypto and stock price lookups",
                    "get_crypto_price",
                    "get_stock_price",
                    "get_markets_overview",
                    "how are the markets",
                    "Market",
                ],
            ),
            (
                "capability map 시장",
                [
                    "Focus: markets",
                    "Everyday ability: Markets",
                    "stock price of AAPL",
                ],
            ),
            (
                "what can Jarvis do research",
                [
                    "Focus: research",
                    "Everyday ability: Research & Web",
                    "web lookup, web search, page fetch/summarize, and research synthesis",
                    "web_lookup",
                    "research",
                    "fetch_page",
                    "web_search",
                    "recent_browser_pages",
                    "summarize_page",
                    "research Zoey OS",
                    "web lookup OpenClaw agent safety",
                    "web search for Jarvis agent safety patterns",
                    "fetch page https://example.com",
                    "recent browser pages",
                    "summarize latest page",
                    "does not run tools",
                ],
            ),
            (
                "what can Jarvis do web lookup",
                [
                    "Focus: research",
                    "Everyday ability: Research & Web",
                    "web_lookup",
                    "web_search",
                    "fetch_page",
                    "does not run tools",
                ],
            ),
            (
                "capability map 웹 검색",
                [
                    "Focus: research",
                    "Everyday ability: Research & Web",
                    "web_search",
                    "research Zoey OS",
                ],
            ),
            (
                "what can Jarvis do writing",
                [
                    "Focus: writing",
                    "Everyday ability: Writing",
                    "paste_text, human_write, compose_and_write",
                    "paste_text",
                    "human_write",
                    "compose_and_write",
                    "write a paragraph about Jarvis and type it",
                    "HIGH_RISK",
                    "does not run tools",
                ],
            ),
            (
                "what can Jarvis do fun",
                [
                    "Focus: fun",
                    "Everyday ability: Fun",
                    "jokes, coin flips, dice rolls, random numbers, option picks, and lightweight playful prompts",
                    "tell_joke",
                    "flip_coin",
                    "roll_dice",
                    "random_number",
                    "choose_option",
                    "flip a coin",
                    "toss a coin",
                    "roll a die",
                    "roll a pair of dice",
                    "roll 2 dice",
                    "pick a random number",
                    "pick a number between one and ten",
                    "choose between pizza and sushi",
                    "help me decide between pizza and sushi",
                    "dad joke",
                    "make me laugh",
                    "does not run tools",
                ],
            ),
            (
                "what can Jarvis do safety",
                [
                    "Focus: safety",
                    "safety and autonomy",
                    "prototype readiness",
                    "setup check",
                    "storage status",
                    "storage recovery plan",
                    "storage recovery check",
                    "computer control status",
                    "autonomy_plan",
                    "readiness_report",
                    "acceptance gate",
                ],
            ),
            (
                "capability map 안전",
                [
                    "Focus: safety",
                    "safety and autonomy",
                    "safety_status",
                ],
            ),
            (
                "what can Jarvis do scheduler",
                [
                    "Focus: scheduler",
                    "scheduler_context_refresh_packet",
                    "schedule assistant basics",
                    "EXTERNAL_SIDE_EFFECT",
                    "resume job State Snapshot",
                    "resume job Conversation Compaction",
                    "list scheduled jobs",
                ],
            ),
            (
                "tool search: approval",
                [
                    "Jarvis tool search: approval",
                    "approval queue",
                    "READ_ONLY",
                    "does not run tools",
                ],
            ),
            (
                "tool detail: run_shell_command",
                [
                    "Jarvis tool detail: run_shell_command",
                    "approval required: yes",
                    "HIGH_RISK",
                    "does not run the tool",
                ],
            ),
            (
                "risk matrix",
                [
                    "Jarvis risk matrix",
                    "Risk totals",
                    "Risk by area",
                    "Approval-gated examples",
                    "approval-gated",
                ],
            ),
            (
                "what permissions do you have?",
                [
                    "Jarvis risk matrix",
                    "Risk totals",
                    "Approval-gated examples",
                    "approval-gated",
                ],
            ),
            (
                "show my permissions",
                [
                    "Jarvis risk matrix",
                    "Risk by area",
                    "Approval-gated examples",
                ],
            ),
        ]
        for case, expected_parts in cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_parts:
                if expected not in result.response:
                    raise SystemExit(f"Capability map missing expected text for '{case}': {expected}")
            if case == "what can Jarvis do personal" and "integration dry run contract: email -> send draft reply; target thread; time today; data draft-only; verification confirm not sent" in result.response:
                raise SystemExit("Capability map still advertises draft-only dry-run as the personal starter example.")
            if case == "what can Jarvis do safety":
                ladder = ["storage status", "storage recovery plan", "storage recovery check"]
                positions = [result.response.find(item) for item in ladder]
                if any(position < 0 for position in positions) or positions != sorted(positions):
                    raise SystemExit(f"Safety capability starters should preserve storage recovery order {ladder}: {positions}")
            metadata = result.tool_results[0].metadata
            if metadata.get("capability_map_handoff_ready"):
                assert_capability_map_handoff(metadata, case)
            if metadata.get("tool_search_handoff_ready"):
                assert_inspection_handoff(metadata, case, "tool_search", status="ok", required_commands={"risk matrix", "capability map"})
                handoff = metadata["tool_search_handoff"]
                if handoff.get("query") != metadata.get("query") or handoff.get("count") != metadata.get("count") or handoff.get("matches") != metadata.get("matches"):
                    raise SystemExit(f"tool_search handoff missed flat metadata parity: {handoff} vs {metadata}")
            if metadata.get("tool_detail_handoff_ready"):
                assert_inspection_handoff(metadata, case, "tool_detail", status="ok", required_commands={"risk matrix", "capability map"})
                handoff = metadata["tool_detail_handoff"]
                for parity_key in ["name", "found", "toolset", "label", "risk", "approval_required", "description"]:
                    if handoff.get(parity_key) != metadata.get(parity_key):
                        raise SystemExit(f"tool_detail handoff missed {parity_key} parity: {handoff} vs {metadata}")
            if metadata.get("risk_matrix_handoff_ready"):
                assert_inspection_handoff(metadata, case, "risk_matrix", status="ok", required_commands={"tool search: approval", "tool detail: run_shell_command", "capability map"})
                handoff = metadata["risk_matrix_handoff"]
                for parity_key in ["tools", "risk_totals", "matrix", "approval_gated", "limit"]:
                    if handoff.get(parity_key) != metadata.get(parity_key):
                        raise SystemExit(f"risk_matrix handoff missed {parity_key} parity: {handoff} vs {metadata}")
            for key in [
                "calls_model",
                "executes_tools",
                "queues_approval",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
                "approves_request",
                "dismisses_request",
                "controls_computer",
                "reads_private_data",
                "writes_files",
                "writes_notes",
                "writes_memory",
                "external_side_effect",
                "requires_approval",
            ]:
                if metadata.get(key) is not False:
                    raise SystemExit(f"Capability tool unsafe metadata {key}: {metadata}")

        readme = Path("README.md").read_text()
        for expected in [
            "what can Jarvis do info",
            "what can Jarvis do productivity",
            "what can Jarvis do utilities",
            "what can Jarvis do markets",
            "what can Jarvis do research",
            "what can Jarvis do writing",
            "what can Jarvis do fun",
            "list matching tools and read-only boundaries",
        ]:
            if expected not in readme:
                raise SystemExit(f"README missed category-focused capability map documentation: {expected}")

        bad_search = runtime.registry.get("tool_search").handler({"query": "", "limit": "bad"})
        if bad_search.ok or bad_search.metadata.get("reason") != "missing_query":
            raise SystemExit("tool_search missing query missed safe refusal metadata.")
        assert_inspection_handoff(bad_search.metadata, "tool_search missing query", "tool_search", status="refused", required_commands={"tool search: approvals", "capability map", "risk matrix"})
        if bad_search.metadata.get("limit") != 12 or bad_search.metadata.get("raw_limit") != "bad":
            raise SystemExit(f"tool_search missing query missed bounded raw limit metadata: {bad_search.metadata}")
        bool_search = runtime.registry.get("tool_search").handler({"query": "", "limit": False})
        if bool_search.ok or bool_search.metadata.get("limit") != 12 or bool_search.metadata.get("raw_limit") != "False":
            raise SystemExit(f"tool_search should treat boolean limits as malformed and preserve raw metadata: {bool_search.metadata}")
        long_bad_search = runtime.registry.get("tool_search").handler({"query": "", "limit": "l" * 200})
        if long_bad_search.metadata.get("raw_limit") != ("l" * 77 + "..."):
            raise SystemExit(f"tool_search missing query did not bound raw limit metadata: {long_bad_search.metadata}")
        path_bad_search = runtime.registry.get("tool_search").handler({"query": "", "limit": "/\x55sers/example/private/capability-limit"})
        if path_bad_search.ok or path_bad_search.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"tool_search should redact path-shaped bad limits: {path_bad_search.metadata}")
        assert_no_local_path(path_bad_search.metadata, "tool_search path-shaped bad limit metadata")
        temp_bad_search = runtime.registry.get("tool_search").handler({"query": "", "limit": "/var/folders/zc/jarvis-capability-limit"})
        if temp_bad_search.ok or temp_bad_search.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"tool_search should redact temp-root bad limits: {temp_bad_search.metadata}")
        assert_no_local_path(temp_bad_search.metadata, "tool_search temp-root bad limit metadata")
        path_query_search = runtime.registry.get("tool_search").handler({"query": "/tmp/jarvis-capability-query", "limit": 3})
        if not path_query_search.ok or "<local-path>" not in path_query_search.output:
            raise SystemExit(f"tool_search should redact path-shaped no-match queries: {path_query_search.output}")
        assert_no_local_path(path_query_search.output, "tool_search path-shaped query output")
        assert_no_local_path(path_query_search.metadata, "tool_search path-shaped query metadata")
        assert_inspection_handoff(path_query_search.metadata, "tool_search path-shaped query", "tool_search", status="empty", required_commands={"capability map", "tool search: approvals", "risk matrix"})

        large_matrix = runtime.registry.get("risk_matrix").handler({"limit": 999999})
        if not large_matrix.ok or large_matrix.metadata.get("limit") != 50:
            raise SystemExit("risk_matrix did not clamp a large limit.")
        assert_inspection_handoff(large_matrix.metadata, "risk_matrix large limit", "risk_matrix", status="ok", required_commands={"tool search: approval", "tool detail: run_shell_command", "capability map"})
        bool_matrix = runtime.registry.get("risk_matrix").handler({"limit": True})
        if not bool_matrix.ok or bool_matrix.metadata.get("limit") != 8 or bool_matrix.metadata.get("raw_limit") != "True":
            raise SystemExit(f"risk_matrix should treat boolean limits as malformed and preserve raw metadata: {bool_matrix.metadata}")
        bad_matrix = runtime.registry.get("risk_matrix").handler({"limit": "bad"})
        if not bad_matrix.ok or bad_matrix.metadata.get("limit") != 8 or bad_matrix.metadata.get("raw_limit") != "bad":
            raise SystemExit(f"risk_matrix did not preserve bounded raw bad limit metadata: {bad_matrix.metadata}")
        path_bad_matrix = runtime.registry.get("risk_matrix").handler({"limit": "/private/tmp/jarvis-risk-matrix-limit"})
        if not path_bad_matrix.ok or path_bad_matrix.metadata.get("limit") != 8 or path_bad_matrix.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"risk_matrix should redact path-shaped bad limits: {path_bad_matrix.metadata}")
        assert_no_local_path(path_bad_matrix.metadata, "risk_matrix path-shaped bad limit metadata")
        temp_bad_matrix = runtime.registry.get("risk_matrix").handler({"limit": "/tmp/jarvis-risk-matrix-limit"})
        if not temp_bad_matrix.ok or temp_bad_matrix.metadata.get("limit") != 8 or temp_bad_matrix.metadata.get("raw_limit") != "<local-path>":
            raise SystemExit(f"risk_matrix should redact temp-root bad limits: {temp_bad_matrix.metadata}")
        assert_no_local_path(temp_bad_matrix.metadata, "risk_matrix temp-root bad limit metadata")

        bad_detail = runtime.registry.get("tool_detail").handler({"name": ""})
        if bad_detail.ok or bad_detail.metadata.get("reason") != "missing_name":
            raise SystemExit("tool_detail missing name missed safe refusal metadata.")
        assert_inspection_handoff(bad_detail.metadata, "tool_detail missing name", "tool_detail", status="refused", required_commands={"tool detail: run_shell_command", "tool search: approvals", "capability map"})
        if bad_detail.metadata.get("raw_name") != "":
            raise SystemExit(f"tool_detail missing name missed bounded raw name metadata: {bad_detail.metadata}")
        path_detail = runtime.registry.get("tool_detail").handler({"name": "/var/folders/zc/jarvis-tool-detail"})
        if not path_detail.ok or "<local-path>" not in path_detail.output:
            raise SystemExit(f"tool_detail should redact path-shaped no-match names: {path_detail.output}")
        assert_no_local_path(path_detail.output, "tool_detail path-shaped name output")
        assert_no_local_path(path_detail.metadata, "tool_detail path-shaped name metadata")
        assert_inspection_handoff(path_detail.metadata, "tool_detail path-shaped name", "tool_detail", status="empty", required_commands={"capability map", "risk matrix"})

        path_focus = runtime.registry.get("capability_map").handler({"focus": "/tmp/jarvis-capability-focus"})
        if not path_focus.ok or "<local-path>" not in path_focus.output:
            raise SystemExit(f"capability_map should redact path-shaped focus values: {path_focus.output}")
        assert_no_local_path(path_focus.output, "capability_map path-shaped focus output")
        assert_no_local_path(path_focus.metadata, "capability_map path-shaped focus metadata")
        assert_capability_map_handoff(path_focus.metadata, "capability_map path-shaped focus", status="empty")
        computer_focus = runtime.registry.get("capability_map").handler({"focus": "computer"})
        if not computer_focus.ok or "<local-path>" not in computer_focus.output:
            raise SystemExit(f"computer capability examples should redact local screenshot paths: {computer_focus.output}")
        assert_no_local_path(computer_focus.output, "computer capability map output")
        assert_capability_map_handoff(computer_focus.metadata, "computer capability map")


if __name__ == "__main__":
    main()
