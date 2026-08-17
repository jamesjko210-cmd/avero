"""Pin the capability cockpit: the visible control plane for what Jarvis can
do, each lane's health, risk, approval need, and example command.

Direction note (2026-07-04): the product moat is a capability cockpit, not a
companion roster. This smoke pins that the cockpit stays read-only, derives
from the live registry + audit trail, treats approval-gate stops as
"awaiting approval" (safety working) rather than failures, tolerates
malformed audit rows, and never leaks local paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.tools.cockpit import CAPABILITY_LANES
from jarvis_v2.tools.registry import Tool
from jarvis_v2.tools import cockpit as cockpit_tools

_LEAK_MARKER = "/\x55sers/hostile/secret-cockpit-row"


class _ExplodingRow:
    def __getitem__(self, key: str):
        raise RuntimeError(f"boom {_LEAK_MARKER}")


class _HostileTruthiness:
    def __bool__(self) -> bool:
        raise RuntimeError(f"truthiness boom {_LEAK_MARKER}")

    def __str__(self) -> str:
        return _LEAK_MARKER


class _ExplodingString:
    def __bool__(self) -> bool:
        return True

    def __str__(self) -> str:
        raise RuntimeError(f"string boom {_LEAK_MARKER}")


class _ExplodingToolName:
    @property
    def name(self):
        raise RuntimeError(f"bad tool name {_LEAK_MARKER}")


class _ExplodingRiskName:
    @property
    def name(self):
        raise RuntimeError(f"bad risk name {_LEAK_MARKER}")


class _BadRiskTool:
    name = "bad_risk_fixture"
    risk = _ExplodingRiskName()


class _SortingRegistry:
    def __init__(self, tools):
        self._tools = {f"fixture_{index}": tool for index, tool in enumerate(tools)}

    def list(self):
        return sorted(self._tools.values(), key=lambda tool: tool.name)


def _run_cockpit(runtime):
    return runtime.registry.get("capability_cockpit").handler({})


class _EmptyStore:
    def recent_tool_runs(self, limit: int = 400):
        return []


def _noop_tool(args: dict) -> ToolResult:
    return ToolResult("noop", True, "ok")


def test_cockpit_reports_all_lanes_read_only() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = _run_cockpit(runtime)
        if not result.ok:
            raise SystemExit("capability_cockpit must succeed on a fresh runtime")
        metadata = result.metadata
        if metadata.get("direction_summary") != "One Jarvis coordinator, visible capability cockpit, internal workers only.":
            raise SystemExit(f"cockpit should expose the research-backed direction summary: {metadata}")
        principles = metadata.get("direction_principles") or []
        expected_direction_phrases = [
            "Show capabilities, health, risk, approvals, smoke coverage, and next commands",
            "subagents/internal workers invisible as personas",
            "Do not add Zoey-style companion identities",
            "Delay broad integration bridges until Korean messaging",
        ]
        for expected in expected_direction_phrases:
            if not any(expected in str(principle) for principle in principles):
                raise SystemExit(f"cockpit direction principles missed {expected!r}: {metadata}")
        if metadata.get("direction_rejects_companion_roster") is not True:
            raise SystemExit(f"cockpit should explicitly reject companion-roster drift: {metadata}")
        if metadata.get("direction_exposes_internal_workers_as_capacity") is not True:
            raise SystemExit(f"cockpit should expose internal workers only as capacity: {metadata}")
        if "Direction: One Jarvis coordinator, visible capability cockpit, internal workers only." not in result.output:
            raise SystemExit("cockpit text should render the direction summary")
        if "Direction guardrail: Do not add Zoey-style companion identities" not in result.output:
            raise SystemExit("cockpit text should render the companion-identity guardrail")
        if "Direction guardrail: Delay broad integration bridges until Korean messaging" not in result.output:
            raise SystemExit("cockpit text should render the integration-sprawl guardrail")
        if metadata["lane_count"] != len(CAPABILITY_LANES):
            raise SystemExit(f"expected {len(CAPABILITY_LANES)} lanes, got {metadata['lane_count']}")
        if sum(int(count) for count in metadata.get("status_counts", {}).values()) != metadata["lane_count"]:
            raise SystemExit(f"cockpit status counts should cover every lane: {metadata}")
        coverage = metadata.get("coverage_summary")
        if not isinstance(coverage, dict):
            raise SystemExit(f"cockpit should expose a structured coverage summary: {metadata}")
        if coverage.get("lane_count") != metadata["lane_count"]:
            raise SystemExit(f"coverage summary lane count should mirror cockpit lane count: {metadata}")
        if coverage.get("tool_ready_count") != metadata["lane_count"]:
            raise SystemExit(f"all fresh-runtime cockpit lanes should have registered tool coverage: {metadata}")
        if coverage.get("smoke_ready_count") != metadata["lane_count"]:
            raise SystemExit(f"all fresh-runtime cockpit lanes should have registered smoke coverage: {metadata}")
        if coverage.get("hidden_smoke_metadata_count") != 0:
            raise SystemExit(f"fresh-runtime cockpit should have no hidden smoke metadata: {metadata}")
        if coverage.get("hidden_tool_metadata_count") != 0:
            raise SystemExit(f"fresh-runtime cockpit should have no hidden tool metadata: {metadata}")
        if coverage.get("approval_required_count") != metadata.get("approval_required_lane_count"):
            raise SystemExit(f"top-level approval count should mirror coverage summary: {metadata}")
        if coverage.get("auto_run_safe_count") != metadata.get("auto_run_safe_lane_count"):
            raise SystemExit(f"top-level auto-run-safe count should mirror coverage summary: {metadata}")
        if coverage.get("approval_required_count", 0) + coverage.get("auto_run_safe_count", 0) != metadata["lane_count"]:
            raise SystemExit(f"coverage summary risk split should cover every lane: {metadata}")
        trust = metadata.get("trust_summary")
        if not isinstance(trust, dict):
            raise SystemExit(f"cockpit should expose a structured earned-trust summary: {metadata}")
        if metadata.get("trust_non_authorizing") is not True or trust.get("trust_non_authorizing") is not True:
            raise SystemExit(f"cockpit trust summary must stay non-authorizing: {metadata}")
        if trust.get("lane_count") != metadata["lane_count"]:
            raise SystemExit(f"trust summary lane count should mirror cockpit lane count: {metadata}")
        if trust.get("trust_ready_lane_count") != metadata.get("trust_ready_lane_count"):
            raise SystemExit(f"top-level trust count should mirror trust summary: {metadata}")
        if trust.get("trust_partial_lane_count") != metadata.get("trust_partial_lane_count"):
            raise SystemExit(f"top-level trust partial count should mirror trust summary: {metadata}")
        if trust.get("trust_ready_check_count") != metadata.get("trust_ready_check_count"):
            raise SystemExit(f"top-level trust ready-check count should mirror trust summary: {metadata}")
        if trust.get("trust_check_count") != metadata.get("trust_check_count"):
            raise SystemExit(f"top-level trust check count should mirror trust summary: {metadata}")
        if metadata.get("trust_ready") is not True or trust.get("trust_ready") is not True:
            raise SystemExit(f"fresh-runtime cockpit trust checklist should be ready: {metadata}")
        contract = metadata.get("earned_trust_contract") or []
        for expected in (
            "registered tools and max risk",
            "approval boundary before execution",
            "aggregate smoke coverage",
            "explicit proof points",
        ):
            if not any(expected in str(item) for item in contract):
                raise SystemExit(f"earned-trust contract missed {expected!r}: {metadata}")
        if "trust checklist" not in str(trust.get("summary") or ""):
            raise SystemExit(f"trust summary should be human-readable: {metadata}")
        if f"Trust summary: {trust['summary']}" not in result.output:
            raise SystemExit("cockpit text should render the earned-trust summary")
        if metadata.get("tool_coverage_ready_lane_count") != metadata["lane_count"]:
            raise SystemExit(f"top-level tool coverage count should cover every lane: {metadata}")
        if metadata.get("smoke_coverage_ready_lane_count") != metadata["lane_count"]:
            raise SystemExit(f"top-level smoke coverage count should cover every lane: {metadata}")
        if metadata.get("coverage_ready") is not True or coverage.get("coverage_ready") is not True:
            raise SystemExit(f"fresh-runtime cockpit coverage should be ready: {metadata}")
        summary_text = coverage.get("summary") or ""
        if "tools " not in summary_text or "smokes " not in summary_text or "approval-gated " not in summary_text:
            raise SystemExit(f"coverage summary should be human-readable: {metadata}")
        if "hidden smoke metadata" in summary_text:
            raise SystemExit(f"healthy cockpit summary should not mention hidden smoke metadata: {metadata}")
        if f"Coverage summary: {summary_text}" not in result.output:
            raise SystemExit("cockpit text should render the coverage summary")
        if not metadata.get("next_command") or metadata.get("next_command_count", 0) < 1:
            raise SystemExit(f"cockpit should expose a top-level next command queue: {metadata}")
        if metadata["next_command"] != metadata["next_commands"][0]:
            raise SystemExit(f"top-level next_command should mirror first queued command: {metadata}")
        queue = metadata.get("next_command_queue") or []
        if len(queue) != metadata["next_command_count"] or metadata["next_commands"] != [entry.get("command") for entry in queue]:
            raise SystemExit(f"cockpit next command queue should match flattened commands: {metadata}")
        for flag in ("calls_model", "executes_tools", "queues_approval", "requires_approval", "external_side_effect", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
            if metadata.get(flag):
                raise SystemExit(f"cockpit boundary flag {flag} must be false")
        for lane in metadata["capability_lanes"]:
            if not lane["example_command"]:
                raise SystemExit(f"lane {lane['key']} missing example command")
            if not lane.get("next_command") or lane.get("next_command_kind") not in {"example", "diagnostic", "approval_review"}:
                raise SystemExit(f"lane {lane['key']} missing typed next command: {lane}")
            if lane["status"] in {"ready", "ok"} and lane.get("next_command") != lane["example_command"]:
                raise SystemExit(f"healthy lane should stage its example command: {lane}")
            if lane["status"] in {"ready", "ok"} and lane.get("next_command_kind") != "example":
                raise SystemExit(f"healthy lane should mark next command as example: {lane}")
            if lane.get("tool_coverage") != "registered" or lane.get("tool_coverage_ready") is not True:
                raise SystemExit(f"lane {lane['key']} tool coverage should be registered: {lane}")
            if not lane.get("registered_tools"):
                raise SystemExit(f"lane {lane['key']} should list registered tools: {lane}")
            if lane["missing_tools"]:
                raise SystemExit(f"lane {lane['key']} references missing tools: {lane['missing_tools']}")
            if lane.get("attention_reasons"):
                raise SystemExit(f"lane {lane['key']} should not carry attention reasons while ready: {lane}")
            if lane.get("smoke_coverage") != "registered" or lane.get("smoke_coverage_ready") is not True:
                raise SystemExit(f"lane {lane['key']} smoke coverage should be registered: {lane}")
            if lane.get("missing_smoke_modules"):
                raise SystemExit(f"lane {lane['key']} has unregistered smoke coverage: {lane}")
            if not lane.get("registered_smoke_modules"):
                raise SystemExit(f"lane {lane['key']} should name aggregate smoke coverage: {lane}")
            if lane.get("trust_non_authorizing") is not True:
                raise SystemExit(f"lane {lane['key']} trust checklist must stay non-authorizing: {lane}")
            checks = lane.get("trust_checklist")
            if not isinstance(checks, list) or len(checks) < 4:
                raise SystemExit(f"lane {lane['key']} should expose a bounded trust checklist: {lane}")
            labels = [str(check.get("label")) for check in checks if isinstance(check, dict)]
            for expected in ("registered tools", "aggregate smoke coverage", "next command"):
                if expected not in labels:
                    raise SystemExit(f"lane {lane['key']} trust checklist missed {expected!r}: {lane}")
            if not any(label in {"approval boundary", "auto-run boundary"} for label in labels):
                raise SystemExit(f"lane {lane['key']} trust checklist missed the execution boundary: {lane}")
            if lane.get("trust_ready") is not True:
                raise SystemExit(f"fresh-runtime lane {lane['key']} trust checklist should be ready: {lane}")
            if lane.get("trust_ready_count") != lane.get("trust_check_count"):
                raise SystemExit(f"lane {lane['key']} trust ready/check counts should match: {lane}")


def test_approval_gate_stop_reads_as_awaiting_approval_not_failure() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        approval_id = runtime.store.add_pending_approval(
            runtime.session_id,
            "send a telegram proof",
            "send_telegram",
            "approval required",
            {"to": "owner", "message": "proof"},
        )
        runtime.store.log_tool_run(
            session_id=runtime.session_id,
            tool_name="send_telegram",
            risk="HIGH_RISK",
            ok=False,
            approved=False,
            approval_id=approval_id,
            output="approval required",
            metadata={"failure_kind": "approval_required"},
        )
        result = _run_cockpit(runtime)
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes["messaging"]
        if lane["status"] != "awaiting approval":
            raise SystemExit(
                f"approval-gate stop must read 'awaiting approval', got {lane['status']!r}"
            )
        if "send_telegram" not in lane["last_approval_hold"]:
            raise SystemExit(f"approval-held lane should preserve the held tool receipt: {lane}")
        if lane["last_approval_id"] != approval_id:
            raise SystemExit(f"approval-held lane should carry the linked approval id: {lane}")
        if lane["last_failure"] or lane["last_failure_kind"]:
            raise SystemExit(f"approval-held lane should not masquerade as a failed run: {lane}")
        expected_commands = [
            f"approval readiness {approval_id}",
            f"approval packet {approval_id}",
            f"approval chain proof {approval_id}",
        ]
        if lane["approval_next_commands"] != expected_commands:
            raise SystemExit(f"approval-held lane should expose review commands: {lane}")
        if lane["next_command"] != f"approval readiness {approval_id}" or lane["next_command_kind"] != "approval_review":
            raise SystemExit(f"approval-held lane should stage approval readiness as next command: {lane}")
        first = result.metadata.get("next_command_queue", [{}])[0]
        if result.metadata.get("next_command") != f"approval readiness {approval_id}" or first.get("kind") != "approval_review":
            raise SystemExit(f"approval-held cockpit should prioritize approval review in the global queue: {result.metadata}")
        if "send_telegram" not in str(first.get("last_approval_hold") or ""):
            raise SystemExit(f"approval-held queue entry should preserve the held tool receipt: {first}")
        if first.get("last_approval_id") != approval_id:
            raise SystemExit(f"approval-held queue entry should carry the linked approval id: {first}")
        if first.get("last_failure") or first.get("last_failure_kind"):
            raise SystemExit(f"approval-held queue entry should not carry failure context: {first}")
        if first.get("approval_next_commands") != expected_commands:
            raise SystemExit(f"approval-held queue entry should expose review commands: {first}")
        if "approval hold: send_telegram" not in result.output or f"`approval packet {approval_id}`" not in result.output:
            raise SystemExit("approval-held receipt and next commands should be visible in cockpit text")
        if "last failure: send_telegram" in result.output:
            raise SystemExit("approval-held cockpit text should not render the approval gate as a failure")
        if f"next: `approval readiness {approval_id}` (approval_review)" not in result.output:
            raise SystemExit("approval-held lane should render the typed next command")
        if "Next command queue:" not in result.output or f"`approval readiness {approval_id}` (approval_review" not in result.output:
            raise SystemExit("approval-held global next command queue should be visible in cockpit text")


def test_approval_gate_aliases_read_as_awaiting_approval() -> None:
    cases = [
        {"failure_stage": "approval required"},
        {"failure_kind": "approval-gate"},
        {"failure_kind": "explicit approval required"},
        {"requires_confirmation": True},
        {"requires_approval": "true"},
    ]
    for metadata in cases:
        with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
            runtime = make_temp_runtime(Path(tmp))
            approval_id = runtime.store.add_pending_approval(
                runtime.session_id,
                "send a telegram proof",
                "send_telegram",
                "approval required",
                {"to": "owner", "message": "proof"},
            )
            runtime.store.log_tool_run(
                session_id=runtime.session_id,
                tool_name="send_telegram",
                risk="HIGH_RISK",
                ok=False,
                approved=False,
                approval_id=approval_id,
                output="approval held",
                metadata=metadata,
            )
            result = _run_cockpit(runtime)
            lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
            lane = lanes["messaging"]
            if lane["status"] != "awaiting approval":
                raise SystemExit(f"approval alias should read as awaiting approval: {metadata!r} -> {lane}")
            if lane["last_failure"] or lane["last_failure_kind"]:
                raise SystemExit(f"approval alias should not populate failure fields: {metadata!r} -> {lane}")
            expected_commands = [
                f"approval readiness {approval_id}",
                f"approval packet {approval_id}",
                f"approval chain proof {approval_id}",
            ]
            if lane["approval_next_commands"] != expected_commands:
                raise SystemExit(f"approval alias should expose bound review commands: {metadata!r} -> {lane}")
            if lane["next_command"] != expected_commands[0] or lane["next_command_kind"] != "approval_review":
                raise SystemExit(f"approval alias should stage bound approval readiness: {metadata!r} -> {lane}")
            if lane["attention_reasons"]:
                raise SystemExit(f"approval alias should not create attention reasons: {metadata!r} -> {lane}")
            if "latest failure" in result.output:
                raise SystemExit(f"approval alias should not render as a failure attention reason: {metadata!r}")


def test_terminal_or_unbound_approval_holds_are_not_current_review_state() -> None:
    for terminal_status in ("dismissed", "approved"):
        with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
            runtime = make_temp_runtime(Path(tmp))
            approval_id = runtime.store.add_pending_approval(
                runtime.session_id,
                "send a telegram proof",
                "send_telegram",
                "approval required",
                {"to": "owner", "message": "proof"},
            )
            runtime.store.log_tool_run(
                session_id=runtime.session_id,
                tool_name="send_telegram",
                risk="HIGH_RISK",
                ok=False,
                approved=False,
                approval_id=approval_id,
                output="approval held",
                metadata={"failure_kind": "approval_required"},
            )
            if not runtime.store.set_pending_approval_status(approval_id, terminal_status):
                raise SystemExit(f"fixture could not set approval {approval_id} to {terminal_status}")
            result = _run_cockpit(runtime)
            lane = {item["key"]: item for item in result.metadata["capability_lanes"]}["messaging"]
            if lane["status"] == "awaiting approval":
                raise SystemExit(f"terminal {terminal_status} approval survived as current review state: {lane}")
            if lane["last_approval_hold"] or lane["last_approval_id"] is not None or lane["approval_next_commands"]:
                raise SystemExit(f"terminal {terminal_status} approval kept review metadata: {lane}")
            if f"approval readiness {approval_id}" in result.output or f"approval packet {approval_id}" in result.output:
                raise SystemExit(f"terminal {terminal_status} approval remained in cockpit commands: {result.output}")

    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        runtime.store.log_tool_run(
            session_id=runtime.session_id,
            tool_name="send_telegram",
            risk="HIGH_RISK",
            ok=False,
            approved=False,
            output="legacy approval held without a durable id",
            metadata={"failure_kind": "approval_required"},
        )
        result = _run_cockpit(runtime)
        lane = {item["key"]: item for item in result.metadata["capability_lanes"]}["messaging"]
        if lane["status"] == "awaiting approval" or lane["approval_next_commands"]:
            raise SystemExit(f"unbound legacy hold should not advertise current approval review: {lane}")


def test_real_failure_reads_as_attention_with_last_failure() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        runtime.store.log_tool_run(
            session_id=runtime.session_id,
            tool_name="send_telegram",
            risk="HIGH_RISK",
            ok=False,
            approved=True,
            output="delivery failed",
            metadata={"failure_kind": "transport_error"},
        )
        result = _run_cockpit(runtime)
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes["messaging"]
        if lane["status"] != "attention":
            raise SystemExit(f"real failure must read 'attention', got {lane['status']!r}")
        if "send_telegram" not in lane["last_failure"]:
            raise SystemExit("last_failure must name the failing tool")
        if lane["last_failure_kind"] != "transport_error":
            raise SystemExit(f"failure kind lost: {lane['last_failure_kind']!r}")
        if "latest failure: transport_error" not in lane["attention_reasons"]:
            raise SystemExit(f"failure attention reason missing: {lane}")
        if lane["next_command"] != "channel health" or lane["next_command_kind"] != "diagnostic":
            raise SystemExit(f"messaging attention should stage channel health diagnostic: {lane}")
        first = result.metadata.get("next_command_queue", [{}])[0]
        if result.metadata.get("next_command") != "channel health" or first.get("kind") != "diagnostic":
            raise SystemExit(f"attention cockpit should prioritize the diagnostic in the global queue: {result.metadata}")
        if "send_telegram" not in str(first.get("last_failure") or "") or first.get("last_failure_kind") != "transport_error":
            raise SystemExit(f"failure queue entry should carry the bounded failure stage: {first}")
        if "latest failure: transport_error" not in result.output:
            raise SystemExit("failure attention reason should be visible in text output")
        if "next: `channel health` (diagnostic)" not in result.output:
            raise SystemExit("attention lane should render the diagnostic next command")


def test_string_false_audit_ok_does_not_become_success() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        runtime.store.recent_tool_runs = lambda limit=400: [
            {
                "tool_name": "send_telegram",
                "ok": "0",
                "created_at": "2026-07-05T02:45:00Z",
                "metadata": json.dumps({"failure_stage": "transport_error"}),
            },
            {
                "tool_name": "daily_brief",
                "ok": "definitely maybe",
                "created_at": f"2026-07-05T02:44:00Z at {_LEAK_MARKER}",
                "metadata": "{}",
            },
        ]
        result = _run_cockpit(runtime)
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        messaging = lanes["messaging"]
        if messaging["status"] != "attention":
            raise SystemExit(f"string '0' audit row should read as attention, got {messaging['status']!r}")
        if "send_telegram" not in messaging["last_failure"] or messaging["last_success"]:
            raise SystemExit(f"string '0' audit row was not treated as a failure: {messaging}")
        if messaging["last_failure_kind"] != "transport_error":
            raise SystemExit(f"string failure metadata lost: {messaging['last_failure_kind']!r}")
        if "latest failure: transport_error" not in messaging["attention_reasons"]:
            raise SystemExit(f"string failure attention reason missing: {messaging}")
        if messaging["next_command"] != "channel health" or messaging["next_command_kind"] != "diagnostic":
            raise SystemExit(f"string failure should stage channel health diagnostic: {messaging}")
        if result.metadata["unreadable_tool_run_rows"] != 1:
            raise SystemExit(f"ambiguous ok text should be hidden as unreadable: {result.metadata}")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if _LEAK_MARKER in blob:
            raise SystemExit("unreadable ambiguous ok row leaked seeded local path text")


def test_malformed_audit_rows_hidden_and_paths_scrubbed() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        hostile_tool_name_row = {
            "tool_name": _HostileTruthiness(),
            "ok": 1,
            "created_at": _HostileTruthiness(),
            "metadata": "{}",
        }
        good_row = {
            "tool_name": "daily_brief",
            "ok": 1,
            "created_at": f"2026-07-04T09:00:00Z at {_LEAK_MARKER}",
            "metadata": "{}",
        }
        runtime.store.recent_tool_runs = lambda limit=400: [_ExplodingRow(), hostile_tool_name_row, good_row]
        result = _run_cockpit(runtime)
        if not result.ok:
            raise SystemExit("cockpit must survive malformed audit rows")
        metadata = result.metadata
        if metadata["unreadable_tool_run_rows"] != 2:
            raise SystemExit(f"expected 2 hidden rows, got {metadata['unreadable_tool_run_rows']}")
        blob = result.output + json.dumps(metadata, ensure_ascii=False)
        if _LEAK_MARKER in blob:
            raise SystemExit("cockpit leaked a raw local path from an audit row")
        lanes = {lane["key"]: lane for lane in metadata["capability_lanes"]}
        if "<local-path>" not in lanes["morning_brief"]["last_success"]:
            raise SystemExit("path-shaped audit text must be redacted, not dropped")
        if "last success: daily_brief @ 2026-07-04T09:00:00Z at <local-path>" not in result.output:
            raise SystemExit("cockpit text should render bounded last-success receipts for healthy lanes")


def test_dashboard_snapshot_carries_cockpit_lanes() -> None:
    from jarvis_v2.ui.status_server import build_status_snapshot

    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        snapshot = build_status_snapshot(runtime)
        if snapshot.capability_cockpit_direction_summary != "One Jarvis coordinator, visible capability cockpit, internal workers only.":
            raise SystemExit(f"status snapshot lost cockpit direction summary: {snapshot.capability_cockpit_direction_summary!r}")
        if not any(
            "Do not add Zoey-style companion identities" in principle
            for principle in snapshot.capability_cockpit_direction_principles
        ):
            raise SystemExit(
                f"status snapshot lost cockpit direction principles: {snapshot.capability_cockpit_direction_principles!r}"
            )
        if not any(
            "Delay broad integration bridges until Korean messaging" in principle
            for principle in snapshot.capability_cockpit_direction_principles
        ):
            raise SystemExit(
                f"status snapshot lost integration-sprawl guardrail: {snapshot.capability_cockpit_direction_principles!r}"
            )
        if len(snapshot.capability_cockpit_lanes) != len(CAPABILITY_LANES):
            raise SystemExit("status snapshot must carry all cockpit lanes")
        if not isinstance(snapshot.capability_cockpit_attention, list):
            raise SystemExit("snapshot attention list malformed")


def test_internal_orchestration_lane_is_visible_but_read_only() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = _run_cockpit(runtime)
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes.get("orchestration")
        if lane is None:
            raise SystemExit("cockpit must expose the internal orchestration lane")
        if lane["title"] != "Internal Orchestration":
            raise SystemExit(f"orchestration lane title drifted: {lane}")
        if lane["risk"] != "READ_ONLY" or lane["approval_required"] is not False:
            raise SystemExit(f"orchestration lane must stay read-only and auto-run safe: {lane}")
        if lane["tool_coverage"] != "registered" or lane["registered_tools"] != ["subagent_fleet_status"]:
            raise SystemExit(f"orchestration lane lost the subagent status tool: {lane}")
        if lane["smoke_coverage"] != "registered" or "smoke_test_subagent_fleet" not in lane["registered_smoke_modules"]:
            raise SystemExit(f"orchestration lane must be pinned by subagent smoke coverage: {lane}")
        if lane["example_command"] != "jarvis status":
            raise SystemExit(f"orchestration lane should reuse the routed status command: {lane}")
        proof_points = " | ".join(lane.get("proof_points") or [])
        for expected in (
            "worker readiness and capacity without spawning tasks",
            "not user-facing companion personas",
            "without granting autonomy, approvals, or tool execution",
        ):
            if expected not in proof_points:
                raise SystemExit(f"orchestration lane lost proof point {expected!r}: {lane}")
        notes = " | ".join(lane.get("guardrail_notes") or [])
        if "read-only" not in notes or "does not approve, dispatch, or run risky actions" not in notes:
            raise SystemExit(f"orchestration lane should name read-only worker boundaries: {lane}")
        if "- Internal Orchestration: ready" not in result.output or "proofs: subagent_fleet_status exposes worker readiness" not in result.output:
            raise SystemExit("orchestration lane should be visible in cockpit text output")


def test_agent_landscape_lane_keeps_research_guidance_visible() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = _run_cockpit(runtime)
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes.get("agent_landscape")
        if lane is None:
            raise SystemExit("cockpit must expose the agent landscape guardrails lane")
        if lane["title"] != "Agent Landscape Guardrails":
            raise SystemExit(f"agent landscape lane title drifted: {lane}")
        if lane["risk"] != "READ_ONLY" or lane["approval_required"] is not False:
            raise SystemExit(f"agent landscape lane must stay read-only and auto-run safe: {lane}")
        expected_tools = ["work_queue", "capability_cockpit", "harness_doctrine"]
        if lane["registered_tools"] != expected_tools or lane["tool_coverage"] != "registered":
            raise SystemExit(f"agent landscape lane should use registered guidance tools: {lane}")
        expected_smokes = ["smoke_test_next_step", "smoke_test_capability_cockpit", "smoke_test_continuity"]
        if lane["registered_smoke_modules"] != expected_smokes or lane["smoke_coverage"] != "registered":
            raise SystemExit(f"agent landscape lane should be pinned by next-step/cockpit/continuity smokes: {lane}")
        if lane["example_command"] != "work queue" or lane["next_command"] != "work queue":
            raise SystemExit(f"agent landscape lane should point to the research-backed work queue: {lane}")
        proof_points = [str(point) for point in lane.get("proof_points") or []]
        expected_proofs = (
            "visible control plane, not companion personas",
            "traces, evals, guardrails, and human-in-the-loop checks",
            "broad privileges, skill supply-chain risk, and persistent prompt-injection",
            "Korean messaging reliability, phone control, morning brief, contact lookup, and operator-real evals",
        )
        for expected in expected_proofs:
            if not any(expected in proof for proof in proof_points):
                raise SystemExit(f"agent landscape lane lost proof point {expected!r}: {lane}")
        notes = [str(note) for note in lane.get("guardrail_notes") or []]
        if not any("avoid autonomy theater" in note and "companion identity sprawl" in note for note in notes):
            raise SystemExit(f"agent landscape lane should reject companion/persona drift: {lane}")
        if not any("does not fetch the web" in note and "authorize integration bridges" in note for note in notes):
            raise SystemExit(f"agent landscape lane should explain the read-only research boundary: {lane}")
        trust_labels = [
            str(check.get("label"))
            for check in (lane.get("trust_checklist") or [])
            if isinstance(check, dict)
        ]
        if "auto-run boundary" not in trust_labels or "explicit proof points" not in trust_labels:
            raise SystemExit(f"agent landscape lane should expose read-only trust and proof checks: {lane}")
        if "- Agent Landscape Guardrails: ready" not in result.output:
            raise SystemExit("agent landscape lane should be visible in cockpit text output")
        if "proofs: Zoey/OpenClaw/Hermes research keeps Jarvis pointed" not in result.output:
            raise SystemExit("agent landscape proof points should be visible in cockpit text output")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if "/\x55sers/" in blob or "/private/" in blob:
            raise SystemExit("agent landscape lane must not leak local paths")


def test_build_guardrails_lane_explains_live_proof_freeze() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = _run_cockpit(runtime)
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes.get("build_guardrails")
        if lane is None:
            raise SystemExit("cockpit must expose build guardrails during the live-proof freeze")
        if lane["title"] != "Build Guardrails":
            raise SystemExit(f"build guardrails title drifted: {lane}")
        if lane["risk"] != "READ_ONLY" or lane["approval_required"] is not False:
            raise SystemExit(f"build guardrails must stay read-only and auto-run safe: {lane}")
        if lane["status"] not in {"ready", "ok"}:
            raise SystemExit(f"build guardrails should not present the freeze as a broken lane: {lane}")
        if lane["tool_coverage"] != "registered":
            raise SystemExit(f"build guardrails must reuse registered diagnostics tools: {lane}")
        if lane["smoke_coverage"] != "registered":
            raise SystemExit(f"build guardrails must be pinned by registered aggregate smokes: {lane}")
        notes = [str(note) for note in lane.get("guardrail_notes") or []]
        if len(notes) != 4:
            raise SystemExit(f"build guardrails should carry exactly four bounded notes: {lane}")
        if not any("live-proof freeze active" in note for note in notes):
            raise SystemExit(f"build guardrails must explain the active live-proof freeze: {lane}")
        if not any(
            "pending live-proof matrix" in note
            and "Kakao" in note
            and "Instagram" in note
            and "Telegram" in note
            and "iMessage" in note
            and "FaceTime" in note
            for note in notes
        ):
            raise SystemExit(f"build guardrails must name the pending live-proof channel categories: {lane}")
        if not any("report live matrix results" in note and "pass/fail" in note for note in notes):
            raise SystemExit(f"build guardrails must explain how to report live matrix results: {lane}")
        if not any("last visible stage/error" in note and "message content" in note for note in notes):
            raise SystemExit(f"build guardrails must preserve safe live-result reporting boundaries: {lane}")
        if not any("local-safe" in note and "adjacent smokes" in note for note in notes):
            raise SystemExit(f"build guardrails must explain the autonomous safe lane: {lane}")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if "guardrail: live-proof freeze active" not in result.output:
            raise SystemExit("build guardrails notes should be visible in cockpit text output")
        if "pending live-proof matrix covers Kakao" not in result.output:
            raise SystemExit("build guardrails matrix note should be visible in cockpit text output")
        if "report live matrix results" not in result.output:
            raise SystemExit("build guardrails report-format note should be visible in cockpit text output")
        if "/\x55sers/" in blob or "/private/" in blob:
            raise SystemExit("build guardrails notes must not leak local paths")


def test_acceptance_harness_lane_exposes_live_check_without_overclaiming() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = _run_cockpit(runtime)
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes.get("acceptance_harness")
        if lane is None:
            raise SystemExit("cockpit must expose the acceptance harness lane")
        if lane["title"] != "Acceptance Harness":
            raise SystemExit(f"acceptance harness title drifted: {lane}")
        if lane["risk"] != "READ_ONLY" or lane["approval_required"] is not False:
            raise SystemExit(f"acceptance harness must stay read-only and auto-run safe: {lane}")
        if lane["status"] not in {"ready", "ok"}:
            raise SystemExit(f"acceptance harness should not present missing live proofs as a broken lane: {lane}")
        expected_tools = ["capability_cockpit", "readiness_report", "channel_health", "jarvis_doctor"]
        if lane["registered_tools"] != expected_tools or lane["tool_coverage"] != "registered":
            raise SystemExit(f"acceptance harness should reuse registered diagnostics tools: {lane}")
        expected_smokes = ["smoke_test_live_check", "smoke_test_capability_cockpit", "smoke_test_chat_timeout"]
        if lane["registered_smoke_modules"] != expected_smokes or lane["smoke_coverage"] != "registered":
            raise SystemExit(f"acceptance harness should be pinned by live_check/cockpit/handoff smokes: {lane}")
        if lane["example_command"] != "live proof status":
            raise SystemExit(f"acceptance harness should advertise the phone/control-plane proof status command: {lane}")
        proofs = [str(point) for point in lane.get("proof_points") or []]
        expected_proofs = (
            "live_check publishes one-screen acceptance rows",
            "acceptance coverage, acceptance gaps, acceptance next",
            "aggregate smoke",
            "aggregate smoke runs serialize through a bounded local suite lock",
            "operator evals",
            "jobs 7-day proof",
            "phone approvals",
            "personal proofs row audits bounded metadata only",
            "scheduled-job freshness row checks last_run_at drift",
            "daemon startup row checks launcher contracts",
            "error guidance row checks recovery wording",
            "daily brief degraded-section row requires recovery-hint proof",
            "acceptance gaps points to acceptance next for prioritized proof guidance",
            "acceptance next safe-report fields list lane, result, evidence, and stage",
            "chat path and mixed conversation rows separate model-backed readiness",
        )
        for expected in expected_proofs:
            if not any(expected in proof for proof in proofs):
                raise SystemExit(f"acceptance harness lost proof point {expected!r}: {lane}")
        notes = [str(note) for note in lane.get("guardrail_notes") or []]
        if not any("diagnostic only" in note and "operator present" in note for note in notes):
            raise SystemExit(f"acceptance harness must say live proof still needs operator present: {lane}")
        if not any("do not replace live delivery" in note or "do not replace live delivery" in note.lower() for note in notes):
            raise SystemExit(f"acceptance harness must avoid overclaiming readiness as delivery proof: {lane}")
        trust_labels = [
            str(check.get("label"))
            for check in (lane.get("trust_checklist") or [])
            if isinstance(check, dict)
        ]
        if "explicit proof points" not in trust_labels or "auto-run boundary" not in trust_labels:
            raise SystemExit(f"acceptance harness trust checklist should include proof and read-only boundary: {lane}")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        for expected in (
            "Acceptance Harness",
            "live_check publishes one-screen acceptance rows",
            "acceptance coverage, acceptance gaps, acceptance next",
            "aggregate smoke runs serialize through a bounded local suite lock",
            "operator evals",
            "jobs 7-day proof",
            "phone approvals",
            "scheduled-job freshness row checks last_run_at drift",
            "daily brief degraded-section row requires recovery-hint proof",
            "acceptance gaps points to acceptance next for prioritized proof guidance",
            "acceptance next safe-report fields list lane, result, evidence, and stage",
            "diagnostic only; run opt-in live_check with operator present",
            "readiness rows do not replace live delivery",
        ):
            if expected not in blob:
                raise SystemExit(f"acceptance harness should be visible in cockpit output/metadata: missed {expected!r}")
        if "/\x55sers/" in blob or "/private/" in blob:
            raise SystemExit("acceptance harness lane must not leak local paths")


def test_operator_workflow_eval_lane_is_visible_and_approval_aware() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = _run_cockpit(runtime)
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        writing = lanes.get("writing")
        if writing is None:
            raise SystemExit("cockpit must expose the writing/type-control lane")
        if writing["title"] != "Writing":
            raise SystemExit(f"writing lane title drifted: {writing}")
        if writing["risk"] != "HIGH_RISK" or writing["approval_required"] is not True:
            raise SystemExit(f"writing lane must surface computer-control approval risk: {writing}")
        expected_writing_tools = ["compose_and_write", "human_write", "paste_text"]
        if writing["registered_tools"] != expected_writing_tools:
            raise SystemExit(f"writing lane tool set drifted: {writing}")
        expected_writing_smokes = ["smoke_test_compose_connector", "smoke_test_writer_connector"]
        if writing["smoke_coverage"] != "registered" or writing["registered_smoke_modules"] != expected_writing_smokes:
            raise SystemExit(f"writing lane must be pinned by compose/writer smokes: {writing}")
        if writing["example_command"] != "write a paragraph about Jarvis and type it":
            raise SystemExit(f"writing lane should advertise the natural write-and-type command: {writing}")
        writing_trust_labels = [
            str(check.get("label"))
            for check in (writing.get("trust_checklist") or [])
            if isinstance(check, dict)
        ]
        if "approval boundary" not in writing_trust_labels or "explicit proof points" not in writing_trust_labels:
            raise SystemExit(f"writing lane should expose approval and proof checks: {writing}")
        writing_proofs = [str(point) for point in writing.get("proof_points") or []]
        expected_writing_proofs = (
            "compose_and_write stops before typing or pasting",
            "human_write and paste_text remain HIGH_RISK",
            "writer failures expose Accessibility recovery guidance",
        )
        for expected in expected_writing_proofs:
            if not any(expected in proof for proof in writing_proofs):
                raise SystemExit(f"writing lane lost proof point {expected!r}: {writing}")
        if not any("approval-gated" in str(note) and "writer/compose smokes" in str(note) for note in writing.get("guardrail_notes") or []):
            raise SystemExit(f"writing lane should explain mocked proof and approval gating: {writing}")
        if "- Writing: ready" not in result.output or "proofs: compose_and_write stops before typing" not in result.output:
            raise SystemExit("writing lane should be visible in cockpit text output")

        research = lanes.get("research_web")
        if research is None:
            raise SystemExit("cockpit must expose the Research & Web lane")
        if research["title"] != "Research & Web":
            raise SystemExit(f"Research & Web lane title drifted: {research}")
        if research["risk"] != "LOCAL_SAFE" or research["approval_required"] is not False:
            raise SystemExit(f"Research & Web lane should stay auto-run safe from the cockpit: {research}")
        expected_research_tools = [
            "web_lookup",
            "research",
            "fetch_page",
            "web_search",
            "recent_browser_pages",
            "summarize_page",
        ]
        if research["registered_tools"] != expected_research_tools:
            raise SystemExit(f"Research & Web lane tool set drifted: {research}")
        expected_research_smokes = ["smoke_test_research_connector", "smoke_test_next_layer", "smoke_test_live_check"]
        if research["smoke_coverage"] != "registered" or research["registered_smoke_modules"] != expected_research_smokes:
            raise SystemExit(f"Research & Web lane must be pinned by research/browser/live_check smokes: {research}")
        if research["example_command"] != "research Zoey OS" or research["next_command"] != "research Zoey OS":
            raise SystemExit(f"Research & Web lane should advertise source-backed research: {research}")
        research_trust_labels = [
            str(check.get("label"))
            for check in (research.get("trust_checklist") or [])
            if isinstance(check, dict)
        ]
        if "auto-run boundary" not in research_trust_labels or "explicit proof points" not in research_trust_labels:
            raise SystemExit(f"Research & Web lane should expose auto-run and proof checks: {research}")
        research_proofs = [str(point) for point in research.get("proof_points") or []]
        expected_research_proofs = (
            "research and web_lookup emit no-authority handoff packets",
            "live_check verifies research/web registration",
            "research failures expose DuckDuckGo Lite recovery guidance",
        )
        for expected in expected_research_proofs:
            if not any(expected in proof for proof in research_proofs):
                raise SystemExit(f"Research & Web lane lost proof point {expected!r}: {research}")
        research_notes = [str(note) for note in research.get("guardrail_notes") or []]
        if not any("external search/page services" in note and "explicitly run" in note for note in research_notes):
            raise SystemExit(f"Research & Web lane should explain when external services are used: {research}")
        if not any("does not fetch pages" in note and "queue approvals" in note for note in research_notes):
            raise SystemExit(f"Research & Web lane should explain the read-only cockpit boundary: {research}")
        if "- Research & Web: ready" not in result.output or "proofs: research and web_lookup emit no-authority" not in result.output:
            raise SystemExit("Research & Web lane should be visible in cockpit text output")

        approvals = lanes.get("approvals")
        if approvals is None:
            raise SystemExit("cockpit must expose the Approvals review lane")
        if approvals["title"] != "Approvals":
            raise SystemExit(f"Approvals lane title drifted: {approvals}")
        if approvals["risk"] != "READ_ONLY" or approvals["approval_required"] is not False:
            raise SystemExit(f"Approvals lane must stay review-only and non-authorizing: {approvals}")
        expected_approval_tools = [
            "list_pending_approvals",
            "inspect_pending_approval",
            "approval_readiness_packet",
            "approval_execution_packet",
            "approval_chain_proof",
            "approval_queue_summary",
            "review_pending_approvals",
            "approval_history",
        ]
        if approvals["registered_tools"] != expected_approval_tools:
            raise SystemExit(f"Approvals lane tool set drifted: {approvals}")
        expected_approval_smokes = ["smoke_test_approvals", "smoke_test_audit"]
        if approvals["smoke_coverage"] != "registered" or approvals["registered_smoke_modules"] != expected_approval_smokes:
            raise SystemExit(f"Approvals lane must be pinned by approval/audit smokes: {approvals}")
        if approvals["example_command"] != "pending approvals" or approvals["next_command"] != "pending approvals":
            raise SystemExit(f"Approvals lane should advertise pending-approval review: {approvals}")
        approval_trust_labels = [
            str(check.get("label"))
            for check in (approvals.get("trust_checklist") or [])
            if isinstance(check, dict)
        ]
        if "auto-run boundary" not in approval_trust_labels or "explicit proof points" not in approval_trust_labels:
            raise SystemExit(f"Approvals lane should expose read-only trust and proof checks: {approvals}")
        approval_proofs = [str(point) for point in approvals.get("proof_points") or []]
        expected_approval_proofs = (
            "approval review tools expose pending requests",
            "approve and dismiss tools are intentionally excluded",
        )
        for expected in expected_approval_proofs:
            if not any(expected in proof for proof in approval_proofs):
                raise SystemExit(f"Approvals lane lost proof point {expected!r}: {approvals}")
        approval_notes = [str(note) for note in approvals.get("guardrail_notes") or []]
        if not any("review only" in note and "explicit owner action" in note for note in approval_notes):
            raise SystemExit(f"Approvals lane should explain the review-only boundary: {approvals}")
        if "- Approvals: ready" not in result.output or "proofs: approval review tools expose pending requests" not in result.output:
            raise SystemExit("Approvals lane should be visible in cockpit text output")

        lane = lanes.get("operator_workflow_evals")
        if lane is None:
            raise SystemExit("cockpit must expose the operator workflow eval pack")
        if lane["title"] != "Operator Workflow Evals":
            raise SystemExit(f"operator workflow lane title drifted: {lane}")
        if lane["risk"] != "HIGH_RISK" or lane["approval_required"] is not True:
            raise SystemExit(f"operator workflow lane must surface send-side approval risk: {lane}")
        if lane["tool_coverage"] != "registered" or lane["tool_coverage_ready"] is not True:
            raise SystemExit(f"operator workflow lane should use registered workflow tools: {lane}")
        expected_tools = ["send_telegram", "run_job_now", "find_contact", "get_markets_overview", "channel_health"]
        if lane["registered_tools"] != expected_tools:
            raise SystemExit(f"operator workflow lane tool set drifted: {lane}")
        if lane["smoke_coverage"] != "registered" or lane["registered_smoke_modules"] != ["smoke_test_operator_evals"]:
            raise SystemExit(f"operator workflow eval pack must be pinned by the aggregate smoke suite: {lane}")
        if lane["example_command"] != "push today's brief to my phone":
            raise SystemExit(f"operator workflow lane should advertise a phone-first proof command: {lane}")
        if lane.get("trust_ready") is not True or lane.get("trust_non_authorizing") is not True:
            raise SystemExit(f"operator workflow lane trust checklist should be ready and non-authorizing: {lane}")
        trust_labels = [
            str(check.get("label"))
            for check in (lane.get("trust_checklist") or [])
            if isinstance(check, dict)
        ]
        if "approval boundary" not in trust_labels:
            raise SystemExit(f"operator workflow lane must make the approval boundary visible: {lane}")
        if "explicit proof points" not in trust_labels:
            raise SystemExit(f"operator workflow lane must make proof points part of earned trust: {lane}")
        if lane.get("trust_check_count") != 5 or lane.get("trust_ready_count") != 5:
            raise SystemExit(f"operator workflow lane should have five ready trust checks: {lane}")
        notes = [str(note) for note in lane.get("guardrail_notes") or []]
        if not any("Korean sends" in note and "markets" in note for note in notes):
            raise SystemExit(f"operator workflow lane should explain the real-task proof pack: {lane}")
        if not any("approval" in note and "receipt" in note for note in notes):
            raise SystemExit(f"operator workflow lane should explain side-effect gating: {lane}")
        proofs = [str(point) for point in lane.get("proof_points") or []]
        expected_proofs = (
            "Korean Telegram send stops at approval",
            "Korean contact lookup resolves",
            "Morning Brief can be pushed",
            "Phone control shortcuts expose",
            "Voice-note bilingual send commands",
            "Markets summaries include Korean output",
            "Clean-state cockpit has zero false attention lanes",
        )
        for expected in expected_proofs:
            if not any(expected in proof for proof in proofs):
                raise SystemExit(f"operator workflow lane lost proof point {expected!r}: {lane}")

        personal = lanes.get("personal_proofs")
        if personal is None:
            raise SystemExit("cockpit must expose the Personal Proofs lane")
        if personal["title"] != "Personal Proofs":
            raise SystemExit(f"Personal Proofs lane title drifted: {personal}")
        if personal["risk"] != "HIGH_RISK" or personal["approval_required"] is not True:
            raise SystemExit(f"Personal Proofs lane must surface write/send approval risk: {personal}")
        expected_personal_tools = [
            "list_events",
            "create_event",
            "update_event",
            "delete_event",
            "read_emails",
            "search_emails",
            "send_email",
            "set_reminder",
            "find_contact",
        ]
        if personal["registered_tools"] != expected_personal_tools:
            raise SystemExit(f"Personal Proofs lane tool set drifted: {personal}")
        expected_personal_smokes = [
            "smoke_test_live_check",
            "smoke_test_calendar_connector",
            "smoke_test_email_connector",
            "smoke_test_reminders",
            "smoke_test_contacts_connector",
        ]
        if personal["smoke_coverage"] != "registered" or personal["registered_smoke_modules"] != expected_personal_smokes:
            raise SystemExit(f"Personal Proofs lane must be pinned by live_check and personal connector smokes: {personal}")
        if personal["example_command"] != "personal proofs status" or personal["next_command"] != "personal proofs status":
            raise SystemExit(f"Personal Proofs lane should advertise proof-status review: {personal}")
        personal_proofs = [str(point) for point in personal.get("proof_points") or []]
        expected_personal_proofs = (
            "personal proofs are still live-acceptance evidence",
            "calendar create/update/delete and email send remain HIGH_RISK",
            "set_reminder proof uses Jarvis's local Telegram reminder path",
            "contact proof requires distinct successful lookups",
        )
        for expected in expected_personal_proofs:
            if not any(expected in proof for proof in personal_proofs):
                raise SystemExit(f"Personal Proofs lane lost proof point {expected!r}: {personal}")
        personal_notes = [str(note) for note in personal.get("guardrail_notes") or []]
        if not any("WS2" in note and "calendar writes" in note and "contact lookup" in note for note in personal_notes):
            raise SystemExit(f"Personal Proofs lane should name the WS2 proof scope: {personal}")
        if not any("does not access accounts or send anything" in note for note in personal_notes):
            raise SystemExit(f"Personal Proofs lane should explain the read-only proof boundary: {personal}")
        if "- Personal Proofs: ready" not in result.output or "proofs: personal proofs are still live-acceptance evidence" not in result.output:
            raise SystemExit("Personal Proofs lane should be visible in cockpit text output")

        learning = lanes.get("learning_loop")
        if learning is None:
            raise SystemExit("cockpit must expose the Learning Loop lane")
        if learning["title"] != "Learning Loop":
            raise SystemExit(f"Learning Loop lane title drifted: {learning}")
        if learning["risk"] != "LOCAL_SAFE" or learning["approval_required"] is not False:
            raise SystemExit(f"Learning Loop lane should stay local-safe and non-approval-gated: {learning}")
        expected_learning_tools = [
            "learning_review",
            "session_learning_preview",
            "after_action_learning_packet",
            "execution_learning_closure_packet",
            "feedback_actions",
            "failure_learning_cockpit",
            "queue_learning_tasks",
        ]
        if learning["registered_tools"] != expected_learning_tools:
            raise SystemExit(f"Learning Loop lane tool set drifted: {learning}")
        if learning["tool_coverage"] != "registered" or learning["tool_coverage_ready"] is not True:
            raise SystemExit(f"Learning Loop lane should use registered learning tools: {learning}")
        if learning["smoke_coverage"] != "registered" or learning["registered_smoke_modules"] != ["smoke_test_learning_review"]:
            raise SystemExit(f"Learning Loop lane must be pinned by learning-review smoke: {learning}")
        if learning["example_command"] != "learning review" or learning["next_command"] != "learning review":
            raise SystemExit(f"Learning Loop lane should point to the read-only review path: {learning}")
        learning_proofs = [str(point) for point in learning.get("proof_points") or []]
        expected_learning_proofs = (
            "learning_review surfaces feedback",
            "bind approved runs to verification, audit, recovery, and learning evidence",
            "repeated failures into reviewable fixes",
            "queue_learning_tasks is LOCAL_SAFE",
        )
        for expected in expected_learning_proofs:
            if not any(expected in proof for proof in learning_proofs):
                raise SystemExit(f"Learning Loop lane lost proof point {expected!r}: {learning}")
        learning_notes = [str(note) for note in learning.get("guardrail_notes") or []]
        if not any("visible review loop" in note and "not autonomous self-modification" in note for note in learning_notes):
            raise SystemExit(f"Learning Loop lane should reject autonomous self-modification: {learning}")
        if not any("non-authorizing" in note and "approvals" in note for note in learning_notes):
            raise SystemExit(f"Learning Loop lane should explain authority boundaries: {learning}")
        if "- Learning Loop: ready" not in result.output or "proofs: learning_review surfaces feedback" not in result.output:
            raise SystemExit("Learning Loop lane should be visible in cockpit text output")

        acceptance_proof_count = 11
        approvals_proof_count = len(expected_approval_proofs)
        writing_proof_count = len(expected_writing_proofs)
        personal_proof_count = len(expected_personal_proofs)
        research_proof_count = len(expected_research_proofs)
        learning_proof_count = 4
        orchestration_proof_count = 3
        agent_landscape_proof_count = 4
        expected_total_proofs = (
            len(expected_proofs)
            + acceptance_proof_count
            + approvals_proof_count
            + writing_proof_count
            + personal_proof_count
            + research_proof_count
            + learning_proof_count
            + orchestration_proof_count
            + agent_landscape_proof_count
        )
        if result.metadata.get("proof_lane_count") != 9 or result.metadata.get("proof_point_count") != expected_total_proofs:
            raise SystemExit(f"cockpit should aggregate operator workflow proof coverage: {result.metadata}")
        proof_summary = result.metadata.get("proof_summary") or []
        if len(proof_summary) != 4:
            raise SystemExit(f"cockpit should expose a compact proof summary: {result.metadata}")
        hidden_count = result.metadata.get("proof_summary_hidden_lane_count")
        if hidden_count != result.metadata.get("proof_lane_count") - len(proof_summary):
            raise SystemExit(f"cockpit should count proof lanes hidden from the compact summary: {result.metadata}")
        hidden_lanes = result.metadata.get("proof_summary_hidden_lanes") or []
        hidden_titles = {entry.get("lane_title") for entry in hidden_lanes if isinstance(entry, dict)}
        for title in ("Learning Loop", "Internal Orchestration", "Agent Landscape Guardrails"):
            if title not in hidden_titles:
                raise SystemExit(f"compact proof summary should advertise hidden {title} lane: {hidden_lanes}")
        writing_summary = next((entry for entry in proof_summary if entry.get("lane_key") == "writing"), {})
        if writing_summary.get("proof_count") != writing_proof_count:
            raise SystemExit(f"writing proof summary drifted: {proof_summary}")
        if not any("compose_and_write stops before typing" in proof for proof in writing_summary.get("sample_proofs") or []):
            raise SystemExit(f"writing proof summary should sample the write-gating proof: {writing_summary}")
        approvals_summary = next((entry for entry in proof_summary if entry.get("lane_key") == "approvals"), {})
        if approvals_summary.get("proof_count") != approvals_proof_count:
            raise SystemExit(f"Approvals proof summary drifted: {proof_summary}")
        if not any("approval review tools expose pending requests" in proof for proof in approvals_summary.get("sample_proofs") or []):
            raise SystemExit(f"Approvals proof summary should sample the review-only proof: {approvals_summary}")
        summary = next((entry for entry in proof_summary if entry.get("lane_key") == "operator_workflow_evals"), {})
        if summary.get("lane_key") != "operator_workflow_evals" or summary.get("proof_count") != len(expected_proofs):
            raise SystemExit(f"operator workflow proof summary drifted: {summary}")
        if not any("Korean Telegram send stops at approval" in proof for proof in summary.get("sample_proofs") or []):
            raise SystemExit(f"operator workflow proof summary should sample the Korean send proof: {summary}")
        acceptance_summary = next((entry for entry in proof_summary if entry.get("lane_key") == "acceptance_harness"), {})
        if acceptance_summary.get("proof_count") != acceptance_proof_count:
            raise SystemExit(f"acceptance harness proof summary drifted: {proof_summary}")
        if "Operator Workflow Evals" not in result.output or "smoke: registered" not in result.output:
            raise SystemExit("operator workflow lane should be visible in cockpit text output")
        if f"Proof coverage: 9 lane(s), {expected_total_proofs} proof point(s)" not in result.output:
            raise SystemExit("cockpit text output should summarize aggregate proof coverage")
        if "Proof summary has 5 more lane(s)" not in result.output or "Internal Orchestration" not in result.output:
            raise SystemExit("cockpit text output should disclose hidden proof lanes")
        if "proofs: Korean Telegram send stops at approval" not in result.output:
            raise SystemExit("operator workflow proof points should be visible in cockpit text output")


def test_malformed_lane_notes_and_proofs_stay_bounded() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": "hostile_text_metadata",
                "title": "Hostile Text Metadata",
                "tools": ["good_fixture"],
                "example": "good fixture",
                "smoke": ["smoke_test_capability_cockpit"],
                "notes": _HostileTruthiness(),
                "proof_points": [_HostileTruthiness(), _ExplodingString(), "safe proof"],
            },
        ]
        tools = [Tool("good_fixture", "fixture", RiskLevel.READ_ONLY, _noop_tool)]
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), lambda: tools)
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must survive malformed note/proof metadata")
        lane = result.metadata["capability_lanes"][0]
        if lane["guardrail_notes"] != ["<local-path>"]:
            raise SystemExit(f"malformed notes should be bounded and redacted: {lane}")
        if lane["proof_points"] != ["<local-path>", "<unreadable>", "safe proof"]:
            raise SystemExit(f"malformed proof points should be bounded and preserved: {lane}")
        if result.metadata.get("proof_lane_count") != 1 or result.metadata.get("proof_point_count") != 3:
            raise SystemExit(f"bounded proof metadata should still count as proof coverage: {result.metadata}")
        summary = result.metadata.get("proof_summary") or []
        if not summary or summary[0].get("sample_proofs") != ["<local-path>", "<unreadable>"]:
            raise SystemExit(f"proof summary should sample bounded proof metadata: {summary}")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if _LEAK_MARKER in blob or "/\x55sers/" in blob:
            raise SystemExit(f"malformed note/proof metadata leaked raw local path text: {blob}")
        if "guardrail: <local-path>" not in result.output or "proofs: <local-path>; <unreadable>; safe proof" not in result.output:
            raise SystemExit("bounded note/proof metadata should remain visible for cockpit diagnosis")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_side_effect_and_personal_lanes_name_connector_smokes() -> None:
    with TemporaryDirectory(prefix="jarvis-cockpit-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = _run_cockpit(runtime)
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        expected = {
            "messaging": [
                "smoke_test_telegram_control",
                "smoke_test_social_connectors",
                "smoke_test_imessage_connector",
                "smoke_test_channel_health",
            ],
            "calls": ["smoke_test_call_connector", "smoke_test_channel_health"],
            "calendar_email": ["smoke_test_calendar_connector", "smoke_test_email_connector"],
            "personal_proofs": [
                "smoke_test_live_check",
                "smoke_test_calendar_connector",
                "smoke_test_email_connector",
                "smoke_test_reminders",
                "smoke_test_contacts_connector",
            ],
            "contacts": ["smoke_test_people", "smoke_test_contacts_connector", "smoke_test_contacts_fuzzy"],
            "markets": ["smoke_test_markets_connector", "smoke_test_currency_connector", "smoke_test_weather_connector"],
            "research_web": ["smoke_test_research_connector", "smoke_test_next_layer", "smoke_test_live_check"],
            "memory": ["smoke_test_core", "smoke_test_memory_stats"],
            "learning_loop": ["smoke_test_learning_review"],
            "diagnostics": ["smoke_test_doctor", "smoke_test_audit", "smoke_test_channel_health"],
        }
        for key, expected_smokes in expected.items():
            lane = lanes.get(key)
            if lane is None:
                raise SystemExit(f"cockpit missed {key!r} lane")
            if lane["smoke_coverage"] != "registered" or lane["registered_smoke_modules"] != expected_smokes:
                raise SystemExit(f"{key} lane must be pinned by exact connector smokes: {lane}")
            if lane["missing_smoke_modules"]:
                raise SystemExit(f"{key} lane should not have missing smoke modules: {lane}")
            if lane["status"] not in {"ready", "ok"}:
                raise SystemExit(f"{key} lane should stay healthy with exact connector smokes: {lane}")


def test_cockpit_max_risk_reports_personal_and_external_levels() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": "personal_only",
                "title": "Personal Only",
                "tools": ["read_private_fixture"],
                "example": "read private fixture",
                "smoke": [],
            },
            {
                "key": "external_only",
                "title": "External Only",
                "tools": ["send_external_fixture"],
                "example": "send external fixture",
                "smoke": [],
            },
        ]
        tools = [
            Tool("read_private_fixture", "fixture", RiskLevel.PERSONAL_DATA, _noop_tool),
            Tool("send_external_fixture", "fixture", RiskLevel.EXTERNAL_SIDE_EFFECT, _noop_tool),
        ]
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), lambda: tools)
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must succeed")
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        if lanes["personal_only"]["risk"] != "PERSONAL_DATA" or lanes["personal_only"]["approval_required"] is not True:
            raise SystemExit(f"PERSONAL_DATA lane risk was under-reported: {lanes['personal_only']}")
        if lanes["external_only"]["risk"] != "EXTERNAL_SIDE_EFFECT" or lanes["external_only"]["approval_required"] is not True:
            raise SystemExit(f"EXTERNAL_SIDE_EFFECT lane risk was under-reported: {lanes['external_only']}")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_partial_or_missing_tool_coverage_marks_lane_attention() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": "partial_tools",
                "title": "Partial Tools",
                "tools": ["present_fixture", "missing_fixture"],
                "example": "partial fixture",
                "smoke": ["smoke_test_capability_cockpit"],
            },
            {
                "key": "missing_tools",
                "title": "Missing Tools",
                "tools": ["missing_only_fixture"],
                "example": "missing fixture",
                "smoke": ["smoke_test_capability_cockpit"],
            },
        ]
        tools = [Tool("present_fixture", "fixture", RiskLevel.READ_ONLY, _noop_tool)]
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), lambda: tools)
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must succeed")
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        partial = lanes["partial_tools"]
        missing = lanes["missing_tools"]
        if partial["status"] != "attention" or "partial_tools" not in result.metadata["attention_lanes"]:
            raise SystemExit(f"partial tool coverage should put lane in attention: {partial}")
        if partial["tool_coverage"] != "partial" or partial["tool_coverage_ready"] is not False:
            raise SystemExit(f"partial tool coverage should be explicit: {partial}")
        if partial["registered_tools"] != ["present_fixture"] or partial["missing_tools"] != ["missing_fixture"]:
            raise SystemExit(f"partial lane should name present and missing tools: {partial}")
        if partial["attention_reasons"] != ["missing tools: missing_fixture"]:
            raise SystemExit(f"partial lane should explain missing tools: {partial}")
        if partial["next_command"] != "capability cockpit" or partial["next_command_kind"] != "diagnostic":
            raise SystemExit(f"partial lane should stage cockpit diagnostic: {partial}")
        if "attention: missing tools: missing_fixture" not in result.output:
            raise SystemExit("partial lane missing visible attention reason")
        if missing["status"] != "missing" or "missing_tools" not in result.metadata["attention_lanes"]:
            raise SystemExit(f"fully missing tools should keep missing status: {missing}")
        if missing["tool_coverage"] != "missing" or missing["tool_coverage_ready"] is not False:
            raise SystemExit(f"missing tool coverage should be explicit: {missing}")
        if missing["registered_tools"] or missing["missing_tools"] != ["missing_only_fixture"]:
            raise SystemExit(f"missing lane should name missing tools only: {missing}")
        if missing["attention_reasons"] != ["all tools missing"]:
            raise SystemExit(f"missing lane should explain missing tools: {missing}")
        if missing["next_command"] != "capability cockpit" or missing["next_command_kind"] != "diagnostic":
            raise SystemExit(f"missing lane should stage cockpit diagnostic: {missing}")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_missing_or_empty_smoke_coverage_marks_lane_attention() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": "missing_smoke",
                "title": "Missing Smoke",
                "tools": ["covered_fixture"],
                "example": "covered fixture",
                "smoke": ["smoke_test_this_module_does_not_exist"],
            },
            {
                "key": "empty_smoke",
                "title": "Empty Smoke",
                "tools": ["uncovered_fixture"],
                "example": "uncovered fixture",
                "smoke": [],
            },
        ]
        tools = [
            Tool("covered_fixture", "fixture", RiskLevel.READ_ONLY, _noop_tool),
            Tool("uncovered_fixture", "fixture", RiskLevel.READ_ONLY, _noop_tool),
        ]
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), lambda: tools)
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must succeed")
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        missing = lanes["missing_smoke"]
        empty = lanes["empty_smoke"]
        if missing["smoke_coverage"] != "unregistered" or missing["smoke_coverage_ready"] is not False:
            raise SystemExit(f"missing smoke module should be unregistered: {missing}")
        if missing["status"] != "attention" or "missing_smoke" not in result.metadata["attention_lanes"]:
            raise SystemExit(f"missing smoke module should put lane in attention: {missing}")
        if "smoke_test_this_module_does_not_exist" not in missing["missing_smoke_modules"]:
            raise SystemExit(f"missing smoke module should be listed: {missing}")
        if missing["attention_reasons"] != ["unregistered smoke: smoke_test_this_module_does_not_exist"]:
            raise SystemExit(f"missing smoke module should explain attention: {missing}")
        if missing["next_command"] != "capability cockpit" or missing["next_command_kind"] != "diagnostic":
            raise SystemExit(f"missing smoke lane should stage cockpit diagnostic: {missing}")
        if empty["smoke_coverage"] != "uncovered" or empty["smoke_coverage_ready"] is not False:
            raise SystemExit(f"empty smoke list should be uncovered: {empty}")
        if empty["status"] != "attention" or "empty_smoke" not in result.metadata["attention_lanes"]:
            raise SystemExit(f"empty smoke list should put lane in attention: {empty}")
        if empty["attention_reasons"] != ["no aggregate smoke coverage"]:
            raise SystemExit(f"empty smoke list should explain attention: {empty}")
        if empty["next_command"] != "capability cockpit" or empty["next_command_kind"] != "diagnostic":
            raise SystemExit(f"empty smoke lane should stage cockpit diagnostic: {empty}")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_malformed_smoke_metadata_is_hidden_without_blinding_valid_smoke() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": "hostile_smoke_metadata",
                "title": "Hostile Smoke Metadata",
                "tools": ["good_fixture"],
                "example": "good fixture",
                "smoke": [_HostileTruthiness(), _ExplodingString(), "smoke_test_capability_cockpit"],
            },
        ]
        tools = [Tool("good_fixture", "fixture", RiskLevel.READ_ONLY, _noop_tool)]
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), lambda: tools)
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must survive malformed smoke metadata")
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes["hostile_smoke_metadata"]
        if lane["smoke_coverage"] != "registered" or lane["smoke_coverage_ready"] is not True:
            raise SystemExit(f"valid smoke module should still register: {lane}")
        if lane["registered_smoke_modules"] != ["smoke_test_capability_cockpit"]:
            raise SystemExit(f"valid smoke module should remain visible: {lane}")
        if lane["missing_smoke_modules"] or lane["smoke_modules"] != ["smoke_test_capability_cockpit"]:
            raise SystemExit(f"malformed smoke metadata should not masquerade as modules: {lane}")
        if lane["hidden_smoke_modules"] != 2:
            raise SystemExit(f"malformed smoke metadata should be counted: {lane}")
        if lane["status"] != "attention" or lane["attention_reasons"] != ["unreadable smoke metadata: 2"]:
            raise SystemExit(f"malformed smoke metadata should be a clear attention reason: {lane}")
        coverage = result.metadata.get("coverage_summary") or {}
        if coverage.get("hidden_smoke_metadata_count") != 2:
            raise SystemExit(f"coverage summary should count hidden smoke metadata: {coverage}")
        if "hidden smoke metadata 2" not in str(coverage.get("summary") or ""):
            raise SystemExit(f"coverage summary text should name hidden smoke metadata: {coverage}")
        if f"Coverage summary: {coverage['summary']}" not in result.output:
            raise SystemExit(f"cockpit output should render hidden smoke metadata in summary: {result.output}")
        if result.metadata.get("coverage_ready") is not False or coverage.get("coverage_ready") is not False:
            raise SystemExit(f"hidden smoke metadata should fail the top-level coverage verdict: {result.metadata}")
        if result.metadata.get("smoke_coverage_ready_lane_count") != 1 or coverage.get("smoke_ready_count") != 1:
            raise SystemExit(f"valid smoke coverage count should remain intact: {result.metadata}")
        if lane["next_command"] != "capability cockpit" or lane["next_command_kind"] != "diagnostic":
            raise SystemExit(f"malformed smoke metadata should stage cockpit diagnostic: {lane}")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if _LEAK_MARKER in blob or "/\x55sers/" in blob:
            raise SystemExit(f"malformed smoke metadata leaked raw local path text: {blob}")
        if "<local-path>" in json.dumps(lane["smoke_modules"], ensure_ascii=False):
            raise SystemExit(f"redaction marker should not appear as a smoke module: {lane}")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_malformed_tool_metadata_is_hidden_without_blinding_valid_tool() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": "hostile_tool_metadata",
                "title": "Hostile Tool Metadata",
                "tools": [_HostileTruthiness(), _ExplodingString(), "good_fixture"],
                "example": "good fixture",
                "smoke": ["smoke_test_capability_cockpit"],
            },
        ]
        tools = [Tool("good_fixture", "fixture", RiskLevel.READ_ONLY, _noop_tool)]
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), lambda: tools)
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must survive malformed tool metadata")
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes["hostile_tool_metadata"]
        if lane["tool_coverage"] != "registered" or lane["tool_coverage_ready"] is not True:
            raise SystemExit(f"valid tool should still register: {lane}")
        if lane["registered_tools"] != ["good_fixture"] or lane["missing_tools"]:
            raise SystemExit(f"malformed tool metadata should not masquerade as tool names: {lane}")
        if lane.get("hidden_tool_metadata") != 2:
            raise SystemExit(f"malformed tool metadata should be counted: {lane}")
        if lane["status"] != "attention" or lane["attention_reasons"] != ["unreadable tool metadata: 2"]:
            raise SystemExit(f"malformed tool metadata should be a clear attention reason: {lane}")
        coverage = result.metadata.get("coverage_summary") or {}
        if coverage.get("hidden_tool_metadata_count") != 2:
            raise SystemExit(f"coverage summary should count hidden tool metadata: {coverage}")
        if "hidden tool metadata 2" not in str(coverage.get("summary") or ""):
            raise SystemExit(f"coverage summary text should name hidden tool metadata: {coverage}")
        if result.metadata.get("coverage_ready") is not False or coverage.get("coverage_ready") is not False:
            raise SystemExit(f"hidden tool metadata should fail the top-level coverage verdict: {result.metadata}")
        if result.metadata.get("tool_coverage_ready_lane_count") != 1 or coverage.get("tool_ready_count") != 1:
            raise SystemExit(f"valid tool coverage count should remain intact: {result.metadata}")
        if f"Coverage summary: {coverage['summary']}" not in result.output:
            raise SystemExit(f"cockpit output should render hidden tool metadata in summary: {result.output}")
        if "attention: unreadable tool metadata: 2" not in result.output:
            raise SystemExit("cockpit output should render hidden tool metadata attention reason")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if _LEAK_MARKER in blob or "/\x55sers/" in blob:
            raise SystemExit(f"malformed tool metadata leaked raw local path text: {blob}")
        if "<local-path>" in json.dumps(lane["registered_tools"] + lane["missing_tools"], ensure_ascii=False):
            raise SystemExit(f"redaction marker should not appear as a tool name: {lane}")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_malformed_registry_tool_does_not_blind_good_tools() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": "good_tool_lane",
                "title": "Good Tool Lane",
                "tools": ["good_fixture"],
                "example": "good fixture",
                "smoke": ["smoke_test_capability_cockpit"],
            },
        ]
        tools = [
            _ExplodingToolName(),
            Tool("good_fixture", "fixture", RiskLevel.READ_ONLY, _noop_tool),
        ]
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), lambda: tools)
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must survive one malformed tool")
        if result.metadata["unreadable_registry_tools"] != 1:
            raise SystemExit(f"malformed registry tool should be counted once: {result.metadata}")
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes["good_tool_lane"]
        if lane["tool_coverage"] != "registered" or lane["registered_tools"] != ["good_fixture"]:
            raise SystemExit(f"good tool should remain visible despite malformed peer: {lane}")
        if lane["status"] not in {"ready", "ok"} or lane["attention_reasons"]:
            raise SystemExit(f"malformed peer should not put the good lane in attention: {lane}")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if _LEAK_MARKER in blob:
            raise SystemExit("malformed registry tool leaked seeded local path text")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_bound_registry_list_failure_preserves_good_tools() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": "bound_registry_good_tool",
                "title": "Bound Registry Good Tool",
                "tools": ["good_fixture"],
                "example": "good fixture",
                "smoke": ["smoke_test_capability_cockpit"],
            },
        ]
        registry = _SortingRegistry(
            [
                _ExplodingToolName(),
                Tool("good_fixture", "fixture", RiskLevel.READ_ONLY, _noop_tool),
            ]
        )
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), registry.list)
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must survive a bound registry.list failure")
        if result.metadata["unreadable_registry_tools"] != 1:
            raise SystemExit(f"bound registry.list failure should count one hidden tool: {result.metadata}")
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes["bound_registry_good_tool"]
        if lane["tool_coverage"] != "registered" or lane["registered_tools"] != ["good_fixture"]:
            raise SystemExit(f"good tool should remain visible after registry.list recovery: {lane}")
        if lane["status"] not in {"ready", "ok"} or lane["attention_reasons"]:
            raise SystemExit(f"registry recovery should not put the good lane in attention: {lane}")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if _LEAK_MARKER in blob:
            raise SystemExit("bound registry.list recovery leaked seeded local path text")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_unreadable_tool_risk_fails_lane_closed() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": "bad_risk_lane",
                "title": "Bad Risk Lane",
                "tools": ["bad_risk_fixture"],
                "example": "bad risk fixture",
                "smoke": ["smoke_test_capability_cockpit"],
            },
        ]
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), lambda: [_BadRiskTool()])
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must survive unreadable tool risk")
        lanes = {lane["key"]: lane for lane in result.metadata["capability_lanes"]}
        lane = lanes["bad_risk_lane"]
        if lane["risk"] != "UNKNOWN" or lane["approval_required"] is not True:
            raise SystemExit(f"unreadable risk must fail closed to approval-required UNKNOWN: {lane}")
        if lane["status"] != "attention" or result.metadata["attention_lanes"] != ["bad_risk_lane"]:
            raise SystemExit(f"unreadable risk should put lane in attention: {lane}")
        if lane["attention_reasons"] != ["unreadable tool risk: bad_risk_fixture"]:
            raise SystemExit(f"unreadable risk should be visible as the attention reason: {lane}")
        if lane["next_command"] != "capability cockpit" or lane["next_command_kind"] != "diagnostic":
            raise SystemExit(f"unreadable risk lane should stage cockpit diagnostic: {lane}")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if _LEAK_MARKER in blob:
            raise SystemExit("unreadable risk exception leaked seeded local path text")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_malformed_lane_spec_text_falls_back_without_leaking() -> None:
    original_lanes = list(cockpit_tools.CAPABILITY_LANES)
    try:
        cockpit_tools.CAPABILITY_LANES[:] = [
            {
                "key": _HostileTruthiness(),
                "title": _HostileTruthiness(),
                "tools": ["missing_fixture"],
                "example": _HostileTruthiness(),
                "diagnostic": _HostileTruthiness(),
                "smoke": [],
            },
        ]
        handler = cockpit_tools.make_capability_cockpit_tool(_EmptyStore(), lambda: [])
        result = handler({})
        if not result.ok:
            raise SystemExit("fixture cockpit must survive malformed lane spec text")
        lanes = result.metadata["capability_lanes"]
        if len(lanes) != 1:
            raise SystemExit(f"malformed lane fixture should produce one lane: {lanes}")
        lane = lanes[0]
        if lane["key"] != "<local-path>" or lane["title"] != "<local-path>":
            raise SystemExit(f"malformed lane labels should be redacted, not dropped: {lane}")
        if lane["status"] != "missing" or lane["next_command"] != "capability cockpit":
            raise SystemExit(f"malformed diagnostic should fall back to safe cockpit command: {lane}")
        if lane["example_command"] != "<local-path>" or result.metadata["attention_lanes"] != ["<local-path>"]:
            raise SystemExit(f"malformed lane metadata should stay bounded and redacted: {result.metadata}")
        blob = result.output + json.dumps(result.metadata, ensure_ascii=False)
        if _LEAK_MARKER in blob or "/\x55sers/" in blob:
            raise SystemExit(f"malformed lane spec leaked raw local path text: {blob}")
        if "<local-path>" not in blob:
            raise SystemExit(f"malformed lane spec should preserve redaction markers for diagnostics: {blob}")
    finally:
        cockpit_tools.CAPABILITY_LANES[:] = original_lanes


def test_next_command_queue_handles_hostile_lane_truthiness() -> None:
    queue = cockpit_tools._next_command_queue(
        [
            {
                "key": _HostileTruthiness(),
                "title": _HostileTruthiness(),
                "status": _HostileTruthiness(),
                "next_command": "channel health",
                "next_command_kind": "diagnostic",
                "approval_required": _HostileTruthiness(),
                "last_failure": _HostileTruthiness(),
                "last_failure_kind": _HostileTruthiness(),
                "last_approval_hold": _HostileTruthiness(),
                "last_approval_id": _HostileTruthiness(),
                "approval_next_commands": [_HostileTruthiness(), "approval packet latest"],
            }
        ]
    )
    if len(queue) != 1:
        raise SystemExit(f"hostile lane should still produce one bounded queue entry: {queue}")
    entry = queue[0]
    if entry["command"] != "channel health" or entry["kind"] != "diagnostic":
        raise SystemExit(f"hostile lane should preserve the safe diagnostic command: {entry}")
    if entry["approval_required"] is not False:
        raise SystemExit(f"hostile approval_required must fail closed to false in queue metadata: {entry}")
    blob = json.dumps(queue, ensure_ascii=False)
    if _LEAK_MARKER in blob or "/\x55sers/" in blob:
        raise SystemExit(f"hostile lane queue leaked raw local path text: {blob}")
    if blob.count("<local-path>") < 6:
        raise SystemExit(f"hostile lane queue should redact stringable local-path markers: {blob}")

    counts = cockpit_tools._lane_status_counts([{"status": _HostileTruthiness()}])
    if counts != {"<local-path>": 1}:
        raise SystemExit(f"hostile status counts should stay bounded: {counts}")

    summary = cockpit_tools._proof_summary(
        [
            {
                "key": _HostileTruthiness(),
                "title": _HostileTruthiness(),
                "proof_points": [_HostileTruthiness(), "safe proof"],
            }
        ]
    )
    summary_blob = json.dumps(summary, ensure_ascii=False)
    if _LEAK_MARKER in summary_blob or "/\x55sers/" in summary_blob:
        raise SystemExit(f"hostile proof summary leaked raw local path text: {summary_blob}")
    if not summary or summary[0]["proof_count"] != 2 or "<local-path>" not in summary_blob:
        raise SystemExit(f"hostile proof summary should preserve bounded proof metadata: {summary}")

    unreadable_queue = cockpit_tools._next_command_queue(
        [
            {
                "key": _ExplodingString(),
                "title": _ExplodingString(),
                "status": _ExplodingString(),
                "next_command": "capability cockpit",
                "next_command_kind": _ExplodingString(),
                "last_failure": _ExplodingString(),
                "last_failure_kind": _ExplodingString(),
                "last_approval_hold": _ExplodingString(),
                "approval_next_commands": [_ExplodingString()],
            }
        ]
    )
    unreadable_blob = json.dumps(unreadable_queue, ensure_ascii=False)
    if _LEAK_MARKER in unreadable_blob or "/\x55sers/" in unreadable_blob:
        raise SystemExit(f"exploding string queue leaked raw local path text: {unreadable_blob}")
    if unreadable_blob.count("<unreadable>") < 6:
        raise SystemExit(f"exploding string queue should preserve unreadable markers: {unreadable_blob}")

    path_command_queue = cockpit_tools._next_command_queue(
        [
            {
                "key": "bad_command",
                "title": "Bad Command",
                "status": "attention",
                "next_command": _HostileTruthiness(),
                "next_command_kind": "diagnostic",
            },
            {
                "key": "safe_command",
                "title": "Safe Command",
                "status": "attention",
                "next_command": "capability cockpit",
                "next_command_kind": "diagnostic",
            },
        ]
    )
    if [entry["command"] for entry in path_command_queue] != ["capability cockpit"]:
        raise SystemExit(f"path-shaped next commands should be hidden from the global queue: {path_command_queue}")
    path_command_blob = json.dumps(path_command_queue, ensure_ascii=False)
    if _LEAK_MARKER in path_command_blob or "/\x55sers/" in path_command_blob or "<local-path>" in path_command_blob:
        raise SystemExit(f"path-shaped next command leaked into queue metadata: {path_command_blob}")


def main() -> None:
    test_cockpit_reports_all_lanes_read_only()
    test_approval_gate_stop_reads_as_awaiting_approval_not_failure()
    test_approval_gate_aliases_read_as_awaiting_approval()
    test_terminal_or_unbound_approval_holds_are_not_current_review_state()
    test_real_failure_reads_as_attention_with_last_failure()
    test_string_false_audit_ok_does_not_become_success()
    test_malformed_audit_rows_hidden_and_paths_scrubbed()
    test_dashboard_snapshot_carries_cockpit_lanes()
    test_internal_orchestration_lane_is_visible_but_read_only()
    test_agent_landscape_lane_keeps_research_guidance_visible()
    test_build_guardrails_lane_explains_live_proof_freeze()
    test_acceptance_harness_lane_exposes_live_check_without_overclaiming()
    test_operator_workflow_eval_lane_is_visible_and_approval_aware()
    test_side_effect_and_personal_lanes_name_connector_smokes()
    test_cockpit_max_risk_reports_personal_and_external_levels()
    test_partial_or_missing_tool_coverage_marks_lane_attention()
    test_missing_or_empty_smoke_coverage_marks_lane_attention()
    test_malformed_smoke_metadata_is_hidden_without_blinding_valid_smoke()
    test_malformed_tool_metadata_is_hidden_without_blinding_valid_tool()
    test_malformed_lane_notes_and_proofs_stay_bounded()
    test_malformed_registry_tool_does_not_blind_good_tools()
    test_bound_registry_list_failure_preserves_good_tools()
    test_unreadable_tool_risk_fails_lane_closed()
    test_malformed_lane_spec_text_falls_back_without_leaking()
    test_next_command_queue_handles_hostile_lane_truthiness()
    print("Capability cockpit smoke passed")


if __name__ == "__main__":
    main()
