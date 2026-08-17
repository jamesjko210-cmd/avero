"""Eval pack built from the operator's real daily workflows (2026-07-04 direction:
"make everyday workflows reliable enough to trust", measured with evals).

Every eval drives `runtime.handle()` — the exact path the dashboard composer,
Telegram bot, and voice input use — in an isolated temp runtime. No network,
no real sends, no real contacts: a queued approval is the SUCCESS state for
send flows, because stopping at the gate is the product working.

Covered lanes:
1. Korean telegram send lifecycle (Hangul integrity + approval gate + dismiss)
2. Korean contact lookup (가상연락처이 resolves with relation)
3. Channel health phrase (the phone "what's my channel state" flow)
4. "What broke" diagnostics (failed run is visible in recent tool runs)
5. Morning brief pushed on demand to the owner phone channel
6. Phone control shortcuts (status / voice / voice setup / voice stop / capabilities / agent research / completion status / safety / privacy / risk / what broke / brief status / channels / readiness / doctor / cockpit / summary / next action / attention / lane status / guardrails)
7. Telegram voice note -> bilingual Korean/English send command -> approval gate
8. Scheduler visibility (list scheduled jobs)
9. Memory write (remember ...)
10. Currency conversion (mocked FX seam — routing + formatting)
11. Weather lookup (mocked wttr seam)
12. Markets summary in English and Korean (mocked crypto/stock seams — the research "markets summary")
13. Korean task/reminder reads (command-first, read-only, no model fallback)

For the utility lanes the external fetch is mocked at the connector's
module seam and RESTORED in a finally block, so no network is hit and no global
state leaks into the rest of the smoke suite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from jarvis_v2.automations import scheduler as scheduler_module
from jarvis_v2.automations import telegram_control as telegram_control_module
from jarvis_v2.automations.scheduler import MORNING_BRIEF_JOB_NAME
from jarvis_v2.scripts.test_runtime import handle_runtime_case, make_temp_runtime
from jarvis_v2.tools import currency_connector as _currency
from jarvis_v2.tools import weather_connector as _weather
from jarvis_v2.tools import markets_connector as _markets
from jarvis_v2.tools import calendar_connector as _calendar
from jarvis_v2.tools import call_connector as _call_connector


class _FakeCalExec:
    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        return self._payload


class _FakeCalEvents:
    def list(self, **kwargs):
        return _FakeCalExec(
            {"items": [{"summary": "Standup", "start": {"dateTime": "2026-07-05T09:00:00+09:00"}}]}
        )


class _FakeCalService:
    def events(self):
        return _FakeCalEvents()


class _ShortcutTool:
    def handler(self, args):
        return SimpleNamespace(
            output="Capability cockpit: all the operator phone lanes visible. Build Guardrails: live-proof freeze active.",
            metadata={
                "next_command": "channel health",
                "next_commands": ["channel health", "approval readiness 7"],
                "next_command_count": 2,
                "lane_count": 7,
                "attention_lane_count": 1,
                "proof_lane_count": 1,
                "proof_point_count": 4,
                "proof_summary": [
                    {
                        "lane_key": "operator_workflow_evals",
                        "lane_title": "Operator Workflow Evals",
                        "proof_count": 4,
                        "sample_proofs": [
                            "Korean Telegram send stops at approval with Hangul recipient/body intact",
                            "Morning Brief can be pushed on demand to the owner phone channel",
                        ],
                    }
                ],
                "status_counts": {"attention": 1, "awaiting approval": 1, "ok": 2, "ready": 3},
                "next_command_queue": [
                    {
                        "command": "channel health",
                        "kind": "diagnostic",
                        "lane_title": "Messaging (KR/EN)",
                        "status": "attention",
                    },
                    {
                        "command": "approval readiness 7",
                        "kind": "approval_review",
                        "lane_title": "Approvals",
                        "status": "awaiting approval",
                    },
                ],
                "capability_lanes": [
                    {
                        "key": "messaging",
                        "title": "Messaging (KR/EN)",
                        "status": "attention",
                        "risk": "HIGH_RISK",
                        "approval_required": True,
                        "tool_coverage": "registered",
                        "smoke_coverage": "registered",
                        "next_command": "channel health",
                        "next_command_kind": "diagnostic",
                        "example_command": "send 가상연락처이 a telegram saying hello",
                        "attention_reasons": ["latest failure: transport_error"],
                        "last_failure": "send_telegram @ 2026-07-06T12:00:00+09:00",
                        "last_failure_kind": "transport_error",
                    },
                    {
                        "key": "approvals",
                        "title": "Approvals",
                        "status": "awaiting approval",
                        "risk": "HIGH_RISK",
                        "approval_required": True,
                        "tool_coverage": "registered",
                        "smoke_coverage": "registered",
                        "next_command": "approval readiness 7",
                        "next_command_kind": "approval_review",
                        "last_approval_hold": "send_telegram @ 2026-07-06T12:01:00+09:00",
                        "approval_next_commands": ["approval readiness 7", "approval packet 7"],
                    },
                    {
                        "key": "voice",
                        "title": "Voice",
                        "status": "ok",
                        "risk": "READ_ONLY",
                        "approval_required": False,
                        "tool_coverage": "registered",
                        "smoke_coverage": "registered",
                        "next_command": "voice setup check",
                        "next_command_kind": "example",
                    },
                    {
                        "key": "build_guardrails",
                        "title": "Build Guardrails",
                        "status": "ready",
                        "risk": "READ_ONLY",
                        "approval_required": False,
                        "tool_coverage": "registered",
                        "smoke_coverage": "registered",
                        "next_command": "readiness report",
                        "next_command_kind": "example",
                        "guardrail_notes": ["live-proof freeze active"],
                    },
                    {
                        "key": "orchestration",
                        "title": "Internal Orchestration",
                        "status": "ready",
                        "risk": "READ_ONLY",
                        "approval_required": False,
                        "tool_coverage": "registered",
                        "smoke_coverage": "registered",
                        "next_command": "jarvis status",
                        "next_command_kind": "example",
                        "example_command": "jarvis status",
                        "guardrail_notes": ["internal subagents are workers, not companion personas"],
                    },
                    {
                        "key": "operator_workflow_evals",
                        "title": "Operator Workflow Evals",
                        "status": "ready",
                        "risk": "HIGH_RISK",
                        "approval_required": True,
                        "tool_coverage": "registered",
                        "smoke_coverage": "registered",
                        "next_command": "channel health",
                        "next_command_kind": "diagnostic",
                        "example_command": "push today's brief to my phone",
                        "proof_points": [
                            "Korean Telegram send stops at approval with Hangul recipient/body intact",
                            "Morning Brief can be pushed on demand to the owner phone channel",
                            "Phone control shortcuts expose status, failures, channels, approvals, cockpit, lanes, and guardrails",
                            "Clean-state cockpit has zero false attention lanes",
                        ],
                        "attention_reasons": [
                            "pins Korean sends, phone brief, channel diagnostics, contacts, markets"
                        ],
                    },
                    {
                        "key": "scheduler",
                        "title": "Scheduler",
                        "status": "ok",
                        "next_command": "list scheduled jobs",
                    },
                ],
            },
        )


class _ShortcutRegistry:
    def get(self, name):
        if name != "capability_cockpit":
            raise KeyError(name)
        return _ShortcutTool()


class _ShortcutRuntime:
    def __init__(self):
        self.inputs: list[str] = []
        self.registry = _ShortcutRegistry()

    def handle(self, text: str, request_token: str | None = None):
        self.inputs.append(text)
        replies = {
            "recent tool runs": "Recent tool runs: send_telegram failed at transport stage.",
            "execution health report": "Jarvis execution health report: recent failure reviewed.",
            "list scheduled jobs": "Scheduled jobs: Morning Brief next run at 09:00.",
            "channel health": "Jarvis channel health report: Telegram owner channel ok.",
            "pending approvals": "Pending approvals: none.",
            "jarvis status": "Jarvis status: ready agents 10 / 10.",
            "voice command cockpit": "Jarvis voice command cockpit: waiting for a confirmed transcript.",
            "voice setup check": "Jarvis voice setup check: read-only, no microphone access requested.",
            "voice stop intent: stop listening": "Jarvis voice stop intent packet: capture action stop current capture only.",
            "capability map": "Jarvis capability map: safe commands and approval-gated actions.",
            "handoff brief": "Jarvis handoff brief: Claude left CODEX_TASKS and live-proof freeze guidance for Codex.",
            "capability map info": "Jarvis capability map: Info covers weather, news, definitions, and Wikipedia.",
            "capability map markets": "Jarvis capability map: Markets covers stocks and crypto lookups.",
            "capability map messages": "Jarvis capability map: Messages & calls are approval-gated and support Korean/English contacts.",
            "capability map memory": "Jarvis capability map: Memory covers durable preferences, people, and search.",
            "capability map notes": "Jarvis capability map: Notes covers human-readable Obsidian note capture.",
            "capability map productivity": "Jarvis capability map: Productivity covers calendar, email, tasks, reminders, and brief surfaces.",
            "capability map research": "Jarvis capability map: Research covers web lookup and source-backed summaries.",
            "capability map utilities": "Jarvis capability map: Utilities covers translation, currency, math, and time helpers.",
            "capability map writing": "Jarvis capability map: Writing covers drafts and local write/type flows.",
            "work queue": (
                "Jarvis work queue: Zoey/OpenClaw/Hermes research points Jarvis toward a visible control plane, "
                "not companion personas. OpenAI Agents SDK/LangGraph/CrewAI/n8n point to traces, evals, guardrails, "
                "and human-in-the-loop checks. OpenClaw/Hermes warn against persistent prompt-injection and "
                "broad ungated integrations. Other Jarvis-style systems reinforce visibility before broad autonomy."
            ),
            "what changed in Jarvis": "Jarvis change report: latest Codex patch is visible for review.",
            "completion claim gate": "Completion claim gate: BLOCKED until live channel proofs and recovery/learning proof debt are reviewed.",
            "build progress": "Jarvis build progress report: current work state, active goals, and safe next build moves are visible.",
            "roadmap": "Jarvis roadmap: next assistant layers are visible without taking action.",
            "agi gates": "AGI gates: readiness is visible, with proof debt still blocking completion claims.",
            "list goals": "Active goals: keep Jarvis build visible until completion evidence is real.",
            "safety status": "Jarvis safety status: approvals guard risky actions.",
            "privacy report": "Jarvis privacy report: private data requires approval.",
            "risk matrix": "Jarvis risk matrix: READ_ONLY and HIGH_RISK counts.",
            "readiness report": "Readiness report: safe-use checks are visible.",
            "memory stats": "Jarvis memory stats: local memory health counters are visible.",
            "learning review": "Jarvis learning review: feedback and learning-loop checks are visible.",
            "setup check": "Jarvis V3 setup check: optional dependencies visible.",
            "jarvis doctor": "Jarvis doctor: diagnostics complete.",
        }
        return SimpleNamespace(response=replies.get(text, f"Unexpected shortcut route: {text}"), tool_results=[])


def eval_korean_send_lifecycle() -> None:
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = runtime.handle("send 가상연락처이 a telegram saying 좋은 아침이에요")
        if "approval" not in result.response.lower():
            raise SystemExit(f"Korean send must stop at the approval gate, got: {result.response[:120]!r}")

        pending = runtime.store.list_pending_approvals(limit=5)
        if len(pending) != 1 or pending[0]["tool_name"] != "send_telegram":
            raise SystemExit(f"expected exactly one queued send_telegram approval, got {[(r['id'], r['tool_name']) for r in pending]}")

        args = json.loads(pending[0]["planned_args"] or "{}")
        if args.get("to") != "가상연락처이":
            raise SystemExit(f"Hangul recipient corrupted in planned args: {args.get('to')!r}")
        if args.get("message") != "좋은 아침이에요":
            raise SystemExit(f"Hangul message corrupted in planned args: {args.get('message')!r}")

        executed = [row for row in runtime.store.recent_tool_runs(limit=10) if row["tool_name"] == "send_telegram" and row["ok"]]
        if executed:
            raise SystemExit("send_telegram must NOT execute without approval")

        approval_id = pending[0]["id"]
        runtime.handle(f"dismiss approval {approval_id}")
        if runtime.store.list_pending_approvals(limit=5):
            raise SystemExit("dismiss must clear the queued Korean send")


def eval_owner_self_telegram_restart_lifecycle() -> None:
    with TemporaryDirectory(prefix="jarvis-owner-telegram-eval-") as tmp:
        root = Path(tmp)
        command = "send a telegram to me saying approval recovery proof"
        first_runtime = make_temp_runtime(root)
        held = first_runtime.handle(command)
        pending = first_runtime.store.list_pending_approvals(limit=5)
        if len(pending) != 1 or pending[0]["tool_name"] != "send_telegram":
            raise SystemExit(f"owner-self Telegram command should queue one send approval: {held}")
        approval_id = int(pending[0]["id"])

        owner_calls = []
        web_calls = []
        original_owner = _call_connector._send_owner_telegram
        original_web = _call_connector._place_telegram_send
        try:
            _call_connector._send_owner_telegram = lambda message: owner_calls.append(message)
            _call_connector._place_telegram_send = lambda to, message: web_calls.append((to, message))

            restarted_runtime = make_temp_runtime(root)
            packet = restarted_runtime.registry.get("approval_execution_packet").handler(
                {"approval_id": approval_id}
            )
            if not packet.ok or packet.metadata.get("approval_readiness_rechecked") is not True:
                raise SystemExit(f"owner-self Telegram last look did not recover after restart: {packet}")
            transition = restarted_runtime.registry.get("approve_pending_approval").handler(
                {"approval_id": approval_id}
            )
            if not transition.ok:
                raise SystemExit(f"owner-self Telegram approval did not transition once: {transition}")
            rerun = restarted_runtime.handle(
                command,
                approved=True,
                approved_approval_id=approval_id,
            )
            successful = [
                item for item in rerun.tool_results
                if item.tool_name == "send_telegram" and item.ok
            ]
            if len(successful) != 1 or owner_calls != ["approval recovery proof"] or web_calls:
                raise SystemExit(
                    "owner-self Telegram approved rerun crossed the wrong adapter or count: "
                    f"owner={owner_calls}, web={web_calls}, result={rerun}"
                )

            repeated = restarted_runtime.handle(command)
            repeated_pending = restarted_runtime.store.list_pending_approvals(limit=5)
            if (
                len(repeated_pending) != 1
                or int(repeated_pending[0]["id"]) == approval_id
                or owner_calls != ["approval recovery proof"]
                or "approval" not in repeated.response.lower()
            ):
                raise SystemExit(
                    "a repeated owner-self request should queue a fresh approval without replaying the old send: "
                    f"pending={[(row['id'], row['status']) for row in repeated_pending]}, result={repeated}"
                )
        finally:
            _call_connector._send_owner_telegram = original_owner
            _call_connector._place_telegram_send = original_web


def eval_contact_lookup_korean() -> None:
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        added = runtime.registry.get("add_person").handler(
            {"name": "가상연락처이", "relation": "synthetic relation", "notes": "Synthetic contact profile"}
        )
        if not added.ok:
            raise SystemExit("add_person seed failed")
        result = runtime.handle("show person 가상연락처이")
        if "가상연락처이" not in result.response or "synthetic relation" not in result.response:
            raise SystemExit(f"Korean contact lookup failed: {result.response[:120]!r}")
        listed = runtime.handle("list people")
        if "가상연락처이" not in listed.response:
            raise SystemExit("list people must include the seeded Korean contact")


def eval_channel_health_phrase() -> None:
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = runtime.handle("channel health")
        if "channel health" not in result.response.lower():
            raise SystemExit(f"channel health phrase must route to the report, got: {result.response[:120]!r}")
        if "read-only" not in result.response.lower():
            raise SystemExit("channel health must declare its read-only boundary")


def eval_what_broke_diagnostics() -> None:
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        runtime.store.log_tool_run(
            session_id=runtime.session_id,
            tool_name="send_telegram",
            risk="HIGH_RISK",
            ok=False,
            approved=True,
            output="delivery failed at transport stage",
            metadata={"failure_kind": "transport_error"},
        )
        result = runtime.handle("recent tool runs")
        low = result.response.lower()
        if "send_telegram" not in low:
            raise SystemExit("the failed send must be visible in recent tool runs")
        if "failed" not in low and "blocked" not in low:
            raise SystemExit(f"failure state must be legible, got: {result.response[:160]!r}")


def eval_scheduler_visibility() -> None:
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = runtime.handle("list scheduled jobs")
        if "scheduled job" not in result.response.lower():
            raise SystemExit(f"scheduler visibility phrase failed: {result.response[:120]!r}")


def eval_korean_task_and_reminder_reads() -> None:
    with TemporaryDirectory(prefix="jarvis-korean-productivity-eval-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        seeded = runtime.registry.get("add_task").handler(
            {"body": "Jarvis 한국어 작업 확인", "due": "today", "priority": "normal"}
        )
        if not seeded.ok:
            raise SystemExit(f"could not seed Korean task eval: {seeded}")

        task_result = runtime.handle("오늘 할 일 뭐야?")
        if [item.tool_name for item in task_result.tool_results] != ["search_tasks"]:
            raise SystemExit(f"Korean dated task read missed the tools route: {task_result}")
        if "Jarvis 한국어 작업 확인" not in task_result.response:
            raise SystemExit(f"Korean dated task read missed the seeded task: {task_result.response!r}")

        reminder_result = runtime.handle("내 리마인더 보여줘")
        if [item.tool_name for item in reminder_result.tool_results] != ["list_reminders"]:
            raise SystemExit(f"Korean reminder read missed the tools route: {reminder_result}")
        if "Did you mean" in reminder_result.response:
            raise SystemExit(f"Korean reminder read should not require an English resend: {reminder_result.response!r}")

        if runtime.store.list_pending_approvals(limit=5):
            raise SystemExit("Korean task/reminder reads must not queue approvals")


@mock.patch.dict(os.environ, {"JARVIS_OWNER_TELEGRAM": "owner-smoke"}, clear=False)
def eval_morning_brief_phone_delivery() -> None:
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        scheduled = runtime.registry.get("schedule_morning_brief").handler({"time": "9:00am"})
        if not scheduled.ok:
            raise SystemExit(f"could not seed Morning Brief job: {scheduled.output[:120]!r}")

        built: list[object] = []
        sent: list[str] = []
        with mock.patch.object(
            scheduler_module,
            "_build_live_daily_brief",
            side_effect=lambda config: built.append(config) or "the operator morning brief: weather, calendar, markets.",
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or {"ok": True, "result": {"message_id": 101}},
        ):
            result = handle_runtime_case(runtime, "push today's brief to my phone", approved=True)

        if not result.verified:
            raise SystemExit(f"phone Morning Brief command should verify: {result.response[:160]!r}")
        if [tool_result.tool_name for tool_result in result.tool_results] != ["run_job_now"]:
            raise SystemExit(f"phone Morning Brief must reuse run_job_now: {result.tool_results}")
        if len(built) != 1:
            raise SystemExit(f"phone Morning Brief should compose exactly once: {built}")
        if sent != ["the operator morning brief: weather, calendar, markets."]:
            raise SystemExit(f"phone Morning Brief should send exactly once to owner channel: {sent}")
        tool_result = result.tool_results[0]
        if not tool_result.ok or "Morning Brief accepted by Telegram API" not in tool_result.output:
            raise SystemExit(f"phone Morning Brief should report Telegram API acceptance: {tool_result.output[:160]!r}")
        if "sent to Telegram" in tool_result.output or "delivered to Telegram" in tool_result.output:
            raise SystemExit(f"phone Morning Brief should not overclaim Telegram delivery: {tool_result.output[:160]!r}")
        if tool_result.metadata.get("job_name") != MORNING_BRIEF_JOB_NAME:
            raise SystemExit(f"phone Morning Brief should preserve job metadata: {tool_result.metadata}")


def eval_phone_control_shortcuts() -> None:
    runtime = _ShortcutRuntime()
    bridge = telegram_control_module.TelegramCommandBridge(
        runtime_factory=lambda: runtime,
        send_func=lambda chat_id, text, markup=None: {"ok": True},
        fetch_func=lambda token, offset, timeout: [],
        chat_action_func=None,
    )
    cases = [
        ("status", "jarvis status", "Jarvis status"),
        ("/voice", "voice command cockpit", "voice command cockpit"),
        ("/voice-setup", "voice setup check", "voice setup check"),
        ("/voice-stop", "voice stop intent: stop listening", "voice stop intent packet"),
        ("capabilities", "capability map", "capability map"),
        ("agent research note", "work queue", "Zoey/OpenClaw/Hermes research"),
        ("AI agent landscape", "work queue", "visible control plane"),
        ("every AI agent research", "work queue", "visible control plane"),
        ("three month plan", "work queue", "visible control plane"),
        ("three months of work", "work queue", "visible control plane"),
        ("Jarvis three month plan", "work queue", "visible control plane"),
        ("what are the three months of work", "work queue", "visible control plane"),
        ("what is the three month plan for Jarvis", "work queue", "visible control plane"),
        ("what should Jarvis focus on for the next three months", "work queue", "visible control plane"),
        ("how do we unlock the moat", "work queue", "visible control plane"),
        ("what unlocks Jarvis moat", "work queue", "visible control plane"),
        ("what should Jarvis build before integrations", "work queue", "visible control plane"),
        ("when should Jarvis add integrations", "work queue", "visible control plane"),
        ("should Jarvis add integrations now", "work queue", "visible control plane"),
        ("other Jarvis models", "work queue", "Other Jarvis-style systems"),
        ("what did you find about Zoey?", "work queue", "visible control plane"),
        ("what did you find about other Jarvis models", "work queue", "Other Jarvis-style systems"),
        ("what should Jarvis copy from OpenClaw", "work queue", "visible control plane"),
        ("what should Jarvis avoid from Zoey", "work queue", "not companion personas"),
        ("what is the Jarvis strategy", "work queue", "visible control plane"),
        ("Jarvis product strategy", "work queue", "visible control plane"),
        ("what is the right move for Jarvis", "work queue", "visible control plane"),
        ("what should we do after Zoey research?", "work queue", "visible control plane"),
        ("should Jarvis use companions", "work queue", "not companion personas"),
        ("should Jarvis copy Zoey?", "work queue", "not companion personas"),
        ("what is Jarvis moat", "work queue", "visible control plane"),
        ("why no companion personas", "work queue", "not companion personas"),
        ("what did Claude leave for Codex?", "handoff brief", "handoff brief"),
        ("what Claude said", "handoff brief", "handoff brief"),
        ("check what Claude has left", "handoff brief", "handoff brief"),
        ("check codex.md tasks", "work queue", "Zoey/OpenClaw/Hermes research"),
        ("what does CODEX_TASKS say?", "work queue", "Zoey/OpenClaw/Hermes research"),
        ("what did Codex modify?", "what changed in Jarvis", "change report"),
        ("what changed after Codex?", "what changed in Jarvis", "change report"),
        ("Lindy research", "work queue", "visible control plane"),
        ("Manus research", "work queue", "visible control plane"),
        ("조이 조사 결과", "work queue", "not companion personas"),
        ("모든 AI 에이전트 조사", "work queue", "visible control plane"),
        ("다른 자비스 모델", "work queue", "Other Jarvis-style systems"),
        ("자비스 차별점", "work queue", "visible control plane"),
        ("자비스 3개월 계획", "work queue", "visible control plane"),
        ("자비스 세 달 계획", "work queue", "visible control plane"),
        ("자비스 다음 3개월 뭐 해", "work queue", "visible control plane"),
        ("자비스 해자 어떻게 열어", "work queue", "visible control plane"),
        ("통합 지금 추가해도 돼", "work queue", "visible control plane"),
        ("통합 언제 추가해", "work queue", "visible control plane"),
        ("자비스 전략", "work queue", "visible control plane"),
        ("자비스 방향", "work queue", "visible control plane"),
        ("조이 이후 뭐 만들까", "work queue", "visible control plane"),
        ("자비스 조이 이후 뭐 만들어", "work queue", "visible control plane"),
        ("조이 따라해야 해", "work queue", "not companion personas"),
        ("자비스 컴패니언 해야 해", "work queue", "not companion personas"),
        ("왜 컴패니언 안 해", "work queue", "not companion personas"),
        ("is Jarvis done?", "completion claim gate", "Completion claim gate"),
        ("completion status", "completion claim gate", "BLOCKED"),
        ("자비스 끝났어?", "completion claim gate", "proof debt"),
        ("AGI progress", "build progress", "build progress report"),
        ("what did you build?", "build progress", "safe next build moves"),
        ("roadmap", "roadmap", "next assistant layers"),
        ("AGI status", "agi gates", "proof debt"),
        ("goals", "list goals", "Active goals"),
        ("AGI 진행상황", "build progress", "build progress report"),
        ("로드맵", "roadmap", "next assistant layers"),
        ("AGI 상태", "agi gates", "proof debt"),
        ("목표 상태", "list goals", "Active goals"),
        ("/safety", "safety status", "safety status"),
        ("is Jarvis safe to use?", "safety status", "safety status"),
        ("안전하게 써도 돼?", "safety status", "safety status"),
        ("what are your limits?", "safety status", "safety status"),
        ("what can you not do?", "safety status", "safety status"),
        ("자비스 한계", "safety status", "safety status"),
        ("뭐 못해", "safety status", "safety status"),
        ("what requires approval?", "safety status", "safety status"),
        ("what tools require approval?", "safety status", "safety status"),
        ("승인 필요한 것", "safety status", "safety status"),
        ("어떤 명령이 승인 필요", "safety status", "safety status"),
        ("/privacy", "privacy report", "privacy report"),
        ("/risk", "risk matrix", "risk matrix"),
        ("which commands are risky?", "risk matrix", "risk matrix"),
        ("which tools are high risk?", "risk matrix", "risk matrix"),
        ("고위험 도구", "risk matrix", "risk matrix"),
        ("what broke", "recent tool runs", "send_telegram failed"),
        ("what is broken?", "recent tool runs", "send_telegram failed"),
        ("can you send messages?", "capability map messages", "Messages & calls"),
        ("can you check email?", "capability map productivity", "Productivity"),
        ("can you use voice?", "voice command cockpit", "voice command cockpit"),
        ("can you check weather?", "capability map info", "Info"),
        ("can you check stock prices?", "capability map markets", "Markets"),
        ("can you translate?", "capability map utilities", "Utilities"),
        ("can you research the web?", "capability map research", "Research"),
        ("can you write text?", "capability map writing", "Writing"),
        ("can you remember things?", "capability map memory", "Memory"),
        ("can you take notes?", "capability map notes", "Notes"),
        ("can you read files?", "safety status", "safety status"),
        ("can you run commands?", "risk matrix", "risk matrix"),
        ("brief status", "list scheduled jobs", "Morning Brief next run"),
        ("channels", "channel health", "channel health report"),
        ("approvals", "pending approvals", "Pending approvals"),
        ("readiness", "readiness report", "Readiness report"),
        ("memory status", "memory stats", "memory stats"),
        ("learning status", "learning review", "learning review"),
        ("setup", "setup check", "setup check"),
        ("doctor", "jarvis doctor", "Jarvis doctor"),
        ("상태 확인", "jarvis status", "Jarvis status"),
        ("기능 알려줘", "capability map", "capability map"),
        ("메시지 보낼 수 있어", "capability map messages", "Messages & calls"),
        ("이메일 확인 가능해", "capability map productivity", "Productivity"),
        ("음성 가능해", "voice command cockpit", "voice command cockpit"),
        ("날씨 확인 가능해", "capability map info", "Info"),
        ("주식 확인 가능해", "capability map markets", "Markets"),
        ("번역 가능해", "capability map utilities", "Utilities"),
        ("웹 검색 가능해", "capability map research", "Research"),
        ("글쓰기 가능해", "capability map writing", "Writing"),
        ("기억할 수 있어", "capability map memory", "Memory"),
        ("메모 가능해", "capability map notes", "Notes"),
        ("파일 읽을 수 있어", "safety status", "safety status"),
        ("명령 실행 가능해", "risk matrix", "risk matrix"),
        ("뭐가 고장났어", "recent tool runs", "send_telegram failed"),
        ("최근 실패", "execution health report", "execution health report"),
        ("브리핑 확인", "list scheduled jobs", "Morning Brief next run"),
        ("did telegram send?", "channel health", "channel health report"),
        ("did the message send?", "channel health", "channel health report"),
        ("채널 상태", "channel health", "channel health report"),
        ("채널 헬스", "channel health", "channel health report"),
        ("텔레그램 보내졌어?", "channel health", "channel health report"),
        ("카톡 보내졌어?", "channel health", "channel health report"),
        ("텔레그램 보내졌나요?", "channel health", "channel health report"),
        ("카톡 갔나요?", "channel health", "channel health report"),
        ("승인 대기", "pending approvals", "Pending approvals"),
        ("준비 상태", "readiness report", "Readiness report"),
        ("기억 상태", "memory stats", "memory stats"),
        ("학습 상태", "learning review", "learning review"),
        ("설정 확인", "setup check", "setup check"),
        ("진단", "jarvis doctor", "Jarvis doctor"),
        ("음성", "voice command cockpit", "voice command cockpit"),
        ("마이크 확인", "voice setup check", "voice setup check"),
        ("음성 중지", "voice stop intent: stop listening", "voice stop intent packet"),
    ]
    for command, routed_to, expected_fragment in cases:
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} must remain read-only without approval buttons")
        if expected_fragment not in reply:
            raise SystemExit(f"{command!r} returned the wrong phone status reply: {reply[:160]!r}")
        if runtime.inputs[-1] != routed_to:
            raise SystemExit(f"{command!r} routed to {runtime.inputs[-1]!r}, expected {routed_to!r}")

    before_cockpit = list(runtime.inputs)
    for command in (
        "cockpit",
        "guardrails",
        "freeze status",
        "what is frozen",
        "what can Codex touch?",
        "what should you not edit?",
        "proofs pending",
        "show guardrails",
        "why frozen?",
        "safe lane",
        "콕핏",
        "가드레일 상태",
        "동결 상태",
        "프리즈 상태",
        "동결 파일",
        "수정 금지 파일",
    ):
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} shortcut must remain read-only without approval buttons")
        if "Capability cockpit" not in reply or "Build Guardrails" not in reply:
            raise SystemExit(f"{command!r} shortcut did not call the guardrail-aware cockpit tool: {reply[:160]!r}")
        if runtime.inputs != before_cockpit:
            raise SystemExit(f"{command!r} shortcut must call the read-only cockpit tool directly")

    for command in (
        "what should I test?",
        "test matrix status",
        "what live proofs are pending?",
        "what is the live test matrix?",
        "검증 대기 뭐야?",
        "테스트 매트릭스 보여줘",
        "라이브 테스트 뭐 해야 해?",
        "테스트 매트릭스 상태",
        "뭘 테스트해야 해?",
    ):
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} shortcut must remain read-only without approval buttons")
        for expected_fragment in (
            "Pending live-proof matrix",
            "KakaoTalk",
            "Telegram",
            "Instagram",
            "iMessage",
            "- Phone:",
            "- FaceTime:",
            "- KakaoTalk call:",
            "- Telegram call:",
            "- Instagram call:",
            "Call proof gate (all fields are required)",
            "recipient confirmation",
            "`call_requested` alone is not live proof",
            "Report format",
            "pass / fail / blocked",
            "last visible stage/error",
            "does not approve, send, call, run live_check",
        ):
            if expected_fragment not in reply:
                raise SystemExit(f"{command!r} shortcut lost live matrix context {expected_fragment!r}: {reply[:220]!r}")
        for forbidden in ("Fixture", "가상연락처일", "가상연락처이", "BotFather"):
            if forbidden in reply:
                raise SystemExit(f"{command!r} shortcut must not echo private live-test targets: {reply[:220]!r}")
        if runtime.inputs != before_cockpit:
            raise SystemExit(f"{command!r} shortcut must call the read-only matrix packet directly")

    for command in ("라이브 증명 상태",):
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} shortcut must remain read-only without approval buttons")
        if (
            "Cockpit lane: Acceptance Harness" not in reply
            and ("Capability cockpit" not in reply or "Build Guardrails" not in reply)
        ):
            raise SystemExit(f"{command!r} shortcut did not call an acceptance/freeze control surface: {reply[:160]!r}")
        if runtime.inputs != before_cockpit:
            raise SystemExit(f"{command!r} shortcut must call the read-only cockpit tool directly")

    before_summary = list(runtime.inputs)
    for command in (
        "cockpit summary",
        "control plane summary",
        "can I trust Jarvis?",
        "trust report",
        "is Jarvis reliable?",
        "자비스 믿어도 돼?",
        "믿을만해",
        "콕핏 요약",
    ):
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} shortcut must remain read-only without approval buttons")
        if "Cockpit summary" not in reply or "7 lane(s)" not in reply:
            raise SystemExit(f"{command!r} shortcut did not surface the compact cockpit overview: {reply[:220]!r}")
        for expected_fragment in (
            "1 needing attention",
            "1 approval-held",
            "attention: 1",
            "awaiting approval: 1",
            "ok: 2",
            "ready: 3",
        ):
            if expected_fragment not in reply:
                raise SystemExit(f"{command!r} summary lost status context {expected_fragment!r}: {reply[:220]!r}")
        for expected_fragment in (
            "proofs: 1 lane(s), 4 proof point(s)",
            "Operator Workflow Evals",
            "Korean Telegram send stops at approval",
            "Morning Brief can be pushed",
        ):
            if expected_fragment not in reply:
                raise SystemExit(f"{command!r} summary lost proof context {expected_fragment!r}: {reply[:220]!r}")
        for expected_fragment in ("channel health", "Messaging (KR/EN)", "approval readiness 7"):
            if expected_fragment not in reply:
                raise SystemExit(f"{command!r} summary lost queue context {expected_fragment!r}: {reply[:220]!r}")
        for expected_fragment in (
            "Approval-held means Jarvis is waiting for owner review",
            "not counted as a tool failure",
        ):
            if expected_fragment not in reply:
                raise SystemExit(f"{command!r} summary lost approval-held semantics {expected_fragment!r}: {reply[:220]!r}")
        if "does not approve" not in reply or "execute" not in reply:
            raise SystemExit(f"{command!r} shortcut should declare the read-only boundary: {reply[:220]!r}")
        if runtime.inputs != before_summary:
            raise SystemExit(f"{command!r} shortcut must not route through runtime.handle")

    before_proofs = list(runtime.inputs)
    for command in ("proofs", "evidence", "what proof do we have?", "증거 상태"):
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} shortcut must remain read-only without approval buttons")
        for expected_fragment in (
            "Cockpit proofs",
            "coverage: 1 lane(s), 4 proof point(s)",
            "Operator Workflow Evals",
            "Korean Telegram send stops at approval",
            "Morning Brief can be pushed",
        ):
            if expected_fragment not in reply:
                raise SystemExit(f"{command!r} proof shortcut lost {expected_fragment!r}: {reply[:220]!r}")
        if "does not approve" not in reply or "execute" not in reply:
            raise SystemExit(f"{command!r} shortcut should declare the read-only boundary: {reply[:220]!r}")
        if runtime.inputs != before_proofs:
            raise SystemExit(f"{command!r} shortcut must not route through runtime.handle")

    before_next_action = list(runtime.inputs)
    for command in ("next cockpit action", "what should I do next?", "다음 행동"):
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} shortcut must remain read-only without approval buttons")
        if "Next cockpit action" not in reply or "channel health" not in reply:
            raise SystemExit(f"{command!r} shortcut did not surface the top cockpit command: {reply[:180]!r}")
        if "Messaging (KR/EN)" not in reply or "diagnostic" not in reply:
            raise SystemExit(f"{command!r} shortcut lost cockpit queue context: {reply[:180]!r}")
        if "approval readiness 7" not in reply:
            raise SystemExit(f"{command!r} shortcut should include the queued approval review command: {reply[:180]!r}")
        if "does not approve" not in reply or "execute" not in reply:
            raise SystemExit(f"{command!r} shortcut should declare the read-only boundary: {reply[:180]!r}")
        if runtime.inputs != before_next_action:
            raise SystemExit(f"{command!r} shortcut must not route through runtime.handle")

    before_attention = list(runtime.inputs)
    for command in ("cockpit attention", "what needs attention?", "주의 상태"):
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} shortcut must remain read-only without approval buttons")
        if "Cockpit attention" not in reply or "Messaging (KR/EN)" not in reply:
            raise SystemExit(f"{command!r} shortcut did not surface cockpit attention: {reply[:180]!r}")
        if "Needs attention:" not in reply or "Approval-held review:" not in reply:
            raise SystemExit(f"{command!r} shortcut should separate failures from approval-held review: {reply[:220]!r}")
        for expected_fragment in (
            "Approval-held means Jarvis is waiting for owner review",
            "not counted as a tool failure",
        ):
            if expected_fragment not in reply:
                raise SystemExit(f"{command!r} attention lost approval-held safety semantics {expected_fragment!r}: {reply[:220]!r}")
        if "channel health" not in reply or "transport_error" not in reply:
            raise SystemExit(f"{command!r} shortcut lost diagnostic context: {reply[:180]!r}")
        if "approval readiness 7" not in reply:
            raise SystemExit(f"{command!r} shortcut should include approval-review context: {reply[:180]!r}")
        if "Scheduler" in reply:
            raise SystemExit(f"{command!r} shortcut should omit healthy cockpit lanes: {reply[:180]!r}")
        if "does not approve" not in reply or "execute" not in reply:
            raise SystemExit(f"{command!r} shortcut should declare the read-only boundary: {reply[:180]!r}")
        if runtime.inputs != before_attention:
            raise SystemExit(f"{command!r} shortcut must not route through runtime.handle")

    before_lane = list(runtime.inputs)
    lane_cases = [
        ("messaging lane", "Cockpit lane: Messaging (KR/EN)", "channel health"),
        ("approvals lane", "Cockpit lane: Approvals", "not counted as a tool failure"),
        ("승인 상태", "Cockpit lane: Approvals", "not counted as a tool failure"),
        ("voice lane", "Cockpit lane: Voice", "voice setup check"),
        ("메시지 상태", "Cockpit lane: Messaging (KR/EN)", "transport_error"),
        ("가드레일 레인", "Cockpit lane: Build Guardrails", "live-proof freeze active"),
        ("agent status", "Cockpit lane: Internal Orchestration", "jarvis status"),
        ("worker status", "Cockpit lane: Internal Orchestration", "internal subagents are workers"),
        ("에이전트 상태", "Cockpit lane: Internal Orchestration", "not companion personas"),
    ]
    for command, expected_heading, expected_fragment in lane_cases:
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} shortcut must remain read-only without approval buttons")
        if expected_heading not in reply or expected_fragment not in reply:
            raise SystemExit(f"{command!r} shortcut did not surface lane detail: {reply[:220]!r}")
        if "does not approve" not in reply or "execute" not in reply:
            raise SystemExit(f"{command!r} shortcut should declare the read-only boundary: {reply[:220]!r}")
        if runtime.inputs != before_lane:
            raise SystemExit(f"{command!r} shortcut must not route through runtime.handle")

    before_eval_lane = list(runtime.inputs)
    for command in (
        "trust tests",
        "operator real workflow tests",
        "is Korean messaging tested",
        "Korean message proof",
        "can Jarvis safely send Korean messages",
        "한국어 메시지 평가",
        "한국어 메시지 증명",
        "한글 전송 증명",
        "가상연락처이 전송 테스트",
        "모닝브리핑 평가",
    ):
        reply, markup = bridge._handle_command(command)
        if markup is not None:
            raise SystemExit(f"{command!r} shortcut must remain read-only without approval buttons")
        if "Cockpit lane: Operator Workflow Evals" not in reply:
            raise SystemExit(f"{command!r} shortcut did not open the eval lane: {reply[:220]!r}")
        for expected_fragment in (
            "HIGH_RISK",
            "approval required: yes",
            "smoke: registered",
            "Korean sends",
            "proofs:",
            "Korean Telegram send stops at approval",
            "Morning Brief can be pushed",
            "Clean-state cockpit has zero false attention lanes",
        ):
            if expected_fragment not in reply:
                raise SystemExit(f"{command!r} eval lane lost {expected_fragment!r}: {reply[:220]!r}")
        if "does not approve" not in reply or "execute" not in reply:
            raise SystemExit(f"{command!r} shortcut should declare the read-only boundary: {reply[:220]!r}")
        if runtime.inputs != before_eval_lane:
            raise SystemExit(f"{command!r} shortcut must not route through runtime.handle")


def eval_telegram_voice_note_bilingual_send_gate() -> None:
    """A Telegram voice note can become a Korean/English send request, but the
    normal approval gate must still be the first side-effect boundary."""
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        tmp_path = Path(tmp)
        runtime = make_temp_runtime(tmp_path / "runtime")
        transcript = "send 가상연락처이 a telegram saying hello 안녕하세요"
        sent: list[tuple[str, str, dict | None]] = []
        old_state = telegram_control_module.STATE_FILE
        old_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
        original_transcribe = telegram_control_module.transcribe_voice_message
        try:
            os.environ["TELEGRAM_BOT_TOKEN"] = "test:token"
            os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
            telegram_control_module.STATE_FILE = tmp_path / "telegram.json"
            telegram_control_module._save_offset(0)
            telegram_control_module.transcribe_voice_message = lambda token, message: (transcript, "")
            update = {
                "update_id": 501,
                "message": {
                    "chat": {"id": 555001},
                    "voice": {"file_id": "voice-file", "file_size": 2048},
                },
            }
            bridge = telegram_control_module.TelegramCommandBridge(
                runtime_factory=lambda: runtime,
                send_func=lambda chat_id, text, markup=None: sent.append((chat_id, text, markup)) or {"ok": True},
                fetch_func=lambda token, offset, timeout: [update],
                chat_action_func=None,
            )
            if bridge.process_once(poll_timeout=0) != 1:
                raise SystemExit("Telegram voice-note send eval did not process exactly one owner update")
        finally:
            telegram_control_module.transcribe_voice_message = original_transcribe
            telegram_control_module.STATE_FILE = old_state
            if old_token is None:
                os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            else:
                os.environ["TELEGRAM_BOT_TOKEN"] = old_token
            if old_owner is None:
                os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
            else:
                os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner

        if not any(f"🎙 Heard: {transcript}" in text for _cid, text, _markup in sent):
            raise SystemExit(f"Telegram voice-note eval did not echo bilingual transcript intact: {sent}")
        pending = runtime.store.list_pending_approvals(limit=5)
        if len(pending) != 1 or pending[0]["tool_name"] != "send_telegram":
            raise SystemExit(f"voice-note send should queue exactly one send_telegram approval: {pending}")
        args = json.loads(pending[0]["planned_args"] or "{}")
        if args.get("to") != "가상연락처이" or args.get("message") != "hello 안녕하세요":
            raise SystemExit(f"voice-note send corrupted bilingual planned args: {args}")
        if not any(markup for _cid, text, markup in sent if "approval" in text.lower()):
            raise SystemExit(f"voice-note send should return approval buttons to the phone: {sent}")
        executed = [
            row for row in runtime.store.recent_tool_runs(limit=10)
            if row["tool_name"] == "send_telegram" and row["ok"]
        ]
        if executed:
            raise SystemExit(f"voice-note send must not execute before approval: {executed}")


def eval_telegram_voice_note_korean_status_shortcut() -> None:
    """A Korean voice note asking for channel status should use the phone
    shortcut layer directly, not a generic chat fallback or approval path."""
    runtime = _ShortcutRuntime()
    transcript = "채널 상태"
    sent: list[tuple[str, str, dict | None]] = []
    old_state = telegram_control_module.STATE_FILE
    old_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    old_owner = os.environ.get("JARVIS_OWNER_TELEGRAM")
    original_transcribe = telegram_control_module.transcribe_voice_message
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        try:
            os.environ["TELEGRAM_BOT_TOKEN"] = "test:token"
            os.environ["JARVIS_OWNER_TELEGRAM"] = "555001"
            telegram_control_module.STATE_FILE = Path(tmp) / "telegram.json"
            telegram_control_module._save_offset(0)
            telegram_control_module.transcribe_voice_message = lambda token, message: (transcript, "")
            update = {
                "update_id": 502,
                "message": {
                    "chat": {"id": 555001},
                    "voice": {"file_id": "voice-file", "file_size": 2048},
                },
            }
            bridge = telegram_control_module.TelegramCommandBridge(
                runtime_factory=lambda: runtime,
                send_func=lambda chat_id, text, markup=None: sent.append((chat_id, text, markup)) or {"ok": True},
                fetch_func=lambda token, offset, timeout: [update],
                chat_action_func=None,
            )
            if bridge.process_once(poll_timeout=0) != 1:
                raise SystemExit("Korean status voice-note eval did not process exactly one owner update")
        finally:
            telegram_control_module.transcribe_voice_message = original_transcribe
            telegram_control_module.STATE_FILE = old_state
            if old_token is None:
                os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            else:
                os.environ["TELEGRAM_BOT_TOKEN"] = old_token
            if old_owner is None:
                os.environ.pop("JARVIS_OWNER_TELEGRAM", None)
            else:
                os.environ["JARVIS_OWNER_TELEGRAM"] = old_owner

    if runtime.inputs != ["channel health"]:
        raise SystemExit(f"Korean status voice note should route directly to channel health: {runtime.inputs}")
    if not any(f"🎙 Heard: {transcript}" in text for _cid, text, _markup in sent):
        raise SystemExit(f"Korean status voice note did not echo transcript intact: {sent}")
    if not any("channel health report" in text for _cid, text, _markup in sent):
        raise SystemExit(f"Korean status voice note did not return channel health: {sent}")
    if any(markup for _cid, _text, markup in sent):
        raise SystemExit(f"Korean read-only voice shortcut must not create approval buttons: {sent}")


def eval_memory_write() -> None:
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        result = runtime.handle("remember that the operator prefers direct answers")
        if "remembered" not in result.response.lower():
            raise SystemExit(f"memory write failed: {result.response[:120]!r}")


def eval_currency_conversion() -> None:
    original = _currency._fetch
    _currency._fetch = lambda amount, src, dst: {"rates": {dst: 1531.23 * (amount / 100.0)}}
    try:
        with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
            runtime = make_temp_runtime(Path(tmp))
            result = runtime.handle("convert 100 USD to KRW")
            low = result.response.lower()
            if "krw" not in low or "usd" not in low:
                raise SystemExit(f"currency conversion did not route/format: {result.response[:120]!r}")
            if "1,531" not in result.response and "153,123" not in result.response:
                raise SystemExit(f"currency conversion lost the mocked rate: {result.response[:120]!r}")
    finally:
        _currency._fetch = original


def eval_weather_lookup() -> None:
    original = _weather._fetch
    _weather._fetch = lambda location: {
        "current_condition": [{"temp_C": "21", "FeelsLikeC": "20", "humidity": "48",
                               "windspeedKmph": "6", "weatherDesc": [{"value": "Sunny"}]}],
        "weather": [{"maxtempC": "26", "mintempC": "16"}],
        "nearest_area": [{"areaName": [{"value": "Seoul"}]}],
    }
    try:
        with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
            runtime = make_temp_runtime(Path(tmp))
            result = runtime.handle("weather in Seoul")
            if "seoul" not in result.response.lower():
                raise SystemExit(f"weather did not route to Seoul: {result.response[:120]!r}")
            if "21" not in result.response and "Sunny" not in result.response:
                raise SystemExit(f"weather lost the mocked reading: {result.response[:120]!r}")
    finally:
        _weather._fetch = original


def eval_markets_summary() -> None:
    orig_crypto, orig_stock = _markets._fetch_crypto_many, _markets._fetch_stock
    _markets._fetch_crypto_many = lambda coin_ids: {
        cid: {"usd": 60000.0, "usd_24h_change": 2.5} for cid in coin_ids
    }
    _markets._fetch_stock = lambda symbol: {"price": 100.0, "prev": 98.0, "symbol": symbol.upper()}
    try:
        with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
            runtime = make_temp_runtime(Path(tmp))
            result = runtime.handle("markets overview")
            low = result.response.lower()
            if "market" not in low and "btc" not in low:
                raise SystemExit(f"markets overview did not route: {result.response[:120]!r}")
            if "60,000" not in result.response and "60000" not in result.response and "$" not in result.response:
                raise SystemExit(f"markets overview lost the mocked prices: {result.response[:150]!r}")
            korean = runtime.handle("markets overview in Korean")
            if "시장 스냅샷" not in korean.response or "투자 조언" not in korean.response:
                raise SystemExit(f"Korean markets overview did not route/localize: {korean.response[:150]!r}")
            if "Markets snapshot:" in korean.response or "Informational only" in korean.response:
                raise SystemExit(f"Korean markets overview should not use English framing: {korean.response[:150]!r}")
            if not korean.tool_results or korean.tool_results[0].metadata.get("response_language") != "ko":
                raise SystemExit(f"Korean markets overview should expose response_language=ko metadata: {korean.tool_results}")
    finally:
        _markets._fetch_crypto_many = orig_crypto
        _markets._fetch_stock = orig_stock


def eval_calendar_brief() -> None:
    original = _calendar._get_readonly_service
    _calendar._get_readonly_service = lambda: _FakeCalService()
    try:
        with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
            runtime = make_temp_runtime(Path(tmp))
            result = runtime.handle("what's on my calendar today")
            if "standup" not in result.response.lower():
                raise SystemExit(f"calendar brief did not surface the mocked event: {result.response[:120]!r}")
            korean = runtime.handle("오늘 일정 알려줘")
            if "standup" not in korean.response.lower():
                raise SystemExit(f"Korean calendar brief lost the mocked event: {korean.response[:120]!r}")
            if (
                [action.tool_name for action in korean.plan.actions] != ["list_events"]
                or korean.plan.actions[0].args != {"range": "today"}
                or korean.metadata.get("runtime_route") != "tools"
            ):
                raise SystemExit(f"Korean calendar brief fell through deterministic routing: {korean}")
            free = runtime.handle("am I free tomorrow")
            if "standup" not in free.response.lower() and "thing" not in free.response.lower():
                raise SystemExit(f"calendar availability query failed: {free.response[:120]!r}")
            korean_free = runtime.handle("내일 시간 있어?")
            if "standup" not in korean_free.response.lower() and "thing" not in korean_free.response.lower():
                raise SystemExit(f"Korean calendar availability query failed: {korean_free.response[:120]!r}")
            if (
                [action.tool_name for action in korean_free.plan.actions] != ["check_availability"]
                or korean_free.plan.actions[0].args != {"text": "am I free tomorrow"}
                or korean_free.metadata.get("runtime_route") != "tools"
            ):
                raise SystemExit(
                    f"Korean calendar availability fell through deterministic routing: {korean_free}"
                )
    finally:
        _calendar._get_readonly_service = original


def eval_clean_state_is_all_clear() -> None:
    """A healthy Jarvis with nothing wrong must LOOK healthy — no false alarms.
    On a fresh runtime the cockpit should show zero attention/missing lanes and
    there should be no pending approvals. Guards against a regression that makes
    the control plane cry wolf."""
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        meta = runtime.registry.get("capability_cockpit").handler({}).metadata
        noisy = [lane["key"] for lane in meta["capability_lanes"] if lane["status"] in ("attention", "missing")]
        if noisy:
            raise SystemExit(f"clean runtime should have no attention lanes, got: {noisy}")
        if meta.get("attention_lane_count", 0) != 0:
            raise SystemExit(f"clean runtime attention_lane_count must be 0, got {meta.get('attention_lane_count')}")
        if runtime.store.list_pending_approvals(limit=5):
            raise SystemExit("clean runtime should have no pending approvals")


def eval_discovery_surface_is_read_only() -> None:
    """"what can you do" and "voice command cockpit" are how the operator discovers
    capabilities. They must route to read-only help/inspection surfaces and
    never queue an approval or execute anything."""
    with TemporaryDirectory(prefix="jarvis-evals-") as tmp:
        runtime = make_temp_runtime(Path(tmp))
        cap = runtime.handle("what can you do")
        if "capabilit" not in cap.response.lower():
            raise SystemExit(f"'what can you do' should surface capabilities: {cap.response[:120]!r}")
        voice = runtime.handle("voice command cockpit")
        if "voice command cockpit" not in voice.response.lower() and "transcript" not in voice.response.lower():
            raise SystemExit(f"voice command cockpit did not route: {voice.response[:120]!r}")
        discovery_cases = {
            "can you send messages": "capability map messages",
            "can you check email": "capability map productivity",
            "can you give morning brief": "capability map productivity",
            "can you check weather": "capability map info",
            "can you read news": "capability map info",
            "can you check stock prices": "capability map markets",
            "can you translate": "capability map utilities",
            "can you research the web": "capability map research",
            "can you write text": "capability map writing",
            "can you remember things": "capability map memory",
            "can you take notes": "capability map notes",
            "can you read files": "safety status",
            "can you run commands": "risk matrix",
            "can you use voice": "voice command cockpit",
            "메시지 보낼 수 있어": "capability map messages",
            "이메일 확인 가능해": "capability map productivity",
            "날씨 확인 가능해": "capability map info",
            "주식 확인 가능해": "capability map markets",
            "번역 가능해": "capability map utilities",
            "웹 검색 가능해": "capability map research",
            "글쓰기 가능해": "capability map writing",
            "기억할 수 있어": "capability map memory",
            "메모 가능해": "capability map notes",
            "파일 읽을 수 있어": "safety status",
            "명령 실행 가능해": "risk matrix",
            "음성 가능해": "voice command cockpit",
        }
        for phrase, expected in discovery_cases.items():
            result = runtime.handle(phrase)
            response = result.response
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"{phrase!r} should redirect to {expected!r}: {response[:160]!r}")
            if result.tool_results:
                raise SystemExit(f"{phrase!r} must stay discovery-only without executing tools: {result.tool_results}")
        broken = runtime.handle("what is broken")
        if "No tool runs logged yet." not in broken.response and "Recent tool runs:" not in broken.response:
            raise SystemExit(f"'what is broken' should show read-only recent tool runs: {broken.response[:160]!r}")
        if len(broken.tool_results) != 1 or getattr(broken.tool_results[0], "tool_name", "") != "recent_tool_runs":
            raise SystemExit(f"'what is broken' should only run recent_tool_runs: {broken.tool_results!r}")
        if runtime.store.list_pending_approvals(limit=5):
            raise SystemExit("discovery/voice surfaces must stay read-only (no approvals queued)")


def main() -> None:
    eval_korean_send_lifecycle()
    eval_owner_self_telegram_restart_lifecycle()
    eval_contact_lookup_korean()
    eval_channel_health_phrase()
    eval_what_broke_diagnostics()
    eval_morning_brief_phone_delivery()
    eval_phone_control_shortcuts()
    eval_telegram_voice_note_bilingual_send_gate()
    eval_telegram_voice_note_korean_status_shortcut()
    eval_scheduler_visibility()
    eval_korean_task_and_reminder_reads()
    eval_memory_write()
    eval_currency_conversion()
    eval_weather_lookup()
    eval_markets_summary()
    eval_calendar_brief()
    eval_clean_state_is_all_clear()
    eval_discovery_surface_is_read_only()
    print("operator-real eval pack passed")


if __name__ == "__main__":
    main()
