"""Smoke tests for the fuzzy "did you mean?" command suggester (no model, no I/O)."""

from __future__ import annotations

from jarvis_v2.agent.command_suggest import suggest_command


def test_typo_near_miss_is_suggested() -> None:
    out = suggest_command("weathr in tokyo")
    if not out or "weather in Tokyo" not in out:
        raise SystemExit(f"typo should suggest the real weather command: {out!r}")
    if "Did you mean" not in out:
        raise SystemExit(f"suggestion should be phrased as a question: {out!r}")


def test_currency_and_market_typos_are_suggested() -> None:
    cur = suggest_command("convrt 100 usd to krw")
    if not cur or "convert 100 USD to KRW" not in cur:
        raise SystemExit(f"currency typo should suggest the real command: {cur!r}")
    stk = suggest_command("stck price of aapl")
    if not stk or "stock price of AAPL" not in stk:
        raise SystemExit(f"market typo should suggest the real command: {stk!r}")


def test_short_planner_alias_typos_are_suggested() -> None:
    cal = suggest_command("calender today")
    if not cal or "calendar today" not in cal:
        raise SystemExit(f"calendar typo should suggest the short planner alias: {cal!r}")
    wiki = suggest_command("wikipeda ada lovelace")
    if not wiki or "wikipedia Ada Lovelace" not in wiki:
        raise SystemExit(f"wikipedia typo should suggest the short planner alias: {wiki!r}")
    reminder = suggest_command("remind me tomorow")
    if not reminder or "remind me tomorrow" not in reminder:
        raise SystemExit(f"reminder typo should suggest the short planner alias: {reminder!r}")


def test_storage_recovery_typos_are_suggested() -> None:
    status = suggest_command("storag status")
    if not status or "storage status" not in status:
        raise SystemExit(f"storage status typo should suggest the storage status command: {status!r}")
    plan = suggest_command("storage recoery plan")
    if not plan or "storage recovery plan" not in plan:
        raise SystemExit(f"storage recovery plan typo should suggest the read-only plan command: {plan!r}")
    check = suggest_command("storage recovery chek")
    if not check or "storage recovery check" not in check:
        raise SystemExit(f"storage recovery check typo should suggest the read-only check command: {check!r}")


def test_setup_diagnostic_typos_are_suggested() -> None:
    cases = {
        "jarvis docter": "jarvis doctor",
        "setup chek": "setup check",
        "readines report": "readiness report",
        "prototype readines": "prototype readiness",
        "safety staus": "safety status",
        "help saftey": "help safety",
        "voice setup chek": "voice setup check",
    }
    for typo, expected in cases.items():
        out = suggest_command(typo)
        if not out or expected not in out:
            raise SystemExit(f"setup/diagnostic typo should suggest {expected!r}: {typo!r} -> {out!r}")


def test_control_help_typos_are_suggested() -> None:
    cases = {
        "help contrl": "help control",
        "help cokpit": "help cockpit",
        "help statuz": "help status",
        "help control plane": "help control",
        "hlep control": "help control",
        "hlep cockpit": "help cockpit",
        "hlep status": "help status",
    }
    wrong_fragments = ["harness control", "harness status", "“cockpit”"]
    for typo, expected in cases.items():
        out = suggest_command(typo)
        if not out or expected not in out:
            raise SystemExit(f"control-help typo should suggest {expected!r}: {typo!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"control-help typo suggested stale/wrong command {wrong!r}: {typo!r} -> {out!r}"
                )


def test_approval_and_proof_typos_are_suggested() -> None:
    cases = {
        "pending aprovals": "pending approvals",
        "aproval review": "approval review",
        "approval readines 1": "approval readiness 1",
        "approval pakcet 1": "approval packet 1",
        "approval chain prof 1": "approval chain proof 1",
        "verificaton receipt 1": "verification receipt 1",
        "execution helth report": "execution health report",
        "recovery closure cheklist": "recovery closure checklist",
        "after action lerning packet 1": "after-action learning packet 1",
    }
    for typo, expected in cases.items():
        out = suggest_command(typo)
        if not out or expected not in out:
            raise SystemExit(f"approval/proof typo should suggest {expected!r}: {typo!r} -> {out!r}")


def test_approval_gate_intents_redirect_to_read_only_approval_commands() -> None:
    cases = {
        "approval queue": "pending approvals",
        "approvals status": "approval summary",
        "approval status": "approval summary",
        "approval report": "approval summary",
        "approval health": "approval summary",
        "approval gate": "approval summary",
        "approval gates": "approval summary",
        "approval gate status": "approval summary",
        "what is blocked by approval": "approval summary",
        "what approvals are waiting": "pending approvals",
        "what is waiting on me": "pending approvals",
        "anything waiting on me": "pending approvals",
        "any approvals pending": "pending approvals",
        "are approvals pending": "pending approvals",
        "anything need approval": "pending approvals",
        "anything needs my approval": "pending approvals",
        "anything pending approval": "pending approvals",
        "approval needed": "pending approvals",
        "blocked by approval": "approval summary",
        "do you need anything from me": "pending approvals",
        "needs approval": "pending approvals",
        "pending review": "pending approvals",
        "pending approval status": "pending approvals",
        "review queue": "pending approvals",
        "show pending approval status": "pending approvals",
        "show pending approvals": "pending approvals",
        "show approvals": "pending approvals",
        "show me approvals": "pending approvals",
        "show me pending approvals": "pending approvals",
        "do you need my approval": "pending approvals",
        "what do you need from me": "pending approvals",
        "what is blocked by me": "pending approvals",
        "what approvals are pending": "pending approvals",
        "what is pending": "pending approvals",
        "what should i review": "pending approvals",
        "which approvals are pending": "pending approvals",
        "waiting for me": "pending approvals",
        "approval latest": "approval readiness latest",
        "latest approval": "approval readiness latest",
        "approval proof": "approval evidence for latest",
        "approval evidence": "approval evidence for latest",
        "approval receipt": "approval evidence for latest",
    }
    wrong_fragments = ["jarvis status", "privacy report", "approval packet 1", "approval review", "setup check"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"approval-gate intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"approval-gate intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_approval_failure_intents_redirect_to_read_only_approval_summary() -> None:
    cases = [
        "approval failed",
        "approval not working",
        "approval problem",
        "approval stuck",
        "approval button failed",
        "approval button not working",
        "approval button stuck",
        "approval queue failed",
        "approval queue stuck",
        "approve failed",
        "approve button not working",
        "why did approval fail",
        "why did approve fail",
    ]
    wrong_fragments = ["channel health", "recent tool runs", "pending approvals", "queued as approval"]
    for phrase in cases:
        out = suggest_command(phrase)
        if not out or "approval summary" not in out:
            raise SystemExit(f"approval failure intent should suggest approval summary: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"approval failure intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_harness_starter_typos_are_suggested() -> None:
    cases = {
        "harnes operations": "harness operations",
        "agi next bild move": "agi next build move",
        "completion clame gate": "completion claim gate",
        "command cokpit": "command cockpit",
        "model routeing status": "model routing status",
        "capabilty map": "capability map",
    }
    for typo, expected in cases.items():
        out = suggest_command(typo)
        if not out or expected not in out:
            raise SystemExit(f"harness starter typo should suggest {expected!r}: {typo!r} -> {out!r}")


def test_completion_and_agi_status_intents_redirect_to_read_only_gate_commands() -> None:
    cases = {
        "is jarvis done": "completion claim gate",
        "is jarvis finished": "completion claim gate",
        "is jarvis complete": "completion claim gate",
        "can jarvis claim done": "completion claim gate",
        "can jarvis claim completion": "completion claim gate",
        "did acceptance pass": "completion claim gate",
        "has jarvis passed acceptance": "completion claim gate",
        "jarvis done status": "completion claim gate",
        "done status": "completion claim gate",
        "jarvis completion status": "completion audit",
        "completion status": "completion audit",
        "completion claim": "completion claim gate",
        "agi status": "agi gates",
        "agi readiness": "agi gates",
        "agi readiness status": "agi gates",
        "agent harness status": "harness status",
        "agent harness readiness": "harness readiness digest",
        "harness readiness": "harness readiness digest",
    }
    wrong_fragments = ["list jarvis notes", "action readiness", "jarvis status"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"completion/AGI intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"completion/AGI intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_progress_roadmap_goal_intents_redirect_to_read_only_control_commands() -> None:
    cases = {
        "agi progress": "build progress",
        "jarvis progress": "build progress",
        "how is jarvis going": "build progress",
        "project status": "build progress",
        "what did you build": "build progress",
        "what have you built": "build progress",
        "roadmap status": "roadmap",
        "what is next on the roadmap": "roadmap",
        "what should jarvis build next": "roadmap",
        "goal status": "list goals",
        "goals status": "list goals",
        "active goals": "list goals",
    }
    wrong_fragments = ["completion claim", "approval summary", "jarvis status", "fallback"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"progress/roadmap/goal intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"progress/roadmap/goal intent suggested stale/wrong command {wrong!r}: "
                    f"{phrase!r} -> {out!r}"
                )


def test_subagent_orchestration_intents_redirect_to_status() -> None:
    exact_command_phrases = {
        "subagent fleet status",
    }
    for phrase in [
        "agent readiness",
        "agent status",
        "agent health",
        "agent fleet health",
        "agents ready",
        "agents health",
        "are agents ready",
        "are my agents ready",
        "how many agents are ready",
        "subagent status",
        "subagent health",
        "subagent readiness",
        "subagent fleet health",
        "subagent fleet readiness",
        "internal orchestration status",
        "internal agents status",
        "internal worker status",
        "orchestration status",
        "tool orchestration status",
        "parallel agent status",
        "parallel agents status",
        "worker fleet status",
        "worker fleet health",
        "worker readiness",
        "worker health",
        "ready agents",
        "ready agents status",
        "ready agent count",
        "what agents are ready",
        "what agents are running",
        "which agents are ready",
        "which agents are running",
        "에이전트 상태",
        "에이전트 준비 상태",
        "에이전트 헬스",
        "서브에이전트 상태",
        "서브에이전트 준비 상태",
        "준비된 에이전트",
        "준비된 에이전트 몇 개",
        "내부 오케스트레이션 상태",
        "워커 상태",
        "워커 준비 상태",
        "워커 헬스",
        "병렬 에이전트 상태",
    ]:
        out = suggest_command(phrase)
        if not out or "subagent fleet status" not in out:
            raise SystemExit(f"subagent/orchestration intent should suggest subagent fleet status: {phrase!r} -> {out!r}")
        if "jarvis status" in out:
            raise SystemExit(f"subagent/orchestration intent suggested stale broad status: {phrase!r} -> {out!r}")

    for phrase in exact_command_phrases:
        out = suggest_command(phrase)
        if out is not None:
            raise SystemExit(f"exact subagent fleet command should route directly, not suggest itself: {phrase!r} -> {out!r}")


def test_model_brain_intents_redirect_to_model_routing_status() -> None:
    cases = [
        "model status",
        "model health",
        "model routing",
        "which model are you using",
        "what model are you using",
        "what brain are you using",
        "brain status",
        "brain health",
        "brain loop status",
        "ai brain status",
        "local model status",
        "ollama status",
        "llm status",
        "planner status",
        "chat model status",
        "chat acceptance latency",
        "chat latency status",
        "chat latency acceptance status",
        "chat history window",
        "chat history messages",
        "chat max history messages",
        "chat max reply tokens",
        "chat p95",
        "chat p95 status",
        "chat p95 target",
        "chat reply token cap",
        "chat response length status",
        "chat speed acceptance status",
        "chat speed settings",
        "chat token cap",
        "chat tuning",
        "chat tuning knobs",
        "conversation acceptance latency",
        "conversation p95 status",
        "conversation speed status",
        "8 second chat target",
        "8 second conversation target",
        "8s chat target",
        "8s conversation target",
        "did chat meet the 8 second target",
        "did chat pass latency",
        "did mixed conversation pass latency",
        "how do I make chat faster",
        "is chat under 8 seconds",
        "is Jarvis under 8 seconds",
        "is mixed conversation under 8 seconds",
        "Jarvis latency status",
        "Jarvis speed status",
        "lower chat history window",
        "lower chat token cap",
        "lower reply token cap",
        "make chat faster",
        "make Jarvis faster",
        "make Jarvis replies shorter",
        "mixed conversation acceptance latency",
        "mixed conversation acceptance status",
        "mixed conversation latency",
        "mixed conversation latency status",
        "mixed conversation p95",
        "mixed conversation p95 status",
        "p95 chat target",
        "p95 latency status",
        "reduce chat history",
        "reduce reply tokens",
        "reply length status",
        "response length status",
        "should we lower chat history",
        "should we lower reply tokens",
        "speed up chat",
        "speed up Jarvis",
        "tune chat speed",
        "tune Jarvis speed",
        "what are the chat speed knobs",
        "what are the chat tuning knobs",
        "under 8 seconds status",
        "why are replies slow",
        "why are chat replies slow",
        "why is chat slow",
        "why is Jarvis slow",
        "planner model status",
        "routing status",
        "대화 지연 상태",
        "자비스 느려",
        "자비스 속도 상태",
        "채팅 느려",
        "채팅 속도 설정",
        "채팅 튜닝",
    ]
    wrong_fragments = [
        "integration status",
        "brain search",
        "planner gap",
        "harness build slice",
        "jarvis status",
        "fallback chat",
        "queued as approval",
    ]
    for phrase in cases:
        out = suggest_command(phrase)
        if not out or "model routing status" not in out:
            raise SystemExit(f"model/brain intent should suggest model routing status: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(f"model/brain intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}")


def test_model_brain_failure_intents_redirect_to_model_routing_status() -> None:
    cases = [
        "model failed",
        "model not working",
        "local model error",
        "ollama failed",
        "ollama not working",
        "llm problem",
        "brain broken",
        "chat not working",
        "chat model failed",
        "planner failed",
        "planner model not working",
        "why did model fail",
        "why did ollama fail",
        "why did planner model fail",
    ]
    wrong_fragments = ["channel health", "recent tool runs", "planner gap", "queued as approval"]
    for phrase in cases:
        out = suggest_command(phrase)
        if not out or "model routing status" not in out:
            raise SystemExit(f"model/brain failure intent should suggest model routing status: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(f"model/brain failure intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}")


def test_eval_verification_intents_redirect_to_read_only_proof_commands() -> None:
    cases = {
        "smoke status": "cockpit",
        "smoke test status": "cockpit",
        "aggregate smoke status": "cockpit",
        "full smoke status": "cockpit",
        "latest aggregate smoke proof": "cockpit",
        "smoke suite status": "cockpit",
        "suite lock status": "cockpit",
        "smoke lock status": "cockpit",
        "smoke suite lock status": "cockpit",
        "acceptance harness status": "cockpit",
        "smoke coverage": "cockpit",
        "test status": "cockpit",
        "test coverage": "cockpit",
        "are tests green": "cockpit",
        "are smokes green": "cockpit",
        "eval status": "cockpit",
        "eval pack": "cockpit",
        "eval pack plan": "cockpit",
        "eval matrix": "cockpit",
        "operator evals": "cockpit",
        "operator eval pack": "cockpit",
        "operator eval pack plan": "cockpit",
        "operator real workflow tests": "cockpit",
        "operator trust tests": "cockpit",
        "operator workflow evals": "cockpit",
        "operator workflow proof": "cockpit",
        "operator workflow tests": "cockpit",
        "operator real tasks": "cockpit",
        "operator task evals": "cockpit",
        "operator workflows": "cockpit",
        "real operator workflows": "cockpit",
        "real operator workflow tests": "cockpit",
        "evaluation status": "cockpit",
        "workflow evals": "cockpit",
        "daily workflow evals": "cockpit",
        "workflow proof status": "cockpit",
        "workflow tests": "cockpit",
        "workflow status": "cockpit",
        "trust tests": "cockpit",
        "trust test status": "cockpit",
        "test matrix": "cockpit",
        "test matrix status": "cockpit",
        "tests pending": "cockpit",
        "live test matrix": "cockpit",
        "live test status": "cockpit",
        "live test results": "cockpit",
        "live matrix status": "cockpit",
        "live proof matrix": "cockpit",
        "live proof results": "cockpit",
        "live proof test status": "cockpit",
        "pending live tests": "cockpit",
        "pending live proofs": "cockpit",
        "channel proof status": "cockpit",
        "channel proof matrix": "cockpit",
        "what live tests are pending": "cockpit",
        "what proof is pending": "cockpit",
        "what proofs are still pending": "cockpit",
        "what live results do you need from me": "cockpit",
        "what results do you need from me": "cockpit",
        "what live proof do you need": "cockpit",
        "how do i report live test results": "cockpit",
        "how should i report live test results": "cockpit",
        "how should i report the live matrix": "cockpit",
        "how to report live test results": "cockpit",
        "live test result format": "cockpit",
        "live test results format": "cockpit",
        "live matrix result format": "cockpit",
        "report live matrix results": "cockpit",
        "report live test results": "cockpit",
        "what format should i use for live test results": "cockpit",
        "what should i send after testing": "cockpit",
        "what should i test": "cockpit",
        "what tests should i run": "cockpit",
        "how do i prove calendar write": "cockpit",
        "how do i test calendar write": "cockpit",
        "how do i prove email send": "cockpit",
        "how do i test email read": "cockpit",
        "how do i prove reminders": "cockpit",
        "how do i test reminders": "cockpit",
        "how do i prove contact lookup": "cockpit",
        "how do i test contact lookup": "cockpit",
        "how do i prove Korean voice": "cockpit",
        "how do i test Telegram voice": "cockpit",
        "how do i prove reboot survival": "cockpit",
        "how do i test daemon recovery": "cockpit",
        "how do i prove morning brief": "cockpit",
        "how do i test morning brief": "cockpit",
        "how do i prove phone approval": "cockpit",
        "how do i test approval flow": "cockpit",
        "what should operator test": "cockpit",
        "what has been tested": "cockpit",
        "what real workflows are tested": "cockpit",
        "what real workflows are covered": "cockpit",
        "what real workflows should Jarvis prove": "cockpit",
        "what should Jarvis prove live": "cockpit",
        "are operator workflows covered": "cockpit",
        "can Jarvis safely send Korean messages": "cockpit",
        "did telegram pass": "cockpit",
        "is telegram proven": "cockpit",
        "telegram proof passed": "cockpit",
        "did kakao pass": "cockpit",
        "is kakao proven": "cockpit",
        "did imessage pass": "cockpit",
        "is imessage proven": "cockpit",
        "did morning brief pass": "cockpit",
        "is morning brief proven": "cockpit",
        "did calendar write pass": "cockpit",
        "is calendar write proven": "cockpit",
        "did email send pass": "cockpit",
        "is email send proven": "cockpit",
        "did reminders pass": "cockpit",
        "is contact lookup proven": "cockpit",
        "did voice pass": "cockpit",
        "is local talk proven": "cockpit",
        "is Korean voice proven": "cockpit",
        "did reboot proof pass": "cockpit",
        "is reboot proven": "cockpit",
        "Hangul message proof": "cockpit",
        "Hangul send proof": "cockpit",
        "is Hangul messaging covered": "cockpit",
        "is Korean message covered": "cockpit",
        "is Korean messaging covered": "cockpit",
        "is Korean messaging tested": "cockpit",
        "is Korean telegram covered": "cockpit",
        "korean message eval": "cockpit",
        "korean message proof": "cockpit",
        "korean message test": "cockpit",
        "korean send proof": "cockpit",
        "korean telegram eval": "cockpit",
        "korean telegram proof": "cockpit",
        "korean telegram test": "cockpit",
        "morning brief eval": "cockpit",
        "contact lookup eval": "cockpit",
        "markets eval": "cockpit",
        "phone eval pack": "cockpit",
        "phone control evals": "cockpit",
        "what is eval pack": "cockpit",
        "what is the eval pack": "cockpit",
        "what is the operator eval pack": "cockpit",
        "proof matrix": "cockpit",
        "show proof matrix": "cockpit",
        "calendar proof status": "cockpit",
        "calendar status": "cockpit",
        "calendar live proof status": "cockpit",
        "calendar write proof status": "cockpit",
        "calendar create update delete proof": "cockpit",
        "email status": "cockpit",
        "email read status": "cockpit",
        "email search status": "cockpit",
        "email send status": "cockpit",
        "email proof status": "cockpit",
        "reminder status": "cockpit",
        "reminders status": "cockpit",
        "reminder proof status": "cockpit",
        "set reminder proof status": "cockpit",
        "contacts status": "cockpit",
        "contacts proof status": "cockpit",
        "contact lookup status": "cockpit",
        "contact lookup proof status": "cockpit",
        "personal proof status": "cockpit",
        "personal proofs status": "cockpit",
        "what personal integration proofs are missing": "cockpit",
        "phone approval status": "cockpit",
        "approval flow status": "cockpit",
        "cockpit status": "cockpit",
        "error guidance status": "cockpit",
        "korean voice status": "cockpit",
        "telegram voice status": "cockpit",
        "what broke status": "recent tool runs",
        "verification matrix": "cockpit",
        "regression status": "cockpit",
        "quality gate": "cockpit",
        "release gate": "cockpit",
        "what have we verified": "cockpit",
        "what is verified": "cockpit",
        "voice proof status": "cockpit",
        "is Korean voice tested": "cockpit",
        "telegram voice proof status": "cockpit",
        "korean voice proof status": "cockpit",
        "verification status": "verification receipt latest",
        "verification proof": "verification receipt latest",
        "proof status": "verification receipt latest",
        "what was verified": "verification receipt latest",
        "latest verification": "verification receipt latest",
        "last verification": "verification receipt latest",
        "evidence": "evidence ledger",
        "proof": "evidence ledger",
        "proofs": "evidence ledger",
        "show proof": "evidence ledger",
        "show proofs": "evidence ledger",
        "show me proof": "evidence ledger",
        "show me proofs": "evidence ledger",
        "show me evidence": "evidence ledger",
        "show the proof": "evidence ledger",
        "show the proofs": "evidence ledger",
        "show the evidence": "evidence ledger",
        "what proof do you have": "evidence ledger",
        "what proof do we have": "evidence ledger",
        "what proofs do we have": "evidence ledger",
        "what proof is there": "evidence ledger",
        "show evidence": "evidence ledger",
        "what evidence do you have": "evidence ledger",
        "what evidence do we have": "evidence ledger",
        "do you have evidence": "evidence ledger",
        "evidence status": "evidence ledger",
        "evidence report": "evidence ledger",
        "completion evidence": "evidence ledger",
        "trust evidence": "evidence ledger",
        "latest proof": "verification receipt latest",
        "latest verification proof": "verification receipt latest",
        "last proof": "verification receipt latest",
        "자비스 신뢰 테스트": "cockpit",
        "신뢰 테스트 상태": "cockpit",
        "제임스 워크플로우 테스트": "cockpit",
        "실제 워크플로우 평가": "cockpit",
        "한국어 메시지 평가": "cockpit",
        "한국어 메시지 검증": "cockpit",
        "한국어 메시지 증명": "cockpit",
        "한국어 메시지 테스트됐어": "cockpit",
        "한국어 텔레그램 평가": "cockpit",
        "한국어 텔레그램 검증": "cockpit",
        "한국어 텔레그램 안전해": "cockpit",
        "한국어 텔레그램 증명": "cockpit",
        "한글 메시지 증명": "cockpit",
        "한글 전송 증명": "cockpit",
        "가상연락처이 메시지 테스트": "cockpit",
        "가상연락처이 전송 증명": "cockpit",
        "가상연락처이 전송 테스트": "cockpit",
        "모닝브리핑 평가": "cockpit",
        "폰 컨트롤 평가": "cockpit",
        "음성 증명 상태": "cockpit",
        "한국어 음성 증명": "cockpit",
        "인수 상태": "cockpit",
        "테스트 초록": "cockpit",
        "증명 매트릭스": "cockpit",
        "검증 매트릭스": "cockpit",
        "캘린더 쓰기 어떻게 증명해": "cockpit",
        "캘린더 쓰기 어떻게 테스트해": "cockpit",
        "이메일 보내기 어떻게 증명해": "cockpit",
        "이메일 읽기 어떻게 테스트해": "cockpit",
        "리마인더 어떻게 증명해": "cockpit",
        "리마인더 어떻게 테스트해": "cockpit",
        "연락처 조회 어떻게 증명해": "cockpit",
        "연락처 조회 어떻게 테스트해": "cockpit",
        "한국어 음성 어떻게 증명해": "cockpit",
        "텔레그램 음성 어떻게 테스트해": "cockpit",
        "재부팅 생존 어떻게 증명해": "cockpit",
        "데몬 복구 어떻게 테스트해": "cockpit",
        "아침 브리핑 어떻게 증명해": "cockpit",
        "아침 브리핑 어떻게 테스트해": "cockpit",
        "폰 승인 어떻게 증명해": "cockpit",
        "승인 흐름 어떻게 테스트해": "cockpit",
    }
    wrong_fragments = [
        "storage status",
        "goal 1 status",
        "integration status",
        "verification packet",
        "chat context",
        "what goals do I have",
        "what goals do i have",
    ]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"eval/verification intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"eval/verification intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_acceptance_finish_line_intents_redirect_to_read_only_cockpit() -> None:
    cases = {
        "august checklist": "cockpit",
        "acceptance coverage": "cockpit",
        "acceptance coverage drift": "cockpit",
        "acceptance coverage status": "cockpit",
        "acceptance harness rows": "cockpit",
        "definition of done": "cockpit",
        "does live_check cover every checklist section": "cockpit",
        "does live_check cover the August checklist": "cockpit",
        "does live_check cover acceptance": "cockpit",
        "finish plan": "cockpit",
        "is live_check missing rows": "cockpit",
        "live_check coverage": "cockpit",
        "live check coverage": "cockpit",
        "live_check rows": "cockpit",
        "live_check missing rows": "cockpit",
        "one screen live_check table": "cockpit",
        "one screen acceptance table": "cockpit",
        "show one screen acceptance table": "cockpit",
        "show finish plan": "cockpit",
        "what is the finish line": "cockpit",
        "what is the August finish line": "cockpit",
        "what remains before Jarvis is done": "cockpit",
        "what remains to finish Jarvis": "cockpit",
        "what's not finished in Jarvis": "cockpit",
        "what is left before August": "cockpit",
        "what is left for August": "cockpit",
        "what's left to finish": "cockpit",
        "what is missing": "cockpit",
        "what is not done": "cockpit",
        "what is still missing": "cockpit",
        "what is still open": "cockpit",
        "what is unfinished": "cockpit",
        "what live proofs are missing": "cockpit",
        "what live tests should I run": "cockpit",
        "what live proofs should I run": "cockpit",
        "what should operator test live": "cockpit",
        "what should I verify live": "cockpit",
        "what open acceptance items remain": "cockpit",
        "what DoD items are open": "cockpit",
        "what acceptance tests are pending": "cockpit",
        "what acceptance proofs are pending": "cockpit",
        "what acceptance proof is next": "completion next proof",
        "what do I need to prove": "cockpit",
        "what should I prove next": "completion next proof",
        "what should I test next": "completion next proof",
        "what proof should I run next": "completion next proof",
        "next live proof": "completion next proof",
        "next live test": "completion next proof",
        "which proof is next": "completion next proof",
        "what is the next acceptance proof": "completion next proof",
        "what should operator test next": "completion next proof",
        "next acceptance test": "completion next proof",
        "what does the operator need to test": "cockpit",
        "what rows are in live_check": "cockpit",
        "what rows does live_check show": "cockpit",
        "what should the operator prove": "cockpit",
        "open checklist items": "cockpit",
        "remaining checklist items": "cockpit",
        "what remains unfinished": "cockpit",
        "unverified checklist items": "cockpit",
        "unfinished Jarvis work": "cockpit",
        "완료 기준": "cockpit",
        "검수 체크리스트": "cockpit",
        "남은 기준": "cockpit",
        "라이브체크 커버리지": "cockpit",
        "라이브 체크 표": "cockpit",
        "라이브체크 항목": "cockpit",
        "원스크린 인수 표": "cockpit",
        "인수 커버리지": "cockpit",
        "뭐 남았어": "cockpit",
        "다음 증명": "completion next proof",
        "다음 라이브 테스트": "completion next proof",
        "다음에 뭐 테스트해": "completion next proof",
        "다음에 뭐 증명해": "completion next proof",
        "뭐 증명해야 해": "cockpit",
    }
    wrong_fragments = ["wikipedia", "dictionary", "completion claim gate", "model routing status", "queued as approval"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"acceptance finish-line intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out.lower():
                raise SystemExit(
                    f"acceptance finish-line intent suggested stale/wrong command "
                    f"{wrong!r}: {phrase!r} -> {out!r}"
                )


def test_cockpit_attention_intents_redirect_to_read_only_cockpit() -> None:
    cases = {
        "cockpit attention": "cockpit",
        "attention": "cockpit",
        "attention status": "cockpit",
        "what needs attention": "cockpit",
        "what needs my attention": "cockpit",
        "what needs review": "cockpit",
        "which lanes need attention": "cockpit",
        "주의 상태": "cockpit",
        "주의 필요": "cockpit",
        "검토 필요": "cockpit",
        "콕핏 주의": "cockpit",
    }
    wrong_fragments = ["readiness report", "integration status", "chat context", "queued as approval"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"cockpit attention intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out.lower():
                raise SystemExit(
                    f"cockpit attention intent suggested stale/wrong command "
                    f"{wrong!r}: {phrase!r} -> {out!r}"
                )


def test_cockpit_lane_intents_redirect_to_read_only_cockpit() -> None:
    cases = {
        "messages lane": "cockpit",
        "messaging lane": "cockpit",
        "calls lane": "cockpit",
        "morning brief lane": "cockpit",
        "briefing lane": "cockpit",
        "contacts lane": "cockpit",
        "calendar email lane": "cockpit",
        "personal lane": "cockpit",
        "personal integrations lane": "cockpit",
        "personal proofs lane": "cockpit",
        "schedule lane": "cockpit",
        "scheduled jobs lane": "cockpit",
        "weather lane": "cockpit",
        "research lane": "cockpit",
        "memory lane": "cockpit",
        "learning lane": "cockpit",
        "diagnostics lane": "cockpit",
        "approvals lane": "cockpit",
        "workflow evals lane": "cockpit",
        "worker lane": "cockpit",
        "voice lane": "cockpit",
        "메시지 레인": "cockpit",
        "통화 레인": "cockpit",
        "브리핑 레인": "cockpit",
        "연락처 레인": "cockpit",
        "캘린더 이메일 레인": "cockpit",
        "개인 증명 레인": "cockpit",
        "스케줄 레인": "cockpit",
        "날씨 레인": "cockpit",
        "연구 레인": "cockpit",
        "기억 레인": "cockpit",
        "학습 레인": "cockpit",
        "진단 레인": "cockpit",
        "승인 레인": "cockpit",
        "워크플로우 평가 레인": "cockpit",
        "워커 레인": "cockpit",
        "음성 레인": "cockpit",
    }
    wrong_fragments = [
        "daily briefing",
        "list scheduled jobs",
        "schedule today",
        "channel health",
        "call fixture",
        "weather in",
        "research ",
        "readiness report",
        "queued as approval",
    ]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"cockpit lane intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out.lower():
                raise SystemExit(
                    f"cockpit lane intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_memory_learning_intents_redirect_to_read_only_memory_commands() -> None:
    cases = {
        "memory status": "memory stats",
        "memory health": "memory stats",
        "memory report": "memory stats",
        "brain memory status": "memory stats",
        "knowledge status": "memory stats",
        "memory review": "learning review",
        "review memories": "learning review",
        "review memory": "learning review",
        "what did you learn": "learning review",
        "what have you learned": "learning review",
        "what should jarvis learn": "learning review",
        "learning status": "learning review",
        "learning loop status": "learning review",
        "learning health": "learning review",
        "learning report": "learning review",
        "learning review status": "learning review",
        "after action learning status": "after-action learning packet",
        "after-action learning status": "after-action learning packet",
        "what did Jarvis learn from the last failure": "after-action learning packet",
        "what did Jarvis learn from last run": "after-action learning packet",
        "is learning debt closed": "execution learning closure",
        "learning debt status": "execution learning closure",
        "what learning debt is open": "execution learning closure",
        "learning loop proof": "execution learning closure",
        "learning proof matrix": "execution learning closure",
        "recovery closure status": "recovery closure checklist",
        "recovery learning status": "recovery closure checklist",
        "is recovery debt closed": "recovery closure checklist",
        "repeated failure status": "repeated failure clusters",
        "failure learning status": "failure learning cockpit",
    }
    wrong_fragments = ["morning startup", "brain search", "integration status"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"memory/learning intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"memory/learning intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_guardrail_and_freeze_intents_redirect_to_cockpit() -> None:
    for phrase in [
        "guardrails",
        "show guardrails",
        "build guardrails",
        "freeze status",
        "freeze list",
        "frozen files",
        "live freeze list",
        "live proof freeze",
        "live proof status",
        "live channel proof status",
        "what is frozen",
        "what files are frozen",
        "what is the freeze list",
        "what is the live test matrix",
        "what live tests are pending",
        "what proof is pending",
        "what proofs are still pending",
        "what live results do you need from me",
        "what results do you need from me",
        "what live proof do you need",
        "what should i test",
        "what tests should i run",
        "what channels should i test",
        "which channels need live proof",
        "what live channels are pending",
        "what should operator test",
        "can codex edit planner",
        "can codex edit the send code",
        "can you edit the planner",
        "can you edit the send code",
        "what can codex touch",
        "what can you edit",
        "what should codex avoid",
        "what should codex leave alone",
        "what should codex not edit",
        "what should codex not touch",
        "what should you not edit",
        "what should you not touch",
        "why frozen",
        "safe lane",
        "codex guardrails",
        "proofs pending",
        "가드레일",
        "프리즈 상태",
        "동결 파일",
        "수정 금지 파일",
        "뭐 테스트해야 해",
        "라이브 테스트 뭐 해야 해",
        "검증 대기 뭐야",
        "테스트 매트릭스 보여줘",
    ]:
        out = suggest_command(phrase)
        if not out or "cockpit" not in out:
            raise SystemExit(f"guardrail/freeze intent should suggest cockpit: {phrase!r} -> {out!r}")


def test_tool_control_plane_intents_redirect_to_read_only_commands() -> None:
    cases = {
        "tool status": "cockpit",
        "tools status": "cockpit",
        "tool health": "cockpit",
        "tools health": "cockpit",
        "tool report": "cockpit",
        "tools report": "cockpit",
        "available tool status": "cockpit",
        "available tools": "list tools",
        "show available tools": "list tools",
        "tool list": "list tools",
        "tools list": "list tools",
        "registered tools": "capability map",
        "tool registry": "capability map",
        "tool registry status": "cockpit",
        "list capabilities": "capability map",
        "capability list": "capability map",
        "capabilities list": "capability map",
        "available capabilities": "capability map",
        "show capabilities": "capability map",
        "capabilities cockpit": "cockpit",
        "capability dashboard": "cockpit",
        "capabilities dashboard": "cockpit",
        "jarvis cockpit": "cockpit",
        "control plane": "cockpit",
        "control plane health": "cockpit",
        "control plane status": "cockpit",
        "control status": "cockpit",
        "open cockpit": "cockpit",
        "open control plane": "cockpit",
        "show me control plane": "cockpit",
        "show me dashboard status": "cockpit",
        "show me the control panel": "cockpit",
        "show me the control plane": "cockpit",
        "show control plane": "cockpit",
        "jarvis control plane": "cockpit",
        "control panel": "cockpit",
        "jarvis control panel": "cockpit",
        "dashboard status": "cockpit",
        "show dashboard status": "cockpit",
        "trust dashboard": "cockpit",
        "safety dashboard": "safety status",
        "risk dashboard": "risk matrix",
        "show risk levels": "risk matrix",
        "what is the risk level": "risk matrix",
        "what tools are available": "list tools",
        "what tools do you have": "list tools",
        "what can your tools do": "capability map",
        "what can jarvis actually do": "capability map",
        "what is jarvis capable of": "capability map",
        "what can you actually do": "capability map",
        "what can u do": "capability map",
        "capability report": "cockpit",
        "capability cockpit status": "cockpit",
        "where is control plane": "cockpit",
        "where is the control plane": "cockpit",
        "도구 상태": "cockpit",
        "툴 상태": "cockpit",
        "도구 헬스": "cockpit",
        "툴 보고서": "cockpit",
        "도구 목록": "list tools",
        "도구 리스트": "list tools",
        "툴 목록": "list tools",
        "툴 리스트": "list tools",
        "사용 가능한 도구": "list tools",
        "가능한 도구": "list tools",
        "어떤 도구 있어": "list tools",
        "도구 확인": "capability map",
        "툴 레지스트리": "capability map",
        "기능 리스트": "capability map",
        "가능한 기능": "capability map",
        "컨트롤 플레인": "cockpit",
        "컨트롤패널": "cockpit",
        "자비스 컨트롤패널": "cockpit",
        "상태 대시보드": "cockpit",
        "능력 대시보드": "cockpit",
        "기능 대시보드": "cockpit",
        "기능 상태": "capability map",
        "능력 상태": "capability map",
        "안전 대시보드": "safety status",
        "리스크 매트릭스": "risk matrix",
        "위험 수준 보여줘": "risk matrix",
    }
    wrong_fragments = [
        "goal 1 status",
        "tool search",
        "integration status",
        "storage status",
        "tool_detail",
        "todo list",
        "terminal dashboard",
        "jarvis doctor",
        "computer control status",
    ]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"tool/control-plane intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"tool/control-plane intent suggested stale/wrong command "
                    f"{wrong!r}: {phrase!r} -> {out!r}"
                )


def test_health_diagnostic_intents_redirect_to_read_only_status_commands() -> None:
    cases = {
        "status report": "cockpit",
        "system status": "jarvis status",
        "system health": "jarvis doctor",
        "jarvis health": "jarvis doctor",
        "jarvis report": "jarvis status",
        "health report": "execution health report",
        "health status": "cockpit",
        "diagnostics": "jarvis doctor",
        "diagnostic status": "jarvis doctor",
        "diagnostics status": "jarvis doctor",
        "diagnostic report": "jarvis doctor",
        "diagnostics report": "jarvis doctor",
        "error status": "jarvis doctor",
        "error report": "jarvis doctor",
        "failure status": "execution health report",
        "failed status": "execution health report",
        "failures": "execution health report",
        "show failures": "execution health report",
        "what just failed": "execution health report",
        "what is healthy": "cockpit",
        "what is unhealthy": "execution health report",
        "is jarvis healthy": "jarvis doctor",
        "is everything healthy": "cockpit",
        "overall health": "cockpit",
        "overall status": "cockpit",
        "전체 상태": "cockpit",
        "전체 헬스": "cockpit",
        "시스템 상태": "jarvis status",
        "시스템 헬스": "jarvis doctor",
        "자비스 헬스": "jarvis doctor",
        "건강 상태": "cockpit",
        "진단 상태": "jarvis doctor",
        "진단 보고서": "jarvis doctor",
    }
    wrong_fragments = ["storage status", "goal 1 status", "wiki", "wikipedia", "tool search", "show alarms", "memory stats"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"health/diagnostic intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"health/diagnostic intent suggested stale/wrong command "
                    f"{wrong!r}: {phrase!r} -> {out!r}"
                )


def test_usage_example_intents_redirect_to_read_only_discovery_commands() -> None:
    cases = {
        "examples": "capability map",
        "example commands": "capability map",
        "sample commands": "capability map",
        "starter commands": "capability map",
        "command examples": "capability map",
        "jarvis examples": "capability map",
        "show examples": "capability map",
        "show commands": "capability map",
        "show shortcuts": "capability map",
        "shortcuts": "capability map",
        "what can i try": "capability map",
        "what should i try": "capability map",
        "how do i use jarvis": "capability map",
        "how to use jarvis": "capability map",
        "jarvis usage": "capability map",
        "usage": "capability map",
        "teach me jarvis": "capability map",
        "beginner commands": "capability map",
        "daily commands": "capability map",
        "common commands": "capability map",
        "owner commands": "capability map",
        "message examples": "capability map",
        "messaging examples": "capability map",
        "send examples": "capability map",
        "sending examples": "capability map",
        "kakao examples": "capability map",
        "imessage examples": "capability map",
        "instagram examples": "capability map",
        "telegram message examples": "capability map",
        "call examples": "capability map",
        "calling examples": "capability map",
        "phone call examples": "capability map",
        "contact examples": "capability map",
        "contacts examples": "capability map",
        "contact lookup examples": "capability map",
        "address book examples": "capability map",
        "voice examples": "voice command cockpit",
        "voice commands": "voice command cockpit",
        "show messaging examples": "capability map",
        "show voice commands": "voice command cockpit",
        "show me capabilities": "capability map",
        "show me the capability map": "capability map",
        "show me what you can do": "capability map",
        "what can i send": "capability map",
        "what can i call": "capability map",
        "what can i ask": "capability map",
        "what can jarvis do": "capability map",
        "what can you do": "capability map",
        "message capabilities": "capability map messages",
        "messaging capabilities": "capability map messages",
        "send capabilities": "capability map messages",
        "kakao capabilities": "capability map messages",
        "telegram capabilities": "capability map messages",
        "imessage capabilities": "capability map messages",
        "instagram capabilities": "capability map messages",
        "call capabilities": "capability map messages",
        "phone call capabilities": "capability map messages",
        "contact capabilities": "capability map messages",
        "contacts capabilities": "capability map messages",
        "contact lookup capabilities": "capability map messages",
        "address book capabilities": "capability map messages",
        "help with contacts": "capability map messages",
        "help with contact lookup": "capability map messages",
        "help with address book": "capability map messages",
        "what can you do with contacts": "capability map messages",
        "what can Jarvis do with contacts": "capability map messages",
        "what can you do with contact lookup": "capability map messages",
        "what can Jarvis do with address book": "capability map messages",
        "contact commands": "capability map",
        "contacts commands": "capability map",
        "contact lookup commands": "capability map",
        "address book commands": "capability map",
        "voice capabilities": "capability map voice",
        "voice command capabilities": "capability map voice",
        "calendar capabilities": "capability map productivity",
        "email capabilities": "capability map productivity",
        "weather capabilities": "capability map info",
        "news capabilities": "capability map info",
        "market capabilities": "capability map markets",
        "markets capabilities": "capability map markets",
        "stock capabilities": "capability map markets",
        "crypto capabilities": "capability map markets",
        "translation capabilities": "capability map utilities",
        "utility capabilities": "capability map utilities",
        "research capabilities": "capability map research",
        "writing capabilities": "capability map writing",
        "task capabilities": "capability map tasks",
        "note capabilities": "capability map notes",
        "memory capabilities": "capability map memory",
        "approval capabilities": "capability map approvals",
        "safety capabilities": "capability map safety",
        "dashboard capabilities": "cockpit",
        "help with messages": "capability map messages",
        "help with voice": "capability map voice",
        "phone examples": "capability map",
        "telegram examples": "capability map",
        "예시": "capability map",
        "명령 예시": "capability map",
        "명령어 예시": "capability map",
        "샘플 명령어": "capability map",
        "자비스 사용법": "capability map",
        "사용법": "capability map",
        "도움말": "help",
        "자비스 도움말": "help",
        "뭐부터 해": "capability map",
        "무엇부터 해": "capability map",
        "시작 명령어": "capability map",
        "자주 쓰는 명령어": "capability map",
        "메시지 예시": "capability map",
        "메세지 예시": "capability map",
        "문자 예시": "capability map",
        "전송 예시": "capability map",
        "카카오 예시": "capability map",
        "카톡 예시": "capability map",
        "아이메시지 예시": "capability map",
        "인스타그램 예시": "capability map",
        "전화 예시": "capability map",
        "통화 예시": "capability map",
        "음성 예시": "voice command cockpit",
        "음성 명령어": "voice command cockpit",
        "뭘 보낼 수 있어": "capability map",
        "뭘 물어볼 수 있어": "capability map",
        "메시지 기능": "capability map messages",
        "메세지 기능": "capability map messages",
        "문자 기능": "capability map messages",
        "전송 기능": "capability map messages",
        "연락처 기능": "capability map messages",
        "연락처 찾기 기능": "capability map messages",
        "연락처 검색 기능": "capability map messages",
        "연락처 조회 기능": "capability map messages",
        "주소록 기능": "capability map messages",
        "주소록 검색 기능": "capability map messages",
        "주소록 조회 기능": "capability map messages",
        "연락처 명령어": "capability map",
        "연락처 검색 명령어": "capability map",
        "연락처 조회 명령어": "capability map",
        "주소록 명령어": "capability map",
        "주소록 검색 명령어": "capability map",
        "주소록 조회 명령어": "capability map",
        "연락처 예시": "capability map",
        "연락처 검색 예시": "capability map",
        "연락처 조회 예시": "capability map",
        "주소록 예시": "capability map",
        "주소록 검색 예시": "capability map",
        "주소록 조회 예시": "capability map",
        "카카오 기능": "capability map messages",
        "카톡 기능": "capability map messages",
        "텔레그램 기능": "capability map messages",
        "아이메시지 기능": "capability map messages",
        "인스타그램 기능": "capability map messages",
        "전화 기능": "capability map messages",
        "통화 기능": "capability map messages",
        "음성 기능": "capability map voice",
        "캘린더 기능": "capability map productivity",
        "일정 기능": "capability map productivity",
        "이메일 기능": "capability map productivity",
        "할일 기능": "capability map tasks",
        "날씨 기능": "capability map info",
        "뉴스 기능": "capability map info",
        "시장 기능": "capability map markets",
        "주식 기능": "capability map markets",
        "코인 기능": "capability map markets",
        "번역 기능": "capability map utilities",
        "검색 기능": "capability map research",
        "글쓰기 기능": "capability map writing",
        "메모 기능": "capability map notes",
        "기억 기능": "capability map memory",
        "승인 기능": "capability map approvals",
        "안전 기능": "capability map safety",
        "대시보드 기능": "cockpit",
        "폰 예시": "capability map",
        "텔레그램 예시": "capability map",
    }
    wrong_fragments = [
        "call_contact",
        "call_telegram",
        "queued as approval",
        "Safety receipt",
        "what should I do now",
        "what changed in Jarvis",
        "jarvis status",
        "dispatch decision",
        "send_imessage",
    ]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"usage/example intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"usage/example intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_daily_operator_intents_redirect_to_read_only_status_commands() -> None:
    cases = {
        "what broke": "recent tool runs",
        "what is broken": "recent tool runs",
        "anything failing": "execution health report",
        "failure details": "execution health report",
        "latest failure": "recent tool runs",
        "show last failure": "recent tool runs",
        "show me failures": "execution health report",
        "show me what broke": "recent tool runs",
        "show me what is broken": "recent tool runs",
        "what's wrong": "recent tool runs",
        "what errors happened": "recent tool runs",
        "what went wrong": "recent tool runs",
        "what needs recovery": "recovery closure checklist",
        "show me what failed": "execution health report",
        "last error": "recent tool runs",
        "last failed tool": "recent tool runs",
        "last failure": "recent tool runs",
        "latest error": "recent tool runs",
        "show last error": "recent tool runs",
        "what failed last": "recent tool runs",
        "what was the last error": "recent tool runs",
        "what was the last failure": "recent tool runs",
        "brief status": "list scheduled jobs",
        "morning brief status": "list scheduled jobs",
        "daily brief status": "list scheduled jobs",
        "channel status": "channel health",
        "channels status": "channel health",
        "what can i say": "capability map",
        "what should i say": "capability map",
        "how do i talk to jarvis": "capability map",
        "status cockpit": "cockpit",
        "cockpit summary": "cockpit",
        "jarvis cockpit summary": "cockpit",
        "control center": "cockpit",
        "jarvis control center": "cockpit",
        "control plane": "cockpit",
        "trust cockpit": "cockpit",
        "trust checklist": "cockpit",
        "earned trust": "cockpit",
        "earned trust checklist": "cockpit",
        "evidence status": "evidence ledger",
        "proofs": "evidence ledger",
        "daemon status": "cockpit",
        "daemon startup status": "cockpit",
        "launchd status": "cockpit",
        "network recovery status": "cockpit",
        "reboot status": "cockpit",
        "will jarvis survive reboot": "cockpit",
        "can i trust jarvis": "cockpit",
        "can i trust you": "cockpit",
        "can jarvis be trusted": "cockpit",
        "why can i trust jarvis": "cockpit",
        "why should i trust jarvis": "cockpit",
        "what makes jarvis reliable": "cockpit",
        "what makes jarvis trustworthy": "cockpit",
        "is jarvis trustworthy": "cockpit",
        "is jarvis reliable": "cockpit",
        "trust status": "cockpit",
        "trust report": "cockpit",
        "trust health": "cockpit",
        "reliability status": "cockpit",
        "reliability report": "cockpit",
        "reliability health": "cockpit",
        "show trust status": "cockpit",
        "show reliability status": "cockpit",
        "is jarvis safe to use": "safety status",
        "can i use jarvis safely": "safety status",
        "safe to use jarvis": "safety status",
        "what are your limits": "safety status",
        "what are jarvis limits": "safety status",
        "what are your boundaries": "safety status",
        "what are the safety boundaries": "safety status",
        "what are you allowed to do": "safety status",
        "what are you not allowed to do": "safety status",
        "what can you not do": "safety status",
        "what cant you do": "safety status",
        "what can jarvis not do": "safety status",
        "dashboard status": "cockpit",
        "capability health": "cockpit",
        "capability lanes": "cockpit",
        "what needs my attention": "cockpit",
        "what needs attention": "cockpit",
        "next cockpit action": "safe next actions",
        "cockpit next action": "safe next actions",
        "what should i run next": "safe next actions",
        "what command should i run next": "safe next actions",
        "what should i check next": "safe next actions",
        "attention needed": "readiness report",
        "ready for me": "readiness report",
        "what should i check": "readiness report",
        "blocked work": "execution health report",
        "blocked status": "execution health report",
        "can i retry": "recovery closure checklist",
        "can i retry now": "recovery closure checklist",
        "recovery status": "recovery closure checklist",
        "retry readiness": "recovery closure checklist",
        "retry status": "recovery closure checklist",
        "safe to retry": "recovery closure checklist",
        "should i retry": "recovery closure checklist",
        "recovery plan": "recovery closure checklist",
        "how do i recover": "recovery closure checklist",
    }
    wrong_fragments = ["what should Jarvis do next", "memory stats", "storage recovery plan", "capability map"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"daily operator intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out and wrong != expected:
                raise SystemExit(f"daily operator intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}")


def test_scheduler_automation_intents_redirect_to_read_only_job_status() -> None:
    cases = {
        "schedule status": "list scheduled jobs",
        "scheduler status": "list scheduled jobs",
        "schedule health": "list scheduled jobs",
        "schedule report": "list scheduled jobs",
        "job status": "list scheduled jobs",
        "jobs status": "list scheduled jobs",
        "scheduled jobs": "list scheduled jobs",
        "scheduled job status": "list scheduled jobs",
        "scheduled jobs status": "list scheduled jobs",
        "scheduled brief status": "list scheduled jobs",
        "what jobs are running": "list scheduled jobs",
        "what is scheduled": "list scheduled jobs",
        "what is the schedule": "list scheduled jobs",
        "what is the morning schedule": "list scheduled jobs",
        "what is running automatically": "list scheduled jobs",
        "what is automated": "list scheduled jobs",
        "when is morning brief": "list scheduled jobs",
        "when is next morning brief": "list scheduled jobs",
        "next morning brief": "list scheduled jobs",
        "morning brief schedule": "list scheduled jobs",
        "morning brief next run": "list scheduled jobs",
        "is morning brief scheduled": "list scheduled jobs",
        "is the brief scheduled": "list scheduled jobs",
        "automation status": "list scheduled jobs",
        "automation health": "list scheduled jobs",
        "automation report": "list scheduled jobs",
        "automations status": "list scheduled jobs",
        "what automations are running": "list scheduled jobs",
        "background jobs": "list scheduled jobs",
        "background job status": "list scheduled jobs",
        "recurring jobs": "list scheduled jobs",
        "recurring job status": "list scheduled jobs",
        "recurring tasks": "list scheduled jobs",
        "recurring task status": "list scheduled jobs",
    }
    wrong_fragments = ["schedule today", "what is on my schedule today", "integration status", "jarvis status", "morning startup"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"scheduler/automation intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"scheduler/automation intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_brief_failure_intents_redirect_to_recent_runs() -> None:
    cases = {
        "brief failed": "recent tool runs",
        "brief error": "recent tool runs",
        "brief issue": "recent tool runs",
        "brief problem": "recent tool runs",
        "brief missing": "recent tool runs",
        "brief didn't arrive": "recent tool runs",
        "brief did not arrive": "recent tool runs",
        "brief not delivered": "recent tool runs",
        "brief ran today": "recent tool runs",
        "brief sent today": "recent tool runs",
        "brief last run": "recent tool runs",
        "morning brief failed": "recent tool runs",
        "morning brief error": "recent tool runs",
        "morning brief issue": "recent tool runs",
        "morning brief problem": "recent tool runs",
        "morning brief missing": "recent tool runs",
        "morning brief didn't arrive": "recent tool runs",
        "morning brief did not send": "recent tool runs",
        "did morning brief send": "recent tool runs",
        "did the morning brief send": "recent tool runs",
        "did my morning brief run": "recent tool runs",
        "did brief run today": "recent tool runs",
        "was morning brief sent": "recent tool runs",
        "was morning brief delivered": "recent tool runs",
        "morning brief last run": "recent tool runs",
        "last morning brief": "recent tool runs",
        "did scheduler run": "recent tool runs",
        "scheduler ran today": "recent tool runs",
        "schedule ran today": "recent tool runs",
        "why did scheduler not run": "recent tool runs",
        "scheduler not running": "recent tool runs",
        "jobs not running": "recent tool runs",
        "where is my morning brief": "recent tool runs",
        "why did morning brief fail": "recent tool runs",
        "why didn't morning brief send": "recent tool runs",
        "why did morning brief not arrive": "recent tool runs",
    }
    wrong_fragments = ["list scheduled jobs", "morning startup", "daily brief", "jarvis status"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"brief failure intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"brief failure intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_messaging_channel_status_intents_redirect_to_read_only_channel_health() -> None:
    cases = {
        "messaging status": "channel health",
        "message status": "channel health",
        "messages status": "channel health",
        "message health": "channel health",
        "can i resend": "channel health",
        "can i send again": "channel health",
        "send status": "channel health",
        "sending status": "channel health",
        "send again": "channel health",
        "should i resend": "channel health",
        "should i send again": "channel health",
        "safe to resend": "channel health",
        "safe to send again": "channel health",
        "delivery status": "channel health",
        "message delivery status": "channel health",
        "message did not send": "channel health",
        "message didn't send": "channel health",
        "message didnt send": "channel health",
        "message not sent": "channel health",
        "message not delivered": "channel health",
        "message failed": "channel health",
        "message error": "channel health",
        "message issue": "channel health",
        "message problem": "channel health",
        "messaging issue": "channel health",
        "send failed": "channel health",
        "send did not work": "channel health",
        "send didn't work": "channel health",
        "send error": "channel health",
        "send problem": "channel health",
        "sending issue": "channel health",
        "call failed": "channel health",
        "call issue": "channel health",
        "call problem": "channel health",
        "phone call failed": "channel health",
        "phone call problem": "channel health",
        "telegram health": "channel health",
        "telegram report": "channel health",
        "telegram channel status": "channel health",
        "telegram failed": "channel health",
        "telegram error": "channel health",
        "telegram broken": "channel health",
        "telegram issue": "channel health",
        "telegram not working": "channel health",
        "telegram problem": "channel health",
        "telegram did not send": "channel health",
        "telegram didn't send": "channel health",
        "telegram didnt send": "channel health",
        "telegram message did not send": "channel health",
        "telegram message not sent": "channel health",
        "telegram message not delivered": "channel health",
        "telegram send failed": "channel health",
        "telegram send did not work": "channel health",
        "telegram send didn't work": "channel health",
        "retry telegram": "channel health",
        "resend telegram": "channel health",
        "send telegram again": "channel health",
        "try telegram again": "channel health",
        "telegram last failure": "channel health",
        "last telegram failure": "channel health",
        "telegram last success": "channel health",
        "telegram success count": "channel health",
        "telegram 7 day success count": "channel health",
        "did telegram send": "channel health",
        "did telegram go through": "channel health",
        "did my telegram go through": "channel health",
        "was telegram sent": "channel health",
        "was telegram delivered": "channel health",
        "did telegram work recently": "channel health",
        "what broke on telegram": "channel health",
        "why did telegram break": "channel health",
        "what happened to telegram": "channel health",
        "what happened with telegram": "channel health",
        "why did telegram fail": "channel health",
        "why did telegram send fail": "channel health",
        "why did telegram not send": "channel health",
        "why did telegram send not work": "channel health",
        "message delivery health": "channel health",
        "channel last failure": "channel health",
        "show channel failures": "channel health",
        "imessage status": "channel health",
        "imessage health": "channel health",
        "imessage failed": "channel health",
        "imessage error": "channel health",
        "imessage issue": "channel health",
        "imessage last failure": "channel health",
        "imessage success count": "channel health",
        "did imessage go through": "channel health",
        "was imessage sent": "channel health",
        "did imessage work recently": "channel health",
        "what broke on imessage": "channel health",
        "imessage did not send": "channel health",
        "imessage didn't send": "channel health",
        "imessage not sent": "channel health",
        "retry imessage": "channel health",
        "resend imessage": "channel health",
        "send imessage again": "channel health",
        "what happened to imessage": "channel health",
        "why did imessage fail": "channel health",
        "why did imessage not send": "channel health",
        "kakao status": "channel health",
        "kakao health": "channel health",
        "kakao failed": "channel health",
        "kakao error": "channel health",
        "kakao issue": "channel health",
        "kakao last failure": "channel health",
        "kakao success count": "channel health",
        "did kakao go through": "channel health",
        "was kakao delivered": "channel health",
        "did kakao work recently": "channel health",
        "what broke on kakao": "channel health",
        "kakao did not send": "channel health",
        "kakao didn't send": "channel health",
        "kakao not sent": "channel health",
        "retry kakao": "channel health",
        "resend kakao": "channel health",
        "send kakao again": "channel health",
        "kakao not working": "channel health",
        "kakao problem": "channel health",
        "what happened with kakao": "channel health",
        "why did kakao fail": "channel health",
        "why did kakao not send": "channel health",
        "instagram status": "channel health",
        "instagram health": "channel health",
        "instagram failed": "channel health",
        "instagram error": "channel health",
        "instagram issue": "channel health",
        "instagram last failure": "channel health",
        "instagram success count": "channel health",
        "did instagram go through": "channel health",
        "was instagram sent": "channel health",
        "did instagram work recently": "channel health",
        "what broke on instagram": "channel health",
        "instagram did not send": "channel health",
        "instagram didn't send": "channel health",
        "instagram not sent": "channel health",
        "retry instagram": "channel health",
        "resend instagram": "channel health",
        "send instagram again": "channel health",
        "what happened to instagram": "channel health",
        "why did instagram fail": "channel health",
        "why did instagram not send": "channel health",
        "why did call fail": "channel health",
        "why did phone call fail": "channel health",
        "why did send fail": "channel health",
        "did it send": "channel health",
        "did it go through": "channel health",
        "did my message send": "channel health",
        "did the message send": "channel health",
        "did my message go through": "channel health",
        "did the message go through": "channel health",
        "was it sent": "channel health",
        "was my message delivered": "channel health",
        "was the message delivered": "channel health",
    }
    wrong_fragments = [
        "storage status",
        "safety status",
        "integration status",
        "goal 1 status",
        "tool search",
        "dispatch decision",
        "pending approvals",
    ]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"messaging/channel status intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"messaging/channel status intent suggested stale/wrong command "
                    f"{wrong!r}: {phrase!r} -> {out!r}"
                )


def test_phone_control_intents_redirect_to_read_only_commands() -> None:
    cases = {
        "phone control center": "cockpit",
        "phone control center plan": "cockpit",
        "phone control status": "cockpit",
        "telegram control center": "cockpit",
        "telegram control status": "cockpit",
        "mobile control": "cockpit",
        "phone commands": "capability map",
        "phone help": "capability map",
        "telegram commands": "capability map",
        "telegram help": "capability map",
        "phone shortcuts": "capability map",
        "telegram shortcuts": "capability map",
        "what can i do from my phone": "capability map",
        "what can i do on telegram": "capability map",
        "what can i do in telegram": "capability map",
        "what commands work on telegram": "capability map",
        "what can i text jarvis": "capability map",
        "what can i send jarvis on telegram": "capability map",
        "what is phone control center": "cockpit",
        "what is the phone control center": "cockpit",
        "mobile status": "channel health",
        "owner phone status": "channel health",
        "phone status": "channel health",
        "phone health": "channel health",
        "telegram status": "channel health",
    }
    wrong_fragments = ["call_contact", "call_telegram", "approval", "what changed in Jarvis"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"phone-control intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"phone-control intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_audit_history_intents_redirect_to_read_only_audit_commands() -> None:
    cases = {
        "audit trail": "recent tool runs",
        "audit log": "recent tool runs",
        "show audit log": "recent tool runs",
        "show audit trail": "recent tool runs",
        "execution audit": "recent tool runs",
        "show execution audit": "recent tool runs",
        "tool log": "recent tool runs",
        "tool logs": "recent tool runs",
        "execution logs": "recent tool runs",
        "show logs": "recent tool runs",
        "recent logs": "recent tool runs",
        "recent runs": "recent tool runs",
        "last run": "recent tool runs",
        "last tool run": "recent tool runs",
        "last actions": "recent tool runs",
        "last thing you did": "recent tool runs",
        "latest run": "recent tool runs",
        "latest tool run": "recent tool runs",
        "latest actions": "recent tool runs",
        "recent tool run": "recent tool runs",
        "run history": "recent tool runs",
        "tool run history": "recent tool runs",
        "tool execution history": "recent tool runs",
        "recent actions": "recent tool runs",
        "recent activity": "recent tool runs",
        "runtime trace latest": "runtime trace receipt",
        "last execution receipt": "verification receipt latest",
        "latest execution receipt": "verification receipt latest",
        "what ran recently": "recent tool runs",
        "what ran last": "recent tool runs",
        "what tool ran last": "recent tool runs",
        "what did jarvis do": "recent tool runs",
        "what did jarvis run last": "recent tool runs",
        "what did you do": "recent tool runs",
        "what changed": "recent tool runs",
        "what changed recently": "recent tool runs",
        "what changed in code": "what changed in Jarvis",
        "what changed after codex": "what changed in Jarvis",
        "what did codex change": "what changed in Jarvis",
        "what did codex just change": "what changed in Jarvis",
        "what did codex just do": "what changed in Jarvis",
        "what did codex modify": "what changed in Jarvis",
        "what did you change": "what changed in Jarvis",
        "what did you just change": "what changed in Jarvis",
        "what did you just do": "what changed in Jarvis",
        "what files did codex edit": "what changed in Jarvis",
        "what files changed": "what changed in Jarvis",
        "which files changed": "what changed in Jarvis",
        "which files did you edit": "what changed in Jarvis",
        "what did you edit": "what changed in Jarvis",
        "show changed files": "what changed in Jarvis",
        "show modified files": "what changed in Jarvis",
        "code changes status": "what changed in Jarvis",
        "diff status": "what changed in Jarvis",
        "patch status": "what changed in Jarvis",
        "what should i review in this change": "what changed in Jarvis",
        "what should claude review": "what changed in Jarvis",
        "anything broken": "recent tool runs",
        "is anything broken": "recent tool runs",
        "why did it fail": "recent tool runs",
        "show receipts": "verification receipt latest",
        "latest receipt": "verification receipt latest",
        "last receipt": "verification receipt latest",
        "recent receipts": "verification receipt latest",
    }
    wrong_fragments = [
        "execution case closure",
        "what should Jarvis do next",
        "what do I need to do",
        "That sounds like it may involve action",
        "fallback chat",
        "next actions",
    ]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"audit/history intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(f"audit/history intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}")


def test_trust_boundary_intents_redirect_to_read_only_safety_commands() -> None:
    cases = {
        "safety": "safety status",
        "safety check": "safety status",
        "is jarvis safe": "safety status",
        "is this safe": "safety status",
        "safe to run": "safety status",
        "approval safety": "safety status",
        "what requires approval": "safety status",
        "what needs approval": "safety status",
        "what actions need approval": "safety status",
        "what commands need approval": "safety status",
        "what tools require approval": "safety status",
        "which tools require approval": "safety status",
        "what is approval gated": "safety status",
        "what is approval-gated": "safety status",
        "what needs my permission": "safety status",
        "what requires my permission": "safety status",
        "redos status": "frozen routing risk report",
        "regex dos status": "frozen routing risk report",
        "regex risk status": "frozen routing risk report",
        "regex safety status": "frozen routing risk report",
        "send call routing risk report": "frozen routing risk report",
        "what is the frozen redos risk": "frozen routing risk report",
        "show frozen routing risk": "frozen routing risk report",
        "planner redos finding": "frozen routing risk report",
        "planner input length guard": "planner input guard report",
        "input length guard status": "planner input guard report",
        "long input safety status": "planner input guard report",
        "long message redos status": "planner input guard report",
        "long command redos risk": "planner input guard report",
        "what is the input length guard decision": "planner input guard report",
        "privacy": "privacy report",
        "privacy status": "privacy report",
        "privacy check": "privacy report",
        "what data can you see": "privacy report",
        "what private data can you see": "privacy report",
        "risk": "risk matrix",
        "risk status": "risk matrix",
        "risk check": "risk matrix",
        "permissions": "risk matrix",
        "permission status": "risk matrix",
        "jarvis permissions": "risk matrix",
        "show my permissions": "risk matrix",
        "show jarvis permissions": "risk matrix",
        "which commands are risky": "risk matrix",
        "which actions are risky": "risk matrix",
        "what permissions do you need": "risk matrix",
        "what permissions do you have": "risk matrix",
        "what permissions does jarvis have": "risk matrix",
        "what permissions do i have": "risk matrix",
        "what are my permissions": "risk matrix",
        "what is high risk": "risk matrix",
        "which tools are high risk": "risk matrix",
        "high risk tools": "risk matrix",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"trust-boundary intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in ["setup check", "jarvis status", "approval history"]:
            if wrong in out:
                raise SystemExit(f"trust-boundary intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}")


def test_next_step_operator_intents_redirect_to_read_only_continuity_commands() -> None:
    cases = {
        "what next": "safe next actions",
        "what should i do next": "safe next actions",
        "next step": "safe next actions",
        "next action": "safe next actions",
        "safe next action": "safe next actions",
        "next safe action": "safe next actions",
        "what's queued": "work queue",
        "what work is queued": "work queue",
        "work queue status": "work queue",
        "work backlog": "work queue",
        "work list": "work queue",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"next-step/operator intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        if "chat context" in out:
            raise SystemExit(f"next-step/operator intent should not suggest vague chat wording: {phrase!r} -> {out!r}")


def test_korean_operator_intents_redirect_to_read_only_status_commands() -> None:
    cases = {
        "뭐 할 수 있어": "capability map",
        "뭐 할 수 있니": "capability map",
        "뭐 할 수 있는지 보여줘": "capability map",
        "뭘 할 수 있어": "capability map",
        "무엇을 할 수 있어": "capability map",
        "무슨 기능 있어": "capability map",
        "자비스, 뭐 할 수 있어?": "capability map",
        "사용 가능한 기능": "capability map",
        "기능 리스트": "capability map",
        "기능 알려줘": "capability map",
        "명령어 보여줘": "capability map",
        "명령어 알려줘": "capability map",
        "명령어 목록": "capability map",
        "뭐가 고장났어": "recent tool runs",
        "뭐가 문제야?": "recent tool runs",
        "무슨 문제 있어": "recent tool runs",
        "무슨 문제야": "recent tool runs",
        "문제 있어?": "recent tool runs",
        "문제 상태": "recent tool runs",
        "감사 기록": "recent tool runs",
        "감사 기록 보여줘": "recent tool runs",
        "실행 기록 보여줘": "recent tool runs",
        "최근에 뭐 했어": "recent tool runs",
        "마지막 실행": "recent tool runs",
        "최근 오류": "recent tool runs",
        "최근 에러": "recent tool runs",
        "마지막 오류": "recent tool runs",
        "마지막 에러": "recent tool runs",
        "고장났어?": "recent tool runs",
        "고장 상태": "recent tool runs",
        "뭐가 실패했어": "execution health report",
        "실패한 거 있어": "execution health report",
        "실패 상태": "execution health report",
        "실패 목록": "execution health report",
        "실패 보여줘": "execution health report",
        "최근 실패": "execution health report",
        "최근 실패 보여줘": "execution health report",
        "왜 실패했어": "recent tool runs",
        "오류 상태": "jarvis doctor",
        "복구 상태": "recovery closure checklist",
        "복구 계획": "recovery closure checklist",
        "재시도 상태": "recovery closure checklist",
        "재시도 준비": "recovery closure checklist",
        "재시도 해도 돼": "recovery closure checklist",
        "주의 필요": "cockpit",
        "주의 상태": "cockpit",
        "콕핏 주의": "cockpit",
        "확인 필요": "readiness report",
        "내가 봐야 할 거 있어": "readiness report",
        "대기 상태": "readiness report",
        "브리핑 상태": "list scheduled jobs",
        "브리핑 확인": "list scheduled jobs",
        "아침 브리핑 상태": "list scheduled jobs",
        "모닝브리프 상태": "list scheduled jobs",
        "모닝 브리핑 상태": "list scheduled jobs",
        "채널 상태": "channel health",
        "채널 확인": "channel health",
        "채널 헬스": "channel health",
        "상태 보여줘": "jarvis status",
        "상태 알려줘": "jarvis status",
        "상태 확인": "jarvis status",
        "자비스 상태 알려줘": "jarvis status",
        "컨트롤 센터": "cockpit",
        "컨트롤 플레인 보여줘": "cockpit",
        "콕핏 보여줘": "cockpit",
        "대시보드 보여줘": "cockpit",
        "대시보드 상태": "cockpit",
        "신뢰 상태": "cockpit",
        "신뢰 보고서": "cockpit",
        "신뢰도 상태": "cockpit",
        "신뢰도 보고서": "cockpit",
        "신뢰할 수 있어": "cockpit",
        "왜 자비스 믿어도 돼": "cockpit",
        "왜 자비스를 믿어도 돼": "cockpit",
        "자비스 신뢰 가능": "cockpit",
        "자비스 신뢰 상태": "cockpit",
        "믿어도 돼": "cockpit",
        "믿을만해": "cockpit",
        "안전하게 써도 돼": "safety status",
        "자비스 안전하게 써도 돼": "safety status",
        "무엇을 못해": "safety status",
        "뭐 못해": "safety status",
        "무엇을 할 수 없어": "safety status",
        "뭘 할 수 없어": "safety status",
        "자비스 한계": "safety status",
        "자비스 경계": "safety status",
        "폰 컨트롤 센터": "cockpit",
        "텔레그램 컨트롤 센터": "cockpit",
        "모바일 컨트롤": "cockpit",
        "폰 컨트롤 상태": "cockpit",
        "텔레그램 컨트롤 상태": "cockpit",
        "폰 명령어": "capability map",
        "폰 도움말": "capability map",
        "폰에서 뭐 할 수 있어": "capability map",
        "텔레그램 명령어": "capability map",
        "텔레그램 도움말": "capability map",
        "텔레그램에서 뭐 할 수 있어": "capability map",
        "폰 단축키": "capability map",
        "텔레그램 단축키": "capability map",
        "폰 상태": "channel health",
        "모바일 상태": "channel health",
        "텔레그램 상태": "channel health",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean operator intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_scheduler_automation_intents_redirect_to_read_only_job_status() -> None:
    cases = {
        "예약 작업": "list scheduled jobs",
        "예약 작업 상태": "list scheduled jobs",
        "스케줄 상태": "list scheduled jobs",
        "스케줄 확인": "list scheduled jobs",
        "스케줄러 상태": "list scheduled jobs",
        "자동화 상태": "list scheduled jobs",
        "자동화 확인": "list scheduled jobs",
        "백그라운드 작업": "list scheduled jobs",
        "백그라운드 작업 상태": "list scheduled jobs",
        "반복 작업": "list scheduled jobs",
        "반복 작업 상태": "list scheduled jobs",
        "아침 브리프 예약": "list scheduled jobs",
        "브리핑 예약": "list scheduled jobs",
        "브리핑 작업 상태": "list scheduled jobs",
        "브리핑 언제야": "list scheduled jobs",
        "다음 브리핑 언제": "list scheduled jobs",
        "모닝브리프 예약 확인": "list scheduled jobs",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean scheduler/automation intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_completion_and_agi_intents_redirect_to_read_only_gate_commands() -> None:
    cases = {
        "AGI 상태": "agi gates",
        "AGI 준비": "agi gates",
        "AGI 준비 상태": "agi gates",
        "AGI 게이트 상태": "agi gates",
        "에이지아이 상태": "agi gates",
        "에이지아이 준비 상태": "agi gates",
        "하네스 상태": "harness status",
        "하네스 준비": "harness readiness digest",
        "하네스 준비 상태": "harness readiness digest",
        "자비스 끝났어": "completion claim gate",
        "자비스 끝났니": "completion claim gate",
        "자비스 완료됐어": "completion claim gate",
        "자비스 완료상태": "completion claim gate",
        "자비스 완성됐어": "completion claim gate",
        "완료 상태": "completion audit",
        "완료 주장": "completion claim gate",
        "완료 게이트": "completion claim gate",
        "완료 증명": "completion next proof",
    }
    wrong_fragments = ["list jarvis notes", "action readiness", "jarvis status"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean completion/AGI intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"Korean completion/AGI intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_korean_progress_roadmap_goal_intents_redirect_to_read_only_control_commands() -> None:
    cases = {
        "AGI 진행상황": "build progress",
        "AGI 진행": "build progress",
        "빌드 상태": "build progress",
        "빌드 진행상황": "build progress",
        "자비스 진행상황": "build progress",
        "진행 보고": "build progress",
        "뭐 만들었어": "build progress",
        "로드맵 보여줘": "roadmap",
        "자비스 로드맵": "roadmap",
        "다음 로드맵": "roadmap",
        "다음에 뭐 만들어": "roadmap",
        "목표 상태": "list goals",
        "활성 목표": "list goals",
    }
    wrong_fragments = ["completion claim", "approval summary", "jarvis status", "fallback"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean progress/roadmap/goal intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"Korean progress/roadmap/goal intent suggested stale/wrong command {wrong!r}: "
                    f"{phrase!r} -> {out!r}"
                )


def test_exact_korean_goal_aliases_do_not_bypass_the_planner() -> None:
    for phrase in ("목표 보여줘", "목표보여줘", "목표 목록", "목표목록"):
        out = suggest_command(phrase)
        if out is not None:
            raise SystemExit(
                f"exact Korean goal alias should reach the planner directly, not a suggestion: "
                f"{phrase!r} -> {out!r}"
            )


def test_korean_messaging_channel_status_intents_redirect_to_read_only_channel_health() -> None:
    cases = {
        "메시지 상태": "channel health",
        "메세지 상태": "channel health",
        "메시징 상태": "channel health",
        "문자 상태": "channel health",
        "전송 상태": "channel health",
        "다시 보내도 돼": "channel health",
        "다시 보내도 되나요": "channel health",
        "다시 보낼까": "channel health",
        "재전송 상태": "channel health",
        "채널 마지막 실패": "channel health",
        "채널 성공 횟수": "channel health",
        "채널 실패 보여줘": "channel health",
        "아이메시지 상태": "channel health",
        "아이메시지 확인": "channel health",
        "아이메시지 마지막 실패": "channel health",
        "아이메시지 성공 횟수": "channel health",
        "카카오 상태": "channel health",
        "카카오 확인": "channel health",
        "카카오 마지막 실패": "channel health",
        "카카오 성공 횟수": "channel health",
        "카카오 갔어": "channel health",
        "카카오 보냈어": "channel health",
        "카카오 보내졌어": "channel health",
        "카카오 다시 보낼까": "channel health",
        "카카오 재시도": "channel health",
        "카카오 재전송": "channel health",
        "카톡 상태": "channel health",
        "카톡 확인": "channel health",
        "카톡 마지막 실패": "channel health",
        "카톡 성공 횟수": "channel health",
        "카톡 갔어": "channel health",
        "카톡 보냈어": "channel health",
        "카톡 보내졌어": "channel health",
        "카톡 다시 보낼까": "channel health",
        "카톡 재시도": "channel health",
        "카톡 재전송": "channel health",
        "카카오 실패": "channel health",
        "카카오 문제": "channel health",
        "카카오 이슈": "channel health",
        "왜 카카오 실패": "channel health",
        "카톡 오류": "channel health",
        "카톡 안됨": "channel health",
        "카톡 문제": "channel health",
        "텔레그램 헬스": "channel health",
        "텔레그램 확인": "channel health",
        "텔레그램 마지막 실패": "channel health",
        "텔레그램 최근 성공": "channel health",
        "텔레그램 성공 횟수": "channel health",
        "텔레그램 최근에 됐어": "channel health",
        "텔레그램 갔어": "channel health",
        "텔레그램 보냈어": "channel health",
        "텔레그램 보내졌어": "channel health",
        "텔레그램 다시 보낼까": "channel health",
        "텔레그램 재시도": "channel health",
        "텔레그램 재전송": "channel health",
        "텔레그램 실패": "channel health",
        "텔레그램 오류": "channel health",
        "텔레그램 문제": "channel health",
        "텔레그램 이슈": "channel health",
        "텔레그램 안됨": "channel health",
        "텔레그램 고장": "channel health",
        "왜 텔레그램 실패": "channel health",
        "인스타그램 상태": "channel health",
        "인스타그램 마지막 실패": "channel health",
        "인스타그램 성공 횟수": "channel health",
        "인스타그램 실패": "channel health",
        "인스타그램 문제": "channel health",
        "인스타 상태": "channel health",
        "인스타 문제": "channel health",
        "아이메시지 실패": "channel health",
        "아이메시지 문제": "channel health",
        "아이메시지 갔어": "channel health",
        "아이메시지 보냈어": "channel health",
        "아이메시지 보내졌어": "channel health",
        "아이메시지 다시 보낼까": "channel health",
        "아이메시지 재시도": "channel health",
        "아이메시지 재전송": "channel health",
        "메시지 실패": "channel health",
        "메시지 문제": "channel health",
        "메시지 갔어": "channel health",
        "메시지 갔나요": "channel health",
        "메시지 보냈어": "channel health",
        "메시지 보내졌어": "channel health",
        "왜 메시지 실패": "channel health",
        "전송 실패": "channel health",
        "전송 문제": "channel health",
        "전송 됐어": "channel health",
        "전송됐어": "channel health",
        "전송됐나요": "channel health",
        "왜 전송 실패": "channel health",
        "통화 실패": "channel health",
        "통화 문제": "channel health",
        "전화 실패": "channel health",
        "전화 문제": "channel health",
        "왜 전화 실패": "channel health",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean messaging/channel status intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_brief_failure_intents_redirect_to_recent_runs() -> None:
    cases = {
        "브리핑 실패": "recent tool runs",
        "브리핑 오류": "recent tool runs",
        "브리핑 문제": "recent tool runs",
        "브리핑 이슈": "recent tool runs",
        "브리핑 안 왔어": "recent tool runs",
        "브리핑 어디": "recent tool runs",
        "오늘 브리핑 보냈어": "recent tool runs",
        "브리핑 실행됐어": "recent tool runs",
        "모닝브리프 실패": "recent tool runs",
        "모닝브리프 오류": "recent tool runs",
        "모닝브리프 문제": "recent tool runs",
        "모닝브리프 안 왔어": "recent tool runs",
        "모닝브리프 어디": "recent tool runs",
        "모닝브리프 보냈어": "recent tool runs",
        "아침브리프 실패": "recent tool runs",
        "아침브리프 문제": "recent tool runs",
        "아침브리프 안 왔어": "recent tool runs",
        "아침 브리핑 보냈어": "recent tool runs",
        "스케줄러 실행됐어": "recent tool runs",
        "예약 작업 안 돌아": "recent tool runs",
        "예약 작업 문제": "recent tool runs",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean brief failure intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_model_brain_intents_redirect_to_model_routing_status() -> None:
    cases = [
        "모델 상태",
        "모델 확인",
        "모델 라우팅",
        "모델 라우팅 상태",
        "어떤 모델 써",
        "무슨 모델 써",
        "브레인 상태",
        "브레인 확인",
        "두뇌 상태",
        "올라마 상태",
        "오라마 상태",
        "라우팅 상태",
        "플래너 상태",
        "플래너 모델 상태",
        "채팅 모델 상태",
        "답변 길이 상태",
        "응답 길이 상태",
        "8초 대화 상태",
        "8초 채팅 상태",
        "대화 p95 상태",
        "대화 8초 목표",
        "채팅 기록 창",
        "채팅 히스토리 상태",
        "채팅 p95 상태",
        "채팅 8초 목표",
        "채팅 토큰 상태",
        "대화 기록 상태",
        "혼합 대화 p95 상태",
        "혼합 대화 지연 상태",
        "혼합 대화 속도 상태",
        "로컬 모델 상태",
        "LLM 상태",
    ]
    for phrase in cases:
        out = suggest_command(phrase)
        if not out or "model routing status" not in out:
            raise SystemExit(f"Korean model/brain intent should suggest model routing status: {phrase!r} -> {out!r}")


def test_korean_model_brain_failure_intents_redirect_to_model_routing_status() -> None:
    cases = [
        "모델 실패",
        "모델 오류",
        "모델 안됨",
        "브레인 문제",
        "두뇌 오류",
        "올라마 실패",
        "올라마 안돼",
        "오라마 오류",
        "로컬 모델 실패",
        "플래너 실패",
        "플래너 모델 안됨",
        "채팅 모델 오류",
        "LLM 실패",
    ]
    for phrase in cases:
        out = suggest_command(phrase)
        if not out or "model routing status" not in out:
            raise SystemExit(
                f"Korean model/brain failure intent should suggest model routing status: "
                f"{phrase!r} -> {out!r}"
            )


def test_korean_eval_verification_intents_redirect_to_read_only_proof_commands() -> None:
    cases = {
        "스모크 상태": "cockpit",
        "스모크 테스트 상태": "cockpit",
        "테스트 상태": "cockpit",
        "테스트 커버리지": "cockpit",
        "평가 상태": "cockpit",
        "제임스 평가": "cockpit",
        "제임스 평가팩": "cockpit",
        "제임스 워크플로우 평가": "cockpit",
        "제임스 워크플로우 테스트": "cockpit",
        "실제 워크플로우 평가": "cockpit",
        "실제 워크플로우 테스트": "cockpit",
        "워크플로우 평가": "cockpit",
        "워크플로우 테스트": "cockpit",
        "테스트 매트릭스": "cockpit",
        "테스트 매트릭스 상태": "cockpit",
        "라이브 테스트 매트릭스": "cockpit",
        "라이브 매트릭스 상태": "cockpit",
        "라이브 증명 상태": "cockpit",
        "채널 증명 상태": "cockpit",
        "채널 증명 매트릭스": "cockpit",
        "회귀 테스트 상태": "cockpit",
        "품질 게이트": "cockpit",
        "뭐가 검증됐어": "cockpit",
        "뭐가 검증됐나요": "cockpit",
        "무엇이 검증됐어": "cockpit",
        "무엇을 검증했어": "cockpit",
        "검증 상태": "verification receipt latest",
        "검증 확인": "verification receipt latest",
        "검증 결과": "verification receipt latest",
        "증거": "evidence ledger",
        "증거 상태": "evidence ledger",
        "증거 있어": "evidence ledger",
        "증거 뭐 있어": "evidence ledger",
        "증거 보여줘": "evidence ledger",
        "증거 보고서": "evidence ledger",
        "증명 상태": "evidence ledger",
        "증명 보여줘": "evidence ledger",
        "근거 상태": "evidence ledger",
        "근거 있어": "evidence ledger",
        "근거 뭐 있어": "evidence ledger",
        "근거 보여줘": "evidence ledger",
        "완료 증거": "evidence ledger",
        "신뢰 증거": "evidence ledger",
        "검증 증거": "verification receipt latest",
        "검증 증명": "verification receipt latest",
        "최근 검증": "verification receipt latest",
        "최신 검증": "verification receipt latest",
        "최근 증거": "verification receipt latest",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean eval/verification intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_memory_learning_intents_redirect_to_read_only_memory_commands() -> None:
    cases = {
        "메모리 상태": "memory stats",
        "메모리 확인": "memory stats",
        "기억 상태": "memory stats",
        "기억 확인": "memory stats",
        "뭘 기억해": "what do you remember",
        "무엇을 기억해": "what do you remember",
        "자비스가 뭘 기억해": "what do you remember",
        "나에 대해 뭘 알아": "what do you know about me",
        "학습 상태": "learning review",
        "학습 확인": "learning review",
        "학습 리뷰": "learning review",
        "학습 루프 상태": "learning review",
        "학습 부채 상태": "execution learning closure",
        "학습 부채 닫혔어": "execution learning closure",
        "학습 증명 매트릭스": "execution learning closure",
        "복구 학습 상태": "recovery closure checklist",
        "복구 부채 상태": "recovery closure checklist",
        "반복 실패 상태": "repeated failure clusters",
        "실패 학습 상태": "failure learning cockpit",
        "무엇을 배웠어": "learning review",
        "뭘 배웠어": "learning review",
        "배운 거": "learning review",
        "기억 리뷰": "learning review",
        "기억 정리": "weak memories",
        "약한 기억": "weak memories",
        "중복 기억": "duplicate memories",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean memory/learning intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_audit_history_intents_redirect_to_read_only_audit_commands() -> None:
    cases = {
        "감사 로그": "recent tool runs",
        "감사 기록": "recent tool runs",
        "실행 로그": "recent tool runs",
        "실행 기록": "recent tool runs",
        "실행 영수증": "verification receipt latest",
        "최근 로그": "recent tool runs",
        "최근 실행": "recent tool runs",
        "최근 실행 기록": "recent tool runs",
        "최근 도구 실행": "recent tool runs",
        "최근 기록": "recent tool runs",
        "최근 작업": "recent tool runs",
        "최근 활동": "recent tool runs",
        "마지막 도구 실행": "recent tool runs",
        "감사 상태": "recent tool runs",
        "도구 실행 기록": "recent tool runs",
        "런타임 추적": "runtime trace receipt",
        "무슨 일 했어": "recent tool runs",
        "뭐 했어": "recent tool runs",
        "영수증": "verification receipt latest",
        "최근 영수증": "verification receipt latest",
        "검증 영수증": "verification receipt latest",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean audit/history intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_trust_boundary_intents_redirect_to_read_only_safety_commands() -> None:
    cases = {
        "안전": "safety status",
        "안전 확인": "safety status",
        "안전 상태": "safety status",
        "자비스 안전": "safety status",
        "승인 안전": "safety status",
        "승인 필요한 것": "safety status",
        "승인이 필요한 것": "safety status",
        "뭐 승인 필요": "safety status",
        "어떤 명령이 승인 필요": "safety status",
        "어떤 도구가 승인 필요": "safety status",
        "권한 필요한 것": "safety status",
        "레도스 상태": "frozen routing risk report",
        "정규식 위험 상태": "frozen routing risk report",
        "동결 라우팅 위험": "frozen routing risk report",
        "전송 라우팅 레도스": "frozen routing risk report",
        "입력 길이 제한 상태": "planner input guard report",
        "긴 입력 안전 상태": "planner input guard report",
        "긴 명령 안전 상태": "planner input guard report",
        "플래너 입력 제한": "planner input guard report",
        "긴 메시지 레도스 위험": "planner input guard report",
        "개인정보": "privacy report",
        "개인정보 확인": "privacy report",
        "개인정보 상태": "privacy report",
        "프라이버시": "privacy report",
        "프라이버시 확인": "privacy report",
        "데이터 접근": "privacy report",
        "위험": "risk matrix",
        "위험 상태": "risk matrix",
        "위험 확인": "risk matrix",
        "리스크": "risk matrix",
        "리스크 상태": "risk matrix",
        "권한": "risk matrix",
        "권한 상태": "risk matrix",
        "위험한 명령": "risk matrix",
        "고위험 도구": "risk matrix",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean trust-boundary intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_approval_gate_intents_redirect_to_read_only_approval_commands() -> None:
    cases = {
        "승인 대기": "pending approvals",
        "승인 목록": "pending approvals",
        "승인 큐": "pending approvals",
        "승인 확인": "pending approvals",
        "승인 필요한 거 있어": "pending approvals",
        "승인할 거 있어": "pending approvals",
        "내가 승인해야 할 거 있어": "pending approvals",
        "뭐 기다려": "pending approvals",
        "나 기다리는 거 있어": "pending approvals",
        "검토할 거 있어": "pending approvals",
        "리뷰 큐": "pending approvals",
        "승인 필요한 작업": "pending approvals",
        "대기 중인 승인": "pending approvals",
        "승인 상태": "approval summary",
        "승인 내역": "approval history",
        "승인 기록": "approval history",
        "승인 증거": "approval evidence for latest",
        "승인 검증": "approval evidence for latest",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean approval-gate intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_approval_failure_intents_redirect_to_read_only_approval_summary() -> None:
    cases = [
        "승인 문제",
        "승인 실패",
        "승인 오류",
        "승인 안됨",
        "승인 안돼",
        "승인 버튼 문제",
        "승인 버튼 실패",
        "승인 버튼 안됨",
        "승인 큐 문제",
        "승인 큐 멈춤",
        "왜 승인 실패",
        "왜 승인 안됨",
        "허가 문제",
        "허가 실패",
        "허가 안됨",
    ]
    wrong_fragments = ["channel health", "recent tool runs", "pending approvals", "queued as approval"]
    for phrase in cases:
        out = suggest_command(phrase)
        if not out or "approval summary" not in out:
            raise SystemExit(f"Korean approval failure intent should suggest approval summary: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"Korean approval failure intent suggested stale/wrong command "
                    f"{wrong!r}: {phrase!r} -> {out!r}"
                )


def test_korean_next_step_intents_redirect_to_read_only_continuity_commands() -> None:
    cases = {
        "다음 뭐": "safe next actions",
        "다음 뭐 해": "safe next actions",
        "다음 뭐 할까": "safe next actions",
        "다음 작업": "safe next actions",
        "다음 할 일": "safe next actions",
        "다음 단계": "safe next actions",
        "안전한 다음 작업": "safe next actions",
        "작업 큐": "work queue",
        "작업 목록": "work queue",
        "작업 현황": "work queue",
        "미션 컨트롤": "mission control",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean next-step intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_korean_voice_intents_redirect_to_read_only_voice_commands() -> None:
    cases = {
        "음성": "voice command cockpit",
        "음성 상태": "voice command cockpit",
        "음성 명령 상태": "voice command cockpit",
        "음성 오류": "voice setup check",
        "음성 안돼": "voice setup check",
        "목소리": "voice command cockpit",
        "목소리 문제": "voice setup check",
        "보이스": "voice command cockpit",
        "마이크": "voice setup check",
        "마이크 확인": "voice setup check",
        "마이크 안돼": "voice setup check",
        "마이크 오류": "voice setup check",
        "마이크 실패": "voice setup check",
        "음성 설정 확인": "voice setup check",
        "음성 워밍업 상태": "voice setup check",
        "음성 준비됐어": "voice setup check",
        "음성 중지": "voice stop intent: stop listening",
        "음성 멈춰": "voice stop intent: stop listening",
        "음성 취소": "voice stop intent: stop listening",
        "음성 음소거": "voice stop intent: stop listening",
        "그만 말해": "voice stop intent: stop listening",
        "말 그만": "voice stop intent: stop listening",
        "자비스 조용히": "voice stop intent: stop listening",
        "목소리 멈춰": "voice stop intent: stop listening",
        "녹음 중지": "voice stop intent: stop listening",
        "녹음 멈춰": "voice stop intent: stop listening",
        "녹음 안됨": "voice setup check",
        "마이크 상태": "voice setup check",
        "마이크 개인정보": "microphone privacy please",
        "마이크 프라이버시": "microphone privacy please",
        "푸시투톡 상태": "voice setup check",
        "푸시투톡 개인정보": "microphone privacy please",
        "위스퍼 상태": "voice setup check",
        "위스퍼 워밍업 상태": "voice setup check",
        "위스퍼 실패": "voice setup check",
        "전사 오류": "voice setup check",
        "다시 녹음": "voice stop intent: stop listening",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"Korean voice intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_voice_failure_intents_redirect_to_voice_setup_check() -> None:
    cases = {
        "mic not working": "voice setup check",
        "microphone broken": "voice setup check",
        "my microphone is not working": "voice setup check",
        "voice broken": "voice setup check",
        "voice input failed": "voice setup check",
        "speech not working": "voice setup check",
        "recording failed": "voice setup check",
        "whisper failed": "voice setup check",
        "transcription not working": "voice setup check",
        "why did voice fail": "voice setup check",
        "why did whisper fail": "voice setup check",
    }
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"voice failure intent should suggest {expected!r}: {phrase!r} -> {out!r}")


def test_voice_control_and_readiness_intents_redirect_to_read_only_voice_commands() -> None:
    cases = {
        "stop talking": "voice stop intent: stop listening",
        "stop speaking": "voice stop intent: stop listening",
        "mute jarvis": "voice stop intent: stop listening",
        "mute speech": "voice stop intent: stop listening",
        "silence voice": "voice stop intent: stop listening",
        "stop voice": "voice stop intent: stop listening",
        "cancel speech": "voice stop intent: stop listening",
        "interrupt speech": "voice stop intent: stop listening",
        "stop listening": "voice stop intent: stop listening",
        "stop recording": "voice stop intent: stop listening",
        "microphone privacy": "microphone privacy please",
        "voice privacy": "microphone privacy please",
        "push to talk privacy": "microphone privacy please",
        "voice warmup status": "voice setup check",
        "whisper warmup status": "voice setup check",
        "is voice ready": "voice setup check",
        "voice input status": "voice setup check",
        "mic status": "voice setup check",
        "push to talk status": "voice setup check",
        "asr status": "voice setup check",
    }
    wrong_fragments = ["stop Jarvis", "list voices", "run_shell_command", "pending approvals"]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"voice control/readiness intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"voice control/readiness intent suggested stale/wrong command {wrong!r}: {phrase!r} -> {out!r}"
                )


def test_help_promised_operator_typos_are_suggested() -> None:
    cases = {
        "risk prefight": "risk preflight",
        "execution governer": "execution governor",
        "voice stop intnt": "voice stop intent",
        "command diagnossis": "command diagnosis",
        "approval histry": "approval history",
    }
    for typo, expected in cases.items():
        out = suggest_command(typo)
        if not out or expected not in out:
            raise SystemExit(f"help-promised operator typo should suggest {expected!r}: {typo!r} -> {out!r}")


def test_continuity_starter_typos_are_suggested() -> None:
    cases = {
        "safe next actons": "safe next actions",
        "session closeuot": "session closeout",
        "work blok checkpoint": "work block checkpoint",
        "prioritty stack": "priority stack",
    }
    for typo, expected in cases.items():
        out = suggest_command(typo)
        if not out or expected not in out:
            raise SystemExit(f"continuity starter typo should suggest {expected!r}: {typo!r} -> {out!r}")


def test_real_conversation_is_not_hijacked() -> None:
    # Genuine conversational prose must fall through to the chat model (None here).
    for prose in [
        "tell me about your day",
        "what is the meaning of life",
        "i had a really long and complicated week at work",
        "do you ever wonder what it all means",
    ]:
        if suggest_command(prose) is not None:
            raise SystemExit(f"real conversation should not be hijacked: {prose!r}")


def test_gibberish_and_empty_return_none() -> None:
    for junk in ["", "   ", "asdfghjkl", "qwertyuiop zxcvbnm"]:
        if suggest_command(junk) is not None:
            raise SystemExit(f"gibberish/empty should return no suggestion: {junk!r}")


def test_exact_command_is_not_echoed_back() -> None:
    # An exact catalog command would already route to a tool; suggesting it back
    # would be noise.
    for exact in [
        "weather in Tokyo",
        "calendar today",
        "wikipedia Ada Lovelace",
        "cockpit",
        "capability cockpit",
        "pending approvals",
        "approval review",
        "approval readiness 1",
        "approval packet 1",
        "approval chain proof 1",
        "verification receipt 1",
        "execution health report",
        "recovery closure checklist",
        "after-action learning packet 1",
        "jarvis doctor",
        "setup check",
        "readiness report",
        "prototype readiness",
        "safety status",
        "help safety",
        "voice setup check",
        "recent tool runs",
        "list scheduled jobs",
        "channel health",
        "privacy report",
        "risk matrix",
        "storage status",
        "storage recovery plan",
        "storage recovery check",
        "jarvis status",
        "harness operations",
        "agi next build move",
        "completion claim gate",
        "command cockpit",
        "model routing status",
        "capability map",
        "risk preflight",
        "execution governor",
        "voice stop intent",
        "command diagnosis",
        "approval history",
        "safe next actions",
        "session closeout",
        "work block checkpoint",
        "priority stack",
    ]:
        if suggest_command(exact) is not None:
            raise SystemExit(f"an exact command should not be suggested back: {exact!r}")


def test_long_input_is_skipped() -> None:
    # Long inputs are almost always real prose, not a mistyped one-liner.
    long_input = "weather forecast for tokyo and seoul and osaka and busan next week please"
    if suggest_command(long_input) is not None:
        raise SystemExit("long prose should be left to the chat fallback")


def test_runtime_uses_suggestion_over_generic_chat() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        res = rt.handle("weathr in tokyo")
        response = getattr(res, "response", "")
        if "Did you mean" not in response or "weather in Tokyo" not in response:
            raise SystemExit(f"runtime should surface the suggestion for a typo: {response!r}")
        meta = getattr(res, "metadata", {}) or {}
        chat_response = meta.get("chat_response") or {}
        if chat_response.get("runtime_route") != "command_suggestion":
            raise SystemExit(f"runtime should tag the suggestion route: {chat_response!r}")


def test_runtime_exact_cockpit_aliases_execute_read_only_cockpit() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        suggestion = rt.handle("control plane")
        if "Did you mean" not in suggestion.response or "cockpit" not in suggestion.response:
            raise SystemExit(f"control-plane phrasing should still suggest cockpit: {suggestion.response!r}")
        if getattr(suggestion, "tool_results", []):
            raise SystemExit("control-plane suggestion should not execute tools before the owner sends cockpit")

        for phrase in [
            "cockpit",
            "capability cockpit",
            "capability_cockpit",
            "show cockpit",
            "show me cockpit",
            "show me the cockpit",
            "where is the cockpit",
            "콕핏",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(f"{phrase!r} should run capability_cockpit once: {results!r}")
            if not res.verified or "Capability cockpit" not in res.response:
                raise SystemExit(f"{phrase!r} should return the cockpit output: {res.response[:160]!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"{phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"{phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"{phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"{phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "cockpit summary",
            "jarvis cockpit summary",
            "capability cockpit plan",
            "phone control center plan",
            "what is capability cockpit plan",
            "what is phone control center",
            "what is the capability cockpit plan",
            "what is the phone control center",
            "eval pack plan",
            "operator eval pack plan",
            "what is eval pack",
            "what is the eval pack",
            "what is the operator eval pack",
            "what real workflows should Jarvis prove",
            "what should Jarvis prove live",
            "trust checklist",
            "earned trust checklist",
            "can I trust Jarvis?",
            "why can I trust Jarvis?",
            "능력 콕핏 계획",
            "폰 컨트롤 센터 계획",
            "평가팩 계획",
            "제임스 평가팩 계획",
            "실제 워크플로우 뭐 증명해",
            "자비스 뭐 증명해야 해",
            "콕핏 요약",
            "신뢰 체크리스트",
            "자비스 믿어도 돼?",
            "믿어도 되는 이유",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(f"trust/cockpit phrase {phrase!r} should run capability_cockpit once: {results!r}")
            if "Capability cockpit" not in res.response:
                raise SystemExit(
                    f"trust/cockpit phrase {phrase!r} should expose the cockpit, not chat/suggestion drift: "
                    f"{res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"trust/cockpit phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"trust/cockpit phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"trust/cockpit phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"trust/cockpit phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")


def test_runtime_exact_agent_moat_plan_aliases_execute_work_queue_read_only() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    phrases = [
        "three month plan",
        "three months of work",
        "Jarvis three month plan",
        "what are the three months of work",
        "what is the three month plan",
        "what is the three month plan for Jarvis",
        "what should Jarvis focus on for the next three months",
        "how do we unlock the moat",
        "what unlocks Jarvis moat",
        "what unlocks the moat",
        "what should Jarvis build before integrations",
        "when should Jarvis add integrations",
        "should Jarvis add integrations now",
        "자비스 3개월 계획",
        "자비스 세 달 계획",
        "자비스 다음 3개월 뭐 해",
        "자비스 해자 어떻게 열어",
        "해자 어떻게 열어",
        "통합 지금 추가해도 돼",
        "통합 언제 추가해",
        "자비스 통합 언제 추가해",
    ]
    wrong_fragments = [
        "Wikipedia",
        "days until",
        "call_contact",
        "find_contact",
        "queued as approval",
        "fallback chat",
        "Did you mean",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in phrases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "work_queue":
                raise SystemExit(f"moat/plan phrase {phrase!r} should run work_queue once: {results!r}")
            if "Jarvis work queue" not in res.response or "visible control plane" not in res.response:
                raise SystemExit(
                    f"moat/plan phrase {phrase!r} should surface the work queue guidance: "
                    f"{res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"moat/plan phrase {phrase!r} must remain read-only without queuing approvals")
            for wrong in wrong_fragments:
                if wrong in res.response:
                    raise SystemExit(
                        f"moat/plan phrase leaked stale path {wrong!r}: "
                        f"{phrase!r} -> {res.response[:240]!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"moat/plan phrase {phrase!r} should execute through the tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"moat/plan phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"moat/plan phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "chat acceptance latency",
            "chat latency status",
            "chat latency acceptance status",
            "chat p95",
            "chat p95 status",
            "chat p95 target",
            "conversation latency status",
            "conversation acceptance latency",
            "conversation p95 status",
            "8 second chat target",
            "8 second conversation target",
            "8s chat target",
            "8s conversation target",
            "did chat meet the 8 second target",
            "did chat pass latency",
            "did mixed conversation pass latency",
            "chat history window",
            "chat history messages",
            "chat max history messages",
            "chat max reply tokens",
            "chat reply token cap",
            "chat response length status",
            "chat speed acceptance status",
            "chat speed settings",
            "chat token cap",
            "chat tuning",
            "chat tuning knobs",
            "conversation speed status",
            "how do I make chat faster?",
            "how fast is chat?",
            "is chat under 8 seconds",
            "is Jarvis under 8 seconds",
            "is chat fast enough?",
            "is mixed conversation under 8 seconds",
            "Jarvis latency status",
            "Jarvis speed status",
            "lower chat history window",
            "lower chat token cap",
            "lower reply token cap",
            "make chat faster",
            "make Jarvis faster",
            "make Jarvis replies shorter",
            "mixed conversation acceptance latency",
            "mixed conversation acceptance status",
            "mixed conversation latency",
            "mixed conversation latency status",
            "mixed conversation p95",
            "mixed conversation p95 status",
            "mixed conversation status",
            "p95 chat target",
            "p95 latency status",
            "reduce chat history",
            "reduce reply tokens",
            "reply length status",
            "response length status",
            "should we lower chat history",
            "should we lower reply tokens",
            "speed up chat",
            "speed up Jarvis",
            "tune chat speed",
            "tune Jarvis speed",
            "what are the chat speed knobs?",
            "what are the chat tuning knobs?",
            "under 8 seconds status",
            "why are replies slow?",
            "why are chat replies slow?",
            "why is chat slow?",
            "why is Jarvis slow?",
            "대화 지연 상태",
            "대화 속도 상태",
            "대화 기록 상태",
            "대화 p95 상태",
            "대화 8초 목표",
            "8초 대화 상태",
            "8초 채팅 상태",
            "답변 길이 상태",
            "응답 길이 상태",
            "자비스 느려",
            "자비스 속도 상태",
            "자비스 왜 느려",
            "채팅 기록 창",
            "채팅 히스토리 상태",
            "채팅 p95 상태",
            "채팅 8초 목표",
            "채팅 느려",
            "채팅 속도 상태",
            "채팅 속도 설정",
            "채팅 토큰 상태",
            "채팅 튜닝",
            "혼합 대화 p95 상태",
            "혼합 대화 지연 상태",
            "혼합 대화 속도 상태",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "model_routing_status":
                raise SystemExit(f"chat-latency phrase {phrase!r} should run model_routing_status once: {results!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"chat-latency phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"chat-latency phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"chat-latency phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"chat-latency phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "diagnostics",
            "diagnostic status",
            "diagnostic report",
            "diagnostics status",
            "diagnostics report",
            "system health",
            "jarvis health",
            "is Jarvis healthy?",
            "error status",
            "error report",
            "시스템 헬스",
            "자비스 헬스",
            "진단 상태",
            "진단 보고서",
            "오류 상태",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "jarvis_doctor":
                raise SystemExit(f"diagnostic phrase {phrase!r} should run jarvis_doctor once: {results!r}")
            if "Jarvis doctor" not in res.response:
                raise SystemExit(
                    f"diagnostic phrase {phrase!r} should expose the doctor report, "
                    f"not a suggestion/chat fallback: {res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"diagnostic phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"diagnostic phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"diagnostic phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"diagnostic phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "health status",
            "overall status",
            "overall health",
            "status report",
            "what is healthy?",
            "is everything healthy?",
            "전체 상태",
            "전체 헬스",
            "건강 상태",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(f"health phrase {phrase!r} should run capability_cockpit once: {results!r}")
            if "Capability cockpit" not in res.response:
                raise SystemExit(
                    f"health phrase {phrase!r} should expose the cockpit, "
                    f"not a suggestion/chat fallback: {res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"health phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"health phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"health phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"health phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "setup ready",
            "setup readiness",
            "is setup ready?",
            "environment status",
            "env status",
            "config status",
            "configuration status",
            "status config",
            "launcher status",
            "bootstrap status",
            "startup status",
            "설정 상태",
            "설정 확인",
            "환경 상태",
            "구성 상태",
            "부트스트랩 상태",
            "시작 상태",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "setup_check":
                raise SystemExit(f"setup readiness phrase {phrase!r} should run setup_check once: {results!r}")
            if "Jarvis V3 setup check" not in res.response:
                raise SystemExit(
                    f"setup readiness phrase {phrase!r} should expose the setup check, "
                    f"not a suggestion/chat fallback: {res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"setup readiness phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"setup readiness phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"setup readiness phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"setup readiness phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )

        for phrase in [
            "voice status",
            "voice readiness",
            "voice setup status",
            "is voice ready?",
            "voice input status",
            "voice warmup status",
            "whisper status",
            "whisper warmup status",
            "mic status",
            "microphone status",
            "push to talk status",
            "ASR status",
            "마이크 상태",
            "마이크 확인",
            "음성 상태",
            "음성 워밍업 상태",
            "음성 입력 상태",
            "음성 준비됐어",
            "위스퍼 상태",
            "위스퍼 워밍업 상태",
            "푸시투톡 상태",
            "전사 상태",
            "녹음 상태",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "voice_setup_check":
                raise SystemExit(f"voice readiness phrase {phrase!r} should run voice_setup_check once: {results!r}")
            if "Jarvis voice setup check" not in res.response:
                raise SystemExit(
                    f"voice readiness phrase {phrase!r} should expose the voice setup check, "
                    f"not a suggestion/chat fallback: {res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"voice readiness phrase {phrase!r} must remain read-only without queuing approvals")
            tool_meta = getattr(results[0], "metadata", {}) or {}
            if tool_meta.get("records_audio") or tool_meta.get("starts_listener"):
                raise SystemExit(f"voice readiness phrase {phrase!r} must not record audio or start a listener: {tool_meta!r}")
            if tool_meta.get("voice_warmup_touches_microphone") is not False:
                raise SystemExit(f"voice readiness phrase {phrase!r} lost warmup microphone boundary: {tool_meta!r}")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"voice readiness phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"voice readiness phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"voice readiness phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )

        for phrase in [
            "reboot status",
            "will Jarvis survive reboot?",
            "daemon startup status",
            "daemon status",
            "launchd status",
            "LaunchAgent status",
            "scheduler daemon status",
            "telegram daemon status",
            "status server daemon status",
            "network recovery status",
            "will Jarvis recover from network loss?",
            "reboot proof matrix",
            "daemon proof matrix",
            "post reboot proof matrix",
            "post-reboot proof matrix",
            "launchd proof matrix",
            "telegram control daemon status",
            "telegram control proof matrix",
            "status server proof matrix",
            "scheduler daemon proof matrix",
            "network loss proof result format",
            "network recovery proof result format",
            "how should I report network recovery proof?",
            "what network recovery should I test?",
            "what should I test after reboot?",
            "재부팅 상태",
            "데몬 상태",
            "런치디 상태",
            "런치에이전트 상태",
            "스케줄러 데몬 상태",
            "텔레그램 데몬 상태",
            "네트워크 복구 상태",
            "재부팅 증명",
            "데몬 증명",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(f"reboot/daemon phrase {phrase!r} should run capability_cockpit once: {results!r}")
            if "Capability cockpit" not in res.response or "Acceptance Harness" not in res.response:
                raise SystemExit(
                    f"reboot/daemon phrase {phrase!r} should expose the acceptance/reliability cockpit, "
                    f"not chat or stale suggestions: {res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"reboot/daemon phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"reboot/daemon phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"reboot/daemon phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"reboot/daemon phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "restart the Telegram control daemon",
            "restart Telegram daemon",
            "텔레그램 제어 데몬 재시작",
            "텔레그램 데몬 재시작",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "telegram_control_restart_guidance":
                raise SystemExit(f"Telegram restart phrase {phrase!r} should run read-only guidance once: {results!r}")
            if (
                "restart was not performed" not in res.response
                or "telegram control daemon status" not in res.response
                or "does not inspect live process state" not in res.response
            ):
                raise SystemExit(f"Telegram restart phrase {phrase!r} missed bounded recovery guidance: {res.response[:240]!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Telegram restart phrase {phrase!r} must remain read-only without queuing approvals")
            tool_meta = getattr(results[0], "metadata", {}) or {}
            if tool_meta.get("restart_performed") is not False or tool_meta.get("controls_computer") is not False:
                raise SystemExit(f"Telegram restart phrase {phrase!r} crossed the daemon-control boundary: {tool_meta!r}")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"Telegram restart phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"Telegram restart phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"Telegram restart phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "restart it",
            "restart this",
            "restart that",
            "재시작해",
            "다시 시작해",
            "그거 재시작해",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "restart_target_clarification":
                raise SystemExit(f"ambiguous restart phrase {phrase!r} should clarify once without guessing: {results!r}")
            if "did not restart anything" not in res.response or "target is ambiguous" not in res.response:
                raise SystemExit(f"ambiguous restart phrase {phrase!r} missed bounded clarification: {res.response[:240]!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"ambiguous restart phrase {phrase!r} must remain read-only without queuing approvals")
            tool_meta = getattr(results[0], "metadata", {}) or {}
            if (
                tool_meta.get("restart_performed") is not False
                or tool_meta.get("restart_target_identified") is not False
                or tool_meta.get("controls_computer") is not False
            ):
                raise SystemExit(f"ambiguous restart phrase {phrase!r} crossed the restart boundary: {tool_meta!r}")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"ambiguous restart phrase {phrase!r} should use the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"ambiguous restart phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"ambiguous restart phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "personal proof matrix",
            "personal proof result format",
            "personal proof status",
            "personal proofs status",
            "personal integration proof status",
            "personal integrations proof status",
            "personal integration proof gaps",
            "personal integrations proof gaps",
            "personal integration proof matrix",
            "personal integrations proof matrix",
            "personal integration result format",
            "personal integrations result format",
            "did telegram pass",
            "is telegram proven",
            "telegram proof passed",
            "did kakao pass",
            "is kakao proven",
            "did imessage pass",
            "is imessage proven",
            "did morning brief pass",
            "is morning brief proven",
            "did calendar write pass",
            "is calendar write proven",
            "did email send pass",
            "is email send proven",
            "did reminders pass",
            "is contact lookup proven",
            "did voice pass",
            "is local talk proven",
            "is korean voice proven",
            "did reboot proof pass",
            "is reboot proven",
            "calendar proof status",
            "calendar status",
            "calendar live proof status",
            "calendar write status",
            "calendar write proof",
            "calendar write proof matrix",
            "calendar write proof status",
            "calendar create update delete proof",
            "calendar create update delete proof matrix",
            "calendar write result format",
            "email status",
            "email read status",
            "email search status",
            "email send status",
            "email proof",
            "email proof status",
            "email read proof matrix",
            "email search proof matrix",
            "email send proof matrix",
            "reminder status",
            "reminders status",
            "reminder proof",
            "reminder proof matrix",
            "reminder proof status",
            "set reminder proof",
            "set reminder proof matrix",
            "set reminder proof status",
            "contacts status",
            "contacts proof",
            "contacts proof status",
            "contact lookup status",
            "contact lookup proof",
            "contact lookup proof matrix",
            "contact lookup proof status",
            "phone approval status",
            "approval flow status",
            "cockpit status",
            "error guidance status",
            "korean voice status",
            "telegram voice status",
            "what personal integration proofs are missing",
            "what personal integrations are unproven",
            "what calendar write should I test",
            "how should I report calendar write proof",
            "what email read should I test",
            "what email search should I test",
            "what email send should I test",
            "how should I report email read proof",
            "how should I report email search proof",
            "how should I report email send proof",
            "what reminder proof should I test",
            "what set reminder proof should I test",
            "how should I report reminder proof",
            "how should I report set reminder proof",
            "what contact lookup should I test",
            "how should I report contact lookup proof",
            "scheduler proof matrix",
            "scheduler proof result format",
            "jobs 7 day proof",
            "jobs seven day proof",
            "scheduled jobs 7 day proof",
            "scheduled jobs streak",
            "scheduled job streak status",
            "scheduler streak status",
            "did scheduled jobs run for 7 days?",
            "are scheduled jobs running daily?",
            "what is the 7 day scheduler proof?",
            "what scheduler streak should I test?",
            "how should I report scheduled job streak proof?",
            "daily streak proof",
            "daily job streak proof",
            "job streak proof matrix",
            "scheduled delivery proof matrix",
            "morning brief delivery proof matrix",
            "daily value proof matrix",
            "approval proof result format",
            "approval buttons proof matrix",
            "approval callback proof matrix",
            "phone approval buttons proof matrix",
            "telegram approval buttons proof matrix",
            "what approval buttons should I test?",
            "how should I report approval callback proof?",
            "phone control proof matrix",
            "phone control proof result format",
            "conversation proof matrix",
            "conversation proof result format",
            "chat latency proof matrix",
            "chat latency proof result format",
            "conversation research proof matrix",
            "conversation research proof result format",
            "mixed conversation proof result format",
            "voice proof result format",
            "voice live proof format",
            "telegram voice proof matrix",
            "telegram voice proof result format",
            "local voice proof matrix",
            "local voice proof result format",
            "korean voice proof matrix",
            "korean voice proof result format",
            "push to talk proof matrix",
            "push-to-talk proof matrix",
            "talk.py proof matrix",
            "spoken reply proof matrix",
            "voice speak proof matrix",
            "reboot proof result format",
            "are error messages actionable",
            "connector error proof matrix",
            "connector recovery proof matrix",
            "do error messages name the fix",
            "do user errors name the fix",
            "error guidance proof result format",
            "error messages status",
            "error recovery proof",
            "error recovery status",
            "how should I report error messages?",
            "how should I report error recovery proof?",
            "is error guidance proven",
            "recovery guidance status",
            "show error guidance",
            "show recovery guidance",
            "user-visible error proof matrix",
            "user visible error proof matrix",
            "what error messages should I test?",
            "what recovery guidance should I test?",
            "what user-facing errors need proof?",
            "what personal proofs should I test?",
            "what personal integrations should I test?",
            "what phone control should I test?",
            "what research proof should I test?",
            "what mixed conversation should I test?",
            "what conversation should I test?",
            "what chat proof should I run?",
            "what voice proofs should I test?",
            "what local voice should I test?",
            "what telegram voice should I test?",
            "what Korean voice should I test?",
            "how should I report conversation proof?",
            "how should I report chat latency proof?",
            "how should I report local voice proof?",
            "how should I report telegram voice proof?",
            "how should I report Korean voice proof?",
            "how should I report error guidance proof?",
            "개인 증명 매트릭스",
            "개인 증명 결과 형식",
            "캘린더 쓰기 증명 매트릭스",
            "이메일 읽기 증명",
            "이메일 검색 증명",
            "이메일 보내기 증명",
            "리마인더 증명 매트릭스",
            "연락처 조회 증명 매트릭스",
            "스케줄 증명 매트릭스",
            "스케줄 증명 결과 형식",
            "일일 가치 증명 매트릭스",
            "승인 증명 매트릭스",
            "승인 결과 형식",
            "폰컨트롤 증명 매트릭스",
            "폰 제어 증명 형식",
            "대화 연구 증명 매트릭스",
            "대화 증명 결과 형식",
            "대화 증명 뭐 해야 해",
            "대화 증명 어떻게 보고해",
            "혼합 대화 증명",
            "혼합 대화 뭐 테스트해",
            "챗 지연 증명 형식",
            "음성 증명 뭐 해야 해",
            "음성 증명 어떻게 보고해",
            "음성 뭐 테스트해",
            "텔레그램 음성 증명",
            "로컬 음성 증명",
            "음성 증명 결과 형식",
            "오류 안내 증명 매트릭스",
            "오류 안내 증명",
            "오류 안내 형식",
            "오류 메시지 상태",
            "오류 복구 상태",
            "복구 안내 증명",
            "재부팅 증명 형식",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(f"proof-reporting phrase {phrase!r} should run capability_cockpit once: {results!r}")
            if "Capability cockpit" not in res.response or "Acceptance Harness" not in res.response:
                raise SystemExit(
                    f"proof-reporting phrase {phrase!r} should expose the acceptance cockpit, "
                    f"not chat, research, call routing, or stale suggestions: {res.response[:240]!r}"
                )
            stale_fragments = (
                "Research is having trouble",
                "risk level HIGH_RISK",
                "Good evening! Here's your brief",
                "Good morning! Here's your brief",
                "I am here. The full local chat model is offline",
                "I am here. The configured chat model did not produce a usable response",
                "That sounds like it may involve action",
            )
            if any(fragment in res.response for fragment in stale_fragments):
                raise SystemExit(f"proof-reporting phrase {phrase!r} leaked a stale action path: {res.response[:240]!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"proof-reporting phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"proof-reporting phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"proof-reporting phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"proof-reporting phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )

        for phrase in ["proofs", "evidence status", "증거 상태"]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "evidence_ledger":
                raise SystemExit(f"proof/evidence phrase {phrase!r} should run evidence_ledger once: {results!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"proof/evidence phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"proof/evidence phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"proof/evidence phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"proof/evidence phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "next cockpit action",
            "cockpit next action",
            "what is Jarvis next step",
            "what is next for Jarvis",
            "what's next for Jarvis",
            "what command should I run",
            "what command should I run now",
            "what should I run next?",
            "what should I run now?",
            "what command should I run next?",
            "what should I check now?",
            "what should I check next?",
            "다음 행동",
            "다음 명령",
            "다음 뭐 해",
            "다음 뭐 해야 해",
            "다음 뭐 할까",
            "다음에 뭐 해?",
            "다음에 뭘 실행해",
            "무슨 명령 실행해",
            "자비스 다음 단계",
            "자비스 다음 뭐 해",
            "자비스 뭐부터 해",
            "뭐 실행해야 해?",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "safe_next_actions":
                raise SystemExit(f"next-control phrase {phrase!r} should run safe_next_actions once: {results!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"next-control phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"next-control phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"next-control phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"next-control phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "messages lane",
            "calls lane",
            "morning brief lane",
            "briefing lane",
            "contacts lane",
            "calendar email lane",
            "personal lane",
            "personal integrations lane",
            "personal proofs lane",
            "schedule lane",
            "scheduled jobs lane",
            "weather lane",
            "research lane",
            "memory lane",
            "learning lane",
            "diagnostics lane",
            "approvals lane",
            "workflow evals lane",
            "worker lane",
            "voice lane",
            "메시지 레인",
            "통화 레인",
            "브리핑 레인",
            "연락처 레인",
            "캘린더 이메일 레인",
            "개인 증명 레인",
            "스케줄 레인",
            "날씨 레인",
            "연구 레인",
            "기억 레인",
            "학습 레인",
            "진단 레인",
            "승인 레인",
            "워크플로우 평가 레인",
            "워커 레인",
            "음성 레인",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(
                    f"lane phrase {phrase!r} should run read-only capability_cockpit once, "
                    f"not its underlying feature tool: {results!r}"
                )
            if "Capability cockpit" not in res.response:
                raise SystemExit(
                    f"lane phrase {phrase!r} should expose the cockpit instead of feature/chat drift: "
                    f"{res.response[:240]!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            if "capability_lanes" not in tool_meta or "lane_count" not in tool_meta:
                raise SystemExit(f"lane phrase {phrase!r} lost cockpit lane metadata: {tool_meta!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"lane phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"lane phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"lane phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "cockpit attention",
            "attention",
            "attention status",
            "what needs attention",
            "what needs my attention",
            "what needs review",
            "which lanes need attention",
            "주의 상태",
            "주의 필요",
            "검토 필요",
            "콕핏 주의",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(
                    f"attention phrase {phrase!r} should run read-only capability_cockpit once: {results!r}"
                )
            if "Capability cockpit" not in res.response:
                raise SystemExit(
                    f"attention phrase {phrase!r} should expose the cockpit instead of chat/readiness drift: "
                    f"{res.response[:240]!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            if "attention_lane_count" not in tool_meta or "capability_lanes" not in tool_meta:
                raise SystemExit(f"attention phrase {phrase!r} lost cockpit attention metadata: {tool_meta!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"attention phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"attention phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"attention phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}")

        for phrase in [
            "acceptance harness status",
            "acceptance coverage",
            "acceptance coverage drift",
            "acceptance coverage status",
            "acceptance harness rows",
            "august checklist",
            "show august checklist",
            "definition of done",
            "does live_check cover every checklist section",
            "does live_check cover the August checklist",
            "does live_check cover acceptance",
            "finish plan",
            "show finish plan",
            "is live_check missing rows",
            "live_check coverage",
            "live check coverage",
            "live_check rows",
            "live_check missing rows",
            "live proof status",
            "one screen live_check table",
            "one screen acceptance table",
            "show one screen acceptance table",
            "smoke status",
            "aggregate smoke status",
            "full smoke status",
            "latest aggregate smoke proof",
            "smoke suite status",
            "suite lock status",
            "smoke lock status",
            "smoke suite lock status",
            "are tests green?",
            "test coverage",
            "what has been tested?",
            "voice proof status",
            "is Korean voice tested?",
            "what is the finish line?",
            "what remains before Jarvis is done?",
            "what remains to finish Jarvis",
            "what's not finished in Jarvis?",
            "what open acceptance items remain?",
            "what DoD items are open?",
            "what acceptance tests are pending?",
            "what acceptance proofs are pending?",
            "what do I need to prove?",
            "what does the operator need to test?",
            "what rows are in live_check",
            "what rows does live_check show",
            "what should the operator prove?",
            "what is left before August?",
            "what is left for August?",
            "whats left to finish?",
            "what is not done?",
            "what is still not done?",
            "what is still open?",
            "what is left unfinished?",
            "what remains unproven?",
            "what live proofs are missing?",
            "what live tests should I run?",
            "what live proofs should I run?",
            "what should operator test live?",
            "what should I verify live?",
            "how do I prove calendar write?",
            "how do I test calendar write?",
            "how do I prove email send?",
            "how do I test email read?",
            "how do I prove reminders?",
            "how do I test reminders?",
            "how do I prove contact lookup?",
            "how do I test contact lookup?",
            "how do I prove Korean voice?",
            "how do I test Telegram voice?",
            "how do I prove reboot survival?",
            "how do I test daemon recovery?",
            "how do I prove morning brief?",
            "how do I test morning brief?",
            "how do I prove phone approval?",
            "how do I test approval flow?",
            "which acceptance items are blocked?",
            "which checklist items are blocked?",
            "open checklist items",
            "remaining checklist items",
            "unverified checklist items",
            "unfinished Jarvis work",
            "Jarvis unfinished work",
            "Jarvis remaining work",
            "완료 기준",
            "검수 체크리스트",
            "남은 기준",
            "라이브체크 커버리지",
            "라이브 체크 표",
            "라이브체크 항목",
            "원스크린 인수 표",
            "인수 커버리지",
            "뭐 남았어?",
            "완료 뭐 남았어?",
            "뭐 증명해야 해?",
            "라이브 증명 상태",
            "인수 상태",
            "스모크 상태",
            "테스트 초록",
            "음성 증명 상태",
            "한국어 음성 증명",
            "캘린더 쓰기 어떻게 증명해",
            "캘린더 쓰기 어떻게 테스트해",
            "이메일 보내기 어떻게 증명해",
            "이메일 읽기 어떻게 테스트해",
            "리마인더 어떻게 증명해",
            "리마인더 어떻게 테스트해",
            "연락처 조회 어떻게 증명해",
            "연락처 조회 어떻게 테스트해",
            "한국어 음성 어떻게 증명해",
            "텔레그램 음성 어떻게 테스트해",
            "재부팅 생존 어떻게 증명해",
            "데몬 복구 어떻게 테스트해",
            "아침 브리핑 어떻게 증명해",
            "아침 브리핑 어떻게 테스트해",
            "폰 승인 어떻게 증명해",
            "승인 흐름 어떻게 테스트해",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(
                    f"acceptance phrase {phrase!r} should run read-only capability_cockpit once: {results!r}"
                )
            if "Capability cockpit" not in res.response or "Acceptance Harness" not in res.response:
                raise SystemExit(
                    f"acceptance phrase {phrase!r} should expose the acceptance harness cockpit, "
                    f"not wiki/dictionary/chat: {res.response[:240]!r}"
                )
            if "Wikipedia" in res.response or "dictionary service" in res.response:
                raise SystemExit(f"acceptance phrase {phrase!r} drifted into info lookup: {res.response[:240]!r}")
            if any(result.tool_name in {"open_application", "wiki_summary"} for result in results):
                raise SystemExit(f"acceptance phrase {phrase!r} drifted into stale action/search routing: {results!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"acceptance phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"acceptance phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"acceptance phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"acceptance phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )

        for phrase in [
            "acceptance next proof",
            "next acceptance proof",
            "next acceptance test",
            "next live proof",
            "next live test",
            "next proof",
            "next proof to run",
            "what acceptance proof is next?",
            "what is the next acceptance proof?",
            "what proof is next?",
            "what proof should I run next?",
            "what should I prove next?",
            "what should I test next?",
            "what should operator test next?",
            "which proof is next?",
            "다음 증명",
            "다음 라이브 테스트",
            "다음에 뭐 테스트해",
            "다음에 뭐 증명해",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "completion_next_proof_packet":
                raise SystemExit(
                    f"next-proof phrase {phrase!r} should run read-only completion_next_proof_packet once: "
                    f"{results!r}"
                )
            if "Jarvis completion next proof packet" not in res.response:
                raise SystemExit(
                    f"next-proof phrase {phrase!r} should expose the next proof packet, "
                    f"not cockpit/chat/wiki drift: {res.response[:240]!r}"
                )
            if "Did you mean" in res.response or "Wikipedia" in res.response:
                raise SystemExit(f"next-proof phrase {phrase!r} drifted into stale routing: {res.response[:240]!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"next-proof phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"next-proof phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"next-proof phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"next-proof phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )

        for phrase in [
            "did acceptance pass",
            "has Jarvis passed acceptance",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "completion_claim_gate":
                raise SystemExit(
                    f"acceptance-verdict phrase {phrase!r} should run completion_claim_gate once: {results!r}"
                )
            if "Jarvis completion claim gate" not in res.response or "BLOCKED" not in res.response:
                raise SystemExit(
                    f"acceptance-verdict phrase {phrase!r} should expose the completion claim gate, "
                    f"not chat or optimistic pass wording: {res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"acceptance-verdict phrase {phrase!r} must stay read-only")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"acceptance-verdict phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"acceptance-verdict phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            if tool_meta.get("authorizes_completion_claim") is not False:
                raise SystemExit(
                    f"acceptance-verdict phrase {phrase!r} must not authorize completion claims: {tool_meta!r}"
                )

        for phrase in [
            "what blocks August completion",
            "what blocks completion",
            "what is blocking August completion?",
            "what is blocking completion?",
            "what is blocking Jarvis from being done?",
            "whats blocking completion?",
            "completion status",
            "Jarvis completion status",
            "완료 뭐 막혀?",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "completion_audit_packet":
                raise SystemExit(
                    f"completion-blocker phrase {phrase!r} should run completion_audit_packet once: {results!r}"
                )
            if "completion audit packet" not in res.response.lower():
                raise SystemExit(
                    f"completion-blocker phrase {phrase!r} should expose the completion audit, "
                    f"not chat/readiness drift: {res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"completion-blocker phrase {phrase!r} must stay read-only")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"completion-blocker phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"completion-blocker phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            if tool_meta.get("authorizes_completion_claim") is not False:
                raise SystemExit(
                    f"completion-blocker phrase {phrase!r} must not authorize completion claims: {tool_meta!r}"
                )

        for phrase in [
            "what is missing",
            "what is still missing",
            "what is unfinished",
            "what remains unfinished",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(
                    f"unfinished-work phrase {phrase!r} should run capability_cockpit once: {results!r}"
                )
            if "Capability cockpit" not in res.response:
                raise SystemExit(
                    f"unfinished-work phrase {phrase!r} should expose the cockpit, "
                    f"not chat/wiki drift: {res.response[:240]!r}"
                )
            if "Did you mean" in res.response or "Wikipedia" in res.response:
                raise SystemExit(f"unfinished-work phrase {phrase!r} drifted into stale routing: {res.response[:240]!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"unfinished-work phrase {phrase!r} must stay read-only")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"unfinished-work phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"unfinished-work phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
                if tool_meta.get(key) is not False:
                    raise SystemExit(
                        f"unfinished-work phrase {phrase!r} must not grant {key}: {tool_meta!r}"
                    )

        for phrase in [
            "what live results do you need from me",
            "what results do you need from me",
            "what format should I use for live test results",
            "how should I report live test results",
            "how do I report live test results",
            "how should I report the live matrix",
            "how to report live test results",
            "live test result format",
            "live test results format",
            "live matrix result format",
            "report live matrix results",
            "report live test results",
            "what should I send after testing",
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "capability_cockpit":
                raise SystemExit(
                    f"live-result-format phrase {phrase!r} should run capability_cockpit once: {results!r}"
                )
            if "Capability cockpit" not in res.response:
                raise SystemExit(
                    f"live-result-format phrase {phrase!r} should expose the cockpit, "
                    f"not chat/wiki drift: {res.response[:240]!r}"
                )
            if "Did you mean" in res.response or "Wikipedia" in res.response:
                raise SystemExit(
                    f"live-result-format phrase {phrase!r} drifted into stale routing: {res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"live-result-format phrase {phrase!r} must stay read-only")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(
                    f"live-result-format phrase {phrase!r} should execute through the normal tool route: {meta!r}"
                )
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"live-result-format phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"live-result-format phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
                if tool_meta.get(key) is not False:
                    raise SystemExit(
                        f"live-result-format phrase {phrase!r} must not grant {key}: {tool_meta!r}"
                    )

        for phrase, expected_tool, expected_snippet in [
            ("code changes status", "build_progress_report", "Jarvis build progress report"),
            ("diff status", "build_progress_report", "Jarvis build progress report"),
            ("patch status", "build_progress_report", "Jarvis build progress report"),
            ("what changed after Codex", "build_progress_report", "Jarvis build progress report"),
            ("what changed in code", "build_progress_report", "Jarvis build progress report"),
            ("what did you change", "build_progress_report", "Jarvis build progress report"),
            ("what did you just change", "build_progress_report", "Jarvis build progress report"),
            ("what did you just do", "build_progress_report", "Jarvis build progress report"),
            ("what did Codex change", "build_progress_report", "Jarvis build progress report"),
            ("what did Codex just change", "build_progress_report", "Jarvis build progress report"),
            ("what did Codex just do", "build_progress_report", "Jarvis build progress report"),
            ("what did Codex modify", "build_progress_report", "Jarvis build progress report"),
            ("what did you edit", "build_progress_report", "Jarvis build progress report"),
            ("what files did Codex edit", "build_progress_report", "Jarvis build progress report"),
            ("what files changed", "build_progress_report", "Jarvis build progress report"),
            ("which files changed", "build_progress_report", "Jarvis build progress report"),
            ("which files did you edit", "build_progress_report", "Jarvis build progress report"),
            ("show changed files", "build_progress_report", "Jarvis build progress report"),
            ("show modified files", "build_progress_report", "Jarvis build progress report"),
            ("what should Claude review", "build_progress_report", "Jarvis build progress report"),
            ("what should I review in this change", "build_progress_report", "Jarvis build progress report"),
            ("what did Claude tell Codex", "handoff_brief", "Jarvis Handoff Brief"),
            ("what did Claude leave", "handoff_brief", "Jarvis Handoff Brief"),
            ("what did Claude leave for Codex", "handoff_brief", "Jarvis Handoff Brief"),
            ("what did Claude hand off", "handoff_brief", "Jarvis Handoff Brief"),
            ("what did Claude say", "handoff_brief", "Jarvis Handoff Brief"),
            ("handoff", "handoff_brief", "Jarvis Handoff Brief"),
            ("handoff brief status", "handoff_brief", "Jarvis Handoff Brief"),
            ("handoff notice", "handoff_brief", "Jarvis Handoff Brief"),
            ("handoff report", "handoff_brief", "Jarvis Handoff Brief"),
            ("handoff status", "handoff_brief", "Jarvis Handoff Brief"),
            ("what is the handoff", "handoff_brief", "Jarvis Handoff Brief"),
            ("check Codex tasks", "work_queue", "Jarvis work queue"),
            ("check current instructions", "work_queue", "Jarvis work queue"),
            ("Codex current instructions", "work_queue", "Jarvis work queue"),
            ("Codex tasks", "work_queue", "Jarvis work queue"),
            ("Codex work queue", "work_queue", "Jarvis work queue"),
            ("current instructions", "work_queue", "Jarvis work queue"),
            ("current work queue", "work_queue", "Jarvis work queue"),
            ("show current instructions", "work_queue", "Jarvis work queue"),
            ("what are current instructions", "work_queue", "Jarvis work queue"),
            ("what is the current work queue", "work_queue", "Jarvis work queue"),
            ("what should Codex do next", "work_queue", "Jarvis work queue"),
            ("what should Codex work on", "work_queue", "Jarvis work queue"),
            ("클로드가 뭐 남겼어", "handoff_brief", "Jarvis Handoff Brief"),
            ("클로드 인수인계", "handoff_brief", "Jarvis Handoff Brief"),
            ("핸드오프 상태", "handoff_brief", "Jarvis Handoff Brief"),
            ("인수인계 상태", "handoff_brief", "Jarvis Handoff Brief"),
            ("코덱스 작업 뭐야", "work_queue", "Jarvis work queue"),
            ("코덱스 작업 큐", "work_queue", "Jarvis work queue"),
            ("현재 지시사항 보여줘", "work_queue", "Jarvis work queue"),
            ("작업 큐", "work_queue", "Jarvis work queue"),
            ("작업 큐 보여줘", "work_queue", "Jarvis work queue"),
            ("현재 작업목록", "work_queue", "Jarvis work queue"),
            ("뭐 수정했어", "build_progress_report", "Jarvis build progress report"),
            ("어떤 파일 바꿨어", "build_progress_report", "Jarvis build progress report"),
            ("변경 파일 보여줘", "build_progress_report", "Jarvis build progress report"),
            ("수정 파일 보여줘", "build_progress_report", "Jarvis build progress report"),
            ("코덱스가 뭐 바꿨어", "build_progress_report", "Jarvis build progress report"),
            ("코덱스 변경사항", "build_progress_report", "Jarvis build progress report"),
            ("최근 변경사항", "build_progress_report", "Jarvis build progress report"),
            ("패치 상태", "build_progress_report", "Jarvis build progress report"),
            ("변경사항 확인", "build_progress_report", "Jarvis build progress report"),
        ]:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != expected_tool:
                raise SystemExit(
                    f"handoff/change phrase {phrase!r} should run {expected_tool} once: {results!r}"
                )
            if expected_snippet not in res.response:
                raise SystemExit(
                    f"handoff/change phrase {phrase!r} should expose {expected_snippet!r}, "
                    f"not chat/wiki drift: {res.response[:240]!r}"
                )
            if "Did you mean" in res.response or "Wikipedia" in res.response:
                raise SystemExit(
                    f"handoff/change phrase {phrase!r} drifted into stale routing: {res.response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"handoff/change phrase {phrase!r} must stay read-only")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(
                    f"handoff/change phrase {phrase!r} should execute through the normal tool route: {meta!r}"
                )
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"handoff/change phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"handoff/change phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
                if tool_meta.get(key) is True:
                    raise SystemExit(
                        f"handoff/change phrase {phrase!r} must not grant {key}: {tool_meta!r}"
                    )


def test_runtime_exact_read_only_tool_name_aliases_execute_status_tools() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "approval_queue_summary": "approval_queue_summary",
        "capability_map": "capability_map",
        "channel_health": "channel_health",
        "completion_claim_gate": "completion_claim_gate",
        "evidence_ledger": "evidence_ledger",
        "execution_health_report": "execution_health_report",
        "harness_status": "harness_status",
        "harness_readiness_digest": "harness_readiness_digest",
        "jarvis_doctor": "jarvis_doctor",
        "jarvis_status": "jarvis_status",
        "learning_review": "learning_review",
        "list_scheduled_jobs": "list_scheduled_jobs",
        "list_tools": "list_tools",
        "memory_stats": "memory_stats",
        "model_routing_status": "model_routing_status",
        "privacy_report": "privacy_report",
        "readiness_report": "readiness_report",
        "recent_tool_runs": "recent_tool_runs",
        "risk_matrix": "risk_matrix",
        "safety_status": "safety_status",
        "storage_recovery_check": "storage_recovery_check",
        "storage_recovery_plan": "storage_recovery_plan",
        "storage_status": "storage_status",
        "show storage stats": "storage_status",
        "storage stats": "storage_status",
        "storage statistics": "storage_status",
        "what is storage status": "storage_status",
        "subagent_fleet_status": "subagent_fleet_status",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected_tool in cases.items():
            tool = rt.registry.get(expected_tool)
            if tool.risk.name != "READ_ONLY":
                raise SystemExit(f"raw tool-name alias must stay read-only: {phrase!r} -> {tool.risk.name}")
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != expected_tool:
                raise SystemExit(f"{phrase!r} should execute {expected_tool!r} once: {results!r}")
            if not res.verified:
                raise SystemExit(f"{phrase!r} should verify through the normal runtime route: {res.response[:160]!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"{phrase!r} must not queue approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"{phrase!r} should execute as a normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"{phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"{phrase!r} should mark exact alias metadata: {planner_metadata!r}")


def test_runtime_exact_registered_read_only_tool_names_execute_without_manual_aliases() -> None:
    from contextlib import nullcontext
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from jarvis_v2.agent.runtime import _RUNTIME_EXACT_TOOL_ALIAS_RAW, _runtime_exact_tool_alias_plan
    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    dynamic_cases = {
        "architecture_map": "architecture_map",
        "agi_gate_report": "agi_gate_report",
        "agi_next_build_move": "agi_next_build_move",
        "after_action_learning_packet": "after_action_learning_packet",
        "approval_history": "approval_history",
        "autonomy_continuation_execution_packet": "autonomy_continuation_execution_packet",
        "autonomy_resume_gate": "autonomy_resume_gate",
        "autonomy_step_closure_packet": "autonomy_step_closure_packet",
        "build_target_packet": "build_target_packet",
        "chat_continuity_brief": "chat_continuity_brief",
        "chat_response_health": "chat_response_health",
        "checkpoint_recovery_cockpit": "checkpoint_recovery_cockpit",
        "checkpoint_recovery_apply_packet": "checkpoint_recovery_apply_packet",
        "checkpoint_recovery_followthrough_packet": "checkpoint_recovery_followthrough_packet",
        "checkpoint_recovery_preview": "checkpoint_recovery_preview",
        "coding_discipline_packet": "coding_discipline_packet",
        "completion_audit_packet": "completion_audit_packet",
        "completion_next_proof_packet": "completion_next_proof_packet",
        "completion_proof_refresh_packet": "completion_proof_refresh_packet",
        "continuation_packet": "continuation_packet",
        "computer_control_readiness": "computer_control_readiness",
        "computer_control_status": "computer_control_status",
        "current_time": "current_time",
        "execution_case_closure_packet": "execution_case_closure_packet",
        "execution_case_evidence_packet": "execution_case_evidence_packet",
        "execution_case_gate": "execution_case_gate",
        "execution_case_review_packet": "execution_case_review_packet",
        "execution_case_timeline": "execution_case_timeline",
        "execution_audit_gate": "execution_audit_gate",
        "execution_learning_closure_packet": "execution_learning_closure_packet",
        "execution_recovery_packet": "execution_recovery_packet",
        "failure_learning_cockpit": "failure_learning_cockpit",
        "feedback_actions": "feedback_actions",
        "flip_coin": "flip_coin",
        "focus_brief": "focus_brief",
        "generate_password": "generate_password",
        "generate_uuid": "generate_uuid",
        "handoff_brief": "handoff_brief",
        "harness_build_slice": "harness_build_slice",
        "harness_completion_assessment": "harness_completion_assessment",
        "harness_operations_brief": "harness_operations_brief",
        "harness_doctrine": "harness_doctrine",
        "inspect_execution_case": "inspect_execution_case",
        "integration_execution_matrix": "integration_execution_matrix",
        "jarvis_help": "jarvis_help",
        "list_decisions": "list_decisions",
        "list_duplicate_memories": "list_duplicate_memories",
        "list_goals": "list_goals",
        "list_people": "list_people",
        "list_preferences": "list_preferences",
        "list_reminders": "list_reminders",
        "list_sessions": "list_sessions",
        "list_skills": "list_skills",
        "list_tasks": "list_tasks",
        "list_voices": "list_voices",
        "list_weak_memories": "list_weak_memories",
        "morning_startup": "morning_startup",
        "next_actions": "next_actions",
        "next_action_packet": "next_action_packet",
        "next_session_plan": "next_session_plan",
        "next_task": "next_task",
        "observe_act_verify_cockpit": "observe_act_verify_cockpit",
        "operator_instruction_supersession_packet": "operator_instruction_supersession_packet",
        "operator_handoff_packet": "operator_handoff_packet",
        "operator_timebox_contract": "operator_timebox_contract",
        "overdue_tasks": "overdue_tasks",
        "priority_goal": "priority_goal",
        "priority_stack": "priority_stack",
        "prototype_readiness_checklist": "prototype_readiness_checklist",
        "recent_conversation": "recent_conversation",
        "recent_memories": "recent_memories",
        "recent_saved_notes": "recent_saved_notes",
        "recovery_closure_checklist": "recovery_closure_checklist",
        "repeated_failure_clusters": "repeated_failure_clusters",
        "return_brief": "return_brief",
        "safe_next_actions": "safe_next_actions",
        "scheduler_context_refresh_packet": "scheduler_context_refresh_packet",
        "session_closeout": "session_closeout",
        "setup_check": "setup_check",
        "status_dashboard": "status_dashboard",
        "system_info": "system_info",
        "task_board": "task_board",
        "task_overview": "task_overview",
        "voice_setup_check": "voice_setup_check",
        "work_block_checkpoint": "work_block_checkpoint",
        "work_queue": "work_queue",
        "work_session_packet": "work_session_packet",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected_tool in dynamic_cases.items():
            if phrase in _RUNTIME_EXACT_TOOL_ALIAS_RAW:
                raise SystemExit(f"{phrase!r} should prove the dynamic registry-backed alias path, not the manual map")
            tool = rt.registry.get(expected_tool)
            if tool.risk.name != "READ_ONLY":
                raise SystemExit(f"dynamic exact raw alias must stay read-only: {phrase!r} -> {tool.risk.name}")
            plan = _runtime_exact_tool_alias_plan(phrase, rt.registry)
            if plan is None or len(plan.actions) != 1 or plan.actions[0].tool_name != expected_tool:
                raise SystemExit(f"{phrase!r} should build a dynamic exact-alias plan: {plan!r}")
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            owner_env = (
                patch.dict("os.environ", {"JARVIS_OWNER_TELEGRAM": "smoke-owner"})
                if expected_tool == "list_reminders"
                else nullcontext()
            )
            with owner_env:
                res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != expected_tool:
                raise SystemExit(f"{phrase!r} should execute {expected_tool!r} once: {results!r}")
            if not res.verified:
                raise SystemExit(f"{phrase!r} should verify through the normal runtime route: {res.response[:160]!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"{phrase!r} must not queue approvals")
            trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"{phrase!r} should mark exact alias metadata: {planner_metadata!r}")

        excluded_plan = _runtime_exact_tool_alias_plan("voice_command_cockpit", rt.registry)
        if excluded_plan is not None:
            raise SystemExit(f"raw excludes must block exact alias execution: {excluded_plan!r}")
        approvals_before = len(rt.store.list_pending_approvals(limit=100))
        excluded_res = rt.handle("voice_command_cockpit")
        approvals_after = len(rt.store.list_pending_approvals(limit=100))
        if getattr(excluded_res, "tool_results", []) or approvals_after != approvals_before:
            raise SystemExit("excluded raw alias must not execute tools or queue approvals")
        excluded_text = excluded_res.response.lower()
        if "needs arguments" not in excluded_text and "did you mean" not in excluded_text:
            raise SystemExit(f"excluded raw alias should remain non-executable guidance: {excluded_res.response!r}")

        blocked_cases = [
            "action_readiness_packet",
            "action_rehearsal",
            "argument_contract_packet",
            "run_shell_command",
            "get_clipboard",
            "write_text_file",
            "voice_command_cockpit",
            "approval_readiness_packet",
            "choose_option",
            "command_cockpit_packet",
            "command_diagnosis",
            "command_intake_packet",
            "computer_task_plan",
            "days_until",
            "dispatch_decision_packet",
            "execution_acceptance_gate",
            "execution_contract",
            "execution_governor_packet",
            "execution_readiness_matrix",
            "failure_promotion_packet",
            "failure_to_test_preview",
            "fetch_page",
            "get_memory",
            "planner_gap_packet",
            "random_number",
            "roll_dice",
            "screen_verification_contract",
            "tool_detail",
            "verification_packet",
            "verification_receipt",
            "voice_confirmation_packet",
            "web_lookup",
        ]
        for blocked in blocked_cases:
            plan = _runtime_exact_tool_alias_plan(blocked, rt.registry)
            if plan is not None:
                raise SystemExit(f"{blocked!r} must not be accepted by the dynamic READ_ONLY exact alias bridge: {plan!r}")

        blocked_hint_cases = {
            "action_readiness_packet": "action readiness: run a script and email me the result",
            "action_rehearsal": "rehearse: run command python3 --version",
            "argument_contract_packet": "argument contract: run command python3 --version",
            "approval_readiness_packet": "approval readiness <approval id>",
            "command_cockpit_packet": "command cockpit: organize my downloads and summarize what changed",
            "command_diagnosis": "command diagnosis: run command python3 --version",
            "command_intake_packet": "command intake: organize my downloads and summarize what changed",
            "dispatch_decision_packet": "dispatch decision: summarize my recent Jarvis work",
            "execution_acceptance_gate": (
                "acceptance gate: organize downloads; evidence recent tool run ok; tests smoke passed"
            ),
            "execution_contract": "execution contract: organize my downloads and summarize what changed",
            "execution_governor_packet": "execution governor: organize my downloads and summarize what changed",
            "execution_readiness_matrix": (
                "execution readiness matrix: organize my downloads and summarize what changed"
            ),
            "failure_promotion_packet": "failure promotion packet: dashboard layout",
            "failure_to_test_preview": "failure to test preview: Jarvis overlapped dashboard text",
            "fetch_page": "fetch page https://example.com",
            "planner_gap_packet": "planner gap: organize my downloads",
            "verification_packet": "verification packet: organize my downloads and summarize what changed",
            "verification_receipt": "verification receipt <run id>",
            "voice_command_cockpit": "voice command cockpit: <transcript>",
        }
        for blocked, expected_example in blocked_hint_cases.items():
            res = rt.handle(blocked)
            results = getattr(res, "tool_results", []) or []
            if results:
                raise SystemExit(f"{blocked!r} should not execute as a no-arg exact alias: {results!r}")
            response = getattr(res, "response", "")
            if "needs arguments" not in response or "Did you mean" not in response or expected_example not in response:
                raise SystemExit(
                    f"{blocked!r} should redirect to an argument-aware command hint "
                    f"{expected_example!r}: {response!r}"
                )
            trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is True:
                raise SystemExit(f"{blocked!r} should not mark exact alias metadata: {planner_metadata!r}")
            chat_response = (getattr(res, "metadata", {}) or {}).get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"{blocked!r} should stay on the command-suggestion route: {chat_response!r}")
            if trace.get("approval_queue_delta") != 0 or trace.get("new_approval_ids"):
                raise SystemExit(f"{blocked!r} should not alter the approval queue: {trace!r}")


def test_runtime_web_search_returns_name_bound_success_without_widening_fetch_page() -> None:
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from jarvis_v2.scripts.test_runtime import make_temp_runtime
    from jarvis_v2.tools import browser

    html = "<html><title>Mock search</title><body>Mocked search result content.</body></html>"
    with tempfile.TemporaryDirectory(prefix="jarvis-web-search-runtime-") as tmp:
        rt = make_temp_runtime(Path(tmp))
        with patch.object(browser, "_fetch", side_effect=lambda url, timeout=12: (html, url)):
            result = rt.handle("web search jarvis browser receipt")

        if len(result.tool_results) != 1:
            raise SystemExit(f"natural web_search route should execute exactly once: {result.tool_results!r}")
        receipt = result.tool_results[0]
        if not receipt.ok or receipt.tool_name != "web_search":
            raise SystemExit(f"web_search should return a successful name-bound receipt: {receipt}")
        if receipt.metadata.get("history_saved") is not True or receipt.metadata.get("writes_database") is not True:
            raise SystemExit(f"web_search receipt should truthfully report persisted history: {receipt.metadata}")
        if receipt.metadata.get("failure_kind") == "tool_result_binding_mismatch":
            raise SystemExit(f"web_search receipt should not fail runtime binding: {receipt.metadata}")
        runs = rt.store.recent_tool_runs(limit=10)
        if not runs or runs[0]["tool_name"] != "web_search" or not runs[0]["ok"]:
            raise SystemExit(f"web_search should leave one successful bound audit row: {runs}")

        hint = rt.handle("fetch_page")
        if hint.tool_results:
            raise SystemExit(f"bare fetch_page must remain a non-executing argument hint: {hint.tool_results!r}")
        if "needs arguments" not in hint.response or "fetch page https://example.com" not in hint.response:
            raise SystemExit(f"bare fetch_page should retain its exact argument hint: {hint.response!r}")


def test_runtime_redirects_argument_required_read_only_command_stubs_before_chat_or_empty_plans() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "approval chain proof": "approval chain proof <approval id>",
        "approval packet": "approval packet <approval id>",
        "approval readiness": "approval readiness <approval id>",
        "choose option": "choose option: <option A> | <option B>",
        "computer task plan": "computer task plan: <task>",
        "days until": "days until 2026-12-31",
        "fetch page": "fetch page https://example.com",
        "get memory": "get memory <query>",
        "screen verification": "screen verification: <expected screen state>",
        "tool detail": "tool detail: <tool name>",
        "verification receipt for": "verification receipt <run id>",
        "voice command cockpit": "voice command cockpit: <transcript>",
        "voice confirmation": "voice confirmation: <transcript>",
        "voice confirmation receipt": "voice confirmation receipt: <transcript> confirmed=true",
        "voice route gate": "voice route gate: <transcript> confirmed=true",
        "voice route proof bundle": "voice route proof bundle: <transcript> confirmed=true",
        "voice runtime bridge": "voice runtime bridge: <transcript> confirmed=true",
        "voice transcript review": "voice transcript review: <transcript>",
        "voice action audit": "voice action audit: <transcript> confirmed=true",
        "voice execution handoff": "voice execution handoff: <transcript> confirmed=true",
        "voice post run closure": "voice post-run closure: <transcript> confirmed=true",
        "voice cycle ledger": "voice cycle ledger: <transcript> confirmed=true",
        "voice audio file gate": "voice audio file gate: /path/to/audio.m4a consent=true",
        "voice reply preview": "voice reply preview: <text>",
        "web lookup": "web lookup <query>",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected_example in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "needs arguments" not in response or "Did you mean" not in response or expected_example not in response:
                raise SystemExit(
                    f"{phrase!r} should redirect to argument-aware command hint "
                    f"{expected_example!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"{phrase!r} should not queue approvals")
            results = getattr(res, "tool_results", []) or []
            if results:
                raise SystemExit(f"{phrase!r} should not execute a read-only tool without arguments: {results!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"{phrase!r} should stay on command-suggestion route: {chat_response!r}")
            trace = meta.get("runtime_trace") or {}
            if trace.get("approval_queue_delta") != 0 or trace.get("new_approval_ids"):
                raise SystemExit(f"{phrase!r} should not alter approval queue: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is True:
                raise SystemExit(f"{phrase!r} should not mark exact alias metadata: {planner_metadata!r}")


def test_runtime_routes_subagent_status_to_dedicated_read_only_tool() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in [
            "subagent status",
            "subagent health",
            "show subagent status",
            "show agent status",
            "agent health",
            "agent fleet health",
            "are agents ready",
            "how many agents are ready",
            "show worker status",
            "worker health",
            "worker fleet health",
            "tool orchestration status",
            "parallel agents status",
            "what agents are ready",
            "which agents are running",
            "에이전트 상태",
            "에이전트 준비 상태",
            "에이전트 헬스",
            "서브에이전트 상태",
            "서브에이전트 준비 상태",
            "준비된 에이전트",
            "준비된 에이전트 몇 개",
            "워커 준비 상태",
            "워커 헬스",
            "병렬 에이전트 상태",
        ]:
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Internal worker fleet status:" not in response or "ready workers:" not in response:
                raise SystemExit(f"runtime should route subagent status to the fleet status tool: {phrase!r} -> {response!r}")
            if "jarvis status" in response:
                raise SystemExit(f"runtime subagent status should not use broad jarvis status: {phrase!r} -> {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            if meta.get("runtime_route") != "tools":
                raise SystemExit(f"runtime should tag subagent status as tools route: {phrase!r} -> {meta!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or tool_results[0].tool_name != "subagent_fleet_status":
                raise SystemExit(f"subagent status should execute exactly the read-only fleet tool: {phrase!r} -> {tool_results!r}")
            if tool_results[0].metadata.get("requires_approval") is not False:
                raise SystemExit(f"subagent fleet status should not require approval: {phrase!r} -> {tool_results[0].metadata!r}")

        direct = rt.handle("subagent fleet status")
        if direct.verified is not True or not direct.tool_results:
            raise SystemExit(f"direct subagent fleet status should execute read-only status tool: {direct}")
        if "Internal worker fleet status:" not in direct.response or "ready workers:" not in direct.response:
            raise SystemExit(f"direct subagent fleet status response missing status summary: {direct.response!r}")


def test_runtime_redirects_model_brain_intents_to_model_routing_status() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = [
        "brain health",
        "planner status",
        "what model are you using",
        "ollama status",
        "model health",
        "local model status",
        "chat model status",
    ]
    wrong_fragments = ["integration status", "brain search", "planner gap"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in cases:
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "model routing status" not in response and "Jarvis model routing status" not in response:
                raise SystemExit(f"runtime should redirect model/brain phrase {phrase!r} to model routing status: {response!r}")
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"runtime model/brain phrase suggested stale/wrong command {wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response and chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag model/brain phrase {phrase!r} as a command suggestion: {chat_response!r}")


def test_runtime_preplanner_redirects_model_brain_failure_intents_to_model_routing_status() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = [
        "model failed",
        "model not working",
        "local model error",
        "ollama failed",
        "chat model not working",
        "planner model failed",
        "why did model fail",
        "why did ollama fail",
        "모델 오류",
        "모델 안됨",
        "브레인 문제",
        "올라마 실패",
        "로컬 모델 실패",
        "플래너 모델 안됨",
        "채팅 모델 오류",
        "LLM 실패",
    ]
    wrong_fragments = ["channel health", "recent tool runs", "queued as approval", "Safety receipt"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or "model routing status" not in response:
                raise SystemExit(
                    f"runtime should redirect model/brain failure phrase {phrase!r} "
                    f"to model routing status: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"model/brain failure phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(
                    "model/brain failure phrase should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"model/brain failure phrase suggested stale/wrong command "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag model/brain failure phrase {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark model/brain failure redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "model/brain failure redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )


def test_runtime_redirects_eval_verification_intents_to_read_only_proof_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "eval status": "cockpit",
        "operator evals": "cockpit",
        "operator workflow evals": "cockpit",
        "operator real tasks": "cockpit",
        "workflow evals": "cockpit",
        "workflow tests": "cockpit",
        "is Korean messaging tested": "cockpit",
        "is Korean telegram covered": "cockpit",
        "Korean message proof": "cockpit",
        "Korean send proof": "cockpit",
        "Hangul send proof": "cockpit",
        "can Jarvis safely send Korean messages": "cockpit",
        "test matrix status": "cockpit",
        "live test matrix": "cockpit",
        "live matrix status": "cockpit",
        "channel proof status": "cockpit",
        "evaluation status": "cockpit",
        "regression status": "cockpit",
        "verification status": "verification receipt latest",
        "verification proof": "verification receipt latest",
        "what was verified": "verification receipt latest",
        "latest verification": "verification receipt latest",
        "evidence": "evidence ledger",
        "show me proof": "evidence ledger",
        "show me evidence": "evidence ledger",
        "what proof do we have": "evidence ledger",
        "what evidence do we have": "evidence ledger",
    }
    wrong_fragments = [
        "storage status",
        "goal 1 status",
        "integration status",
        "verification packet",
        "what goals do I have",
        "what goals do i have",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect eval/proof phrase {phrase!r} to {expected!r}: {response!r}")
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"runtime eval/proof phrase suggested stale/wrong command {wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag eval/proof phrase {phrase!r} as a command suggestion: {chat_response!r}")
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(f"eval/proof redirect should not alter approval queue: {phrase!r} -> {runtime_trace}")
            if runtime_trace.get("planned_actions") or runtime_trace.get("tool_results"):
                raise SystemExit(f"eval/proof redirect should not execute tools: {phrase!r} -> {runtime_trace}")


def test_runtime_redirects_approval_gate_intents_to_read_only_approval_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    direct_summary_cases = {
        "approval summary",
        "approval dashboard",
        "approvals status",
        "approval status",
        "approval report",
        "approval health",
        "approval gate",
        "approval gates",
        "approval gate status",
        "what is blocked by approval",
        "blocked by approval",
    }
    cases = {
        "approval latest": "approval readiness latest",
        "approval proof": "approval evidence for latest",
        "approval evidence": "approval evidence for latest",
        "approval receipt": "approval evidence for latest",
    }
    direct_read_only_cases = {
        "approval queue": "No pending approvals.",
        "pending approvals": "No pending approvals.",
        "show approvals": "No pending approvals.",
        "show me approvals": "No pending approvals.",
        "show me pending approvals": "No pending approvals.",
        "what approvals are waiting": "No pending approvals.",
        "are approvals pending": "No pending approvals.",
        "anything waiting on me": "No pending approvals.",
        "any approvals pending": "No pending approvals.",
        "anything need approval": "No pending approvals.",
        "anything needs my approval": "No pending approvals.",
        "anything pending approval": "No pending approvals.",
        "approval needed": "No pending approvals.",
        "do you need anything from me": "No pending approvals.",
        "do you need me for anything": "No pending approvals.",
        "do you need my approval": "No pending approvals.",
        "needs approval": "No pending approvals.",
        "pending approval": "No pending approvals.",
        "pending approval status": "No pending approvals.",
        "pending review": "No pending approvals.",
        "review queue": "No pending approvals.",
        "show pending approval status": "No pending approvals.",
        "show pending approvals": "No pending approvals.",
        "what do you need from me": "No pending approvals.",
        "what is blocked by me": "No pending approvals.",
        "what is pending": "No pending approvals.",
        "what is pending approval": "No pending approvals.",
        "what is waiting on me": "No pending approvals.",
        "what should i review": "No pending approvals.",
        "waiting for me": "No pending approvals.",
        "what approvals are pending": "No pending approvals.",
        "which approvals are pending": "No pending approvals.",
    }
    wrong_fragments = ["jarvis status", "privacy report", "approval packet 1", "approval review", "setup check"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in direct_read_only_cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if expected not in response:
                raise SystemExit(f"runtime should execute read-only approval queue phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"read-only approval queue phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "list_pending_approvals":
                raise SystemExit(f"approval queue phrase should only run list_pending_approvals: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"read-only approval queue phrase should stay ungated and queue-neutral: {runtime_trace!r}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"approval queue phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase in direct_summary_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Approval queue summary:" not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute approval summary phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"read-only approval summary phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "approval_queue_summary":
                raise SystemExit(f"approval summary phrase should only run approval_queue_summary: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"read-only approval summary phrase should stay ungated and queue-neutral: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"approval summary phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect approval-gate phrase {phrase!r} to {expected!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"approval-gate phrase should not queue approvals before the owner sends the suggested command: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(f"approval-gate phrase should not execute tools before suggestion acceptance: {phrase!r}")
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"runtime approval-gate phrase suggested stale/wrong command {wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag approval-gate phrase {phrase!r} as a command suggestion: {chat_response!r}")
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(f"approval-gate redirect should not alter approval queue: {phrase!r} -> {runtime_trace}")


def test_runtime_executes_approval_failure_intents_as_read_only_summary() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = [
        "approval broken",
        "approval failed",
        "approval not working",
        "approval button stuck",
        "approval queue failed",
        "approve button not working",
        "why did approval fail",
        "승인 실패",
        "승인 안됨",
        "승인 버튼 안돼",
        "승인 큐 멈춤",
        "왜 승인 실패",
        "허가 오류",
    ]
    wrong_fragments = [
        "Did you mean",
        "channel health",
        "recent tool runs",
        "queued as approval",
        "Safety receipt",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Approval queue summary:" not in response:
                raise SystemExit(
                    f"runtime should execute approval failure phrase {phrase!r} "
                    f"as approval_queue_summary: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"approval failure phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "approval_queue_summary":
                raise SystemExit(f"approval failure phrase should only run approval_queue_summary: {phrase!r} {tool_results!r}")
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"approval failure phrase returned stale/wrong command "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            runtime_trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or runtime_trace.get("route") != "tools":
                raise SystemExit(f"approval failure phrase should execute through normal tool route: {phrase!r} {meta!r}")
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "approval failure summary should not require approval or alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"approval failure phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")


def test_runtime_redirects_memory_learning_intents_to_read_only_memory_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "memory status": ("memory_stats", "No memories yet."),
        "memory health": ("memory_stats", "No memories yet."),
        "memory report": ("memory_stats", "No memories yet."),
        "memory check": ("memory_stats", "No memories yet."),
        "knowledge status": ("memory_stats", "No memories yet."),
        "brain memory status": ("memory_stats", "No memories yet."),
        "memory review": ("learning_review", "Jarvis learning review:"),
        "review memories": ("learning_review", "Jarvis learning review:"),
        "review memory": ("learning_review", "Jarvis learning review:"),
        "what did you learn": ("learning_review", "Jarvis learning review:"),
        "what have you learned": ("learning_review", "Jarvis learning review:"),
        "what should Jarvis learn": ("learning_review", "Jarvis learning review:"),
        "learning status": ("learning_review", "Jarvis learning review:"),
        "learning loop status": ("learning_review", "Jarvis learning review:"),
        "learning health": ("learning_review", "Jarvis learning review:"),
        "after action learning status": ("after_action_learning_packet", "Jarvis after-action learning packet:"),
        "after-action learning status": ("after_action_learning_packet", "Jarvis after-action learning packet:"),
        "what did Jarvis learn from the last failure": ("after_action_learning_packet", "Jarvis after-action learning packet:"),
        "what did Jarvis learn from last run": ("after_action_learning_packet", "Jarvis after-action learning packet:"),
        "is learning debt closed": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "learning debt status": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "what learning debt is open": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "learning loop proof": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "learning proof matrix": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "recovery closure status": ("recovery_closure_checklist", "Jarvis recovery closure checklist:"),
        "recovery learning status": ("recovery_closure_checklist", "Jarvis recovery closure checklist:"),
        "is recovery debt closed": ("recovery_closure_checklist", "Jarvis recovery closure checklist:"),
        "repeated failure status": ("repeated_failure_clusters", "Jarvis repeated-failure cluster report:"),
        "failure learning status": ("failure_learning_cockpit", "Jarvis failure learning cockpit:"),
        "what does Jarvis remember": ("recent_memories", "No memories yet."),
        "기억 상태": ("memory_stats", "No memories yet."),
        "기억 확인": ("memory_stats", "No memories yet."),
        "메모리 상태": ("memory_stats", "No memories yet."),
        "메모리 확인": ("memory_stats", "No memories yet."),
        "학습 상태": ("learning_review", "Jarvis learning review:"),
        "학습 확인": ("learning_review", "Jarvis learning review:"),
        "학습 루프 상태": ("learning_review", "Jarvis learning review:"),
        "학습 부채 상태": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "학습 부채 닫혔어": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "학습 증명 매트릭스": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "복구 학습 상태": ("recovery_closure_checklist", "Jarvis recovery closure checklist:"),
        "복구 부채 상태": ("recovery_closure_checklist", "Jarvis recovery closure checklist:"),
        "반복 실패 상태": ("repeated_failure_clusters", "Jarvis repeated-failure cluster report:"),
        "실패 학습 상태": ("failure_learning_cockpit", "Jarvis failure learning cockpit:"),
        "기억 리뷰": ("learning_review", "Jarvis learning review:"),
        "무엇을 배웠어": ("learning_review", "Jarvis learning review:"),
        "뭘 배웠어": ("learning_review", "Jarvis learning review:"),
        "뭘 기억해": ("recent_memories", "No memories yet."),
        "자비스가 뭘 기억해": ("recent_memories", "No memories yet."),
        "기억 정리": ("list_weak_memories", "No weak-looking memories found."),
        "약한 기억": ("list_weak_memories", "No weak-looking memories found."),
        "중복 기억": ("list_duplicate_memories", "No duplicate-looking memories found."),
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, (expected_tool, expected_fragment) in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != expected_tool:
                raise SystemExit(
                    f"runtime should execute {expected_tool!r} for memory/learning phrase "
                    f"{phrase!r}: {results!r}"
                )
            if "Did you mean" in response:
                raise SystemExit(f"runtime should execute memory/learning phrase, not suggest: {phrase!r} -> {response!r}")
            if expected_fragment not in response:
                raise SystemExit(
                    f"runtime memory/learning phrase {phrase!r} should expose {expected_tool!r} output: "
                    f"{response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"memory/learning phrase {phrase!r} must remain read-only without queuing approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(f"memory/learning phrase {phrase!r} should execute through the normal tool route: {meta!r}")
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"memory/learning phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"memory/learning phrase {phrase!r} should mark the exact alias route: {planner_metadata!r}"
                )


def test_runtime_redirects_freeze_status_to_cockpit() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    phrases = [
        "freeze status",
        "live freeze list",
        "live proof freeze",
        "what is frozen",
        "what files are frozen",
        "can codex edit planner",
        "can codex edit the send code",
        "can you edit the planner",
        "can you edit the send code",
        "what can codex touch",
        "what can you edit",
        "what should codex avoid",
        "what should codex leave alone",
        "what should codex not edit",
        "what should codex not touch",
        "what should you not edit",
        "what should you not touch",
        "show guardrails",
        "live channel proof status",
        "proofs pending",
        "what is the live test matrix",
        "what live tests are pending",
        "what proof is pending",
        "what proofs are still pending",
        "what live proof do you need",
        "what should i test",
        "what tests should i run",
        "what should operator test",
        "what channels should i test",
        "which channels need live proof",
        "what live channels are pending",
        "what is the freeze list",
        "가드레일",
        "프리즈 상태",
        "동결 파일",
        "수정 금지 파일",
        "뭐 테스트해야 해",
        "라이브 테스트 뭐 해야 해",
        "어떤 채널 테스트해",
        "라이브 채널 뭐 테스트해",
        "검증 대기 뭐야",
        "테스트 매트릭스 보여줘",
    ]
    wrong_fragments = [
        "Wikipedia",
        "what should I do now",
        "That sounds like it may involve action",
        "fallback chat",
        "Safety receipt",
        "queued as approval",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in phrases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or "cockpit" not in response:
                raise SystemExit(f"runtime should redirect freeze/guardrail phrase {phrase!r} to cockpit: {response!r}")
            if getattr(res, "tool_results", []):
                raise SystemExit(f"freeze/guardrail redirect should not execute tools: {phrase!r}")
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"freeze/guardrail redirect should not queue approvals: {phrase!r} {approvals_before}->{approvals_after}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"freeze/guardrail redirect leaked stale/wrong route {wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag the guardrail redirect route: {phrase!r} -> {chat_response!r}")
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(f"runtime should mark guardrail redirect as pre-planner: {phrase!r} -> {chat_response!r}")
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    f"freeze/guardrail redirect should not alter approval queue metadata: {phrase!r} -> {runtime_trace}"
                )


def test_runtime_redirects_daily_operator_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "what can i say": "capability map",
        "control center": "cockpit",
        "can jarvis be trusted": "cockpit",
        "is jarvis reliable": "cockpit",
        "is jarvis trustworthy": "cockpit",
        "trust status": "cockpit",
        "trust report": "cockpit",
        "trust health": "cockpit",
        "reliability status": "cockpit",
        "reliability report": "cockpit",
        "reliability health": "cockpit",
        "show trust status": "cockpit",
        "show reliability status": "cockpit",
        "is jarvis safe to use": "safety status",
        "can i use jarvis safely": "safety status",
        "safe to use jarvis": "safety status",
        "dashboard status": "cockpit",
        "capability health": "cockpit",
        "attention needed": "readiness report",
        "ready for me": "readiness report",
        "blocked work": "execution health report",
        "can i retry": "recovery closure checklist",
        "can i retry now": "recovery closure checklist",
        "retry readiness": "recovery closure checklist",
        "retry status": "recovery closure checklist",
        "safe to retry": "recovery closure checklist",
        "should i retry": "recovery closure checklist",
    }
    direct_recent_run_cases = {
        "anything broken",
        "audit trail",
        "broken status",
        "error summary",
        "execution audit",
        "is anything broken",
        "last error",
        "last tool run",
        "last failed tool",
        "last failure",
        "latest error",
        "latest tool run",
        "latest failure",
        "recent errors",
        "recent tool run",
        "show audit trail",
        "show execution audit",
        "show last error",
        "show last failure",
        "show me what broke",
        "show me what is broken",
        "tool execution history",
        "tool run history",
        "what broke",
        "what broke status",
        "what did jarvis run last",
        "what errors happened",
        "what errors are there",
        "what is broken",
        "what is wrong",
        "what ran last",
        "what tool ran last",
        "what's broken",
        "what's wrong",
        "what failed last",
        "what was the last error",
        "what was the last failure",
        "what went wrong",
        "whats broken",
        "whats wrong",
        "why did it fail",
    }
    direct_read_only_cases = {
        "anything failing": "execution health report",
        "failure details": "execution health report",
        "failure report": "execution health report",
        "failure status": "execution health report",
        "recent failures": "execution health report",
        "show me failures": "execution health report",
        "show me what failed": "execution health report",
        "what failed": "execution health report",
        "what failed recently": "execution health report",
        "what is failing": "execution health report",
        "what's failing": "execution health report",
        "whats failing": "execution health report",
    }
    direct_recovery_cases = {
        "how do i recover",
        "recovery plan",
        "recovery status",
        "what needs recovery",
    }
    direct_trace_cases = {
        "runtime trace latest",
    }
    direct_verification_receipt_cases = {
        "last execution receipt",
        "latest execution receipt",
    }
    direct_cockpit_cases = {
        "cockpit attention",
        "cockpit status",
        "cockpit summary",
        "attention",
        "attention status",
        "approval flow status",
        "approval buttons proof matrix",
        "approval callback proof matrix",
        "error guidance status",
        "korean voice status",
        "phone approval status",
        "phone approval result format",
        "phone approval buttons proof matrix",
        "phone approve deny proof matrix",
        "telegram voice status",
        "telegram approval buttons proof matrix",
        "telegram approval result format",
        "what approval buttons should i test",
        "what approval callback should i test",
        "what approve deny buttons should i test",
        "how should i report approval callback proof",
        "how should i report phone approval proof",
        "승인 버튼 증명 매트릭스",
        "폰 승인 버튼 증명",
        "텔레그램 승인 증명",
        "승인 콜백 증명",
        "승인 거절 증명",
        "can i trust jarvis",
        "can i trust you",
        "trust checklist",
        "earned trust",
        "earned trust checklist",
        "why can i trust jarvis",
        "why should i trust jarvis",
        "what makes jarvis reliable",
        "what makes jarvis trustworthy",
        "what needs attention",
        "what needs my attention",
        "what needs review",
        "which lanes need attention",
    }
    wrong_fragments = ["what should Jarvis do next", "memory stats", "storage recovery plan"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in direct_recent_run_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if (
                ("No tool runs logged yet." not in response and "Recent tool runs:" not in response)
                or "Did you mean" in response
            ):
                raise SystemExit(f"runtime should execute recent-run failure phrase {phrase!r}: {response!r}")
            if "Wikipedia" in response:
                raise SystemExit(f"recent-run failure phrase should not route to Wikipedia: {phrase!r} -> {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"recent-run failure phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "recent_tool_runs":
                raise SystemExit(f"recent-run phrase should only run recent_tool_runs: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"recent-run failure phrase should not require approval: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"recent-run failure phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase in direct_trace_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Jarvis runtime trace receipt:" not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute trace phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"trace phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "runtime_trace_receipt":
                raise SystemExit(f"trace phrase should only run runtime_trace_receipt: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"trace phrase should not require approval: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"trace phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase in direct_verification_receipt_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Jarvis verification receipt:" not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute verification-receipt phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"verification-receipt phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "verification_receipt":
                raise SystemExit(f"verification-receipt phrase should only run verification_receipt: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"verification-receipt phrase should not require approval: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"verification-receipt phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}"
                )
        for phrase, expected in direct_read_only_cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if expected not in response:
                raise SystemExit(f"runtime should execute read-only failure phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"read-only failure phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "execution_health_report":
                raise SystemExit(f"failure phrase should only run execution_health_report: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"read-only failure phrase should not require approval: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"failure phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase in direct_recovery_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Jarvis recovery closure checklist:" not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute recovery phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"recovery phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "recovery_closure_checklist":
                raise SystemExit(
                    f"recovery phrase should only run recovery_closure_checklist: {phrase!r} {tool_results!r}"
                )
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"recovery phrase should not require approval: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"recovery phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase in direct_cockpit_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Capability cockpit" not in response:
                raise SystemExit(f"runtime should execute cockpit attention phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"cockpit attention phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "capability_cockpit":
                raise SystemExit(f"cockpit attention phrase should only run capability_cockpit: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"cockpit attention phrase should not require approval: {phrase!r} {runtime_trace}")
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect {phrase!r} to {expected!r}: {response!r}")
            for wrong in wrong_fragments:
                if wrong in response and wrong != expected:
                    raise SystemExit(
                        f"runtime daily operator phrase suggested stale/wrong command {wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag {phrase!r} as a command suggestion: {chat_response!r}")


def test_runtime_preplanner_redirects_phone_control_intents_before_call_routing() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "phone control center": "cockpit",
        "phone control status": "cockpit",
        "phone commands": "capability map",
        "phone help": "capability map",
        "phone shortcuts": "capability map",
        "telegram control center": "cockpit",
        "telegram control status": "cockpit",
        "telegram commands": "capability map",
        "telegram help": "capability map",
        "telegram shortcuts": "capability map",
        "what can i do from my phone": "capability map",
        "what can i do on telegram": "capability map",
        "what can i do in telegram": "capability map",
        "what commands work on telegram": "capability map",
        "what can i text jarvis": "capability map",
        "what can i send jarvis on telegram": "capability map",
        "폰 컨트롤 센터": "cockpit",
        "폰 컨트롤 상태": "cockpit",
        "폰 명령어": "capability map",
        "폰 도움말": "capability map",
        "폰에서 뭐 할 수 있어": "capability map",
        "텔레그램 명령어": "capability map",
        "텔레그램 도움말": "capability map",
        "텔레그램에서 뭐 할 수 있어": "capability map",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect phone-control phrase {phrase!r} to {expected!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"phone-control phrase should not queue approvals: {phrase!r} {approvals_before}->{approvals_after}")
            if getattr(res, "tool_results", []):
                raise SystemExit(f"phone-control phrase should not execute tools before the user sends the suggested command: {phrase!r}")
            if any(fragment in response for fragment in ["call_contact", "call_telegram", "Safety receipt", "queued as approval"]):
                raise SystemExit(f"phone-control phrase leaked old call/approval route: {phrase!r} -> {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag phone-control phrase {phrase!r} as a command suggestion: {chat_response!r}")
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(f"runtime should mark phone-control redirect as pre-planner: {phrase!r} -> {chat_response!r}")
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(f"phone-control redirect should not alter approval queue: {phrase!r} -> {runtime_trace}")


def test_runtime_preplanner_redirects_usage_example_intents_before_call_or_chat_drift() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "examples": "capability map",
        "example commands": "capability map",
        "sample commands": "capability map",
        "starter commands": "capability map",
        "command examples": "capability map",
        "jarvis examples": "capability map",
        "show examples": "capability map",
        "show commands": "capability map",
        "show shortcuts": "capability map",
        "shortcuts": "capability map",
        "what can i try": "capability map",
        "what should i try": "capability map",
        "how do i use jarvis": "capability map",
        "how to use jarvis": "capability map",
        "jarvis usage": "capability map",
        "usage": "capability map",
        "teach me jarvis": "capability map",
        "beginner commands": "capability map",
        "daily commands": "capability map",
        "common commands": "capability map",
        "owner commands": "capability map",
        "message examples": "capability map",
        "messaging examples": "capability map",
        "send examples": "capability map",
        "sending examples": "capability map",
        "kakao examples": "capability map",
        "imessage examples": "capability map",
        "instagram examples": "capability map",
        "telegram message examples": "capability map",
        "call examples": "capability map",
        "calling examples": "capability map",
        "phone call examples": "capability map",
        "contact examples": "capability map",
        "contacts examples": "capability map",
        "contact lookup examples": "capability map",
        "address book examples": "capability map",
        "voice examples": "voice command cockpit",
        "voice commands": "voice command cockpit",
        "show messaging examples": "capability map",
        "show voice commands": "voice command cockpit",
        "show me what you can do": "capability map",
        "what can i send": "capability map",
        "what can i call": "capability map",
        "what can i ask": "capability map",
        "what can jarvis do": "capability map",
        "what can you do": "capability map",
        "what can you do with contacts": "capability map messages",
        "what can jarvis do with contacts": "capability map messages",
        "what can you do with contact lookup": "capability map messages",
        "what can jarvis do with address book": "capability map messages",
        "message capabilities": "capability map messages",
        "messaging capabilities": "capability map messages",
        "send capabilities": "capability map messages",
        "kakao capabilities": "capability map messages",
        "telegram capabilities": "capability map messages",
        "imessage capabilities": "capability map messages",
        "instagram capabilities": "capability map messages",
        "call capabilities": "capability map messages",
        "phone call capabilities": "capability map messages",
        "contact capabilities": "capability map messages",
        "contacts capabilities": "capability map messages",
        "contact lookup capabilities": "capability map messages",
        "address book capabilities": "capability map messages",
        "help with contacts": "capability map messages",
        "help with contact lookup": "capability map messages",
        "help with address book": "capability map messages",
        "contact commands": "capability map",
        "contacts commands": "capability map",
        "contact lookup commands": "capability map",
        "address book commands": "capability map",
        "voice capabilities": "capability map voice",
        "voice command capabilities": "capability map voice",
        "calendar capabilities": "capability map productivity",
        "email capabilities": "capability map productivity",
        "weather capabilities": "capability map info",
        "news capabilities": "capability map info",
        "market capabilities": "capability map markets",
        "markets capabilities": "capability map markets",
        "stock capabilities": "capability map markets",
        "crypto capabilities": "capability map markets",
        "translation capabilities": "capability map utilities",
        "utility capabilities": "capability map utilities",
        "research capabilities": "capability map research",
        "writing capabilities": "capability map writing",
        "task capabilities": "capability map tasks",
        "note capabilities": "capability map notes",
        "memory capabilities": "capability map memory",
        "approval capabilities": "capability map approvals",
        "safety capabilities": "capability map safety",
        "dashboard capabilities": "cockpit",
        "help with messages": "capability map messages",
        "help with voice": "capability map voice",
        "phone examples": "capability map",
        "telegram examples": "capability map",
        "예시": "capability map",
        "명령 예시": "capability map",
        "명령어 예시": "capability map",
        "샘플 명령어": "capability map",
        "자비스 사용법": "capability map",
        "사용법": "capability map",
        "도움말": "help",
        "자비스 도움말": "help",
        "뭐부터 해": "capability map",
        "무엇부터 해": "capability map",
        "시작 명령어": "capability map",
        "자주 쓰는 명령어": "capability map",
        "메시지 예시": "capability map",
        "메세지 예시": "capability map",
        "문자 예시": "capability map",
        "전송 예시": "capability map",
        "카카오 예시": "capability map",
        "카톡 예시": "capability map",
        "아이메시지 예시": "capability map",
        "인스타그램 예시": "capability map",
        "전화 예시": "capability map",
        "통화 예시": "capability map",
        "음성 예시": "voice command cockpit",
        "음성 명령어": "voice command cockpit",
        "뭘 보낼 수 있어": "capability map",
        "뭘 물어볼 수 있어": "capability map",
        "메시지 기능": "capability map messages",
        "메세지 기능": "capability map messages",
        "문자 기능": "capability map messages",
        "전송 기능": "capability map messages",
        "카카오 기능": "capability map messages",
        "카톡 기능": "capability map messages",
        "텔레그램 기능": "capability map messages",
        "아이메시지 기능": "capability map messages",
        "인스타그램 기능": "capability map messages",
        "전화 기능": "capability map messages",
        "통화 기능": "capability map messages",
        "음성 기능": "capability map voice",
        "캘린더 기능": "capability map productivity",
        "일정 기능": "capability map productivity",
        "이메일 기능": "capability map productivity",
        "할일 기능": "capability map tasks",
        "날씨 기능": "capability map info",
        "뉴스 기능": "capability map info",
        "시장 기능": "capability map markets",
        "주식 기능": "capability map markets",
        "코인 기능": "capability map markets",
        "번역 기능": "capability map utilities",
        "검색 기능": "capability map research",
        "글쓰기 기능": "capability map writing",
        "메모 기능": "capability map notes",
        "기억 기능": "capability map memory",
        "승인 기능": "capability map approvals",
        "안전 기능": "capability map safety",
        "대시보드 기능": "cockpit",
        "폰 예시": "capability map",
        "텔레그램 예시": "capability map",
    }
    wrong_fragments = [
        "call_contact",
        "call_telegram",
        "phone call",
        "queued as approval",
        "Safety receipt",
        "what should I do now",
        "what changed in Jarvis",
        "jarvis status",
        "dispatch decision",
        "send_imessage",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect usage/example phrase {phrase!r} to {expected!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"usage/example phrase should not queue approvals: {phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(f"usage/example phrase should not execute tools before the suggested command: {phrase!r}")
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(f"usage/example phrase leaked old route {wrong!r}: {phrase!r} -> {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag usage/example phrase {phrase!r} as a command suggestion: {chat_response!r}")
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(f"runtime should mark usage/example redirect as pre-planner: {phrase!r} -> {chat_response!r}")
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(f"usage/example redirect should not alter approval queue: {phrase!r} -> {runtime_trace}")


def test_runtime_preplanner_redirects_messaging_channel_status_before_dispatch_or_model_routing() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "can i resend": "channel health",
        "can i send again": "channel health",
        "send status": "channel health",
        "sending status": "channel health",
        "send again": "channel health",
        "should i resend": "channel health",
        "should i send again": "channel health",
        "safe to resend": "channel health",
        "safe to send again": "channel health",
        "delivery status": "channel health",
        "message delivery status": "channel health",
        "message failed": "channel health",
        "message error": "channel health",
        "message issue": "channel health",
        "message problem": "channel health",
        "messaging issue": "channel health",
        "send failed": "channel health",
        "send error": "channel health",
        "send problem": "channel health",
        "sending issue": "channel health",
        "call failed": "channel health",
        "call issue": "channel health",
        "call problem": "channel health",
        "phone call failed": "channel health",
        "phone call problem": "channel health",
        "telegram report": "channel health",
        "telegram failed": "channel health",
        "telegram error": "channel health",
        "telegram broken": "channel health",
        "telegram issue": "channel health",
        "telegram not working": "channel health",
        "telegram problem": "channel health",
        "telegram send failed": "channel health",
        "retry telegram": "channel health",
        "resend telegram": "channel health",
        "send telegram again": "channel health",
        "try telegram again": "channel health",
        "telegram last failure": "channel health",
        "last telegram failure": "channel health",
        "telegram last success": "channel health",
        "telegram success count": "channel health",
        "telegram 7 day success count": "channel health",
        "did telegram send": "channel health",
        "did telegram go through": "channel health",
        "did my telegram go through": "channel health",
        "was telegram sent": "channel health",
        "was telegram delivered": "channel health",
        "did telegram work recently": "channel health",
        "what broke on telegram": "channel health",
        "why did telegram break": "channel health",
        "what happened to telegram": "channel health",
        "what happened with telegram": "channel health",
        "why did telegram fail": "channel health",
        "why did telegram send fail": "channel health",
        "message delivery health": "channel health",
        "channel last failure": "channel health",
        "show channel failures": "channel health",
        "imessage failed": "channel health",
        "imessage error": "channel health",
        "imessage issue": "channel health",
        "imessage last failure": "channel health",
        "imessage success count": "channel health",
        "did imessage go through": "channel health",
        "was imessage sent": "channel health",
        "did imessage work recently": "channel health",
        "what broke on imessage": "channel health",
        "what happened to imessage": "channel health",
        "why did imessage fail": "channel health",
        "retry imessage": "channel health",
        "resend imessage": "channel health",
        "send imessage again": "channel health",
        "kakao failed": "channel health",
        "kakao error": "channel health",
        "kakao issue": "channel health",
        "kakao last failure": "channel health",
        "kakao success count": "channel health",
        "did kakao go through": "channel health",
        "was kakao delivered": "channel health",
        "did kakao work recently": "channel health",
        "what broke on kakao": "channel health",
        "kakao not working": "channel health",
        "kakao problem": "channel health",
        "what happened with kakao": "channel health",
        "why did kakao fail": "channel health",
        "retry kakao": "channel health",
        "resend kakao": "channel health",
        "send kakao again": "channel health",
        "instagram failed": "channel health",
        "instagram error": "channel health",
        "instagram issue": "channel health",
        "instagram last failure": "channel health",
        "instagram success count": "channel health",
        "did instagram go through": "channel health",
        "was instagram sent": "channel health",
        "did instagram work recently": "channel health",
        "what broke on instagram": "channel health",
        "what happened to instagram": "channel health",
        "why did instagram fail": "channel health",
        "retry instagram": "channel health",
        "resend instagram": "channel health",
        "send instagram again": "channel health",
        "why did call fail": "channel health",
        "why did phone call fail": "channel health",
        "why did send fail": "channel health",
        "did it send": "channel health",
        "did it go through": "channel health",
        "did my message send": "channel health",
        "did the message send": "channel health",
        "did my message go through": "channel health",
        "did the message go through": "channel health",
        "was it sent": "channel health",
        "was my message delivered": "channel health",
        "was the message delivered": "channel health",
        "메시지 실패": "channel health",
        "메시지 안 보내짐": "channel health",
        "메시지 못 보냄": "channel health",
        "전송 실패": "channel health",
        "다시 보내도 돼": "channel health",
        "다시 보내도 되나요": "channel health",
        "다시 보낼까": "channel health",
        "재전송 상태": "channel health",
        "채널 마지막 실패": "channel health",
        "채널 성공 횟수": "channel health",
        "채널 실패 보여줘": "channel health",
        "전송 안됨": "channel health",
        "전송 안돼": "channel health",
        "전송 안 보내짐": "channel health",
        "전송 못 보냄": "channel health",
        "아이메시지 마지막 실패": "channel health",
        "아이메시지 성공 횟수": "channel health",
        "아이메시지 실패": "channel health",
        "아이메시지 갔어": "channel health",
        "아이메시지 보냈어": "channel health",
        "아이메시지 보내졌어": "channel health",
        "아이메시지 다시 보낼까": "channel health",
        "아이메시지 재시도": "channel health",
        "아이메시지 재전송": "channel health",
        "카카오 마지막 실패": "channel health",
        "카카오 성공 횟수": "channel health",
        "카카오 실패": "channel health",
        "카카오 갔어": "channel health",
        "카카오 보냈어": "channel health",
        "카카오 보내졌어": "channel health",
        "카카오 다시 보낼까": "channel health",
        "카카오 재시도": "channel health",
        "카카오 재전송": "channel health",
        "카카오 문제": "channel health",
        "카카오 이슈": "channel health",
        "왜 카카오 실패": "channel health",
        "왜 전송 안됨": "channel health",
        "왜 전송 안 보내짐": "channel health",
        "왜 메시지 안 보내짐": "channel health",
        "카톡 마지막 실패": "channel health",
        "카톡 성공 횟수": "channel health",
        "카톡 오류": "channel health",
        "카톡 안됨": "channel health",
        "카톡 갔어": "channel health",
        "카톡 보냈어": "channel health",
        "카톡 보내졌어": "channel health",
        "카톡 다시 보낼까": "channel health",
        "카톡 재시도": "channel health",
        "카톡 재전송": "channel health",
        "카톡 문제": "channel health",
        "텔레그램 확인": "channel health",
        "텔레그램 마지막 실패": "channel health",
        "텔레그램 최근 성공": "channel health",
        "텔레그램 성공 횟수": "channel health",
        "텔레그램 최근에 됐어": "channel health",
        "텔레그램 갔어": "channel health",
        "텔레그램 보냈어": "channel health",
        "텔레그램 보내졌어": "channel health",
        "텔레그램 다시 보낼까": "channel health",
        "텔레그램 재시도": "channel health",
        "텔레그램 재전송": "channel health",
        "텔레그램 실패": "channel health",
        "텔레그램 오류": "channel health",
        "텔레그램 문제": "channel health",
        "텔레그램 이슈": "channel health",
        "텔레그램 안됨": "channel health",
        "텔레그램 고장": "channel health",
        "왜 텔레그램 실패": "channel health",
        "왜 텔레그램 안 보내짐": "channel health",
        "왜 텔레그램 안 보내졌어": "channel health",
        "인스타그램 마지막 실패": "channel health",
        "인스타그램 성공 횟수": "channel health",
        "인스타그램 실패": "channel health",
        "인스타그램 문제": "channel health",
        "메시지 갔어": "channel health",
        "메시지 갔나요": "channel health",
        "메시지 보냈어": "channel health",
        "메시지 보내졌어": "channel health",
        "전송 됐어": "channel health",
        "전송됐어": "channel health",
        "전송됐나요": "channel health",
        "통화 실패": "channel health",
        "통화 문제": "channel health",
        "전화 실패": "channel health",
        "전화 문제": "channel health",
        "왜 전화 실패": "channel health",
    }
    wrong_fragments = [
        "dispatch_decision_packet",
        "Safety receipt",
        "queued as approval",
        "storage status",
        "safety status",
        "integration status",
        "goal 1 status",
        "tool search",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect messaging/channel status phrase {phrase!r} "
                    f"to {expected!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"messaging/channel status phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(
                    "messaging/channel status phrase should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"messaging/channel status phrase leaked stale/wrong route "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag messaging/channel status phrase {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark messaging/channel status redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "messaging/channel status redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )


def test_runtime_preplanner_redirects_tool_control_plane_intents_before_tool_detail() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "tool status": "cockpit",
        "tools status": "cockpit",
        "tool health": "cockpit",
        "tools health": "cockpit",
        "tool report": "cockpit",
        "tools report": "cockpit",
        "available tool status": "cockpit",
        "available tools": "list tools",
        "show available tools": "list tools",
        "tool list": "list tools",
        "tools list": "list tools",
        "tool registry": "capability map",
        "tool registry status": "cockpit",
        "list capabilities": "capability map",
        "capability list": "capability map",
        "capabilities list": "capability map",
        "available capabilities": "capability map",
        "show capabilities": "capability map",
        "capabilities cockpit": "cockpit",
        "capability dashboard": "cockpit",
        "capabilities dashboard": "cockpit",
        "jarvis cockpit": "cockpit",
        "control plane": "cockpit",
        "control plane health": "cockpit",
        "control plane status": "cockpit",
        "control status": "cockpit",
        "open cockpit": "cockpit",
        "open control plane": "cockpit",
        "show me control plane": "cockpit",
        "show me dashboard status": "cockpit",
        "show me the control panel": "cockpit",
        "show me the control plane": "cockpit",
        "show control plane": "cockpit",
        "jarvis control plane": "cockpit",
        "control panel": "cockpit",
        "jarvis control panel": "cockpit",
        "dashboard status": "cockpit",
        "show dashboard status": "cockpit",
        "trust dashboard": "cockpit",
        "safety dashboard": "safety status",
        "risk dashboard": "risk matrix",
        "show risk levels": "risk matrix",
        "what is the risk level": "risk matrix",
        "what tools are available": "list tools",
        "what tools do you have": "list tools",
        "what can jarvis actually do": "capability map",
        "what is jarvis capable of": "capability map",
        "what can you actually do": "capability map",
        "what can u do": "capability map",
        "capability report": "cockpit",
        "capability cockpit status": "cockpit",
        "where is control plane": "cockpit",
        "where is the control plane": "cockpit",
        "도구 상태": "cockpit",
        "툴 상태": "cockpit",
        "도구 보고서": "cockpit",
        "도구 목록": "list tools",
        "도구 리스트": "list tools",
        "툴 목록": "list tools",
        "툴 리스트": "list tools",
        "사용 가능한 도구": "list tools",
        "가능한 도구": "list tools",
        "툴 레지스트리": "capability map",
        "기능 리스트": "capability map",
        "가능한 기능": "capability map",
        "컨트롤 플레인": "cockpit",
        "컨트롤패널": "cockpit",
        "자비스 컨트롤패널": "cockpit",
        "상태 대시보드": "cockpit",
        "능력 대시보드": "cockpit",
        "기능 대시보드": "cockpit",
        "기능 상태": "capability map",
        "능력 상태": "capability map",
        "안전 대시보드": "safety status",
        "리스크 매트릭스": "risk matrix",
        "위험 수준 보여줘": "risk matrix",
    }
    wrong_fragments = [
        "goal 1 status",
        "tool search",
        "tool_detail",
        "Tool `status`",
        "Tool `registry`",
        "execution health report",
        "todo list",
        "terminal dashboard",
        "jarvis doctor",
        "computer control status",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect tool/control-plane phrase {phrase!r} "
                    f"to {expected!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"tool/control-plane phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(
                    "tool/control-plane phrase should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"tool/control-plane phrase suggested stale/wrong command "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag tool/control-plane phrase {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark tool/control-plane redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "tool/control-plane redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )


def test_runtime_redirects_control_help_suggestions_before_fallback_chat() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "help control plane": "help control",
        "hlep control": "help control",
        "hlep cockpit": "help cockpit",
        "hlep status": "help status",
    }
    direct_help = {
        "help control plane": "Jarvis help: control",
    }
    wrong_fragments = [
        "harness control",
        "harness status",
        "I am here",
        "model_or_fallback_chat",
        "queued as approval",
        "Safety receipt",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            direct_expected = direct_help.get(phrase)
            is_direct_help = bool(direct_expected and direct_expected in response)
            if not is_direct_help and ("Did you mean" not in response or expected not in response):
                raise SystemExit(
                    f"runtime should redirect control-help phrase {phrase!r} "
                    f"to {expected!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"control-help phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            tool_results = getattr(res, "tool_results", []) or []
            if is_direct_help:
                if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "jarvis_help":
                    raise SystemExit(
                        "valid control-help alias should execute jarvis_help once: "
                        f"{phrase!r} -> {tool_results!r}"
                    )
            elif tool_results:
                raise SystemExit(
                    "control-help phrase should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"control-help phrase leaked stale/wrong route "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if not is_direct_help and chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag control-help phrase {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "control-help redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )


def test_runtime_preplanner_redirects_health_diagnostic_intents_before_status_drift() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    direct_system_cases = {
        "system status": ("system_info", "System:"),
    }
    cases = {
        "jarvis report": "jarvis status",
        "시스템 상태": "jarvis status",
    }
    direct_health_cases = {
        "health report": "execution health report",
        "failure status": "execution health report",
        "failed status": "execution health report",
        "failures": "execution health report",
        "show failures": "execution health report",
        "what just failed": "execution health report",
        "what is unhealthy": "execution health report",
    }
    wrong_fragments = [
        "storage status",
        "goal 1 status",
        "Wikipedia",
        "wiki_summary",
        "system_info",
        "Tool `status`",
        "show alarms",
        "memory stats",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in direct_health_cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if expected not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute health/diagnostic phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"health/diagnostic phrase should not queue approvals: {phrase!r}")
            if any(wrong in response for wrong in wrong_fragments):
                raise SystemExit(f"health/diagnostic phrase leaked stale/wrong route: {phrase!r} -> {response!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "execution_health_report":
                raise SystemExit(
                    f"health/diagnostic phrase should only run execution_health_report: {phrase!r} {tool_results!r}"
                )
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"health/diagnostic phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"health/diagnostic phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase, (expected_tool, expected) in direct_system_cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if expected not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute status phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"status phrase should not queue approvals: {phrase!r}")
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(f"status phrase leaked stale/wrong command {wrong!r}: {phrase!r} -> {response!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != expected_tool:
                raise SystemExit(f"status phrase should only run {expected_tool}: {phrase!r} {tool_results!r}")
            meta = getattr(res, "metadata", {}) or {}
            runtime_trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or runtime_trace.get("route") != "tools":
                raise SystemExit(f"status phrase should execute through the normal tool route: {phrase!r} {meta!r}")
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"status phrase should stay read-only: {phrase!r} {runtime_trace}")
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect health/diagnostic phrase {phrase!r} "
                    f"to {expected!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"health/diagnostic phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(
                    "health/diagnostic phrase should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"health/diagnostic phrase suggested stale/wrong command "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag health/diagnostic phrase {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark health/diagnostic redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "health/diagnostic redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )


def test_runtime_redirects_scheduler_automation_intents_to_read_only_job_status() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = [
        "brief status",
        "daily brief status",
        "schedule status",
        "scheduler status",
        "scheduler health",
        "scheduler report",
        "job status",
        "jobs status",
        "scheduled jobs",
        "scheduled brief status",
        "scheduled job status",
        "scheduled jobs status",
        "what jobs are running",
        "what is scheduled",
        "what is the schedule",
        "what is the morning schedule",
        "automation status",
        "automations status",
        "background job status",
        "recurring tasks",
        "recurring task status",
        "when is morning brief",
        "when is next morning brief",
        "next morning brief",
        "morning brief status",
        "morning brief health",
        "morning brief schedule",
        "morning brief next run",
        "is morning brief scheduled",
        "is the brief scheduled",
        "예약 작업 상태",
        "스케줄 상태",
        "자동화 상태",
        "백그라운드 작업",
        "반복 작업",
        "브리핑 상태",
        "브리핑 예약",
        "브리핑 언제야",
        "다음 브리핑 언제",
        "모닝브리프 상태",
        "모닝브리프 일정",
        "모닝브리프 예약 확인",
    ]
    wrong_fragments = [
        "Did you mean",
        "Good evening! Here's your brief",
        "Good morning! Here's your brief",
        "list_events",
        "schedule today",
        "what is on my schedule today",
        "integration status",
        "jarvis status",
        "morning startup",
        "daily_briefing",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != "list_scheduled_jobs":
                raise SystemExit(
                    f"runtime should execute read-only list_scheduled_jobs for "
                    f"scheduler/automation phrase {phrase!r}: {results!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"scheduler/automation phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"scheduler/automation phrase returned stale/wrong output "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            if "No scheduled jobs." not in response and "Scheduler boundary:" not in response:
                raise SystemExit(
                    f"scheduler/automation phrase should expose scheduled-job status safely: "
                    f"{phrase!r} -> {response[:240]!r}"
                )
            meta = getattr(res, "metadata", {}) or {}
            runtime_trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or runtime_trace.get("route") != "tools":
                raise SystemExit(
                    f"scheduler/automation phrase should execute through the normal tool route: "
                    f"{phrase!r} -> {meta!r}"
                )
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "scheduler/automation exact command should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )
            if runtime_trace.get("approval_required") is True:
                raise SystemExit(f"scheduler/automation phrase should not require approval: {phrase!r} -> {runtime_trace!r}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"scheduler/automation phrase {phrase!r} should mark the exact alias route: "
                    f"{planner_metadata!r}"
                )


def test_runtime_preplanner_redirects_brief_failure_intents_to_recent_runs() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "brief failed": "recent tool runs",
        "brief error": "recent tool runs",
        "brief issue": "recent tool runs",
        "brief problem": "recent tool runs",
        "brief missing": "recent tool runs",
        "brief didn't arrive": "recent tool runs",
        "brief not delivered": "recent tool runs",
        "brief ran today": "recent tool runs",
        "brief sent today": "recent tool runs",
        "brief last run": "recent tool runs",
        "morning brief failed": "recent tool runs",
        "morning brief issue": "recent tool runs",
        "morning brief problem": "recent tool runs",
        "morning brief missing": "recent tool runs",
        "morning brief didn't arrive": "recent tool runs",
        "morning brief did not send": "recent tool runs",
        "did morning brief send": "recent tool runs",
        "did the morning brief send": "recent tool runs",
        "did my morning brief run": "recent tool runs",
        "did brief run today": "recent tool runs",
        "was morning brief sent": "recent tool runs",
        "was morning brief delivered": "recent tool runs",
        "morning brief last run": "recent tool runs",
        "last morning brief": "recent tool runs",
        "did scheduler run": "recent tool runs",
        "scheduler ran today": "recent tool runs",
        "schedule ran today": "recent tool runs",
        "why did scheduler not run": "recent tool runs",
        "scheduler not running": "recent tool runs",
        "jobs not running": "recent tool runs",
        "where is my morning brief": "recent tool runs",
        "why did morning brief fail": "recent tool runs",
        "why didn't morning brief send": "recent tool runs",
        "why did morning brief not arrive": "recent tool runs",
        "브리핑 실패": "recent tool runs",
        "브리핑 문제": "recent tool runs",
        "브리핑 안 왔어": "recent tool runs",
        "브리핑 어디": "recent tool runs",
        "오늘 브리핑 보냈어": "recent tool runs",
        "브리핑 실행됐어": "recent tool runs",
        "모닝브리프 실패": "recent tool runs",
        "모닝브리프 문제": "recent tool runs",
        "모닝브리프 안 왔어": "recent tool runs",
        "모닝브리프 어디": "recent tool runs",
        "모닝브리프 보냈어": "recent tool runs",
        "아침브리프 실패": "recent tool runs",
        "아침브리프 문제": "recent tool runs",
        "아침브리프 안 왔어": "recent tool runs",
        "아침 브리핑 보냈어": "recent tool runs",
        "스케줄러 실행됐어": "recent tool runs",
        "예약 작업 안 돌아": "recent tool runs",
        "예약 작업 문제": "recent tool runs",
    }
    wrong_fragments = ["list scheduled jobs", "morning startup", "queued as approval", "Safety receipt"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect brief failure phrase {phrase!r} "
                    f"to {expected!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"brief failure phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(
                    "brief failure phrase should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"brief failure phrase suggested stale/wrong command "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag brief failure phrase {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark brief failure redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "brief failure redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )


def test_runtime_redirects_audit_history_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "execution logs": "recent tool runs",
        "what did jarvis do": "recent tool runs",
        "recent actions": "recent tool runs",
        "last actions": "recent tool runs",
        "last thing you did": "recent tool runs",
        "what ran recently": "recent tool runs",
        "recent runs": "recent tool runs",
        "감사 기록 보여줘": "recent tool runs",
        "실행 기록 보여줘": "recent tool runs",
        "최근에 뭐 했어": "recent tool runs",
        "latest receipt": "verification receipt latest",
    }
    wrong_fragments = [
        "execution case closure",
        "what should Jarvis do next",
        "what do I need to do",
        "That sounds like it may involve action",
        "fallback chat",
        "next actions",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect audit/history phrase {phrase!r} to {expected!r}: {response!r}")
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"runtime audit/history phrase suggested stale/wrong command {wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag audit/history phrase {phrase!r} as a command suggestion: {chat_response!r}")

        for phrase in ("what changed", "what changed recently", "recent activity"):
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Jarvis activity digest" not in response:
                raise SystemExit(f"{phrase!r} should run the read-only activity digest directly: {response!r}")
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or getattr(results[0], "tool_name", None) != "activity_digest":
                raise SystemExit(f"{phrase!r} should execute exactly activity_digest: {results!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"{phrase!r} should not queue approvals: {approvals_before}->{approvals_after}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") == "command_suggestion":
                raise SystemExit(f"{phrase!r} should not be suggestion-gated: {chat_response!r}")


def test_runtime_redirects_trust_boundary_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "safety check": ("safety status", "Jarvis safety status"),
        "privacy status": ("privacy report", "Jarvis privacy boundary report"),
        "risk status": ("risk matrix", "Jarvis risk matrix"),
        "approval safety": ("safety status", "safety status"),
        "what are your limits": ("safety status", "Jarvis safety status"),
        "what are jarvis limits": ("safety status", "Jarvis safety status"),
        "what are your boundaries": ("safety status", "Jarvis safety status"),
        "what are the safety boundaries": ("safety status", "Jarvis safety status"),
        "what are you allowed to do": ("safety status", "Jarvis safety status"),
        "what are you not allowed to do": ("safety status", "Jarvis safety status"),
        "what can you not do": ("safety status", "Jarvis safety status"),
        "what cant you do": ("safety status", "Jarvis safety status"),
        "what can jarvis not do": ("safety status", "Jarvis safety status"),
        "what requires approval": ("safety status", "Jarvis safety status"),
        "what needs approval": ("safety status", "Jarvis safety status"),
        "what actions need approval": ("safety status", "Jarvis safety status"),
        "what commands need approval": ("safety status", "Jarvis safety status"),
        "what tools require approval": ("safety status", "Jarvis safety status"),
        "which tools require approval": ("safety status", "Jarvis safety status"),
        "what is approval gated": ("safety status", "Jarvis safety status"),
        "what needs my permission": ("safety status", "Jarvis safety status"),
        "what requires my permission": ("safety status", "Jarvis safety status"),
        "redos status": ("frozen routing risk report", "Jarvis frozen routing risk report"),
        "regex safety status": ("frozen routing risk report", "Jarvis frozen routing risk report"),
        "send call routing risk report": ("frozen routing risk report", "Jarvis frozen routing risk report"),
        "what is the frozen redos risk": ("frozen routing risk report", "Jarvis frozen routing risk report"),
        "input length guard status": ("planner input guard report", "Jarvis planner input guard report"),
        "long input safety status": ("planner input guard report", "Jarvis planner input guard report"),
        "long message redos status": ("planner input guard report", "Jarvis planner input guard report"),
        "what is the input length guard decision": ("planner input guard report", "Jarvis planner input guard report"),
        "which commands are risky": ("risk matrix", "Jarvis risk matrix"),
        "which actions are risky": ("risk matrix", "Jarvis risk matrix"),
        "what permissions do you need": ("risk matrix", "Jarvis risk matrix"),
        "what permissions do you have": ("risk matrix", "Jarvis risk matrix"),
        "what permissions does Jarvis have": ("risk matrix", "Jarvis risk matrix"),
        "what permissions do I have": ("risk matrix", "Jarvis risk matrix"),
        "what are my permissions": ("risk matrix", "Jarvis risk matrix"),
        "show my permissions": ("risk matrix", "Jarvis risk matrix"),
        "what is high risk": ("risk matrix", "Jarvis risk matrix"),
        "which tools are high risk": ("risk matrix", "Jarvis risk matrix"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, (expected_command, expected_report) in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if expected_command not in response and expected_report not in response:
                raise SystemExit(
                    f"runtime should route trust-boundary phrase {phrase!r} "
                    f"to {expected_command!r} or its report: {response!r}"
                )
            for wrong in ["setup check", "jarvis status", "approval history"]:
                if wrong in response:
                    raise SystemExit(f"runtime trust-boundary phrase suggested stale/wrong command {wrong!r}: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response and chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should not use generic chat for trust-boundary phrase {phrase!r}: {chat_response!r}")


def test_runtime_preplanner_redirects_capability_questions_before_action_planning() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "can you send messages": "capability map messages",
        "can jarvis send messages": "capability map messages",
        "can you send telegram messages": "capability map messages",
        "can you send kakao messages": "capability map messages",
        "can you message people": "capability map messages",
        "can you text people": "capability map messages",
        "can you call people": "capability map messages",
        "can you call my contacts": "capability map messages",
        "can you check weather": "capability map info",
        "can you get weather": "capability map info",
        "can you tell me weather": "capability map info",
        "can you read news": "capability map info",
        "can you summarize news": "capability map info",
        "can you look up wikipedia": "capability map info",
        "can you define words": "capability map info",
        "can you tell me bitcoin price": "capability map markets",
        "can you check stock prices": "capability map markets",
        "can you translate": "capability map utilities",
        "can you convert currency": "capability map utilities",
        "can you calculate things": "capability map utilities",
        "can you set timers": "capability map productivity",
        "can you set reminders": "capability map productivity",
        "can you read my calendar": "capability map productivity",
        "can you check email": "capability map productivity",
        "can you give morning brief": "capability map productivity",
        "can you brief me every morning": "capability map productivity",
        "can you research the web": "capability map research",
        "can you browse the web": "capability map research",
        "can you search the internet": "capability map research",
        "can you write text": "capability map writing",
        "can you write emails": "capability map writing",
        "can you remember things": "capability map memory",
        "can you search memory": "capability map memory",
        "can you take notes": "capability map notes",
        "can you read files": "safety status",
        "can you run commands": "risk matrix",
        "can you use voice": "voice command cockpit",
        "can I talk to you": "voice command cockpit",
        "can you hear me": "voice command cockpit",
        "how do I use voice": "voice command cockpit",
    }
    direct_recent_run_cases = {
        "what is broken",
        "what's broken",
    }
    wrong_tool_fragments = [
        "read_emails",
        "daily_briefing",
        "get_weather",
        "get_news",
        "wiki_summary",
        "get_crypto_price",
        "get_stock_price",
        "translate",
        "send_telegram",
        "send_kakao",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in direct_recent_run_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if (
                ("No tool runs logged yet." not in response and "Recent tool runs:" not in response)
                or "Did you mean" in response
            ):
                raise SystemExit(f"runtime should execute broken-state capability phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"broken-state capability phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "recent_tool_runs":
                raise SystemExit(f"broken-state phrase should only run recent_tool_runs: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"broken-state phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"broken-state phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect capability question {phrase!r} "
                    f"to {expected!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"capability question should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(
                    "capability question should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            for wrong in wrong_tool_fragments:
                if wrong in response:
                    raise SystemExit(f"capability question leaked wrong action/tool {wrong!r}: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag capability question {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark capability question redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )


def test_runtime_preplanner_redirects_korean_capability_questions_before_chat_fallback() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "날씨 확인 가능해": "capability map info",
        "뉴스 읽을 수 있어": "capability map info",
        "뉴스 요약 가능해": "capability map info",
        "위키 찾아볼 수 있어": "capability map info",
        "주식 확인 가능해": "capability map markets",
        "비트코인 가격 볼 수 있어": "capability map markets",
        "번역 가능해": "capability map utilities",
        "환율 계산 가능해": "capability map utilities",
        "계산 가능해": "capability map utilities",
        "타이머 설정 가능해": "capability map productivity",
        "리마인더 가능해": "capability map productivity",
        "웹 검색 가능해": "capability map research",
        "글쓰기 가능해": "capability map writing",
        "기억할 수 있어": "capability map memory",
        "메모 가능해": "capability map notes",
        "파일 읽을 수 있어": "safety status",
        "명령 실행 가능해": "risk matrix",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect Korean capability question {phrase!r} "
                    f"to {expected!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"Korean capability question should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(
                    "Korean capability question should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag Korean capability question {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark Korean capability redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )


def test_runtime_redirects_next_step_operator_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "what next": "safe next actions",
        "next step": "Jarvis next action packet",
        "what's queued": "work queue",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if expected.lower() not in response.lower():
                raise SystemExit(f"runtime should route next-step phrase {phrase!r} to {expected!r}: {response!r}")
            if "chat context" in response:
                raise SystemExit(f"runtime should avoid vague chat/next-actions wording for {phrase!r}: {response!r}")
            if phrase == "next step" and (
                "Recommended move:" not in response
                or "Execution health:" not in response
                or "This packet is planning only" not in response
            ):
                raise SystemExit(f"runtime next-step packet lost its one-card planning boundary: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response and chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should not use generic chat for next-step phrase {phrase!r}: {chat_response!r}")


def test_runtime_redirects_korean_operator_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "뭐 할 수 있어": "capability map",
        "뭐 할 수 있는지 보여줘": "capability map",
        "재시도 상태": "recovery closure checklist",
        "재시도 준비": "recovery closure checklist",
        "재시도 해도 돼": "recovery closure checklist",
        "내가 봐야 할 거 있어": "readiness report",
        "대기 상태": "readiness report",
        "컨트롤 센터": "cockpit",
        "컨트롤 플레인 보여줘": "cockpit",
        "대시보드 보여줘": "cockpit",
        "신뢰 상태": "cockpit",
        "신뢰 보고서": "cockpit",
        "신뢰도 상태": "cockpit",
        "신뢰도 보고서": "cockpit",
        "신뢰할 수 있어": "cockpit",
        "왜 자비스 믿어도 돼": "cockpit",
        "왜 자비스를 믿어도 돼": "cockpit",
        "자비스 신뢰 가능": "cockpit",
        "자비스 신뢰 상태": "cockpit",
        "믿어도 돼": "cockpit",
        "믿을만해": "cockpit",
        "안전하게 써도 돼": "safety status",
        "자비스 안전하게 써도 돼": "safety status",
        "무엇을 못해": "safety status",
        "뭐 못해": "safety status",
        "무엇을 할 수 없어": "safety status",
        "뭘 할 수 없어": "safety status",
        "자비스 한계": "safety status",
        "자비스 경계": "safety status",
        "연락처 기능": "capability map messages",
        "연락처 검색 기능": "capability map messages",
        "연락처 조회 기능": "capability map messages",
        "연락처 명령어": "capability map",
        "연락처 검색 명령어": "capability map",
        "연락처 조회 명령어": "capability map",
        "연락처 예시": "capability map",
        "연락처 검색 예시": "capability map",
        "연락처 조회 예시": "capability map",
        "주소록 기능": "capability map messages",
        "주소록 검색 기능": "capability map messages",
        "주소록 조회 기능": "capability map messages",
        "주소록 명령어": "capability map",
        "주소록 검색 명령어": "capability map",
        "주소록 조회 명령어": "capability map",
        "주소록 예시": "capability map",
        "주소록 검색 예시": "capability map",
        "주소록 조회 예시": "capability map",
    }
    direct_cockpit_cases = {
        "주의 필요",
        "주의 상태",
        "검토 필요",
        "콕핏 주의",
        "콕핏 보여줘",
        "콕핏 상태",
        "대시보드 상태",
        "상태판 보여줘",
        "자비스 상태판",
        "신뢰 체크리스트",
        "자비스 믿어도 돼",
        "믿어도 되는 이유",
        "자비스 믿어도 되는 이유",
        "자비스 신뢰 체크리스트",
    }
    direct_recent_run_cases = {
        "감사 로그",
        "감사 상태",
        "고장났어?",
        "고장 상태",
        "도구 실행 기록",
        "마지막 에러",
        "마지막 오류",
        "마지막 실행",
        "마지막 도구 실행",
        "뭐가 고장났어",
        "뭐가 문제야?",
        "뭐가 안돼",
        "무슨 문제 있어",
        "무슨 문제야",
        "문제 뭐야",
        "문제 상태",
        "문제 있어?",
        "실행 기록",
        "오류 뭐야",
        "왜 실패했어",
        "최근 도구 실행",
        "최근 에러",
        "최근 실행 기록",
        "최근 오류",
    }
    direct_trace_cases = {
        "런타임 추적",
    }
    direct_verification_receipt_cases = {
        "실행 영수증",
    }
    direct_failure_cases = {
        "뭐가 실패했어",
        "실패 목록",
        "실패 보여줘",
        "실패 상태",
        "실패한 거 있어",
        "최근 실패",
        "마지막 실패",
    }
    direct_recovery_cases = {
        "복구 상태",
        "복구 계획",
        "뭐 복구해야 해",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in direct_cockpit_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Capability cockpit" not in response:
                raise SystemExit(f"runtime should execute Korean cockpit attention phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean cockpit attention phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "capability_cockpit":
                raise SystemExit(
                    f"Korean cockpit attention phrase should only run capability_cockpit: {phrase!r} {tool_results!r}"
                )
        for phrase in direct_recent_run_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if (
                ("No tool runs logged yet." not in response and "Recent tool runs:" not in response)
                or "Did you mean" in response
            ):
                raise SystemExit(f"runtime should execute Korean recent-run phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean recent-run phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "recent_tool_runs":
                raise SystemExit(f"Korean recent-run phrase should only run recent_tool_runs: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"Korean recent-run phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"Korean recent-run phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase in direct_trace_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Jarvis runtime trace receipt:" not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute Korean trace phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean trace phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "runtime_trace_receipt":
                raise SystemExit(f"Korean trace phrase should only run runtime_trace_receipt: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"Korean trace phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"Korean trace phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase in direct_verification_receipt_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Jarvis verification receipt:" not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute Korean verification-receipt phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean verification-receipt phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "verification_receipt":
                raise SystemExit(
                    f"Korean verification-receipt phrase should only run verification_receipt: {phrase!r} {tool_results!r}"
                )
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"Korean verification-receipt phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"Korean verification-receipt phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}"
                )
        for phrase in direct_failure_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Jarvis execution health report:" not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute Korean failure-health phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean failure-health phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "execution_health_report":
                raise SystemExit(
                    f"Korean failure-health phrase should only run execution_health_report: {phrase!r} {tool_results!r}"
                )
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"Korean failure-health phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"Korean failure-health phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase in direct_recovery_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Jarvis recovery closure checklist:" not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute Korean recovery phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean recovery phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "recovery_closure_checklist":
                raise SystemExit(
                    f"Korean recovery phrase should only run recovery_closure_checklist: {phrase!r} {tool_results!r}"
                )
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"Korean recovery phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"Korean recovery phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect Korean phrase {phrase!r} to {expected!r}: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag Korean phrase {phrase!r} as a command suggestion: {chat_response!r}")


def test_runtime_routes_korean_readonly_status_surfaces_to_existing_tools() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "저장소 상태": "storage_status",
        "저장소 통계": "storage_status",
        "저장소 확인": "storage_status",
        "스토리지 상태": "storage_status",
        "스토리지 통계": "storage_status",
        "DB 상태": "storage_status",
        "디비 상태": "storage_status",
        "데이터베이스 상태": "storage_status",
        "설치 상태": "setup_check",
        "셋업 상태": "setup_check",
        "환경 확인": "setup_check",
        "부트 상태": "setup_check",
        "콕핏 보여줘": "capability_cockpit",
        "콕핏 상태": "capability_cockpit",
        "대시보드 상태": "capability_cockpit",
        "상태판 보여줘": "capability_cockpit",
        "자비스 상태판": "capability_cockpit",
        "서브에이전트 몇 개": "subagent_fleet_status",
        "서브에이전트 개수": "subagent_fleet_status",
        "준비된 에이전트 몇 개": "subagent_fleet_status",
        "준비된 에이전트 개수": "subagent_fleet_status",
        "워커 몇 개": "subagent_fleet_status",
        "워커 개수": "subagent_fleet_status",
        "작업자 몇 개": "subagent_fleet_status",
        "작업자 개수": "subagent_fleet_status",
        "내부 오케스트레이션 확인": "subagent_fleet_status",
    }
    disallowed_true_flags = {
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
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected_tool in cases.items():
            tool = rt.registry.get(expected_tool)
            if tool.risk.name != "READ_ONLY":
                raise SystemExit(f"Korean status surface must route to read-only tool: {phrase!r} -> {tool.risk.name}")
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or getattr(results[0], "tool_name", "") != expected_tool:
                raise SystemExit(
                    f"Korean status surface should run {expected_tool!r} exactly once: "
                    f"{phrase!r} -> {results!r}"
                )
            if not res.verified:
                raise SystemExit(f"Korean status surface should verify through runtime: {phrase!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean status surface must not queue approvals: {phrase!r}")
            meta = getattr(res, "metadata", {}) or {}
            runtime_trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or runtime_trace.get("route") != "tools":
                raise SystemExit(f"Korean status surface should execute as a normal tool route: {phrase!r} {meta!r}")
            if (
                runtime_trace.get("approval_required") is True
                or runtime_trace.get("approval_queue_delta") != 0
                or runtime_trace.get("new_approval_ids")
            ):
                raise SystemExit(f"Korean status surface should stay read-only: {phrase!r} {runtime_trace!r}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"Korean status surface should mark exact-alias routing: {phrase!r} {planner_metadata!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            bad_flags = sorted(flag for flag in disallowed_true_flags if tool_meta.get(flag) is True)
            if bad_flags:
                raise SystemExit(f"Korean status surface exposed side-effect flags {bad_flags}: {phrase!r}")


def test_runtime_routes_channel_status_surfaces_to_channel_health() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "message status",
        "message check",
        "messages status",
        "messaging status",
        "messaging check",
        "message channel status",
        "check message",
        "check message status",
        "check messaging",
        "channel check",
        "check channel",
        "check channel status",
        "check channel health",
        "telegram status",
        "telegram check",
        "telegram health",
        "telegram channel status",
        "check telegram",
        "check telegram status",
        "check telegram health",
        "imessage status",
        "iMessage status",
        "imessage check",
        "imessage health",
        "imessage channel status",
        "check imessage",
        "check imessage status",
        "kakao status",
        "kakao check",
        "kakao health",
        "kakao channel status",
        "check kakao",
        "check kakao status",
        "instagram status",
        "instagram check",
        "instagram health",
        "instagram channel status",
        "check instagram",
        "check instagram status",
        "mobile status",
        "owner phone status",
        "phone status",
        "phone check",
        "phone health",
        "phone channel status",
        "phone call status",
        "check phone",
        "check phone status",
        "call status",
        "call check",
        "call health",
        "call channel status",
        "check call",
        "check call status",
        "facetime status",
        "FaceTime status",
        "facetime check",
        "facetime health",
        "facetime channel status",
        "check facetime",
        "check facetime status",
    }
    disallowed_true_flags = {
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
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        tool = rt.registry.get("channel_health")
        if tool.risk.name != "READ_ONLY":
            raise SystemExit(f"channel_health must stay read-only for channel status aliases: {tool.risk.name}")
        for phrase in cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or getattr(results[0], "tool_name", "") != "channel_health":
                raise SystemExit(
                    f"channel status phrase should run channel_health exactly once: {phrase!r} -> {results!r}"
                )
            if not res.verified:
                raise SystemExit(f"channel status phrase should verify through runtime: {phrase!r}")
            response = getattr(res, "response", "")
            if "Safety receipt:" in response or "queued as approval" in response or "Did you mean" in response:
                raise SystemExit(f"channel status phrase should not drift to approval/suggestion: {phrase!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"channel status phrase must not queue approvals: {phrase!r}")
            meta = getattr(res, "metadata", {}) or {}
            runtime_trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or runtime_trace.get("route") != "tools":
                raise SystemExit(f"channel status phrase should execute as a normal tool route: {phrase!r} {meta!r}")
            if (
                runtime_trace.get("approval_required") is True
                or runtime_trace.get("approval_queue_delta") != 0
                or runtime_trace.get("new_approval_ids")
            ):
                raise SystemExit(f"channel status phrase should stay read-only: {phrase!r} {runtime_trace!r}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"channel status phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            bad_flags = sorted(flag for flag in disallowed_true_flags if tool_meta.get(flag) is True)
            if bad_flags:
                raise SystemExit(f"channel status phrase exposed side-effect flags {bad_flags}: {phrase!r}")


def test_runtime_routes_korean_channel_status_surfaces_to_channel_health() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "채널 상태",
        "채널 확인",
        "메시지 상태",
        "메세지 상태",
        "메시지 채널 상태",
        "문자 상태",
        "문자 채널 상태",
        "전송 상태",
        "전송 채널 상태",
        "발송 채널 상태",
        "텔레그램 상태",
        "텔레그램 채널 상태",
        "카카오 상태",
        "카카오톡 상태",
        "카카오 채널 상태",
        "카톡 상태",
        "인스타 상태",
        "인스타그램 상태",
        "아이메시지 상태",
        "iMessage 상태",
        "imessage 상태",
        "통화 상태",
        "통화 채널 상태",
        "전화 상태",
        "전화 채널 상태",
        "콜 상태",
        "페이스타임 상태",
        "facetime 상태",
        "폰 상태",
        "모바일 상태",
    }
    disallowed_true_flags = {
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
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        tool = rt.registry.get("channel_health")
        if tool.risk.name != "READ_ONLY":
            raise SystemExit(f"channel_health must stay read-only for Korean channel status aliases: {tool.risk.name}")
        for phrase in cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or getattr(results[0], "tool_name", "") != "channel_health":
                raise SystemExit(
                    f"Korean channel status phrase should run channel_health exactly once: "
                    f"{phrase!r} -> {results!r}"
                )
            if not res.verified:
                raise SystemExit(f"Korean channel status phrase should verify through runtime: {phrase!r}")
            response = getattr(res, "response", "")
            if "Safety receipt:" in response or "queued as approval" in response or "Did you mean" in response:
                raise SystemExit(f"Korean channel status phrase should not drift to approval/suggestion: {phrase!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean channel status phrase must not queue approvals: {phrase!r}")
            meta = getattr(res, "metadata", {}) or {}
            runtime_trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or runtime_trace.get("route") != "tools":
                raise SystemExit(f"Korean channel status phrase should execute as a normal tool route: {phrase!r} {meta!r}")
            if (
                runtime_trace.get("approval_required") is True
                or runtime_trace.get("approval_queue_delta") != 0
                or runtime_trace.get("new_approval_ids")
            ):
                raise SystemExit(f"Korean channel status phrase should stay read-only: {phrase!r} {runtime_trace!r}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"Korean channel status phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}"
                )
            tool_meta = getattr(results[0], "metadata", {}) or {}
            bad_flags = sorted(flag for flag in disallowed_true_flags if tool_meta.get(flag) is True)
            if bad_flags:
                raise SystemExit(f"Korean channel status phrase exposed side-effect flags {bad_flags}: {phrase!r}")


def test_runtime_preplanner_redirects_korean_completion_and_agi_before_fallback_chat() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "AGI 상태": "agi gates",
        "AGI 준비 상태": "agi gates",
        "에이지아이 준비 상태": "agi gates",
        "하네스 상태": "harness status",
        "하네스 준비 상태": "harness readiness digest",
        "자비스 끝났어": "completion claim gate",
        "자비스 완료됐어": "completion claim gate",
        "자비스 완료상태": "completion claim gate",
        "완료 상태": "completion audit",
        "완료 주장": "completion claim gate",
        "완료 증명": "completion next proof",
    }
    wrong_fragments = ["list jarvis notes", "action readiness", "fallback"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect Korean completion/AGI phrase {phrase!r} to {expected!r}: {response!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"runtime Korean completion/AGI phrase suggested stale/wrong command "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag Korean completion/AGI phrase {phrase!r} as a command suggestion: "
                    f"{chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark Korean completion/AGI redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    f"Korean completion/AGI redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )
            if runtime_trace.get("planned_actions") or runtime_trace.get("tool_results"):
                raise SystemExit(
                    f"Korean completion/AGI redirect should not execute tools: {phrase!r} -> {runtime_trace}"
                )
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("authorizes_completion_claim") is not False:
                raise SystemExit(
                    f"Korean completion/AGI redirect must not authorize completion claims: "
                    f"{phrase!r} -> {planner_metadata}"
                )


def test_runtime_redirects_progress_roadmap_and_goal_status_before_fallback_chat() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "AGI progress": "build progress",
        "how is Jarvis going": "build progress",
        "roadmap status": "roadmap",
        "goal status": "list goals",
        "AGI 진행상황": "build progress",
        "자비스 진행상황": "build progress",
        "로드맵 보여줘": "roadmap",
        "다음에 뭐 만들어": "roadmap",
        "목표 상태": "list goals",
        "활성 목표": "list goals",
    }
    wrong_fragments = ["completion claim", "approval summary", "fallback"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect progress/roadmap/goal phrase {phrase!r} to {expected!r}: "
                    f"{response!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"runtime progress/roadmap/goal phrase suggested stale/wrong command "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag progress/roadmap/goal phrase {phrase!r} as a command suggestion: "
                    f"{chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    f"progress/roadmap/goal redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )
            if runtime_trace.get("planned_actions") or runtime_trace.get("tool_results"):
                raise SystemExit(
                    f"progress/roadmap/goal redirect should not execute tools: {phrase!r} -> {runtime_trace}"
                )

        direct_goal = rt.handle("active goals")
        direct_response = getattr(direct_goal, "response", "")
        direct_trace = (getattr(direct_goal, "metadata", {}) or {}).get("runtime_trace") or {}
        direct_actions = direct_trace.get("planned_actions") or []
        if "Did you mean" in direct_response or [a.get("tool_name") for a in direct_actions] != ["list_goals"]:
            raise SystemExit(
                "runtime should execute exact read-only goal aliases directly, not suggestion-gate them: "
                f"{direct_response!r} / {direct_actions!r}"
            )
        if direct_trace.get("approval_queue_delta") != 0 or direct_trace.get("new_approval_ids"):
            raise SystemExit(f"direct read-only goal alias should not alter approval queue: {direct_trace}")


def test_runtime_redirects_korean_model_brain_intents_to_model_routing_status() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = [
        "모델 상태",
        "브레인 상태",
        "어떤 모델 써",
        "올라마 상태",
        "플래너 상태",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in cases:
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or "model routing status" not in response:
                raise SystemExit(f"runtime should redirect Korean model/brain phrase {phrase!r} to model routing status: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag Korean model/brain phrase {phrase!r} as a command suggestion: {chat_response!r}")


def test_runtime_redirects_korean_eval_verification_intents_to_read_only_proof_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "평가 상태": "cockpit",
        "제임스 평가": "cockpit",
        "제임스 워크플로우 평가": "cockpit",
        "실제 워크플로우 테스트": "cockpit",
        "테스트 매트릭스 상태": "cockpit",
        "테스트 매트릭스 보여줘": "cockpit",
        "뭐 테스트해야 해": "cockpit",
        "라이브 테스트 뭐 해야 해": "cockpit",
        "검증 대기 뭐야": "cockpit",
        "라이브 매트릭스 상태": "cockpit",
        "라이브 테스트 결과": "cockpit",
        "라이브 테스트 결과 보고": "cockpit",
        "라이브 결과 보고 방법": "cockpit",
        "라이브 결과 형식": "cockpit",
        "라이브 테스트 결과 형식": "cockpit",
        "테스트 결과 어떻게 보고해": "cockpit",
        "테스트 결과 어떻게 보내": "cockpit",
        "테스트 결과 형식": "cockpit",
        "채널 증명 상태": "cockpit",
        "한국어 메시지 검증": "cockpit",
        "한국어 메시지 증명": "cockpit",
        "한국어 메시지 테스트됐어": "cockpit",
        "한국어 텔레그램 안전해": "cockpit",
        "한글 전송 증명": "cockpit",
        "가상연락처이 전송 테스트": "cockpit",
        "뭐가 검증됐어": "cockpit",
        "무엇이 검증됐어": "cockpit",
        "무엇을 검증했어": "cockpit",
        "검증 상태": "verification receipt latest",
        "검증 확인": "verification receipt latest",
        "증거": "evidence ledger",
        "증거 있어": "evidence ledger",
        "증거 뭐 있어": "evidence ledger",
        "증거 보여줘": "evidence ledger",
        "증명 보여줘": "evidence ledger",
        "근거 보여줘": "evidence ledger",
        "근거 있어": "evidence ledger",
        "검증 증거": "verification receipt latest",
        "최근 검증": "verification receipt latest",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect Korean eval/proof phrase {phrase!r} to {expected!r}: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag Korean eval/proof phrase {phrase!r} as a command suggestion: {chat_response!r}")


def test_runtime_preplanner_redirects_korean_evidence_phrasing_before_fallback_chat() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "증거": "evidence ledger",
        "증거 있어": "evidence ledger",
        "증거 뭐 있어": "evidence ledger",
        "증거 보여줘": "evidence ledger",
        "증명 보여줘": "evidence ledger",
        "근거 있어": "evidence ledger",
        "근거 뭐 있어": "evidence ledger",
        "근거 보여줘": "evidence ledger",
        "완료 증거": "evidence ledger",
        "신뢰 증거": "evidence ledger",
        "검증 증거": "verification receipt latest",
        "뭐가 검증됐어": "cockpit",
        "무엇이 검증됐어": "cockpit",
        "무엇을 검증했어": "cockpit",
        "최근 검증": "verification receipt latest",
    }
    wrong_fragments = ["fallback", "chat context", "verification packet", "goal 1 status"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect Korean evidence phrase {phrase!r} to {expected!r}: {response!r}")
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(f"runtime Korean evidence phrase suggested stale/wrong command {wrong!r}: {phrase!r} -> {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag Korean evidence phrase {phrase!r} as a command suggestion: {chat_response!r}")
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(f"runtime should mark Korean evidence redirect as pre-planner: {phrase!r} -> {chat_response!r}")
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(f"Korean evidence redirect should not alter approval queue: {phrase!r} -> {runtime_trace}")
            if runtime_trace.get("planned_actions") or runtime_trace.get("tool_results"):
                raise SystemExit(f"Korean evidence redirect should not execute tools: {phrase!r} -> {runtime_trace}")


def test_runtime_redirects_korean_memory_learning_intents_to_read_only_memory_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "메모리 상태": ("memory_stats", "No memories yet."),
        "기억 상태": ("memory_stats", "No memories yet."),
        "뭘 기억해": ("recent_memories", "No memories yet."),
        "학습 상태": ("learning_review", "Jarvis learning review:"),
        "학습 확인": ("learning_review", "Jarvis learning review:"),
        "학습 루프 상태": ("learning_review", "Jarvis learning review:"),
        "학습 부채 상태": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "학습 부채 닫혔어": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "학습 증명 매트릭스": ("execution_learning_closure_packet", "Jarvis execution learning closure packet:"),
        "복구 학습 상태": ("recovery_closure_checklist", "Jarvis recovery closure checklist:"),
        "복구 부채 상태": ("recovery_closure_checklist", "Jarvis recovery closure checklist:"),
        "반복 실패 상태": ("repeated_failure_clusters", "Jarvis repeated-failure cluster report:"),
        "실패 학습 상태": ("failure_learning_cockpit", "Jarvis failure learning cockpit:"),
        "무엇을 배웠어": ("learning_review", "Jarvis learning review:"),
        "기억 리뷰": ("learning_review", "Jarvis learning review:"),
        "기억 정리": ("list_weak_memories", "No weak-looking memories found."),
        "중복 기억": ("list_duplicate_memories", "No duplicate-looking memories found."),
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        broad_profile = rt.handle("나에 대해 뭘 알아")
        broad_response = getattr(broad_profile, "response", "")
        broad_results = getattr(broad_profile, "tool_results", []) or []
        if len(broad_results) != 1 or broad_results[0].tool_name != "chat_context":
            raise SystemExit(
                f"broad Korean profile-memory phrase should directly read bounded context: {broad_results!r}"
            )
        if "Did you mean" in broad_response or "Local personal state" not in broad_response:
            raise SystemExit(f"broad Korean profile-memory phrase should not require an English resend: {broad_response!r}")
        broad_meta = getattr(broad_profile, "metadata", {}) or {}
        broad_trace = broad_meta.get("runtime_trace") or {}
        if broad_meta.get("runtime_route") != "tools" or broad_trace.get("route") != "tools":
            raise SystemExit(
                f"broad Korean profile-memory phrase should use the normal tool route: {broad_meta!r}"
            )
        if broad_trace.get("approval_required") is True or broad_trace.get("new_approval_ids"):
            raise SystemExit(f"broad Korean profile-memory phrase must remain read-only: {broad_trace!r}")

        for phrase, (expected_tool, expected_fragment) in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            results = getattr(res, "tool_results", []) or []
            if len(results) != 1 or results[0].tool_name != expected_tool:
                raise SystemExit(
                    f"runtime should execute {expected_tool!r} for Korean memory/learning phrase "
                    f"{phrase!r}: {results!r}"
                )
            if "Did you mean" in response or expected_fragment not in response:
                raise SystemExit(
                    f"runtime Korean memory/learning phrase {phrase!r} should expose {expected_tool!r} output: "
                    f"{response[:240]!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean memory/learning phrase {phrase!r} must remain read-only without approvals")
            meta = getattr(res, "metadata", {}) or {}
            trace = meta.get("runtime_trace") or {}
            if meta.get("runtime_route") != "tools" or trace.get("route") != "tools":
                raise SystemExit(
                    f"Korean memory/learning phrase {phrase!r} should execute through normal tool route: {meta!r}"
                )
            if trace.get("approval_required") is True or trace.get("new_approval_ids"):
                raise SystemExit(f"Korean memory/learning phrase {phrase!r} should not require approval: {trace!r}")
            planner_metadata = trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(
                    f"Korean memory/learning phrase {phrase!r} should mark exact alias route: {planner_metadata!r}"
                )


def test_runtime_redirects_korean_audit_history_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "최근 실행": "recent tool runs",
        "무슨 일 했어": "recent tool runs",
        "영수증": "verification receipt latest",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect Korean audit/history phrase {phrase!r} to {expected!r}: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag Korean audit/history phrase {phrase!r} as a command suggestion: {chat_response!r}")


def test_runtime_redirects_korean_trust_boundary_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "안전 확인": ("safety status", "Jarvis safety status"),
        "개인정보 확인": ("privacy report", "Jarvis privacy boundary report"),
        "위험 상태": ("risk matrix", "Jarvis risk matrix"),
        "무엇을 못해": ("safety status", "Jarvis safety status"),
        "뭐 못해": ("safety status", "Jarvis safety status"),
        "무엇을 할 수 없어": ("safety status", "Jarvis safety status"),
        "뭘 할 수 없어": ("safety status", "Jarvis safety status"),
        "자비스 한계": ("safety status", "Jarvis safety status"),
        "자비스 경계": ("safety status", "Jarvis safety status"),
        "승인 필요한 것": ("safety status", "Jarvis safety status"),
        "승인이 필요한 것": ("safety status", "Jarvis safety status"),
        "뭐 승인 필요": ("safety status", "Jarvis safety status"),
        "어떤 명령이 승인 필요": ("safety status", "Jarvis safety status"),
        "어떤 도구가 승인 필요": ("safety status", "Jarvis safety status"),
        "권한 필요한 것": ("safety status", "Jarvis safety status"),
        "레도스 상태": ("frozen routing risk report", "Jarvis frozen routing risk report"),
        "정규식 위험 상태": ("frozen routing risk report", "Jarvis frozen routing risk report"),
        "동결 라우팅 위험": ("frozen routing risk report", "Jarvis frozen routing risk report"),
        "전송 라우팅 레도스": ("frozen routing risk report", "Jarvis frozen routing risk report"),
        "입력 길이 제한 상태": ("planner input guard report", "Jarvis planner input guard report"),
        "긴 입력 안전 상태": ("planner input guard report", "Jarvis planner input guard report"),
        "긴 명령 안전 상태": ("planner input guard report", "Jarvis planner input guard report"),
        "플래너 입력 제한": ("planner input guard report", "Jarvis planner input guard report"),
        "긴 메시지 레도스 위험": ("planner input guard report", "Jarvis planner input guard report"),
        "위험한 명령": ("risk matrix", "Jarvis risk matrix"),
        "고위험 도구": ("risk matrix", "Jarvis risk matrix"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, (expected_command, expected_report) in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if expected_command not in response and expected_report not in response:
                raise SystemExit(
                    f"runtime should route Korean trust-boundary phrase {phrase!r} "
                    f"to {expected_command!r} or its report: {response!r}"
                )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response and chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should not use generic chat for Korean trust-boundary phrase {phrase!r}: {chat_response!r}")


def test_runtime_redirects_korean_approval_gate_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    direct_summary_cases = {
        "승인 상태",
        "승인 대시보드",
    }
    direct_pending_cases = {
        "승인 대기",
        "승인 대기 뭐 있어",
        "승인 목록",
        "승인 큐",
        "승인 확인",
        "승인 뭐 남았어",
        "승인 필요한 거 있어",
        "승인할 거 있어",
        "내가 승인해야 할 거 있어",
        "뭐 기다려",
        "나 기다리는 거 있어",
        "검토할 거 있어",
        "리뷰 큐",
        "승인 필요한 작업",
        "대기 중인 승인",
    }
    cases = {
        "승인 내역": "approval history",
        "승인 증거": "approval evidence for latest",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase in direct_summary_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Approval queue summary:" not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute Korean approval summary phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean approval summary phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "approval_queue_summary":
                raise SystemExit(f"Korean approval summary phrase should only run approval_queue_summary: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"Korean approval summary phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"Korean approval summary phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase in direct_pending_cases:
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "No pending approvals." not in response or "Did you mean" in response:
                raise SystemExit(f"runtime should execute Korean pending-approval phrase {phrase!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"Korean pending-approval phrase should not queue approvals: {phrase!r}")
            tool_results = getattr(res, "tool_results", [])
            if len(tool_results) != 1 or getattr(tool_results[0], "tool_name", "") != "list_pending_approvals":
                raise SystemExit(f"Korean pending-approval phrase should only run list_pending_approvals: {phrase!r} {tool_results!r}")
            runtime_trace = (getattr(res, "metadata", {}) or {}).get("runtime_trace") or {}
            if runtime_trace.get("approval_required") is True or runtime_trace.get("approval_queue_delta") != 0:
                raise SystemExit(f"Korean pending-approval phrase should stay read-only: {phrase!r} {runtime_trace}")
            planner_metadata = runtime_trace.get("planner_metadata") or {}
            if planner_metadata.get("runtime_exact_tool_alias") is not True:
                raise SystemExit(f"Korean pending-approval phrase should mark exact-alias routing: {phrase!r} {planner_metadata!r}")
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect Korean approval-gate phrase {phrase!r} to {expected!r}: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag Korean approval-gate phrase {phrase!r} as a command suggestion: {chat_response!r}")


def test_runtime_redirects_korean_next_step_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "미션 컨트롤": "mission control",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect Korean next-step phrase {phrase!r} to {expected!r}: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag Korean next-step phrase {phrase!r} as a command suggestion: {chat_response!r}")


def test_runtime_redirects_korean_voice_intents_to_existing_commands() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "음성": "voice command cockpit",
        "마이크 개인정보": "microphone privacy please",
        "음성 중지": "voice stop intent: stop listening",
        "그만 말해": "voice stop intent: stop listening",
    }
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            res = rt.handle(phrase)
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect Korean voice phrase {phrase!r} to {expected!r}: {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag Korean voice phrase {phrase!r} as a command suggestion: {chat_response!r}")


def test_runtime_preplanner_redirects_voice_control_and_readiness_before_chat_or_voice_execution() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "stop talking": "voice stop intent: stop listening",
        "mute jarvis": "voice stop intent: stop listening",
        "stop voice": "voice stop intent: stop listening",
        "cancel speech": "voice stop intent: stop listening",
        "microphone privacy": "microphone privacy please",
    }
    wrong_fragments = ["stop Jarvis", "list voices", "Safety receipt", "queued as approval"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(f"runtime should redirect voice phrase {phrase!r} to {expected!r}: {response!r}")
            if approvals_after != approvals_before:
                raise SystemExit(f"voice phrase should not queue approvals: {phrase!r} {approvals_before}->{approvals_after}")
            if getattr(res, "tool_results", []):
                raise SystemExit(f"voice phrase should not execute tools before the suggested command: {phrase!r}")
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(f"voice phrase suggested stale/wrong route {wrong!r}: {phrase!r} -> {response!r}")
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(f"runtime should tag voice phrase {phrase!r} as a command suggestion: {chat_response!r}")
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(f"runtime should mark voice redirect as pre-planner: {phrase!r} -> {chat_response!r}")
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(f"voice redirect should not alter approval queue: {phrase!r} -> {runtime_trace}")


def test_runtime_preplanner_redirects_voice_failure_intents_to_voice_setup_check() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "mic not working": "voice setup check",
        "my microphone is not working": "voice setup check",
        "voice broken": "voice setup check",
        "voice input failed": "voice setup check",
        "whisper failed": "voice setup check",
        "transcription not working": "voice setup check",
        "why did voice fail": "voice setup check",
        "마이크 안돼": "voice setup check",
        "음성 오류": "voice setup check",
        "목소리 문제": "voice setup check",
        "녹음 안됨": "voice setup check",
        "위스퍼 실패": "voice setup check",
        "전사 오류": "voice setup check",
    }
    wrong_fragments = ["channel health", "pending approvals", "queued as approval", "Safety receipt"]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect voice failure phrase {phrase!r} "
                    f"to {expected!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"voice failure phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(
                    "voice failure phrase should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"voice failure phrase suggested stale/wrong command "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag voice failure phrase {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark voice failure redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "voice failure redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )


def test_handoff_and_work_queue_intents_redirect_to_read_only_commands() -> None:
    cases = {
        "what did claude leave": "handoff brief",
        "what did claude leave for codex": "handoff brief",
        "what did claude hand off": "handoff brief",
        "what claude said": "handoff brief",
        "check what claude left": "handoff brief",
        "check what claude has left": "handoff brief",
        "what has claude left": "handoff brief",
        "what is the handoff": "handoff brief",
        "handoff status": "handoff brief",
        "handoff notice": "handoff brief",
        "show handoff notice": "handoff brief",
        "codex tasks": "work queue",
        "check codex tasks": "work queue",
        "check codex.md tasks": "work queue",
        "current instructions": "work queue",
        "what are current instructions": "work queue",
        "what does CODEX_TASKS say": "work queue",
        "what should codex work on": "work queue",
        "what should codex do next": "work queue",
        "what is the current work queue": "work queue",
        "agent landscape research": "work queue",
        "agent research note": "work queue",
        "AI agent landscape": "work queue",
        "every AI agent research": "work queue",
        "how do we unlock the moat": "work queue",
        "Jarvis three month plan": "work queue",
        "other Jarvis models": "work queue",
        "other Jarvis models research": "work queue",
        "three month plan": "work queue",
        "three months of work": "work queue",
        "what did the agent research say": "work queue",
        "what did you find about AI agents": "work queue",
        "what did you find about other Jarvis models": "work queue",
        "what about other Jarvis models": "work queue",
        "what did you find about Zoey": "work queue",
        "what did you learn from Zoey": "work queue",
        "Zoey research": "work queue",
        "Zoey OS research": "work queue",
        "OpenClaw research": "work queue",
        "Hermes research": "work queue",
        "Lindy research": "work queue",
        "Manus research": "work queue",
        "Devin research": "work queue",
        "what should Jarvis borrow from OpenClaw": "work queue",
        "what should Jarvis borrow from Lindy": "work queue",
        "what should Jarvis copy from OpenClaw": "work queue",
        "what should Jarvis copy from Lindy": "work queue",
        "what should Jarvis borrow from Zoey": "work queue",
        "what should Jarvis avoid from Zoey": "work queue",
        "what should Jarvis avoid from AI agents": "work queue",
        "what should Jarvis do after the research": "work queue",
        "what is the Jarvis strategy": "work queue",
        "Jarvis strategy": "work queue",
        "Jarvis product strategy": "work queue",
        "Jarvis direction": "work queue",
        "what is the strategy after the research": "work queue",
        "what is the right move for Jarvis": "work queue",
        "what is the three month plan": "work queue",
        "what is the three month plan for Jarvis": "work queue",
        "what are the three months of work": "work queue",
        "what should Jarvis build before integrations": "work queue",
        "what should Jarvis focus on for the next three months": "work queue",
        "what should we do after Zoey research": "work queue",
        "what should Jarvis build after Zoey research": "work queue",
        "should Jarvis use companions": "work queue",
        "should Jarvis have companions": "work queue",
        "should Jarvis add companion personas": "work queue",
        "should Jarvis copy Zoey": "work queue",
        "should Jarvis copy Lindy": "work queue",
        "should Jarvis copy OpenClaw": "work queue",
        "what was the AI agent research recommendation": "work queue",
        "what did research say to avoid": "work queue",
        "what is Jarvis moat": "work queue",
        "what is the Jarvis moat": "work queue",
        "Jarvis moat": "work queue",
        "what unlocks Jarvis moat": "work queue",
        "what unlocks the moat": "work queue",
        "when should Jarvis add integrations": "work queue",
        "should Jarvis add integrations now": "work queue",
        "why no companions": "work queue",
        "why no companion personas": "work queue",
        "why not companion personas": "work queue",
        "what did Claude say about Zoey": "work queue",
        "what did Claude recommend about companions": "work queue",
        "Claude가 뭐 남겼어": "handoff brief",
        "클로드 인수인계": "handoff brief",
        "코덱스 작업 뭐야": "work queue",
        "현재 지시사항 보여줘": "work queue",
        "작업 큐 보여줘": "work queue",
        "조이 조사 결과": "work queue",
        "Zoey 조사 결과": "work queue",
        "에이전트 조사 결과": "work queue",
        "모든 AI 에이전트 조사": "work queue",
        "다른 자비스 모델": "work queue",
        "자비스 모델 조사": "work queue",
        "자비스 차별점": "work queue",
        "자비스 해자": "work queue",
        "자비스 3개월 계획": "work queue",
        "자비스 세 달 계획": "work queue",
        "자비스 다음 3개월 뭐 해": "work queue",
        "자비스 해자 어떻게 열어": "work queue",
        "해자 어떻게 열어": "work queue",
        "통합 지금 추가해도 돼": "work queue",
        "통합 언제 추가해": "work queue",
        "자비스 통합 언제 추가해": "work queue",
        "조이에서 피해야 할 것": "work queue",
        "자비스 전략": "work queue",
        "자비스 방향": "work queue",
        "조이 이후 뭐 만들까": "work queue",
        "자비스 조이 이후 뭐 만들어": "work queue",
        "조이 따라해야 해": "work queue",
        "조이 따라할까": "work queue",
        "자비스 조이 따라해야 해": "work queue",
        "자비스 컴패니언 해야 해": "work queue",
        "자비스 동반자 해야 해": "work queue",
        "린디 연구": "work queue",
        "마누스 연구": "work queue",
        "오픈클로 연구": "work queue",
        "헤르메스 연구": "work queue",
        "왜 동반자 안 해": "work queue",
        "왜 컴패니언 안 해": "work queue",
        "동반자 페르소나 피해야 해": "work queue",
        "동반자 레이어 해야 해": "work queue",
        "컴패니언 페르소나 피해야 해": "work queue",
        "컴패니언 레이어 해야 해": "work queue",
    }
    wrong_fragments = [
        "wikipedia",
        "safe next actions",
        "cockpit",
        "channel health",
        "pending approvals",
        "research is having trouble",
        "what should Jarvis do next",
        "fallback chat",
    ]
    for phrase, expected in cases.items():
        out = suggest_command(phrase)
        if not out or expected not in out:
            raise SystemExit(f"handoff/work-queue intent should suggest {expected!r}: {phrase!r} -> {out!r}")
        for wrong in wrong_fragments:
            if wrong in out:
                raise SystemExit(
                    f"handoff/work-queue intent suggested stale/wrong command "
                    f"{wrong!r}: {phrase!r} -> {out!r}"
                )


def test_runtime_preplanner_redirects_handoff_and_work_queue_before_chat_or_wikipedia() -> None:
    import tempfile
    from pathlib import Path

    from jarvis_v2.scripts.test_runtime import make_temp_runtime

    cases = {
        "agent landscape research": "work queue",
        "agent research note": "work queue",
        "AI agent landscape": "work queue",
        "every AI agent research": "work queue",
        "other Jarvis models": "work queue",
        "other Jarvis models research": "work queue",
        "what did the agent research say": "work queue",
        "what did you find about AI agents": "work queue",
        "what did you find about other Jarvis models": "work queue",
        "what about other Jarvis models": "work queue",
        "what did you find about Zoey": "work queue",
        "what did you learn from Zoey": "work queue",
        "Zoey research": "work queue",
        "Zoey OS research": "work queue",
        "OpenClaw research": "work queue",
        "Hermes research": "work queue",
        "Lindy research": "work queue",
        "Manus research": "work queue",
        "Devin research": "work queue",
        "what should Jarvis borrow from OpenClaw": "work queue",
        "what should Jarvis borrow from Lindy": "work queue",
        "what should Jarvis copy from OpenClaw": "work queue",
        "what should Jarvis copy from Lindy": "work queue",
        "what should Jarvis borrow from Zoey": "work queue",
        "what should Jarvis avoid from Zoey": "work queue",
        "what should Jarvis avoid from AI agents": "work queue",
        "what should Jarvis do after the research": "work queue",
        "what is the Jarvis strategy": "work queue",
        "Jarvis strategy": "work queue",
        "Jarvis product strategy": "work queue",
        "Jarvis direction": "work queue",
        "what is the strategy after the research": "work queue",
        "what is the right move for Jarvis": "work queue",
        "what should we do after Zoey research": "work queue",
        "what should Jarvis build after Zoey research": "work queue",
        "should Jarvis use companions": "work queue",
        "should Jarvis have companions": "work queue",
        "should Jarvis add companion personas": "work queue",
        "should Jarvis copy Zoey": "work queue",
        "should Jarvis copy Lindy": "work queue",
        "should Jarvis copy OpenClaw": "work queue",
        "what was the AI agent research recommendation": "work queue",
        "what did research say to avoid": "work queue",
        "what is Jarvis moat": "work queue",
        "what is the Jarvis moat": "work queue",
        "Jarvis moat": "work queue",
        "why no companions": "work queue",
        "why no companion personas": "work queue",
        "why not companion personas": "work queue",
        "what did Claude say about Zoey": "work queue",
        "what did Claude recommend about companions": "work queue",
        "조이 조사 결과": "work queue",
        "Zoey 조사 결과": "work queue",
        "에이전트 조사 결과": "work queue",
        "모든 AI 에이전트 조사": "work queue",
        "다른 자비스 모델": "work queue",
        "자비스 모델 조사": "work queue",
        "자비스 차별점": "work queue",
        "자비스 해자": "work queue",
        "조이에서 피해야 할 것": "work queue",
        "자비스 전략": "work queue",
        "자비스 방향": "work queue",
        "조이 이후 뭐 만들까": "work queue",
        "자비스 조이 이후 뭐 만들어": "work queue",
        "조이 따라해야 해": "work queue",
        "조이 따라할까": "work queue",
        "자비스 조이 따라해야 해": "work queue",
        "자비스 컴패니언 해야 해": "work queue",
        "자비스 동반자 해야 해": "work queue",
        "린디 연구": "work queue",
        "마누스 연구": "work queue",
        "오픈클로 연구": "work queue",
        "헤르메스 연구": "work queue",
        "왜 동반자 안 해": "work queue",
        "왜 컴패니언 안 해": "work queue",
        "동반자 페르소나 피해야 해": "work queue",
        "동반자 레이어 해야 해": "work queue",
        "컴패니언 페르소나 피해야 해": "work queue",
        "컴패니언 레이어 해야 해": "work queue",
    }
    wrong_fragments = [
        "Wikipedia is having trouble",
        "fallback chat",
        "model_unavailable",
        "safe next actions",
        "channel health",
        "queued as approval",
        "research is having trouble",
        "what should Jarvis do next",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rt = make_temp_runtime(Path(tmp))
        for phrase, expected in cases.items():
            approvals_before = len(rt.store.list_pending_approvals(limit=100))
            res = rt.handle(phrase)
            approvals_after = len(rt.store.list_pending_approvals(limit=100))
            response = getattr(res, "response", "")
            if "Did you mean" not in response or expected not in response:
                raise SystemExit(
                    f"runtime should redirect handoff/work-queue phrase {phrase!r} "
                    f"to {expected!r}: {response!r}"
                )
            if approvals_after != approvals_before:
                raise SystemExit(
                    f"handoff/work-queue phrase should not queue approvals: "
                    f"{phrase!r} {approvals_before}->{approvals_after}"
                )
            if getattr(res, "tool_results", []):
                raise SystemExit(
                    "handoff/work-queue phrase should not execute tools before the user "
                    f"sends the suggested command: {phrase!r}"
                )
            for wrong in wrong_fragments:
                if wrong in response:
                    raise SystemExit(
                        f"handoff/work-queue phrase leaked stale path "
                        f"{wrong!r}: {phrase!r} -> {response!r}"
                    )
            meta = getattr(res, "metadata", {}) or {}
            chat_response = meta.get("chat_response") or {}
            if chat_response.get("runtime_route") != "command_suggestion":
                raise SystemExit(
                    f"runtime should tag handoff/work-queue phrase {phrase!r} "
                    f"as a command suggestion: {chat_response!r}"
                )
            if chat_response.get("pre_planner_command_suggestion") is not True:
                raise SystemExit(
                    f"runtime should mark handoff/work-queue redirect as pre-planner: "
                    f"{phrase!r} -> {chat_response!r}"
                )
            runtime_trace = meta.get("runtime_trace") or {}
            if runtime_trace.get("approval_queue_delta") != 0 or runtime_trace.get("new_approval_ids"):
                raise SystemExit(
                    "handoff/work-queue redirect should not alter approval queue: "
                    f"{phrase!r} -> {runtime_trace}"
                )


def main() -> None:
    test_typo_near_miss_is_suggested()
    test_currency_and_market_typos_are_suggested()
    test_short_planner_alias_typos_are_suggested()
    test_storage_recovery_typos_are_suggested()
    test_setup_diagnostic_typos_are_suggested()
    test_control_help_typos_are_suggested()
    test_approval_and_proof_typos_are_suggested()
    test_approval_gate_intents_redirect_to_read_only_approval_commands()
    test_approval_failure_intents_redirect_to_read_only_approval_summary()
    test_harness_starter_typos_are_suggested()
    test_completion_and_agi_status_intents_redirect_to_read_only_gate_commands()
    test_progress_roadmap_goal_intents_redirect_to_read_only_control_commands()
    test_subagent_orchestration_intents_redirect_to_status()
    test_model_brain_intents_redirect_to_model_routing_status()
    test_model_brain_failure_intents_redirect_to_model_routing_status()
    test_eval_verification_intents_redirect_to_read_only_proof_commands()
    test_acceptance_finish_line_intents_redirect_to_read_only_cockpit()
    test_cockpit_attention_intents_redirect_to_read_only_cockpit()
    test_cockpit_lane_intents_redirect_to_read_only_cockpit()
    test_memory_learning_intents_redirect_to_read_only_memory_commands()
    test_guardrail_and_freeze_intents_redirect_to_cockpit()
    test_tool_control_plane_intents_redirect_to_read_only_commands()
    test_health_diagnostic_intents_redirect_to_read_only_status_commands()
    test_usage_example_intents_redirect_to_read_only_discovery_commands()
    test_daily_operator_intents_redirect_to_read_only_status_commands()
    test_scheduler_automation_intents_redirect_to_read_only_job_status()
    test_brief_failure_intents_redirect_to_recent_runs()
    test_messaging_channel_status_intents_redirect_to_read_only_channel_health()
    test_phone_control_intents_redirect_to_read_only_commands()
    test_audit_history_intents_redirect_to_read_only_audit_commands()
    test_trust_boundary_intents_redirect_to_read_only_safety_commands()
    test_next_step_operator_intents_redirect_to_read_only_continuity_commands()
    test_korean_operator_intents_redirect_to_read_only_status_commands()
    test_korean_scheduler_automation_intents_redirect_to_read_only_job_status()
    test_korean_completion_and_agi_intents_redirect_to_read_only_gate_commands()
    test_korean_progress_roadmap_goal_intents_redirect_to_read_only_control_commands()
    test_exact_korean_goal_aliases_do_not_bypass_the_planner()
    test_korean_messaging_channel_status_intents_redirect_to_read_only_channel_health()
    test_korean_brief_failure_intents_redirect_to_recent_runs()
    test_korean_model_brain_intents_redirect_to_model_routing_status()
    test_korean_model_brain_failure_intents_redirect_to_model_routing_status()
    test_korean_eval_verification_intents_redirect_to_read_only_proof_commands()
    test_korean_memory_learning_intents_redirect_to_read_only_memory_commands()
    test_korean_audit_history_intents_redirect_to_read_only_audit_commands()
    test_korean_trust_boundary_intents_redirect_to_read_only_safety_commands()
    test_korean_approval_gate_intents_redirect_to_read_only_approval_commands()
    test_korean_approval_failure_intents_redirect_to_read_only_approval_summary()
    test_korean_next_step_intents_redirect_to_read_only_continuity_commands()
    test_korean_voice_intents_redirect_to_read_only_voice_commands()
    test_voice_failure_intents_redirect_to_voice_setup_check()
    test_voice_control_and_readiness_intents_redirect_to_read_only_voice_commands()
    test_handoff_and_work_queue_intents_redirect_to_read_only_commands()
    test_help_promised_operator_typos_are_suggested()
    test_continuity_starter_typos_are_suggested()
    test_real_conversation_is_not_hijacked()
    test_gibberish_and_empty_return_none()
    test_exact_command_is_not_echoed_back()
    test_long_input_is_skipped()
    test_runtime_uses_suggestion_over_generic_chat()
    test_runtime_exact_cockpit_aliases_execute_read_only_cockpit()
    test_runtime_exact_agent_moat_plan_aliases_execute_work_queue_read_only()
    test_runtime_exact_read_only_tool_name_aliases_execute_status_tools()
    test_runtime_exact_registered_read_only_tool_names_execute_without_manual_aliases()
    test_runtime_redirects_argument_required_read_only_command_stubs_before_chat_or_empty_plans()
    test_runtime_web_search_returns_name_bound_success_without_widening_fetch_page()
    test_runtime_routes_subagent_status_to_dedicated_read_only_tool()
    test_runtime_redirects_model_brain_intents_to_model_routing_status()
    test_runtime_preplanner_redirects_model_brain_failure_intents_to_model_routing_status()
    test_runtime_redirects_eval_verification_intents_to_read_only_proof_commands()
    test_runtime_redirects_approval_gate_intents_to_read_only_approval_commands()
    test_runtime_executes_approval_failure_intents_as_read_only_summary()
    test_runtime_redirects_memory_learning_intents_to_read_only_memory_commands()
    test_runtime_redirects_freeze_status_to_cockpit()
    test_runtime_redirects_daily_operator_intents_to_existing_commands()
    test_runtime_preplanner_redirects_phone_control_intents_before_call_routing()
    test_runtime_preplanner_redirects_usage_example_intents_before_call_or_chat_drift()
    test_runtime_preplanner_redirects_messaging_channel_status_before_dispatch_or_model_routing()
    test_runtime_preplanner_redirects_tool_control_plane_intents_before_tool_detail()
    test_runtime_redirects_control_help_suggestions_before_fallback_chat()
    test_runtime_preplanner_redirects_health_diagnostic_intents_before_status_drift()
    test_runtime_redirects_scheduler_automation_intents_to_read_only_job_status()
    test_runtime_preplanner_redirects_brief_failure_intents_to_recent_runs()
    test_runtime_redirects_audit_history_intents_to_existing_commands()
    test_runtime_redirects_trust_boundary_intents_to_existing_commands()
    test_runtime_preplanner_redirects_capability_questions_before_action_planning()
    test_runtime_preplanner_redirects_korean_capability_questions_before_chat_fallback()
    test_runtime_redirects_next_step_operator_intents_to_existing_commands()
    test_runtime_redirects_korean_operator_intents_to_existing_commands()
    test_runtime_preplanner_redirects_korean_completion_and_agi_before_fallback_chat()
    test_runtime_redirects_progress_roadmap_and_goal_status_before_fallback_chat()
    test_runtime_redirects_korean_model_brain_intents_to_model_routing_status()
    test_runtime_redirects_korean_eval_verification_intents_to_read_only_proof_commands()
    test_runtime_preplanner_redirects_korean_evidence_phrasing_before_fallback_chat()
    test_runtime_redirects_korean_memory_learning_intents_to_read_only_memory_commands()
    test_runtime_redirects_korean_audit_history_intents_to_existing_commands()
    test_runtime_redirects_korean_trust_boundary_intents_to_existing_commands()
    test_runtime_redirects_korean_approval_gate_intents_to_existing_commands()
    test_runtime_routes_korean_readonly_status_surfaces_to_existing_tools()
    test_runtime_routes_channel_status_surfaces_to_channel_health()
    test_runtime_routes_korean_channel_status_surfaces_to_channel_health()
    test_runtime_redirects_korean_next_step_intents_to_existing_commands()
    test_runtime_redirects_korean_voice_intents_to_existing_commands()
    test_runtime_preplanner_redirects_voice_control_and_readiness_before_chat_or_voice_execution()
    test_runtime_preplanner_redirects_voice_failure_intents_to_voice_setup_check()
    test_runtime_preplanner_redirects_handoff_and_work_queue_before_chat_or_wikipedia()
    print("Command suggest smoke passed")


if __name__ == "__main__":
    main()
