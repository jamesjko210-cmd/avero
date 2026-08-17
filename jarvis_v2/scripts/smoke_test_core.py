from __future__ import annotations

from datetime import datetime
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfoNotFoundError

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.v3_commands import V3_DASHBOARD_COMMAND, V3_DASHBOARD_INFO_COMMAND


def assert_route(command: str, tool_name: str, args: dict | None = None) -> None:
    plan = RuleBasedPlanner().plan(command)
    actions = plan.actions
    if len(actions) != 1 or actions[0].tool_name != tool_name:
        raise SystemExit(f"{command!r} should route to {tool_name}: {[(a.tool_name, a.args) for a in actions]}")
    expected_args = args or {}
    if actions[0].args != expected_args:
        raise SystemExit(f"{command!r} should pass {expected_args}: {actions[0].args}")


def test_memory_read_alias_routes() -> None:
    for command in (
        "memories please",
        "memory list please",
        "show latest memories",
        "facts please",
        "remembered facts please",
    ):
        assert_route(command, "recent_memories", {"limit": 10})


def test_korean_time_and_weather_aliases_route_without_chat() -> None:
    for command in (
        "지금 몇 시야",
        "몇 시예요?",
        "지금 몇 시인지 알려줘",
        "현재 시간은?",
        "현재 시간 뭐예요?",
        "현재 시간 알려줘",
        "시간 확인해 주세요",
        "나는 지금 한국어를 말하고 있어, 지금 몇시야?",
        "지금 며시아",
        "나는 지금 한국에 살고 있어, 지금 며시아?",
    ):
        assert_route(command, "current_time", {"locale": "ko"})
    for command, location in (
        ("서울은 지금 몇 시야", "Seoul"),
        ("도쿄 현재 시간은?", "Tokyo"),
        ("뉴욕 시간 알려줘", "New York"),
        ("런던은 몇 시예요?", "London"),
    ):
        assert_route(command, "current_time", {"location": location, "locale": "ko"})
    for command in ("지금 날씨야", "오늘 날씨 어때?", "날씨 알려줘", "현재 날씨 보여 주세요"):
        assert_route(command, "get_weather")
    assert_route("search memories for jarvis", "search_memory", {"query": "jarvis"})
    assert_route("search memory for jarvis please", "search_memory", {"query": "jarvis"})
    assert_route("memories about Jarvis please", "search_memory", {"query": "Jarvis"})
    assert_route("what do you remember about jarvis please", "search_memory", {"query": "jarvis"})
    assert_route("what do you know about Maya", "search_memory", {"query": "Maya"})
    for command in (
        "recent facts please",
        "what facts do you know",
    ):
        assert_route(command, "recent_memories", {"limit": 10})


def test_korean_email_aliases_route_without_chat() -> None:
    for command, args in (
        ("최근 이메일 보여줘", {}),
        ("이메일 확인해줘", {}),
        ("안 읽은 이메일 확인해줘", {"unread": True}),
        ("새 메일 보여줘", {"unread": True}),
    ):
        assert_route(command, "read_emails", args)
    for command, args in (
        ("이메일에서 영수증 찾아줘", {"query": "영수증"}),
        ("invoice 이메일 찾아줘", {"query": "invoice"}),
        ("이메일에서 project alpha 검색해줘", {"query": "project alpha"}),
    ):
        assert_route(command, "search_emails", args)


def test_korean_calendar_period_aliases_route_without_chat() -> None:
    for command, range_name in (
        ("오늘 오전 일정 보여줘", "today_morning"),
        ("내일 오후 캘린더 확인해줘", "tomorrow_afternoon"),
        ("내일 저녁 스케줄 알려줘", "tomorrow_evening"),
    ):
        assert_route(command, "list_events", {"range": range_name})
    for command, range_name in (
        ("이번 주 일정 보여줘", "week"),
        ("다음주 캘린더 확인해줘", "next_week"),
        ("이번 달 스케줄 알려줘", "month"),
        ("다음달 일정 보여줘", "next_month"),
        ("이번 주말 일정", "weekend"),
        ("다음 주말 캘린더", "next_weekend"),
    ):
        assert_route(command, "list_events", {"range": range_name})


def test_free_text_routes_preserve_regular_whitespace() -> None:
    cases = (
        (
            "add task reconcile ledger:  col A     col B     col C",
            "add_task",
            "body",
            "reconcile ledger:  col A     col B     col C",
        ),
        (
            "remember that the wifi password is:      hunter2spaces",
            "remember",
            "body",
            "the wifi password is:      hunter2spaces",
        ),
        (
            "remind me to review     the     aligned      columns in 5 minutes",
            "set_reminder",
            "text",
            "remind me to review     the     aligned      columns in 5 minutes",
        ),
        (
            "translate hello     world to Korean",
            "translate",
            "request",
            "translate hello     world to Korean",
        ),
        (
            "add task keep     spacing and then show my goals",
            "add_task",
            "body",
            "keep     spacing",
        ),
    )
    for command, tool_name, argument, expected in cases:
        plan = RuleBasedPlanner().plan(command)
        if len(plan.actions) != 1 or plan.actions[0].tool_name != tool_name:
            raise SystemExit(f"{command!r} should route to {tool_name}: {plan.actions}")
        actual = plan.actions[0].args.get(argument)
        if actual != expected:
            raise SystemExit(
                f"{command!r} should preserve free-text whitespace in {argument}: "
                f"got {actual!r}, expected {expected!r}"
            )


def test_planner_truncates_compound_and_then_sentences() -> None:
    # Real gap found live 2026-07-10 (rounds 33-35): individually patching
    # every free-text capture group and bare exact-match trigger against
    # compound "X and then Y" sentences does not scale across a file this
    # large -- per-tool fixes this session found the bug independently in
    # add_task, create_reminder, find_contact, note/task/file search,
    # create_goal, decision recording, and export_state_snapshot, plus
    # untested cases here in translate/research (raw text bled whole into
    # the tool arg) and bare exact-match triggers like roadmap/readiness/
    # safety/list-goals/preferences/tasks (which require NO trailing text at
    # all, so the whole match failed and a LATER, unrelated pattern silently
    # claimed the second clause instead -- worse than clause-bleed, since the
    # first, explicitly-requested action never ran at all). `RuleBasedPlanner
    # .plan()` now truncates at the first top-level " and then " before any
    # matching runs, closing this class of bug once for the whole file. This
    # test exercises that central behavior directly rather than relying only
    # on the scattered per-tool tests added alongside the original fixes.
    assert_route(
        "translate hello to french and then check my email",
        "translate",
        {"request": "translate hello to french"},
    )
    assert_route(
        "research quantum computing and then check my email",
        "research",
        {"query": "quantum computing"},
    )
    assert_route("roadmap report and then show my tasks", "roadmap_report", {})
    assert_route("readiness report and then show weather", "readiness_report", {})
    assert_route("safety status and then check email", "safety_status", {})
    assert_route("list my goals and then check weather", "list_goals", {"status": "active"})
    assert_route("show my preferences and then check weather", "list_preferences", {})
    assert_route("list my tasks and then check weather", "list_tasks", {"status": "open"})
    assert_route("save state and then show my calendar", "export_state_snapshot", {})
    # Deliberately scoped to " and then " only, NOT bare " and " (no "then"):
    # bare "and" is legitimate content in many working commands today, and
    # must NOT be truncated.
    assert_route("add 5 and 10 and 15", "calculate", {"expression": "5 plus 10 plus 15"})
    assert_route("search for apples and bananas", "web_lookup", {"query": "apples and bananas"})
    convert_actions = RuleBasedPlanner().plan("convert 100 fahrenheit to celsius and kelvin").actions
    if [a.tool_name for a in convert_actions] != ["convert_units"] or convert_actions[0].args.get("to_unit") != "celsius":
        raise SystemExit(f"bare 'and' compound conversion target should still resolve to its first unit: {convert_actions}")
    factorial_actions = RuleBasedPlanner().plan("what's 5 factorial and 6 factorial").actions
    if [a.tool_name for a in factorial_actions] != ["calculate"] or "factorial and" not in factorial_actions[0].args.get("expression", ""):
        raise SystemExit(f"bare 'and' factorial chain should not be truncated by the 'and then' guard: {factorial_actions}")


def test_and_then_split_is_not_vulnerable_to_pathological_whitespace() -> None:
    # Real gap found live 2026-07-10 (round 37, self-audit of round 35): the
    # centralized "and then" truncation added in round 35 originally used
    # unbounded `\s+and\s+then\s+`, which suffers catastrophic O(n^2)
    # backtracking on long whitespace-heavy input containing no "and then" --
    # bounding it to \s{1,10} fixed that one regex in isolation. But testing
    # the SAME pathological input through the full plan() pipeline still took
    # 34.6s, proving a different, pre-existing regex was the real bottleneck.
    # Root cause: _strip_trailing_politeness() (called unconditionally on
    # EVERY plan() invocation, before any routing) used
    # `(?:[\s,.;!?]*(?:please|pls|thanks|thank you)[\s,.;!?]*)+$` -- a
    # nested-quantifier ReDoS shape (inner `*` can match empty, so the outer
    # `+` has exponentially many ways to partition a long non-matching
    # whitespace run). Because this runs on every single command with no
    # gate, it was the highest-impact vulnerability found this session: any
    # user input ending in a long run of whitespace/punctuation could hang
    # the whole planner, independent of the rest of the sentence. Fixed by
    # bounding both quantifiers (`{0,20}` / `{1,10}`), which cannot blow up
    # and comfortably covers any realistic trailing-politeness phrase. This
    # test exercises both the entry-point split AND the politeness stripper
    # directly, since the pipeline-level regression is what actually caught
    # the pre-existing bug that the isolated round-35 test missed.
    import time

    from jarvis_v2.agent.planner import _strip_trailing_politeness

    pathological = "add task buy" + " " * 20000 + "milk"
    start = time.monotonic()
    RuleBasedPlanner().plan(pathological)
    elapsed = time.monotonic() - start
    if elapsed > 2.0:
        raise SystemExit(
            f"planner took {elapsed:.2f}s on pathological whitespace input (budget 2.0s) -- "
            "likely a regression of the round-37 ReDoS mitigations"
        )

    trailing_pathological = "hello" + " ,.;!?" * 50000
    start = time.monotonic()
    _strip_trailing_politeness(trailing_pathological)
    elapsed2 = time.monotonic() - start
    if elapsed2 > 2.0:
        raise SystemExit(
            f"_strip_trailing_politeness took {elapsed2:.2f}s on pathological trailing "
            "punctuation/whitespace (budget 2.0s) -- likely a regression of the round-37 "
            "nested-quantifier ReDoS fix"
        )

    politeness_cases = [
        ("remind me to call mom please", "remind me to call mom"),
        ("what's the weather thanks", "what's the weather"),
        ("add task buy milk, thank you", "add task buy milk"),
        ("please please please thanks", ""),
        ("do the thing pls, thanks!", "do the thing"),
        ("no trailing politeness here", "no trailing politeness here"),
    ]
    for inp, expected in politeness_cases:
        got = _strip_trailing_politeness(inp)
        if got != expected:
            raise SystemExit(
                f"_strip_trailing_politeness regressed on {inp!r}: got {got!r}, expected {expected!r}"
            )


def test_specialist_packet_request_is_not_vulnerable_to_pathological_input() -> None:
    # Real gap found live 2026-07-10 (round 38, follow-up sweep after round
    # 37's fix): a lightweight static grep for the same nested-quantifier
    # shape (`[*+])[*+]`) elsewhere in planner.py surfaced a second, WORSE
    # ReDoS in `_clean_specialist_packet_request()` (line ~176):
    # `re.fullmatch(r"(?:please|pls|thanks|thank you|[\s,.;:!?]+)+", ...)`.
    # Unlike round 37's finding, the vulnerable branch here
    # (`[\s,.;:!?]+`) can match ONE OR MORE (not zero) characters, but it is
    # a bare, unanchored alternative inside the repeated group with no
    # required literal -- the classic `(a+)+` catastrophic-backtracking
    # shape. Confirmed EXPONENTIAL (not just O(n^2)): plain trailing-nonmatch
    # input hung at n=24 chars already (0.67s and climbing fast: 0.04s at
    # n=20, 0.17s at n=22, 0.67s at n=24 -- roughly doubling every 2 chars).
    # A first attempt to fix this the same way as round 37 (bounding both
    # quantifiers to `{1,20}`) was NOT sufficient and was itself caught
    # live: bounding the OUTER repetition count does not prevent
    # combinatorial blowup when the INNER quantifier is also variable-length
    # with no forced literal anchor -- the engine still enumerates
    # combinatorially many ways to partition the matched span across
    # repetitions before concluding failure. Confirmed this empirically (not
    # just in theory) by watching real backgrounded test processes accumulate
    # 88-120 CPU-minutes each against the "fixed" {1,20} pattern before being
    # killed. Root-caused and fixed properly by eliminating the
    # repeated-alternation-with-bare-filler shape entirely: strip separator
    # characters and politeness words in two separate, unnested, single-pass
    # `re.sub` calls (no repetition operator wraps an alternation at all),
    # then check whether anything is left. This is O(n) by construction --
    # verified 100,000 pathological chars complete in effectively 0ms, not
    # just "under some budget". Also worth noting for future audits: bounding
    # quantifiers (round 37's fix shape) is only safe when at least one
    # branch of the repeated alternation cannot match empty AND no branch is
    # a bare variable-length filler with nothing anchoring it -- bounding
    # alone is not a universal fix for every nested-quantifier shape.
    import signal
    import time

    from jarvis_v2.agent.planner import _clean_specialist_packet_request, DEFAULT_HARNESS_PACKET_REQUEST

    class _TooSlow(Exception):
        pass

    def _alarm(signum, frame):
        raise _TooSlow()

    previous_handler = signal.signal(signal.SIGALRM, _alarm)
    try:
        for n in (24, 1000, 20000, 100000):
            pathological = " " * n + "X"
            signal.setitimer(signal.ITIMER_REAL, 2.0)
            start = time.monotonic()
            try:
                _clean_specialist_packet_request(pathological)
            except _TooSlow:
                raise SystemExit(
                    f"_clean_specialist_packet_request took >2.0s on {n}-char pathological "
                    "input -- likely a regression of the round-38 ReDoS fix"
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            elapsed = time.monotonic() - start
            if elapsed > 2.0:
                raise SystemExit(
                    f"_clean_specialist_packet_request took {elapsed:.2f}s on {n}-char "
                    "pathological input (budget 2.0s) -- likely a regression of the round-38 ReDoS fix"
                )
    finally:
        signal.signal(signal.SIGALRM, previous_handler)

    cases = [
        ("please", DEFAULT_HARNESS_PACKET_REQUEST),
        ("please, thanks!", DEFAULT_HARNESS_PACKET_REQUEST),
        ("pls thank you", DEFAULT_HARNESS_PACKET_REQUEST),
        ("", DEFAULT_HARNESS_PACKET_REQUEST),
        ("   ", DEFAULT_HARNESS_PACKET_REQUEST),
        ("please please please thanks", DEFAULT_HARNESS_PACKET_REQUEST),
        ("for the migration status", "the migration status"),
        ("check the database schema", "check the database schema"),
        ("pleasant weather today", "pleasant weather today"),
    ]
    for inp, expected in cases:
        got = _clean_specialist_packet_request(inp)
        if got != expected:
            raise SystemExit(
                f"_clean_specialist_packet_request regressed on {inp!r}: got {got!r}, expected {expected!r}"
            )


def test_strip_trailing_politeness_suffix_only_helper() -> None:
    # Real gap found live 2026-07-10 (round 43), discovered while investigating
    # why "show my recent notes please"-style phrasing (working bare, per round
    # 40) still fell through to chat with "please" attached. Root cause: the
    # global politeness-retry mechanism in RuleBasedPlanner.plan() strips BOTH
    # leading words ("show"/"please"/"just"/etc, via LEADING_POLITE_WRAPPER_RE)
    # AND trailing politeness, but only trusts the retry's result for tools in
    # POLITE_COMMAND_RETRY_TOOLS -- a deliberately narrow allowlist that
    # excludes list_jarvis_notes/list_goals/list_preferences/list_people/
    # list_skills (see project_jarvis_strip_trailing_politeness_leading_strip
    # memory for why the leading-strip side effect makes broadening that
    # allowlist unsafe: verb-only entries like "show jarvis notes" have no
    # bare "jarvis notes" fallback in their exact-match sets, so a naive full
    # strip would silently break them). Fixed by extracting the trailing-only
    # half of _strip_trailing_politeness into its own reusable helper,
    # _strip_trailing_politeness_suffix_only(), and using THAT (not the full
    # leading+trailing function) as a local, per-block strip in the 5 affected
    # exact-match sets. This is safe specifically because it's a no-op when
    # there's no trailing please/pls/thanks to strip, so it cannot change the
    # outcome for any phrase that doesn't end in one -- verified directly here
    # rather than just trusting that argument.
    from jarvis_v2.agent.planner import _strip_trailing_politeness_suffix_only

    noop_cases = [
        "show jarvis notes",
        "show latest jarvis notes",
        "my active goals",
        "who do you know",
    ]
    for text in noop_cases:
        got = _strip_trailing_politeness_suffix_only(text)
        if got != text:
            raise SystemExit(
                f"_strip_trailing_politeness_suffix_only should be a no-op with no trailing "
                f"please/pls/thanks present: {text!r} -> {got!r}"
            )

    strips_cases = [
        ("show my recent notes please", "show my recent notes"),
        ("show my active goals please", "show my active goals"),
        ("show my recent preferences please", "show my recent preferences"),
        ("show my recent people please", "show my recent people"),
        ("show my recent skills please", "show my recent skills"),
    ]
    for text, expected in strips_cases:
        got = _strip_trailing_politeness_suffix_only(text)
        if got != expected:
            raise SystemExit(
                f"_strip_trailing_politeness_suffix_only should strip trailing please only: "
                f"{text!r} -> {got!r}, expected {expected!r}"
            )

    routed_cases = [
        ("show my recent notes please", "list_jarvis_notes"),
        ("show my active goals please", "list_goals"),
        ("show my recent preferences please", "list_preferences"),
        ("show my recent people please", "list_people"),
        ("show my recent skills please", "list_skills"),
        # Leading-strip-sensitive entries must still work unaffected.
        ("show jarvis notes", "list_jarvis_notes"),
        ("show latest jarvis notes", "list_jarvis_notes"),
    ]
    for command, expected_tool in routed_cases:
        assert_route(command, expected_tool, {} if expected_tool != "list_goals" else {"status": "active"})


def test_non_frozen_lazy_capture_patterns_are_bounded() -> None:
    # Real gap found live 2026-07-10 (round 44): a static-signature sweep for
    # the round-43 finding's exact shape (an unbounded lazy `(?P<x>.+?)`
    # capture required to be followed by a specific later literal, in an
    # unanchored re.search) turned up 18 candidates. 14 were inside the
    # frozen send/call routing block (already flagged, not touched -- see
    # project_jarvis_frozen_zone_redos_finding memory). Of the remaining 4
    # non-frozen call sites, all 4 were genuinely vulnerable (confirmed via
    # direct timing, not just visual inspection): preference_match,
    # save_skill_match, write_match, and 2 of transform_text_match's 5
    # chained alternatives (the other 3 have mode-word-first shapes and were
    # confirmed already fast). Growth was quadratic-ish, not exponential like
    # round 43's kakao/telegram finding, but still genuinely unbounded before
    # this fix: transform_text's "make X mode" alternative alone took 12.2s
    # on 20,000 non-matching whitespace characters. Bounded each vulnerable
    # lazy capture to a generous cap no legitimate preference key/skill name/
    # file path/text-to-transform would ever need (100-2000 chars depending
    # on the field). Important nuance, not swept under the rug: bounding the
    # OUTER lazy capture alone does not fully eliminate the cost for inputs
    # far beyond the bound -- a SUBSEQUENT \s+ in the same pattern can still
    # backtrack against whatever of the long input remains after the bounded
    # group, so e.g. transform's "make" pattern still takes ~12s at 100,000
    # pathological characters even after bounding its capture to 2000. This
    # is a real, verified, substantial improvement (was unboundedly bad
    # before, is now capped and vastly faster for realistic-to-moderate
    # input lengths) but not a complete fix for arbitrarily large adversarial
    # input -- that would need an entry-point-level total-input-length guard
    # (same architectural shape as round 35's "and then" truncation), which
    # is a bigger, more consequential change intentionally left for explicit
    # discussion rather than added unilaterally here. This test asserts the
    # verified, bounded improvement, not a claim of full closure.
    # The later entry-point guard caps every regex-visible command at 4,000
    # characters and separately exercises much larger hostile inputs below.
    # Keep this isolated raw-pattern check at that reachable maximum; the old
    # 20,000-character fixture measured an impossible post-guard state and
    # became a stable false failure on Python 3.14 despite the real route
    # remaining bounded.
    import re
    import time

    bounded_patterns = {
        "preference_match": (
            r"^(?:set|save|remember)\s+preference\s+(?P<key>.{1,100}?)\s+(?:to|as|=)\s+(?P<value>.{1,1000}?)(?:\s+category\s+(?P<category>.+))?$",
            "set preference word",
        ),
        "save_skill_match": (
            r"^save\s+skill\s+(?P<name>.{1,100}?)\s+(?:when|trigger)\s+(?P<trigger>.{1,300}?)\s+(?:do|procedure|:)\s*(?P<body>.+)$",
            "save skill word",
        ),
        "write_match": (
            r"^(?:write|create|make)\s+(?:a\s+|an\s+)?(?:file\s+)?(?:called\s+)?(?P<path>.{1,300}?)\s+(?:with|as|:)\s*(?P<content>.+)$",
            "write word",
        ),
        "transform_make": (
            r"^make\s+(?P<text>.{1,2000}?)\s+(?P<mode>uppercase|lowercase|title\s?case|titlecase|sentence case|capitalized|slugified|slug|no spaces|camel\s?case|pascal\s?case|snake\s?case|kebab\s?case)(?: please)?[\?\.!]*$",
            "make word",
        ),
        "transform_convert": (
            r"^(?:convert|turn)\s+(?P<text>.{1,2000}?)\s+(?:to|into)\s+(?P<mode>uppercase|lowercase|title\s?case|titlecase|sentence case|capitalized|camel\s?case|pascal\s?case|snake\s?case|kebab\s?case|slugified|slug|no spaces|remove spaces|trim spaces|remove punctuation|strip punctuation)(?: please)?[\?\.!]*$",
            "convert word",
        ),
    }
    max_reachable_chars = 4000
    budget_s = 1.0
    for label, (pattern, prefix) in bounded_patterns.items():
        pathological = (prefix + " " * max_reachable_chars + "end")[:max_reachable_chars]
        start = time.monotonic()
        re.search(pattern, pathological, re.IGNORECASE | re.DOTALL)
        elapsed = time.monotonic() - start
        if elapsed > budget_s:
            raise SystemExit(
                f"{label} took {elapsed:.2f}s at the 4,000-char planner cap (budget {budget_s}s) -- "
                "likely a regression of the round-44 bounded-lazy-capture fix"
            )

    from jarvis_v2.agent.planner import RuleBasedPlanner

    planner = RuleBasedPlanner()
    correctness_cases = [
        ("set preference tone to direct", "set_preference"),
        ("save skill test when trigger fires do the thing", "save_skill"),
        ("write test.txt with hello world", "write_text_file"),
        ("create test.txt with hello world", "write_text_file"),
        ("make hello world uppercase", "transform_text"),
        ("convert hello world to uppercase", "transform_text"),
        ("set preference response style to direct and warm category communication", "set_preference"),
    ]
    for command, expected_tool in correctness_cases:
        plan = planner.plan(command)
        if [a.tool_name for a in plan.actions] != [expected_tool]:
            raise SystemExit(
                f"round-44 bounded-lazy-capture fix regressed routing for {command!r}: {plan.actions}"
            )


def test_entry_point_guard_bounds_frozen_zone_redos() -> None:
    # Fable-authorized fix (2026-07-10 checkpoint) for the round-43/44 frozen-
    # zone finding: the kakao/telegram/imessage "trailing-service" send
    # patterns use an unbounded lazy `(?P<to>.+?)` followed by a required
    # literal, which catastrophically backtracks on inputs containing long
    # whitespace runs -- fresh timing at the checkpoint confirmed the raw
    # kakao regex TIMES OUT (>4s) at just 500 internal spaces and the full
    # pipeline hung at ~1000. Because those patterns sit inside the frozen
    # send/call routing block, the fix is an entry-point guard in plan()
    # (`_normalize_planner_input`) that collapses whitespace runs of 5+ to a
    # single space and caps matching length at 4000 chars BEFORE any pattern
    # runs -- bounding the whole file's worst case, frozen block included,
    # without editing a single frozen line. The runtime models the ORIGINAL
    # message (runtime.py handle() -> chat.respond(user_input)), so chat
    # fidelity is unaffected by the cap; only inline captured content past
    # 4000 chars is truncated, a rare edge documented at the guard.
    import signal
    import time

    class _TooSlow(Exception):
        pass

    def _alarm(signum, frame):
        raise _TooSlow()

    previous_handler = signal.signal(signal.SIGALRM, _alarm)
    try:
        pathological_cases = {
            "kakao single-run": "send x" + " " * 1000 + "y",
            "kakao huge-run": "send" + " " * 50000 + "x",
            "telegram trailing": "message x" + " " * 5000 + "on telegram saying hi",
            "multi-run": ("word" + " " * 9) * 1000,
            "dense multi-run": ("a" + " " * 6) * 2000,
            "degenerate connectors": "send " + "to " * 5000,
        }
        for label, pathological in pathological_cases.items():
            signal.setitimer(signal.ITIMER_REAL, 3.0)
            start = time.monotonic()
            try:
                RuleBasedPlanner().plan(pathological)
            except _TooSlow:
                raise SystemExit(
                    f"planner exceeded 3.0s on {label!r} pathological input -- "
                    "likely a regression of the entry-point ReDoS guard"
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            elapsed = time.monotonic() - start
            if elapsed > 3.0:
                raise SystemExit(
                    f"planner took {elapsed:.2f}s on {label!r} (budget 3.0s) -- "
                    "likely a regression of the entry-point ReDoS guard"
                )
    finally:
        signal.signal(signal.SIGALRM, previous_handler)

    # The guard must not change routing or captured content for normal input,
    # including typical 2-4 space runs (only 5+ runs are collapsed).
    plan = RuleBasedPlanner().plan("send fixture a kakao saying hello world")
    if [a.tool_name for a in plan.actions] != ["send_kakao"] or plan.actions[0].args != {
        "to": "fixture",
        "message": "hello world",
    }:
        raise SystemExit(f"guard changed normal kakao routing/capture: {plan.actions}")
    plan = RuleBasedPlanner().plan("add task buy    milk")
    if [a.tool_name for a in plan.actions] != ["add_task"]:
        raise SystemExit(f"guard must leave 2-4 space runs untouched: {plan.actions}")


def test_korean_list_command_parity() -> None:
    # Fable checkpoint plan item B2 (2026-07-10): every probed Korean
    # list-command phrasing fell through to chat, where the model FABRICATED
    # capability denials ("Sorry, I'm not capable of showing your tasks
    # here") for tools Jarvis has -- the same hallucination class as the
    # English round-39 "list my open tasks" bug, across all six list
    # families. Fixed with exact-match Korean aliases in the planner's list
    # sets (planner lane; runtime.py's alias layer is Codex's active lane).
    # Unspaced goal forms are included because Korean chat commonly omits
    # spaces, and "목표목록"/"목표보여줘" were previously (only) caught by the
    # runtime's compact-matching PRE-PLANNER SUGGESTION set, which shadowed
    # the new planner route into a two-turn "did you mean" -- that shadow was
    # removed per the precedent documented at runtime.py's
    # _GOAL_STATUS_PRE_PLANNER comment (suggestions are protective only for
    # genuinely-ambiguous phrases like bare "목표", which stays suggested).
    for command, tool_name, args in [
        ("내 작업 보여줘", "list_tasks", {"status": "open"}),
        ("내 열린 작업 보여줘", "list_tasks", {"status": "open"}),
        ("할 일 목록", "list_tasks", {"status": "open"}),
        ("작업 목록", "list_tasks", {"status": "open"}),
        ("내 노트 보여줘", "list_jarvis_notes", None),
        ("최근 메모 보여줘", "list_jarvis_notes", None),
        ("메모 목록", "list_jarvis_notes", None),
        ("내 목표 보여줘", "list_goals", {"status": "active"}),
        ("목표 목록", "list_goals", {"status": "active"}),
        ("목표목록", "list_goals", {"status": "active"}),
        ("목표보여줘", "list_goals", {"status": "active"}),
        ("내 스킬 보여줘", "list_skills", None),
        ("스킬 목록", "list_skills", None),
        ("내 선호 보여줘", "list_preferences", None),
        ("선호 목록", "list_preferences", None),
        ("사람 목록", "list_people", None),
        ("아는 사람 보여줘", "list_people", None),
    ]:
        plan = RuleBasedPlanner().plan(command)
        if [a.tool_name for a in plan.actions] != [tool_name]:
            raise SystemExit(
                f"Korean list parity regressed: {command!r} should route to {tool_name}: "
                f"{[(a.tool_name, a.args) for a in plan.actions]}"
            )
        if args is not None and plan.actions[0].args != args:
            raise SystemExit(f"{command!r} should pass {args}: {plan.actions[0].args}")
    # Korean negation forms must keep falling to chat -- verified clean at the
    # checkpoint (B1: no Korean action pattern matches negated phrasings), and
    # the new aliases above must not change that.
    for command in ("하지 마", "작업 보여주지 마", "날씨 확인하지 마"):
        plan = RuleBasedPlanner().plan(command)
        if plan.actions:
            raise SystemExit(
                f"Korean negation must not execute an action: {command!r} -> {plan.actions}"
            )


def test_planner_routes_negated_requests_to_chat() -> None:
    # Real gap found live 2026-07-10 (round 36): none of this file's action-
    # matching patterns account for an explicit negation/cancellation
    # lead-in, so a negated request would silently execute the OPPOSITE of
    # what was asked -- "don't remind me to call mom" created a reminder
    # anyway, "never mind the weather" / "no need to check the weather"
    # still fetched the weather, and "don't tell me the weather" ran
    # get_weather with empty args. More severe than the round-33/34/35
    # clause-bleed bugs (those produced wrong DATA; this produces the WRONG
    # ACTION, including an unwanted persisted reminder the user explicitly
    # declined). Deterministic regex matching can't reliably understand
    # negation scope, so an unambiguous negation/cancellation lead-in now
    # routes to chat instead of guessing which action to skip.
    for command in (
        "don't add a task to buy milk",
        "don't remind me to call mom",
        "I don't need a reminder to call mom",
        "no need to check the weather",
        "don't search my notes for meeting",
        "never mind the weather",
        "cancel that, don't add the task",
        "actually don't create that goal",
        "don't tell me the weather",
    ):
        plan = RuleBasedPlanner().plan(command)
        if plan.actions or not plan.needs_model:
            raise SystemExit(f"negated request should route to chat instead of executing an action: {command!r} -> {plan.actions}")
    # Scoped tightly to the START of the command so it does NOT catch "don't"
    # appearing later as legitimate content (e.g. inside a task body).
    for command, expected_tool in (
        ("add task buy milk", "add_task"),
        ("add task buy milk, don't forget the eggs", "add_task"),
        ("remind me to not forget the meeting", "create_reminder"),
        ("remind me to call mom", "create_reminder"),
        ("search my notes for meeting", "search_jarvis_notes"),
    ):
        plan = RuleBasedPlanner().plan(command)
        if [a.tool_name for a in plan.actions] != [expected_tool]:
            raise SystemExit(f"negation guard should not catch 'don't' appearing later as content: {command!r} -> {plan.actions}")


def assert_boundary(metadata: dict, label: str) -> None:
    for key in (
        "calls_model",
        "executes_tools",
        "queues_approval",
        "controls_computer",
        "executes_side_effect",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
    ):
        if metadata.get(key):
            raise SystemExit(f"{label} unexpectedly set {key}: {metadata}")


def assert_no_local_path(value: object, label: str) -> None:
    text = str(value)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in text:
            raise SystemExit(f"{label} leaked a local path: {text}")


def test_timezone_database_failures_name_recovery() -> None:
    with TemporaryDirectory(prefix="jarvis-core-timezone-recovery-") as temp:
        runtime = make_temp_runtime(Path(temp))
        current_time = runtime.registry.get("current_time").handler
        time_difference = runtime.registry.get("time_difference").handler

        with patch(
            "jarvis_v2.tools.registry.ZoneInfo",
            side_effect=ZoneInfoNotFoundError("synthetic timezone failure"),
        ):
            unavailable_time = current_time({"location": "Seoul"})
            unavailable_difference = time_difference(
                {"source": "Seoul", "target": "London"}
            )

        class _NoOffsetTime:
            @staticmethod
            def utcoffset():
                return None

        class _NoOffsetDatetime:
            @staticmethod
            def now(_timezone):
                return _NoOffsetTime()

        with patch("jarvis_v2.tools.registry.datetime", _NoOffsetDatetime):
            unavailable_offset = time_difference(
                {"source": "Seoul", "target": "London"}
            )

        for label, result, expected_status in (
            ("current timezone database", unavailable_time, "timezone_unavailable"),
            ("time-difference timezone database", unavailable_difference, "timezone_unavailable"),
            ("time-difference timezone offset", unavailable_offset, "offset_unavailable"),
        ):
            if result.ok:
                raise SystemExit(f"{label} failure should fail closed: {result}")
            metadata = result.metadata
            status = metadata.get("time_status") or metadata.get("time_difference_status")
            if status != expected_status:
                raise SystemExit(f"{label} failure missed status metadata: {metadata}")
            for expected in ("setup check", "tzdata", "retry", "UTC"):
                if expected not in result.output:
                    raise SystemExit(
                        f"{label} failure missed recovery text {expected!r}: {result.output}"
                    )
            if metadata.get("next_command") != "setup check":
                raise SystemExit(f"{label} failure missed next command: {metadata}")
            if metadata.get("recovery_commands") != ["setup check", "time in UTC"]:
                raise SystemExit(f"{label} failure missed bounded recovery commands: {metadata}")
            if metadata.get("retry_requires_timezone_data_repair") is not True:
                raise SystemExit(f"{label} failure should require timezone repair: {metadata}")
            assert_boundary(metadata, label)
            assert_no_local_path(result.output, f"{label} output")
            assert_no_local_path(metadata, f"{label} metadata")


def test_current_time_formats_korean_locale_without_changing_english() -> None:
    class _FixedDatetime:
        @staticmethod
        def now(timezone=None):
            return datetime(2026, 8, 3, 23, 7, tzinfo=timezone)

    with TemporaryDirectory(prefix="jarvis-core-korean-time-") as temp:
        runtime = make_temp_runtime(Path(temp))
        current_time = runtime.registry.get("current_time").handler
        with patch("jarvis_v2.tools.registry.datetime", _FixedDatetime):
            local_korean = current_time({"locale": "ko"})
            seoul_korean = current_time({"location": "Seoul", "locale": "ko"})
            local_english = current_time({})

    if local_korean.output != "현재 시간은 2026년 8월 3일 월요일 오후 11시 7분입니다.":
        raise SystemExit(f"local Korean time output drifted: {local_korean.output!r}")
    if seoul_korean.output != "서울의 현재 시간은 2026년 8월 3일 월요일 오후 11시 7분입니다. (KST, Asia/Seoul)":
        raise SystemExit(f"Seoul Korean time output drifted: {seoul_korean.output!r}")
    if local_english.output != "Monday, August 03, 2026 at 11:07 PM":
        raise SystemExit(f"English time output changed: {local_english.output!r}")
    if local_korean.metadata.get("locale") != "ko" or seoul_korean.metadata.get("locale") != "ko":
        raise SystemExit("Korean time results should report locale=ko")
    if local_english.metadata.get("locale") != "en":
        raise SystemExit("English time result should report locale=en")


def main() -> None:
    test_memory_read_alias_routes()
    test_korean_time_and_weather_aliases_route_without_chat()
    test_korean_email_aliases_route_without_chat()
    test_korean_calendar_period_aliases_route_without_chat()
    test_free_text_routes_preserve_regular_whitespace()
    test_planner_truncates_compound_and_then_sentences()
    test_and_then_split_is_not_vulnerable_to_pathological_whitespace()
    test_specialist_packet_request_is_not_vulnerable_to_pathological_input()
    test_strip_trailing_politeness_suffix_only_helper()
    test_non_frozen_lazy_capture_patterns_are_bounded()
    test_entry_point_guard_bounds_frozen_zone_redos()
    test_korean_list_command_parity()
    test_planner_routes_negated_requests_to_chat()
    test_timezone_database_failures_name_recovery()
    test_current_time_formats_korean_locale_without_changing_english()
    with TemporaryDirectory(prefix="jarvis-core-") as temp:
        runtime = make_temp_runtime(Path(temp))
        cases = [
            "remember that Jarvis V2 has a separated executive core",
            "search memory for separated executive core",
            "what do you remember about separated executive core",
            "what do you remember",
            "what time is it",
            "what date is today",
            "today date",
            "what day is today",
            "what's today's date",
            "what day of the week is it",
            "what month is it",
            "what year is it",
            "what time zone am i in",
            "current timezone",
            "what timezone is seoul",
            "seoul timezone",
            "time difference between seoul and london",
            "time difference seoul london",
            "how many hours ahead is seoul from london",
            "seoul to london time difference",
            "time difference between seoul and atlantis",
            "what date is tomorrow",
            "what's tomorrow's date",
            "tomorrow date",
            "what day was yesterday",
            "what date is the day after tomorrow",
            "what day is the day after tomorrow",
            "day after tomorrow date",
            "what date was the day before yesterday",
            "what day was the day before yesterday",
            "day before yesterday date",
            "what date is in two days",
            "what day is in 3 days",
            "what date was 2 days ago",
            "what date is next week",
            "what day is next friday",
            "date next monday",
            # Real gap found live 2026-07-10: "what's the date in 10 days" (no
            # "is" before "in") fell through to the generic calculate() tool
            # with expression "the date in 10 days" (a SyntaxError, refused
            # cleanly but the wrong tool) instead of relative_date -- the
            # connector-word alternation only recognized "is/was/for/on", not
            # "in", even though the target grammar already supports "in N days".
            "what's the date in 10 days",
            "what time is it in Seoul",
            "time in Tokyo",
            "Tokyo time",
            "current time Seoul",
            "local time in London",
            "what time Seoul",
            "what time is Seoul",
            "London current time",
            "time in Atlantis",
            "daily note Jarvis V2 smoke test completed.",
        ]
        relative_date_cases = {
            "what date is tomorrow",
            "what's tomorrow's date",
            "tomorrow date",
            "what day was yesterday",
            "what date is the day after tomorrow",
            "what day is the day after tomorrow",
            "day after tomorrow date",
            "what date was the day before yesterday",
            "what day was the day before yesterday",
            "day before yesterday date",
            "what date is in two days",
            "what day is in 3 days",
            "what date was 2 days ago",
            "what date is next week",
            "what day is next friday",
            "date next monday",
            "what's the date in 10 days",
        }
        for case in cases:
            result = runtime.handle(case)
            expected_failure = case in {"time in Atlantis", "time difference between seoul and atlantis"}
            status = "ok" if result.verified else "failed"
            print(f"[{status}] {case}")
            print(result.response)
            print()
            if result.verified == expected_failure:
                raise SystemExit(f"Expected '{case}' to run.")
            if case in relative_date_cases and [tool.tool_name for tool in result.tool_results] != ["relative_date"]:
                raise SystemExit(f"Relative-date case should route to relative_date: {case!r} -> {result.tool_results}")
            if result.tool_results:
                metadata = result.tool_results[0].metadata
                assert_boundary(metadata, case)
                if case in {"what time is it in Seoul", "current time Seoul", "what time Seoul", "what time is Seoul"}:
                    if metadata.get("timezone") != "Asia/Seoul" or metadata.get("location") != "Seoul":
                        raise SystemExit(f"Seoul time missed timezone metadata: {metadata}")
                    if "Seoul:" not in result.response or "Asia/Seoul" not in result.response:
                        raise SystemExit(f"Seoul time output missed location/timezone: {result.response}")
                if case in {"what time zone am i in", "current timezone"}:
                    if metadata.get("timezone_requested") is not True or not metadata.get("timezone"):
                        raise SystemExit(f"local timezone query missed timezone metadata: {metadata}")
                    if "Local timezone:" not in result.response or "Current local time:" not in result.response:
                        raise SystemExit(f"local timezone output missed timezone label: {result.response}")
                if case in {"what timezone is seoul", "seoul timezone"}:
                    if metadata.get("timezone_requested") is not True or metadata.get("timezone") != "Asia/Seoul" or metadata.get("location") != "Seoul":
                        raise SystemExit(f"Seoul timezone query missed timezone metadata: {metadata}")
                    if "Seoul:" not in result.response or "Asia/Seoul" not in result.response:
                        raise SystemExit(f"Seoul timezone output missed location/timezone: {result.response}")
                if case in {
                    "time difference between seoul and london",
                    "time difference seoul london",
                    "how many hours ahead is seoul from london",
                    "seoul to london time difference",
                }:
                    if metadata.get("time_difference_status") != "ok":
                        raise SystemExit(f"time difference missed ok status: {metadata}")
                    if metadata.get("source") != "Seoul" or metadata.get("target") != "London":
                        raise SystemExit(f"time difference missed source/target labels: {metadata}")
                    if metadata.get("source_timezone") != "Asia/Seoul" or metadata.get("target_timezone") != "Europe/London":
                        raise SystemExit(f"time difference missed timezone metadata: {metadata}")
                    if "Seoul" not in result.response or "London" not in result.response or "ahead" not in result.response:
                        raise SystemExit(f"time difference output missed comparison: {result.response}")
                if case in {"time in Tokyo", "Tokyo time"}:
                    if metadata.get("timezone") != "Asia/Tokyo" or metadata.get("location") != "Tokyo":
                        raise SystemExit(f"Tokyo time missed timezone metadata: {metadata}")
                    if "Tokyo:" not in result.response or "Asia/Tokyo" not in result.response:
                        raise SystemExit(f"Tokyo time output missed location/timezone: {result.response}")
                if case in {"local time in London", "London current time"}:
                    if metadata.get("timezone") != "Europe/London" or metadata.get("location") != "London":
                        raise SystemExit(f"London time missed timezone metadata: {metadata}")
                    if "London:" not in result.response or "Europe/London" not in result.response:
                        raise SystemExit(f"London time output missed location/timezone: {result.response}")
                if case == "time in Atlantis":
                    if metadata.get("time_status") != "unknown_timezone" or metadata.get("timezone_lookup_supported") is not False:
                        raise SystemExit(f"unknown timezone should fail closed with safe metadata: {metadata}")
                if case == "time difference between seoul and atlantis":
                    if metadata.get("time_difference_status") != "unknown_timezone" or metadata.get("source_lookup_supported") is not True or metadata.get("target_lookup_supported") is not False:
                        raise SystemExit(f"unknown time difference target should fail closed with safe metadata: {metadata}")
                if case in {"what date is tomorrow", "what's tomorrow's date", "tomorrow date"}:
                    if metadata.get("relative_date_status") != "ok" or metadata.get("day_offset") != 1:
                        raise SystemExit(f"tomorrow relative date missed metadata: {metadata}")
                    if "Tomorrow is" not in result.response or not metadata.get("target_date") or not metadata.get("weekday"):
                        raise SystemExit(f"tomorrow relative date output missed date fields: {result.response} {metadata}")
                if case == "what day was yesterday":
                    if metadata.get("relative_date_status") != "ok" or metadata.get("day_offset") != -1:
                        raise SystemExit(f"yesterday relative date missed metadata: {metadata}")
                    if "Yesterday is" not in result.response or not metadata.get("target_date") or not metadata.get("weekday"):
                        raise SystemExit(f"yesterday relative date output missed date fields: {result.response} {metadata}")
                if case in {"what date is the day after tomorrow", "what day is the day after tomorrow", "day after tomorrow date"}:
                    if metadata.get("relative_date_status") != "ok" or metadata.get("day_offset") != 2:
                        raise SystemExit(f"day-after-tomorrow relative date missed metadata: {metadata}")
                    if "The day after tomorrow is" not in result.response or not metadata.get("target_date") or not metadata.get("weekday"):
                        raise SystemExit(f"day-after-tomorrow relative date output missed date fields: {result.response} {metadata}")
                if case in {"what date was the day before yesterday", "what day was the day before yesterday", "day before yesterday date"}:
                    if metadata.get("relative_date_status") != "ok" or metadata.get("day_offset") != -2:
                        raise SystemExit(f"day-before-yesterday relative date missed metadata: {metadata}")
                    if "The day before yesterday is" not in result.response or not metadata.get("target_date") or not metadata.get("weekday"):
                        raise SystemExit(f"day-before-yesterday relative date output missed date fields: {result.response} {metadata}")
                if case == "what date is in two days":
                    if metadata.get("relative_date_status") != "ok" or metadata.get("target") != "in two days" or metadata.get("day_offset") != 2:
                        raise SystemExit(f"in-two-days relative date missed metadata: {metadata}")
                    if "In 2 days is" not in result.response or not metadata.get("target_date") or not metadata.get("weekday"):
                        raise SystemExit(f"in-two-days relative date output missed date fields: {result.response} {metadata}")
                if case == "what day is in 3 days":
                    if metadata.get("relative_date_status") != "ok" or metadata.get("target") != "in 3 days" or metadata.get("day_offset") != 3:
                        raise SystemExit(f"in-3-days relative date missed metadata: {metadata}")
                    if "In 3 days is" not in result.response or not metadata.get("target_date") or not metadata.get("weekday"):
                        raise SystemExit(f"in-3-days relative date output missed date fields: {result.response} {metadata}")
                if case == "what date was 2 days ago":
                    if metadata.get("relative_date_status") != "ok" or metadata.get("target") != "2 days ago" or metadata.get("day_offset") != -2:
                        raise SystemExit(f"2-days-ago relative date missed metadata: {metadata}")
                    if "2 days ago is" not in result.response or not metadata.get("target_date") or not metadata.get("weekday"):
                        raise SystemExit(f"2-days-ago relative date output missed date fields: {result.response} {metadata}")
                if case == "what date is next week":
                    if metadata.get("relative_date_status") != "ok" or metadata.get("target") != "next week" or metadata.get("day_offset") != 7:
                        raise SystemExit(f"next-week relative date missed metadata: {metadata}")
                    if "One week from now is" not in result.response or not metadata.get("target_date") or not metadata.get("weekday"):
                        raise SystemExit(f"next-week relative date output missed date fields: {result.response} {metadata}")
                if case == "what day is next friday":
                    if metadata.get("relative_date_status") != "ok" or metadata.get("target") != "next friday" or metadata.get("day_offset", 0) <= 0:
                        raise SystemExit(f"next friday relative date missed metadata: {metadata}")
                    if "Next Friday is" not in result.response or not metadata.get("target_date") or metadata.get("weekday") != "Friday":
                        raise SystemExit(f"next friday relative date output missed weekday/date: {result.response} {metadata}")
                if case == "date next monday":
                    if metadata.get("relative_date_status") != "ok" or metadata.get("target") != "next monday" or metadata.get("day_offset", 0) <= 0:
                        raise SystemExit(f"next monday relative date missed metadata: {metadata}")
                    if "Next Monday is" not in result.response or not metadata.get("target_date") or metadata.get("weekday") != "Monday":
                        raise SystemExit(f"next monday relative date output missed weekday/date: {result.response} {metadata}")
                if case.startswith("remember") and not metadata.get("writes_memory"):
                    raise SystemExit("remember should declare memory writes.")
                if case.startswith("remember"):
                    handoff = metadata.get("memory_write_handoff")
                    if not isinstance(handoff, dict) or handoff.get("source") != "remember":
                        raise SystemExit(f"remember missed structured write handoff metadata: {metadata}")
                    if handoff.get("ready_for_operator") is not True or handoff.get("memory_id") != metadata.get("memory_id"):
                        raise SystemExit(f"remember handoff missed readiness/id parity: {handoff}")
                    if handoff.get("category") != "facts" or handoff.get("title") != "Jarvis V2 has a separated executive core":
                        raise SystemExit(f"remember handoff missed category/title parity: {handoff}")
                    path_text = str(metadata.get("note_path") or "")
                    path_display = metadata.get("path_display")
                    if not path_text or not Path(path_text).exists():
                        raise SystemExit(f"remember should preserve exact note path metadata: {metadata}")
                    if path_text in result.response:
                        raise SystemExit("remember should not print the raw local note path.")
                    assert_no_local_path(result.response, "remember output")
                    if not isinstance(path_display, str) or not path_display.startswith("Memory Tree/"):
                        raise SystemExit(f"remember missed vault-relative path display metadata: {metadata}")
                    if handoff.get("path_display") != path_display:
                        raise SystemExit(f"remember handoff path display should match metadata: {handoff}")
                    if f"search memory for {handoff.get('title')}" not in handoff.get("next_commands", []):
                        raise SystemExit(f"remember handoff missed search follow-up command: {handoff}")
                    boundaries = handoff.get("boundaries")
                    if not isinstance(boundaries, dict) or boundaries.get("writes_memory") is not True:
                        raise SystemExit(f"remember handoff missed write boundary: {handoff}")
                    for key in ("queues_approval", "controls_computer", "external_side_effect", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
                        if boundaries.get(key):
                            raise SystemExit(f"remember handoff should keep {key}=False: {handoff}")
                if (case.startswith("search") or case.startswith("what do you remember about")) and metadata.get("writes_memory"):
                    raise SystemExit("search memory should be read-only.")
                if case.startswith("search") or case.startswith("what do you remember about"):
                    handoff = metadata.get("memory_search_handoff")
                    if not isinstance(handoff, dict) or handoff.get("source") != "search_memory":
                        raise SystemExit(f"search memory missed structured handoff metadata: {metadata}")
                    if handoff.get("ready_for_operator") is not True or handoff.get("count") != metadata.get("count"):
                        raise SystemExit(f"search memory handoff missed readiness/count parity: {handoff}")
                    if handoff.get("query") != "separated executive core" or handoff.get("limit") != 5:
                        raise SystemExit(f"search memory handoff missed query/limit parity: {handoff}")
                    if handoff.get("memory_ids") != [1] or handoff.get("first_memory_id") != 1:
                        raise SystemExit(f"search memory handoff missed memory id parity: {handoff}")
                    rows = handoff.get("rows")
                    if not isinstance(rows, list) or rows[0].get("title") != "Jarvis V2 has a separated executive core":
                        raise SystemExit(f"search memory handoff missed compact row: {handoff}")
                    if "search memory for separated executive core" not in handoff.get("next_commands", []):
                        raise SystemExit(f"search memory handoff missed next command: {handoff}")
                    boundaries = handoff.get("boundaries")
                    if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
                        raise SystemExit(f"search memory handoff missed read-only boundary: {handoff}")
                    for key in ("writes_files", "writes_memory", "writes_notes", "queues_approval", "controls_computer", "external_side_effect", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
                        if boundaries.get(key):
                            raise SystemExit(f"search memory handoff should keep {key}=False: {handoff}")
                if case == "what do you remember":
                    handoff = metadata.get("recent_memories_handoff")
                    if not isinstance(handoff, dict) or handoff.get("source") != "recent_memories":
                        raise SystemExit(f"recent memories missed structured handoff metadata: {metadata}")
                    if handoff.get("ready_for_operator") is not True or handoff.get("count") != metadata.get("count"):
                        raise SystemExit(f"recent memories handoff missed readiness/count parity: {handoff}")
                    if handoff.get("limit") != 10 or handoff.get("memory_ids") != [1] or handoff.get("first_memory_id") != 1:
                        raise SystemExit(f"recent memories handoff missed limit/id parity: {handoff}")
                    if "what do you remember" not in handoff.get("next_commands", []):
                        raise SystemExit(f"recent memories handoff missed next command: {handoff}")
                    if handoff.get("boundaries", {}).get("read_only") is not True or handoff.get("boundaries", {}).get("writes_memory"):
                        raise SystemExit(f"recent memories handoff should stay read-only: {handoff}")
                    for key in ("writes_files", "writes_memory", "writes_notes", "queues_approval", "controls_computer", "external_side_effect", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
                        if handoff.get("boundaries", {}).get(key):
                            raise SystemExit(f"recent memories handoff should keep {key}=False: {handoff}")
                if case.startswith("daily note") and not metadata.get("writes_notes"):
                    raise SystemExit("daily note should declare note writes.")
                if case.startswith("daily note"):
                    path_text = str(metadata.get("path") or "")
                    path_display = metadata.get("path_display")
                    if not path_text or not Path(path_text).exists():
                        raise SystemExit(f"daily note should preserve exact saved path metadata: {metadata}")
                    if path_text in result.response:
                        raise SystemExit("daily note should not print the raw local note path.")
                    assert_no_local_path(result.response, "daily note output")
                    if not isinstance(path_display, str) or not path_display.startswith("Daily/"):
                        raise SystemExit(f"daily note missed vault-relative display metadata: {metadata}")
                    if f"Daily note updated: {path_display}" not in result.response:
                        raise SystemExit("daily note should print a vault-relative saved-note label.")
                    handoff = metadata.get("daily_note_handoff")
                    if not isinstance(handoff, dict) or handoff.get("source") != "write_daily_note":
                        raise SystemExit(f"daily note missed structured handoff metadata: {metadata}")
                    if handoff.get("ready_for_operator") is not True or handoff.get("path_display") != path_display:
                        raise SystemExit(f"daily note handoff missed readiness/path parity: {handoff}")
                    if handoff.get("heading") != "Quick Capture" or "Jarvis V2 smoke test completed." not in str(handoff.get("body_preview") or ""):
                        raise SystemExit(f"daily note handoff missed heading/body preview: {handoff}")
                    if "export state" not in handoff.get("next_commands", []) or "what do you remember" not in handoff.get("next_commands", []):
                        raise SystemExit(f"daily note handoff missed next commands: {handoff}")
                    boundaries = handoff.get("boundaries")
                    if not isinstance(boundaries, dict) or boundaries.get("writes_notes") is not True:
                        raise SystemExit(f"daily note handoff missed write boundary: {handoff}")
                    for key in ("queues_approval", "controls_computer", "external_side_effect", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
                        if boundaries.get(key):
                            raise SystemExit(f"daily note handoff should keep {key}=False: {handoff}")

        long = runtime.handle("remember that " + ("Jarvis bounds core memory writes safely. " * 80))
        if not long.verified:
            raise SystemExit("Long remember should run.")
        metadata = long.tool_results[0].metadata
        assert_boundary(metadata, "long remember")
        if not metadata.get("writes_memory") or not metadata.get("writes_notes"):
            raise SystemExit("Long remember should retain memory/note write metadata.")
        long_handoff = metadata.get("memory_write_handoff")
        if not isinstance(long_handoff, dict) or long_handoff.get("memory_id") != metadata.get("memory_id"):
            raise SystemExit(f"Long remember should retain write handoff id parity: {metadata}")
        if len(str(long_handoff.get("body_preview") or "")) > 220:
            raise SystemExit(f"Long remember handoff should bound body preview: {long_handoff}")
        if not str(long_handoff.get("path_display") or "").startswith("Memory Tree/"):
            raise SystemExit(f"Long remember handoff should keep safe path display: {long_handoff}")

        for target in ["/private/tmp/relative-date", "next smarchday", "in 500 days"]:
            refused_date = runtime.registry.get("relative_date").handler({"target": target})
            if refused_date.ok or refused_date.metadata.get("relative_date_status") != "unsupported_target":
                raise SystemExit(f"relative_date should refuse unsupported targets safely: {refused_date.output} {refused_date.metadata}")
            assert_boundary(refused_date.metadata, f"relative_date refusal {target}")
            assert_no_local_path(refused_date.output, f"relative_date refusal {target}")
            assert_no_local_path(refused_date.metadata, f"relative_date refusal metadata {target}")

        path_cases = []
        for value in [
            "/\x55sers/example/private/memory-category",
            "/var/folders/zc/jarvis/memory-category",
            "/tmp/jarvis-memory-category",
        ]:
            path_cases.append(("path category", {"category": value, "title": "Safe title", "body": "Safe body"}, "invalid_category", "raw_category"))
        for value in [
            "/private/tmp/memory-title",
            "/var/folders/zc/jarvis/memory-title",
            "/tmp/jarvis-memory-title",
        ]:
            path_cases.append(("path title", {"category": "facts", "title": value, "body": "Safe body"}, "invalid_title", "raw_title"))
        for value in [
            "Remember /\x55sers/example/private/memory-body",
            "Remember /var/folders/zc/jarvis/memory-body",
            "Remember /tmp/jarvis-memory-body",
        ]:
            path_cases.append(("path body", {"category": "facts", "title": "Safe title", "body": value}, "invalid_body", "raw_body"))

        for label, args, reason, raw_key in path_cases:
            bad_memory = runtime.registry.get("remember").handler(args)
            if bad_memory.ok or bad_memory.metadata.get("reason") != reason:
                raise SystemExit(f"remember {label} should include safe refusal metadata: {bad_memory.metadata}")
            raw_value = str(bad_memory.metadata.get(raw_key) or "")
            if "<local-path>" not in raw_value:
                raise SystemExit(f"remember {label} leaked local path in metadata: {bad_memory.metadata}")
            assert_no_local_path(raw_value, f"remember {label} metadata")
            if bad_memory.metadata.get("writes_files") or bad_memory.metadata.get("writes_memory") or bad_memory.metadata.get("writes_notes"):
                raise SystemExit(f"remember {label} should not write: {bad_memory.metadata}")
            if "memory_write_handoff" in bad_memory.metadata:
                raise SystemExit(f"remember {label} should not emit write handoff on refusal: {bad_memory.metadata}")
            assert_no_local_path(bad_memory.output, f"remember {label} output")

        direct_search = runtime.registry.get("search_memory").handler({"query": "separated", "limit": "bad"})
        if not direct_search.ok or direct_search.metadata.get("limit") != 5:
            raise SystemExit(f"search_memory should sanitize malformed limits: {direct_search.metadata}")
        if direct_search.metadata.get("memory_search_handoff", {}).get("limit") != 5:
            raise SystemExit(f"search_memory handoff should use sanitized limit: {direct_search.metadata}")
        direct_search_large = runtime.registry.get("search_memory").handler({"query": "separated", "limit": 999999})
        if not direct_search_large.ok or direct_search_large.metadata.get("limit") != 100:
            raise SystemExit(f"search_memory should clamp large limits: {direct_search_large.metadata}")
        empty_search = runtime.registry.get("search_memory").handler({"query": "zzzzmissing"})
        empty_handoff = empty_search.metadata.get("memory_search_handoff")
        if not empty_search.ok or not isinstance(empty_handoff, dict):
            raise SystemExit(f"empty search_memory missed handoff metadata: {empty_search.metadata}")
        if empty_handoff.get("count") != 0 or empty_handoff.get("memory_ids") != [] or empty_handoff.get("first_memory_id") is not None:
            raise SystemExit(f"empty search_memory handoff should preserve empty state: {empty_handoff}")
        if empty_handoff.get("boundaries", {}).get("read_only") is not True or empty_handoff.get("boundaries", {}).get("writes_memory"):
            raise SystemExit(f"empty search_memory handoff should stay read-only: {empty_handoff}")
        for key in ("writes_files", "writes_memory", "writes_notes", "queues_approval", "controls_computer", "external_side_effect", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
            if empty_handoff.get("boundaries", {}).get(key):
                raise SystemExit(f"empty search_memory handoff should keep {key}=False: {empty_handoff}")

        direct_recent = runtime.registry.get("recent_memories").handler({"limit": False})
        if not direct_recent.ok or direct_recent.metadata.get("limit") != 10:
            raise SystemExit(f"recent_memories should treat boolean limits as malformed defaults: {direct_recent.metadata}")
        if direct_recent.metadata.get("recent_memories_handoff", {}).get("limit") != 10:
            raise SystemExit(f"recent_memories handoff should use sanitized limit: {direct_recent.metadata}")
        empty_runtime = make_temp_runtime(Path(temp) / "empty")
        empty_recent = empty_runtime.registry.get("recent_memories").handler({})
        empty_recent_handoff = empty_recent.metadata.get("recent_memories_handoff")
        if not empty_recent.ok or not isinstance(empty_recent_handoff, dict):
            raise SystemExit(f"empty recent_memories missed handoff metadata: {empty_recent.metadata}")
        if empty_recent_handoff.get("count") != 0 or empty_recent_handoff.get("memory_ids") != [] or empty_recent_handoff.get("first_memory_id") is not None:
            raise SystemExit(f"empty recent_memories handoff should preserve empty state: {empty_recent_handoff}")
        if empty_recent_handoff.get("boundaries", {}).get("read_only") is not True or empty_recent_handoff.get("boundaries", {}).get("writes_files"):
            raise SystemExit(f"empty recent_memories handoff should stay read-only: {empty_recent_handoff}")
        for key in ("writes_files", "writes_memory", "writes_notes", "queues_approval", "controls_computer", "external_side_effect", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
            if empty_recent_handoff.get("boundaries", {}).get(key):
                raise SystemExit(f"empty recent_memories handoff should keep {key}=False: {empty_recent_handoff}")

        direct_status = runtime.registry.get("jarvis_status").handler({})
        if not direct_status.ok:
            raise SystemExit(f"jarvis_status should run read-only: {direct_status}")
        assert_boundary(direct_status.metadata, "jarvis_status")
        for expected in [
            "Jarvis status:",
            "active project: Jarvis V3",
            V3_DASHBOARD_COMMAND,
            V3_DASHBOARD_INFO_COMMAND,
            "Obsidian root: <local-path>",
        ]:
            if expected not in direct_status.output:
                raise SystemExit(f"jarvis_status missed expected safe status text {expected!r}: {direct_status.output}")
        if direct_status.metadata.get("project_name") != "Jarvis V3":
            raise SystemExit(f"jarvis_status missed project name metadata: {direct_status.metadata}")
        for path_key in ("project_root", "dashboard_launcher", "obsidian_root"):
            if direct_status.metadata.get(path_key) != "<local-path>":
                raise SystemExit(f"jarvis_status should expose redacted {path_key}: {direct_status.metadata}")
        if direct_status.metadata.get("path_metadata_redacted") is not True:
            raise SystemExit(f"jarvis_status should declare redacted path metadata: {direct_status.metadata}")
        if direct_status.metadata.get("dashboard_launcher_exists") is not True:
            raise SystemExit(f"jarvis_status missed dashboard launcher existence metadata: {direct_status.metadata}")
        assert_no_local_path(direct_status.output, "jarvis_status output")
        assert_no_local_path(direct_status.metadata, "jarvis_status metadata")

        empty_daily_note = runtime.registry.get("write_daily_note").handler({"body": ""})
        if empty_daily_note.ok:
            raise SystemExit("write_daily_note empty body should fail.")
        if empty_daily_note.metadata.get("writes_files") or empty_daily_note.metadata.get("writes_memory") or empty_daily_note.metadata.get("writes_notes"):
            raise SystemExit(f"write_daily_note empty body should not write: {empty_daily_note.metadata}")
        if "daily_note_handoff" in empty_daily_note.metadata:
            raise SystemExit(f"write_daily_note empty body should not emit write handoff: {empty_daily_note.metadata}")


if __name__ == "__main__":
    main()
