"""Structural hang guard for the planner: a deterministic adversarial corpus
run through the real ``RuleBasedPlanner().plan()`` under hard per-case
timeouts.

Why this exists (Fable checkpoint plan item A, 2026-07-10): rounds 37, 38, 43,
and 44 each found a separate catastrophic-backtracking (ReDoS-shaped) regex in
``jarvis_v2/agent/planner.py`` by hand — different pattern shapes, same failure
mode: one pathological input hangs the whole planner. Per-pattern fixes and the
Fable-authorized entry-point guard (``_normalize_planner_input``) closed the
known cases, but the file has hundreds of patterns and gains more every round
from two concurrent authors (Claude and Codex). This module converts that
whack-a-mole into a standing structural property: for every prefix x filler
shape x length in a fixed corpus, planning must complete within a hard budget.
Any future pattern that reintroduces the failure mode fails HERE, loudly, at
the next aggregate run -- instead of hanging a live Jarvis turn.

Design notes:
- Real ``RuleBasedPlanner``, no mocks: the property being asserted is about the
  real matching pipeline, entry-point guard included.
- SIGALRM per case: a genuinely-hung regex cannot be interrupted by a wall-clock
  check after the fact; the alarm aborts it mid-backtrack so the suite fails
  with a named case instead of hanging the runner.
- Deterministic corpus (no randomness): reproducible failures, stable runtime.
  Filler shapes are chosen from the empirically-found trigger classes:
  whitespace runs (rounds 35/37/43), repeated connector tokens ("to " -- the
  kakao/telegram trailing-service shape), separator/punctuation runs (round 38),
  and word soup with many separate short whitespace runs (the multi-run shape
  found at the Fable checkpoint, which whitespace-collapse alone did not fix).
- Budgets are generous (2.5s/case) so slow-CI noise does not flake the suite;
  a real catastrophic pattern blows past any budget by orders of magnitude.
"""
from __future__ import annotations

import signal
import time

from jarvis_v2.agent.planner import RuleBasedPlanner

PER_CASE_BUDGET_SECONDS = 2.5
TOTAL_BUDGET_SECONDS = 60.0

# Command lead-ins spanning the planner's major routing families. Each is a
# plausible start of a real command whose free-text tail an attacker (or an
# accidental paste) controls.
PREFIXES = [
    "",
    "send ",
    "send x ",
    "message ",
    "text fixture ",
    "remind me to ",
    "add task ",
    "add note ",
    "remember that ",
    "search my notes for ",
    "search for ",
    "find file ",
    "find contact ",
    "create goal ",
    "record decision ",
    "translate ",
    "research ",
    "what's the weather in ",
    "calculate ",
    "convert ",
    "call ",
    "don't ",
    "save skill ",
    "set preference ",
]

# (label, builder) -- builder receives a target length and returns filler text.
FILLER_SHAPES = [
    ("ws-run", lambda n: " " * n),
    ("to-repeat", lambda n: "to " * (n // 3)),
    ("word-soup", lambda n: ("word" + " " * 9) * (n // 13)),
    ("punct-run", lambda n: ",.;!? " * (n // 6)),
]

LENGTHS = [1200, 18000]

# Historical regressions, verbatim shapes from rounds 37/38/43/44 and the Fable
# checkpoint -- kept explicit so a regression names the exact prior incident.
KNOWN_BAD_CASES = {
    "round-37 add_task ws-run": "add task buy" + " " * 20000 + "milk",
    "round-38 politeness punct": "hello" + " ,.;!?" * 8000,
    "round-43 kakao ws-run": "send x" + " " * 1000 + "y",
    "checkpoint multi-run": ("a" + " " * 6) * 2000,
    "checkpoint degenerate-to": "send " + "to " * 5000,
    "round-44 transform tail": "make this" + " " * 20000 + "formal",
}


class _CaseTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ANN001 - signal signature
    raise _CaseTimeout()


def _run_case(planner: RuleBasedPlanner, label: str, text: str) -> float:
    signal.setitimer(signal.ITIMER_REAL, PER_CASE_BUDGET_SECONDS)
    start = time.monotonic()
    try:
        planner.plan(text)
    except _CaseTimeout:
        raise SystemExit(
            f"planner hang guard: case {label!r} (len {len(text)}) exceeded the "
            f"{PER_CASE_BUDGET_SECONDS}s budget -- a catastrophic-backtracking "
            "pattern has been (re)introduced in planner.py; see "
            "FABLE_CHECKPOINT_2026-07-10.md plan item A and the "
            "project_jarvis_planner_redos_class memory for the fix playbook"
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return time.monotonic() - start


def _assert_chat_fallthrough_planning_stays_fast(planner: RuleBasedPlanner) -> float:
    # Regression guard for the regex-cache latency fix (planner.py's import-time
    # re._MAXCACHE bump, Fable checkpoint plan item C1). A chat-routed turn falls
    # through the entire matcher chain, touching the planner's full ~590-710
    # distinct patterns. If those overflow the re cache and recompile each call,
    # this jumps from ~1ms to 60-123ms. The budget is deliberately generous (25ms
    # -- 20x the observed ~1ms, but 2.5x+ under the broken 60ms+) so slow CI
    # never flakes it while a removed/regressed cache fix still fails loudly.
    warm_inputs = [
        "tell me something interesting about the world please",
        "how are you doing today my friend",
        "what do you think about all of this",
    ]
    for text in warm_inputs:
        planner.plan(text)
    iterations = 25
    start = time.monotonic()
    for i in range(iterations):
        planner.plan(warm_inputs[i % len(warm_inputs)])
    per_call_ms = (time.monotonic() - start) / iterations * 1000
    if per_call_ms > 25.0:
        raise SystemExit(
            f"planner hang guard: chat-fallthrough plan() averaged {per_call_ms:.1f}ms "
            "(budget 25ms) -- the regex-cache latency fix in planner.py (re._MAXCACHE "
            "bump) has likely regressed; the planner's distinct-pattern count exceeds "
            "the re cache and every chat turn is recompiling patterns"
        )
    return per_call_ms


def main() -> None:
    previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    planner = RuleBasedPlanner()
    fallthrough_ms = _assert_chat_fallthrough_planning_stays_fast(planner)
    cases = 0
    slowest: tuple[float, str] = (0.0, "")
    suite_start = time.monotonic()
    try:
        for label, text in KNOWN_BAD_CASES.items():
            elapsed = _run_case(planner, label, text)
            cases += 1
            if elapsed > slowest[0]:
                slowest = (elapsed, label)
        for prefix in PREFIXES:
            for shape_label, build in FILLER_SHAPES:
                for length in LENGTHS:
                    label = f"{prefix.strip() or '<bare>'}/{shape_label}/{length}"
                    elapsed = _run_case(planner, label, prefix + build(length))
                    cases += 1
                    if elapsed > slowest[0]:
                        slowest = (elapsed, label)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
    total = time.monotonic() - suite_start
    if total > TOTAL_BUDGET_SECONDS:
        raise SystemExit(
            f"planner hang guard: corpus total {total:.1f}s exceeded the "
            f"{TOTAL_BUDGET_SECONDS}s budget -- no single case hung, but overall "
            "matching cost regressed; profile before raising this budget"
        )
    print(
        f"Planner hang guard passed: {cases} adversarial cases, total {total:.1f}s, "
        f"slowest {slowest[0]*1000:.0f}ms ({slowest[1]}); "
        f"chat-fallthrough plan() {fallthrough_ms:.1f}ms/call"
    )


if __name__ == "__main__":
    main()
