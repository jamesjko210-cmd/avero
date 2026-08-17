"""Smoke tests for the fun (joke) connector (mocked fetch, no network)."""

from __future__ import annotations

from typing import Any

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.config import load_config
from jarvis_v2.tools._http import HttpError
from jarvis_v2.tools import fun_connector as fc

NO_AUTHORITY_FLAGS = (
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
)


def _tools():
    return {t.name: t for t in fc.make_fun_tools(load_config())}


def _assert_joke_handoff(
    metadata: dict[str, Any],
    label: str,
    *,
    status: str,
    reason: str = "",
    joke_returned: bool = False,
    joke_chars: int = 0,
    retry_safe: bool = False,
) -> dict[str, Any]:
    handoff = metadata.get("joke_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing joke_handoff: {metadata}")
    if metadata.get("joke_handoff_ready") is not True:
        raise SystemExit(f"{label} missing joke handoff readiness flag: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} nested joke handoff_ready missing: {handoff}")
    if metadata.get("ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should be ready for operator clients: {metadata}")
    if metadata.get("joke_ready_for_operator") != handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} joke ready alias parity failed: {metadata} vs {handoff}")
    if metadata.get("state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should report no state change: {metadata}")
    if metadata.get("changed") != [] or handoff.get("changed") != []:
        raise SystemExit(f"{label} should expose an empty changed list: {metadata}")
    if metadata.get("joke_state_changed") != handoff.get("state_changed") or metadata.get("joke_changed") != handoff.get("changed"):
        raise SystemExit(f"{label} joke state alias parity failed: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False in flat and nested metadata: {metadata}")
        alias = f"joke_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} joke no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if metadata.get("content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep joke content out of handoff metadata: {metadata}")
    if metadata.get("joke_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} joke content-in-handoff alias parity failed: {metadata} vs {handoff}")
    if handoff.get("source") != "tell_joke" or handoff.get("status") != status:
        raise SystemExit(f"{label} has wrong handoff status/source: {handoff}")
    if reason and handoff.get("reason") != reason:
        raise SystemExit(f"{label} has wrong handoff reason: {handoff}")
    if handoff.get("joke_returned") is not joke_returned or handoff.get("joke_chars") != joke_chars:
        raise SystemExit(f"{label} has wrong joke availability metadata: {handoff}")
    if handoff.get("joke_returned") != metadata.get("joke_returned"):
        raise SystemExit(f"{label} should mirror joke_returned: {handoff} vs {metadata}")
    if handoff.get("joke_chars") != metadata.get("joke_chars"):
        raise SystemExit(f"{label} should mirror joke_chars: {handoff} vs {metadata}")
    if handoff.get("exception_type", "") != metadata.get("exception_type", ""):
        raise SystemExit(f"{label} should mirror exception type: {handoff} vs {metadata}")
    if handoff.get("content_in_metadata") is not False:
        raise SystemExit(f"{label} should exclude joke content from metadata: {handoff}")
    if metadata.get("joke_content_in_metadata") != handoff.get("content_in_metadata"):
        raise SystemExit(f"{label} joke content-in-metadata alias parity failed: {metadata} vs {handoff}")
    if handoff.get("retry_safe") is not retry_safe:
        raise SystemExit(f"{label} has wrong retry safety: {handoff}")
    if handoff.get("next_safe_command") != "tell me a joke":
        raise SystemExit(f"{label} has wrong next command: {handoff}")
    expected_commands = ["tell me a joke"]
    if handoff.get("next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} has wrong safe-command list: {handoff}")
    if handoff.get("next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} has wrong safe-command count: {handoff}")
    if metadata.get("joke_next_safe_command") != handoff.get("next_safe_command"):
        raise SystemExit(f"{label} joke next-command alias parity failed: {metadata} vs {handoff}")
    if metadata.get("joke_next_safe_commands") != expected_commands:
        raise SystemExit(f"{label} joke next-commands alias parity failed: {metadata} vs {handoff}")
    if metadata.get("joke_next_safe_command_count") != len(expected_commands):
        raise SystemExit(f"{label} joke next-command count alias parity failed: {metadata} vs {handoff}")

    boundaries = handoff.get("boundaries")
    expected = {
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
    if boundaries != expected:
        raise SystemExit(f"{label} has wrong boundaries: {boundaries}")
    if metadata.get("joke_boundaries") != boundaries:
        raise SystemExit(f"{label} joke boundary alias parity failed: {metadata} vs {handoff}")
    return handoff


def _assert_local_fun_handoff(metadata: dict[str, Any], label: str, *, source: str, result: Any) -> dict[str, Any]:
    handoff = metadata.get("fun_handoff")
    if not isinstance(handoff, dict):
        raise SystemExit(f"{label} missing fun_handoff: {metadata}")
    if metadata.get("fun_handoff_ready") is not True or handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} should be handoff-ready: {metadata}")
    if metadata.get("fun_ready_for_operator") is not True or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} should be operator-ready: {metadata}")
    if handoff.get("source") != source or handoff.get("status") != "ok":
        raise SystemExit(f"{label} has wrong handoff source/status: {handoff}")
    if metadata.get("state_changed") is not False or handoff.get("state_changed") is not False:
        raise SystemExit(f"{label} should not change state: {metadata}")
    if metadata.get("fun_state_changed") != handoff.get("state_changed") or metadata.get("fun_changed") != []:
        raise SystemExit(f"{label} fun state aliases wrong: {metadata} vs {handoff}")
    for key in NO_AUTHORITY_FLAGS:
        if metadata.get(key) is not False or handoff.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False: {metadata}")
        alias = f"fun_{key}"
        if metadata.get(alias) is not False or metadata.get(alias) != handoff.get(key):
            raise SystemExit(f"{label} fun no-authority alias parity failed for {alias}: {metadata} vs {handoff}")
    if metadata.get("calls_external_service") is not False:
        raise SystemExit(f"{label} should be local-only: {metadata}")
    if metadata.get("executes_side_effect") is not False or metadata.get("external_side_effect") is not False:
        raise SystemExit(f"{label} should not claim side effects: {metadata}")
    if metadata.get("reads_personal_data") is not False or metadata.get("writes_files") is not False or metadata.get("controls_computer") is not False:
        raise SystemExit(f"{label} should not read personal data, write files, or control computer: {metadata}")
    if metadata.get("content_in_handoff") is not False or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} should keep output content out of handoff flags: {metadata}")
    if metadata.get("fun_content_in_handoff") != handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} fun content alias parity failed: {metadata} vs {handoff}")
    if handoff.get("result") != str(result):
        raise SystemExit(f"{label} should preserve bounded result in handoff: {handoff}")
    if not isinstance(handoff.get("next_safe_command"), str) or not handoff["next_safe_command"]:
        raise SystemExit(f"{label} should expose a next safe command: {handoff}")
    expected_commands = [handoff["next_safe_command"]]
    if handoff.get("next_safe_commands") != expected_commands or handoff.get("next_safe_command_count") != 1:
        raise SystemExit(f"{label} has wrong next command list/count: {handoff}")
    if metadata.get("fun_next_safe_commands") != expected_commands or metadata.get("fun_next_safe_command_count") != 1:
        raise SystemExit(f"{label} next-command alias parity failed: {metadata} vs {handoff}")
    boundaries = handoff.get("boundaries")
    expected = {
        "calls_model": False,
        "calls_external_service": False,
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
    if boundaries != expected or metadata.get("fun_boundaries") != boundaries:
        raise SystemExit(f"{label} has wrong local boundaries: {metadata} vs {handoff}")
    return handoff


def test_is_local_safe() -> None:
    if _tools()["tell_joke"].risk != RiskLevel.LOCAL_SAFE:
        raise SystemExit("tell_joke should be LOCAL_SAFE")
    if _tools()["flip_coin"].risk != RiskLevel.READ_ONLY:
        raise SystemExit("flip_coin should be READ_ONLY")
    if _tools()["roll_dice"].risk != RiskLevel.READ_ONLY:
        raise SystemExit("roll_dice should be READ_ONLY")
    if _tools()["random_number"].risk != RiskLevel.READ_ONLY:
        raise SystemExit("random_number should be READ_ONLY")
    if _tools()["choose_option"].risk != RiskLevel.READ_ONLY:
        raise SystemExit("choose_option should be READ_ONLY")


def test_tells_joke() -> None:
    fc._fetch_joke = lambda: "Why don't skeletons fight? They don't have the guts."  # type: ignore
    out = _tools()["tell_joke"].handler({})
    if not out.ok or "skeletons" not in out.output:
        raise SystemExit(f"joke output wrong: {out.output}")
    if out.metadata.get("joke_returned") is not True or out.metadata.get("joke_chars") != len(out.output):
        raise SystemExit(f"joke success should preserve returned-joke metadata: {out.metadata}")
    _assert_joke_handoff(
        out.metadata,
        "joke success",
        status="ok",
        joke_returned=True,
        joke_chars=len(out.output),
    )


def test_handles_empty_joke() -> None:
    fc._fetch_joke = lambda: ""  # type: ignore
    out = _tools()["tell_joke"].handler({})
    if not out.ok or "fresh out of jokes" not in out.output:
        raise SystemExit(f"empty joke should be a clean success: {out.output}")
    if out.metadata.get("joke_returned") is not False:
        raise SystemExit(f"empty joke should preserve no-joke metadata: {out.metadata}")
    _assert_joke_handoff(
        out.metadata,
        "empty joke",
        status="empty",
        reason="empty_response",
        retry_safe=True,
    )


def _assert_joke_recovery_output(output: str, label: str) -> None:
    expected = ["joke service", "icanhazdadjoke.com", "setup check", "tell me a joke"]
    missing = [item for item in expected if item not in output]
    if missing:
        raise SystemExit(f"{label} missed recovery guidance {missing}: {output!r}")


def test_handles_error() -> None:
    def boom():
        raise RuntimeError("/\x55sers/example/private/offline")

    fc._fetch_joke = boom  # type: ignore
    out = _tools()["tell_joke"].handler({})
    if out.ok or "joke service" not in out.output:
        raise SystemExit(f"error not handled: {out.output}")
    _assert_joke_recovery_output(out.output, "joke generic error")
    if "offline" in out.output or "/\x55sers/operator" in out.output or "Joke error" in out.output:
        raise SystemExit(f"joke error should not leak raw fetch exceptions: {out.output}")
    if out.metadata.get("joke_returned") is not False:
        raise SystemExit(f"joke error should preserve no-joke metadata: {out.metadata}")
    if out.metadata.get("exception_type") != "RuntimeError":
        raise SystemExit(f"joke error should preserve bounded diagnostic metadata: {out.metadata}")
    if out.metadata.get("error_type") != "RuntimeError":
        raise SystemExit(f"joke error should preserve allowlisted diagnostic metadata: {out.metadata}")
    _assert_joke_handoff(
        out.metadata,
        "joke error",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )


def test_handles_http_5xx_with_recovery_guidance() -> None:
    def boom():
        raise HttpError(503, "/private/tmp/joke-service")

    fc._fetch_joke = boom  # type: ignore
    out = _tools()["tell_joke"].handler({})
    if out.ok:
        raise SystemExit(f"HTTP 5xx joke error should fail closed: {out.output}")
    _assert_joke_recovery_output(out.output, "joke HTTP 5xx")
    if "/private/tmp" in out.output or "joke-service" in out.output:
        raise SystemExit(f"HTTP 5xx joke error should not leak raw details: {out.output}")
    if out.metadata.get("exception_type") != "HttpError" or out.metadata.get("error_type") != "HttpError":
        raise SystemExit(f"HTTP 5xx joke error should preserve bounded diagnostic metadata: {out.metadata}")
    _assert_joke_handoff(
        out.metadata,
        "joke HTTP 5xx",
        status="unavailable",
        reason="fetch_error",
        retry_safe=True,
    )


def test_flip_coin_is_local_read_only() -> None:
    old_choice = fc.random.choice
    try:
        fc.random.choice = lambda choices: "heads"  # type: ignore
        out = _tools()["flip_coin"].handler({})
    finally:
        fc.random.choice = old_choice  # type: ignore
    if not out.ok or out.output != "Coin flip: heads.":
        raise SystemExit(f"coin output wrong: {out.output}")
    if out.metadata.get("result") != "heads" or out.metadata.get("choices") != ["heads", "tails"]:
        raise SystemExit(f"coin metadata wrong: {out.metadata}")
    _assert_local_fun_handoff(out.metadata, "coin flip", source="flip_coin", result="heads")


def test_roll_dice_is_local_read_only() -> None:
    rolls = iter([3, 5])
    old_randint = fc.random.randint
    try:
        fc.random.randint = lambda low, high: next(rolls)  # type: ignore
        out = _tools()["roll_dice"].handler({"count": 2, "sides": 6})
    finally:
        fc.random.randint = old_randint  # type: ignore
    if not out.ok or out.output != "Rolled 2d6: 3, 5 (total 8).":
        raise SystemExit(f"dice output wrong: {out.output}")
    if out.metadata.get("rolls") != [3, 5] or out.metadata.get("result") != 8 or out.metadata.get("sides") != 6 or out.metadata.get("count") != 2:
        raise SystemExit(f"dice metadata wrong: {out.metadata}")
    _assert_local_fun_handoff(out.metadata, "dice roll", source="roll_dice", result=8)


def test_roll_dice_bounds_args() -> None:
    old_randint = fc.random.randint
    try:
        fc.random.randint = lambda low, high: high  # type: ignore
        out = _tools()["roll_dice"].handler({"count": 999, "sides": 999})
    finally:
        fc.random.randint = old_randint  # type: ignore
    if out.metadata.get("count") != 12 or out.metadata.get("sides") != 100:
        raise SystemExit(f"dice should clamp count/sides: {out.metadata}")
    if out.metadata.get("result") != 1200:
        raise SystemExit(f"bounded dice result wrong: {out.metadata}")


def test_random_number_is_local_read_only() -> None:
    old_randint = fc.random.randint
    try:
        fc.random.randint = lambda low, high: high  # type: ignore
        out = _tools()["random_number"].handler({"min": 5, "max": 9})
    finally:
        fc.random.randint = old_randint  # type: ignore
    if not out.ok or out.output != "Random number (5-9): 9.":
        raise SystemExit(f"random number output wrong: {out.output}")
    if out.metadata.get("result") != 9 or out.metadata.get("min") != 5 or out.metadata.get("max") != 9:
        raise SystemExit(f"random number metadata wrong: {out.metadata}")
    _assert_local_fun_handoff(out.metadata, "random number", source="random_number", result=9)


def test_random_number_bounds_and_swaps_args() -> None:
    old_randint = fc.random.randint
    try:
        fc.random.randint = lambda low, high: low  # type: ignore
        out = _tools()["random_number"].handler({"min": 9999999, "max": -9999999})
    finally:
        fc.random.randint = old_randint  # type: ignore
    if not out.ok:
        raise SystemExit(f"bounded random number failed: {out.output}")
    if out.metadata.get("min") != -1_000_000 or out.metadata.get("max") != 1_000_000 or out.metadata.get("result") != -1_000_000:
        raise SystemExit(f"random number should clamp and swap bounds: {out.metadata}")


def test_choose_option_is_local_read_only() -> None:
    old_choice = fc.random.choice
    try:
        fc.random.choice = lambda choices: choices[-1]  # type: ignore
        out = _tools()["choose_option"].handler({"options": ["pizza", "sushi"]})
    finally:
        fc.random.choice = old_choice  # type: ignore
    if not out.ok or out.output != "I pick: sushi.":
        raise SystemExit(f"choose option output wrong: {out.output}")
    if out.metadata.get("result") != "sushi" or out.metadata.get("options") != ["pizza", "sushi"] or out.metadata.get("option_count") != 2:
        raise SystemExit(f"choose option metadata wrong: {out.metadata}")
    _assert_local_fun_handoff(out.metadata, "choose option", source="choose_option", result="sushi")


def test_choose_option_refuses_unclear_options() -> None:
    out = _tools()["choose_option"].handler({"options": ["pizza"]})
    if out.ok or out.metadata.get("reason") != "missing_options":
        raise SystemExit(f"choose option should refuse fewer than two options: {out.output} {out.metadata}")
    _assert_local_fun_handoff(out.metadata, "choose option refusal", source="choose_option", result="")


def test_planner_routes_joke() -> None:
    p = RuleBasedPlanner()
    for q in ["joke", "joke please", "funny joke", "tell joke", "tell me a joke", "tell me something funny", "make me laugh", "got any jokes"]:
        if [a.tool_name for a in p.plan(q).actions] != ["tell_joke"]:
            raise SystemExit(f"joke route missed: {q!r}")
    for q in [
        "flip a coin",
        "coin flip",
        "coin toss",
        "heads or tails",
        "pick heads or tails",
        "choose heads or tails",
        "flip heads or tails",
        "random heads or tails",
        "toss heads or tails",
        "toss a coin",
        "toss coin",
    ]:
        if [a.tool_name for a in p.plan(q).actions] != ["flip_coin"]:
            raise SystemExit(f"coin route missed: {q!r}")
    random_cases = {
        "pick a number": {},
        "pick a random number": {},
        "random number": {},
        "random 1 to 10": {"min": 1, "max": 10},
        "random number 1 to 10": {"min": 1, "max": 10},
        "pick random number 1 to 10": {"min": 1, "max": 10},
        "give me a random number 1-10": {"min": 1, "max": 10},
        "choose a number": {},
        "choose a number between 1 and 10": {"min": 1, "max": 10},
        "choose a number from 1 to 10": {"min": 1, "max": 10},
        "random number between -5 and 5": {"min": -5, "max": 5},
        "pick a number between 1 and 100": {"min": 1, "max": 100},
        "pick a number from 1 through 100": {"min": 1, "max": 100},
        "pick a number between one and ten": {"min": 1, "max": 10},
        "pick random number one to ten": {"min": 1, "max": 10},
        "choose a number from one through ten": {"min": 1, "max": 10},
        "give me a random number between negative five and five": {"min": -5, "max": 5},
        "random number between minus five and five": {"min": -5, "max": 5},
        "pick a number between twenty one and thirty": {"min": 21, "max": 30},
        "random number from one hundred to one hundred five": {"min": 100, "max": 105},
    }
    for q, expected_args in random_cases.items():
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["random_number"] or actions[0].args != expected_args:
            raise SystemExit(f"random number route missed: {q!r} -> {[(a.tool_name, a.args) for a in actions]}")
    if [a.tool_name for a in p.plan("pick one of one and ten").actions] != ["choose_option"]:
        raise SystemExit("word-number random route should not hijack explicit option picking")
    choose_cases = {
        "choose between pizza and sushi": {"options": ["pizza", "sushi"]},
        "pick between tea or coffee": {"options": ["tea", "coffee"]},
        "help me choose between red and blue": {"options": ["red", "blue"]},
        "help me decide between red and blue": {"options": ["red", "blue"]},
        "choose one between apples and oranges": {"options": ["apples", "oranges"]},
        "choose pizza or sushi": {"options": ["pizza", "sushi"]},
        "pick pizza or sushi": {"options": ["pizza", "sushi"]},
        "should I choose pizza or sushi": {"options": ["pizza", "sushi"]},
        "could you pick tea or coffee": {"options": ["tea", "coffee"]},
        "decide between pizza and sushi": {"options": ["pizza", "sushi"]},
        "pick one: pizza, sushi, ramen": {"options": ["pizza", "sushi", "ramen"]},
        "choose one: red or blue": {"options": ["red", "blue"]},
        "choose one of pizza, sushi, ramen": {"options": ["pizza", "sushi", "ramen"]},
        "pick one of tea or coffee": {"options": ["tea", "coffee"]},
        "help me choose one of red or blue": {"options": ["red", "blue"]},
        "pick: tea or coffee": {"options": ["tea", "coffee"]},
        "pick one from: pizza, sushi, ramen": {"options": ["pizza", "sushi", "ramen"]},
        "help me choose: red or blue": {"options": ["red", "blue"]},
        "choose between: pizza or sushi": {"options": ["pizza", "sushi"]},
        "choose between pizza, sushi, and ramen": {"options": ["pizza", "sushi", "ramen"]},
        "choose from red, blue, green": {"options": ["red", "blue", "green"]},
    }
    for q, expected_args in choose_cases.items():
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["choose_option"] or actions[0].args != expected_args:
            raise SystemExit(f"choose option route missed: {q!r} -> {[(a.tool_name, a.args) for a in actions]}")
    for q in ["choose pizza", "pick pizza", "decide pizza", "should I choose pizza", "choose: pizza", "pick one: pizza", "choose one of pizza", "choose one of these", "choose from pizza sushi ramen"]:
        if p.plan(q).actions:
            raise SystemExit(f"choose option route should not overclaim unclear choices: {q!r}")
    if p.plan("number 1 to 10").actions:
        raise SystemExit("random number route should not overclaim bare numeric ranges")
    dice_cases = {
        "roll a die": {},
        "roll dice": {},
        "roll dice please": {},
        "roll a pair of dice": {"count": 2},
        "roll pair of dice": {"count": 2},
        "roll 2 dice": {"count": 2},
        "roll two dice": {"count": 2},
        "roll a six sided die": {"sides": 6},
        "roll two six sided dice": {"count": 2, "sides": 6},
        "roll a d20": {"sides": 20},
        "roll d20": {"sides": 20},
        "roll two d20": {"count": 2, "sides": 20},
        "roll three twenty sided dice": {"count": 3, "sides": 20},
        "roll one hundred sided die": {"count": 1, "sides": 100},
        "roll 2d6": {"count": 2, "sides": 6},
    }
    for q, expected_args in dice_cases.items():
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["roll_dice"] or actions[0].args != expected_args:
            raise SystemExit(f"dice route missed: {q!r} -> {[(a.tool_name, a.args) for a in actions]}")


def main() -> None:
    test_is_local_safe()
    test_tells_joke()
    test_handles_empty_joke()
    test_handles_error()
    test_handles_http_5xx_with_recovery_guidance()
    test_flip_coin_is_local_read_only()
    test_roll_dice_is_local_read_only()
    test_roll_dice_bounds_args()
    test_random_number_is_local_read_only()
    test_random_number_bounds_and_swaps_args()
    test_choose_option_is_local_read_only()
    test_choose_option_refuses_unclear_options()
    test_planner_routes_joke()
    print("Fun connector smoke passed")


if __name__ == "__main__":
    main()
