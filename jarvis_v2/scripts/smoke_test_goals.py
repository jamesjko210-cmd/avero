from __future__ import annotations

import re
from tempfile import TemporaryDirectory
from pathlib import Path

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.tools.goals import _goal_handoff_metadata, _metadata_bool


class HostileRow:
    def __init__(self, marker: str):
        self.marker = marker

    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        return self.marker


def assert_safe(metadata: dict, label: str) -> None:
    for key in (
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "controls_computer",
        "reads_private_data",
    ):
        if metadata.get(key):
            raise SystemExit(f"{label} unexpectedly set {key}: {metadata}")


def assert_vault_relative_receipt(result, root: Path, prefix: str, label: str) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    response = result.response if hasattr(result, "response") else result.output
    path_text = str(metadata.get("path") or "")
    path_display = metadata.get("path_display")
    receipt_line = response.split("\n", 1)[0]
    if path_text:
        raise SystemExit(f"{label} should not retain an absolute saved path: {metadata}")
    if any(fragment in receipt_line for fragment in (str(root), "/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} receipt should not expose local temp or user paths.")
    if not isinstance(path_display, str) or not path_display.startswith(prefix):
        raise SystemExit(f"{label} missed vault-relative display metadata: {metadata}")
    if re.search(r"[0-9a-f]{32}", path_display):
        raise SystemExit(f"{label} exposed the stable store identity in display metadata: {metadata}")
    if path_display not in receipt_line:
        raise SystemExit(f"{label} should print the vault-relative saved-note label.")


def assert_goal_exact_metadata_bool() -> None:
    for value in ("true", "false", "yes", "0", 1, 0, [], ["value"], None, object()):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"goal metadata bool should fail closed for {value!r}")
    if _metadata_bool(True) is not True:
        raise SystemExit("goal metadata bool should preserve exact True")
    if _metadata_bool(False) is not False:
        raise SystemExit("goal metadata bool should preserve exact False")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("goal metadata bool should honor the explicit default")


def assert_goal_malformed_handoff_flags() -> None:
    handoff = {
        "source": "goal_status",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["goal 1 status"],
        "boundaries": {"read_only": True},
    }
    metadata = _goal_handoff_metadata("goal_status_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("goal_status_state_changed") is not False:
        raise SystemExit(f"malformed goal state_changed should fail closed: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("goal_status_content_in_handoff") is not False:
        raise SystemExit(f"malformed goal content_in_handoff should fail closed: {metadata}")


def assert_goal_next_commands_fail_closed() -> None:
    handoff = {
        "source": "goal_status",
        "state_changed": False,
        "changed": [],
        "content_in_handoff": True,
        "next_commands": {
            "show": "goal 1 status",
            "path": "/private/tmp/jarvis/goal-command",
            "number": 7,
            "blank": "",
        },
        "boundaries": {"read_only": True},
    }
    metadata = _goal_handoff_metadata("goal_status_handoff", handoff)
    if handoff.get("next_commands") != {"show": "goal 1 status", "blank": ""}:
        raise SystemExit(f"goal handoff should drop malformed/path next commands: {handoff}")
    if handoff.get("next_safe_commands") != ["goal 1 status"] or metadata.get("next_safe_commands") != ["goal 1 status"]:
        raise SystemExit(f"goal handoff next-safe commands should keep only sanitized commands: {metadata}")
    if handoff.get("hidden_next_command_count") != 2:
        raise SystemExit(f"goal handoff should count hidden malformed/path commands: {handoff}")
    if "/private/tmp" in str(handoff) or "/private/tmp" in str(metadata):
        raise SystemExit(f"goal handoff should not leak hidden local paths: {metadata}")


def assert_planner_routes_goal_phone_aliases() -> None:
    planner = RuleBasedPlanner()
    for text in (
        "show my goals",
        "what are my goals",
        "what goals do I have",
        "my goals",
        "goals please",
        "goal list please",
        "show goals",
        "show latest goals",
        # Real bug found live 2026-07-08: "active goals" phrasing fell through to
        # a "did you mean 'show my goals'?" suggestion instead of routing directly.
        "active goals",
        "show active goals",
        "list active goals",
        "my active goals",
        # Real gap found live 2026-07-10, same class as the round-39 "list my
        # open tasks" bug, surfaced via a WS4 mixed-conversation
        # remeasurement: "my active goals" alone worked, but every
        # show/list-prefixed combination of the same qualifier fell through
        # to chat.
        "show my active goals",
        "list my active goals",
        "current goals",
        "show current goals",
        "list current goals",
        "my current goals",
        "show my current goals",
        "list my current goals",
        # Real gap found live 2026-07-10 (round 43): trailing "please" broke
        # this whole exact-match set, same class as the notes fix -- see
        # smoke_test_notes.py's matching comment for the full root cause
        # (list_goals isn't in POLITE_COMMAND_RETRY_TOOLS, so the global
        # politeness-retry mechanism silently discarded a successful match).
        # Fixed with a local trailing-only strip.
        "show my active goals please",
        "projects list",
        "show my projects",
        "project list please",
        "what projects do I have",
    ):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("list_goals", {"status": "active"})]:
            raise SystemExit(f"planner missed goal/project list alias {text!r}: {plan.actions}")
    for text in ("what are my next actions", "show my next actions", "show latest next actions", "next actions please", "what should I do next today"):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("next_actions", {})]:
            raise SystemExit(f"planner missed next-actions alias {text!r}: {plan.actions}")
    # Real bug found live 2026-07-08: "next step" phrasing fell through to a
    # "did you mean 'what's my next move'?" suggestion -- only "next move" was
    # a recognized synonym, even though "next step" is at least as natural.
    for text in ("next step", "my next step", "what is my next step", "what's my next step"):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [("next_action_packet", {})]:
            raise SystemExit(f"planner missed next-step alias {text!r}: {plan.actions}")
    for text in ("new goal get fit because health", "goal get fit because health"):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [
            ("create_goal", {"title": "get fit", "purpose": "health", "horizon": ""})
        ]:
            raise SystemExit(f"planner missed goal-create alias {text!r}: {plan.actions}")
    # Real gap found live 2026-07-09: "create a goal to learn spanish" and
    # "set a goal to learn spanish" fell through because create_goal_match only
    # recognized "create/add/start/new goal <title>" with no article and no "to"
    # lead-in before the title, and "set" was not a recognized verb at all.
    for text in ("create a goal to learn spanish", "set a goal to learn spanish", "add a goal to learn spanish"):
        plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in plan.actions] != [
            ("create_goal", {"title": "learn spanish", "purpose": "", "horizon": ""})
        ]:
            raise SystemExit(f"planner missed natural goal-create phrasing {text!r}: {plan.actions}")
    # Real gap found live 2026-07-10, compound-sentence clause-bleed class
    # (same as this session's round-33 add_task/create_reminder/find_contact
    # fixes): "create goal learn spanish and then show my tasks" would have
    # written a goal literally titled "learn spanish and then show my tasks"
    # instead of "learn spanish", silently dropping the second intent.
    for text in ("create goal learn spanish and then show my tasks", "add goal learn spanish and then check weather"):
        compound_goal_plan = planner.plan(text)
        if [(a.tool_name, a.args) for a in compound_goal_plan.actions] != [
            ("create_goal", {"title": "learn spanish", "purpose": "", "horizon": ""})
        ]:
            raise SystemExit(f"planner should stop goal title at a compound-sentence boundary: {text!r} -> {compound_goal_plan.actions}")
    # Real gap found live 2026-07-10: "search for the/a goal about X" leaked
    # to a public web_lookup search instead of staying local -- no dedicated
    # keyword-search-by-content tool exists for goals (only
    # list_goals/goal_status), so the fix is a targeted exclusion in the
    # generic web-search matchers rather than a new route; this should now
    # match "find the goal about X"'s existing behavior of falling through to
    # chat instead of leaking to the web.
    for text in ("search for the goal about fitness", "search for a goal about fitness", "find the goal about fitness"):
        plan = planner.plan(text)
        if [a.tool_name for a in plan.actions] == ["web_lookup"]:
            raise SystemExit(f"goal query should not leak to a public web search: {text!r} -> {plan.actions}")


def assert_runtime_routes_goal_listing_without_suggestion_shadow() -> None:
    # Real gap found live 2026-07-09: `assert_planner_routes_goal_phone_aliases`
    # above only exercises the bare RuleBasedPlanner, so it could not catch that
    # jarvis_v2/agent/runtime.py's `_GOAL_STATUS_PRE_PLANNER` set intercepted
    # these SAME already-fixed aliases at the runtime layer BEFORE the planner
    # ever ran, turning a working one-shot command into a "Did you mean 'list
    # goals'? Send that and I'll run it." suggestion loop. This is the same
    # underlying bug as the 2026-07-08 "active goals" fix referenced above, just
    # reintroduced one layer up -- a bare-planner test can never catch a
    # runtime-level suggestion shadow, so this test drives real JarvisRuntime.
    with TemporaryDirectory(prefix="jarvis-goals-runtime-alias-") as temp:
        root = Path(temp)
        runtime = JarvisRuntime(
            JarvisConfig(
                data_dir=root,
                db_path=root / "jarvis.sqlite",
                obsidian_vault=root / "Vault",
                obsidian_root="Jarvis",
                use_model_planner=False,
            )
        )
        for text in (
            "active goals",
            "goals",
            "show goals",
            "show my goals",
            "what are my goals",
            "what goals do i have",
            "how many goals do i have",
            "how many goals",
            "count goals",
            "goal count",
            "goals count",
            "목표 몇 개",
            "목표 개수",
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [a.get("tool_name") for a in planned] != ["list_goals"]:
                raise SystemExit(f"runtime suggestion-shadow regressed for {text!r}: {planned} / {result.response!r}")
            if text in {"how many goals do i have", "how many goals", "count goals", "goal count", "goals count", "목표 몇 개", "목표 개수"}:
                if "Count: 0" not in result.response:
                    raise SystemExit(f"goal count alias should answer with a count for {text!r}: {result.response!r}")
                assert_goal_list_handoff(result.tool_results[0].metadata, text)
                assert_safe(result.tool_results[0].metadata, text)
        if runtime.store.list_goals(limit=100):
            raise SystemExit("empty goal count aliases must not create or mutate goals.")
        # "goal status" / "goals status" are deliberately still suggestion-shadowed
        # (they do not resolve as cleanly through the planner), so this asserts the
        # protective behavior stays in place rather than being over-removed.
        ambiguous_result = runtime.handle("goal status")
        ambiguous_planned = (ambiguous_result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
        if ambiguous_planned:
            raise SystemExit(f"'goal status' should stay suggestion-gated, not auto-execute: {ambiguous_planned}")


def assert_runtime_routes_goal_step_listing_aliases() -> None:
    # Real gap found 2026-07-10: the read-only `goal_status` command already
    # prints steps, but natural "list/show steps for goal 1" phrasings fell to
    # chat or a two-turn suggestion instead of the existing status surface.
    with TemporaryDirectory(prefix="jarvis-goal-step-alias-") as temp:
        root = Path(temp)
        runtime = JarvisRuntime(
            JarvisConfig(
                data_dir=root,
                db_path=root / "jarvis.sqlite",
                obsidian_vault=root / "Vault",
                obsidian_root="Jarvis",
                use_model_planner=False,
            )
        )
        runtime.handle("new goal build jarvis because trust")
        runtime.handle("add step to goal 1: verify goal-step aliases")
        for text in (
            "list steps for goal 1",
            "show steps for goal 1",
            "steps for goal 1",
            "goal 1 steps",
            "show goal 1 steps",
            "목표 1 단계 보여줘",
            "목표 1 단계 목록",
            "단계 목록 목표 1",
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [action.get("tool_name") for action in planned] != ["goal_status"]:
                raise SystemExit(f"runtime missed goal-step listing alias {text!r}: {planned} / {result.response!r}")
            if len(result.tool_results) != 1:
                raise SystemExit(f"goal-step listing alias should execute one tool for {text!r}: {result.tool_results}")
            metadata = result.tool_results[0].metadata
            if metadata.get("goal_id") != 1 or metadata.get("steps") != 1:
                raise SystemExit(f"goal-step listing alias missed goal status metadata for {text!r}: {metadata}")
            if "verify goal-step aliases" not in result.response:
                raise SystemExit(f"goal-step listing alias should show existing steps for {text!r}: {result.response!r}")
            assert_goal_status_handoff(metadata, text)
            assert_safe(metadata, text)


def assert_goal_contract(
    metadata: dict,
    handoff: dict,
    label: str,
    *,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
) -> None:
    handoff_key = next((key for key, value in metadata.items() if key.endswith("_handoff") and value is handoff), "")
    if not handoff_key:
        raise SystemExit(f"{label} could not discover goal handoff key: {metadata}")
    prefix = handoff_key.removesuffix("_handoff")
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        expected_next = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        expected_next = [str(value) for value in raw_next if str(value or "").strip()]
    expected_first = expected_next[0] if expected_next else ""
    for key, expected in (
        ("handoff_ready", True),
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} missed goal handoff {key}={expected}: handoff={handoff}")
        if key != "handoff_ready" and metadata.get(key) != expected:
            raise SystemExit(f"{label} missed goal contract {key}={expected}: metadata={metadata} handoff={handoff}")
    for key, expected in (
        (f"{handoff_key}_ready", True),
        (f"{prefix}_handoff_ready", True),
        (f"{prefix}_ready_for_operator", True),
        (f"{prefix}_state_changed", state_changed),
        (f"{prefix}_changed", changed),
        (f"{prefix}_content_in_handoff", content_in_handoff),
        (f"{prefix}_authorizes_execution", False),
        (f"{prefix}_authorizes_completion_claim", False),
        (f"{prefix}_approval_granted", False),
    ):
        if metadata.get(key) != expected:
            raise SystemExit(f"{label} missed goal prefixed alias {key}={expected}: metadata={metadata}")
    for container, container_label in ((handoff, "handoff"), (metadata, "metadata")):
        if container.get("next_safe_command") != expected_first:
            raise SystemExit(f"{label} {container_label} next_safe_command mismatch: {container}")
        if container.get("next_safe_commands") != expected_next:
            raise SystemExit(f"{label} {container_label} next_safe_commands mismatch: {container}")
        if container.get("next_safe_command_count") != len(expected_next):
            raise SystemExit(f"{label} {container_label} next_safe_command_count mismatch: {container}")
    for key, expected in (
        (f"{prefix}_next_safe_command", expected_first),
        (f"{prefix}_next_safe_commands", expected_next),
        (f"{prefix}_next_safe_command_count", len(expected_next)),
    ):
        if metadata.get(key) != expected:
            raise SystemExit(f"{label} missed goal prefixed safe-command alias {key}: metadata={metadata}")


def assert_goal_export_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("goal_export_handoff")
    if not isinstance(handoff, dict) or metadata.get("goal_export_handoff_ready") is not True:
        raise SystemExit(f"{label} missed goal export handoff: {metadata}")
    goal = handoff.get("goal") or {}
    steps = handoff.get("steps") or []
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    path_display = str(handoff.get("path_display") or "")
    if handoff.get("source") != "export_goal":
        raise SystemExit(f"{label} export handoff source diverged: {handoff}")
    assert_goal_contract(metadata, handoff, label, state_changed=True, changed=["goal_export"], content_in_handoff=True)
    if handoff.get("goal_id") != metadata.get("goal_id") or goal.get("id") != metadata.get("goal_id"):
        raise SystemExit(f"{label} export handoff goal id parity failed: {metadata}")
    if goal.get("step_count") != metadata.get("steps") or len(steps) != metadata.get("steps"):
        raise SystemExit(f"{label} export handoff step count parity failed: {metadata}")
    if goal.get("open_step_count") != metadata.get("open_steps"):
        raise SystemExit(f"{label} export handoff open step count parity failed: {metadata}")
    if path_display != metadata.get("path_display") or any(fragment in path_display for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} export handoff leaked/missed safe path: {handoff}")
    goal_id = metadata.get("goal_id")
    if next_commands.get("show_goal") != f"goal {goal_id} status":
        raise SystemExit(f"{label} export handoff missed show command: {handoff}")
    if next_commands.get("export_again") != f"export goal {goal_id} to obsidian":
        raise SystemExit(f"{label} export handoff missed export-again command: {handoff}")
    for step in steps:
        if step.get("goal_id") != goal_id:
            raise SystemExit(f"{label} export handoff step goal id parity failed: {handoff}")
        if not isinstance(step.get("body_chars"), int) or step.get("body_chars") <= 0:
            raise SystemExit(f"{label} export handoff missed step body length: {handoff}")
    for key in ["exports_goal", "writes_files", "writes_notes"]:
        if boundaries.get(key) is not True:
            raise SystemExit(f"{label} export handoff boundary {key} should be true: {handoff}")
    for key in ["writes_memory", "adds_goal", "adds_step", "changes_goal_status", "completes_goal", "completes_step", "calls_model", "executes_tools", "queues_approval", "authorizes_execution", "authorizes_completion_claim", "approval_granted", "controls_computer", "reads_private_data", "reads_personal_data", "executes_side_effect"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} export handoff boundary {key} should be false: {handoff}")


def assert_goal_status_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("goal_status_handoff")
    if not isinstance(handoff, dict) or metadata.get("goal_status_handoff_ready") is not True:
        raise SystemExit(f"{label} missed goal status handoff: {metadata}")
    goal = handoff.get("goal") or {}
    steps = handoff.get("steps") or []
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "goal_status":
        raise SystemExit(f"{label} status handoff source diverged: {handoff}")
    assert_goal_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=True)
    if handoff.get("goal_id") != metadata.get("goal_id") or goal.get("id") != metadata.get("goal_id"):
        raise SystemExit(f"{label} status handoff goal id parity failed: {metadata}")
    if goal.get("status") != metadata.get("status"):
        raise SystemExit(f"{label} status handoff goal status parity failed: {metadata}")
    if goal.get("step_count") != metadata.get("steps") or len(steps) != metadata.get("steps"):
        raise SystemExit(f"{label} status handoff step count parity failed: {metadata}")
    if goal.get("open_step_count") != metadata.get("open_steps"):
        raise SystemExit(f"{label} status handoff open step count parity failed: {metadata}")
    goal_id = metadata.get("goal_id")
    if next_commands.get("export_goal") != f"export goal {goal_id} to obsidian":
        raise SystemExit(f"{label} status handoff missed export command: {handoff}")
    if next_commands.get("add_step") != f"add step to goal {goal_id}: <next step>":
        raise SystemExit(f"{label} status handoff missed add-step command: {handoff}")
    if goal.get("next_step_id") and next_commands.get("complete_next_step") != f"complete goal step {goal.get('next_step_id')}":
        raise SystemExit(f"{label} status handoff missed next-step completion command: {handoff}")
    for step in steps:
        if step.get("goal_id") != goal_id:
            raise SystemExit(f"{label} status handoff step goal id parity failed: {handoff}")
        if not isinstance(step.get("body_chars"), int) or step.get("body_chars") <= 0:
            raise SystemExit(f"{label} status handoff missed step body length: {handoff}")
    if boundaries.get("reads_goal") is not True:
        raise SystemExit(f"{label} status handoff reads_goal boundary should be true: {handoff}")
    for key in ["writes_files", "writes_memory", "writes_notes", "exports_goal", "adds_goal", "adds_step", "changes_goal_status", "completes_goal", "completes_step", "calls_model", "executes_tools", "queues_approval", "authorizes_execution", "authorizes_completion_claim", "approval_granted", "controls_computer", "reads_private_data", "reads_personal_data", "executes_side_effect"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} status handoff boundary {key} should be false: {handoff}")


def assert_goal_list_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("goal_list_handoff")
    if not isinstance(handoff, dict) or metadata.get("goal_list_handoff_ready") is not True:
        raise SystemExit(f"{label} missed goal list handoff: {metadata}")
    goals = handoff.get("goals") or []
    goal_ids = handoff.get("goal_ids") or []
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "list_goals":
        raise SystemExit(f"{label} list handoff source diverged: {handoff}")
    assert_goal_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=bool(goals))
    if handoff.get("status") != metadata.get("status"):
        raise SystemExit(f"{label} list handoff status parity failed: {handoff}")
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} list handoff limit parity failed: {handoff}")
    if handoff.get("goal_count") != metadata.get("count") or len(goals) != metadata.get("count"):
        raise SystemExit(f"{label} list handoff count parity failed: {metadata}")
    if goal_ids != [goal.get("id") for goal in goals]:
        raise SystemExit(f"{label} list handoff goal ids diverged: {handoff}")
    if goals and handoff.get("first_goal_id") != goals[0].get("id"):
        raise SystemExit(f"{label} list handoff first goal id diverged: {handoff}")
    if not goals and handoff.get("first_goal_id") is not None:
        raise SystemExit(f"{label} empty list handoff should not set first goal id: {handoff}")
    first_goal_id = handoff.get("first_goal_id")
    expected_show = f"goal {first_goal_id} status" if first_goal_id is not None else ""
    expected_export = f"export goal {first_goal_id} to obsidian" if first_goal_id is not None else ""
    if next_commands.get("show_first_goal") != expected_show:
        raise SystemExit(f"{label} list handoff missed show-first command: {handoff}")
    if next_commands.get("export_first_goal") != expected_export:
        raise SystemExit(f"{label} list handoff missed export-first command: {handoff}")
    if next_commands.get("next_actions") != "next actions" or next_commands.get("list_active") != "list goals":
        raise SystemExit(f"{label} list handoff missed navigation commands: {handoff}")
    for goal in goals:
        for key in ["id", "title", "status", "step_count", "open_step_count", "done_step_count"]:
            if key not in goal:
                raise SystemExit(f"{label} list handoff goal missed {key}: {handoff}")
    if boundaries.get("reads_goals") is not True:
        raise SystemExit(f"{label} list handoff reads_goals boundary should be true: {handoff}")
    for key in ["writes_files", "writes_memory", "writes_notes", "exports_goal", "adds_goal", "adds_step", "changes_goal_status", "completes_goal", "completes_step", "calls_model", "executes_tools", "queues_approval", "authorizes_execution", "authorizes_completion_claim", "approval_granted", "controls_computer", "reads_private_data", "reads_personal_data", "executes_side_effect"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} list handoff boundary {key} should be false: {handoff}")


def assert_goal_next_actions_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("goal_next_actions_handoff")
    if not isinstance(handoff, dict) or metadata.get("goal_next_actions_handoff_ready") is not True:
        raise SystemExit(f"{label} missed goal next-actions handoff: {metadata}")
    actions = handoff.get("actions") or []
    goal_ids = handoff.get("goal_ids") or []
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    if handoff.get("source") != "next_actions":
        raise SystemExit(f"{label} next-actions handoff source diverged: {handoff}")
    assert_goal_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=bool(actions))
    if handoff.get("limit") != metadata.get("limit"):
        raise SystemExit(f"{label} next-actions handoff limit parity failed: {handoff}")
    if handoff.get("action_count") != metadata.get("count") or len(actions) != metadata.get("count"):
        raise SystemExit(f"{label} next-actions handoff count parity failed: {metadata}")
    if goal_ids != [action.get("goal_id") for action in actions]:
        raise SystemExit(f"{label} next-actions handoff goal ids diverged: {handoff}")
    if actions:
        first = actions[0]
        if handoff.get("first_goal_id") != first.get("goal_id") or handoff.get("first_step_id") != first.get("step_id"):
            raise SystemExit(f"{label} next-actions handoff first ids diverged: {handoff}")
        if next_commands.get("show_first_goal") != first.get("show_goal_command"):
            raise SystemExit(f"{label} next-actions handoff missed show-first command: {handoff}")
        if next_commands.get("complete_first_step") != first.get("complete_step_command"):
            raise SystemExit(f"{label} next-actions handoff missed complete-first command: {handoff}")
        if next_commands.get("add_step_to_first_goal") != first.get("add_step_command"):
            raise SystemExit(f"{label} next-actions handoff missed add-step command: {handoff}")
    else:
        if handoff.get("first_goal_id") is not None or handoff.get("first_step_id") is not None:
            raise SystemExit(f"{label} empty next-actions handoff should not set first ids: {handoff}")
        for key in ["show_first_goal", "complete_first_step", "add_step_to_first_goal"]:
            if next_commands.get(key) != "":
                raise SystemExit(f"{label} empty next-actions command {key} should be blank: {handoff}")
    if next_commands.get("list_goals") != "list goals" or next_commands.get("create_goal") != "create goal <title> because <purpose>":
        raise SystemExit(f"{label} next-actions handoff missed navigation commands: {handoff}")
    for action in actions:
        for key in ["goal_id", "goal_title", "goal_status", "action", "has_open_step", "show_goal_command", "complete_step_command", "add_step_command"]:
            if key not in action:
                raise SystemExit(f"{label} next-actions action missed {key}: {handoff}")
    for key in ["reads_goals", "reads_steps"]:
        if boundaries.get(key) is not True:
            raise SystemExit(f"{label} next-actions handoff boundary {key} should be true: {handoff}")
    for key in ["writes_files", "writes_memory", "writes_notes", "exports_goal", "adds_goal", "adds_step", "changes_goal_status", "completes_goal", "completes_step", "calls_model", "executes_tools", "queues_approval", "authorizes_execution", "authorizes_completion_claim", "approval_granted", "controls_computer", "reads_private_data", "reads_personal_data", "executes_side_effect"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} next-actions handoff boundary {key} should be false: {handoff}")


def assert_goal_mutation_handoff(metadata: dict, label: str, *, source: str, mutation: str, changed: list[str]) -> None:
    handoff = metadata.get("goal_mutation_handoff")
    if not isinstance(handoff, dict) or metadata.get("goal_mutation_handoff_ready") is not True:
        raise SystemExit(f"{label} missed goal mutation handoff: {metadata}")
    goal = handoff.get("goal") or {}
    steps = handoff.get("steps") or []
    next_commands = handoff.get("next_commands") or {}
    boundaries = handoff.get("boundaries") or {}
    path_display = str(handoff.get("path_display") or "")
    if handoff.get("source") != source or handoff.get("mutation") != mutation:
        raise SystemExit(f"{label} mutation handoff source/mutation diverged: {handoff}")
    assert_goal_contract(metadata, handoff, label, state_changed=True, changed=changed, content_in_handoff=True)
    if handoff.get("goal_id") != metadata.get("goal_id") or goal.get("id") != metadata.get("goal_id"):
        raise SystemExit(f"{label} mutation handoff goal id parity failed: {metadata}")
    if metadata.get("step_id") is not None and handoff.get("step_id") != metadata.get("step_id"):
        raise SystemExit(f"{label} mutation handoff step id parity failed: {metadata}")
    if handoff.get("changed") != changed:
        raise SystemExit(f"{label} mutation handoff changed fields diverged: {handoff}")
    if path_display != metadata.get("path_display") or any(fragment in path_display for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} mutation handoff leaked/missed safe path: {handoff}")
    if len(steps) != goal.get("step_count"):
        raise SystemExit(f"{label} mutation handoff step count diverged: {handoff}")
    goal_id = metadata.get("goal_id")
    if next_commands.get("show_goal") != f"goal {goal_id} status":
        raise SystemExit(f"{label} mutation handoff missed show command: {handoff}")
    if next_commands.get("export_goal") != f"export goal {goal_id} to obsidian":
        raise SystemExit(f"{label} mutation handoff missed export command: {handoff}")
    for step in steps:
        if step.get("goal_id") != goal_id:
            raise SystemExit(f"{label} mutation handoff step goal id parity failed: {handoff}")
        if not isinstance(step.get("body_chars"), int) or step.get("body_chars") <= 0:
            raise SystemExit(f"{label} mutation handoff missed step body length: {handoff}")
    for key in ["writes_files", "writes_memory", "writes_notes"]:
        if boundaries.get(key) is not True:
            raise SystemExit(f"{label} mutation handoff boundary {key} should be true: {handoff}")
    expected_true = {
        "adds_goal": mutation == "goal_create",
        "adds_step": mutation == "step_create",
        "changes_goal_status": mutation == "status_update",
        "completes_goal": mutation == "status_update" and metadata.get("status") == "done",
        "completes_step": mutation == "step_completion",
    }
    for key, expected in expected_true.items():
        if boundaries.get(key) is not expected:
            raise SystemExit(f"{label} mutation handoff boundary {key} should be {expected}: {handoff}")
    for key in ["exports_goal", "calls_model", "executes_tools", "queues_approval", "authorizes_execution", "authorizes_completion_claim", "approval_granted", "controls_computer", "reads_private_data", "reads_personal_data", "executes_side_effect"]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} mutation handoff boundary {key} should be false: {handoff}")


def assert_goal_refusal_handoff(metadata: dict, label: str, *, source: str, mutation: str, reason: str) -> None:
    handoff = metadata.get("goal_refusal_handoff")
    if not isinstance(handoff, dict) or metadata.get("goal_refusal_handoff_ready") is not True:
        raise SystemExit(f"{label} missed goal refusal handoff: {metadata}")
    boundaries = handoff.get("boundaries") or {}
    next_commands = handoff.get("next_commands") or {}
    if metadata.get("goal_mutation_handoff_ready") is not False:
        raise SystemExit(f"{label} refusal should explicitly mark mutation handoff not ready: {metadata}")
    if handoff.get("source") != source or handoff.get("mutation") != mutation or handoff.get("reason") != reason:
        raise SystemExit(f"{label} refusal handoff source/mutation/reason diverged: {handoff}")
    assert_goal_contract(
        metadata,
        handoff,
        label,
        state_changed=False,
        changed=[],
        content_in_handoff="raw_field" in handoff,
    )
    if handoff.get("changed") != [] or handoff.get("refused") is not True:
        raise SystemExit(f"{label} refusal handoff should report no changed fields: {handoff}")
    if handoff.get("goal_id") != metadata.get("goal_id") or handoff.get("step_id") != metadata.get("step_id"):
        raise SystemExit(f"{label} refusal handoff id parity failed: {metadata}")
    raw_field = handoff.get("raw_field")
    if raw_field and handoff.get("raw_value") != metadata.get(f"raw_{raw_field}"):
        raise SystemExit(f"{label} refusal handoff raw value parity failed: {metadata}")
    if "list_goals" not in next_commands or "next_actions" not in next_commands or "retry" not in next_commands:
        raise SystemExit(f"{label} refusal handoff missed recovery commands: {handoff}")
    for key in [
        "writes_files",
        "writes_memory",
        "writes_notes",
        "adds_goal",
        "adds_step",
        "changes_goal_status",
        "completes_goal",
        "completes_step",
        "exports_goal",
        "calls_model",
        "executes_tools",
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "controls_computer",
        "reads_private_data",
        "reads_personal_data",
        "executes_side_effect",
    ]:
        if boundaries.get(key) is not False:
            raise SystemExit(f"{label} refusal handoff boundary {key} should be false: {handoff}")


def assert_goal_readonly_reports_tolerate_malformed_rows() -> None:
    marker = "GOAL_HOSTILE_ROW_SHOULD_NOT_LEAK"
    with TemporaryDirectory(prefix="jarvis-goals-hostile-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)
        created = runtime.registry.get("create_goal").handler({
            "title": "Build safe goal handoffs",
            "purpose": "keep planning readable",
        })
        if not created.ok:
            raise SystemExit(f"hostile-row fixture goal creation failed: {created}")
        stepped = runtime.registry.get("add_goal_step").handler({"goal_id": 1, "body": "review malformed goal rows"})
        if not stepped.ok:
            raise SystemExit(f"hostile-row fixture step creation failed: {stepped}")

        readable_goals = runtime.store.list_goals(status="active", limit=100)
        readable_steps = runtime.store.list_goal_steps(1)

        def hostile_goals(status=None, limit=25):  # noqa: ANN001
            return [HostileRow(marker), *readable_goals]

        def hostile_steps(goal_id):  # noqa: ANN001
            return [HostileRow(marker), *readable_steps]

        runtime.store.list_goals = hostile_goals  # type: ignore[method-assign]
        runtime.store.list_goal_steps = hostile_steps  # type: ignore[method-assign]

        listed = runtime.registry.get("list_goals").handler({"status": "active", "limit": 5})
        list_metadata = listed.metadata
        if not listed.ok:
            raise SystemExit(f"list_goals should tolerate malformed goal rows: {listed}")
        if list_metadata.get("count") != 1 or list_metadata.get("readable_goal_rows") != 1:
            raise SystemExit(f"list_goals should preserve readable goal count: {list_metadata}")
        if list_metadata.get("unreadable_goal_rows") != 1 or list_metadata.get("unreadable_goal_step_rows") != 1:
            raise SystemExit(f"list_goals missed unreadable goal/step counters: {list_metadata}")
        if "Build safe goal handoffs" not in listed.output:
            raise SystemExit("list_goals should preserve readable goal titles behind malformed rows.")
        for expected in ("unreadable goal row(s) hidden for safety", "unreadable goal step row(s) hidden for safety"):
            if expected not in listed.output:
                raise SystemExit(f"list_goals missed safe malformed-row diagnostic: {expected}")
        if marker in listed.output or marker in str(list_metadata):
            raise SystemExit("list_goals leaked raw malformed row text.")
        assert_goal_list_handoff(list_metadata, "malformed-row list_goals")
        assert_safe(list_metadata, "malformed-row list_goals")

        next_actions = runtime.registry.get("next_actions").handler({"limit": 5})
        next_metadata = next_actions.metadata
        if not next_actions.ok:
            raise SystemExit(f"next_actions should tolerate malformed goal rows: {next_actions}")
        if next_metadata.get("count") != 1 or next_metadata.get("readable_goal_rows") != 1:
            raise SystemExit(f"next_actions should preserve readable action count: {next_metadata}")
        if next_metadata.get("unreadable_goal_rows") != 1 or next_metadata.get("unreadable_goal_step_rows") != 1:
            raise SystemExit(f"next_actions missed unreadable goal/step counters: {next_metadata}")
        if "review malformed goal rows" not in next_actions.output:
            raise SystemExit("next_actions should preserve readable next steps behind malformed rows.")
        for expected in ("unreadable active goal row(s) hidden for safety", "unreadable goal step row(s) hidden for safety"):
            if expected not in next_actions.output:
                raise SystemExit(f"next_actions missed safe malformed-row diagnostic: {expected}")
        if marker in next_actions.output or marker in str(next_metadata):
            raise SystemExit("next_actions leaked raw malformed row text.")
        assert_goal_next_actions_handoff(next_metadata, "malformed-row next_actions")
        assert_safe(next_metadata, "malformed-row next_actions")


def assert_missing_goal_and_step_id_recovery() -> None:
    with TemporaryDirectory(prefix="jarvis-missing-goal-recovery-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)
        private_step = "PRIVATE_GOAL_STEP_SHOULD_NOT_APPEAR"
        cases = (
            (
                "goal_status",
                {"goal_id": 9999},
                "goal_read",
                ["list goals", "goal <correct goal id> status"],
            ),
            (
                "add_goal_step",
                {"goal_id": 9999, "body": private_step},
                "step_create",
                [
                    "list goals",
                    "goal <correct goal id> status",
                    "add step to goal <correct goal id>: <next step>",
                ],
            ),
            (
                "set_goal_status",
                {"goal_id": 9999, "status": "done"},
                "status_update",
                [
                    "list goals",
                    "goal <correct goal id> status",
                    "goal <correct goal id> <active|paused|done|dropped>",
                ],
            ),
            (
                "export_goal",
                {"goal_id": 9999},
                "goal_export",
                [
                    "list goals",
                    "goal <correct goal id> status",
                    "export goal <correct goal id> to obsidian",
                ],
            ),
        )
        for tool_name, args, mutation, expected_commands in cases:
            result = runtime.registry.get(tool_name).handler(args)
            if result.ok or result.metadata.get("reason") != "missing_goal":
                raise SystemExit(f"{tool_name} should fail closed for a missing goal: {result}")
            for token in ("Run `list goals`", "refresh goal IDs", "goal <correct goal id> status", "normal policy"):
                if token not in result.output:
                    raise SystemExit(f"{tool_name} missing-goal output missed {token!r}: {result.output}")
            metadata = result.metadata
            if metadata.get("recovery_commands") != expected_commands:
                raise SystemExit(f"{tool_name} missing-goal recovery order drifted: {metadata}")
            for key in ("retry_requires_goal_refresh", "retry_requires_corrected_id", "recovery_commands_require_normal_policy"):
                if metadata.get(key) is not True:
                    raise SystemExit(f"{tool_name} missing-goal recovery missed {key}: {metadata}")
            for key in (
                "authorizes_retry",
                "authorizes_goal_mutation",
                "writes_files",
                "writes_memory",
                "writes_notes",
                "authorizes_execution",
                "authorizes_completion_claim",
            ):
                if metadata.get(key):
                    raise SystemExit(f"{tool_name} missing-goal recovery unexpectedly set {key}: {metadata}")
            assert_goal_refusal_handoff(
                metadata,
                f"{tool_name} missing goal",
                source=tool_name,
                mutation=mutation,
                reason="missing_goal",
            )
            assert_safe(metadata, f"{tool_name} missing goal")
            if private_step in f"{result.output}\n{metadata}":
                raise SystemExit(f"{tool_name} missing-goal recovery leaked supplied step text")

        missing_step = runtime.registry.get("complete_goal_step").handler({"step_id": 9999})
        if missing_step.ok or missing_step.metadata.get("reason") != "missing_step":
            raise SystemExit(f"complete_goal_step should fail closed for a missing step: {missing_step}")
        for token in ("Run `next actions`", "refresh goal-step IDs", "complete goal step <correct step id>", "normal policy"):
            if token not in missing_step.output:
                raise SystemExit(f"missing goal-step output missed {token!r}: {missing_step.output}")
        step_metadata = missing_step.metadata
        if step_metadata.get("recovery_commands") != ["next actions", "complete goal step <correct step id>"]:
            raise SystemExit(f"missing goal-step recovery order drifted: {step_metadata}")
        for key in ("retry_requires_goal_step_refresh", "retry_requires_corrected_id", "recovery_commands_require_normal_policy"):
            if step_metadata.get(key) is not True:
                raise SystemExit(f"missing goal-step recovery missed {key}: {step_metadata}")
        for key in ("authorizes_retry", "authorizes_goal_mutation", "writes_files", "writes_memory", "writes_notes"):
            if step_metadata.get(key):
                raise SystemExit(f"missing goal-step recovery unexpectedly set {key}: {step_metadata}")
        assert_goal_refusal_handoff(
            step_metadata,
            "missing goal step",
            source="complete_goal_step",
            mutation="step_completion",
            reason="missing_step",
        )
        assert_safe(step_metadata, "missing goal step")


def main() -> None:
    assert_missing_goal_and_step_id_recovery()
    assert_goal_exact_metadata_bool()
    assert_goal_malformed_handoff_flags()
    assert_goal_next_commands_fail_closed()
    assert_planner_routes_goal_phone_aliases()
    assert_runtime_routes_goal_listing_without_suggestion_shadow()
    assert_runtime_routes_goal_step_listing_aliases()
    assert_goal_readonly_reports_tolerate_malformed_rows()
    with TemporaryDirectory(prefix="jarvis-goals-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)

        cases = [
            "create goal Build Jarvis V2 because make a personal assistant with memory and tools by this month",
            "add step to goal 1: define the brain loop",
            "add step to goal 1: wire goals into Obsidian",
            "complete goal step 1",
            "goal 1 status",
            "next actions",
            "list goals",
            "export goal 1 to obsidian",
            "goal 1 done",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1000])
            print()
            if case == "create goal Build Jarvis V2 because make a personal assistant with memory and tools by this month":
                metadata = result.tool_results[0].metadata
                if metadata.get("goal_id") != 1 or metadata.get("writes_files") is not True:
                    raise SystemExit("Create goal missed durable goal metadata.")
                if metadata.get("reads_private_data") is not False or metadata.get("controls_computer") is not False:
                    raise SystemExit("Create goal missed safety metadata.")
                if not metadata.get("writes_notes") or not metadata.get("writes_memory"):
                    raise SystemExit("Create goal missed note/memory write metadata.")
                assert_goal_mutation_handoff(metadata, case, source="create_goal", mutation="goal_create", changed=["goal"])
            if case == "add step to goal 1: define the brain loop":
                assert_goal_mutation_handoff(result.tool_results[0].metadata, case, source="add_goal_step", mutation="step_create", changed=["steps"])
            if case == "add step to goal 1: wire goals into Obsidian":
                assert_goal_mutation_handoff(result.tool_results[0].metadata, case, source="add_goal_step", mutation="step_create", changed=["steps"])
            if case == "complete goal step 1":
                assert_goal_mutation_handoff(result.tool_results[0].metadata, case, source="complete_goal_step", mutation="step_completion", changed=["steps"])
            if case == "goal 1 status":
                metadata = result.tool_results[0].metadata
                if metadata.get("steps") != 2 or metadata.get("writes_files") is not False:
                    raise SystemExit("Goal status missed read-only metadata.")
                assert_goal_status_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "next actions":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 10 or metadata.get("count") != 1:
                    raise SystemExit("Next actions missed limit/count metadata.")
                assert_goal_next_actions_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "list goals":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 25 or metadata.get("count") != 1 or metadata.get("status") != "active":
                    raise SystemExit("List goals missed default limit/count/status metadata.")
                assert_goal_list_handoff(metadata, case)
                assert_safe(metadata, case)
            if case == "export goal 1 to obsidian":
                metadata = result.tool_results[0].metadata
                if metadata.get("goal_id") != 1 or metadata.get("writes_files") is not True:
                    raise SystemExit(f"Runtime export_goal missed durable metadata: {metadata}")
                if not metadata.get("writes_notes") or metadata.get("writes_memory") is not False:
                    raise SystemExit(f"Runtime export_goal exposed incorrect note/memory write metadata: {metadata}")
                if len(str(metadata.get("content_sha256") or "")) != 64 or len(str(metadata.get("source_revision") or "")) != 64:
                    raise SystemExit(f"Runtime export_goal missed publication evidence: {metadata}")
                assert_goal_export_handoff(metadata, case)
                assert_vault_relative_receipt(result, root, "Projects/", case)
            if case == "goal 1 done":
                assert_goal_mutation_handoff(result.tool_results[0].metadata, case, source="set_goal_status", mutation="status_update", changed=["status"])

        store_identity = runtime.store.get_store_identity()
        goal_note = (
            root
            / "Vault"
            / "Jarvis"
            / "Projects"
            / f"Build Jarvis V2 [1-{store_identity}].md"
        )
        if not goal_note.exists():
            raise SystemExit(f"Expected goal note missing: {goal_note}")
        text = goal_note.read_text(encoding="utf-8")
        if "define the brain loop" not in text or "wire goals into Obsidian" not in text:
            raise SystemExit("Goal note did not contain expected steps.")

        direct_list = runtime.registry.get("list_goals").handler({"limit": "bad"})
        if not direct_list.ok or direct_list.metadata.get("limit") != 25 or direct_list.metadata.get("raw_limit") != "bad":
            raise SystemExit("list_goals did not sanitize a bad limit.")
        if direct_list.metadata.get("writes_files") is not False or direct_list.metadata.get("reads_private_data") is not False:
            raise SystemExit("list_goals missed read-only safety metadata.")
        assert_goal_list_handoff(direct_list.metadata, "direct list goals")
        assert_safe(direct_list.metadata, "direct list goals")
        direct_list_bool_limit = runtime.registry.get("list_goals").handler({"limit": False})
        if not direct_list_bool_limit.ok or direct_list_bool_limit.metadata.get("limit") != 25 or direct_list_bool_limit.metadata.get("raw_limit") != "False":
            raise SystemExit(f"list_goals should treat boolean limits as malformed defaults: {direct_list_bool_limit.metadata}")
        if direct_list_bool_limit.metadata.get("writes_files") or direct_list_bool_limit.metadata.get("writes_memory"):
            raise SystemExit(f"list_goals boolean limit should stay read-only: {direct_list_bool_limit.metadata}")
        assert_goal_list_handoff(direct_list_bool_limit.metadata, "direct list goals bool limit")
        assert_safe(direct_list_bool_limit.metadata, "direct list goals bool limit")
        direct_list_long_limit = runtime.registry.get("list_goals").handler({"limit": "l" * 200})
        if direct_list_long_limit.metadata.get("raw_limit") != ("l" * 79 + "…"):
            raise SystemExit(f"list_goals did not bound raw bad limit metadata: {direct_list_long_limit.metadata}")
        assert_goal_list_handoff(direct_list_long_limit.metadata, "direct list goals long limit")
        for path_limit in (
            "/\x55sers/example/private/goal-limit",
            "/var/folders/zc/jarvis/goal-limit",
            "/tmp/jarvis/goal-limit",
        ):
            path_bad_list_limit = runtime.registry.get("list_goals").handler({"limit": path_limit})
            if path_bad_list_limit.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"list_goals should redact path-shaped bad limits: {path_bad_list_limit.metadata}")
            assert_goal_list_handoff(path_bad_list_limit.metadata, "path bad list goals limit")

        empty_active_list = runtime.registry.get("list_goals").handler({"status": "active", "limit": 5})
        if not empty_active_list.ok or empty_active_list.metadata.get("count") != 0 or empty_active_list.metadata.get("status") != "active":
            raise SystemExit(f"list_goals active filter should produce an empty active handoff after goal completion: {empty_active_list.metadata}")
        assert_goal_list_handoff(empty_active_list.metadata, "empty active list goals")
        assert_safe(empty_active_list.metadata, "empty active list goals")

        bad_list_status = runtime.registry.get("list_goals").handler({"status": "archived"})
        if bad_list_status.ok or "Goal status must be active" not in bad_list_status.output:
            raise SystemExit("list_goals should reject unsupported status filters.")
        if bad_list_status.metadata.get("reason") != "bad_status" or bad_list_status.metadata.get("raw_status") != "archived":
            raise SystemExit(f"list_goals bad status should preserve bounded raw status: {bad_list_status.metadata}")
        if bad_list_status.metadata.get("writes_files") or bad_list_status.metadata.get("queues_approval"):
            raise SystemExit("list_goals bad status should stay read-only and approval-free.")
        assert_safe(bad_list_status.metadata, "bad list goals status")
        for path_status in (
            "/private/tmp/jarvis-goal-status",
            "/var/folders/zc/jarvis/goal-status",
            "/tmp/jarvis/goal-status",
        ):
            path_bad_list_status = runtime.registry.get("list_goals").handler({"status": path_status})
            if path_bad_list_status.ok or path_bad_list_status.metadata.get("raw_status") != "<local-path>":
                raise SystemExit(f"list_goals should redact path-shaped bad statuses: {path_bad_list_status.metadata}")
            assert_safe(path_bad_list_status.metadata, "path bad list goals status")

        direct_next = runtime.registry.get("next_actions").handler({"limit": "bad"})
        if not direct_next.ok or direct_next.metadata.get("limit") != 10 or direct_next.metadata.get("raw_limit") != "bad":
            raise SystemExit("next_actions did not sanitize a bad limit.")
        assert_goal_next_actions_handoff(direct_next.metadata, "direct next actions")
        assert_safe(direct_next.metadata, "direct next actions")
        direct_next_bool = runtime.registry.get("next_actions").handler({"limit": True})
        if not direct_next_bool.ok or direct_next_bool.metadata.get("limit") != 10 or direct_next_bool.metadata.get("raw_limit") != "True":
            raise SystemExit(f"next_actions should treat boolean limits as malformed defaults: {direct_next_bool.metadata}")
        assert_goal_next_actions_handoff(direct_next_bool.metadata, "direct next actions bool limit")
        assert_safe(direct_next_bool.metadata, "direct next actions bool limit")
        for path_limit in (
            "/private/tmp/jarvis-next-limit",
            "/var/folders/zc/jarvis/next-limit",
            "/tmp/jarvis/next-limit",
        ):
            path_bad_next_limit = runtime.registry.get("next_actions").handler({"limit": path_limit})
            if not path_bad_next_limit.ok or path_bad_next_limit.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"next_actions should redact path-shaped bad limits: {path_bad_next_limit.metadata}")
            assert_goal_next_actions_handoff(path_bad_next_limit.metadata, "path bad next actions limit")
            assert_safe(path_bad_next_limit.metadata, "path bad next actions limit")
        direct_next_large = runtime.registry.get("next_actions").handler({"limit": 999999})
        if not direct_next_large.ok or direct_next_large.metadata.get("limit") != 200:
            raise SystemExit("next_actions did not clamp a large limit.")
        assert_goal_next_actions_handoff(direct_next_large.metadata, "direct next actions large")
        assert_safe(direct_next_large.metadata, "direct next actions large")

        bad_goal = runtime.registry.get("goal_status").handler({"goal_id": "bad"})
        if bad_goal.ok or "must be a number" not in bad_goal.output:
            raise SystemExit("goal_status did not handle a bad id cleanly.")
        if bad_goal.metadata.get("raw_id") != "bad":
            raise SystemExit(f"goal_status should preserve bounded raw bad id metadata: {bad_goal.metadata}")
        assert_safe(bad_goal.metadata, "bad goal status")
        bool_goal = runtime.registry.get("goal_status").handler({"goal_id": True})
        if bool_goal.ok or "must be a number" not in bool_goal.output:
            raise SystemExit("goal_status should reject boolean ids instead of coercing them to goal ids.")
        if bool_goal.metadata.get("raw_id") != "True":
            raise SystemExit(f"goal_status should preserve boolean raw id metadata: {bool_goal.metadata}")
        assert_safe(bool_goal.metadata, "bool goal status")
        for path_id in (
            "/\x55sers/example/private/goal-id",
            "/var/folders/zc/jarvis/goal-id",
            "/tmp/jarvis/goal-id",
        ):
            path_bad_goal = runtime.registry.get("goal_status").handler({"goal_id": path_id})
            if path_bad_goal.ok or path_bad_goal.metadata.get("raw_id") != "<local-path>":
                raise SystemExit(f"goal_status should redact path-shaped bad ids: {path_bad_goal.metadata}")
            assert_safe(path_bad_goal.metadata, "path bad goal status")

        for bad_numeric_id in (0, -1):
            bad_numeric_goal = runtime.registry.get("goal_status").handler({"goal_id": bad_numeric_id})
            if bad_numeric_goal.ok or "positive number" not in bad_numeric_goal.output:
                raise SystemExit("goal_status should reject non-positive ids before lookup.")
            if bad_numeric_goal.metadata.get("raw_id") != str(bad_numeric_id):
                raise SystemExit(f"goal_status should preserve non-positive raw id metadata: {bad_numeric_goal.metadata}")
            if bad_numeric_goal.metadata.get("writes_files") or bad_numeric_goal.metadata.get("writes_memory") or bad_numeric_goal.metadata.get("writes_notes"):
                raise SystemExit(f"goal_status non-positive id should not write: {bad_numeric_goal.metadata}")
            assert_safe(bad_numeric_goal.metadata, "bad numeric goal status")

        direct_goal_status = runtime.registry.get("goal_status").handler({"goal_id": 1})
        if not direct_goal_status.ok or direct_goal_status.metadata.get("goal_id") != 1:
            raise SystemExit(f"direct goal_status should inspect goal #1: {direct_goal_status.metadata}")
        assert_goal_status_handoff(direct_goal_status.metadata, "direct goal status")
        assert_safe(direct_goal_status.metadata, "direct goal status")

        bad_step = runtime.registry.get("complete_goal_step").handler({"step_id": "bad"})
        if bad_step.ok or "must be a number" not in bad_step.output:
            raise SystemExit("complete_goal_step did not handle a bad id cleanly.")
        if bad_step.metadata.get("raw_id") != "bad":
            raise SystemExit(f"complete_goal_step should preserve bounded raw bad id metadata: {bad_step.metadata}")
        assert_goal_refusal_handoff(bad_step.metadata, "bad complete goal step", source="complete_goal_step", mutation="step_completion", reason="bad_step_id")
        assert_safe(bad_step.metadata, "bad complete goal step")
        for path_id in (
            "/private/tmp/jarvis-step-id",
            "/var/folders/zc/jarvis/step-id",
            "/tmp/jarvis/step-id",
        ):
            path_bad_step = runtime.registry.get("complete_goal_step").handler({"step_id": path_id})
            if path_bad_step.ok or path_bad_step.metadata.get("raw_id") != "<local-path>":
                raise SystemExit(f"complete_goal_step should redact path-shaped bad ids: {path_bad_step.metadata}")
            assert_goal_refusal_handoff(path_bad_step.metadata, "path bad complete goal step", source="complete_goal_step", mutation="step_completion", reason="bad_step_id")
            assert_safe(path_bad_step.metadata, "path bad complete goal step")

        for bad_numeric_id in (0, -1):
            bad_numeric_step = runtime.registry.get("complete_goal_step").handler({"step_id": bad_numeric_id})
            if bad_numeric_step.ok or "positive number" not in bad_numeric_step.output:
                raise SystemExit("complete_goal_step should reject non-positive ids before mutation.")
            if bad_numeric_step.metadata.get("raw_id") != str(bad_numeric_id):
                raise SystemExit(f"complete_goal_step should preserve non-positive raw id metadata: {bad_numeric_step.metadata}")
            if bad_numeric_step.metadata.get("writes_files") or bad_numeric_step.metadata.get("writes_memory") or bad_numeric_step.metadata.get("writes_notes"):
                raise SystemExit(f"complete_goal_step non-positive id should not write: {bad_numeric_step.metadata}")
            assert_goal_refusal_handoff(bad_numeric_step.metadata, "bad numeric complete goal step", source="complete_goal_step", mutation="step_completion", reason="bad_step_id")
            assert_safe(bad_numeric_step.metadata, "bad numeric complete goal step")

        exported = runtime.registry.get("export_goal").handler({"goal_id": 1})
        if not exported.ok or exported.metadata.get("writes_files") is not True:
            raise SystemExit("export_goal missed write metadata.")
        if not exported.metadata.get("writes_notes") or exported.metadata.get("writes_memory") is not False:
            raise SystemExit("export_goal exposed incorrect note/memory write metadata.")
        assert_goal_export_handoff(exported.metadata, "direct export_goal")
        assert_vault_relative_receipt(exported, root, "Projects/", "direct export_goal")

        bad_export = runtime.registry.get("export_goal").handler({"goal_id": 0})
        if bad_export.ok or "positive number" not in bad_export.output:
            raise SystemExit("export_goal should reject non-positive ids before writing.")
        if bad_export.metadata.get("raw_id") != "0" or bad_export.metadata.get("writes_files"):
            raise SystemExit(f"export_goal non-positive id should preserve raw id and not write: {bad_export.metadata}")
        assert_safe(bad_export.metadata, "bad export goal")
        bool_export = runtime.registry.get("export_goal").handler({"goal_id": True})
        if bool_export.ok or "must be a number" not in bool_export.output:
            raise SystemExit("export_goal should reject boolean ids before writing.")
        if bool_export.metadata.get("raw_id") != "True" or bool_export.metadata.get("writes_files"):
            raise SystemExit(f"export_goal boolean id should preserve raw id and not write: {bool_export.metadata}")
        assert_safe(bool_export.metadata, "bool export goal")
        for path_id in (
            "/\x55sers/example/private/export-goal-id",
            "/var/folders/zc/jarvis/export-goal-id",
            "/tmp/jarvis/export-goal-id",
        ):
            path_bad_export = runtime.registry.get("export_goal").handler({"goal_id": path_id})
            if path_bad_export.ok or path_bad_export.metadata.get("raw_id") != "<local-path>" or path_bad_export.metadata.get("writes_files"):
                raise SystemExit(f"export_goal should redact path-shaped bad ids and not write: {path_bad_export.metadata}")
            assert_safe(path_bad_export.metadata, "path bad export goal")

        long_goal = runtime.registry.get("create_goal").handler({
            "title": "Jarvis " * 80,
            "purpose": "Bound long goal text safely. " * 80,
            "horizon": "this month " * 50,
        })
        if not long_goal.ok:
            raise SystemExit("Long goal creation should run.")
        if long_goal.metadata.get("title_chars", 9999) > 180 or long_goal.metadata.get("purpose_chars", 9999) > 620:
            raise SystemExit("create_goal did not bound long fields.")
        assert_goal_mutation_handoff(long_goal.metadata, "long create goal", source="create_goal", mutation="goal_create", changed=["goal"])

        long_step = runtime.registry.get("add_goal_step").handler({"goal_id": 1, "body": "step " * 500})
        if not long_step.ok or long_step.metadata.get("body_chars", 9999) > 620:
            raise SystemExit("add_goal_step did not bound long step bodies.")
        if not long_step.metadata.get("writes_notes") or not long_step.metadata.get("writes_memory"):
            raise SystemExit("add_goal_step missed note/memory write metadata.")
        assert_goal_mutation_handoff(long_step.metadata, "long add goal step", source="add_goal_step", mutation="step_create", changed=["steps"])

        bad_status = runtime.registry.get("set_goal_status").handler({"goal_id": 1, "status": "unknown"})
        if bad_status.ok:
            raise SystemExit("set_goal_status should reject unknown statuses.")
        if bad_status.metadata.get("reason") != "bad_status" or bad_status.metadata.get("raw_status") != "unknown":
            raise SystemExit(f"set_goal_status bad status should preserve bounded raw status: {bad_status.metadata}")
        assert_goal_refusal_handoff(bad_status.metadata, "bad goal status update", source="set_goal_status", mutation="status_update", reason="bad_status")
        assert_safe(bad_status.metadata, "bad goal status update")
        for path_status in (
            "/private/tmp/jarvis-set-goal-status",
            "/var/folders/zc/jarvis/set-goal-status",
            "/tmp/jarvis/set-goal-status",
        ):
            path_bad_status = runtime.registry.get("set_goal_status").handler({"goal_id": 1, "status": path_status})
            if path_bad_status.ok or path_bad_status.metadata.get("raw_status") != "<local-path>":
                raise SystemExit(f"set_goal_status should redact path-shaped bad statuses: {path_bad_status.metadata}")
            assert_goal_refusal_handoff(path_bad_status.metadata, "path bad goal status update", source="set_goal_status", mutation="status_update", reason="bad_status")
            assert_safe(path_bad_status.metadata, "path bad goal status update")

        bad_status_id = runtime.registry.get("set_goal_status").handler({"goal_id": 0, "status": "done"})
        if bad_status_id.ok or "positive number" not in bad_status_id.output:
            raise SystemExit("set_goal_status should reject non-positive ids before mutation.")
        if bad_status_id.metadata.get("raw_id") != "0" or bad_status_id.metadata.get("writes_files"):
            raise SystemExit(f"set_goal_status non-positive id should preserve raw id and not write: {bad_status_id.metadata}")
        assert_goal_refusal_handoff(bad_status_id.metadata, "bad goal status id", source="set_goal_status", mutation="status_update", reason="bad_goal_id")
        assert_safe(bad_status_id.metadata, "bad goal status id")
        for path_id in (
            "/\x55sers/example/private/status-goal-id",
            "/var/folders/zc/jarvis/status-goal-id",
            "/tmp/jarvis/status-goal-id",
        ):
            path_bad_status_id = runtime.registry.get("set_goal_status").handler({"goal_id": path_id, "status": "done"})
            if path_bad_status_id.ok or path_bad_status_id.metadata.get("raw_id") != "<local-path>" or path_bad_status_id.metadata.get("writes_files"):
                raise SystemExit(f"set_goal_status should redact path-shaped bad ids and not write: {path_bad_status_id.metadata}")
            assert_goal_refusal_handoff(path_bad_status_id.metadata, "path bad goal status id", source="set_goal_status", mutation="status_update", reason="bad_goal_id")
            assert_safe(path_bad_status_id.metadata, "path bad goal status id")

        missing_title = runtime.registry.get("create_goal").handler({"title": ""})
        if missing_title.ok or missing_title.metadata.get("reason") != "missing_title" or missing_title.metadata.get("raw_title") != "":
            raise SystemExit(f"create_goal missing title should preserve bounded raw title metadata: {missing_title.metadata}")
        assert_goal_refusal_handoff(missing_title.metadata, "missing goal title", source="create_goal", mutation="goal_create", reason="missing_title")
        assert_safe(missing_title.metadata, "missing goal title")
        for path_title in (
            "/\x55sers/example/private/goal-title",
            "/var/folders/zc/jarvis/goal-title",
            "/tmp/jarvis/goal-title",
        ):
            path_bad_title = runtime.registry.get("create_goal").handler({"title": path_title})
            if path_bad_title.ok or path_bad_title.metadata.get("reason") != "invalid_title" or path_bad_title.metadata.get("raw_title") != "<local-path>":
                raise SystemExit(f"create_goal should reject and redact path-shaped goal titles: {path_bad_title.metadata}")
            if path_bad_title.metadata.get("writes_files") or path_bad_title.metadata.get("writes_memory") or path_bad_title.metadata.get("writes_notes"):
                raise SystemExit(f"create_goal path-shaped title should not write: {path_bad_title.metadata}")
            if any(fragment in path_bad_title.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"create_goal path-shaped title leaked output: {path_bad_title.output}")
            assert_goal_refusal_handoff(path_bad_title.metadata, "path bad goal title", source="create_goal", mutation="goal_create", reason="invalid_title")
            assert_safe(path_bad_title.metadata, "path bad goal title")

        missing_step_body = runtime.registry.get("add_goal_step").handler({"goal_id": 1, "body": ""})
        if missing_step_body.ok or missing_step_body.metadata.get("reason") != "missing_body":
            raise SystemExit(f"add_goal_step missing body should return a local refusal: {missing_step_body.metadata}")
        assert_goal_refusal_handoff(missing_step_body.metadata, "missing goal step body", source="add_goal_step", mutation="step_create", reason="missing_body")
        assert_safe(missing_step_body.metadata, "missing goal step body")


if __name__ == "__main__":
    main()
