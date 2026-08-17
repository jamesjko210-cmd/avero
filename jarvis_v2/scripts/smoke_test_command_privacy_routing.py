from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.memory.store import TaskRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime


CASES = (
    ("what's on my clipboard?", "get_clipboard", {}),
    ("what's on my clipboard.", "get_clipboard", {}),
    ("what's on my clipboard!", "get_clipboard", {}),
    ("what is on my clipboard???", "get_clipboard", {}),
    ("what is in the clipboard", "get_clipboard", {}),
    ("search my tasks for approval", "search_tasks", {"query": "approval", "status": "all"}),
    ("search my tasks about approval please", "search_tasks", {"query": "approval", "status": "all"}),
    ("read my note Projects/Plan", "read_jarvis_note", {"path": "Projects/Plan"}),
    ("read my note Projects/Plan?", "read_jarvis_note", {"path": "Projects/Plan"}),
    ("read my notes Projects/Plan please", "read_jarvis_note", {"path": "Projects/Plan"}),
    ("read my note: Projects/Plan", "read_jarvis_note", {"path": "Projects/Plan"}),
    ("read my note, Projects/Plan", "read_jarvis_note", {"path": "Projects/Plan"}),
    ("read my Jarvis note Projects/Plan", "read_jarvis_note", {"path": "Projects/Plan"}),
)


def _assert_plan_routes() -> None:
    planner = RuleBasedPlanner()
    forbidden = {"list_events", "web_lookup", "read_text_file"}
    for command, expected_tool, expected_args in CASES:
        plan = planner.plan(command)
        if len(plan.actions) != 1:
            raise SystemExit(f"privacy route did not produce one action: {command!r} -> {plan}")
        action = plan.actions[0]
        if action.tool_name != expected_tool or action.args != expected_args:
            raise SystemExit(
                f"privacy route drifted: {command!r} -> {action.tool_name} {action.args}"
            )
        if action.tool_name in forbidden:
            raise SystemExit(f"privacy route escaped to a broader tool: {command!r} -> {action}")


def _assert_runtime_boundaries() -> None:
    with TemporaryDirectory(prefix="jarvis-command-privacy-routing-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.add_task(TaskRecord("review approval routing"))
        runtime.vault.create_note("Projects/Plan.md", "Scoped plan body.", heading="Plan")
        results = {command: runtime.handle(command) for command, _, _ in CASES}

        clipboard = results["what's on my clipboard?"]
        clipboard_item = clipboard.tool_results[0]
        if (
            clipboard.plan.actions[0].tool_name != "get_clipboard"
            or clipboard_item.tool_name != "get_clipboard"
            or clipboard_item.metadata.get("requires_confirmation") is not True
            or clipboard_item.metadata.get("executed_handler") is not False
        ):
            raise SystemExit(f"clipboard alias bypassed its personal-data approval hold: {clipboard}")

        tasks = results["search my tasks for approval"]
        if (
            tasks.plan.actions[0].tool_name != "search_tasks"
            or tasks.tool_results[0].tool_name != "search_tasks"
            or tasks.tool_results[0].metadata.get("executes_side_effect") is not False
            or "review approval routing" not in tasks.response
        ):
            raise SystemExit(f"private task search escaped its local read path: {tasks}")

        note = results["read my note Projects/Plan"]
        if (
            note.plan.actions[0].tool_name != "read_jarvis_note"
            or note.tool_results[0].tool_name != "read_jarvis_note"
            or note.tool_results[0].metadata.get("requires_confirmation") is True
            or "Scoped plan body." not in note.response
        ):
            raise SystemExit(f"Jarvis-note alias escaped to a personal file read: {note}")

        pending = runtime.store.list_pending_approvals(limit=10)
        if not pending or any(row["tool_name"] != "get_clipboard" for row in pending):
            raise SystemExit(f"privacy aliases queued the wrong approvals: {pending}")
        forbidden_runs = {
            row["tool_name"]
            for row in runtime.store.recent_tool_runs(limit=20)
            if row["tool_name"] in {"list_events", "web_lookup", "read_text_file"}
        }
        if forbidden_runs:
            raise SystemExit(f"privacy aliases invoked broader tools: {sorted(forbidden_runs)}")


def main() -> None:
    _assert_plan_routes()
    _assert_runtime_boundaries()
    print("command privacy routing smoke passed")


if __name__ == "__main__":
    main()
