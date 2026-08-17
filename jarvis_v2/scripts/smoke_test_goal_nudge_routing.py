from __future__ import annotations

from jarvis_v2.agent.planner import RuleBasedPlanner


ROUTE_CASES = (
    ("schedule goal nudge", "schedule_goal_nudge", {}),
    ("schedule goal nudge.", "schedule_goal_nudge", {}),
    ("schedule stale goal", "schedule_goal_nudge", {}),
    ("Jarvis, schedule goal nudge", "schedule_goal_nudge", {}),
    ("Jarvis, please schedule goal nudge", "schedule_goal_nudge", {}),
    ("Okay, Jarvis, schedule goal nudge", "schedule_goal_nudge", {}),
    ("Hey, Jarvis, please schedule goal nudge", "schedule_goal_nudge", {}),
    ("Hey! Jarvis, please schedule goal nudge", "schedule_goal_nudge", {}),
    ("schedule a goal nudge", "schedule_goal_nudge", {}),
    ("schedule a daily goal nudge", "schedule_goal_nudge", {}),
    ("schedule goal nudges every day", "schedule_goal_nudge", {}),
    ("schedule the goal nudge daily", "schedule_goal_nudge", {}),
    ("schedule goal nudge daily.", "schedule_goal_nudge", {}),
    ("please schedule stale goals every day", "schedule_goal_nudge", {}),
    ("goal nudge", "goal_nudge", {}),
    ("run goal nudge now", "run_job_now", {"name": "Goal Nudge"}),
    ("run goal nudge now.", "run_job_now", {"name": "Goal Nudge"}),
    ("run the goal nudge job now", "run_job_now", {"name": "Goal Nudge"}),
    ("run job goal nudge now", "run_job_now", {"name": "Goal Nudge"}),
    ("Jarvis, run the goal nudge now", "run_job_now", {"name": "Goal Nudge"}),
    ("Jarvis, please run the goal nudge now", "run_job_now", {"name": "Goal Nudge"}),
)

UNSUPPORTED_SCHEDULES = (
    "schedule goal nudge every hour",
    "schedule goal nudges every hour",
    "schedule goal nudge every week",
    "schedule goal nudge once tomorrow",
)


def test_goal_nudge_route_precedence() -> None:
    planner = RuleBasedPlanner()
    for command, expected_tool, expected_args in ROUTE_CASES:
        plan = planner.plan(command)
        if len(plan.actions) != 1:
            raise SystemExit(f"goal nudge route produced the wrong action count: {command!r} -> {plan}")
        action = plan.actions[0]
        if action.tool_name != expected_tool or action.args != expected_args:
            raise SystemExit(
                "goal nudge route precedence drifted: "
                f"{command!r} -> {action.tool_name} {action.args}; "
                f"expected {expected_tool} {expected_args}"
            )

    for command in UNSUPPORTED_SCHEDULES:
        plan = planner.plan(command)
        if len(plan.actions) != 1 or plan.actions[0].tool_name != "respond":
            raise SystemExit(f"unsupported goal nudge cadence must clarify: {command!r} -> {plan}")
        response = plan.actions[0].args.get("text")
        if not isinstance(response, str) or "daily" not in response.lower():
            raise SystemExit(f"goal nudge cadence clarification is not actionable: {command!r} -> {plan}")

    for command in (
        "Jarvis, don't schedule goal nudge",
        "Jarvis, do not run goal nudge now",
        "Okay, Jarvis, don't schedule goal nudge",
        "Hey, Jarvis, could you please not schedule goal nudge",
        "Hey! Jarvis, please do not schedule goal nudge",
    ):
        plan = planner.plan(command)
        if plan.actions or plan.needs_model is not True:
            raise SystemExit(f"negated goal nudge command must not execute: {command!r} -> {plan}")


def main() -> None:
    test_goal_nudge_route_precedence()
    print("Goal nudge routing smoke passed")


if __name__ == "__main__":
    main()
