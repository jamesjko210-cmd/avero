"""Offline acceptance proof for V3 primary-storage persistence across restart.

The parent process creates an empty temporary V3 storage boundary, then starts
two independent Python processes.  The first writes through ``JarvisRuntime``;
the second constructs a new runtime and proves the same tasks, goals, notes,
memory, preference, and planning context are still available.  The fixture
disables storage fallback and model/network routing so a passing result can
only come from the configured temporary primary SQLite database and vault.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_BODY = "verify the isolated V3 restart boundary"
GOAL_TITLE = "Prove durable V3 restart"
GOAL_STEP = "start a fresh runtime and inspect primary storage"
NOTE_BODY = "clean restart acceptance note survived in the V3 vault"
MEMORY_BODY = "clean restart acceptance memory survived in V3 primary storage"
PREFERENCE_KEY = "restart_mode"
PREFERENCE_VALUE = "durable-primary-only"


def _persisted_daily_plan(runtime: Any) -> tuple[str, str]:
    daily_files = sorted((runtime.vault.root_path / "Daily").glob("*.md"))
    if len(daily_files) != 1:
        raise AssertionError(f"expected one persisted daily-plan note, found {len(daily_files)}")
    text = daily_files[0].read_text(encoding="utf-8")
    for marker in (TASK_BODY, GOAL_TITLE, GOAL_STEP):
        if marker not in text:
            raise AssertionError(f"persisted daily-plan note missed durable context {marker!r}")
    relative = str(daily_files[0].relative_to(runtime.vault.root_path))
    digest = hashlib.sha256(daily_files[0].read_bytes()).hexdigest()
    return relative, digest


def _assert_primary_storage(runtime: Any, root: Path) -> None:
    expected_data = root / "primary-data"
    expected_db = expected_data / "jarvis-v3-primary.sqlite"
    expected_vault = root / "primary-vault"
    expected_note_root = expected_vault / "Jarvis"
    if runtime.config.data_dir.resolve() != expected_data.resolve():
        raise AssertionError(f"runtime selected the wrong data directory: {runtime.config.data_dir}")
    if runtime.config.db_path.resolve() != expected_db.resolve():
        raise AssertionError(f"runtime selected the wrong primary database: {runtime.config.db_path}")
    if runtime.store.db_path.resolve() != expected_db.resolve():
        raise AssertionError(f"memory store is not bound to the configured primary database: {runtime.store.db_path}")
    if runtime.config.obsidian_vault.resolve() != expected_vault.resolve():
        raise AssertionError(f"runtime selected the wrong vault: {runtime.config.obsidian_vault}")
    if runtime.vault.root_path.resolve() != expected_note_root.resolve():
        raise AssertionError(f"runtime note root is not the configured V3 vault: {runtime.vault.root_path}")
    if runtime.storage_fallback is not None:
        raise AssertionError(f"runtime unexpectedly activated fallback storage: {runtime.storage_fallback}")
    if runtime.store.read_only_startup:
        raise AssertionError("runtime opened primary storage in read-only startup mode")

    with runtime.store.connect() as conn:
        rows = conn.execute("PRAGMA database_list").fetchall()
    main_files = [Path(str(row[2])).resolve() for row in rows if str(row[1]) == "main"]
    if main_files != [expected_db.resolve()]:
        raise AssertionError(f"SQLite main database is not the configured primary file: {main_files}")


def _handle(runtime: Any, command: str, expected_tool: str) -> Any:
    result = runtime.handle(command)
    tools = [item.tool_name for item in result.tool_results]
    if not result.verified or tools != [expected_tool]:
        raise AssertionError(
            f"{command!r} did not complete through {expected_tool}: "
            f"verified={result.verified}, tools={tools}, response={result.response!r}"
        )
    trace = result.metadata.get("runtime_trace") or {}
    if runtime.storage_fallback is not None:
        raise AssertionError(f"{command!r} activated fallback storage: {runtime.storage_fallback}")
    if trace.get("approval_required") or trace.get("queued_approval_ids"):
        raise AssertionError(f"offline persistence command unexpectedly reached approval: {trace}")
    return result


def _assert_durable_rows(runtime: Any) -> int:
    tasks = runtime.store.list_tasks(status="open", limit=100)
    matching_tasks = [row for row in tasks if str(row["body"]) == TASK_BODY]
    if len(matching_tasks) != 1 or str(matching_tasks[0]["priority"]) != "high":
        raise AssertionError(f"durable task is missing or changed: {[dict(row) for row in tasks]}")

    goals = runtime.store.list_goals(status="active", limit=100)
    matching_goals = [row for row in goals if str(row["title"]) == GOAL_TITLE]
    if len(matching_goals) != 1:
        raise AssertionError(f"durable goal is missing or duplicated: {[dict(row) for row in goals]}")
    goal_id = int(matching_goals[0]["id"])
    steps = runtime.store.list_goal_steps(goal_id)
    if len(steps) != 1 or str(steps[0]["body"]) != GOAL_STEP or str(steps[0]["status"]) != "open":
        raise AssertionError(f"durable goal step is missing or changed: {[dict(row) for row in steps]}")

    memories = runtime.store.list_memories(category="facts", limit=100)
    if sum(str(row["body"]) == MEMORY_BODY for row in memories) != 1:
        raise AssertionError(f"durable memory is missing or duplicated: {[dict(row) for row in memories]}")

    preferences = runtime.store.list_preferences(category="reliability", status="active", limit=100)
    matches = [
        row
        for row in preferences
        if str(row["key"]) == PREFERENCE_KEY and str(row["value"]) == PREFERENCE_VALUE
    ]
    if len(matches) != 1:
        raise AssertionError(f"durable preference is missing or duplicated: {[dict(row) for row in preferences]}")

    note = runtime.vault.root_path / "Inbox" / "Captured Notes.md"
    if not note.is_file() or NOTE_BODY not in note.read_text(encoding="utf-8"):
        raise AssertionError("durable Jarvis note is missing from the configured primary vault")
    return goal_id


def _writer(root: Path) -> dict[str, Any]:
    from jarvis_v2.agent.runtime import JarvisRuntime

    runtime = JarvisRuntime()
    _assert_primary_storage(runtime, root)
    _handle(runtime, f"add task {TASK_BODY} priority high", "add_task")
    _handle(runtime, f"create goal {GOAL_TITLE}", "create_goal")
    goal_id = _assert_goal_id(runtime)
    _handle(runtime, f"add step to goal {goal_id}: {GOAL_STEP}", "add_goal_step")
    _handle(runtime, f"take a note {NOTE_BODY}", "write_jarvis_note")
    _handle(runtime, f"remember that {MEMORY_BODY}", "remember")
    _handle(
        runtime,
        f"set preference {PREFERENCE_KEY} to {PREFERENCE_VALUE} category reliability",
        "set_preference",
    )
    _assert_durable_rows(runtime)

    planning = _handle(runtime, "daily plan", "daily_plan")
    for marker in (TASK_BODY, GOAL_TITLE, GOAL_STEP):
        if marker not in planning.response:
            raise AssertionError(f"pre-restart daily planning missed durable context {marker!r}")
    daily_plan_relative, daily_plan_digest = _persisted_daily_plan(runtime)
    return {
        "phase": "writer",
        "session_id": runtime.session_id,
        "goal_id": goal_id,
        "db_exists": runtime.config.db_path.is_file(),
        "store_identity": runtime.store.store_instance_identity(),
        "daily_plan_relative": daily_plan_relative,
        "daily_plan_digest": daily_plan_digest,
        "fallback_active": runtime.storage_fallback is not None,
    }


def _assert_goal_id(runtime: Any) -> int:
    goals = [
        row
        for row in runtime.store.list_goals(status="active", limit=100)
        if str(row["title"]) == GOAL_TITLE
    ]
    if len(goals) != 1:
        raise AssertionError(f"could not resolve freshly-created acceptance goal: {[dict(row) for row in goals]}")
    return int(goals[0]["id"])


def _reader(root: Path) -> dict[str, Any]:
    from jarvis_v2.agent.runtime import JarvisRuntime

    runtime = JarvisRuntime()
    _assert_primary_storage(runtime, root)
    goal_id = _assert_durable_rows(runtime)
    daily_plan_relative, daily_plan_digest = _persisted_daily_plan(runtime)

    read_cases = (
        ("list tasks", "list_tasks", TASK_BODY),
        ("list goals", "list_goals", GOAL_TITLE),
        (f"goal {goal_id} status", "goal_status", GOAL_STEP),
        ("read jarvis note Inbox/Captured Notes.md", "read_jarvis_note", NOTE_BODY),
        ("what do you remember about clean restart acceptance", "search_memory", MEMORY_BODY),
        ("list preferences", "list_preferences", PREFERENCE_VALUE),
    )
    for command, tool, marker in read_cases:
        result = _handle(runtime, command, tool)
        if marker not in result.response:
            raise AssertionError(f"post-restart {tool} missed persisted marker {marker!r}: {result.response}")

    planning = _handle(runtime, "daily plan", "daily_plan")
    for marker in (TASK_BODY, GOAL_TITLE, GOAL_STEP):
        if marker not in planning.response:
            raise AssertionError(f"post-restart daily planning missed durable context {marker!r}")

    next_action = _handle(runtime, "next action packet", "next_action_packet")
    if TASK_BODY not in next_action.response:
        raise AssertionError(f"post-restart next-action planning missed the durable task: {next_action.response}")

    readiness = _handle(runtime, "readiness report", "readiness_report")
    if "Runtime storage fallback: ok | inactive" not in readiness.response:
        raise AssertionError(f"post-restart readiness did not prove fallback inactive: {readiness.response}")
    readiness_metadata = readiness.tool_results[0].metadata
    if readiness_metadata.get("storage_runtime_fallback_active") is not False:
        raise AssertionError(f"readiness metadata did not prove primary storage: {readiness_metadata}")

    status = _handle(runtime, "jarvis status", "jarvis_status")
    status_metadata = status.tool_results[0].metadata
    if (
        status_metadata.get("project_name") != "Jarvis V3"
        or status_metadata.get("memories") is not True
        or status_metadata.get("active_goals") != 1
    ):
        raise AssertionError(f"post-restart status missed durable V3 state: {status_metadata}")

    return {
        "phase": "reader",
        "session_id": runtime.session_id,
        "goal_id": goal_id,
        "store_identity": runtime.store.store_instance_identity(),
        "daily_plan_relative": daily_plan_relative,
        "daily_plan_digest_before_regeneration": daily_plan_digest,
        "fallback_active": runtime.storage_fallback is not None,
        "primary_rows_verified": True,
        "planning_verified": True,
        "readiness_verified": True,
        "status_verified": True,
    }


def _child_env(root: Path) -> dict[str, str]:
    home = root / "empty-home"
    temp = root / "process-temp"
    data = root / "primary-data"
    vault = root / "primary-vault"
    empty_env = root / "empty-v3.env"
    for directory in (home, temp, data, vault):
        directory.mkdir(parents=True, exist_ok=True)
    empty_env.write_text("", encoding="utf-8")
    empty_env.chmod(0o600)
    return {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(temp),
        "JARVIS_V3_ENV": str(empty_env),
        "JARVIS_DATA_DIR": str(data),
        "JARVIS_DB_PATH": str(data / "jarvis-v3-primary.sqlite"),
        "JARVIS_OBSIDIAN_VAULT": str(vault),
        "JARVIS_OBSIDIAN_ROOT": "Jarvis",
        "JARVIS_STORAGE_FALLBACK_DIR": str(root / "forbidden-fallback"),
        "JARVIS_DISABLE_STORAGE_FALLBACK": "1",
        "JARVIS_USE_MODEL_PLANNER": "0",
        "JARVIS_MODEL_PROVIDER": "invalid",
        "JARVIS_WATCHED_DIRS": "",
        "JARVIS_V3_ENABLE_DAEMONS": "0",
        "JARVIS_V3_ENABLE_SCHEDULER": "0",
    }


def _run_phase(root: Path, phase: str, env: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "jarvis_v2.scripts.smoke_test_v3_clean_restart_persistence", "--phase", phase, "--root", str(root)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90.0,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"V3 clean-restart {phase} process failed with {result.returncode}: "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"V3 clean-restart {phase} process returned no receipt")
    try:
        receipt = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"V3 clean-restart {phase} receipt was malformed: {lines[-1]!r}") from exc
    if receipt.get("phase") != phase or receipt.get("fallback_active") is not False:
        raise SystemExit(f"V3 clean-restart {phase} receipt did not prove primary storage: {receipt}")
    return receipt


def test_clean_restart_primary_storage_persistence() -> None:
    with TemporaryDirectory(prefix="jarvis-v3-clean-restart-") as temp:
        root = Path(temp)
        env = _child_env(root)
        writer = _run_phase(root, "writer", env)
        reader = _run_phase(root, "reader", env)
        if writer.get("session_id") == reader.get("session_id"):
            raise SystemExit("writer and reader unexpectedly reused the same runtime session identity")
        if writer.get("goal_id") != reader.get("goal_id"):
            raise SystemExit(f"goal identity changed across restart: {writer} -> {reader}")
        if writer.get("store_identity") != reader.get("store_identity"):
            raise SystemExit(f"primary SQLite identity changed across restart: {writer} -> {reader}")
        if (
            writer.get("daily_plan_relative") != reader.get("daily_plan_relative")
            or writer.get("daily_plan_digest") != reader.get("daily_plan_digest_before_regeneration")
        ):
            raise SystemExit(f"persisted planning artifact changed before restart recovery: {writer} -> {reader}")
        if not writer.get("db_exists") or reader.get("primary_rows_verified") is not True:
            raise SystemExit(f"primary database proof is incomplete: {writer} -> {reader}")
        if (root / "forbidden-fallback").exists():
            raise SystemExit("disabled fallback storage was created during clean-restart acceptance")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase", choices=("writer", "reader"))
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.phase:
        if args.root is None:
            raise SystemExit("--root is required for a child phase")
        receipt = _writer(args.root) if args.phase == "writer" else _reader(args.root)
        print(json.dumps(receipt, sort_keys=True))
        return
    test_clean_restart_primary_storage_persistence()
    print("V3 clean-restart primary-storage persistence smoke passed")


if __name__ == "__main__":
    main()
