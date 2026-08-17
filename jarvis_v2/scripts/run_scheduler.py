from __future__ import annotations

import time

from jarvis_v2.config import load_config
from jarvis_v2.automations.scheduler import Scheduler, _safe_job_failure_code
from jarvis_v2.memory.goal_projection import reconcile_pending_goal_projections
from jarvis_v2.memory.memory_projection import reconcile_pending_memory_projections
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.profile_projection import reconcile_pending_profile_projections
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.scripts.startup import is_startup_storage_error, print_startup_failure
from jarvis_v2.scripts.daemon_gate import require_v3_daemon_enable, require_v3_scheduler_enable


def main() -> None:
    require_v3_daemon_enable("scheduler")
    require_v3_scheduler_enable()
    try:
        config = load_config()
    except Exception as exc:
        if not (is_startup_storage_error(exc) or isinstance(exc, UnicodeError)):
            raise
        print_startup_failure(exc, program="Jarvis scheduler")
        raise SystemExit(3) from None

    try:
        store = MemoryStore(config.db_path)
        vault = ObsidianVault(config.obsidian_vault, config.obsidian_root)
        store.init()
        vault.init()
        reconcile_pending_profile_projections(store, vault, limit=20)
        reconcile_pending_goal_projections(store, vault, limit=20)
        reconcile_pending_memory_projections(store, vault, limit=20)
        scheduler = Scheduler(store, vault, config)
    except Exception as exc:
        if not is_startup_storage_error(exc):
            raise
        print_startup_failure(exc, program="Jarvis scheduler")
        raise SystemExit(3) from None

    print("Jarvis V3 scheduler running. Press Ctrl+C to stop.")
    while True:
        try:
            output = scheduler.run_due_jobs()
            if output != "No jobs due.":
                print(output)
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
            return
        except Exception as exc:
            print(f"Scheduler tick failed: {_safe_job_failure_code(exc)}. Retrying in 60 seconds.")
        try:
            reconcile_pending_goal_projections(store, vault, limit=20)
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
            return
        except Exception as exc:
            print(
                "Goal projection recovery failed: "
                f"{_safe_job_failure_code(exc)}. Retrying in 60 seconds."
            )
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
            return


if __name__ == "__main__":
    main()
