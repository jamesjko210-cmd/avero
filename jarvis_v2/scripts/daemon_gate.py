"""Fail-closed V3 background-service activation boundary."""

from __future__ import annotations

import os


V3_DAEMON_ENABLE_ENV = "JARVIS_V3_ENABLE_DAEMONS"
V3_SCHEDULER_ENABLE_ENV = "JARVIS_V3_ENABLE_SCHEDULER"


def v3_daemons_enabled() -> bool:
    """Return true only for the exact supervised V3 daemon opt-in."""

    return os.getenv(V3_DAEMON_ENABLE_ENV, "") == "1"


def require_v3_daemon_enable(service: str) -> None:
    """Refuse before runtime, credentials, state, or network are touched."""
    if v3_daemons_enabled():
        return
    print(
        f"Jarvis V3 {service} daemon is disabled before startup. "
        f"Set {V3_DAEMON_ENABLE_ENV}=1 only for a supervised V3 service cutover after "
        "isolated storage, credentials, service labels, and rollback are verified. "
        "No runtime was constructed and no external action was attempted.",
        flush=True,
    )
    raise SystemExit(4)


def v3_scheduler_enabled() -> bool:
    """Return true only for the exact, separately supervised scheduler opt-in."""

    return os.getenv(V3_SCHEDULER_ENABLE_ENV, "") == "1"


def require_v3_scheduler_enable() -> None:
    """Refuse scheduler execution unless its service-specific opt-in is exact."""

    if v3_scheduler_enabled():
        return
    print(
        "Jarvis V3 scheduler execution is disabled before startup. "
        f"Set {V3_SCHEDULER_ENABLE_ENV}=1 only for a separately supervised scheduler "
        "cutover after jobs, delivery targets, isolated state, and rollback are verified. "
        "No scheduler was constructed and no scheduled job was run.",
        flush=True,
    )
    raise SystemExit(4)
