from __future__ import annotations

from pathlib import Path
import re

from jarvis_v2.scripts import bootstrap_memory, startup
from jarvis_v2.tools import storage
from jarvis_v2.v3_commands import (
    V3_BOOTSTRAP_CHECK_COMMAND,
    V3_BOOTSTRAP_WRITE_COMMAND,
    V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND,
    V3_CALENDAR_AUTH_READONLY_COMMAND,
    V3_DASHBOARD_COMMAND,
    V3_DASHBOARD_INFO_COMMAND,
    V3_DASHBOARD_MODULE_COMMAND,
    V3_DIAGNOSE_COMMAND,
    V3_DIAGNOSE_PYTHON_COMMAND,
    V3_DIAGNOSE_PYTHON_JSON_COMMAND,
    V3_ENV_COMMAND_PREFIX,
    V3_HUD_COMMAND,
    V3_HUD_MUTED_COMMAND,
    V3_LIVE_CHECK_COMMAND,
    V3_ONE_SHOT_TIME_COMMAND,
    V3_PROJECT_PYTHON,
    V3_ROOT_LAUNCHER,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPERATOR_RUNBOOKS = ("V3_SUPERVISED_PROOF_RUNBOOK.md",)
RUNBOOK_SHELL_COMMAND_RE = re.compile(
    r"(?:\./launch_jarvis_v3(?:_[a-z0-9_]+)?\.py|"
    r"(?:\.venv/bin/)?python3 -m jarvis_v2\.scripts\.google_calendar_(?:readonly_auth|reauth))"
)
PRODUCTION_FILES = (
    "jarvis_v2/agent/runtime.py",
    "jarvis_v2/scripts/bootstrap_memory.py",
    "jarvis_v2/scripts/hud.py",
    "jarvis_v2/tools/storage.py",
    "jarvis_v2/tools/readiness.py",
    "jarvis_v2/scripts/live_check.py",
    "jarvis_v2/tools/registry.py",
    "jarvis_v2/tools/help.py",
    "jarvis_v2/tools/doctor.py",
    "jarvis_v2/ui/status_server.py",
)
FORBIDDEN_BY_FILE = {
    "jarvis_v2/agent/runtime.py": ("`python3 -m jarvis_v2.scripts.bootstrap_memory --check`",),
    "jarvis_v2/scripts/bootstrap_memory.py": (
        'BOOTSTRAP_CHECK_COMMAND = "python3 -m',
        'BOOTSTRAP_WRITE_COMMAND = "python3 -m',
    ),
    "jarvis_v2/scripts/hud.py": (
        "Run `python3 -m jarvis_v2.scripts.bootstrap_memory",
        "\n    python3 -m jarvis_v2.scripts.hud",
    ),
    "jarvis_v2/tools/storage.py": (
        'BOOTSTRAP_CHECK_COMMAND = "python3 -m',
        'BOOTSTRAP_WRITE_COMMAND = "python3 -m',
    ),
    "jarvis_v2/tools/readiness.py": ("then run `python3 -m jarvis_v2.scripts.bootstrap_memory",),
    "jarvis_v2/scripts/live_check.py": (
        "JARVIS_LIVE_CHECK=1 python3 -m jarvis_v2.scripts.live_check",
        '("python3 launch_jarvis_v3_dashboard.py",)',
    ),
    "jarvis_v2/tools/registry.py": (
        "dashboard command: `python3 launch_jarvis_v3_dashboard.py`",
        "ask-Jarvis dashboard command: `python3 -m jarvis_v2.scripts.ask",
        'dashboard_launch_command="python3 launch_jarvis_v3_dashboard.py"',
        'dashboard_ask_command=\'python3 -m jarvis_v2.scripts.ask',
        '"python3 launch_jarvis_v3_dashboard.py\\n\\n"',
        '"python3 -m jarvis_v2.scripts.run_status_server\\n\\n"',
    ),
    "jarvis_v2/tools/help.py": (
        '"terminal one-shot: python3 -m jarvis_v2.scripts.ask',
        '"terminal dashboard: python3 launch_jarvis_v3_dashboard.py',
        '"terminal dashboard help: python3 -m jarvis_v2.scripts.ask',
        '"terminal diagnose: python3 -m jarvis_v2.scripts.ask',
        '"terminal send: python3 -m jarvis_v2.scripts.ask',
    ),
    "jarvis_v2/tools/doctor.py": (
        "Install optional computer-control deps: python3 -m pip",
        "dashboard command: `python3 launch_jarvis_v3_dashboard.py`",
        "ask-Jarvis command: `python3 -m jarvis_v2.scripts.ask",
        "equivalent module command: `python3 -m jarvis_v2.scripts.run_status_server`",
        'dashboard_launch_command="python3 launch_jarvis_v3_dashboard.py"',
        'dashboard_ask_command=\'python3 -m jarvis_v2.scripts.ask',
        'dashboard_module_command="python3 -m jarvis_v2.scripts.run_status_server"',
    ),
    "jarvis_v2/ui/status_server.py": (
        'or "python3 -m jarvis_v2.scripts.bootstrap_memory --check"',
        'or "python3 -m jarvis_v2.scripts.bootstrap_memory"',
    ),
}


def _assert(condition: bool, message: object) -> None:
    if not condition:
        raise SystemExit(str(message))


def test_canonical_command_contract() -> None:
    _assert(V3_ENV_COMMAND_PREFIX == 'JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env"', V3_ENV_COMMAND_PREFIX)
    _assert(V3_PROJECT_PYTHON == f"{V3_ENV_COMMAND_PREFIX} .venv/bin/python3", V3_PROJECT_PYTHON)
    _assert(V3_ROOT_LAUNCHER == f"{V3_ENV_COMMAND_PREFIX} ./launch_jarvis_v3.py", V3_ROOT_LAUNCHER)
    for command in (
        V3_BOOTSTRAP_CHECK_COMMAND,
        V3_BOOTSTRAP_WRITE_COMMAND,
        V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND,
        V3_CALENDAR_AUTH_READONLY_COMMAND,
        V3_DASHBOARD_COMMAND,
        V3_DASHBOARD_INFO_COMMAND,
        V3_DASHBOARD_MODULE_COMMAND,
        V3_DIAGNOSE_COMMAND,
        V3_DIAGNOSE_PYTHON_COMMAND,
        V3_DIAGNOSE_PYTHON_JSON_COMMAND,
        V3_HUD_COMMAND,
        V3_HUD_MUTED_COMMAND,
        V3_LIVE_CHECK_COMMAND,
        V3_ONE_SHOT_TIME_COMMAND,
    ):
        _assert(command.startswith(V3_ENV_COMMAND_PREFIX), command)
        _assert("python3 launch_jarvis_v3" not in command, command)
    _assert(startup.V3_BOOTSTRAP_CHECK_COMMAND == V3_BOOTSTRAP_CHECK_COMMAND, "startup command drift")
    _assert(bootstrap_memory.BOOTSTRAP_CHECK_COMMAND == V3_BOOTSTRAP_CHECK_COMMAND, "bootstrap check drift")
    _assert(bootstrap_memory.BOOTSTRAP_WRITE_COMMAND == V3_BOOTSTRAP_WRITE_COMMAND, "bootstrap write drift")
    _assert(storage.BOOTSTRAP_CHECK_COMMAND == V3_BOOTSTRAP_CHECK_COMMAND, "storage check drift")
    _assert(storage.BOOTSTRAP_WRITE_COMMAND == V3_BOOTSTRAP_WRITE_COMMAND, "storage write drift")


def test_production_guidance_inventory() -> None:
    _assert(set(PRODUCTION_FILES) == set(FORBIDDEN_BY_FILE), "production command inventory drift")
    for relative in PRODUCTION_FILES:
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_BY_FILE[relative]:
            _assert(forbidden not in source, f"{relative} retained bare V3 terminal guidance: {forbidden}")


def test_operator_runbook_environment_custody() -> None:
    for relative in OPERATOR_RUNBOOKS:
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        matched = 0
        for line_number, line in enumerate(source.splitlines(), start=1):
            for match in RUNBOOK_SHELL_COMMAND_RE.finditer(line):
                matched += 1
                prefix = line[: match.start()]
                _assert(
                    V3_ENV_COMMAND_PREFIX in prefix,
                    f"{relative}:{line_number} retained a bare V3 launcher/auth command",
                )
        _assert(matched >= 7, f"{relative} operator command inventory unexpectedly shrank: {matched}")
        for expected in (
            "normal macOS Terminal shell prompt",
            "Jarvis's `You:` prompt",
        ):
            _assert(expected in source, f"{relative} lost prompt-location guidance: {expected}")
        _assert(
            V3_CALENDAR_AUTH_READONLY_COMMAND in source,
            f"{relative} lost the root read-only Calendar auth launcher",
        )
        _assert(
            ".venv/bin/python3 -m jarvis_v2.scripts.google_calendar_" not in source,
            f"{relative} retained a direct Calendar auth module command",
        )
        _assert(
            "explicitly forbids the launcher's\n`full-access` mode" in source,
            f"{relative} lost the full-access proof prohibition",
        )


def main() -> None:
    test_canonical_command_contract()
    test_production_guidance_inventory()
    test_operator_runbook_environment_custody()
    print("V3 command guidance smoke passed")


if __name__ == "__main__":
    main()
