"""Canonical terminal commands for the Jarvis V3 project.

Keep this module dependency-light: startup, tools, scripts, and the dashboard
all import these strings while handling recovery failures.
"""

from __future__ import annotations


V3_ENV_COMMAND_PREFIX = 'JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env"'
V3_PROJECT_PYTHON = f"{V3_ENV_COMMAND_PREFIX} .venv/bin/python3"
V3_ROOT_LAUNCHER = f"{V3_ENV_COMMAND_PREFIX} ./launch_jarvis_v3.py"

V3_BOOTSTRAP_CHECK_COMMAND = (
    f"{V3_PROJECT_PYTHON} -m jarvis_v2.scripts.bootstrap_memory --check"
)
V3_BOOTSTRAP_WRITE_COMMAND = f"{V3_PROJECT_PYTHON} -m jarvis_v2.scripts.bootstrap_memory"
V3_DIAGNOSE_COMMAND = f'{V3_ROOT_LAUNCHER} --diagnose "what time is it"'
V3_DASHBOARD_COMMAND = f"{V3_ENV_COMMAND_PREFIX} ./launch_jarvis_v3_dashboard.py"
V3_CALENDAR_AUTH_READONLY_COMMAND = (
    f"{V3_ENV_COMMAND_PREFIX} ./launch_jarvis_v3_calendar_auth.py readonly"
)
V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND = (
    f"{V3_ENV_COMMAND_PREFIX} ./launch_jarvis_v3_calendar_auth.py full-access"
)
V3_CALENDAR_AUTH_TERMINAL_LOCATION = (
    "at the normal macOS Terminal shell prompt—not in Telegram and not at "
    "Jarvis's `You:` prompt"
)
V3_DASHBOARD_INFO_COMMAND = f'{V3_ROOT_LAUNCHER} "status dashboard"'
V3_DASHBOARD_MODULE_COMMAND = f"{V3_PROJECT_PYTHON} -m jarvis_v2.scripts.run_status_server"
V3_HUD_COMMAND = f"{V3_PROJECT_PYTHON} -m jarvis_v2.scripts.hud"
V3_HUD_MUTED_COMMAND = (
    f"{V3_ENV_COMMAND_PREFIX} JARVIS_HUD_SPEAK=0 .venv/bin/python3 -m jarvis_v2.scripts.hud"
)
V3_ONE_SHOT_TIME_COMMAND = f'{V3_ROOT_LAUNCHER} "what time is it"'
V3_DIAGNOSE_PYTHON_COMMAND = (
    f'{V3_ROOT_LAUNCHER} --diagnose "run command python3 --version"'
)
V3_DIAGNOSE_PYTHON_JSON_COMMAND = (
    f'{V3_ROOT_LAUNCHER} --diagnose --json "run command python3 --version"'
)
V3_LIVE_CHECK_COMMAND = (
    f"{V3_ENV_COMMAND_PREFIX} JARVIS_LIVE_CHECK=1 .venv/bin/python3 "
    "-m jarvis_v2.scripts.live_check --contact NAME"
)
