from __future__ import annotations

import io
import json
import os
import plistlib
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from jarvis_v2.agent.types import RiskLevel
from jarvis_v2.agent.types import ToolResult
from jarvis_v2.scripts import live_check
from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.contacts_connector import ContactMatch
from jarvis_v2.v3_commands import (
    V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND,
    V3_CALENDAR_AUTH_READONLY_COMMAND,
    V3_CALENDAR_AUTH_TERMINAL_LOCATION,
)


IS_PUBLIC_CANDIDATE = structural_public_candidate_profile(live_check.REPO_ROOT) is not None


def _chat_message_row(response: dict[str, object], *, role: str = "assistant", content: str = "") -> dict[str, object]:
    return {
        "role": role,
        "content": content,
        "metadata": json.dumps({"runtime_route": "chat", "chat_response": response}),
    }


def _tool_message_row(
    tool_name: str,
    *,
    toolset: str = "",
    content: str = "",
    route: str = "tools",
) -> dict[str, object]:
    action: dict[str, object] = {"tool_name": tool_name}
    if toolset:
        action["toolset"] = toolset
    return {
        "role": "assistant",
        "content": content,
        "metadata": json.dumps(
            {
                "runtime_route": route,
                "runtime_trace": {
                    "route": route,
                    "planned_actions": [action],
                },
            }
        ),
    }


def _tool_run_row(
    tool_name: str,
    *,
    ok: bool = True,
    approved: bool = False,
    metadata: dict[str, object] | None = None,
    output: str = "PRIVATE TOOL OUTPUT SHOULD NOT APPEAR",
) -> dict[str, object]:
    return {
        "tool_name": tool_name,
        "ok": 1 if ok else 0,
        "approved": 1 if approved else 0,
        "output": output,
        "metadata": json.dumps(metadata or {}),
    }


class _FakeTool:
    def __init__(self, name: str, risk: RiskLevel) -> None:
        self.name = name
        self.risk = risk


class _FakeRegistry:
    def __init__(self, risks: dict[str, RiskLevel]) -> None:
        self._tools = {name: _FakeTool(name, risk) for name, risk in risks.items()}

    def get(self, name: str) -> _FakeTool:
        return self._tools[name]


class _FakePath:
    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists


def _expected_personal_risks(**overrides: RiskLevel) -> dict[str, RiskLevel]:
    values = {name: getattr(RiskLevel, risk_name) for name, risk_name in live_check.PERSONAL_INTEGRATION_RISKS.items()}
    values.update(overrides)
    return values


def _expected_research_risks(**overrides: RiskLevel) -> dict[str, RiskLevel]:
    values = {name: getattr(RiskLevel, risk_name) for name, risk_name in live_check.RESEARCH_TOOL_RISKS.items()}
    values.update(overrides)
    return values


def _write_startup_contract(root: Path, *, telegram_module: str = "jarvis_v2.scripts.run_telegram_control", keep_alive: bool = True) -> Path:
    (root / "jarvis_v2" / "scripts").mkdir(parents=True)
    (root / "launch_jarvis_v3_dashboard.py").write_text(
        "from jarvis_v2.scripts.run_status_server import main\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    (root / "jarvis_v2" / "scripts" / "run_telegram_control.py").write_text(
        "TelegramCommandBridge = object\n"
        "def _start_scheduler_ticker(config):\n"
        "    pass\n"
        "def main():\n"
        "    config = object()\n"
        "    _start_scheduler_ticker(config)\n"
        "    bridge = TelegramCommandBridge()\n"
        "    bridge.run_forever()\n",
        encoding="utf-8",
    )
    (root / "jarvis_v2" / "scripts" / "run_scheduler.py").write_text(
        "import time\n"
        "from jarvis_v2.automations.scheduler import Scheduler\n"
        "def main():\n"
        "    scheduler = Scheduler(None, None)\n"
        "    scheduler.run_due_jobs()\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    account_log_directory = live_check._account_log_directory()
    telegram_log = os.fspath(account_log_directory / "telegram.log")
    imessage_log = os.fspath(account_log_directory / "imessage.log")
    dashboard_log = os.fspath(account_log_directory / "dashboard.log")
    telegram_plist = {
        "Label": "com.jarvis-v3.telegram",
        "ProgramArguments": ["/opt/homebrew/bin/python3", "-m", telegram_module],
        "WorkingDirectory": str(root),
        "EnvironmentVariables": {
            "PYTHONPATH": str(root),
            "PATH": live_check.STARTUP_LAUNCHD_PATH,
        },
        "RunAtLoad": True,
        "KeepAlive": keep_alive,
        "ProcessType": "Background",
        "StandardOutPath": telegram_log,
        "StandardErrorPath": telegram_log,
    }
    imessage_plist = {
        "Label": "com.jarvis-v3.imessage",
        "ProgramArguments": ["/opt/homebrew/bin/python3", "-m", "jarvis_v2.scripts.run_imessage_control"],
        "WorkingDirectory": str(root),
        "EnvironmentVariables": {
            "PYTHONPATH": str(root),
            "PATH": live_check.STARTUP_LAUNCHD_PATH,
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": imessage_log,
        "StandardErrorPath": imessage_log,
    }
    dashboard_plist = {
        "Label": "com.jarvis-v3.dashboard",
        "ProgramArguments": ["/opt/homebrew/bin/python3", "-m", "jarvis_v2.scripts.run_status_server"],
        "WorkingDirectory": str(root),
        "EnvironmentVariables": {
            "PYTHONPATH": str(root),
            "PATH": live_check.STARTUP_LAUNCHD_PATH,
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "ThrottleInterval": 30,
        "StandardOutPath": dashboard_log,
        "StandardErrorPath": dashboard_log,
    }
    with (root / "com.jarvis-v3.telegram.plist").open("wb") as handle:
        plistlib.dump(telegram_plist, handle)
    with (root / "com.jarvis-v3.imessage.plist").open("wb") as handle:
        plistlib.dump(imessage_plist, handle)
    with (root / "com.jarvis-v3.dashboard.plist").open("wb") as handle:
        plistlib.dump(dashboard_plist, handle)
    launch_agents = root / "LaunchAgents"
    launch_agents.mkdir()
    return launch_agents.resolve()


def _write_daemon_source_freshness_contract(root: Path, *, modified_at: float) -> None:
    for contract in live_check.DAEMON_SOURCE_FRESHNESS_CONTRACTS.values():
        for relative_path in contract["sources"]:
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("# service source fixture\n", encoding="utf-8")
            os.utime(path, (modified_at, modified_at))
        for relative_dir in contract.get("source_dirs", ()):
            path = root / relative_dir / "source_fixture.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("# service source directory fixture\n", encoding="utf-8")
            os.utime(path, (modified_at, modified_at))


_CLEAN_SOURCE_HEAD = "a" * 40


def _stable_clean_git_reader(
    _root: Path,
) -> tuple[live_check.DaemonSourceGitSnapshot, bool]:
    return live_check.DaemonSourceGitSnapshot(
        head=_CLEAN_SOURCE_HEAD,
        clean=True,
    ), True


def _write_daemon_restart_doc(root: Path, *, complete: bool = True) -> None:
    if complete:
        text = """
## Diagnostics

### Daemon Recovery

The readiness checks above never restart daemons, call `launchctl`, send messages, or prove reboot survival.
Diagnostics never execute these commands.

```bash
launchctl kickstart -k gui/$(id -u)/com.jarvis-v3.telegram
```

```bash
launchctl kickstart -k gui/$(id -u)/com.jarvis-v3.imessage
```

```bash
launchctl kickstart -k gui/$(id -u)/com.jarvis-v3.dashboard
```

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_dashboard.py
```

On any dashboard `401` response, sign in as `jarvis` with the configured
`JARVIS_STATUS_AUTH_TOKEN`. If that token is unavailable, stop the local dashboard, run
`python3 -m jarvis_v2.scripts.setup_status_auth` to create a fresh unique owner-only token,
and restart the dashboard. The generic response never distinguishes credential errors or
echoes credentials or request headers.

Check daemon logs at `Library/Logs/JarvisV3/telegram.log`,
`Library/Logs/JarvisV3/imessage.log`, and `Library/Logs/JarvisV3/dashboard.log`.
Live reboot proof still needs operator present.
"""
    else:
        text = """
## Diagnostics

### Daemon Recovery

restart private /\x55sers/example/daemon SHOULD NOT APPEAR
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _install_primary_startup_contracts(
    root: Path,
    launch_agents: Path,
    *,
    activate_telegram: bool = True,
) -> Path | None:
    selector = root / "runtime.env"
    if activate_telegram:
        selector.write_text("", encoding="utf-8")
        selector.chmod(0o600)
        selector = selector.resolve()
    for filename in ("com.jarvis-v3.telegram.plist", "com.jarvis-v3.dashboard.plist"):
        with (root / filename).open("rb") as handle:
            installed = plistlib.load(handle)
        if activate_telegram:
            installed["EnvironmentVariables"][live_check.V3_DAEMON_ENABLE_ENV] = "1"
            installed["EnvironmentVariables"][live_check.STARTUP_SELECTED_ENV] = str(selector)
            installed["EnvironmentVariables"][live_check.V3_SCHEDULER_ENABLE_ENV] = "0"
        installed_path = launch_agents / filename
        with installed_path.open("wb") as handle:
            plistlib.dump(installed, handle)
        installed_path.chmod(0o600)
    return selector if activate_telegram else None


def _loaded_job_snapshot(
    contract: dict[str, object],
    *,
    root: Path,
    selector: Path,
    pid: int,
    state: str = "running",
) -> live_check.LaunchdJobSnapshot:
    with (root / Path(str(contract["path"])).name).open("rb") as handle:
        payload = plistlib.load(handle)
    environment = dict(payload["EnvironmentVariables"])
    environment[live_check.V3_DAEMON_ENABLE_ENV] = "1"
    environment[live_check.STARTUP_SELECTED_ENV] = str(selector)
    environment[live_check.V3_SCHEDULER_ENABLE_ENV] = "0"
    return live_check.LaunchdJobSnapshot(
        label=str(contract["label"]),
        pid=pid,
        state=state,
        program="/opt/homebrew/bin/python3",
        program_arguments=("/opt/homebrew/bin/python3", "-m", str(contract["module"])),
        working_directory=str(root),
        daemon_enable="1",
        selected_environment=str(selector),
        scheduler_enable="0",
        environment=tuple(sorted(environment.items())),
    )


def _proven_primary_startup_readers(root: Path, selector: Path):
    expected = [
        contract
        for contract in live_check.STARTUP_PLIST_CONTRACTS.values()
        if contract.get("primary")
    ]
    jobs = [
        _loaded_job_snapshot(
            contract,
            root=root,
            selector=selector,
            pid=4100 + index,
        )
        for index, contract in enumerate(expected)
    ]
    processes = [
        live_check.DaemonProcessSnapshot(
            module=str(contract["module"]),
            started_at_epoch=1_800_000_000.0,
            pid=4100 + index,
        )
        for index, contract in enumerate(expected)
    ]
    return (
        lambda labels: (jobs if tuple(labels) == tuple(job.label for job in jobs) else [], True),
        lambda: (processes, True),
    )


def test_channel_health_live_check_is_metadata_only() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-channel-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.log_tool_run(
            "smoke",
            "send_telegram",
            "HIGH_RISK",
            True,
            True,
            "LIVE CHECK OUTPUT SHOULD NOT APPEAR",
            metadata={
                "to": "BotFather SHOULD NOT APPEAR",
                "message": "secret message SHOULD NOT APPEAR",
            },
        )

        ok, detail = live_check._check_channel_health(runtime.config)

        if ok is not None:
            raise SystemExit(f"partial channel evidence must remain unproven: {ok} {detail}")
        for expected in ("channels with 7d success", "remaining channels are unproven", "content suppressed"):
            if expected not in detail:
                raise SystemExit(f"live_check channel health detail missed {expected!r}: {detail}")
        for forbidden in ("SHOULD NOT APPEAR", "BotFather", "secret message", "LIVE CHECK OUTPUT"):
            if forbidden in detail:
                raise SystemExit(f"live_check channel health leaked sensitive fixture {forbidden!r}: {detail}")


def test_channel_health_live_check_rejects_malformed_channel_metadata() -> None:
    result = ToolResult(
        "channel_health",
        True,
        "SHOULD NOT APPEAR",
        {
            "content_suppressed": True,
            "reads_message_content": False,
            "channel_count": "BotFather SHOULD NOT APPEAR",
            "rows_reviewed": "/\x55sers/example/channel-audit.sqlite SHOULD NOT APPEAR",
            "row_sample_truncated": True,
        },
    )
    with (
        mock.patch("jarvis_v2.memory.store.MemoryStore", return_value=object()),
        mock.patch("jarvis_v2.tools.channel_health.make_channel_health_tools", return_value=[lambda _args: result]),
    ):
        ok, detail = live_check._check_channel_health(config=SimpleNamespace(db_path="unused"))
    if ok is not False or detail != "channel health metadata is malformed":
        raise SystemExit(f"malformed channel-health metadata should fail closed: {ok} {detail}")
    for forbidden in ("BotFather", "/\x55sers/operator", "channel-audit.sqlite", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"channel health corrupt-count detail leaked seeded value {forbidden!r}: {detail}")


def test_channel_health_live_check_does_not_pass_without_recent_success() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-channel-empty-") as temp:
        runtime = make_temp_runtime(Path(temp))
        ok, detail = live_check._check_channel_health(runtime.config)
    if ok is not None:
        raise SystemExit(f"empty channel history must remain unproven: {ok} {detail}")
    for expected in ("0/8 channels with 7d success", "no recent channel success", "content suppressed"):
        if expected not in detail:
            raise SystemExit(f"empty channel-health detail missed {expected!r}: {detail}")


def test_channel_health_live_check_flags_newer_failure_than_success() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-channel-regression-") as temp:
        runtime = make_temp_runtime(Path(temp))
        success_id = runtime.store.log_tool_run(
            "smoke",
            "send_telegram",
            "HIGH_RISK",
            True,
            True,
            "SUCCESS SHOULD NOT APPEAR",
            metadata={"to": "BotFather SHOULD NOT APPEAR"},
        )
        runtime.store.log_tool_run(
            "smoke",
            "send_telegram",
            "HIGH_RISK",
            False,
            True,
            "FAILURE SHOULD NOT APPEAR",
            metadata={"telegram_send_stage": "telegram_enter_did_not_send"},
        )
        earlier = (datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)).replace(microsecond=0).isoformat() + "Z"
        with runtime.store.connect() as conn:
            conn.execute("UPDATE tool_runs SET created_at = ? WHERE id = ?", (earlier, success_id))
        ok, detail = live_check._check_channel_health(runtime.config)
    if ok is not False:
        raise SystemExit(f"newer channel failure must not pass health: {ok} {detail}")
    for expected in ("1/8 channels with 7d success", "newer failure than their last success"):
        if expected not in detail:
            raise SystemExit(f"regressed channel-health detail missed {expected!r}: {detail}")
    for forbidden in ("BotFather", "SUCCESS SHOULD NOT APPEAR", "FAILURE SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"regressed channel-health detail leaked {forbidden!r}: {detail}")


def test_channel_health_live_check_requires_complete_consistent_success_evidence() -> None:
    from jarvis_v2.tools.channel_health import CHANNELS

    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    channels = [
        {
            "channel": channel_id,
            "success_count_7d": 1,
            "last_success_at": now,
            "last_failure_at": "never",
        }
        for channel_id, _label, _tools in CHANNELS
    ]
    result = ToolResult(
        "channel_health",
        True,
        "SHOULD NOT APPEAR",
        {
            "content_suppressed": True,
            "reads_message_content": False,
            "channel_count": len(channels),
            "rows_reviewed": len(channels),
            "channels": channels,
            "row_sample_truncated": False,
        },
    )
    with (
        mock.patch("jarvis_v2.memory.store.MemoryStore", return_value=object()),
        mock.patch("jarvis_v2.tools.channel_health.make_channel_health_tools", return_value=[lambda _args: result]),
    ):
        ok, detail = live_check._check_channel_health(config=SimpleNamespace(db_path="unused"))
    if ok is not True or "8/8 channels with 7d success" not in detail:
        raise SystemExit(f"complete current channel proof should pass: {ok} {detail}")

    channels[0]["last_success_at"] = "never"
    with (
        mock.patch("jarvis_v2.memory.store.MemoryStore", return_value=object()),
        mock.patch("jarvis_v2.tools.channel_health.make_channel_health_tools", return_value=[lambda _args: result]),
    ):
        ok, detail = live_check._check_channel_health(config=SimpleNamespace(db_path="unused"))
    if ok is not False or detail != "channel health metadata is malformed":
        raise SystemExit(f"count without current success timestamp must fail closed: {ok} {detail}")


def test_channel_health_live_check_redacts_exception_details() -> None:
    with mock.patch(
        "jarvis_v2.memory.store.MemoryStore",
        side_effect=RuntimeError("BotFather secret message SHOULD NOT APPEAR"),
    ):
        ok, detail = live_check._check_channel_health(config=SimpleNamespace(db_path="unused"))
    if ok is not False or "check local Jarvis storage" not in detail:
        raise SystemExit(f"channel-health exception should name safe recovery: {ok} {detail}")
    for forbidden in ("BotFather", "secret message", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"channel-health exception leaked {forbidden!r}: {detail}")


def test_chat_path_live_check_reports_model_ready_without_content_leakage() -> None:
    class ChatStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def recent_messages(self, limit: int = 30, session_id: str | None = None) -> list[dict[str, object]]:
            if limit < 100 or session_id is not None:
                raise AssertionError("chat path live_check should inspect a bounded all-session window")
            return [
                {"role": "user", "content": "가상연락처이 SHOULD NOT APPEAR", "metadata": "{}"},
                _chat_message_row(
                    {
                        "runtime_route": "command_suggestion",
                        "pre_planner_command_suggestion": True,
                        "source": "model",
                        "used_model": True,
                        "latency_ms": 999,
                    }
                ),
                {
                    "role": "assistant",
                    "content": "tool reply SHOULD NOT APPEAR",
                    "metadata": json.dumps(
                        {
                            "runtime_route": "tools",
                            "chat_response": {"source": "model", "used_model": True, "latency_ms": 111},
                        }
                    ),
                },
                _chat_message_row(
                    {
                        "source": "model",
                        "used_model": True,
                        "used_fallback": False,
                        "latency_ms": 120.2,
                        "model_error": "/\x55sers/example/private/chat-error SHOULD NOT APPEAR",
                    },
                    content="assistant content SHOULD NOT APPEAR",
                ),
                _chat_message_row({"source": "model", "used_model": True, "duration_ms": 150}),
                _chat_message_row(
                    {
                        "source": "/private/tmp/hostile-source SHOULD NOT APPEAR",
                        "used_model": True,
                        "latency_ms": 180,
                    }
                ),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", ChatStore):
        ok, detail = live_check._check_chat_path_health(SimpleNamespace(db_path="unused"))
    if ok is not True:
        raise SystemExit(f"chat path model-backed window should pass: {ok} {detail}")
    for expected in (
        "3 recent ordinary chat turn(s)",
        "model 3",
        "fallback 0",
        "latency samples 3",
        "p50/p95",
        "sources model=2, unknown=1",
        "metadata-only",
    ):
        if expected not in detail:
            raise SystemExit(f"chat path detail missed {expected!r}: {detail}")
    for forbidden in (
        "가상연락처이",
        "tool reply",
        "assistant content",
        "/\x55sers/operator",
        "/private/tmp",
        "chat-error",
        "hostile-source",
        "SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(f"chat path detail leaked seeded value {forbidden!r}: {detail}")


def test_chat_path_live_check_flags_fallback_and_empty_windows_safely() -> None:
    class FallbackStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def recent_messages(self, limit: int = 30, session_id: str | None = None) -> list[dict[str, object]]:
            return [
                _chat_message_row(
                    {
                        "source": "fallback",
                        "used_model": False,
                        "used_fallback": True,
                        "latency_ms": 42,
                        "model_error": "/\x55sers/example/private/fallback-error SHOULD NOT APPEAR",
                    },
                    content="fallback answer SHOULD NOT APPEAR",
                )
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", FallbackStore):
        ok, detail = live_check._check_chat_path_health(SimpleNamespace(db_path="unused"))
    if ok is not False or "no model-backed ordinary chat" not in detail:
        raise SystemExit(f"chat path fallback-only window should fail safely: {ok} {detail}")
    for expected in ("1 recent ordinary chat turn(s)", "model 0", "fallback 1", "metadata-only"):
        if expected not in detail:
            raise SystemExit(f"fallback chat path detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "fallback-error", "fallback answer", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"fallback chat path detail leaked seeded value {forbidden!r}: {detail}")

    class EmptyStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def recent_messages(self, limit: int = 30, session_id: str | None = None) -> list[dict[str, object]]:
            return [
                _chat_message_row({"runtime_route": "command_suggestion", "source": "model", "used_model": True}),
                {"role": "assistant", "content": "bad metadata SHOULD NOT APPEAR", "metadata": "not json"},
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", EmptyStore):
        ok, detail = live_check._check_chat_path_health(SimpleNamespace(db_path="unused"))
    if ok is not None or "no stored ordinary chat turns" not in detail:
        raise SystemExit(f"chat path empty window should be a skipped/not-yet-proven row: {ok} {detail}")
    for forbidden in ("bad metadata", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"empty chat path detail leaked seeded value {forbidden!r}: {detail}")


def test_mixed_conversation_live_check_reports_ready_without_content_leakage() -> None:
    class MixedStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_sessions(self, limit: int = 20) -> list[dict[str, object]]:
            if limit < 20:
                raise AssertionError("mixed conversation live_check should inspect recent sessions")
            return [
                {"session_id": "seeded-session-id-SHOULD-NOT-APPEAR", "messages": 24},
            ]

        def recent_messages(self, limit: int = 30, session_id: str | None = None) -> list[dict[str, object]]:
            if limit < 80 or session_id != "seeded-session-id-SHOULD-NOT-APPEAR":
                raise AssertionError("mixed conversation live_check should inspect a bounded session window")
            return [
                {"role": "user", "content": "가상연락처이 SHOULD NOT APPEAR", "metadata": "{}"},
                _chat_message_row(
                    {
                        "source": "model",
                        "used_model": True,
                        "used_fallback": False,
                        "latency_ms": 1100,
                        "model_error": "/\x55sers/example/private/mixed-chat SHOULD NOT APPEAR",
                    },
                    content="chat answer SHOULD NOT APPEAR",
                ),
                _tool_message_row("research", toolset="browser", content="research content SHOULD NOT APPEAR"),
                _tool_message_row("web_lookup", toolset="browser"),
                _tool_message_row("list_events", toolset="personal", content="calendar content SHOULD NOT APPEAR"),
                _tool_message_row("check_availability", toolset="personal"),
                _tool_message_row("add_task", toolset="tasks", content="task content SHOULD NOT APPEAR"),
                _tool_message_row("list_tasks", toolset="tasks"),
                _chat_message_row({"source": "model", "used_model": True, "latency_ms": 2100}),
                _chat_message_row({"source": "model", "used_model": True, "duration_ms": 3100}),
                _tool_message_row("weather", toolset="info"),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", MixedStore):
        ok, detail = live_check._check_mixed_conversation_health(SimpleNamespace(db_path="unused"))
    if ok is not True:
        raise SystemExit(f"mixed conversation model-backed window should pass: {ok} {detail}")
    for expected in (
        "best recent session: 10 assistant turn(s)",
        "lanes chat=3, research=2, calendar=2, tasks=2",
        "chat model 3",
        "fallback 0",
        "latency samples 3",
        "chat p95",
        "metadata-only",
        "mixed-session proof ready",
    ):
        if expected not in detail:
            raise SystemExit(f"mixed conversation detail missed {expected!r}: {detail}")
    for forbidden in (
        "seeded-session-id",
        "가상연락처이",
        "chat answer",
        "research content",
        "calendar content",
        "task content",
        "/\x55sers/operator",
        "mixed-chat",
        "SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(f"mixed conversation detail leaked seeded value {forbidden!r}: {detail}")


def test_mixed_conversation_live_check_reports_incomplete_without_content_leakage() -> None:
    class IncompleteStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_sessions(self, limit: int = 20) -> list[dict[str, object]]:
            return [{"session_id": "incomplete-private-session SHOULD NOT APPEAR", "messages": 4}]

        def recent_messages(self, limit: int = 30, session_id: str | None = None) -> list[dict[str, object]]:
            return [
                _chat_message_row(
                    {
                        "source": "model",
                        "used_model": True,
                        "latency_ms": 120,
                        "model_error": "/private/tmp/incomplete-mixed SHOULD NOT APPEAR",
                    },
                    content="incomplete mixed answer SHOULD NOT APPEAR",
                ),
                _tool_message_row("list_tasks", toolset="tasks", content="private tasks SHOULD NOT APPEAR"),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", IncompleteStore):
        ok, detail = live_check._check_mixed_conversation_health(SimpleNamespace(db_path="unused"))
    if ok is not None or "not yet proven" not in detail:
        raise SystemExit(f"mixed conversation incomplete window should be skipped: {ok} {detail}")
    for expected in (
        "best recent session: 2 assistant turn(s)",
        "chat=1",
        "tasks=1",
        "metadata-only",
        "proof path: collect one session with chat, research, calendar, tasks",
        "at least 3 chat latency samples",
        "model routing status",
        "rerun the mixed conversation proof and live_check",
    ):
        if expected not in detail:
            raise SystemExit(f"incomplete mixed conversation detail missed {expected!r}: {detail}")
    for forbidden in (
        "incomplete-private-session",
        "incomplete mixed answer",
        "private tasks",
        "/private/tmp",
        "incomplete-mixed",
        "SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(f"incomplete mixed conversation detail leaked seeded value {forbidden!r}: {detail}")


def test_mixed_conversation_live_check_names_proof_path_for_missing_latency_samples() -> None:
    class SparseLatencyStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_sessions(self, limit: int = 20) -> list[dict[str, object]]:
            return [{"session_id": "sparse-latency-private-session SHOULD NOT APPEAR", "messages": 12}]

        def recent_messages(self, limit: int = 30, session_id: str | None = None) -> list[dict[str, object]]:
            return [
                _chat_message_row(
                    {
                        "source": "model",
                        "used_model": True,
                        "latency_ms": 900,
                        "model_error": "/\x55sers/example/private/sparse-latency SHOULD NOT APPEAR",
                    },
                    content="sparse latency answer SHOULD NOT APPEAR",
                ),
                _tool_message_row("research", toolset="browser"),
                _tool_message_row("web_lookup", toolset="browser"),
                _tool_message_row("fetch_page", toolset="browser"),
                _tool_message_row("list_events", toolset="personal"),
                _tool_message_row("check_availability", toolset="personal"),
                _tool_message_row("list_calendars", toolset="personal"),
                _tool_message_row("add_task", toolset="tasks"),
                _tool_message_row("task_board", toolset="tasks"),
                _tool_message_row("list_tasks", toolset="tasks"),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", SparseLatencyStore):
        ok, detail = live_check._check_mixed_conversation_health(SimpleNamespace(db_path="unused"))
    if ok is not None or "collect 2 more chat latency sample(s)" not in detail:
        raise SystemExit(f"sparse-latency mixed conversation should be skipped safely: {ok} {detail}")
    for expected in (
        "best recent session: 10 assistant turn(s)",
        "lanes chat=1, research=3, calendar=3, tasks=3",
        "latency samples 1",
        "metadata-only",
        "proof path: collect one session with chat, research, calendar, tasks",
        "at least 3 chat latency samples",
        "model routing status",
        "rerun the mixed conversation proof and live_check",
    ):
        if expected not in detail:
            raise SystemExit(f"sparse-latency proof detail missed {expected!r}: {detail}")
    for forbidden in (
        "sparse-latency-private-session",
        "sparse latency answer",
        "/\x55sers/operator",
        "sparse-latency",
        "SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(f"sparse-latency detail leaked seeded value {forbidden!r}: {detail}")


def test_mixed_conversation_live_check_flags_latency_regression_safely() -> None:
    class SlowStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_sessions(self, limit: int = 20) -> list[dict[str, object]]:
            return [{"session_id": "slow-session SHOULD NOT APPEAR", "messages": 24}]

        def recent_messages(self, limit: int = 30, session_id: str | None = None) -> list[dict[str, object]]:
            return [
                _chat_message_row({"source": "model", "used_model": True, "latency_ms": 9000}, content="slow one"),
                _chat_message_row({"source": "model", "used_model": True, "latency_ms": 10000}, content="slow two"),
                _chat_message_row({"source": "model", "used_model": True, "latency_ms": 11000}, content="slow three"),
                _tool_message_row("research", toolset="browser"),
                _tool_message_row("web_search", toolset="browser"),
                _tool_message_row("list_events", toolset="personal"),
                _tool_message_row("check_availability", toolset="personal"),
                _tool_message_row("add_task", toolset="tasks"),
                _tool_message_row("task_board", toolset="tasks"),
                _tool_message_row("weather", toolset="info"),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", SlowStore):
        ok, detail = live_check._check_mixed_conversation_health(SimpleNamespace(db_path="unused"))
    if ok is not False or "above 8000ms target" not in detail:
        raise SystemExit(f"mixed conversation slow window should fail safely: {ok} {detail}")
    for expected in (
        "best recent session: 10 assistant turn(s)",
        "latency samples 3",
        "chat p95",
        "metadata-only",
        "model routing status",
        "JARVIS_CHAT_MAX_REPLY_TOKENS",
        "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
        "rerun the mixed conversation proof and live_check",
    ):
        if expected not in detail:
            raise SystemExit(f"slow mixed conversation detail missed {expected!r}: {detail}")
    for forbidden in ("slow-session", "slow one", "slow two", "slow three", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"slow mixed conversation detail leaked seeded value {forbidden!r}: {detail}")


def test_personal_integration_live_check_reports_ready_without_secret_leakage() -> None:
    registry = _FakeRegistry(_expected_personal_risks())
    with (
        mock.patch("jarvis_v2.tools.registry.build_core_registry", return_value=registry),
        mock.patch("jarvis_v2.tools.calendar_connector._creds_file", return_value=_FakePath(True)),
        mock.patch("jarvis_v2.tools.calendar_connector._readonly_token_file", return_value=_FakePath(True)),
        mock.patch(
            "jarvis_v2.tools.email_connector._credential_status",
            return_value={
                "gmail_address": "the operator.secret@example.com SHOULD NOT APPEAR",
                "app_password": "SECRET SHOULD NOT APPEAR",
                "address_configured": True,
                "password_configured": True,
                "address_valid": True,
                "password_valid": True,
                "valid": True,
            },
        ),
        mock.patch("jarvis_v2.scripts.live_check.shutil.which", return_value="/\x55sers/example/private/osascript SHOULD NOT APPEAR"),
    ):
        ok, detail = live_check._check_personal_integration_readiness(SimpleNamespace(db_path=Path("/tmp/jarvis.sqlite"), obsidian_vault=Path("/tmp/vault"), obsidian_root="Jarvis"))
    if ok is not True:
        raise SystemExit(f"personal integration readiness should pass when gates and setup are ready: {ok} {detail}")
    for expected in (
        "tool gates ready: 15/15",
        "calendar OAuth ready",
        "Gmail valid",
        "reminders osascript present",
        "EXTERNAL_SIDE_EFFECT or HIGH_RISK and approval-gated",
        "no handlers run",
    ):
        if expected not in detail:
            raise SystemExit(f"personal integration detail missed {expected!r}: {detail}")
    for forbidden in ("the operator.secret@example.com", "SECRET SHOULD NOT APPEAR", "/\x55sers/operator", "osascript SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"personal integration readiness leaked secret/path detail {forbidden!r}: {detail}")


def test_personal_integration_live_check_reports_missing_setup_without_values() -> None:
    registry = _FakeRegistry(_expected_personal_risks())
    with (
        mock.patch("jarvis_v2.tools.registry.build_core_registry", return_value=registry),
        mock.patch("jarvis_v2.tools.calendar_connector._creds_file", return_value=_FakePath(False)),
        mock.patch("jarvis_v2.tools.calendar_connector._readonly_token_file", return_value=_FakePath(False)),
        mock.patch(
            "jarvis_v2.tools.email_connector._credential_status",
            return_value={
                "gmail_address": "/\x55sers/example/private/mail SHOULD NOT APPEAR",
                "app_password": "",
                "address_configured": True,
                "password_configured": False,
                "address_valid": False,
                "password_valid": False,
                "valid": False,
            },
        ),
        mock.patch("jarvis_v2.scripts.live_check.shutil.which", return_value=None),
    ):
        ok, detail = live_check._check_personal_integration_readiness(SimpleNamespace(db_path=Path("/tmp/jarvis.sqlite"), obsidian_vault=Path("/tmp/vault"), obsidian_root="Jarvis"))
    if ok is not None:
        raise SystemExit(f"missing personal integration setup should be not-yet-proven, not pass/fail hard: {ok} {detail}")
    for expected in (
        "tool gates ready: 15/15",
        "calendar OAuth missing OAuth file(s)",
        "Gmail missing env",
        "reminders osascript missing",
        "live proof still needs operator present",
    ):
        if expected not in detail:
            raise SystemExit(f"missing setup detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "private/mail", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"missing setup detail leaked seeded value {forbidden!r}: {detail}")


def test_personal_integration_live_check_flags_missing_or_wrong_risk_gates() -> None:
    missing_risks = _expected_personal_risks()
    missing_risks.pop("send_email")
    missing_registry = _FakeRegistry(missing_risks)
    with mock.patch("jarvis_v2.tools.registry.build_core_registry", return_value=missing_registry):
        ok, detail = live_check._check_personal_integration_readiness(SimpleNamespace(db_path=Path("/tmp/jarvis.sqlite"), obsidian_vault=Path("/tmp/vault"), obsidian_root="Jarvis"))
    if ok is not False or "missing 1 tool(s): send_email" not in detail:
        raise SystemExit(f"missing send_email gate should fail safely: {ok} {detail}")

    wrong_registry = _FakeRegistry(_expected_personal_risks(create_event=RiskLevel.LOCAL_SAFE))
    with mock.patch("jarvis_v2.tools.registry.build_core_registry", return_value=wrong_registry):
        ok, detail = live_check._check_personal_integration_readiness(SimpleNamespace(db_path=Path("/tmp/jarvis.sqlite"), obsidian_vault=Path("/tmp/vault"), obsidian_root="Jarvis"))
    if ok is not False or "wrong risk 1 tool(s): create_event:LOCAL_SAFE" not in detail:
        raise SystemExit(f"wrong create_event risk should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"wrong-risk detail leaked seeded value {forbidden!r}: {detail}")


def test_personal_proof_history_reports_ready_without_content_leakage() -> None:
    class FakeStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def recent_tool_runs(self, limit: int = 100) -> list[dict[str, object]]:
            if limit < live_check.PERSONAL_PROOF_RECENT_LIMIT:
                raise AssertionError("personal proof history should inspect the configured bounded window")
            return [
                _tool_run_row("list_events", metadata={"calendar_id": "evt-secret SHOULD NOT APPEAR"}),
                _tool_run_row("create_event", approved=True, metadata={"title": "doctor SHOULD NOT APPEAR"}),
                _tool_run_row("update_event", approved=True, metadata={"event_id": "evt-secret SHOULD NOT APPEAR"}),
                _tool_run_row("delete_event", approved=True, metadata={"event_id": "evt-secret SHOULD NOT APPEAR"}),
                _tool_run_row("read_emails", metadata={"sender": "the operator@example.com SHOULD NOT APPEAR"}),
                _tool_run_row("search_emails", metadata={"query": "private invoice SHOULD NOT APPEAR"}),
                _tool_run_row("send_email", approved=True, metadata={"to": "the operator@example.com SHOULD NOT APPEAR"}),
                _tool_run_row(
                    "set_reminder",
                    approved=True,
                    metadata={
                        "message": "secret reminder SHOULD NOT APPEAR",
                        "set_reminder_handoff": {
                            "status": "scheduled",
                            "message": "secret reminder SHOULD NOT APPEAR",
                        },
                    },
                ),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", FakeStore):
        ok, detail = live_check._check_personal_proof_history(SimpleNamespace(db_path="unused"))
    if ok is not True:
        raise SystemExit(f"personal proof history ready window should pass: {ok} {detail}")
    for expected in (
        "reviewed 8 personal audit row(s)",
        "calendar read=yes",
        "calendar write 3/3",
        "email read=yes",
        "email search/body=yes",
        "email send=yes",
        "approved set_reminder scheduling=yes",
        "content suppressed",
        "reminder phone delivery and target/content details still need operator acceptance review",
    ):
        if expected not in detail:
            raise SystemExit(f"personal proof ready detail missed {expected!r}: {detail}")
    for forbidden in (
        "SHOULD NOT APPEAR",
        "the operator@example.com",
        "private invoice",
        "doctor",
        "secret reminder",
        "evt-secret",
    ):
        if forbidden in detail:
            raise SystemExit(f"personal proof ready detail leaked seeded value {forbidden!r}: {detail}")

    class UnapprovedReminderStore(FakeStore):
        def recent_tool_runs(self, limit: int = 100) -> list[dict[str, object]]:
            rows = super().recent_tool_runs(limit)
            for row in rows:
                if row.get("tool_name") == "set_reminder":
                    row["approved"] = 0
            return rows

    with mock.patch("jarvis_v2.memory.store.MemoryStore", UnapprovedReminderStore):
        unapproved_ok, unapproved_detail = live_check._check_personal_proof_history(
            SimpleNamespace(db_path="unused")
        )
    if unapproved_ok is True or "approved set_reminder scheduling=no" not in unapproved_detail:
        raise SystemExit(
            "legacy unapproved reminder success must not satisfy live proof: "
            f"{unapproved_ok} {unapproved_detail}"
        )


def test_personal_proof_history_reports_missing_without_content_leakage() -> None:
    class FakeStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def recent_tool_runs(self, limit: int = 100) -> list[dict[str, object]]:
            return [
                _tool_run_row("list_events", metadata={"calendar_id": "private calendar SHOULD NOT APPEAR"}),
                _tool_run_row(
                    "send_email",
                    ok=False,
                    approved=True,
                    metadata={"to": "the operator@example.com SHOULD NOT APPEAR", "subject": "secret SHOULD NOT APPEAR"},
                ),
                _tool_run_row(
                    "create_event",
                    approved=False,
                    metadata={"title": "doctor appointment SHOULD NOT APPEAR"},
                ),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", FakeStore):
        ok, detail = live_check._check_personal_proof_history(SimpleNamespace(db_path="unused"))
    if ok is not None:
        raise SystemExit(f"personal proof missing window should be not-yet-proven: {ok} {detail}")
    for expected in (
        "reviewed 3 personal audit row(s)",
        "calendar read=yes",
        "calendar write 0/3",
        "email send=no",
        "failed/blocked rows 1",
        "not yet proven",
        "calendar create/update/delete",
        "email read/search",
        "approved set_reminder scheduling proof",
    ):
        if expected not in detail:
            raise SystemExit(f"personal proof missing detail missed {expected!r}: {detail}")
    for forbidden in (
        "SHOULD NOT APPEAR",
        "private calendar",
        "the operator@example.com",
        "secret",
        "doctor appointment",
    ):
        if forbidden in detail:
            raise SystemExit(f"personal proof missing detail leaked seeded value {forbidden!r}: {detail}")


def test_personal_proof_history_requires_set_reminder_not_macos_create_without_content_leakage() -> None:
    class FakeStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def recent_tool_runs(self, limit: int = 100) -> list[dict[str, object]]:
            return [
                _tool_run_row("list_events", metadata={"calendar_id": "evt-secret SHOULD NOT APPEAR"}),
                _tool_run_row("create_event", approved=True, metadata={"title": "doctor SHOULD NOT APPEAR"}),
                _tool_run_row("update_event", approved=True, metadata={"event_id": "evt-secret SHOULD NOT APPEAR"}),
                _tool_run_row("delete_event", approved=True, metadata={"event_id": "evt-secret SHOULD NOT APPEAR"}),
                _tool_run_row("read_emails", metadata={"sender": "the operator@example.com SHOULD NOT APPEAR"}),
                _tool_run_row("search_emails", metadata={"query": "private invoice SHOULD NOT APPEAR"}),
                _tool_run_row("send_email", approved=True, metadata={"to": "the operator@example.com SHOULD NOT APPEAR"}),
                _tool_run_row("create_reminder", approved=True, metadata={"title": "wrong path SHOULD NOT APPEAR"}),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", FakeStore):
        ok, detail = live_check._check_personal_proof_history(SimpleNamespace(db_path="unused"))
    if ok is not None:
        raise SystemExit(f"macOS create_reminder should not satisfy set_reminder proof: {ok} {detail}")
    for expected in (
        "reviewed 7 personal audit row(s)",
        "approved set_reminder scheduling=no",
        "not yet proven",
        "approved set_reminder scheduling proof",
    ):
        if expected not in detail:
            raise SystemExit(f"set_reminder proof detail missed {expected!r}: {detail}")
    for forbidden in ("wrong path", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"set_reminder proof detail leaked seeded value {forbidden!r}: {detail}")


def test_personal_proof_history_reports_empty_window_safely() -> None:
    class FakeStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def recent_tool_runs(self, limit: int = 100) -> list[dict[str, object]]:
            return [
                _tool_run_row("get_weather", metadata={"location": "Seoul SHOULD NOT APPEAR"}),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", FakeStore):
        ok, detail = live_check._check_personal_proof_history(SimpleNamespace(db_path="unused"))
    if ok is not None:
        raise SystemExit(f"personal proof empty window should be not-yet-proven: {ok} {detail}")
    for expected in ("reviewed 0 personal audit row(s)", "no personal live-proof audit rows yet", "operator present"):
        if expected not in detail:
            raise SystemExit(f"personal proof empty detail missed {expected!r}: {detail}")
    for forbidden in ("Seoul", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"personal proof empty detail leaked seeded value {forbidden!r}: {detail}")


def test_research_live_check_reports_ready_without_network_or_content_leakage() -> None:
    registry = _FakeRegistry(_expected_research_risks())
    with mock.patch("jarvis_v2.tools.registry.build_core_registry", return_value=registry):
        ok, detail = live_check._check_research_readiness(SimpleNamespace(db_path=Path("/tmp/jarvis.sqlite"), obsidian_vault=Path("/tmp/vault"), obsidian_root="Jarvis"))
    if ok is not True:
        raise SystemExit(f"research readiness should pass when routes and gates are ready: {ok} {detail}")
    for expected in (
        "tool gates ready: 5/5",
        "routes ready: 7/7",
        "READ_ONLY/LOCAL_SAFE",
        "no web fetch",
        "model call",
        "note write",
        "approval run",
    ):
        if expected not in detail:
            raise SystemExit(f"research readiness detail missed {expected!r}: {detail}")
    for forbidden in ("Ada Lovelace", "Mongolia", "/\x55sers/", "/private/", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"research readiness leaked query/content/path detail {forbidden!r}: {detail}")


def test_research_live_check_flags_missing_or_wrong_risk_gates() -> None:
    missing_risks = _expected_research_risks()
    missing_risks.pop("web_search")
    missing_registry = _FakeRegistry(missing_risks)
    with mock.patch("jarvis_v2.tools.registry.build_core_registry", return_value=missing_registry):
        ok, detail = live_check._check_research_readiness(SimpleNamespace(db_path=Path("/tmp/jarvis.sqlite"), obsidian_vault=Path("/tmp/vault"), obsidian_root="Jarvis"))
    if ok is not False or "missing 1 tool(s): web_search" not in detail:
        raise SystemExit(f"missing web_search gate should fail safely: {ok} {detail}")

    missing_extract_risks = _expected_research_risks()
    missing_extract_risks.pop("extract_links")
    missing_extract_registry = _FakeRegistry(missing_extract_risks)
    with mock.patch("jarvis_v2.tools.registry.build_core_registry", return_value=missing_extract_registry):
        ok, detail = live_check._check_research_readiness(SimpleNamespace(db_path=Path("/tmp/jarvis.sqlite"), obsidian_vault=Path("/tmp/vault"), obsidian_root="Jarvis"))
    if ok is not False or "missing 1 tool(s): extract_links" not in detail:
        raise SystemExit(f"missing extract_links gate should fail safely: {ok} {detail}")

    wrong_registry = _FakeRegistry(_expected_research_risks(research=RiskLevel.HIGH_RISK))
    with mock.patch("jarvis_v2.tools.registry.build_core_registry", return_value=wrong_registry):
        ok, detail = live_check._check_research_readiness(SimpleNamespace(db_path=Path("/tmp/jarvis.sqlite"), obsidian_vault=Path("/tmp/vault"), obsidian_root="Jarvis"))
    if ok is not False or "wrong risk 1 tool(s): research:HIGH_RISK" not in detail:
        raise SystemExit(f"wrong research risk should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"research wrong-risk detail leaked seeded value {forbidden!r}: {detail}")


def test_research_live_check_flags_route_drift_without_query_leakage() -> None:
    class WrongPlanner:
        def plan(self, _phrase: str):
            return SimpleNamespace(
                actions=[
                    SimpleNamespace(
                        tool_name="web_lookup",
                        args={"query": "/\x55sers/example/private/research-query SHOULD NOT APPEAR"},
                    )
                ]
            )

    registry = _FakeRegistry(_expected_research_risks())
    with (
        mock.patch("jarvis_v2.tools.registry.build_core_registry", return_value=registry),
        mock.patch("jarvis_v2.agent.planner.RuleBasedPlanner", WrongPlanner),
    ):
        ok, detail = live_check._check_research_readiness(SimpleNamespace(db_path=Path("/tmp/jarvis.sqlite"), obsidian_vault=Path("/tmp/vault"), obsidian_root="Jarvis"))
    if ok is not False:
        raise SystemExit(f"research route drift should fail safely: {ok} {detail}")
    for expected in ("route drift", "research", "web_lookup", "web_search", "fix research routing/tool gates"):
        if expected not in detail:
            raise SystemExit(f"research route-drift detail missed {expected!r}: {detail}")
    for forbidden in (
        "/\x55sers/operator",
        "private/research-query",
        "SHOULD NOT APPEAR",
        "Ada Lovelace",
        "Mongolia",
    ):
        if forbidden in detail:
            raise SystemExit(f"research route-drift detail leaked seeded value {forbidden!r}: {detail}")


def test_personal_route_live_check_reports_ready_without_handler_or_content_leakage() -> None:
    ok, detail = live_check._check_personal_route_readiness()
    if ok is not True:
        raise SystemExit(f"personal route readiness should pass in current wiring: {ok} {detail}")
    for expected in (
        "routes ready: 14/14",
        "direct personal paths 14",
        "dispatch-held risky paths 0",
        "do not fall to chat",
        "no handlers",
        "approvals",
        "account reads",
        "sends run",
    ):
        if expected not in detail:
            raise SystemExit(f"personal route readiness detail missed {expected!r}: {detail}")
    for forbidden in (
        "dentist",
        "Lunch with Sam",
        "alex@example.com",
        "invoice",
        "review approval safety",
        "/\x55sers/",
        "/private/",
        "SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(f"personal route readiness leaked command/content detail {forbidden!r}: {detail}")


def test_personal_route_live_check_flags_drift_without_content_leakage() -> None:
    class WrongPlanner:
        def plan(self, _phrase: str):
            return SimpleNamespace(
                actions=[
                    SimpleNamespace(
                        tool_name="read_emails",
                        args={"query": "/\x55sers/example/private/personal-route SHOULD NOT APPEAR"},
                    )
                ]
            )

    with mock.patch("jarvis_v2.agent.planner.RuleBasedPlanner", WrongPlanner):
        ok, detail = live_check._check_personal_route_readiness()
    if ok is not False:
        raise SystemExit(f"personal route drift should fail safely: {ok} {detail}")
    for expected in (
        "route drift",
        "send_email",
        "create_event",
        "arg drift",
        "read_emails",
        "fix personal command routing before live write proofs",
    ):
        if expected not in detail:
            raise SystemExit(f"personal route-drift detail missed {expected!r}: {detail}")
    for forbidden in (
        "/\x55sers/operator",
        "private/personal-route",
        "SHOULD NOT APPEAR",
        "dentist",
        "alex@example.com",
        "Lunch with Sam",
        "invoice",
    ):
        if forbidden in detail:
            raise SystemExit(f"personal route-drift detail leaked seeded value {forbidden!r}: {detail}")


def test_guardrail_control_plane_live_check_is_read_only_and_bounded() -> None:
    ok, detail = live_check._check_guardrail_control_plane()
    if ok is not True:
        raise SystemExit(f"guardrail control plane should pass in current wiring: {detail}")
    for expected in (
        "ready",
        "guardrails",
        "freeze",
        "cockpit",
        "live-proof freeze",
        "planner/send/call/HUD",
        "live-test-result",
        "live-result-reporting",
        "result-format",
    ):
        if expected not in detail:
            raise SystemExit(f"guardrail control plane detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "send_telegram", "approval #"):
        if forbidden in detail:
            raise SystemExit(f"guardrail control plane detail leaked unsafe fragment {forbidden!r}: {detail}")


def test_guardrail_control_plane_live_check_exception_is_redacted() -> None:
    seeded_error = "/\x55sers/example/private/guardrail-plane SECRET SHOULD NOT APPEAR"
    with mock.patch(
        "jarvis_v2.agent.command_suggest.suggest_command",
        side_effect=RuntimeError(seeded_error),
    ):
        ok, detail = live_check._check_guardrail_control_plane()
    if ok is not False or "RuntimeError" not in detail or "<local-path>" not in detail:
        raise SystemExit(f"guardrail control plane exception should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "guardrail-plane", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"guardrail control plane exception leaked seeded detail {forbidden!r}: {detail}")


def test_proof_ledger_control_plane_live_check_is_read_only_and_bounded() -> None:
    ok, detail = live_check._check_proof_ledger_control_plane()
    if ok is not True:
        raise SystemExit(f"proof ledger control plane should pass in current wiring: {detail}")
    for expected in ("ready", "proof", "evidence", "read-only ledger", "receipt surfaces"):
        if expected not in detail:
            raise SystemExit(f"proof ledger control plane detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "send_telegram", "approval #"):
        if forbidden in detail:
            raise SystemExit(f"proof ledger control plane detail leaked unsafe fragment {forbidden!r}: {detail}")


def test_proof_ledger_control_plane_live_check_exception_is_redacted() -> None:
    seeded_error = "/\x55sers/example/private/proof-ledger SECRET SHOULD NOT APPEAR"
    with mock.patch(
        "jarvis_v2.agent.command_suggest.suggest_command",
        side_effect=RuntimeError(seeded_error),
    ):
        ok, detail = live_check._check_proof_ledger_control_plane()
    if ok is not False or "RuntimeError" not in detail or "<local-path>" not in detail:
        raise SystemExit(f"proof ledger control plane exception should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "proof-ledger", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"proof ledger control plane exception leaked seeded detail {forbidden!r}: {detail}")


def test_completion_phone_gate_live_check_is_read_only_and_bounded() -> None:
    ok, detail = live_check._check_completion_phone_gate()
    if ok is not True:
        raise SystemExit(f"completion phone gate should pass in current wiring: {detail}")
    for expected in ("ready", "completion phone aliases", "claim gate", "blocked evidence", "no approval buttons"):
        if expected not in detail:
            raise SystemExit(f"completion phone gate detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "send_telegram", "approval #"):
        if forbidden in detail:
            raise SystemExit(f"completion phone gate detail leaked unsafe fragment {forbidden!r}: {detail}")


def test_completion_phone_gate_live_check_exception_is_redacted() -> None:
    seeded_error = "/\x55sers/example/private/completion-phone-gate SECRET SHOULD NOT APPEAR"
    with mock.patch(
        "jarvis_v2.automations.telegram_control.TelegramCommandBridge",
        side_effect=RuntimeError(seeded_error),
    ):
        ok, detail = live_check._check_completion_phone_gate()
    if ok is not False or "RuntimeError" not in detail or "<local-path>" not in detail:
        raise SystemExit(f"completion phone gate exception should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "completion-phone-gate", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"completion phone gate exception leaked seeded detail {forbidden!r}: {detail}")


def test_learning_recovery_control_plane_live_check_is_read_only_and_bounded() -> None:
    ok, detail = live_check._check_learning_recovery_control_plane()
    if ok is not True:
        raise SystemExit(f"learning/recovery control plane should pass in current wiring: {detail}")
    for expected in ("ready", "learning", "recovery", "read-only proof surfaces"):
        if expected not in detail:
            raise SystemExit(f"learning/recovery control plane detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "send_telegram", "approval #"):
        if forbidden in detail:
            raise SystemExit(f"learning/recovery control plane detail leaked unsafe fragment {forbidden!r}: {detail}")


def test_learning_recovery_control_plane_live_check_exception_is_redacted() -> None:
    seeded_error = "/\x55sers/example/private/learning-recovery SECRET SHOULD NOT APPEAR"
    with mock.patch(
        "jarvis_v2.agent.command_suggest.suggest_command",
        side_effect=RuntimeError(seeded_error),
    ):
        ok, detail = live_check._check_learning_recovery_control_plane()
    if ok is not False or "RuntimeError" not in detail or "<local-path>" not in detail:
        raise SystemExit(f"learning/recovery control plane exception should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "learning-recovery", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"learning/recovery control plane exception leaked seeded detail {forbidden!r}: {detail}")


def test_phone_approval_flow_live_check_is_read_only_and_bounded() -> None:
    ok, detail = live_check._check_phone_approval_flow()
    if ok is not True:
        raise SystemExit(f"phone approval flow should pass in current wiring: {detail}")
    for expected in (
        "ready",
        "phone approval buttons",
        "owner approve/deny callbacks",
        "fake runtime",
        "non-owner callback ignored",
        "read-only local check",
        "no live approval",
    ):
        if expected not in detail:
            raise SystemExit(f"phone approval flow detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "send_telegram", "approve:7", "deny:7", "approval #", "555001", "999999"):
        if forbidden in detail:
            raise SystemExit(f"phone approval flow detail leaked unsafe fragment {forbidden!r}: {detail}")


def test_phone_approval_flow_live_check_exception_is_redacted() -> None:
    seeded_error = "/\x55sers/example/private/phone-approval-flow SECRET SHOULD NOT APPEAR"
    with mock.patch(
        "jarvis_v2.automations.telegram_control.TelegramCommandBridge",
        side_effect=RuntimeError(seeded_error),
    ):
        ok, detail = live_check._check_phone_approval_flow()
    if ok is not False or "RuntimeError" not in detail or "<local-path>" not in detail:
        raise SystemExit(f"phone approval flow exception should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "phone-approval-flow", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"phone approval flow exception leaked seeded detail {forbidden!r}: {detail}")


def test_phone_control_center_live_check_is_read_only_and_bounded() -> None:
    ok, detail = live_check._check_phone_control_center()
    if ok is not True:
        raise SystemExit(f"phone control center should pass in current wiring: {detail}")
    for expected in (
        "ready",
        "phone control",
        "shortcuts",
        "voice",
        "capabilities",
        "agent research/Zoey",
        "safety",
        "privacy",
        "risk",
        "setup",
        "doctor",
        "memory/state",
        "learning loop",
        "progress/roadmap/goals",
        "AGI gates",
        "worker readiness/internal workers",
        "completion-blocker proof shortcuts",
        "live-test matrix/report-format packet",
        "personal-proof matrix/report-format packet",
        "scheduler-proof matrix/report-format packet",
        "approval-proof matrix/report-format packet",
        "phone-control proof matrix/report-format packet",
        "reboot-proof matrix/report-format packet",
        "mixed-conversation proof matrix/report-format packet",
        "voice-proof matrix/report-format packet",
        "acceptance harness/live proof status",
        "acceptance-gaps proof shortcuts",
        "acceptance-next proof shortcuts",
        "aggregate-smoke proof shortcuts",
        "reboot/daemon proof shortcuts",
        "voice-proof shortcuts",
        "compact proof view",
        "hidden proof-lane discovery",
        "delivery-status/did-it-send/call-connected",
        "what-can-you-do discovery",
        "trust checklist",
        "help discovery",
        "guardrails/freeze",
        "approval-held attention split",
        "trust/eval",
        "read-only",
    ):
        if expected not in detail:
            raise SystemExit(f"phone control center detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "send_telegram", "approval #", "transport failed"):
        if forbidden in detail:
            raise SystemExit(f"phone control center detail leaked unsafe fragment {forbidden!r}: {detail}")


def test_phone_control_center_live_check_exception_is_redacted() -> None:
    seeded_error = "/\x55sers/example/private/phone-control-center SECRET SHOULD NOT APPEAR"
    with mock.patch(
        "jarvis_v2.automations.telegram_control.TelegramCommandBridge",
        side_effect=RuntimeError(seeded_error),
    ):
        ok, detail = live_check._check_phone_control_center()
    if ok is not False or "RuntimeError" not in detail or "<local-path>" not in detail:
        raise SystemExit(f"phone control center exception should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "phone-control-center", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"phone control center exception leaked seeded detail {forbidden!r}: {detail}")


def test_voice_warmup_live_check_reports_opt_in_boundary() -> None:
    original_transcriber = os.environ.get("JARVIS_VOICE_WARMUP")
    try:
        os.environ.pop("JARVIS_VOICE_WARMUP", None)
        ok, detail = live_check._check_voice_warmup()
        if ok is not None or "disabled" not in detail or "JARVIS_VOICE_WARMUP=1" not in detail:
            raise SystemExit(f"disabled voice warmup should be a skipped opt-in row: {ok} {detail}")

        os.environ["JARVIS_VOICE_WARMUP"] = "1"
        with mock.patch("jarvis_v2.tools.voice._voice_local_transcriber_source", return_value="in_process_transcriber"):
            ok, detail = live_check._check_voice_warmup()
        if ok is not True:
            raise SystemExit(f"enabled voice warmup with a transcriber should pass: {detail}")
        for expected in ("enabled", "generated silence", "no microphone"):
            if expected not in detail:
                raise SystemExit(f"enabled voice warmup detail missed {expected!r}: {detail}")

        with mock.patch("jarvis_v2.tools.voice._voice_local_transcriber_source", return_value="not_configured"):
            ok, detail = live_check._check_voice_warmup()
        if ok is not False or "no local audio transcriber" not in detail:
            raise SystemExit(f"enabled warmup without a transcriber should fail clearly: {ok} {detail}")
    finally:
        if original_transcriber is None:
            os.environ.pop("JARVIS_VOICE_WARMUP", None)
        else:
            os.environ["JARVIS_VOICE_WARMUP"] = original_transcriber


def test_voice_live_check_suppresses_unknown_transcriber_source() -> None:
    seeded_source = "/\x55sers/example/private/whisper-model SHOULD NOT APPEAR"
    with mock.patch("jarvis_v2.tools.voice._voice_local_transcriber_source", return_value=seeded_source):
        ok, detail = live_check._check_voice_transcription()
    if ok is not False or "unknown transcriber source" not in detail:
        raise SystemExit(f"unknown transcriber source should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "private/whisper-model", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"unknown transcriber detail leaked seeded source {forbidden!r}: {detail}")

    original_warmup = os.environ.get("JARVIS_VOICE_WARMUP")
    try:
        os.environ["JARVIS_VOICE_WARMUP"] = "1"
        with mock.patch("jarvis_v2.tools.voice._voice_local_transcriber_source", return_value=seeded_source):
            ok, detail = live_check._check_voice_warmup()
    finally:
        if original_warmup is None:
            os.environ.pop("JARVIS_VOICE_WARMUP", None)
        else:
            os.environ["JARVIS_VOICE_WARMUP"] = original_warmup
    if ok is not False or "transcriber source is unknown" not in detail:
        raise SystemExit(f"unknown warmup transcriber source should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "private/whisper-model", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"unknown warmup detail leaked seeded source {forbidden!r}: {detail}")


def test_dashboard_voice_live_check_reuses_shared_hud_pipeline() -> None:
    ok, detail = live_check._check_dashboard_voice_wiring()
    if ok is not True:
        raise SystemExit(f"dashboard voice live check should pass in current wiring: {detail}")
    if "HudRecorder" not in detail or "stop_speech" not in detail:
        raise SystemExit(f"dashboard voice live check should name the shared helpers: {detail}")


def test_local_talk_live_check_reports_ready_without_mic_or_runtime() -> None:
    with mock.patch("jarvis_v2.scripts.live_check.shutil.which", return_value="/usr/local/bin/ffmpeg"):
        ok, detail = live_check._check_local_talk_readiness()
    if ok is not True:
        raise SystemExit(f"local talk live check should pass in current wiring: {detail}")
    for expected in (
        "ready",
        "static contract",
        "--speak",
        "--list-mics",
        "--mic",
        "default recording cap 120s",
        "approvals stay held",
        "no microphone",
        "ffmpeg",
        "transcription",
        "runtime turn",
        "speech",
        "account access",
    ):
        if expected not in detail:
            raise SystemExit(f"local talk detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "SECRET SHOULD NOT APPEAR", "clip.wav", "You:"):
        if forbidden in detail:
            raise SystemExit(f"local talk detail leaked implementation/private detail {forbidden!r}: {detail}")


def test_local_talk_live_check_requires_ffmpeg_before_live_mic_proof() -> None:
    with mock.patch("jarvis_v2.scripts.live_check.shutil.which", return_value=None):
        ok, detail = live_check._check_local_talk_readiness()
    if ok is not False:
        raise SystemExit(f"local talk without ffmpeg should fail readiness: {ok} {detail}")
    for expected in (
        "ffmpeg is unavailable",
        "install it",
        "rerun local talk readiness",
        "no microphone",
        "transcription",
        "runtime turn",
        "speech",
        "account access",
    ):
        if expected not in detail:
            raise SystemExit(f"ffmpeg readiness detail missed {expected!r}: {detail}")
    for forbidden in ("/usr/local", "/\x55sers/", "/private/", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"ffmpeg readiness detail leaked local/private state {forbidden!r}: {detail}")


def test_local_talk_live_check_flags_source_drift_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-talk-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "scripts"
        path.mkdir(parents=True)
        (path / "talk.py").write_text(
            "def broken():\n"
            "    return '/\x55sers/example/private/talk-source SECRET SHOULD NOT APPEAR'\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_local_talk_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"local talk drift should fail safely: {ok} {detail}")
    for expected in ("talk readiness drift", "argparse --speak", "runtime bridge", "approval hold"):
        if expected not in detail:
            raise SystemExit(f"local talk drift detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "private/talk-source", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"local talk drift detail leaked seeded value {forbidden!r}: {detail}")


def test_telegram_voice_live_check_reports_ready_without_live_voice_or_runtime() -> None:
    ok, detail = live_check._check_telegram_voice_readiness()
    if ok is not True:
        raise SystemExit(f"telegram voice live check should pass in current wiring: {detail}")
    for expected in (
        "ready",
        "static contract",
        "voice/audio",
        "20MB cap",
        "Telegram getFile",
        "temp voice file cleanup",
        "local transcriber",
        "safe transcript fallback",
        "heard receipt",
        "owner lock",
        "approval-gated runtime handoff",
        "no Telegram polling/download",
        "microphone",
        "transcription",
        "runtime turn",
        "account access",
        "send run",
    ):
        if expected not in detail:
            raise SystemExit(f"telegram voice detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "SECRET SHOULD NOT APPEAR", "file_id", "voice.oga", "🎙 Heard: "):
        if forbidden in detail:
            raise SystemExit(f"telegram voice detail leaked implementation/private detail {forbidden!r}: {detail}")


def test_telegram_voice_live_check_flags_source_drift_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-tg-voice-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "def broken():\n"
            "    return '/\x55sers/example/private/tg-voice-source SECRET SHOULD NOT APPEAR'\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_telegram_voice_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"telegram voice drift should fail safely: {ok} {detail}")
    for expected in (
        "telegram voice readiness drift",
        "voice/audio detection",
        "telegram getFile",
        "safe transcript fallback",
        "owner lock",
        "runtime handoff",
    ):
        if expected not in detail:
            raise SystemExit(f"telegram voice drift detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "private/tg-voice-source", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"telegram voice drift detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_reports_ready_without_leakage() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-guidance-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up on the Mac. "
            "Install the `whisper` CLI (or set JARVIS_VOICE_WHISPER_MODEL_PATH) "
            "and try again. /\x55sers/example/private/voice SECRET SHOULD NOT APPEAR\"\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not True:
        raise SystemExit(f"error guidance live check should pass with recovery contracts intact: {detail}")
    for expected in (
        "error guidance ready",
        "Calendar least-privilege root-launcher reauth",
        "missing setup",
        "generic recovery",
        "HTTP not-found",
        "service",
        "timeout",
        "Gmail",
        "reminders",
        "voice",
        "Ollama",
        "chat/planner/status",
        "common info connectors name service-specific fixes",
        "weather",
        "news",
        "currency",
        "translation",
        "sun",
        "history",
        "Wikipedia",
        "research",
        "dictionary",
        "air quality",
        "holidays",
        "markets",
        "jokes",
        "daily brief degraded-section guidance",
        "weather/calendar/reminders/news fixes",
        "compose/write/OCR",
        "Telegram voice setup guidance",
        "timezone data",
        "startup recovery audit storage",
        "missing runtime-trace metadata",
        "no account access",
        "web fetch",
        "model call",
        "transcription",
        "approval",
        "typing",
        "pasting",
        "OCR execution",
        "send run",
    ):
        if expected not in detail:
            raise SystemExit(f"error guidance detail missed {expected!r}: {detail}")
    for forbidden in (
        "/\x55sers/operator",
        "private/voice",
        "SECRET SHOULD NOT APPEAR",
        "Bad Request",
        "HTTP Error",
        "market backend",
        "joke backend",
        "SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(f"error guidance detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_matches_actual_repository() -> None:
    blocked_call = mock.Mock(side_effect=AssertionError("error-guidance proof attempted I/O"))
    with (
        mock.patch.object(Path, "read_text", blocked_call),
        mock.patch("builtins.open", blocked_call),
        mock.patch("socket.socket", blocked_call),
        mock.patch("subprocess.run", blocked_call),
        mock.patch("subprocess.Popen", blocked_call),
        mock.patch("urllib.request.urlopen", blocked_call),
    ):
        ok, detail = live_check._check_error_guidance_readiness(
            root=live_check.REPO_ROOT
        )
    if ok is not True:
        raise SystemExit(
            "error guidance synthetic fixtures masked actual repository drift: "
            f"{detail}"
        )
    for expected in (
        "error guidance ready: 55 recovery contract(s)",
        "reminders",
        "voice",
        "Ollama",
        "name a next fix",
        "no account access",
        "send run",
    ):
        if expected not in detail:
            raise SystemExit(
                f"actual repository error-guidance proof missed {expected!r}: {detail}"
            )
    for forbidden in (
        *live_check.ERROR_GUIDANCE_FORBIDDEN_FRAGMENTS,
        "GMAIL_APP_PASSWORD",
        "TELEGRAM_BOT_TOKEN",
        "SECRET SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(
                f"actual repository error-guidance proof leaked {forbidden!r}: {detail}"
            )


def test_error_guidance_live_check_rejects_contract_count_shrink() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-guidance-count-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
            "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
            encoding="utf-8",
        )
        with mock.patch.object(
            live_check,
            "ERROR_GUIDANCE_EXPECTED_CONTRACT_COUNT",
            56,
        ):
            ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"error-guidance contract shrink should fail closed: {ok} {detail}")
    for expected in ("error guidance drift", "recovery coverage count changed", "55/56"):
        if expected not in detail:
            raise SystemExit(
                f"error-guidance contract shrink detail missed {expected!r}: {detail}"
            )
    for forbidden in ("/\x55sers/", "/private/", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(
                f"error-guidance contract shrink leaked {forbidden!r}: {detail}"
            )


def test_error_guidance_inventory_rejects_same_count_substitution() -> None:
    labels = tuple(sorted(live_check.ERROR_GUIDANCE_EXPECTED_LABELS))
    substituted = labels[:-1] + (labels[0],)
    problem = live_check._error_guidance_inventory_problem(substituted)
    if problem is None:
        raise SystemExit("same-count recovery-contract substitution should fail closed")
    for expected in ("duplicate recovery contract", "missing recovery contract"):
        if expected not in problem:
            raise SystemExit(
                f"recovery-contract substitution detail missed {expected!r}: {problem}"
            )


def test_error_guidance_live_check_requires_visible_recovery_wording() -> None:
    from jarvis_v2.tools import tasks as tasks_module

    actual = tasks_module._missing_task_result(
        tool_name="inspect_task",
        task_id=987654,
        mutation="task_read",
        retry_command="show task <correct task id>",
    )
    hidden_guidance = SimpleNamespace(
        output="Task not found.",
        metadata=actual.metadata,
    )
    with mock.patch.object(
        tasks_module,
        "_missing_task_result",
        return_value=hidden_guidance,
    ):
        ok, detail = live_check._check_error_guidance_readiness()
    if ok is not False:
        raise SystemExit(f"metadata-only recovery wording should fail closed: {ok} {detail}")
    for expected in ("error guidance drift", "Task-id not-found guidance visible output", "missing"):
        if expected not in detail:
            raise SystemExit(
                f"metadata-only recovery detail missed {expected!r}: {detail}"
            )


def test_error_guidance_live_check_flags_recovery_metadata_drift() -> None:
    from jarvis_v2.tools import tasks as tasks_module

    unsafe_result = SimpleNamespace(
        output="Task not found. Run `show task <correct task id>` before retry.",
        metadata={
            "retry_requires_task_refresh": False,
            "authorizes_retry": True,
        },
    )
    with mock.patch.object(
        tasks_module,
        "_missing_task_result",
        return_value=unsafe_result,
    ):
        ok, detail = live_check._check_error_guidance_readiness()
    if ok is not False:
        raise SystemExit(f"recovery metadata drift should fail closed: {ok} {detail}")
    for expected in ("error guidance drift", "Task-id not-found guidance", "missing"):
        if expected not in detail:
            raise SystemExit(
                f"recovery metadata drift detail missed {expected!r}: {detail}"
            )
    for forbidden in ("/\x55sers/", "/private/", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(
                f"recovery metadata drift leaked {forbidden!r}: {detail}"
            )


def test_error_guidance_live_check_checks_each_combined_result() -> None:
    from jarvis_v2.tools import tasks as tasks_module

    original = tasks_module._note_task_not_found_result

    def one_unsafe_result(*, tool_name: str, raw_path: str, metadata=None):
        result = original(tool_name=tool_name, raw_path=raw_path, metadata=metadata)
        if tool_name != "import_tasks_from_note":
            return result
        unsafe_metadata = dict(result.metadata)
        unsafe_metadata["retry_requires_note_refresh"] = False
        unsafe_metadata["authorizes_retry"] = True
        return SimpleNamespace(output=result.output, metadata=unsafe_metadata)

    with mock.patch.object(
        tasks_module,
        "_note_task_not_found_result",
        side_effect=one_unsafe_result,
    ):
        ok, detail = live_check._check_error_guidance_readiness()
    if ok is not False:
        raise SystemExit(f"combined recovery metadata drift should fail closed: {ok} {detail}")
    for expected in (
        "error guidance drift",
        "Note-to-task not-found guidance result 2 recovery metadata",
        "missing",
    ):
        if expected not in detail:
            raise SystemExit(
                f"combined recovery metadata drift detail missed {expected!r}: {detail}"
            )


def test_error_guidance_live_check_flags_calendar_drift_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-guidance-calendar-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
            "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
            encoding="utf-8",
        )
        with mock.patch(
            "jarvis_v2.tools.calendar_connector._calendar_error",
            return_value=(
                f"{V3_CALENDAR_AUTH_READONLY_COMMAND} {V3_CALENDAR_AUTH_TERMINAL_LOCATION} "
                "google_calendar_reauth Retrying will not help not connected yet Google credentials run setup "
            ),
        ):
            ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"calendar guidance drift should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "calendar dead-token guidance", "leaked raw detail"):
        if expected not in detail:
            raise SystemExit(f"calendar guidance drift detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "google_calendar_reauth"):
        if forbidden in detail:
            raise SystemExit(f"calendar guidance drift detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_rejects_broadened_calendar_auth_safely() -> None:
    from jarvis_v2.tools import brief_tools as brief_module

    with TemporaryDirectory(prefix="jarvis-live-check-guidance-calendar-broad-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
            "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
            encoding="utf-8",
        )
        original = brief_module._daily_brief_recovery_line(["weather", "calendar", "reminders", "news"])
        with mock.patch.object(
            brief_module,
            "_daily_brief_recovery_line",
            lambda unavailable: f"{original} {V3_CALENDAR_AUTH_FULL_ACCESS_COMMAND}",
        ):
            ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"broadened Calendar auth guidance should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "daily brief degraded-section guidance", "leaked raw detail"):
        if expected not in detail:
            raise SystemExit(f"broadened Calendar auth detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "full-access"):
        if forbidden in detail:
            raise SystemExit(f"broadened Calendar auth detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_flags_calendar_generic_drift_safely() -> None:
    def calendar_error_fixture(exc: Exception) -> str:
        text = str(exc)
        if "invalid_grant" in text:
            return (
                f"{V3_CALENDAR_AUTH_READONLY_COMMAND} "
                f"{V3_CALENDAR_AUTH_TERMINAL_LOCATION} Retrying will not help"
            )
        if "Google credentials not found" in text:
            return "not connected yet Google credentials run setup"
        return "Google Calendar is having trouble. Try again later."

    with TemporaryDirectory(prefix="jarvis-live-check-guidance-calendar-generic-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
            "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
            encoding="utf-8",
        )
        with mock.patch("jarvis_v2.tools.calendar_connector._calendar_error", calendar_error_fixture):
            ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"calendar generic guidance drift should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "calendar generic recovery guidance", "missing"):
        if expected not in detail:
            raise SystemExit(f"calendar generic guidance drift detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "SHOULD NOT APPEAR", "raw calendar"):
        if forbidden in detail:
            raise SystemExit(f"calendar generic guidance drift detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_flags_voice_setup_drift_safely() -> None:
    from jarvis_v2.automations import telegram_control as telegram_control_module

    with mock.patch.object(
        telegram_control_module,
        "_telegram_voice_setup_guidance",
        lambda: "/\x55sers/example/private/voice-setup SECRET SHOULD NOT APPEAR",
    ):
        ok, detail = live_check._check_error_guidance_readiness()
    if ok is not False:
        raise SystemExit(f"voice guidance drift should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "telegram voice setup guidance", "missing"):
        if expected not in detail:
            raise SystemExit(f"voice guidance drift detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "voice-setup", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"voice guidance drift detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_flags_ollama_fallback_drift_safely() -> None:
    from jarvis_v2.agent import chat as chat_module

    with mock.patch.object(
        chat_module.ChatBrain,
        "_fallback_response",
        lambda self, *args, **kwargs: (
            "/\x55sers/example/private/model SECRET SHOULD NOT APPEAR"
        ),
    ):
        ok, detail = live_check._check_error_guidance_readiness()
    if ok is not False:
        raise SystemExit(f"Ollama fallback guidance drift should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "Ollama chat fallback guidance", "missing"):
        if expected not in detail:
            raise SystemExit(f"Ollama guidance drift detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "model SECRET", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"Ollama guidance drift detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_flags_planner_fallback_drift_safely() -> None:
    """"Ollama planner fallback guidance" is behavior-based (calls the real,
    actually-installed _model_exception_recovery_hint), so drift is simulated by
    monkeypatching that function's return value, not by writing a fixture file."""
    from jarvis_v2.agent import model_planner as model_planner_module

    with TemporaryDirectory(prefix="jarvis-live-check-guidance-planner-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
            "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
            encoding="utf-8",
        )
        with mock.patch.object(
            model_planner_module,
            "_model_exception_recovery_hint",
            lambda model: "/\x55sers/example/private/planner SECRET SHOULD NOT APPEAR",
        ):
            ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"Ollama planner fallback guidance drift should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "Ollama planner fallback guidance", "missing"):
        if expected not in detail:
            raise SystemExit(f"Ollama planner guidance drift detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "planner SECRET", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"Ollama planner guidance drift detail leaked seeded value {forbidden!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "planner SECRET", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"Ollama planner guidance drift detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_flags_user_visible_recovery_drift_safely() -> None:
    from jarvis_v2.tools import apple_reminders, model_status, voice

    cases = (
        (
            apple_reminders,
            "_apple_reminders_recovery_message",
            "Apple Reminders permission guidance",
        ),
        (voice, "_voice_error", "voice recovery guidance"),
        (model_status, "_ollama_unreachable_guidance", "Ollama status guidance"),
    )
    for module, attribute, label in cases:
        with TemporaryDirectory(prefix="jarvis-live-check-guidance-behavior-") as temp:
            root = Path(temp)
            path = root / "jarvis_v2" / "automations"
            path.mkdir(parents=True)
            (path / "telegram_control.py").write_text(
                "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
                "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                module,
                attribute,
                lambda *args, **kwargs: "/\x55sers/example/private/recovery SECRET SHOULD NOT APPEAR",
            ):
                ok, detail = live_check._check_error_guidance_readiness(root=root)
        if ok is not False:
            raise SystemExit(f"{label} behavior drift should fail safely: {ok} {detail}")
        for expected in ("error guidance drift", label):
            if expected not in detail:
                raise SystemExit(f"{label} behavior drift detail missed {expected!r}: {detail}")
        if "missing" not in detail and "leaked raw detail" not in detail:
            raise SystemExit(f"{label} behavior drift did not name the failed contract: {detail}")
        for forbidden in ("/\x55sers/operator", "private/recovery", "SECRET SHOULD NOT APPEAR"):
            if forbidden in detail:
                raise SystemExit(f"{label} behavior drift leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_flags_compose_guidance_drift_safely() -> None:
    from jarvis_v2.tools import compose_connector as compose_module

    with TemporaryDirectory(prefix="jarvis-live-check-guidance-compose-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
            "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
            encoding="utf-8",
        )
        with mock.patch.object(
            compose_module,
            "_compose_model_recovery_hint",
            lambda config: "/\x55sers/example/private/compose-model SECRET SHOULD NOT APPEAR",
        ):
            ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"compose guidance drift should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "compose-and-write model guidance", "missing"):
        if expected not in detail:
            raise SystemExit(f"compose guidance drift detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "compose-model SECRET", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"compose guidance drift detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_flags_daily_brief_guidance_drift_safely() -> None:
    from jarvis_v2.tools import brief_tools as brief_module

    with TemporaryDirectory(prefix="jarvis-live-check-guidance-brief-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
            "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
            encoding="utf-8",
        )
        with mock.patch.object(
            brief_module,
            "_daily_brief_recovery_line",
            lambda unavailable: "/\x55sers/example/private/daily-brief SECRET SHOULD NOT APPEAR",
        ):
            ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"daily brief guidance drift should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "daily brief degraded-section guidance", "missing"):
        if expected not in detail:
            raise SystemExit(f"daily brief guidance drift detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "daily-brief SECRET", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"daily brief guidance drift detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_flags_writer_guidance_drift_safely() -> None:
    from jarvis_v2.tools import writer_connector as writer_module

    with TemporaryDirectory(prefix="jarvis-live-check-guidance-writer-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
            "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
            encoding="utf-8",
        )
        with mock.patch.object(
            writer_module,
            "_write_failure_message",
            lambda mode: "/private/tmp/writer-accessibility SECRET SHOULD NOT APPEAR",
        ):
            ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"writer guidance drift should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "writer type recovery guidance", "missing"):
        if expected not in detail:
            raise SystemExit(f"writer guidance drift detail missed {expected!r}: {detail}")
    for forbidden in ("/private/tmp", "writer-accessibility", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"writer guidance drift detail leaked seeded value {forbidden!r}: {detail}")


def test_error_guidance_live_check_flags_ocr_guidance_drift_safely() -> None:
    from jarvis_v2.tools import ocr as ocr_module

    with TemporaryDirectory(prefix="jarvis-live-check-guidance-ocr-") as temp:
        root = Path(temp)
        path = root / "jarvis_v2" / "automations"
        path.mkdir(parents=True)
        (path / "telegram_control.py").write_text(
            "MESSAGE = \"Voice transcription isn't set up. Install the `whisper` CLI "
            "or set JARVIS_VOICE_WHISPER_MODEL_PATH and try again.\"\n",
            encoding="utf-8",
        )
        with mock.patch.object(
            ocr_module,
            "_ocr_unavailable_message",
            lambda: "/var/folders/ocr-swift SECRET SHOULD NOT APPEAR",
        ):
            ok, detail = live_check._check_error_guidance_readiness(root=root)
    if ok is not False:
        raise SystemExit(f"OCR guidance drift should fail safely: {ok} {detail}")
    for expected in ("error guidance drift", "OCR unavailable guidance", "missing"):
        if expected not in detail:
            raise SystemExit(f"OCR guidance drift detail missed {expected!r}: {detail}")
    for forbidden in ("/var/folders", "ocr-swift", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"OCR guidance drift detail leaked seeded value {forbidden!r}: {detail}")


def test_aggregate_smoke_proof_live_check_reports_logged_green_without_running_tests() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-aggregate-") as temp:
        root = Path(temp)
        _write_aggregate_smoke_source(root, datetime(2026, 7, 8, 10, 45, tzinfo=UTC))
        (root / "CODEX_TASKS.md").write_text(
            "# CODEX_TASKS\n\n"
            "private note /\x55sers/example/private/smoke SECRET SHOULD NOT APPEAR\n"
            "- 2026-07-08 19:46 KST (Codex): DONE WS7/WS6 planner audit; "
            "python3 -m jarvis_v2.scripts.smoke_test_all green (110 modules, 196.0s).\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_aggregate_smoke_proof(root=root)

    if ok is not True:
        raise SystemExit(f"aggregate smoke proof should pass with logged green proof: {detail}")
    for expected in ("aggregate smoke proof ready", "smoke_test_all", "110 modules", "duration 196.0s", "read-only", "did not run tests"):
        if expected not in detail:
            raise SystemExit(f"aggregate smoke proof detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "private/smoke", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"aggregate smoke proof detail leaked seeded value {forbidden!r}: {detail}")


def _write_aggregate_smoke_source(root: Path, modified_at: datetime) -> None:
    source = root / "jarvis_v2" / "scripts" / "smoke_test_all.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# aggregate smoke source fixture\n", encoding="utf-8")
    timestamp = modified_at.timestamp()
    os.utime(source, (timestamp, timestamp))


def test_aggregate_smoke_proof_live_check_requires_timestamped_evidence() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-aggregate-tail-") as temp:
        root = Path(temp)
        _write_aggregate_smoke_source(root, datetime(2026, 7, 8, 10, 45, tzinfo=UTC))
        (root / "CODEX_TASKS.md").write_text(
            "All 112 smoke modules passed. Duration: 201.5s\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_aggregate_smoke_proof(root=root)

    if ok is not None:
        raise SystemExit(f"aggregate smoke proof without a timestamp should stay unproven: {detail}")
    if "timestamp missing" not in detail or "log a KST timestamp" not in detail:
        raise SystemExit(f"aggregate timestamp-missing detail was incomplete: {detail}")


def test_aggregate_smoke_proof_live_check_rejects_stale_evidence() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-aggregate-stale-") as temp:
        root = Path(temp)
        _write_aggregate_smoke_source(root, datetime(2026, 7, 8, 10, 47, tzinfo=UTC))
        (root / "CODEX_TASKS.md").write_text(
            "- 2026-07-08 19:46 KST (Codex): "
            "python3 -m jarvis_v2.scripts.smoke_test_all green (112 modules, 201.5s).\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_aggregate_smoke_proof(root=root)

    if ok is not False:
        raise SystemExit(f"aggregate smoke proof older than source should fail: {detail}")
    if "predates current source" not in detail or "rerun the aggregate suite before DONE" not in detail:
        raise SystemExit(f"aggregate stale-proof detail was incomplete: {detail}")


def test_aggregate_smoke_proof_live_check_flags_missing_and_low_counts_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-aggregate-missing-") as temp:
        root = Path(temp)
        (root / "CODEX_TASKS.md").write_text(
            "No green aggregate proof yet. /private/tmp/secret SHOULD NOT APPEAR\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_aggregate_smoke_proof(root=root)
    if ok is not False:
        raise SystemExit(f"missing aggregate smoke proof should fail safely: {ok} {detail}")
    for expected in ("aggregate smoke proof missing", "smoke_test_all", "run the aggregate suite before DONE"):
        if expected not in detail:
            raise SystemExit(f"missing aggregate smoke detail missed {expected!r}: {detail}")
    for forbidden in ("/private/tmp", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"missing aggregate smoke detail leaked seeded value {forbidden!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-aggregate-low-") as temp:
        root = Path(temp)
        (root / "CODEX_TASKS.md").write_text(
            "- 2026-07-08 19:20 KST (Codex): python3 -m jarvis_v2.scripts.smoke_test_all green (94 modules, 88.1s).\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_aggregate_smoke_proof(root=root)
    if ok is not False:
        raise SystemExit(f"low aggregate smoke proof should fail safely: {ok} {detail}")
    for expected in ("aggregate smoke proof below target", "94 module(s)", "at least 110", "rerun the aggregate suite before DONE"):
        if expected not in detail:
            raise SystemExit(f"low aggregate smoke detail missed {expected!r}: {detail}")


def test_aggregate_smoke_proof_picks_the_latest_timestamp_despite_handoff_order() -> None:
    """Proof freshness must not depend on a handoff preserving prepend order."""
    with TemporaryDirectory(prefix="jarvis-live-check-aggregate-order-") as temp:
        root = Path(temp)
        _write_aggregate_smoke_source(root, datetime(2026, 7, 8, 12, 27, tzinfo=UTC))
        (root / "CODEX_TASKS.md").write_text(
            "# CODEX_TASKS\n\n"
            "## Autonomous Maintenance Log\n"
            "- 2026-07-08 21:28 KST (Codex): older entry placed first; "
            "python3 -m jarvis_v2.scripts.smoke_test_all green (110 modules, 240.2s).\n"
            "- 2026-07-08 21:29 KST (Codex): newer handoff proof placed second; "
            "python3 -m jarvis_v2.scripts.smoke_test_all green (111 modules, 241.3s).\n"
            "- 2026-07-06 14:49 KST (Codex): much older entry; "
            "python3 -m jarvis_v2.scripts.smoke_test_all green, 90 modules.\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_aggregate_smoke_proof(root=root)
    if ok is not True:
        raise SystemExit(f"aggregate smoke proof should pass using the newest timestamp: {detail}")
    if "111 modules" not in detail or "duration 241.3s" not in detail:
        raise SystemExit(f"aggregate smoke proof picked the wrong timestamped entry: {detail}")
    if "110 modules" in detail or "90 modules" in detail or "240.2s" in detail:
        raise SystemExit(f"aggregate smoke proof leaked an older entry's numbers: {detail}")


def _write_operator_eval_sources(root: Path, *, omit_token: str = "") -> None:
    script_dir = root / "jarvis_v2" / "scripts"
    script_dir.mkdir(parents=True)
    fragments: list[str] = []
    for tokens in live_check.OPERATOR_WORKFLOW_EVAL_REQUIRED_TEXT.values():
        fragments.extend(tokens)
    if omit_token:
        fragments = [fragment for fragment in fragments if fragment != omit_token]
    (script_dir / "smoke_test_operator_evals.py").write_text(
        "\n".join(fragments)
        + "\n/\x55sers/example/private/eval-source SECRET SHOULD NOT APPEAR\n",
        encoding="utf-8",
    )


def test_operator_workflow_eval_proof_reports_logged_green_without_running_evals() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-operator-evals-") as temp:
        root = Path(temp)
        _write_operator_eval_sources(root)
        (root / "CODEX_TASKS.md").write_text(
            "# CODEX_TASKS\n\n"
            "private note /\x55sers/example/private/eval-log SECRET SHOULD NOT APPEAR\n"
            "- 2026-07-08 23:21 KST (Codex): focused "
            "`python3 -m jarvis_v2.scripts.smoke_test_operator_evals` green; "
            "aggregate smoke stayed green.\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_operator_workflow_eval_proof(root=root)

    if ok is not True:
        raise SystemExit(f"operator workflow eval proof should pass with logged green proof: {detail}")
    for expected in (
        "operator workflow eval proof ready",
        "operator-real workflow lane(s)",
        "smoke_test_operator_evals",
        "latest logged focused proof is green",
        "read-only source/log check only",
        "did not run evals",
        "send",
        "approve",
        "real contacts",
    ):
        if expected not in detail:
            raise SystemExit(f"operator workflow eval proof detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "private/eval", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"operator workflow eval proof detail leaked seeded value {forbidden!r}: {detail}")


def test_operator_workflow_eval_proof_flags_missing_or_drift_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-operator-evals-missing-") as temp:
        root = Path(temp)
        _write_operator_eval_sources(root)
        (root / "CODEX_TASKS.md").write_text(
            "No focused eval proof yet. /private/tmp/eval SECRET SHOULD NOT APPEAR\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_operator_workflow_eval_proof(root=root)
    if ok is not False:
        raise SystemExit(f"missing operator workflow proof should fail safely: {ok} {detail}")
    for expected in ("operator workflow eval proof missing", "smoke_test_operator_evals", "run the focused eval pack before DONE"):
        if expected not in detail:
            raise SystemExit(f"missing operator eval proof detail missed {expected!r}: {detail}")
    for forbidden in ("/private/tmp", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"missing operator eval proof detail leaked seeded value {forbidden!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-operator-evals-drift-") as temp:
        root = Path(temp)
        _write_operator_eval_sources(root, omit_token="queued approval is the SUCCESS state")
        (root / "CODEX_TASKS.md").write_text(
            "- 2026-07-08 23:21 KST (Codex): "
            "python3 -m jarvis_v2.scripts.smoke_test_operator_evals green.\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_operator_workflow_eval_proof(root=root)
    if ok is not False:
        raise SystemExit(f"drifted operator workflow eval pack should fail safely: {ok} {detail}")
    for expected in ("operator workflow eval pack drift", "contract(s) missing", "repair smoke_test_operator_evals"):
        if expected not in detail:
            raise SystemExit(f"drifted operator eval proof detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "queued approval is the SUCCESS state", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"drifted operator eval proof detail leaked seeded value {forbidden!r}: {detail}")


def _write_acceptance_plan_sections(root: Path, *, omit: str = "") -> None:
    sections = (
        "A. Conversation & research",
        "B. Channels",
        "C. Personal integrations",
        "D. Voice",
        "E. Autonomy & daily value",
        "F. Reliability",
    )
    lines = ["# Jarvis V2 Finish Plan\n"]
    for section in sections:
        if section == omit:
            continue
        lines.append(f"### {section}\n")
        for anchor, _rows in live_check.ACCEPTANCE_HARNESS_ITEM_ROWS[section]:
            lines.append(f"- [ ] {anchor} fixture item\n")
    lines.append("/\x55sers/example/private/plan SECRET SHOULD NOT APPEAR\n")
    (root / "FINISH_PLAN_AUGUST.md").write_text("\n".join(lines), encoding="utf-8")


def test_acceptance_harness_coverage_reports_section_map_without_live_actions() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not True:
        raise SystemExit(f"acceptance coverage should pass with all DoD sections mapped: {detail}")
    personal_rows = dict(live_check.ACCEPTANCE_HARNESS_ITEM_ROWS["C. Personal integrations"])
    if personal_rows.get("Contacts lookup") != ("macOS Contacts",):
        raise SystemExit(
            "Contacts DoD must use its direct read-only Contacts evidence, not the unrelated personal audit-history row"
        )
    channel_rows = dict(live_check.ACCEPTANCE_HARNESS_ITEM_ROWS["B. Channels"])
    if channel_rows.get("Calls (phone/FaceTime") != (
        "call readiness",
        "channel health",
    ):
        raise SystemExit(
            "Calls DoD must use the no-call target/gate preflight plus content-suppressed health"
        )
    for expected in (
        "acceptance coverage ready",
        "6/6 DoD section(s)",
        "diagnostic row(s)",
        "28/28 DoD checklist item(s)",
        "pinned to existing row evidence",
        "conversation/research",
        "channels",
        "channel health, send-resolution preflight, call-target readiness, and Telegram-owner readiness",
        "personal integrations",
        "voice",
        "daily value",
        "reliability",
        "one-screen table budget",
        "direct Contacts lookup coverage",
        "scheduled-job freshness, 7-day proof, and phone approval-button flow",
        "acceptance coverage, open-gap visibility, next-proof guidance",
        "aggregate smoke proof",
        "operator-real workflow eval proof",
        "error-guidance recovery checks",
        "guardrail-plane shortcuts",
        "proof-ledger shortcuts",
        "completion-claim gating",
        "learning/recovery loop checks",
        "read-only coverage map",
        "not a live proof",
        "operator-gated",
        "sends/calls/approvals/microphone/reboot",
    ):
        if expected not in detail:
            raise SystemExit(f"acceptance coverage detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "private/plan", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"acceptance coverage detail leaked seeded value {forbidden!r}: {detail}")


def test_acceptance_harness_coverage_flags_plan_or_row_drift_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-missing-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root, omit="D. Voice")
        ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing acceptance section should fail safely: {ok} {detail}")
    for expected in ("acceptance coverage drift", "missing DoD section", "D. Voice", "repair live_check coverage"):
        if expected not in detail:
            raise SystemExit(f"missing acceptance section detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "private/plan", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"missing acceptance section detail leaked seeded value {forbidden!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-row-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        with mock.patch.object(
            live_check,
            "ACCEPTANCE_HARNESS_SECTION_ROWS",
            {"B. Channels": ("missing channel row SHOULD NOT APPEAR",)},
        ):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing acceptance row mapping should fail safely: {ok} {detail}")
    for expected in ("acceptance coverage drift", "missing live_check row mapping", "B. Channels", "repair live_check coverage"):
        if expected not in detail:
            raise SystemExit(f"missing acceptance row detail missed {expected!r}: {detail}")
    if "SHOULD NOT APPEAR" in detail:
        raise SystemExit(f"missing acceptance row detail leaked seeded value: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-send-resolution-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_send_resolution = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "send resolution"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_send_resolution):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing send resolution row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "B. Channels",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing send resolution row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-telegram-owner-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_telegram_owner = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "Telegram owner"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_telegram_owner):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing Telegram owner row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "B. Channels",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing Telegram owner row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-job-proof-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_streak = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "jobs 7-day proof"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_streak):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing jobs 7-day proof row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "E. Autonomy & daily value",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing jobs 7-day row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-phone-approvals-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_phone_approvals = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "phone approvals"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_phone_approvals):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing phone approvals row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "E. Autonomy & daily value",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing phone approvals row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-operator-evals-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_operator_evals = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "operator evals"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_operator_evals):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing operator evals row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "F. Reliability",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing operator evals row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-contacts-coverage-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_contacts = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "macOS Contacts"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_contacts):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing macOS Contacts row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "C. Personal integrations",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing Contacts row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-self-coverage-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_gap_row = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "acceptance gaps"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_gap_row):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing acceptance gaps row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "F. Reliability",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing acceptance gaps row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-next-coverage-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_next_row = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "acceptance next"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_next_row):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing acceptance next row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "F. Reliability",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing acceptance next row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-completion-gate-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_completion_gate = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "completion gate"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_completion_gate):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing completion gate row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "F. Reliability",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing completion gate row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-learning-loop-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_without_learning_loop = tuple(
            label for label in live_check.LIVE_CHECK_TABLE_LABELS if label != "learning loop"
        )
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_without_learning_loop):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing learning loop row should fail safely: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "missing live_check row mapping",
        "F. Reliability",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"missing learning loop row detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-duplicate-row-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        labels_with_duplicate = live_check.LIVE_CHECK_TABLE_LABELS + ("channel health",)
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", labels_with_duplicate):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"duplicate live_check row label should fail safely: {ok} {detail}")
    for expected in ("acceptance coverage drift", "duplicate live_check row label", "repair live_check coverage"):
        if expected not in detail:
            raise SystemExit(f"duplicate live_check row detail missed {expected!r}: {detail}")
    if "channel health" in detail:
        raise SystemExit(f"duplicate live_check row detail should not echo row labels: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-acceptance-row-budget-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        extra_rows = tuple(f"extra safe row {i} SHOULD NOT APPEAR" for i in range(10))
        with mock.patch.object(live_check, "LIVE_CHECK_TABLE_LABELS", live_check.LIVE_CHECK_TABLE_LABELS + extra_rows):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"oversized live_check table should fail safely: {ok} {detail}")
    for expected in ("acceptance coverage drift", "one-screen row budget exceeded", "repair live_check coverage"):
        if expected not in detail:
            raise SystemExit(f"oversized live_check table detail missed {expected!r}: {detail}")
    if "SHOULD NOT APPEAR" in detail:
        raise SystemExit(f"oversized live_check table detail leaked row labels: {detail}")


def test_acceptance_item_map_detects_requirement_drift_without_leakage() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-item-anchor-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        plan_path = root / "FINISH_PLAN_AUGUST.md"
        plan_text = plan_path.read_text(encoding="utf-8")
        plan_path.write_text(
            plan_text.replace(
                "- [ ] Korean voice input fixture item",
                "- [ ] NEW private voice proof SHOULD NOT APPEAR",
                1,
            ),
            encoding="utf-8",
        )
        ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"replaced acceptance item should fail item coverage: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "checklist item anchor mismatch",
        "D. Voice",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"acceptance item anchor drift detail missed {expected!r}: {detail}")
    if "SHOULD NOT APPEAR" in detail or "NEW private voice proof" in detail:
        raise SystemExit(f"acceptance item anchor drift leaked checklist text: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-item-count-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        plan_path = root / "FINISH_PLAN_AUGUST.md"
        plan_text = plan_path.read_text(encoding="utf-8")
        plan_path.write_text(
            plan_text.replace(
                "### B. Channels",
                "- [ ] NEW private conversation proof SHOULD NOT APPEAR\n### B. Channels",
                1,
            ),
            encoding="utf-8",
        )
        ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"new unmapped acceptance item should fail item coverage: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "checklist item count mismatch",
        "A. Conversation & research",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"acceptance item count drift detail missed {expected!r}: {detail}")
    if "SHOULD NOT APPEAR" in detail or "NEW private conversation proof" in detail:
        raise SystemExit(f"acceptance item count drift leaked checklist text: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-item-row-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        drifted_map = dict(live_check.ACCEPTANCE_HARNESS_ITEM_ROWS)
        drifted_map["F. Reliability"] = tuple(
            (anchor, ("missing private row SHOULD NOT APPEAR",)) if anchor == "Every error message" else (anchor, rows)
            for anchor, rows in drifted_map["F. Reliability"]
        )
        with mock.patch.object(live_check, "ACCEPTANCE_HARNESS_ITEM_ROWS", drifted_map):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)

    if ok is not False:
        raise SystemExit(f"missing item-level live_check row should fail coverage: {ok} {detail}")
    for expected in (
        "acceptance coverage drift",
        "checklist item row mapping(s) missing",
        "F. Reliability",
        "repair live_check coverage",
    ):
        if expected not in detail:
            raise SystemExit(f"acceptance item row drift detail missed {expected!r}: {detail}")
    if "SHOULD NOT APPEAR" in detail or "missing private row" in detail:
        raise SystemExit(f"acceptance item row drift leaked row text: {detail}")


def test_acceptance_item_map_requires_one_evidence_mapping_per_plan_item() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-item-empty-rows-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        drifted_map = dict(live_check.ACCEPTANCE_HARNESS_ITEM_ROWS)
        drifted_map["F. Reliability"] = tuple(
            (anchor, ()) if anchor == "Every error message" else (anchor, rows)
            for anchor, rows in drifted_map["F. Reliability"]
        )
        with mock.patch.object(live_check, "ACCEPTANCE_HARNESS_ITEM_ROWS", drifted_map):
            ok, detail = live_check._check_acceptance_harness_coverage(root=root)
    if ok is not False:
        raise SystemExit(f"empty checklist evidence mapping should fail closed: {ok} {detail}")
    for expected in ("acceptance coverage drift", "checklist item row mapping(s) missing", "F. Reliability"):
        if expected not in detail:
            raise SystemExit(f"empty checklist evidence detail missed {expected!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-item-one-to-one-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        plan_path = root / "FINISH_PLAN_AUGUST.md"
        lines = plan_path.read_text(encoding="utf-8").splitlines()
        first = next(index for index, line in enumerate(lines) if "`live_check` covers" in line)
        second = next(index for index, line in enumerate(lines) if "Full smoke aggregate" in line)
        lines[first] = "- [ ] `live_check` covers Full smoke aggregate fixture item"
        lines[second] = "- [ ] unmapped private reliability item SHOULD NOT APPEAR"
        plan_path.write_text("\n".join(lines), encoding="utf-8")
        ok, detail = live_check._check_acceptance_harness_coverage(root=root)
    if ok is not False:
        raise SystemExit(f"multi-anchor and unmapped checklist items should fail closed: {ok} {detail}")
    for expected in ("acceptance coverage drift", "checklist item anchor mismatch", "F. Reliability"):
        if expected not in detail:
            raise SystemExit(f"one-to-one checklist detail missed {expected!r}: {detail}")
    for forbidden in ("SHOULD NOT APPEAR", "unmapped private reliability"):
        if forbidden in detail:
            raise SystemExit(f"one-to-one checklist detail leaked seeded text {forbidden!r}: {detail}")


def test_acceptance_harness_coverage_matches_actual_finish_plan() -> None:
    plan_path = live_check.REPO_ROOT / "FINISH_PLAN_AUGUST.md"
    if IS_PUBLIC_CANDIDATE:
        if plan_path.exists():
            raise SystemExit(
                "public candidate retained private operational reference: FINISH_PLAN_AUGUST.md"
            )
        return
    blocked_call = mock.Mock(side_effect=AssertionError("acceptance coverage attempted I/O"))
    with (
        mock.patch("socket.socket", blocked_call),
        mock.patch("subprocess.run", blocked_call),
        mock.patch("subprocess.Popen", blocked_call),
        mock.patch("urllib.request.urlopen", blocked_call),
    ):
        ok, detail = live_check._check_acceptance_harness_coverage(root=live_check.REPO_ROOT)
    if ok is not True:
        raise SystemExit(f"actual finish plan is not covered by live_check: {ok} {detail}")
    for expected in ("6/6 DoD section(s)", "28/28 DoD checklist item(s)", "read-only coverage map"):
        if expected not in detail:
            raise SystemExit(f"actual finish-plan coverage missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"actual finish-plan coverage leaked {forbidden!r}: {detail}")


def test_acceptance_malformed_heading_prefix_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-heading-prefix-") as temp:
        root = Path(temp)
        _write_acceptance_plan_sections(root)
        plan_path = root / "FINISH_PLAN_AUGUST.md"
        plan_text = plan_path.read_text(encoding="utf-8").replace(
            "### B. Channels",
            "### B. ChannelsAndSomethingElse",
            1,
        )
        plan_path.write_text(plan_text, encoding="utf-8")
        ok, detail = live_check._check_acceptance_harness_coverage(root=root)
        _counts, missing = live_check._acceptance_checkbox_counts(plan_text)
    if ok is not False or "missing DoD section(s): B. Channels" not in detail:
        raise SystemExit(f"malformed heading prefix should fail coverage: {ok} {detail}")
    if "B. Channels" not in missing:
        raise SystemExit(f"malformed heading prefix should leave channels missing: {missing}")
    for forbidden in ("ChannelsAndSomethingElse", "/\x55sers/", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"malformed heading detail leaked {forbidden!r}: {detail}")


def _write_acceptance_plan_checkboxes(root: Path, *, missing: str = "", all_done: bool = False, no_items: bool = False) -> None:
    sections = (
        "A. Conversation & research",
        "B. Channels",
        "C. Personal integrations",
        "D. Voice",
        "E. Autonomy & daily value",
        "F. Reliability",
    )
    lines = ["# Jarvis V2 Finish Plan\n"]
    for index, section in enumerate(sections):
        if section == missing:
            continue
        lines.append(f"### {section}\n")
        if no_items:
            continue
        if all_done:
            lines.append("- [x] DONE private item SHOULD NOT APPEAR\n")
        elif index in (1, 3, 5):
            lines.append("- [ ] OPEN private item SHOULD NOT APPEAR\n")
        else:
            lines.append("- [x] DONE private item SHOULD NOT APPEAR\n")
            lines.append("- [ ] OPEN private item SHOULD NOT APPEAR\n")
    lines.append("/\x55sers/example/private/acceptance SECRET SHOULD NOT APPEAR\n")
    (root / "FINISH_PLAN_AUGUST.md").write_text("\n".join(lines), encoding="utf-8")


def test_acceptance_open_items_reports_remaining_gaps_without_item_leakage() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-gaps-open-") as temp:
        root = Path(temp)
        _write_acceptance_plan_checkboxes(root)
        ok, detail = live_check._check_acceptance_open_items(root=root)

    if ok is not False:
        raise SystemExit(f"open acceptance items should keep the row red: {ok} {detail}")
    for expected in (
        "acceptance gaps remain",
        "6 open / 3 done",
        "A. Conversation & research 1",
        "B. Channels 1",
        "D. Voice 1",
        "F. Reliability 1",
        "operator-present live proofs still required",
        "run `acceptance next`",
        "prioritized operator-present proof lane",
        "safe report format",
        "read-only count only",
        "no live sends/calls/approvals/microphone/reboot",
    ):
        if expected not in detail:
            raise SystemExit(f"open acceptance gaps detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "private/acceptance", "SECRET SHOULD NOT APPEAR", "OPEN private", "DONE private"):
        if forbidden in detail:
            raise SystemExit(f"open acceptance gaps detail leaked seeded value {forbidden!r}: {detail}")


def test_acceptance_open_items_distinguishes_offline_engineering_debt() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-gaps-offline-") as temp:
        root = Path(temp)
        _write_acceptance_plan_checkboxes(root)
        plan_path = root / "FINISH_PLAN_AUGUST.md"
        plan_text = plan_path.read_text(encoding="utf-8").replace(
            "### F. Reliability\n",
            (
                "### F. Reliability\n"
                "- [ ] Every error message a user can see names the fix "
                "SECRET SHOULD NOT APPEAR\n"
            ),
            1,
        )
        plan_path.write_text(plan_text, encoding="utf-8")
        ok, detail = live_check._check_acceptance_open_items(root=root)

    if ok is not False:
        raise SystemExit(
            f"offline engineering debt should keep acceptance gaps open: {ok} {detail}"
        )
    for expected in (
        "7 open / 3 done",
        "operator-present live proofs still required (6)",
        "offline engineering debt remains (1)",
        "error guidance 1",
        "run `acceptance next`",
        "read-only count only",
    ):
        if expected not in detail:
            raise SystemExit(
                f"offline acceptance-debt detail missed {expected!r}: {detail}"
            )
    for forbidden in (
        "SECRET SHOULD NOT APPEAR",
        "Every error message",
        "/\x55sers/operator",
        "OPEN private",
        "DONE private",
    ):
        if forbidden in detail:
            raise SystemExit(
                f"offline acceptance-debt detail leaked {forbidden!r}: {detail}"
            )


def test_annotated_acceptance_heading_keeps_channel_counts_and_priority() -> None:
    plan = """# Jarvis V2 Finish Plan

### A. Conversation & research
- [x] DONE private item SHOULD NOT APPEAR
### B. Channels (BLOCKED on the operator's live tests — fixture)
- [ ] OPEN private channel one SHOULD NOT APPEAR
- [ ] OPEN private channel two SHOULD NOT APPEAR
### C. Personal integrations
- [ ] OPEN private item SHOULD NOT APPEAR
### D. Voice
- [ ] OPEN private item SHOULD NOT APPEAR
### E. Autonomy & daily value
- [ ] OPEN private item SHOULD NOT APPEAR
### F. Reliability
- [ ] OPEN private item SHOULD NOT APPEAR
/\x55sers/example/private/acceptance SECRET SHOULD NOT APPEAR
"""
    if live_check._canonical_acceptance_section("B. Channels (BLOCKED on the operator)") != "B. Channels":
        raise SystemExit("annotated canonical acceptance heading was not recognized")
    if live_check._canonical_acceptance_section("B. ChannelsAndSomethingElse") is not None:
        raise SystemExit("non-canonical heading prefix should not be accepted")

    with TemporaryDirectory(prefix="jarvis-live-check-gaps-annotated-") as temp:
        root = Path(temp)
        (root / "FINISH_PLAN_AUGUST.md").write_text(plan, encoding="utf-8")
        counts, missing = live_check._acceptance_checkbox_counts(plan)
        gap_ok, gap_detail = live_check._check_acceptance_open_items(root=root)
        next_ok, next_detail = live_check._check_acceptance_next_action(root=root)

    if missing:
        raise SystemExit(f"annotated heading incorrectly left sections missing: {missing}")
    if counts["B. Channels"] != {"open": 2, "done": 0}:
        raise SystemExit(f"annotated channel counts were dropped: {counts['B. Channels']}")
    if gap_ok is not False:
        raise SystemExit(f"annotated open checklist should keep gaps red: {gap_ok} {gap_detail}")
    for expected in ("acceptance gaps remain", "6 open / 1 done", "B. Channels 2"):
        if expected not in gap_detail:
            raise SystemExit(f"annotated acceptance gaps detail missed {expected!r}: {gap_detail}")
    if next_ok is not None:
        raise SystemExit(f"annotated channel lane should be the pending next proof: {next_ok} {next_detail}")
    for expected in ("next acceptance lane: B. Channels", "2 open item(s) in this lane"):
        if expected not in next_detail:
            raise SystemExit(f"annotated acceptance next detail missed {expected!r}: {next_detail}")
    for detail in (gap_detail, next_detail):
        for forbidden in (
            "/\x55sers/operator",
            "private/acceptance",
            "SECRET SHOULD NOT APPEAR",
            "OPEN private",
            "DONE private",
        ):
            if forbidden in detail:
                raise SystemExit(f"annotated acceptance detail leaked seeded value {forbidden!r}: {detail}")


def test_acceptance_open_items_reports_clear_checklist_without_live_actions() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-gaps-clear-") as temp:
        root = Path(temp)
        _write_acceptance_plan_checkboxes(root, all_done=True)
        ok, detail = live_check._check_acceptance_open_items(root=root)

    if ok is not True:
        raise SystemExit(f"clear acceptance checklist should pass: {ok} {detail}")
    for expected in (
        "acceptance gaps clear",
        "0 open / 6 done",
        "all 6 sections counted",
        "final acceptance review",
        "operator present",
        "read-only count only",
    ):
        if expected not in detail:
            raise SystemExit(f"clear acceptance gaps detail missed {expected!r}: {detail}")
    for forbidden in ("SHOULD NOT APPEAR", "DONE private", "/\x55sers/operator"):
        if forbidden in detail:
            raise SystemExit(f"clear acceptance gaps detail leaked seeded value {forbidden!r}: {detail}")


def test_acceptance_open_items_flags_missing_or_empty_checklist_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-gaps-missing-") as temp:
        root = Path(temp)
        _write_acceptance_plan_checkboxes(root, missing="E. Autonomy & daily value")
        ok, detail = live_check._check_acceptance_open_items(root=root)

    if ok is not False:
        raise SystemExit(f"missing acceptance section should fail the gaps row safely: {ok} {detail}")
    for expected in ("acceptance gaps unavailable", "missing DoD section", "E. Autonomy & daily value", "repair finish-plan headings"):
        if expected not in detail:
            raise SystemExit(f"missing acceptance gaps detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"missing acceptance gaps detail leaked seeded value {forbidden!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-gaps-empty-") as temp:
        root = Path(temp)
        _write_acceptance_plan_checkboxes(root, no_items=True)
        ok, detail = live_check._check_acceptance_open_items(root=root)

    if ok is not False:
        raise SystemExit(f"empty acceptance checklist should fail the gaps row safely: {ok} {detail}")
    for expected in ("acceptance gaps unavailable", "no DoD checkbox items found", "repair finish-plan checklist"):
        if expected not in detail:
            raise SystemExit(f"empty acceptance gaps detail missed {expected!r}: {detail}")


def test_acceptance_next_action_reports_priority_lane_without_item_leakage() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-next-open-") as temp:
        root = Path(temp)
        _write_acceptance_plan_checkboxes(root)
        ok, detail = live_check._check_acceptance_next_action(root=root)

    if ok is not None:
        raise SystemExit(f"open acceptance next action should be not-yet-proven, not pass/fail: {ok} {detail}")
    if "; ;" in detail:
        raise SystemExit(f"acceptance next guidance should not contain a duplicated separator: {detail}")
    if "; read-only guidance only" not in detail:
        raise SystemExit(f"acceptance next guidance should separate the report format from its boundary: {detail}")
    for expected in (
        "next acceptance lane: B. Channels",
        "live channel matrix",
        "pass/fail",
        "approval shown",
        "stage/error",
        "do not include secrets or message content",
        "safe report fields: lane, result",
        "approval shown if relevant",
        "bounded evidence id or live_check row",
        "last visible stage/error",
        "omit secrets",
        "message content",
        "transcripts",
        "contact handles",
        "tokens",
        "local paths",
        "screenshots with private text",
        "1 open item(s) in this lane",
        "read-only guidance only",
        "does not send/call/approve",
        "operator-present proof",
    ):
        if expected not in detail:
            raise SystemExit(f"acceptance next detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "private/acceptance", "SECRET SHOULD NOT APPEAR", "OPEN private", "DONE private"):
        if forbidden in detail:
            raise SystemExit(f"acceptance next detail leaked seeded value {forbidden!r}: {detail}")


def test_acceptance_next_action_reports_clear_checklist_without_live_actions() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-next-clear-") as temp:
        root = Path(temp)
        _write_acceptance_plan_checkboxes(root, all_done=True)
        ok, detail = live_check._check_acceptance_next_action(root=root)

    if ok is not True:
        raise SystemExit(f"clear acceptance next action should pass: {ok} {detail}")
    for expected in ("acceptance next clear", "no open DoD checkbox", "6 done", "final acceptance review", "read-only guidance only"):
        if expected not in detail:
            raise SystemExit(f"clear acceptance next detail missed {expected!r}: {detail}")
    for forbidden in ("SHOULD NOT APPEAR", "DONE private", "/\x55sers/operator"):
        if forbidden in detail:
            raise SystemExit(f"clear acceptance next detail leaked seeded value {forbidden!r}: {detail}")


def test_acceptance_next_action_flags_missing_or_empty_checklist_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-next-missing-") as temp:
        root = Path(temp)
        _write_acceptance_plan_checkboxes(root, missing="B. Channels")
        ok, detail = live_check._check_acceptance_next_action(root=root)

    if ok is not False:
        raise SystemExit(f"missing acceptance section should fail the next row safely: {ok} {detail}")
    for expected in ("acceptance next unavailable", "missing DoD section", "B. Channels", "repair finish-plan headings"):
        if expected not in detail:
            raise SystemExit(f"missing acceptance next detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "SECRET SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"missing acceptance next detail leaked seeded value {forbidden!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-next-empty-") as temp:
        root = Path(temp)
        _write_acceptance_plan_checkboxes(root, no_items=True)
        ok, detail = live_check._check_acceptance_next_action(root=root)

    if ok is not False:
        raise SystemExit(f"empty acceptance checklist should fail the next row safely: {ok} {detail}")
    for expected in ("acceptance next unavailable", "no DoD checkbox items found", "repair finish-plan checklist"):
        if expected not in detail:
            raise SystemExit(f"empty acceptance next detail missed {expected!r}: {detail}")


def test_call_readiness_binds_native_target_without_call_or_app_control() -> None:
    with TemporaryDirectory(prefix="jarvis-live-check-call-readiness-") as temp:
        runtime = make_temp_runtime(Path(temp))
        private_name = "Private Recipient SHOULD NOT APPEAR"
        private_phone = "+821012345678"
        with mock.patch(
            "jarvis_v2.tools.contacts_connector.resolve_contact",
            return_value=[ContactMatch(private_name, phone=private_phone)],
        ):
            ok, detail = live_check._check_call_readiness(
                runtime.config,
                ("acceptance target SHOULD NOT APPEAR",),
            )
    if ok is not True:
        raise SystemExit(f"exact native call target should be preflight-ready: {ok} {detail}")
    for expected in (
        "4/4 call tools HIGH_RISK, strict-argument, and approval-gated",
        "approval card must bind the exact target + effective mode",
        "1/1 native phone/FaceTime target binding(s) ready",
        "supplied names and handles hidden",
        "account and call-button availability still need operator-present proof",
        "call_requested is not recipient-confirmed proof",
        "known-not-started and outcome-unknown are distinct",
        "never retried automatically",
        "no call, approval, browser, clipboard, or app control ran",
    ):
        if expected not in detail:
            raise SystemExit(f"call readiness detail missed {expected!r}: {detail}")
    for forbidden in (
        private_name,
        private_phone,
        "acceptance target",
        "SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(f"call readiness leaked target detail {forbidden!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-call-email-only-") as temp:
        runtime = make_temp_runtime(Path(temp))
        with mock.patch(
            "jarvis_v2.tools.contacts_connector.resolve_contact",
            return_value=[
                ContactMatch(
                    "Email-only Recipient SHOULD NOT APPEAR",
                    email="private@example.com",
                )
            ],
        ):
            ok, detail = live_check._check_call_readiness(
                runtime.config,
                ("email-only target SHOULD NOT APPEAR",),
            )
    if ok is not False or "need contact repair" not in detail:
        raise SystemExit(f"email-only phone target should fail before live proof: {ok} {detail}")
    for forbidden in ("Email-only Recipient", "private@example.com", "email-only target"):
        if forbidden in detail:
            raise SystemExit(f"failed call readiness leaked target detail {forbidden!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-call-gate-drift-") as temp:
        runtime = make_temp_runtime(Path(temp))
        drifted = [
            _FakeTool("call_contact", RiskLevel.READ_ONLY),
            _FakeTool("call_kakao", RiskLevel.HIGH_RISK),
            _FakeTool("call_instagram", RiskLevel.HIGH_RISK),
        ]
        with mock.patch(
            "jarvis_v2.tools.call_connector.make_call_tools",
            return_value=drifted,
        ):
            ok, detail = live_check._check_call_readiness(runtime.config, ())
    if ok is not False or "1 missing, 1 not HIGH_RISK" not in detail:
        raise SystemExit(f"call gate drift should fail closed: {ok} {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-call-contract-drift-") as temp:
        runtime = make_temp_runtime(Path(temp))
        from jarvis_v2.tools import call_connector

        real_tools = call_connector.make_call_tools(runtime.config)
        drifted = [
            replace(tool, argument_contract=None)
            if tool.name == "call_contact"
            else tool
            for tool in real_tools
        ]
        with mock.patch(
            "jarvis_v2.tools.call_connector.make_call_tools",
            return_value=drifted,
        ):
            ok, detail = live_check._check_call_readiness(runtime.config, ())
    if ok is not False or "call argument-contract drift: 1/4" not in detail:
        raise SystemExit(f"call contract drift should fail closed: {ok} {detail}")


def test_live_check_exception_details_are_redacted() -> None:
    seeded_path = "/\x55sers/example/private/live-check-secret.db SHOULD NOT APPEAR"
    with mock.patch("jarvis_v2.tools.contacts_connector.resolve_contact", side_effect=RuntimeError(seeded_path)):
        ok, detail = live_check._check_contacts("Fixture")
    if ok is not False:
        raise SystemExit(f"contact exception should fail safely: {ok} {detail}")
    if "RuntimeError" not in detail or "<local-path>" not in detail:
        raise SystemExit(f"contact exception detail missed bounded diagnostic: {detail}")
    for forbidden in ("/\x55sers/operator", "private/live-check-secret.db", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"contact exception leaked seeded private detail {forbidden!r}: {detail}")

    with TemporaryDirectory(prefix="jarvis-live-check-redact-") as temp:
        runtime = make_temp_runtime(Path(temp))
        store_error = OSError("/private/tmp/jarvis-live-scheduler.sqlite TOKEN SHOULD NOT APPEAR")
        with mock.patch("jarvis_v2.memory.store.MemoryStore", side_effect=store_error):
            ok, detail = live_check._check_morning_brief_schedule(runtime.config)
    if ok is not False or "scheduler store unreadable" not in detail:
        raise SystemExit(f"scheduler exception should fail safely: {ok} {detail}")
    if "OSError" not in detail or "<local-path>" not in detail:
        raise SystemExit(f"scheduler exception detail missed bounded diagnostic: {detail}")
    for forbidden in ("/private/tmp", "TOKEN SHOULD NOT APPEAR", "jarvis-live-scheduler.sqlite"):
        if forbidden in detail:
            raise SystemExit(f"scheduler exception leaked seeded private detail {forbidden!r}: {detail}")


def test_live_check_contacts_distinguishes_store_state() -> None:
    with (
        mock.patch("jarvis_v2.tools.contacts_connector.clear_contact_cache"),
        mock.patch("jarvis_v2.tools.contacts_connector.resolve_contact", return_value=[]),
        mock.patch("jarvis_v2.tools.contacts_connector.contacts_store_status", return_value="empty"),
    ):
        ok, detail = live_check._check_contacts("Fixture")
    if ok is not False or "no saved contacts" not in detail:
        raise SystemExit(f"empty Contacts store should name the sync blocker: {ok} {detail}")

    with (
        mock.patch("jarvis_v2.tools.contacts_connector.clear_contact_cache"),
        mock.patch("jarvis_v2.tools.contacts_connector.resolve_contact", return_value=[]),
        mock.patch("jarvis_v2.tools.contacts_connector.contacts_store_status", return_value="available"),
    ):
        ok, detail = live_check._check_contacts("Fixture")
    if ok is not False or "try a name you know exists" not in detail:
        raise SystemExit(f"missing contact should remain distinct from setup failure: {ok} {detail}")

    with (
        mock.patch("jarvis_v2.tools.contacts_connector.clear_contact_cache"),
        mock.patch("jarvis_v2.tools.contacts_connector.resolve_contact", return_value=[]),
        mock.patch(
            "jarvis_v2.tools.contacts_connector.contacts_store_status",
            side_effect=RuntimeError("/\x55sers/example/private/contacts SHOULD NOT APPEAR"),
        ),
    ):
        ok, detail = live_check._check_contacts("Fixture")
    if ok is not False or "couldn't access macOS Contacts" not in detail:
        raise SystemExit(f"failed Contacts status probe should return bounded recovery: {ok} {detail}")
    for forbidden in ("/\x55sers/", "private/contacts", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"failed Contacts status probe leaked private detail {forbidden!r}: {detail}")


def test_live_check_contact_matrix_is_bounded_and_hides_supplied_names() -> None:
    contacts = live_check._arg_contacts(
        [
            "--contact",
            "Fixture Example SHOULD NOT APPEAR",
            "--contact=가상연락처일 SHOULD NOT APPEAR",
            "-c",
            "가상연락처이 SHOULD NOT APPEAR",
            "--contact",
            "FIXTURE EXAMPLE SHOULD NOT APPEAR",
        ]
    )
    if contacts != (
        "Fixture Example SHOULD NOT APPEAR",
        "가상연락처일 SHOULD NOT APPEAR",
        "가상연락처이 SHOULD NOT APPEAR",
    ):
        raise SystemExit(f"contact matrix should preserve unique supplied queries: {contacts!r}")
    calls: list[str] = []

    def lookup(contact: str) -> tuple[bool | None, str]:
        calls.append(contact)
        return True, "private detail SHOULD NOT APPEAR"

    ok, detail = live_check._check_contact_matrix(contacts, lookup, label="contact lookup(s)")
    if ok is not True or "3/3 contact lookup(s) ready" not in detail or "supplied names hidden" not in detail:
        raise SystemExit(f"contact matrix should report aggregate success: {ok} {detail}")
    if calls != list(contacts):
        raise SystemExit(f"contact matrix should check each unique supplied query once: {calls!r}")
    for forbidden in ("Fixture Example", "가상연락처일", "가상연락처이", "private detail", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"contact matrix leaked supplied/private detail {forbidden!r}: {detail}")

    def mixed(_: str) -> tuple[bool | None, str]:
        return False, "private detail SHOULD NOT APPEAR"

    ok, detail = live_check._check_contact_matrix(contacts, mixed, label="send preflight(s)")
    if ok is not False or "0/3 send preflight(s) ready" not in detail or "3 need attention" not in detail:
        raise SystemExit(f"contact matrix should surface aggregate failure: {ok} {detail}")
    for forbidden in ("Fixture Example", "가상연락처일", "가상연락처이", "private detail", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"failed contact matrix leaked supplied/private detail {forbidden!r}: {detail}")

    over_limit = tuple(f"contact-{index}" for index in range(live_check.MAX_LIVE_CHECK_CONTACT_QUERIES + 1))
    calls.clear()
    ok, detail = live_check._check_contact_matrix(over_limit, lookup, label="contact lookup(s)")
    if ok is not False or "contact preflight limit is" not in detail or "rerun with fewer supplied names" not in detail:
        raise SystemExit(f"over-limit contact matrix should refuse before lookup: {ok} {detail}")
    if calls:
        raise SystemExit(f"over-limit contact matrix should not query any contact: {calls!r}")


def test_live_check_telegram_failure_details_are_redacted() -> None:
    with (
        mock.patch("jarvis_v2.scripts.live_check.load_config", return_value=SimpleNamespace()),
        mock.patch("jarvis_v2.automations.telegram_control._bot_token", return_value="token-present"),
        mock.patch("jarvis_v2.automations.telegram_control._owner_chat_id", return_value="12345"),
        mock.patch(
            "jarvis_v2.automations.telegram_control.send_message",
            return_value={"ok": False, "error": "failed reading /var/folders/zc/jarvis-token.txt SHOULD NOT APPEAR"},
        ),
    ):
        ok, detail = live_check._check_telegram(send=True)
    if ok is not False or "send failed" not in detail:
        raise SystemExit(f"telegram send failure should fail safely: {ok} {detail}")
    if "<local-path>" not in detail:
        raise SystemExit(f"telegram send failure missed redacted path marker: {detail}")
    for forbidden in ("/var/folders", "jarvis-token.txt", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"telegram failure leaked seeded private detail {forbidden!r}: {detail}")


def test_live_check_success_details_suppress_private_values() -> None:
    matches = [
        ContactMatch("Fixture Example SHOULD NOT APPEAR", phone="+1 555 123 4567", email="fixture.secret@example.com"),
    ]
    with (
        mock.patch("jarvis_v2.tools.contacts_connector.clear_contact_cache"),
        mock.patch("jarvis_v2.tools.contacts_connector.resolve_contact", return_value=matches),
    ):
        ok, detail = live_check._check_contacts("fixture")
    if ok is not True or "1 match(es)" not in detail or "phone handle" not in detail:
        raise SystemExit(f"contact success should stay useful without raw values: {ok} {detail}")
    for forbidden in ("Fixture Example", "+1 555 123 4567", "fixture.secret@example.com", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"contact success leaked private value {forbidden!r}: {detail}")

    with (
        mock.patch("jarvis_v2.tools.contacts_connector.looks_like_handle", return_value=False),
        mock.patch("jarvis_v2.tools.contacts_connector.clear_contact_cache"),
        mock.patch("jarvis_v2.tools.contacts_connector.resolve_contact", return_value=matches),
    ):
        ok, detail = live_check._check_send_resolution("fixture")
    if ok is not True or "single matched contact" not in detail or "phone handle" not in detail:
        raise SystemExit(f"send-resolution success should describe the decision: {ok} {detail}")
    for forbidden in ("Fixture Example", "+1 555 123 4567", "fixture.secret@example.com", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"send-resolution success leaked private value {forbidden!r}: {detail}")

    ambiguous = [
        ContactMatch("Fixture Example SHOULD NOT APPEAR", phone="+1 555 123 4567"),
        ContactMatch("Fixture Kim SHOULD NOT APPEAR", phone="+1 555 999 0000"),
    ]
    with (
        mock.patch("jarvis_v2.tools.contacts_connector.looks_like_handle", return_value=False),
        mock.patch("jarvis_v2.tools.contacts_connector.clear_contact_cache"),
        mock.patch("jarvis_v2.tools.contacts_connector.resolve_contact", return_value=ambiguous),
    ):
        ok, detail = live_check._check_send_resolution("fixture")
    if ok is not None or "would BLOCK" not in detail or "2 matching contacts" not in detail:
        raise SystemExit(f"ambiguous send-resolution should block without listing names: {ok} {detail}")
    for forbidden in ("Fixture Example", "Fixture Kim", "+1 555", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"ambiguous send-resolution leaked private value {forbidden!r}: {detail}")

    with mock.patch("jarvis_v2.tools.contacts_connector.looks_like_handle", return_value=True):
        ok, detail = live_check._check_send_resolution("+1 555 123 4567")
    if ok is not True or "phone handle" not in detail:
        raise SystemExit(f"direct handle should report handle kind only: {ok} {detail}")
    if "+1 555 123 4567" in detail:
        raise SystemExit(f"direct handle preview leaked raw handle: {detail}")


def test_live_check_telegram_success_suppresses_owner_id() -> None:
    with (
        mock.patch("jarvis_v2.scripts.live_check.load_config", return_value=SimpleNamespace()),
        mock.patch("jarvis_v2.automations.telegram_control._bot_token", return_value="token-present"),
        mock.patch("jarvis_v2.automations.telegram_control._owner_chat_id", return_value="123456789 SHOULD NOT APPEAR"),
    ):
        ok, detail = live_check._check_telegram(send=False)
    if ok is not True or "owner chat id present" not in detail:
        raise SystemExit(f"telegram env check should report owner configured: {ok} {detail}")
    for forbidden in ("123456789", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"telegram env detail leaked owner id {forbidden!r}: {detail}")

    with (
        mock.patch("jarvis_v2.scripts.live_check.load_config", return_value=SimpleNamespace()),
        mock.patch("jarvis_v2.automations.telegram_control._bot_token", return_value="token-present"),
        mock.patch("jarvis_v2.automations.telegram_control._owner_chat_id", return_value="123456789 SHOULD NOT APPEAR"),
        mock.patch("jarvis_v2.automations.telegram_control.send_message", return_value={"ok": True}),
    ):
        ok, detail = live_check._check_telegram(send=True)
    if ok is not True or "accepted" not in detail or "owner chat id" not in detail:
        raise SystemExit(f"telegram send check should report API acceptance without owner id: {ok} {detail}")
    if "delivered" in detail.casefold():
        raise SystemExit(f"telegram send check should not overclaim phone delivery: {detail}")
    for forbidden in ("123456789", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"telegram send detail leaked owner id {forbidden!r}: {detail}")


def test_live_check_telegram_read_probe_is_read_only_and_private_safe() -> None:
    with (
        mock.patch("jarvis_v2.scripts.live_check.load_config", return_value=SimpleNamespace()),
        mock.patch("jarvis_v2.automations.telegram_control._bot_token", return_value="token-present"),
        mock.patch(
            "jarvis_v2.automations.telegram_control._owner_chat_id",
            return_value="123456789 SHOULD NOT APPEAR",
        ),
        mock.patch(
            "jarvis_v2.automations.telegram_control._api_call",
            return_value={
                "ok": True,
                "result": {
                    "id": 987654321,
                    "username": "private_bot SHOULD NOT APPEAR",
                },
            },
        ) as api_call,
        mock.patch("jarvis_v2.automations.telegram_control.send_message") as send_message,
    ):
        ok, detail = live_check._check_telegram(send=False, probe=True)
    if ok is not True or "read-only Telegram API identity check succeeded" not in detail:
        raise SystemExit(f"Telegram read probe should report bounded success: {ok} {detail}")
    api_call.assert_called_once_with("token-present", "getMe", {}, 10.0)
    send_message.assert_not_called()
    for forbidden in ("123456789", "987654321", "private_bot", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"Telegram read probe leaked private identity {forbidden!r}: {detail}")

    seeded_error = "TLS failed at /\x55sers/example/private/token.pem SHOULD NOT APPEAR"
    with (
        mock.patch("jarvis_v2.scripts.live_check.load_config", return_value=SimpleNamespace()),
        mock.patch("jarvis_v2.automations.telegram_control._bot_token", return_value="token-present"),
        mock.patch("jarvis_v2.automations.telegram_control._owner_chat_id", return_value="12345"),
        mock.patch(
            "jarvis_v2.automations.telegram_control._api_call",
            return_value={"ok": False, "error": seeded_error},
        ),
        mock.patch("jarvis_v2.automations.telegram_control.send_message") as send_message,
    ):
        ok, detail = live_check._check_telegram(send=False, probe=True)
    if ok is not False or "setup check" not in detail or "network/TLS setup" not in detail:
        raise SystemExit(f"Telegram read probe failure missed actionable recovery: {ok} {detail}")
    if "<local-path>" not in detail:
        raise SystemExit(f"Telegram read probe failure missed path redaction: {detail}")
    send_message.assert_not_called()
    for forbidden in ("/\x55sers/operator", "token.pem", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"Telegram read probe failure leaked private detail {forbidden!r}: {detail}")


def test_live_check_telegram_loads_project_env_before_preflight() -> None:
    with (
        mock.patch("jarvis_v2.scripts.live_check.load_config", return_value=SimpleNamespace()) as load_config,
        mock.patch("jarvis_v2.automations.telegram_control._bot_token", return_value=""),
    ):
        ok, detail = live_check._check_telegram(send=False)
    if ok is not False or "no bot token" not in detail:
        raise SystemExit(f"missing Telegram token should remain a bounded preflight failure: {ok} {detail}")
    load_config.assert_called_once_with()


def test_live_check_rows_redact_detail_before_printing() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        live_check._row("test row", True, "ready via /\x55sers/example/private/bin/whisper SECRET SHOULD NOT APPEAR")
    output = stdout.getvalue()
    if "<local-path>" not in output:
        raise SystemExit(f"row output missed redacted path marker: {output}")
    for forbidden in ("/\x55sers/operator", "private/bin/whisper", "SECRET SHOULD NOT APPEAR"):
        if forbidden in output:
            raise SystemExit(f"row output leaked seeded private detail {forbidden!r}: {output}")


def test_live_check_unprintable_diagnostics_fail_closed() -> None:
    class ExplodingText:
        def __str__(self) -> str:
            raise RuntimeError("/\x55sers/example/private/stringify SHOULD NOT APPEAR")

        def __bool__(self) -> bool:
            raise RuntimeError("/private/tmp/bool SHOULD NOT APPEAR")

    class ExplodingError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("/\x55sers/example/private/exception SHOULD NOT APPEAR")

    detail = live_check._safe_detail(ExplodingText())
    if "<unprintable ExplodingText>" not in detail:
        raise SystemExit(f"unprintable detail should use a bounded placeholder: {detail}")
    exception_detail = live_check._safe_exception(ExplodingError())
    if "ExplodingError" not in exception_detail or "<unprintable ExplodingError>" not in exception_detail:
        raise SystemExit(f"unprintable exception should use type plus placeholder: {exception_detail}")
    handle_kind = live_check._handle_kind(ExplodingText())
    if handle_kind != "no phone/email handle":
        raise SystemExit(f"unprintable handle should collapse to no handle: {handle_kind}")
    if live_check._safe_iso_datetime(ExplodingText()) != "invalid_timestamp":
        raise SystemExit("unprintable timestamp should collapse to invalid_timestamp")
    if live_check._job_enabled({"enabled": ExplodingText()}) is not False:
        raise SystemExit("unprintable enabled flag should fail closed to disabled")
    if live_check._voice_source_label(ExplodingText()) != "unknown transcriber source":
        raise SystemExit("unprintable voice source should collapse to unknown source")
    for output in (detail, exception_detail, handle_kind):
        for forbidden in ("/\x55sers/operator", "/private/tmp", "SHOULD NOT APPEAR", "stringify", "exception"):
            if forbidden in output:
                raise SystemExit(f"unprintable diagnostic leaked seeded detail {forbidden!r}: {output}")


def test_live_check_malformed_mappings_fail_closed() -> None:
    class ExplodingMapping:
        def get(self, _key: str, _default: object = None) -> object:
            raise RuntimeError("/\x55sers/example/private/metadata-get SHOULD NOT APPEAR")

        def __getitem__(self, _key: str) -> object:
            raise RuntimeError("/private/tmp/metadata-item SHOULD NOT APPEAR")

    private_brief = "Good morning! SHOULD NOT APPEAR\nCalendar private detail SHOULD NOT APPEAR"
    daily_tool = SimpleNamespace(
        name="daily_briefing",
        handler=lambda _args: ToolResult(
            "daily_briefing",
            True,
            private_brief,
            ExplodingMapping(),
        ),
    )
    with mock.patch("jarvis_v2.tools.brief_tools.make_brief_tools", return_value=[daily_tool]):
        ok, detail = live_check._check_daily_brief(config=object())
    if ok is not True or "content suppressed" not in detail:
        raise SystemExit(f"daily brief malformed metadata should still compose safely: {ok} {detail}")
    if "section(s) available" in detail or "unavailable" in detail:
        raise SystemExit(f"daily brief malformed metadata should not show section counts: {detail}")
    for forbidden in ("Good morning", "Calendar private", "metadata-get", "metadata-item", "SHOULD NOT APPEAR", "/\x55sers/operator", "/private/tmp"):
        if forbidden in detail:
            raise SystemExit(f"daily brief malformed metadata leaked seeded detail {forbidden!r}: {detail}")

    channel_result = ToolResult("channel_health", True, "SHOULD NOT APPEAR", ExplodingMapping())
    with (
        mock.patch("jarvis_v2.memory.store.MemoryStore", return_value=object()),
        mock.patch("jarvis_v2.tools.channel_health.make_channel_health_tools", return_value=[lambda _args: channel_result]),
    ):
        ok, detail = live_check._check_channel_health(config=SimpleNamespace(db_path="unused"))
    if ok is not False or "metadata did not prove content suppression" not in detail:
        raise SystemExit(f"channel health malformed metadata should fail safely: {ok} {detail}")
    for forbidden in ("metadata-get", "metadata-item", "SHOULD NOT APPEAR", "/\x55sers/operator", "/private/tmp"):
        if forbidden in detail:
            raise SystemExit(f"channel health malformed metadata leaked seeded detail {forbidden!r}: {detail}")

    class ExplodingName:
        def __str__(self) -> str:
            raise RuntimeError("/\x55sers/example/private/job-name SHOULD NOT APPEAR")

    class ExplodingDict(dict):
        def get(self, _key: str, _default: object = None) -> object:
            raise RuntimeError("/private/tmp/job-get SHOULD NOT APPEAR")

    class MalformedJobStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[object]:
            return [
                {"name": ExplodingName(), "enabled": 1, "next_run_at": "/\x55sers/example/job-next SHOULD NOT APPEAR"},
                ExplodingDict({"name": "Morning Brief", "enabled": 1}),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", MalformedJobStore):
        ok, detail = live_check._check_morning_brief_schedule(config=SimpleNamespace(db_path="unused"))
    if ok is not None or "not scheduled" not in detail:
        raise SystemExit(f"malformed scheduler mappings should be ignored safely: {ok} {detail}")
    for forbidden in ("job-name", "job-next", "job-get", "SHOULD NOT APPEAR", "/\x55sers/operator", "/private/tmp"):
        if forbidden in detail:
            raise SystemExit(f"malformed scheduler mapping leaked seeded detail {forbidden!r}: {detail}")


def test_live_check_daily_brief_success_suppresses_brief_content() -> None:
    private_brief = "\n".join(
        [
            "Good morning! SHOULD NOT APPEAR",
            "Calendar: dental appointment with Dr. Secret at 09:00 SHOULD NOT APPEAR",
            "Email: bank code 123456 SHOULD NOT APPEAR",
        ]
    )
    tool = SimpleNamespace(
        name="daily_briefing",
        handler=lambda _args: ToolResult(
            "daily_briefing",
            True,
            private_brief,
            {"available_count": 2, "unavailable_count": 1, "daily_brief_recovery_hint_count": 1},
        ),
    )
    with mock.patch("jarvis_v2.tools.brief_tools.make_brief_tools", return_value=[tool]):
        ok, detail = live_check._check_daily_brief(config=object())
    if ok is not True:
        raise SystemExit(f"daily brief diagnostic should pass with mocked content: {detail}")
    for expected in (
        "composed",
        "chars",
        "line(s)",
        "2 section(s) available",
        "1 unavailable",
        "1 recovery hint(s)",
        "content suppressed",
    ):
        if expected not in detail:
            raise SystemExit(f"daily brief diagnostic missed {expected!r}: {detail}")
    for forbidden in ("Good morning", "dental appointment", "Dr. Secret", "bank code", "123456", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"daily brief diagnostic leaked private content {forbidden!r}: {detail}")


def test_live_check_daily_brief_unavailable_requires_recovery_hint_proof_without_leakage() -> None:
    private_brief = "\n".join(
        [
            "Good morning! SHOULD NOT APPEAR",
            "Unavailable: weather, calendar.",
            "Recovery: /\x55sers/example/private/raw backend SHOULD NOT APPEAR",
        ]
    )
    too_few_hint_tool = SimpleNamespace(
        name="daily_briefing",
        handler=lambda _args: ToolResult(
            "daily_briefing",
            True,
            private_brief,
            {
                "available_count": 1,
                "unavailable_count": 2,
                "daily_brief_recovery_hint_count": 1,
                "daily_brief_recovery_hints": {
                    "weather": "/private/tmp/raw hint SHOULD NOT APPEAR",
                },
            },
        ),
    )
    with mock.patch("jarvis_v2.tools.brief_tools.make_brief_tools", return_value=[too_few_hint_tool]):
        ok, detail = live_check._check_daily_brief(config=object())
    if ok is not False:
        raise SystemExit(f"daily brief missing recovery hint proof should fail safely: {ok} {detail}")
    for expected in ("2 unavailable section(s)", "only 1 recovery hint(s)", "content suppressed"):
        if expected not in detail:
            raise SystemExit(f"daily brief missing recovery hint proof detail missed {expected!r}: {detail}")
    for forbidden in (
        "Good morning",
        "Unavailable: weather",
        "Recovery:",
        "raw backend",
        "raw hint",
        "SHOULD NOT APPEAR",
        "/\x55sers/operator",
        "/private/tmp",
    ):
        if forbidden in detail:
            raise SystemExit(f"daily brief missing recovery hint proof leaked seeded detail {forbidden!r}: {detail}")


def test_live_check_daily_brief_rejects_malformed_output_and_counts() -> None:
    class PrivateObject:
        def __str__(self) -> str:
            return "/\x55sers/example/private/daily-brief SHOULD NOT APPEAR"

    bad_tool = SimpleNamespace(
        name="daily_briefing",
        handler=lambda _args: ToolResult(
            "daily_briefing",
            True,
            PrivateObject(),
            {"available_count": 1, "unavailable_count": 0},
        ),
    )
    with mock.patch("jarvis_v2.tools.brief_tools.make_brief_tools", return_value=[bad_tool]):
        ok, detail = live_check._check_daily_brief(config=object())
    if ok is not False or "non-text content" not in detail:
        raise SystemExit(f"daily brief malformed output should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "daily-brief", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"daily brief malformed output leaked seeded detail {forbidden!r}: {detail}")

    private_brief = "Good morning! SHOULD NOT APPEAR\nCalendar: private appointment SHOULD NOT APPEAR"
    corrupt_count_tool = SimpleNamespace(
        name="daily_briefing",
        handler=lambda _args: ToolResult(
            "daily_briefing",
            True,
            private_brief,
            {
                "available_count": True,
                "unavailable_count": "/private/tmp/brief-count SHOULD NOT APPEAR",
            },
        ),
    )
    with mock.patch("jarvis_v2.tools.brief_tools.make_brief_tools", return_value=[corrupt_count_tool]):
        ok, detail = live_check._check_daily_brief(config=object())
    if ok is not True or "content suppressed" not in detail:
        raise SystemExit(f"daily brief corrupt count metadata should still compose safely: {ok} {detail}")
    if "section(s) available" in detail or "unavailable" in detail:
        raise SystemExit(f"daily brief corrupt count metadata should not show section counts: {detail}")
    for forbidden in ("Good morning", "private appointment", "/private/tmp", "brief-count", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"daily brief corrupt count metadata leaked seeded detail {forbidden!r}: {detail}")


def test_live_check_morning_brief_schedule_suppresses_bad_next_run() -> None:
    class FakeStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "Morning Brief",
                    "enabled": 1,
                    "next_run_at": "/\x55sers/example/calendar dental appointment SHOULD NOT APPEAR",
                }
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", FakeStore):
        ok, detail = live_check._check_morning_brief_schedule(config=SimpleNamespace(db_path="unused"))
    if ok is not False or "invalid_timestamp" not in detail:
        raise SystemExit(f"enabled bad Morning Brief timestamp should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "dental appointment", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"bad Morning Brief timestamp leaked seeded detail {forbidden!r}: {detail}")

    class PausedStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "Morning Brief",
                    "enabled": 0,
                    "next_run_at": "/private/tmp/paused-secret SHOULD NOT APPEAR",
                }
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", PausedStore):
        ok, detail = live_check._check_morning_brief_schedule(config=SimpleNamespace(db_path="unused"))
    if ok is not None or "paused" not in detail or "invalid_timestamp" not in detail:
        raise SystemExit(f"paused bad Morning Brief timestamp should skip safely: {ok} {detail}")
    for forbidden in ("/private/tmp", "paused-secret", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"paused Morning Brief timestamp leaked seeded detail {forbidden!r}: {detail}")


def test_live_check_morning_brief_schedule_survives_malformed_rows() -> None:
    class ExplodingRow:
        def __getitem__(self, _key: str) -> object:
            raise TypeError("/\x55sers/example/exploding-row SHOULD NOT APPEAR")

    class MalformedStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[object]:
            return [
                {
                    "enabled": 1,
                    "next_run_at": "/\x55sers/example/missing-name SHOULD NOT APPEAR",
                },
                ExplodingRow(),
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", MalformedStore):
        ok, detail = live_check._check_morning_brief_schedule(config=SimpleNamespace(db_path="unused"))
    if ok is not None or "not scheduled" not in detail:
        raise SystemExit(f"malformed non-matching job rows should be ignored safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "missing-name", "exploding-row", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"malformed job rows leaked seeded detail {forbidden!r}: {detail}")

    class MissingTimestampStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return [{"name": "Morning Brief", "enabled": 1}]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", MissingTimestampStore):
        ok, detail = live_check._check_morning_brief_schedule(config=SimpleNamespace(db_path="unused"))
    if ok is not False or "invalid_timestamp" not in detail:
        raise SystemExit(f"matching job without next_run_at should fail safely: {ok} {detail}")


def test_live_check_scheduled_job_freshness_reports_running_jobs_safely() -> None:
    now = datetime(2026, 7, 8, 9, 0, 0)

    class FreshStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "Morning Brief SHOULD NOT APPEAR",
                    "enabled": 1,
                    "interval_minutes": 1440,
                    "last_run_at": "2026-07-08T08:55:00",
                    "next_run_at": "2026-07-09T09:00:00",
                },
                {
                    "name": "State Snapshot SHOULD NOT APPEAR",
                    "enabled": 1,
                    "interval_minutes": 60,
                    "last_run_at": "2026-07-08T08:30:00",
                    "next_run_at": "2026-07-08T09:30:00",
                },
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", FreshStore):
        ok, detail = live_check._check_scheduled_job_freshness(SimpleNamespace(db_path="unused"), now=now)
    if ok is not True:
        raise SystemExit(f"fresh scheduled jobs should pass: {ok} {detail}")
    for expected in ("2 enabled job(s) fresh", "interval+10m", "next due 2026-07-08T09:30:00"):
        if expected not in detail:
            raise SystemExit(f"fresh scheduled job detail missed {expected!r}: {detail}")
    for forbidden in ("Morning Brief", "State Snapshot", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"fresh scheduled job detail leaked job names/content {forbidden!r}: {detail}")


def test_live_check_scheduled_job_freshness_reports_first_run_without_overclaiming() -> None:
    now = datetime(2026, 7, 8, 9, 0, 0)

    class FirstRunStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "Morning Brief private name SHOULD NOT APPEAR",
                    "enabled": 1,
                    "interval_minutes": 1440,
                    "last_run_at": "",
                    "next_run_at": "2026-07-08T09:30:00",
                }
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", FirstRunStore):
        ok, detail = live_check._check_scheduled_job_freshness(SimpleNamespace(db_path="unused"), now=now)
    if ok is not None:
        raise SystemExit(f"first-run scheduled jobs should be a skipped/not-yet-proven row: {ok} {detail}")
    for expected in ("1 enabled job(s)", "awaiting first recorded run", "no overdue jobs", "next due 2026-07-08T09:30:00"):
        if expected not in detail:
            raise SystemExit(f"first-run scheduled job detail missed {expected!r}: {detail}")
    for forbidden in ("Morning Brief", "private name", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"first-run scheduled job detail leaked job names/content {forbidden!r}: {detail}")


def test_live_check_scheduled_job_freshness_flags_stale_and_invalid_metadata_safely() -> None:
    now = datetime(2026, 7, 8, 9, 0, 0)

    class StaleStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "/\x55sers/example/private/stale-job SHOULD NOT APPEAR",
                    "enabled": 1,
                    "interval_minutes": 60,
                    "last_run_at": "2026-07-08T07:00:00",
                    "next_run_at": "2026-07-08T07:30:00",
                },
                {
                    "name": "Fresh job SHOULD NOT APPEAR",
                    "enabled": 1,
                    "interval_minutes": 60,
                    "last_run_at": "2026-07-08T08:30:00",
                    "next_run_at": "2026-07-08T09:30:00",
                },
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", StaleStore):
        ok, detail = live_check._check_scheduled_job_freshness(SimpleNamespace(db_path="unused"), now=now)
    if ok is not False:
        raise SystemExit(f"stale scheduled jobs should fail the freshness diagnostic: {ok} {detail}")
    for expected in ("2 enabled job(s)", "1 stale", "1 overdue next_run", "1 old last_run", "check the scheduler daemon"):
        if expected not in detail:
            raise SystemExit(f"stale scheduled job detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "stale-job", "Fresh job", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"stale scheduled job detail leaked job names/content {forbidden!r}: {detail}")

    class InvalidStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "/private/tmp/invalid-job SHOULD NOT APPEAR",
                    "enabled": 1,
                    "interval_minutes": "/\x55sers/example/private/interval SHOULD NOT APPEAR",
                    "last_run_at": "2026-07-08T08:30:00",
                    "next_run_at": "2026-07-08T09:30:00",
                }
            ]

    with mock.patch("jarvis_v2.memory.store.MemoryStore", InvalidStore):
        ok, detail = live_check._check_scheduled_job_freshness(SimpleNamespace(db_path="unused"), now=now)
    if ok is not False or "invalid freshness metadata" not in detail:
        raise SystemExit(f"invalid scheduled job metadata should fail safely: {ok} {detail}")
    for forbidden in ("/private/tmp", "/\x55sers/operator", "invalid-job", "interval", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"invalid scheduled job detail leaked job names/content {forbidden!r}: {detail}")


def test_live_check_network_recovery_reports_only_content_free_evidence() -> None:
    clean_snapshot = {
        "state_counts": {
            "prepared": 0,
            "claimed": 0,
            "sending": 0,
            "accepted": 2,
            "rejected": 0,
            "uncertain": 0,
            "failed": 0,
        },
        "chunk_receipts": 2,
        "occurrences": 2,
        "active_occurrences": 0,
        "uncertain_occurrences": 0,
        "failed_occurrences": 0,
        "orphaned_chunks": 0,
        "content_bearing_chunks": 0,
        "counts_saturated": False,
        "oldest_active_at": "/\x55sers/example/private/timestamp SHOULD NOT APPEAR",
    }

    class SnapshotStore:
        snapshot: object = clean_snapshot

        def __init__(self, _db_path: object) -> None:
            pass

        def scheduled_delivery_operational_snapshot(self) -> object:
            return self.snapshot

    config = SimpleNamespace(db_path="unused")
    with mock.patch("jarvis_v2.memory.store.MemoryStore", SnapshotStore):
        ok, detail = live_check._check_network_recovery_readiness(config)
    if ok is not None or "no unresolved recovery state" not in detail or "operator-present proof" not in detail:
        raise SystemExit(f"clean delivery ledger must remain unproven, not ready: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "timestamp", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"network recovery clean detail leaked private state {forbidden!r}: {detail}")

    rejected_snapshot = dict(clean_snapshot)
    rejected_snapshot["state_counts"] = dict(clean_snapshot["state_counts"], rejected=1)
    rejected_snapshot["chunk_receipts"] = 3
    rejected_snapshot["occurrences"] = 3
    rejected_snapshot["active_occurrences"] = 1
    SnapshotStore.snapshot = rejected_snapshot
    with mock.patch("jarvis_v2.memory.store.MemoryStore", SnapshotStore):
        ok, detail = live_check._check_network_recovery_readiness(config)
    if ok is not False or "1 retryable scheduled-delivery rejection" not in detail:
        raise SystemExit(f"rejected recovery must fail closed: {ok} {detail}")

    uncertain_snapshot = dict(clean_snapshot)
    uncertain_snapshot["state_counts"] = dict(clean_snapshot["state_counts"], uncertain=1, failed=1)
    uncertain_snapshot["chunk_receipts"] = 4
    uncertain_snapshot["occurrences"] = 4
    uncertain_snapshot["uncertain_occurrences"] = 1
    uncertain_snapshot["failed_occurrences"] = 1
    SnapshotStore.snapshot = uncertain_snapshot
    with mock.patch("jarvis_v2.memory.store.MemoryStore", SnapshotStore):
        ok, detail = live_check._check_network_recovery_readiness(config)
    if ok is not False or "1 scheduled-delivery outcome(s) need manual review" not in detail:
        raise SystemExit(f"uncertain recovery must fail closed before ordinary failures: {ok} {detail}")

    active_snapshot = dict(clean_snapshot)
    active_snapshot["state_counts"] = dict(clean_snapshot["state_counts"], sending=1)
    active_snapshot["chunk_receipts"] = 3
    active_snapshot["occurrences"] = 3
    active_snapshot["active_occurrences"] = 1
    active_snapshot["content_bearing_chunks"] = 1
    SnapshotStore.snapshot = active_snapshot
    with mock.patch("jarvis_v2.memory.store.MemoryStore", SnapshotStore):
        ok, detail = live_check._check_network_recovery_readiness(config)
    if ok is not None or "1 scheduled-delivery occurrence(s) still active" not in detail:
        raise SystemExit(f"active recovery should remain unproven: {ok} {detail}")


def test_live_check_network_recovery_rejects_malformed_or_unreadable_evidence_safely() -> None:
    class SnapshotStore:
        snapshot: object = {"state_counts": {"accepted": "/private/tmp/secret SHOULD NOT APPEAR"}}

        def __init__(self, _db_path: object) -> None:
            pass

        def scheduled_delivery_operational_snapshot(self) -> object:
            return self.snapshot

    config = SimpleNamespace(db_path="unused")
    with mock.patch("jarvis_v2.memory.store.MemoryStore", SnapshotStore):
        ok, detail = live_check._check_network_recovery_readiness(config)
    if ok is not False or "ledger is malformed" not in detail:
        raise SystemExit(f"malformed recovery ledger must fail safely: {ok} {detail}")
    for forbidden in ("/private/tmp", "secret", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"malformed recovery detail leaked private state {forbidden!r}: {detail}")

    class ExplodingStore(SnapshotStore):
        def scheduled_delivery_operational_snapshot(self) -> object:
            raise RuntimeError("/\x55sers/example/private/recovery SHOULD NOT APPEAR")

    with mock.patch("jarvis_v2.memory.store.MemoryStore", ExplodingStore):
        ok, detail = live_check._check_network_recovery_readiness(config)
    if ok is not False or "ledger unreadable: RuntimeError: <local-path>" not in detail:
        raise SystemExit(f"unreadable recovery ledger must fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "private", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"unreadable recovery detail leaked private state {forbidden!r}: {detail}")


def _scheduled_streak_history(
    start_day: int,
    days: int,
    *,
    status: str = "ok",
    job_type: str,
    schedule_identity_revision: int,
    interval_minutes: int,
) -> str:
    events = []
    cursor = datetime(2026, 7, start_day, 9, 0, 0)
    end = cursor + timedelta(days=days)
    index = 0
    while cursor < end:
        events.append(
            {
                "occurrence_key": f"legacy:{schedule_identity_revision:032x}{index:032x}",
                "date": cursor.date().isoformat(),
                "ran_at": cursor.isoformat(),
                "status": status,
                "job_type": job_type,
                "schedule_identity_revision": schedule_identity_revision,
            }
        )
        cursor += timedelta(minutes=interval_minutes)
        index += 1
    return json.dumps({"run_history_schema": 3, "run_history": events})


def _scheduled_streak_rows(*, days: int = 7, status: str = "ok", malformed: object | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (job_type, interval_minutes) in enumerate(
        live_check.SCHEDULED_JOB_EXPECTED_INTERVALS.items()
    ):
        schedule_identity_revision = index + 1
        metadata = (
            malformed
            if malformed is not None and index == 0
            else _scheduled_streak_history(
                2,
                days,
                status=status,
                job_type=job_type,
                schedule_identity_revision=schedule_identity_revision,
                interval_minutes=interval_minutes,
            )
        )
        rows.append(
            {
                "name": f"/\x55sers/example/private/job-{index} SHOULD NOT APPEAR",
                "enabled": 1,
                "interval_minutes": interval_minutes,
                "job_type": job_type,
                "schedule_identity_revision": schedule_identity_revision,
                "last_run_at": "2026-07-08T09:00:00",
                "next_run_at": "2026-07-09T09:00:00",
                "metadata": metadata,
            }
        )
    return rows


def test_live_check_scheduled_job_streak_reports_ready_without_job_leakage() -> None:
    now = datetime(2026, 7, 9, 9, 5, 0)

    class ReadyStreakStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return _scheduled_streak_rows(days=7)

    with mock.patch("jarvis_v2.memory.store.MemoryStore", ReadyStreakStore):
        ok, detail = live_check._check_scheduled_job_streak(SimpleNamespace(db_path="unused"), now=now)
    if ok is not True:
        raise SystemExit(f"7-day scheduled-job streak should pass: {ok} {detail}")
    for expected in ("8/8 expected enabled job(s)", "7 consecutive complete day(s)", "2026-07-09"):
        if expected not in detail:
            raise SystemExit(f"ready scheduled-job streak detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/operator", "job-0", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"ready scheduled-job streak detail leaked job names/content {forbidden!r}: {detail}")


def test_live_check_scheduled_job_streak_reports_incomplete_without_overclaiming() -> None:
    now = datetime(2026, 7, 9, 9, 5, 0)

    class IncompleteStreakStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return _scheduled_streak_rows(days=3)

    with mock.patch("jarvis_v2.memory.store.MemoryStore", IncompleteStreakStore):
        ok, detail = live_check._check_scheduled_job_streak(SimpleNamespace(db_path="unused"), now=now)
    if ok is not None:
        raise SystemExit(f"incomplete scheduled-job streak should be not-yet-proven: {ok} {detail}")
    for expected in (
        "8/8 expected enabled job(s)",
        "retained proof spans up to",
        "but is not consecutive",
        "job cadence gap(s)",
        "need 7 consecutive complete day(s)",
    ):
        if expected not in detail:
            raise SystemExit(f"incomplete scheduled-job streak detail missed {expected!r}: {detail}")
    if "consecutive complete day(s) recorded" in detail:
        raise SystemExit(f"gapped scheduled-job proof overclaimed continuity: {detail}")
    for forbidden in ("/\x55sers/operator", "job-0", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"incomplete scheduled-job streak detail leaked job names/content {forbidden!r}: {detail}")


def test_live_check_scheduled_job_streak_flags_failed_or_invalid_history_safely() -> None:
    now = datetime(2026, 7, 9, 9, 5, 0)

    class FailedStreakStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return _scheduled_streak_rows(days=7, status="failed")

    with mock.patch("jarvis_v2.memory.store.MemoryStore", FailedStreakStore):
        ok, detail = live_check._check_scheduled_job_streak(SimpleNamespace(db_path="unused"), now=now)
    if ok is not False or "failed scheduled-job run event" not in detail:
        raise SystemExit(f"failed scheduled-job history should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/operator", "job-0", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"failed scheduled-job streak detail leaked job names/content {forbidden!r}: {detail}")

    class InvalidStreakStore:
        def __init__(self, _db_path: object) -> None:
            pass

        def list_jobs(self) -> list[dict[str, object]]:
            return _scheduled_streak_rows(malformed={"run_history": [{"date": "/private/tmp/bad SHOULD NOT APPEAR"}]})

    with mock.patch("jarvis_v2.memory.store.MemoryStore", InvalidStreakStore):
        ok, detail = live_check._check_scheduled_job_streak(SimpleNamespace(db_path="unused"), now=now)
    if ok is not False or "invalid run-history event" not in detail:
        raise SystemExit(f"invalid scheduled-job history should fail safely: {ok} {detail}")
    for forbidden in ("/private/tmp", "/\x55sers/operator", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"invalid scheduled-job streak detail leaked metadata content {forbidden!r}: {detail}")


def test_daemon_source_freshness_reports_current_and_stale_code_safely() -> None:
    process_started_at = 200.0
    snapshots = [
        live_check.DaemonProcessSnapshot(
            module=str(contract["module"]),
            started_at_epoch=process_started_at,
        )
        for contract in live_check.DAEMON_SOURCE_FRESHNESS_CONTRACTS.values()
    ]
    with TemporaryDirectory(prefix="jarvis-daemon-source-fresh-") as temp:
        root = Path(temp)
        _write_daemon_source_freshness_contract(root, modified_at=100.0)
        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=snapshots,
            git_reader=_stable_clean_git_reader,
        )
        if (
            ok is not True
            or "2/2 primary daemons" not in detail
            or "clean stable HEAD" not in detail
            or "loaded-memory identity is not attested" not in detail
        ):
            raise SystemExit(f"current daemon source should report ready: {ok} {detail}")

        stale_path = root / "jarvis_v2" / "tools" / "weather_connector.py"
        stale_path.write_text("# changed production connector\n", encoding="utf-8")
        os.utime(stale_path, (300.0, 300.0))
        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=snapshots,
            git_reader=_stable_clean_git_reader,
        )
    if ok is not False or "stale service source: Telegram, dashboard" not in detail:
        raise SystemExit(f"stale daemon source should fail explicitly: {ok} {detail}")
    if "restart manually" not in detail or "rerun live_check" not in detail:
        raise SystemExit(f"stale daemon source missed bounded recovery: {detail}")
    for forbidden in (
        "/\x55sers/",
        "/private/",
        "/tmp/",
        str(root),
        "telegram_control.py",
        "weather_connector.py",
        "requirements.txt",
        "200",
        "300",
        "SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(f"daemon source freshness leaked path/process detail {forbidden!r}: {detail}")


def test_daemon_source_freshness_tracks_runtime_closure_without_docs_noise() -> None:
    process_started_at = 200.0
    snapshots = [
        live_check.DaemonProcessSnapshot(
            module=str(contract["module"]),
            started_at_epoch=process_started_at,
        )
        for contract in live_check.DAEMON_SOURCE_FRESHNESS_CONTRACTS.values()
    ]
    with TemporaryDirectory(prefix="jarvis-daemon-runtime-source-") as temp:
        root = Path(temp)
        _write_daemon_source_freshness_contract(root, modified_at=100.0)

        readme = root / "README.md"
        readme.write_text("documentation-only change\n", encoding="utf-8")
        os.utime(readme, (300.0, 300.0))
        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=snapshots,
            git_reader=_stable_clean_git_reader,
        )
        if ok is not True:
            raise SystemExit(f"documentation-only change should not stale daemons: {ok} {detail}")

        dashboard_source = root / "jarvis_v2" / "ui" / "status_config.py"
        dashboard_source.write_text("# changed dashboard dependency\n", encoding="utf-8")
        os.utime(dashboard_source, (300.0, 300.0))
        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=snapshots,
            git_reader=_stable_clean_git_reader,
        )
    if ok is not False or "stale service source: dashboard" not in detail:
        raise SystemExit(f"dashboard runtime dependency should stale dashboard only: {ok} {detail}")
    if "Telegram" in detail:
        raise SystemExit(f"dashboard-only dependency incorrectly staled Telegram: {detail}")
    for forbidden in (
        "/\x55sers/",
        "/private/",
        "/tmp/",
        str(root),
        "status_config.py",
        "README.md",
        "200",
        "300",
    ):
        if forbidden in detail:
            raise SystemExit(f"runtime-closure freshness leaked detail {forbidden!r}: {detail}")


def test_daemon_source_freshness_flags_missing_duplicate_and_unavailable_state() -> None:
    telegram_module = str(
        live_check.DAEMON_SOURCE_FRESHNESS_CONTRACTS["Telegram"]["module"]
    )
    dashboard_module = str(
        live_check.DAEMON_SOURCE_FRESHNESS_CONTRACTS["dashboard"]["module"]
    )
    with TemporaryDirectory(prefix="jarvis-daemon-source-state-") as temp:
        root = Path(temp)
        _write_daemon_source_freshness_contract(root, modified_at=100.0)
        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=[
                live_check.DaemonProcessSnapshot(
                    module=telegram_module,
                    started_at_epoch=200.0,
                )
            ],
            git_reader=_stable_clean_git_reader,
        )
        if ok is not False or "not running: dashboard" not in detail:
            raise SystemExit(f"missing primary daemon should fail explicitly: {ok} {detail}")

        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=[
                live_check.DaemonProcessSnapshot(
                    module=telegram_module,
                    started_at_epoch=200.0,
                ),
                live_check.DaemonProcessSnapshot(
                    module=dashboard_module,
                    started_at_epoch=200.0,
                ),
                live_check.DaemonProcessSnapshot(
                    module=dashboard_module,
                    started_at_epoch=201.0,
                ),
            ],
            git_reader=_stable_clean_git_reader,
        )
        if ok is not False or "multiple primary daemon processes detected: dashboard" not in detail:
            raise SystemExit(f"duplicate primary daemon should fail explicitly: {ok} {detail}")

        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=[],
            process_scan_ok=False,
        )
        if ok is not None or "freshness unavailable" not in detail:
            raise SystemExit(f"unavailable process scan should stay unproven: {ok} {detail}")
    for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root), "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"daemon source-state detail leaked {forbidden!r}: {detail}")


def test_daemon_source_git_reader_is_bounded_read_only_and_fail_closed() -> None:
    root = Path("/reviewed/jarvis-v3")
    clean_output = f"# branch.oid {_CLEAN_SOURCE_HEAD}\n# branch.head main\n"
    tracked_dirty_output = clean_output + "1 .M N... 100644 100644 100644 a b private.py\n"
    untracked_output = clean_output + "? private-untracked-token.txt\n"
    for output, expected_clean in (
        (clean_output, True),
        (tracked_dirty_output, False),
        (untracked_output, False),
    ):
        completed = SimpleNamespace(returncode=0, stdout=output)
        with mock.patch(
            "jarvis_v2.scripts.live_check.subprocess.run",
            return_value=completed,
        ) as run:
            snapshot, read_ok = live_check._read_daemon_source_git_state(root)
        if (
            not read_ok
            or not isinstance(snapshot, live_check.DaemonSourceGitSnapshot)
            or snapshot.head != _CLEAN_SOURCE_HEAD
            or snapshot.clean is not expected_clean
        ):
            raise SystemExit(f"daemon source Git status parse drifted: {read_ok} {snapshot}")
        command = run.call_args.args[0]
        if command != [
            "/usr/bin/git",
            "-C",
            str(root),
            "status",
            "--porcelain=v2",
            "--branch",
            "--untracked-files=normal",
            "--no-ahead-behind",
        ]:
            raise SystemExit(f"daemon source Git reader used unexpected argv: {command}")
        if (
            run.call_args.kwargs.get("timeout") != 3.0
            or run.call_args.kwargs.get("check") is not False
        ):
            raise SystemExit(f"daemon source Git reader lost bounded execution: {run.call_args}")
        for forbidden in ("commit", "checkout", "reset", "clean", "restore", "switch"):
            if forbidden in command:
                raise SystemExit(f"daemon source Git reader used mutating verb {forbidden!r}")

    failures = (
        mock.Mock(return_value=SimpleNamespace(returncode=1, stdout="private failure")),
        mock.Mock(side_effect=subprocess.TimeoutExpired(["git"], 3.0)),
        mock.Mock(
            return_value=SimpleNamespace(
                returncode=0,
                stdout=clean_output + "? " + "x" * live_check.DAEMON_SOURCE_GIT_STATUS_MAX_CHARS,
            )
        ),
        mock.Mock(
            return_value=SimpleNamespace(
                returncode=0,
                stdout="# branch.oid invalid\n? private-untracked-token.txt\n",
            )
        ),
    )
    for runner in failures:
        with mock.patch("jarvis_v2.scripts.live_check.subprocess.run", runner):
            snapshot, read_ok = live_check._read_daemon_source_git_state(root)
        if read_ok or snapshot is not None:
            raise SystemExit(f"daemon source Git reader did not fail closed: {read_ok} {snapshot}")


def test_daemon_source_freshness_requires_stable_clean_git_head() -> None:
    snapshots = [
        live_check.DaemonProcessSnapshot(
            module=str(contract["module"]),
            started_at_epoch=200.0,
        )
        for contract in live_check.DAEMON_SOURCE_FRESHNESS_CONTRACTS.values()
    ]
    with TemporaryDirectory(prefix="jarvis-daemon-git-boundary-") as temp:
        root = Path(temp)
        _write_daemon_source_freshness_contract(root, modified_at=100.0)
        backdated = root / "jarvis_v2" / "tools" / "weather_connector.py"
        backdated.write_text("# uncommitted content hidden by old mtime\n", encoding="utf-8")
        os.utime(backdated, (100.0, 100.0))

        dirty = live_check.DaemonSourceGitSnapshot(head=_CLEAN_SOURCE_HEAD, clean=False)
        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=snapshots,
            git_reader=lambda _root: (dirty, True),
        )
        if ok is not False or "worktree is not clean" not in detail:
            raise SystemExit(f"backdated uncommitted source should fail closed: {ok} {detail}")

        heads = iter(
            (
                (live_check.DaemonSourceGitSnapshot(head="a" * 40, clean=True), True),
                (live_check.DaemonSourceGitSnapshot(head="b" * 40, clean=True), True),
            )
        )
        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=snapshots,
            git_reader=lambda _root: next(heads),
        )
        if ok is not False or "Git state changed during inspection" not in detail:
            raise SystemExit(f"source HEAD drift should fail closed: {ok} {detail}")

        reads = iter(
            (
                (live_check.DaemonSourceGitSnapshot(head=_CLEAN_SOURCE_HEAD, clean=True), True),
                (None, False),
            )
        )
        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=snapshots,
            git_reader=lambda _root: next(reads),
        )
        if ok is not False or "Git state unavailable after inspection" not in detail:
            raise SystemExit(f"second Git read failure should fail closed: {ok} {detail}")

        ok, detail = live_check._check_daemon_source_freshness(
            root,
            process_snapshots=snapshots,
            git_reader=lambda _root: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(["git"], 3.0)
            ),
        )
        if ok is not False or "Git state unavailable" not in detail:
            raise SystemExit(f"Git reader timeout should fail closed: {ok} {detail}")
    for forbidden in (
        "/\x55sers/",
        "/private/",
        "/tmp/",
        str(root),
        "weather_connector.py",
        "private-untracked-token",
        _CLEAN_SOURCE_HEAD,
    ):
        if forbidden in detail:
            raise SystemExit(f"daemon source Git refusal leaked {forbidden!r}: {detail}")


def test_daemon_source_freshness_process_scan_is_read_only_and_content_free() -> None:
    seeded_output = (
        "39567 Thu Jul 30 07:58:53 2026 /opt/homebrew/bin/python3 "
        "-m jarvis_v2.scripts.run_telegram_control\n"
        "39596 Thu Jul 30 07:59:00 2026 /opt/homebrew/bin/python3 "
        "-m jarvis_v2.scripts.run_status_server\n"
        "99998 Thu Jul 30 07:59:30 2026 /private/attacker.py --flag "
        "-m jarvis_v2.scripts.run_status_server\n"
        "99999 Thu Jul 30 08:00:00 2026 /private/SHOULD_NOT_APPEAR "
        "--token SECRET SHOULD NOT APPEAR\n"
    )
    completed = SimpleNamespace(returncode=0, stdout=seeded_output)
    with mock.patch(
        "jarvis_v2.scripts.live_check.subprocess.run",
        return_value=completed,
    ) as run:
        snapshots, scan_ok = live_check._read_primary_daemon_processes()
    if not scan_ok or {snapshot.module for snapshot in snapshots} != {
        str(contract["module"])
        for contract in live_check.DAEMON_SOURCE_FRESHNESS_CONTRACTS.values()
    }:
        raise SystemExit(f"daemon process scan missed primary modules: {scan_ok} {snapshots}")
    if len(snapshots) != 2:
        raise SystemExit(f"daemon process scan retained unrelated process content: {snapshots}")
    if {snapshot.pid for snapshot in snapshots} != {39567, 39596}:
        raise SystemExit(f"daemon process scan lost PID provenance: {snapshots}")
    command = run.call_args.args[0]
    if command != ["/bin/ps", "-axo", "pid=,lstart=,command="]:
        raise SystemExit(f"daemon process scan invoked unexpected command: {command}")
    if run.call_args.kwargs.get("timeout") != 3.0 or run.call_args.kwargs.get("check") is not False:
        raise SystemExit(f"daemon process scan lost bounded read-only execution: {run.call_args}")


def test_daemon_startup_live_check_reports_ready_without_path_leakage() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-ready-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        selector = _install_primary_startup_contracts(root, launch_agents)
        launchd_reader, process_reader = _proven_primary_startup_readers(root, selector)
        with mock.patch.dict(
            os.environ,
            {live_check.STARTUP_SELECTED_ENV: str(selector)},
        ):
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
                launchd_reader=launchd_reader,
                process_reader=process_reader,
            )
    if ok is not True:
        raise SystemExit(f"daemon startup readiness should pass with installed primary contract: {ok} {detail}")
    for expected in (
        "startup wiring ready",
        "3 launcher(s)",
        "3 inert LaunchAgent template(s)",
        "RunAtLoad+KeepAlive",
        "Telegram daemon owns scheduler ticker",
        "2/2 primary LaunchAgent installed",
        "2/2 explicitly activated",
        "private recovery contract present",
        "scheduler activation remains separate",
        "2/2 activated primary launchd jobs own stable running service processes",
        "no process control run",
        "live reboot proof still needs operator present",
    ):
        if expected not in detail:
            raise SystemExit(f"daemon startup readiness detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root), "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"daemon startup readiness leaked path/private detail {forbidden!r}: {detail}")


def test_daemon_startup_live_check_defers_without_launchd_pid_provenance() -> None:
    cases = (
        ("inspection-unavailable", "provenance inspection unavailable"),
        ("job-not-running", "job state is not proven running"),
        ("pid-not-owned", "PID ownership of the expected service process is not proven"),
    )
    for label, expected in cases:
        with TemporaryDirectory(prefix=f"jarvis-daemon-provenance-{label}-") as temp:
            root = Path(temp)
            launch_agents = _write_startup_contract(root)
            _write_daemon_restart_doc(root)
            selector = _install_primary_startup_contracts(root, launch_agents)
            contracts = [
                contract
                for contract in live_check.STARTUP_PLIST_CONTRACTS.values()
                if contract.get("primary")
            ]
            jobs = [
                _loaded_job_snapshot(
                    contract,
                    root=root,
                    selector=selector,
                    pid=5200 + index,
                    state="waiting" if label == "job-not-running" else "running",
                )
                for index, contract in enumerate(contracts)
            ]
            processes = [
                live_check.DaemonProcessSnapshot(
                    module=str(contract["module"]),
                    started_at_epoch=1_800_000_000.0,
                    pid=6200 + index if label == "pid-not-owned" else 5200 + index,
                )
                for index, contract in enumerate(contracts)
            ]
            launchd_reader = (
                (lambda _labels: ([], False))
                if label == "inspection-unavailable"
                else (lambda _labels, rows=jobs: (rows, True))
            )
            process_reader = (
                (lambda: ([], False))
                if label == "inspection-unavailable"
                else (lambda rows=processes: (rows, True))
            )
            with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selector)}):
                ok, detail = live_check._check_daemon_startup_readiness(
                    root=root,
                    launch_agents_dir=launch_agents,
                    launchd_reader=launchd_reader,
                    process_reader=process_reader,
                )
        if ok is not None or expected not in detail:
            raise SystemExit(f"{label} launchd provenance should defer: {ok} {detail}")
        for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root), "SHOULD NOT APPEAR"):
            if forbidden in detail:
                raise SystemExit(f"{label} launchd provenance detail leaked {forbidden!r}: {detail}")


def test_launchd_primary_job_inspection_is_read_only_and_bounded() -> None:
    labels = ("com.jarvis-v3.telegram", "com.jarvis-v3.dashboard")
    modules = (
        "jarvis_v2.scripts.run_telegram_control",
        "jarvis_v2.scripts.run_status_server",
    )
    selector = "/reviewed/v3/runtime.env"
    working_directory = "/reviewed/v3"
    def launchctl_output(module: str, pid: int) -> str:
        return (
            "state = running\n"
            f"pid = {pid}\n"
            "program = /opt/homebrew/bin/python3\n"
            "arguments = {\n"
            "  /opt/homebrew/bin/python3\n"
            "  -m\n"
            f"  {module}\n"
            "}\n"
            f"working directory = {working_directory}\n"
            "environment = {\n"
            f"  {live_check.V3_DAEMON_ENABLE_ENV} => 1\n"
            f"  {live_check.STARTUP_SELECTED_ENV} => {selector}\n"
            f"  {live_check.V3_SCHEDULER_ENABLE_ENV} => 0\n"
            "}\n"
        )
    outputs = (
        SimpleNamespace(returncode=0, stdout=launchctl_output(modules[0], 7101)),
        SimpleNamespace(returncode=0, stdout=launchctl_output(modules[1], 7102)),
    )
    with (
        mock.patch.object(live_check.sys, "platform", "darwin"),
        mock.patch("jarvis_v2.scripts.live_check.subprocess.run", side_effect=outputs) as run,
    ):
        snapshots, inspection_ok = live_check._read_launchd_primary_jobs(labels)
    if not inspection_ok or [(row.label, row.pid, row.state) for row in snapshots] != [
        (labels[0], 7101, "running"),
        (labels[1], 7102, "running"),
    ]:
        raise SystemExit(f"read-only launchd inspection result drifted: {inspection_ok} {snapshots}")
    for row, module in zip(snapshots, modules, strict=True):
        if (
            row.program != "/opt/homebrew/bin/python3"
            or row.program_arguments != ("/opt/homebrew/bin/python3", "-m", module)
            or row.working_directory != working_directory
            or row.daemon_enable != "1"
            or row.selected_environment != selector
            or row.scheduler_enable != "0"
            or row.environment != (
                (live_check.V3_DAEMON_ENABLE_ENV, "1"),
                (live_check.V3_SCHEDULER_ENABLE_ENV, "0"),
                (live_check.STARTUP_SELECTED_ENV, selector),
            )
        ):
            raise SystemExit(f"launchd loaded-contract metadata drifted: {row}")
    valid_output = launchctl_output(modules[0], 7101)
    missing_program = valid_output.replace("program = /opt/homebrew/bin/python3\n", "")
    duplicate_program = valid_output.replace(
        "program = /opt/homebrew/bin/python3\n",
        "program = /opt/homebrew/bin/python3\nprogram = /opt/homebrew/bin/python3\n",
    )
    wrong_program = valid_output.replace(
        "program = /opt/homebrew/bin/python3\n",
        "program = /stale/python\n",
    )
    if live_check._launchctl_loaded_contract(missing_program)[0] != "":
        raise SystemExit("launchd parser accepted a missing effective Program field")
    if live_check._launchctl_loaded_contract(duplicate_program)[0] != "":
        raise SystemExit("launchd parser accepted duplicate effective Program fields")
    if live_check._launchctl_loaded_contract(wrong_program)[0] != "/stale/python":
        raise SystemExit("launchd parser failed to retain a wrong Program for exact rejection")
    if run.call_count != len(labels):
        raise SystemExit(f"launchd inspection call count drifted: {run.call_count}")
    for call, label in zip(run.call_args_list, labels, strict=True):
        command = call.args[0]
        if command != ["/bin/launchctl", "print", f"gui/{os.getuid()}/{label}"]:
            raise SystemExit(f"launchd inspection used unexpected command: {command}")
        if call.kwargs.get("timeout") != 3.0 or call.kwargs.get("check") is not False:
            raise SystemExit(f"launchd inspection lost bounded execution: {call}")
        for forbidden in ("load", "unload", "bootstrap", "bootout", "kickstart", "enable", "disable"):
            if forbidden in command:
                raise SystemExit(f"launchd inspection used control verb {forbidden!r}: {command}")


def test_daemon_startup_live_check_rejects_stale_loaded_launchd_contract() -> None:
    with TemporaryDirectory(prefix="jarvis-loaded-contract-drift-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        selector = _install_primary_startup_contracts(root, launch_agents)
        valid_launchd_reader, process_reader = _proven_primary_startup_readers(root, selector)
        labels = tuple(
            str(contract["label"])
            for contract in live_check.STARTUP_PLIST_CONTRACTS.values()
            if contract.get("primary")
        )
        valid_jobs, _ = valid_launchd_reader(labels)
        stale_environment = dict(valid_jobs[0].environment)
        stale_environment["PYTHONPATH"] = "/stale/source"
        unexpected_environment = dict(valid_jobs[0].environment)
        unexpected_environment["PYTHONHOME"] = "/stale/python-home"
        wrong_xpc_environment = dict(valid_jobs[0].environment)
        wrong_xpc_environment["XPC_SERVICE_NAME"] = "com.unreviewed.service"
        mutations = {
            "arguments": replace(
                valid_jobs[0],
                program_arguments=valid_jobs[0].program_arguments + ("--unexpected",),
            ),
            "program-missing": replace(valid_jobs[0], program=""),
            "program-wrong": replace(valid_jobs[0], program="/stale/python"),
            "working-directory": replace(valid_jobs[0], working_directory="/stale/source"),
            "daemon-missing": replace(valid_jobs[0], daemon_enable=None),
            "daemon-whitespace": replace(valid_jobs[0], daemon_enable=" 1 "),
            "selector": replace(valid_jobs[0], selected_environment="/stale/runtime.env"),
            "scheduler-missing": replace(valid_jobs[0], scheduler_enable=None),
            "scheduler-enabled": replace(valid_jobs[0], scheduler_enable="1"),
            "scheduler-whitespace": replace(valid_jobs[0], scheduler_enable=" 0 "),
            "pythonpath": replace(
                valid_jobs[0],
                environment=tuple(sorted(stale_environment.items())),
            ),
            "unexpected-pythonhome": replace(
                valid_jobs[0],
                environment=tuple(sorted(unexpected_environment.items())),
            ),
            "wrong-xpc-service": replace(
                valid_jobs[0],
                environment=tuple(sorted(wrong_xpc_environment.items())),
            ),
            "duplicate-extra-environment": replace(
                valid_jobs[0],
                environment=valid_jobs[0].environment
                + (("XPC_SERVICE_NAME", "one"), ("XPC_SERVICE_NAME", "two")),
            ),
        }
        for label, first_job in mutations.items():
            jobs = [first_job, *valid_jobs[1:]]
            with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selector)}):
                ok, detail = live_check._check_daemon_startup_readiness(
                    root=root,
                    launch_agents_dir=launch_agents,
                    launchd_reader=lambda _labels, rows=jobs: (rows, True),
                    process_reader=process_reader,
                )
            if ok is not False or "loaded launchd job configuration does not match" not in detail:
                raise SystemExit(f"{label} loaded launchd drift should fail closed: {ok} {detail}")
            for forbidden in (
                "/\x55sers/",
                "/private/",
                "/tmp/",
                str(root),
                "/stale/",
            ):
                if forbidden in detail:
                    raise SystemExit(f"{label} loaded launchd drift leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_allows_duplicate_free_launchd_environment_extras() -> None:
    with TemporaryDirectory(prefix="jarvis-loaded-contract-extra-env-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        selector = _install_primary_startup_contracts(root, launch_agents)
        valid_launchd_reader, process_reader = _proven_primary_startup_readers(root, selector)
        labels = tuple(
            str(contract["label"])
            for contract in live_check.STARTUP_PLIST_CONTRACTS.values()
            if contract.get("primary")
        )
        valid_jobs, _ = valid_launchd_reader(labels)
        jobs = []
        for job in valid_jobs:
            environment = dict(job.environment)
            environment["XPC_SERVICE_NAME"] = job.label
            jobs.append(replace(job, environment=tuple(sorted(environment.items()))))
        with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selector)}):
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
                launchd_reader=lambda _labels: (jobs, True),
                process_reader=process_reader,
            )
    if ok is not True:
        raise SystemExit(f"duplicate-free launchd environment extras should be allowed: {ok} {detail}")
    for forbidden in ("XPC_SERVICE_NAME", "/\x55sers/", "/private/", "/tmp/", str(root)):
        if forbidden in detail:
            raise SystemExit(f"launchd environment extra leaked into readiness detail: {detail}")


def test_daemon_startup_live_check_allows_reviewed_exact_program_contract() -> None:
    with TemporaryDirectory(prefix="jarvis-reviewed-program-contract-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        for contract in live_check.STARTUP_PLIST_CONTRACTS.values():
            path = root / Path(str(contract["path"])).name
            with path.open("rb") as handle:
                payload = plistlib.load(handle)
            payload["Program"] = payload["ProgramArguments"][0]
            with path.open("wb") as handle:
                plistlib.dump(payload, handle)
        selector = _install_primary_startup_contracts(root, launch_agents)
        launchd_reader, process_reader = _proven_primary_startup_readers(root, selector)
        with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selector)}):
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
                launchd_reader=launchd_reader,
                process_reader=process_reader,
            )
    if ok is not True:
        raise SystemExit(f"reviewed exact Program contract should be allowed: {ok} {detail}")


def test_daemon_startup_live_check_rejects_unsafe_installed_file_custody() -> None:
    for label in ("group-readable", "owner-executable", "hard-linked", "symlink"):
        with TemporaryDirectory(prefix=f"jarvis-daemon-custody-{label}-") as temp:
            root = Path(temp)
            launch_agents = _write_startup_contract(root)
            _write_daemon_restart_doc(root)
            selector = _install_primary_startup_contracts(root, launch_agents)
            installed = launch_agents / "com.jarvis-v3.telegram.plist"
            if label == "group-readable":
                installed.chmod(0o640)
            elif label == "owner-executable":
                installed.chmod(0o700)
            elif label == "hard-linked":
                os.link(installed, launch_agents / "linked-copy.plist")
            else:
                installed.unlink()
                installed.symlink_to(root / "com.jarvis-v3.telegram.plist")
            with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selector)}):
                ok, detail = live_check._check_daemon_startup_readiness(
                    root=root,
                    launch_agents_dir=launch_agents,
                )
        if ok is not False or "installed LaunchAgent unreadable/malformed" not in detail:
            raise SystemExit(f"{label} installed plist custody should fail closed: {ok} {detail}")
        for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root)):
            if forbidden in detail:
                raise SystemExit(f"{label} installed plist custody leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_linked_selected_environment() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-selector-hardlink-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        selector = _install_primary_startup_contracts(root, launch_agents)
        os.link(selector, root / "selector-copy.env")
        with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selector)}):
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
            )
    if ok is not False or "selected V3 environment has unsafe custody" not in detail:
        raise SystemExit(f"linked selected environment should fail closed: {ok} {detail}")
    for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root), "selector-copy"):
        if forbidden in detail:
            raise SystemExit(f"linked selected environment leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_symlinked_selected_environment_parent() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-selector-parent-") as temp:
        root = Path(temp).resolve()
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        _install_primary_startup_contracts(root, launch_agents)
        real_parent = root / "real-private-parent"
        real_parent.mkdir()
        real_selector = real_parent / "runtime.env"
        real_selector.write_text("", encoding="utf-8")
        real_selector.chmod(0o600)
        linked_parent = root / "linked-private-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        selected = linked_parent / "runtime.env"
        for filename in ("com.jarvis-v3.telegram.plist", "com.jarvis-v3.dashboard.plist"):
            installed_path = launch_agents / filename
            with installed_path.open("rb") as handle:
                installed = plistlib.load(handle)
            installed["EnvironmentVariables"][live_check.STARTUP_SELECTED_ENV] = str(selected)
            with installed_path.open("wb") as handle:
                plistlib.dump(installed, handle)
        with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selected)}):
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
            )
    if ok is not False or "selected V3 environment has unsafe custody" not in detail:
        raise SystemExit(f"symlink-parent selected environment should fail closed: {ok} {detail}")
    for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root), "linked-private"):
        if forbidden in detail:
            raise SystemExit(f"symlink-parent selected environment leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_unsafe_launchagents_directory_custody() -> None:
    for label in ("writable", "symlink-parent"):
        with TemporaryDirectory(prefix=f"jarvis-launchagents-custody-{label}-") as temp:
            root = Path(temp)
            launch_agents = _write_startup_contract(root)
            _write_daemon_restart_doc(root)
            if label == "writable":
                launch_agents.chmod(0o777)
                inspected = launch_agents
            else:
                real_agents = root / "real-launch-agents"
                launch_agents.rename(real_agents)
                launch_agents.symlink_to(real_agents, target_is_directory=True)
                inspected = launch_agents
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=inspected,
            )
        expected = "unsafe custody" if label == "writable" else "directory is malformed"
        if ok is not False or expected not in detail:
            raise SystemExit(f"{label} LaunchAgents directory custody should fail closed: {ok} {detail}")
        for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root), "real-launch"):
            if forbidden in detail:
                raise SystemExit(f"{label} LaunchAgents custody leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_uses_account_database_home() -> None:
    with TemporaryDirectory(prefix="jarvis-account-home-launchagents-") as temp:
        base = Path(temp).resolve()
        root = base / "repo"
        root.mkdir()
        account_home = base / "account-home"
        account_home.mkdir()
        account = SimpleNamespace(pw_dir=str(account_home))
        with mock.patch.object(live_check.pwd, "getpwuid", return_value=account):
            _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        launch_agents = account_home / "Library" / "LaunchAgents"
        launch_agents.mkdir(parents=True)
        launch_agents.chmod(0o700)
        selector = _install_primary_startup_contracts(root, launch_agents)
        launchd_reader, process_reader = _proven_primary_startup_readers(root, selector)
        with (
            mock.patch.object(live_check.pwd, "getpwuid", return_value=account),
            mock.patch.dict(
                os.environ,
                {
                    "HOME": str(base / "unreviewed-home-SHOULD-NOT-APPEAR"),
                    live_check.STARTUP_SELECTED_ENV: str(selector),
                },
            ),
        ):
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launchd_reader=launchd_reader,
                process_reader=process_reader,
            )
        linked_home = base / "linked-account-home"
        linked_home.symlink_to(account_home, target_is_directory=True)
        linked_account = SimpleNamespace(pw_dir=str(linked_home))
        with (
            mock.patch.object(live_check.pwd, "getpwuid", return_value=linked_account),
            mock.patch.dict(
                os.environ,
                {live_check.STARTUP_SELECTED_ENV: str(selector)},
            ),
        ):
            linked_ok, linked_detail = live_check._check_daemon_startup_readiness(
                root=root,
                launchd_reader=launchd_reader,
                process_reader=process_reader,
            )
    if ok is not True:
        raise SystemExit(f"account-database LaunchAgent root should be accepted: {ok} {detail}")
    for forbidden in ("unreviewed-home", "/\x55sers/", "/private/", "/tmp/", str(root)):
        if forbidden in detail:
            raise SystemExit(f"account-home LaunchAgent proof leaked {forbidden!r}: {detail}")
    if linked_ok is not False or "directory is malformed" not in linked_detail:
        raise SystemExit(
            f"symlinked account-database home should fail custody: {linked_ok} {linked_detail}"
        )

    relative_account = SimpleNamespace(pw_dir="relative-private-home")
    with mock.patch.object(live_check.pwd, "getpwuid", return_value=relative_account):
        try:
            live_check._account_launch_agents_directory()
        except OSError:
            pass
        else:
            raise SystemExit("relative account home should fail closed")


def test_daemon_log_custody_is_owner_only_and_no_follow() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-log-custody-") as temp:
        directory = Path(temp).resolve() / "JarvisV3"
        directory.mkdir(mode=live_check.STARTUP_LOG_DIRECTORY_MODE)
        filenames = tuple(
            str(contract["log_filename"])
            for contract in live_check.STARTUP_PLIST_CONTRACTS.values()
        )
        for filename in filenames:
            path = directory / filename
            path.write_text("", encoding="utf-8")
            path.chmod(live_check.STARTUP_LOG_FILE_MODE)
        if not live_check._owner_only_log_custody(directory, filenames):
            raise SystemExit("owner-only daemon log custody should pass")

        first = directory / filenames[0]
        first.chmod(0o644)
        if live_check._owner_only_log_custody(directory, filenames):
            raise SystemExit("shared daemon log permissions passed custody")
        first.chmod(live_check.STARTUP_LOG_FILE_MODE)
        first.unlink()
        outside = Path(temp) / "outside.log"
        outside.write_text("", encoding="utf-8")
        outside.chmod(live_check.STARTUP_LOG_FILE_MODE)
        first.symlink_to(outside)
        if live_check._owner_only_log_custody(directory, filenames):
            raise SystemExit("symlinked daemon log passed custody")
        first.unlink()
        first.write_text("", encoding="utf-8")
        first.chmod(live_check.STARTUP_LOG_FILE_MODE)
        directory.chmod(0o755)
        if live_check._owner_only_log_custody(directory, filenames):
            raise SystemExit("shared daemon log directory passed custody")


def test_private_plist_read_rejects_content_race_and_oversize() -> None:
    with TemporaryDirectory(prefix="jarvis-stable-private-plist-") as temp:
        directory = Path(temp).resolve()
        path = directory / "service.plist"
        original = {"Label": "reviewed"}
        changed = {"Label": "changed-SHOULD-NOT-APPEAR"}
        with path.open("wb") as handle:
            plistlib.dump(original, handle)
        path.chmod(0o600)
        real_loads = plistlib.loads

        def mutate_during_parse(raw: bytes) -> object:
            payload = real_loads(raw)
            with path.open("wb") as handle:
                plistlib.dump(changed, handle)
            path.chmod(0o600)
            return payload

        directory_fd = live_check._open_absolute_directory_fd(directory)
        try:
            with mock.patch.object(
                live_check.plistlib,
                "loads",
                side_effect=mutate_during_parse,
            ):
                try:
                    live_check._load_private_plist_at(directory_fd, path.name)
                except OSError:
                    pass
                else:
                    raise SystemExit("same-inode plist mutation during parse was accepted")

            path.write_bytes(b"x" * (live_check.STARTUP_PLIST_MAX_BYTES + 1))
            path.chmod(0o600)
            try:
                live_check._load_private_plist_at(directory_fd, path.name)
            except OSError:
                pass
            else:
                raise SystemExit("oversized installed plist was accepted")
        finally:
            os.close(directory_fd)


def test_launchd_primary_provenance_requires_closing_observation() -> None:
    with TemporaryDirectory(prefix="jarvis-closing-launchd-proof-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        selector = _install_primary_startup_contracts(root, launch_agents)
        valid_launchd_reader, process_reader = _proven_primary_startup_readers(root, selector)
        labels = tuple(
            str(contract["label"])
            for contract in live_check.STARTUP_PLIST_CONTRACTS.values()
            if contract.get("primary")
        )
        valid_jobs, _ = valid_launchd_reader(labels)
        closing_jobs = [replace(valid_jobs[0], pid=valid_jobs[0].pid + 99), *valid_jobs[1:]]
        observations = iter(((valid_jobs, True), (closing_jobs, True)))
        with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selector)}):
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
                launchd_reader=lambda _labels: next(observations),
                process_reader=process_reader,
            )
    if ok is not None or "PID ownership" not in detail:
        raise SystemExit(f"changed closing launchd observation should remain unproven: {ok} {detail}")
    for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root)):
        if forbidden in detail:
            raise SystemExit(f"closing launchd refusal leaked {forbidden!r}: {detail}")


def test_launchd_primary_provenance_rejects_duplicate_processes() -> None:
    with TemporaryDirectory(prefix="jarvis-duplicate-primary-proof-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        selector = _install_primary_startup_contracts(root, launch_agents)
        launchd_reader, valid_process_reader = _proven_primary_startup_readers(root, selector)
        processes, scanned = valid_process_reader()
        if not scanned:
            raise SystemExit("valid primary process fixture was not trusted")
        duplicate = replace(processes[0], pid=processes[0].pid + 99)

        with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selector)}):
            stable_ok, stable_detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
                launchd_reader=launchd_reader,
                process_reader=lambda: ([*processes, duplicate], True),
            )
        if stable_ok is not False or "multiple processes claim" not in stable_detail:
            raise SystemExit(
                "stable duplicate primary process should fail closed: "
                f"{stable_ok} {stable_detail}"
            )

        process_observations = iter(
            (
                (processes, True),
                ([*processes, duplicate], True),
            )
        )
        with mock.patch.dict(os.environ, {live_check.STARTUP_SELECTED_ENV: str(selector)}):
            closing_ok, closing_detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
                launchd_reader=launchd_reader,
                process_reader=lambda: next(process_observations),
            )
    if closing_ok is not False or "multiple processes claim" not in closing_detail:
        raise SystemExit(
            "closing duplicate primary process should fail closed: "
            f"{closing_ok} {closing_detail}"
        )
    for detail in (stable_detail, closing_detail):
        for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root)):
            if forbidden in detail:
                raise SystemExit(f"duplicate primary process refusal leaked {forbidden!r}: {detail}")


def test_private_file_custody_rejects_foreign_owner_snapshot() -> None:
    with TemporaryDirectory(prefix="jarvis-private-custody-owner-") as temp:
        path = Path(temp) / "private.plist"
        path.write_text("private", encoding="utf-8")
        path.chmod(0o600)
        fd = os.open(path, os.O_RDONLY)
        try:
            current = os.fstat(fd)
            fields = list(current)
            fields[4] = os.getuid() + 1
            foreign_owner = os.stat_result(fields)
            with mock.patch("jarvis_v2.scripts.live_check.os.fstat", return_value=foreign_owner):
                if live_check._private_regular_fd_custody(fd):
                    raise SystemExit("foreign-owner stable fd custody must fail closed")
        finally:
            os.close(fd)
        directory_fd = os.open(Path(temp), os.O_RDONLY)
        try:
            current = os.fstat(directory_fd)
            fields = list(current)
            fields[4] = os.getuid() + 1
            foreign_owner = os.stat_result(fields)
            with mock.patch("jarvis_v2.scripts.live_check.os.fstat", return_value=foreign_owner):
                if live_check._launch_agents_directory_custody(directory_fd):
                    raise SystemExit("foreign-owner LaunchAgents directory custody must fail closed")
        finally:
            os.close(directory_fd)


def test_daemon_startup_live_check_skips_when_primary_launchagent_not_installed() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-skip-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        ok, detail = live_check._check_daemon_startup_readiness(root=root, launch_agents_dir=launch_agents)
    if ok is not None:
        raise SystemExit(f"daemon startup without installed primary should be not-yet-proven: {ok} {detail}")
    for expected in (
        "startup wiring ready",
        "0/2 primary LaunchAgent installed",
        "0/2 explicitly activated",
        "private recovery contract present",
        "service activation/reboot deferred and not proven",
    ):
        if expected not in detail:
            raise SystemExit(f"daemon startup skip detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root)):
        if forbidden in detail:
            raise SystemExit(f"daemon startup skip detail leaked path {forbidden!r}: {detail}")


def test_daemon_startup_live_check_flags_malformed_contracts_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-drift-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(
            root,
            telegram_module="jarvis_v2.scripts.wrong_private_module_SHOULD_NOT_APPEAR",
            keep_alive=False,
        )
        _write_daemon_restart_doc(root)
        (root / "jarvis_v2" / "scripts" / "run_telegram_control.py").write_text(
            "def main():\n"
            "    secret = '/\x55sers/example/private/daemon SHOULD NOT APPEAR'\n"
            "    return secret\n",
            encoding="utf-8",
        )
        ok, detail = live_check._check_daemon_startup_readiness(root=root, launch_agents_dir=launch_agents)
    if ok is not False:
        raise SystemExit(f"malformed daemon startup contract should fail safely: {ok} {detail}")
    for expected in ("drifted 1 launcher contract(s)", "Telegram", "repair startup wiring"):
        if expected not in detail:
            raise SystemExit(f"daemon startup malformed detail missed {expected!r}: {detail}")
    for forbidden in (
        "/\x55sers/operator",
        "/private/",
        "/tmp/",
        str(root),
        "wrong_private_module",
        "SHOULD NOT APPEAR",
    ):
        if forbidden in detail:
            raise SystemExit(f"daemon startup malformed detail leaked seeded detail {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_unreadable_installed_primary_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-installed-malformed-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        (launch_agents / "com.jarvis-v3.telegram.plist").write_text(
            "installed secret /\x55sers/operator SHOULD NOT APPEAR",
            encoding="utf-8",
        )
        ok, detail = live_check._check_daemon_startup_readiness(root=root, launch_agents_dir=launch_agents)
    if ok is not False or "installed LaunchAgent unreadable/malformed" not in detail:
        raise SystemExit(f"unreadable installed LaunchAgent should fail safely: {ok} {detail}")
    if len(detail) > live_check.MAX_DIAGNOSTIC_CHARS:
        raise SystemExit(f"unreadable installed LaunchAgent detail was unbounded: {len(detail)} {detail}")
    for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root), "SHOULD NOT APPEAR", "installed secret"):
        if forbidden in detail:
            raise SystemExit(f"unreadable installed LaunchAgent detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_installed_primary_drift_safely() -> None:
    drift_cases = {
        "Label": "com.private.wrong.SHOULD_NOT_APPEAR",
        "Program": "/opt/homebrew/bin/python3",
        "MachServices": {"com.unreviewed.SHOULD_NOT_APPEAR": True},
        "ProgramArguments": ["/private/wrong/python", "-m", "private.wrong.SHOULD_NOT_APPEAR"],
        "WorkingDirectory": "/\x55sers/example/private/SHOULD_NOT_APPEAR",
        "EnvironmentVariables": {
            "PYTHONPATH": "/\x55sers/example/private/SHOULD_NOT_APPEAR",
            "TOKEN": "SHOULD NOT APPEAR",
        },
        "RunAtLoad": False,
        "KeepAlive": False,
    }
    for field, drifted_value in drift_cases.items():
        with TemporaryDirectory(prefix="jarvis-daemon-startup-installed-drift-") as temp:
            root = Path(temp)
            launch_agents = _write_startup_contract(root)
            _write_daemon_restart_doc(root)
            with (root / "com.jarvis-v3.telegram.plist").open("rb") as handle:
                installed = plistlib.load(handle)
            installed[field] = drifted_value
            installed_path = launch_agents / "com.jarvis-v3.telegram.plist"
            with installed_path.open("wb") as handle:
                plistlib.dump(installed, handle)
            installed_path.chmod(0o600)
            ok, detail = live_check._check_daemon_startup_readiness(root=root, launch_agents_dir=launch_agents)
        expected = (
            "installed LaunchAgent environment is not allowlisted"
            if field == "EnvironmentVariables"
            else "installed LaunchAgent drifted field(s): TopLevelKeys"
            if field == "MachServices"
            else f"installed LaunchAgent drifted field(s): {field}"
        )
        if ok is not False or expected not in detail:
            raise SystemExit(f"installed LaunchAgent {field} drift should fail safely: {ok} {detail}")
        if len(detail) > live_check.MAX_DIAGNOSTIC_CHARS:
            raise SystemExit(f"installed LaunchAgent drift detail was unbounded: {len(detail)} {detail}")
        for forbidden in (
            "/\x55sers/",
            "/private/",
            "/tmp/",
            str(root),
            "SHOULD NOT APPEAR",
            "private.wrong",
            "com.private",
        ):
            if forbidden in detail:
                raise SystemExit(f"installed LaunchAgent drift detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_non_mapping_installed_primary_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-installed-list-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        with (launch_agents / "com.jarvis-v3.telegram.plist").open("wb") as handle:
            plistlib.dump(["/\x55sers/example/private/SHOULD NOT APPEAR"], handle)
        (launch_agents / "com.jarvis-v3.telegram.plist").chmod(0o600)
        ok, detail = live_check._check_daemon_startup_readiness(root=root, launch_agents_dir=launch_agents)
    if ok is not False or "installed LaunchAgent unreadable/malformed" not in detail:
        raise SystemExit(f"non-mapping installed LaunchAgent should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/", "/private/", str(root), "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"non-mapping installed LaunchAgent detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_stale_template_directory_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-template-directory-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        template_path = root / "com.jarvis-v3.telegram.plist"
        with template_path.open("rb") as handle:
            template = plistlib.load(handle)
        template["WorkingDirectory"] = "/\x55sers/example/old/private/SHOULD NOT APPEAR"
        with template_path.open("wb") as handle:
            plistlib.dump(template, handle)
        ok, detail = live_check._check_daemon_startup_readiness(root=root, launch_agents_dir=launch_agents)
    if ok is not False or "drifted 1 contract(s): Telegram LaunchAgent" not in detail:
        raise SystemExit(f"stale template directory should fail safely: {ok} {detail}")
    for forbidden in ("/\x55sers/", "/private/", str(root), "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"stale template directory detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_flags_missing_restart_documentation_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-doc-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root, complete=False)
        ok, detail = live_check._check_daemon_startup_readiness(root=root, launch_agents_dir=launch_agents)
    if ok is not False:
        raise SystemExit(f"missing daemon restart docs should fail safely: {ok} {detail}")
    for expected in (
        "daemon recovery boundary documentation missing or malformed",
        "repair the bounded recovery contract before reboot proof",
    ):
        if expected not in detail:
            raise SystemExit(f"daemon startup doc detail missed {expected!r}: {detail}")
    for forbidden in (
        "/\x55sers/operator",
        "/private/",
        "/tmp/",
        str(root),
        "SHOULD NOT APPEAR",
        "jarvis_v2_telegram.log",
        "launchctl kickstart",
    ):
        if forbidden in detail:
            raise SystemExit(f"daemon startup doc detail leaked seeded/detail value {forbidden!r}: {detail}")


def test_startup_recovery_documentation_classifier_normalizes_only_presentation() -> None:
    if live_check._startup_recovery_documentation_mode(live_check.REPO_ROOT) != "sanitized":
        raise SystemExit("current public recovery wording was not recognized exactly")

    with TemporaryDirectory(prefix="jarvis-recovery-doc-classifier-") as temp:
        root = Path(temp)
        valid = "\n\n".join(live_check.STARTUP_PUBLIC_RECOVERY_BOUNDARY)
        (root / "README.md").write_text(valid.upper().replace(" ", "  \n"), encoding="utf-8")
        if live_check._startup_recovery_documentation_mode(root) != "sanitized":
            raise SystemExit("case or line wrapping changed exact public recovery semantics")

        for token in live_check.STARTUP_PUBLIC_RECOVERY_BOUNDARY:
            weakened = valid.replace(token, "recovery boundary omitted", 1)
            (root / "README.md").write_text(weakened, encoding="utf-8")
            if live_check._startup_recovery_documentation_mode(root) is not None:
                raise SystemExit("public recovery classifier accepted missing semantic evidence")


def test_daemon_startup_live_check_defers_sanitized_candidate_without_private_recovery() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-sanitized-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        for contract in live_check.STARTUP_PLIST_CONTRACTS.values():
            (root / contract["path"]).unlink()
        (root / "README.md").write_text(
            "\n".join(live_check.STARTUP_PUBLIC_RECOVERY_BOUNDARY),
            encoding="utf-8",
        )
        with mock.patch(
            "jarvis_v2.scripts.live_check._valid_public_candidate_root",
            return_value=True,
        ):
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
            )
    if ok is not None:
        raise SystemExit(f"sanitized candidate should defer startup proof: {ok} {detail}")
    for expected in (
        "startup/reboot deferred",
        "validated manual launcher(s)",
        "intentionally omits service templates and private recovery commands",
        "activation not proven",
        "no process control run",
    ):
        if expected not in detail:
            raise SystemExit(f"sanitized candidate detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root), "launchctl kickstart"):
        if forbidden in detail:
            raise SystemExit(f"sanitized candidate detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_distinguishes_inert_installed_template() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-inert-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        _install_primary_startup_contracts(
            root,
            launch_agents,
            activate_telegram=False,
        )
        ok, detail = live_check._check_daemon_startup_readiness(
            root=root,
            launch_agents_dir=launch_agents,
        )
    if ok is not None:
        raise SystemExit(f"inert installed template should remain unproven: {ok} {detail}")
    for expected in (
        "3 inert LaunchAgent template(s)",
        "2/2 primary LaunchAgent installed",
        "0/2 explicitly activated",
        "service activation/reboot deferred and not proven",
    ):
        if expected not in detail:
            raise SystemExit(f"inert installed detail missed {expected!r}: {detail}")
    for forbidden in ("/\x55sers/", "/private/", "/tmp/", str(root)):
        if forbidden in detail:
            raise SystemExit(f"inert installed detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_activation_value_secret_and_scheduler_key() -> None:
    cases = {
        "invalid": (
            {live_check.V3_DAEMON_ENABLE_ENV: "true"},
            "installed LaunchAgent activation value is invalid",
        ),
        "secret": (
            {
                live_check.V3_DAEMON_ENABLE_ENV: "1",
                "PRIVATE_TOKEN_SHOULD_NOT_APPEAR": "/\x55sers/operator/private/SHOULD NOT APPEAR",
            },
            "installed LaunchAgent environment is not allowlisted",
        ),
        "scheduler": (
            {
                live_check.V3_DAEMON_ENABLE_ENV: "1",
                live_check.V3_SCHEDULER_ENABLE_ENV: "1",
            },
            "scheduler-disable value is invalid; exact value 0 is required",
        ),
        "scheduler-whitespace": (
            {live_check.V3_SCHEDULER_ENABLE_ENV: " 0 "},
            "scheduler-disable value is invalid; exact value 0 is required",
        ),
        "scheduler-missing": (
            {},
            "missing the explicit scheduler-disable value; exact value 0 is required",
        ),
    }
    for label, (activation, expected) in cases.items():
        with TemporaryDirectory(prefix=f"jarvis-daemon-startup-activation-{label}-") as temp:
            root = Path(temp)
            launch_agents = _write_startup_contract(root)
            _write_daemon_restart_doc(root)
            selector = _install_primary_startup_contracts(root, launch_agents)
            path = launch_agents / "com.jarvis-v3.telegram.plist"
            with path.open("rb") as handle:
                installed = plistlib.load(handle)
            installed["EnvironmentVariables"].update(activation)
            if label == "scheduler-missing":
                installed["EnvironmentVariables"].pop(live_check.V3_SCHEDULER_ENABLE_ENV)
            with path.open("wb") as handle:
                plistlib.dump(installed, handle)
            with mock.patch.dict(
                os.environ,
                {live_check.STARTUP_SELECTED_ENV: str(selector)},
            ):
                ok, detail = live_check._check_daemon_startup_readiness(
                    root=root,
                    launch_agents_dir=launch_agents,
                )
        if ok is not False or expected not in detail:
            raise SystemExit(f"{label} activation should fail closed: {ok} {detail}")
        if len(detail) > live_check.MAX_DIAGNOSTIC_CHARS:
            raise SystemExit(f"{label} activation detail was unbounded: {len(detail)} {detail}")
        for forbidden in (
            "/\x55sers/",
            "/private/",
            "/tmp/",
            str(root),
            "PRIVATE_TOKEN",
            "SHOULD NOT APPEAR",
        ):
            if forbidden in detail:
                raise SystemExit(f"{label} activation detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_activated_checked_in_template() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-template-activation-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        template_path = root / "com.jarvis-v3.telegram.plist"
        with template_path.open("rb") as handle:
            template = plistlib.load(handle)
        template["EnvironmentVariables"].update(
            {
                live_check.V3_DAEMON_ENABLE_ENV: "1",
                "PRIVATE_TOKEN_SHOULD_NOT_APPEAR": "/private/SHOULD NOT APPEAR",
            }
        )
        with template_path.open("wb") as handle:
            plistlib.dump(template, handle)
        ok, detail = live_check._check_daemon_startup_readiness(
            root=root,
            launch_agents_dir=launch_agents,
        )
    if ok is not False or "drifted 1 contract(s): Telegram LaunchAgent" not in detail:
        raise SystemExit(f"activated checked-in template should fail closed: {ok} {detail}")
    for forbidden in ("/private/", str(root), "PRIVATE_TOKEN", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"activated template detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_malformed_installed_path_safely() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-installed-path-") as temp:
        root = Path(temp)
        _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        malformed = root / "not-a-directory-PRIVATE-SHOULD-NOT-APPEAR"
        malformed.write_text(
            "/\x55sers/operator/private/SHOULD NOT APPEAR",
            encoding="utf-8",
        )
        ok, detail = live_check._check_daemon_startup_readiness(
            root=root,
            launch_agents_dir=malformed,
        )
    if ok is not False or "installed LaunchAgent directory is malformed" not in detail:
        raise SystemExit(f"malformed installed path should fail closed: {ok} {detail}")
    for forbidden in ("/\x55sers/", "/private/", str(root), "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"malformed installed path detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_rejects_optional_installed_environment_drift() -> None:
    with TemporaryDirectory(prefix="jarvis-daemon-startup-optional-environment-") as temp:
        root = Path(temp)
        launch_agents = _write_startup_contract(root)
        _write_daemon_restart_doc(root)
        selector = root / "runtime.env"
        selector.write_text("", encoding="utf-8")
        selector.chmod(0o600)
        selector = selector.resolve()
        with (root / "com.jarvis-v3.imessage.plist").open("rb") as handle:
            installed = plistlib.load(handle)
        installed["EnvironmentVariables"].update(
            {
                live_check.V3_DAEMON_ENABLE_ENV: "1",
                live_check.STARTUP_SELECTED_ENV: str(selector),
                live_check.V3_SCHEDULER_ENABLE_ENV: "0",
            }
        )
        installed_path = launch_agents / "com.jarvis-v3.imessage.plist"
        with installed_path.open("wb") as handle:
            plistlib.dump(installed, handle)
        installed_path.chmod(0o600)
        with mock.patch.dict(
            os.environ,
            {live_check.STARTUP_SELECTED_ENV: str(selector)},
        ):
            inert_ok, inert_detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
            )
        if inert_ok is not None or "0/2 primary LaunchAgent installed" not in inert_detail:
            raise SystemExit(
                f"valid optional installed contract should not count as primary: {inert_ok} {inert_detail}"
            )
        installed["EnvironmentVariables"]["PRIVATE_TOKEN_SHOULD_NOT_APPEAR"] = (
            "/private/SHOULD NOT APPEAR"
        )
        with installed_path.open("wb") as handle:
            plistlib.dump(installed, handle)
        with mock.patch.dict(
            os.environ,
            {live_check.STARTUP_SELECTED_ENV: str(selector)},
        ):
            ok, detail = live_check._check_daemon_startup_readiness(
                root=root,
                launch_agents_dir=launch_agents,
            )
    if ok is not False or "installed LaunchAgent environment is not allowlisted" not in detail:
        raise SystemExit(f"optional installed environment drift should fail closed: {ok} {detail}")
    for forbidden in ("/private/", str(root), "PRIVATE_TOKEN", "SHOULD NOT APPEAR"):
        if forbidden in detail:
            raise SystemExit(f"optional installed environment detail leaked {forbidden!r}: {detail}")


def test_daemon_startup_live_check_binds_selected_environment_privately() -> None:
    cases = (
        ("missing", False, "missing the selected V3 environment binding"),
        ("mismatch", False, "does not match this diagnostic session"),
        ("unsafe", False, "selected V3 environment has unsafe custody"),
        ("unavailable", None, "environment binding is not proven"),
    )
    for label, expected_ok, expected_detail in cases:
        with TemporaryDirectory(prefix=f"jarvis-daemon-startup-selector-{label}-") as temp:
            root = Path(temp)
            launch_agents = _write_startup_contract(root)
            _write_daemon_restart_doc(root)
            selector = _install_primary_startup_contracts(root, launch_agents)
            current_selector = str(selector)
            if label == "missing":
                for filename in ("com.jarvis-v3.telegram.plist", "com.jarvis-v3.dashboard.plist"):
                    path = launch_agents / filename
                    with path.open("rb") as handle:
                        installed = plistlib.load(handle)
                    installed["EnvironmentVariables"].pop(live_check.STARTUP_SELECTED_ENV)
                    with path.open("wb") as handle:
                        plistlib.dump(installed, handle)
            elif label == "mismatch":
                other = root / "other-private-SHOULD-NOT-APPEAR.env"
                other.write_text("", encoding="utf-8")
                other.chmod(0o600)
                current_selector = str(other)
            elif label == "unsafe":
                selector.chmod(0o644)
            elif label == "unavailable":
                current_selector = ""
            with mock.patch.dict(
                os.environ,
                {live_check.STARTUP_SELECTED_ENV: current_selector},
            ):
                ok, detail = live_check._check_daemon_startup_readiness(
                    root=root,
                    launch_agents_dir=launch_agents,
                )
        if ok is not expected_ok or expected_detail not in detail:
            raise SystemExit(f"{label} selected-environment binding drifted: {ok} {detail}")
        if len(detail) > live_check.MAX_DIAGNOSTIC_CHARS:
            raise SystemExit(f"{label} selected-environment detail was unbounded: {len(detail)} {detail}")
        for forbidden in (
            "/\x55sers/",
            "/private/",
            "/tmp/",
            str(root),
            "other-private",
            "SHOULD-NOT-APPEAR",
        ):
            if forbidden in detail:
                raise SystemExit(f"{label} selected-environment detail leaked {forbidden!r}: {detail}")


def test_live_check_main_survives_config_failure_and_redacts() -> None:
    seeded_error = "/\x55sers/example/private/live-config.env SECRET SHOULD NOT APPEAR"
    stdout = io.StringIO()
    patches = (
        mock.patch.dict(os.environ, {"JARVIS_LIVE_CHECK": "1", "JARVIS_LIVE_CHECK_SEND": "0"}),
        mock.patch.object(sys, "argv", ["live_check.py"]),
        mock.patch("jarvis_v2.scripts.live_check.load_config", side_effect=RuntimeError(seeded_error)),
        mock.patch("jarvis_v2.scripts.live_check._check_voice_transcription", return_value=(True, "voice ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_voice_warmup", return_value=(None, "warmup skipped")),
        mock.patch("jarvis_v2.scripts.live_check._check_dashboard_voice_wiring", return_value=(True, "dashboard ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_local_talk_readiness", return_value=(True, "local talk ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_telegram_voice_readiness", return_value=(True, "telegram voice ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_image_ocr", return_value=(True, "ocr ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_research_readiness", return_value=(True, "research ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_mixed_conversation_health", return_value=(True, "mixed conversation ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_personal_route_readiness", return_value=(True, "personal routes ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_guardrail_control_plane", return_value=(True, "guardrails ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_proof_ledger_control_plane", return_value=(True, "proof ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_completion_phone_gate", return_value=(True, "completion ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_learning_recovery_control_plane", return_value=(True, "learning ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_phone_approval_flow", return_value=(True, "phone approvals ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_phone_control_center", return_value=(True, "phone control ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_daemon_startup_readiness", return_value=(True, "daemon startup ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_operator_workflow_eval_proof", return_value=(True, "operator evals ok")),
        mock.patch("jarvis_v2.scripts.live_check._check_telegram", return_value=(True, "telegram env ok")),
    )
    with ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        try:
            with redirect_stdout(stdout):
                live_check.main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise SystemExit(f"live_check main should always exit 0 on config failure, got {exc.code}") from exc

    output = stdout.getvalue()
    status_markers = (" ✓ ok", " ✗ FAIL", " – skip")
    printed_labels = []
    for line in output.splitlines():
        for marker in status_markers:
            if marker in line:
                printed_labels.append(line.split(marker, 1)[0].strip())
                break
    if tuple(printed_labels) != live_check.LIVE_CHECK_TABLE_LABELS:
        raise SystemExit(
            "live_check config-failure output row contract drifted: "
            f"expected {live_check.LIVE_CHECK_TABLE_LABELS!r}, got {tuple(printed_labels)!r}"
        )
    if len(printed_labels) > live_check.LIVE_CHECK_ONE_SCREEN_MAX_ROWS:
        raise SystemExit(
            "live_check config-failure output exceeded one-screen row budget: "
            f"{len(printed_labels)}/{live_check.LIVE_CHECK_ONE_SCREEN_MAX_ROWS}"
        )
    for expected in ("config unavailable", "RuntimeError", "<local-path>"):
        if expected not in output:
            raise SystemExit(f"live_check config-failure output missed {expected!r}: {output}")
    for forbidden in ("/\x55sers/operator", "live-config.env", "SECRET SHOULD NOT APPEAR"):
        if forbidden in output:
            raise SystemExit(f"live_check config-failure output leaked seeded detail {forbidden!r}: {output}")


def main() -> None:
    test_channel_health_live_check_is_metadata_only()
    test_channel_health_live_check_rejects_malformed_channel_metadata()
    test_channel_health_live_check_does_not_pass_without_recent_success()
    test_channel_health_live_check_flags_newer_failure_than_success()
    test_channel_health_live_check_requires_complete_consistent_success_evidence()
    test_channel_health_live_check_redacts_exception_details()
    test_chat_path_live_check_reports_model_ready_without_content_leakage()
    test_chat_path_live_check_flags_fallback_and_empty_windows_safely()
    test_mixed_conversation_live_check_reports_ready_without_content_leakage()
    test_mixed_conversation_live_check_reports_incomplete_without_content_leakage()
    test_mixed_conversation_live_check_names_proof_path_for_missing_latency_samples()
    test_mixed_conversation_live_check_flags_latency_regression_safely()
    test_personal_integration_live_check_reports_ready_without_secret_leakage()
    test_personal_integration_live_check_reports_missing_setup_without_values()
    test_personal_integration_live_check_flags_missing_or_wrong_risk_gates()
    test_personal_proof_history_reports_ready_without_content_leakage()
    test_personal_proof_history_reports_missing_without_content_leakage()
    test_personal_proof_history_requires_set_reminder_not_macos_create_without_content_leakage()
    test_personal_proof_history_reports_empty_window_safely()
    test_research_live_check_reports_ready_without_network_or_content_leakage()
    test_research_live_check_flags_missing_or_wrong_risk_gates()
    test_research_live_check_flags_route_drift_without_query_leakage()
    test_personal_route_live_check_reports_ready_without_handler_or_content_leakage()
    test_personal_route_live_check_flags_drift_without_content_leakage()
    test_guardrail_control_plane_live_check_is_read_only_and_bounded()
    test_guardrail_control_plane_live_check_exception_is_redacted()
    test_proof_ledger_control_plane_live_check_is_read_only_and_bounded()
    test_proof_ledger_control_plane_live_check_exception_is_redacted()
    test_completion_phone_gate_live_check_is_read_only_and_bounded()
    test_completion_phone_gate_live_check_exception_is_redacted()
    test_learning_recovery_control_plane_live_check_is_read_only_and_bounded()
    test_learning_recovery_control_plane_live_check_exception_is_redacted()
    test_phone_approval_flow_live_check_is_read_only_and_bounded()
    test_phone_approval_flow_live_check_exception_is_redacted()
    test_phone_control_center_live_check_is_read_only_and_bounded()
    test_phone_control_center_live_check_exception_is_redacted()
    test_voice_warmup_live_check_reports_opt_in_boundary()
    test_voice_live_check_suppresses_unknown_transcriber_source()
    test_dashboard_voice_live_check_reuses_shared_hud_pipeline()
    test_local_talk_live_check_reports_ready_without_mic_or_runtime()
    test_local_talk_live_check_requires_ffmpeg_before_live_mic_proof()
    test_local_talk_live_check_flags_source_drift_safely()
    test_telegram_voice_live_check_reports_ready_without_live_voice_or_runtime()
    test_telegram_voice_live_check_flags_source_drift_safely()
    test_error_guidance_live_check_reports_ready_without_leakage()
    test_error_guidance_live_check_matches_actual_repository()
    test_error_guidance_live_check_rejects_contract_count_shrink()
    test_error_guidance_inventory_rejects_same_count_substitution()
    test_error_guidance_live_check_requires_visible_recovery_wording()
    test_error_guidance_live_check_flags_recovery_metadata_drift()
    test_error_guidance_live_check_checks_each_combined_result()
    test_error_guidance_live_check_flags_calendar_drift_safely()
    test_error_guidance_live_check_rejects_broadened_calendar_auth_safely()
    test_error_guidance_live_check_flags_calendar_generic_drift_safely()
    test_error_guidance_live_check_flags_voice_setup_drift_safely()
    test_error_guidance_live_check_flags_ollama_fallback_drift_safely()
    test_error_guidance_live_check_flags_planner_fallback_drift_safely()
    test_error_guidance_live_check_flags_user_visible_recovery_drift_safely()
    test_error_guidance_live_check_flags_compose_guidance_drift_safely()
    test_error_guidance_live_check_flags_daily_brief_guidance_drift_safely()
    test_error_guidance_live_check_flags_writer_guidance_drift_safely()
    test_error_guidance_live_check_flags_ocr_guidance_drift_safely()
    test_aggregate_smoke_proof_live_check_reports_logged_green_without_running_tests()
    test_aggregate_smoke_proof_live_check_requires_timestamped_evidence()
    test_aggregate_smoke_proof_live_check_rejects_stale_evidence()
    test_aggregate_smoke_proof_live_check_flags_missing_and_low_counts_safely()
    test_aggregate_smoke_proof_picks_the_latest_timestamp_despite_handoff_order()
    test_operator_workflow_eval_proof_reports_logged_green_without_running_evals()
    test_operator_workflow_eval_proof_flags_missing_or_drift_safely()
    test_acceptance_harness_coverage_reports_section_map_without_live_actions()
    test_acceptance_harness_coverage_flags_plan_or_row_drift_safely()
    test_acceptance_item_map_detects_requirement_drift_without_leakage()
    test_acceptance_item_map_requires_one_evidence_mapping_per_plan_item()
    test_acceptance_harness_coverage_matches_actual_finish_plan()
    test_acceptance_malformed_heading_prefix_fails_closed()
    test_acceptance_open_items_reports_remaining_gaps_without_item_leakage()
    test_acceptance_open_items_distinguishes_offline_engineering_debt()
    test_annotated_acceptance_heading_keeps_channel_counts_and_priority()
    test_acceptance_open_items_reports_clear_checklist_without_live_actions()
    test_acceptance_open_items_flags_missing_or_empty_checklist_safely()
    test_acceptance_next_action_reports_priority_lane_without_item_leakage()
    test_acceptance_next_action_reports_clear_checklist_without_live_actions()
    test_acceptance_next_action_flags_missing_or_empty_checklist_safely()
    test_call_readiness_binds_native_target_without_call_or_app_control()
    test_live_check_exception_details_are_redacted()
    test_live_check_contacts_distinguishes_store_state()
    test_live_check_contact_matrix_is_bounded_and_hides_supplied_names()
    test_live_check_telegram_failure_details_are_redacted()
    test_live_check_success_details_suppress_private_values()
    test_live_check_telegram_success_suppresses_owner_id()
    test_live_check_telegram_read_probe_is_read_only_and_private_safe()
    test_live_check_telegram_loads_project_env_before_preflight()
    test_live_check_rows_redact_detail_before_printing()
    test_live_check_unprintable_diagnostics_fail_closed()
    test_live_check_malformed_mappings_fail_closed()
    test_live_check_daily_brief_success_suppresses_brief_content()
    test_live_check_daily_brief_unavailable_requires_recovery_hint_proof_without_leakage()
    test_live_check_daily_brief_rejects_malformed_output_and_counts()
    test_live_check_morning_brief_schedule_suppresses_bad_next_run()
    test_live_check_morning_brief_schedule_survives_malformed_rows()
    test_live_check_scheduled_job_freshness_reports_running_jobs_safely()
    test_live_check_scheduled_job_freshness_reports_first_run_without_overclaiming()
    test_live_check_scheduled_job_freshness_flags_stale_and_invalid_metadata_safely()
    test_live_check_network_recovery_reports_only_content_free_evidence()
    test_live_check_network_recovery_rejects_malformed_or_unreadable_evidence_safely()
    test_live_check_scheduled_job_streak_reports_ready_without_job_leakage()
    test_live_check_scheduled_job_streak_reports_incomplete_without_overclaiming()
    test_live_check_scheduled_job_streak_flags_failed_or_invalid_history_safely()
    test_daemon_source_freshness_reports_current_and_stale_code_safely()
    test_daemon_source_freshness_tracks_runtime_closure_without_docs_noise()
    test_daemon_source_freshness_flags_missing_duplicate_and_unavailable_state()
    test_daemon_source_git_reader_is_bounded_read_only_and_fail_closed()
    test_daemon_source_freshness_requires_stable_clean_git_head()
    test_daemon_source_freshness_process_scan_is_read_only_and_content_free()
    test_daemon_startup_live_check_reports_ready_without_path_leakage()
    test_daemon_startup_live_check_defers_without_launchd_pid_provenance()
    test_launchd_primary_job_inspection_is_read_only_and_bounded()
    test_daemon_startup_live_check_rejects_stale_loaded_launchd_contract()
    test_daemon_startup_live_check_allows_duplicate_free_launchd_environment_extras()
    test_daemon_startup_live_check_allows_reviewed_exact_program_contract()
    test_daemon_startup_live_check_rejects_unsafe_installed_file_custody()
    test_daemon_startup_live_check_rejects_linked_selected_environment()
    test_daemon_startup_live_check_rejects_symlinked_selected_environment_parent()
    test_daemon_startup_live_check_rejects_unsafe_launchagents_directory_custody()
    test_daemon_startup_live_check_uses_account_database_home()
    test_daemon_log_custody_is_owner_only_and_no_follow()
    test_private_plist_read_rejects_content_race_and_oversize()
    test_launchd_primary_provenance_requires_closing_observation()
    test_launchd_primary_provenance_rejects_duplicate_processes()
    test_private_file_custody_rejects_foreign_owner_snapshot()
    test_daemon_startup_live_check_skips_when_primary_launchagent_not_installed()
    test_daemon_startup_live_check_flags_malformed_contracts_safely()
    test_daemon_startup_live_check_rejects_unreadable_installed_primary_safely()
    test_daemon_startup_live_check_rejects_installed_primary_drift_safely()
    test_daemon_startup_live_check_rejects_non_mapping_installed_primary_safely()
    test_daemon_startup_live_check_rejects_stale_template_directory_safely()
    test_daemon_startup_live_check_flags_missing_restart_documentation_safely()
    test_startup_recovery_documentation_classifier_normalizes_only_presentation()
    test_daemon_startup_live_check_defers_sanitized_candidate_without_private_recovery()
    test_daemon_startup_live_check_distinguishes_inert_installed_template()
    test_daemon_startup_live_check_rejects_activation_value_secret_and_scheduler_key()
    test_daemon_startup_live_check_rejects_activated_checked_in_template()
    test_daemon_startup_live_check_rejects_malformed_installed_path_safely()
    test_daemon_startup_live_check_rejects_optional_installed_environment_drift()
    test_daemon_startup_live_check_binds_selected_environment_privately()
    test_live_check_main_survives_config_failure_and_redacts()
    print("Live check diagnostics smoke passed")


if __name__ == "__main__":
    main()
