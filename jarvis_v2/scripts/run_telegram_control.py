"""Run the owner-locked Jarvis V2 Telegram control bridge in the foreground.

This is the remote phone-control daemon: it long-polls Telegram, runs the
owner's commands through the V2 runtime on this Mac, and replies to the phone.
Every side effect still goes through the V2 safety harness.

Requires (in .env or environment):
- TELEGRAM_BOT_TOKEN  (from @BotFather)
- JARVIS_OWNER_TELEGRAM  (your numeric chat id; run telegram_whoami to find it)
"""

from __future__ import annotations

from jarvis_v2.config import load_config
from jarvis_v2.agent.model_provider import generate_model_text
from jarvis_v2.automations.telegram_control import (
    TelegramCommandBridge,
    _bot_token,
    _load_offset_snapshot,
    _owner_chat_id,
)
from jarvis_v2.automations.telegram_offset_state import TelegramOffsetStateError
from jarvis_v2.scripts.daemon_gate import (
    require_v3_daemon_enable,
    v3_daemons_enabled,
    v3_scheduler_enabled,
)


def _warm_model(config) -> None:
    """Probe the Ollama loopback daemon before the first command."""
    if config.model_provider != "ollama":
        print(
            "Ollama loopback daemon warmup skipped: configured model provider is not Ollama; "
            "no request was sent.",
            flush=True,
        )
        return
    try:
        generate_model_text(
            provider="ollama",
            model=config.chat_model,
            messages=[{"role": "user", "content": "ready?"}],
            timeout_seconds=60,
            max_output_tokens=1,
            temperature=0,
            keep_alive="30m",
        )
        print(
            f"Ollama loopback daemon responded to the warmup request for {config.chat_model}; "
            "this does not verify where model execution occurred.",
            flush=True,
        )
    except Exception as exc:
        print(
            f"Ollama loopback daemon warmup did not complete. Diagnostic: {type(exc).__name__}. "
            "Jarvis will retry the daemon on the first command. If commands also fail, run "
            "`model status`, then repair the local Ollama service before retrying.",
            flush=True,
        )


def _start_scheduler_ticker(config) -> bool:
    """Run due scheduled jobs (e.g. the 09:00 Morning Brief) once a minute.

    The Scheduler and its jobs already exist; this ticker is what actually
    fires them — without it the Morning Brief would never send. The thread
    uses its OWN store/vault and never lets a job error kill the loop.
    """
    if not (v3_daemons_enabled() and v3_scheduler_enabled()):
        print(
            "Jarvis V3 scheduler ticker remains disabled; Telegram control will run without "
            "scheduled-job execution. Set JARVIS_V3_ENABLE_SCHEDULER=1 only during a separately "
            "supervised scheduler cutover.",
            flush=True,
        )
        return False

    import threading
    import time

    def tick_loop() -> None:
        from jarvis_v2.automations.scheduler import Scheduler
        from jarvis_v2.memory.obsidian import ObsidianVault
        from jarvis_v2.memory.store import MemoryStore

        scheduler = None
        initialization_failed = False
        tick_failed = False
        while True:
            if scheduler is None:
                try:
                    store = MemoryStore(config.db_path)
                    vault = ObsidianVault(config.obsidian_vault, config.obsidian_root)
                    store.init()
                    vault.init()
                    scheduler = Scheduler(store, vault, config)
                except Exception:
                    if not initialization_failed:
                        print(
                            "Scheduler ticker initialization unavailable; retrying in 60 seconds. "
                            "If this persists, run `setup check` and repair local storage access.",
                            flush=True,
                        )
                    initialization_failed = True
                    time.sleep(60)
                    continue
                if initialization_failed:
                    print("Scheduler ticker initialization recovered.", flush=True)
                print("Scheduler ticker running (checks due jobs every 60s).", flush=True)

            try:
                output = scheduler.run_due_jobs()
                if tick_failed:
                    print("Scheduler ticker tick recovered.", flush=True)
                tick_failed = False
                if output and output != "No jobs due.":
                    stamp = time.strftime("%m-%d %H:%M")
                    print(f"[scheduler {stamp}] {output[:300]}", flush=True)
            except Exception:
                if not tick_failed:
                    print(
                        "Scheduler ticker tick unavailable; retrying in 60 seconds. If this "
                        "persists, run `list scheduled jobs` and `setup check`, then repair the "
                        "reported scheduler or storage issue.",
                        flush=True,
                    )
                tick_failed = True
            time.sleep(60)

    threading.Thread(target=tick_loop, name="JarvisSchedulerTicker", daemon=True).start()
    return True


def main() -> None:
    require_v3_daemon_enable("Telegram")
    config = load_config()
    token = _bot_token()
    if not token:
        print(
            "Telegram control disabled: TELEGRAM_BOT_TOKEN is not set in .env. "
            "Set it from @BotFather, then restart the Jarvis Telegram service.",
            flush=True,
        )
        raise SystemExit(1)
    owner = _owner_chat_id()
    if not owner:
        print(
            "Telegram control disabled: JARVIS_OWNER_TELEGRAM is not set in .env. "
            "Run telegram_whoami, set the returned owner id, then restart the Jarvis Telegram service.",
            flush=True,
        )
        print("Run: python3 -m jarvis_v2.scripts.telegram_whoami", flush=True)
        raise SystemExit(1)

    try:
        _load_offset_snapshot(token, owner)
    except TelegramOffsetStateError as exc:
        print(
            "Telegram control disabled: offset custody validation failed before model or "
            f"scheduler startup ({exc.reason}). Run `setup check`, repair the reported "
            "state issue, then restart the Jarvis Telegram service.",
            flush=True,
        )
        raise SystemExit(1) from None

    _warm_model(config)
    _start_scheduler_ticker(config)
    print("Jarvis V3 Telegram control running (owner locked). Press Ctrl+C to stop.", flush=True)
    bridge = TelegramCommandBridge()
    try:
        bridge.run_forever()
    except KeyboardInterrupt:
        print("\nTelegram control stopped.", flush=True)


if __name__ == "__main__":
    main()
