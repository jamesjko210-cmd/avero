"""Run the legacy owner-locked Jarvis V2 iMessage control bridge in the foreground.

Telegram is the primary remote command channel. This legacy/dev-only launcher is
kept for local testing of the iMessage bridge: it polls the Messages database,
runs owner commands prefixed with "Jarvis" through the V2 runtime, and replies
over iMessage. Every side effect still goes through the V2 safety harness.

Requires:
- JARVIS_OWNER_IMESSAGE set (via .env or environment)
- Full Disk Access for the Python interpreter running this script
- Automation permission for Messages (granted on first send)
"""

from __future__ import annotations

from jarvis_v2.config import load_config
from jarvis_v2.agent.model_provider import generate_model_text
from jarvis_v2.automations.imessage_control import IMessageCommandBridge, _owner_identifier
from jarvis_v2.scripts.daemon_gate import require_v3_daemon_enable


def _warm_model(config) -> None:
    """Probe the Ollama loopback daemon before the first phone command."""
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
            "Jarvis will retry the daemon on the first command.",
            flush=True,
        )


def main() -> None:
    require_v3_daemon_enable("iMessage")
    config = load_config()  # loads .env so JARVIS_OWNER_IMESSAGE is available
    owner = _owner_identifier()
    if not owner:
        print("iMessage control disabled: JARVIS_OWNER_IMESSAGE is not set in .env.", flush=True)
        raise SystemExit(1)

    _warm_model(config)
    print("Jarvis V3 iMessage control running (owner locked). Press Ctrl+C to stop.", flush=True)
    print("Text yourself: 'Jarvis what time is it' to test.")
    bridge = IMessageCommandBridge()
    try:
        bridge.run_forever()
    except KeyboardInterrupt:
        print("\niMessage control stopped.")


if __name__ == "__main__":
    main()
