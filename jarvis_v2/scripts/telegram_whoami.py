"""Print the Telegram chat id(s) that have messaged the bot.

Run this AFTER creating the bot and sending it any message ("hi") from your
phone, so you can copy your numeric chat id into JARVIS_OWNER_TELEGRAM.
"""

from __future__ import annotations

from jarvis_v2.config import load_config
from jarvis_v2.automations.telegram_control import _bot_token, get_updates


def main() -> None:
    load_config()  # loads .env so TELEGRAM_BOT_TOKEN is available
    token = _bot_token()
    if not token:
        print(
            "TELEGRAM_BOT_TOKEN is not set in .env. Create or inspect the bot in @BotFather, "
            "set TELEGRAM_BOT_TOKEN in .env, then re-run telegram_whoami."
        )
        raise SystemExit(1)

    updates = get_updates(token, offset=None, timeout=0)
    if not updates:
        print("No messages seen yet. Send your bot any message from your phone, then re-run this.")
        return

    seen: dict[str, str] = {}
    for update in updates:
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        name = chat.get("first_name") or chat.get("username") or "?"
        if chat_id is not None:
            seen[str(chat_id)] = str(name)

    print("Chat ids that messaged your bot:")
    for chat_id, name in seen.items():
        print(f"  {chat_id}  ({name})")
    print("\nAdd the right one to .env as:  JARVIS_OWNER_TELEGRAM=<chat_id>")


if __name__ == "__main__":
    main()
