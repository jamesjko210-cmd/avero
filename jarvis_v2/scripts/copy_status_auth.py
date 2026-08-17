"""Copy the local dashboard password without displaying or logging it."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jarvis_v2.env import read_env_values
from jarvis_v2.ui.status_config import STATUS_AUTH_ENV, status_auth_token_is_valid


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PBCOPY = "/usr/bin/pbcopy"
_COPY_TIMEOUT_SECONDS = 5


class StatusAuthCopyError(RuntimeError):
    """Bounded refusal that contains no password or local-path detail."""


@dataclass(frozen=True)
class StatusAuthCopyResult:
    copied: bool


def _selected_env_path(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser()
    configured = os.getenv("JARVIS_V3_ENV", "").strip()
    return Path(configured).expanduser() if configured else _PROJECT_ROOT / ".env"


def copy_status_auth(env_path: Path | None = None) -> StatusAuthCopyResult:
    """Validate one selected env file, then copy its dashboard password."""

    selected = _selected_env_path(env_path)
    try:
        values = read_env_values(
            selected,
            reject_duplicate_keys=frozenset({STATUS_AUTH_ENV}),
            require_owner_only=True,
        )
    except FileNotFoundError:
        raise StatusAuthCopyError("the selected environment file does not exist") from None
    except PermissionError:
        raise StatusAuthCopyError("the selected environment file has unsafe custody") from None
    except UnicodeError:
        raise StatusAuthCopyError("the selected environment file is not valid UTF-8") from None
    except ValueError:
        raise StatusAuthCopyError("multiple dashboard authentication entries are not allowed") from None
    except OSError:
        raise StatusAuthCopyError("the selected environment path is not a secure regular file") from None

    token = values.get(STATUS_AUTH_ENV, "")
    if not status_auth_token_is_valid(token):
        raise StatusAuthCopyError("the dashboard authentication entry is missing or invalid")

    try:
        completed = subprocess.run(
            [_PBCOPY],
            input=token.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_COPY_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise StatusAuthCopyError("the local clipboard could not be updated") from None
    if completed.returncode != 0:
        raise StatusAuthCopyError("the local clipboard could not be updated")
    return StatusAuthCopyResult(copied=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the owner-only Jarvis environment file and copy its dashboard "
            "password to the local clipboard without displaying it."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Environment file to read (defaults to JARVIS_V3_ENV or the project .env).",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        copy_status_auth(args.env_file)
    except StatusAuthCopyError as exc:
        print(f"Dashboard password copy refused: {exc}.", file=os.sys.stderr)
        raise SystemExit(2) from None
    print(
        "Dashboard password copied to the local clipboard without displaying it. "
        "Paste it only into the dashboard password field, then clear the clipboard."
    )


if __name__ == "__main__":
    main()
