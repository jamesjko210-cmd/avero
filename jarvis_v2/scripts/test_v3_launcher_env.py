from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from jarvis_v2.scripts.v3_python_runtime import V3_INHERITED_CUSTODY_KEYS


def scrubbed_v3_launcher_env(
    environment: Mapping[str, str] | None = None,
    *,
    selected_env: Path | None = None,
) -> dict[str, str]:
    """Return a child env with inherited V3 custody deliberately removed."""

    child = dict(os.environ if environment is None else environment)
    for key in V3_INHERITED_CUSTODY_KEYS:
        child.pop(key, None)
    child.pop("JARVIS_V3_ENV", None)
    if selected_env is not None:
        child["JARVIS_V3_ENV"] = str(selected_env)
    return child


def empty_v3_launcher_env(
    selected_env: Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Create a private empty selector after deliberately scrubbing inherited custody."""

    selected_env.write_text("", encoding="utf-8")
    selected_env.chmod(0o600)
    return scrubbed_v3_launcher_env(environment, selected_env=selected_env)
