#!/usr/bin/env python3
"""Run Jarvis V3 local push-to-talk with spoken replies by default."""

import sys
from pathlib import Path

from jarvis_v2.scripts.v3_python_runtime import reexec_with_v3_python


reexec_with_v3_python(Path(__file__))

from jarvis_v2.scripts.talk import main  # noqa: E402  (re-exec must happen first)


if __name__ == "__main__":
    if "--speak" not in sys.argv[1:] and "--no-speak" not in sys.argv[1:]:
        sys.argv.append("--speak")
    main()
