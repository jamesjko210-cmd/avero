#!/usr/bin/env python3
"""Send one terminal message to Jarvis V3 using the dependency-complete interpreter."""

from pathlib import Path

from jarvis_v2.scripts.v3_python_runtime import reexec_with_v3_python


reexec_with_v3_python(Path(__file__))

from jarvis_v2.scripts.ask import main  # noqa: E402  (re-exec must happen first)


if __name__ == "__main__":
    main()
