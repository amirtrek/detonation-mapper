"""Root entrypoint for `python main.py` usage."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cli import main
from progress import progress, progress_kv


if __name__ == "__main__":
    progress("[main] Process starting.")
    progress_kv("[main] Python environment", executable=sys.executable, cwd=str(Path.cwd()))
    progress_kv("[main] Import paths", project_root=str(PROJECT_ROOT), src_path=str(SRC_PATH))
    progress_kv("[main] Raw argv", argv=sys.argv)
    raise SystemExit(main())
