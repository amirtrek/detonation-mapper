"""Helpers for flushed progress output."""

from __future__ import annotations

import sys
from itertools import count
from typing import Any

_STEP_COUNTER = count(1)


def progress(message: str) -> None:
    """Emit a progress line immediately without mixing with stdout results."""

    step = next(_STEP_COUNTER)
    print(f"[step {step:05d}] {message}", file=sys.stderr, flush=True)


def progress_kv(label: str, **values: Any) -> None:
    """Reserved for verbose tracing; intentionally silent by default."""

    _ = (label, values)
