"""
State persistence — save and load agent progress to/from disk.

BUG FIXED: original json.dump could fail when FAISS metadata leaked numpy
arrays (np.float32, np.int64) into the findings dict.  All non-serialisable
types are now coerced via a custom JSON encoder before writing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_STATE_PATH = Path("storage/state.json")


# ------------------------------------------------------------------
# Custom encoder — handles numpy scalars / arrays
# ------------------------------------------------------------------

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def save_state(
    tasks: list[dict],
    findings: list[dict],
    memory_state: dict | None = None,
) -> None:
    """
    Persist current agent state to ``storage/state.json``.

    Parameters
    ----------
    tasks : list[dict]
        Task list with status markers.
    findings : list[dict]
        Accumulated research findings.
    memory_state : dict, optional
        Serialisable snapshot from :meth:`VectorStore.get_serialisable_state`.
    """
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "tasks": tasks,
        "findings": findings,
        "completed_tasks": [
            t["task"] for t in tasks if t.get("status") == "completed"
        ],
        "memory": memory_state or {},
    }

    with open(_STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, cls=_SafeEncoder)


def load_state() -> dict | None:
    """
    Load a previously saved state.

    Returns ``None`` if no state file exists yet.
    """
    if not _STATE_PATH.exists():
        return None

    with open(_STATE_PATH, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"[state_manager] Corrupt state file, ignoring: {exc}")
            return None


def clear_state() -> None:
    """Delete the saved state file (used in tests and fresh runs)."""
    if _STATE_PATH.exists():
        os.remove(_STATE_PATH)
