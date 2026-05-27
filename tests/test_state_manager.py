"""
Tests for storage.state_manager

Verifies numpy-safe serialisation (the original crash bug) and round-trips.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from storage.state_manager import save_state, load_state, clear_state


@pytest.fixture(autouse=True)
def clean_state():
    """Delete state file before and after every test."""
    clear_state()
    yield
    clear_state()


class TestSaveAndLoad:
    def test_round_trip(self):
        tasks = [{"task": "find stuff", "status": "completed"}]
        findings = [{"task": "find stuff", "results": []}]
        save_state(tasks, findings)
        loaded = load_state()
        assert loaded is not None
        assert loaded["tasks"] == tasks
        assert loaded["findings"] == findings

    def test_completed_tasks_populated(self):
        tasks = [
            {"task": "a", "status": "completed"},
            {"task": "b", "status": "pending"},
        ]
        save_state(tasks, [])
        loaded = load_state()
        assert "a" in loaded["completed_tasks"]
        assert "b" not in loaded["completed_tasks"]

    def test_numpy_types_dont_crash(self):
        """
        BUG FIX: original json.dump raised TypeError on numpy scalars.
        Findings can carry np.float32 values from FAISS metadata leakage.
        """
        findings = [{
            "task": "test",
            "results": [{
                "confidence": np.float32(0.75),       # was crashing
                "relevance": np.float64(0.85),
                "count": np.int64(42),
                "embedding_slice": np.array([0.1, 0.2], dtype=np.float32),
            }]
        }]
        # Should not raise
        save_state([], findings)
        loaded = load_state()
        r = loaded["findings"][0]["results"][0]
        assert abs(r["confidence"] - 0.75) < 0.01

    def test_memory_state_saved(self):
        mem = {"metadata": [{"summary": "cached thing"}]}
        save_state([], [], memory_state=mem)
        loaded = load_state()
        assert loaded["memory"] == mem


class TestLoadState:
    def test_returns_none_when_no_file(self):
        assert load_state() is None

    def test_returns_none_on_corrupt_json(self, tmp_path):
        state_file = Path("storage/state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("{ not valid json }", encoding="utf-8")
        result = load_state()
        assert result is None


class TestClearState:
    def test_clear_removes_file(self):
        save_state([], [])
        assert load_state() is not None
        clear_state()
        assert load_state() is None

    def test_clear_is_idempotent(self):
        clear_state()  # Should not raise even if file doesn't exist
        clear_state()
