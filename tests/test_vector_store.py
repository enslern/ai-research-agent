"""
Tests for memory.vector_store.VectorStore

Covers the critical nested-def bug fix and all public methods.
The SentenceTransformer model is mocked to avoid network downloads.
"""

import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


def _make_mock_model():
    """Return a mock SentenceTransformer that produces deterministic 384-d vectors."""
    mock = MagicMock()
    # encode returns a (1, 384) float32 array based on string hash — deterministic
    def encode(texts, **kwargs):
        result = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            result.append(rng.random(384).astype(np.float32))
        return np.array(result, dtype=np.float32)
    mock.encode.side_effect = encode
    return mock


@pytest.fixture
def store():
    with patch("memory.vector_store.SentenceTransformer", return_value=_make_mock_model()):
        from memory.vector_store import VectorStore
        return VectorStore()


def _make_summary(n: int = 1) -> dict:
    return {
        "url": f"https://example.com/article-{n}",
        "summary": f"This is research summary number {n} about topic alpha beta gamma.",
        "source_title": f"Article {n}",
        "task": "test task",
        "confidence": 0.75,
        "evaluation": {"relevance": 0.8, "credibility": 0.7},
    }


class TestAddMemory:
    def test_adds_entry(self, store):
        store.add_memory(_make_summary(1))
        assert len(store) == 1

    def test_ignores_empty_summary(self, store):
        store.add_memory({"summary": "", "url": "x", "source_title": "x"})
        assert len(store) == 0

    def test_stores_only_safe_fields(self, store):
        store.add_memory(_make_summary(1))
        entry = store.metadata[0]
        for key in ("url", "summary", "source_title", "task", "confidence"):
            assert key in entry

    def test_multiple_entries(self, store):
        for i in range(5):
            store.add_memory(_make_summary(i))
        assert len(store) == 5


class TestSearchMemory:
    def test_returns_empty_when_no_memory(self, store):
        results = store.search_memory("anything")
        assert results == []

    def test_returns_results_after_add(self, store):
        """Core regression test for the nested-def bug."""
        store.add_memory(_make_summary(1))
        results = store.search_memory("research summary topic", top_k=1)
        # The original nested-def bug caused this to always return [] — now must return 1
        assert len(results) == 1

    def test_top_k_respected(self, store):
        for i in range(10):
            store.add_memory(_make_summary(i))
        results = store.search_memory("research", top_k=3)
        assert len(results) <= 3

    def test_result_has_expected_keys(self, store):
        store.add_memory(_make_summary(1))
        results = store.search_memory("research", top_k=1)
        assert results, "Expected at least one result"
        for key in ("url", "summary", "source_title"):
            assert key in results[0]

    def test_semantically_irrelevant_query_still_returns_results(self, store):
        """FAISS always returns the closest match even if semantically irrelevant."""
        store.add_memory(_make_summary(1))
        results = store.search_memory("completely unrelated banana", top_k=1)
        assert len(results) == 1


class TestPersistence:
    def test_serialisable_state_is_json_safe(self, store):
        store.add_memory(_make_summary(1))
        state = store.get_serialisable_state()
        serialised = json.dumps(state)  # must not raise
        assert serialised

    def test_load_from_state_restores_search(self, store):
        store.add_memory(_make_summary(1))
        state = store.get_serialisable_state()

        with patch("memory.vector_store.SentenceTransformer", return_value=_make_mock_model()):
            from memory.vector_store import VectorStore
            new_store = VectorStore()

        new_store.load_from_state(state)
        results = new_store.search_memory("research summary", top_k=1)
        assert len(results) == 1
