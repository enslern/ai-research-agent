"""
Vector memory store using FAISS + SentenceTransformers.

Stores research summaries as embeddings so the agent can
recall relevant past findings instead of re-fetching URLs.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """In-memory semantic search over past research summaries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        # Stores serialisable dicts only — no numpy arrays
        self.metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_memory(self, summary_data: dict) -> None:
        """
        Encode summary text and add to the FAISS index.

        Only stores JSON-safe scalar fields so state_manager
        can serialise memory without issues.
        """
        text = summary_data.get("summary", "")
        if not text:
            return

        embedding = self._encode(text)
        self.index.add(embedding)

        # Store only serialisable fields
        safe_entry = {
            "url": summary_data.get("url", ""),
            "summary": summary_data.get("summary", ""),
            "source_title": summary_data.get("source_title", ""),
            "task": summary_data.get("task", ""),
            "confidence": summary_data.get("confidence", 0.0),
            "evaluation": summary_data.get("evaluation", {}),
        }
        self.metadata.append(safe_entry)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search_memory(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Return up to *top_k* past summaries most relevant to *query*.

        BUG FIXED: the original had search_memory defined *inside* itself
        (nested def), so the outer method body never ran — it just returned
        None every time.  The actual search logic now lives at the correct
        indentation level.
        """
        if len(self.metadata) == 0:
            return []

        query_embedding = self._encode(query)

        # FAISS search returns (distances, indices) — shapes (1, top_k)
        actual_k = min(top_k, len(self.metadata))
        distances, indices = self.index.search(query_embedding, actual_k)

        results = []
        for idx in indices[0]:
            if idx == -1:                      # FAISS sentinel for "no result"
                continue
            if idx >= len(self.metadata):      # Guard against index mismatch
                continue
            results.append(self.metadata[idx])

        return results

    # ------------------------------------------------------------------
    # Persistence helpers (called by state_manager)
    # ------------------------------------------------------------------

    def get_serialisable_state(self) -> dict:
        """Return metadata list — safe to json.dump."""
        return {"metadata": self.metadata}

    def load_from_state(self, state: dict) -> None:
        """Re-populate index from a previously saved state dict."""
        for entry in state.get("metadata", []):
            self.add_memory(entry)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        embedding = self.model.encode([text])
        return np.array(embedding, dtype="float32")

    def __len__(self) -> int:
        return len(self.metadata)
