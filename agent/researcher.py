"""
Researcher — orchestrates the per-task research loop.

BUG FIXED: when a memory hit occurred, the cached summary was returned
as-is.  If that entry was stored before the "confidence" key was added
(or if it simply wasn't set), main.py's print loop raised a KeyError.
All results — whether from memory or fresh web fetch — are now guaranteed
to carry a "confidence" key.
"""

from __future__ import annotations

from browser.search import search_web
from browser.scrapper import read_webpage
from agent.summariser import summarize_webpage
from agent.evaluator import evaluate_summary
from agent.confidence import calculate_confidence
from memory.vector_store import VectorStore
from storage.state_manager import save_state

# Thresholds for accepting a source
_MIN_RELEVANCE = 0.60
_MIN_CREDIBILITY = 0.50


def run_research(tasks: list[dict], memory: VectorStore) -> list[dict]:
    """
    Process every pending task and return a list of findings.

    Parameters
    ----------
    tasks : list[dict]
        Task dicts with ``"task"`` and ``"status"`` keys.
    memory : VectorStore
        Shared vector memory — read before web search, written after.

    Returns
    -------
    list[dict]
        ``[{"task": str, "results": list[dict]}, …]``
    """
    findings: list[dict] = []

    for current_task in tasks:
        if current_task["status"] != "pending":
            continue

        task_name: str = current_task["task"]
        print(f"\n{'='*60}")
        print(f"Researching: {task_name}")
        print("=" * 60)

        relevant_results: list[dict] = []

        # ----------------------------------------------------------
        # STEP 1 — Memory lookup
        # ----------------------------------------------------------
        memory_hits = memory.search_memory(task_name, top_k=3)

        if memory_hits:
            print(f"[memory] Found {len(memory_hits)} cached result(s).")
            for hit in memory_hits:
                # BUG FIX: ensure confidence key always present
                if "confidence" not in hit:
                    hit["confidence"] = calculate_confidence(
                        hit.get("evaluation", {}), []
                    )
                relevant_results.append(hit)
        else:
            print("[memory] No cache hit — fetching from web.")

            # ----------------------------------------------------------
            # STEP 2 — Web search
            # ----------------------------------------------------------
            urls = search_web(task_name)

            if not urls:
                print("[search] No URLs returned.")
            
            for url in urls:
                result = _process_url(url, task_name, relevant_results)
                if result is not None:
                    relevant_results.append(result)
                    memory.add_memory(result)

        findings.append({"task": task_name, "results": relevant_results})

        current_task["status"] = "completed"
        print(f"\n[researcher] Completed: {task_name}")

        # ----------------------------------------------------------
        # STEP 3 — Persist state after every task
        # ----------------------------------------------------------
        save_state(tasks, findings, memory.get_serialisable_state())
        print("[state] State saved.")

    return findings


def _process_url(
    url: str,
    task_name: str,
    existing_results: list[dict],
) -> dict | None:
    """
    Fetch, summarise, evaluate, and score a single URL.

    Returns the enriched summary dict, or None if the URL should be skipped.
    """
    print(f"  → Reading: {url}")

    webpage_data = read_webpage(url)
    if not webpage_data:
        print(f"  ✗ No content: {url}")
        return None

    # Summarise
    summary = summarize_webpage(webpage_data, url)
    summary["task"] = task_name

    # Evaluate
    evaluation = evaluate_summary(summary, task_name)
    summary["evaluation"] = evaluation

    # Score confidence
    confidence = calculate_confidence(evaluation, existing_results)
    summary["confidence"] = confidence

    relevance = evaluation.get("relevance", 0)
    credibility = evaluation.get("credibility", 0)

    print(
        f"  relevance={relevance:.2f}  "
        f"credibility={credibility:.2f}  "
        f"confidence={confidence:.2f}"
    )

    if relevance > _MIN_RELEVANCE and credibility > _MIN_CREDIBILITY:
        print("  ✓ Good source — saved.")
        return summary

    print("  ✗ Low quality — skipped.")
    return None
