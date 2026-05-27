"""
AI Research Agent — entry point.

Usage
-----
    python main.py

Set the research goal in the ``GOAL`` constant below, or pass it as a
command-line argument:

    python main.py "your research goal here"
"""

from __future__ import annotations

import sys

from agent.planner import build_plan
from agent.researcher import run_research
from agent.reporter import generate_report, print_summary
from memory.vector_store import VectorStore
from storage.state_manager import load_state, clear_state


GOAL = "how to get good at solving rubiks cube"


def main(goal: str = GOAL) -> None:
    print("\n" + "=" * 60)
    print("AI RESEARCH AGENT")
    print("=" * 60)
    print(f"Goal: {goal}\n")

    memory = VectorStore()

    # ------------------------------------------------------------------
    # Resume or start fresh
    # ------------------------------------------------------------------
    saved = load_state()

    if saved:
        print("Resuming previous run …")
        tasks = saved["tasks"]
        # Reload memory from saved state
        memory.load_from_state(saved.get("memory", {}))
    else:
        tasks = build_plan(goal)

    print(f"\nTasks ({len(tasks)}):")
    for t in tasks:
        status_icon = "✓" if t["status"] == "completed" else "○"
        print(f"  {status_icon} {t['task']}")

    pending = sum(1 for t in tasks if t["status"] == "pending")
    if pending == 0:
        print("\nAll tasks already completed.  Run with --fresh to restart.")
        return

    # ------------------------------------------------------------------
    # Run research
    # ------------------------------------------------------------------
    findings = run_research(tasks, memory)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print_summary(findings)

    report_path = generate_report(goal, findings)
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] == "--fresh":
        print("Clearing previous state …")
        clear_state()
        args = args[1:]

    goal = " ".join(args) if args else GOAL
    main(goal)
