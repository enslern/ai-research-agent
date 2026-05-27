"""
Reporter — synthesises raw findings into a structured markdown report.

This module was completely absent from the original codebase.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent._llm import get_client

_MODEL = "llama-3.3-70b-versatile"
_REPORTS_DIR = Path("reports")


def generate_report(goal: str, findings: list[dict]) -> str:
    """
    Generate a markdown research report from *findings*.

    Returns
    -------
    str
        Absolute path to the saved report file.
    """
    _REPORTS_DIR.mkdir(exist_ok=True)

    sections = _build_sections(findings)
    narrative = _synthesise(goal, sections)
    report_md = _format_report(goal, findings, narrative)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = _REPORTS_DIR / f"report_{timestamp}.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"\n[reporter] Report saved → {report_path}")
    return str(report_path)


def print_summary(findings: list[dict]) -> None:
    """Pretty-print findings to stdout (no LLM call)."""
    print("\n" + "=" * 60)
    print("RESEARCH FINDINGS")
    print("=" * 60)

    for finding in findings:
        print(f"\n📌 Task: {finding['task']}")
        results = finding.get("results", [])
        if not results:
            print("   No results found.")
            continue
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] {r.get('source_title', 'Untitled')}")
            print(f"      URL        : {r.get('url', 'N/A')}")
            print(f"      Confidence : {r.get('confidence', 0):.2f}")
            print(f"      Summary    : {r.get('summary', '')[:200]}…")


def _build_sections(findings: list[dict]) -> str:
    lines: list[str] = []
    for finding in findings:
        lines.append(f"\n## {finding['task']}")
        for r in finding.get("results", []):
            lines.append(f"\n### {r.get('source_title', 'Source')}")
            lines.append(f"URL: {r.get('url', '')}")
            lines.append(f"Confidence: {r.get('confidence', 0):.2f}")
            lines.append(f"\n{r.get('summary', 'No summary available.')}")
    return "\n".join(lines)


def _synthesise(goal: str, sections: str) -> str:
    prompt = (
        "You are a senior research analyst.\n\n"
        f"Research Goal:\n{goal}\n\n"
        "Below are summarised findings from multiple sources.\n\n"
        f"{sections}\n\n"
        "Write a cohesive, well-structured research synthesis in markdown.\n"
        "Include:\n"
        "1. An executive summary (3-5 sentences)\n"
        "2. Key findings per sub-question\n"
        "3. Conflicting evidence or gaps\n"
        "4. Actionable conclusions\n\n"
        "Be analytical and precise. Do not pad with filler."
    )

    try:
        response = get_client().chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[reporter] LLM synthesis failed: {exc}")
        return sections  # Fallback: raw summaries


def _format_report(goal: str, findings: list[dict], narrative: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_sources = sum(len(f.get("results", [])) for f in findings)

    header = (
        f"# Research Report\n\n"
        f"**Goal:** {goal}\n\n"
        f"**Generated:** {timestamp}\n\n"
        f"**Tasks completed:** {len(findings)}\n\n"
        f"**Sources analysed:** {total_sources}\n\n"
        "---\n\n"
    )

    appendix = ["\n---\n\n## Source Appendix\n"]
    for finding in findings:
        appendix.append(f"\n### {finding['task']}\n")
        for r in finding.get("results", []):
            appendix.append(
                f"- [{r.get('source_title', r.get('url', 'Unknown'))}]"
                f"({r.get('url', '#')}) "
                f"*(confidence: {r.get('confidence', 0):.2f})*"
            )

    return header + narrative + "\n".join(appendix)
