"""
Evaluator — scores a summary for relevance, credibility, and authority.

BUG FIXED: original used google.genai with a Groq key — wrong client.
Everything is now on Groq via the lazy OpenAI-compatible client.
"""

import json

from agent._llm import get_client

_MODEL = "llama-3.3-70b-versatile"

_TRUSTED_DOMAINS: dict[str, float] = {
    "arxiv.org": 0.95,
    "github.com": 0.85,
    "wikipedia.org": 0.75,
    "stackoverflow.com": 0.80,
    "medium.com": 0.60,
    "reddit.com": 0.50,
    "news.ycombinator.com": 0.70,
    ".gov": 0.90,
    ".edu": 0.88,
}

_FALLBACK_EVALUATION: dict = {
    "relevance": 0.5,
    "credibility": 0.5,
    "authority": 0.4,
    "needs_more_research": True,
}


def get_authority_score(url: str) -> float:
    """Return a heuristic domain-authority score for *url*."""
    for domain, score in _TRUSTED_DOMAINS.items():
        if domain in url:
            return score
    return 0.40


def evaluate_summary(summary_data: dict, research_goal: str) -> dict:
    """
    Score a summary for relevance and credibility, blended with a
    rule-based authority score.

    Returns
    -------
    dict
        Keys: ``relevance``, ``credibility``, ``authority``, ``needs_more_research``.
    """
    prompt = (
        "You are a research quality evaluator.\n\n"
        f"Research Goal:\n{research_goal}\n\n"
        f"Source Title:\n{summary_data.get('source_title', 'Unknown')}\n\n"
        f"URL:\n{summary_data.get('url', '')}\n\n"
        f"Summary:\n{summary_data.get('summary', '')}\n\n"
        "Evaluate this source and return ONLY valid JSON (no markdown fences):\n\n"
        "{\n"
        '    "relevance": <float 0.0-1.0>,\n'
        '    "credibility": <float 0.0-1.0>,\n'
        '    "needs_more_research": <true|false>\n'
        "}"
    )

    try:
        response = get_client().chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )

        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        evaluation: dict = json.loads(raw)
        evaluation["relevance"] = float(evaluation.get("relevance", 0.5))
        evaluation["credibility"] = float(evaluation.get("credibility", 0.5))
        evaluation["needs_more_research"] = bool(evaluation.get("needs_more_research", True))

    except Exception as exc:  # noqa: BLE001
        print(f"[evaluator] Evaluation failed: {exc}")
        evaluation = dict(_FALLBACK_EVALUATION)

    evaluation["authority"] = get_authority_score(summary_data.get("url", ""))
    return evaluation
