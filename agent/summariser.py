"""
Summariser — condenses raw webpage content into a research-ready summary.

Uses Groq (llama-3.3-70b-versatile) via the OpenAI-compatible SDK.
"""

from agent._llm import get_client

_MODEL = "llama-3.3-70b-versatile"
_MAX_PARAGRAPH_CHARS = 6000


def summarize_webpage(webpage_data: dict, url: str) -> dict:
    """
    Summarise *webpage_data* into a concise, fact-focused paragraph.

    Parameters
    ----------
    webpage_data : dict
        Keys: ``"title"`` (str), ``"headings"`` (list[str]),
        ``"paragraphs"`` (list[str]).
    url : str
        Source URL.

    Returns
    -------
    dict
        ``{"url", "summary", "source_title"}``
    """
    paragraphs = webpage_data.get("paragraphs", [])
    headings = webpage_data.get("headings", [])
    title = webpage_data.get("title", "Untitled")

    body = "\n".join(paragraphs)
    if len(body) > _MAX_PARAGRAPH_CHARS:
        body = body[:_MAX_PARAGRAPH_CHARS] + "\n[content truncated]"

    heading_block = "\n".join(f"- {h}" for h in headings[:10]) if headings else "N/A"

    prompt = (
        "You are a research assistant.\n\n"
        "Summarise the following webpage into a concise, fact-focused paragraph "
        "of 150 words or fewer.\n\n"
        "Focus on:\n"
        "- Key ideas and claims\n"
        "- Important facts and statistics\n"
        "- Relevant insights\n\n"
        f"Title: {title}\n\n"
        f"Headings:\n{heading_block}\n\n"
        f"Content:\n{body}"
    )

    try:
        response = get_client().chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        summary = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[summariser] LLM call failed for {url}: {exc}")
        summary = f"Summary unavailable ({exc})"

    return {
        "url": url,
        "summary": summary,
        "source_title": title,
    }
