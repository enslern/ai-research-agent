"""
Web search via Tavily.

BUG FIXED: API key was hardcoded in the source file.
It is now loaded from the TAVILY_API_KEY environment variable.
"""

import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))


def search_web(query: str, max_results: int = 5) -> list[str]:
    """
    Search the web for *query* and return a list of URLs.

    Parameters
    ----------
    query : str
        The search query.
    max_results : int
        Maximum number of URLs to return (default 5).

    Returns
    -------
    list[str]
        Ordered list of result URLs, most relevant first.
    """
    try:
        response = _client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
        )
        return [r["url"] for r in response.get("results", [])]
    except Exception as exc:  # noqa: BLE001
        print(f"[search] Web search failed for '{query}': {exc}")
        return []
