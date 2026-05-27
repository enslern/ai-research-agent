"""
Web scraper using Playwright + readability-lxml + BeautifulSoup.

Launches a headless Chromium instance, fetches the page, cleans it with
Mozilla's Readability algorithm, and returns structured text content.
"""

from __future__ import annotations

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
from readability import Document

_PAGE_TIMEOUT_MS = 15_000   # 15 s — avoid hanging on slow pages
_MAX_PARAGRAPHS = 50        # Cap paragraphs to keep summaries focused


def read_webpage(url: str) -> dict | None:
    """
    Fetch and parse *url*, returning structured text content.

    Returns
    -------
    dict or None
        ``{"title": str, "headings": list[str], "paragraphs": list[str]}``
        Returns ``None`` if the page could not be fetched or yielded no content.
    """
    html = _fetch_html(url)
    if html is None:
        return None

    return _parse_html(html)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _fetch_html(url: str) -> str | None:
    """Fetch raw HTML via headless Chromium.  Returns None on failure."""
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=_PAGE_TIMEOUT_MS,
            )
            html = page.content()
            browser.close()
        return html
    except PlaywrightTimeout:
        print(f"[scraper] Timeout fetching {url}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[scraper] Failed to fetch {url}: {exc}")
        return None


def _parse_html(html: str) -> dict | None:
    """Extract title, headings, and paragraphs from raw HTML."""
    try:
        doc = Document(html)
        title = doc.title() or "Untitled"
        cleaned_html = doc.summary()
    except Exception as exc:  # noqa: BLE001
        print(f"[scraper] Readability failed: {exc}")
        return None

    soup = BeautifulSoup(cleaned_html, "html.parser")

    headings = [
        tag.get_text(strip=True)
        for tag in soup.find_all(["h1", "h2", "h3"])
        if tag.get_text(strip=True)
    ]

    paragraphs = [
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ][:_MAX_PARAGRAPHS]

    if not paragraphs:
        return None

    return {"title": title, "headings": headings, "paragraphs": paragraphs}
