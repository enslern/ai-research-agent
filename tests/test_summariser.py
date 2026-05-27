"""Tests for agent.summariser.summarize_webpage"""

from unittest.mock import MagicMock, patch
import pytest

from agent.summariser import summarize_webpage


def _mock_client(text: str):
    mc = MagicMock()
    mc.chat.completions.create.return_value.choices[0].message.content = text
    return mc


_SAMPLE_PAGE = {
    "title": "How to Learn Fast",
    "headings": ["Introduction", "Key Techniques"],
    "paragraphs": ["Spaced repetition is effective.", "Active recall beats passive review."],
}


class TestSummarizeWebpage:
    def test_returns_expected_keys(self):
        with patch("agent.summariser.get_client", return_value=_mock_client("Summary here.")):
            result = summarize_webpage(_SAMPLE_PAGE, "https://example.com")
        assert "url" in result
        assert "summary" in result
        assert "source_title" in result

    def test_url_preserved(self):
        url = "https://example.com/article"
        with patch("agent.summariser.get_client", return_value=_mock_client("Summary.")):
            result = summarize_webpage(_SAMPLE_PAGE, url)
        assert result["url"] == url

    def test_title_preserved(self):
        with patch("agent.summariser.get_client", return_value=_mock_client("Summary.")):
            result = summarize_webpage(_SAMPLE_PAGE, "https://x.com")
        assert result["source_title"] == "How to Learn Fast"

    def test_api_failure_returns_fallback(self):
        mc = MagicMock()
        mc.chat.completions.create.side_effect = Exception("API error")
        with patch("agent.summariser.get_client", return_value=mc):
            result = summarize_webpage(_SAMPLE_PAGE, "https://x.com")
        assert "unavailable" in result["summary"].lower()

    def test_long_content_truncated(self):
        long_page = {
            "title": "Long Article",
            "headings": [],
            "paragraphs": ["x" * 10_000],
        }
        captured = []

        def capture(*args, **kwargs):
            captured.append(kwargs.get("messages", [{}])[0].get("content", ""))
            r = MagicMock()
            r.choices[0].message.content = "Summary."
            return r

        mc = MagicMock()
        mc.chat.completions.create.side_effect = capture
        with patch("agent.summariser.get_client", return_value=mc):
            summarize_webpage(long_page, "https://x.com")

        assert "[content truncated]" in captured[0]
