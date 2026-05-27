"""Tests for agent.evaluator"""

from unittest.mock import MagicMock, patch
import json
import pytest

from agent.evaluator import get_authority_score, evaluate_summary


def _mock_client(payload: dict):
    mc = MagicMock()
    mc.chat.completions.create.return_value.choices[0].message.content = json.dumps(payload)
    return mc


class TestGetAuthorityScore:
    def test_known_high_authority(self):
        assert get_authority_score("https://arxiv.org/abs/1234") == 0.95

    def test_known_mid_authority(self):
        assert get_authority_score("https://medium.com/article") == 0.60

    def test_unknown_domain(self):
        assert get_authority_score("https://someblog.io/post") == 0.40

    def test_gov_domain(self):
        assert get_authority_score("https://data.gov/dataset") == 0.90

    def test_edu_domain(self):
        assert get_authority_score("https://mit.edu/course") == 0.88


class TestEvaluateSummary:
    def test_successful_evaluation(self):
        payload = {"relevance": 0.9, "credibility": 0.8, "needs_more_research": False}
        with patch("agent.evaluator.get_client", return_value=_mock_client(payload)):
            result = evaluate_summary(
                {"url": "https://arxiv.org/abs/1", "source_title": "Test", "summary": "test"},
                "test goal",
            )
        assert result["relevance"] == 0.9
        assert result["credibility"] == 0.8
        assert result["needs_more_research"] is False
        assert result["authority"] == 0.95  # arxiv.org

    def test_authority_blended_in(self):
        payload = {"relevance": 0.5, "credibility": 0.5, "needs_more_research": True}
        with patch("agent.evaluator.get_client", return_value=_mock_client(payload)):
            result = evaluate_summary(
                {"url": "https://reddit.com/r/test", "source_title": "T", "summary": "s"},
                "goal",
            )
        assert result["authority"] == 0.50  # reddit.com

    def test_fallback_on_api_failure(self):
        mc = MagicMock()
        mc.chat.completions.create.side_effect = Exception("API down")
        with patch("agent.evaluator.get_client", return_value=mc):
            result = evaluate_summary(
                {"url": "https://example.com", "source_title": "T", "summary": "s"},
                "goal",
            )
        assert "relevance" in result
        assert "credibility" in result
        assert "authority" in result
        assert "needs_more_research" in result

    def test_malformed_json_uses_fallback(self):
        mc = MagicMock()
        mc.chat.completions.create.return_value.choices[0].message.content = "not json {{"
        with patch("agent.evaluator.get_client", return_value=mc):
            result = evaluate_summary({"url": "x", "source_title": "T", "summary": "s"}, "goal")
        assert result["relevance"] == 0.5
