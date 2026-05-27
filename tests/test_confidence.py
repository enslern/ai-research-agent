"""
Tests for agent.confidence.calculate_confidence

Covers the first-result inflation bug fix.
"""

import pytest
from agent.confidence import calculate_confidence


def _eval(relevance=0.8, credibility=0.7, authority=0.6):
    return {
        "relevance": relevance,
        "credibility": credibility,
        "authority": authority,
        "needs_more_research": True,
    }


class TestCalculateConfidence:
    def test_returns_float(self):
        result = calculate_confidence(_eval(), [])
        assert isinstance(result, float)

    def test_result_in_range(self):
        result = calculate_confidence(_eval(), [])
        assert 0.0 <= result <= 1.0

    def test_rounded_to_2dp(self):
        result = calculate_confidence(_eval(), [])
        assert result == round(result, 2)

    def test_bug_fix_no_inflation_on_empty_results(self):
        """
        BUG FIX: original used agreement_score=0.5 when existing_results=[].
        Now it must be 0.0, so first-result confidence is lower.
        """
        result_empty = calculate_confidence(_eval(0.8, 0.7, 0.6), [])
        result_with_one = calculate_confidence(_eval(0.8, 0.7, 0.6), [{"x": 1}])
        # Having corroborating evidence must increase confidence
        assert result_with_one > result_empty

    def test_agreement_grows_with_more_results(self):
        base_eval = _eval(0.8, 0.7, 0.6)
        scores = [
            calculate_confidence(base_eval, [{}] * n)
            for n in range(6)
        ]
        # Confidence must be non-decreasing as evidence accumulates
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_agreement_saturates_at_5(self):
        base_eval = _eval(0.8, 0.7, 0.6)
        score_5 = calculate_confidence(base_eval, [{}] * 5)
        score_10 = calculate_confidence(base_eval, [{}] * 10)
        assert score_5 == score_10

    def test_high_quality_source(self):
        result = calculate_confidence(_eval(1.0, 1.0, 1.0), [{}] * 5)
        assert result > 0.9

    def test_low_quality_source(self):
        result = calculate_confidence(_eval(0.0, 0.0, 0.0), [])
        assert result == 0.0

    def test_missing_keys_default_gracefully(self):
        """Partial evaluation dict should not raise."""
        result = calculate_confidence({}, [])
        assert 0.0 <= result <= 1.0
