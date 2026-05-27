"""
Tests for agent.planner.build_plan
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from agent.planner import build_plan


def _mock_client(content: str):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    mock_client.chat.completions.create.return_value = mock_resp
    return mock_client


class TestBuildPlan:
    def test_returns_correct_structure(self):
        with patch("agent.planner.get_client", return_value=_mock_client(json.dumps(["Q1", "Q2", "Q3"]))):
            tasks = build_plan("test goal")
        assert len(tasks) == 3
        for task in tasks:
            assert "task" in task
            assert task["status"] == "pending"

    def test_strips_markdown_fences(self):
        fenced = '```json\n["question 1", "question 2"]\n```'
        with patch("agent.planner.get_client", return_value=_mock_client(fenced)):
            tasks = build_plan("goal")
        assert len(tasks) == 2

    def test_retries_on_failure(self):
        call_count = 0
        good_client = _mock_client(json.dumps(["Q1"]))

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Rate limited")
            return good_client.chat.completions.create.return_value

        failing_client = MagicMock()
        failing_client.chat.completions.create.side_effect = side_effect

        with patch("agent.planner.get_client", return_value=failing_client), \
             patch("agent.planner.time.sleep"):
            tasks = build_plan("goal")

        assert len(tasks) == 1
        assert call_count == 3

    def test_raises_after_max_retries(self):
        bad_client = MagicMock()
        bad_client.chat.completions.create.side_effect = Exception("always fails")

        with patch("agent.planner.get_client", return_value=bad_client), \
             patch("agent.planner.time.sleep"):
            with pytest.raises(RuntimeError, match="Planner failed"):
                build_plan("goal")

    def test_filters_empty_questions(self):
        with patch("agent.planner.get_client", return_value=_mock_client(json.dumps(["Q1", "", "  ", "Q2"]))):
            tasks = build_plan("goal")
        assert len(tasks) == 2
