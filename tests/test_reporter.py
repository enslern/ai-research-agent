"""Tests for agent.reporter"""

import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from agent.reporter import generate_report, print_summary, _build_sections


SAMPLE_FINDINGS = [
    {
        "task": "What is spaced repetition?",
        "results": [
            {
                "url": "https://example.com/sr",
                "source_title": "Spaced Repetition Guide",
                "summary": "Spaced repetition is a learning technique.",
                "confidence": 0.85,
            }
        ],
    },
    {
        "task": "How many hours to study per day?",
        "results": [],
    },
]


def _mock_client(text: str):
    mc = MagicMock()
    mc.chat.completions.create.return_value.choices[0].message.content = text
    return mc


class TestBuildSections:
    def test_includes_task_names(self):
        sections = _build_sections(SAMPLE_FINDINGS)
        assert "What is spaced repetition?" in sections

    def test_includes_source_titles(self):
        sections = _build_sections(SAMPLE_FINDINGS)
        assert "Spaced Repetition Guide" in sections

    def test_handles_empty_results(self):
        sections = _build_sections(SAMPLE_FINDINGS)
        assert "How many hours" in sections


class TestGenerateReport:
    def test_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("agent.reporter.get_client", return_value=_mock_client("# Synthesis\n\nAnalysis.")):
            path = generate_report("test goal", SAMPLE_FINDINGS)
        assert os.path.exists(path)

    def test_report_contains_goal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("agent.reporter.get_client", return_value=_mock_client("Synthesis.")):
            path = generate_report("my unique research goal", SAMPLE_FINDINGS)
        content = Path(path).read_text()
        assert "my unique research goal" in content

    def test_report_contains_source_appendix(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("agent.reporter.get_client", return_value=_mock_client("Synthesis.")):
            path = generate_report("goal", SAMPLE_FINDINGS)
        content = Path(path).read_text()
        assert "Source Appendix" in content

    def test_fallback_on_llm_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mc = MagicMock()
        mc.chat.completions.create.side_effect = Exception("down")
        with patch("agent.reporter.get_client", return_value=mc):
            path = generate_report("goal", SAMPLE_FINDINGS)
        assert os.path.exists(path)


class TestPrintSummary:
    def test_no_exception_on_empty_findings(self, capsys):
        print_summary([])
        captured = capsys.readouterr()
        assert "RESEARCH FINDINGS" in captured.out

    def test_prints_task_names(self, capsys):
        print_summary(SAMPLE_FINDINGS)
        captured = capsys.readouterr()
        assert "What is spaced repetition?" in captured.out
