"""
Planner — breaks a high-level research goal into discrete sub-tasks.

Uses Groq (llama-3.3-70b-versatile) via the OpenAI-compatible SDK.
"""

import json
import time

from agent._llm import get_client

_MODEL = "llama-3.3-70b-versatile"
_MAX_RETRIES = 3
_RETRY_DELAY = 40  # seconds — Groq rate-limit window


def build_plan(goal: str, num_tasks: int = 5) -> list[dict]:
    """
    Decompose *goal* into *num_tasks* concrete research questions.

    Returns
    -------
    list[dict]
        Each dict has keys ``"task"`` (str) and ``"status"`` (``"pending"``).

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    prompt = (
        f"Break this research goal into {num_tasks} clear, specific research questions.\n\n"
        f"Goal:\n{goal}\n\n"
        "Return ONLY a valid JSON array of strings — no markdown, no explanation.\n\n"
        'Example: ["question 1", "question 2"]'
    )

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = get_client().chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            raw = response.choices[0].message.content or ""
            raw = raw.replace("```json", "").replace("```", "").strip()

            questions: list[str] = json.loads(raw)

            if not isinstance(questions, list):
                raise ValueError("LLM did not return a JSON list.")

            return [
                {"task": str(q).strip(), "status": "pending"}
                for q in questions
                if str(q).strip()
            ]

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[planner] Attempt {attempt}/{_MAX_RETRIES} failed: {exc}")
            if attempt < _MAX_RETRIES:
                print(f"[planner] Retrying in {_RETRY_DELAY}s …")
                time.sleep(_RETRY_DELAY)

    raise RuntimeError(
        f"Planner failed after {_MAX_RETRIES} attempts. Last error: {last_error}"
    )
