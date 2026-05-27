"""
Confidence scorer — combines evaluation signals into a single 0-1 score.

BUG FIXED: original defaulted agreement_score to 0.5 when there were zero
existing results, which *inflated* the confidence of the very first result
found.  The fixed version defaults to 0.0 (no corroboration yet) so early
results are scored conservatively and confidence grows as evidence accumulates.
"""


def calculate_confidence(
    evaluation: dict,
    existing_results: list,
) -> float:
    """
    Compute a weighted confidence score for a single research result.

    Formula
    -------
    confidence = 0.40 * relevance
               + 0.35 * credibility
               + 0.15 * authority
               + 0.10 * agreement

    Where:
    - ``agreement`` grows from 0 → 1 as corroborating results accumulate
      (saturates at 5 results).  Starts at **0.0**, not 0.5, so the first
      result is scored on its own merits only.

    Parameters
    ----------
    evaluation : dict
        Output of :func:`agent.evaluator.evaluate_summary`.
    existing_results : list
        Results already collected for this task (used to gauge agreement).

    Returns
    -------
    float
        Confidence in [0.0, 1.0] rounded to 2 decimal places.
    """
    relevance = float(evaluation.get("relevance", 0.5))
    credibility = float(evaluation.get("credibility", 0.5))
    authority = float(evaluation.get("authority", 0.4))

    # BUG FIX: was 0.5 when len == 0, inflating first-result confidence
    agreement = min(len(existing_results) / 5.0, 1.0) if existing_results else 0.0

    confidence = (
        0.40 * relevance
        + 0.35 * credibility
        + 0.15 * authority
        + 0.10 * agreement
    )

    return round(min(max(confidence, 0.0), 1.0), 2)
