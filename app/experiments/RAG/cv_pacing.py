"""Defaults and backoff for RAG CV pipelines (RPM / 429 safety across backends)."""

CV_DEFAULT_SLEEP_SECONDS = 12.0
CV_FOLD_GAP_MULTIPLIER = 3.0


def rate_limit_backoff_seconds(attempt_index: int) -> float:
    """
    Exponential wait after a rate-limited failure before the next attempt.

    ``attempt_index`` is 0-based within the retry loop (first backoff uses 0).
    Capped so a single call does not sleep unbounded.
    """
    base = 10.0
    cap = 120.0
    return min(cap, base * (2**attempt_index))


def generic_retry_backoff_seconds(attempt_index: int) -> float:
    """Shorter backoff for non-429 transient errors."""
    return min(60.0, 5.0 * (2**attempt_index))
