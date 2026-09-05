"""
Shared batching helper for LLM-based scoring agents at scale.

Every agent that sends a company list to an LLM in a single prompt risks
silent truncation once that list gets large — this happened once already
in Report Agent at just 20 companies (5-of-20 profiles returned). At 500
companies, an un-batched prompt across Moat/Growth/Risk/Exposure is even
more likely to truncate, time out, or degrade in quality.

BATCH_SIZE=25 keeps each prompt (and each expected structured response)
small enough that a model returning fewer results than requested is
immediately obvious and logged, rather than silently absorbed into a
"looks about right" total.
"""

import re
import time

BATCH_SIZE = 25


def chunk(items: list, size: int = BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def invoke_with_retry(structured_llm, prompt, max_retries: int = 4, default_backoff: float = 20.0):
    """
    Wraps a structured_llm.invoke() call with retry-on-rate-limit handling.

    Gemini's free tier caps requests at 15/minute per model. With five
    scoring agents each issuing ~20 batches, it's normal to exceed that
    ceiling within the first minute of a run — every batch that got
    throttled previously failed outright with no retry, silently sending
    that whole batch of companies to the fallback default. This is what
    caused 125 companies to fall back to defaults in one run (vs. 25 the
    run before) even though nothing about the data or matching logic
    changed — it was five agents' worth of parallel calls all hitting the
    same 15/min ceiling at once.

    On a 429/RESOURCE_EXHAUSTED error, Gemini's own error payload includes
    its suggested wait time (e.g. "Please retry in 43.65s" /
    RetryInfo.retryDelay). This parses that value and sleeps for it (plus
    a small buffer) rather than guessing a fixed backoff, so the retry is
    timed to actually land after the quota window rolls over instead of
    retrying too early and immediately hitting another 429.

    Non-rate-limit errors are NOT retried here — they re-raise immediately
    so the caller's existing except block logs and falls back exactly as
    before. This only changes behavior for the 429 case.
    """
    attempt = 0
    while True:
        try:
            return structured_llm.invoke(prompt)
        except Exception as e:
            msg = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg

            if not is_rate_limit or attempt >= max_retries:
                raise

            wait = default_backoff
            match = re.search(r"retry in ([\d.]+)\s*s", msg, re.IGNORECASE)
            if not match:
                match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s?'", msg)
            if match:
                wait = float(match.group(1)) + 2  # small buffer past Gemini's own estimate

            attempt += 1
            print(f"[Rate Limit] quota hit (attempt {attempt}/{max_retries}) — "
                  f"waiting {wait:.0f}s before retrying this batch.")
            time.sleep(wait)