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

BATCH_SIZE = 25


def chunk(items: list, size: int = BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i:i + size]
