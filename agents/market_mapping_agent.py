"""
MARKET MAPPING AGENT

Fix from previous version: this agent used to analyze a single company's
"spend layers" and write its result to state["market_mapping_result"],
which nothing downstream ever read — Company Ingestion always fell back to
its own hardcoded segment list, making this agent a no-op in the graph.

This version does what the project brief's Section 4 table actually asks:
"Maps AI Factory spend across infrastructure layers" — a one-time mapping
of the whole value chain, not a per-company lookup — and writes the result
to state["segments"], which Company Ingestion now genuinely reads.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Real spend-share breakdown derived from the Stargate reference material
# (Servers, Networking, Power [UPS+turbines+switchgear+generators+PDUs],
# Cooling [towers+chillers+CRAHs], Engineering & Construction), rolled up
# into the 5 segments the rest of the pipeline uses. Total reference spend
# ~= $68.4B; percentages below are each layer's share of that total.
SEGMENT_SPEND_SHARE = {
    "Compute / Servers": 71.2,
    "Networking": 12.3,
    "Power Infrastructure": 9.7,
    "Cooling Systems": 3.1,
    "Engineering & Construction": 3.7,
}

SEGMENTS = list(SEGMENT_SPEND_SHARE.keys())


def market_mapping_node(state: dict) -> dict:
    """
    No LLM call needed here — the value-chain layer split is a fixed
    reference mapping (from the project's own source material), not
    something that needs to be re-derived at runtime. Keeping this
    deterministic also means it can't silently drift between pipeline runs.
    """
    return {
        "segments": SEGMENTS,
        "market_mapping_result": {
            "segment_spend_share_pct": SEGMENT_SPEND_SHARE,
            "note": "Derived from AI Factory equipment cost reference data (Stargate breakdown).",
        },
    }