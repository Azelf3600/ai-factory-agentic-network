"""
CROSS-VALIDATION AGENT

Runs after Moat / Margin / Growth / Risk / Exposure, before Ranking.

Per project spec Section 4: "Agents cross-validate assumptions to prevent
single-factor bias." Previously nothing did this — the five scoring agents
ran independently and only ever met at Ranking, which only checks for
*missing* data (unmatched), never *inconsistent* data across agents.

Deliberately rules-based, not a second LLM call re-judging another agent's
output. An LLM "does this look right?" pass would just be a second opinion,
not a validation — it has no more claim to truth than the first agent's
guess. Deterministic rules over the agents' own numbers is the only way to
actually catch (moat=5, growth=3%) type contradictions consistently.

This agent does NOT change any score. It only produces flags for the
Ranking/Report stage to surface — score correction stays a human/analyst
decision, same as the "unmatched" flag pattern.

Add new rules as you observe real inconsistent outputs in test runs — the
four below are a starting set, not exhaustive.
"""

import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.utils import build_lookup_key, _get  # noqa: E402


def _index_by_key(records: List[Any]) -> Dict[str, dict]:
    index = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        key = build_lookup_key(r.get("company"), r.get("ticker"))
        if key:
            index[key] = r
    return index


# Segments where a >40% operating margin (margin_score == 5) is atypical
# enough to warrant a second look rather than accepting the LLM's recall
# at face value. Compute/Servers and Networking (software-heavy, IP-licensing
# names) are excluded since >40% margins are common and plausible there.
MARGIN_IMPLAUSIBLE_SEGMENTS = {"Power Infrastructure", "Cooling Systems", "Engineering & Construction"}


def _rule_high_moat_low_growth(moat: int, growth: float) -> str | None:
    """A dominant/defensible moat (4-5) paired with a low 3yr AI-driven CAGR
    (<10%) is internally inconsistent for an AI-Factory-growth thesis
    specifically: strong moat should be translating into captured growth,
    not just defensibility with no upside. Worth a rationale check."""
    if moat is not None and moat >= 4 and growth is not None and growth < 10:
        return (f"Moat={moat} (strong/dominant) but 3yr AI-driven CAGR={growth}% "
                f"(low) — check whether the Growth Agent's rationale accounts for "
                f"the moat, or whether the Moat Agent's rationale actually supports "
                f"an AI-Factory growth thesis for this company.")
    return None


def _rule_low_moat_high_growth(moat: int, growth: float) -> str | None:
    """The inverse case: a weak/no moat (0-1) paired with a very high CAGR
    (>40%) implies growth with no durability — plausible short-term but
    worth flagging since it's the kind of number a hype-prone LLM call is
    most likely to produce without real backing."""
    if moat is not None and moat <= 1 and growth is not None and growth > 40:
        return (f"Moat={moat} (weak/no defensibility) but 3yr AI-driven CAGR={growth}% "
                f"(very high) — high growth with no moat is fragile; verify this "
                f"isn't an unsupported hype number.")
    return None


def _rule_implausible_margin_for_segment(margin_score: int, segment: str) -> str | None:
    """margin_score == 5 means the LLM reported >40% operating margin. That's
    common in Compute/Servers (chip IP, software) but atypical in Power,
    Cooling, or Engineering & Construction, which tend to be lower-margin,
    capital-intensive, competitively-bid businesses. Flag rather than trust
    the LLM's recall silently."""
    if margin_score == 5 and segment in MARGIN_IMPLAUSIBLE_SEGMENTS:
        return (f"Margin score 5 (>40% operating margin) reported in '{segment}', "
                f"a segment where margins that high are atypical — verify the "
                f"Margin Agent's operating_margin_pct against a real filing before "
                f"trusting this score.")
    return None


def _rule_exposure_growth_mismatch(exposure_pct: float, growth: float) -> str | None:
    """Very low AI revenue exposure (<15%) paired with a very high AI-driven
    CAGR (>30%) is a mismatch: if AI Factory work is a small sliver of this
    company's revenue today, it's unlikely to swing overall company growth
    that far on its own within 3 years, unless that sliver itself is
    forecast to grow explosively — worth a rationale check either way."""
    if exposure_pct is not None and exposure_pct < 15 and growth is not None and growth > 30:
        return (f"AI revenue exposure={exposure_pct}% (low) but 3yr AI-driven "
                f"CAGR={growth}% (high) — a small AI-exposed revenue base "
                f"driving that much company-wide growth is a mismatch worth checking.")
    return None


def cross_validation_node(state: dict) -> dict:
    companies = state.get("companies", [])
    moat_idx = _index_by_key(state.get("moat_scores", []))
    margin_idx = _index_by_key(state.get("margin_scores", []))
    growth_idx = _index_by_key(state.get("growth_forecasts", []))
    exposure_idx = _index_by_key(state.get("ai_revenue_exposures", []))

    flags: List[dict] = []

    for c in companies:
        name = _get(c, "name", "Unknown")
        ticker = _get(c, "ticker")
        segment = _get(c, "segment", "Compute / Servers")

        key = build_lookup_key(name, ticker)

        moat_rec = moat_idx.get(key)
        margin_rec = margin_idx.get(key)
        growth_rec = growth_idx.get(key)
        exposure_rec = exposure_idx.get(key)

        moat_val = (moat_rec or {}).get("score")
        margin_score = (margin_rec or {}).get("score")
        growth_val = (growth_rec or {}).get("cagr_pct")
        exposure_pct = (exposure_rec or {}).get("exposure_pct")

        checks = [
            ("high_moat_low_growth", _rule_high_moat_low_growth(moat_val, growth_val)),
            ("low_moat_high_growth", _rule_low_moat_high_growth(moat_val, growth_val)),
            ("implausible_margin_for_segment", _rule_implausible_margin_for_segment(margin_score, segment)),
            ("exposure_growth_mismatch", _rule_exposure_growth_mismatch(exposure_pct, growth_val)),
        ]

        for rule_name, detail in checks:
            if detail:
                flags.append({
                    "company": name,
                    "ticker": ticker or "N/A",
                    "rule": rule_name,
                    "severity": "warning",
                    "detail": detail,
                })

    if flags:
        print(f"[Cross-Validation Agent] {len(flags)} flag(s) raised across {len(companies)} companies.")

    return {"cross_validation_flags": flags}
