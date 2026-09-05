"""RANKING AGENT"""

import os
import sys
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import RankedCompany  # noqa: F401 (kept for typing/reference)
from agents.utils import build_lookup_key, _get


def _index_by_key(records: List[dict]) -> Dict[str, dict]:
    """
    Keys every record by build_lookup_key(name, ticker). Safe to do a plain
    exact-match index here because fill_missing_with_fallback already
    normalized each record's company/ticker to the company's own canonical
    values (including anything recovered via fuzzy match upstream).
    """
    index = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        key = build_lookup_key(r.get("company"), r.get("ticker"))
        if key:
            index[key] = r
    return index


def calculate_tafgs(moat: int, margin_score: int, growth_cagr_pct: float, discount_pct: float) -> float:
    """
    Project spec Section 2: TAFGS = (Moat x Operating Margin Score) x Forecast
    AI-Driven Growth, then risk-discounted. Raw composite for relative
    ranking, not bounded to 0-100.
    """
    base_score = moat * margin_score * growth_cagr_pct
    return round(base_score * (1.0 - discount_pct), 2)


def ranking_node(state: dict) -> dict:
    companies = state.get("companies", [])

    moat_idx = _index_by_key(state.get("moat_scores", []))
    margin_idx = _index_by_key(state.get("margin_scores", []))
    growth_idx = _index_by_key(state.get("growth_forecasts", []))
    risk_idx = _index_by_key(state.get("risk_adjustments", []))
    exposure_idx = _index_by_key(state.get("ai_revenue_exposures", []))

    ranked_items = []
    for c in companies:
        name = _get(c, "name", "Unknown")
        ticker = _get(c, "ticker")
        segment = _get(c, "segment", "Compute / Servers")

        key = build_lookup_key(name, ticker)
        moat_rec = moat_idx.get(key)
        margin_rec = margin_idx.get(key)
        growth_rec = growth_idx.get(key)
        risk_rec = risk_idx.get(key)
        exposure_rec = exposure_idx.get(key)

        moat_val = (moat_rec or {}).get("score", 3)
        margin_pct = (margin_rec or {}).get("operating_margin_pct", 20.0)
        margin_score = (margin_rec or {}).get("score", 3)
        growth_val = (growth_rec or {}).get("cagr_pct", 20.0)
        risk_discount = (risk_rec or {}).get("discount_pct", 0.05)
        risk_notes = (risk_rec or {}).get("risk_notes", "Standard execution risk applied")

        # Exposure now comes from the dedicated agent. Fall back to the
        # Company's own ingestion-time placeholder only if the exposure
        # agent has no record at all for this ticker.
        if exposure_rec:
            exposure_pct = exposure_rec.get("exposure_pct", 50.0)
            exposure_source = exposure_rec.get("source", "estimated")
        else:
            exposure_pct = _get(c, "ai_revenue_exposure_pct", 50.0)
            exposure_source = "placeholder"

        unmatched = any([
            moat_rec is None or moat_rec.get("unmatched", False),
            margin_rec is None or margin_rec.get("unmatched", False),
            growth_rec is None or growth_rec.get("unmatched", False),
            risk_rec is None or risk_rec.get("unmatched", False),
            exposure_rec is None or exposure_rec.get("unmatched", False),
        ])

        tafgs = calculate_tafgs(moat_val, margin_score, growth_val, risk_discount)

        ranked_items.append({
            "rank": 0,  # assigned after sort
            "company": name,
            "ticker": ticker or "N/A",
            "segment": segment,
            "ai_revenue_exposure_pct": exposure_pct,
            "ai_revenue_exposure_source": exposure_source,
            "moat_score": moat_val,
            "operating_margin_pct": margin_pct,
            "margin_score": margin_score,
            "growth_cagr_pct": growth_val,
            "risk_notes": risk_notes,
            "tafgs_score": tafgs,
            "unmatched": unmatched,
        })

    ranked_items.sort(key=lambda x: x["tafgs_score"], reverse=True)
    for i, item in enumerate(ranked_items):
        item["rank"] = i + 1

    return {"rankings": ranked_items}