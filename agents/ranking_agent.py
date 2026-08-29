"""RANKING AGENT"""

import os
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import RankedCompany

def calculate_tafgs(moat: int, margin_pct: float, growth_cagr: float, discount_pct: float) -> float:
    moat_pts = (moat / 5.0) * 30.0
    margin_score = min(max(int(margin_pct // 10), 0), 5)
    margin_pts = (margin_score / 5.0) * 30.0
    growth_pts = min(growth_cagr / 100.0, 1.0) * 40.0
    
    base_score = moat_pts + margin_pts + growth_pts
    return round(base_score * (1.0 - discount_pct), 2)

def ranking_node(state: dict) -> dict:
    companies = state.get("companies", [])
    moat_map = {m.get("company"): m.get("score", 3) for m in state.get("moat_scores", []) if isinstance(m, dict)}
    margin_map = {m.get("company"): m.get("operating_margin_pct", 20.0) for m in state.get("margin_scores", []) if isinstance(m, dict)}
    growth_map = {g.get("company"): g.get("cagr_pct", 20.0) for g in state.get("growth_forecasts", []) if isinstance(g, dict)}
    
    risk_discount_map = {r.get("company"): r.get("discount_pct", 0.05) for r in state.get("risk_adjustments", []) if isinstance(r, dict)}
    risk_notes_map = {r.get("company"): r.get("risk_notes", "Standard execution risk applied") for r in state.get("risk_adjustments", []) if isinstance(r, dict)}

    ranked_items = []
    for idx, c in enumerate(companies):
        name = c.get("name") if isinstance(c, dict) else getattr(c, "name", "Unknown")
        m_val = moat_map.get(name, 3)
        mg_val = margin_map.get(name, 20.0)
        g_val = growth_map.get(name, 20.0)
        r_val = risk_discount_map.get(name, 0.05)
        r_notes = risk_notes_map.get(name, "Standard execution risk applied")
        
        tafgs = calculate_tafgs(m_val, mg_val, g_val, r_val)

        ranked_items.append({
            "rank": idx + 1,
            "company": name,
            "ticker": c.get("ticker", "N/A") if isinstance(c, dict) else "N/A",
            "segment": c.get("segment", "Compute / Servers") if isinstance(c, dict) else "Compute / Servers",
            "ai_revenue_exposure_pct": c.get("ai_revenue_exposure_pct", 50.0) if isinstance(c, dict) else 50.0,
            "moat_score": m_val,
            "operating_margin_pct": mg_val,
            "growth_cagr_pct": g_val,
            "risk_notes": r_notes,
            "tafgs_score": tafgs
        })

    ranked_items.sort(key=lambda x: x["tafgs_score"], reverse=True)
    for i, item in enumerate(ranked_items):
        item["rank"] = i + 1

    return {"rankings": ranked_items}