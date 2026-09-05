"""
Real-data layer for Margin Analysis. Pulls actual trailing operating margin
from yfinance where available, so the pipeline stops relying on an LLM's
guess/recall for a number that's actually a reported, verifiable figure.

Does NOT require scaling the dataset — works against the current 30-company
seed set as-is. Falls back to None (caller sends to LLM) for anything
yfinance can't resolve: some non-US listings (SU.PA, 6367.T, ABBN.SW,
ENR.DE, RR.L, WSP.TO) can have gaps, delayed data, or different field
availability than US large-caps.
"""

import yfinance as yf


def fetch_real_operating_margin(ticker: str) -> dict | None:
    """
    Returns {"operating_margin_pct": float, "source": "real"} if yfinance
    has a usable figure, else None (caller falls back to LLM estimate).
    """
    if not ticker or ticker == "N/A":
        return None
    try:
        info = yf.Ticker(ticker).info
        margin = info.get("operatingMargins")
        if margin is None:
            return None
        return {
            "operating_margin_pct": round(margin * 100, 1),
            "source": "real",
        }
    except Exception as e:
        print(f"[Real Margin Fetch] {ticker} failed: {e}")
        return None
