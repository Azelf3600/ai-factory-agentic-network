"""
RANKING AGENT (Assigned: Malones)

Function:
  Consolidates scores across Moat, Margin, Growth, and Risk Adjustment agents,
  computes the Total AI Factory Growth Score (TAFGS), and produces a ranked leaderboard
  of evaluated AI infrastructure companies.

Reads from shared state:
  - companies (list)
  - moat_scores (list)
  - margin_scores (list)
  - growth_forecasts (list)
  - risk_adjustments (list)
  - company_name (str, fallback for single-company standalone execution)

Writes to shared state:
  - state["ranking_result"] (Naming Guide Sheet contract)
  - state["rankings"] (schema.py PipelineState contract)
"""

import os
import sys
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# Ensure parent directory is in sys.path so schema.py can be imported anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, RankedCompany

# Fallback for Colab Secret loading
if "GOOGLE_API_KEY" not in os.environ:
    try:
        from google.colab import userdata

        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
    except ImportError:
        pass

# ---- 1. Set up the LLM ----
llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)


# ---- 2. Define agent-specific structured output schema ----
class RankingAgentOutput(BaseModel):
    rankings: List[RankedCompany] = Field(
        description="List of companies ranked in descending order by final TAFGS score."
    )
    top_pick_summary: str = Field(
        description="2-3 sentence executive synthesis on why the top-ranked company leads the AI factory ecosystem."
    )
    methodology_notes: str = Field(
        description="Brief note on how moat, margin, growth, and risk haircut were synthesized."
    )


# ---- 3. Helper function to compute TAFGS deterministically ----
def calculate_tafgs(moat: int, margin: int, growth_cagr: float, discount_pct: float) -> float:
    """
    Computes TAFGS on a 0-100 base scale:
      - Moat Score (0-5): weighted at 30% -> (moat / 5) * 30
      - Margin Score (0-5): weighted at 30% -> (margin / 5) * 30
      - Growth CAGR (%): weighted at 40% -> capped at 100% CAGR max for 40 pts
      - Risk Discount: reduces total base score by discount_pct (e.g. 0.10 = 10% haircut)
    """
    moat_pts = (moat / 5.0) * 30.0
    margin_pts = (margin / 5.0) * 30.0
    growth_pts = min(growth_cagr / 100.0, 1.0) * 40.0

    raw_score = moat_pts + margin_pts + growth_pts
    final_score = raw_score * (1.0 - discount_pct)
    return round(final_score, 2)


# ---- 4. Node function ----
def ranking_node(state: dict) -> dict:
    # 1. Gather upstream data collections from state
    companies = state.get("companies", [])
    moat_list = state.get("moat_scores", [])
    margin_list = state.get("margin_scores", [])
    growth_list = state.get("growth_forecasts", [])
    risk_list = state.get("risk_adjustments", [])

    # Handle fallback for standalone single-company testing if state lists are empty
    if not companies and "company_name" in state:
        comp_name = state.get("company_name", "NVIDIA")
        companies = [{"name": comp_name, "ticker": "NVDA", "segment": "Compute/GPUs"}]

    # Convert list lookups into helper dictionaries by company name
    moat_map = {
        (m.get("company") if isinstance(m, dict) else getattr(m, "company", "")): (
            m.get("score", 3) if isinstance(m, dict) else getattr(m, "score", 3)
        )
        for m in moat_list
    }
    margin_map = {
        (m.get("company") if isinstance(m, dict) else getattr(m, "company", "")): (
            m.get("score", 3) if isinstance(m, dict) else getattr(m, "score", 3)
        )
        for m in margin_list
    }
    growth_map = {
        (g.get("company") if isinstance(g, dict) else getattr(g, "company", "")): (
            g.get("cagr_pct", 25.0) if isinstance(g, dict) else getattr(g, "cagr_pct", 25.0)
        )
        for g in growth_list
    }
    risk_map = {
        (r.get("company") if isinstance(r, dict) else getattr(r, "company", "")): (
            r.get("discount_pct", 0.0) if isinstance(r, dict) else getattr(r, "discount_pct", 0.0)
        )
        for r in risk_list
    }

    # 2. Build preliminary score objects programmatically
    computed_candidates = []
    for comp in companies:
        name = comp.get("name") if isinstance(comp, dict) else getattr(comp, "name", "Unknown")
        ticker = comp.get("ticker") if isinstance(comp, dict) else getattr(comp, "ticker", None)
        segment = comp.get("segment") if isinstance(comp, dict) else getattr(comp, "segment", "General AI")

        m_score = moat_map.get(name, 3)
        mg_score = margin_map.get(name, 3)
        g_cagr = growth_map.get(name, 20.0)
        r_discount = risk_map.get(name, 0.0)

        score = calculate_tafgs(m_score, mg_score, g_cagr, r_discount)

        computed_candidates.append(
            {
                "company": name,
                "ticker": ticker,
                "segment": segment,
                "moat_score": m_score,
                "margin_score": mg_score,
                "growth_cagr_pct": g_cagr,
                "tafgs_score": score,
            }
        )

    # Sort descending by calculated TAFGS score
    computed_candidates.sort(key=lambda x: x["tafgs_score"], reverse=True)

    # 3. Prompt Gemini to validate the leaderboard and draft executive commentary
    prompt = f"""
You are the Ranking Agent in an AI Factory Growth Equity pipeline.

Below is the calculated leaderboard based on the formula:
TAFGS = [(Moat/5 * 30) + (Margin/5 * 30) + (Min(CAGR, 100)/100 * 40)] * (1 - RiskDiscount)

Candidate Metrics:
{computed_candidates}

Instructions:
1. Verify the rank sequence (1 to N) for each company according to its tafgs_score.
2. Return the structured list of ranked companies matching the RankedCompany schema.
3. Provide a top pick executive summary and methodology note.
"""

    structured_llm = llm.with_structured_output(RankingAgentOutput)
    result: RankingAgentOutput = structured_llm.invoke(prompt)

    # Naming Guide Contract: populate assigned individual result key
    state["ranking_result"] = result.model_dump()

    # PipelineState Contract: populate master rankings list
    state["rankings"] = [r.model_dump() for r in result.rankings]

    return state


# ---- 5. Standalone execution graph ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("ranking", ranking_node)
    graph.set_entry_point("ranking")
    graph.add_edge("ranking", END)
    app = graph.compile()

    # Test standalone run with mock upstream state
    mock_state = {
        "companies": [
            {"name": "NVIDIA", "ticker": "NVDA", "segment": "Compute/GPUs"},
            {"name": "Arista Networks", "ticker": "ANET", "segment": "Networking"},
        ],
        "moat_scores": [
            {"company": "NVIDIA", "score": 5, "rationale": "CUDA monopoly"},
            {"company": "Arista Networks", "score": 4, "rationale": "EOS software lock-in"},
        ],
        "margin_scores": [
            {"company": "NVIDIA", "score": 5, "operating_margin_pct": 55.0, "source": "real"},
            {"company": "Arista Networks", "score": 4, "operating_margin_pct": 42.0, "source": "real"},
        ],
        "growth_forecasts": [
            {"company": "NVIDIA", "cagr_pct": 45.0, "rationale": "Data center compute demand"},
            {"company": "Arista Networks", "cagr_pct": 22.0, "rationale": "Ethernet switching growth"},
        ],
        "risk_adjustments": [
            {"company": "NVIDIA", "discount_pct": 0.05, "risk_notes": "Cyclicality risk"},
            {"company": "Arista Networks", "discount_pct": 0.0, "risk_notes": "Low execution risk"},
        ],
    }

    output = app.invoke(mock_state)
    print("\n--- Ranking Agent Output ---")
    print(output)
