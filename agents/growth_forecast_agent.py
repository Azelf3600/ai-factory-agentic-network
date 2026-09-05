"""GROWTH FORECAST AGENT"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, GrowthForecast  # noqa: E402
from agents.utils import fill_missing_with_fallback  # noqa: E402

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)


class GrowthBatchOutput(BaseModel):
    analysis: List[GrowthForecast]


def _fallback_growth(company: dict) -> dict:
    return {
        "company": company.get("name", "Unknown"),
        "ticker": company.get("ticker"),
        "cagr_pct": 15.0,
        "rationale": "FALLBACK: no LLM result returned for this company — conservative industry-baseline estimate.",
    }


def growth_forecast_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    company_dicts = [c if isinstance(c, dict) else c.model_dump() for c in companies]
    company_lines = "\n".join(
        f"- {c.get('name')} (ticker: {c.get('ticker', 'unknown')})" for c in company_dicts
    )

    structured_llm = llm.with_structured_output(GrowthBatchOutput, method="json_schema")
    prompt = f"""
Project a 3-year AI-driven revenue CAGR (%) for each company below, based on
AI Factory capex exposure, backlog growth, and hyperscaler/sovereign AI
commitments. Give a single point-estimate, not a range. Be disciplined —
avoid hype-driven numbers; if AI exposure is minimal, the CAGR should reflect
that. Return each company's exact ticker symbol as given (do not invent or omit it).

Companies:
{company_lines}
"""

    try:
        result: GrowthBatchOutput = structured_llm.invoke(prompt)
        results = [g.model_dump() for g in result.analysis]
    except Exception as e:
        print(f"[Growth Forecast Error]: {e}")
        results = []

    results = fill_missing_with_fallback(company_dicts, results, _fallback_growth, agent_label="Growth Forecast Agent")
    return {"growth_forecasts": results}