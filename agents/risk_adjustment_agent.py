"""RISK ADJUSTMENT AGENT"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, RiskAdjustment  # noqa: E402
from agents.utils import fill_missing_with_fallback  # noqa: E402

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)


class RiskBatchOutput(BaseModel):
    analysis: List[RiskAdjustment]


def _fallback_risk(company: dict) -> dict:
    return {
        "company": company.get("name", "Unknown"),
        "ticker": company.get("ticker"),
        "discount_pct": 0.1,
        "risk_notes": "FALLBACK: no LLM result returned for this company — conservative default discount applied.",
    }


def risk_adjustment_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    company_dicts = [c if isinstance(c, dict) else c.model_dump() for c in companies]
    company_lines = "\n".join(
        f"- {c.get('name')} (ticker: {c.get('ticker', 'unknown')})" for c in company_dicts
    )

    structured_llm = llm.with_structured_output(RiskBatchOutput, method="json_schema")
    prompt = f"""
For each company below, provide a risk discount (discount_pct, 0.0 to 0.2)
reflecting execution risk, cyclicality, and customer concentration, plus a
short note explaining the main risk factor. Return each company's exact
ticker symbol as given (do not invent or omit it).

Companies:
{company_lines}
"""

    try:
        result: RiskBatchOutput = structured_llm.invoke(prompt)
        results = [r.model_dump() for r in result.analysis]
    except Exception as e:
        print(f"[Risk Adjustment Error]: {e}")
        results = []

    results = fill_missing_with_fallback(company_dicts, results, _fallback_risk, agent_label="Risk Adjustment Agent")
    return {"risk_adjustments": results}