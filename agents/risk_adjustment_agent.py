"""RISK ADJUSTMENT AGENT"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, RiskAdjustment

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)

class RiskBatchOutput(BaseModel):
    analysis: List[RiskAdjustment]

def risk_adjustment_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    comp_names = [c.get("name") if isinstance(c, dict) else getattr(c, "name", "Unknown") for c in companies]
    structured_llm = llm.with_structured_output(RiskBatchOutput, method="json_schema")

    prompt = f"Provide risk discount_pct (0.0 to 0.2) and notes for: {', '.join(comp_names)}"

    try:
        result: RiskBatchOutput = structured_llm.invoke(prompt)
        return {"risk_adjustments": [r.model_dump() for r in result.analysis]}
    except Exception as e:
        print(f"[Risk Adjustment Error]: {e}")
        fallback = [RiskAdjustment(company=name, discount_pct=0.05, risk_notes="General cyclicality risk").model_dump() for name in comp_names]
        return {"risk_adjustments": fallback}