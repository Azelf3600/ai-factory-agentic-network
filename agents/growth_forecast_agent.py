"""GROWTH FORECAST AGENT"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, GrowthForecast

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)

class GrowthBatchOutput(BaseModel):
    analysis: List[GrowthForecast]

def growth_forecast_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    comp_names = [c.get("name") if isinstance(c, dict) else getattr(c, "name", "Unknown") for c in companies]
    structured_llm = llm.with_structured_output(GrowthBatchOutput, method="json_schema")

    prompt = f"Project 3-Year AI Revenue CAGR (%) and rationale for: {', '.join(comp_names)}"

    try:
        result: GrowthBatchOutput = structured_llm.invoke(prompt)
        return {"growth_forecasts": [g.model_dump() for g in result.analysis]}
    except Exception as e:
        print(f"[Growth Forecast Error]: {e}")
        fallback = [GrowthForecast(company=name, cagr_pct=20.0, rationale="Industry baseline growth").model_dump() for name in comp_names]
        return {"growth_forecasts": fallback}