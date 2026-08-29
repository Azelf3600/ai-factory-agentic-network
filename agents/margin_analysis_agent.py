"""MARGIN ANALYSIS AGENT"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, MarginScore

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)

class MarginBatchOutput(BaseModel):
    analysis: List[MarginScore]

def margin_analysis_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    comp_names = [c.get("name") if isinstance(c, dict) else getattr(c, "name", "Unknown") for c in companies]
    structured_llm = llm.with_structured_output(MarginBatchOutput, method="json_schema")

    prompt = f"Provide Operating Margin % and Margin Score (0-5) for: {', '.join(comp_names)}"

    try:
        result: MarginBatchOutput = structured_llm.invoke(prompt)
        return {"margin_scores": [m.model_dump() for m in result.analysis]}
    except Exception as e:
        print(f"[Margin Analysis Error]: {e}")
        fallback = [MarginScore(company=name, score=3, operating_margin_pct=25.0, source="estimated").model_dump() for name in comp_names]
        return {"margin_scores": fallback}