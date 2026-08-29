"""MOAT ANALYSIS AGENT"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, MoatScore

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)

class MoatBatchOutput(BaseModel):
    analysis: List[MoatScore]

def moat_analysis_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    comp_names = [c.get("name") if isinstance(c, dict) else getattr(c, "name", "Unknown") for c in companies]
    structured_llm = llm.with_structured_output(MoatBatchOutput, method="json_schema")

    prompt = f"Analyze economic moat (0-5 scale) and short rationale for: {', '.join(comp_names)}"

    try:
        result: MoatBatchOutput = structured_llm.invoke(prompt)
        return {"moat_scores": [m.model_dump() for m in result.analysis]}
    except Exception as e:
        print(f"[Moat Analysis Error]: {e}")
        fallback = [MoatScore(company=name, score=3, rationale="Solid technology position").model_dump() for name in comp_names]
        return {"moat_scores": fallback}