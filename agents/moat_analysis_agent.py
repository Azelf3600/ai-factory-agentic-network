"""MOAT ANALYSIS AGENT"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, MoatScore  # noqa: E402
from agents.utils import fill_missing_with_fallback  # noqa: E402

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)


class MoatBatchOutput(BaseModel):
    analysis: List[MoatScore]


def _fallback_moat(company: dict) -> dict:
    return {
        "company": company.get("name", "Unknown"),
        "ticker": company.get("ticker"),
        "score": 3,
        "rationale": "FALLBACK: no LLM result returned for this company — treat as placeholder, not a real assessment.",
    }


def moat_analysis_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    company_dicts = [c if isinstance(c, dict) else c.model_dump() for c in companies]

    # Send name + ticker explicitly so the model can echo the ticker back —
    # tickers rarely get reworded the way company names do.
    company_lines = "\n".join(
        f"- {c.get('name')} (ticker: {c.get('ticker', 'unknown')})" for c in company_dicts
    )

    structured_llm = llm.with_structured_output(MoatBatchOutput, method="json_schema")
    prompt = f"""
Analyze economic moat (0-5 scale) for each of the following companies.
For EACH company, return its exact ticker symbol as given below (do not
invent or omit it) along with a short rationale (architectural lock-in,
ecosystem dominance, switching costs, or supply-chain bottleneck position).

Companies:
{company_lines}
"""

    try:
        result: MoatBatchOutput = structured_llm.invoke(prompt)
        results = [m.model_dump() for m in result.analysis]
    except Exception as e:
        print(f"[Moat Analysis Error]: {e}")
        results = []

    results = fill_missing_with_fallback(company_dicts, results, _fallback_moat, agent_label="Moat Analysis Agent")
    return {"moat_scores": results}