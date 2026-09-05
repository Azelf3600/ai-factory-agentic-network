"""
AI REVENUE EXPOSURE AGENT

Previously every company carried a hardcoded 50.0 ai_revenue_exposure_pct
placeholder set at ingestion time — Nvidia and Cisco got the same number,
with zero research behind either. This agent actually estimates the figure
per company, following the same batch-LLM + ticker-matched-fallback pattern
as Moat/Margin/Growth/Risk.

Same caveat as Margin analysis: LLM recall of a specific % for less-covered
companies is a genuine estimate, not a verified fact — "source" tracks that
distinction the same way Margin's does.
"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, AIRevenueExposure  # noqa: E402
from agents.utils import fill_missing_with_fallback  # noqa: E402
from agents.batch_utils import chunk, BATCH_SIZE  # noqa: E402

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)


class ExposureBatchOutput(BaseModel):
    analysis: List[AIRevenueExposure]


def _fallback_exposure(company: dict) -> dict:
    return {
        "company": company.get("name", "Unknown"),
        "ticker": company.get("ticker"),
        "exposure_pct": 50.0,
        "rationale": "FALLBACK: no LLM result returned for this company — treat as placeholder, not a real assessment.",
        "source": "estimated",
    }


def ai_revenue_exposure_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    company_dicts = [c if isinstance(c, dict) else c.model_dump() for c in companies]

    structured_llm = llm.with_structured_output(ExposureBatchOutput, method="json_schema")
    all_results: List[dict] = []

    for batch in chunk(company_dicts, BATCH_SIZE):
        company_lines = "\n".join(
            f"- {c.get('name')} (ticker: {c.get('ticker', 'unknown')}, segment: {c.get('segment', 'unknown')})"
            for c in batch
        )

        prompt = f"""
For each company below, estimate the percentage of its TOTAL revenue that
comes specifically from AI Factory / hyperscale data center builds (compute,
networking, power, cooling, or construction tied to AI infrastructure) —
not overall tech revenue, and not company-wide growth. A pure-play AI chip
company might be 80-100%; a large diversified industrial conglomerate where
data centers are one small end-market might be 5-15%. Be disciplined and
specific per company rather than defaulting to a round number for everyone.
Mark source as "real" only if this is based on a disclosed segment
breakdown you're confident about; otherwise "estimated". Return each
company's exact ticker as given (do not invent or omit it).

Companies:
{company_lines}
"""
        try:
            result: ExposureBatchOutput = structured_llm.invoke(prompt)
            batch_results = [r.model_dump() for r in result.analysis]
            if len(batch_results) != len(batch):
                print(f"[AI Revenue Exposure WARNING] batch of {len(batch)} companies returned "
                      f"{len(batch_results)} results — some may be missing and will fall back to default.")
            all_results.extend(batch_results)
        except Exception as e:
            print(f"[AI Revenue Exposure Error] batch starting with {batch[0].get('name')}: {e}")

    results = fill_missing_with_fallback(company_dicts, all_results, _fallback_exposure, agent_label="AI Revenue Exposure Agent")
    return {"ai_revenue_exposures": results}