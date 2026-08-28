"""
COMPANY INGESTION AGENT

Function: Identifies eligible public companies globally for the AI Factory
Growth Equity pipeline, segment by segment (e.g. Networking, Compute/GPUs,
Cooling, Hyperscaler).

Reads `segments` from the shared PipelineState, asks Gemini to identify
publicly traded companies with real AI-infrastructure exposure in each
segment, validates the result against a structured Pydantic schema, and
writes:
  - state["company_ingestion_result"]  -> this agent's own raw output (dict)
  - state["companies"]                 -> the shared `companies` list every
                                           downstream agent reads from

Run this file directly to confirm your environment + API key work.
"""

import json
import os
import sys
from typing import List, Optional

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Ensure parent directory is in sys.path so schema.py can be imported anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL

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

# Fallback segment list, used only if the pipeline hasn't set `segments` yet.
DEFAULT_SEGMENTS = ["Networking", "Compute/GPUs", "Cooling", "Hyperscaler"]


# ---- 2. Define this agent's OWN structured output shape ----
class EligibleCompany(BaseModel):
    """A single publicly traded company judged eligible for the dataset."""
    name: str
    ticker: Optional[str] = Field(
        default=None, description="Primary exchange ticker symbol, e.g. NVDA"
    )
    exchange: Optional[str] = Field(
        default=None, description="Primary listing exchange, e.g. NASDAQ, NYSE, TSE, LSE"
    )
    country: Optional[str] = Field(
        default=None, description="Country of headquarters / primary listing"
    )
    segment: str = Field(
        description="AI infrastructure segment, e.g. Networking, Compute/GPUs, Cooling, Hyperscaler"
    )
    is_public: bool = True
    description: Optional[str] = Field(
        default=None, description="One-sentence description of what the company does"
    )
    eligibility_rationale: str = Field(
        description="Why this company qualifies: public listing + genuine AI-infrastructure revenue exposure"
    )


class CompanyIngestionOutput(BaseModel):
    """This agent's full structured output for a single segment."""
    segment: str
    companies: List[EligibleCompany]
    companies_identified: int
    notes: Optional[str] = Field(
        default=None, description="Any caveats, e.g. thin coverage or ambiguous segment fit"
    )


# ---- 3. The node function itself ----
def company_ingestion_node(state: dict) -> dict:
    segments = state.get("segments") or DEFAULT_SEGMENTS

    structured_llm = llm.with_structured_output(CompanyIngestionOutput)

    segment_results = []
    for segment in segments:
        prompt = (
            "You are the Company Ingestion Agent for an AI Factory Growth "
            "Equity research pipeline. Identify up to 8 PUBLICLY TRADED "
            f"companies, from ANY country/exchange globally, that have real, "
            f"material revenue exposure to the '{segment}' segment of AI "
            "infrastructure build-out (data centers, GPUs/accelerators, "
            "networking, power, cooling, or hyperscale cloud capex). "
            "Only include companies that are actually publicly listed today "
            "with a real ticker symbol — do not invent tickers, and do not "
            "include private companies or subsidiaries. For each company, "
            "give a one-sentence description and a short rationale for why "
            "it is eligible."
        )
        result: CompanyIngestionOutput = structured_llm.invoke(prompt)
        segment_results.append(result.model_dump())

    total_companies = sum(r["companies_identified"] for r in segment_results)

    # RESULT_KEY must be a dict (the test harness's Test 5 requires
    # isinstance(result_data, dict)), so wrap the per-segment list rather
    # than assigning it directly.
    state["company_ingestion_result"] = {
        "segments_processed": segments,
        "segment_results": segment_results,
        "total_companies_identified": total_companies,
    }

    # Also populate the shared `companies` field per the PipelineState
    # contract — this is the agent responsible for building that list.
    companies = state.get("companies", [])
    for seg_result in segment_results:
        for c in seg_result["companies"]:
            companies.append(
                {
                    "name": c["name"],
                    "ticker": c.get("ticker"),
                    "segment": c["segment"],
                    "is_public": c.get("is_public", True),
                    "description": c.get("description"),
                }
            )
    state["companies"] = companies

    return state


# ---- 4. Minimal graph to run this node standalone ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("company_ingestion", company_ingestion_node)
    graph.set_entry_point("company_ingestion")
    graph.add_edge("company_ingestion", END)
    app = graph.compile()

    output = app.invoke({"segments": ["Compute/GPUs", "Networking"]})
    print("\n--- Company Ingestion Agent Output ---")
    print(json.dumps(output["company_ingestion_result"], indent=2))
    print(f"\nTotal companies added to shared state: {len(output['companies'])}")