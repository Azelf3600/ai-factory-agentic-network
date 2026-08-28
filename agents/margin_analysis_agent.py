"""
MARGIN ANALYSIS AGENT

Function:
  Evaluates a company's financial operating margin profile, assigning a score
  from 0 to 5 based on industry margins.

Reads from shared state:
  - company_name (str)
  - companies (list, optional) — pulls segment/description context if available

Writes to shared state delta:
  - margin_analysis_result (dict)
  - margin_scores (list with single item for operator.add reducer)
"""

import os
import sys
from typing import List, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
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


# ---- 2. Define agent-specific structured output schema ----
class MarginAnalysisOutput(BaseModel):
    company_received: str
    operating_margin_pct: float = Field(
        description="Recent operating margin percentage (operating income / revenue * 100)."
    )
    source: Literal["real", "estimated"] = Field(
        description="Whether the margin figure comes from verified financial reports ('real') or model estimation ('estimated')."
    )
    margin_score: int = Field(
        ge=0,
        le=5,
        description=(
            "Margin Score from 0 to 5: "
            "0 = Negative margin (<0%), 1 = Low (0-10%), 2 = Moderate (10-20%), "
            "3 = Good (20-30%), 4 = High (30-40%), 5 = Exceptional (>40%)."
        ),
    )
    rationale: str = Field(
        description="1-2 sentence rationale justifying the score and operating margin profile."
    )


# ---- 3. Node function ----
def margin_analysis_node(state: dict) -> dict:
    company_name = state.get("company_name", "Unknown Company")

    segment = "AI Infrastructure"
    description = ""
    for c in state.get("companies", []):
        name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
        if name == company_name:
            segment = c.get("segment") if isinstance(c, dict) else getattr(c, "segment", segment)
            description = c.get("description") if isinstance(c, dict) else getattr(c, "description", "") or ""
            break

    prompt = f"""
You are the Margin Analysis Agent in an AI Factory Analysis research pipeline.

Evaluate the operating margin profile for the following company:

Company: {company_name}
Segment: {segment}
Description: {description or 'No additional description provided.'}

Instructions:
1. Identify or estimate the company's recent operating margin percentage (operating income / revenue * 100).
2. Indicate whether this number is 'real' (based on verified public financial filings) or 'estimated'.
3. Normalize the operating margin into a Margin Score from 0 to 5 using this benchmark scale:
   - 0: Negative operating margin (< 0%)
   - 1: Low operating margin (0% to 10%)
   - 2: Moderate operating margin (10% to 20%)
   - 3: Good operating margin (20% to 30%)
   - 4: High operating margin (30% to 40%)
   - 5: Exceptional operating margin (> 40%)
4. Provide a clear 1-2 sentence rationale tying the operating margin to the score.
"""

    structured_llm = llm.with_structured_output(MarginAnalysisOutput)
    result: MarginAnalysisOutput = structured_llm.invoke(prompt)

    margin_entry = {
        "company": result.company_received or company_name,
        "score": result.margin_score,
        "operating_margin_pct": result.operating_margin_pct,
        "source": result.source,
    }

    # Return state delta only to avoid parallel execution collisions
    return {
        "margin_analysis_result": result.model_dump(),
        "margin_scores": [margin_entry],
    }


# ---- 4. Standalone execution graph ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("margin_analysis", margin_analysis_node)
    graph.set_entry_point("margin_analysis")
    graph.add_edge("margin_analysis", END)
    app = graph.compile()

    output = app.invoke({"company_name": "NVIDIA"})
    print("\n--- Margin Analysis Agent Output ---")
    print(output)