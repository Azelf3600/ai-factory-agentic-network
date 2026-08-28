"""
MOAT ANALYSIS AGENT (Assigned: Dullo)

Function:
  Evaluates a company's competitive moat depth, technological differentiation,
  and ecosystem lock-in within the AI infrastructure ecosystem, assigning a score
  from 0 (no moat) to 5 (impenetrable ecosystem lock-in).

Reads from shared state:
  - company_name (str)
  - companies (list, optional) — pulls segment/description context if available

Writes to shared state delta:
  - moat_analysis_result (dict)
  - moat_scores (list with single item for operator.add reducer)
"""

import os
import sys
from typing import List, Optional

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
class MoatAnalysisOutput(BaseModel):
    company_received: str
    moat_score: int = Field(
        ge=0,
        le=5,
        description=(
            "Competitive moat score from 0 to 5: "
            "0 = Commodity/No Moat, 1 = Weak, 2 = Moderate, 3 = Strong, "
            "4 = Very Strong, 5 = Impenetrable Ecosystem Lock-in/Monopoly."
        ),
    )
    differentiation_factors: List[str] = Field(
        description="2-4 key drivers of product or technological differentiation (e.g., proprietary architecture, patents, scale)."
    )
    ecosystem_lock_in: str = Field(
        description="1-2 sentences detailing software lock-in, switching costs, developer mindset, or network effects."
    )
    main_threats_to_moat: List[str] = Field(
        description="2-3 potential threats or emerging open-source/competitor alternatives that could erode this moat."
    )
    rationale: str = Field(
        description="1-2 sentence overall justification tying the moat score to the company's competitive position."
    )


# ---- 3. Node function ----
def moat_analysis_node(state: dict) -> dict:
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
You are the Moat Analysis Agent in an AI Factory Analysis research pipeline.

Evaluate the competitive moat depth and ecosystem lock-in for the following company:

Company: {company_name}
Segment: {segment}
Description: {description or 'No additional description provided.'}

Instructions:
1. Assess product differentiation, proprietary technology/IP, supply chain dominance, and network effects.
2. Evaluate ecosystem lock-in (e.g., proprietary software stacks, switching costs, developer community standards).
3. Assign an integer Moat Score from 0 to 5:
   - 0: Pure commodity with no pricing power or differentiation.
   - 1: Minimal differentiation; easily replicable by competitors.
   - 2: Moderate moat; decent brand or spec advantage, but faces strong substitutes.
   - 3: Strong moat; proprietary technology with high customer switching costs.
   - 4: Very strong moat; clear market leadership, high margins, and strong ecosystem lock-in.
   - 5: Impenetrable moat; massive network effects, platform monopoly, or insurmountable IP/scale advantage.
4. Identify main differentiation drivers, ecosystem lock-in dynamics, key threats, and provide a clear rationale.
"""

    structured_llm = llm.with_structured_output(MoatAnalysisOutput)
    result: MoatAnalysisOutput = structured_llm.invoke(prompt)

    moat_entry = {
        "company": result.company_received or company_name,
        "score": result.moat_score,
        "rationale": result.rationale,
    }

    # Return state delta only to avoid parallel execution collisions
    return {
        "moat_analysis_result": result.model_dump(),
        "moat_scores": [moat_entry],
    }


# ---- 4. Standalone execution graph ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("moat_analysis", moat_analysis_node)
    graph.set_entry_point("moat_analysis")
    graph.add_edge("moat_analysis", END)
    app = graph.compile()

    output = app.invoke({"company_name": "NVIDIA"})
    print("\n--- Moat Analysis Agent Output ---")
    print(output)