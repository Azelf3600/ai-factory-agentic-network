"""
RISK ADJUSTMENT AGENT (Assigned: Salengga)

Function:
  Applies execution, cyclicality, and concentration discounts to a
  company's valuation/attractiveness score, producing a structured
  risk-adjustment profile.

Reads from shared state:
  - company_name (str)
  - business_summary (str, optional)

Writes to shared state delta:
  - risk_adjustment_result (dict)
  - risk_adjustments (list with single item for operator.add reducer)
"""

import os
import sys
from typing import List

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


# ---- 2. Define agent-specific output schema ----
class RiskAdjustmentOutput(BaseModel):
    company_received: str

    execution_risk_discount_pct: float = Field(
        description="Discount (0-100) applied for execution risk."
    )
    execution_risk_notes: str = Field(
        description="1-2 sentence justification for execution risk."
    )

    cyclicality_risk_discount_pct: float = Field(
        description="Discount (0-100) applied for cyclicality risk."
    )
    cyclicality_risk_notes: str = Field(
        description="1-2 sentence justification for cyclicality risk."
    )

    concentration_risk_discount_pct: float = Field(
        description="Discount (0-100) applied for concentration risk."
    )
    concentration_risk_notes: str = Field(
        description="1-2 sentence justification for concentration risk."
    )

    total_risk_adjustment_pct: float = Field(
        description="Combined overall discount (0-100) to apply to the company's valuation/score."
    )
    key_risk_factors: List[str] = Field(
        description="3-5 short bullet-style strings naming the biggest specific risk drivers."
    )
    confidence: str = Field(
        description="One of: Low, Medium, High."
    )


# ---- 3. Node function ----
def risk_adjustment_node(state: dict) -> dict:
    company_name = state.get("company_name", "Unknown Company")
    business_summary = state.get("business_summary", "")

    context_block = (
        f"Additional business context from prior analysis:\n{business_summary}\n"
        if business_summary
        else "No additional business context was provided; use general knowledge about this company and its industry.\n"
    )

    prompt = f"""
You are the Risk Adjustment Agent in an AI Factory Analysis pipeline.

Evaluate '{company_name}' and assign discount percentages (0-100) across:
1. EXECUTION RISK — management credibility, delivery track record, operational complexity.
2. CYCLICALITY RISK — exposure to macro/industry cycles, capex sensitivity.
3. CONCENTRATION RISK — customer, supplier, or product line dependence.

{context_block}

Provide discount percentages, short justifications, a total combined total_risk_adjustment_pct,
key risk factors, and confidence level.
"""

    structured_llm = llm.with_structured_output(RiskAdjustmentOutput)
    result: RiskAdjustmentOutput = structured_llm.invoke(prompt)

    risk_entry = {
        "company": result.company_received or company_name,
        "discount_pct": result.total_risk_adjustment_pct / 100.0,  # e.g., 10.0 -> 0.10 haircut
        "risk_notes": (
            f"Execution: {result.execution_risk_notes} | "
            f"Cyclicality: {result.cyclicality_risk_notes} | "
            f"Concentration: {result.concentration_risk_notes}"
        ),
    }

    # Return state delta only to avoid parallel execution collisions
    return {
        "risk_adjustment_result": result.model_dump(),
        "risk_adjustments": [risk_entry],
    }


# ---- 4. Standalone execution graph ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("risk_adjustment", risk_adjustment_node)
    graph.set_entry_point("risk_adjustment")
    graph.add_edge("risk_adjustment", END)
    app = graph.compile()

    output = app.invoke({"company_name": "NVIDIA"})
    print("\n--- Risk Adjustment Agent Output ---")
    print(output)