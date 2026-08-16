"""
RISK ADJUSTMENT AGENT

Function:
  Applies execution, cyclicality, and concentration discounts to a
  company's valuation/attractiveness score, producing a structured
  risk-adjustment profile that downstream agents (e.g. a valuation
  or scoring agent) can consume.

Reads from shared state:
  - company_name (str)
  - business_summary (str, optional) — pass in output from an upstream
    "business analysis" agent if you have one; falls back to a generic
    prompt if not present.

Writes to shared state:
  - state["risk_adjustment_result"]
"""

import os
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List

# Fallback for Colab Secret loading
if "GOOGLE_API_KEY" not in os.environ:
    try:
        from google.colab import userdata
        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
    except ImportError:
        pass

# ---- 1. Set up the LLM ----
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",  # Active model for your API key
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)


# ---- 2. Define this agent's OWN structured output shape ----
class RiskAdjustmentOutput(BaseModel):
    company_received: str

    # --- Execution risk ---
    execution_risk_discount_pct: float = Field(
        description=(
            "Discount (0-100) applied for execution risk: management track "
            "record, delivery/operational risk, history of missed guidance "
            "or failed rollouts."
        )
    )
    execution_risk_notes: str = Field(
        description="1-2 sentence justification for the execution risk discount."
    )

    # --- Cyclicality risk ---
    cyclicality_risk_discount_pct: float = Field(
        description=(
            "Discount (0-100) applied for cyclicality risk: sensitivity to "
            "macro/industry cycles, demand volatility, capex cycle exposure."
        )
    )
    cyclicality_risk_notes: str = Field(
        description="1-2 sentence justification for the cyclicality risk discount."
    )

    # --- Concentration risk ---
    concentration_risk_discount_pct: float = Field(
        description=(
            "Discount (0-100) applied for concentration risk: customer "
            "concentration, supplier concentration, geographic or single "
            "product-line dependence."
        )
    )
    concentration_risk_notes: str = Field(
        description="1-2 sentence justification for the concentration risk discount."
    )

    # --- Aggregate ---
    total_risk_adjustment_pct: float = Field(
        description=(
            "Combined overall discount (0-100) to apply to the company's "
            "valuation/score, synthesizing all three risk factors "
            "(not necessarily a simple sum)."
        )
    )
    key_risk_factors: List[str] = Field(
        description="3-5 short bullet-style strings naming the biggest specific risk drivers."
    )
    confidence: str = Field(
        description="One of: Low, Medium, High — confidence in this risk assessment given available info."
    )


# ---- 3. The node function itself ----
def risk_adjustment_node(state: dict) -> dict:
    company_name = state.get("company_name", "Unknown Company")
    business_summary = state.get("business_summary", "")

    context_block = (
        f"Additional business context from prior analysis:\n{business_summary}\n"
        if business_summary
        else "No additional business context was provided; use general "
             "knowledge about this company and its industry.\n"
    )

    prompt = f"""
You are the Risk Adjustment Agent in an AI Factory Analysis pipeline.

Your job is to evaluate '{company_name}' and assign discount percentages
(0-100, where 0 = no discount/no risk and 100 = maximum discount/extreme risk)
across three risk categories:

1. EXECUTION RISK — management credibility, delivery track record,
   operational complexity, history of execution stumbles.
2. CYCLICALITY RISK — exposure to macroeconomic or industry cycles,
   demand volatility, capex/spending cycle sensitivity.
3. CONCENTRATION RISK — customer concentration, supplier concentration,
   geographic concentration, or reliance on a single product/segment.

{context_block}

For each category, provide a discount percentage and a short justification.
Then provide a total_risk_adjustment_pct that reflects your overall judgment
of combined risk (this should account for overlap between factors, not just
sum the three numbers). List the 3-5 most important specific risk drivers,
and state your confidence level (Low, Medium, High) given the information
available.
"""

    structured_llm = llm.with_structured_output(RiskAdjustmentOutput)
    result: RiskAdjustmentOutput = structured_llm.invoke(prompt)

    state["risk_adjustment_result"] = result.model_dump()
    return state


# ---- 4. Minimal graph to run this node standalone ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("risk_adjustment", risk_adjustment_node)
    graph.set_entry_point("risk_adjustment")
    graph.add_edge("risk_adjustment", END)
    app = graph.compile()

    output = app.invoke({"company_name": "NVIDIA"})
    print("\n--- Risk Adjustment Agent Output ---")
    print(output)