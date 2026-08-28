"""
GROWTH FORECAST AGENT

Function:
  Evaluates a company's projected AI-driven revenue CAGR over the next 3 years.

Reads from shared state:
  - company_name (str)
  - companies (list, optional) — pulls segment/description context if available

Writes to shared state delta:
  - growth_forecast_result (dict)
  - growth_forecasts (list with single item for operator.add reducer)
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
class GrowthForecastOutput(BaseModel):
    company: str = Field(description="Name of the evaluated company.")
    cagr_pct: float = Field(description="Projected 3-year AI-driven revenue CAGR percentage.")
    key_growth_drivers: List[str] = Field(description="2-3 key drivers of growth.")
    key_risks: List[str] = Field(description="Material risks that could hinder growth.")
    rationale: str = Field(description="1-2 sentence concise summary of the forecast.")


# ---- 3. Node function ----
def growth_forecast_node(state: dict) -> dict:
    company_name = state.get("company_name", "Unknown Company")

    segment = "Unknown segment"
    description = ""
    for c in state.get("companies", []):
        name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
        if name == company_name:
            segment = c.get("segment") if isinstance(c, dict) else getattr(c, "segment", segment)
            description = c.get("description") if isinstance(c, dict) else getattr(c, "description", "") or ""
            break

    prompt = f"""
You are a growth-equity analyst specializing in AI infrastructure and AI-driven
business transformation. Evaluate the following company and project its
AI-driven revenue CAGR (compound annual growth rate) over the next 3 years.

Company: {company_name}
Segment: {segment}
Description: {description or "No additional description available."}

Instructions:
- Focus specifically on growth driven by AI demand (e.g. AI infrastructure
  buildout, AI-enabled product lines, AI-driven demand for the company's
  core offering) rather than the company's total revenue growth.
- Give a single point-estimate CAGR percentage (e.g. 22.5), not a range.
- Identify the 2-3 key drivers behind that number.
- Note any material risks that could cause the company to fall short of it.
- Be a disciplined analyst: avoid inflated, hype-driven numbers. If AI
  exposure is minimal or speculative, the CAGR should reflect that.
"""

    structured_llm = llm.with_structured_output(GrowthForecastOutput)
    result: GrowthForecastOutput = structured_llm.invoke(prompt)

    forecast_entry = {
        "company": result.company or company_name,
        "cagr_pct": result.cagr_pct,
        "rationale": result.rationale,
    }

    # Return state delta only to avoid parallel execution collisions
    return {
        "growth_forecast_result": result.model_dump(),
        "growth_forecasts": [forecast_entry],
    }


# ---- 4. Standalone execution graph ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("growth_forecast", growth_forecast_node)
    graph.set_entry_point("growth_forecast")
    graph.add_edge("growth_forecast", END)
    app = graph.compile()

    output = app.invoke({"company_name": "NVIDIA"})
    print("\n--- Growth Forecast Agent Output ---")
    print(output)