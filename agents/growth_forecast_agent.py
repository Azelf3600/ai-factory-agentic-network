"""
GROWTH FORECAST AGENT
Projects AI-driven 3-year CAGR for a given company.

Reads:  state["companies"] (looks up the target company by name) and
        state["company_name"] (the company currently being evaluated)
Writes: appends a GrowthForecast to state["growth_forecasts"]
"""

import os
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional

# Fallback for Colab Secret loading
if "GOOGLE_API_KEY" not in os.environ:
    try:
        from google.colab import userdata
        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
    except ImportError:
        pass

# ---- 1. Set up the LLM ----
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",  # Active model for your API key
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)


# ---- 2. Define this agent's structured output shape ----
# Mirrors GrowthForecast in the shared PipelineState schema, plus a couple
# of intermediate fields the LLM can reason over before committing to a
# single cagr_pct number.
class GrowthForecastOutput(BaseModel):
    company: str
    cagr_pct: float = Field(
        description="Projected AI-driven revenue CAGR over the next 3 years, as a percentage (e.g. 24.5 for 24.5%)."
    )
    key_growth_drivers: str = Field(
        description="Short summary of the 2-3 main factors driving the AI-related growth projection."
    )
    key_risks_to_forecast: Optional[str] = Field(
        default=None,
        description="Short summary of factors that could cause the company to miss this CAGR."
    )
    rationale: str = Field(
        description="1-2 sentence justification tying the CAGR number to the growth drivers."
    )


# ---- 3. The node function itself ----
def growth_forecast_node(state: dict) -> dict:
    company_name = state.get("company_name", "Unknown Company")

    # Pull extra context about the company if it's already in the shared
    # companies list (populated upstream by the Company Ingestion Agent).
    segment = "Unknown segment"
    description = ""
    for c in state.get("companies", []):
        # companies may be dicts (from .model_dump()) or Company objects
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

    # Append (never overwrite) to the shared growth_forecasts field,
    # matching the GrowthForecast shape in PipelineState.
    forecast_entry = {
        "company": result.company or company_name,
        "cagr_pct": result.cagr_pct,
        "rationale": result.rationale,
    }

    state.setdefault("growth_forecasts", [])
    state["growth_forecasts"].append(forecast_entry)

    # Keep the full structured output around too, in case downstream
    # agents or the final report want the extra drivers/risks detail.
    state["growth_forecast_result"] = result.model_dump()

    return state


# ---- 4. Minimal graph to run this node standalone ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("growth_forecast", growth_forecast_node)
    graph.set_entry_point("growth_forecast")
    graph.add_edge("growth_forecast", END)
    app = graph.compile()

    test_state = {
        "company_name": "NVIDIA",
        "companies": [
            {
                "name": "NVIDIA",
                "ticker": "NVDA",
                "segment": "Compute/GPUs",
                "is_public": True,
                "description": "Designs GPUs and AI accelerator hardware/software powering AI training and inference.",
            }
        ],
    }

    output = app.invoke(test_state)
    print("\n--- Growth Forecast Agent Output ---")
    print(output)