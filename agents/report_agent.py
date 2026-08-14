"""
REPORT AGENT — Generates an investor-ready Top 20 output.

1. Read candidate companies from state
2. Call Gemini with Pydantic structured output validation
3. Save structured report directly into state["report_result"]
"""

import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Fallback for Colab Secret loading
if "GOOGLE_API_KEY" not in os.environ:
    try:
        from google.colab import userdata
        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
    except ImportError:
        pass

# ---- 1. Set up the LLM ----
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Active stable model for your API key
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.2,
    max_retries=5
)


# ---- 2. Define structured output schema ----
class RankedCompany(BaseModel):
    rank: int = Field(description="Ranking position from 1 to 20.")
    company_name: str = Field(description="Name of the evaluated company.")
    core_innovation: str = Field(description="Key AI technology or competitive advantage.")
    market_potential: str = Field(description="TAM/SAM or growth trajectory potential.")
    investment_thesis: str = Field(description="Concise investment thesis.")
    risk_level: str = Field(description="Risk assessment rating (Low, Medium, High).")
    key_metrics: Dict[str, str] = Field(description="Key company metrics.")

class Top20ReportSchema(BaseModel):
    report_title: str = Field(description="Title of the report.")
    executive_summary: str = Field(description="High-level synthesis of top companies.")
    macro_investment_trends: List[str] = Field(description="Key market trends.")
    top_20_companies: List[RankedCompany] = Field(description="List of top companies ranked.")
    key_risk_factors: List[str] = Field(description="Industry-wide risk factors.")
    recommendation_next_steps: str = Field(description="Actionable next steps.")


# ---- 3. The node function (report_node) ----
def report_node(state: dict) -> dict:
    """
    Produces investor-ready Top 20 output and updates state['report_result'].
    """
    companies_data = state.get("candidate_companies", [])
    analysis_context = state.get("analysis_data", {})

    structured_llm = llm.with_structured_output(Top20ReportSchema)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a VC Investment Partner generating an investor-ready Top 20 report."),
        ("human", "Evaluate these companies and generate the report:\n{companies}\nContext: {context}")
    ])

    formatted_prompt = prompt.format_messages(
        companies=str(companies_data),
        context=str(analysis_context)
    )

    time.sleep(2)  # Cooldown to avoid 429 errors

    validated_output: Top20ReportSchema = structured_llm.invoke(formatted_prompt)

    # Save to exact required key: state["report_result"]
    state["report_result"] = validated_output.model_dump()
    return state


# ---- 4. Minimal graph to run standalone ----
if __name__ == "__main__":
    print("--- Running report_agent.py (report_node) Standalone ---")

    graph = StateGraph(dict)
    graph.add_node("report_node", report_node)
    graph.set_entry_point("report_node")
    graph.add_edge("report_node", END)
    app = graph.compile()

    output = app.invoke({
        "candidate_companies": [
            {"name": "NVIDIA", "ticker": "NVDA"},
            {"name": "AMD", "ticker": "AMD"}
        ]
    })

    print("\n--- Output Saved to state['report_result'] ---")
    print("Report Title:", output.get("report_result", {}).get("report_title"))