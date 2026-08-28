"""
REPORT AGENT — Generates an investor-ready Top 20 output.

1. Read candidate companies and analysis from state
2. Call Gemini with Pydantic structured output validation
3. Format output and write directly to state["final_report"]
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
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
    max_retries=5,
)


# ---- 2. Define structured output schema ----
class RankedCompany(BaseModel):
    rank: int = Field(description="Ranking position from 1 to 20.")
    company_name: str = Field(description="Name of the evaluated company.")
    core_innovation: str = Field(
        description="Key AI technology or competitive advantage."
    )
    market_potential: str = Field(
        description="TAM/SAM or growth trajectory potential."
    )
    investment_thesis: str = Field(description="Concise investment thesis.")
    risk_level: str = Field(
        description="Risk assessment rating (Low, Medium, High)."
    )
    key_metrics: Dict[str, str] = Field(description="Key company metrics.")


class Top20ReportSchema(BaseModel):
    report_title: str = Field(description="Title of the report.")
    executive_summary: str = Field(
        description="High-level synthesis of top companies."
    )
    macro_investment_trends: List[str] = Field(
        description="Key market trends."
    )
    top_20_companies: List[RankedCompany] = Field(
        description="List of top companies ranked."
    )
    key_risk_factors: List[str] = Field(description="Industry-wide risk factors.")
    recommendation_next_steps: str = Field(description="Actionable next steps.")


# ---- 3. The node function (report_node) ----
def report_node(state: dict) -> dict:
    """Produces investor-ready Top 20 output and updates state['final_report']."""
    # Pull data using the keys defined in PipelineState
    companies_data = state.get("companies", [])
    rankings_data = state.get("rankings", [])
    moat_scores = state.get("moat_scores", [])
    margin_scores = state.get("margin_scores", [])

    context_summary = {
        "rankings": rankings_data,
        "moats": moat_scores,
        "margins": margin_scores,
    }

    structured_llm = llm.with_structured_output(Top20ReportSchema)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a VC Investment Partner generating an investor-ready Top 20 report.",
            ),
            (
                "human",
                "Evaluate these companies and generate the report:\nCompanies: {companies}\nContext: {context}",
            ),
        ]
    )

    formatted_prompt = prompt.format_messages(
        companies=str(companies_data), context=str(context_summary)
    )

    time.sleep(2)  # Cooldown to avoid 429 rate limit errors

    validated_output: Top20ReportSchema = structured_llm.invoke(formatted_prompt)

    # Format structured schema object into clean text for state['final_report']
    report_dict = validated_output.model_dump()
    formatted_report_text = f"# {report_dict['report_title']}\n\n"
    formatted_report_text += f"## Executive Summary\n{report_dict['executive_summary']}\n\n"
    
    formatted_report_text += "## Macro Investment Trends\n"
    for trend in report_dict['macro_investment_trends']:
        formatted_report_text += f"- {trend}\n"
        
    formatted_report_text += "\n## Top Ranked Companies\n"
    for comp in report_dict['top_20_companies']:
        formatted_report_text += f"### Rank {comp['rank']}: {comp['company_name']}\n"
        formatted_report_text += f"- **Thesis:** {comp['investment_thesis']}\n"
        formatted_report_text += f"- **Core Innovation:** {comp['core_innovation']}\n"
        formatted_report_text += f"- **Market Potential:** {comp['market_potential']}\n"
        formatted_report_text += f"- **Risk Level:** {comp['risk_level']}\n\n"

    formatted_report_text += "## Industry Risk Factors\n"
    for risk in report_dict['key_risk_factors']:
        formatted_report_text += f"- {risk}\n"

    formatted_report_text += f"\n## Recommendation & Next Steps\n{report_dict['recommendation_next_steps']}\n"

    # Return key matching PipelineState schema
    return {
        "final_report": formatted_report_text
    }


# ---- 4. Minimal graph to run standalone ----
if __name__ == "__main__":
    print("--- Running report_agent.py (report_node) Standalone ---")

    graph = StateGraph(dict)
    graph.add_node("report_node", report_node)
    graph.set_entry_point("report_node")
    graph.add_edge("report_node", END)
    app = graph.compile()

    output = app.invoke(
        {
            "companies": [
                {"name": "NVIDIA", "ticker": "NVDA"},
                {"name": "AMD", "ticker": "AMD"},
            ],
            "rankings": [{"company": "NVIDIA", "rank": 1}],
        }
    )

    print("\n--- Output Saved to state['final_report'] ---")
    print(output.get("final_report"))