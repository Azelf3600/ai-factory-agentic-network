"""
Report Synthesis Agent for the AI Factory Growth Equity Pipeline.

Aggregates structured data across all pipeline stages into a comprehensive,
markdown-formatted investment report fulfilling key deliverable requirements.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import DEFAULT_MODEL, PipelineState

REPORT_PROMPT_TEMPLATE = """ You are a Principal Growth Equity Analyst synthesizing an executive investment report for an AI Infrastructure fund.

Synthesize the provided pipeline state into a clean Markdown report. Ensure ALL required sections and deliverables are fully represented.

### MANDATORY REPORT STRUCTURE:

# AI Infrastructure & Hardware Ecosystem: Top 20 Strategic Investment Opportunities

## 1. Executive Summary
Provide a high-level overview of current AI infrastructure spending patterns and investment priorities.

## 2. AI Factory Value-Chain Mapping
Provide estimated capital distribution weights (% share of AI Factory dollar spend) across the 5 core segments:
- **Compute / Servers (GPUs, AI servers)**: [Estimated % Spend]
- **Networking (Ethernet, InfiniBand, optical)**: [Estimated % Spend]
- **Power Infrastructure (generators, UPS, switchgear)**: [Estimated % Spend]
- **Cooling Systems (liquid cooling, chillers)**: [Estimated % Spend]
- **Engineering & Construction (design, commissioning)**: [Estimated % Spend]

*Rank Data Center Infrastructure priorities based on current capital allocation bottlenecks.*

## 3. Top 20 AI Factory Growth Ranking (Master Table)
Render a Markdown table containing ALL 20 entries ranked by Total AI Factory Growth Score (TAFGS).
| Rank | Company | Ticker | Primary Segment | AI Rev Exposure (%) | Moat Score (0-5) | Op Margin (%) | 3-Yr CAGR (%) | TAFGS Score |
|---|---|---|---|---|---|---|---|---|

## 4. Detailed Company Profiles (Top 20)
For EACH of the 20 ranked entries, construct a structured profile:

### Rank [Rank Number]: [Company Name] ([Ticker if public])
- **Primary AI Factory Role:** [Role/Segment]
- **Moat & Differentiation Narrative:** [Qualitative moat narrative]
- **Operating Margin Profile:** [Normalized operating margin % and source context]
- **AI-Driven Growth Catalysts (2026–2029):** [3-year growth drivers and forecast details]
- **Key Risks:** [Execution, customer concentration, cyclicality, supply risks]
- **Final Growth Score (TAFGS):** [Score]

## 5. Industry Risk Factors & Recommendation
Synthesize macroeconomic risk factors and outline explicit due-diligence recommendations.

---
Pipeline State Data:
Companies & Rankings Data:
{rankings_json}
Moat Scores:
{moats_json}
Margin Scores:
{margins_json}
Growth Forecasts:
{growth_json}
Risk Adjustments:
{risk_json}
"""


def report_agent_node(state: PipelineState) -> dict:
    """Agent node that formats pipeline metrics into the final executive report."""
    llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL, temperature=0.2)

    # Serialize internal pipeline state data for LLM context
    rankings_json = json.dumps(state.get("rankings", []), indent=2, default=str)
    moats_json = json.dumps(state.get("moat_scores", []), indent=2, default=str)
    margins_json = json.dumps(state.get("margin_scores", []), indent=2, default=str)
    growth_json = json.dumps(
        state.get("growth_forecasts", []), indent=2, default=str
    )
    risk_json = json.dumps(
        state.get("risk_adjustments", []), indent=2, default=str
    )

    prompt = REPORT_PROMPT_TEMPLATE.format(
        rankings_json=rankings_json,
        moats_json=moats_json,
        margins_json=margins_json,
        growth_json=growth_json,
        risk_json=risk_json,
    )

    response = llm.invoke(prompt)

    # Return updated state key dictionary expected by LangGraph
    return {"final_report": response.content}