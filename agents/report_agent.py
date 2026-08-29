"""REPORT SYNTHESIS AGENT"""

import os
import sys
import json
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, PipelineState

def report_agent_node(state: PipelineState) -> dict:
    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.2,
    )
    rankings_json = json.dumps(state.get("rankings", []), indent=2)

    prompt = f"""
You are a Lead Growth Equity Analyst. Generate a comprehensive Executive AI Infrastructure Report based on this dataset of 20 evaluated companies:

{rankings_json}

Ensure your output contains these 3 distinct sections:

1. AI FACTORY VALUE-CHAIN MAPPING
- Breakdown of AI Factory capital expenditure across the 5 core layers: Compute/Servers (~60%), Networking (~15%), Power Infrastructure (~15%), Cooling Systems (~7%), and Engineering & Construction (~3%).

2. MASTER RANKINGS TABLE
- A clean markdown table of all 20 companies sorted by TAFGS Score containing: Rank, Company, Ticker, Segment, Moat Score, Op Margin %, CAGR %, and Final TAFGS Score.

3. TOP 20 COMPANY PROFILES
For EACH of the 20 companies, provide a structured profile including:
- Rank & TAFGS Score
- Primary AI Factory Role
- Moat & Differentiation Narrative
- Operating Margin Profile
- AI-Driven Growth Catalysts (2026–2029)
- Key Risks (e.g., customer concentration, cyclicality, supply chain bottleneck)
"""

    try:
        response = llm.invoke(prompt)
        
        if isinstance(response.content, str):
            report_text = response.content
        elif isinstance(response.content, list):
            report_text = "\n".join([
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in response.content
            ])
        else:
            report_text = str(response.content)

        return {"final_report": report_text}
    except Exception as e:
        print(f"[Report Synthesis Error]: {e}")
        return {"final_report": "Error generating report."}