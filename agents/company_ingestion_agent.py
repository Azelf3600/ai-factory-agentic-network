"""COMPANY INGESTION AGENT"""

import os
import sys
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)

DEFAULT_SEGMENTS = [
    "Compute / Servers",
    "Networking",
    "Power Infrastructure",
    "Cooling Systems",
    "Engineering & Construction"
]

class EligibleCompany(BaseModel):
    name: str
    ticker: Optional[str] = None
    segment: str
    is_public: bool = True
    description: Optional[str] = None

class CompanyIngestionOutput(BaseModel):
    segment: str
    companies: List[EligibleCompany]

def company_ingestion_node(state: dict) -> dict:
    segments = state.get("segments") or DEFAULT_SEGMENTS
    structured_llm = llm.with_structured_output(CompanyIngestionOutput, method="json_schema")

    companies_list = []
    seen_names = set()

    for segment in segments:
        prompt = (
            f"Identify 4 distinct PUBLICLY TRADED companies operating in the '{segment}' "
            f"AI data center infrastructure layer. Include ticker symbols."
        )
        try:
            result: CompanyIngestionOutput = structured_llm.invoke(prompt)
            for c in result.companies:
                if c.name not in seen_names:
                    seen_names.add(c.name)
                    companies_list.append({
                        "name": c.name,
                        "ticker": c.ticker or "N/A",
                        "segment": segment,
                        "is_public": c.is_public,
                        "description": c.description or "",
                        "ai_revenue_exposure_pct": 50.0
                    })
        except Exception as e:
            print(f"[Company Ingestion Error] Segment {segment}: {e}")

    # Fallback to guarantee exactly 20 companies across the 5 specific value-chain layers
    if len(companies_list) < 20:
        fallback_companies = [
            # Compute / Servers (2026-2029 Leaders)
            {"name": "NVIDIA", "ticker": "NVDA", "segment": "Compute / Servers", "ai_revenue_exposure_pct": 85.0},
            {"name": "Advanced Micro Devices", "ticker": "AMD", "segment": "Compute / Servers", "ai_revenue_exposure_pct": 55.0},
            {"name": "Super Micro Computer", "ticker": "SMCI", "segment": "Compute / Servers", "ai_revenue_exposure_pct": 75.0},
            {"name": "Dell Technologies", "ticker": "DELL", "segment": "Compute / Servers", "ai_revenue_exposure_pct": 35.0},

            # Networking
            {"name": "Broadcom", "ticker": "AVGO", "segment": "Networking", "ai_revenue_exposure_pct": 60.0},
            {"name": "Arista Networks", "ticker": "ANET", "segment": "Networking", "ai_revenue_exposure_pct": 50.0},
            {"name": "Marvell Technology", "ticker": "MRVL", "segment": "Networking", "ai_revenue_exposure_pct": 55.0},
            {"name": "Cisco Systems", "ticker": "CSCO", "segment": "Networking", "ai_revenue_exposure_pct": 20.0},

            # Power Infrastructure
            {"name": "Eaton Corporation", "ticker": "ETN", "segment": "Power Infrastructure", "ai_revenue_exposure_pct": 40.0},
            {"name": "GE Vernova", "ticker": "GEV", "segment": "Power Infrastructure", "ai_revenue_exposure_pct": 35.0},
            {"name": "Schneider Electric", "ticker": "SBGSY", "segment": "Power Infrastructure", "ai_revenue_exposure_pct": 30.0},
            {"name": "Cummins", "ticker": "CMI", "segment": "Power Infrastructure", "ai_revenue_exposure_pct": 25.0},

            # Cooling Systems
            {"name": "Vertiv Holdings", "ticker": "VRT", "segment": "Cooling Systems", "ai_revenue_exposure_pct": 70.0},
            {"name": "Modine Manufacturing", "ticker": "MOD", "segment": "Cooling Systems", "ai_revenue_exposure_pct": 45.0},
            {"name": "nVent Electric", "ticker": "NVT", "segment": "Cooling Systems", "ai_revenue_exposure_pct": 40.0},
            {"name": "Trane Technologies", "ticker": "TT", "segment": "Cooling Systems", "ai_revenue_exposure_pct": 25.0},

            # Engineering & Construction
            {"name": "Quanta Services", "ticker": "PWR", "segment": "Engineering & Construction", "ai_revenue_exposure_pct": 30.0},
            {"name": "EMCOR Group", "ticker": "EME", "segment": "Engineering & Construction", "ai_revenue_exposure_pct": 35.0},
            {"name": "AECOM", "ticker": "ACM", "segment": "Engineering & Construction", "ai_revenue_exposure_pct": 25.0},
            {"name": "Sterling Infrastructure", "ticker": "STRL", "segment": "Engineering & Construction", "ai_revenue_exposure_pct": 40.0}
        ]
        
        for fb in fallback_companies:
            if fb["name"] not in seen_names:
                seen_names.add(fb["name"])
                companies_list.append(fb)

    return {"companies": companies_list[:20]}