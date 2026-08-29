"""MARKET MAPPING AGENT"""

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

class InfrastructureLayerSpend(BaseModel):
    layer_name: str
    estimated_share_pct: Optional[float] = None
    rationale: str
    key_vendors_or_partners: List[str] = []

class MarketMappingOutput(BaseModel):
    company_received: str
    ai_factory_spend_summary: str
    infrastructure_layers: List[InfrastructureLayerSpend]
    primary_layer_of_strength: str
    confidence_level: str

def market_mapping_node(state: dict) -> dict:
    company_name = state.get("company_name", "NVIDIA")
    structured_llm = llm.with_structured_output(MarketMappingOutput, method="json_schema")
    prompt = f"Evaluate {company_name} across AI infrastructure spend layers (Compute, Networking, Power, Cooling)."

    try:
        result: MarketMappingOutput = structured_llm.invoke(prompt)
        return {"market_mapping_result": result.model_dump()}
    except Exception as e:
        print(f"[Market Mapping Error]: {e}")
        return {
            "market_mapping_result": {
                "company_received": company_name,
                "ai_factory_spend_summary": "Standard mapping overview.",
                "infrastructure_layers": [],
                "primary_layer_of_strength": "Compute",
                "confidence_level": "Medium"
            }
        }