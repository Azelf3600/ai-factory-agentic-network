"""
MARKET MAPPING AGENT
Maps AI Factory spend across infrastructure layers (e.g. compute/silicon,
networking/interconnect, power & cooling, storage, orchestration/software,
and services) for a given company.
"""

import os
from typing import List, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

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


# ---- 2. Define this agent's structured output shape ----
class InfrastructureLayerSpend(BaseModel):
    layer_name: str = Field(
        description="Infrastructure layer, e.g. Compute/Silicon, Networking, "
        "Power & Cooling, Storage, Orchestration/Software, Services"
    )
    estimated_share_pct: Optional[float] = Field(
        default=None,
        description="Estimated % of total AI Factory spend allocated to this "
        "layer, if inferable. Null if not enough info.",
    )
    rationale: str = Field(
        description="Brief justification for this layer's estimated spend/role"
    )
    key_vendors_or_partners: List[str] = Field(
        default_factory=list,
        description="Named vendors, suppliers, or internal products tied to "
        "this layer for this company",
    )


class MarketMappingOutput(BaseModel):
    company_received: str
    ai_factory_spend_summary: str = Field(
        description="1-2 sentence overview of how this company's AI infrastructure "
        "spend is distributed"
    )
    infrastructure_layers: List[InfrastructureLayerSpend] = Field(
        description="Breakdown of spend/positioning across AI infrastructure layers"
    )
    primary_layer_of_strength: str = Field(
        description="The single layer where this company has the strongest "
        "market position or spend concentration"
    )
    confidence_level: str = Field(
        description="One of: Low, Medium, High — confidence in this mapping "
        "given available public information"
    )


# ---- 3. The node function itself ----
MARKET_MAPPING_PROMPT = """You are an infrastructure market analyst specializing in AI Factory
(AI datacenter / AI infrastructure) spend analysis.

Evaluate the company below and map its AI Factory-related activity across these
infrastructure layers:
- Compute/Silicon (GPUs, ASICs, CPUs, accelerators)
- Networking/Interconnect (NICs, switches, optical, fabric)
- Power & Cooling (power delivery, liquid cooling, energy procurement)
- Storage (high-performance storage for AI workloads)
- Orchestration/Software (schedulers, MLOps, cluster management)
- Services (systems integration, colocation, consulting)

Company: {company_name}

For each layer that is materially relevant to this company (as a buyer, seller,
or both), provide an estimated share of AI Factory spend if you can reasonably
infer it, a rationale, and any named vendors or partners you're aware of. Omit
layers that are not materially relevant rather than forcing a number. Then
identify the company's primary layer of strength and state your overall
confidence level (Low, Medium, High) given the availability of public
information about this company.
"""


def market_mapping_node(state: dict) -> dict:
    company_name = state.get("company_name", "Unknown Company")

    structured_llm = llm.with_structured_output(MarketMappingOutput)
    result: MarketMappingOutput = structured_llm.invoke(
        MARKET_MAPPING_PROMPT.format(company_name=company_name)
    )

    state["market_mapping_result"] = result.model_dump()
    return state


# ---- 4. Minimal graph to run this node standalone ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("market_mapping", market_mapping_node)
    graph.set_entry_point("market_mapping")
    graph.add_edge("market_mapping", END)
    app = graph.compile()

    output = app.invoke({"company_name": "NVIDIA"})
    print("\n--- Market Mapping Agent Output ---")
    print(output)
