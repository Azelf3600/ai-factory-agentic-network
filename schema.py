"""
Shared state schema for the AI Factory Growth Equity pipeline.

This is the contract every agent reads from and writes to.
Do NOT change field names/types without syncing the whole team first —
this is exactly the kind of change that breaks everyone else's agent.
"""

import operator
from typing import Annotated, Any, List, Literal, Optional, TypedDict, Union
from pydantic import BaseModel, Field

# ==============================================================================
# CENTRALIZED PIPELINE CONFIGURATION
# ==============================================================================
# Update this single constant to change the model across all agent nodes.
DEFAULT_MODEL = "gemini-3.1-flash-lite"


# ==============================================================================
# SHARED PIPELINE SCHEMAS
# ==============================================================================
class Company(BaseModel):
    """One company entry in the master dataset. Built by the Company Ingestion Agent."""
    name: str
    ticker: Optional[str] = None
    segment: Literal[
        "Compute / Servers",
        "Networking",
        "Power Infrastructure",
        "Cooling Systems",
        "Engineering & Construction"
    ]
    ai_revenue_exposure_pct: float = Field(
        default=0.0, 
        description="Revenue exposure to AI Factory builds (% of total revenue, e.g., 65.0 for 65%)"
    )
    is_public: bool = True
    description: Optional[str] = None


class MoatScore(BaseModel):
    """Output of the Moat Analysis Agent, one per company."""
    company: str
    score: int = Field(ge=0, le=5)
    rationale: str


class MarginScore(BaseModel):
    """Output of the Margin Analysis Agent, one per company."""
    company: str
    score: int = Field(ge=0, le=5)
    operating_margin_pct: Optional[float] = None
    source: Literal["real", "estimated"] = "estimated"


class GrowthForecast(BaseModel):
    """Output of the Growth Forecast Agent, one per company."""
    company: str
    cagr_pct: float
    rationale: str


class RiskAdjustment(BaseModel):
    """Output of the Risk Adjustment Agent, one per company."""
    company: str
    discount_pct: float = 0.0  # applied to final TAFGS, e.g. 0.10 = 10% haircut
    risk_notes: str


class RankedCompany(BaseModel):
    """One row of the final Top 20 output. Built by the Ranking Agent."""
    rank: int
    company: str
    ticker: Optional[str] = None
    segment: str
    ai_revenue_exposure_pct: float
    moat_score: int
    operating_margin_pct: float
    growth_cagr_pct: float
    risk_notes: Optional[str] = None
    tafgs_score: float  # Total AI Factory Growth Score


class PipelineState(TypedDict, total=False):
    """
    The full shared state object passed through the LangGraph pipeline.
    Uses TypedDict with operator.add reducers to safely handle concurrent
    appends during parallel agent execution without state update collisions.
    """
    company_name: str
    segments: List[str]
    companies: List[Union[Company, dict, Any]]
    moat_scores: Annotated[List[Union[MoatScore, dict, Any]], operator.add]
    margin_scores: Annotated[List[Union[MarginScore, dict, Any]], operator.add]
    growth_forecasts: Annotated[List[Union[GrowthForecast, dict, Any]], operator.add]
    risk_adjustments: Annotated[List[Union[RiskAdjustment, dict, Any]], operator.add]
    rankings: List[Union[RankedCompany, dict, Any]]
    final_report: str