"""
Shared state schema for the AI Factory Growth Equity pipeline.
This is the contract every agent reads from and writes to.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, Union
from pydantic import BaseModel, Field

# Centralized default model name
DEFAULT_MODEL = "gemini-3.1-flash-lite"


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
    ] = "Compute / Servers"
    ai_revenue_exposure_pct: float = Field(
        default=50.0,
        description="Placeholder revenue-exposure figure set at ingestion. The real "
                     "figure now comes from the AI Revenue Exposure Agent — this field "
                     "is only used as a last-resort fallback if that agent has no "
                     "record at all for a company."
    )
    ai_revenue_exposure_source: Literal["placeholder", "estimated", "researched"] = Field(
        default="placeholder",
        description="Whether ai_revenue_exposure_pct is a real researched figure, an "
                     "LLM estimate, or just the ingestion-time default.",
    )
    is_public: bool = True
    description: Optional[str] = None


class MoatScore(BaseModel):
    """Output of the Moat Analysis Agent, one per company."""
    company: str
    ticker: Optional[str] = None
    score: int = Field(ge=0, le=5)
    rationale: str


class MarginScore(BaseModel):
    """Output of the Margin Analysis Agent, one per company."""
    company: str
    ticker: Optional[str] = None
    score: int = Field(ge=0, le=5)
    operating_margin_pct: float = 20.0
    source: Literal["real", "estimated"] = "estimated"


class GrowthForecast(BaseModel):
    """Output of the Growth Forecast Agent, one per company."""
    company: str
    ticker: Optional[str] = None
    cagr_pct: float
    rationale: str


class RiskAdjustment(BaseModel):
    """Output of the Risk Adjustment Agent, one per company."""
    company: str
    ticker: Optional[str] = None
    discount_pct: float = 0.05
    risk_notes: str


class AIRevenueExposure(BaseModel):
    """
    Output of the AI Revenue Exposure Agent, one per company.
    Fills the gap where every company previously carried a hardcoded 50.0
    placeholder from Company Ingestion with no actual research behind it.
    """
    company: str
    ticker: Optional[str] = None
    exposure_pct: float = Field(ge=0.0, le=100.0)
    rationale: str
    source: Literal["real", "estimated"] = "estimated"


class RankedCompany(BaseModel):
    """One row of the final output table. Built by the Ranking Agent."""
    rank: int
    company: str
    ticker: Optional[str] = None
    segment: str
    ai_revenue_exposure_pct: float
    ai_revenue_exposure_source: str = "placeholder"
    moat_score: int
    operating_margin_pct: float
    margin_score: int
    growth_cagr_pct: float
    risk_notes: Optional[str] = None
    tafgs_score: float  # Total AI Factory Growth Score
    unmatched: bool = False  # True if one or more agents fell back to defaults for this company


class PipelineState(TypedDict, total=False):
    """
    The full shared state object passed through the LangGraph pipeline.
    Uses TypedDict with operator.add reducers to safely handle concurrent
    appends during parallel agent execution without state update collisions.

    NOTE: every key a node returns MUST be declared here. LangGraph builds
    its state channels from this schema — a node returning a key that isn't
    declared gets silently dropped, not errored. (This previously happened
    to market_mapping_result.) Declare new state keys here the moment a
    node starts returning them.
    """
    company_name: str
    segments: List[str]
    market_mapping_result: Dict[str, Any]
    companies: List[Union[Company, dict, Any]]
    moat_scores: Annotated[List[Union[MoatScore, dict, Any]], operator.add]
    margin_scores: Annotated[List[Union[MarginScore, dict, Any]], operator.add]
    growth_forecasts: Annotated[List[Union[GrowthForecast, dict, Any]], operator.add]
    risk_adjustments: Annotated[List[Union[RiskAdjustment, dict, Any]], operator.add]
    ai_revenue_exposures: Annotated[List[Union[AIRevenueExposure, dict, Any]], operator.add]
    rankings: List[Union[RankedCompany, dict, Any]]
    final_report: str