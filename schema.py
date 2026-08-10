"""
Shared state schema for the AI Factory Growth Equity pipeline.

This is the contract every agent reads from and writes to.
Do NOT change field names/types without syncing the whole team first —
this is exactly the kind of change that breaks everyone else's agent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Company(BaseModel):
    """One company entry in the master dataset. Built by the Company Ingestion Agent."""
    name: str
    ticker: Optional[str] = None
    segment: str  # e.g. "Networking", "Compute/GPUs", "Cooling", "Hyperscaler"
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
    moat_score: int
    margin_score: int
    growth_cagr_pct: float
    tafgs_score: float


class PipelineState(BaseModel):
    """
    The full shared state object passed through the LangGraph pipeline.
    Each agent node reads what it needs and appends its own output —
    nobody should overwrite another agent's section of this object.
    """
    segments: List[str] = []
    companies: List[Company] = []
    moat_scores: List[MoatScore] = []
    margin_scores: List[MarginScore] = []
    growth_forecasts: List[GrowthForecast] = []
    risk_adjustments: List[RiskAdjustment] = []
    rankings: List[RankedCompany] = []
    final_report: str = ""
