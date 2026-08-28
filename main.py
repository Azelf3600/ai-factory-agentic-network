"""
ASYNC MAIN PIPELINE ORCHESTRATOR
"""

import os
import sys
from langgraph.graph import StateGraph, END

# Import centralized configuration and schema
from schema import DEFAULT_MODEL, PipelineState

# Import all 8 agent node functions
from agents.market_mapping_agent import market_mapping_node
from agents.company_ingestion_agent import company_ingestion_node
from agents.moat_analysis_agent import moat_analysis_node
from agents.margin_analysis_agent import margin_analysis_node
from agents.growth_forecast_agent import growth_forecast_node
from agents.risk_adjustment_agent import risk_adjustment_node
from agents.ranking_agent import ranking_node
from agents.report_agent import report_node


def build_pipeline_graph():
    """Constructs and compiles the parallel AI Factory analysis graph."""
    # Use PipelineState schema instead of generic dict
    workflow = StateGraph(PipelineState)

    # 1. Register all 8 agent nodes
    workflow.add_node("market_mapping", market_mapping_node)
    workflow.add_node("company_ingestion", company_ingestion_node)
    workflow.add_node("moat_analysis", moat_analysis_node)
    workflow.add_node("margin_analysis", margin_analysis_node)
    workflow.add_node("growth_forecast", growth_forecast_node)
    workflow.add_node("risk_adjustment", risk_adjustment_node)
    workflow.add_node("ranking", ranking_node)
    workflow.add_node("report", report_node)

    # 2. Stage 1: Market & Company Setup
    workflow.set_entry_point("market_mapping")
    workflow.add_edge("market_mapping", "company_ingestion")

    # 3. Stage 2: Fan-Out (Parallel execution of evaluation agents)
    workflow.add_edge("company_ingestion", "moat_analysis")
    workflow.add_edge("company_ingestion", "margin_analysis")
    workflow.add_edge("company_ingestion", "growth_forecast")
    workflow.add_edge("company_ingestion", "risk_adjustment")

    # 4. Stage 3: Fan-In (Consolidate parallel outputs into ranking)
    workflow.add_edge("moat_analysis", "ranking")
    workflow.add_edge("margin_analysis", "ranking")
    workflow.add_edge("growth_forecast", "ranking")
    workflow.add_edge("risk_adjustment", "ranking")

    # 5. Stage 4: Final Synthesis & Reporting
    workflow.add_edge("ranking", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


if __name__ == "__main__":
    print("Initializing Parallel AI Factory Growth Equity Pipeline...")
    app = build_pipeline_graph()

    initial_state = {
        "company_name": "NVIDIA",
        "segments": [],
        "companies": [],
        "moat_scores": [],
        "margin_scores": [],
        "growth_forecasts": [],
        "risk_adjustments": [],
        "rankings": [],
        "final_report": "",
    }

    print("Executing parallel graph...")
    final_output = app.invoke(initial_state)

    print("\n==========================================")
    print("         FINAL INVESTOR REPORT            ")
    print("==========================================\n")
    print(final_output.get("final_report", "No report generated."))
