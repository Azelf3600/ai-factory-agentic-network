"""
MAIN PIPELINE ORCHESTRATOR
"""

import os
import sys
from langgraph.graph import StateGraph, END

from schema import PipelineState

from agents.market_mapping_agent import market_mapping_node
from agents.company_ingestion_agent import company_ingestion_node
from agents.moat_analysis_agent import moat_analysis_node
from agents.margin_analysis_agent import margin_analysis_node
from agents.growth_forecast_agent import growth_forecast_node
from agents.risk_adjustment_agent import risk_adjustment_node
from agents.ai_revenue_exposure_agent import ai_revenue_exposure_node
from agents.cross_validation_agent import cross_validation_node
from agents.ranking_agent import ranking_node
from agents.report_agent import report_agent_node


def build_pipeline_graph():
    """Constructs and compiles the parallel AI Factory analysis graph."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("market_mapping", market_mapping_node)
    workflow.add_node("company_ingestion", company_ingestion_node)
    workflow.add_node("moat_analysis", moat_analysis_node)
    workflow.add_node("margin_analysis", margin_analysis_node)
    workflow.add_node("growth_forecast", growth_forecast_node)
    workflow.add_node("risk_adjustment", risk_adjustment_node)
    workflow.add_node("ai_revenue_exposure", ai_revenue_exposure_node)
    workflow.add_node("cross_validation", cross_validation_node)
    workflow.add_node("ranking", ranking_node)
    workflow.add_node("report", report_agent_node)

    workflow.set_entry_point("market_mapping")
    workflow.add_edge("market_mapping", "company_ingestion")

    workflow.add_edge("company_ingestion", "moat_analysis")
    workflow.add_edge("company_ingestion", "margin_analysis")
    workflow.add_edge("company_ingestion", "growth_forecast")
    workflow.add_edge("company_ingestion", "risk_adjustment")
    workflow.add_edge("company_ingestion", "ai_revenue_exposure")

    # Cross-validation runs only after ALL five scoring agents have written
    # their results — LangGraph waits for every incoming edge before firing
    # a node, so this join is automatic as long as all five point to it.
    workflow.add_edge("moat_analysis", "cross_validation")
    workflow.add_edge("margin_analysis", "cross_validation")
    workflow.add_edge("growth_forecast", "cross_validation")
    workflow.add_edge("risk_adjustment", "cross_validation")
    workflow.add_edge("ai_revenue_exposure", "cross_validation")

    workflow.add_edge("cross_validation", "ranking")
    workflow.add_edge("ranking", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# FIX: app.py does `from main import graph`, but this compiled graph was
# previously only ever built inside `if __name__ == "__main__":` below —
# which does NOT run when another module (like Streamlit's app.py) imports
# this file. That left no module-level `graph` name to import at all,
# causing "cannot import name 'graph' from 'main'" the moment app.py tried
# to import it.
#
# Building it here, at module level, means `graph` exists as soon as
# `main.py` is imported by anything (Streamlit, a test file, a notebook),
# not just when this file is run directly via `python main.py`.
graph = build_pipeline_graph()


if __name__ == "__main__":
    print("Initializing AI Factory Growth Equity Pipeline...")
    app = graph  # module-level `graph` above is the same compiled pipeline

    initial_state = {
        "segments": [],
        "companies": [],
        "moat_scores": [],
        "margin_scores": [],
        "growth_forecasts": [],
        "risk_adjustments": [],
        "ai_revenue_exposures": [],
        "cross_validation_flags": [],
        "rankings": [],
        "final_report": "",
    }

    print("Executing graph...")
    final_output = app.invoke(initial_state)

    print("\n==========================================")
    print("         FINAL INVESTOR REPORT            ")
    print("==========================================\n")
    print(final_output.get("final_report", "No report generated."))

    unmatched = [r for r in final_output.get("rankings", []) if r.get("unmatched")]
    if unmatched:
        print(f"\n[WARNING] {len(unmatched)} companies had incomplete agent data: "
              f"{', '.join(r['company'] for r in unmatched)}")

    cv_flags = final_output.get("cross_validation_flags", [])
    if cv_flags:
        print(f"\n[CROSS-VALIDATION] {len(cv_flags)} flag(s) raised:")
        for f in cv_flags:
            print(f"  - {f['company']} ({f['ticker']}) [{f['rule']}]: {f['detail']}")