import os
import streamlit as st
import pandas as pd

# 1. Environment & API Key Setup
st.set_page_config(
    page_title="AI Infrastructure Growth Equity Analyzer",
    page_icon="⚡",
    layout="wide"
)

# Ensure keys are mirrored in os.environ for agents checking either variable name
api_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
else:
    st.error("API Key missing! Please set GOOGLE_API_KEY or GEMINI_API_KEY in Streamlit Secrets.")

# Import Graph Pipeline
try:
    from main import graph
except ImportError as e:
    st.error(f"Failed to import graph pipeline: {e}")

# 2. Header & Overview
st.title("⚡ AI Infrastructure Value Chain Analyzer")
st.markdown("""
*Multi-Agent Financial Analysis Pipeline evaluating AI Data Center Infrastructure companies across 5 core segments.*
""")

# 3. Sidebar Controls
st.sidebar.header("Pipeline Configuration")
run_button = st.sidebar.button("Run Full Pipeline", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("Segment Focus")
ALL_SEGMENTS = [
    "Compute / Servers",
    "Networking",
    "Power Infrastructure",
    "Cooling Systems",
    "Engineering & Construction"
]
selected_segments = st.sidebar.multiselect(
    "Filter Segments",
    ALL_SEGMENTS,
    default=ALL_SEGMENTS
)

# 4. Pipeline Execution
if run_button:
    # NOTE: previously this filter had no effect downstream — Company
    # Ingestion's dataset-loading path (the one used whenever a curated
    # JSON dataset is present) never consulted state["segments"], and
    # Market Mapping's node unconditionally overwrote whatever was passed
    # in here with its own fixed 5-segment list. Both are now fixed
    # (agents/company_ingestion_agent.py and agents/market_mapping_agent.py)
    # so this selection actually reaches the company universe.
    initial_state = {"segments": selected_segments}

    with st.status("Executing Multi-Agent Pipeline...", expanded=True) as status:
        st.write(" Mapping AI Factory Spend Layers...")
        try:
            st.write(" Ingesting Company Universe & Applying Segment Folds...")
            st.write(" Running Parallel Evaluation Agents (Moat, Margin, Growth, Risk, Exposure)...")
            st.write(" Executing Deterministic Cross-Validation Rules...")

            final_state = graph.invoke(initial_state)

            st.write(" Calculating TAFGS & Final Rankings...")
            st.write(" Synthesizing Executive Report & Profiles...")

            st.session_state["pipeline_results"] = final_state
            status.update(label="Pipeline Run Complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Pipeline Execution Failed", state="error", expanded=True)
            st.error(f"Error during execution: {e}")

# 5. Output Rendering
if "pipeline_results" in st.session_state:
    results = st.session_state["pipeline_results"]
    rankings = results.get("rankings", [])
    cv_flags = results.get("cross_validation_flags", [])
    final_report = results.get("final_report", "")

    # KPI Summary Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Companies Analyzed", len(rankings))
    with col2:
        avg_tafgs = round(sum(r["tafgs_score"] for r in rankings) / len(rankings), 2) if rankings else 0
        st.metric("Average TAFGS Score", avg_tafgs)
    with col3:
        st.metric("Cross-Validation Flags", len(cv_flags))
    with col4:
        # FIX: ranking_agent previously never carried the Margin Analysis
        # Agent's "real" vs "estimated" source tag into ranked_items, and
        # this read the wrong key ("margin_source" wasn't populated at
        # all) — so this metric always showed 0/N. ranking_agent.py now
        # emits "margin_source" per company, and this reads that field.
        real_margin_cnt = sum(1 for r in rankings if r.get("margin_source") == "real")
        st.metric("Verified Margins (yfinance)", f"{real_margin_cnt}/{len(rankings)}")

    # Visual Tabs
    tab_rankings, tab_flags, tab_report = st.tabs([" Master Rankings", " Cross-Validation Audit", " Executive Report"])

    # Tab 1: Master Table
    with tab_rankings:
        df = pd.DataFrame(rankings)
        if not df.empty:
            display_cols = [
                "rank", "company", "ticker", "segment", "tafgs_score",
                "moat_score", "margin_score", "operating_margin_pct",
                "growth_cagr_pct", "ai_revenue_exposure_pct",
                "risk_discount_pct"
            ]
            available_cols = [c for c in display_cols if c in df.columns]

            st.dataframe(
                df[available_cols],
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "tafgs_score": st.column_config.NumberColumn("TAFGS Score", format="%.2f"),
                    "operating_margin_pct": st.column_config.NumberColumn("Operating Margin", format="%.1f%%"),
                    "growth_cagr_pct": st.column_config.NumberColumn("3Yr CAGR", format="%.1f%%"),
                    "ai_revenue_exposure_pct": st.column_config.NumberColumn("AI Exposure", format="%.0f%%"),
                    # risk_discount_pct is stored as a 0.0-0.2 fraction (not
                    # a 0-100 value like the other %-fields above), so this
                    # uses Streamlit's built-in "percent" format, which
                    # scales automatically, instead of a printf %% string
                    # that would render 0.05 as "0%".
                    "risk_discount_pct": st.column_config.NumberColumn("Risk Discount", format="percent"),
                },
                use_container_width=True,
                hide_index=True
            )

    # Tab 2: Cross Validation Audit
    with tab_flags:
        if cv_flags:
            st.warning(f"Raised {len(cv_flags)} validation warnings across the dataset:")
            flags_df = pd.DataFrame(cv_flags)
            st.dataframe(
                flags_df[["company", "ticker", "rule", "severity", "detail"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No cross-validation flags raised for this run.")

    # Tab 3: Report & Downloads
    with tab_report:
        st.download_button(
            label="Download Markdown Report",
            data=final_report,
            file_name="AI_Infrastructure_Executive_Report.md",
            mime="text/markdown"
        )
        st.markdown(final_report)