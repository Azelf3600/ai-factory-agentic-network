"""REPORT SYNTHESIS AGENT"""

import os
import sys
import json
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, PipelineState  # noqa: E402

TOP_N_PROFILES = 20     # per spec Section 3.3 — "Top 20 AI Factory Growth Ranking"
PROFILE_BATCH_SIZE = 5  # small batches so the LLM can't quietly truncate to an "excerpt"

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.2,
)


class CompanyProfile(BaseModel):
    company: str
    ticker: str
    primary_role: str
    moat_narrative: str
    margin_profile: str
    growth_catalysts: str
    key_risks: str


class ProfileBatchOutput(BaseModel):
    profiles: List[CompanyProfile]


def _chunk(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _build_master_table(rankings: List[dict]) -> str:
    """
    Built deterministically from real data, not LLM-transcribed — removes
    any chance of the model mis-copying scores it already has as structured
    data (this is what caused the wrong margin brackets to show up in the
    table even after the underlying score was briefly fixed elsewhere).
    """
    header = "| Rank | Company | Ticker | Segment | Moat | Margin | Op Margin % | AI Rev Exposure % | CAGR % | TAFGS |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in rankings:
        flag = " *" if r.get("unmatched") else ""
        exposure = r.get("ai_revenue_exposure_pct", 50.0)
        exposure_label = f"{exposure:.0f}%" if r.get("ai_revenue_exposure_source") != "placeholder" else "n/a*"
        rows.append(
            f"| {r['rank']} | {r['company']}{flag} | {r.get('ticker', 'N/A')} | {r['segment']} | "
            f"{r['moat_score']} | {r['margin_score']} | {r['operating_margin_pct']:.1f}% | "
            f"{exposure_label} | {r['growth_cagr_pct']:.1f}% | {r['tafgs_score']:.2f} |"
        )
    table = "\n".join(rows)
    footnote = ("\n\n\\* Flagged rows had at least one agent fall back to a default "
                "(unmatched result), or AI revenue exposure has not yet been reliably estimated.")
    return table + footnote


def _generate_profiles(rankings_top_n: List[dict]) -> str:
    """
    Generates the Top-N profiles in small batches so every single company
    actually gets a profile, instead of one giant call that quietly
    produces only a partial 'excerpt' (observed at 5-of-20 previously).
    """
    structured_llm = llm.with_structured_output(ProfileBatchOutput, method="json_schema")
    all_profiles: List[CompanyProfile] = []

    for batch in _chunk(rankings_top_n, PROFILE_BATCH_SIZE):
        batch_json = json.dumps(batch, indent=2)
        prompt = f"""
You are a Lead Growth Equity Analyst. Write a structured profile for EVERY
SINGLE company in this batch — do not skip any, do not summarize only the
top ones, this batch is intentionally small so you can cover all of them:

{batch_json}

For each company provide:
- primary_role: their role in the AI Factory value chain, one sentence.
- moat_narrative: differentiation / lock-in, 1-2 sentences.
- margin_profile: comment on their operating margin, 1 sentence.
- growth_catalysts: 2026-2029 AI-driven growth catalysts, 1-2 sentences.
- key_risks: customer concentration, cyclicality, or supply chain bottleneck, 1-2 sentences.

If ai_revenue_exposure_source is "placeholder" for a company, do not state
an exposure % as fact anywhere in that company's profile — say it hasn't
been reliably estimated yet.
"""
        try:
            result: ProfileBatchOutput = structured_llm.invoke(prompt)
            batch_profiles = result.profiles
            if len(batch_profiles) != len(batch):
                print(f"[Report Agent WARNING] batch of {len(batch)} companies returned "
                      f"{len(batch_profiles)} profiles — some may be missing.")
            all_profiles.extend(batch_profiles)
        except Exception as e:
            print(f"[Report Agent Error] profile batch starting with {batch[0].get('company')}: {e}")
            # Fallback: emit a stub profile per company so nobody silently
            # disappears from the report even if this batch's LLM call fails.
            for c in batch:
                all_profiles.append(CompanyProfile(
                    company=c.get("company", "Unknown"),
                    ticker=c.get("ticker", "N/A"),
                    primary_role=c.get("segment", "Unknown"),
                    moat_narrative="Profile generation failed for this company — see rankings table for scores.",
                    margin_profile=f"{c.get('operating_margin_pct', 0):.1f}% operating margin.",
                    growth_catalysts="Not generated (LLM error).",
                    key_risks="Not generated (LLM error).",
                ))

    # Render in rank order, matched back to the original ranking dicts.
    by_company = {p.company: p for p in all_profiles}
    sections = []
    for r in rankings_top_n:
        p = by_company.get(r["company"])
        if p is None:
            continue
        sections.append(f"""#### {r['rank']}. {p.company} ({p.ticker}) | TAFGS: {r['tafgs_score']:.2f}
*   **Role:** {p.primary_role}
*   **Moat:** {p.moat_narrative}
*   **Margin:** {p.margin_profile}
*   **Growth Catalysts:** {p.growth_catalysts}
*   **Risks:** {p.key_risks}
""")
    return "\n".join(sections)


def report_agent_node(state: PipelineState) -> dict:
    rankings = state.get("rankings", [])
    top_n = rankings[:TOP_N_PROFILES]

    unmatched = [r for r in rankings if r.get("unmatched")]
    unmatched_note = ""
    if unmatched:
        names = ", ".join(r["company"] for r in unmatched)
        unmatched_note = (
            f"\n**Data quality note:** the following companies had at least one agent "
            f"fall back to a default and should be treated as lower-confidence: {names}\n"
        )

    spend_share = state.get("market_mapping_result", {}).get("segment_spend_share_pct", {})
    spend_share_lines = "\n".join(f"- {seg}: ~{pct}%" for seg, pct in spend_share.items()) or \
        "- (segment spend share unavailable — Market Mapping agent did not run)"

    master_table = _build_master_table(rankings)
    profiles_md = _generate_profiles(top_n)

    report_text = f"""# Executive AI Infrastructure Report

## 1. AI Factory Value-Chain Mapping
Spend-share breakdown, derived from AI Factory equipment cost reference data:

{spend_share_lines}
{unmatched_note}
## 2. Master Rankings Table

{master_table}

## 3. Top {len(top_n)} Company Profiles

{profiles_md}
"""
    return {"final_report": report_text}