"""
MARGIN ANALYSIS AGENT

Fix from previous version: the LLM was asked to both estimate the operating
margin % AND classify it into the spec's 0-5 bracket in the same call.
Checked against its own numbers, it violated the bracket rule on ~20% of
companies in a 30-company test run (e.g. reporting 42% margin but scoring
it 4 instead of 5, or 9% margin scored 2 instead of 1). LLMs are unreliable
at self-applying simple arithmetic rules consistently across a batch, even
when the rule is stated plainly in the prompt.

Fix: keep the LLM for what it's actually useful for — estimating/recalling
the operating margin % — but always recompute the 0-5 score deterministically
in code from that %, overriding whatever bracket the LLM assigned.
"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, MarginScore  # noqa: E402
from agents.utils import fill_missing_with_fallback  # noqa: E402

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)


class MarginBatchOutput(BaseModel):
    analysis: List[MarginScore]


def _bracket_margin_score(operating_margin_pct: float) -> int:
    """
    Deterministic bracket per project spec Section 2.2:
        >40%    -> 5
        30-40%  -> 4
        20-30%  -> 3
        10-20%  -> 2
        <10%    -> 1
    Never delegate this classification to the LLM — see module docstring.
    """
    if operating_margin_pct > 40:
        return 5
    elif operating_margin_pct >= 30:
        return 4
    elif operating_margin_pct >= 20:
        return 3
    elif operating_margin_pct >= 10:
        return 2
    else:
        return 1


def _fallback_margin(company: dict) -> dict:
    return {
        "company": company.get("name", "Unknown"),
        "ticker": company.get("ticker"),
        "score": 3,
        "operating_margin_pct": 20.0,
        "source": "estimated",
    }


def margin_analysis_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    company_dicts = [c if isinstance(c, dict) else c.model_dump() for c in companies]
    company_lines = "\n".join(
        f"- {c.get('name')} (ticker: {c.get('ticker', 'unknown')})" for c in company_dicts
    )

    structured_llm = llm.with_structured_output(MarginBatchOutput, method="json_schema")
    prompt = f"""
Provide the operating margin % for each company below. Give your best
estimate of the actual reported operating margin. Return each company's
exact ticker symbol as given (do not invent or omit it). Mark source as
"real" only if you are confident in the actual reported operating margin;
otherwise mark it "estimated". You may also provide a score field, but it
will be recalculated deterministically from operating_margin_pct
afterward, so focus your effort on getting the margin % itself right.

Companies:
{company_lines}
"""

    try:
        result: MarginBatchOutput = structured_llm.invoke(prompt)
        results = [m.model_dump() for m in result.analysis]
    except Exception as e:
        print(f"[Margin Analysis Error]: {e}")
        results = []

    # Always override the LLM's own bracket classification with a
    # deterministic recomputation from the margin % it reported.
    for r in results:
        r["score"] = _bracket_margin_score(r.get("operating_margin_pct", 20.0))

    results = fill_missing_with_fallback(company_dicts, results, _fallback_margin, agent_label="Margin Analysis Agent")
    return {"margin_scores": results}