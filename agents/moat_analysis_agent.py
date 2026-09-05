"""MOAT ANALYSIS AGENT

FIX (this version): the prompt previously asked for "economic moat (0-5)"
with no anchor beyond the four bullet categories. In practice this let the
LLM score general corporate moat — a regulated utility's monopoly, an
industrial conglomerate's brand — rather than moat *specifically within
the AI Factory value chain*, per spec Section 2.1. This showed up as a
systematic pattern in Cross-Validation's high_moat_low_growth flags: at
both 95 and 250 companies, ~22-23% of the universe (utilities, general
semiconductor incumbents, general tech/IT-services giants) got moat=4-5
paired with single-digit AI-driven CAGR, because their moat rationale was
never actually about AI Factory positioning to begin with.

The prompt now explicitly instructs the model to score moat ONLY as it
applies to AI Factory / hyperscale data center infrastructure specifically,
gives a worked contrast example (regulated utility vs. NVIDIA/TSM), and
requires the rationale to name the specific AI Factory role the moat
protects — not just restate general corporate strength. This doesn't
eliminate high_moat_low_growth flags (some will be legitimate — a real
bottleneck position that genuinely hasn't translated to growth yet is
worth flagging), but it should sharply cut the ones that are really just
mislabeled general-corporate-moat scores.
"""

import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL, MoatScore  # noqa: E402
from agents.utils import fill_missing_with_fallback  # noqa: E402
from agents.batch_utils import chunk, BATCH_SIZE, invoke_with_retry  # noqa: E402

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)


class MoatBatchOutput(BaseModel):
    analysis: List[MoatScore]


def _fallback_moat(company: dict) -> dict:
    return {
        "company": company.get("name", "Unknown"),
        "ticker": company.get("ticker"),
        "score": 3,
        "rationale": "FALLBACK: no LLM result returned for this company — treat as placeholder, not a real assessment.",
    }


def moat_analysis_node(state: dict) -> dict:
    companies = state.get("companies", [])
    if not companies and "company_name" in state:
        companies = [{"name": state["company_name"]}]

    company_dicts = [c if isinstance(c, dict) else c.model_dump() for c in companies]

    structured_llm = llm.with_structured_output(MoatBatchOutput, method="json_schema")
    all_results: List[dict] = []

    for batch in chunk(company_dicts, BATCH_SIZE):
        # Send name + ticker explicitly so the model can echo the ticker back —
        # tickers rarely get reworded the way company names do.
        company_lines = "\n".join(
            f"- {c.get('name')} (ticker: {c.get('ticker', 'unknown')}, segment: {c.get('segment', 'unknown')})"
            for c in batch
        )

        prompt = f"""
Score each company below on economic moat (0-5), but ONLY as that moat
applies to AI Factory / hyperscale AI data center infrastructure
specifically — NOT the company's general corporate moat.

This distinction matters. A company can have a powerful moat in its core
business while having little or no moat in AI Factory infrastructure —
score the latter, not the former. For example:
- A regulated electric utility may have an unassailable monopoly in its
  service territory, but that monopoly protects its general power business,
  not a differentiated or defensible position supplying AI data center
  demand specifically. Being a bystander that benefits from nearby AI
  Factory load growth is NOT a moat — score it low (0-2) unless the
  company has a genuinely differentiated role (e.g. an exclusive
  large-scale power purchase agreement structure, unique grid-interconnect
  capacity, or proprietary generation technology that AI Factory
  developers specifically compete for access to).
- A large, diversified industrial or tech conglomerate may dominate its
  primary market, but if AI Factory infrastructure is a small, generic
  slice of a much bigger business with no special lock-in there, score it
  low-to-moderate (1-3), not high, even if the company is a household name.
- Contrast that with NVIDIA (CUDA ecosystem lock-in specific to AI compute)
  or TSMC (the sole fab capable of producing leading-edge AI chips at
  volume) — these are moats that exist BECAUSE of, and ONLY WITHIN, the AI
  Factory value chain. That is what a 4-5 score should represent:
  architectural lock-in, ecosystem dominance/design wins, switching costs,
  or scarcity/bottleneck position specifically within AI Factory compute,
  networking, power, cooling, or construction.

For EACH company, return its exact ticker symbol as given below (do not
invent or omit it), a score (0-5), and a rationale. The rationale MUST
name the specific AI Factory role the moat protects (e.g. "sole supplier
of X to hyperscale GPU clusters") — a rationale that only restates general
corporate strength without naming an AI-Factory-specific mechanism is not
acceptable and should push the score down, not up.

Companies:
{company_lines}
"""
        try:
            result: MoatBatchOutput = invoke_with_retry(structured_llm, prompt)
            batch_results = [m.model_dump() for m in result.analysis]
            if len(batch_results) != len(batch):
                print(f"[Moat Analysis WARNING] batch of {len(batch)} companies returned "
                      f"{len(batch_results)} results — some may be missing and will fall back to default.")
            all_results.extend(batch_results)
        except Exception as e:
            print(f"[Moat Analysis Error] batch starting with {batch[0].get('name')}: {e}")

    results = fill_missing_with_fallback(company_dicts, all_results, _fallback_moat, agent_label="Moat Analysis Agent")
    return {"moat_scores": results}