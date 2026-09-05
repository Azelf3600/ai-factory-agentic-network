"""
COMPANY INGESTION AGENT

Fixes from previous version:
1. Now loads the curated data/companies.json dataset first (reproducible,
   vetted, excludes private companies per project scope) instead of
   regenerating a company list via LLM on every run.
2. If the dataset file is missing, falls back to LLM generation per
   segment — but round-robins collection instead of appending each
   segment's results in full then slicing to [:N], which used to let
   later segments (e.g. Engineering & Construction) get cut off entirely.
3. Static hardcoded fallback (used only if the LLM path also fails)
   is now segment-balanced by construction.
4. FIX: _load_static_dataset previously only ever checked DATASET_PATH
   (companies_full_universe.json). FALLBACK_DATASET_PATH (companies.json)
   was defined but dead code — if the full-universe file was missing, the
   loader jumped straight to LLM generation instead of trying the fallback
   dataset, contradicting the docstring above. Now it actually tries both,
   in order.
5. Every ingested company is tagged with ai_revenue_exposure_source so
   downstream agents/report know 50.0 is a placeholder, not a research
   finding.

SEGMENT-FOLD SCOPE DECISION: data/companies.json was built with 7 segments
(including a "Compute Platforms & Hyperscalers" segment for Microsoft/
Amazon/TSMC/Broadcom/Meta). schema.py, ranking, and report converged on 5
segments per the project brief's Section 3.1 table. This loader folds the
7 segments down to 5 (hyperscaler segment -> "Compute / Servers", both
power segments -> "Power Infrastructure") so nothing crashes downstream.
The buyer/operator vs. component-supplier distinction this fold collapses
is preserved structurally via the `ai_factory_role` / `original_segment`
fields below, not just described in prose.

FIX (this version) — SEGMENT FILTER WAS DEAD ON THE DATASET PATH: the
dataset-loading branch (the one actually used whenever a curated JSON file
is present, i.e. essentially every real run) never consulted
state["segments"] at all — it always returned every company in the file
regardless of what the caller asked for. Only the LLM-generation fallback
branch (used when no dataset file exists) ever looked at `segments`. In
practice this meant the Streamlit sidebar's "Filter Segments" multiselect
had no effect on a normal run.

Now `company_ingestion_node` filters `dataset_companies` down to the
requested `segments` (matched against each company's already-folded
5-segment `segment` field, since that's what the rest of the pipeline and
the UI both operate on) before returning, on both the dataset path and the
LLM path. If the filter would produce an empty list (e.g. a stale/bad
segment name was passed in), it falls back to the unfiltered set rather
than silently returning zero companies, and logs why.
"""

import json
import os
import sys
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import DEFAULT_MODEL  # noqa: E402

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)

DEFAULT_SEGMENTS = [
    "Compute / Servers",
    "Networking",
    "Power Infrastructure",
    "Cooling Systems",
    "Engineering & Construction",
]

# RESOLVED (per activity Section 3.1 — 5 segments only): the Day 2 dataset's
# "Compute Platforms & Hyperscalers" segment (Microsoft/Amazon/TSMC/Broadcom/
# Meta) folds into "Compute / Servers" for ranking/report purposes. The
# buyer/operator vs. component-supplier distinction this fold collapses is
# preserved structurally via ai_factory_role below, not just described in
# prose.
SEGMENT_FOLD_MAP = {
    "Compute / AI Servers & GPUs": "Compute / Servers",
    "Compute Platforms & Hyperscalers": "Compute / Servers",
    "Networking": "Networking",
    "Power - Generators & Turbines": "Power Infrastructure",
    "Power - UPS, Switchgear & PDUs": "Power Infrastructure",
    "Cooling": "Cooling Systems",
    "Engineering & Construction": "Engineering & Construction",
}

# The one original_segment value that represents a buyer/operator role
# rather than a component-supplier role within the AI Factory value chain.
BUYER_OPERATOR_ORIGINAL_SEGMENT = "Compute Platforms & Hyperscalers"

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "companies_full_universe.json")
FALLBACK_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "companies.json")

# Hardcoded, segment-balanced last-resort list. Used only if both dataset
# files are missing AND the LLM generation path also fails outright.
STATIC_FALLBACK_COMPANIES = [
    {"name": "NVIDIA", "ticker": "NVDA", "segment": "Compute / Servers"},
    {"name": "Super Micro Computer", "ticker": "SMCI", "segment": "Compute / Servers"},
    {"name": "Arista Networks", "ticker": "ANET", "segment": "Networking"},
    {"name": "Vertiv Holdings", "ticker": "VRT", "segment": "Power Infrastructure"},
    {"name": "Modine Manufacturing", "ticker": "MOD", "segment": "Cooling Systems"},
    {"name": "Fluor Corporation", "ticker": "FLR", "segment": "Engineering & Construction"},
]


class EligibleCompany(BaseModel):
    name: str
    ticker: Optional[str] = None
    segment: str
    is_public: bool = True
    description: Optional[str] = None


class CompanyIngestionOutput(BaseModel):
    segment: str
    companies: List[EligibleCompany]


def _load_dataset_file(path: str) -> Optional[List[dict]]:
    """Load and fold a single dataset file. Returns None if it doesn't exist."""
    if not os.path.isfile(path):
        return None

    with open(path) as f:
        raw = json.load(f)

    companies = []
    for c in raw.get("companies", []):
        if not c.get("is_public", True):
            continue  # private companies excluded from ranking per project scope
        original_segment = c["segment"]
        folded_segment = SEGMENT_FOLD_MAP.get(original_segment, "Compute / Servers")
        ai_factory_role = (
            "buyer_operator" if original_segment == BUYER_OPERATOR_ORIGINAL_SEGMENT
            else "component_supplier"
        )
        companies.append({
            "name": c["name"],
            "ticker": c.get("ticker"),
            "segment": folded_segment,
            "original_segment": original_segment,
            "ai_factory_role": ai_factory_role,
            "is_public": True,
            "description": c.get("description", ""),
            "ai_revenue_exposure_pct": c.get("ai_revenue_exposure_pct", 50.0),
            "ai_revenue_exposure_source": "researched" if "ai_revenue_exposure_pct" in c else "placeholder",
        })
    return companies


def _load_static_dataset() -> Optional[List[dict]]:
    """
    Try the primary curated dataset first, then the fallback dataset.
    Returns None only if neither file exists, so callers fall through to
    LLM generation.
    """
    companies = _load_dataset_file(DATASET_PATH)
    if companies is not None:
        return companies

    print(f"[Company Ingestion] {DATASET_PATH} not found — trying fallback dataset.")
    return _load_dataset_file(FALLBACK_DATASET_PATH)


def _filter_by_segments(companies: List[dict], requested_segments: Optional[List[str]]) -> List[dict]:
    """
    Restricts `companies` to those whose (already-folded) 5-segment
    `segment` field is in `requested_segments`. This is the filter the
    Streamlit "Filter Segments" multiselect (and any other caller) relies
    on — previously it was applied nowhere on the dataset-loading path.

    If requested_segments is empty/None, or filtering would zero out the
    whole list (e.g. a stale segment name was passed in), returns
    `companies` unfiltered rather than silently producing an empty
    universe, and logs why.
    """
    if not requested_segments:
        return companies

    requested_set = set(requested_segments)
    filtered = [c for c in companies if c.get("segment") in requested_set]

    if not filtered:
        print(f"[Company Ingestion] Requested segments {requested_segments} matched 0 "
              f"companies — ignoring the filter and returning the full universe instead.")
        return companies

    print(f"[Company Ingestion] Filtered to {len(filtered)}/{len(companies)} companies "
          f"for requested segments: {requested_segments}")
    return filtered


def _generate_via_llm(segments: List[str], per_segment: int = 4) -> List[dict]:
    """
    Round-robins across segments so no segment can be starved by an
    earlier segment eating the whole company budget.
    """
    structured_llm = llm.with_structured_output(CompanyIngestionOutput, method="json_schema")
    by_segment = {}

    for segment in segments:
        prompt = (
            f"Identify {per_segment} distinct PUBLICLY TRADED companies operating in the "
            f"'{segment}' AI data center infrastructure layer. Include ticker symbols."
        )
        try:
            result: CompanyIngestionOutput = structured_llm.invoke(prompt)
            by_segment[segment] = [
                {
                    "name": c.name,
                    "ticker": c.ticker or "N/A",
                    "segment": segment,
                    "original_segment": segment,
                    "ai_factory_role": "component_supplier",
                    "is_public": c.is_public,
                    "description": c.description or "",
                    "ai_revenue_exposure_pct": 50.0,
                    "ai_revenue_exposure_source": "placeholder",
                }
                for c in result.companies
            ]
        except Exception as e:
            print(f"[Company Ingestion Error] Segment {segment}: {e}")
            by_segment[segment] = []

    # Round-robin flatten so every segment gets fair representation
    # even if the total exceeds or falls short of a target count.
    companies_list = []
    seen_names = set()
    max_len = max((len(v) for v in by_segment.values()), default=0)
    for i in range(max_len):
        for segment in segments:
            seg_list = by_segment.get(segment, [])
            if i < len(seg_list) and seg_list[i]["name"] not in seen_names:
                seen_names.add(seg_list[i]["name"])
                companies_list.append(seg_list[i])

    if not companies_list:
        print("[Company Ingestion] LLM generation returned nothing — using static fallback list.")
        companies_list = [
            {**c, "is_public": True, "description": "",
             "original_segment": c["segment"], "ai_factory_role": "component_supplier",
             "ai_revenue_exposure_pct": 50.0, "ai_revenue_exposure_source": "placeholder"}
            for c in STATIC_FALLBACK_COMPANIES
        ]

    return companies_list


def company_ingestion_node(state: dict) -> dict:
    requested_segments = state.get("segments") or None

    dataset_companies = _load_static_dataset()
    if dataset_companies:
        print(f"[Company Ingestion] Loaded {len(dataset_companies)} companies from dataset file.")
        dataset_companies = _filter_by_segments(dataset_companies, requested_segments)
        n_buyer_operator = sum(1 for c in dataset_companies if c["ai_factory_role"] == "buyer_operator")
        if n_buyer_operator:
            print(f"[Company Ingestion] {n_buyer_operator} companies tagged 'buyer_operator' "
                  f"(hyperscalers folded into Compute / Servers — see original_segment field).")
        return {"companies": dataset_companies}

    print("[Company Ingestion] No dataset file found — generating via LLM instead.")
    segments_for_llm = requested_segments or DEFAULT_SEGMENTS
    companies_list = _generate_via_llm(segments_for_llm)
    return {"companies": companies_list}