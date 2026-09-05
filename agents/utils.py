"""
Shared helpers used by every scoring agent (Moat, Margin, Growth, Risk)
and the Ranking Agent.
"""

import difflib
from typing import Any, Dict, List, Optional


def _get(obj: Any, field: str, default=None):
    """Read a field whether obj is a dict or a pydantic model instance."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def build_lookup_key(name: Optional[str], ticker: Optional[str]) -> str:
    """Prefer ticker as the canonical key; fall back to normalized name."""
    if ticker:
        return ticker.strip().upper()
    return (name or "").strip().upper()


def match_company_result(
    company: Dict[str, Any],
    results: List[Any],
    name_field: str = "company",
    ticker_field: str = "ticker",
    fuzzy_cutoff: float = 0.82,
) -> tuple[Optional[Any], bool]:
    """Find the agent result entry that corresponds to `company`."""
    target_ticker = (company.get("ticker") or "").strip().upper()
    target_name = (company.get("name") or "").strip().upper()

    # 1. Exact ticker match
    if target_ticker:
        for r in results:
            r_ticker = (_get(r, ticker_field) or "").strip().upper()
            if r_ticker and r_ticker == target_ticker:
                return r, True

    # 2. Exact name match
    for r in results:
        r_name = (_get(r, name_field) or "").strip().upper()
        if r_name and r_name == target_name:
            return r, True

    # 3. Fuzzy name match
    candidate_names = [(_get(r, name_field) or "") for r in results]
    close = difflib.get_close_matches(company.get("name", ""), candidate_names, n=1, cutoff=fuzzy_cutoff)
    if close:
        for r in results:
            if _get(r, name_field) == close[0]:
                print(f"[match warning] fuzzy-matched '{company.get('name')}' -> '{close[0]}'")
                return r, True

    # 4. No match found
    print(f"[match FAILED] no result found for {company.get('name')} ({company.get('ticker')})")
    return None, False


def fill_missing_with_fallback(
    companies: List[Dict[str, Any]],
    results: List[Any],
    make_fallback,
    agent_label: str = "agent",
) -> List[Dict[str, Any]]:
    """
    Returns exactly one entry per company in `companies`, in the same order,
    every entry a plain dict.

    FIX vs previous version: matched entries (including fuzzy matches) are
    now normalized to carry the company's own canonical name/ticker instead
    of whatever string the LLM returned. Previously a fuzzy match here
    (e.g. "NVIDIA Corporation" -> "NVIDIA") was recovered at this stage but
    then lost again downstream in ranking_agent.py, which joins by exact
    ticker/name and had no way to know a fuzzy match had happened —
    the recovered result silently turned into a ranking-stage default with
    no warning. Normalizing the key here means any consumer doing a simple
    exact-match join will always succeed for anything this function
    resolved, matched or fallback.

    Also tags every entry with an explicit unmatched: bool, so downstream
    code (ranking_agent's data-quality flag, the report's "unmatched"
    footnote) can check a real field instead of sniffing a "FALLBACK:"
    text prefix — which margin's fallback function never set anyway.

    Also dedupes: if two companies would otherwise fuzzy-match to the same
    result object, only the first claims it; the second gets its own
    fallback rather than silently sharing another company's score.
    """
    output = []
    claimed_ids = set()

    for c in companies:
        match, matched = match_company_result(c, results)

        if matched and id(match) not in claimed_ids:
            claimed_ids.add(id(match))
            entry = dict(match) if isinstance(match, dict) else match.model_dump()
            entry["company"] = c.get("name")
            entry["ticker"] = c.get("ticker")
            entry["unmatched"] = False
            output.append(entry)
        else:
            if matched:
                print(f"[{agent_label}] result for '{c.get('name')}' was already claimed by "
                      f"another company (duplicate match) — applying fallback default instead.")
            else:
                print(f"[{agent_label}] no result for '{c.get('name')}' ({c.get('ticker')}) "
                      f"— applying fallback default.")
            fb = dict(make_fallback(c))
            fb["unmatched"] = True
            output.append(fb)

    return output