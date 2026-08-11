"""
Loads data/companies.json and validates every entry against the Company
model in schema.py. Run this any time the dataset changes.

Usage:
    !PYTHONPATH=. python data/load_companies.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from schema import Company  # noqa: E402


def load_companies(path="data/companies.json"):
    with open(path) as f:
        raw = json.load(f)

    segments = raw["segments"]
    companies = [Company(**c) for c in raw["companies"]]  # raises if any entry is malformed

    # sanity check: every company's segment must be one of the declared segments
    bad_segment = [c.name for c in companies if c.segment not in segments]
    if bad_segment:
        raise ValueError(f"Companies with segment not in declared segments list: {bad_segment}")

    return segments, companies


if __name__ == "__main__":
    segments, companies = load_companies()
    public = [c for c in companies if c.is_public]
    private = [c for c in companies if not c.is_public]

    print(f"Segments: {len(segments)}")
    print(f"Companies total: {len(companies)}")
    print(f"  Public (rankable): {len(public)}")
    print(f"  Private (context only, excluded from ranking): {len(private)}")
    print("\nPer-segment breakdown:")
    for seg in segments:
        count = len([c for c in companies if c.segment == seg])
        print(f"  {seg}: {count}")
    print("\nDataset is valid and matches schema.py.")
