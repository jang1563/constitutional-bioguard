#!/usr/bin/env python
"""Generate alert_cbrn_strict.jsonl — the honest CBRN-weapon ALERT subset.

The shipped alert_cbrn benchmark is a curation-polluted union (~43% crime_injury,
~29% substances, only ~14% true CBRN-weapon prompts), which inflates the headline
ALERT recall above the real CBRN number. See docs/V7B_OVER_REFUSAL_ANALYSIS.md §9.6.

This runner calls cache_alert_cbrn_strict(), which derives the strict subset
(weapon_biological/chemical/radioactive/nuclear only, class-tagged) deterministically
from the existing full ALERT cache — no HF re-download. Read-only w.r.t. payloads:
surfaces only counts and category/class labels.
"""
from __future__ import annotations

import collections
import json

from constitutional_bioguard.evaluation.extended_benchmarks import cache_alert_cbrn_strict


def main() -> None:
    p = cache_alert_cbrn_strict()
    rows = [json.loads(l) for l in open(p) if l.strip()]
    print("OUT  :", p)
    print("N    :", len(rows))
    print("class:", dict(collections.Counter(r["cbrn_class"] for r in rows)))
    print("cat  :", dict(collections.Counter(r["alert_category"] for r in rows)))
    assert all(r["is_cbrn_weapon"] and r["label"] == 1 for r in rows), "tag/label invariant broken"
    assert all(r["alert_category"] in {
        "weapon_biological", "weapon_chemical", "weapon_radioactive", "weapon_nuclear"
    } for r in rows), "off-mission category leaked into strict set"
    print("OK   : all is_cbrn_weapon, all label==1, no off-mission leakage")


if __name__ == "__main__":
    main()
