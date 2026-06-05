#!/usr/bin/env python
"""Per-category breakdown of /no_think ALERT-CBRN recall misses (informs Phase 2).

ALERT-CBRN /no_think flagged 0.646 of harmful items (1486 misses / 4198) with a
clean parser (unknown_rate 0.0007 -- no colloquial confound, unlike WildGuard).
This script joins the saved predictions to the ALERT source by index and reports
recall per `alert_category`, so Phase 2's recall-preserving data design can target
the specific CBRN sub-types the model under-flags, rather than blanket-augmenting.

Treats the data as opaque: only category labels, source tags, and binary
predictions are surfaced -- never prompt/response payload text.

Pure post-hoc join + metric aggregation. No model, no GPU.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR

SRC = DATA_EXTERNAL / "alert_cbrn.jsonl"
PRED = METRICS_DIR / "v7b_nothink_eval_alert_cbrn.json"


def main():
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    pred = json.load(open(PRED))["predictions"]
    assert len(rows) == len(pred), (len(rows), len(pred))

    print("=== ALERT-CBRN /no_think miss breakdown ===")
    print("n:", len(rows))
    print("alert_category dist:",
          dict(Counter(r["alert_category"] for r in rows).most_common()))
    print("source dist:",
          dict(Counter(str(r.get("source")) for r in rows).most_common()))
    print()

    # All ALERT items are label=1 (harmful) -> recall == flag rate.
    def table(keyfn, title):
        agg = defaultdict(lambda: [0, 0, 0, 0])  # n, flagged, resp_unharmful, resp_unknown
        for r, p in zip(rows, pred):
            k = keyfn(r)
            b = agg[k]
            b[0] += 1
            b[1] += p["pred"]
            if p.get("response_harm") == "unharmful":
                b[2] += 1
            if p.get("response_harm") == "unknown":
                b[3] += 1
        hdr = "{:<30}{:>6}{:>6}{:>8}{:>6}{:>7}{:>5}".format(
            title, "n", "flag", "recall", "miss", "unhrm", "unk")
        print(hdr)
        print("-" * len(hdr))
        for k, (n, fl, uh, uk) in sorted(agg.items(), key=lambda kv: kv[1][0] - kv[1][1],
                                         reverse=True):
            # sort by absolute miss count (n - flagged) descending
            miss = n - fl
            print("{:<30}{:>6}{:>6}{:>8.3f}{:>6}{:>7}{:>5}".format(
                k, n, fl, fl / n if n else 0, miss, uh, uk))
        N = sum(b[0] for b in agg.values())
        F = sum(b[1] for b in agg.values())
        print("{:<30}{:>6}{:>6}{:>8.3f}{:>6}".format("OVERALL", N, F, F / N, N - F))
        print()

    table(lambda r: r["alert_category"], "alert_category")
    if "filter_reason" in rows[0]:
        # filter_reason can be long free text; bucket by first 40 chars
        table(lambda r: (str(r.get("filter_reason")) or "none")[:40], "filter_reason[:40]")


if __name__ == "__main__":
    main()
