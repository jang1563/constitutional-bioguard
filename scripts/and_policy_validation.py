#!/usr/bin/env python
# AND policy validation on multiple expert/proxy legit-bio over-refusal sets.
# Question: does AND preserve the 15x over-ref reduction (vs single head) beyond the
# n=201 expert bridge set?
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).parent))
from dual_mode_guard import DualModeGuard, ROOT


def ci(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


SETS = [
    ("expert n=201 (bridge orig, no resp)", "data/bio_overrefusal_queries.jsonl"),
    ("expert n=181 +safe-resp (LARGE)", "data/processed/expert_legit_with_safe_responses.jsonl"),
    ("borderline n=79 +safe-resp (Step2)", "data/processed/borderline_with_safe_responses.jsonl"),
    ("FORTRESS safe held-out n=184", "data/external/fortress_safe_heldout.jsonl"),
    ("OR-Bench bio n=740 (no resp)", "data/external/orbench_bio.jsonl"),
]


def main():
    g = DualModeGuard()
    print(f"{'set':<32} {'policy':<14} {'over-ref':>10} {'95% CI':>16} {'vs v8bh':>10}")
    print("-" * 86)
    for name, path in SETS:
        rows = [json.loads(l) for l in open(ROOT / path) if l.strip()]
        Q = [r["query"] for r in rows]
        R = [r.get("response") or "" for r in rows]
        has_resp = any(R)
        pp, pr = g.score_batch(Q, R if has_resp else None)
        n = len(rows)
        for pol, flag in [("prompt_only", pp >= 0.5),
                          ("response_only", (pr if pr is not None else np.zeros_like(pp)) >= 0.5),
                          ("and", (pp >= 0.5) & ((pr if pr is not None else np.zeros_like(pp)) >= 0.5))]:
            if pol == "response_only" and pr is None:
                continue
            if pol == "and" and pr is None:
                continue
            k = int(flag.sum())
            r = k / n
            lo, hi = ci(k, n)
            ref = ""
            if pol == "and" and pr is not None:
                v8_or = float((pr >= 0.5).mean())
                ref = f"{v8_or/r:.1f}x" if r > 0 else "inf"
            print(f"{name[:32]:<32} {pol:<14} {r:>10.3f} [{lo:.3f},{hi:.3f}]  {ref:>10}")
        print()


if __name__ == "__main__":
    main()
