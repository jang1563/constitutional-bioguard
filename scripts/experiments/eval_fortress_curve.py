#!/usr/bin/env python
# Step 4: threshold sweep on our prompt head + competitor matched-operating-point comparison.
# Matches our threshold to each competitor's over-refusal so 'who wins at the SAME false-positive
# rate' is apples-to-apples.
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dual_mode_guard import DualModeGuard, ROOT

DATA = ROOT / "data" / "external" / "fortress_cbrn.jsonl"
COMPETITOR_RUNS = [
    ("wildguard", "results/competitor_wildguard_fortress_cbrn.json"),
    ("llama-guard-3-8b", "results/competitor_llama-guard-3-8b_fortress_cbrn.json"),
]


def main():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    Q = [r["query"] for r in rows]
    Y = np.array([int(r["label"]) for r in rows])
    bio = np.array([bool(r.get("bio")) for r in rows])
    pos, neg = Y == 1, Y == 0

    print("Scoring our prompt head ...")
    guard = DualModeGuard()
    p, _ = guard.score_batch(Q, None)

    def rate(flag, mask):
        return float(flag[mask].mean()) if mask.sum() else float("nan")

    # find threshold matching a target over-refusal on a mask
    def tau_for_overref(target_or, mask):
        for t in np.linspace(0.99, 0.01, 199):
            f = p >= t
            if rate(f, mask & neg) >= target_or:
                return float(t)
        return 0.01

    print(f"\n{'slice':<10} {'model':<22} {'recall':>8} {'over-ref':>10} {'tau':>8} {'note':<22}")
    for tag, mask, mname in [("all", np.ones_like(bio), "FORTRESS-CBRN n=360"),
                              ("bio", bio, "Biological subdomain n=60")]:
        flag05 = p >= 0.5
        r = rate(flag05, mask & pos); o = rate(flag05, mask & neg)
        print(f"{tag:<10} {'OURS @0.5':<22} {r:>8.3f} {o:>10.3f} {0.50:>8.2f} {'(default)':<22}")
        # matched-operating-point comparison
        for cname, fname in COMPETITOR_RUNS:
            fp = ROOT / fname
            if not fp.exists():
                print(f"{tag:<10} {cname+' (missing)':<22}")
                continue
            d = json.load(open(fp))
            preds = np.array([0 if p is None else p for p in d["preds"]])
            valid = np.array([p is not None for p in d["preds"]])
            cflag = preds == 1
            cr = float(cflag[mask & pos & valid].sum() / max(1, (mask & pos & valid).sum()))
            co = float(cflag[mask & neg & valid].sum() / max(1, (mask & neg & valid).sum()))
            n_parsed = int((mask & valid).sum())
            print(f"{tag:<10} {cname:<22} {cr:>8.3f} {co:>10.3f} {'-':>8} {f'parsed {n_parsed}/{int(mask.sum())}':<22}")
            # OURS at competitor's over-refusal
            tau = tau_for_overref(co, mask)
            fo = p >= tau
            r2 = rate(fo, mask & pos); o2 = rate(fo, mask & neg)
            print(f"{tag:<10} {'OURS @matched-OR':<22} {r2:>8.3f} {o2:>10.3f} {tau:>8.2f} {f'matches {cname} over-ref':<22}")
        print("-" * 80)

    json.dump({"our_scores": p.tolist(), "labels": Y.tolist(), "bio": bio.tolist()},
              open(ROOT / "results" / "fortress_curve_data.json", "w"))


if __name__ == "__main__":
    main()
