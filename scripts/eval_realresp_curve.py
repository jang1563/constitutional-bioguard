#!/usr/bin/env python
# Step 4: matched-operating-point comparison on real_response_bio (response-harm).
# Our RESPONSE head (v8b encoder, continuous scores) swept to each competitor's over-refusal so
# 'who has higher recall at the SAME false-positive rate' is apples-to-apples.
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dual_mode_guard import DualModeGuard, ROOT

import argparse
COMP_MODELS = ["wildguard", "llama-guard-3-8b", "shieldgemma-9b", "qwen3guard-8b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "external" / "real_response_bio.jsonl"))
    args = ap.parse_args()
    DATA = Path(args.data)
    stem = DATA.stem
    COMP = [(m, f"results/competitor_{m}_{stem}.json") for m in COMP_MODELS]
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    Q = [r["query"] for r in rows]
    R = [r.get("response") or "" for r in rows]
    Y = np.array([int(r["label"]) for r in rows])
    pos, neg = Y == 1, Y == 0

    print("Scoring our heads ...")
    guard = DualModeGuard()
    p_prompt, p_resp = guard.score_batch(Q, R)

    def rec_or(score, tau):
        f = score >= tau
        return float(f[pos].mean()), float(f[neg].mean())

    def tau_for(score, target_or):
        # lowest tau (highest recall) whose over-refusal <= target
        best = 0.999
        for t in np.linspace(0.001, 0.999, 999):
            if rec_or(score, t)[1] <= target_or:
                best = t
                break
        return best

    print(f"\nn={len(rows)} (harm {int(pos.sum())} / benign {int(neg.sum())})")
    print(f"{'model':<26}{'recall':>9}{'over-ref':>10}{'tau':>7}")
    r0, o0 = rec_or(p_resp, 0.5)
    print(f"{'OURS response_only @0.5':<26}{r0:>9.3f}{o0:>10.3f}{0.5:>7.2f}")
    for cname, fname in COMP:
        if not (ROOT / fname).exists():
            continue
        d = json.load(open(ROOT / fname))
        preds = np.array([0 if x is None else x for x in d["preds"]])
        valid = np.array([x is not None for x in d["preds"]])
        cr = float((preds[pos & valid] == 1).mean())
        co = float((preds[neg & valid] == 1).mean())
        print(f"{cname:<26}{cr:>9.3f}{co:>10.3f}{'-':>7}  parsed {int(valid.sum())}/{len(rows)}")
        tau = tau_for(p_resp, co)
        rr, oo = rec_or(p_resp, tau)
        win = "WE WIN" if rr > cr else "we lose"
        print(f"{'  OURS @matched-OR':<26}{rr:>9.3f}{oo:>10.3f}{tau:>7.2f}  vs {cname}: {win} (recall {rr:.3f} vs {cr:.3f})")

    # response head ROC sanity: a few sweep points (confirm NOT saturated)
    print("\nresponse-head sweep (recall / over-ref):")
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        r, o = rec_or(p_resp, t)
        print(f"  tau={t:.1f}: recall={r:.3f} over-ref={o:.3f}")
    json.dump({"p_resp": p_resp.tolist(), "p_prompt": p_prompt.tolist(), "labels": Y.tolist()},
              open(ROOT / "results" / "realresp_curve_data.json", "w"))


if __name__ == "__main__":
    main()
