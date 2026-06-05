#!/usr/bin/env python
# Step 3: distribution-free over-refusal bound for the dual-mode guard via Learn-then-Test.
# Select the response-head threshold tau* that GUARANTEES over-refusal (benign flag rate) <= alpha
# with confidence 1-delta, using an exact Clopper-Pearson upper bound on a calibration set, scanned
# fixed-sequence from high tau (low FPR) downward on the monotone FPR(tau) family. Report the
# guaranteed recall at tau* on held-out harmful + empirical over-refusal on held-out benign.
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).parent.parent


def cp_upper(k, n, delta):
    # Clopper-Pearson upper confidence bound on a binomial proportion.
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(ROOT / "results" / "realresp_curve_data.json"))
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.scores))
    p = np.array(d["p_resp"])
    Y = np.array(d["labels"])
    rng = np.random.RandomState(args.seed)

    benign = np.where(Y == 0)[0]
    harm = np.where(Y == 1)[0]
    rng.shuffle(benign)
    rng.shuffle(harm)
    # split benign: calibration (select tau) + test (validate FPR). harmful: test recall.
    bcal, btest = benign[: len(benign) // 2], benign[len(benign) // 2:]
    print(f"calibration benign={len(bcal)}, test benign={len(btest)}, test harmful={len(harm)}, "
          f"delta={args.delta}")
    print(f"\n{'alpha':>6}{'tau*':>8}{'UCB(FPR)':>10}{'test over-ref':>14}{'recall@tau*':>13}")
    out = {}
    for alpha in (0.05, 0.10, 0.15, 0.20):
        # fixed-sequence: high tau (low FPR) -> low; keep while CP-UCB on calib FPR <= alpha
        tau_star = 1.0
        for tau in np.linspace(0.99, 0.01, 197):
            k = int((p[bcal] >= tau).sum())
            if cp_upper(k, len(bcal), args.delta) <= alpha:
                tau_star = float(tau)
            else:
                break
        kcal = int((p[bcal] >= tau_star).sum())
        ucb = cp_upper(kcal, len(bcal), args.delta)
        test_or = float((p[btest] >= tau_star).mean())
        recall = float((p[harm] >= tau_star).mean())
        print(f"{alpha:>6.2f}{tau_star:>8.3f}{ucb:>10.3f}{test_or:>14.3f}{recall:>13.3f}")
        out[f"alpha_{alpha}"] = {"tau_star": tau_star, "ucb_fpr": ucb,
                                 "test_over_refusal": test_or, "recall": recall}
    print(f"\nGuarantee: at tau*, P(true over-refusal <= alpha) >= {1-args.delta:.2f} "
          "(Clopper-Pearson, fixed-sequence on monotone FPR). test over-ref validates the bound holds.")
    json.dump(out, open(ROOT / "results" / "conformal_bound.json", "w"), indent=2)


if __name__ == "__main__":
    main()
