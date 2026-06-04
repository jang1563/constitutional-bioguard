#!/usr/bin/env python
# Integrity-review fixes: confidence intervals (Clopper-Pearson), McNemar paired tests, AUPRC for
# the footprint claim, and paired recall+over-refusal per policy. Produces the corrected numbers
# the docs must use.
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta, binomtest
from scipy.stats import chi2

ROOT = Path(__file__).parent.parent
COMP = ["wildguard", "llama-guard-3-8b", "shieldgemma-9b", "qwen3guard-8b"]


def cp_ci(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def rate_ci(flag, mask):
    k = int(flag[mask].sum())
    n = int(mask.sum())
    lo, hi = cp_ci(k, n)
    return k / n if n else float("nan"), lo, hi, n


def mcnemar(a, b, y):
    # paired test: a,b binary preds, y labels; compare on POSITIVES (recall) -> discordant pairs
    pos = y == 1
    a_only = int(((a == 1) & (b == 0) & pos).sum())
    b_only = int(((a == 0) & (b == 1) & pos).sum())
    nd = a_only + b_only
    if nd == 0:
        return 1.0, a_only, b_only
    # exact binomial (two-sided) on discordant
    p = binomtest(a_only, nd, 0.5).pvalue
    return p, a_only, b_only


def auprc(scores, y):
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp)
    rec = tp / max(1, ys.sum())
    # average precision
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


def comp_preds(stem):
    out = {}
    for g in COMP:
        fp = ROOT / "results" / f"competitor_{g}_{stem}.json"
        if fp.exists():
            d = json.load(open(fp))
            out[g] = np.array([0 if x is None else x for x in d["preds"]])
    return out


print("=" * 72)
print("(1) FORTRESS-bio recall with 95% Clopper-Pearson CI + McNemar vs ours (n=30 harmful)")
rows = [json.loads(l) for l in open(ROOT / "data/external/fortress_cbrn.jsonl") if l.strip()]
Y = np.array([int(r["label"]) for r in rows])
bio = np.array([bool(r.get("bio")) for r in rows])
cur = json.load(open(ROOT / "results/fortress_curve_data.json"))
ours_p = np.array(cur["our_scores"])
ours_flag = (ours_p >= 0.5).astype(int)
posbio = bio & (Y == 1)
r, lo, hi, n = rate_ci(ours_flag, posbio)
print(f"  OURS prompt   recall={r:.3f} [{lo:.3f},{hi:.3f}] n={n}")
cp = comp_preds("fortress_cbrn")
for g, pr in cp.items():
    r, lo, hi, n = rate_ci(pr == 1, posbio)
    p, ao, bo = mcnemar(ours_flag, (pr == 1).astype(int), Y * bio)  # restrict to bio via Y*bio trick
    print(f"  {g:<14} recall={r:.3f} [{lo:.3f},{hi:.3f}]  McNemar vs ours p={p:.3f} (ours+{ao}/them+{bo})")

print("=" * 72)
print("(2) RESPONSE-harm recall (real_response_bio_large) with 95% CI (n=343 harmful)")
rrows = [json.loads(l) for l in open(ROOT / "data/external/real_response_bio_large.jsonl") if l.strip()]
RY = np.array([int(r["label"]) for r in rrows])
ours = json.load(open(ROOT / "results/v8bh_compare.json"))
pv = np.array(ours["large_v8bd"])
ours_rflag = (pv >= 0.5).astype(int)
rp = RY == 1
r, lo, hi, n = rate_ci(ours_rflag, rp)
print(f"  OURS v8bh @0.5  recall={r:.3f} [{lo:.3f},{hi:.3f}] over-ref(benign)={ (ours_rflag[RY==0].mean()):.3f}")
cpr = comp_preds("real_response_bio_large")
for g, pr in cpr.items():
    r, lo, hi, n = rate_ci(pr == 1, rp)
    p, ao, bo = mcnemar(ours_rflag, (pr == 1).astype(int), RY)
    print(f"  {g:<14} recall={r:.3f} [{lo:.3f},{hi:.3f}] over-ref={ (pr[RY==0]==1).mean():.3f}  McNemar p={p:.3f}")

print("=" * 72)
print("(3) PAIRED recall + over-refusal per DualModeGuard policy (real_response_bio_large)")
pp = np.array(ours["large_v8bd"])  # response head v8bh
# prompt scores on same set
prm = json.load(open(ROOT / "results/dualmode_step2.json"))  # may differ set; use v8bh_compare proxy
# use realresp_curve_data for prompt scores aligned to real set
rc = json.load(open(ROOT / "results/realresp_curve_data.json"))
pprompt = np.array(rc["p_prompt"])
fr = pp >= 0.5
fpm = pprompt >= 0.5
for name, flag in [("prompt_only", fpm), ("response_only", fr), ("and", fpm & fr), ("or", fpm | fr)]:
    rr, rlo, rhi, _ = rate_ci(flag, RY == 1)
    orr, olo, ohi, _ = rate_ci(flag, RY == 0)
    print(f"  {name:<14} recall={rr:.3f} [{rlo:.3f},{rhi:.3f}]  over-ref={orr:.3f} [{olo:.3f},{ohi:.3f}]")

print("=" * 72)
print("(4) FOOTPRINT: AUPRC + recall, student (prompt head) vs teacher on bio_clean_eval")
st = json.load(open(ROOT / "results/dualmode_step2.json"))
sp = np.array(st["scores"]["p_prompt"])
sy = np.array(st["scores"]["label"])
print(f"  student AUPRC={auprc(sp, sy):.3f}  recall@0.5={ (sp[sy==1]>=0.5).mean():.3f}")
th = json.load(open(ROOT / "results/v7c_prob_scores_aug2ln.json"))
tp = np.array([x["p_harmful"] for x in th["pos"]])
tn = np.array([x["p_harmful"] for x in th["neg"]])
ts = np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])
tsc = np.concatenate([tp, tn])
print(f"  teacher AUPRC={auprc(tsc, ts):.3f}  recall@0.5={ (tp>=0.5).mean():.3f}")
print("=" * 72)
