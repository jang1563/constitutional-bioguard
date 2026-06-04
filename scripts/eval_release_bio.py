#!/usr/bin/env python
"""Phase R.1 release-readiness: bio recall at scale + response axis, OOD-tagged.

Replaces the n=9 wildguard OOD bio cell. Recall sets are tagged IN-DIST (used in
pdual_v3 POS_FILES: saladbench_cbrn/alert_cbrn/advbench_bio/harmbench_bio/
jailbreakbench_bio/clearharm_bio) vs OOD (held out from the prompt head). The
lexicon is rule-based so lex recall is OOD on EVERY set; only the learned head
carries train/test contamination, so we break out lex / learned / hybrid per set.
Headline OOD bio recall = scisafeeval_bio + simple_safety_bio (neither trained).
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np  # noqa: E402

from constitutional_bioguard.config import DATA_PROCESSED, DATA_EXTERNAL  # noqa: E402
from constitutional_bioguard.dual_mode import DualModeGuard  # noqa: E402


def load(name):
    for d in (DATA_PROCESSED, DATA_EXTERNAL):
        p = d / f"{name}.jsonl"
        if p.exists():
            return [json.loads(line) for line in open(p) if line.strip()]
    return []


def qfield(r):
    for k in ("query", "prompt", "text", "instruction", "behavior", "goal"):
        if k in r and r[k]:
            return str(r[k])
    return ""


def boot_ci(flags, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(flags, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    means = [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# all label=1 bio-harm prompt sets; tag = OOD (held out of pdual) vs in-dist (trained)
RECALL_SETS = [
    ("scisafeeval_bio", "OOD"),
    ("simple_safety_bio", "OOD"),
    ("harmbench_bio", "in-dist"),
    ("advbench_bio", "in-dist"),
    ("clearharm_bio", "in-dist"),
    ("alert_cbrn_strict", "in-dist"),
    ("saladbench_cbrn_strict", "in-dist"),
]


def main():
    guard = DualModeGuard()
    out = {}

    print("=== PROMPT-AXIS BIO RECALL (label=1; higher=better) ===")
    print(f"  {'tag':8} {'set':24} {'n':>5}  {'recall':>6} {'[95% CI]':>15}  "
          f"{'lex':>5} {'learn':>5}")
    for name, tag in RECALL_SETS:
        rows = load(name)
        if not rows:
            print(f"  {tag:8} {name:24}  MISSING")
            continue
        qs = [qfield(r) for r in rows]
        v = guard.classify_batch(qs)
        flags = np.array([int(x.prompt_flag) for x in v])
        src = Counter(x.prompt_source for x in v)
        lex_rec = float(np.mean([x.prompt_source in ("lex", "both") for x in v]))
        learn_rec = float(np.mean([x.prompt_source in ("learned", "both") for x in v]))
        lo, hi = boot_ci(flags)
        print(f"  {tag:8} {name:24} {len(qs):5}  {flags.mean():.3f} "
              f"[{lo:.3f},{hi:.3f}]  {lex_rec:.3f} {learn_rec:.3f}")
        out[name] = {"tag": tag, "n": len(qs), "recall": float(flags.mean()),
                     "ci95": [lo, hi], "lex_rec": lex_rec, "learned_rec": learn_rec,
                     "sources": dict(src)}

    # OOD headline: pooled scisafeeval + simple_safety
    ood = [n for n, t in RECALL_SETS if t == "OOD" and n in out]
    if ood:
        pooled = []
        for name in ood:
            rows = load(name)
            qs = [qfield(r) for r in rows]
            v = guard.classify_batch(qs)
            pooled.extend(int(x.prompt_flag) for x in v)
        lo, hi = boot_ci(pooled)
        print(f"\n  >>> OOD POOLED ({'+'.join(ood)}) n={len(pooled)} "
              f"recall={np.mean(pooled):.3f} [{lo:.3f},{hi:.3f}]")
        out["_ood_pooled"] = {"sets": ood, "n": len(pooled),
                              "recall": float(np.mean(pooled)), "ci95": [lo, hi]}

    # response axis on real_response_bio (mixed labels -> TPR & FPR)
    rr = load("real_response_bio")
    if rr:
        qs = [qfield(r) for r in rr]
        rs = [str(r.get("response", "")) for r in rr]
        labs = np.array([int(r.get("label", 0)) for r in rr])
        v = guard.classify_batch(qs, rs)
        pf = np.array([int(x.prompt_flag) for x in v])
        rf = np.array([int(x.response_flag) for x in v])
        jf = np.array([int(x.joint_flag) for x in v])

        def rates(mask):
            return {ax: round(float(a[mask].mean()), 3)
                    for ax, a in (("prompt", pf), ("response", rf), ("joint", jf))}
        print(f"\n=== RESPONSE AXIS real_response_bio (n={len(rr)}, "
              f"pos={int(labs.sum())}) ===")
        tpr = rates(labs == 1)
        fpr = rates(labs == 0)
        print(f"  TPR {tpr}")
        print(f"  FPR {fpr}")
        out["real_response_bio"] = {"n": len(rr), "pos": int(labs.sum()),
                                    "TPR": tpr, "FPR": fpr}

    op = DATA_PROCESSED.parent / "results" / "release_bio_eval.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w"), indent=2)
    print(f"\nSaved: {op}")


if __name__ == "__main__":
    main()
