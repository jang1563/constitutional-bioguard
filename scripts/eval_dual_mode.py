#!/usr/bin/env python
"""End-to-end dual-mode bio guard eval (P3).

Reports per-axis (prompt-only, response-only) and joint metrics on the same
wildguard_test set used for the standalone evals, plus the response held-out v2
eval set. Confirms the hybrid prompt-head + v2 response-head pipeline produces
sensible joint decisions.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from constitutional_bioguard.config import CONFIGS_DIR, DATA_EXTERNAL, DATA_PROCESSED
from constitutional_bioguard.dual_mode import DualModeGuard

BIO_KW = [k.lower() for k in json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]


def is_bio(t):
    return any(k in str(t or "").lower() for k in BIO_KW)


_CYBER = ["malware", "ransomware", "computer virus", "cyber", "hacker",
          "zero-day", "antivirus", "firewall", "network security", "payload"]
_GAME = ["plague inc", "strategy game"]


def is_cyber_or_game(t):
    tl = str(t or "").lower()
    return any(k in tl for k in _CYBER) or any(k in tl for k in _GAME)


def main():
    guard = DualModeGuard()
    out = {}

    # 1) OOD wildguard_test prompt-only (clean bio-harm subset)
    wg = [json.loads(l) for l in open(DATA_EXTERNAL / "wildguard_test.jsonl") if l.strip()]
    def phl(r):
        v = r.get("prompt_harm_label")
        return v.lower() if isinstance(v, str) else None

    bio_harm = [r.get("prompt","") for r in wg
                if phl(r) == "harmful" and is_bio(r.get("prompt",""))
                and not is_cyber_or_game(r.get("prompt",""))]
    nonbio_harm = [r.get("prompt","") for r in wg
                   if phl(r) == "harmful" and not is_bio(r.get("prompt",""))]
    bio_benign = [r.get("prompt","") for r in wg
                  if phl(r) == "unharmful" and is_bio(r.get("prompt",""))]

    def prompt_axis(name, qs):
        if not qs: return
        v = guard.classify_batch(qs)
        pf = np.array([int(x.prompt_flag) for x in v])
        srcs = [x.prompt_source for x in v]
        from collections import Counter
        cnt = Counter(srcs)
        print(f"  {name} (n={len(qs)}): prompt_flag_rate={pf.mean():.3f}  sources={dict(cnt)}")
        out[name] = {"n": len(qs), "flag_rate": float(pf.mean()), "sources": dict(cnt)}

    print("=== prompt-axis OOD wildguard_test (cleaned) ===")
    prompt_axis("bio_harm  ", bio_harm)
    prompt_axis("nonbio_harm", nonbio_harm)
    prompt_axis("bio_benign", bio_benign)

    # 2) In-dist pdual_val (positives + bio benign negatives)
    val = [json.loads(l) for l in open(DATA_PROCESSED / "pdual_val.jsonl") if l.strip()]
    pos = [r["query"] for r in val if int(r["label"]) == 1]
    neg = [r["query"] for r in val if int(r["label"]) == 0]
    print("\n=== prompt-axis in-dist pdual_val ===")
    prompt_axis("pos       ", pos)
    prompt_axis("neg       ", neg)

    # 3) Joint (prompt + response) on a bio-eval set if available
    test_jsonls = [
        DATA_PROCESSED / "pair_test.jsonl",
        DATA_PROCESSED / "v2_test.jsonl",
        DATA_PROCESSED / "test.jsonl",
    ]
    test_path = next((p for p in test_jsonls if p.exists()), None)
    if test_path:
        rows = [json.loads(l) for l in open(test_path) if l.strip()]
        rows = rows[:500]  # cap for speed
        qs = [r.get("query", r.get("prompt", "")) for r in rows]
        rs = [r.get("response", "") for r in rows]
        labs = np.array([int(r.get("label", 0)) for r in rows])
        v = guard.classify_batch(qs, rs)
        pf = np.array([int(x.prompt_flag) for x in v])
        rf = np.array([int(x.response_flag) for x in v])
        jf = np.array([int(x.joint_flag) for x in v])
        # joint TPR on label=1 (true unsafe responses), joint FPR on label=0
        if labs.sum():
            tpr_p = pf[labs == 1].mean()
            tpr_r = rf[labs == 1].mean()
            tpr_j = jf[labs == 1].mean()
        else:
            tpr_p = tpr_r = tpr_j = float("nan")
        if (labs == 0).sum():
            fpr_p = pf[labs == 0].mean()
            fpr_r = rf[labs == 0].mean()
            fpr_j = jf[labs == 0].mean()
        else:
            fpr_p = fpr_r = fpr_j = float("nan")
        print(f"\n=== joint axis on {test_path.name} (n={len(rows)}, pos={int(labs.sum())}) ===")
        print(f"  TPR  prompt={tpr_p:.3f}  response={tpr_r:.3f}  joint={tpr_j:.3f}")
        print(f"  FPR  prompt={fpr_p:.3f}  response={fpr_r:.3f}  joint={fpr_j:.3f}")
        out["joint"] = {"n": len(rows), "pos": int(labs.sum()),
                        "TPR": {"prompt": float(tpr_p), "response": float(tpr_r), "joint": float(tpr_j)},
                        "FPR": {"prompt": float(fpr_p), "response": float(fpr_r), "joint": float(fpr_j)}}
    else:
        print("\n(joint axis skipped: no pair_test/v2_test/test jsonl found)")

    # 4) save
    out_path = DATA_PROCESSED.parent / "results" / "dual_mode_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
