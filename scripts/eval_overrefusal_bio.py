#!/usr/bin/env python
"""Benign-bio over-refusal eval for the dual-mode PROMPT axis (P3 review item a).

The wildguard bio_benign cell in eval_dual_mode was n=3 -- statistically
meaningless. This measures prompt-axis over-refusal on real benign-bio and
dual-use-bio prompt sets at scale. Lower flag_rate = better; every prompt here
is one we should NOT flag (WMDP-bio stems are the dual-use boundary -- sparse
flagging tolerated, not zero). Source breakdown shows whether a flag came from
the lexicon (lex) or the learned head (learned).
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np  # noqa: E402

from constitutional_bioguard.config import (  # noqa: E402
    CONFIGS_DIR, DATA_EXTERNAL, DATA_PROCESSED,
)
from constitutional_bioguard.dual_mode import DualModeGuard, has_bio_context  # noqa: E402

BIO_KW = [k.lower() for k in
          json.load(open(CONFIGS_DIR / "bio_keywords_adv.json")).get("keywords", [])]


def is_bio(t):
    return any(k in str(t or "").lower() for k in BIO_KW)


def find(name):
    for d in (DATA_PROCESSED, DATA_EXTERNAL):
        p = d / f"{name}.jsonl"
        if p.exists():
            return p
    return None


def load(name):
    p = find(name)
    if not p:
        print(f"  ({name} NOT FOUND)")
        return []
    return [json.loads(line) for line in open(p) if line.strip()]


def report(guard, name, prompts, out):
    if not prompts:
        print(f"  {name}: (empty after filter)")
        return
    v = guard.classify_batch(prompts)
    pf = np.array([int(x.prompt_flag) for x in v])
    bc = np.array([int(has_bio_context(p)) for p in prompts])
    src = Counter(x.prompt_source for x in v)
    print(f"  {name} (n={len(prompts)}, bio_ctx={bc.mean():.2f}): "
          f"flag_rate={pf.mean():.3f}  sources={dict(src)}")
    out[name.strip()] = {"n": len(prompts), "flag_rate": float(pf.mean()),
                         "bio_ctx_rate": float(bc.mean()), "sources": dict(src)}


def main():
    guard = DualModeGuard()
    out = {}

    print("=== benign-bio over-refusal (prompt axis; lower=better) ===")

    # 1) OR-Bench health: benign-but-toxic-looking health/bio prompts (label==0)
    orh = [r for r in load("or_bench_health") if int(r.get("label", 0)) == 0]
    report(guard, "orbench_health_benign    ", [r["query"] for r in orh], out)
    report(guard, "orbench_health_benign_bio",
           [r["query"] for r in orh if is_bio(r["query"])], out)

    # 2) OR-Bench-Hard-1k: benign-but-scary, bio subset (label==0)
    orhd = [r for r in load("or_bench_hard_1k")
            if int(r.get("label", 0)) == 0 and is_bio(r.get("query", ""))]
    report(guard, "orbench_hard_benign_bio  ", [r["query"] for r in orhd], out)

    # 3) WMDP-bio: dual-use research stems (dedup by question_idx) -- boundary set
    wm = load("wmdp_bio")
    seen, stems = set(), []
    for r in wm:
        qi = r.get("question_idx", r.get("query", ""))
        if qi in seen:
            continue
        seen.add(qi)
        stems.append(r.get("query", ""))
    report(guard, "wmdp_bio_stems_dualuse   ", stems, out)

    # 4) XSTest: safe-bio contrast prompts (label==0, is_bio)
    xs = load("xstest")
    xs_safe = [r["query"] for r in xs
               if int(r.get("label", 0)) == 0 and is_bio(r.get("query", ""))]
    report(guard, "xstest_safe_bio          ", xs_safe, out)

    op = DATA_PROCESSED.parent / "results" / "overrefusal_bio_eval.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w"), indent=2)
    print(f"\nSaved: {op}")


if __name__ == "__main__":
    main()
