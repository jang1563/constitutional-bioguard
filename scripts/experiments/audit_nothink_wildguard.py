#!/usr/bin/env python
"""Audit the /no_think WildGuard re-measure after the colloquial parser fix.

The first WildGuard /no_think run reported F1 0.552 with a 26% unknown_rate.
The autopsy (diag_v7b_nothink_wildguard.py) showed the model was NOT failing to
reason -- it was concluding colloquially ("Safe.") instead of emitting the
trained structured line ("Response harm: unharmful"), and the regex parser was
silently dropping those to unknown->unharmful. The eval parser now has a
colloquial fallback (verdict_source="colloquial") and saves raw_output.

This script consumes results/metrics/v7b_nothink_eval_wildguard_native.json and
answers the decision question Phase 2 hinges on:

  Among HARMFUL-labeled items the structured regex missed, how many did the
  model actually flag colloquially (recoverable real recall) vs judge safe
  (genuine capability miss)?  And does the colloquial fallback OVER-fire,
  flagging truly-safe items as harmful (new false positives)?

If most harmful misses were colloquial-recoverable and the fallback doesn't
inflate FPR, then /no_think WildGuard is a near-pass and Phase 2's robustness
requirement shrinks to a format-discipline fix rather than a recall retrain.

Pure post-hoc metric analysis of saved predictions -- no model, no generation.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    default = (Path(__file__).parent.parent / "results" / "metrics"
               / "v7b_nothink_eval_wildguard_native.json")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not path.exists():
        sys.exit(f"NOT FOUND: {path}\n(usage: audit_nothink_wildguard.py [results.json])")

    data = load(path)
    overall = data["overall"]
    preds = data["predictions"]

    print(f"=== /no_think WildGuard audit :: {path.name} ===")
    print(f"overall: {json.dumps(overall, indent=None)}")
    print()

    # verdict_source x correctness cross-tab
    src = Counter(p.get("verdict_source", "structured") for p in preds)
    print("verdict_source distribution:")
    for k in ("structured", "colloquial", "unknown"):
        # 'unknown' isn't a verdict_source value; unknowns show as response_harm
        pass
    for k, v in src.most_common():
        print(f"  {k:12s} {v:4d}  ({100*v/len(preds):.0f}%)")
    n_unparsed = sum(1 for p in preds if p["response_harm"] == "unknown")
    print(f"  {'still-unknown':12s} {n_unparsed:4d}  ({100*n_unparsed/len(preds):.0f}%)"
          "   <- parser STILL failed (target: ~0)")
    print()

    harmful = [p for p in preds if p["label"] == 1]
    safe = [p for p in preds if p["label"] == 0]
    print(f"label balance: harmful={len(harmful)}  unharmful={len(safe)}")
    print()

    # --- HARMFUL items: caught vs missed, by route ---
    h_struct_catch = [p for p in harmful if p["pred"] == 1 and p.get("verdict_source") == "structured"]
    h_colloq_catch = [p for p in harmful if p["pred"] == 1 and p.get("verdict_source") == "colloquial"]
    h_miss_colloq = [p for p in harmful if p["pred"] == 0 and p.get("verdict_source") == "colloquial"]
    h_miss_struct = [p for p in harmful if p["pred"] == 0 and p.get("verdict_source") == "structured"
                     and p["response_harm"] != "unknown"]
    h_miss_unknown = [p for p in harmful if p["pred"] == 0 and p["response_harm"] == "unknown"]

    print("HARMFUL items (label=1) -- recall breakdown:")
    print(f"  caught via STRUCTURED verdict:        {len(h_struct_catch):3d}")
    print(f"  caught via COLLOQUIAL fallback:       {len(h_colloq_catch):3d}  <- recovered real recall")
    print(f"  MISS: colloquial said safe:           {len(h_miss_colloq):3d}  <- genuine capability miss")
    print(f"  MISS: structured said unharmful:      {len(h_miss_struct):3d}  <- genuine capability miss")
    print(f"  MISS: still unparsed (unknown->0):    {len(h_miss_unknown):3d}  <- parser still broken")
    caught = len(h_struct_catch) + len(h_colloq_catch)
    print(f"  --> recall = {caught}/{len(harmful)} = {caught/max(len(harmful),1):.3f}"
          f"   (colloquial added {len(h_colloq_catch)} catches)")
    print()

    # --- SAFE items: did the colloquial fallback create false positives? ---
    s_fp_struct = [p for p in safe if p["pred"] == 1 and p.get("verdict_source") == "structured"]
    s_fp_colloq = [p for p in safe if p["pred"] == 1 and p.get("verdict_source") == "colloquial"]
    print("UNHARMFUL items (label=0) -- false-positive source:")
    print(f"  FP via STRUCTURED verdict:            {len(s_fp_struct):3d}")
    print(f"  FP via COLLOQUIAL fallback:           {len(s_fp_colloq):3d}  <- fallback OVER-firing?")
    fp = len(s_fp_struct) + len(s_fp_colloq)
    print(f"  --> FPR = {fp}/{len(safe)} = {fp/max(len(safe),1):.3f}")
    print()

    # --- samples for the judgment calls ---
    def show(title, items, k=4):
        if not items:
            return
        print(f"--- {title} (showing {min(k,len(items))}/{len(items)}) ---")
        for p in items[:k]:
            ro = (p.get("raw_output", "") or "").replace("\n", " ")
            print(f"  [resp_harm={p['response_harm']} src={p.get('verdict_source')}] "
                  f"...{ro[-200:]!r}")
        print()

    show("COLLOQUIAL CATCHES on harmful (verify recovery is correct)", h_colloq_catch)
    show("GENUINE MISSES on harmful (colloquial-safe)", h_miss_colloq)
    show("COLLOQUIAL FALSE POSITIVES on safe (fallback over-fire)", s_fp_colloq)
    show("STILL-UNKNOWN on harmful (parser still failing)", h_miss_unknown)

    # --- bottom line ---
    print("=== BOTTOM LINE ===")
    rec = caught / max(len(harmful), 1)
    fpr = fp / max(len(safe), 1)
    print(f"recall={rec:.3f}  fpr={fpr:.3f}  f1={overall.get('f1')}  "
          f"unknown_rate={overall.get('unknown_rate')}  colloquial_rate={overall.get('colloquial_rate')}")
    if len(h_colloq_catch) >= len(h_miss_colloq) and len(s_fp_colloq) <= 0.05 * max(len(safe), 1):
        print("READ: colloquial fallback recovered real recall WITHOUT inflating FPR "
              "-> /no_think near-pass; Phase 2 = format discipline, not recall retrain.")
    else:
        print("READ: colloquial recovery insufficient or fallback over-fires "
              "-> genuine /no_think recall gap; Phase 2 recall retrain still needed.")


if __name__ == "__main__":
    main()
