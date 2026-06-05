#!/usr/bin/env python
"""Stratify Phase 3 OOD predictions by SaladBench/ALERT category to demonstrate
v4's bio-selectivity vs generalist baselines.

Output: per-category flag rate per model, with bio vs non-bio aggregate.

Key claim to verify:
  v4 (bio specialist) should have HIGH flag rate on O39 (Biological/Chemical)
  but LOW flag rate on non-bio CBRN categories (drugs, weapons, terrorism,
  malware, critical infra).

  WildGuard 7B / LLaMA-Guard 3 8B (generalist) should have HIGH flag rate
  on ALL CBRN categories indiscriminately.

This is the competitive-advantage demonstration for a bio specialist.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR


def load_predictions(model_prefix: str, bench: str) -> list[dict]:
    """Load per-item predictions, choosing the file with the most predictions."""
    candidates = [
        METRICS_DIR / f"v5_eval_{model_prefix}_{bench}.json",
        METRICS_DIR / f"phase3_{model_prefix}_{bench}.json",
    ]
    if model_prefix == "v4":
        candidates.append(METRICS_DIR / f"v4_eval_{bench}.json")
        candidates.append(METRICS_DIR / f"v4_compare_{bench}.json")
    # Pick the file with most predictions (avoid 1000-cap truncation)
    best = []
    for fp in candidates:
        if not fp.exists():
            continue
        d = json.load(open(fp))
        preds = d.get("predictions", [])
        if len(preds) > len(best):
            best = preds
    return best


def stratify_saladbench(models: list[str]) -> dict:
    fp = DATA_EXTERNAL / "saladbench_cbrn.jsonl"
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    print(f"\nSaladBench-CBRN: {len(rows)} items, {len(set(r.get('salad_category_3','') for r in rows))} unique categories")

    # Bio = O39, Non-bio = everything else in CBRN-flagged set
    bio_indices = [i for i, r in enumerate(rows)
                   if "O39" in r.get("salad_category_3", "")]
    nonbio_indices = [i for i, r in enumerate(rows)
                      if "O39" not in r.get("salad_category_3", "")]
    print(f"  Bio (O39): {len(bio_indices)} items")
    print(f"  Non-bio:   {len(nonbio_indices)} items")

    # Per-category breakdown
    by_cat = defaultdict(list)
    for i, r in enumerate(rows):
        cat = r.get("salad_category_3", "_uncat") or "_uncat"
        by_cat[cat].append(i)

    report = {"benchmark": "saladbench_cbrn", "n_total": len(rows),
              "models": {}}
    for model in models:
        preds = load_predictions(model, "saladbench_cbrn")
        if len(preds) != len(rows):
            print(f"  Skipping {model}: predictions={len(preds)} != n_rows={len(rows)}")
            continue
        flags = [p["pred"] for p in preds]

        # Bio vs non-bio aggregate
        bio_flag = sum(flags[i] for i in bio_indices)
        nonbio_flag = sum(flags[i] for i in nonbio_indices)
        bio_rate = bio_flag / max(len(bio_indices), 1)
        nonbio_rate = nonbio_flag / max(len(nonbio_indices), 1)

        # Per-category
        per_cat = {}
        for cat, idx_list in sorted(by_cat.items(), key=lambda x: -len(x[1])):
            cat_flag = sum(flags[i] for i in idx_list)
            per_cat[cat] = {
                "n": len(idx_list),
                "n_flag": cat_flag,
                "flag_rate": round(cat_flag / len(idx_list), 4),
            }

        report["models"][model] = {
            "bio_only_flag_rate": round(bio_rate, 4),
            "non_bio_flag_rate": round(nonbio_rate, 4),
            "bio_selectivity_ratio": (
                round(bio_rate / nonbio_rate, 4) if nonbio_rate > 0 else float("inf")
            ),
            "n_bio_flagged": bio_flag,
            "n_bio_total": len(bio_indices),
            "n_non_bio_flagged": nonbio_flag,
            "n_non_bio_total": len(nonbio_indices),
            "per_category": per_cat,
        }
    return report


def stratify_alert(models: list[str]) -> dict:
    fp = DATA_EXTERNAL / "alert_cbrn.jsonl"
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    cats = Counter(r.get("alert_category", "") for r in rows)
    print(f"\nALERT-CBRN: {len(rows)} items, {len(cats)} categories")
    for c, n in cats.most_common(10):
        print(f"  {n:5} {c}")

    # ALERT bio = "weapon_biological" or similar
    def is_bio(r):
        c = r.get("alert_category", "").lower()
        return ("biolog" in c) or ("bio_" in c) or c.endswith("_bio")
    bio_indices = [i for i, r in enumerate(rows) if is_bio(r)]
    nonbio_indices = [i for i, r in enumerate(rows) if not is_bio(r)]
    print(f"  Bio:      {len(bio_indices)} items")
    print(f"  Non-bio:  {len(nonbio_indices)} items")

    report = {"benchmark": "alert_cbrn", "n_total": len(rows),
              "models": {}}
    for model in models:
        preds = load_predictions(model, "alert_cbrn")
        if len(preds) != len(rows):
            print(f"  Skipping {model}: preds={len(preds)} != n_rows={len(rows)}")
            continue
        flags = [p["pred"] for p in preds]
        bio_flag = sum(flags[i] for i in bio_indices)
        nonbio_flag = sum(flags[i] for i in nonbio_indices)
        bio_rate = bio_flag / max(len(bio_indices), 1)
        nonbio_rate = nonbio_flag / max(len(nonbio_indices), 1)
        report["models"][model] = {
            "bio_only_flag_rate": round(bio_rate, 4),
            "non_bio_flag_rate": round(nonbio_rate, 4),
            "bio_selectivity_ratio": (
                round(bio_rate / nonbio_rate, 4) if nonbio_rate > 0 else float("inf")
            ),
            "n_bio_flagged": bio_flag,
            "n_bio_total": len(bio_indices),
            "n_non_bio_flagged": nonbio_flag,
            "n_non_bio_total": len(nonbio_indices),
        }
    return report


def main():
    models = ["v3", "v4", "wildguard_7b", "llama_guard_3_8b", "v5_baseline", "v5"]

    salad = stratify_saladbench(models)
    alert = stratify_alert(models)

    # Print comparison
    print("\n" + "=" * 80)
    print("BIO-SELECTIVITY: Bio-only flag rate vs Non-bio flag rate")
    print("=" * 80)
    print(f"\n{'Model':<22} {'SaladBench bio%':>16} {'non-bio%':>10} {'ratio':>8}  |  {'ALERT bio%':>12} {'non-bio%':>10}")
    print('-' * 100)
    for model in models:
        s = salad["models"].get(model, {})
        a = alert["models"].get(model, {})
        if not s and not a:
            continue
        def fmt(d, k):
            v = d.get(k)
            if v is None: return "n/a"
            if k.endswith("_rate"):
                return f"{v*100:.1f}%"
            return f"{v:.2f}"
        line = f"{model:<22}"
        line += f" {fmt(s, 'bio_only_flag_rate'):>15}"
        line += f" {fmt(s, 'non_bio_flag_rate'):>10}"
        line += f" {fmt(s, 'bio_selectivity_ratio'):>8}"
        line += f"  |  {fmt(a, 'bio_only_flag_rate'):>12}"
        line += f" {fmt(a, 'non_bio_flag_rate'):>10}"
        print(line)

    out = METRICS_DIR / "stratified_bio_selectivity.json"
    with open(out, "w") as f:
        json.dump({"saladbench": salad, "alert": alert}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
