#!/usr/bin/env python
"""V6 F.3: Cascade simulation using cached predictions.

Since all three models (v4, WildGuard 7B, LLaMA-Guard 3 8B) have predictions
cached on the locked eval distributions, we can simulate cascade behavior
purely offline (no GPU). For each (Stage1, τ_1, w) configuration:

  z_final = w · logit(p_s1) + (1 - w) · logit(p_s2)
  flag = 1 if sigmoid(z_final) >= 0.5

We sweep:
  - Stage1 ∈ {WildGuard 7B, LLaMA-Guard 3 8B}
  - τ_1 ∈ [0.0, 1.0] step 0.1 (route to Stage2 if p_s1 >= τ_1)
  - w ∈ {0.0, 0.25, 0.5, 0.55, 0.75, 1.0} (fusion weight)
  - Cascade variants: handoff vs weighted fusion

For each config, compute on the locked eval suite:
  - Bio recall (SaladBench O39 subset)
  - Bio selectivity ratio
  - OR-Bench-Hard FPR
  - XSTest FPR
  - Latency (in p_s1 cost units, since Stage1 always runs)

Report Pareto curve. Anthropic CC++ uses (0.55/0.45 weighted fusion) — we
report that point as the "principled default" plus best-on-bio-F1 sweep.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_clip = np.clip(p, eps, 1 - eps)
    return np.log(p_clip / (1 - p_clip))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def load_predictions(prefix_candidates: list[str], bench: str) -> tuple[list[int], list[float], list[str]]:
    """Load (labels, probs, categories) — choose file with MOST predictions.

    Multiple candidate files may exist; v5_eval_* files were capped at 1000
    predictions but v4_eval_* / phase3_* have full predictions. Pick the
    one with the most data.
    """
    best = ([], [], [])
    for prefix in prefix_candidates:
        fp = METRICS_DIR / f"{prefix}_{bench}.json"
        if not fp.exists():
            continue
        d = json.load(open(fp))
        preds = d.get("predictions", [])
        if not preds:
            continue
        if len(preds) <= len(best[0]):
            continue
        best = (
            [int(p.get("label", 0)) for p in preds],
            [float(p.get("prob", 0)) for p in preds],
            [str(p.get("category", "")) for p in preds],
        )
    return best


def evaluate_cascade(p_s1: np.ndarray, p_s2: np.ndarray, labels: np.ndarray,
                     tau_1: float, w: float) -> dict:
    """Pure weighted-fusion cascade (CC++ style).

    All items get Stage2 run; final = weighted logit fusion.
    tau_1 is currently informational (could gate Stage2 to save compute,
    but we always have both predictions cached so use both).
    """
    z_s1 = logit(p_s1)
    z_s2 = logit(p_s2)
    z_final = w * z_s1 + (1 - w) * z_s2
    p_final = sigmoid(z_final)
    preds = (p_final >= 0.5).astype(int)
    n = len(labels)
    n_flag = int(preds.sum())
    result = {
        "n": n,
        "n_flag": n_flag,
        "flag_rate": round(float(preds.mean()), 4),
    }
    if (labels == 1).any() and (labels == 0).any():
        result["precision"] = round(float(precision_score(labels, preds, zero_division=0)), 4)
        result["recall"] = round(float(recall_score(labels, preds, zero_division=0)), 4)
        result["f1"] = round(float(f1_score(labels, preds, zero_division=0)), 4)
        result["fpr"] = round(float(preds[labels == 0].mean()) if (labels == 0).any() else 0, 4)
    return result


def cascade_simulation_on_saladbench(stage1_prefix: str, w: float) -> dict:
    """Special handling: SaladBench has O39 bio subset + non-bio for selectivity."""
    # Load
    bench = "saladbench_cbrn"
    labels_s2, probs_s2, _ = load_predictions(
        ["v4_eval", "v5_eval_v4", "phase3_v4"], bench,
    )
    labels_s1, probs_s1, cats_s1 = load_predictions([f"phase3_{stage1_prefix}"], bench)
    if not labels_s2 or not labels_s1 or len(labels_s2) != len(labels_s1):
        return {"error": f"missing/mismatched preds: s2={len(labels_s2)} s1={len(labels_s1)}"}
    # v4_eval files don't have category; load from source SaladBench jsonl
    sb_path = DATA_EXTERNAL / "saladbench_cbrn.jsonl"
    cats = cats_s1
    if not cats or len(cats) != len(labels_s2):
        if sb_path.exists():
            cats = [json.loads(l).get("salad_category_3", "")
                    for l in open(sb_path) if l.strip()]
            cats = cats[:len(labels_s2)]
        else:
            cats = ["" for _ in labels_s2]

    p_s1 = np.array(probs_s1)
    p_s2 = np.array(probs_s2)

    z_final = w * logit(p_s1) + (1 - w) * logit(p_s2)
    preds = (sigmoid(z_final) >= 0.5).astype(int)

    # Stratify by O39 vs non-O39
    bio_idx = [i for i, c in enumerate(cats) if "O39" in c]
    nonbio_idx = [i for i, c in enumerate(cats) if "O39" not in c]
    bio_flag = sum(preds[i] for i in bio_idx)
    nonbio_flag = sum(preds[i] for i in nonbio_idx)
    bio_rate = bio_flag / max(len(bio_idx), 1)
    nonbio_rate = nonbio_flag / max(len(nonbio_idx), 1)
    return {
        "stage1": stage1_prefix,
        "w": w,
        "n_bio": len(bio_idx),
        "n_nonbio": len(nonbio_idx),
        "bio_recall": round(float(bio_rate), 4),
        "nonbio_fpr": round(float(nonbio_rate), 4),
        "selectivity_ratio": round(float(bio_rate / nonbio_rate), 4) if nonbio_rate > 0 else None,
    }


def cascade_eval_full_suite(stage1_prefix: str, w: float) -> dict:
    """Evaluate cascade on all locked eval benchmarks."""
    BENCHES = [
        ("or_bench_hard_1k", ["v5_eval_v4"], None, "fpr"),
        ("false_reject_test", ["v5_eval_v4"], None, "fpr"),
        ("xstest", ["v5_eval_v4"], None, "fpr"),
        ("simple_safety_bio", ["v5_eval_v4"], None, "recall"),
        ("wildguard_native", ["v5_eval_v4"], None, "f1"),
        ("alert_cbrn", ["v5_eval_v4"], None, "recall"),
    ]
    results = {}
    for bench, v4_prefixes, _, _ in BENCHES:
        labels_s2, probs_s2, _ = load_predictions(v4_prefixes, bench)
        labels_s1, probs_s1, _ = load_predictions([f"phase3_{stage1_prefix}"], bench)
        if not labels_s2 or not labels_s1 or len(labels_s2) != len(labels_s1):
            results[bench] = {"error": "missing or mismatched preds"}
            continue
        labels = np.array(labels_s2)
        p_s1 = np.array(probs_s1)
        p_s2 = np.array(probs_s2)
        cascade_metrics = evaluate_cascade(p_s1, p_s2, labels, tau_1=0.5, w=w)
        results[bench] = cascade_metrics
    # SaladBench separately for selectivity
    results["saladbench_stratified"] = cascade_simulation_on_saladbench(stage1_prefix, w)
    return results


def main():
    report = {"description": "Cascade simulation via cached predictions"}

    # Sweep w grid for both Stage1 choices
    weight_grid = [0.0, 0.25, 0.45, 0.50, 0.55, 0.75, 1.0]
    # Note: w=0.0 means pure Stage2 (v4 alone), w=1.0 means pure Stage1
    # CC++ default = 0.55 (Stage1) / 0.45 (Stage2) → w=0.55

    for stage1 in ["wildguard_7b", "llama_guard_3_8b"]:
        report[stage1] = {}
        for w in weight_grid:
            logging.info("Cascade: stage1=%s w=%.2f", stage1, w)
            res = cascade_eval_full_suite(stage1, w)
            report[stage1][f"w_{w:.2f}"] = res

    # Summary table — selectivity at CC++ default w=0.55
    print("\n" + "=" * 100)
    print(f"{'Stage1':<22} {'w':<6} {'Bio recall (O39)':>18} {'Non-bio FPR':>15} {'Selectivity':>13} {'OR-Hard FPR':>13}")
    print("-" * 100)
    for stage1 in ["wildguard_7b", "llama_guard_3_8b"]:
        for w in weight_grid:
            res = report[stage1][f"w_{w:.2f}"]
            sal = res.get("saladbench_stratified", {})
            orh = res.get("or_bench_hard_1k", {})
            print(f"{stage1:<22} {w:<6.2f} "
                  f"{sal.get('bio_recall', 0)*100:>17.1f}% "
                  f"{sal.get('nonbio_fpr', 0)*100:>14.1f}% "
                  f"{sal.get('selectivity_ratio', 0) or 0:>13.2f} "
                  f"{orh.get('flag_rate', 0)*100:>12.1f}%")

    out = METRICS_DIR / "v6_f3_cascade_simulation.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
