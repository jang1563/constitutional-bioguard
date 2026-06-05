#!/usr/bin/env python
"""Phase 2 evaluation: v3 + WildGuard + LLaMA-Guard on extended benchmarks.

Phase 2A:
  - HarmBench FULL (per-category: 7 semantic categories)
  - AdvBench FULL (all harm types)
  - WildGuardTest with native prompt_harm_label

Phase 2B:
  - XSTest (over-refusal benchmark: 250 SAFE + 200 UNSAFE)
  - BeaverTails subset (multi-category harm)

Outputs:
  - results/metrics/phase2_{model}_{benchmark}.json (per-benchmark)
  - results/metrics/phase2_{model}_summary.json (aggregated per model)
  - results/metrics/phase2_domain_coverage.json (matrix data for heatmap)

Usage:
    python scripts/experiments/evaluate_phase2.py --model v3
    python scripts/experiments/evaluate_phase2.py --model wildguard_7b
    python scripts/experiments/evaluate_phase2.py --model llama_guard_3_8b
    python scripts/experiments/evaluate_phase2.py --model all  # sequential, big job
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from constitutional_bioguard.config import (
    DATA_EXTERNAL,
    METRICS_DIR,
    MODELS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Benchmark loaders ───────────────────────────────────────────────────────

def load_harmbench_full() -> list[dict]:
    path = DATA_EXTERNAL / "harmbench_full.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            "Run: python scripts/experiments/cache_phase2_data.py harmbench_full"
        )
    return load_jsonl(path)


def load_advbench_full() -> list[dict]:
    path = DATA_EXTERNAL / "advbench_full.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            "Run: python scripts/experiments/cache_phase2_data.py advbench_full"
        )
    return load_jsonl(path)


def load_xstest() -> list[dict]:
    path = DATA_EXTERNAL / "xstest.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            "Run: python scripts/experiments/cache_phase2_data.py xstest"
        )
    return load_jsonl(path)


def load_beavertails() -> list[dict]:
    path = DATA_EXTERNAL / "beavertails_subset.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            "Run: python scripts/experiments/cache_phase2_data.py beavertails"
        )
    return load_jsonl(path)


def load_wildguard_native() -> list[dict]:
    """Load WildGuardTest with native labels (mixed harm/safe)."""
    path = DATA_EXTERNAL / "wildguard_test.jsonl"
    rows = load_jsonl(path)
    # Filter to items with prompt_harm_label set (handle None/missing safely)
    filtered = []
    for r in rows:
        lbl = r.get("prompt_harm_label")
        if not isinstance(lbl, str):
            continue
        if lbl.lower() not in ("harmful", "unharmful"):
            continue
        r["query"] = r.get("prompt", "")
        r["label"] = 1 if lbl.lower() == "harmful" else 0
        filtered.append(r)
    return filtered


# ── Inference dispatch ──────────────────────────────────────────────────────

def predict_v3(rows: list[dict]) -> tuple[list[int], list[float]]:
    """Run v3 inference on rows."""
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer,
        predict_batch,
    )
    model_dir = MODELS_DIR / "deberta_bioguard_v3_balanced"
    model, tokenizer = load_model_and_tokenizer(model_dir)
    queries = [r.get("query") or r.get("prompt", "") for r in rows]
    responses = [r.get("response", "") for r in rows]
    preds = predict_batch(
        model=model, tokenizer=tokenizer,
        queries=queries, responses=responses,
        normalize=True,
    )
    del model, tokenizer
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pred_labels = [p[0] for p in preds]
    probs = [p[2] for p in preds]
    return pred_labels, probs


def predict_baseline(model_name: str, rows: list[dict]) -> tuple[list[int], list[float]]:
    """Run external baseline (WildGuard or LLaMA-Guard) inference."""
    from constitutional_bioguard.evaluation.external_baselines import get_baseline
    clf = get_baseline(model_name)
    clf.load()
    queries = [r.get("query") or r.get("prompt", "") for r in rows]
    responses = [r.get("response", "") for r in rows]
    preds = clf.predict_batch(queries, responses)
    clf.unload()
    pred_labels = [p[0] for p in preds]
    probs = [p[1] for p in preds]
    return pred_labels, probs


def predict(model: str, rows: list[dict]) -> tuple[list[int], list[float]]:
    if model == "v3":
        return predict_v3(rows)
    return predict_baseline(model, rows)


# ── Metric computation ──────────────────────────────────────────────────────

def compute_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    probs: list[float],
) -> dict:
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    n = len(y_true)

    result = {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_flagged": int(y_pred.sum()),
        "flag_rate": round(float(y_pred.mean()), 4),
    }
    if n_pos > 0 and n_neg > 0:
        result.update({
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "fpr": round(float(y_pred[y_true == 0].mean()), 4),
        })
        try:
            result["auroc"] = round(float(roc_auc_score(y_true, probs)), 4)
            result["auprc"] = round(float(average_precision_score(y_true, probs)), 4)
            result["auprc_random_baseline"] = round(n_pos / n, 4)
        except Exception:
            pass
    elif n_pos > 0:  # all positive
        result["recall"] = result["flag_rate"]
    elif n_neg > 0:  # all negative
        result["fpr"] = result["flag_rate"]
    return result


def evaluate_per_category(
    rows: list[dict],
    pred_labels: list[int],
    probs: list[float],
    category_field: str,
) -> dict:
    """Per-category breakdown."""
    by_cat: dict[str, dict] = {}
    cats = set(r.get(category_field, "_unknown") for r in rows)
    for cat in cats:
        idx = [i for i, r in enumerate(rows) if r.get(category_field, "_unknown") == cat]
        if not idx:
            continue
        sub_true = [rows[i]["label"] for i in idx]
        sub_pred = [pred_labels[i] for i in idx]
        sub_probs = [probs[i] for i in idx]
        by_cat[cat] = compute_metrics(sub_true, sub_pred, sub_probs)
    return by_cat


# ── Main eval driver ────────────────────────────────────────────────────────

BENCHMARKS = {
    "harmbench_full":     (load_harmbench_full, "semantic_category"),
    "advbench_full":      (load_advbench_full, None),
    "xstest":             (load_xstest, "type"),
    "beavertails":        (load_beavertails, "primary_category"),
    "wildguard_native":   (load_wildguard_native, "subcategory"),
}


def evaluate_model(model: str) -> dict:
    """Run model on all Phase 2 benchmarks."""
    logger.info("=" * 60)
    logger.info("PHASE 2: %s", model)
    logger.info("=" * 60)

    summary = {"model": model, "benchmarks": {}}

    for bench_name, (loader, cat_field) in BENCHMARKS.items():
        logger.info("\n[%s on %s]", model, bench_name)
        try:
            rows = loader()
        except Exception as e:
            logger.error("Failed to load %s: %s", bench_name, e)
            summary["benchmarks"][bench_name] = {"error": str(e)}
            continue

        n_pos = sum(r.get("label", 0) for r in rows)
        logger.info(
            "  %d items, %d positive, %d negative",
            len(rows), n_pos, len(rows) - n_pos,
        )

        pred_labels, probs = predict(model, rows)
        true_labels = [r["label"] for r in rows]

        overall = compute_metrics(true_labels, pred_labels, probs)
        per_cat = (
            evaluate_per_category(rows, pred_labels, probs, cat_field)
            if cat_field else None
        )

        result = {
            "overall": overall,
            "by_category": per_cat,
        }
        # Save per-item predictions WITH metadata for cascade simulation.
        # Critical: include category fields so downstream routing can use the
        # actual semantic category (NOT the model's own confidence) as the
        # gate signal — avoids circular dependency in calibrated routing.
        result["predictions"] = [
            {
                "label": t,
                "pred": p,
                "prob": float(pr),
                "category": r.get(cat_field, "") if cat_field else "",
                "subcategory": r.get("subcategory", ""),
                "semantic_category": r.get("semantic_category", ""),
                "primary_category": r.get("primary_category", ""),
                "type": r.get("type", ""),
                "adversarial": r.get("adversarial", False),
            }
            for r, t, p, pr in zip(rows, true_labels, pred_labels, probs)
        ]

        out_path = METRICS_DIR / f"phase2_{model}_{bench_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("  Saved: %s", out_path)

        # Log overall + first few category stats
        if "f1" in overall:
            logger.info(
                "  Overall: F1=%.4f AUROC=%s recall=%.4f fpr=%.4f",
                overall["f1"], overall.get("auroc", "N/A"),
                overall["recall"], overall["fpr"],
            )
        else:
            logger.info("  Overall flag_rate: %.4f (%d/%d)",
                        overall["flag_rate"], overall["n_flagged"], overall["n"])

        if per_cat:
            logger.info("  By category (top 5 by n):")
            for cat, m in sorted(
                per_cat.items(), key=lambda x: -x[1]["n"]
            )[:5]:
                if "f1" in m:
                    logger.info(
                        "    %-40s n=%d F1=%.4f recall=%.4f",
                        cat, m["n"], m["f1"], m["recall"],
                    )
                else:
                    logger.info(
                        "    %-40s n=%d flag_rate=%.4f",
                        cat, m["n"], m["flag_rate"],
                    )

        # Drop predictions from summary to keep it small
        summary["benchmarks"][bench_name] = {
            "overall": overall, "by_category": per_cat,
        }

    summary_path = METRICS_DIR / f"phase2_{model}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\nSummary: %s", summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["v3", "wildguard_7b", "llama_guard_3_8b", "all"],
    )
    args = parser.parse_args()

    if args.model == "all":
        for m in ["v3", "wildguard_7b", "llama_guard_3_8b"]:
            evaluate_model(m)
    else:
        evaluate_model(args.model)


if __name__ == "__main__":
    main()
