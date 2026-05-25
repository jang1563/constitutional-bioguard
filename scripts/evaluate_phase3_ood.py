#!/usr/bin/env python
"""Phase 3: OOD bio benchmark evaluation (SaladBench, ALERT, OR-Bench, SST).

Runs v3 + WildGuard + LLaMA-Guard 3 on independent OOD bio benchmarks.
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
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from constitutional_bioguard.config import DATA_EXTERNAL, METRICS_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(true_labels: list[int], pred_labels: list[int],
                    probs: list[float]) -> dict:
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    n = len(y_true)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    n_flagged = int(y_pred.sum())
    result = {"n": n, "n_positive": n_pos, "n_negative": n_neg,
              "n_flagged": n_flagged,
              "flag_rate": round(float(n_flagged / n) if n else 0.0, 4)}
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
    elif n_pos > 0:
        result["recall"] = result["flag_rate"]
    elif n_neg > 0:
        result["fpr"] = result["flag_rate"]
    return result


BENCHMARKS_PHASE3 = {
    "saladbench_cbrn":   "saladbench_cbrn.jsonl",
    "alert_cbrn":        "alert_cbrn.jsonl",
    "or_bench_health":   "or_bench_health.jsonl",
    "simple_safety_bio": "simple_safety_bio.jsonl",
}


def predict(model: str, rows: list[dict]) -> tuple[list[int], list[float]]:
    queries = [r.get("query", "") for r in rows]
    responses = [r.get("response", "") for r in rows]
    if model == "v3":
        from constitutional_bioguard.evaluation.evaluate_classifier import (
            load_model_and_tokenizer, predict_batch,
        )
        m, t = load_model_and_tokenizer(MODELS_DIR / "deberta_bioguard_v3_balanced")
        preds = predict_batch(model=m, tokenizer=t, queries=queries,
                              responses=responses, normalize=True)
        del m, t
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [p[0] for p in preds], [p[2] for p in preds]
    else:
        from constitutional_bioguard.evaluation.external_baselines import get_baseline
        clf = get_baseline(model)
        clf.load()
        preds = clf.predict_batch(queries, responses)
        clf.unload()
        return [p[0] for p in preds], [p[1] for p in preds]


def evaluate_model(model: str):
    logger.info("=" * 60)
    logger.info("PHASE 3 OOD: %s", model)
    logger.info("=" * 60)
    summary = {"model": model, "benchmarks": {}}
    for bench, fname in BENCHMARKS_PHASE3.items():
        fp = DATA_EXTERNAL / fname
        if not fp.exists():
            logger.warning("Skip %s: cache not found at %s", bench, fp)
            summary["benchmarks"][bench] = {"error": "cache not found"}
            continue
        logger.info("\n[%s on %s]", model, bench)
        rows = load_jsonl(fp)
        if not rows:
            summary["benchmarks"][bench] = {"error": "empty"}
            continue
        pl, probs = predict(model, rows)
        true_labels = [r["label"] for r in rows]
        m = compute_metrics(true_labels, pl, probs)
        result = {
            "overall": m,
            "predictions": [
                {"label": t, "pred": p, "prob": float(pr),
                 "category": r.get("salad_category_3") or r.get("alert_category")
                             or r.get("or_category") or r.get("sst_category", "")}
                for r, t, p, pr in zip(rows, true_labels, pl, probs)
            ],
        }
        out_path = METRICS_DIR / f"phase3_{model}_{bench}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if "f1" in m:
            logger.info("  F1=%.4f AUROC=%.4f AUPRC=%.4f recall=%.4f fpr=%.4f",
                        m["f1"], m.get("auroc", 0), m.get("auprc", 0),
                        m["recall"], m["fpr"])
        else:
            logger.info("  flag_rate=%.4f (%d/%d)",
                        m["flag_rate"], m["n_flagged"], m["n"])
        summary["benchmarks"][bench] = m

    out = METRICS_DIR / f"phase3_{model}_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\nSummary: %s", out)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["v3", "wildguard_7b", "llama_guard_3_8b", "all"])
    args = parser.parse_args()
    if args.model == "all":
        for m in ["v3", "wildguard_7b", "llama_guard_3_8b"]:
            evaluate_model(m)
    else:
        evaluate_model(args.model)


if __name__ == "__main__":
    main()
