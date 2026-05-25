#!/usr/bin/env python
"""Run external safety classifier baselines on the same benchmarks as v3.

Baselines:
  - LLaMA-Guard 3 8B (Meta) — standard general-purpose safety classifier
  - WildGuard 7B (AI2) — trained on WildGuardMix distribution

Benchmarks:
  - BioThreat-Eval (558 expert-labelled bio queries+responses)
  - HarmBench bio held-out (UNSAFE)
  - AdvBench bio held-out (UNSAFE)
  - WildGuardMix held-out (SAFE non-bio; cross-domain FAR)
  - LAB-Bench held-out (SAFE bio-legitimate)

Output: results/metrics/baseline_{model_name}_{benchmark}.json
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
    f1_score, precision_score, recall_score, roc_auc_score,
)

from constitutional_bioguard.config import (
    DATA_EXTERNAL, METRICS_DIR,
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


def load_biothreat() -> tuple[list[str], list[str], list[int]]:
    """Load BioThreat-Eval (558 items)."""
    from constitutional_bioguard.evaluation.external_validation import (
        derive_ground_truth_label, format_for_classifier,
        load_evaluation_results, load_query_bank,
    )
    eval_results = load_evaluation_results()
    query_bank = {q["query_id"]: q for q in load_query_bank()}
    valid_entries = []
    pairs = []
    for entry in eval_results:
        q, r = format_for_classifier(entry, query_bank)
        if q.strip() or r.strip():
            valid_entries.append(entry)
            pairs.append((q, r))
    queries = [p[0] for p in pairs]
    responses = [p[1] for p in pairs]
    labels = [derive_ground_truth_label(e, "threat_level_4") for e in valid_entries]
    return queries, responses, labels


def load_benchmark(name: str) -> tuple[list[str], list[str], list[int]]:
    """Load a benchmark by name. Returns (queries, responses, labels)."""
    if name == "biothreat":
        return load_biothreat()

    splits = DATA_EXTERNAL / "v3_splits"
    path_map = {
        "harmbench_bio_ho": splits / "harmbench_bio_held_out.jsonl",
        "advbench_bio_ho": splits / "advbench_bio_held_out.jsonl",
        "wildguard_test_ho": splits / "wildguard_test_held_out.jsonl",
        "lab_bench_ho": splits / "lab_bench_held_out.jsonl",
        "wmdp_cyber_ho": splits / "wmdp_cyber_held_out.jsonl",
        "wmdp_chem_ho": splits / "wmdp_chem_held_out.jsonl",
    }
    if name not in path_map:
        raise ValueError(f"Unknown benchmark: {name}")
    rows = load_jsonl(path_map[name])
    queries = [r.get("query") or r.get("prompt", "") for r in rows]
    responses = [r.get("response", "") for r in rows]
    labels = [int(r.get("label", 0)) for r in rows]
    return queries, responses, labels


def compute_metrics(true_labels: list[int], pred_labels: list[int],
                    prob_unsafe: list[float], n: int) -> dict:
    """Compute classification metrics. Handles both single-class and mixed."""
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    n_flagged = int(y_pred.sum())

    result = {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_flagged_unsafe": n_flagged,
        "flag_rate": round(float(n_flagged / n) if n else 0.0, 4),
    }

    if n_pos > 0 and n_neg > 0:
        # Mixed labels: full classification metrics
        result.update({
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "fpr": round(float(y_pred[y_true == 0].mean()), 4),
        })
        try:
            result["auroc"] = round(float(roc_auc_score(y_true, prob_unsafe)), 4)
        except Exception:
            pass
    elif n_pos > 0:
        # Only positive labels: flag_rate == recall
        result["recall"] = result["flag_rate"]
    else:
        # Only negative labels: flag_rate == FAR
        result["fpr"] = result["flag_rate"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=str, required=True,
        choices=["llama_guard_3_8b", "wildguard_7b"],
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="+",
        default=[
            "biothreat", "harmbench_bio_ho", "advbench_bio_ho",
            "wildguard_test_ho", "lab_bench_ho",
        ],
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Cap items per benchmark (for smoke test)",
    )
    args = parser.parse_args()

    from constitutional_bioguard.evaluation.external_baselines import (
        get_baseline,
    )

    logger.info("=" * 60)
    logger.info("BASELINE: %s", args.baseline)
    logger.info("=" * 60)

    classifier = get_baseline(args.baseline)
    classifier.load()

    overall = {"baseline": args.baseline, "model_id": classifier.model_id}

    for bench in args.benchmarks:
        logger.info("\n[%s on %s]", args.baseline, bench)
        try:
            queries, responses, labels = load_benchmark(bench)
        except Exception as e:
            logger.error("Failed to load %s: %s", bench, e)
            overall[bench] = {"error": str(e)}
            continue

        if args.max_items is not None:
            queries = queries[: args.max_items]
            responses = responses[: args.max_items]
            labels = labels[: args.max_items]

        logger.info("  %d items, %d positive, %d negative",
                    len(labels), sum(labels), len(labels) - sum(labels))

        preds = classifier.predict_batch(queries, responses)
        pred_labels = [p[0] for p in preds]
        probs = [p[1] for p in preds]

        m = compute_metrics(labels, pred_labels, probs, len(labels))
        overall[bench] = m

        out_path = METRICS_DIR / f"baseline_{args.baseline}_{bench}.json"
        with open(out_path, "w") as f:
            json.dump({
                "baseline": args.baseline,
                "benchmark": bench,
                "metrics": m,
                "predictions": list(zip(labels, pred_labels, probs)),
            }, f, indent=2)
        logger.info("  Saved: %s", out_path)

        # Print summary
        if "f1" in m:
            logger.info(
                "  F1=%.4f AUROC=%s recall=%.4f fpr=%.4f",
                m["f1"], m.get("auroc", "N/A"), m["recall"], m["fpr"],
            )
        else:
            logger.info(
                "  flag_rate=%.4f (%d/%d)",
                m["flag_rate"], m["n_flagged_unsafe"], m["n"],
            )

    classifier.unload()

    # Save overall summary
    summary_path = METRICS_DIR / f"baseline_{args.baseline}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2)
    logger.info("\nSummary: %s", summary_path)


if __name__ == "__main__":
    main()
