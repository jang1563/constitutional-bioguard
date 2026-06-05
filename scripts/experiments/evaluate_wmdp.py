"""Evaluate Constitutional BioGuard on WMDP-Bio (multiple-choice questions).

# NOTE: WMDP uses MCQ format; correct=UNSAFE is an approximation.
# The classifier was trained on query+response pairs (open-ended text), not
# multiple-choice options. These pseudo-labels treat the correct answer as
# UNSAFE (dangerous knowledge) and wrong answers as SAFE (distractors).
# Results should be interpreted with that caveat in mind.

Usage:
    python scripts/experiments/evaluate_wmdp.py [--model-dir PATH] [--limit N]

    --limit N  Process only the first N questions (useful for quick testing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.evaluate_classifier import (
    load_model_and_tokenizer,
    predict_batch,
)


def build_inputs(
    question: str, choices: list[str], answer: int
) -> list[tuple[str, int]]:
    """Build (text, expected_label) pairs for a single MCQ item.

    Correct answer  → label 1 (UNSAFE): dangerous knowledge being tested.
    Wrong answers   → label 0 (SAFE):   distractor / incorrect information.
    """
    pairs = []
    for i, choice in enumerate(choices):
        label = 1 if i == answer else 0
        pairs.append((question, choice, label))
    return pairs


def evaluate_wmdp(model_dir: Path, limit: int | None = None) -> dict:
    """Run BioGuard on WMDP-Bio and return metrics dict."""
    # ── Load dataset ──────────────────────────────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' library is required. Install with: pip install datasets"
        ) from exc

    print("Downloading cais/wmdp (wmdp-bio split) …")
    dataset = load_dataset("cais/wmdp", "wmdp-bio", split="test")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"Using first {len(dataset)} examples (--limit {limit})")
    else:
        print(f"Loaded {len(dataset)} examples.")

    # ── Build inputs ──────────────────────────────────────────────────────────
    all_queries: list[str] = []
    all_responses: list[str] = []
    all_labels: list[int] = []

    for item in dataset:
        question: str = item["question"]
        choices: list[str] = item["choices"]
        answer: int = int(item["answer"])
        triples = build_inputs(question, choices, answer)
        for q, r, label in triples:
            all_queries.append(q)
            all_responses.append(r)
            all_labels.append(label)

    print(f"Total inputs (question × choices): {len(all_queries)}")
    print(f"  Expected UNSAFE (correct answers): {sum(all_labels)}")
    print(f"  Expected SAFE   (wrong answers)  : {len(all_labels) - sum(all_labels)}")

    # ── Load model & predict ──────────────────────────────────────────────────
    print(f"\nLoading model from: {model_dir}")
    model, tokenizer = load_model_and_tokenizer(model_dir)

    print("Running inference …")
    preds_and_probs = predict_batch(
        model=model,
        tokenizer=tokenizer,
        queries=all_queries,
        responses=all_responses,
        normalize=True,
    )

    pred_labels = np.array([p[0] for p in preds_and_probs])
    probs_unsafe = np.array([p[2] for p in preds_and_probs])
    true_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    precision = float(precision_score(true_labels, pred_labels, zero_division=0))
    recall = float(recall_score(true_labels, pred_labels, zero_division=0))
    f1 = float(f1_score(true_labels, pred_labels, zero_division=0))
    accuracy = float(accuracy_score(true_labels, pred_labels))

    auroc = 0.0
    if len(set(true_labels)) > 1:
        auroc = float(roc_auc_score(true_labels, probs_unsafe))

    # False-positive rate (safe examples misclassified as unsafe)
    safe_mask = true_labels == 0
    fpr = float(pred_labels[safe_mask].mean()) if safe_mask.sum() > 0 else 0.0

    # Per-question accuracy: was the correct answer flagged UNSAFE?
    n_correct_flagged = 0
    n_questions = len(dataset)
    idx = 0
    for item in dataset:
        n_choices = len(item["choices"])
        answer = int(item["answer"])
        # The correct-answer prediction is at offset `answer` within this question's block
        correct_idx = idx + answer
        if pred_labels[correct_idx] == 1:
            n_correct_flagged += 1
        idx += n_choices

    results = {
        "dataset": "cais/wmdp (wmdp-bio, test split)",
        "n_questions": n_questions,
        "n_total_inputs": len(all_labels),
        "n_unsafe_expected": int(sum(all_labels)),
        "n_safe_expected": int(len(all_labels) - sum(all_labels)),
        "note": (
            "NOTE: WMDP uses MCQ format; correct=UNSAFE is an approximation. "
            "The classifier was trained on query+response pairs."
        ),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "auroc": round(auroc, 4),
            "fpr": round(fpr, 4),
        },
        "wmdp_specific": {
            "correct_answers_flagged_unsafe": n_correct_flagged,
            "correct_answers_flagged_unsafe_pct": round(
                n_correct_flagged / n_questions * 100, 1
            ),
        },
    }

    print("\n── WMDP-Bio Evaluation Results ──────────────────────────────")
    print(f"  Questions          : {n_questions}")
    print(f"  Total inputs       : {len(all_labels)}")
    print(f"  Precision          : {precision:.4f}")
    print(f"  Recall             : {recall:.4f}")
    print(f"  F1                 : {f1:.4f}")
    print(f"  Accuracy           : {accuracy:.4f}")
    print(f"  AUROC              : {auroc:.4f}")
    print(f"  FPR (wrong→UNSAFE) : {fpr:.4f}")
    print(
        f"  Correct answers flagged UNSAFE: "
        f"{n_correct_flagged}/{n_questions} "
        f"({n_correct_flagged / n_questions * 100:.1f}%)"
    )

    # ── Save results ──────────────────────────────────────────────────────────
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "wmdp_bio_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Constitutional BioGuard on WMDP-Bio"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR / "deberta_bioguard_v1",
        help="Path to the trained model directory (default: models/deberta_bioguard_v1/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N questions (omit for full dataset)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_wmdp(args.model_dir, limit=args.limit)
