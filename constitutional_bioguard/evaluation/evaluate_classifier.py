"""Internal evaluation of the trained bio-safety classifier.

Computes precision, recall, F1, AUROC, accuracy, and FPR on the held-out
test set. Also computes per-NSABB-category metrics using metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from constitutional_bioguard.config import (
    DATA_PROCESSED,
    METRICS_DIR,
    MODELS_DIR,
    NSABB_CATEGORIES,
)
from constitutional_bioguard.models import (
    ClassificationMetrics,
    EvaluationReport,
    NSABBCategory,
    PerCategoryMetrics,
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_dir: Optional[Path] = None):
    """Load the trained DeBERTa model and tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = model_dir or MODELS_DIR / "deberta_bioguard_v1"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return model, tokenizer


def predict_batch(
    texts: list[str],
    model,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    device: Optional[str] = None,
    normalize: bool = True,
) -> list[tuple[int, float, float]]:
    """Run inference on a batch of texts.

    Args:
        texts: Input strings in "[CLS] query [SEP] response [SEP]" format.
        normalize: If True, apply encoding normalization before tokenization.
            Mitigates ROT13, base64, URL-encode, hex, homoglyph attacks.
            Default: True.

    Returns:
        List of (predicted_label, confidence, prob_unsafe) tuples.
        prob_unsafe is the class-1 (UNSAFE) probability, used for AUROC.
    """
    if normalize:
        from constitutional_bioguard.preprocessing import normalize_text
        texts = [normalize_text(t) for t in texts]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_labels = probs.argmax(dim=-1)
            confidences = probs.max(dim=-1).values
            probs_class1 = probs[:, 1]

        for label, conf, p1 in zip(
            pred_labels.cpu().numpy(),
            confidences.cpu().numpy(),
            probs_class1.cpu().numpy(),
        ):
            predictions.append((int(label), float(conf), float(p1)))

    return predictions


def evaluate_test_set(
    test_file: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> EvaluationReport:
    """Evaluate the classifier on the test set.

    Returns:
        EvaluationReport with internal metrics and per-category breakdown.
    """
    test_file = test_file or DATA_PROCESSED / "test.jsonl"

    # Load test data
    records = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    texts = [r["text"] for r in records]
    true_labels = np.array([r["label"] for r in records])
    categories = [r.get("category", "") for r in records]
    fine_labels = [r.get("fine_label", "") for r in records]

    logger.info("Evaluating on %d test examples...", len(texts))

    # Predict (single pass returns labels, confidences, and class-1 probs)
    model, tokenizer = load_model_and_tokenizer(model_dir)
    preds_and_confs = predict_batch(texts, model, tokenizer)
    pred_labels = np.array([p[0] for p in preds_and_confs])
    confidences = np.array([p[1] for p in preds_and_confs])
    probs_unsafe = np.array([p[2] for p in preds_and_confs])

    overall = _compute_metrics(true_labels, pred_labels, probs_unsafe)
    logger.info("Overall: F1=%.4f, AUROC=%.4f, FPR=%.4f",
                overall.f1, overall.auroc, overall.fpr)

    # Per-category metrics (exclude benign examples whose category is a placeholder)
    per_category = []
    for cat in NSABB_CATEGORIES:
        mask = np.array([
            c == cat and fl != "benign"
            for c, fl in zip(categories, fine_labels)
        ])
        if mask.sum() == 0:
            continue
        cat_metrics = _compute_metrics(
            true_labels[mask], pred_labels[mask], probs_unsafe[mask]
        )
        per_category.append(
            PerCategoryMetrics(
                category=NSABBCategory(cat),
                metrics=cat_metrics,
            )
        )
        logger.info("  %s: F1=%.4f (n=%d)", cat, cat_metrics.f1, mask.sum())

    report = EvaluationReport(
        model_name="deberta_bioguard_v1",
        internal_metrics=overall,
        per_category_metrics=per_category,
    )

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = METRICS_DIR / "internal_evaluation.json"
    with open(metrics_file, "w") as f:
        f.write(report.model_dump_json(indent=2))
    logger.info("Saved evaluation report to %s", metrics_file)

    # Save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_file = METRICS_DIR / "confusion_matrix.json"
    with open(cm_file, "w") as f:
        json.dump({"matrix": cm.tolist(), "labels": ["SAFE", "UNSAFE"]}, f, indent=2)

    # Save classification report
    report_text = classification_report(
        true_labels, pred_labels, target_names=["SAFE", "UNSAFE"]
    )
    report_file = METRICS_DIR / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    return report


def _compute_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probs: np.ndarray,
) -> ClassificationMetrics:
    """Compute ClassificationMetrics from arrays."""
    safe_mask = true_labels == 0
    fpr = float(pred_labels[safe_mask].mean()) if safe_mask.sum() > 0 else 0.0

    auroc = 0.0
    if len(set(true_labels)) > 1:
        auroc = float(roc_auc_score(true_labels, probs))

    return ClassificationMetrics(
        precision=float(precision_score(true_labels, pred_labels, zero_division=0)),
        recall=float(recall_score(true_labels, pred_labels, zero_division=0)),
        f1=float(f1_score(true_labels, pred_labels, zero_division=0)),
        auroc=auroc,
        accuracy=float(accuracy_score(true_labels, pred_labels)),
        fpr=fpr,
        n_samples=len(true_labels),
        n_positive=int((true_labels == 1).sum()),
        n_negative=int((true_labels == 0).sum()),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_test_set()
