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
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    return model, tokenizer


def predict_batch(
    texts: list[str] | None = None,
    model=None,
    tokenizer=None,
    max_length: int = 512,
    batch_size: int = 32,
    device: Optional[str] = None,
    normalize: bool = True,
    *,
    queries: list[str] | None = None,
    responses: list[str] | None = None,
) -> list[tuple[int, float, float]]:
    """Run inference on a batch of texts or query/response pairs.

    Accepts either ``texts`` (single-string mode) or ``queries`` +
    ``responses`` (sentence-pair mode).  Sentence-pair mode uses the
    tokenizer's native pair encoding which inserts proper [SEP] tokens
    and allows smarter truncation.

    Returns:
        List of (predicted_label, confidence, prob_unsafe) tuples.
    """
    use_pairs = queries is not None and responses is not None
    if not use_pairs and texts is None:
        raise ValueError("Provide either 'texts' or 'queries'+'responses'")
    n = len(queries) if use_pairs else len(texts)

    if normalize:
        from constitutional_bioguard.preprocessing import normalize_text
        if use_pairs:
            queries = [normalize_text(q) for q in queries]
            responses = [normalize_text(r) for r in responses]
        else:
            texts = [normalize_text(t) for t in texts]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    predictions = []
    for i in range(0, n, batch_size):
        if use_pairs:
            batch_q = queries[i : i + batch_size]
            batch_r = responses[i : i + batch_size]
            inputs = tokenizer(
                batch_q,
                batch_r,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
        else:
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


# ---------------------------------------------------------------------------
# Sliding-window inference for long inputs
# ---------------------------------------------------------------------------

def _tokenize_length(tokenizer, text: str) -> int:
    """Return the number of tokens for *text* (no special tokens)."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def predict_sliding(
    model,
    tokenizer,
    query: str,
    response: str,
    max_length: int = 512,
    stride: int = 256,
    device: Optional[str] = None,
    normalize: bool = True,
    aggregation: str = "max",
    threshold: float = 0.5,
) -> tuple[int, float, float, int]:
    """Classify a single query/response pair with sliding-window inference.

    When the combined input exceeds *max_length* tokens the response is
    split into overlapping windows (with *stride* token overlap).  The
    query is prepended to every window so local context is always present.

    Aggregation:
        ``"max"``  — use ``max(p_unsafe)`` across windows.
                     Conservative; suited for safety classifiers.
        ``"mean"`` — use ``mean(p_unsafe)``.

    Returns:
        ``(label, confidence, p_unsafe, n_windows)``
    """
    if normalize:
        from constitutional_bioguard.preprocessing import normalize_text
        query = normalize_text(query)
        response = normalize_text(response)

    # How many tokens does the query consume?  Reserve 3 for [CLS] + 2×[SEP].
    q_ids = tokenizer.encode(query, add_special_tokens=False)
    r_ids = tokenizer.encode(response, add_special_tokens=False)
    special_budget = 3  # [CLS] … [SEP] … [SEP]
    r_budget = max_length - len(q_ids) - special_budget

    if r_budget < 1:
        # Query alone exceeds limit — fall back to simple truncation
        preds = predict_batch(
            model=model, tokenizer=tokenizer,
            queries=[query], responses=[response],
            max_length=max_length, normalize=False, device=device,
        )
        return (*preds[0], 1)

    if len(r_ids) <= r_budget:
        # Fits in a single window
        preds = predict_batch(
            model=model, tokenizer=tokenizer,
            queries=[query], responses=[response],
            max_length=max_length, normalize=False, device=device,
        )
        return (*preds[0], 1)

    # --- multiple windows needed ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    window_probs: list[float] = []
    start = 0
    while start < len(r_ids):
        end = start + r_budget
        chunk_ids = r_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        inputs = tokenizer(
            query, chunk_text,
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
            window_probs.append(float(probs[0, 1]))

        if end >= len(r_ids):
            break
        start += stride

    n_windows = len(window_probs)
    if aggregation == "max":
        p_unsafe = max(window_probs)
    else:
        p_unsafe = sum(window_probs) / n_windows

    label = 1 if p_unsafe >= threshold else 0
    confidence = p_unsafe if label == 1 else (1.0 - p_unsafe)
    return (label, confidence, p_unsafe, n_windows)


def predict_batch_sliding(
    model,
    tokenizer,
    queries: list[str],
    responses: list[str],
    max_length: int = 512,
    stride: int = 256,
    device: Optional[str] = None,
    normalize: bool = True,
    aggregation: str = "max",
    threshold: float = 0.5,
) -> list[tuple[int, float, float, int]]:
    """Batch wrapper around :func:`predict_sliding`.

    Returns list of ``(label, confidence, p_unsafe, n_windows)`` tuples.
    """
    results = []
    for q, r in zip(queries, responses):
        results.append(predict_sliding(
            model, tokenizer, q, r,
            max_length=max_length, stride=stride,
            device=device, normalize=normalize,
            aggregation=aggregation, threshold=threshold,
        ))
    return results


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
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    true_labels = np.array([r["label"] for r in records])
    categories = [r.get("category", "") for r in records]
    fine_labels = [r.get("fine_label", "") for r in records]

    logger.info("Evaluating on %d test examples...", len(records))

    model, tokenizer = load_model_and_tokenizer(model_dir)
    has_pairs = "query" in records[0] and "response" in records[0]
    if has_pairs:
        preds_and_confs = predict_batch(
            model=model, tokenizer=tokenizer,
            queries=[r["query"] for r in records],
            responses=[r["response"] for r in records],
        )
    else:
        preds_and_confs = predict_batch(
            [r["text"] for r in records], model, tokenizer,
        )
    pred_labels = np.array([p[0] for p in preds_and_confs])
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
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    logger.info("Saved evaluation report to %s", metrics_file)

    # Save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_file = METRICS_DIR / "confusion_matrix.json"
    with open(cm_file, "w", encoding="utf-8") as f:
        json.dump({"matrix": cm.tolist(), "labels": ["SAFE", "UNSAFE"]}, f, indent=2)

    # Save classification report
    report_text = classification_report(
        true_labels, pred_labels, target_names=["SAFE", "UNSAFE"]
    )
    report_file = METRICS_DIR / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
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


def calibrate_threshold(
    val_file: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    target_metric: str = "f1",
) -> dict:
    """Find the optimal classification threshold on the validation set.

    Sweeps thresholds from 0.1 to 0.9 and picks the one that maximises
    ``target_metric`` (default: F1).  Saves ``calibration.json`` in the
    model directory.

    Returns:
        Dict with optimal threshold, metric curve, and target_metric value.
    """
    val_file = val_file or DATA_PROCESSED / "val.jsonl"
    model_dir_path = model_dir or (MODELS_DIR / "deberta_bioguard_v1")

    records = []
    with open(val_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    true_labels = np.array([r["label"] for r in records])

    model, tokenizer = load_model_and_tokenizer(model_dir)
    has_pairs = records and "query" in records[0] and "response" in records[0]
    if has_pairs:
        preds_and_confs = predict_batch(
            model=model, tokenizer=tokenizer,
            queries=[r["query"] for r in records],
            responses=[r["response"] for r in records],
        )
    else:
        preds_and_confs = predict_batch(
            [r["text"] for r in records], model, tokenizer,
        )
    probs_unsafe = np.array([p[2] for p in preds_and_confs])

    thresholds = np.arange(0.10, 0.91, 0.01)
    best_threshold = 0.5
    best_score = -1.0
    curve = []

    for t in thresholds:
        preds = (probs_unsafe >= t).astype(int)
        f1 = float(f1_score(true_labels, preds, zero_division=0))
        prec = float(precision_score(true_labels, preds, zero_division=0))
        rec = float(recall_score(true_labels, preds, zero_division=0))
        safe_mask = true_labels == 0
        fpr_val = float(preds[safe_mask].mean()) if safe_mask.sum() > 0 else 0.0

        score = {"f1": f1, "precision": prec, "recall": rec}[target_metric]
        curve.append({
            "threshold": round(float(t), 2),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "fpr": round(fpr_val, 4),
        })
        if score > best_score:
            best_score = score
            best_threshold = round(float(t), 2)

    result = {
        "optimal_threshold": best_threshold,
        "target_metric": target_metric,
        "best_score": round(best_score, 4),
        "default_threshold": 0.5,
        "n_val_samples": len(true_labels),
        "curve": curve,
    }

    cal_file = model_dir_path / "calibration.json"
    with open(cal_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Calibration: optimal threshold=%.2f (%s=%.4f) saved to %s",
        best_threshold, target_metric, best_score, cal_file,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_test_set()
