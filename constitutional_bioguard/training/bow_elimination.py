"""WS-2: Bag-of-words elimination.

A failure mode for a biosafety classifier is latching onto surface vocabulary —
flagging an exchange UNSAFE because it contains words like "pathogen" or
"synthesis" rather than because the *intent* is unsafe. A classifier that has
learned the keyword shortcut scores well in-distribution but generalises poorly,
which is a plausible contributor to BioGuard v1's external Cohen kappa gap
(0.414 on BioThreat-Eval).

Bag-of-words elimination (McKenzie et al. 2025, arXiv:2506.10805) diagnoses and
mitigates this: train a bag-of-words linear model on the training data and
remove examples whose label is *trivially predictable from surface tokens
alone*. A DeBERTa classifier trained on the filtered set is pushed to learn
intent rather than vocabulary.

This module is pure analysis over the training JSONL — no model inference, no
GPU. It uses cross-validated bag-of-words predictions so an example is judged
by a model that never saw it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from constitutional_bioguard.config import DATA_PROCESSED

logger = logging.getLogger(__name__)


def load_records(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bow_predictability(
    texts: list[str],
    labels: list[int],
    *,
    cv_folds: int = 5,
    max_features: int = 5000,
    confidence: float = 0.90,
    seed: int = 42,
) -> dict:
    """Score how predictable each label is from surface tokens alone.

    Trains a bag-of-words logistic-regression model with cross-validated
    predictions, so every example is scored by a model that did not train on
    it. An example is "keyword-trivial" when the BoW model predicts its true
    label with probability >= ``confidence``.

    Args:
        texts: input strings.
        labels: 0/1 labels aligned with ``texts``.
        cv_folds: cross-validation folds for out-of-fold prediction.
        max_features: vocabulary cap for the bag-of-words representation.
        confidence: probability threshold above which a correct BoW
            prediction marks the example as keyword-trivial.
        seed: random seed.

    Returns:
        Dict with per-example BoW probabilities, a boolean ``trivial`` mask,
        the bag-of-words cross-validated AUROC, and summary counts.
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "bag-of-words elimination needs scikit-learn: pip install scikit-learn"
        ) from exc

    if len(texts) != len(labels):
        raise ValueError("texts and labels must be aligned")

    vectorizer = CountVectorizer(
        max_features=max_features, lowercase=True, stop_words="english"
    )
    features = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, random_state=seed)

    # Out-of-fold predicted probabilities — honest per-example scores.
    proba = cross_val_predict(
        clf, features, labels, cv=cv_folds, method="predict_proba"
    )
    p_unsafe = [float(row[1]) for row in proba]

    bow_auroc = float(roc_auc_score(labels, p_unsafe))

    trivial = []
    for p, y in zip(p_unsafe, labels):
        p_true = p if y == 1 else (1.0 - p)
        trivial.append(p_true >= confidence)

    n_trivial = sum(trivial)
    return {
        "p_unsafe": p_unsafe,
        "trivial": trivial,
        "bow_cv_auroc": round(bow_auroc, 6),
        "n_total": len(texts),
        "n_trivial": n_trivial,
        "trivial_fraction": round(n_trivial / len(texts), 6),
        "confidence_threshold": confidence,
    }


def eliminate_bow_examples(
    train_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    report_file: Optional[Path] = None,
    *,
    text_field: str = "text",
    confidence: float = 0.90,
    dry_run: bool = True,
) -> dict:
    """WS-2 entry point: diagnose and optionally filter keyword-trivial examples.

    Args:
        train_file: input training JSONL.
        output_file: where to write the filtered JSONL (kept examples only).
        report_file: where to write the elimination report JSON.
        text_field: which record field to use as the bag-of-words input.
        confidence: keyword-triviality threshold.
        dry_run: when True (default) only the report is written — no filtered
            dataset is produced. Filtering the training set is a deliberate
            action that should be reviewed before retraining.

    Returns:
        The elimination report dict.
    """
    train_file = train_file or DATA_PROCESSED / "train.jsonl"
    output_file = output_file or DATA_PROCESSED / "train_bow_filtered.jsonl"
    report_file = report_file or DATA_PROCESSED / "bow_elimination_report.json"

    records = load_records(train_file)
    texts = [str(r.get(text_field, "")) for r in records]
    labels = [int(r["label"]) for r in records]

    result = bow_predictability(texts, labels, confidence=confidence)

    # Per-category breakdown of triviality, if category metadata is present.
    by_category: dict[str, dict] = {}
    for rec, is_trivial in zip(records, result["trivial"]):
        cat = rec.get("category", "unknown")
        slot = by_category.setdefault(cat, {"n": 0, "n_trivial": 0})
        slot["n"] += 1
        slot["n_trivial"] += int(is_trivial)
    for cat, slot in by_category.items():
        slot["trivial_fraction"] = round(slot["n_trivial"] / slot["n"], 6)

    report = {
        "workstream": "WS-2 bag-of-words elimination",
        "train_file": str(train_file),
        "text_field": text_field,
        "bow_cv_auroc": result["bow_cv_auroc"],
        "n_total": result["n_total"],
        "n_trivial": result["n_trivial"],
        "trivial_fraction": result["trivial_fraction"],
        "confidence_threshold": confidence,
        "by_category": by_category,
        "interpretation": (
            "A high bag-of-words cross-validated AUROC means surface tokens "
            "alone already separate the classes well — the synthetic data is "
            "lexically easy. Examples flagged 'trivial' can be removed so the "
            "DeBERTa classifier is pushed toward intent over vocabulary. A "
            "high trivial fraction is itself evidence that v1's strong "
            "in-distribution F1 is partly a keyword shortcut, consistent with "
            "the external kappa gap (see docs/V2_DESIGN.md WS-2)."
        ),
        "dry_run": dry_run,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved bag-of-words elimination report to %s", report_file)

    if not dry_run:
        kept = [
            rec
            for rec, is_trivial in zip(records, result["trivial"])
            if not is_trivial
        ]
        with open(output_file, "w") as f:
            for rec in kept:
                f.write(json.dumps(rec) + "\n")
        report["n_kept"] = len(kept)
        report["output_file"] = str(output_file)
        logger.info(
            "Wrote %d filtered examples (removed %d) to %s",
            len(kept),
            result["n_trivial"],
            output_file,
        )
    else:
        logger.info(
            "Dry run: %d of %d examples are keyword-trivial (%.1f%%). "
            "Re-run with dry_run=False to write the filtered dataset.",
            result["n_trivial"],
            result["n_total"],
            100 * result["trivial_fraction"],
        )

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eliminate_bow_examples()
