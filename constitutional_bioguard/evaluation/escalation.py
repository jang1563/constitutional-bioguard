"""WS-1: First-stage escalation calibration.

BioGuard v2 repositions the classifier as the cheap first stage of a
Constitutional Classifiers++ style cascade (arXiv:2601.04603). A first-stage
screener is not calibrated to a balanced F1 operating point; it is calibrated
to **escalation recall under a compute budget**: flag (escalate) as much of the
genuinely unsafe traffic as possible while keeping the escalation rate — the
fraction of all traffic sent to the expensive second stage — within budget.

Because escalation is not refusal, a first stage can tolerate a high
false-positive rate; the second stage resolves those. The operating point is
therefore chosen by a recall constraint, following the Neyman-Pearson framing
of Hua et al. 2025 (arXiv:2507.15886): for a calibrated probabilistic score,
thresholding the score is the optimal test for a given error constraint.

Key correctness note — escalation rate is base-rate dependent
-------------------------------------------------------------
The escalation rate (flag rate) is::

    P(flag) = recall * pi + fpr * (1 - pi)

where ``pi`` is the prior probability that an exchange is unsafe. The validation
set is class-balanced from synthetic generation and is ~68% unsafe, which is
not representative of production traffic where unsafe exchanges are rare.
Escalation rate must therefore be evaluated against an *assumed production base
rate*, not the empirical validation base rate. This module reports escalation
rate across a range of base rates and applies the go/no-go gate at a realistic
(low) base rate.

This module derives escalation operating points from the validation
threshold-sweep curve produced by ``evaluate_classifier.calibrate_threshold``.
It does not require model inference.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from constitutional_bioguard.config import DATA_PROCESSED, METRICS_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

# Base rates (prior P(unsafe)) at which to report escalation rate. The first is
# the synthetic validation empirical rate (unrealistic); the rest bracket
# plausible production traffic where unsafe exchanges are rare.
DEFAULT_BASE_RATES = [0.50, 0.10, 0.05, 0.01, 0.005]

# Base rate used for the go/no-go gate — a deliberately conservative estimate
# of unsafe-exchange prevalence in production traffic.
GATE_BASE_RATE = 0.01


def count_val_labels(val_file: Optional[Path] = None) -> tuple[int, int]:
    """Count positive (UNSAFE) and negative (SAFE) examples in the val set.

    Returns:
        (n_positive, n_negative)
    """
    val_file = val_file or DATA_PROCESSED / "val.jsonl"
    n_pos = n_neg = 0
    with open(val_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label = json.loads(line).get("label")
            if label == 1:
                n_pos += 1
            elif label == 0:
                n_neg += 1
    return n_pos, n_neg


def escalation_rate(recall: float, fpr: float, base_rate: float) -> float:
    """Flag rate at an assumed unsafe-exchange prior.

    ``P(flag) = recall * base_rate + fpr * (1 - base_rate)``.
    """
    return recall * base_rate + fpr * (1.0 - base_rate)


def escalation_rate_table(
    recall: float, fpr: float, base_rates: Optional[list[float]] = None
) -> dict[str, float]:
    """Escalation rate across a range of assumed base rates."""
    base_rates = base_rates or DEFAULT_BASE_RATES
    return {
        f"base_rate_{br:g}": round(escalation_rate(recall, fpr, br), 6)
        for br in base_rates
    }


def find_escalation_threshold(
    curve: list[dict],
    target_recall: float = 0.98,
    max_escalation_rate: float = 0.15,
    gate_base_rate: float = GATE_BASE_RATE,
) -> dict:
    """Select the first-stage operating point for a recall target.

    Picks the **highest** threshold that still achieves ``target_recall`` — the
    highest threshold minimises FPR (and thus escalation rate / second-stage
    cost) among all thresholds meeting the recall constraint. This is the
    Neyman-Pearson-style choice: the most efficient recall-constrained point.

    The go/no-go gate evaluates escalation rate at ``gate_base_rate`` — a
    realistic low prior — not at the unrealistic validation base rate.

    Args:
        curve: threshold-sweep curve (``calibration.json`` ``curve`` field),
            each entry with ``threshold``, ``recall``, ``fpr``, ``precision``.
        target_recall: minimum recall the first stage must achieve.
        max_escalation_rate: escalation-rate budget for the go/no-go gate.
        gate_base_rate: assumed production prior P(unsafe) for the gate.

    Returns:
        Report dict with the chosen operating point, escalation rate across
        base rates, and a ``gate_passed`` boolean (WS-1 gate, V2_DESIGN.md).
    """
    qualifying = [e for e in curve if e["recall"] >= target_recall]

    if not qualifying:
        best = max(curve, key=lambda e: e["recall"])
        return {
            "target_recall": target_recall,
            "max_escalation_rate": max_escalation_rate,
            "gate_base_rate": gate_base_rate,
            "achievable": False,
            "best_recall_available": best["recall"],
            "gate_passed": False,
            "note": (
                "No threshold reaches the target recall; the first stage "
                "cannot meet the escalation-recall requirement."
            ),
        }

    # Highest threshold meeting the recall target = lowest FPR = cheapest.
    chosen = max(qualifying, key=lambda e: e["threshold"])
    gate_esc = escalation_rate(chosen["recall"], chosen["fpr"], gate_base_rate)
    gate_passed = gate_esc <= max_escalation_rate

    return {
        "target_recall": target_recall,
        "max_escalation_rate": max_escalation_rate,
        "gate_base_rate": gate_base_rate,
        "achievable": True,
        "operating_point": {
            "threshold": chosen["threshold"],
            "recall": chosen["recall"],
            "fpr": chosen["fpr"],
            "precision": chosen.get("precision"),
        },
        "escalation_rate_by_base_rate": escalation_rate_table(
            chosen["recall"], chosen["fpr"]
        ),
        "escalation_rate_at_gate_base_rate": round(gate_esc, 6),
        "gate_passed": gate_passed,
        "note": (
            f"Gate PASSED: at an assumed production base rate of "
            f"{gate_base_rate:g}, escalation rate is {gate_esc:.4f} <= "
            f"{max_escalation_rate} budget."
            if gate_passed
            else f"Gate FAILED: escalation rate {gate_esc:.4f} at base rate "
            f"{gate_base_rate:g} exceeds the {max_escalation_rate} budget."
        ),
    }


def curve_recall_span(curve: list[dict]) -> float:
    """Range of recall values covered by the threshold sweep."""
    recalls = [e["recall"] for e in curve]
    return max(recalls) - min(recalls)


def au_prc(curve: list[dict], min_recall_span: float = 0.5) -> Optional[float]:
    """Approximate area under the precision-recall curve from the sweep.

    Returns ``None`` (not a fabricated number) when the sweep does not span
    enough of the recall axis to make trapezoidal integration meaningful.

    The v1 ``calibration.json`` sweep was produced to find an F1-optimal
    threshold; on the over-separable synthetic validation set the model is
    extremely confident, so recall barely varies across thresholds and the
    sweep covers only a narrow recall band. A faithful AU-PRC needs raw
    per-example validation scores spanning recall 0..1; the v2 calibration run
    will persist those. Until then this returns ``None`` with the span so the
    caller can report the limitation honestly rather than a broken value.

    Args:
        curve: sweep entries with ``precision`` and ``recall``.
        min_recall_span: minimum recall range required to integrate.

    Returns:
        Approximate AU-PRC, or ``None`` if the sweep span is insufficient.
    """
    span = curve_recall_span(curve)
    if span < min_recall_span:
        logger.warning(
            "AU-PRC not computed: sweep spans only %.4f of the recall axis "
            "(need >= %.2f). Requires raw validation scores.",
            span,
            min_recall_span,
        )
        return None

    pts = sorted(
        ((e["recall"], e["precision"]) for e in curve), key=lambda p: p[0]
    )
    area = 0.0
    for (r0, p0), (r1, p1) in zip(pts, pts[1:]):
        area += (r1 - r0) * (p0 + p1) / 2.0
    return round(area, 6)


def temperature_scale_nll(logits_unsafe: list[float], labels: list[int]) -> dict:
    """Fit a scalar temperature on validation scores to calibrate confidence.

    Raw encoder softmax scores are typically overconfident; temperature scaling
    (Guo et al. 2017; recommended for cascade routing by Gatekeeper,
    arXiv:2502.19335) divides logits by a learned scalar T before the sigmoid,
    minimising negative log-likelihood on the validation set.

    This operates on raw per-example scores. The v1 ``calibration.json`` stores
    only the derived sweep curve, not raw scores, so this is provided for the
    v2 calibration run, which will persist raw validation logits.

    Args:
        logits_unsafe: raw pre-sigmoid logit for the UNSAFE class, per example.
        labels: 0/1 ground-truth labels.

    Returns:
        Dict with the fitted ``temperature`` and the NLL before/after.
    """
    if not logits_unsafe or len(logits_unsafe) != len(labels):
        raise ValueError("logits and labels must be non-empty and aligned")

    def nll(temp: float) -> float:
        total = 0.0
        for z, y in zip(logits_unsafe, labels):
            p = 1.0 / (1.0 + math.exp(-z / temp))
            p = min(max(p, 1e-12), 1 - 1e-12)
            total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return total / len(labels)

    best_t, best_nll = 1.0, nll(1.0)
    for i in range(60):  # T in 0.5 .. 3.45
        t = 0.5 + 0.05 * i
        val = nll(t)
        if val < best_nll:
            best_t, best_nll = t, val

    return {
        "temperature": round(best_t, 4),
        "nll_before": round(nll(1.0), 6),
        "nll_after": round(best_nll, 6),
    }


def run_escalation_calibration(
    calibration_file: Optional[Path] = None,
    val_file: Optional[Path] = None,
    target_recall: float = 0.98,
    max_escalation_rate: float = 0.15,
    output_file: Optional[Path] = None,
) -> dict:
    """WS-1 entry point: derive and persist the first-stage operating point.

    Args:
        calibration_file: path to ``calibration.json`` (sweep curve source).
        val_file: path to ``val.jsonl`` (for positive/negative counts).
        target_recall: minimum first-stage recall.
        max_escalation_rate: escalation-rate budget for the go/no-go gate.
        output_file: where to write the escalation report JSON.

    Returns:
        The escalation report dict.
    """
    calibration_file = calibration_file or (
        MODELS_DIR / "deberta_bioguard_v1" / "calibration.json"
    )
    output_file = output_file or METRICS_DIR / "escalation_calibration.json"

    with open(calibration_file) as f:
        calibration = json.load(f)
    curve = calibration["curve"]

    n_pos, n_neg = count_val_labels(val_file)
    val_base_rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else 0.0
    logger.info(
        "Validation set: %d positive, %d negative (empirical base rate %.3f)",
        n_pos,
        n_neg,
        val_base_rate,
    )

    operating = find_escalation_threshold(
        curve, target_recall, max_escalation_rate
    )
    prc = au_prc(curve)

    report = {
        "workstream": "WS-1 escalation calibration",
        "n_val_positive": n_pos,
        "n_val_negative": n_neg,
        "val_empirical_base_rate": round(val_base_rate, 6),
        "au_prc": prc,
        "au_prc_note": (
            None
            if prc is not None
            else "Sweep curve does not span enough of the recall axis; "
            "AU-PRC requires raw validation scores (v2 calibration run)."
        ),
        "escalation_operating_point": operating,
        "f1_optimal_threshold_for_reference": calibration.get(
            "optimal_threshold"
        ),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved escalation calibration report to %s", output_file)

    op = operating.get("operating_point")
    if op:
        logger.info(
            "Operating point: threshold=%.2f recall=%.4f fpr=%.4f "
            "(gate %s)",
            op["threshold"],
            op["recall"],
            op["fpr"],
            "PASSED" if operating["gate_passed"] else "FAILED",
        )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_escalation_calibration()
