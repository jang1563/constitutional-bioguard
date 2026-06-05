#!/usr/bin/env python
"""Controlled response-harm baseline comparison vs v7.C.

Runs WildGuard-7B (response-harm-only signal) and LLaMA-Guard-3-8B on the EXACT
same wildguard_native item set + `response_harm_label` gold that
eval_v7b_qwen3_cot.py uses (n=1709, pos_rate ~0.166), so the resulting F1 is
apples-to-apples with v7.C's response_harm F1 (0.703).

Why this script exists:
  Phase-2 baselines scored WG/LG against `prompt_harm_label` (pos_rate 0.443),
  AND WildGuard was run in UNION mode (harmful_request OR harmful_response).
  Neither matches a response-harm classifier like v7.C -> the phase-2 F1s
  (WG 0.880, LG3 0.413) are NOT comparable. This script fixes both axes:
    - gold = response_harm_label (identical to eval_v7b)
    - WildGuard signal = "response" (harmful_response ONLY)
    - LLaMA-Guard-3 is natively a response/conversation classifier -> used as-is.

Content-blind: writes ONLY metrics + numeric label/pred/prob arrays, never
query/response payload text. Row order matches eval_v7b's loader, so the arrays
align item-for-item with v7.C's predictions (enables a later paired test).

Output: results/metrics/{out_prefix}_{model}_wildguard_native.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_wildguard_native_response_harm():
    """Replicate eval_v7b_qwen3_cot.py's wildguard_native loader EXACTLY."""
    from constitutional_bioguard.config import DATA_EXTERNAL
    fp = DATA_EXTERNAL / "wildguard_test.jsonl"
    if not fp.exists():
        logger.error(f"missing {fp}")
        sys.exit(1)
    with open(fp) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows
            if isinstance(r.get("response_harm_label"), str)
            and r["response_harm_label"].lower() in ("harmful", "unharmful")]
    for r in rows:
        r["query"] = r.get("prompt", r.get("query", ""))
        r["label"] = 1 if r["response_harm_label"].lower() == "harmful" else 0
    return rows


def compute_metrics(y_true, y_pred, y_prob):
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    yt = np.array(y_true)
    yp = np.array(y_pred)
    n_pos = int((yt == 1).sum())
    n_neg = int((yt == 0).sum())
    out = {
        "n": int(len(yt)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_flag": int(yp.sum()),
        "flag_rate": round(float(yp.mean()), 4),
    }
    if n_pos and n_neg:
        out["precision"] = round(float(precision_score(yt, yp, zero_division=0)), 4)
        out["recall"] = round(float(recall_score(yt, yp, zero_division=0)), 4)
        out["f1"] = round(float(f1_score(yt, yp, zero_division=0)), 4)
        out["fpr"] = round(float(yp[yt == 0].mean()), 4)
        try:
            out["auroc"] = round(float(roc_auc_score(yt, y_prob)), 4)
            out["auprc"] = round(float(average_precision_score(yt, y_prob)), 4)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description="Controlled response-harm baseline eval")
    ap.add_argument("--models", nargs="+",
                    default=["wildguard_7b", "llama_guard_3_8b"])
    ap.add_argument("--out-prefix", default="v7c_baseline_rh")
    ap.add_argument("--expect-n", type=int, default=1709,
                    help="Assert item-set size matches v7.C (drift guard)")
    args = ap.parse_args()

    from constitutional_bioguard.config import METRICS_DIR
    from constitutional_bioguard.evaluation.external_baselines import get_baseline
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_wildguard_native_response_harm()
    queries = [r["query"] for r in rows]
    responses = [r.get("response", "") for r in rows]
    labels = [int(r["label"]) for r in rows]
    n_pos = sum(labels)
    logger.info(
        f"wildguard_native (response_harm): n={len(rows)} n_pos={n_pos} "
        f"pos_rate={n_pos/len(rows):.4f}"
    )
    assert len(rows) == args.expect_n, (
        f"item-set drift: expected {args.expect_n}, got {len(rows)} "
        f"(NOT comparable to v7.C — abort)"
    )

    summary = {}
    for name in args.models:
        clf = get_baseline(name)
        signal = None
        if name in ("wildguard_7b", "wildguard"):
            clf.signal = "response"      # harmful_response ONLY (match v7.C task)
            signal = "response"
            logger.info("WildGuard signal = RESPONSE-only (not union)")
        logger.info(f"=== {name}: loading ===")
        clf.load()
        logger.info(f"=== {name}: predicting on {len(queries)} items ===")
        preds = clf.predict_batch(queries, responses)
        clf.unload()

        pred_labels = [int(p[0]) for p in preds]
        probs = [float(p[1]) for p in preds]
        m = compute_metrics(labels, pred_labels, probs)
        m["model"] = name
        m["wildguard_signal"] = signal
        m["gold_label"] = "response_harm_label"
        logger.info(
            f"{name} :: f1={m.get('f1')} precision={m.get('precision')} "
            f"recall={m.get('recall')} fpr={m.get('fpr')}"
        )

        out = {
            "overall": m,
            # numeric only (content-blind); aligned with eval_v7b row order
            "labels": labels,
            "preds": pred_labels,
            "probs": probs,
        }
        fp = METRICS_DIR / f"{args.out_prefix}_{name}_wildguard_native.json"
        with open(fp, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"wrote {fp}")
        summary[name] = m

    with open(METRICS_DIR / f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Summary (wildguard_native, response_harm gold, n=%d) ===" % len(rows))
    for name, m in summary.items():
        print(f"{name}: f1={m.get('f1')} P={m.get('precision')} "
              f"R={m.get('recall')} fpr={m.get('fpr')} "
              f"signal={m.get('wildguard_signal')}")


if __name__ == "__main__":
    main()
