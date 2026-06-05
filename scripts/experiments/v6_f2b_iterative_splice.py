#!/usr/bin/env python
"""V6 F.2b: Iterative SPLICE — multi-rank concept erasure.

F.2 rank-1 SPLICE failed (concept AUROC stayed at 0.9999 after projection).
v4 encodes the "refusal+compliance hybrid" concept along multiple directions
(robustly, due to B.2 quadruplet training). Rank-1 erasure of the
best-LSQ direction isn't enough.

Iterative approach (INLP-style with task protection):
  for k = 1 to K_max:
    fit classifier on current X to predict concept
    if AUROC ≤ 0.65: stop (concept erased)
    if task AUROC after k iters drops below 0.80: stop (about to damage task)
    extract classifier's weight direction → orthogonalize against task direction
    project: X ← X @ (I - w_k_orth w_k_orth.T)

Outputs:
  models/deberta_bioguard_v4_splice/splice_projector_multirank.pt
  data/metrics/v6_f2b_iterative_splice.json
"""
from __future__ import annotations

import json
import logging
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from constitutional_bioguard.config import METRICS_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Re-use template phrases from v6_f2
from v6_f2_fit_splice import (
    COMPLIANCE_PHRASES, REFUSAL_PHRASES, DEFLECTION_PHRASES,
    HYBRID_PHRASES, SCIENTIFIC_PHRASES,
    load_biothreat_items, build_splice_probe_set, extract_v4_embeddings,
)


def fit_iterative_splice(
    X: np.ndarray, c: np.ndarray, y: np.ndarray,
    K_max: int = 16,
    concept_target: float = 0.65,
    task_floor: float = 0.80,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """Iteratively remove concept directions while protecting task direction."""
    d = X.shape[1]
    X_cur = X.copy()
    P_cumulative = np.eye(d, dtype=np.float32)
    history = []

    # First: compute task direction once (in original space) — we protect against this
    y_t = y.astype(np.float32) - y.mean()
    X_centered = X - X.mean(axis=0, keepdims=True)
    XtX = X_centered.T @ X_centered + ridge * np.eye(d, dtype=np.float32)
    w_t = np.linalg.solve(XtX, X_centered.T @ y_t)
    w_t_norm = w_t / (np.linalg.norm(w_t) + 1e-12)

    for k in range(K_max):
        # Measure current concept and task probe AUROC
        Xtr, Xte, ctr, cte = train_test_split(X_cur, c, test_size=0.3, random_state=42, stratify=c)
        clf_c = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ctr)
        concept_auroc = float(roc_auc_score(cte, clf_c.predict_proba(Xte)[:, 1]))

        Xtr_y, Xte_y, ytr, yte = train_test_split(X_cur, y, test_size=0.3, random_state=42, stratify=y)
        clf_y = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_y, ytr)
        task_auroc = float(roc_auc_score(yte, clf_y.predict_proba(Xte_y)[:, 1]))

        logger.info("  iter %2d:  concept AUROC = %.4f  task AUROC = %.4f", k, concept_auroc, task_auroc)
        history.append({"iter": k, "concept_auroc": round(concept_auroc, 4),
                        "task_auroc": round(task_auroc, 4)})

        if concept_auroc <= concept_target:
            logger.info("  STOP: concept AUROC <= %.2f reached at iter %d", concept_target, k)
            history[-1]["stop_reason"] = "concept_erased"
            break
        if task_auroc < task_floor:
            logger.info("  STOP: task AUROC dropped below %.2f at iter %d (about to damage task)", task_floor, k)
            history[-1]["stop_reason"] = "task_floor_hit"
            break

        # Get classifier direction (the LR coefficients, normalized)
        w_c = clf_c.coef_[0].astype(np.float32)

        # Orthogonalize against task direction
        proj_t = float(np.dot(w_c, w_t_norm))
        w_c_orth = w_c - proj_t * w_t_norm
        w_c_norm = w_c_orth / (np.linalg.norm(w_c_orth) + 1e-12)

        # Update cumulative projector and current X
        P_step = np.eye(d, dtype=np.float32) - np.outer(w_c_norm, w_c_norm)
        P_cumulative = P_step @ P_cumulative
        X_cur = X_cur @ P_step.T

    return P_cumulative, {"history": history, "final_iter": k,
                           "final_concept_auroc": history[-1]["concept_auroc"],
                           "final_task_auroc": history[-1]["task_auroc"]}


def main():
    queries = load_biothreat_items()
    probe_items = build_splice_probe_set(queries, n_per_template=200)

    v4_dir = MODELS_DIR / "deberta_bioguard_v4_response_diverse"
    X = extract_v4_embeddings(v4_dir, probe_items)
    c = np.array([it["concept_label"] for it in probe_items], dtype=np.int32)
    y = np.array([it["task_label"] for it in probe_items], dtype=np.int32)
    logger.info("Embeddings: %s | concept pos: %d/%d | task pos: %d/%d",
                X.shape, int(c.sum()), len(c), int(y.sum()), len(y))

    P_multi, info = fit_iterative_splice(
        X, c, y,
        K_max=16,
        concept_target=0.65,
        task_floor=0.80,
    )
    logger.info("\nFinal iterative SPLICE:\n%s", json.dumps(info, indent=2))

    # Save multi-rank projector
    splice_dir = MODELS_DIR / "deberta_bioguard_v4_splice"
    splice_dir.mkdir(parents=True, exist_ok=True)
    if not (splice_dir / "config.json").exists():
        # First copy weights
        for fp in v4_dir.iterdir():
            if fp.is_file():
                shutil.copy(fp, splice_dir / fp.name)
    P_tensor = torch.tensor(P_multi, dtype=torch.float32)
    torch.save(P_tensor, splice_dir / "splice_projector_multirank.pt")
    logger.info("Saved multi-rank projector to %s",
                splice_dir / "splice_projector_multirank.pt")

    final_decision = (
        "PASS" if info["final_concept_auroc"] <= 0.65 and info["final_task_auroc"] >= 0.80
        else "FAIL"
    )
    provenance = {
        "version": "v4_splice_multirank",
        "base_model": str(v4_dir),
        "method": "Iterative SPLICE (INLP-style with task protection)",
        "K_max": 16,
        "concept_target": 0.65,
        "task_floor": 0.80,
        "iterative_history": info,
        "final_decision": final_decision,
    }
    with open(splice_dir / "v4_splice_multirank_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    out = METRICS_DIR / "v6_f2b_iterative_splice.json"
    with open(out, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Saved %s", out)
    logger.info("Final decision: %s", final_decision)


if __name__ == "__main__":
    main()
