"""SPLICE projector for linear concept erasure on classifier head input.

Reference: SPLICE (arXiv:2506.10703, 2025) -- closed-form linear concept
erasure that preserves target covariance, successor to LEACE (Belrose 2023).

Key constraint for v5: apply only to the classifier head's input, not the
encoder. The encoder must keep the compliance-template feature for the model
to USE that information; we're forbidding the head from anchoring decisions
on it.

Operationally:
  1. After encoder training, collect [CLS] embeddings on a held-out probe set
     with binary labels for the spurious concept (e.g., compliance template
     present yes/no).
  2. Fit the SPLICE projector P (orthogonal projection removing the concept's
     least-squares-best linear direction) that:
       - Projects out the compliance-feature direction
       - Preserves variance in directions correlated with the task label
  3. Bolt P as a frozen layer between encoder output and classifier head
     input. Inference: cls_head(P @ encoder_out[:, 0, :]).

Implementation: this file provides the projector fit + a thin frozen nn.Module
wrapper for inference-time integration.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def fit_splice_projector(
    embeddings: np.ndarray,
    concept_labels: np.ndarray,
    task_labels: np.ndarray | None = None,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Fit SPLICE projector matrix P (D x D).

    Approach (LEACE-style with task-preservation): project out the
    least-squares-best linear direction that predicts `concept_labels`
    from `embeddings`. If `task_labels` provided, weight the projection
    to minimize damage to task-correlated variance.

    For v5 (D=768 DeBERTa-base, n=1400 probe set), this is essentially
    a one-shot least-squares fit. We use the simplest LEACE form:

      P = I - U @ U.T

    where U is the orthonormal basis of the concept's best linear direction.

    Args:
        embeddings: (n, D) float array of [CLS] embeddings
        concept_labels: (n,) binary array (1 = concept present, 0 = absent)
        task_labels: optional (n,) binary array for task preservation
        ridge: regularization for the pseudoinverse

    Returns:
        P: (D, D) projector matrix.
    """
    n, D = embeddings.shape
    logger.info("Fitting SPLICE projector: n=%d D=%d concept_pos=%d task_pos=%s",
                n, D, int(concept_labels.sum()),
                int(task_labels.sum()) if task_labels is not None else "n/a")

    # Center embeddings
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    y_c = concept_labels.astype(np.float32) - concept_labels.mean()

    # Least-squares: find direction w that best predicts concept from X
    # w = (X.T X + ridge I)^-1 X.T y_c
    XtX = X.T @ X + ridge * np.eye(D, dtype=np.float32)
    Xty = X.T @ y_c
    w = np.linalg.solve(XtX, Xty)  # (D,)

    # Normalize w to unit length
    w_norm = w / (np.linalg.norm(w) + 1e-12)

    # If task_labels provided, ensure we don't project out the task direction
    if task_labels is not None:
        y_t = task_labels.astype(np.float32) - task_labels.mean()
        w_task = np.linalg.solve(XtX, X.T @ y_t)
        w_task_norm = w_task / (np.linalg.norm(w_task) + 1e-12)
        # Make w orthogonal to w_task: w_concept_orth = w - (w · w_task) * w_task
        dot = float(np.dot(w_norm, w_task_norm))
        w_concept_orth = w_norm - dot * w_task_norm
        w_norm = w_concept_orth / (np.linalg.norm(w_concept_orth) + 1e-12)
        logger.info("  Concept-task alignment before orthogonalization: %.4f", dot)

    # Projector: I - w w^T
    P = np.eye(D, dtype=np.float32) - np.outer(w_norm, w_norm)
    logger.info("Projector fit. ||P|| frob = %.4f, trace(I-P) = %.4f",
                float(np.linalg.norm(P, 'fro')), float(D - np.trace(P)))

    # Sanity: predict concept from PX -- AUROC should drop significantly
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        Xte_proj = (X @ P.T)  # apply projector
        if concept_labels.sum() > 5 and (1 - concept_labels).sum() > 5:
            Xtr, Xte, ytr, yte = train_test_split(
                Xte_proj,
                concept_labels,
                test_size=0.3,
                random_state=42,
                stratify=concept_labels,
            )
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, ytr)
            auroc_after = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
            Xtr, Xte, ytr, yte = train_test_split(
                X,
                concept_labels,
                test_size=0.3,
                random_state=42,
                stratify=concept_labels,
            )
            clf2 = LogisticRegression(max_iter=2000, C=1.0)
            clf2.fit(Xtr, ytr)
            auroc_before = roc_auc_score(yte, clf2.predict_proba(Xte)[:, 1])
            logger.info(
                "  Concept AUROC: before=%.4f after-projection=%.4f",
                auroc_before,
                auroc_after,
            )
    except Exception as e:
        logger.warning("Probe sanity-check failed: %s", e)

    return P


class SPLICEProjector(nn.Module):
    """Frozen linear projector between encoder and classifier head.

    Loads SPLICE matrix from a .pt file and applies it to the [CLS]
    embedding at inference time (and during the head's fine-tuning if any).

    Usage:
        # Fit
        P = fit_splice_projector(embeddings, concept_labels, task_labels)
        torch.save(torch.tensor(P), 'splice_P.pt')

        # Bolt onto model
        projector = SPLICEProjector.from_file('splice_P.pt')
        model.deberta.set_output_projector(projector)  # custom wiring
    """

    def __init__(self, P: torch.Tensor):
        super().__init__()
        self.register_buffer("P", P)  # frozen, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bsz, dim) -- apply P from the right: x @ P.T
        return x @ self.P.T

    @classmethod
    def from_file(cls, path: str | Path) -> "SPLICEProjector":
        P = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(P, torch.Tensor):
            P = torch.tensor(P)
        return cls(P)
