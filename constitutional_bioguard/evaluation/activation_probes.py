"""WS-3: Linear activation probes for open-weight LLMs.

Replicates the CC++ finding (Cunningham, Wei et al. 2026) that linear
probes trained on a language model's intermediate activations ensemble
complementarily with external fine-tuned classifiers.

Since we cannot probe Claude's internals, we use open-weight models
(Llama-3.1-8B, Gemma-2-9B) as proxies — testing whether the
*architectural* claim (probes capture orthogonal signal) transfers
to the biosafety domain.

Two probe types:
  - **Mean probe**: average hidden states across all tokens at a target
    layer, then train a logistic classifier. Simple, robust baseline.
  - **Suffix probe**: append a short classification instruction, take
    the hidden state of the final token at the target layer. Captures
    the model's "summary" representation conditioned on the prompt.

Ensemble: combine probe logits with BioGuard classifier logits via
learned or fixed weighting; measure AU-PRC and error decorrelation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# ── Probe configuration ─────────────────────────────────────────────────────

# Target layer as fraction of total depth (~40%, the band identified by
# McKenzie et al. as carrying the most linearly-decodable concepts).
DEFAULT_LAYER_FRACTION = 0.4

SUFFIX_INSTRUCTION = (
    "Based on the above exchange, classify whether the content poses a "
    "biological dual-use safety concern. Classification:"
)

SUPPORTED_MODELS = {
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "n_layers": 32,
        "hidden_dim": 4096,
    },
    "gemma-2-9b": {
        "hf_id": "google/gemma-2-9b",
        "n_layers": 42,
        "hidden_dim": 3584,
    },
}


@dataclass
class ProbeResult:
    """Results from a single probe or ensemble evaluation."""

    au_prc: float
    auroc: float
    f1: float
    tpr_at_1pct_fpr: float
    n_total: int
    probe_type: str  # "mean", "suffix", "ensemble", "bioguard_only"
    model_name: str
    target_layer: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "au_prc": round(self.au_prc, 6),
            "auroc": round(self.auroc, 6),
            "f1": round(self.f1, 6),
            "tpr_at_1pct_fpr": round(self.tpr_at_1pct_fpr, 6),
            "n_total": self.n_total,
            "probe_type": self.probe_type,
            "model_name": self.model_name,
            "target_layer": self.target_layer,
            **self.extra,
        }


# ── Activation extraction ───────────────────────────────────────────────────

def get_target_layer(model_name: str, fraction: float = DEFAULT_LAYER_FRACTION) -> int:
    """Compute the target layer index from a depth fraction."""
    info = SUPPORTED_MODELS[model_name]
    layer = int(info["n_layers"] * fraction)
    return min(layer, info["n_layers"] - 1)


def load_model_for_probing(
    model_name: str,
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
):
    """Load an open-weight model configured to output hidden states.

    Returns (model, tokenizer).  Model is in eval mode with gradients
    disabled.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info = SUPPORTED_MODELS[model_name]
    hf_id = info["hf_id"]

    logger.info("Loading %s for activation probing...", hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=device,
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def format_exchange(query: str, response: str) -> str:
    """Format a query-response pair for the LLM input."""
    return f"User: {query}\nAssistant: {response}"


def extract_activations_batch(
    model,
    tokenizer,
    texts: list[str],
    target_layer: int,
    probe_type: str = "mean",
    batch_size: int = 4,
    max_length: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract hidden-state activations from a target layer.

    Args:
        model: HuggingFace causal LM with output_hidden_states=True.
        tokenizer: Corresponding tokenizer.
        texts: Input texts (already formatted as exchanges).
        target_layer: Which layer's hidden states to extract.
        probe_type: "mean" (average all tokens) or "suffix" (last token).
        batch_size: Inference batch size.
        max_length: Max token length.
        device: Override device.

    Returns:
        np.ndarray of shape (n_texts, hidden_dim).
    """
    if device is None:
        device = next(model.parameters()).device

    all_activations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # hidden_states is a tuple of (n_layers + 1) tensors
        # Index 0 = embedding output, index k = layer k output
        hidden = outputs.hidden_states[target_layer + 1]  # +1 for embedding

        attention_mask = inputs["attention_mask"]

        if probe_type == "mean":
            # Mean-pool over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        elif probe_type == "suffix":
            # Take the last non-padding token's hidden state
            seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
            pooled = hidden[
                torch.arange(hidden.size(0), device=device), seq_lengths
            ]
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

        all_activations.append(pooled.cpu().float().numpy())

        if (i // batch_size) % 10 == 0:
            logger.info(
                "Extracted activations: %d / %d", min(i + batch_size, len(texts)), len(texts)
            )

    return np.concatenate(all_activations, axis=0)


# ── Probe training ──────────────────────────────────────────────────────────

def train_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 1000,
) -> LogisticRegressionCV:
    """Train a logistic regression probe with cross-validated regularization.

    Args:
        activations: (n_samples, hidden_dim) array.
        labels: (n_samples,) binary labels.
        max_iter: Maximum solver iterations.

    Returns:
        Fitted LogisticRegressionCV model.
    """
    logger.info(
        "Training linear probe: %d samples, %d features, class balance %.1f%% positive",
        activations.shape[0],
        activations.shape[1],
        100 * labels.mean(),
    )
    probe = LogisticRegressionCV(
        Cs=10,
        cv=5,
        scoring="average_precision",
        solver="lbfgs",
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
    )
    probe.fit(activations, labels)
    logger.info("Probe trained. Best C=%.4f", probe.C_[0])
    return probe


# ── Evaluation ──────────────────────────────────────────────────────────────

def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """Compute TPR at a given FPR threshold."""
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    # Find the largest TPR where FPR <= target
    valid = fpr_arr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr_arr[valid][-1])


def evaluate_probe(
    probe,
    activations: np.ndarray,
    labels: np.ndarray,
    probe_type: str,
    model_name: str,
    target_layer: int,
) -> ProbeResult:
    """Evaluate a trained probe on held-out data."""
    probs = probe.predict_proba(activations)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return ProbeResult(
        au_prc=float(average_precision_score(labels, probs)),
        auroc=float(roc_auc_score(labels, probs)),
        f1=float(f1_score(labels, preds)),
        tpr_at_1pct_fpr=tpr_at_fpr(labels, probs, 0.01),
        n_total=len(labels),
        probe_type=probe_type,
        model_name=model_name,
        target_layer=target_layer,
    )


def evaluate_ensemble(
    probe_probs: np.ndarray,
    bioguard_probs: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    target_layer: int,
    weight_probe: float = 0.5,
) -> tuple[ProbeResult, dict]:
    """Evaluate the probe + BioGuard ensemble.

    Uses a simple weighted average of probabilities. Also computes
    the Spearman rank correlation of per-example errors to measure
    complementarity.

    Returns:
        (ProbeResult for ensemble, complementarity_stats dict)
    """
    # Weighted ensemble
    ensemble_probs = weight_probe * probe_probs + (1 - weight_probe) * bioguard_probs
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    result = ProbeResult(
        au_prc=float(average_precision_score(labels, ensemble_probs)),
        auroc=float(roc_auc_score(labels, ensemble_probs)),
        f1=float(f1_score(labels, ensemble_preds)),
        tpr_at_1pct_fpr=tpr_at_fpr(labels, ensemble_probs, 0.01),
        n_total=len(labels),
        probe_type="ensemble",
        model_name=model_name,
        target_layer=target_layer,
        extra={"weight_probe": weight_probe},
    )

    # Complementarity: Spearman correlation of per-example errors
    probe_errors = np.abs(probe_probs - labels)
    bioguard_errors = np.abs(bioguard_probs - labels)
    corr, p_value = spearmanr(probe_errors, bioguard_errors)

    complementarity = {
        "spearman_error_correlation": round(float(corr), 6),
        "spearman_p_value": float(p_value),
        "interpretation": (
            "Low correlation (<0.3) indicates complementary errors — "
            "the probe and classifier fail on different examples, "
            "supporting ensemble benefit."
        ),
    }

    return result, complementarity


# ── Weight sweep ─────────────────────────────────────────────────────────────

def sweep_ensemble_weights(
    probe_probs: np.ndarray,
    bioguard_probs: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    target_layer: int,
    weights: Optional[list[float]] = None,
) -> list[dict]:
    """Sweep probe weight from 0 to 1 and return AU-PRC at each point."""
    if weights is None:
        weights = [round(w * 0.1, 1) for w in range(11)]

    results = []
    for w in weights:
        combo = w * probe_probs + (1 - w) * bioguard_probs
        ap = float(average_precision_score(labels, combo))
        results.append({"weight_probe": w, "au_prc": round(ap, 6)})

    return results


# ── BioGuard predictions ────────────────────────────────────────────────────

def get_bioguard_probs(
    queries: list[str],
    responses: list[str],
    model_dir: Optional[Path] = None,
) -> np.ndarray:
    """Get BioGuard classifier probabilities for query-response pairs.

    Returns:
        np.ndarray of shape (n,) with P(UNSAFE) for each pair.
    """
    from constitutional_bioguard.evaluation.evaluate_classifier import (
        load_model_and_tokenizer,
        predict_batch,
    )

    model, tokenizer = load_model_and_tokenizer(model_dir)
    preds_and_confs = predict_batch(
        model=model, tokenizer=tokenizer,
        queries=queries, responses=responses,
    )
    # predict_batch returns list of (label, confidence, prob_unsafe)
    probs = [p[2] for p in preds_and_confs]  # prob_unsafe directly
    return np.array(probs)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_labeled_data(data_file: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Load query-response pairs and labels from a JSONL file.

    Returns:
        (queries, responses, labels) where labels is a binary ndarray.
    """
    queries, responses, labels = [], [], []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            # Parse "query [SEP] response" format
            if "[SEP]" in text:
                parts = text.split("[SEP]", 1)
                q = parts[0].strip()
                r = parts[1].strip() if len(parts) > 1 else ""
            else:
                q = text
                r = ""
            queries.append(q)
            responses.append(r)
            labels.append(int(row.get("label", 0)))
    return queries, responses, np.array(labels)


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_probe_pipeline(
    model_name: str = "llama-3.1-8b",
    train_file: Optional[Path] = None,
    test_file: Optional[Path] = None,
    bioguard_model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    layer_fraction: float = DEFAULT_LAYER_FRACTION,
    batch_size: int = 4,
    device: str = "auto",
) -> dict:
    """Run the full WS-3 probe pipeline.

    Steps:
        1. Load data (train + test splits)
        2. Load open-weight LLM
        3. Extract activations (mean + suffix probes)
        4. Train linear probes
        5. Get BioGuard predictions
        6. Evaluate probes, BioGuard, and ensembles
        7. Compute complementarity statistics
        8. Save results

    Returns:
        Full results dict.
    """
    from constitutional_bioguard.config import DATA_PROCESSED, METRICS_DIR, MODELS_DIR

    train_file = train_file or DATA_PROCESSED / "train.jsonl"
    test_file = test_file or DATA_PROCESSED / "test.jsonl"
    bioguard_model_dir = bioguard_model_dir or MODELS_DIR / "deberta_bioguard_v1"
    output_dir = output_dir or METRICS_DIR

    target_layer = get_target_layer(model_name, layer_fraction)
    logger.info(
        "WS-3 probe pipeline: model=%s, layer=%d (%.0f%% depth)",
        model_name, target_layer, layer_fraction * 100,
    )

    # 1. Load data
    logger.info("Loading data...")
    train_q, train_r, train_labels = load_labeled_data(train_file)
    test_q, test_r, test_labels = load_labeled_data(test_file)
    logger.info("Train: %d, Test: %d", len(train_labels), len(test_labels))

    # Format as exchanges
    train_texts = [format_exchange(q, r) for q, r in zip(train_q, train_r)]
    test_texts = [format_exchange(q, r) for q, r in zip(test_q, test_r)]

    # 2. Load model
    model, tokenizer = load_model_for_probing(model_name, device=device)

    # 3. Extract activations
    probe_results = {}
    for probe_type in ["mean", "suffix"]:
        logger.info("Extracting %s activations (layer %d)...", probe_type, target_layer)

        if probe_type == "suffix":
            # Append classification instruction for suffix probe
            train_input = [t + "\n" + SUFFIX_INSTRUCTION for t in train_texts]
            test_input = [t + "\n" + SUFFIX_INSTRUCTION for t in test_texts]
        else:
            train_input = train_texts
            test_input = test_texts

        train_acts = extract_activations_batch(
            model, tokenizer, train_input, target_layer,
            probe_type=probe_type, batch_size=batch_size,
        )
        test_acts = extract_activations_batch(
            model, tokenizer, test_input, target_layer,
            probe_type=probe_type, batch_size=batch_size,
        )

        # 4. Train probe
        logger.info("Training %s probe...", probe_type)
        probe = train_probe(train_acts, train_labels)

        # 5. Evaluate probe alone
        result = evaluate_probe(
            probe, test_acts, test_labels,
            probe_type=probe_type,
            model_name=model_name,
            target_layer=target_layer,
        )
        probe_results[probe_type] = {
            "result": result,
            "test_probs": probe.predict_proba(test_acts)[:, 1],
        }
        logger.info(
            "%s probe: AU-PRC=%.4f, AUROC=%.4f, TPR@1%%FPR=%.4f",
            probe_type, result.au_prc, result.auroc, result.tpr_at_1pct_fpr,
        )

    # Free LLM memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 6. Get BioGuard predictions
    logger.info("Getting BioGuard predictions...")
    bioguard_probs = get_bioguard_probs(test_q, test_r, bioguard_model_dir)
    bioguard_preds = (bioguard_probs >= 0.5).astype(int)
    bioguard_result = ProbeResult(
        au_prc=float(average_precision_score(test_labels, bioguard_probs)),
        auroc=float(roc_auc_score(test_labels, bioguard_probs)),
        f1=float(f1_score(test_labels, bioguard_preds)),
        tpr_at_1pct_fpr=tpr_at_fpr(test_labels, bioguard_probs, 0.01),
        n_total=len(test_labels),
        probe_type="bioguard_only",
        model_name="deberta-v3-base",
        target_layer=-1,
    )

    # 7. Ensemble evaluation
    ensemble_results = {}
    complementarity_stats = {}
    for probe_type in ["mean", "suffix"]:
        probe_probs = probe_results[probe_type]["test_probs"]

        # Sweep weights
        sweep = sweep_ensemble_weights(
            probe_probs, bioguard_probs, test_labels,
            model_name=model_name, target_layer=target_layer,
        )
        best_weight = max(sweep, key=lambda x: x["au_prc"])["weight_probe"]

        # Evaluate at best weight
        ens_result, comp_stats = evaluate_ensemble(
            probe_probs, bioguard_probs, test_labels,
            model_name=model_name, target_layer=target_layer,
            weight_probe=best_weight,
        )
        ensemble_results[probe_type] = {
            "result": ens_result,
            "weight_sweep": sweep,
            "best_weight": best_weight,
        }
        complementarity_stats[probe_type] = comp_stats

        logger.info(
            "Ensemble (%s, w=%.1f): AU-PRC=%.4f (probe=%.4f, bioguard=%.4f)",
            probe_type, best_weight, ens_result.au_prc,
            probe_results[probe_type]["result"].au_prc,
            bioguard_result.au_prc,
        )

    # 8. Compile and save
    full_results = {
        "experiment": "WS-3 activation-probe ensemble",
        "model_name": model_name,
        "target_layer": target_layer,
        "layer_fraction": layer_fraction,
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "bioguard": bioguard_result.to_dict(),
        "probes": {
            pt: probe_results[pt]["result"].to_dict()
            for pt in ["mean", "suffix"]
        },
        "ensembles": {
            pt: {
                **ensemble_results[pt]["result"].to_dict(),
                "best_weight": ensemble_results[pt]["best_weight"],
                "weight_sweep": ensemble_results[pt]["weight_sweep"],
            }
            for pt in ["mean", "suffix"]
        },
        "complementarity": complementarity_stats,
        "go_no_go": _evaluate_gate(
            probe_results, ensemble_results, bioguard_result, complementarity_stats
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"probe_ensemble_{model_name.replace('/', '_')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info("Saved results to %s", out_file)

    return full_results


def _evaluate_gate(
    probe_results: dict,
    ensemble_results: dict,
    bioguard_result: ProbeResult,
    complementarity_stats: dict,
) -> dict:
    """Evaluate the WS-3 go/no-go gate.

    Gate passes if:
      1. Ensemble AU-PRC > max(probe AU-PRC, BioGuard AU-PRC) by >0.01
      2. Error correlation < 0.3 (complementary signals)
    """
    best_single = bioguard_result.au_prc
    for pt in ["mean", "suffix"]:
        best_single = max(best_single, probe_results[pt]["result"].au_prc)

    best_ensemble = max(
        ensemble_results[pt]["result"].au_prc for pt in ["mean", "suffix"]
    )
    best_ensemble_type = max(
        ["mean", "suffix"],
        key=lambda pt: ensemble_results[pt]["result"].au_prc,
    )

    margin = best_ensemble - best_single
    error_corr = complementarity_stats[best_ensemble_type]["spearman_error_correlation"]

    passed = margin > 0.01 and error_corr < 0.3

    return {
        "passed": passed,
        "best_single_au_prc": round(best_single, 6),
        "best_ensemble_au_prc": round(best_ensemble, 6),
        "margin": round(margin, 6),
        "best_ensemble_type": best_ensemble_type,
        "error_correlation": round(error_corr, 6),
        "criteria": {
            "margin_threshold": 0.01,
            "correlation_threshold": 0.3,
        },
        "verdict": (
            "PASS: Ensemble shows complementary improvement — CC++ "
            "probe-classifier complementarity transfers to biosafety."
            if passed
            else "FAIL: Ensemble does not meaningfully improve over best "
            "single component. Report as negative result."
        ),
    }
