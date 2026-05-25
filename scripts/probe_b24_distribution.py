#!/usr/bin/env python
"""B.2.4 Probability distribution histograms across benchmarks.

If v3 has stable bio detection, prob_unsafe should be bimodal on bio
benchmarks (regardless of distribution). If v3 has distribution-specific
calibration (shortcut on training distribution), histograms will differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from constitutional_bioguard.config import FIGURES_DIR, METRICS_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions(model: str, bench: str, phase: int = 2) -> list[dict]:
    """Load predictions for a model/bench combo. Returns list of dicts with
    keys label, pred, prob.
    """
    prefix = "phase2" if phase == 2 else "baseline"
    fp = METRICS_DIR / f"{prefix}_{model}_{bench}.json"
    if not fp.exists():
        return []
    d = json.load(fp.open())
    preds = d.get("predictions", [])
    if not preds:
        return []
    if isinstance(preds[0], dict):
        return [
            {"label": p.get("label", 0), "pred": p.get("pred", 0),
             "prob": p.get("prob", 0.0)}
            for p in preds
        ]
    return [{"label": p[0], "pred": p[1], "prob": p[2]} for p in preds]


def plot_distribution(model: str, model_label: str) -> Path:
    """Plot prob_unsafe distributions for one model across benchmarks."""

    # Bio benchmarks with mixed labels
    bench_specs = [
        ("biothreat", "BioThreat-Eval (in-distribution-ish)", 1),
        ("wildguard_native", "WildGuardTest native (OOD)", 2),
        ("beavertails", "BeaverTails (non-bio OOD)", 2),
        ("xstest", "XSTest (compliance template)", 2),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for i, (bench, title, phase) in enumerate(bench_specs):
        ax = axes[i]
        preds = load_predictions(model, bench, phase=phase)
        if not preds:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue

        probs_safe = [p["prob"] for p in preds if p["label"] == 0]
        probs_unsafe = [p["prob"] for p in preds if p["label"] == 1]

        bins = np.linspace(0, 1, 41)
        if probs_safe:
            ax.hist(probs_safe, bins=bins, alpha=0.5, label=f"SAFE (n={len(probs_safe)})",
                    color="#2ca02c", density=True)
        if probs_unsafe:
            ax.hist(probs_unsafe, bins=bins, alpha=0.5, label=f"UNSAFE (n={len(probs_unsafe)})",
                    color="#d62728", density=True)
        ax.axvline(0.5, linestyle="--", color="black", alpha=0.5, label="Threshold 0.5")
        ax.set_xlabel("prob_unsafe")
        ax.set_ylabel("Density")
        ax.set_title(f"{title}\n(n={len(preds)})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"B.2.4 — Probability distribution across benchmarks: {model_label}",
        fontsize=14,
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"phase3_probe_b24_distribution_{model}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_all_models_overlay() -> Path:
    """Overlay all 3 models on bio benchmark to see calibration differences."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bench_specs = [
        ("biothreat", "BioThreat-Eval", 1),
        ("wildguard_native", "WildGuardTest native (OOD bio mixed)", 2),
    ]
    colors = {"v3": "#2ca02c", "wildguard_7b": "#1f77b4", "llama_guard_3_8b": "#ff7f0e"}
    labels = {"v3": "v3 (184M)", "wildguard_7b": "WildGuard (7B)", "llama_guard_3_8b": "LLaMA-Guard 3 (8B)"}

    for ax, (bench, title, phase) in zip(axes, bench_specs):
        bins = np.linspace(0, 1, 31)
        for model_key in ["v3", "wildguard_7b", "llama_guard_3_8b"]:
            preds = load_predictions(model_key, bench, phase=phase)
            if not preds:
                continue
            probs = [p["prob"] for p in preds]
            ax.hist(probs, bins=bins, alpha=0.4, label=labels[model_key],
                    color=colors[model_key], density=True)
        ax.axvline(0.5, linestyle="--", color="black", alpha=0.5)
        ax.set_xlabel("prob_unsafe")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "B.2.4 Probability distributions (all items, all 3 models)\n"
        "Bimodality + calibration comparison across distributions",
        fontsize=12,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "phase3_probe_b24_overlay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main():
    print("B.2.4 Probability distribution probe")
    for model, label in [("v3", "v3 (184M)"), ("wildguard_7b", "WildGuard (7B)"),
                          ("llama_guard_3_8b", "LLaMA-Guard 3 (8B)")]:
        p = plot_distribution(model, label)
        print(f"  Saved: {p}")
    p_overlay = plot_all_models_overlay()
    print(f"  Saved overlay: {p_overlay}")


if __name__ == "__main__":
    main()
