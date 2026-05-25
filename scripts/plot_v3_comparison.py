#!/usr/bin/env python
"""Plot the three-way v3 comparison: bio recall vs cross-domain FAR.

Creates two figures:
  1. v3_comparison_bars.png — bar chart of FAR + recall across benchmarks
  2. v3_pareto.png — Pareto plot (bio recall vs avg cross-domain FAR)
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


def load_summary() -> dict:
    """Combine v3_eval_summary.json + old biothreat results."""
    summary_path = METRICS_DIR / "v3_eval_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    # Use existing compare_*_biothreat.json files if v3 summary missing them
    for tag in ["a_full", "v2", "v3"]:
        if (
            f"biothreat_{tag}" not in summary
            or summary[f"biothreat_{tag}"].get("recall") is None
        ):
            fp = METRICS_DIR / f"v3_compare_{tag}_biothreat.json"
            if not fp.exists():
                fp = METRICS_DIR / f"compare_{tag}_biothreat.json"
            if fp.exists():
                try:
                    with open(fp) as f:
                        d = json.load(f)
                    m = d["by_strategy"]["threat_level_4"]["variant_a"][
                        "metrics_at_0.5"
                    ]
                    summary[f"biothreat_{tag}"] = {
                        "f1": m["f1"],
                        "auroc": m["auroc"],
                        "recall": m["recall"],
                        "precision": m["precision"],
                        "fpr": m["fpr"],
                    }
                except Exception:
                    pass
    return summary


def plot_comparison_bars(summary: dict) -> Path:
    """Bar chart: cross-domain FAR + bio recall, 3 models side by side."""
    models = ["a_full", "v2", "v3"]
    model_labels = ["v1 (A_full)", "v2", "v3 (balanced)"]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]  # red, blue, green

    far_benchmarks = [
        ("wildguard_test", "WildGuard"),
        ("lab_bench", "LAB-Bench"),
        ("wmdp_cyber", "WMDP-Cyber"),
        ("wmdp_chem", "WMDP-Chem"),
        ("med_qa_test", "MedQA"),
        ("pubmed_qa_pqa_labeled", "PubMedQA"),
    ]
    recall_benchmarks = [
        ("biothreat", "BioThreat-Eval", "recall"),
        ("harmbench_bio_ho", "HarmBench bio", "far"),
        ("advbench_bio_ho", "AdvBench bio", "far"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: cross-domain FAR (lower = better)
    ax = axes[0]
    n = len(far_benchmarks)
    x = np.arange(n)
    w = 0.27
    for i, (m, lbl) in enumerate(zip(models, model_labels)):
        vals = [
            summary.get(f"{b}_{m}", {}).get("far", 0) * 100
            for b, _ in far_benchmarks
        ]
        ax.bar(x + (i - 1) * w, vals, w, label=lbl, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in far_benchmarks], rotation=30, ha="right")
    ax.set_ylabel("False Alarm Rate (%)")
    ax.set_title("Cross-domain FAR (lower is better)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="10% threshold")

    # Right panel: bio recall (higher = better)
    ax = axes[1]
    n = len(recall_benchmarks)
    x = np.arange(n)
    for i, (m, lbl) in enumerate(zip(models, model_labels)):
        vals = []
        for b, _, field in recall_benchmarks:
            v = summary.get(f"{b}_{m}", {}).get(field, 0)
            vals.append((v or 0) * 100)
        ax.bar(x + (i - 1) * w, vals, w, label=lbl, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n, _ in recall_benchmarks], rotation=20, ha="right")
    ax.set_ylabel("Flag Rate / Recall (%)")
    ax.set_title("Bio Recall (higher is better)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 105)

    plt.suptitle(
        "Three-way model comparison: v1 (shortcut) -> v2 (recall collapse) -> v3 (balanced)",
        fontsize=13,
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "v3_comparison_bars.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_pareto(summary: dict) -> Path:
    """Pareto scatter: avg bio recall vs avg cross-domain FAR."""
    far_keys = [
        "wildguard_test",
        "lab_bench",
        "wmdp_cyber",
        "wmdp_chem",
        "med_qa_test",
        "pubmed_qa_pqa_labeled",
    ]
    bio_keys = [
        ("biothreat", "recall"),
        ("harmbench_bio_ho", "far"),
        ("advbench_bio_ho", "far"),
    ]
    models = [("a_full", "v1 (A_full)", "#d62728"),
              ("v2", "v2", "#1f77b4"),
              ("v3", "v3 (balanced)", "#2ca02c")]

    fig, ax = plt.subplots(figsize=(8, 7))
    for m, lbl, c in models:
        fars = [
            summary.get(f"{k}_{m}", {}).get("far", 0) * 100 for k in far_keys
        ]
        bios = [
            (summary.get(f"{k}_{m}", {}).get(f, 0) or 0) * 100
            for k, f in bio_keys
        ]
        avg_far = np.mean(fars) if fars else 0
        avg_bio = np.mean(bios) if bios else 0
        ax.scatter(avg_far, avg_bio, s=300, color=c, label=lbl, edgecolor="black", zorder=3)
        ax.annotate(
            lbl, (avg_far, avg_bio),
            xytext=(10, 10), textcoords="offset points",
            fontsize=11, fontweight="bold",
        )

    # Ideal region
    ax.axvspan(0, 10, alpha=0.05, color="green")
    ax.axhspan(50, 100, alpha=0.05, color="green")

    ax.set_xlabel("Avg cross-domain FAR (%)  (lower is better)")
    ax.set_ylabel("Avg bio recall / flag rate (%)  (higher is better)")
    ax.set_title("v3 Pareto: bio recall vs cross-domain FAR")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, max(60, max(np.mean([
        summary.get(f"{k}_a_full", {}).get("far", 0) * 100 for k in far_keys
    ]), 60) + 5))
    ax.set_ylim(-5, 105)

    out_path = FIGURES_DIR / "v3_pareto.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    summary = load_summary()
    p1 = plot_comparison_bars(summary)
    p2 = plot_pareto(summary)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
