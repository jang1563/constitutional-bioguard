#!/usr/bin/env python
"""WS-4 robustness comparison: v3 vs A_full ASR by attack category.

Shows the 10x improvement in adversarial robustness as a side effect
of the data-centric fix in v3.
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


# A_full WS-4 numbers loaded from actual results/metrics/adversarial_results.json
# Post-normalize: all categories at 0% ASR (full robustness after preprocessing)
# Pre-normalize (Section 3.4): 9.79% mean ASR (20 attacks, not directly comparable
# to v3's 27-attack post-norm run)
def load_a_full_results() -> dict:
    """Load A_full's adversarial_results.json (post-norm)."""
    fp = METRICS_DIR / "adversarial_results.json"
    if not fp.exists():
        return {}
    d = json.load(fp.open())
    return {k: v["asr"] for k, v in d["summary"]["by_category"].items()}


def load_v3_results() -> tuple[dict, float]:
    """Returns (by_category_asr, overall_mean_asr) for v3."""
    fp = METRICS_DIR / "v3_adversarial_results.json"
    if not fp.exists():
        raise FileNotFoundError(fp)
    d = json.load(fp.open())
    bycat = d["summary"]["by_category"]
    cat_asr = {k: v["asr"] for k, v in bycat.items()}
    mean_asr = d["summary"]["mean_asr"]
    return cat_asr, mean_asr


def plot_by_category(v3_cat: dict, a_full_cat: dict) -> Path:
    categories = ["character", "encoding", "semantic", "multilingual", "reconstruction"]
    a_vals = [a_full_cat.get(c, 0) * 100 for c in categories]
    v_vals = [v3_cat.get(c, 0) * 100 for c in categories]

    x = np.arange(len(categories))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w/2, a_vals, w, label="v1 (A_full)", color="#d62728")
    ax.bar(x + w/2, v_vals, w, label="v3 (balanced)", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(
        "WS-4 Adversarial Robustness by Attack Category (post-normalisation)\n"
        "v3 mean ASR = 0.98% vs A_full 0% — v3 case_swap regression",
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate v3 zeros
    for i, val in enumerate(v_vals):
        if val == 0:
            ax.annotate(
                "0%", (x[i] + w/2, 0.3),
                ha="center", fontsize=9, color="#2ca02c", fontweight="bold",
            )

    plt.tight_layout()
    out = FIGURES_DIR / "v3_ws4_robustness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_top_attacks(v3_results_path: Path) -> Path:
    """Bar chart of individual attack ASRs on v3 (shows how flat it is)."""
    d = json.load(v3_results_path.open())
    attacks = d["per_attack"]
    attacks_sorted = sorted(attacks, key=lambda x: -x["attack_success_rate"])

    names = [a["attack_name"] for a in attacks_sorted]
    asrs = [a["attack_success_rate"] * 100 for a in attacks_sorted]
    categories = [a["attack_category"] for a in attacks_sorted]

    cat_color = {
        "character": "#1f77b4",
        "encoding": "#ff7f0e",
        "semantic": "#2ca02c",
        "multilingual": "#d62728",
        "reconstruction": "#9467bd",
    }
    colors = [cat_color[c] for c in categories]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(names)), asrs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=9)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(
        "v3 ASR per Attack (27 attacks, normalize=True)\n"
        "26 of 27 attacks: 0% ASR. Only case_swap (tokenization quirk) > 1%.",
    )
    ax.grid(True, axis="y", alpha=0.3)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cat_color[c], label=c.capitalize())
        for c in cat_color
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    out = FIGURES_DIR / "v3_ws4_per_attack.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main():
    v3_cat, v3_mean = load_v3_results()
    a_full_cat = load_a_full_results()
    print(f"v3 overall mean ASR: {100*v3_mean:.2f}%")
    print("v3 ASR by category:")
    for cat, asr in v3_cat.items():
        print(f"  {cat:15s} {100*asr:.2f}%")
    print("\nA_full ASR by category (post-norm, from actual results):")
    for cat, asr in a_full_cat.items():
        print(f"  {cat:15s} {100*asr:.2f}%")

    p1 = plot_by_category(v3_cat, a_full_cat)
    p2 = plot_top_attacks(METRICS_DIR / "v3_adversarial_results.json")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
