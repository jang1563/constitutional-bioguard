"""Evaluation visualization: confusion matrix, ROC curve, per-category F1,
adversarial heatmap, and over-refusal analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constitutional_bioguard.config import FIGURES_DIR, FIGURE_DPI, FIGURE_FONT_SIZE, METRICS_DIR

logger = logging.getLogger(__name__)

# Global style
plt.rcParams.update({
    "font.size": FIGURE_FONT_SIZE,
    "figure.dpi": FIGURE_DPI,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": FIGURE_DPI,
})


def plot_confusion_matrix(output_dir: Optional[Path] = None) -> Path:
    """Plot the confusion matrix from evaluation results."""
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_file = METRICS_DIR / "confusion_matrix.json"
    with open(cm_file, encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["matrix"])
    labels = data["labels"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Bio-Safety Classifier Confusion Matrix")

    filepath = output_dir / "confusion_matrix.png"
    fig.savefig(filepath)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", filepath)
    return filepath


def plot_roc_curve(output_dir: Optional[Path] = None) -> Path:
    """Plot the ROC curve from test set predictions."""
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load internal evaluation for AUROC
    eval_file = METRICS_DIR / "internal_evaluation.json"
    with open(eval_file, encoding="utf-8") as f:
        data = json.load(f)

    auroc = data["internal_metrics"]["auroc"]

    # For a full ROC curve, we'd need the raw predictions.
    # Here we create a summary plot with the AUROC value.
    fig, ax = plt.subplots(figsize=(6, 5))

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUROC=0.50)")

    # Approximate ROC point from metrics
    fpr = data["internal_metrics"]["fpr"]
    recall = data["internal_metrics"]["recall"]
    ax.plot(fpr, recall, "ro", markersize=10, label=f"Operating point")
    ax.annotate(
        f"FPR={fpr:.3f}\nRecall={recall:.3f}",
        (fpr, recall),
        textcoords="offset points",
        xytext=(15, -15),
        fontsize=9,
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"Bio-Safety Classifier ROC (AUROC={auroc:.4f})")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    filepath = output_dir / "roc_curve.png"
    fig.savefig(filepath)
    plt.close(fig)
    logger.info("Saved ROC curve to %s", filepath)
    return filepath


def plot_per_category_f1(output_dir: Optional[Path] = None) -> Path:
    """Bar chart of F1 scores per NSABB category."""
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_file = METRICS_DIR / "internal_evaluation.json"
    with open(eval_file, encoding="utf-8") as f:
        data = json.load(f)

    categories = []
    f1_scores = []
    for cat_data in data["per_category_metrics"]:
        cat_name = cat_data["category"].replace("_", " ").title()
        categories.append(cat_name)
        f1_scores.append(cat_data["metrics"]["f1"])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(categories))
    bars = ax.barh(categories, f1_scores, color=colors)

    # Add value labels
    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=9,
        )

    # Target line
    ax.axvline(x=0.85, color="red", linestyle="--", alpha=0.5, label="Target (0.85)")

    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Category F1 Score (NSABB Categories)")
    ax.set_xlim([0, 1.05])
    ax.legend()

    filepath = output_dir / "per_category_f1.png"
    fig.savefig(filepath)
    plt.close(fig)
    logger.info("Saved per-category F1 chart to %s", filepath)
    return filepath


def plot_adversarial_heatmap(output_dir: Optional[Path] = None) -> Path:
    """Heatmap of attack success rates by attack type and category."""
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    adv_file = METRICS_DIR / "adversarial_results.json"
    with open(adv_file, encoding="utf-8") as f:
        results = json.load(f)

    # Organize by category
    categories = sorted(set(r["attack_category"] for r in results))
    attack_names = [r["attack_name"] for r in results]
    asr_values = [r["attack_success_rate"] * 100 for r in results]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    # Color by category
    cat_colors = {
        "character": "#2196F3",
        "encoding": "#FF9800",
        "semantic": "#4CAF50",
        "multilingual": "#9C27B0",
    }
    colors = [cat_colors.get(r["attack_category"], "#666") for r in results]

    bars = ax.bar(range(len(attack_names)), asr_values, color=colors)

    ax.set_xticks(range(len(attack_names)))
    ax.set_xticklabels(
        [n.replace("_", "\n") for n in attack_names],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Adversarial Robustness: Attack Success Rate by Type")

    # Target line
    ax.axhline(y=15, color="red", linestyle="--", alpha=0.5, label="Target (<15%)")

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=cat.title())
        for cat, color in cat_colors.items()
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color="red", linestyle="--", alpha=0.5, label="Target")
    )
    ax.legend(handles=legend_elements, loc="upper right")

    filepath = output_dir / "adversarial_heatmap.png"
    fig.savefig(filepath)
    plt.close(fig)
    logger.info("Saved adversarial heatmap to %s", filepath)
    return filepath


def plot_overrefusal_analysis(output_dir: Optional[Path] = None) -> Path:
    """Bar chart of FPR by benign source category."""
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    overrefusal_file = METRICS_DIR / "overrefusal_results.json"
    with open(overrefusal_file, encoding="utf-8") as f:
        data = json.load(f)

    per_source = data.get("per_source", {})
    if not per_source:
        logger.warning("No per-source data for over-refusal plot")
        return output_dir / "overrefusal_analysis.png"

    sources = list(per_source.keys())
    fprs = [per_source[s]["fpr"] * 100 for s in sources]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("Reds_r", len(sources))
    bars = ax.bar(sources, fprs, color=colors)

    for bar, fpr in zip(bars, fprs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{fpr:.2f}%",
            ha="center",
            fontsize=9,
        )

    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="Target (<2%)")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("Over-Refusal: FPR on Benign Queries by Source")
    ax.legend()

    filepath = output_dir / "overrefusal_analysis.png"
    fig.savefig(filepath)
    plt.close(fig)
    logger.info("Saved over-refusal analysis to %s", filepath)
    return filepath


def generate_all_figures(output_dir: Optional[Path] = None) -> list[Path]:
    """Generate all evaluation figures."""
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    plot_fns = [
        ("confusion_matrix", plot_confusion_matrix),
        ("roc_curve", plot_roc_curve),
        ("per_category_f1", plot_per_category_f1),
        ("adversarial_heatmap", plot_adversarial_heatmap),
        ("overrefusal_analysis", plot_overrefusal_analysis),
    ]

    for name, fn in plot_fns:
        try:
            path = fn(output_dir)
            figures.append(path)
            logger.info("Generated figure: %s", name)
        except FileNotFoundError as e:
            logger.warning("Skipping %s (missing data): %s", name, e)
        except Exception as e:
            logger.error("Failed to generate %s: %s", name, e)

    return figures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_figures()
