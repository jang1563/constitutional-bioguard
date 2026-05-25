#!/usr/bin/env python
"""Phase 2 visualisations: Domain Coverage Heatmap + Over-refusal + Cascade.

Inputs (from results/metrics/):
  - phase2_v3_summary.json
  - phase2_wildguard_7b_summary.json
  - phase2_llama_guard_3_8b_summary.json
  - phase2_{model}_{benchmark}.json (per-bench, for category breakdown)

Outputs (to results/figures/):
  - phase2_domain_coverage_heatmap.png
  - phase2_xstest_overrefusal.png
  - phase2_cascade_simulation.png
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

MODELS = [
    ("v3", "v3 (184M)", "#2ca02c"),
    ("wildguard_7b", "WildGuard (7B)", "#1f77b4"),
    ("llama_guard_3_8b", "LLaMA-Guard 3 (8B)", "#ff7f0e"),
]

# Stated domain claims for the Domain Coverage matrix
DOMAIN_CLAIMS = {
    "v3":              {"chemical_biological"},  # only bio
    "wildguard_7b":    None,                      # general (None = all)
    "llama_guard_3_8b": None,                     # general
}


def load_summary(model: str) -> dict:
    p = METRICS_DIR / f"phase2_{model}_summary.json"
    if not p.exists():
        return {}
    return json.load(p.open())


def plot_harmbench_heatmap() -> Path:
    """Heatmap: rows = models, cols = HarmBench categories, cells = recall."""
    summaries = {m: load_summary(m) for m, _, _ in MODELS}

    # Gather all categories across models
    all_cats: set[str] = set()
    for s in summaries.values():
        bench = s.get("benchmarks", {}).get("harmbench_full", {})
        cats = bench.get("by_category", {}) or {}
        all_cats.update(cats.keys())
    categories = sorted([c for c in all_cats if c])

    # Build matrix: M (rows) x C (cols) of recall (flag_rate for label=1)
    M = len(MODELS)
    C = len(categories)
    mat = np.full((M, C), np.nan)
    sample_sizes = np.zeros((M, C), dtype=int)
    for mi, (model_key, _, _) in enumerate(MODELS):
        s = summaries.get(model_key, {})
        bench = s.get("benchmarks", {}).get("harmbench_full", {})
        cats = bench.get("by_category", {}) or {}
        for ci, cat in enumerate(categories):
            m = cats.get(cat, {})
            sample_sizes[mi, ci] = m.get("n", 0)
            # All HarmBench items are label=1, so flag_rate = recall
            if "recall" in m:
                mat[mi, ci] = m["recall"] * 100
            elif "flag_rate" in m:
                mat[mi, ci] = m["flag_rate"] * 100

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * C + 3), 0.9 * M + 2.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    # Annotate cells with value + sample size
    for mi in range(M):
        for ci in range(C):
            if not np.isnan(mat[mi, ci]):
                color = "white" if mat[mi, ci] < 30 or mat[mi, ci] > 80 else "black"
                ax.text(
                    ci, mi, f"{mat[mi, ci]:.0f}%\n(n={sample_sizes[mi, ci]})",
                    ha="center", va="center", color=color, fontsize=9,
                )

    ax.set_xticks(range(C))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in categories],
        rotation=20, ha="right", fontsize=9,
    )
    ax.set_yticks(range(M))
    ax.set_yticklabels([lbl for _, lbl, _ in MODELS])

    # Mark stated coverage
    for mi, (mk, _, _) in enumerate(MODELS):
        claims = DOMAIN_CLAIMS.get(mk)
        for ci, cat in enumerate(categories):
            if claims is None or cat in claims:
                ax.add_patch(plt.Rectangle(
                    (ci - 0.5, mi - 0.5), 1, 1,
                    fill=False, edgecolor="blue", linewidth=2,
                ))

    ax.set_title(
        "HarmBench Domain Coverage: Recall by Category (blue box = stated coverage)\n"
        "v3 is bio-only by design; WildGuard + LLaMA-Guard are general-purpose",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, label="Recall / Flag Rate (%)")
    plt.tight_layout()
    out = FIGURES_DIR / "phase2_domain_coverage_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_xstest_overrefusal() -> Path:
    """XSTest: over-refusal rate (FAR on SAFE prompts) vs UNSAFE recall."""
    summaries = {m: load_summary(m) for m, _, _ in MODELS}

    over_refusal = []  # FAR on SAFE-type prompts
    correct_refusal = []  # recall on UNSAFE prompts
    for mk, _, _ in MODELS:
        s = summaries.get(mk, {})
        bench = s.get("benchmarks", {}).get("xstest", {})
        by_type = bench.get("by_category", {}) or {}
        # XSTest types: "safe" categories vs "contrast/unsafe"
        safe_far = 0.0
        unsafe_recall = 0.0
        n_safe = n_unsafe = 0
        for t, m in by_type.items():
            n = m.get("n", 0)
            if "unsafe" in t.lower() or "contrast" in t.lower():
                unsafe_recall += m.get("flag_rate", 0) * n
                n_unsafe += n
            else:
                safe_far += m.get("flag_rate", 0) * n
                n_safe += n
        over_refusal.append(safe_far / max(n_safe, 1) * 100)
        correct_refusal.append(unsafe_recall / max(n_unsafe, 1) * 100)

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(MODELS))
    w = 0.38
    ax.bar(x - w/2, over_refusal, w, label="Over-refusal FAR on SAFE prompts", color="#d62728")
    ax.bar(x + w/2, correct_refusal, w, label="Correct refusal recall on UNSAFE prompts", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in MODELS])
    ax.set_ylabel("Rate (%)")
    ax.set_title(
        "XSTest: Over-refusal vs Correct Refusal\n"
        "(Lower red = better calibration; higher green = better safety)",
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate values
    for i, (over, corr) in enumerate(zip(over_refusal, correct_refusal)):
        ax.text(i - w/2, over + 1, f"{over:.1f}%", ha="center", fontsize=10)
        ax.text(i + w/2, corr + 1, f"{corr:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "phase2_xstest_overrefusal.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def simulate_cascade() -> Path:
    """Simulate WildGuard Stage 1 -> v3 Stage 2 cascade on HarmBench bio.

    Stage 1 (WildGuard): broad recall, escalate to Stage 2 if "looks bio".
    Stage 2 (v3): bio specialist, makes final decision.

    Compare:
      - WildGuard alone (Stage 1 final)
      - v3 alone (Stage 2 final)
      - Cascade: WildGuard catches non-bio, v3 catches bio
    """
    # Load per-bench predictions
    p_v3 = METRICS_DIR / "phase2_v3_harmbench_full.json"
    p_wg = METRICS_DIR / "phase2_wildguard_7b_harmbench_full.json"
    if not (p_v3.exists() and p_wg.exists()):
        print(f"Skipping cascade sim: need {p_v3.name} and {p_wg.name}")
        return None

    v3_data = json.load(p_v3.open())
    wg_data = json.load(p_wg.open())

    v3_preds = v3_data.get("predictions", [])
    wg_preds = wg_data.get("predictions", [])
    if not v3_preds or len(v3_preds) != len(wg_preds):
        print("Cascade sim: prediction lengths mismatch or missing")
        return None

    # Simulation: cascade keeps max of two preds (any flag → flag)
    # This is OR-cascade — boost overall recall
    n = len(v3_preds)
    n_caught_v3 = sum(1 for t, p, _ in v3_preds if t == 1 and p == 1)
    n_caught_wg = sum(1 for t, p, _ in wg_preds if t == 1 and p == 1)
    n_caught_or = sum(
        1 for (t, p1, _), (_, p2, _) in zip(v3_preds, wg_preds)
        if t == 1 and (p1 == 1 or p2 == 1)
    )
    n_total_pos = sum(1 for t, _, _ in v3_preds if t == 1)
    n_total_neg = sum(1 for t, _, _ in v3_preds if t == 0)

    # AND cascade — boost precision
    n_caught_and = sum(
        1 for (t, p1, _), (_, p2, _) in zip(v3_preds, wg_preds)
        if t == 1 and (p1 == 1 and p2 == 1)
    )
    n_false_v3 = sum(1 for t, p, _ in v3_preds if t == 0 and p == 1)
    n_false_wg = sum(1 for t, p, _ in wg_preds if t == 0 and p == 1)
    n_false_or = sum(
        1 for (t, p1, _), (_, p2, _) in zip(v3_preds, wg_preds)
        if t == 0 and (p1 == 1 or p2 == 1)
    )
    n_false_and = sum(
        1 for (t, p1, _), (_, p2, _) in zip(v3_preds, wg_preds)
        if t == 0 and (p1 == 1 and p2 == 1)
    )

    rows = [
        ("v3 alone (184M)",      n_caught_v3,  n_total_pos, n_false_v3,  n_total_neg),
        ("WildGuard alone (7B)", n_caught_wg,  n_total_pos, n_false_wg,  n_total_neg),
        ("OR cascade",           n_caught_or,  n_total_pos, n_false_or,  n_total_neg),
        ("AND cascade",          n_caught_and, n_total_pos, n_false_and, n_total_neg),
    ]

    labels = [r[0] for r in rows]
    recalls = [r[1] / max(r[2], 1) * 100 for r in rows]
    fars = [r[3] / max(r[4], 1) * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w/2, recalls, w, label="Recall (%)", color="#2ca02c")
    ax.bar(x + w/2, fars, w, label="FAR (%)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title(
        "Cascade Simulation on HarmBench Full\n"
        "OR = boost recall (catch more harm); AND = boost precision (reduce FAR)",
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (r, f) in enumerate(zip(recalls, fars)):
        ax.text(i - w/2, r + 1, f"{r:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, f + 1, f"{f:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "phase2_cascade_simulation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main():
    p1 = plot_harmbench_heatmap()
    p2 = plot_xstest_overrefusal()
    p3 = simulate_cascade()
    print(f"Saved heatmap: {p1}")
    print(f"Saved xstest plot: {p2}")
    if p3:
        print(f"Saved cascade sim: {p3}")


if __name__ == "__main__":
    main()
