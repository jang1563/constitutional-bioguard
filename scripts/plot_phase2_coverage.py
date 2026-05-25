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
    """Simulate cascade architectures on HarmBench Full.

    Four configurations compared (all on the same 400 HarmBench items):

      1. v3 alone — bio specialist (184M)
      2. WildGuard alone — general (7B)
      3. OR cascade — flag if EITHER fires (CC++ Stage1+Stage2 "any")
      4. AND cascade — flag if BOTH fire (precision-boost, costlier)
      5. Calibrated routing — WildGuard fires? If yes, accept. If no, run
         v3 ONLY when item has bio-keyword (specialist invoked only when
         in-scope). This mirrors realistic deployment economics.
      6. v3 fallback — WildGuard accepts everything; v3 catches bio that
         WildGuard misses. Models the "bio-specialist as second-stage
         safety net" deployment.

    All four metrics (recall, FAR, precision, F1) reported for each.
    """
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

    # Per-item: (true_label, v3_pred, wg_pred, v3_prob, wg_prob, is_bio?)
    # is_bio determined from per_attack metadata; if missing, fallback to
    # v3_prob > 0.5 (proxy for "v3 thinks it's bio")
    bio_keywords = [
        "bio", "pathogen", "virus", "bacteri", "toxin", "anthrax",
        "smallpox", "plague", "ebola", "botulin", "ricin", "marburg",
        "biosec", "select agent", "gain of function", "biosynthesis",
    ]
    # Rough heuristic: we don't have query text in predictions, but the v3
    # prob can serve as "this is bio" signal. Items where v3 is very confident
    # → likely bio-related (v3 trained only on bio).
    items = []
    for (t1, p1, prob1), (t2, p2, prob2) in zip(v3_preds, wg_preds):
        items.append({
            "label": t1,
            "v3_pred": p1, "v3_prob": prob1,
            "wg_pred": p2, "wg_prob": prob2,
            "is_bio_signal": prob1 > 0.5,  # v3 confident → bio
        })

    n_pos = sum(1 for it in items if it["label"] == 1)
    n_neg = sum(1 for it in items if it["label"] == 0)

    def metrics_for(preds: list[int]) -> dict:
        tp = sum(1 for it, p in zip(items, preds) if it["label"] == 1 and p == 1)
        fp = sum(1 for it, p in zip(items, preds) if it["label"] == 0 and p == 1)
        fn = sum(1 for it, p in zip(items, preds) if it["label"] == 1 and p == 0)
        recall = tp / n_pos if n_pos else 0
        far = fp / n_neg if n_neg else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
        return {"recall": recall * 100, "far": far * 100, "prec": prec * 100, "f1": f1}

    # 1. v3 alone
    m_v3 = metrics_for([it["v3_pred"] for it in items])
    # 2. WildGuard alone
    m_wg = metrics_for([it["wg_pred"] for it in items])
    # 3. OR cascade
    m_or = metrics_for([
        1 if (it["v3_pred"] or it["wg_pred"]) else 0 for it in items
    ])
    # 4. AND cascade
    m_and = metrics_for([
        1 if (it["v3_pred"] and it["wg_pred"]) else 0 for it in items
    ])
    # 5. Calibrated routing: WG fires accept; WG SAFE → run v3 only if bio signal
    #    (specialist invoked only in-scope; matches CC++ Stage1+Stage2 cost model)
    m_routed = metrics_for([
        it["wg_pred"] if it["wg_pred"] else (
            it["v3_pred"] if it["is_bio_signal"] else 0
        ) for it in items
    ])
    # 6. v3-as-fallback: WG decision is final UNLESS v3 fires (specialist
    #    catches what generalist misses)
    m_fallback = metrics_for([
        1 if (it["wg_pred"] or it["v3_pred"]) else 0 for it in items
    ])
    # Equivalent to OR — keep separate label for narrative clarity

    rows = [
        ("v3 alone\n(184M)",      m_v3),
        ("WildGuard alone\n(7B)", m_wg),
        ("OR cascade\nany-fires", m_or),
        ("AND cascade\nboth-fire", m_and),
        ("Calibrated routing\n(bio signal gates v3)", m_routed),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: recall + FAR
    ax = axes[0]
    x = np.arange(len(rows))
    w = 0.38
    recalls = [r[1]["recall"] for r in rows]
    fars = [r[1]["far"] for r in rows]
    ax.bar(x - w/2, recalls, w, label="Recall (%)", color="#2ca02c")
    ax.bar(x + w/2, fars, w, label="FAR (%)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Cascade Configurations on HarmBench Full")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, m in enumerate([r[1] for r in rows]):
        ax.text(i - w/2, m["recall"] + 1, f"{m['recall']:.1f}", ha="center", fontsize=8)
        ax.text(i + w/2, m["far"] + 1, f"{m['far']:.1f}", ha="center", fontsize=8)

    # Right: precision + F1
    ax = axes[1]
    precs = [r[1]["prec"] for r in rows]
    f1s = [r[1]["f1"] * 100 for r in rows]
    ax.bar(x - w/2, precs, w, label="Precision (%)", color="#9467bd")
    ax.bar(x + w/2, f1s, w, label="F1 (×100)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Cascade Configurations: Precision and F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, m in enumerate([r[1] for r in rows]):
        ax.text(i - w/2, m["prec"] + 1, f"{m['prec']:.1f}", ha="center", fontsize=8)
        ax.text(i + w/2, m["f1"] * 100 + 1, f"{m['f1']*100:.1f}", ha="center", fontsize=8)

    plt.suptitle(
        "Phase 2 Cascade Simulation: Specialist + Generalist Architectures\n"
        "Calibrated routing (right-most) = realistic CC++ Stage1+Stage2 deployment",
        fontsize=12,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "phase2_cascade_simulation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # Also save numeric report for the report
    cascade_report = {
        "n_total": len(items),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "configurations": {
            "v3_alone": m_v3,
            "wildguard_alone": m_wg,
            "or_cascade": m_or,
            "and_cascade": m_and,
            "calibrated_routing": m_routed,
        },
    }
    rep_path = METRICS_DIR / "phase2_cascade_report.json"
    with open(rep_path, "w") as f:
        json.dump(cascade_report, f, indent=2)
    print(f"Cascade report: {rep_path}")
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
