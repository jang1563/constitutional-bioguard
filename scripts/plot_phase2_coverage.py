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

    Five configurations compared (all on the same 400 HarmBench items):

      1. v3 alone — bio specialist (184M)
      2. WildGuard alone — general (7B)
      3. OR cascade — flag if EITHER fires (max-recall)
      4. AND cascade — flag if BOTH fire (max-precision)
      5. Semantic-routed cascade — WildGuard's decision is FINAL, EXCEPT
         on items whose semantic_category is "chemical_biological", where
         v3 is invoked as a second opinion (OR-combined with WG). This is
         the realistic CC++ Stage1+Stage2 model: specialist invoked only
         in-scope, gate uses ITEM METADATA (not model's own confidence).

    Critical fix from earlier version: routing decision is based on
    semantic_category, NOT v3's own probability. Using v3's own probability
    as the gate makes the routing equivalent to OR-cascade (circular dep).

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

    # New schema: predictions are dicts with metadata. Fall back to tuple
    # schema for backward compatibility.
    def normalise(p):
        if isinstance(p, dict):
            return p
        # Tuple schema: (label, pred, prob)
        return {"label": p[0], "pred": p[1], "prob": p[2],
                "category": "", "semantic_category": ""}

    v3_items = [normalise(p) for p in v3_preds]
    wg_items = [normalise(p) for p in wg_preds]

    # Build joint item list with both predictions + shared metadata
    items = []
    for v, w in zip(v3_items, wg_items):
        cat = (
            v.get("category") or v.get("semantic_category")
            or w.get("category") or w.get("semantic_category")
            or ""
        )
        items.append({
            "label": v["label"],
            "v3_pred": v["pred"],
            "v3_prob": v["prob"],
            "wg_pred": w["pred"],
            "wg_prob": w["prob"],
            "semantic_category": cat.lower(),
            # in_bio_scope is METADATA-driven, not model-driven:
            "in_bio_scope": cat.lower() == "chemical_biological",
        })

    n_pos = sum(1 for it in items if it["label"] == 1)
    n_neg = sum(1 for it in items if it["label"] == 0)
    n_bio = sum(1 for it in items if it["in_bio_scope"])

    def metrics_for(preds: list[int]) -> dict:
        tp = sum(1 for it, p in zip(items, preds) if it["label"] == 1 and p == 1)
        fp = sum(1 for it, p in zip(items, preds) if it["label"] == 0 and p == 1)
        recall = tp / n_pos if n_pos else 0
        far = fp / n_neg if n_neg else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
        return {
            "recall": recall * 100, "far": far * 100,
            "prec": prec * 100, "f1": f1,
        }

    # 1. v3 alone
    m_v3 = metrics_for([it["v3_pred"] for it in items])
    # 2. WildGuard alone
    m_wg = metrics_for([it["wg_pred"] for it in items])
    # 3. OR cascade — max recall
    m_or = metrics_for([
        1 if (it["v3_pred"] or it["wg_pred"]) else 0 for it in items
    ])
    # 4. AND cascade — max precision
    m_and = metrics_for([
        1 if (it["v3_pred"] and it["wg_pred"]) else 0 for it in items
    ])
    # 5. SEMANTIC-routed cascade — true Stage1+Stage2 with metadata gate
    #    On bio items: OR (WG OR v3). On non-bio items: WG only.
    m_routed = metrics_for([
        (1 if (it["wg_pred"] or it["v3_pred"]) else 0)
        if it["in_bio_scope"]
        else it["wg_pred"]
        for it in items
    ])

    rows = [
        ("v3 alone\n(184M)",                                m_v3),
        ("WildGuard alone\n(7B)",                            m_wg),
        ("OR cascade\nany-fires",                            m_or),
        ("AND cascade\nboth-fire",                           m_and),
        (f"Semantic routing\n(v3 only on bio, n_bio={n_bio})", m_routed),
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
        "Semantic routing (right-most) gates v3 on bio category via metadata, "
        "not model confidence — true Stage1+Stage2 deployment",
        fontsize=12,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "phase2_cascade_simulation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # Per-category cascade breakdown — show where routing actually helps
    cascade_per_cat: dict[str, dict] = {}
    cats = set(it["semantic_category"] for it in items if it["semantic_category"])
    for cat in sorted(cats):
        cat_idx = [i for i, it in enumerate(items) if it["semantic_category"] == cat]
        if len(cat_idx) < 5:  # too small for stats
            continue
        sub = [items[i] for i in cat_idx]
        cat_metrics = {
            "n": len(sub),
            "in_bio_scope": sub[0]["in_bio_scope"],
            "v3_alone_recall": sum(s["v3_pred"] for s in sub if s["label"] == 1)
                              / max(sum(1 for s in sub if s["label"] == 1), 1) * 100,
            "wg_alone_recall": sum(s["wg_pred"] for s in sub if s["label"] == 1)
                              / max(sum(1 for s in sub if s["label"] == 1), 1) * 100,
            "routed_recall": sum(
                (1 if (s["wg_pred"] or s["v3_pred"]) else 0)
                if s["in_bio_scope"] else s["wg_pred"]
                for s in sub if s["label"] == 1
            ) / max(sum(1 for s in sub if s["label"] == 1), 1) * 100,
        }
        cascade_per_cat[cat] = cat_metrics

    cascade_report = {
        "n_total": len(items),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_in_bio_scope": n_bio,
        "configurations": {
            "v3_alone": m_v3,
            "wildguard_alone": m_wg,
            "or_cascade": m_or,
            "and_cascade": m_and,
            "semantic_routing": m_routed,
        },
        "per_category": cascade_per_cat,
        "routing_method": (
            "semantic_category == 'chemical_biological' → OR cascade; "
            "else WildGuard only. Gate uses item METADATA (not model "
            "confidence), avoiding circular dependency."
        ),
    }
    rep_path = METRICS_DIR / "phase2_cascade_report.json"
    with open(rep_path, "w") as f:
        json.dump(cascade_report, f, indent=2)
    print(f"Cascade report: {rep_path}")
    return out


def plot_beavertails_heatmap() -> Path:
    """BeaverTails uses REAL LLM responses — shows v3's true domain boundary."""
    summaries = {m: load_summary(m) for m, _, _ in MODELS}

    # Use top categories by N
    all_cats: set[str] = set()
    cat_n: dict[str, int] = {}
    for s in summaries.values():
        bench = s.get("benchmarks", {}).get("beavertails", {})
        cats = bench.get("by_category", {}) or {}
        for c, m in cats.items():
            all_cats.add(c)
            cat_n[c] = max(cat_n.get(c, 0), m.get("n", 0))
    # Pick top 8 categories by n
    categories = sorted(
        [c for c in all_cats if cat_n.get(c, 0) >= 50],
        key=lambda c: -cat_n[c],
    )[:8]

    M = len(MODELS)
    C = len(categories)
    mat = np.full((M, C), np.nan)
    sample_sizes = np.zeros((M, C), dtype=int)
    for mi, (model_key, _, _) in enumerate(MODELS):
        s = summaries.get(model_key, {})
        bench = s.get("benchmarks", {}).get("beavertails", {})
        cats = bench.get("by_category", {}) or {}
        for ci, cat in enumerate(categories):
            m = cats.get(cat, {})
            sample_sizes[mi, ci] = m.get("n", 0)
            if "flag_rate" in m:
                mat[mi, ci] = m["flag_rate"] * 100

    fig, ax = plt.subplots(figsize=(max(10, 1.0 * C + 3), 0.9 * M + 2.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    # RdYlGn_r: 0 = green (good for SAFE-ish), 100 = red (bad if items mostly SAFE)
    # BeaverTails items are mixed; for SAFE category items, low flag rate = good.

    for mi in range(M):
        for ci in range(C):
            if not np.isnan(mat[mi, ci]):
                color = "white" if mat[mi, ci] > 50 else "black"
                ax.text(
                    ci, mi, f"{mat[mi, ci]:.1f}%\n(n={sample_sizes[mi, ci]})",
                    ha="center", va="center", color=color, fontsize=9,
                )

    ax.set_xticks(range(C))
    ax.set_xticklabels(
        [c.replace(",", "\n")[:25] for c in categories],
        rotation=20, ha="right", fontsize=8,
    )
    ax.set_yticks(range(M))
    ax.set_yticklabels([lbl for _, lbl, _ in MODELS])

    # Mark v3's stated coverage with green border (none of these
    # categories overlap with v3's bio scope, so no boxes)
    ax.set_title(
        "BeaverTails Flag Rate per Category (REAL LLM responses)\n"
        "v3 is silent on non-bio harm (by design); generalists fire broadly",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, label="Flag Rate (%)")
    plt.tight_layout()
    out = FIGURES_DIR / "phase2_beavertails_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main():
    p1 = plot_harmbench_heatmap()
    p2 = plot_xstest_overrefusal()
    p3 = simulate_cascade()
    p4 = plot_beavertails_heatmap()
    print(f"Saved heatmap (HarmBench, all UNSAFE w/ compliance template): {p1}")
    print(f"Saved xstest plot: {p2}")
    if p3:
        print(f"Saved cascade sim: {p3}")
    print(f"Saved heatmap (BeaverTails, real responses): {p4}")


if __name__ == "__main__":
    main()
