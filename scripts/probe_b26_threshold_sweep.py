#!/usr/bin/env python
"""B.2.6 Threshold sweep across distributions.

For each benchmark, plot F1 / Precision / Recall as a function of threshold.
If v3 has a single distribution-invariant optimal threshold, curves align.
If optimal thresholds drift per distribution, v3 has distribution-specific
calibration (not a feature shortcut, but a calibration shortcut).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from constitutional_bioguard.config import FIGURES_DIR, METRICS_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions(model: str, bench: str, phase: int = 2) -> list[dict]:
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
            {"label": p.get("label", 0), "prob": p.get("prob", 0.0)}
            for p in preds
        ]
    return [{"label": p[0], "prob": p[2]} for p in preds]


def sweep_thresholds(preds: list[dict]) -> dict:
    if not preds:
        return {}
    y = np.array([p["label"] for p in preds])
    probs = np.array([p["prob"] for p in preds])
    if len(set(y)) < 2:
        return {}
    thresholds = np.arange(0.05, 1.0, 0.025)
    rows = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        if pred.sum() == 0:
            f1, prec, rec = 0.0, 0.0, 0.0
        else:
            f1 = f1_score(y, pred, zero_division=0)
            prec = precision_score(y, pred, zero_division=0)
            rec = recall_score(y, pred, zero_division=0)
        rows.append({"threshold": t, "f1": f1, "precision": prec, "recall": rec})
    best = max(rows, key=lambda r: r["f1"])
    return {
        "thresholds": [r["threshold"] for r in rows],
        "f1": [r["f1"] for r in rows],
        "precision": [r["precision"] for r in rows],
        "recall": [r["recall"] for r in rows],
        "best_threshold": best["threshold"],
        "best_f1": best["f1"],
    }


def plot_sweep(model: str, model_label: str) -> Path:
    bench_specs = [
        ("biothreat", "BioThreat-Eval", 1),
        ("wildguard_native", "WildGuardTest native", 2),
        ("beavertails", "BeaverTails", 2),
        ("xstest", "XSTest", 2),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    best_points = []

    for i, (bench, title, phase) in enumerate(bench_specs):
        preds = load_predictions(model, bench, phase=phase)
        s = sweep_thresholds(preds)
        if not s:
            continue
        ax.plot(s["thresholds"], s["f1"],
                label=f"{title} (best t={s['best_threshold']:.2f}, F1={s['best_f1']:.3f})",
                color=colors[i], linewidth=2)
        ax.scatter([s["best_threshold"]], [s["best_f1"]],
                   color=colors[i], s=100, zorder=5, edgecolor="black", marker="*")
        best_points.append((title, s["best_threshold"], s["best_f1"]))

    ax.axvline(0.5, linestyle="--", color="black", alpha=0.5, label="Default threshold 0.5")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_title(
        f"B.2.6 Threshold sweep: {model_label}\n"
        f"If optimal thresholds cluster, calibration is stable; if spread, distribution-specific",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    out = FIGURES_DIR / f"phase3_probe_b26_sweep_{model}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # Also save report
    return out, best_points


def main():
    print("B.2.6 Threshold sweep probe")
    print("=" * 80)
    all_best = {}
    for model, label in [("v3", "v3 (184M)"), ("wildguard_7b", "WildGuard (7B)"),
                          ("llama_guard_3_8b", "LLaMA-Guard 3 (8B)")]:
        path, best = plot_sweep(model, label)
        print(f"\n## {label}")
        print(f"  Saved: {path}")
        for title, t, f1 in best:
            print(f"    {title:30s}  best_threshold={t:.2f}  best_F1={f1:.4f}")
        all_best[model] = best

    # Save numeric report
    rep_path = METRICS_DIR / "phase3_probe_b26_threshold_sweep.json"
    with open(rep_path, "w") as f:
        json.dump(
            {model: [{"benchmark": t, "best_threshold": th, "best_f1": f1}
                     for t, th, f1 in lst]
             for model, lst in all_best.items()},
            f, indent=2,
        )
    print(f"\nNumeric report: {rep_path}")


if __name__ == "__main__":
    main()
