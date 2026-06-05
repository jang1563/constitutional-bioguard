#!/usr/bin/env python3
"""Size-peer Pareto plot for the dual-mode response head (v8bh).

Recall vs over-refusal on real bio responses (n=554, 343 harm / 211 benign),
all guards scored on the SAME items. Numbers are the authoritative figures from
docs/MODEL_CARD.md / the HF response-head card. The point of the figure is the
honest headline: the response head is in the recall band of 7-9B guards but is
Pareto-dominated by the openly-available Qwen3Guard-0.6B.

Run: python scripts/plot_size_peer_pareto.py
Out: results/figures/size_peer_pareto.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (name, params_billion, recall, over_refusal)  -- source: docs/MODEL_CARD.md, n=554
DATA = [
    ("Qwen3Guard-0.6B", 0.6, 0.933, 0.142),
    ("response head (v8bh)", 0.184, 0.921, 0.194),
    ("WildGuard-7B", 7.0, 0.904, 0.100),
    ("Granite-Guardian-2B", 2.0, 0.880, 0.123),
    ("Llama-Guard-3-8B", 8.0, 0.851, 0.052),
    ("ShieldGemma-9B", 9.0, 0.615, 0.033),
]
OURS = "response head (v8bh)"
DOMINATOR = "Qwen3Guard-0.6B"


def pareto_frontier(pts):
    """Non-dominated set: maximize recall, minimize over-refusal."""
    front = []
    for name, _, rec, ov in pts:
        dominated = any(
            (o_ov <= ov and o_rec >= rec) and (o_ov < ov or o_rec > rec)
            for o_name, _, o_rec, o_ov in pts if o_name != name
        )
        if not dominated:
            front.append((name, rec, ov))
    return sorted(front, key=lambda t: t[2])  # by over-refusal


def main():
    out = Path("results/figures/size_peer_pareto.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=160)

    # Pareto frontier line (lower-left envelope of the "better" corner)
    front = pareto_frontier(DATA)
    ax.plot([ov for _, _, ov in front], [rec for _, rec, _ in front],
            ls="--", lw=1.3, color="#9aa0a6", zorder=1, label="Pareto frontier")

    # per-label offset (dx_pts, dy_pts, ha) hand-tuned to avoid overlap
    label_pos = {
        "Qwen3Guard-0.6B": (0, 15, "center"),
        "response head (v8bh)": (0, -34, "center"),
        "WildGuard-7B": (12, 9, "left"),
        "Granite-Guardian-2B": (12, -26, "left"),
        "Llama-Guard-3-8B": (12, 9, "left"),
        "ShieldGemma-9B": (12, 9, "left"),
    }
    for name, pb, rec, ov in DATA:
        is_ours = name == OURS
        is_dom = name == DOMINATOR
        color = "#d11149" if is_ours else ("#2e8b57" if is_dom else "#5f6b7a")
        size = 120 + 900 * (pb ** 0.5) / (9.0 ** 0.5)  # area grows with sqrt(params)
        ax.scatter(ov, rec, s=size, color=color, alpha=0.85,
                   edgecolors="black", linewidths=1.4 if (is_ours or is_dom) else 0.6,
                   zorder=3)
        dx, dy, ha = label_pos[name]
        ax.annotate(f"{name}\n({pb:g}B · R={rec:.3f} · OR={ov:.3f})",
                    (ov, rec), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.2, ha=ha,
                    fontweight="bold" if (is_ours or is_dom) else "normal",
                    color="black")

    # dominance arrow: ours -> Qwen
    o = next(d for d in DATA if d[0] == OURS)
    q = next(d for d in DATA if d[0] == DOMINATOR)
    ax.annotate("", xy=(q[3], q[2]), xytext=(o[3], o[2]),
                arrowprops=dict(arrowstyle="-|>", color="#d11149", lw=1.6,
                                ls=(0, (4, 3)), alpha=0.8), zorder=2)
    ax.text((o[3] + q[3]) / 2, (o[2] + q[2]) / 2 - 0.024,
            "Pareto-dominated\nby Qwen3Guard",
            fontsize=8.0, color="#d11149", ha="center", style="italic", zorder=4)

    # "better" direction cues -- best corner is top-left
    ax.annotate("better →\n(higher recall)", xy=(0.012, 0.945), fontsize=8,
                color="#3c4043", rotation=90, va="top")
    ax.annotate("← better (lower over-refusal)", xy=(0.02, 0.585), fontsize=8,
                color="#3c4043")

    ax.set_xlabel("Over-refusal  (FPR on benign bio responses)", fontsize=10.5)
    ax.set_ylabel("Recall  (harmful bio responses caught)", fontsize=10.5)
    ax.set_title("Size-peer comparison on bio response-harm (n=554)\n"
                 "bubble area ∝ parameters · same items for all guards",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlim(0.0, 0.235)
    ax.set_ylim(0.56, 0.98)
    ax.grid(True, ls=":", lw=0.6, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print("Pareto frontier:", [n for n, _, _ in front])


if __name__ == "__main__":
    main()
