#!/usr/bin/env python3
"""Generate the GitHub social-preview (Open Graph) card.

1280x640 (GitHub's recommended 2:1). The card leads with the honest framing:
a dual-mode guard whose durable contribution is the evaluation discipline.

Run: python scripts/make_social_preview.py
Out: docs/assets/social_preview.png
Then upload manually: GitHub repo -> Settings -> General -> Social preview -> Edit.
(GitHub has no API for the social-preview image; it is web-UI only.)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG = "#0d1117"        # GitHub dark
FG = "#e6edf3"        # near-white
MUTE = "#9aa4af"      # muted gray
CRIMSON = "#ff5d73"
GREEN = "#3fb950"
AMBER = "#d29922"


def main():
    out = Path("docs/assets/social_preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # left accent bar
    ax.add_patch(Rectangle((0.0, 0.0), 0.018, 1.0, color=CRIMSON, zorder=2))

    # eyebrow
    ax.text(0.07, 0.86, "AI SAFETY  ·  BIOSECURITY  ·  EVALUATION", color=CRIMSON,
            fontsize=14.5, fontweight="bold")

    # title
    ax.text(0.07, 0.70, "Constitutional BioGuard", color=FG, fontsize=46,
            fontweight="bold")

    # tagline
    ax.text(0.07, 0.585,
            "A dual-mode biosafety content classifier —\nand an honest evaluation case study.",
            color=MUTE, fontsize=20, linespacing=1.25, va="top")

    # three honest bullets with colored dots
    bullets = [
        (GREEN, "Dual-mode guard: response + prompt heads (2x184M DeBERTa-v3),"
                " constitution-driven data."),
        (AMBER, "5 self-audits — each reversed one of the project's own headline claims."),
        (CRIMSON, "Honest negative result: Pareto-dominated by a smaller open model"
                  " (Qwen3Guard-0.6B)."),
    ]
    y = 0.40
    for color, text in bullets:
        ax.plot(0.085, y, marker="o", markersize=11, color=color, zorder=3)
        ax.text(0.11, y, text, color=FG, fontsize=15.5, va="center")
        y -= 0.085

    # footer
    ax.text(0.07, 0.075, "github.com/jang1563/constitutional-bioguard", color=MUTE,
            fontsize=14.5, fontweight="bold")
    ax.text(0.07, 0.03, "JangKeun Kim · Weill Cornell Medicine", color=MUTE, fontsize=13)

    fig.savefig(out, facecolor=BG, bbox_inches=None)
    print(f"wrote {out}  (1280x640)")


if __name__ == "__main__":
    main()
