"""
Generate the static Open Graph preview image for social sharing.

Jordan (1996-97 Bulls, red) vs Kobe (2008-09 Lakers, gold).

Run from project root:
    python -m scripts.generate_og_image
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import matplotlib.pyplot as plt
from src.shot_chart_comparison import build_comparison, TEAM_COLORS
from src.config import OUTPUTS_DIR


def main() -> None:
    color_a = TEAM_COLORS["CHI"][0]   # Bulls primary (red)
    color_b = TEAM_COLORS["LAL"][1]   # Lakers alternate (gold)

    fig = build_comparison(
        "Michael Jordan", "Kobe Bryant",
        season_a="1996-97", season_b="2008-09",
        color_a=color_a, color_b=color_b,
        save=False,
    )

    fig.text(
        0.98, 0.02, "statshot.io",
        ha="right", va="bottom",
        fontsize=11, color="#e56020", alpha=0.6,
        fontweight="bold", fontstyle="italic",
        transform=fig.transFigure,
    )

    out_path = OUTPUTS_DIR / "og-preview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"OG preview image saved: {out_path}")


if __name__ == "__main__":
    main()
