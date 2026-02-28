"""
Where Your Dollar Goes: Consumer spending treemap from BLS-style data.
Uses data/consumer_expenditure_bls.csv (sourced from BLS Consumer Expenditure Survey).
Saves to outputs/spending_treemap.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import numpy as np

from src.config import DATA_DIR, OUTPUTS_DIR

BLS_CSV = DATA_DIR / "consumer_expenditure_bls.csv"


def load_bls_data() -> pd.DataFrame:
    """Load BLS-style consumer expenditure by category."""
    if not BLS_CSV.exists():
        raise FileNotFoundError(f"Data file not found: {BLS_CSV}")
    return pd.read_csv(BLS_CSV)


def run() -> Path:
    """Build treemap of average U.S. consumer spending by category; save to outputs."""
    df = load_bls_data()
    df = df.sort_values("amount_usd", ascending=False).reset_index(drop=True)

    sizes = df["amount_usd"].values
    labels = [f"{row['category']}\n${row['amount_usd']:,}" for _, row in df.iterrows()]

    # Color gradient (blues/greens)
    n = len(sizes)
    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.5 * i / max(n, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85, pad=True, text_kwargs={"fontsize": 8}, ax=ax)
    ax.set_title("Where Your Dollar Goes — Average U.S. Consumer Spending (BLS-style, 2023)\nSource: BLS Consumer Expenditure Survey", fontsize=12)
    ax.axis("off")

    total = sizes.sum()
    ax.text(0.5, -0.02, f"Total average annual expenditures: ${total:,.0f}", transform=ax.transAxes, ha="center", fontsize=10)

    out_path = OUTPUTS_DIR / "spending_treemap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run()
