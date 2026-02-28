"""
Stock market / economic indicator dashboard: S&P 500 and VIX from yfinance.
Optional FRED series (e.g. unemployment) if FRED_API_KEY is set.
Saves to outputs/stock_dashboard.png
"""
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from src.config import OUTPUTS_DIR

# FRED is optional (free API key at https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = os.environ.get("FRED_API_KEY")


def fetch_sp500_vix(period: str = "2y") -> tuple[pd.Series, pd.Series]:
    """Fetch S&P 500 and VIX from Yahoo Finance. period e.g. '2y', '5y'."""
    sp = yf.Ticker("^GSPC")
    vix = yf.Ticker("^VIX")
    sp_hist = sp.history(period=period)
    vix_hist = vix.history(period=period)
    sp_close = sp_hist["Close"] if not sp_hist.empty else pd.Series(dtype=float)
    vix_close = vix_hist["Close"] if not vix_hist.empty else pd.Series(dtype=float)
    return sp_close, vix_close


def fetch_fred_series(series_id: str, period: str = "2y") -> pd.Series | None:
    """Fetch one FRED series if FRED_API_KEY is set. Returns None otherwise."""
    if not FRED_API_KEY:
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        end = datetime.now()
        start = end - timedelta(days=730 if period == "2y" else 1825)
        s = fred.get_series(series_id, start=start, end=end)
        return s
    except Exception:
        return None


def run(period: str = "2y") -> Path:
    """Build S&P 500 + VIX dashboard; save to outputs. Returns path to image."""
    sp, vix = fetch_sp500_vix(period=period)
    if sp.empty or vix.empty:
        raise ValueError("No data from yfinance for S&P 500 or VIX.")

    # Align to common index (business days); strip timezone for matplotlib
    common_idx = sp.index.intersection(vix.index)
    sp = sp.reindex(common_idx).ffill().dropna()
    vix = vix.reindex(common_idx).ffill().dropna()
    common_idx = sp.index.intersection(vix.index)
    sp, vix = sp.loc[common_idx], vix.loc[common_idx]
    if sp.index.tz is not None:
        sp = sp.tz_localize(None)
    if vix.index.tz is not None:
        vix = vix.tz_localize(None)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 0.8]})

    # S&P 500 (ensure numeric and timezone-naive for matplotlib)
    ax1 = axes[0]
    sp_vals = np.asarray(sp, dtype=float)
    ax1.fill_between(sp.index, sp_vals, alpha=0.3, color="#2ecc71")
    ax1.plot(sp.index, sp_vals, color="#27ae60", linewidth=1.2, label="S&P 500")
    ax1.set_ylabel("S&P 500", fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("S&P 500 and VIX (Fear & Greed proxy)", fontsize=12)

    # VIX
    ax2 = axes[1]
    vix_vals = np.asarray(vix, dtype=float)
    ax2.fill_between(vix.index, vix_vals, alpha=0.3, color="#e74c3c")
    ax2.plot(vix.index, vix_vals, color="#c0392b", linewidth=1.2, label="VIX")
    ax2.set_ylabel("VIX", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.tight_layout()

    out_path = OUTPUTS_DIR / "stock_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run()
