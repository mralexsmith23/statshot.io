"""
NBA Shot Chart Heatmap — pulls shot data via nba_api and plots on a half-court.
Saves to outputs/nba_shot_chart_<player_name>.png
"""
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from nba_api.stats.endpoints import shotchartdetail, commonplayerinfo
from nba_api.stats.static import players

from src.config import OUTPUTS_DIR

# NBA half-court dimensions (feet). API LOC_X/LOC_Y are in 1/10 foot from basket center.
HALF_COURT_LENGTH_FT = 47
COURT_WIDTH_FT = 50
HOOP_OFF_FLOOR_FT = 10  # rim height for reference; we draw 2D from above
# Shot chart coordinates: origin at basket, X = sideline direction, Y = toward half-court
# Typical scale: 1 unit ≈ 0.1 ft (so 470 = 47 ft)
SCALE = 0.1  # 1 API unit = 0.1 feet


def get_player_id(name: str) -> int | None:
    """Resolve player name to NBA player ID."""
    matches = players.find_players_by_full_name(name)
    if not matches:
        return None
    return matches[0]["id"]


def fetch_shot_chart(player_id: int, season: str = "2025-26") -> pd.DataFrame:
    """Fetch shot chart detail for a player. season e.g. '2024-25'."""
    # Get team_id for the player (required by API)
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    info_df = info.get_data_frames()[0]
    if info_df.empty:
        return pd.DataFrame()
    team_id = int(info_df["TEAM_ID"].iloc[0])
    time.sleep(0.6)  # rate limit
    shot = shotchartdetail.ShotChartDetail(
        team_id=int(team_id),
        player_id=int(player_id),
        season_nullable=season,
        context_measure_simple="FGA",
    )
    return shot.get_data_frames()[0]


def draw_half_court(ax: plt.Axes, color: str = "black", lw: float = 1.5) -> None:
    """Draw NBA half-court outline on ax (view from above, basket at bottom)."""
    # All in feet: basket at (0, 0), court extends to y = 47, x from -25 to 25
    ax.set_xlim(-30, 30)
    ax.set_ylim(-5, 52)
    ax.set_aspect("equal")
    ax.axis("off")

    # Outer court rectangle (half court)
    rect = mpatches.Rectangle((-COURT_WIDTH_FT / 2, 0), COURT_WIDTH_FT, HALF_COURT_LENGTH_FT, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)
    # Half-court line (sideline to sideline)
    ax.axhline(y=HALF_COURT_LENGTH_FT, xmin=0, xmax=1, color=color, lw=lw)
    # Center circle (radius 6 ft, center at (0, 47))
    circle = mpatches.Circle((0, HALF_COURT_LENGTH_FT), 6, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(circle)
    # Hoop (radius 1.5 ft at (0, 0) — we use 0,0 as basket center)
    hoop = mpatches.Circle((0, 0), 1.5, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(hoop)
    # Backboard (4 ft wide, 4 ft in front of hoop)
    ax.plot([-2, 2], [4, 4], color=color, lw=lw)
    # Key (paint) — 16 ft wide, 19 ft deep (from baseline)
    key = mpatches.Rectangle((-8, 0), 16, 19, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(key)
    # Three-point arc (23.75 ft at corners, 22 ft at top; we approximate with circle 23.75)
    # Arc center at basket, radius 23.75; only the part in front of the backboard
    arc = mpatches.Arc((0, 0), 23.75 * 2, 23.75 * 2, theta1=0, theta2=180, edgecolor=color, linewidth=lw)
    ax.add_patch(arc)
    # Corner 3s (two short lines from baseline)
    ax.plot([-25, -22], [0, 0], color=color, lw=lw)
    ax.plot([22, 25], [0, 0], color=color, lw=lw)


def api_to_court_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Convert API LOC_X, LOC_Y (1/10 ft) to court coordinates in feet. Basket at (0,0), Y toward half-court."""
    out = df.copy()
    # API: LOC_X positive = right, LOC_Y positive = toward half-court (typical)
    out["x_ft"] = out["LOC_X"] * SCALE
    out["y_ft"] = out["LOC_Y"] * SCALE
    return out


def plot_shot_chart(df: pd.DataFrame, player_name: str, season: str, ax: plt.Axes | None = None) -> plt.Figure:
    """Plot shot chart heatmap (2D density) on half-court. Returns figure."""
    if df.empty or "LOC_X" not in df.columns or "LOC_Y" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, "No shot data", ha="center", va="center", fontsize=14)
        return fig

    df = api_to_court_xy(df)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    draw_half_court(ax)
    # 2D histogram / density for heatmap
    x, y = df["x_ft"], df["y_ft"]
    # Clip to court for cleaner plot
    x = np.clip(x, -25, 25)
    y = np.clip(y, 0, 47)
    h = ax.hist2d(x, y, bins=25, cmap="YlOrRd", cmin=1, alpha=0.85)
    plt.colorbar(h[3], ax=ax, label="Shot attempts")
    ax.set_title(f"{player_name} — Shot chart {season}\n(heatmap by attempt density)", fontsize=12)
    return fig


def run(player_name: str = "Stephen Curry", season: str = "2025-26") -> Path:
    """Fetch data, plot shot chart, save to outputs. Returns path to saved image."""
    player_id = get_player_id(player_name)
    if player_id is None:
        raise ValueError(f"Player not found: {player_name}")

    df = fetch_shot_chart(player_id, season=season)
    if df.empty:
        raise ValueError(f"No shot chart data for {player_name} ({season})")

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_shot_chart(df, player_name, season, ax=ax)
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in player_name).strip().replace(" ", "_")
    out_path = OUTPUTS_DIR / f"nba_shot_chart_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run()
