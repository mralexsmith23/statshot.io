"""
Head-to-head NBA shot chart — editorial-quality smoothed FG% heatmap.

Volume-weighted opacity: vivid where both players shoot frequently,
faded where data is sparse. Heatmap clipped to court boundaries.

Usage:
    python -m src.shot_chart_comparison "Devin Booker" "Luka Doncic"
    python -m src.shot_chart_comparison "LeBron James" "Kevin Durant" --season 2015-16
"""
from __future__ import annotations

import datetime
import pathlib
import sys
from io import BytesIO

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc
from nba_api.stats.static import players
from scipy.ndimage import gaussian_filter

from src.cache import load_shots
from src.config import OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------
_FONT_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Inter" / "extras" / "ttf"
_FONTS_LOADED = False


def _ensure_fonts():
    global _FONTS_LOADED
    if _FONTS_LOADED:
        return
    if _FONT_DIR.exists():
        for ttf in _FONT_DIR.glob("Inter*.ttf"):
            fm.fontManager.addfont(str(ttf))
    _FONTS_LOADED = True


def _fp(size: int, weight: str = "regular") -> dict:
    _ensure_fonts()
    return dict(fontfamily="Inter", fontsize=size, fontweight=weight)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCALE = 0.1
COURT_W = 25               # physical court half-width (ft)
COURT_TOP = 26             # just past three-point arc (23.75 ft); heatmap and court match
BASELINE_Y = -4.75         # baseline y in basket-centered coordinates
FT_Y = 14.25               # free-throw line y (19 ft from baseline, 14.25 from basket)
GRID_W = 27                # histogram extends past sideline to capture corner 3 data
GRID_RES = 80
SIGMA = 3.5
MIN_WEIGHT = 0.03
CAP = 0.10
ALPHA_FLOOR = 0.12
TEAM_COLORS: dict[str, tuple[str, str]] = {
    "ATL": ("#E03A3E", "#FDB927"), "BOS": ("#007A33", "#BA9653"), "BKN": ("#000000", "#FFFFFF"),
    "CHA": ("#1D1160", "#00788C"), "CHI": ("#CE1141", "#000000"), "CLE": ("#860038", "#FDBB30"),
    "DAL": ("#00538C", "#B8C4CA"), "DEN": ("#0E2240", "#FEC524"), "DET": ("#C8102E", "#1D42BA"),
    "GSW": ("#1D428A", "#FFC72C"), "HOU": ("#CE1141", "#000000"), "IND": ("#002D62", "#FDBB30"),
    "LAC": ("#C8102E", "#1D428A"), "LAL": ("#552583", "#FDB927"), "MEM": ("#5D76A9", "#12173F"),
    "MIA": ("#98002E", "#F9A01B"), "MIL": ("#00471B", "#EEE1C6"), "MIN": ("#0C2340", "#236192"),
    "NOP": ("#0C2340", "#C8102E"), "NYK": ("#006BB6", "#F58426"), "OKC": ("#007AC1", "#EF6100"),
    "ORL": ("#0077C0", "#000000"), "PHI": ("#006BB6", "#ED174C"), "PHX": ("#1D1160", "#E56020"),
    "POR": ("#E03A3E", "#000000"), "SAC": ("#5A2D81", "#63727A"), "SAS": ("#C4CED4", "#000000"),
    "TOR": ("#CE1141", "#000000"), "UTA": ("#002B5C", "#F9A01B"), "WAS": ("#002B5C", "#E31837"),
    "NJN": ("#002A60", "#CD1041"), "SEA": ("#00653A", "#FFC200"), "VAN": ("#00B2A9", "#DA2E20"),
    "NOH": ("#0C2340", "#C8102E"), "NOK": ("#0C2340", "#C8102E"), "CHH": ("#00778B", "#280071"),
}
FALLBACK_A = ("#1a6faf", "#78b7e8")
FALLBACK_B = ("#c0392b", "#e8a09a")


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def _color_distance(c1: str, c2: str) -> float:
    r1 = np.array(mcolors.to_rgb(c1)) * 255
    r2 = np.array(mcolors.to_rgb(c2)) * 255
    return float(np.linalg.norm(r1 - r2))


def _hue_distance(c1: str, c2: str) -> float:
    """Angular hue distance in degrees (0-180). Uses colorsys HSV."""
    import colorsys
    h1 = colorsys.rgb_to_hsv(*mcolors.to_rgb(c1))[0] * 360
    h2 = colorsys.rgb_to_hsv(*mcolors.to_rgb(c2))[0] * 360
    d = abs(h1 - h2)
    return min(d, 360 - d)


def _luminance(c: str) -> float:
    r, g, b = mcolors.to_rgb(c)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _usable(c: str) -> bool:
    return _luminance(c) < 0.78


def _resolve_colors(abbr_a: str, abbr_b: str) -> tuple[str, str]:
    tc = TEAM_COLORS
    pri_a = tc.get(abbr_a, FALLBACK_A)[0]
    sec_a = tc.get(abbr_a, FALLBACK_A)[1]
    pri_b = tc.get(abbr_b, FALLBACK_B)[0]
    sec_b = tc.get(abbr_b, FALLBACK_B)[1]
    cands_a = [c for c in [pri_a, sec_a] if _usable(c)] or [FALLBACK_A[0]]
    cands_b = [c for c in [pri_b, sec_b] if _usable(c)] or [FALLBACK_B[0]]

    best_pair, best_score = None, -1
    for ca in cands_a:
        for cb in cands_b:
            rgb_d = _color_distance(ca, cb)
            hue_d = _hue_distance(ca, cb)
            if rgb_d < 80:
                continue
            score = rgb_d + hue_d * 2
            if score > best_score:
                best_score = score
                best_pair = (ca, cb)

    if best_pair and best_score > 200:
        return best_pair
    return "#1a6faf", "#e87d2f"


def _make_diverging_cmap(color_a: str, color_b: str):
    rgb_a = mcolors.to_rgb(color_a)
    rgb_b = mcolors.to_rgb(color_b)
    mid = (0.88, 0.88, 0.88)
    blend_a = tuple(0.25 * mid[i] + 0.75 * rgb_a[i] for i in range(3))
    blend_b = tuple(0.25 * mid[i] + 0.75 * rgb_b[i] for i in range(3))
    cdict = {
        "red":   [(0, rgb_a[0], rgb_a[0]), (0.30, blend_a[0], blend_a[0]),
                  (0.5, mid[0], mid[0]),
                  (0.70, blend_b[0], blend_b[0]), (1, rgb_b[0], rgb_b[0])],
        "green": [(0, rgb_a[1], rgb_a[1]), (0.30, blend_a[1], blend_a[1]),
                  (0.5, mid[1], mid[1]),
                  (0.70, blend_b[1], blend_b[1]), (1, rgb_b[1], rgb_b[1])],
        "blue":  [(0, rgb_a[2], rgb_a[2]), (0.30, blend_a[2], blend_a[2]),
                  (0.5, mid[2], mid[2]),
                  (0.70, blend_b[2], blend_b[2]), (1, rgb_b[2], rgb_b[2])],
    }
    return mcolors.LinearSegmentedColormap("diverging", cdict, N=256)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def _resolve(name: str) -> int:
    matches = players.find_players_by_full_name(name)
    if not matches:
        raise ValueError(f"Player not found: {name}")
    return matches[0]["id"]


def _fetch_shots(player_id: int, season: str) -> pd.DataFrame:
    return load_shots(player_id, season, allow_api=True)


def _get_team_abbr(df: pd.DataFrame) -> str:
    if "TEAM_ID" in df.columns and not df.empty:
        from nba_api.stats.static import teams as nba_teams
        tid = int(df["TEAM_ID"].iloc[0])
        for t in nba_teams.get_teams():
            if t["id"] == tid:
                return t["abbreviation"]
    return ""



_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _fetch_image(url: str) -> np.ndarray | None:
    try:
        import requests
        from PIL import Image
        r = requests.get(url, headers=_HEADERS, timeout=5)
        if r.status_code == 200:
            return np.array(Image.open(BytesIO(r.content)).convert("RGBA"))
    except Exception:
        pass
    return None


def _player_headshot(pid: int) -> np.ndarray | None:
    return _fetch_image(f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png")


def _team_logo(abbr: str) -> np.ndarray | None:
    return _fetch_image(
        f"https://a.espncdn.com/i/teamlogos/nba/500/{abbr.lower()}.png"
    )


# ---------------------------------------------------------------------------
# Smoothed FG% surface — bins at GRID_W (wide), display at COURT_W
# ---------------------------------------------------------------------------
def _smooth_fg_pct(df: pd.DataFrame, res: int = GRID_RES, sigma: float = SIGMA):
    x = np.clip(df["LOC_X"].values * SCALE, -GRID_W, GRID_W)
    y = np.clip(df["LOC_Y"].values * SCALE, 0, COURT_TOP)
    made = df["SHOT_MADE_FLAG"].values.astype(float)

    x_edges = np.linspace(-GRID_W, GRID_W, res + 1)
    y_edges = np.linspace(0, COURT_TOP, res + 1)

    fga, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    fgm, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=made)

    has_shots = fga > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_raw = np.where(has_shots, fgm / fga, 0.0)
    weight = has_shots.astype(float)

    smoothed_val = gaussian_filter(pct_raw * weight, sigma=sigma, mode="nearest")
    smoothed_wt = gaussian_filter(weight, sigma=sigma, mode="nearest")
    smoothed_fga = gaussian_filter(fga.astype(float), sigma=sigma, mode="nearest")

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_smooth = np.where(smoothed_wt > MIN_WEIGHT, smoothed_val / smoothed_wt, np.nan)
    return pct_smooth, smoothed_fga


def _crop_to_court(arr: np.ndarray) -> np.ndarray:
    """Crop array from GRID_W extent to COURT_W extent along the x axis (axis 0)."""
    res = arr.shape[0]
    x_centers = np.linspace(-GRID_W, GRID_W, res)
    mask = (x_centers >= -COURT_W) & (x_centers <= COURT_W)
    return arr[mask]


def _diff_surface(df_a: pd.DataFrame, df_b: pd.DataFrame):
    pct_a, fga_a = _smooth_fg_pct(df_a)
    pct_b, fga_b = _smooth_fg_pct(df_b)

    diff = np.full_like(pct_a, 0.0)
    both = (~np.isnan(pct_a)) & (~np.isnan(pct_b))
    diff[both] = pct_a[both] - pct_b[both]

    only_a = (~np.isnan(pct_a)) & (np.isnan(pct_b))
    diff[only_a] = np.clip(pct_a[only_a] - 0.45, -CAP, CAP)
    only_b = (np.isnan(pct_a)) & (~np.isnan(pct_b))
    diff[only_b] = np.clip(0.45 - pct_b[only_b], -CAP, CAP)

    volume = fga_a + fga_b
    return diff, volume


# ---------------------------------------------------------------------------
# Zone stats
# ---------------------------------------------------------------------------
def _zone_splits(df: pd.DataFrame) -> dict:
    total = len(df)
    made = int(df["SHOT_MADE_FLAG"].sum())
    fg_pct = made / total * 100 if total else 0

    is_3 = df["SHOT_TYPE"].str.contains("3PT", na=False)
    fga_3 = int(is_3.sum())
    fgm_3 = int(df.loc[is_3, "SHOT_MADE_FLAG"].sum())
    pct_3 = fgm_3 / fga_3 * 100 if fga_3 else 0

    paint_mask = df["SHOT_ZONE_BASIC"].isin(["Restricted Area", "In The Paint (Non-RA)"])
    fga_paint = int(paint_mask.sum())
    fgm_paint = int(df.loc[paint_mask, "SHOT_MADE_FLAG"].sum())
    pct_paint = fgm_paint / fga_paint * 100 if fga_paint else 0

    mid_mask = df["SHOT_ZONE_BASIC"] == "Mid-Range"
    fga_mid = int(mid_mask.sum())
    fgm_mid = int(df.loc[mid_mask, "SHOT_MADE_FLAG"].sum())
    pct_mid = fgm_mid / fga_mid * 100 if fga_mid else 0

    return {
        "fga": total, "fg_pct": fg_pct,
        "paint_pct": pct_paint, "mid_pct": pct_mid, "three_pct": pct_3,
    }


# ---------------------------------------------------------------------------
# Headline — driven by the diff surface
# ---------------------------------------------------------------------------
def _generate_headline(name_a: str, name_b: str,
                       diff: np.ndarray, volume: np.ndarray,
                       stats_a: dict, stats_b: dict) -> tuple[str, str]:
    last_a = name_a.split()[-1]
    last_b = name_b.split()[-1]
    res = diff.shape[0]

    xc = np.linspace(-GRID_W, GRID_W, res)
    yc = np.linspace(0, COURT_TOP, res)
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    dist = np.sqrt(xx**2 + yy**2)

    paint_mask = (np.abs(xx) <= 8) & (yy <= FT_Y)
    three_mask = dist > 23.75
    mid_mask = ~paint_mask & ~three_mask

    vol_threshold = np.percentile(volume[volume > 0], 20) if (volume > 0).any() else 0
    significant = volume > vol_threshold

    def _zone_adv(mask):
        cells = mask & significant
        if cells.sum() < 5:
            return 0.0
        return float(np.sum(diff[cells] * volume[cells]) / np.sum(volume[cells]))

    adv = {"the paint": _zone_adv(paint_mask),
           "mid-range": _zone_adv(mid_mask),
           "from deep": _zone_adv(three_mask)}

    a_zones = [z for z, v in adv.items() if v > 0.005]
    b_zones = [z for z, v in adv.items() if v < -0.005]

    if len(a_zones) == 3:
        headline = f"{last_a} has the shooting edge from everywhere"
    elif len(b_zones) == 3:
        headline = f"{last_b} has the shooting edge from everywhere"
    elif a_zones and b_zones:
        a_best = max(a_zones, key=lambda z: adv[z])
        b_best = min(b_zones, key=lambda z: adv[z])
        headline = f"{last_a} rules {a_best}, {last_b} strikes {b_best}"
    elif a_zones:
        a_best = max(a_zones, key=lambda z: adv[z])
        headline = f"{last_a} dominates {a_best} in a tight battle"
    elif b_zones:
        b_best = min(b_zones, key=lambda z: adv[z])
        headline = f"{last_b} dominates {b_best} in a tight battle"
    else:
        headline = f"{last_a} and {last_b} are dead even across the court"

    subtitle = (f"Where each player shoots better — and how much it matters  ·  "
                f"{stats_a['fga']:,} vs {stats_b['fga']:,} FGA")

    return headline, subtitle


# ---------------------------------------------------------------------------
# Court drawing
# ---------------------------------------------------------------------------
def _draw_court(ax, color="white", lw=1.6, alpha=0.7, zorder=10):
    CW = COURT_W
    BL = BASELINE_Y
    FT = FT_Y
    kw = dict(edgecolor=color, linewidth=lw, zorder=zorder, fill=False, alpha=alpha)
    lkw = dict(color=color, lw=lw, alpha=alpha, zorder=zorder)

    # Baseline & sidelines
    ax.plot([-CW, CW], [BL, BL], **lkw)
    ax.plot([-CW, -CW], [BL, COURT_TOP], **lkw)
    ax.plot([CW, CW], [BL, COURT_TOP], **lkw)

    # Hoop & backboard
    ax.add_patch(mpatches.Circle((0, 0), 0.75, **kw))
    ax.plot([-3, 3], [-0.75, -0.75], **lkw)

    # Paint: 16 ft wide, baseline to free-throw line
    ax.add_patch(mpatches.Rectangle((-8, BL), 16, FT - BL, **kw))

    # Free-throw circle — solid half toward halfcourt, dashed half toward basket
    ax.add_patch(Arc((0, FT), 12, 12, theta1=0, theta2=180, **kw))
    ax.add_patch(Arc((0, FT), 12, 12, theta1=180, theta2=360, linestyle="--",
                      edgecolor=color, linewidth=lw * 0.6, alpha=alpha * 0.6,
                      zorder=zorder, fill=False))

    # Restricted area (4 ft radius)
    ax.add_patch(Arc((0, 0), 8, 8, theta1=0, theta2=180, **kw))

    # Three-point line: arc + vertical corner lines
    corner_y = np.sqrt(23.75**2 - 22**2)
    theta_corner = np.degrees(np.arctan2(corner_y, 22))
    ax.add_patch(Arc((0, 0), 23.75 * 2, 23.75 * 2,
                      theta1=theta_corner, theta2=180 - theta_corner, **kw))
    ax.plot([-22, -22], [BL, corner_y], **lkw)
    ax.plot([22, 22], [BL, corner_y], **lkw)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def build_comparison(
    player_a: str,
    player_b: str,
    season: str | None = None,
    season_a: str | None = None,
    season_b: str | None = None,
    color_a: str | None = None,
    color_b: str | None = None,
    save: bool = True,
) -> plt.Figure:
    _ensure_fonts()

    szn_a = season_a or season or "2025-26"
    szn_b = season_b or season or "2025-26"

    pid_a = _resolve(player_a)
    pid_b = _resolve(player_b)
    df_a = _fetch_shots(pid_a, szn_a)
    df_b = _fetch_shots(pid_b, szn_b)
    if df_a.empty:
        raise ValueError(f"No shot data for {player_a} ({szn_a})")
    if df_b.empty:
        raise ValueError(f"No shot data for {player_b} ({szn_b})")

    abbr_a, abbr_b = _get_team_abbr(df_a), _get_team_abbr(df_b)
    if color_a is None or color_b is None:
        color_a, color_b = _resolve_colors(abbr_a, abbr_b)
    cmap = _make_diverging_cmap(color_a, color_b)

    stats_a, stats_b = _zone_splits(df_a), _zone_splits(df_b)
    diff, volume = _diff_surface(df_a, df_b)
    headline, subtitle = _generate_headline(player_a, player_b, diff, volume, stats_a, stats_b)

    diff_c = _crop_to_court(diff)
    vol_c = _crop_to_court(volume)

    display = np.where(np.isnan(diff_c), 0.0, -diff_c)
    norm = mcolors.TwoSlopeNorm(vmin=-CAP, vcenter=0, vmax=CAP)
    rgba = cmap(norm(display.T))

    vol_t = vol_c.T
    vol_max = np.percentile(vol_t[vol_t > 0], 95) if (vol_t > 0).any() else 1.0
    alpha_map = np.clip(vol_t / vol_max, 0.0, 1.0)
    alpha_map = ALPHA_FLOOR + alpha_map * (1.0 - ALPHA_FLOOR)
    alpha_map = gaussian_filter(alpha_map, sigma=2.0, mode="nearest")
    rgba[:, :, 3] = alpha_map

    img_head_a = _player_headshot(pid_a)
    img_head_b = _player_headshot(pid_b)
    img_logo_a = _team_logo(abbr_a)
    img_logo_b = _team_logo(abbr_b)

    # ===== FIGURE =====
    fig = plt.figure(figsize=(11, 10), facecolor="white")

    # --- Headline ---
    fig.text(0.50, 0.980, headline,
             ha="center", va="center", color="#1a1a1a", **_fp(24, "bold"))
    fig.text(0.50, 0.948, subtitle,
             ha="center", va="center", color="#888888", **_fp(11, "regular"))

    # --- Player A header ---
    if img_head_a is not None:
        ax_img_a = fig.add_axes([0.045, 0.843, 0.12, 0.090])
        ax_img_a.imshow(img_head_a)
        ax_img_a.axis("off")
    if img_logo_a is not None:
        ax_logo_a = fig.add_axes([0.165, 0.846, 0.055, 0.055])
        ax_logo_a.imshow(img_logo_a)
        ax_logo_a.axis("off")

    cross_era = szn_a != szn_b

    tx_a = 0.230
    name_label_a = f"{player_a}  ({szn_a})" if cross_era else player_a
    fig.text(tx_a, 0.912, name_label_a, ha="left", va="center", color=color_a, **_fp(20, "bold"))
    fig.text(tx_a, 0.890,
             f"{stats_a['fga']:,} FGA  ·  {stats_a['fg_pct']:.1f}% FG",
             ha="left", va="center", color="#555555", **_fp(12, "medium"))
    fig.text(tx_a, 0.870,
             f"Paint {stats_a['paint_pct']:.0f}%  ·  Mid {stats_a['mid_pct']:.0f}%  ·  3PT {stats_a['three_pct']:.0f}%",
             ha="left", va="center", color="#999999", **_fp(10, "regular"))

    fig.text(0.50, 0.890, "vs", ha="center", va="center", color="#cccccc", **_fp(18, "medium"))

    # --- Player B header ---
    if img_head_b is not None:
        ax_img_b = fig.add_axes([0.535, 0.843, 0.12, 0.090])
        ax_img_b.imshow(img_head_b)
        ax_img_b.axis("off")
    if img_logo_b is not None:
        ax_logo_b = fig.add_axes([0.655, 0.846, 0.055, 0.055])
        ax_logo_b.imshow(img_logo_b)
        ax_logo_b.axis("off")

    tx_b = 0.98
    name_label_b = f"{player_b}  ({szn_b})" if cross_era else player_b
    fig.text(tx_b, 0.912, name_label_b, ha="right", va="center", color=color_b, **_fp(20, "bold"))
    fig.text(tx_b, 0.890,
             f"{stats_b['fga']:,} FGA  ·  {stats_b['fg_pct']:.1f}% FG",
             ha="right", va="center", color="#555555", **_fp(12, "medium"))
    fig.text(tx_b, 0.870,
             f"Paint {stats_b['paint_pct']:.0f}%  ·  Mid {stats_b['mid_pct']:.0f}%  ·  3PT {stats_b['three_pct']:.0f}%",
             ha="right", va="center", color="#999999", **_fp(10, "regular"))

    # --- Court + heatmap ---
    court_rect = [0.04, 0.120, 0.92, 0.708]

    ax_bg = fig.add_axes(court_rect)
    ax_bg.set_facecolor("#e8e5e1")
    ax_bg.set_xlim(-COURT_W - 3, COURT_W + 3)
    ax_bg.set_ylim(-5.2, COURT_TOP + 1)
    ax_bg.set_aspect("equal")
    ax_bg.axis("off")

    ax = fig.add_axes(court_rect)
    ax.set_xlim(-COURT_W - 3, COURT_W + 3)
    ax.set_ylim(-5.2, COURT_TOP + 1)
    ax.set_facecolor("none")
    ax.axis("off")

    ax.imshow(
        rgba, origin="lower",
        extent=(-COURT_W, COURT_W, 0, COURT_TOP),
        aspect="equal", interpolation="gaussian",
        zorder=2,
    )

    _draw_court(ax, color="white", lw=1.8, alpha=0.65, zorder=10)

    # --- Legend bar ---
    ax_leg = fig.add_axes([0.18, 0.090, 0.64, 0.015])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_leg.imshow(gradient, aspect="auto", cmap=cmap, extent=(0, 1, 0, 1), origin="lower")
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    for spine in ax_leg.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.5)

    fig.text(0.17, 0.098, f"← {player_a.split()[-1]} better",
             ha="right", va="center", color=color_a, **_fp(12, "semibold"))
    fig.text(0.83, 0.098, f"{player_b.split()[-1]} better →",
             ha="left", va="center", color=color_b, **_fp(12, "semibold"))

    pct_cap = int(CAP * 100)
    ticks = [
        (0.18, f"+{pct_cap}%"),
        (0.34, f"+{pct_cap // 2}%"),
        (0.50, "even"),
        (0.66, f"+{pct_cap // 2}%"),
        (0.82, f"+{pct_cap}%"),
    ]
    for xp, label in ticks:
        fig.text(xp, 0.072, label, ha="center", va="center", color="#999999", **_fp(10, "regular"))

    season_label = f"{szn_a} vs {szn_b}" if cross_era else szn_a
    fig.text(0.50, 0.050, f"{season_label} Regular Season  ·  Source: NBA Stats API",
             ha="center", va="center", color="#bbbbbb", **_fp(9, "regular"))

    if save:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_a = player_a.replace(" ", "_")
        safe_b = player_b.replace(" ", "_")
        out_path = OUTPUTS_DIR / f"comparison_{safe_a}_vs_{safe_b}_{ts}.png"
        fig.savefig(out_path, dpi=180, facecolor="white")
        print(f"Saved: {out_path}")

    return fig


def run():
    args = sys.argv[1:]
    if len(args) < 2:
        print('Usage: python -m src.shot_chart_comparison "Player A" "Player B" '
              '[--season 2025-26] [--season-a 2007-08] [--season-b 2015-16]')
        sys.exit(1)
    player_a, player_b = args[0], args[1]

    def _flag(name: str) -> str | None:
        if name in args:
            idx = args.index(name)
            if idx + 1 < len(args):
                return args[idx + 1]
        return None

    season = _flag("--season")
    season_a = _flag("--season-a")
    season_b = _flag("--season-b")

    fig = build_comparison(player_a, player_b,
                           season=season, season_a=season_a, season_b=season_b)
    plt.close(fig)


if __name__ == "__main__":
    run()
