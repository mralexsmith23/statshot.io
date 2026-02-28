"""
Geographic bubble map: Birthplaces of active NBA players (by country).
Style inspired by NHL birthplaces map. Uses nba_api; caches results in data/.
Saves to outputs/geo_bubble_map_nba_birthplaces.png
"""
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from nba_api.stats.endpoints import commonallplayers, commonplayerinfo

from src.config import DATA_DIR, OUTPUTS_DIR

# Country name (from API) -> (lat, lon) for bubble placement. Centroids for visual clarity.
COUNTRY_CENTROIDS = {
    "USA": (39.8283, -98.5795),
    "United States": (39.8283, -98.5795),
    "Canada": (56.1304, -106.3468),
    "France": (46.2276, 2.2137),
    "Serbia": (44.0165, 21.0059),
    "Australia": (-25.2744, 133.7751),
    "Slovenia": (46.1512, 14.9955),
    "Germany": (51.1657, 10.4515),
    "Nigeria": (9.0820, 8.6753),
    "Cameroon": (6.6111, 20.9394),
    "Democratic Republic of the Congo": (-4.0383, 21.7587),
    "Congo": (-0.2280, 15.8277),
    "Senegal": (14.4974, -14.4524),
    "Bahamas": (25.0343, -77.3963),
    "Jamaica": (18.1096, -77.2975),
    "Dominican Republic": (18.7357, -70.1627),
    "Puerto Rico": (18.2208, -66.5901),
    "Virgin Islands": (18.3358, -64.8963),
    "U.S. Virgin Islands": (18.3358, -64.8963),
    "Argentina": (-38.4161, -63.6167),
    "Brazil": (-14.2350, -51.9253),
    "Venezuela": (6.4238, -66.5897),
    "Colombia": (4.5709, -74.2973),
    "Mexico": (23.6345, -102.5528),
    "Lithuania": (55.1694, 23.8813),
    "Latvia": (56.8796, 24.6032),
    "Estonia": (58.5953, 25.0136),
    "Greece": (39.0742, 21.8243),
    "Turkey": (38.9637, 35.2433),
    "Ukraine": (48.3794, 31.1656),
    "Russia": (61.5240, 105.3188),
    "Georgia": (42.3154, 43.3569),
    "Israel": (31.0461, 34.8516),
    "Croatia": (45.1000, 15.2000),
    "Bosnia & Herzegovina": (43.9159, 17.6791),
    "Bosnia and Herzegovina": (43.9159, 17.6791),
    "Slovakia": (48.6690, 19.6990),
    "Czech Republic": (49.8175, 15.4730),
    "Italy": (41.8719, 12.5674),
    "Spain": (40.4637, -3.7492),
    "Switzerland": (46.8182, 8.2275),
    "United Kingdom": (55.3781, -3.4360),
    "England": (52.3555, -1.1743),
    "Scotland": (56.4907, -4.2026),
    "New Zealand": (-40.9006, 174.8860),
    "China": (35.8617, 104.1954),
    "Japan": (36.2048, 138.2529),
    "South Sudan": (6.8770, 31.3070),
    "Sudan": (12.8628, 30.2176),
    "Egypt": (26.8206, 30.8025),
    "Tunisia": (33.8869, 9.5375),
    "Mali": (17.5707, -3.9962),
    "Ghana": (7.9465, -1.0232),
    "Belize": (17.1899, -88.4976),
    "Austria": (47.5162, 14.5501),
    "Netherlands": (52.1326, 5.2913),
    "Poland": (51.9194, 19.1451),
    "Montenegro": (42.7087, 19.3744),
    "North Macedonia": (41.5124, 21.7453),
    "Serbia and Montenegro": (43.7361, 20.4573),
    "Belgium": (50.5039, 4.4699),
    "Ireland": (53.1424, -7.6921),
    "Iceland": (64.9631, -19.0208),
    "Finland": (61.9241, 25.7482),
    "Sweden": (62.1982, 17.5514),
    "Denmark": (56.2639, 9.5018),
    "Haiti": (18.9712, -72.2852),
    "Trinidad and Tobago": (10.6918, -61.2225),
    "Antigua and Barbuda": (17.0608, -61.7964),
    "Saint Lucia": (13.9094, -60.9789),
    "Barbados": (13.1939, -59.5432),
    "Cuba": (21.5218, -77.7812),
    "Panama": (8.5380, -80.7821),
    "Uruguay": (-32.5228, -55.7658),
    "Chile": (-35.6751, -71.5430),
    "Peru": (-9.1900, -75.0152),
    "Ecuador": (-1.8312, -78.1834),
    "Spain (Catalonia)": (41.5912, 1.5209),
    "Lebanon": (33.8547, 35.8623),
    "Iran": (32.4279, 53.6880),
    "India": (20.5937, 78.9629),
    "Philippines": (12.8797, 121.7740),
    "Taiwan": (23.6978, 120.9605),
    "South Korea": (35.9078, 127.7669),
    "Indonesia": (-0.7893, 113.9213),
}

CACHE_RAW = DATA_DIR / "nba_birthplaces_raw.csv"
CACHE_AGG = DATA_DIR / "nba_birthplaces_by_country.csv"


def fetch_birthplaces_from_api(season: str = "2025-26") -> pd.DataFrame:
    """Fetch all current players and their country; save to cache. Returns aggregated by country."""
    players_df = commonallplayers.CommonAllPlayers(is_only_current_season=1, season=season).get_data_frames()[0]
    if players_df.empty:
        return pd.DataFrame()

    rows = []
    for i, row in players_df.iterrows():
        pid = row["PERSON_ID"]
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=pid)
            info_df = info.get_data_frames()[0]
            if not info_df.empty:
                country = info_df["COUNTRY"].iloc[0]
                if pd.notna(country) and str(country).strip():
                    rows.append({"player_id": pid, "country": str(country).strip()})
        except Exception:
            pass
        time.sleep(0.55)

    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    raw.to_csv(CACHE_RAW, index=False)
    agg = raw.groupby("country").size().reset_index(name="count")
    agg.to_csv(CACHE_AGG, index=False)
    return agg


def load_birthplaces_by_country(use_cache: bool = True, season: str = "2025-26") -> pd.DataFrame:
    """Load birthplaces by country: from cache if present and use_cache, else fetch from API."""
    if use_cache and CACHE_AGG.exists():
        return pd.read_csv(CACHE_AGG)
    return fetch_birthplaces_from_api(season=season)


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Add lat, lon from COUNTRY_CENTROIDS. Unknown countries get (0,0) and are filtered or kept."""
    def lat_lon(c):
        return COUNTRY_CENTROIDS.get(c, COUNTRY_CENTROIDS.get(c.strip(), (None, None)))
    df = df.copy()
    df["lat"] = df["country"].apply(lambda c: lat_lon(c)[0])
    df["lon"] = df["country"].apply(lambda c: lat_lon(c)[1])
    return df.dropna(subset=["lat", "lon"])


def draw_world_outline(ax: plt.Axes) -> None:
    """Simple world map: draw continents outline from Natural Earth (or skip if unavailable)."""
    try:
        import geopandas as gpd
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
        world.boundary.plot(ax=ax, linewidth=0.4, color="gray", zorder=0)
        return
    except Exception:
        pass
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)


def run(use_cache: bool = True, season: str = "2025-26") -> Path:
    """Build bubble map of NBA player birthplaces by country; save to outputs. Returns path to image."""
    df = load_birthplaces_by_country(use_cache=use_cache, season=season)
    if df.empty:
        raise ValueError("No birthplace data. Run with use_cache=False to fetch from API.")

    df = add_coordinates(df)
    if df.empty:
        raise ValueError("No coordinates for any country. Add more COUNTRY_CENTROIDS.")

    # Bubble size scale: area proportional to count; scale so max is readable
    min_s, max_s = 30, 1200
    c = df["count"]
    if c.max() == c.min():
        sizes = np.full(len(df), 200)
    else:
        sizes = min_s + (c - c.min()) / (c.max() - c.min()) * (max_s - min_s)

    fig, ax = plt.subplots(figsize=(14, 8))
    draw_world_outline(ax)
    ax.scatter(df["lon"], df["lat"], s=sizes, alpha=0.6, c="#4a90d9", edgecolors="white", linewidths=0.5, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend: bubble sizes for 1, 10, 20 (like NHL map)
    legend_counts = [1, 10, 20]
    scale = (max_s - min_s) / (c.max() - c.min()) if c.max() > c.min() else 50
    legend_sizes = [min_s + (n - c.min()) * scale for n in legend_counts]
    legend_sizes = [max(min_s, s) for s in legend_sizes]
    for val, sz in zip(legend_counts, legend_sizes):
        ax.scatter([], [], s=sz, c="#4a90d9", alpha=0.6, edgecolors="white", label=f"{val} player(s)")
    ax.legend(scatterpoints=1, frameon=True, loc="lower left", labelspacing=2)

    # Top 10 countries
    top = df.nlargest(10, "count")
    text = "Top 10 countries (active NBA players)\n" + "\n".join(
        f"{i+1}. {row['country']}: {int(row['count'])}" for i, row in top.iterrows()
    )
    ax.text(0.98, 0.98, text, transform=ax.transAxes, fontsize=8, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.set_title("Birthplaces of active NBA players (by country)\nSource: NBA API", fontsize=12)
    out_path = OUTPUTS_DIR / "geo_bubble_map_nba_birthplaces.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run(use_cache=True)
