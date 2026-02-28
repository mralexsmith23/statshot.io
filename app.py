"""
NBA Analytics Explorer — Multi-view Streamlit app.
Run with: streamlit run app.py
"""
import re
import time
import unicodedata
from io import BytesIO
from urllib.parse import quote
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from streamlit_searchbox import st_searchbox
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playergamelogs,
    playercareerstats,
    leaguedashteamstats,
    leaguestandingsv3,
    leagueleaders,
)
from src.cache import load_shots
from src.shot_chart_comparison import TEAM_COLORS, FALLBACK_A, FALLBACK_B

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StatShot — NBA Head-to-Head Shot Charts",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Open Graph meta tags — social sharing previews
# ---------------------------------------------------------------------------
st.markdown(
    """<meta property="og:title" content="StatShot — NBA Head-to-Head Shot Charts" />
<meta property="og:description" content="Compare any two NBA players' shooting — any era, any season. Interactive FG% heatmaps with team colors." />
<meta property="og:type" content="website" />
<meta property="og:url" content="https://statshot.io" />
<meta property="og:image" content="https://statshot.io/og-preview.png" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="StatShot — NBA Head-to-Head Shot Charts" />
<meta name="twitter:description" content="Compare any two NBA players' shooting — any era, any season." />
<meta name="twitter:image" content="https://statshot.io/og-preview.png" />""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — About the Creator
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<h3 style="margin-bottom:0.2rem;">🎯 Stat<span style="color:#e56020;">Shot</span></h3>',
        unsafe_allow_html=True,
    )
    st.caption("Compare any two NBA players — any era, any season.")
    st.divider()
    st.markdown("### About the Creator")
    st.markdown(
        "Built by **Alex Smith** — finance leader, data builder, Suns fan.\n\n"
        "I build tools that turn messy data into clear decisions, "
        "whether it's a $200M budget model or an NBA shot chart."
    )
    st.markdown(
        "[alexsmith.finance](https://alexsmith.finance)  \n"
        "[LinkedIn](https://www.linkedin.com/in/alexwesleysmith/)  \n"
        "[Let's Work Together](mailto:mralexsmith@gmail.com)"
    )
    st.divider()
    st.caption("Data: NBA Stats API  ·  Built with Python & Streamlit")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HALF_COURT_LENGTH_FT = 47
COURT_WIDTH_FT = 50
SCALE = 0.1
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2025, 1995, -1)]

NBA_TEAMS = {t["id"]: t for t in teams.get_teams()}
ABBR_TO_NAME = {t["abbreviation"]: t["full_name"] for t in NBA_TEAMS.values()}
CONFERENCES = {"East": [], "West": []}
DIVISIONS = {}
for t in NBA_TEAMS.values():
    conf = t.get("conference", "")
    div = t.get("division", "")
    if conf:
        CONFERENCES.setdefault(conf, []).append(t["full_name"])
    if div:
        DIVISIONS.setdefault(div, []).append(t["full_name"])


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_all_player_names() -> list[str]:
    return sorted(p["full_name"] for p in players.get_players())


_ALL_NAMES: list[str] = []


def _strip_accents(s: str) -> str:
    """Remove diacritical marks so 'doncic' matches 'Dončić'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _search_players(query: str) -> list[str]:
    """Accent- and case-insensitive search for player names."""
    global _ALL_NAMES
    if not _ALL_NAMES:
        _ALL_NAMES = get_all_player_names()
    if not query:
        return _ALL_NAMES[:20]
    q = _strip_accents(query).lower()
    return [n for n in _ALL_NAMES if q in _strip_accents(n).lower()][:20]


@st.cache_data(show_spinner=False)
def resolve_player_id(name: str) -> int | None:
    matches = players.find_players_by_full_name(name)
    return matches[0]["id"] if matches else None


SHOT_DATA_FIRST_SEASON = 1996

@st.cache_data(show_spinner="Loading player seasons...", ttl=3600)
def fetch_player_seasons(player_id: int) -> list[str]:
    """Return list of seasons (e.g. ['2024-25', '2023-24', ...]) a player has data for,
    filtered to seasons where NBA shot-location data exists (1996-97+)."""
    for attempt in range(3):
        try:
            time.sleep(0.6 + attempt * 1.5)
            career = playercareerstats.PlayerCareerStats(
                player_id=player_id, timeout=60,
            )
            df = career.get_data_frames()[0]
            if df.empty:
                return []
            all_seasons = sorted(df["SEASON_ID"].unique(), reverse=True)
            return [s for s in all_seasons if int(s[:4]) >= SHOT_DATA_FIRST_SEASON]
        except Exception:
            if attempt == 2:
                raise
    return []


@st.cache_data(show_spinner="Fetching shot data...", ttl=600)
def fetch_shot_chart(player_id: int, season: str) -> pd.DataFrame:
    return load_shots(player_id, season, allow_api=True)


@st.cache_data(show_spinner="Fetching game logs...", ttl=600)
def fetch_player_game_logs(player_id: int, season: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            time.sleep(0.6 + attempt * 1.5)
            logs = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                season_nullable=season,
                timeout=60,
            )
            return logs.get_data_frames()[0]
        except Exception:
            if attempt == 2:
                raise
    return pd.DataFrame()


@st.cache_data(show_spinner="Fetching career stats...", ttl=600)
def fetch_career_stats(player_id: int) -> pd.DataFrame:
    for attempt in range(3):
        try:
            time.sleep(0.6 + attempt * 1.5)
            career = playercareerstats.PlayerCareerStats(
                player_id=player_id, timeout=60,
            )
            return career.get_data_frames()[0]
        except Exception:
            if attempt == 2:
                raise
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_team_for_season(player_id: int, season: str) -> str | None:
    """Return team abbreviation for the given player in the given season, or None if not found."""
    career = fetch_career_stats(player_id)
    if career.empty or "SEASON_ID" not in career.columns or "TEAM_ABBREVIATION" not in career.columns:
        return None
    row = career[career["SEASON_ID"] == season]
    if row.empty:
        return None
    return str(row["TEAM_ABBREVIATION"].iloc[0])


@st.cache_data(show_spinner="Fetching team stats...", ttl=600)
def fetch_league_team_stats(season: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            time.sleep(0.6 + attempt * 1.5)
            ts = leaguedashteamstats.LeagueDashTeamStats(season=season, timeout=60)
            return ts.get_data_frames()[0]
        except Exception:
            if attempt == 2:
                raise
    return pd.DataFrame()


@st.cache_data(show_spinner="Fetching standings...", ttl=600)
def fetch_standings(season: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            time.sleep(0.6 + attempt * 1.5)
            s = leaguestandingsv3.LeagueStandingsV3(season=season, timeout=60)
            return s.get_data_frames()[0]
        except Exception:
            if attempt == 2:
                raise
    return pd.DataFrame()


@st.cache_data(show_spinner="Fetching league leaders...", ttl=600)
def fetch_league_leaders(season: str, stat_category: str = "PTS") -> pd.DataFrame:
    for attempt in range(3):
        try:
            time.sleep(0.6 + attempt * 1.5)
            ll = leagueleaders.LeagueLeaders(
                season=season, stat_category_abbreviation=stat_category, timeout=60,
            )
            return ll.get_data_frames()[0]
        except Exception:
            if attempt == 2:
                raise
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_half_court(ax, color="black", lw=1.5):
    ax.set_xlim(-30, 30)
    ax.set_ylim(-5, 52)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(mpatches.Rectangle((-COURT_WIDTH_FT / 2, 0), COURT_WIDTH_FT, HALF_COURT_LENGTH_FT, fill=False, edgecolor=color, linewidth=lw))
    ax.axhline(y=HALF_COURT_LENGTH_FT, xmin=0, xmax=1, color=color, lw=lw)
    ax.add_patch(mpatches.Circle((0, HALF_COURT_LENGTH_FT), 6, fill=False, edgecolor=color, linewidth=lw))
    ax.add_patch(mpatches.Circle((0, 0), 1.5, fill=False, edgecolor=color, linewidth=lw))
    ax.plot([-2, 2], [4, 4], color=color, lw=lw)
    ax.add_patch(mpatches.Rectangle((-8, 0), 16, 19, fill=False, edgecolor=color, linewidth=lw))
    ax.add_patch(mpatches.Arc((0, 0), 23.75 * 2, 23.75 * 2, theta1=0, theta2=180, edgecolor=color, linewidth=lw))
    ax.plot([-25, -22], [0, 0], color=color, lw=lw)
    ax.plot([22, 25], [0, 0], color=color, lw=lw)


def build_shot_chart(df, player_name, season, show_makes, show_misses):
    df = df.copy()
    df["x_ft"] = df["LOC_X"] * SCALE
    df["y_ft"] = df["LOC_Y"] * SCALE
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_half_court(ax)
    if show_makes and show_misses:
        subset = df
    elif show_makes:
        subset = df[df["SHOT_MADE_FLAG"] == 1]
    elif show_misses:
        subset = df[df["SHOT_MADE_FLAG"] == 0]
    else:
        subset = df.iloc[0:0]
    if subset.empty:
        ax.text(0, 25, "No shots to display", ha="center", va="center", fontsize=16, color="gray")
        return fig
    x = np.clip(subset["x_ft"], -25, 25)
    y = np.clip(subset["y_ft"], 0, 47)
    h = ax.hist2d(x, y, bins=30, cmap="YlOrRd", cmin=1, alpha=0.85)
    plt.colorbar(h[3], ax=ax, label="Shot attempts", shrink=0.7)
    ax.set_title(f"{player_name} — {season}", fontsize=14, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------
SHOW_ALL_TABS = False

st.markdown(
    '<h1 style="margin-bottom:0;">🎯 Stat<span style="color:#e56020;">Shot</span></h1>'
    '<p style="margin-top:0; font-size:1.15rem; color:#666;">'
    'NBA head-to-head shot charts — compare any two players, any era.'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:0.85rem; color:#999; margin-top:-0.5rem;">'
    'Powered by NBA Stats API  ·  <a href="https://statshot.io" style="color:#e56020; text-decoration:none;">statshot.io</a>'
    '</p>',
    unsafe_allow_html=True,
)

if SHOW_ALL_TABS:
    tab_compare, tab_shot, tab_trends, tab_teams, tab_leaders = st.tabs([
        "Head-to-Head",
        "Shot Chart",
        "Player Trends",
        "Teams & Standings",
        "League Leaders",
    ])
else:
    tab_compare, = st.tabs(["Head-to-Head"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 0 — Head-to-Head Comparison
# ═══════════════════════════════════════════════════════════════════════════

# --- Check for shareable URL params ---
qp = st.query_params
url_player_a = qp.get("a", "")
url_season_a = qp.get("sa", "")
url_player_b = qp.get("b", "")
url_season_b = qp.get("sb", "")
url_color_a = qp.get("ca", "")   # "alt" → Alternate, anything else → Primary
url_color_b = qp.get("cb", "")
auto_generate = all([url_player_a, url_season_a, url_player_b, url_season_b])

with tab_compare:
    st.subheader("Head-to-Head FG% Comparison")
    st.caption("Smoothed heatmap colored by the player who shoots better in each zone, using their team colors")

    col_a, col_szn_a = st.columns([3, 2])
    with col_a:
        default_a = url_player_a or "Stephen Curry"
        cmp_player_a = st_searchbox(_search_players, label="Player A (type to search)", default=default_a, key="cmp_a", clear_on_submit=False)
    with col_szn_a:
        default_sa = url_season_a or SEASONS[0]
        sa_index = SEASONS.index(default_sa) if default_sa in SEASONS else 0
        cmp_season_a = st.selectbox("Season A", SEASONS, index=sa_index, key="cmp_season_a")

    col_b, col_szn_b = st.columns([3, 2])
    with col_b:
        default_b = url_player_b or "Luka Doncic"
        cmp_player_b = st_searchbox(_search_players, label="Player B (type to search)", default=default_b, key="cmp_b", clear_on_submit=False)
    with col_szn_b:
        default_sb = url_season_b or SEASONS[0]
        sb_index = SEASONS.index(default_sb) if default_sb in SEASONS else 0
        cmp_season_b = st.selectbox("Season B", SEASONS, index=sb_index, key="cmp_season_b")

    col_color_a, col_color_b = st.columns(2)
    with col_color_a:
        ca_idx = 1 if url_color_a == "alt" else 0
        color_pref_a = st.radio("Player A color", ["Primary", "Alternate"], index=ca_idx, key="color_pref_a", horizontal=True)
    with col_color_b:
        cb_idx = 1 if url_color_b == "alt" else 0
        color_pref_b = st.radio("Player B color", ["Primary", "Alternate"], index=cb_idx, key="color_pref_b", horizontal=True)

    run_now = st.button("Generate Comparison", type="primary", key="btn_cmp") or auto_generate

    if auto_generate:
        cmp_player_a = url_player_a or cmp_player_a
        cmp_player_b = url_player_b or cmp_player_b
        cmp_season_a = url_season_a or cmp_season_a
        cmp_season_b = url_season_b or cmp_season_b

    if run_now:
        if cmp_player_a == cmp_player_b and cmp_season_a == cmp_season_b:
            st.error("Pick two different players or different seasons.")
        else:
            try:
                with st.spinner("Crunching shot data — this may take a few seconds..."):
                    pid_a = resolve_player_id(cmp_player_a) if cmp_player_a else None
                    pid_b = resolve_player_id(cmp_player_b) if cmp_player_b else None
                    abbr_a = fetch_team_for_season(pid_a, cmp_season_a) if pid_a else None
                    abbr_b = fetch_team_for_season(pid_b, cmp_season_b) if pid_b else None
                    colors_a = TEAM_COLORS.get(abbr_a, FALLBACK_A) if abbr_a else FALLBACK_A
                    colors_b = TEAM_COLORS.get(abbr_b, FALLBACK_B) if abbr_b else FALLBACK_B
                    color_idx_a = 0 if color_pref_a == "Primary" else 1
                    color_idx_b = 0 if color_pref_b == "Primary" else 1
                    color_a_hex = colors_a[color_idx_a]
                    color_b_hex = colors_b[color_idx_b]

                    from src.shot_chart_comparison import build_comparison
                    fig = build_comparison(
                        cmp_player_a, cmp_player_b,
                        season_a=cmp_season_a, season_b=cmp_season_b,
                        color_a=color_a_hex, color_b=color_b_hex,
                        save=False,
                    )
                safe = lambda s: re.sub(r"[^\w\-]", "_", (s or "").strip()).strip("_") or "Player"
                name_a = safe(cmp_player_a)
                name_b = safe(cmp_player_b)
                season_a_safe = safe(cmp_season_a)
                season_b_safe = safe(cmp_season_b)
                file_name = f"{name_a}_{season_a_safe}_vs_{name_b}_{season_b_safe}.png"
                fig.text(
                    0.98, 0.02, "statshot.io",
                    ha="right", va="bottom",
                    fontsize=11, color="#e56020", alpha=0.6,
                    fontweight="bold", fontstyle="italic",
                    transform=fig.transFigure,
                )
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=180, facecolor="white")
                buf.seek(0)

                ca_param = "alt" if color_pref_a == "Alternate" else "pri"
                cb_param = "alt" if color_pref_b == "Alternate" else "pri"
                share_url = (
                    f"https://statshot.io/?a={quote(cmp_player_a)}"
                    f"&sa={quote(cmp_season_a)}"
                    f"&b={quote(cmp_player_b)}"
                    f"&sb={quote(cmp_season_b)}"
                    f"&ca={ca_param}&cb={cb_param}"
                )
                share_text = f"{cmp_player_a} vs {cmp_player_b} — who shoots better?"

                share_cols = st.columns([1, 1, 1, 1, 2])
                with share_cols[0]:
                    st.download_button(
                        "📥 Download",
                        data=buf.getvalue(),
                        file_name=file_name,
                        mime="image/png",
                        key="cmp_download",
                    )
                with share_cols[1]:
                    x_url = (
                        f"https://twitter.com/intent/tweet"
                        f"?text={quote(share_text + ' 🎯')}"
                        f"&url={quote(share_url)}"
                    )
                    st.link_button("𝕏 Post to X", x_url)
                with share_cols[2]:
                    fb_url = f"https://www.facebook.com/sharer/sharer.php?u={quote(share_url)}"
                    st.link_button("📘 Facebook", fb_url)
                with share_cols[3]:
                    reddit_url = (
                        f"https://www.reddit.com/submit"
                        f"?url={quote(share_url)}"
                        f"&title={quote(share_text + ' | StatShot')}"
                    )
                    st.link_button("🏀 Reddit", reddit_url)

                team_name_a = ABBR_TO_NAME.get(abbr_a, abbr_a or "Unknown")
                team_name_b = ABBR_TO_NAME.get(abbr_b, abbr_b or "Unknown")
                st.markdown(
                    f'<p style="font-size:0.85rem; color:#888; margin:0.5rem 0 0.25rem;">'
                    f'<span style="display:inline-block;width:14px;height:14px;background:{color_a_hex};'
                    f'border:1px solid #ccc;border-radius:3px;vertical-align:middle;margin-right:4px;"></span>'
                    f'{team_name_a} ({abbr_a or "?"}) — {color_pref_a}'
                    f'&nbsp;&nbsp;vs&nbsp;&nbsp;'
                    f'<span style="display:inline-block;width:14px;height:14px;background:{color_b_hex};'
                    f'border:1px solid #ccc;border-radius:3px;vertical-align:middle;margin-right:4px;"></span>'
                    f'{team_name_b} ({abbr_b or "?"}) — {color_pref_b}'
                    f'</p>',
                    unsafe_allow_html=True,
                )

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                    st.error(
                        "The NBA Stats API is slow right now — please try again in a moment. "
                        "(This usually resolves on retry.)"
                    )
                else:
                    st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# HIDDEN TABS — set SHOW_ALL_TABS = True when ready to launch
# ═══════════════════════════════════════════════════════════════════════════
if SHOW_ALL_TABS:

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — Shot Chart
    # ═══════════════════════════════════════════════════════════════════════
    with tab_shot:
        st.subheader("Player Shot Chart")

        col_p, col_s, col_f1, col_f2 = st.columns([3, 2, 1, 1])
        with col_p:
            shot_player = st_searchbox(_search_players, label="Player (type to search)", default="Stephen Curry", key="shot_player", clear_on_submit=False)
        with col_s:
            shot_season = st.selectbox("Season", SEASONS, index=0, key="shot_season")
        with col_f1:
            shot_makes = st.checkbox("Makes", value=True, key="shot_makes")
        with col_f2:
            shot_misses = st.checkbox("Misses", value=True, key="shot_misses")

        if st.button("Generate Shot Chart", type="primary", key="btn_shot"):
            pid = resolve_player_id(shot_player)
            if pid is None:
                st.error(f"Player not found: {shot_player}")
            else:
                df = fetch_shot_chart(pid, shot_season)
                if df.empty:
                    st.warning(f"No shot data for {shot_player} in {shot_season}. Try another season.")
                else:
                    headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
                    col_info, col_chart = st.columns([1, 2.5])
                    with col_info:
                        st.image(headshot_url, width=260)
                        total = len(df)
                        made = int(df["SHOT_MADE_FLAG"].sum())
                        pct = made / total * 100 if total else 0
                        m1, m2, m3 = st.columns(3)
                        m1.metric("FGA", f"{total:,}")
                        m2.metric("FGM", f"{made:,}")
                        m3.metric("FG%", f"{pct:.1f}%")

                        zones = df.groupby("SHOT_ZONE_BASIC").agg(
                            FGA=("SHOT_ATTEMPTED_FLAG", "sum"),
                            FGM=("SHOT_MADE_FLAG", "sum"),
                        ).reset_index()
                        zones["FG%"] = (zones["FGM"] / zones["FGA"] * 100).round(1)
                        zones = zones.sort_values("FGA", ascending=False)
                        st.dataframe(zones, hide_index=True, use_container_width=True)

                    with col_chart:
                        fig = build_shot_chart(df, shot_player, shot_season, shot_makes, shot_misses)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — Player Trends (game log)
    # ═══════════════════════════════════════════════════════════════════════
    with tab_trends:
        st.subheader("Player Season Trends")

        col_pt, col_st, col_stat = st.columns([3, 2, 2])
        with col_pt:
            trend_player = st_searchbox(_search_players, label="Player (type to search)", default="LeBron James", key="trend_player", clear_on_submit=False)
        with col_st:
            trend_season = st.selectbox("Season", SEASONS, index=0, key="trend_season")
        with col_stat:
            trend_stat = st.selectbox("Stat to plot", ["PTS", "AST", "REB", "FG3M", "STL", "BLK", "PLUS_MINUS", "MIN"], key="trend_stat")

        show_career = st.checkbox("Also show career averages by season", key="show_career")

        if st.button("Load Player Trends", type="primary", key="btn_trends"):
            pid = resolve_player_id(trend_player)
            if pid is None:
                st.error(f"Player not found: {trend_player}")
            else:
                logs = fetch_player_game_logs(pid, trend_season)
                if logs.empty:
                    st.warning(f"No game logs for {trend_player} in {trend_season}.")
                else:
                    logs = logs.sort_values("GAME_DATE")
                    headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"

                    col_photo, col_summary = st.columns([1, 3])
                    with col_photo:
                        st.image(headshot_url, width=200)
                        st.metric("Games played", len(logs))
                        if trend_stat in logs.columns:
                            st.metric(f"Avg {trend_stat}", f"{logs[trend_stat].mean():.1f}")
                            st.metric(f"Max {trend_stat}", f"{logs[trend_stat].max():.0f}")

                    with col_summary:
                        if trend_stat in logs.columns:
                            fig, ax = plt.subplots(figsize=(12, 4))
                            vals = logs[trend_stat].values.astype(float)
                            games = range(1, len(vals) + 1)
                            ax.bar(games, vals, color="#4a90d9", alpha=0.7, width=0.8)
                            rolling = pd.Series(vals).rolling(window=5, min_periods=1).mean()
                            ax.plot(games, rolling, color="#e74c3c", linewidth=2, label="5-game rolling avg")
                            ax.set_xlabel("Game #")
                            ax.set_ylabel(trend_stat)
                            ax.set_title(f"{trend_player} — {trend_stat} per game ({trend_season})", fontweight="bold")
                            ax.legend()
                            ax.grid(axis="y", alpha=0.3)
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        else:
                            st.warning(f"Column {trend_stat} not found in game logs.")

                    if show_career:
                        career = fetch_career_stats(pid)
                        if not career.empty and trend_stat in career.columns:
                            career = career.sort_values("SEASON_ID")
                            fig2, ax2 = plt.subplots(figsize=(12, 4))
                            seasons_list = career["SEASON_ID"].astype(str).values
                            gp = career["GP"].values.astype(float)
                            totals = career[trend_stat].values.astype(float)
                            per_game = np.where(gp > 0, totals / gp, 0)
                            ax2.plot(seasons_list, per_game, marker="o", color="#27ae60", linewidth=2)
                            ax2.fill_between(seasons_list, per_game, alpha=0.15, color="#27ae60")
                            ax2.set_xlabel("Season")
                            ax2.set_ylabel(f"{trend_stat} per game")
                            ax2.set_title(f"{trend_player} — Career {trend_stat}/game", fontweight="bold")
                            ax2.grid(axis="y", alpha=0.3)
                            plt.xticks(rotation=45, ha="right")
                            st.pyplot(fig2, use_container_width=True)
                            plt.close(fig2)

                    st.subheader("Game log")
                    display_cols = ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "AST", "REB", "FG_PCT", "FG3M", "FG3A", "STL", "BLK", "TOV", "PLUS_MINUS"]
                    available = [c for c in display_cols if c in logs.columns]
                    st.dataframe(logs[available].reset_index(drop=True), hide_index=True, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — Teams & Standings
    # ═══════════════════════════════════════════════════════════════════════
    with tab_teams:
        st.subheader("Team Stats & Standings")
        team_season = st.selectbox("Season", SEASONS, index=0, key="team_season")

        view_mode = st.radio("View by", ["All teams", "Conference", "Division"], horizontal=True, key="team_view")

        if st.button("Load Team Data", type="primary", key="btn_teams"):
            team_df = fetch_league_team_stats(team_season)
            standings_df = fetch_standings(team_season)

            if team_df.empty:
                st.warning("No team stats returned.")
            else:
                if not standings_df.empty:
                    conf_map = {}
                    div_map = {}
                    for _, row in standings_df.iterrows():
                        tid = row.get("TeamID")
                        conf_map[tid] = row.get("Conference", "")
                        div_map[tid] = row.get("Division", "")
                    team_df["Conference"] = team_df["TEAM_ID"].map(conf_map)
                    team_df["Division"] = team_df["TEAM_ID"].map(div_map)

                if view_mode == "Conference" and "Conference" in team_df.columns:
                    conf_pick = st.selectbox("Conference", sorted(team_df["Conference"].dropna().unique()), key="conf_pick")
                    team_df = team_df[team_df["Conference"] == conf_pick]
                elif view_mode == "Division" and "Division" in team_df.columns:
                    div_pick = st.selectbox("Division", sorted(team_df["Division"].dropna().unique()), key="div_pick")
                    team_df = team_df[team_df["Division"] == div_pick]

                team_df = team_df.sort_values("W_PCT", ascending=False)
                stat_col = st.selectbox("Stat to compare", ["W_PCT", "PTS", "REB", "AST", "FG_PCT", "FG3_PCT", "STL", "BLK", "TOV", "PLUS_MINUS"], key="team_stat")

                fig, ax = plt.subplots(figsize=(14, 6))
                abbrevs = team_df["TEAM_ABBREVIATION"].values
                vals = team_df[stat_col].values.astype(float)
                sort_idx = np.argsort(vals)[::-1]
                abbrevs, vals = abbrevs[sort_idx], vals[sort_idx]

                colors = ["#2ecc71" if v >= np.median(vals) else "#e74c3c" for v in vals]
                ax.barh(range(len(abbrevs)), vals, color=colors, alpha=0.85)
                ax.set_yticks(range(len(abbrevs)))
                ax.set_yticklabels(abbrevs)
                ax.invert_yaxis()
                ax.set_xlabel(stat_col)
                title_suffix = ""
                if view_mode == "Conference":
                    title_suffix = f" — {conf_pick}"
                elif view_mode == "Division":
                    title_suffix = f" — {div_pick}"
                ax.set_title(f"NBA {stat_col} by Team ({team_season}){title_suffix}", fontweight="bold")
                ax.grid(axis="x", alpha=0.3)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.subheader("Full team stats")
                show_cols = ["TEAM_NAME", "W", "L", "W_PCT", "PTS", "REB", "AST", "FG_PCT", "FG3_PCT", "STL", "BLK", "TOV", "PLUS_MINUS"]
                if "Conference" in team_df.columns:
                    show_cols = ["TEAM_NAME", "Conference", "Division"] + show_cols[1:]
                available = [c for c in show_cols if c in team_df.columns]
                st.dataframe(team_df[available].reset_index(drop=True), hide_index=True, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 — League Leaders
    # ═══════════════════════════════════════════════════════════════════════
    with tab_leaders:
        st.subheader("League Leaders")
        col_ls, col_lc, col_ln = st.columns([2, 2, 1])
        with col_ls:
            leader_season = st.selectbox("Season", SEASONS, index=0, key="leader_season")
        with col_lc:
            leader_cat = st.selectbox("Stat category", ["PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT", "EFF"], key="leader_cat")
        with col_ln:
            top_n = st.slider("Top N", 5, 30, 15, key="leader_n")

        if st.button("Load Leaders", type="primary", key="btn_leaders"):
            ldf = fetch_league_leaders(leader_season, leader_cat)
            if ldf.empty:
                st.warning("No data returned.")
            else:
                ldf = ldf.head(top_n)
                fig, ax = plt.subplots(figsize=(12, max(4, top_n * 0.35)))
                names = ldf["PLAYER"].values
                vals = ldf[leader_cat].values.astype(float)
                colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(vals)))[::-1]
                ax.barh(range(len(names)), vals, color=colors, alpha=0.9)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels([f"{i+1}. {n}" for i, n in enumerate(names)])
                ax.invert_yaxis()
                ax.set_xlabel(leader_cat)
                ax.set_title(f"NBA {leader_cat} Leaders ({leader_season})", fontweight="bold")
                ax.grid(axis="x", alpha=0.3)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.subheader("Leaderboard")
                show_cols = ["RANK", "PLAYER", "TEAM", "GP", "MIN", "PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT", "EFF"]
                available = [c for c in show_cols if c in ldf.columns]
                st.dataframe(ldf[available].reset_index(drop=True), hide_index=True, use_container_width=True)
