"""
Shot chart data cache — read from local Parquet files first, fall back to live API.

Cache layout:  data/shot_cache/{player_id}_{season}.parquet
"""
from __future__ import annotations

import time
import logging

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail

from src.config import SHOT_CACHE_DIR

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _cache_path(player_id: int, season: str):
    return SHOT_CACHE_DIR / f"{player_id}_{season}.parquet"


def load_shots(player_id: int, season: str, *, allow_api: bool = True) -> pd.DataFrame:
    """Load shot chart data, preferring local cache.

    If *allow_api* is False and the cache misses, returns an empty DataFrame
    instead of hitting the network (useful in deployed contexts where you want
    to guarantee zero API calls).
    """
    path = _cache_path(player_id, season)
    if path.exists():
        return pd.read_parquet(path)

    if not allow_api:
        return pd.DataFrame()

    df = fetch_shots_live(player_id, season)
    if not df.empty:
        df.to_parquet(path, index=False)
        logger.info("Cached %d shots for player %s / %s", len(df), player_id, season)
    return df


def fetch_shots_live(player_id: int, season: str) -> pd.DataFrame:
    """Hit the NBA Stats API with retries and rate-limit sleeps."""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(0.8 + attempt * 1.5)
            shot = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=int(player_id),
                season_nullable=season,
                context_measure_simple="FGA",
            )
            return shot.get_data_frames()[0]
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
    return pd.DataFrame()


def is_cached(player_id: int, season: str) -> bool:
    return _cache_path(player_id, season).exists()
