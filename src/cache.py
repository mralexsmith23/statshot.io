"""
Shot chart data cache — read from local Parquet files first, fall back to live API.

Cache layout:  data/shot_cache/{player_id}_{season}.parquet

Guarantees
----------
* Completed seasons are cached forever (immutable data).
* Current-season files auto-refresh after CACHE_MAX_AGE_HOURS.
* Corrupt files are deleted and transparently re-fetched.
* Only the columns the app actually uses are persisted (zstd-compressed).
* Every file carries a ``cached_at`` timestamp in parquet metadata.
"""
from __future__ import annotations

import datetime
import time
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from requests.exceptions import RequestException

from src.config import (
    SHOT_CACHE_DIR,
    CACHE_MAX_AGE_HOURS,
    SHOT_KEEP_COLUMNS,
    SHOT_REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cache_path(player_id: int, season: str) -> Path:
    return SHOT_CACHE_DIR / f"{player_id}_{season}.parquet"


def _season_is_complete(season: str) -> bool:
    """An NBA season like '2024-25' ends by late June of the second year."""
    try:
        end_year = int(season[:4]) + 1
    except (ValueError, IndexError):
        return False
    return datetime.date.today() > datetime.date(end_year, 7, 1)


def _is_stale(path: Path, season: str) -> bool:
    if _season_is_complete(season):
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours > CACHE_MAX_AGE_HOURS


def _trim(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns the app needs."""
    available = [c for c in SHOT_KEEP_COLUMNS if c in df.columns]
    return df[available]


def _validate(df: pd.DataFrame) -> bool:
    """Return True when the DataFrame has the columns the app requires."""
    if df.empty:
        return True
    return SHOT_REQUIRED_COLUMNS.issubset(df.columns)


def _safe_read(path: Path) -> pd.DataFrame | None:
    """Read a parquet file, returning None (and deleting the file) if corrupt."""
    try:
        return pd.read_parquet(path)
    except Exception:
        logger.warning("Corrupt cache file %s — deleting", path)
        path.unlink(missing_ok=True)
        return None


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a trimmed DataFrame with zstd compression and a cached_at timestamp."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"cached_at"] = datetime.datetime.utcnow().isoformat().encode()
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, str(path), compression="zstd")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_shots(
    player_id: int,
    season: str,
    *,
    allow_api: bool = True,
) -> pd.DataFrame:
    """Load shot chart data, preferring local cache.

    * Completed-season files never expire.
    * Current-season files re-fetch after ``CACHE_MAX_AGE_HOURS``.
    * Corrupt files are transparently deleted and re-fetched.

    If *allow_api* is False and the cache misses, returns an empty DataFrame
    instead of hitting the network (useful in deployed contexts where you want
    to guarantee zero API calls).
    """
    path = _cache_path(player_id, season)

    if path.exists():
        if not _is_stale(path, season):
            df = _safe_read(path)
            if df is not None:
                return df
        else:
            logger.info("Stale cache for %s/%s — will re-fetch", player_id, season)

    if not allow_api:
        return pd.DataFrame()

    df = fetch_shots_live(player_id, season)
    if not df.empty and _validate(df):
        trimmed = _trim(df)
        _write_parquet(trimmed, path)
        logger.info("Cached %d shots for player %s / %s", len(trimmed), player_id, season)
        return trimmed

    if not _validate(df):
        logger.warning(
            "API response for %s/%s failed validation — not caching", player_id, season,
        )
    return df


def fetch_shots_live(player_id: int, season: str) -> pd.DataFrame:
    """Hit the NBA Stats API with retries on transient network errors."""
    from nba_api.stats.endpoints import shotchartdetail

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
        except (RequestException, ConnectionError, TimeoutError):
            logger.warning(
                "Transient error fetching %s/%s (attempt %d/%d)",
                player_id, season, attempt + 1, MAX_RETRIES,
            )
            if attempt == MAX_RETRIES - 1:
                raise
        except Exception:
            logger.exception(
                "Non-retryable error for %s/%s — failing immediately",
                player_id, season,
            )
            raise
    return pd.DataFrame()


def is_cached(player_id: int, season: str) -> bool:
    """Return True if a *fresh* cache file exists for this player-season."""
    path = _cache_path(player_id, season)
    return path.exists() and not _is_stale(path, season)


# ---------------------------------------------------------------------------
# Cache management utilities
# ---------------------------------------------------------------------------
def cache_stats() -> dict:
    """Return a summary of the shot cache directory."""
    files = list(SHOT_CACHE_DIR.glob("*.parquet"))
    if not files:
        return {"files": 0, "total_bytes": 0, "total_mb": 0.0}

    sizes = [f.stat().st_size for f in files]
    mtimes = [f.stat().st_mtime for f in files]
    return {
        "files": len(files),
        "total_bytes": sum(sizes),
        "total_mb": round(sum(sizes) / (1024 * 1024), 2),
        "oldest": datetime.datetime.fromtimestamp(min(mtimes)).isoformat(),
        "newest": datetime.datetime.fromtimestamp(max(mtimes)).isoformat(),
    }


def purge_stale(max_age_days: int = 90, *, dry_run: bool = True) -> list[str]:
    """Remove parquet files for *completed* seasons older than *max_age_days*.

    Current-season files are never purged here (they auto-refresh via
    ``_is_stale``).  Returns a list of deleted (or would-delete) file names.
    """
    cutoff = time.time() - (max_age_days * 86400)
    removed: list[str] = []
    for f in SHOT_CACHE_DIR.glob("*.parquet"):
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        season = parts[1]
        if not _season_is_complete(season):
            continue
        if f.stat().st_mtime < cutoff:
            removed.append(f.name)
            if not dry_run:
                f.unlink(missing_ok=True)
                logger.info("Purged old cache file: %s", f.name)
    return removed
