"""
Pre-fetch shot chart data for all active NBA players across multiple seasons.

Run from project root:
    python -m scripts.prefetch_shots
    python -m scripts.prefetch_shots --seasons 2025-26 2024-25 2023-24 2022-23

Takes ~8-12 minutes per season at ~1 request/sec for ~450+ players.
"""
from __future__ import annotations

import argparse
import sys
import time

from nba_api.stats.static import players

sys.path.insert(0, ".")
from src.cache import load_shots, is_cached  # noqa: E402

DEFAULT_SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23"]


def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), flush=True)


def prefetch_active(seasons: list[str]) -> None:
    active = [p for p in players.get_players() if p["is_active"]]
    total_players = len(active)
    total_combos = total_players * len(seasons)
    cached = 0
    fetched = 0
    errors = 0
    idx = 0

    _safe_print(f"Pre-fetching {total_players} active players x {len(seasons)} seasons = {total_combos} combos...")

    for season in seasons:
        _safe_print(f"\n=== Season: {season} ===")
        season_fetched = 0
        season_cached = 0

        for i, p in enumerate(active, 1):
            pid = p["id"]
            name = p["full_name"]
            idx += 1

            if is_cached(pid, season):
                cached += 1
                season_cached += 1
                continue

            try:
                df = load_shots(pid, season, allow_api=True)
                rows = len(df)
                fetched += 1
                season_fetched += 1
                if rows > 0:
                    _safe_print(f"  [{i}/{total_players}] {name} {season} -- fetched {rows} shots")
            except Exception as exc:
                errors += 1
                _safe_print(f"  [{i}/{total_players}] {name} {season} -- ERROR: {exc}")
                time.sleep(2)

        _safe_print(f"  {season} done: {season_cached} cached, {season_fetched} fetched")

    _safe_print(f"\nAll done. Cached: {cached}  Fetched: {fetched}  Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-fetch shot data for active players")
    parser.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help="NBA seasons to cache (e.g. 2025-26 2024-25)",
    )
    args = parser.parse_args()
    prefetch_active(args.seasons)
