"""
Pre-fetch shot chart data for all active NBA players in the current season.

Run from project root:
    python -m scripts.prefetch_shots
    python -m scripts.prefetch_shots --season 2024-25

Takes ~8-12 minutes at ~1 request/sec for ~450 players.
"""
from __future__ import annotations

import argparse
import sys
import time

from nba_api.stats.static import players

sys.path.insert(0, ".")
from src.cache import load_shots, is_cached  # noqa: E402


def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), flush=True)


def prefetch_active(season: str) -> None:
    active = [p for p in players.get_players() if p["is_active"]]
    total = len(active)
    cached = 0
    fetched = 0
    errors = 0

    _safe_print(f"Pre-fetching {total} active players for {season}...")

    for i, p in enumerate(active, 1):
        pid = p["id"]
        name = p["full_name"]

        if is_cached(pid, season):
            cached += 1
            _safe_print(f"  [{i}/{total}] {name} -- already cached")
            continue

        try:
            df = load_shots(pid, season, allow_api=True)
            rows = len(df)
            fetched += 1
            _safe_print(f"  [{i}/{total}] {name} -- fetched {rows} shots")
        except Exception as exc:
            errors += 1
            _safe_print(f"  [{i}/{total}] {name} -- ERROR: {exc}")
            time.sleep(2)

    _safe_print(f"\nDone. Cached: {cached}  Fetched: {fetched}  Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-fetch shot data for active players")
    parser.add_argument("--season", default="2025-26", help="NBA season (e.g. 2025-26)")
    args = parser.parse_args()
    prefetch_active(args.season)
