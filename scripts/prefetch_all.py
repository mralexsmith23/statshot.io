"""
Pre-fetch shot chart data for EVERY NBA player across ALL seasons (1996-97+).

Builds the complete historical dataset by:
  1. Iterating every player in the nba_api database
  2. Discovering each player's seasons via career stats
  3. Fetching + caching shot data for every player-season combo

Features:
  - 3-second pacing between API calls to avoid rate-limiting
  - JSON progress file for resume capability (restart picks up where it left off)
  - Batch logging with ETA estimates
  - Skips players/seasons that are already cached

Run from project root:
    python -m scripts.prefetch_all
    python -m scripts.prefetch_all --delay 5          # slower pacing
    python -m scripts.prefetch_all --resume            # pick up from last run (default)
    python -m scripts.prefetch_all --no-resume         # start fresh
    python -m scripts.prefetch_all --batch-size 500    # stop after N fetches
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats

sys.path.insert(0, ".")
from src.cache import load_shots, is_cached  # noqa: E402

SHOT_DATA_FIRST_SEASON = 1996
PROGRESS_FILE = Path("data/prefetch_progress.json")


def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), flush=True)


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _get_all_seasons(player_id: int, delay: float) -> list[str]:
    """Fetch all seasons for a player from 1996-97 onward."""
    try:
        time.sleep(delay)
        career = playercareerstats.PlayerCareerStats(
            player_id=player_id, timeout=60,
        )
        df = career.get_data_frames()[0]
        if df.empty:
            return []
        all_szns = sorted(df["SEASON_ID"].unique(), reverse=True)
        return [s for s in all_szns if int(s[:4]) >= SHOT_DATA_FIRST_SEASON]
    except Exception as exc:
        _safe_print(f"      Could not fetch career seasons: {exc}")
        return []


def prefetch_all(
    delay: float = 3.0,
    resume: bool = True,
    batch_size: int | None = None,
) -> None:
    all_players = sorted(players.get_players(), key=lambda p: p["full_name"])
    total_players = len(all_players)

    progress = _load_progress() if resume else {}
    completed_keys: set[str] = set(progress.get("completed", []))
    failed_keys: list[str] = progress.get("failed", [])

    stats = {
        "players_processed": 0,
        "already_cached": 0,
        "fetched": 0,
        "empty": 0,
        "errors": 0,
        "skipped_no_seasons": 0,
    }
    api_calls = 0
    start_time = time.time()

    _safe_print(f"{'='*60}")
    _safe_print(f"StatShot — Full Dataset Build")
    _safe_print(f"{'='*60}")
    _safe_print(f"Total players in database: {total_players}")
    _safe_print(f"Already completed (from prior runs): {len(completed_keys)}")
    _safe_print(f"API delay: {delay}s between requests")
    if batch_size:
        _safe_print(f"Batch size: will stop after {batch_size} API fetches")
    _safe_print(f"Progress file: {PROGRESS_FILE}")
    _safe_print(f"{'='*60}\n")

    for player_idx, p in enumerate(all_players, 1):
        pid = p["id"]
        name = p["full_name"]

        if batch_size and stats["fetched"] >= batch_size:
            _safe_print(f"\nBatch limit reached ({batch_size} fetches). Saving progress...")
            break

        player_key = str(pid)
        if player_key in completed_keys:
            continue

        stats["players_processed"] += 1

        seasons = _get_all_seasons(pid, delay)
        api_calls += 1

        if not seasons:
            stats["skipped_no_seasons"] += 1
            completed_keys.add(player_key)
            if player_idx % 50 == 0:
                _save_progress({"completed": list(completed_keys), "failed": failed_keys})
            continue

        _safe_print(
            f"[{player_idx}/{total_players}] {name} — "
            f"{len(seasons)} seasons ({seasons[-1]} to {seasons[0]})"
        )

        player_had_errors = False
        for szn in seasons:
            combo_key = f"{pid}_{szn}"

            if combo_key in completed_keys:
                stats["already_cached"] += 1
                continue

            if is_cached(pid, szn):
                stats["already_cached"] += 1
                completed_keys.add(combo_key)
                continue

            try:
                df = load_shots(pid, szn, allow_api=True)
                api_calls += 1
                if df.empty:
                    stats["empty"] += 1
                    _safe_print(f"    {szn} — 0 shots")
                else:
                    stats["fetched"] += 1
                    _safe_print(f"    {szn} — {len(df)} shots cached")
                completed_keys.add(combo_key)
            except Exception as exc:
                stats["errors"] += 1
                player_had_errors = True
                err_msg = str(exc)
                if len(err_msg) > 80:
                    err_msg = err_msg[:80] + "..."
                _safe_print(f"    {szn} — ERROR: {err_msg}")
                failed_keys.append(combo_key)
                time.sleep(delay * 2)

            if batch_size and stats["fetched"] >= batch_size:
                break

        if not player_had_errors:
            completed_keys.add(player_key)

        if player_idx % 20 == 0:
            _save_progress({"completed": list(completed_keys), "failed": failed_keys})
            elapsed = time.time() - start_time
            rate = api_calls / elapsed * 3600 if elapsed > 0 else 0
            _safe_print(
                f"\n  --- Progress: {stats['fetched']} fetched, "
                f"{stats['already_cached']} cached, "
                f"{stats['errors']} errors, "
                f"{rate:.0f} API calls/hr, "
                f"elapsed {elapsed/60:.0f}min ---\n"
            )

    _save_progress({"completed": list(completed_keys), "failed": failed_keys})

    elapsed = time.time() - start_time
    _safe_print(f"\n{'='*60}")
    _safe_print(f"DONE — Full Dataset Build")
    _safe_print(f"{'='*60}")
    _safe_print(f"Players processed this run: {stats['players_processed']}")
    _safe_print(f"Player-seasons fetched:     {stats['fetched']}")
    _safe_print(f"Player-seasons cached:      {stats['already_cached']}")
    _safe_print(f"Player-seasons empty:       {stats['empty']}")
    _safe_print(f"No seasons (pre-1996):      {stats['skipped_no_seasons']}")
    _safe_print(f"Errors:                     {stats['errors']}")
    _safe_print(f"Total API calls:            {api_calls}")
    _safe_print(f"Elapsed:                    {elapsed/60:.1f} min")
    _safe_print(f"Progress saved to:          {PROGRESS_FILE}")
    if failed_keys:
        _safe_print(f"\nFailed combos ({len(failed_keys)}) saved in progress file.")
        _safe_print("Re-run with --resume to retry them.")
    _safe_print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-fetch shot data for every NBA player, all seasons (1996-97+)",
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Seconds between API calls (default: 3.0)",
    )
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="Resume from last progress checkpoint (default)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Start fresh, ignore previous progress",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Stop after N API fetches (useful for running in chunks)",
    )
    args = parser.parse_args()
    prefetch_all(delay=args.delay, resume=args.resume, batch_size=args.batch_size)
