"""
Pre-fetch shot chart data for every NBA player across all seasons (1996-97+).

Optimised pipeline:
  1. Single CommonAllPlayers call to get FROM_YEAR / TO_YEAR for everyone
  2. Skip ~2,200 pre-1996 players with zero API calls
  3. Generate season list from year ranges (no per-player career-stats call)
  4. Fetch + cache shot data for every player-season combo
  5. Adaptive back-off on consecutive errors to avoid API bans

Run from project root:
    python -m scripts.prefetch_all
    python -m scripts.prefetch_all --delay 5
    python -m scripts.prefetch_all --no-resume
    python -m scripts.prefetch_all --batch-size 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import commonallplayers

sys.path.insert(0, ".")
from src.cache import load_shots, is_cached  # noqa: E402

SHOT_DATA_FIRST_SEASON = 1996
PROGRESS_FILE = Path("data/prefetch_progress.json")

BASE_DELAY = 4.5
MAX_BACKOFF = 30.0
BACKOFF_STEP = 3.0


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


def _seasons_for_player(from_year: int, to_year: int) -> list[str]:
    """Generate NBA season strings (e.g. '2020-21') for each year the player
    was active during the shot-data era, newest first."""
    start = max(from_year, SHOT_DATA_FIRST_SEASON)
    seasons = []
    for yr in range(to_year, start - 1, -1):
        seasons.append(f"{yr}-{str(yr + 1)[-2:]}")
    return seasons


def _fetch_shot_data_roster() -> list[dict]:
    """One API call to get all players with year ranges, filtered to 1996+."""
    _safe_print("Fetching full player roster from NBA API (one-time call)...")
    cap = commonallplayers.CommonAllPlayers(
        is_only_current_season=0, timeout=120,
    )
    df = cap.get_data_frames()[0]
    roster = []
    for _, row in df.iterrows():
        to_year = int(row["TO_YEAR"])
        if to_year < SHOT_DATA_FIRST_SEASON:
            continue
        roster.append({
            "id": int(row["PERSON_ID"]),
            "name": row["DISPLAY_FIRST_LAST"],
            "from_year": int(row["FROM_YEAR"]),
            "to_year": to_year,
        })
    roster.sort(key=lambda p: p["name"])
    return roster


def prefetch_all(
    delay: float = BASE_DELAY,
    resume: bool = True,
    batch_size: int | None = None,
) -> None:
    roster = _fetch_shot_data_roster()
    total = len(roster)

    progress = _load_progress() if resume else {}
    completed_keys: set[str] = set(progress.get("completed", []))
    failed_keys: list[str] = progress.get("failed", [])

    stats = {
        "fetched": 0,
        "already_cached": 0,
        "empty": 0,
        "errors": 0,
        "skipped_complete": 0,
    }
    api_calls = 0
    consecutive_errors = 0
    current_delay = delay
    start_time = time.time()

    _safe_print(f"\n{'=' * 60}")
    _safe_print("StatShot  -  Full Dataset Build (v2)")
    _safe_print(f"{'=' * 60}")
    _safe_print(f"Shot-data-era players: {total}")
    _safe_print(f"Already completed:     {len(completed_keys)}")
    _safe_print(f"Base API delay:        {delay}s")
    if batch_size:
        _safe_print(f"Batch size:            {batch_size}")
    _safe_print(f"Progress file:         {PROGRESS_FILE}")
    _safe_print(f"{'=' * 60}\n")

    for idx, player in enumerate(roster, 1):
        pid = player["id"]
        name = player["name"]
        player_key = str(pid)

        if player_key in completed_keys:
            stats["skipped_complete"] += 1
            continue

        if batch_size and stats["fetched"] >= batch_size:
            _safe_print(f"\nBatch limit reached ({batch_size} fetches). Saving...")
            break

        seasons = _seasons_for_player(player["from_year"], player["to_year"])
        if not seasons:
            completed_keys.add(player_key)
            continue

        _safe_print(
            f"[{idx}/{total}] {name}  -  "
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

            time.sleep(current_delay)
            try:
                df = load_shots(pid, szn, allow_api=True)
                api_calls += 1
                consecutive_errors = 0
                if current_delay > delay:
                    current_delay = max(delay, current_delay - 1.0)

                if df.empty:
                    stats["empty"] += 1
                    _safe_print(f"    {szn}  -  0 shots")
                else:
                    stats["fetched"] += 1
                    _safe_print(f"    {szn}  -  {len(df)} shots cached")
                completed_keys.add(combo_key)

            except Exception as exc:
                stats["errors"] += 1
                consecutive_errors += 1
                player_had_errors = True
                err_msg = str(exc)[:80]
                _safe_print(f"    {szn}  -  ERROR: {err_msg}")
                failed_keys.append(combo_key)

                current_delay = min(MAX_BACKOFF, current_delay + BACKOFF_STEP)
                _safe_print(f"    (backoff -> {current_delay:.1f}s delay)")

                if consecutive_errors >= 8:
                    _safe_print(
                        "\n*** 8 consecutive errors — cooling off for 90s ***"
                    )
                    time.sleep(90)
                    consecutive_errors = 0
                    current_delay = delay

            if batch_size and stats["fetched"] >= batch_size:
                break

        if not player_had_errors:
            completed_keys.add(player_key)

        if idx % 15 == 0:
            _save_progress({
                "completed": list(completed_keys),
                "failed": failed_keys,
            })
            elapsed = time.time() - start_time
            rate = api_calls / elapsed * 3600 if elapsed > 0 else 0
            _safe_print(
                f"\n  --- {stats['fetched']} fetched | "
                f"{stats['already_cached']} cached | "
                f"{stats['errors']} errors | "
                f"{rate:.0f} calls/hr | "
                f"{elapsed / 60:.0f}m elapsed ---\n"
            )

    _save_progress({"completed": list(completed_keys), "failed": failed_keys})

    elapsed = time.time() - start_time
    _safe_print(f"\n{'=' * 60}")
    _safe_print("DONE  -  Full Dataset Build")
    _safe_print(f"{'=' * 60}")
    _safe_print(f"Player-seasons fetched:  {stats['fetched']}")
    _safe_print(f"Player-seasons cached:   {stats['already_cached']}")
    _safe_print(f"Empty (0 shots):         {stats['empty']}")
    _safe_print(f"Errors:                  {stats['errors']}")
    _safe_print(f"Players skipped (done):  {stats['skipped_complete']}")
    _safe_print(f"Total API calls:         {api_calls}")
    _safe_print(f"Elapsed:                 {elapsed / 60:.1f} min")
    if failed_keys:
        _safe_print(f"\n{len(failed_keys)} failed combos saved. Re-run with --resume to retry.")
    _safe_print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-fetch shot data for every NBA player (1996-97+)",
    )
    parser.add_argument(
        "--delay", type=float, default=BASE_DELAY,
        help=f"Seconds between API calls (default: {BASE_DELAY})",
    )
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Stop after N API fetches",
    )
    args = parser.parse_args()
    prefetch_all(delay=args.delay, resume=args.resume, batch_size=args.batch_size)
