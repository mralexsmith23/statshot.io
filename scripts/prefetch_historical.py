"""
Pre-fetch shot chart data for historical legends across ALL their seasons
(1996-97 onward, when NBA shot-location data became available).

Run from project root:
    python -m scripts.prefetch_historical

Historical data never changes, so this only needs to run once per player-season.
Uses the career stats API once per player to discover every season, then caches
all shot data locally.
"""
from __future__ import annotations

import sys
import time

from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats

sys.path.insert(0, ".")
from src.cache import load_shots, is_cached  # noqa: E402

SHOT_DATA_FIRST_SEASON = 1996

LEGENDS: list[str] = [
    # 90s / 2000s icons
    "Michael Jordan",
    "Kobe Bryant",
    "Shaquille O'Neal",
    "Tim Duncan",
    "Allen Iverson",
    "Ray Allen",
    "Vince Carter",
    "Tracy McGrady",
    "Steve Nash",
    "Dirk Nowitzki",
    "Paul Pierce",
    "Kevin Garnett",
    "Reggie Miller",
    "Gary Payton",
    "Jason Kidd",
    "Alonzo Mourning",
    "Hakeem Olajuwon",
    "Karl Malone",
    "John Stockton",
    "Scottie Pippen",
    "Patrick Ewing",
    "David Robinson",
    "Charles Barkley",
    "Rasheed Wallace",
    "Ben Wallace",
    "Chauncey Billups",
    "Richard Hamilton",
    # 2000s / 2010s stars
    "Dwyane Wade",
    "Carmelo Anthony",
    "Chris Paul",
    "LeBron James",
    "Kevin Durant",
    "Stephen Curry",
    "Russell Westbrook",
    "James Harden",
    "Kawhi Leonard",
    "Paul George",
    "Kyrie Irving",
    "Klay Thompson",
    "Draymond Green",
    "DeMar DeRozan",
    "Blake Griffin",
    "Chris Bosh",
    "Dwight Howard",
    "Tony Parker",
    "Manu Ginobili",
    "Rajon Rondo",
    "Derrick Rose",
    "John Wall",
    "Bradley Beal",
    "Kemba Walker",
    "Kyle Lowry",
    "Marc Gasol",
    "Pau Gasol",
    "Amar'e Stoudemire",
    "Yao Ming",
    # 2010s / 2020s stars (non-active or recently retired)
    "Giannis Antetokounmpo",
    "Nikola Jokic",
    "Luka Doncic",
    "Jayson Tatum",
    "Joel Embiid",
    "Damian Lillard",
    "Devin Booker",
    "Jimmy Butler",
    "Anthony Davis",
    "Shai Gilgeous-Alexander",
    "Donovan Mitchell",
    "Trae Young",
    "Ja Morant",
    "Zion Williamson",
    "Anthony Edwards",
    "LaMelo Ball",
    "Chet Holmgren",
    "Victor Wembanyama",
    "Paolo Banchero",
    "Tyrese Haliburton",
    "Tyrese Maxey",
    "Jalen Brunson",
    "De'Aaron Fox",
    "Bam Adebayo",
    "Domantas Sabonis",
    "Karl-Anthony Towns",
    "Rudy Gobert",
]


def _p(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), flush=True)


def _resolve_id(name: str) -> int | None:
    matches = players.find_players_by_full_name(name)
    return matches[0]["id"] if matches else None


def _get_all_seasons(player_id: int) -> list[str]:
    """Fetch all seasons for a player from 1996-97 onward."""
    try:
        time.sleep(0.6)
        career = playercareerstats.PlayerCareerStats(
            player_id=player_id, timeout=60,
        )
        df = career.get_data_frames()[0]
        if df.empty:
            return []
        all_szns = sorted(df["SEASON_ID"].unique(), reverse=True)
        return [s for s in all_szns if int(s[:4]) >= SHOT_DATA_FIRST_SEASON]
    except Exception as exc:
        _p(f"    Could not fetch career seasons: {exc}")
        return []


def prefetch_legends() -> None:
    fetched = 0
    cached = 0
    errors = 0
    skipped_empty = 0
    total_combos = 0

    _p(f"Discovering seasons for {len(LEGENDS)} legends...")

    for legend_idx, name in enumerate(LEGENDS, 1):
        pid = _resolve_id(name)
        if pid is None:
            _p(f"  [{legend_idx}/{len(LEGENDS)}] WARNING: could not resolve '{name}', skipping")
            continue

        seasons = _get_all_seasons(pid)
        if not seasons:
            _p(f"  [{legend_idx}/{len(LEGENDS)}] {name} — no seasons found (pre-1996 player?)")
            continue

        _p(f"  [{legend_idx}/{len(LEGENDS)}] {name} — {len(seasons)} seasons: {seasons[0]} to {seasons[-1]}")

        for szn in seasons:
            total_combos += 1
            if is_cached(pid, szn):
                cached += 1
                continue

            try:
                df = load_shots(pid, szn, allow_api=True)
                if df.empty:
                    skipped_empty += 1
                    _p(f"    {name} {szn} — 0 shots (no data)")
                else:
                    fetched += 1
                    _p(f"    {name} {szn} — fetched {len(df)} shots")
            except Exception as exc:
                errors += 1
                _p(f"    {name} {szn} — ERROR: {exc}")
                time.sleep(3)

    _p(f"\nDone. Total: {total_combos}  Cached: {cached}  Fetched: {fetched}  Empty: {skipped_empty}  Errors: {errors}")


if __name__ == "__main__":
    prefetch_legends()
