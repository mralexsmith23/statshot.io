"""
Pre-fetch shot chart data for historical legends across their key seasons.

Run from project root:
    python -m scripts.prefetch_historical

Historical data never changes, so this only needs to run once per player-season.
"""
from __future__ import annotations

import sys
import time

from nba_api.stats.static import players

sys.path.insert(0, ".")
from src.cache import load_shots, is_cached  # noqa: E402

# (player_name, list_of_seasons_to_cache)
# Covers the most-requested cross-era comparisons.
LEGENDS: list[tuple[str, list[str]]] = [
    ("Michael Jordan",    ["1996-97", "1997-98", "1995-96", "1992-93", "1990-91"]),
    ("Kobe Bryant",       ["2005-06", "2006-07", "2007-08", "2008-09", "2012-13"]),
    ("LeBron James",      ["2008-09", "2012-13", "2015-16", "2017-18", "2019-20"]),
    ("Kevin Durant",      ["2013-14", "2016-17", "2017-18", "2020-21", "2023-24"]),
    ("Stephen Curry",     ["2015-16", "2016-17", "2017-18", "2020-21", "2023-24"]),
    ("Shaquille O'Neal",  ["1999-00", "2000-01", "2001-02", "2002-03"]),
    ("Tim Duncan",        ["2002-03", "2003-04", "2004-05", "2006-07"]),
    ("Dirk Nowitzki",     ["2005-06", "2006-07", "2010-11", "2013-14"]),
    ("Dwyane Wade",       ["2005-06", "2008-09", "2010-11", "2012-13"]),
    ("Allen Iverson",     ["2000-01", "2001-02", "2004-05", "2005-06"]),
    ("Steve Nash",        ["2004-05", "2005-06", "2006-07", "2009-10"]),
    ("Ray Allen",         ["2000-01", "2004-05", "2005-06", "2007-08"]),
    ("Carmelo Anthony",   ["2006-07", "2008-09", "2012-13", "2013-14"]),
    ("Paul Pierce",       ["2001-02", "2005-06", "2007-08", "2008-09"]),
    ("Tracy McGrady",     ["2002-03", "2003-04", "2004-05"]),
    ("Vince Carter",      ["1999-00", "2000-01", "2004-05", "2006-07"]),
    ("Chris Paul",        ["2007-08", "2008-09", "2014-15", "2020-21"]),
    ("Russell Westbrook", ["2014-15", "2015-16", "2016-17", "2017-18"]),
    ("James Harden",      ["2014-15", "2017-18", "2018-19", "2019-20"]),
    ("Kawhi Leonard",     ["2015-16", "2016-17", "2018-19", "2020-21"]),
    ("Giannis Antetokounmpo", ["2018-19", "2019-20", "2020-21", "2022-23"]),
    ("Nikola Jokic",      ["2020-21", "2021-22", "2022-23", "2023-24"]),
    ("Luka Doncic",       ["2019-20", "2020-21", "2021-22", "2023-24"]),
    ("Damian Lillard",    ["2015-16", "2017-18", "2019-20", "2020-21"]),
    ("Devin Booker",      ["2019-20", "2020-21", "2022-23", "2023-24"]),
    ("Jimmy Butler",      ["2016-17", "2019-20", "2021-22", "2022-23"]),
    ("Anthony Davis",     ["2017-18", "2019-20", "2020-21", "2023-24"]),
    ("Kyrie Irving",      ["2014-15", "2016-17", "2020-21", "2023-24"]),
    ("Paul George",       ["2013-14", "2018-19", "2019-20", "2023-24"]),
    ("Joel Embiid",       ["2018-19", "2020-21", "2021-22", "2022-23"]),
    ("Shai Gilgeous-Alexander", ["2021-22", "2022-23", "2023-24", "2024-25"]),
    ("Jayson Tatum",      ["2020-21", "2021-22", "2022-23", "2023-24"]),
]


def _p(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), flush=True)


def _resolve_id(name: str) -> int | None:
    matches = players.find_players_by_full_name(name)
    return matches[0]["id"] if matches else None


def prefetch_legends() -> None:
    total_combos = sum(len(szns) for _, szns in LEGENDS)
    fetched = 0
    cached = 0
    errors = 0
    idx = 0

    _p(f"Pre-fetching {total_combos} player-season combos for {len(LEGENDS)} legends...")

    for name, seasons in LEGENDS:
        pid = _resolve_id(name)
        if pid is None:
            _p(f"  WARNING: could not resolve '{name}', skipping")
            errors += len(seasons)
            idx += len(seasons)
            continue

        for szn in seasons:
            idx += 1
            if is_cached(pid, szn):
                cached += 1
                _p(f"  [{idx}/{total_combos}] {name} {szn} -- already cached")
                continue

            try:
                df = load_shots(pid, szn, allow_api=True)
                fetched += 1
                _p(f"  [{idx}/{total_combos}] {name} {szn} -- fetched {len(df)} shots")
            except Exception as exc:
                errors += 1
                _p(f"  [{idx}/{total_combos}] {name} {szn} -- ERROR: {exc}")
                time.sleep(2)

    _p(f"\nDone. Cached: {cached}  Fetched: {fetched}  Errors: {errors}")


if __name__ == "__main__":
    prefetch_legends()
