"""ESPN weekly projected and actual stat lines (unofficial kona_player_info endpoint).

One response holds every week of the season, so a season costs one request. QB/RB/WR/TE only:
ESPN's K and DEF stat ids are unverified, so those positions come from Sleeper.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import polars as pl

from ffcalc import cache

URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3"
FILTER = {
    "players": {
        "filterSlotIds": {"value": [0, 2, 4, 6]},
        "limit": 450,
        "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
    }
}
POSITION_IDS = {1, 2, 3, 4}  # QB RB WR TE (defaultPositionId)

# ESPN stat id -> Sleeper stat key (same vocabulary as ffcalc.scoring). All other ids are ignored.
STAT_MAP = {
    "0": "pass_att", "1": "pass_cmp", "3": "pass_yd", "4": "pass_td", "20": "pass_int",
    "19": "pass_2pt", "23": "rush_att", "24": "rush_yd", "25": "rush_td", "26": "rush_2pt",
    "53": "rec", "42": "rec_yd", "43": "rec_td", "44": "rec_2pt", "58": "rec_tgt",
    "72": "fum_lost",
}  # fmt: skip

SCHEMA = {
    "source": pl.String,
    "season": pl.Int64,
    "week": pl.Int64,
    "sleeper_id": pl.String,
    "stat": pl.String,
    "value": pl.Float64,
}


def current_season() -> int:
    t = date.today()
    return t.year if t.month >= 3 else t.year - 1


def _rows(raw: dict, season: int, source_id: int, id_map: dict[str, str], source: str):
    """Yield (week, sleeper_id, stat, value) for weekly lines; count unmapped players."""
    out, unmapped = [], set()
    for entry in raw.get("players", []):
        p = entry.get("player") or {}
        if p.get("defaultPositionId") not in POSITION_IDS:
            continue
        sid = id_map.get(str(p.get("id", entry.get("id"))))
        for s in p.get("stats", []):
            if (
                s.get("statSourceId") != source_id
                or s.get("statSplitTypeId") != 1
                or s.get("seasonId") != season
                or not s.get("scoringPeriodId")
            ):
                continue
            if sid is None:
                unmapped.add(p.get("id"))
                continue
            for k, v in s["stats"].items():
                if k in STAT_MAP and v:
                    out.append((source, season, s["scoringPeriodId"], sid, STAT_MAP[k], float(v)))
    return out, unmapped


class EspnClient:
    unmapped: set[Any]  # ESPN ids dropped (no sleeper id) by the last projections()/actuals() call

    def __init__(self, season_now: int | None = None) -> None:
        self._season_now = season_now  # pin in tests; default is the wall clock
        self.unmapped = set()

    def _now(self) -> int:
        return current_season() if self._season_now is None else self._season_now

    def season_raw(self, season: int) -> dict:
        ttl = cache.TTL["history" if season < self._now() else "projections"]

        def fetch() -> dict:
            return cache.get_json(
                URL.format(season=season),
                params={"view": "kona_player_info"},
                headers={"X-Fantasy-Filter": json.dumps(FILTER)},
            )

        return cache.cached_json(f"espn/season_{season}", ttl, fetch)

    def _frame(self, season: int, table: pl.DataFrame, source_id: int, source: str) -> pl.DataFrame:
        id_map = dict(
            table.filter(pl.col("espn_id").is_not_null())
            .select("espn_id", "sleeper_id")
            .iter_rows()
        )
        raw = self.season_raw(season)
        rows, unmapped = _rows(raw, season, source_id, id_map, source)
        self.unmapped |= unmapped
        return pl.DataFrame(rows, schema=SCHEMA, orient="row")

    def projections(self, season: int, table: pl.DataFrame) -> pl.DataFrame:
        self.unmapped = set()
        df = self._frame(season, table, 1, "espn")
        if season == self._now():
            # first week with no actuals yet; earlier weeks are no longer forecasts
            played = self._frame(season, table, 0, "espn_actual")
            cur = max(played["week"].to_list(), default=0) + 1
            for week in df.filter(pl.col("week") >= cur)["week"].unique().to_list():
                cache.snapshot("espn", season, week, df.filter(pl.col("week") == week))
        return df

    def actuals(self, season: int, table: pl.DataFrame) -> pl.DataFrame:
        self.unmapped = set()
        return self._frame(season, table, 0, "espn_actual")
