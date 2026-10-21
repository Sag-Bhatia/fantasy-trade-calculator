"""Sleeper: league data (official API) and weekly projections/stats (unofficial)."""

from __future__ import annotations

import re
from typing import Any

import polars as pl

from ffcalc import cache

API = "https://api.sleeper.app/v1"
PROJ_API = "https://api.sleeper.app"
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

# Pre-computed totals and ranks; NOT pts_allow_* (defense scoring buckets).
_DROP_KEYS = {"pts_ppr", "pts_std", "pts_half_ppr", "pts_idp"}
_DROP_PREFIXES = ("adp_", "pos_adp", "pos_rank", "rank_")


_USERNAME = re.compile(r"[A-Za-z0-9_.-]{1,40}")


def check_league_id(league_id: str) -> str:
    """League ids are long numbers; anything else never reaches a URL or cache key."""
    from ffcalc.league import LeagueNotFound  # local: league.py imports this module

    if not (isinstance(league_id, str) and league_id.isascii() and league_id.isdigit()):
        raise LeagueNotFound(league_id)
    return league_id


def check_username(username: str) -> str:
    if not (isinstance(username, str) and _USERNAME.fullmatch(username)):
        raise ValueError("A Sleeper username is 1-40 letters, digits, '_', '.' or '-' (no spaces).")
    return username


def _pos_params() -> list[tuple[str, str]]:
    return [("season_type", "regular"), *(("position[]", p) for p in FANTASY_POSITIONS)]


class SleeperClient:
    def _league_get(self, key: str, path: str, ttl: float | None = None) -> Any:
        return cache.cached_json(
            f"sleeper/{key}",
            cache.TTL["league"] if ttl is None else ttl,
            lambda: cache.get_json(f"{API}{path}"),
        )

    def state(self) -> dict:
        return self._league_get("state", "/state/nfl")

    def league(self, league_id: str) -> dict:
        check_league_id(league_id)
        return self._league_get(f"league_{league_id}", f"/league/{league_id}")

    def rosters(self, league_id: str) -> list[dict]:
        check_league_id(league_id)
        return self._league_get(f"rosters_{league_id}", f"/league/{league_id}/rosters")

    def users(self, league_id: str) -> list[dict]:
        check_league_id(league_id)
        return self._league_get(f"users_{league_id}", f"/league/{league_id}/users")

    def _week_ttl(self, week: int) -> float:
        # Finished weeks never change, except the latest one: stat corrections land days later.
        return cache.TTL["history" if week < int(self.state()["week"]) - 1 else "league"]

    def matchups(self, league_id: str, week: int) -> list[dict]:
        check_league_id(league_id)
        return self._league_get(
            f"matchups_{league_id}_{week}",
            f"/league/{league_id}/matchups/{week}",
            self._week_ttl(week),
        )

    def transactions(self, league_id: str, week: int) -> list[dict]:
        check_league_id(league_id)
        return self._league_get(
            f"transactions_{league_id}_{week}",
            f"/league/{league_id}/transactions/{week}",
            self._week_ttl(week),
        )

    def user(self, username: str) -> dict | None:
        check_username(username)
        return self._league_get(f"user_{username}", f"/user/{username}")

    def user_leagues(self, user_id: str, season: int) -> list[dict]:
        return self._league_get(
            f"user_leagues_{user_id}_{int(season)}", f"/user/{user_id}/leagues/nfl/{int(season)}"
        )

    def players(self) -> dict[str, dict]:
        # Sleeper asks for at most one dump per day; the 24 h TTL enforces that.
        return cache.cached_json(
            "sleeper/players",
            cache.TTL["players"],
            lambda: cache.get_json(f"{API}/players/nfl"),
        )

    def _is_past(self, season: int, week: int | None = None) -> bool:
        cur = self.state()
        if season < int(cur["season"]):
            return True
        return week is not None and season == int(cur["season"]) and week < int(cur["week"])

    def projections(self, season: int, week: int) -> list[dict]:
        url = f"{PROJ_API}/projections/nfl/{season}/{week}"
        if self._is_past(season, week):
            # past weeks: no raw cache (~2 MB each); ffcalc.history keeps compact frames
            return cache.get_json(url, params=_pos_params())

        def fetch() -> list[dict]:
            rows = cache.get_json(url, params=_pos_params())
            cache.snapshot("sleeper", season, week, stat_frame(rows, "sleeper"))
            return rows

        return cache.cached_json(f"sleeper/proj_{season}_{week}", cache.TTL["projections"], fetch)

    def stats(self, season: int, week: int) -> list[dict]:
        url = f"{PROJ_API}/stats/nfl/{season}/{week}"
        if season < int(self.state()["season"]):
            return cache.get_json(url, params=_pos_params())
        ttl = cache.TTL["history" if self._is_past(season, week) else "projections"]
        return cache.cached_json(
            f"sleeper/stats_{season}_{week}", ttl, lambda: cache.get_json(url, params=_pos_params())
        )


def stat_frame(rows: list[dict], source: str) -> pl.DataFrame:
    """Long format: one row per non-zero numeric stat of each player-week."""
    out = [
        (source, int(r["season"]), int(r["week"]), str(r["player_id"]), k, float(v))
        for r in rows
        for k, v in (r.get("stats") or {}).items()
        if isinstance(v, int | float)
        and v != 0
        and k not in _DROP_KEYS
        and not k.startswith(_DROP_PREFIXES)
    ]
    return pl.DataFrame(
        out,
        schema={
            "source": pl.String,
            "season": pl.Int64,
            "week": pl.Int64,
            "sleeper_id": pl.String,
            "stat": pl.String,
            "value": pl.Float64,
        },
        orient="row",
    )


def game_frame(rows: list[dict]) -> pl.DataFrame:
    out = [
        (
            int(r["season"]),
            int(r["week"]),
            str(r["player_id"]),
            r.get("team"),
            r.get("opponent"),
            r.get("game_id"),
        )
        for r in rows
    ]
    return pl.DataFrame(
        out,
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "sleeper_id": pl.String,
            "team": pl.String,
            "opponent": pl.String,
            "game_id": pl.String,
        },
        orient="row",
    )
