"""Fake data clients that read tests/fixtures/ (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from ffcalc.clients.espn import EspnClient
from ffcalc.clients.fantasycalc import values_frame
from ffcalc.sources import Sources

FIXTURES = Path(__file__).parent / "fixtures"


def _load(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


class FakeSleeper:
    """Same methods as SleeperClient; every league id returns the fixture league."""

    def __init__(self) -> None:
        self.dir = FIXTURES / "sleeper"

    def _f(self, name: str, default: Any = None) -> Any:
        return _load(self.dir / f"{name}.json", default)

    def state(self) -> dict:
        return self._f("state")

    def league(self, league_id: str) -> dict:
        return self._f("league")

    def rosters(self, league_id: str) -> list[dict]:
        return self._f("rosters", [])

    def users(self, league_id: str) -> list[dict]:
        return self._f("users", [])

    def matchups(self, league_id: str, week: int) -> list[dict]:
        return self._f(f"matchups_{week}", [])

    def transactions(self, league_id: str, week: int) -> list[dict]:
        return self._f(f"transactions_{week}", [])

    def user(self, username: str) -> dict | None:
        # fixture users are matched on display name (the fixture has no usernames)
        for u in self.users(""):
            if u["display_name"].lower() == username.lower():
                return {"user_id": u["user_id"], "username": u["display_name"]}
        return None

    def user_leagues(self, user_id: str, season: int) -> list[dict]:
        lg = self.league("")
        return [lg] if int(lg["season"]) == season else []

    def players(self) -> dict[str, dict]:
        return self._f("players", {})

    def projections(self, season: int, week: int) -> list[dict]:
        return self._f(f"projections_{season}_{week}", [])

    def stats(self, season: int, week: int) -> list[dict]:
        return self._f(f"stats_{season}_{week}", [])


class FakeFantasyCalc:
    def values(
        self, num_teams: int, num_qbs: int, ppr: float, dynasty: bool = False
    ) -> pl.DataFrame:
        # one fixture (14 teams, 1 QB, PPR 1); other settings return it unchanged
        return values_frame(_load(FIXTURES / "fantasycalc" / "values_14_1_1.json", []))


class FakeEspn(EspnClient):
    """Real parsing code over a trimmed real response (seasons 2024 and 2026 only)."""

    def __init__(self) -> None:
        super().__init__(season_now=2026)  # never the wall clock

    def season_raw(self, season: int) -> dict:
        return _load(FIXTURES / "espn" / f"season_{season}.json", {"players": []})


NFL_FIXTURES = Path(__file__).parent / "fixtures" / "nflverse"


def _load_nfl(name: str) -> pl.DataFrame:
    return pl.read_parquet(NFL_FIXTURES / f"{name}.parquet")


class FakeNflverse:
    def _seasons(self, name: str, seasons: list[int]) -> pl.DataFrame:
        df = _load_nfl(name)
        return df.filter(pl.col("season").is_in(seasons)) if "season" in df.columns else df

    def player_stats(self, seasons: list[int]) -> pl.DataFrame:
        return self._seasons("player_stats", seasons)

    def schedules(self, seasons: list[int]) -> pl.DataFrame:
        return self._seasons("schedules", seasons)

    def injuries(self, seasons: list[int]) -> pl.DataFrame:
        return self._seasons("injuries", seasons)

    def snap_counts(self, seasons: list[int]) -> pl.DataFrame:
        return self._seasons("snap_counts", seasons)

    def ff_playerids(self) -> pl.DataFrame:
        return _load_nfl("ff_playerids")


def sample_sleeper_players() -> dict[str, dict]:
    """~27 real Sleeper player objects (trimmed keys), incl. DEF rows and a namesake WR."""
    return json.loads((NFL_FIXTURES / "sleeper_players_sample.json").read_text())


def fake_sources() -> Sources:
    return Sources(
        sleeper=FakeSleeper(),
        fantasycalc=FakeFantasyCalc(),
        espn=FakeEspn(),
        nflverse=FakeNflverse(),
    )
