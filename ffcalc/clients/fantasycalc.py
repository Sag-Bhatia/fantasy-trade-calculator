"""FantasyCalc market trade values (public, unofficial)."""

from __future__ import annotations

import polars as pl

from ffcalc import cache

URL = "https://api.fantasycalc.com/values/current"

SCHEMA = {
    "sleeper_id": pl.String,
    "name": pl.String,
    "position": pl.String,
    "team": pl.String,
    "value": pl.Int64,
    "overall_rank": pl.Int64,
    "position_rank": pl.Int64,
    "trend_30d": pl.Int64,
}


def values_frame(rows: list[dict]) -> pl.DataFrame:
    """Raw FantasyCalc rows -> flat frame; rows without a sleeperId are dropped."""
    out = [
        (
            str(p["sleeperId"]),
            p.get("name"),
            p.get("position"),
            p.get("maybeTeam"),
            r["value"],
            r.get("overallRank"),
            r.get("positionRank"),
            r.get("trend30Day"),
        )
        for r in rows
        if (p := r["player"]).get("sleeperId")
    ]
    return pl.DataFrame(out, schema=SCHEMA, orient="row")


class FantasyCalcClient:
    def values(
        self, num_teams: int, num_qbs: int, ppr: float, dynasty: bool = False
    ) -> pl.DataFrame:
        params = {
            "isDynasty": str(dynasty).lower(),
            "numQbs": num_qbs,
            "numTeams": num_teams,
            "ppr": ppr,
        }
        rows = cache.cached_json(
            f"fantasycalc/values_{int(dynasty)}_{num_teams}_{num_qbs}_{ppr}",
            cache.TTL["fantasycalc"],
            lambda: cache.get_json(URL, params=params),
        )
        return values_frame(rows)
