"""Blend projection sources into one stat line per player-week, then score it (spec §3–4).

Sources: Sleeper (all positions), ESPN (QB/RB/WR/TE), the in-house model (QB/RB/WR/TE),
and an optional FantasyPros CSV. Per position, each source's weight is the inverse of
its backtested MAE (`data/cache/blend_weights.json`, written by ffcalc.backtest), equal
until a backtest has run. A missing source is dropped and the rest renormalized.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from . import cache, model
from .clients.sleeper import stat_frame
from .history import HIST_SEASONS
from .scoring import score_lines

log = logging.getLogger(__name__)

SOURCES = ("sleeper", "espn", "inhouse", "fantasypros")
LINE_SCHEMA = {
    "source": pl.String, "season": pl.Int64, "week": pl.Int64,
    "sleeper_id": pl.String, "stat": pl.String, "value": pl.Float64,
}  # fmt: skip


def weights_path() -> Any:
    return cache.data_dir() / "cache" / "blend_weights.json"


def load_weights() -> dict[str, dict[str, float]]:
    """position → source → weight. Empty dict means equal weights."""
    try:
        return json.loads(weights_path().read_text())
    except (OSError, ValueError):
        return {}


def blend(
    lines: pl.DataFrame, positions: pl.DataFrame, weights: Mapping[str, Mapping[str, float]]
) -> pl.DataFrame:
    """Weighted mean stat line per (season, week, sleeper_id) over the sources present.

    lines: LINE_SCHEMA; positions: sleeper_id, position.
    Returns season, week, sleeper_id, stat, value, n_sources.
    """
    w_rows = [(p, s, float(v)) for p, d in weights.items() for s, v in d.items()]
    w = pl.DataFrame(
        w_rows, schema={"position": pl.String, "source": pl.String, "w": pl.Float64}, orient="row"
    )
    keys = ["season", "week", "sleeper_id"]
    present = (
        lines.select(*keys, "source")
        .unique()
        .join(positions, on="sleeper_id", how="left")
        .join(w, on=["position", "source"], how="left")
        .with_columns(pl.col("w").fill_null(1.0).clip(1e-6))  # never divide by 0
    )
    # a source only counts toward a stat it models (ESPN/in-house have no first-down or
    # bonus keys); a covered stat missing from a line is a real 0
    cover = lines.select("source", "stat").unique()
    denom = (
        present.select(*keys, "source", "w")
        .join(cover, on="source")
        .group_by(*keys, "stat")
        .agg(pl.col("w").sum().alias("wsum"))
    )
    n_src = present.group_by(keys).agg(pl.len().alias("n_sources"))
    return (
        lines.join(present.select(*keys, "source", "w"), on=[*keys, "source"])
        .group_by(*keys, "stat")
        .agg((pl.col("value") * pl.col("w")).sum().alias("v"))
        .join(denom, on=[*keys, "stat"])
        .join(n_src, on=keys)
        .select(*keys, "stat", (pl.col("v") / pl.col("wsum")).alias("value"), "n_sources")
    )


@dataclass
class Projections:
    points: pl.DataFrame  # season, week, sleeper_id, mean, n_sources
    by_source: pl.DataFrame  # season, week, sleeper_id, source, points
    games: pl.DataFrame  # season, week, team, opponent, implied (team schedule)


def team_games(schedules: pl.DataFrame, season: int) -> pl.DataFrame:
    return model.implied_totals(schedules.filter(pl.col("season") == season))


def inhouse_lines(
    src: Any, season: int, current_week: int, weeks: Iterable[int], games: pl.DataFrame,
    table: pl.DataFrame, trained: Mapping[str, Any],
) -> pl.DataFrame:  # fmt: skip
    """In-house model stat lines for `weeks`, features as of `current_week`."""
    act = model.actual_wide(src.sleeper, season, range(1, current_week))
    if act.is_empty():
        return pl.DataFrame(schema=LINE_SCHEMA)
    feats = model.features_as_of(act, season, current_week)
    if feats.is_empty():
        return pl.DataFrame(schema=LINE_SCHEMA)
    # the player's current team (trades/signings) decides which game he's in
    feats = feats.drop("team").join(table.select("sleeper_id", "team"), on="sleeper_id")
    want = games.filter(pl.col("week").is_in(list(weeks))).select(
        "season", "week", "team", "implied"
    )
    g = feats.select("sleeper_id", "team").join(want, on="team").drop("team")
    return model.predict(trained, feats, g).select(list(LINE_SCHEMA))


def build(
    src: Any,
    season: int,
    current_week: int,
    weeks: Iterable[int],
    scoring: Mapping[str, float],
    table: pl.DataFrame,
    fp_lines: pl.DataFrame | None = None,
    weights: Mapping[str, Mapping[str, float]] | None = None,
) -> Projections:
    weeks = list(weeks)
    parts = []
    for w in weeks:
        try:
            parts.append(stat_frame(src.sleeper.projections(season, w), "sleeper"))
        except Exception as e:
            log.warning("Sleeper projections %s wk %s unavailable: %s", season, w, e)
    if src.espn is not None:
        try:
            parts.append(src.espn.projections(season, table).filter(pl.col("week").is_in(weeks)))
        except Exception as e:
            log.warning("ESPN projections unavailable: %s", e)
    games = pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64, "team": pl.String,
                                 "opponent": pl.String, "implied": pl.Float64})  # fmt: skip
    if src.nflverse is not None:
        try:
            games = team_games(src.nflverse.schedules([season]), season)
            trained = model.load_or_train(
                src.sleeper, src.nflverse.schedules(list(HIST_SEASONS)), HIST_SEASONS
            )
            parts.append(inhouse_lines(src, season, current_week, weeks, games, table, trained))
        except Exception as e:
            log.warning("in-house model unavailable: %s", e)
    if fp_lines is not None and not fp_lines.is_empty():
        parts.append(
            fp_lines.with_columns(pl.lit(season).alias("season")).select(list(LINE_SCHEMA))
        )
    parts = [p.select(list(LINE_SCHEMA)) for p in parts if not p.is_empty()]
    if not parts:
        raise RuntimeError("No projection source is reachable right now. Try again in a minute.")
    lines = pl.concat(parts, how="vertical_relaxed")
    # players on a bye get no game; drop any source line for a team without a game that week
    if not games.is_empty():
        on_field = table.select("sleeper_id", "team").join(games.select("week", "team"), on="team")
        lines = lines.join(
            on_field.select("sleeper_id", "week"), on=["sleeper_id", "week"], how="semi"
        )
    positions = table.select("sleeper_id", "position")
    blended = blend(lines, positions, load_weights() if weights is None else weights)
    pts = score_lines(blended.drop("n_sources"), scoring).rename({"points": "mean"})
    pts = pts.join(blended.select("season", "week", "sleeper_id", "n_sources").unique(),
                   on=["season", "week", "sleeper_id"])  # fmt: skip
    by_source = score_lines(lines, scoring)
    return Projections(pts.with_columns(pl.col("mean").clip(0.0)), by_source, games)
