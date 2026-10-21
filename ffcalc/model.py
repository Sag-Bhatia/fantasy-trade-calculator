"""In-house projection model (spec §4).

Ridge regression per position and stat component, trained on Sleeper's historical
actual stat lines (usage: targets, carries, pass attempts, snap share; production) plus
the Vegas implied team total from nflverse schedules. Features use only weeks before
the week being predicted. Predicts the stat line *if he plays*; it's one source in the
blend, scored with league settings like every other source.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import polars as pl

from . import cache
from .history import week_frames

log = logging.getLogger(__name__)

TARGETS = {
    "QB": ["pass_att", "pass_cmp", "pass_yd", "pass_td", "pass_int",
           "rush_att", "rush_yd", "rush_td", "fum_lost"],
    "RB": ["rush_att", "rush_yd", "rush_td", "rec_tgt", "rec", "rec_yd", "rec_td", "fum_lost"],
    "WR": ["rec_tgt", "rec", "rec_yd", "rec_td", "rush_att", "rush_yd", "rush_td", "fum_lost"],
    "TE": ["rec_tgt", "rec", "rec_yd", "rec_td", "fum_lost"],
}  # fmt: skip
USAGE = ["pass_att", "pass_yd", "pass_td", "rush_att", "rush_yd", "rush_td",
         "rec_tgt", "rec", "rec_yd", "rec_td", "snap_share"]  # fmt: skip
STATS = sorted({s for v in TARGETS.values() for s in v} | {"off_snp", "tm_off_snp"})
MIN_GAMES = 2
MODEL_VERSION = 1  # bump when features/targets change, so the cached fit is ignored
LAMBDA = 3.0
TEAM_FIX = {"LA": "LAR", "WSH": "WAS", "JAC": "JAX", "LVR": "LV"}  # nflverse → Sleeper


def implied_totals(schedules: pl.DataFrame) -> pl.DataFrame:
    """season, week, team (Sleeper code), opponent, implied (Vegas team total, may be null)."""
    s = schedules.filter(pl.col("game_type") == "REG")
    fix = lambda c: pl.col(c).replace(TEAM_FIX)  # noqa: E731
    home = s.select(
        "season", "week", fix("home_team").alias("team"), fix("away_team").alias("opponent"),
        ((pl.col("total_line") + pl.col("spread_line")) / 2).alias("implied"),
    )  # fmt: skip
    away = s.select(
        "season", "week", fix("away_team").alias("team"), fix("home_team").alias("opponent"),
        ((pl.col("total_line") - pl.col("spread_line")) / 2).alias("implied"),
    )  # fmt: skip
    return pl.concat([home, away]).with_columns(
        pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)
    )


def actual_wide(sleeper: Any, season: int, weeks: Iterable[int]) -> pl.DataFrame:
    """season, week, sleeper_id, position, team, <STATS> for every player-week that played."""
    out = []
    for w in weeks:
        try:
            lines, meta = week_frames(sleeper, season, w, "stats")
        except Exception as e:
            log.warning("no stats for %s wk %s: %s", season, w, e)
            continue
        if meta.is_empty():
            continue
        wide = lines.filter(pl.col("stat").is_in(STATS)).pivot(
            on="stat", index="sleeper_id", values="value", aggregate_function="sum"
        )
        for st in STATS:
            if st not in wide.columns:
                wide = wide.with_columns(pl.lit(0.0).alias(st))
        out.append(
            meta.filter(pl.col("position").is_in(list(TARGETS)))
            .select("sleeper_id", "position", "team")
            .join(wide, on="sleeper_id", how="left")
            .with_columns(pl.lit(season).alias("season"), pl.lit(w).alias("week"))
        )
    if not out:
        return pl.DataFrame()
    df = pl.concat(out, how="diagonal_relaxed").with_columns(pl.col(STATS).fill_null(0.0))
    return df.with_columns(
        (
            pl.col("off_snp")
            / pl.when(pl.col("tm_off_snp") > 0).then(pl.col("tm_off_snp")).otherwise(None)
        )
        .fill_null(0.0)
        .alias("snap_share")
    )


def features_as_of(act: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Per player: season-to-date and last-3 averages using games before `week`."""
    prior = act.filter((pl.col("season") == season) & (pl.col("week") < week)).sort("week")
    if prior.is_empty():
        return pl.DataFrame()
    return (
        prior.group_by("sleeper_id")
        .agg(
            pl.col("position").last(),
            pl.col("team").last(),
            pl.len().alias("games"),
            *[pl.col(c).mean().alias(f"std_{c}") for c in USAGE],
            *[pl.col(c).tail(3).mean().alias(f"l3_{c}") for c in USAGE],
        )
        .filter(pl.col("games") >= MIN_GAMES)
    )


FEATS = ["games", *[f"std_{c}" for c in USAGE], *[f"l3_{c}" for c in USAGE], "implied"]


def _design(df: pl.DataFrame, mean_implied: float) -> np.ndarray:
    x = df.with_columns(pl.col("implied").fill_null(mean_implied)).select(FEATS).to_numpy()
    return np.c_[np.ones(len(x)), x.astype(float)]


def train(act: pl.DataFrame, implied: pl.DataFrame, seasons: Iterable[int]) -> dict[str, Any]:
    """Ridge coefficients per position and target stat."""
    rows = []
    for s in seasons:
        for w in range(MIN_GAMES + 2, 19):
            f = features_as_of(act, s, w)
            if f.is_empty():
                continue
            y = act.filter((pl.col("season") == s) & (pl.col("week") == w)).drop("position", "team")
            rows.append(
                f.join(y, on="sleeper_id")
                .join(
                    implied.filter(pl.col("season") == s),
                    left_on=["team", "week"],
                    right_on=["team", "week"],
                    how="left",
                )  # fmt: skip
                .drop("season_right", "opponent", strict=False)
            )
    data = pl.concat(rows, how="diagonal_relaxed")
    mean_implied = float(implied["implied"].mean() or 22.5)  # type: ignore[arg-type]
    model: dict[str, Any] = {"mean_implied": mean_implied, "coef": {}, "n": {}}
    for pos, targets in TARGETS.items():
        d = data.filter(pl.col("position") == pos)
        x = _design(d, mean_implied)
        mu, sd = x[:, 1:].mean(0), x[:, 1:].std(0) + 1e-9
        xs = np.c_[np.ones(len(x)), (x[:, 1:] - mu) / sd]
        reg = LAMBDA * np.eye(xs.shape[1])
        reg[0, 0] = 0.0
        coefs = {}
        for t in targets:
            yv = d[t].to_numpy().astype(float)
            beta = np.linalg.solve(xs.T @ xs + reg, xs.T @ yv)
            coefs[t] = beta.tolist()
        model["coef"][pos] = {"mu": mu.tolist(), "sd": sd.tolist(), "beta": coefs}
        model["n"][pos] = d.height
    return model


def predict(model: Mapping[str, Any], feats: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """feats: features_as_of(...); games: sleeper_id, season, week, implied (one row per
    player-week to predict). Returns long lines: source="inhouse", season, week, sleeper_id,
    stat, value."""
    out = []
    df = feats.join(games, on="sleeper_id")
    for pos, m in model["coef"].items():
        d = df.filter(pl.col("position") == pos)
        if d.is_empty():
            continue
        x = _design(d, model["mean_implied"])
        xs = np.c_[np.ones(len(x)), (x[:, 1:] - np.array(m["mu"])) / np.array(m["sd"])]
        for stat, beta in m["beta"].items():
            v = np.clip(xs @ np.array(beta), 0.0, None)
            out.append(
                pl.DataFrame(
                    {
                        "source": "inhouse",
                        "season": d["season"],
                        "week": d["week"],
                        "sleeper_id": d["sleeper_id"],
                        "stat": stat,
                        "value": v,
                    }
                )  # fmt: skip
            )
    if not out:
        return pl.DataFrame(
            schema={
                "source": pl.String,
                "season": pl.Int64,
                "week": pl.Int64,
                "sleeper_id": pl.String,
                "stat": pl.String,
                "value": pl.Float64,
            }  # fmt: skip
        )
    return pl.concat(out).with_columns(pl.col("value").fill_nan(0.0)).filter(pl.col("value") > 0)


def load_or_train(sleeper: Any, schedules: pl.DataFrame, seasons: Iterable[int]) -> dict[str, Any]:
    seasons = list(seasons)
    key = f"model_v{MODEL_VERSION}_{'_'.join(map(str, seasons))}"

    def fit() -> dict[str, Any]:
        act = pl.concat(
            [actual_wide(sleeper, s, range(1, 19)) for s in seasons], how="diagonal_relaxed"
        )
        return train(act, implied_totals(schedules), seasons)

    return cache.cached_json(key, cache.TTL["history"], fit)


def dumps(model: Mapping[str, Any]) -> str:
    return json.dumps(model)
