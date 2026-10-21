"""Stat line → fantasy points for a league's own scoring settings (spec §3).

Sleeper stat keys and `scoring_settings` keys are the same vocabulary, so points are a
dot product over shared keys. Bucket stats (`pts_allow_21_27`, `bonus_rec_yd_100`, ...)
arrive as 0/1 indicators (or fractions in projections) and score the same way.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

# Projections give only bucketed kicker stats while leagues may score the total or a
# different bucket; derive a missing key from its parts (never when the line has it).
# key → {part: sign}. fgm_50p includes 60+ makes, so 50-59 = 50p − 60p.
_MISS = ("fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39", "fgmiss_40_49", "fgmiss_50p")
_DERIVED: dict[str, dict[str, float]] = {
    "fgmiss": dict.fromkeys(_MISS, 1.0),
    "fgm_50_59": {"fgm_50p": 1.0, "fgm_60p": -1.0},
}


def complete(stats: Mapping[str, float]) -> dict[str, float]:
    out = dict(stats)
    for key, parts in _DERIVED.items():
        if key not in out and any(p in out for p in parts):
            out[key] = sum(out.get(p, 0.0) * sign for p, sign in parts.items())
    return out


def points(stats: Mapping[str, float], scoring: Mapping[str, float]) -> float:
    return float(sum(v * scoring[k] for k, v in complete(stats).items() if k in scoring))


def score_lines(lines: pl.DataFrame, scoring: Mapping[str, float]) -> pl.DataFrame:
    """Long stat lines (…, stat, value) → one row per remaining key with a `points` column."""
    weights = pl.DataFrame(
        {"stat": list(scoring), "weight": [float(v) for v in scoring.values()]},
        schema={"stat": pl.String, "weight": pl.Float64},
    )
    keys = [c for c in lines.columns if c not in ("stat", "value")]
    derived = []
    for key, parts in _DERIVED.items():
        has = lines.group_by(keys).agg((pl.col("stat") == key).any().alias("has"))
        sign = pl.col("stat").replace_strict(parts, default=0.0, return_dtype=pl.Float64)
        part = (
            lines.filter(pl.col("stat").is_in(list(parts)))
            .group_by(keys)
            .agg((pl.col("value") * sign).sum())
            .join(has, on=keys, how="left", nulls_equal=True)
            .filter(~pl.col("has").fill_null(False))
            .select(*keys, pl.lit(key).alias("stat"), "value")
        )
        derived.append(part)
    lines = pl.concat([lines.select(*keys, "stat", "value"), *derived], how="vertical_relaxed")
    return (
        lines.join(weights, on="stat", how="left")
        .with_columns((pl.col("value") * pl.col("weight").fill_null(0.0)).alias("p"))
        .group_by(keys, maintain_order=True)
        .agg(pl.col("p").sum().alias("points"))
    )


# Sleeper's default full-PPR settings, used only to sanity-check against `pts_ppr`.
STANDARD_PPR: dict[str, float] = {
    "pass_yd": 0.04, "pass_td": 4, "pass_int": -1, "pass_2pt": 2,
    "rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2,
    "rec": 1, "rec_yd": 0.1, "rec_td": 6, "rec_2pt": 2,
    "fum_lost": -2, "fum_rec_td": 6,
    "fgm_0_19": 3, "fgm_20_29": 3, "fgm_30_39": 3, "fgm_40_49": 4, "fgm_50p": 5,
    "fgmiss": -1, "xpm": 1, "xpmiss": -1,
    "def_td": 6, "sack": 1, "int": 2, "fum_rec": 2, "safe": 2, "ff": 1, "blk_kick": 2,
    "def_st_td": 6, "def_st_fum_rec": 1, "def_st_ff": 1, "st_td": 6, "st_fum_rec": 1, "st_ff": 1,
    "pts_allow_0": 10, "pts_allow_1_6": 7, "pts_allow_7_13": 4, "pts_allow_14_20": 1,
    "pts_allow_21_27": 0, "pts_allow_28_34": -1, "pts_allow_35p": -4,
}  # fmt: skip
