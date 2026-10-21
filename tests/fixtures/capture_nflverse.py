"""Regenerate tests/fixtures/nflverse/*.parquet from real nflverse data. Manual only (hits network).

Run: uv run python tests/fixtures/capture_nflverse.py
"""
# ruff: noqa: E402, I001

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ffcalc  # noqa: F401  (injects truststore before downloads)
import nflreadpy as nfl
import polars as pl

OUT = Path(__file__).parent / "nflverse"
SEASON, WEEKS = 2025, [1, 2, 3]
SKILL = ["QB", "RB", "WR", "TE", "K"]

stats = nfl.load_player_stats([SEASON]).filter(
    (pl.col("season_type") == "REG") & pl.col("week").is_in(WEEKS) & pl.col("position").is_in(SKILL)
)
# ~150 players: the top scorers over the three weeks.
top = (
    stats.group_by("player_id")
    .agg(pl.col("fantasy_points_ppr").sum().alias("pts"))
    .sort("pts", descending=True)
    .head(150)["player_id"]
    .to_list()
)
stats = stats.filter(pl.col("player_id").is_in(top))
stats.write_parquet(OUT / "player_stats.parquet")

ids = nfl.load_ff_playerids()
keep = ids.filter(pl.col("sleeper_id").is_not_null() & pl.col("position").is_in(SKILL))
# every captured stat player, plus a few hundred other skill players by recency (draft year)
in_stats = keep.filter(pl.col("gsis_id").is_in(top))
rest = keep.filter(~pl.col("gsis_id").is_in(top) | pl.col("gsis_id").is_null())
rest = rest.sort("draft_year", descending=True, nulls_last=True).head(250)
pl.concat([in_stats, rest]).write_parquet(OUT / "ff_playerids.parquet")

nfl.load_schedules([SEASON]).write_parquet(OUT / "schedules.parquet")
nfl.load_injuries([SEASON]).filter(pl.col("week").is_in(WEEKS)).write_parquet(
    OUT / "injuries.parquet"
)
nfl.load_snap_counts([SEASON]).filter(pl.col("week").is_in(WEEKS)).write_parquet(
    OUT / "snap_counts.parquet"
)
