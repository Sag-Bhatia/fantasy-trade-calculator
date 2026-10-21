"""Accuracy and calibration on real past seasons → REPORT.md (spec §11).

`uv run python -m ffcalc.backtest [--league ID] [--quick]`

Every number in REPORT.md is computed here from real data; nothing is typed in by hand.

1. Projection accuracy: MAE per position for Sleeper, ESPN, the in-house model, the blend
   and naive baselines, on one common sample of fantasy-relevant player-weeks that played.
   Walk-forward: the in-house model and blend weights for season s use only seasons < s.
2. Distribution calibration: how often actual scores land below the predicted 10/50/90%
   quantiles (tables built from seasons < s).
3. Simulator calibration: the test league has no past seasons on Sleeper, so we use
   synthetic leagues built from real seasons (spec §11.3 fallback): 12 teams drafted by
   real week-1 projections, lineups set on real weekly projections, scored with real
   actual points. At weeks 6/9/12 the simulator predicts playoff odds from what was
   known then; we score them against what really happened (Brier score).
4. Trade replay: in those leagues, random 1-for-1 trades at each snapshot — does the side
   the model favors really score more starter points the rest of the way?
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from . import engine, history, ids, model, projections
from .lineup import optimal_lineup
from .models import POSITIONS, League, Team
from .scoring import STANDARD_PPR, score_lines
from .simulate import (
    Config,
    Outcomes,
    SimInputs,
    Tables,
    simulate_league,
    simulate_players,
    team_points,
)

log = logging.getLogger(__name__)

SEASONS = (2022, 2023, 2024, 2025)
EVAL_SEASONS = (2023, 2024, 2025)  # each needs at least one earlier season to learn from
SNAPSHOTS = (6, 9, 12)
LEAGUES_PER_SEASON = 20
N_TEAMS = 12
SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
BENCH = 6
DRAFT_LIMITS = {"QB": 2, "RB": 6, "WR": 6, "TE": 2, "K": 1, "DEF": 1}
SIMS = 2000
TRADES_PER_SNAPSHOT = 4
SEED = 2026


# ----------------------------------------------------------------- 1. accuracy


def _points(lines: pl.DataFrame, scoring: Mapping[str, float], name: str) -> pl.DataFrame:
    keys = ["season", "week", "sleeper_id"]
    return score_lines(lines.select(*keys, "stat", "value"), scoring).rename({"points": name})


def espn_points(
    src: Any, season: int, table: pl.DataFrame, scoring: Mapping[str, float]
) -> pl.DataFrame:
    try:
        return _points(src.espn.projections(season, table), scoring, "espn")
    except Exception as e:
        log.warning("ESPN %s unavailable: %s", season, e)
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "sleeper_id": pl.String,
                "espn": pl.Float64,
            }
        )


def inhouse_points(src: Any, season: int, scoring: Mapping[str, float], acts: dict[int, pl.DataFrame],
                   implied: pl.DataFrame) -> pl.DataFrame:  # fmt: skip
    """Walk-forward in-house projections for every week of `season`."""
    prior = [s for s in acts if s < season]
    if not prior:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "sleeper_id": pl.String,
                "inhouse": pl.Float64,
            }
        )
    trained = model.train(
        pl.concat([acts[s] for s in prior], how="diagonal_relaxed"), implied, prior
    )
    out = []
    act = acts[season]
    for w in range(model.MIN_GAMES + 1, 19):
        feats = model.features_as_of(act, season, w)
        if feats.is_empty():
            continue
        g = (
            feats.select("sleeper_id", "team")
            .join(implied.filter((pl.col("season") == season) & (pl.col("week") == w)), on="team")
            .select("sleeper_id", "season", "week", "implied")
        )
        out.append(model.predict(trained, feats, g))
    if not out:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "sleeper_id": pl.String,
                "inhouse": pl.Float64,
            }
        )
    return _points(pl.concat(out), scoring, "inhouse")


def baselines(pw: pl.DataFrame) -> pl.DataFrame:
    """Season-to-date and last-3 average of actual points in games played before the week."""
    played = pw.filter(pl.col("played")).sort("sleeper_id", "season", "week")
    by = ["sleeper_id", "season"]
    cum = played.with_columns(
        (pl.col("actual").cum_sum().over(by) - pl.col("actual")).alias("_sum"),
        (pl.col("actual").cum_count().over(by) - 1).alias("_n"),
        pl.col("actual").shift(1).rolling_mean(3, min_samples=1).over(by).alias("last3"),
    ).with_columns(
        pl.when(pl.col("_n") > 0)
        .then(pl.col("_sum") / pl.col("_n"))
        .otherwise(None)
        .alias("season_avg")
    )
    return cum.select("season", "week", "sleeper_id", "season_avg", "last3")


@dataclass
class Accuracy:
    table: pl.DataFrame  # position, source, mae, n
    weights: dict[str, dict[str, float]]
    coverage: pl.DataFrame  # position, below_q10, below_q50, below_q90, n
    notes: list[str] = field(default_factory=list)


def accuracy(
    src: Any, scoring: Mapping[str, float], table: pl.DataFrame
) -> tuple[Accuracy, pl.DataFrame]:
    pw = history.player_weeks(src.sleeper, scoring, SEASONS)
    schedules = src.nflverse.schedules(list(SEASONS))
    implied = model.implied_totals(schedules)
    acts = {s: model.actual_wide(src.sleeper, s, range(1, 19)) for s in SEASONS}
    keys = ["season", "week", "sleeper_id"]
    espn = pl.concat([espn_points(src, s, table, scoring) for s in EVAL_SEASONS])
    inhouse = pl.concat([inhouse_points(src, s, scoring, acts, implied) for s in EVAL_SEASONS])
    d = (
        pw.rename({"proj": "sleeper"})
        .join(baselines(pw), on=keys, how="left")
        .join(espn.unique(keys), on=keys, how="left")
        .join(inhouse.unique(keys), on=keys, how="left")
    )
    d = d.filter(pl.col("season").is_in(EVAL_SEASONS) & pl.col("played"))
    floor = pl.col("position").replace_strict(
        history.MIN_PROJ, default=99.0, return_dtype=pl.Float64
    )
    d = d.filter(pl.col("sleeper") >= floor)

    sources = ["sleeper", "espn", "inhouse"]
    rows = []
    weights: dict[str, dict[str, float]] = {}
    for pos in POSITIONS:
        g = d.filter(pl.col("position") == pos)
        have = [s for s in sources if s in g.columns and g[s].drop_nulls().len() > 100]
        common = g.drop_nulls([*have, "season_avg", "last3"])
        if common.is_empty():
            continue
        # walk-forward blend: weights for season s from MAE in seasons < s (equal at first)
        blend = np.zeros(common.height)
        for s in EVAL_SEASONS:
            m = (common["season"] == s).to_numpy()
            prior = common.filter(pl.col("season") < s)
            if prior.height > 50:
                w = {
                    src_: 1 / float((prior[src_] - prior["actual"]).abs().to_numpy().mean())
                    for src_ in have
                }
            else:
                w = dict.fromkeys(have, 1.0)
            tot = sum(w.values())
            blend[m] = (
                sum(common.filter(pl.col("season") == s)[k].to_numpy() * v for k, v in w.items())
                / tot
            )
        common = common.with_columns(pl.Series("blend", blend))
        for name in [*have, "blend", "season_avg", "last3"]:
            mae = float((common[name] - common["actual"]).abs().mean())  # type: ignore[arg-type]
            rows.append((pos, name, mae, common.height))
        weights[pos] = {
            k: round(1 / float((common[k] - common["actual"]).abs().to_numpy().mean()), 4)
            for k in have
        }  # type: ignore[arg-type]
    acc = pl.DataFrame(
        rows,
        schema={"position": pl.String, "source": pl.String, "mae": pl.Float64, "n": pl.Int64},
        orient="row",
    )
    return Accuracy(acc, weights, coverage(src, scoring, pw)), pw


def coverage(src: Any, scoring: Mapping[str, float], pw: pl.DataFrame) -> pl.DataFrame:
    """Share of actual scores below the predicted 10/50/90% quantiles (walk-forward)."""
    rows = []
    for s in EVAL_SEASONS:
        prior = pw.filter(pl.col("season") < s)
        edges, resid_q, _, _ = history.residual_tables(prior)
        cur = history._relevant(pw.filter((pl.col("season") == s) & pl.col("played")))
        for p, pos in enumerate(POSITIONS):
            g = cur.filter(pl.col("position") == pos)
            if g.is_empty():
                continue
            proj, act = g["proj"].to_numpy(), g["actual"].to_numpy()
            tier = np.searchsorted(edges[p], proj, side="right")
            q = resid_q[p][tier]  # (n, QGRID)
            rows.append(
                (pos, s, *(float(np.mean(act < proj + q[:, k])) for k in (10, 50, 90)), len(act))
            )
    df = pl.DataFrame(rows, schema={"position": pl.String, "season": pl.Int64, "below_q10": pl.Float64,
                                    "below_q50": pl.Float64, "below_q90": pl.Float64, "n": pl.Int64}, orient="row")  # fmt: skip
    return (
        df.group_by("position")
        .agg(
            *[
                (pl.col(c) * pl.col("n")).sum() / pl.col("n").sum()
                for c in ("below_q10", "below_q50", "below_q90")
            ],
            pl.col("n").sum(),
        )
        .sort(pl.col("position").replace_strict({p: i for i, p in enumerate(POSITIONS)}))
    )


# ----------------------------------------------------------------- 3. synthetic leagues


def round_robin(n: int, weeks: range, rng: np.random.Generator) -> dict[int, list[tuple[int, int]]]:
    ids_ = list(rng.permutation(np.arange(1, n + 1)))
    out = {}
    for wk in weeks:
        k = wk % (n - 1)
        r = ids_[:1] + ids_[1:][k:] + ids_[1:][:k]
        out[wk] = [(int(r[i]), int(r[n - 1 - i])) for i in range(n // 2)]
    return out


def draft(pool: pl.DataFrame, rng: np.random.Generator) -> list[list[str]]:
    """Snake draft by noisy week-1 projection with simple position caps."""
    order = pool.with_columns(
        (pl.col("w1") * pl.Series(rng.lognormal(0, 0.15, pool.height))).alias("k")
    ).sort("k", descending=True)
    avail = list(order.select("sleeper_id", "position").iter_rows())
    rosters: list[list[str]] = [[] for _ in range(N_TEAMS)]
    counts = [dict.fromkeys(DRAFT_LIMITS, 0) for _ in range(N_TEAMS)]
    size = len(SLOTS) + BENCH
    for rnd in range(size):
        seq = range(N_TEAMS) if rnd % 2 == 0 else reversed(range(N_TEAMS))
        for t in seq:
            left = size - len(rosters[t])
            need = [p for p in ("K", "DEF") if counts[t][p] == 0]
            for i, (pid, pos) in enumerate(avail):
                if counts[t][pos] >= DRAFT_LIMITS[pos]:
                    continue
                if pos not in need and left <= len(need):
                    continue  # save the last picks for K/DEF
                rosters[t].append(pid)
                counts[t][pos] += 1
                avail.pop(i)
                break
    return rosters


@dataclass
class SeasonData:
    season: int
    pw: pl.DataFrame
    table: pl.DataFrame  # sleeper_id, position, team, injury_status
    games: pl.DataFrame
    proj: dict[tuple[str, int], float]  # (id, week) → Sleeper projection
    actual: dict[tuple[str, int], float]
    tables: Tables


def season_data(
    src: Any, scoring: Mapping[str, float], pw: pl.DataFrame, season: int
) -> SeasonData:
    s = pw.filter(pl.col("season") == season)
    team = (
        s.filter(pl.col("team").is_not_null())
        .group_by("sleeper_id", "position", "team")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .unique("sleeper_id", keep="first")
        .select("sleeper_id", "position", "team", pl.lit(None, pl.String).alias("injury_status"))
    )
    games = projections.team_games(src.nflverse.schedules([season]), season)
    prior_pw = pw.filter(pl.col("season") < season)
    edges, resid_q, floor, scored = history.residual_tables(prior_pw)
    same, opp, _ = history.correlation_tables(scored, prior_pw)
    hazard, pmf = history.injury_tables(prior_pw)
    tables = Tables(edges, resid_q, floor, same, opp, hazard, pmf, source=f"sleeper < {season}")
    proj = {(a, b): c for a, b, c in s.select("sleeper_id", "week", "proj").iter_rows()}
    actual = {
        (a, b): c
        for a, b, c in s.filter(pl.col("played")).select("sleeper_id", "week", "actual").iter_rows()
    }
    return SeasonData(season, s, team, games, proj, actual, tables)


def real_outcome(
    sd: SeasonData, league: League, rosters: list[list[str]]
) -> tuple[np.ndarray, Any]:
    """Real starter points per team-week (lineups set on that week's projection) and the
    real final standings/bracket."""
    weeks = list(range(1, 18))
    universe = sorted({p for r in rosters for p in r})
    index = {p: i for i, p in enumerate(universe)}
    pos = np.array([POSITIONS.index(sd.table.filter(pl.col("sleeper_id") == p)["position"][0])
                    if p in set(sd.table["sleeper_id"]) else 2 for p in universe])  # fmt: skip
    exp = np.array([[sd.proj.get((p, w), 0.0) for w in weeks] for p in universe], np.float32)
    pts = np.array([[sd.actual.get((p, w), 0.0) for w in weeks] for p in universe], np.float32)
    out = Outcomes(pts.T[None], (exp.T == 0)[None], exp)
    inp = SimInputs(universe, pos, weeks, exp, (exp > 0).astype(float),
                    np.full(exp.shape, -1), np.full(exp.shape, -1), np.full(exp.shape, -1))  # fmt: skip
    cfg = Config([np.array([index[p] for p in r], np.intp) for r in rosters], np.array([], np.intp))
    tp = team_points(out, inp, cfg, league.slots)
    res = simulate_league(out, inp, league, cfg, tp)
    return tp[0], res


def forecast_inputs(sd: SeasonData, league: League, n: int) -> Any:
    """Projection frame for weeks n..17 using only what was known before week n: each
    player's latest positive Sleeper projection (at or before week n) for every week his
    team plays; week n uses week n's own projection (it knows who's ruled out)."""
    last: dict[str, float] = {}
    for (p, w), v in sorted(sd.proj.items(), key=lambda kv: kv[0][1]):
        if w <= n and v > 0:
            last[p] = v
    playing = {(t, w) for t, w in sd.games.select("team", "week").iter_rows()}
    team_of = dict(sd.table.select("sleeper_id", "team").iter_rows())
    rows = []
    for p, v in last.items():
        for w in range(n, 18):
            if (team_of.get(p), w) not in playing:
                continue
            m = sd.proj.get((p, n), 0.0) if w == n else v
            rows.append((sd.season, w, p, float(m), 1))
    pts = pl.DataFrame(rows, schema={"season": pl.Int64, "week": pl.Int64, "sleeper_id": pl.String,
                                     "mean": pl.Float64, "n_sources": pl.Int64}, orient="row")  # fmt: skip
    return projections.Projections(pts, pl.DataFrame(), sd.games)


@dataclass
class Calibration:
    preds: pl.DataFrame  # season, league, snapshot, team, p, made
    trades: pl.DataFrame  # season, league, snapshot, predicted_gain_a, realized_gain_a


def calibrate(src: Any, pw: pl.DataFrame, leagues: int, progress: Any = print) -> Calibration:
    rng = np.random.default_rng(SEED)
    preds, trades = [], []
    for season in EVAL_SEASONS:
        sd = season_data(src, STANDARD_PPR, pw, season)
        w1 = sd.pw.filter(pl.col("week") == 1).select("sleeper_id", pl.col("proj").alias("w1"))
        pool = w1.join(sd.table, on="sleeper_id").filter(pl.col("w1") > 0)
        for k in range(leagues):
            rosters = draft(pool, rng)
            sched = round_robin(N_TEAMS, range(1, 15), rng)
            teams = [Team(i + 1, None, f"T{i + 1}", list(r)) for i, r in enumerate(rosters)]
            base = League(f"syn{season}-{k}", "synthetic", season, N_TEAMS, SLOTS, BENCH, 0, {}, 6, 15, 0,
                          False, 0, 1, teams, sched, {})  # fmt: skip
            tp_real, real = real_outcome(sd, base, rosters)
            made = real.made_playoffs[0]
            for n in SNAPSHOTS:
                lg = _snapshot_league(base, tp_real, n)
                proj = forecast_inputs(sd, lg, n)
                inp, cfg, index = engine.build_inputs(lg, sd.table, proj, sd.tables)
                out = simulate_players(inp, sd.tables, SIMS, SEED + k)
                res = simulate_league(out, inp, lg, cfg)
                for t in range(N_TEAMS):
                    preds.append((season, k, n, t, float(res.playoff_prob[t]), bool(made[t])))
                trades += _replay_trades(rng, sd, lg, inp, cfg, index, out, rosters, n, season, k)
            progress(f"calibration {season} league {k + 1}/{leagues}")
    p = pl.DataFrame(preds, schema={"season": pl.Int64, "league": pl.Int64, "snapshot": pl.Int64,
                                    "team": pl.Int64, "p": pl.Float64, "made": pl.Boolean}, orient="row")  # fmt: skip
    tr = pl.DataFrame(trades, schema={"season": pl.Int64, "league": pl.Int64, "snapshot": pl.Int64,
                                      "predicted": pl.Float64, "realized": pl.Float64}, orient="row")  # fmt: skip
    return Calibration(p, tr)


def _snapshot_league(base: League, tp_real: np.ndarray, n: int) -> League:
    """The league as it stood before week n (real results for weeks 1..n-1)."""
    teams = []
    for i, t in enumerate(base.teams):
        w = l_ = 0
        pf = 0.0
        for wk in range(1, n):
            for a, b in base.schedule[wk]:
                if t.roster_id not in (a, b):
                    continue
                me, them = tp_real[i, wk - 1], tp_real[(b if a == t.roster_id else a) - 1, wk - 1]
                w += me > them
                l_ += me < them
                pf += float(me)
        teams.append(
            Team(
                t.roster_id,
                None,
                t.name,
                list(t.players),
                wins=int(w),
                losses=int(l_),
                points_for=pf,
            )
        )
    return League(base.league_id, base.name, base.season, base.num_teams, base.slots, base.bench_size, 0, {},
                  base.playoff_teams, base.playoff_week_start, 0, False, 0, n, teams, base.schedule, {})  # fmt: skip


def _replay_trades(rng: np.random.Generator, sd: SeasonData, lg: League, inp: SimInputs, cfg: Config,
                   index: dict[str, int], out: Outcomes, rosters: list[list[str]], n: int, season: int,
                   k: int) -> list[tuple]:  # fmt: skip
    """Random 1-for-1 trades: model's predicted Δ starter points for team A vs what the
    real games gave (lineups set on each week's real projection)."""
    res = []
    regular = [i for i, w in enumerate(inp.weeks) if w <= lg.last_regular_week]
    for _ in range(TRADES_PER_SNAPSHOT):
        a, b = rng.choice(N_TEAMS, 2, replace=False)
        pa = rng.choice([p for p in rosters[a] if p in index])
        same = [p for p in rosters[b] if p in index and inp.pos[index[p]] == inp.pos[index[pa]]]
        if not same:
            continue
        pb = rng.choice(same)
        new = [r.copy() for r in cfg.rosters]
        new[a] = np.array([i for i in new[a] if i != index[pa]] + [index[pb]], np.intp)
        new[b] = np.array([i for i in new[b] if i != index[pb]] + [index[pa]], np.intp)
        tb = team_points(out, inp, Config([cfg.rosters[a]], cfg.fa_pool), lg.slots)[:, 0][
            :, regular
        ]
        ta = team_points(out, inp, Config([new[a]], cfg.fa_pool), lg.slots)[:, 0][:, regular]
        predicted = float((ta - tb).sum(1).mean())
        realized = _real_gain(sd, lg, [rosters[a], [*[p for p in rosters[a] if p != pa], pb]], n)
        res.append((season, k, n, predicted, realized))
    return res


def _real_gain(sd: SeasonData, lg: League, two: list[list[str]], n: int) -> float:
    """Real starter points (weeks n..end of regular season) of roster 2 minus roster 1."""
    totals = []
    for roster in two:
        pos = [
            POSITIONS.index(x)
            for x in sd.table.filter(pl.col("sleeper_id").is_in(roster)).sort("sleeper_id")[
                "position"
            ]
        ]
        ids_ = sorted(set(roster) & set(sd.table["sleeper_id"]))
        tot = 0.0
        for w in range(n, lg.last_regular_week + 1):
            e = [sd.proj.get((p, w), 0.0) for p in ids_]
            _, picks = optimal_lineup(e, pos, lg.slots)
            tot += sum(sd.actual.get((ids_[i], w), 0.0) for i, _ in picks)
        totals.append(tot)
    return totals[1] - totals[0]


# ----------------------------------------------------------------- report


def brier(df: pl.DataFrame) -> float:
    return float(((df["p"] - df["made"].cast(pl.Float64)) ** 2).mean())  # type: ignore[arg-type]


def reliability(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns((pl.col("p") * 10).floor().clip(0, 9).cast(pl.Int64).alias("bin"))
        .group_by("bin")
        .agg(
            pl.col("p").mean().alias("predicted"),
            pl.col("made").cast(pl.Float64).mean().alias("actual"),
            pl.len().alias("n"),
        )
        .sort("bin")
    )


def _md(df: pl.DataFrame, fmt: Mapping[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = [
            format(v, fmt.get(c, "")) if isinstance(v, float) else str(v)
            for c, v in zip(cols, row, strict=True)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    acc: Accuracy, cal: Calibration, league: League | None, path: str = "REPORT.md"
) -> str:
    p = cal.preds
    climat = p.with_columns(pl.lit(6 / N_TEAMS).alias("p"))
    b, b0 = brier(p), brier(climat)
    tr = cal.trades.filter(pl.col("predicted").abs() > 0)
    hit = (
        float(((tr["predicted"] > 0) == (tr["realized"] > 0)).to_numpy().mean())
        if tr.height
        else float("nan")
    )
    se = math.sqrt(hit * (1 - hit) / tr.height) if tr.height else float("nan")
    big = tr.filter(pl.col("predicted").abs() >= 10)
    hit_big = (
        float(((big["predicted"] > 0) == (big["realized"] > 0)).to_numpy().mean())
        if big.height
        else float("nan")
    )
    wide = acc.table.pivot(on="source", index="position", values="mae")
    n = acc.table.group_by("position").agg(pl.col("n").first())
    wide = wide.join(n, on="position").sort(
        pl.col("position").replace_strict({q: i for i, q in enumerate(POSITIONS)})
    )
    lines = [
        "# REPORT — accuracy and calibration",
        "",
        f"Generated by `uv run python -m ffcalc.backtest` on {time.strftime('%Y-%m-%d %H:%M')}. "
        "Every number below is computed from real past NFL seasons by that script; none are typed in.",
        "",
        "## 1. Projection accuracy (mean absolute error, fantasy points per player-week)",
        "",
        f"Seasons {', '.join(map(str, EVAL_SEASONS))}, weeks 1–18, fantasy-relevant player-weeks that played, "
        "one common sample per position (rows where every listed source had a projection). "
        "Scored with standard full-PPR settings. Lower is better. "
        "`blend` uses inverse-MAE weights learned only from earlier seasons (equal weights in the first season). "
        "`season_avg` / `last3` are naive baselines: the player's average actual points so far this season / over his last 3 games "
        "(so a player's first game of each season is excluded from the common sample). "
        "ESPN isn't used for K or DEF (its stat ids for those are unverified); the in-house model covers QB/RB/WR/TE.",
        "",
        _md(wide, dict.fromkeys(wide.columns, ".2f")),
        "",
        "Blend weights written to `data/cache/blend_weights.json` (inverse MAE, all seasons above):",
        "",
        "```json",
        json.dumps(acc.weights, indent=1),
        "```",
        "",
        "## 2. Distribution calibration",
        "",
        "Share of actual scores that fell below the predicted 10th / 50th / 90th percentile "
        "(tables learned only from earlier seasons). Perfect calibration is 0.10 / 0.50 / 0.90.",
        "",
        _md(acc.coverage, {"below_q10": ".3f", "below_q50": ".3f", "below_q90": ".3f"}),
        "",
        "## 3. Simulator calibration (playoff odds)",
        "",
    ]
    if league is not None and league.previous_league_id:
        lines.append(
            f"League history exists (previous_league_id {league.previous_league_id}) but replaying it is not built; synthetic leagues used."
        )
    else:
        lines.append(
            "**The test league has no earlier seasons on Sleeper, so this uses the synthetic-league fallback** (spec §11.3): "
            f"{p.select(pl.struct('season', 'league').n_unique()).item() // len(EVAL_SEASONS)} leagues per season × {len(EVAL_SEASONS)} seasons, {N_TEAMS} teams, "
            "drafted by real week-1 projections with noise, standard lineup, 6-team playoffs weeks 15–17, rosters fixed all season (no waivers/trades). "
            "Lineups are set on each week's real Sleeper projection and scored with real actual points. "
            f"At weeks {', '.join(map(str, SNAPSHOTS))} the simulator ({SIMS:,} sims) predicts each team's playoff odds using only "
            "what was known then (each player's latest projection; tables from earlier seasons only)."
        )
    lines += [
        "",
        f"- Team-snapshots scored: **{p.height:,}**",
        f"- Brier score: **{b:.4f}** (lower is better; 0 is perfect)",
        f"- Brier score of the naive constant forecast (every team {6 / N_TEAMS:.0%}): {b0:.4f}",
        f"- Brier skill score vs naive: **{1 - b / b0:.3f}** (1 = perfect, 0 = no better than naive)",
        "",
        "By snapshot week:",
        "",
        _md(
            p.group_by("snapshot")
            .agg(
                pl.len().alias("n"),
                ((pl.col("p") - pl.col("made").cast(pl.Float64)) ** 2).mean().alias("brier"),
            )
            .sort("snapshot"),
            {"brier": ".4f"},
        ),
        "",
        "Reliability (when the model says X%, how often did it happen?):",
        "",
        _md(reliability(p), {"predicted": ".3f", "actual": ".3f"}),
        "",
        "## 4. Trade replay",
        "",
        "No real trade history exists for this league, so trades are replayed inside the synthetic leagues: "
        f"{TRADES_PER_SNAPSHOT} random same-position 1-for-1 trades per snapshot. The model predicts team A's change in "
        "rest-of-regular-season starter points; we check whether team A really gained points (real lineups, real scores).",
        "",
        f"- Trades replayed: **{tr.height:,}**",
        f"- Model picked the side that really gained points: **{hit:.1%}** (± {se:.1%} SE; 50% = coin flip)",
        f"- When the model predicted a gap of at least 10 points: **{hit_big:.1%}** of {big.height:,} trades",
        f"- Correlation of predicted vs realized gain: {float(np.corrcoef(tr['predicted'], tr['realized'])[0, 1]) if tr.height > 2 else float('nan'):.3f}",
        "",
        "## What this does not show",
        "",
        "- Real leagues have waivers, trades and managers who don't always start their best lineup; the synthetic leagues don't.",
        "- FantasyCalc market values have no history, so the market check and acceptance heuristic are not backtested.",
        "- ESPN K/DEF and the FantasyPros CSV are not part of the accuracy comparison.",
    ]
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ffcalc.backtest")
    ap.add_argument("--league", default="1400221038771437568", help="league whose history to check")
    ap.add_argument(
        "--leagues", type=int, default=LEAGUES_PER_SEASON, help="synthetic leagues per season"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    from .league import load_league
    from .sources import live

    src = live()
    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", file=sys.stderr)  # noqa: E731
    try:
        league = load_league(args.league, src.sleeper)
    except Exception:
        league = None
    say("player table")
    table = ids.player_table(src.sleeper.players(), src.nflverse.ff_playerids())
    say("projection accuracy")
    acc, pw = accuracy(src, STANDARD_PPR, table)
    projections.weights_path().parent.mkdir(parents=True, exist_ok=True)
    projections.weights_path().write_text(json.dumps(acc.weights))
    say("simulator calibration")
    cal = calibrate(src, pw, args.leagues, say)
    write_report(acc, cal, league)
    say("wrote REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
