"""Variance, correlation and injury tables from real past seasons (spec §4, §6).

Source: Sleeper's archived weekly projections vs actual stat lines (same ids, same stat
keys), scored with the league's own settings. nflverse injury reports give the
chance a player with a given game status suits up.

`uv run python -m ffcalc history` builds and caches the tables; the app builds them on
first use. Tests use small real fixtures.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl
from scipy.special import ndtri

from . import cache
from .clients.sleeper import stat_frame
from .models import POSITIONS, ROLE_DEPTH, ROLES
from .scoring import STANDARD_PPR, points, score_lines
from .simulate import NPOS, NROLES, QGRID, Tables

log = logging.getLogger(__name__)

HIST_SEASONS = (2022, 2023, 2024, 2025)
TABLES_VERSION = 3  # bump when table-building logic changes, so old caches are ignored
WEEKS = range(1, 19)
TIERS = 5
MAX_OUT = 8  # longest absence tracked; longer ones are counted as 8
SHRINK = 150  # correlation shrinkage toward 0: n / (n + SHRINK)
SNAPSHOTS = range(3, 15)  # weeks a season-long forecast is frozen at, for `persistence`
STATUSES = ("Questionable", "Doubtful", "Out")
# Projected points (league scoring) below which a player-week isn't fantasy-relevant.
MIN_PROJ = {"QB": 8.0, "RB": 4.0, "WR": 4.0, "TE": 3.0, "K": 4.0, "DEF": 3.0}

_META = {
    "sleeper_id": pl.String,
    "position": pl.String,
    "team": pl.String,
    "opponent": pl.String,
    "game_id": pl.String,
    "played": pl.Boolean,
}


def _meta(rows: list[dict], kind: str) -> pl.DataFrame:
    out = []
    for r in rows:
        pos = (r.get("player") or {}).get("position")
        if pos not in POSITIONS:
            continue
        st = r.get("stats") or {}
        if kind == "proj" and points(st, STANDARD_PPR) < 0.5:  # relevance gate only
            continue
        played = bool(st.get("gp")) or (pos == "DEF" and r.get("game_id") is not None)
        if kind == "stats" and not played:
            continue
        out.append(
            (str(r["player_id"]), pos, r.get("team"), r.get("opponent"), r.get("game_id"), played)
        )
    return pl.DataFrame(out, schema=_META, orient="row")


def week_frames(
    sleeper: Any, season: int, week: int, kind: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(stat lines, meta) for one past week, cached compactly forever."""
    key = f"history/sleeper_{kind}/{season}_{week}"
    holder: dict[str, pl.DataFrame] = {}

    def rows() -> list[dict]:
        if "rows" not in holder:
            fetch = sleeper.projections if kind == "proj" else sleeper.stats
            holder["rows"] = fetch(season, week)  # type: ignore[assignment]
        return holder["rows"]  # type: ignore[return-value]

    meta = cache.cached_frame(key + "_meta", cache.TTL["history"], lambda: _meta(rows(), kind))
    keep = set(meta["sleeper_id"].to_list())
    lines = cache.cached_frame(
        key + "_lines",
        cache.TTL["history"],
        lambda: (
            stat_frame([r for r in rows() if str(r["player_id"]) in keep], kind)
            .drop("source")
            .with_columns(pl.col("value").cast(pl.Float32))
        ),
    )
    return lines, meta


def player_weeks(
    sleeper: Any, scoring: Mapping[str, float], seasons: Iterable[int], weeks: Iterable[int] = WEEKS
) -> pl.DataFrame:
    """One row per player-week that was projected OR played: season, week, sleeper_id,
    position, team, opponent, game_id, proj (0 if unprojected), actual (0 if he didn't
    play), played, team_played."""
    frames = []
    for s in seasons:
        for w in weeks:
            try:
                pl_, pm = week_frames(sleeper, s, w, "proj")
                sl, sm = week_frames(sleeper, s, w, "stats")
            except Exception as e:  # a missing week shouldn't sink the whole build
                log.warning("history %s wk %s unavailable: %s", s, w, e)
                continue
            if pm.is_empty() or sm.is_empty():
                continue
            proj = score_lines(pl_.select("sleeper_id", "stat", "value"), scoring)
            act = score_lines(sl.select("sleeper_id", "stat", "value"), scoring)
            teams_played = sm["team"].drop_nulls().unique()
            keys = ["sleeper_id", "position", "team", "opponent", "game_id"]
            f = (
                pm.select(keys)
                .join(
                    sm.select(*keys, "played"),
                    on="sleeper_id",
                    how="full",
                    coalesce=True,
                    suffix="_s",
                )  # fmt: skip
                .with_columns(
                    *[pl.col(k).fill_null(pl.col(f"{k}_s")) for k in keys[1:]],
                    pl.col("played").fill_null(False),
                )
                .drop([f"{k}_s" for k in keys[1:]])
                .join(proj.rename({"points": "proj"}), on="sleeper_id", how="left")
                .join(act.rename({"points": "actual"}), on="sleeper_id", how="left")
                .with_columns(
                    pl.col("proj").fill_null(0.0),
                    pl.col("actual").fill_null(0.0),
                    pl.col("team").is_in(teams_played.implode()).alias("team_played"),
                    pl.lit(s).alias("season"),
                    pl.lit(w).alias("week"),
                )
            )
            frames.append(f)
    if not frames:
        raise RuntimeError("no historical weeks available")
    # fixed row order: rank ties and anything built from this table are reproducible
    return pl.concat(frames, how="vertical_relaxed").sort("season", "week", "sleeper_id")


def _relevant(pw: pl.DataFrame) -> pl.DataFrame:
    floor = pl.col("position").replace_strict(MIN_PROJ, default=99.0, return_dtype=pl.Float64)
    return pw.filter(pl.col("proj") >= floor)


def _ugrid() -> np.ndarray:
    return np.clip(np.linspace(0, 1, QGRID), 0.002, 0.998)


def residual_tables(pw: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    """tier edges (NPOS, T-1), residual quantiles (NPOS, T, QGRID), floors, and the played
    relevant rows with their bucket and PIT z-score (for correlations)."""
    rel = _relevant(pw).filter(pl.col("played"))
    edges = np.zeros((NPOS, TIERS - 1))
    resid_q = np.zeros((NPOS, TIERS, QGRID))
    floor = np.zeros(NPOS)
    parts = []
    for p, name in enumerate(POSITIONS):
        d = rel.filter(pl.col("position") == name)
        if d.height < 50:
            # shortcut: a position with < 50 rows gets a wide normal; only hit on tiny fixtures
            log.warning("only %d %s rows; using a wide normal", d.height, name)
            resid_q[p] = 6.0 * ndtri(_ugrid())[None, :]
            continue
        proj = d["proj"].to_numpy()
        edges[p] = np.quantile(proj, np.linspace(0, 1, TIERS + 1)[1:-1])
        tier = np.searchsorted(edges[p], proj, side="right")
        resid = d["actual"].to_numpy() - proj
        floor[p] = np.quantile(d["actual"].to_numpy(), 0.002)
        z = np.zeros(len(resid))
        for t in range(TIERS):
            m = tier == t
            resid_q[p, t] = np.quantile(resid[m], _ugrid())
            ranks = resid[m].argsort().argsort()
            z[m] = ndtri((ranks + 0.5) / m.sum())
        parts.append(d.with_columns(pl.Series("z", z)))
    return edges, resid_q, floor, pl.concat(parts) if parts else rel.with_columns(z=pl.lit(0.0))


def persistence(scored: pl.DataFrame) -> np.ndarray:
    """(NPOS,) how much of a player's miss carries over from week to week, as a correlation
    of normal scores. The simulator holds today's projection fixed for every later week, so
    this is measured the same way: residuals of later weeks against the projection known at
    a snapshot week. A player who turns out better (a new role, a breakout) beats a frozen
    projection week after week; against each week's own projection it barely shows."""
    rel = scored.select("season", "week", "sleeper_id", "position", "proj", "actual")
    parts = []
    for n in SNAPSHOTS:
        known = (
            rel.filter(pl.col("week") <= n)
            .sort("week")
            .group_by("season", "sleeper_id")
            .agg(pl.col("proj").last().alias("p0"))
        )
        parts.append(
            rel.filter(pl.col("week") > n)
            .join(known, on=["season", "sleeper_id"])
            .with_columns((pl.col("actual") - pl.col("p0")).alias("r"))
            .select("season", "sleeper_id", "position", "r")
            .with_columns(pl.lit(n).alias("snap"))
        )
    frozen = pl.concat(parts)
    out = np.zeros(NPOS)
    for p, name in enumerate(POSITIONS):
        d = frozen.filter(pl.col("position") == name)
        if d.height < 50:
            continue  # shortcut: no evidence → weeks stay independent; only hit on tiny fixtures
        r = d["r"].to_numpy()
        z = ndtri((r.argsort().argsort() + 0.5) / len(r))
        g = (
            d.with_columns(pl.Series("z", z))
            .group_by("season", "sleeper_id", "snap")
            .agg(
                pl.col("z").sum().alias("s"),
                (pl.col("z") ** 2).sum().alias("q"),
                pl.len().alias("k"),
            )
        )
        pairs = float((g["k"] * (g["k"] - 1)).sum())
        if pairs:
            out[p] = float((g["s"] ** 2 - g["q"]).sum()) / pairs  # mean product over week pairs
    return np.clip(out, 0.0, 0.5)


def assign_roles(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Role index (-1 if none) by projection rank at position within team and `by` keys."""
    depth = pl.col("position").replace_strict(ROLE_DEPTH, return_dtype=pl.Int64)
    rank = pl.col("proj").rank("ordinal", descending=True).over([*by, "team", "position"])
    role_name = (
        pl.when(pl.col("position").is_in(["QB", "TE"]))
        .then(pl.col("position") + "1")
        .when(pl.col("position").is_in(["K", "DEF"]))
        .then(pl.col("position"))
        .otherwise(pl.col("position") + rank.cast(pl.String))
    )
    index = {r: i for i, r in enumerate(ROLES)}
    return df.with_columns(
        pl.when(rank <= depth)
        .then(role_name.replace_strict(index, default=-1, return_dtype=pl.Int64))
        .otherwise(-1)
        .alias("role")
    )


def correlation_tables(
    scored: pl.DataFrame, pw: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray, int]:
    """Role-pair residual correlations within NFL games (same team / opponents)."""
    roles = assign_roles(pw.filter(pl.col("team").is_not_null()), ["season", "week"]).select(
        "season", "week", "sleeper_id", "role"
    )
    d = (
        scored.join(roles, on=["season", "week", "sleeper_id"])
        .filter((pl.col("role") >= 0) & pl.col("game_id").is_not_null())
        .select("season", "week", "game_id", "team", "role", "z", "sleeper_id")
    )
    pairs = d.join(d, on=["season", "week", "game_id"], suffix="_b").filter(
        pl.col("sleeper_id") < pl.col("sleeper_id_b")
    )
    same = np.zeros((NROLES, NROLES))
    opp = np.zeros((NROLES, NROLES))
    for tgt, cond in (
        (same, pl.col("team") == pl.col("team_b")),
        (opp, pl.col("team") != pl.col("team_b")),
    ):
        agg = (
            pairs.filter(cond)
            .group_by("role", "role_b")
            .agg((pl.col("z") * pl.col("z_b")).sum().alias("s"), pl.len().alias("n"))
        )
        s = np.zeros((NROLES, NROLES))
        n = np.zeros((NROLES, NROLES))
        for a, b, sv, nv in agg.iter_rows():
            s[a, b] += sv
            n[a, b] += nv
            if a != b:
                s[b, a] += sv
                n[b, a] += nv
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(n > 0, s / n, 0.0) * n / (n + SHRINK)
        tgt[:] = np.clip(r, -0.9, 0.9)
    np.fill_diagonal(same, 0.0)
    return same, opp, pairs.height


def relevant_seasons(pw: pl.DataFrame) -> pl.DataFrame:
    """(sleeper_id, season, team) for players fantasy-relevant at some point that season;
    team = the team he played most for."""
    return (
        _relevant(pw)
        .select("sleeper_id", "season")
        .unique()
        .join(
            pw.filter(pl.col("team").is_not_null())
            .group_by("sleeper_id", "season", "team")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .unique(["sleeper_id", "season"], keep="first"),
            on=["sleeper_id", "season"],
        )
        .select("sleeper_id", "season", "team")
    )


def injury_tables(pw: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per position: chance a relevant player who played misses his team's next game, and
    how many team games he then misses. Weeks are his team's real game weeks, so a bye is
    never counted as an absence."""
    hazard = np.zeros(NPOS)
    pmf = np.zeros((NPOS, MAX_OUT))
    team_weeks: dict[tuple[int, str], list[int]] = {}
    for season, team, week in (
        pw.filter(pl.col("team_played"))
        .select("season", "team", "week")
        .unique()
        .sort("week")
        .iter_rows()
    ):
        team_weeks.setdefault((season, team), []).append(week)
    played = {
        (sid, season): set(g["week"].to_list())
        for (sid, season), g in pw.filter(pl.col("played")).group_by("sleeper_id", "season")
    }
    pos_of = dict(pw.select("sleeper_id", "position").unique("sleeper_id").iter_rows())
    proj_of = {
        (a, b, c): d for a, b, c, d in pw.select("sleeper_id", "season", "week", "proj").iter_rows()
    }
    events = np.zeros(NPOS)
    opps = np.zeros(NPOS)
    counts = np.zeros((NPOS, MAX_OUT))
    for sid, season, team in relevant_seasons(pw).iter_rows():
        p = POSITIONS.index(pos_of[sid])
        weeks = team_weeks.get((season, team), [])
        mine = played.get((sid, season), set())
        for i in range(len(weeks) - 1):
            w = weeks[i]
            if w not in mine or proj_of.get((sid, season, w), 0.0) < MIN_PROJ[pos_of[sid]]:
                continue
            opps[p] += 1
            if weeks[i + 1] in mine:
                continue
            events[p] += 1
            j = i + 1
            while j < len(weeks) and weeks[j] not in mine:
                j += 1
            counts[p, min(j - i - 1, MAX_OUT) - 1] += 1
    for p in range(NPOS):
        hazard[p] = events[p] / opps[p] if opps[p] else 0.0
        pmf[p] = counts[p] / counts[p].sum() if counts[p].sum() else np.eye(MAX_OUT)[0]
    return hazard, pmf


def status_tables(
    pw: pl.DataFrame, injuries: pl.DataFrame, table: pl.DataFrame
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """P(plays | game status, position) for fantasy-relevant players whose team played,
    and his mean multiplier given he plays (actual/projection vs healthy players)."""
    play: dict[str, list[float]] = {}
    mult: dict[str, list[float]] = {}
    ids = table.select("gsis_id", "sleeper_id").drop_nulls().unique("gsis_id")
    inj = (
        injuries.filter(pl.col("report_status").is_in(STATUSES))
        .select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            "gsis_id",
            "report_status",
        )
        .join(ids, on="gsis_id")
        .unique(["season", "week", "sleeper_id"])
    )
    rel = relevant_seasons(pw)
    games = pw.filter(pl.col("team_played")).select("season", "week", "team").unique()
    pos_of = pw.select("sleeper_id", "position").unique("sleeper_id")
    # every (relevant player, team game) with his status, whether or not he was projected;
    # only weeks after he played his team's previous game as a fantasy-relevant regular
    floor = pl.col("position").replace_strict(MIN_PROJ, default=99.0, return_dtype=pl.Float64)
    by = ["sleeper_id", "season"]
    grid = (
        rel.join(games, on=["season", "team"])
        .join(pos_of, on="sleeper_id")
        .join(
            pw.select("season", "week", "sleeper_id", "played", "proj", "actual"),
            on=["season", "week", "sleeper_id"],
            how="left",
        )  # fmt: skip
        .with_columns(pl.col("played").fill_null(False), pl.col("proj").fill_null(0.0))
        .sort("sleeper_id", "season", "week")
        .with_columns(
            (pl.col("played").shift(1).over(by) & (pl.col("proj").shift(1).over(by) >= floor))
            .fill_null(False)
            .alias("regular")
        )
        .filter(pl.col("regular"))
        .join(inj, on=["season", "week", "sleeper_id"], how="left")
    )
    base = grid.filter(pl.col("report_status").is_null() & pl.col("played") & (pl.col("proj") > 0))
    for status in STATUSES:
        play[status], mult[status] = [], []
        pooled = grid.filter(pl.col("report_status") == status)
        for name in POSITIONS:
            g = pooled.filter(pl.col("position") == name)
            # shortcut: positions are pooled when a cell has < 30 rows (K/DEF statuses are rare)
            small = g.height < 30
            src = pooled if small else g
            play[status].append(float(src["played"].to_numpy().mean()) if src.height else 0.0)
            gp = src.filter(pl.col("played") & (pl.col("proj") > 0))
            b = base if small else base.filter(pl.col("position") == name)
            gp_proj, b_proj = gp["proj"].to_numpy().sum(), b["proj"].to_numpy().sum()
            if gp.height and b.height and gp_proj > 0 and b_proj > 0:
                ratio = gp["actual"].to_numpy().sum() / gp_proj
                ratio_base = b["actual"].to_numpy().sum() / b_proj
                mult[status].append(float(ratio / ratio_base))
            else:
                mult[status].append(1.0)
    return play, mult


def build_tables(
    sleeper: Any,
    scoring: Mapping[str, float],
    seasons: Iterable[int] = HIST_SEASONS,
    injuries: pl.DataFrame | None = None,
    table: pl.DataFrame | None = None,
    weeks: Iterable[int] = WEEKS,
) -> tuple[Tables, dict[str, Any]]:
    """Tables plus a summary (sample sizes) for NOTES/REPORT."""
    seasons = list(seasons)
    pw = player_weeks(sleeper, scoring, seasons, weeks)
    edges, resid_q, floor, scored = residual_tables(pw)
    same, opp, n_pairs = correlation_tables(scored, pw)
    hazard, pmf = injury_tables(pw)
    tables = Tables(edges, resid_q, floor, same, opp, hazard, pmf, source=f"sleeper {seasons}")
    tables.persist = persistence(scored)
    if injuries is not None and table is not None and not injuries.is_empty():
        tables.status_play, tables.status_mult = status_tables(pw, injuries, table)
    summary = {
        "seasons": seasons,
        "player_weeks": pw.height,
        "relevant_played": scored.height,
        "game_pairs": n_pairs,
        "hazard": dict(zip(POSITIONS, np.round(hazard, 4).tolist(), strict=True)),
        "persist": dict(zip(POSITIONS, np.round(tables.persist, 3).tolist(), strict=True)),
        "status_play": tables.status_play,
    }
    return tables, summary


def _key(scoring: Mapping[str, float], seasons: Iterable[int]) -> str:
    blob = json.dumps([TABLES_VERSION, sorted(scoring.items()), list(seasons)]).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def save_tables(tables: Tables, scoring: Mapping[str, float], seasons: Iterable[int]) -> None:
    p = cache.data_dir() / "cache" / f"tables_{_key(scoring, seasons)}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(tables).items()}
    p.write_text(json.dumps(d))


def load_tables(
    scoring: Mapping[str, float], seasons: Iterable[int] = HIST_SEASONS
) -> Tables | None:
    p = cache.data_dir() / "cache" / f"tables_{_key(scoring, seasons)}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    arrays = {k: np.array(v) for k, v in d.items() if isinstance(v, list)}
    return Tables(**{**d, **arrays})
