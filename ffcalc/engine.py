"""Load a league and everything the simulator needs, once."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from . import history, ids, projections
from .league import load_league
from .models import POSITIONS, SLOT_ELIGIBILITY, League
from .simulate import Config, SimInputs, Tables, placeholder_tables, playoff_rounds

log = logging.getLogger(__name__)

FA_PER_POS = 8
IR_WEEKS = 4  # shortcut: IR/PUP players assumed out 4 more weeks; Sleeper gives no return date
OUT_STATUSES = {"Out", "IR", "PUP", "Sus", "NA", "DNR", "COV"}


@dataclass
class Context:
    league: League
    table: pl.DataFrame  # player table (ids.player_table)
    proj: projections.Projections
    tables: Tables
    inputs: SimInputs
    base: Config
    index: dict[str, int]  # sleeper_id → universe index
    warnings: list[str] = field(default_factory=list)
    market: pl.DataFrame = field(default_factory=pl.DataFrame)  # FantasyCalc values (may be empty)
    src: Any = None

    def ids_of(self, idx: np.ndarray) -> list[str]:
        return [self.inputs.ids[i] for i in idx]

    def name(self, sleeper_id: str) -> str:
        row = self.table.filter(pl.col("sleeper_id") == sleeper_id)
        return row["name"][0] if row.height else sleeper_id

    def team_index(self, roster_id: int) -> int:
        return [t.roster_id for t in self.league.teams].index(roster_id)


def sim_weeks(league: League) -> list[int]:
    last = playoff_rounds(league)[-1][-1]
    return list(range(league.current_week, last + 1))


def get_tables(
    src: Any, league: League, table: pl.DataFrame, build: bool, warn: list[str]
) -> Tables:
    t = history.load_tables(league.scoring)
    if t is not None:
        return t
    if build:
        try:
            inj = src.nflverse.injuries(list(history.HIST_SEASONS)) if src.nflverse else None
            t, _ = history.build_tables(src.sleeper, league.scoring, injuries=inj, table=table)
            history.save_tables(t, league.scoring, history.HIST_SEASONS)
            return t
        except Exception as e:
            log.warning("history tables unavailable: %s", e)
    warn.append("Score spreads and injury rates are rough defaults (history not built yet).")
    return placeholder_tables()


def build_inputs(
    league: League, table: pl.DataFrame, proj: projections.Projections, tables: Tables,
    fa_per_pos: int = FA_PER_POS,
) -> tuple[SimInputs, Config, dict[str, int]]:  # fmt: skip
    weeks = sim_weeks(league)
    pts = proj.points.filter(pl.col("week").is_in(weeks))
    # only positions we can score and start (IDP and unknown players are left out)
    known = set(table.filter(pl.col("position").is_in(list(POSITIONS)))["sleeper_id"].to_list())
    rostered = {p for t in league.teams for p in t.players if p in known}
    ros = (
        pts.filter(pl.col("week") <= league.last_regular_week)
        .group_by("sleeper_id")
        .agg(pl.col("mean").sum().alias("ros"))
    )
    fa = (
        table.filter(~pl.col("sleeper_id").is_in(list(rostered)) & pl.col("team").is_not_null())
        .join(ros, on="sleeper_id")
        .sort("ros", descending=True)
        .group_by("position", maintain_order=True)
        .head(fa_per_pos)
    )
    universe = sorted(rostered | set(fa["sleeper_id"].to_list()))
    index = {p: i for i, p in enumerate(universe)}
    n, w_count = len(universe), len(weeks)
    col = {w: i for i, w in enumerate(weeks)}

    info = pl.DataFrame({"sleeper_id": universe}).join(
        table.select("sleeper_id", "position", "team", "injury_status"), on="sleeper_id", how="left"
    )
    pos_idx = {p: i for i, p in enumerate(POSITIONS)}
    pos = np.array([pos_idx[p] for p in info["position"].to_list()])
    mean = np.zeros((n, w_count))
    for sid, wk, m in pts.select("sleeper_id", "week", "mean").iter_rows():
        if sid in index:
            mean[index[sid], col[wk]] = m

    healthy_mean = mean.copy()
    p_play = (mean > 0).astype(float)
    out_weeks = np.zeros(n, int)
    for i, (status, p) in enumerate(zip(info["injury_status"].to_list(), pos, strict=True)):
        if status in tables.status_play:
            p_play[i, 0] *= tables.status_play[status][p]
            mean[i, 0] *= tables.status_mult[status][p]
        if status in OUT_STATUSES:
            p_play[i, 0] = 0.0
            out_weeks[i] = IR_WEEKS if status in ("IR", "PUP") else 1

    # games, NFL teams and correlation roles per week
    teams = sorted(t for t in table["team"].drop_nulls().unique().to_list())
    tix = {t: i for i, t in enumerate(teams)}
    game = np.full((n, w_count), -1)
    team = np.full((n, w_count), -1)
    role = np.full((n, w_count), -1)
    g = proj.games.filter(pl.col("week").is_in(weeks))
    keys = {(wk, tm): f"{wk}:{'-'.join(sorted((tm, opp or '')))}"
            for wk, tm, opp in g.select("week", "team", "opponent").iter_rows()}  # fmt: skip
    gnum = {k: i for i, k in enumerate(sorted(set(keys.values())))}
    gid = {k: gnum[v] for k, v in keys.items()}
    roles = history.assign_roles(
        pts.join(table.select("sleeper_id", "position", "team"), on="sleeper_id")
        .rename({"mean": "proj"})
        .filter(pl.col("team").is_not_null()),
        ["week"],
    )
    for sid, wk, tm, r in roles.select("sleeper_id", "week", "team", "role").iter_rows():
        if sid not in index:
            continue
        i, c = index[sid], col[wk]
        team[i, c] = tix.get(tm, -1)
        game[i, c] = gid.get((wk, tm), -1)
        role[i, c] = r
    inputs = SimInputs(
        universe, pos, weeks, mean, p_play, game, team, role, out_weeks, healthy_mean
    )
    rosters = [
        np.array([index[p] for p in t.players if p in index], dtype=np.intp) for t in league.teams
    ]
    cfg = Config(rosters, np.array([index[p] for p in fa["sleeper_id"].to_list()], dtype=np.intp))
    return inputs, cfg, index


def prepare(
    league_id: str,
    src: Any = None,
    fp_lines: pl.DataFrame | None = None,
    build_history: bool = True,
    progress: Callable[[str], None] = lambda _: None,
) -> Context:
    if src is None:
        from .sources import live

        src = live()
    warnings: list[str] = []
    progress("Loading league")
    league = load_league(league_id, src.sleeper)
    progress("Loading players")
    table = ids.player_table(
        src.sleeper.players(), src.nflverse.ff_playerids() if src.nflverse else pl.DataFrame()
    )
    progress("Blending projections (the very first run downloads past seasons: 2–3 minutes)")
    proj = projections.build(
        src, league.season, league.current_week, sim_weeks(league), league.scoring, table, fp_lines
    )
    progress("Loading score distributions")
    tables = get_tables(src, league, table, build_history, warnings)
    inputs, cfg, index = build_inputs(league, table, proj, tables)
    rostered = [p for t in league.teams for p in t.players]
    skipped = [p for p in rostered if p not in index]
    if skipped:
        warnings.append(f"{len(skipped)} rostered player(s) at unsupported positions are ignored.")
    blank = [p for p in rostered if p in index and inputs.mean[index[p]].sum() == 0]
    if blank:
        names = table.filter(pl.col("sleeper_id").is_in(blank))["name"].to_list()
        warnings.append(
            f"No projections for {len(blank)} rostered player(s), counted as 0: "
            + ", ".join(names[:6])
            + ("…" if len(names) > 6 else "")
        )
    unknown = sorted({s for s in league.slots if s not in SLOT_ELIGIBILITY})
    if unknown:
        warnings.append(f"Roster slots not supported and left empty: {', '.join(unknown)}.")
    if league.trade_deadline and league.current_week > league.trade_deadline:
        warnings.append(f"This league's trade deadline (week {league.trade_deadline}) has passed.")
    progress("Loading market values")
    from . import market

    vals = market.values(src, league)
    if vals.is_empty():
        warnings.append("Market values (FantasyCalc) are unavailable right now.")
    return Context(league, table, proj, tables, inputs, cfg, index, warnings, vals, src)
