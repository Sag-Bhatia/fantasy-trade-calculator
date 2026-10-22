"""Trade evaluation, decomposition and verdict (spec §7)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .engine import Context
from .lineup import optimal_lineup
from .simulate import (
    Config,
    Outcomes,
    SeasonResult,
    SimInputs,
    Tables,
    paired_delta,
    simulate_league,
    simulate_players,
    team_points,
    team_points_one,
)

PLAYOFF_WEIGHT = 2.0  # playoff-week points count double in the trade score (spec §7)
VERDICT_PP = 0.005  # ±0.5 percentage points of title odds
VERDICT_SE = 2.0
COMPONENTS = ("Starters", "Depth", "Roster cuts", "Schedule & byes", "Injury risk", "Playoff weeks")


class TradeError(ValueError):
    pass


@dataclass
class Trade:
    team_a: int  # roster_id of the user's team
    team_b: int
    a_gives: list[str]
    b_gives: list[str]


@dataclass
class Move:
    roster_id: int
    drops: list[str] = field(default_factory=list)  # players released for roster space


@dataclass
class Delta:
    value: float
    se: float


@dataclass
class TeamResult:
    roster_id: int
    name: str
    playoff_before: float
    playoff_after: float
    title_before: float
    title_after: float
    playoff: Delta
    title: Delta
    points_per_week: Delta  # starting-lineup points, rest of season
    score: float  # playoff-weighted Δ expected points, rest of season
    decomposition: dict[str, float]
    moves: Move


@dataclass
class TradeResult:
    trade: Trade
    a: TeamResult
    b: TeamResult
    verdict: str
    sims: int
    seed: int
    warnings: list[str]
    before: SeasonResult
    after: SeasonResult
    market: Any = None  # market.MarketCheck when FantasyCalc values are available
    acceptance: float | None = None  # HEURISTIC chance the partner says yes (spec §8)


# ------------------------------------------------------------------ roster moves


def week_weights(ctx: Context) -> np.ndarray:
    lg = ctx.league
    return np.array([PLAYOFF_WEIGHT if w > lg.last_regular_week else 1.0 for w in ctx.inputs.weeks])


def lineup_value(ctx: Context, roster: np.ndarray, expected: np.ndarray | None = None) -> float:
    """Playoff-weighted sum of weekly optimal expected lineups (no randomness)."""
    e = ctx.inputs.expected if expected is None else expected
    pos = ctx.inputs.pos[roster]
    ww = week_weights(ctx)
    return float(
        sum(
            ww[w] * optimal_lineup(list(e[roster, w]), list(pos), ctx.league.slots)[0]
            for w in range(e.shape[1])
        )
    )


def _ros(ctx: Context, i: int) -> float:
    return float((ctx.inputs.expected[i] * week_weights(ctx)).sum())


def apply_trade(ctx: Context, trade: Trade, cfg: Config | None = None) -> tuple[Config, list[Move]]:
    """Swap players, then cut the player the lineup misses least while a roster is over the
    limit. A spot the trade opens stays open: a waiver pickup isn't part of the trade, since
    the team could make it anyway. IR players stay off the active count (a traded IR player
    goes to the new team's IR if it has a free slot)."""
    cfg = cfg or ctx.base
    lg = ctx.league
    ids = [t.roster_id for t in lg.teams]
    if trade.team_a not in ids or trade.team_b not in ids:
        raise TradeError("That team isn't in this league.")
    ia, ib = ctx.team_index(trade.team_a), ctx.team_index(trade.team_b)
    if ia == ib:
        raise TradeError("Pick two different teams.")
    if not trade.a_gives and not trade.b_gives:
        raise TradeError("The trade is empty.")
    if len({*trade.a_gives, *trade.b_gives}) != len(trade.a_gives) + len(trade.b_gives):
        raise TradeError("A player is listed twice.")
    rosters = [r.copy() for r in cfg.rosters]
    fa = list(cfg.fa_pool)
    for team_i, gives in ((ia, trade.a_gives), (ib, trade.b_gives)):
        for p in gives:
            if p not in ctx.index or ctx.index[p] not in rosters[team_i]:
                raise TradeError(f"{ctx.name(p)} isn't on {lg.teams[team_i].name}'s roster.")
    ga = [ctx.index[p] for p in trade.a_gives]
    gb = [ctx.index[p] for p in trade.b_gives]
    reserve = {t: {ctx.index[p] for p in lg.teams[t].reserve if p in ctx.index} for t in (ia, ib)}
    rosters[ia] = np.array([i for i in rosters[ia] if i not in ga] + gb, dtype=np.intp)
    rosters[ib] = np.array([i for i in rosters[ib] if i not in gb] + ga, dtype=np.intp)
    was_ir = reserve[ia] | reserve[ib]
    for t, received in ((ia, gb), (ib, ga)):
        reserve[t] = {i for i in reserve[t] if i in rosters[t]}
        for i in received:  # a traded IR player keeps an IR spot if the new team has one
            if i in was_ir and len(reserve[t]) < lg.ir_slots:
                reserve[t].add(i)

    moves = []
    for team_i, received in ((ia, gb), (ib, ga)):
        move = Move(lg.teams[team_i].roster_id)
        active = [i for i in rosters[team_i] if i not in reserve[team_i]]
        while len(active) > lg.roster_size:
            # cut the player the lineup misses least (a bench player), never one just received
            cands = [i for i in active if i not in received]
            worst = max(
                cands,
                key=lambda i: (
                    lineup_value(ctx, np.array([j for j in active if j != i], dtype=np.intp)),
                    -_ros(ctx, i),
                ),
            )
            active.remove(worst)
            move.drops.append(ctx.inputs.ids[worst])
            fa.append(worst)
        rosters[team_i] = np.array(active + sorted(reserve[team_i]), dtype=np.intp)
        moves.append(move)
    return Config(rosters, np.array(sorted(set(fa)), dtype=np.intp)), moves


# ------------------------------------------------------------------ decomposition


def _stage_inputs(ctx: Context, stage: int) -> tuple[SimInputs, Tables]:
    """Model features switched on cumulatively (spec §7): 0 starters only (everyone always
    healthy, average week) · 1 depth (real in-season injury rates) · 2 roster cuts (the
    trade's forced cuts; same inputs) · 3 schedule & byes (real weekly projections) ·
    4 injury risk (current injury news) · 5 playoff-week weighting (same inputs)."""
    inp, tab = ctx.inputs, copy.copy(ctx.tables)
    if stage >= 4:
        return inp, tab
    base = inp.healthy_mean if inp.healthy_mean is not None else inp.mean
    mean = base.copy()
    if stage < 3:  # each player's average week, no byes, no matchups
        has = base > 0
        avg = np.where(has.any(1), (base * has).sum(1) / np.maximum(has.sum(1), 1), 0.0)
        mean = np.repeat(avg[:, None], base.shape[1], axis=1)
    if stage == 0:
        tab.hazard = np.zeros_like(tab.hazard)
    p_play = (mean > 0).astype(float)
    out_weeks = np.zeros_like(inp.out_weeks)
    return replace(inp, mean=mean, p_play=p_play, out_weeks=out_weeks, healthy_mean=None), tab


def decompose(ctx: Context, cfg_before: Config, cfg_after: Config, teams: list[int],
              moves: list[Move], seed: int, sims: int, full: Outcomes | None = None,
              ) -> dict[int, dict[str, float]]:  # fmt: skip
    """Per team index: Δ playoff-weighted expected points by component.

    Each component is the change in the trade's value when its model feature is switched
    on (stage order fixed, so it depends on that order); components sum to the total.
    All stages share the seed and sim count, and the last two reuse the headline outcomes
    (`full`), so the total equals the headline trade score exactly. Weekly free-agent fill
    of empty slots is on throughout; "Roster cuts" is the trade's forced cuts."""
    ww = week_weights(ctx)
    no_moves = _without_moves(ctx, cfg_before, cfg_after, teams, moves)
    vals: dict[int, list[float]] = {t: [] for t in teams}
    out = None
    for stage in range(6):
        inp, tab = _stage_inputs(ctx, stage)
        if stage >= 4 and full is not None:
            out = full
        elif stage != 2 or out is None:  # stage 2 changes rosters, not inputs
            out = simulate_players(inp, tab, sims, seed)
        weights = ww if stage == 5 else np.ones_like(ww)
        after = cfg_after if stage >= 2 else no_moves
        for t in teams:
            pa = team_points_one(out, inp, after.rosters[t], after.fa_pool, ctx.league.slots)
            pb = team_points_one(
                out, inp, cfg_before.rosters[t], cfg_before.fa_pool, ctx.league.slots
            )
            vals[t].append(float(((pa - pb).mean(0) * weights).sum()))
    return {
        t: {name: v - (vals[t][k - 1] if k else 0.0) for k, (name, v) in enumerate(zip(COMPONENTS, vals[t], strict=True))}  # noqa: E501
        for t in teams
    }  # fmt: skip


def _without_moves(
    ctx: Context, before: Config, after: Config, teams: list[int], moves: list[Move]
) -> Config:
    """After-trade rosters with the trade's cuts put back."""
    rosters = [r.copy() for r in after.rosters]
    for t, move in zip(teams, moves, strict=True):
        rosters[t] = np.concatenate([rosters[t], [ctx.index[p] for p in move.drops]]).astype(
            np.intp
        )
    return Config(rosters, before.fa_pool)


# ------------------------------------------------------------------ evaluation


def verdict(title: Delta) -> str:
    if title.value > VERDICT_PP and title.value > VERDICT_SE * title.se:
        return "Accept"
    if title.value < -VERDICT_PP and title.value < -VERDICT_SE * title.se:
        return "Decline"
    return "Roughly even"


def evaluate(ctx: Context, trade: Trade, sims: int = 10_000, seed: int = 7,
             decomposition: bool = True,
             progress: Callable[[str], None] = lambda _: None) -> TradeResult:  # fmt: skip
    progress("Making both rosters legal")
    cfg_after, moves = apply_trade(ctx, trade)
    return evaluate_config(ctx, trade, cfg_after, moves, sims, seed, decomposition, progress)


def evaluate_config(ctx: Context, trade: Trade, cfg_after: Config, moves: list[Move],
                    sims: int = 10_000, seed: int = 7, decomposition: bool = True,
                    progress: Callable[[str], None] = lambda _: None,
                    ) -> TradeResult:  # fmt: skip
    """Compare `cfg_after` with the league as it is, on one shared set of player outcomes."""
    moves = moves or [Move(trade.team_a), Move(trade.team_b)]
    progress(f"Simulating {sims:,} seasons")
    out = simulate_players(ctx.inputs, ctx.tables, sims, seed)
    tp_before = team_points(out, ctx.inputs, ctx.base, ctx.league.slots)
    tp_after = tp_before.copy()
    ia, ib = ctx.team_index(trade.team_a), ctx.team_index(trade.team_b)
    # shortcut: other teams keep their lineups even if the trade moves a free agent they'd
    # have streamed into an empty slot; upgrade if FA contention ever matters
    for t in (ia, ib):  # only the two trading teams' lineups change
        tp_after[:, t] = team_points_one(
            out, ctx.inputs, cfg_after.rosters[t], cfg_after.fa_pool, ctx.league.slots
        )
    before = simulate_league(out, ctx.inputs, ctx.league, ctx.base, tp_before)
    after = simulate_league(out, ctx.inputs, ctx.league, cfg_after, tp_after)
    progress("Working out why")
    dec = (
        decompose(ctx, ctx.base, cfg_after, [ia, ib], moves, seed, sims, out)
        if decomposition
        else {ia: {}, ib: {}}
    )
    ww = week_weights(ctx)
    results = []
    for t, move in zip((ia, ib), moves, strict=True):
        team = ctx.league.teams[t]
        ppw = paired_delta(tp_after[:, t].mean(1), tp_before[:, t].mean(1))
        score = float(((tp_after[:, t] - tp_before[:, t]).mean(0) * ww).sum())
        results.append(
            TeamResult(
                team.roster_id,
                team.name,
                float(before.made_playoffs[:, t].mean()),
                float(after.made_playoffs[:, t].mean()),
                float(before.champion[:, t].mean()),
                float(after.champion[:, t].mean()),
                Delta(*paired_delta(after.made_playoffs[:, t], before.made_playoffs[:, t])),
                Delta(*paired_delta(after.champion[:, t], before.champion[:, t])),
                Delta(*ppw),
                score,
                dec[t],
                move,
            )  # fmt: skip
        )
    res = TradeResult(
        trade, results[0], results[1], verdict(results[0].title), sims, seed,
        list(ctx.warnings), before, after,
    )  # fmt: skip
    if not ctx.market.is_empty():
        from . import market

        mc = market.market_check(trade, ctx.market, market.fc_params(ctx.league))
        gap_b = (mc.a_gives.total - mc.b_gives.total) / max(mc.b_gives.total, 1)
        res.market = mc
        res.acceptance = market.acceptance(res.b.title.value * 100, gap_b)
    return res
