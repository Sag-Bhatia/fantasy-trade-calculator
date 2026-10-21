"""Trade finder: trades that raise both teams' title odds (spec §9).

1. Candidates: 1-for-1, 2-for-1, 1-for-2 and 2-for-2 deals between my team and each
   other team, using each side's most valuable non-K/DEF players.
2. Prune: market-value band (±30%, FantasyCalc when both sides are valued, else
   rest-of-season points) and positional need — both teams' expected starting lineups
   must improve (fast, vectorized, no randomness).
3. Screen the survivors at 1,000 sims on one shared set of player outcomes.
4. Confirm the best 10 at 10,000 sims; keep trades where both teams' title odds rise by
   more than one standard error, ranked by the smaller of the two gains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .engine import Context
from .lineup import lineup_points
from .simulate import (
    Config,
    Outcomes,
    SeasonResult,
    paired_delta,
    simulate_league,
    simulate_players,
    team_points,
    team_points_one,
)
from .trade import Delta, Move, Trade, TradeError, apply_trade, week_weights

TOP_PLAYERS = 8  # per roster, by rest-of-season expected points
BAND = 0.30
SCREEN_SIMS = 1_000
CONFIRM_SIMS = 10_000
SCREEN_KEEP = 40
CONFIRM_KEEP = 10
SEED = 11


@dataclass
class Suggestion:
    trade: Trade
    a_title: Delta
    b_title: Delta
    a_playoff: Delta
    b_playoff: Delta
    a_points: float  # Δ starter points per week
    b_points: float
    moves: list[Move]
    market_gap: float | None  # for my team: (value received − value given) / value given


def _value(ctx: Context) -> dict[str, float]:
    """Market value per player (FantasyCalc), 0 when unknown."""
    if ctx.market.is_empty():
        return {}
    return {str(k): float(v) for k, v in ctx.market.select("sleeper_id", "value").iter_rows()}


def _ros(ctx: Context) -> np.ndarray:
    return (ctx.inputs.expected * week_weights(ctx)[None, :]).sum(axis=1)


def _expected_lineup(ctx: Context, roster: np.ndarray) -> float:
    """Playoff-weighted expected starting points, everyone healthy as projected (fast).
    Players known to be out (IR, Out) don't count."""
    roster = roster[ctx.inputs.out_weeks[roster] == 0]
    e = ctx.inputs.expected[roster]  # (n, W)
    pos = ctx.inputs.pos[roster]
    ww = week_weights(ctx)
    total = 0.0
    for w in range(e.shape[1]):
        total += ww[w] * float(
            lineup_points(
                e[None, :, w], np.ones((1, len(roster)), bool), e[:, w], pos, ctx.league.slots
            )[0]
        )
    return total


def candidates(ctx: Context, me: int) -> list[Trade]:
    ros = _ros(ctx)
    val = _value(ctx)
    k_def = {4, 5}  # K, DEF position indexes

    def top(roster_id: int) -> list[str]:
        t = ctx.league.team(roster_id)
        ps = [p for p in t.players if p in ctx.index and ctx.inputs.pos[ctx.index[p]] not in k_def
              and p not in t.reserve]  # fmt: skip
        return sorted(ps, key=lambda p: -ros[ctx.index[p]])[:TOP_PLAYERS]

    def worth(ps: tuple[str, ...]) -> tuple[float, bool]:
        if val and all(val.get(p, 0) > 0 for p in ps):
            return sum(val[p] for p in ps), True
        return sum(float(ros[ctx.index[p]]) for p in ps), False

    mine = top(me)
    out: list[Trade] = []
    for t in ctx.league.teams:
        if t.roster_id == me:
            continue
        theirs = top(t.roster_id)
        for na, nb in ((1, 1), (2, 1), (1, 2), (2, 2)):
            for give in combinations(mine, na):
                wg, mg = worth(give)
                for get in combinations(theirs, nb):
                    wt, mt = worth(get)
                    if mg != mt:  # one side unvalued by the market: compare points instead
                        wg = sum(float(ros[ctx.index[p]]) for p in give)
                        wt = sum(float(ros[ctx.index[p]]) for p in get)
                    if wg <= 0 or wt <= 0 or abs(wt - wg) / max(wg, wt) > BAND:
                        continue
                    out.append(Trade(me, t.roster_id, list(give), list(get)))
    return out


def _need_filter(ctx: Context, trades: list[Trade]) -> list[tuple[float, Trade]]:
    """Keep trades that improve both teams' expected lineups.

    shortcut: ignores the trade's own pickup/cut (apply_trade is ~100x slower); a few
    trades that only pay off after the pickup are pruned. Upgrade if the finder gets faster."""
    base = {}
    kept = []
    for tr in trades:
        ia, ib = ctx.team_index(tr.team_a), ctx.team_index(tr.team_b)
        for i in (ia, ib):
            if i not in base:
                base[i] = _expected_lineup(ctx, ctx.base.rosters[i])
        ga = {ctx.index[p] for p in tr.a_gives}
        gb = {ctx.index[p] for p in tr.b_gives}
        ra = np.array([i for i in ctx.base.rosters[ia] if i not in ga] + list(gb), np.intp)
        rb = np.array([i for i in ctx.base.rosters[ib] if i not in gb] + list(ga), np.intp)
        da = _expected_lineup(ctx, ra) - base[ia]
        db = _expected_lineup(ctx, rb) - base[ib]
        if da > 0 and db > 0:
            kept.append((min(da, db), tr))
    kept.sort(key=lambda x: -x[0])
    return kept


def _deltas(ctx: Context, out: Outcomes, tp_base: np.ndarray, base_res: SeasonResult, tr: Trade,
            cfg: Config) -> tuple[Delta, Delta, Delta, Delta, float, float]:  # fmt: skip
    ia, ib = ctx.team_index(tr.team_a), ctx.team_index(tr.team_b)
    tp = tp_base.copy()
    for t in (ia, ib):
        tp[:, t] = team_points_one(out, ctx.inputs, cfg.rosters[t], cfg.fa_pool, ctx.league.slots)
    res = simulate_league(out, ctx.inputs, ctx.league, cfg, tp)
    d = []
    for t in (ia, ib):
        d.append(Delta(*paired_delta(res.champion[:, t], base_res.champion[:, t])))
    p = [
        Delta(*paired_delta(res.made_playoffs[:, t], base_res.made_playoffs[:, t]))
        for t in (ia, ib)
    ]
    pts = [float((tp[:, t] - tp_base[:, t]).mean()) for t in (ia, ib)]
    return d[0], d[1], p[0], p[1], pts[0], pts[1]


def find_trades(
    ctx: Context,
    me: int,
    progress: Callable[[str], None] = lambda _: None,
    screen_sims: int = SCREEN_SIMS,
    confirm_sims: int = CONFIRM_SIMS,
) -> list[Suggestion]:
    progress("Listing possible deals")
    cands = candidates(ctx, me)
    progress(f"Checking positional fit for {len(cands):,} deals")
    # shortcut: only the 3 × SCREEN_KEEP best positional fits are simulated (runtime budget)
    fit = _need_filter(ctx, cands)[: SCREEN_KEEP * 3]
    if not fit:
        return []
    progress(f"Simulating {len(fit)} promising deals")
    out = simulate_players(ctx.inputs, ctx.tables, screen_sims, SEED)
    tp_base = team_points(out, ctx.inputs, ctx.base, ctx.league.slots)
    base_res = simulate_league(out, ctx.inputs, ctx.league, ctx.base, tp_base)
    screened = []
    for _, tr in fit:
        try:
            cfg, moves = apply_trade(ctx, tr)
        except TradeError:
            continue
        da, db, *_ = _deltas(ctx, out, tp_base, base_res, tr, cfg)
        if da.value > 0 and db.value > 0:
            screened.append((min(da.value, db.value), tr, cfg, moves))
    screened.sort(key=lambda x: -x[0])
    screened = screened[:CONFIRM_KEEP]
    if not screened:
        return []
    progress(f"Confirming the best {len(screened)} with {confirm_sims:,} seasons")
    out = simulate_players(ctx.inputs, ctx.tables, confirm_sims, SEED + 1)
    tp_base = team_points(out, ctx.inputs, ctx.base, ctx.league.slots)
    base_res = simulate_league(out, ctx.inputs, ctx.league, ctx.base, tp_base)
    val = _value(ctx)
    found = []
    for _, tr, cfg, moves in screened:
        da, db, pa, pb, xa, xb = _deltas(ctx, out, tp_base, base_res, tr, cfg)
        if da.value > da.se and db.value > db.se:
            gap = None
            if val and all(val.get(p, 0) > 0 for p in tr.a_gives + tr.b_gives):
                give = sum(val[p] for p in tr.a_gives)
                get = sum(val[p] for p in tr.b_gives)
                gap = (get - give) / give if give else None
            found.append(Suggestion(tr, da, db, pa, pb, xa, xb, moves, gap))
    found.sort(key=lambda s: -min(s.a_title.value, s.b_title.value))
    return found
