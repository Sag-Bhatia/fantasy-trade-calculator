"""Trade finder: trades that raise both teams' title odds (spec §9).

1. Candidates: 1-for-1, 2-for-1, 1-for-2 and 2-for-2 deals between my team and each
   other team, using each side's most valuable non-K/DEF players.
2. Prune: market-value band (±30%, FantasyCalc when both sides are valued, else
   rest-of-season points) and positional need — both teams' expected starting lineups
   must improve (fast, vectorized, no randomness).
3. Screen the best fits per partner at 1,000 sims on one shared set of player outcomes,
   ranked by the smaller of the two teams' gains in expected starter points (precise even
   at 1,000 sims; title odds at that size are ±1.5 points, too noisy to rank by).
4. Pick a varied shortlist — the best version of each deal (no throw-in variants of the
   same core trade) and at most two per partner — and confirm it at 10,000 sims; keep
   trades where both teams' title odds rise by more than one standard error, ranked by the
   smaller of the two gains.

`balance` ("Make it fair") runs steps 3–4 on small tweaks of one trade instead: one more
player from either roster, or one fewer from a side sending two or more.
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
PER_TEAM_SCREEN = 10  # best positional fits simulated per partner
MAX_PER_TEAM = 2  # suggestions per partner
CONFIRM_KEEP = 12
BALANCE_KEEP = 5  # tweaks confirmed by `balance`
MAX_SIDE = 3  # players one side can send in a tweaked trade
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


def _top(ctx: Context, roster_id: int, ros: np.ndarray) -> list[str]:
    """A roster's most valuable tradeable players: no K/DEF, nobody on IR."""
    k_def = {4, 5}  # K, DEF position indexes
    t = ctx.league.team(roster_id)
    ps = [p for p in t.players if p in ctx.index and ctx.inputs.pos[ctx.index[p]] not in k_def
          and p not in t.reserve]  # fmt: skip
    return sorted(ps, key=lambda p: -ros[ctx.index[p]])[:TOP_PLAYERS]


def candidates(ctx: Context, me: int) -> list[Trade]:
    ros = _ros(ctx)
    val = _value(ctx)

    def top(roster_id: int) -> list[str]:
        return _top(ctx, roster_id, ros)

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


def _variant(tr: Trade, picked: list[Trade]) -> bool:
    """Same partner and overlapping players on both sides: a throw-in version of a deal
    already picked (e.g. "A + B for C" vs "A + B for C + D")."""
    return any(
        t.team_b == tr.team_b
        and set(t.a_gives) & set(tr.a_gives)
        and set(t.b_gives) & set(tr.b_gives)
        for t in picked
    )


def diverse(ranked: list[Trade], n: int, per_team: int) -> list[Trade]:
    """Best-first picks with no variants of an earlier pick and at most per_team per partner."""
    picked: list[Trade] = []
    per: dict[int, int] = {}
    for tr in ranked:
        if len(picked) == n:
            break
        if per.get(tr.team_b, 0) >= per_team or _variant(tr, picked):
            continue
        picked.append(tr)
        per[tr.team_b] = per.get(tr.team_b, 0) + 1
    return picked


Screened = tuple[float, Trade, Config, list[Move]]


def _screen(ctx: Context, trades: list[Trade], sims: int) -> list[Screened]:
    """Trades whose both starting lineups really get better in `sims` shared seasons,
    best (largest smaller gain) first."""
    out = simulate_players(ctx.inputs, ctx.tables, sims, SEED)
    tp_base = team_points(out, ctx.inputs, ctx.base, ctx.league.slots)
    base_res = simulate_league(out, ctx.inputs, ctx.league, ctx.base, tp_base)
    screened = []
    for tr in trades:
        try:
            cfg, moves = apply_trade(ctx, tr)
        except TradeError:
            continue
        *_, xa, xb = _deltas(ctx, out, tp_base, base_res, tr, cfg)
        if xa > 0 and xb > 0:
            screened.append((min(xa, xb), tr, cfg, moves))
    screened.sort(key=lambda x: -x[0])
    return screened


def _confirm(ctx: Context, screened: list[Screened], sims: int) -> list[Suggestion]:
    """Re-run at `sims`; keep trades where both title odds rise by more than one standard
    error, best (largest smaller gain) first."""
    out = simulate_players(ctx.inputs, ctx.tables, sims, SEED + 1)
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
    # shortcut: only each partner's PER_TEAM_SCREEN best positional fits are simulated
    # (runtime budget); the cap also keeps one team's many combinations from crowding out the rest
    fit = []
    per: dict[int, int] = {}
    for _, tr in _need_filter(ctx, cands):
        if per.get(tr.team_b, 0) < PER_TEAM_SCREEN:
            fit.append(tr)
            per[tr.team_b] = per.get(tr.team_b, 0) + 1
    if not fit:
        return []
    progress(f"Simulating {len(fit)} promising deals")
    screened = _screen(ctx, fit, screen_sims)
    keep = {id(t) for t in diverse([x[1] for x in screened], CONFIRM_KEEP, MAX_PER_TEAM)}
    screened = [x for x in screened if id(x[1]) in keep]
    if not screened:
        return []
    progress(f"Confirming the best {len(screened)} with {confirm_sims:,} seasons")
    return _confirm(ctx, screened, confirm_sims)


def tweaks(ctx: Context, tr: Trade) -> list[Trade]:
    """One-step changes to a trade: either side sends one more of its best players, or a
    side sending two or more keeps one of them."""
    ros = _ros(ctx)
    used = set(tr.a_gives) | set(tr.b_gives)
    out = []
    for a_side in (True, False):
        for p in _top(ctx, tr.team_a if a_side else tr.team_b, ros):
            give, get = list(tr.a_gives), list(tr.b_gives)
            (give if a_side else get).append(p)
            if p not in used and len(give) <= MAX_SIDE and len(get) <= MAX_SIDE:
                out.append(Trade(tr.team_a, tr.team_b, give, get))
    if len(tr.a_gives) > 1:
        out += [Trade(tr.team_a, tr.team_b, [x for x in tr.a_gives if x != p], list(tr.b_gives))
                for p in tr.a_gives]  # fmt: skip
    if len(tr.b_gives) > 1:
        out += [Trade(tr.team_a, tr.team_b, list(tr.a_gives), [x for x in tr.b_gives if x != p])
                for p in tr.b_gives]  # fmt: skip
    return out


def balance(
    ctx: Context,
    tr: Trade,
    progress: Callable[[str], None] = lambda _: None,
    screen_sims: int = SCREEN_SIMS,
    confirm_sims: int = CONFIRM_SIMS,
) -> list[Suggestion]:
    """Small changes to `tr` that raise both teams' title odds ("Make it fair")."""
    options = tweaks(ctx, tr)
    progress(f"Trying {len(options)} small changes to the trade")
    screened = _screen(ctx, options, screen_sims)[:BALANCE_KEEP]
    if not screened:
        return []
    progress(f"Confirming the best {len(screened)} with {confirm_sims:,} seasons")
    return _confirm(ctx, screened, confirm_sims)
