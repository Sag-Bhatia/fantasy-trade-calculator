import numpy as np
import polars as pl
import pytest

from ffcalc import engine
from ffcalc.simulate import Config
from ffcalc.trade import (
    COMPONENTS,
    Delta,
    Trade,
    TradeError,
    apply_trade,
    evaluate,
    evaluate_config,
    verdict,
)
from tests.fakes import fake_sources

SIMS = 1500


@pytest.fixture(scope="module")
def ctx():
    import os
    import tempfile

    os.environ["FFCALC_DATA"] = tempfile.mkdtemp()
    return engine.prepare("1400221038771437568", src=fake_sources(), build_history=False)


def top(ctx, roster_id, position, n=1, skip_reserve=True):
    """The n best (rest-of-season expected) players at a position on a team."""
    t = ctx.league.team(roster_id)
    pool = [p for p in t.players if not (skip_reserve and p in t.reserve)]
    rows = ctx.table.filter(pl.col("sleeper_id").is_in(pool) & (pl.col("position") == position))
    ids = rows["sleeper_id"].to_list()
    ids.sort(key=lambda p: -ctx.inputs.expected[ctx.index[p]].sum())
    return ids[:n]


def test_no_op_config_gives_exactly_zero(ctx):
    a, b = ctx.league.teams[0].roster_id, ctx.league.teams[1].roster_id
    noop = Config([r.copy() for r in ctx.base.rosters], ctx.base.fa_pool.copy())
    res = evaluate_config(ctx, Trade(a, b, [], []), noop, [], sims=SIMS, decomposition=False)
    for side in (res.a, res.b):
        assert side.playoff.value == 0 and side.title.value == 0 and side.points_per_week.value == 0
        assert side.score == 0


def test_same_seed_same_result(ctx):
    t = Trade(1, 2, top(ctx, 1, "WR"), top(ctx, 2, "WR"))
    r1 = evaluate(ctx, t, sims=SIMS, decomposition=False)
    r2 = evaluate(ctx, t, sims=SIMS, decomposition=False)
    assert r1.a.title == r2.a.title and r1.b.playoff == r2.b.playoff


def test_swapping_sides_swaps_results(ctx):
    x, y = top(ctx, 1, "RB"), top(ctx, 2, "WR")
    r1 = evaluate(ctx, Trade(1, 2, x, y), sims=SIMS, decomposition=False)
    r2 = evaluate(ctx, Trade(2, 1, y, x), sims=SIMS, decomposition=False)
    assert r1.a.title == r2.b.title and r1.b.title == r2.a.title
    assert r1.a.playoff == r2.b.playoff


def test_decomposition_sums_to_score(ctx):
    t = Trade(1, 2, top(ctx, 1, "RB"), top(ctx, 2, "TE"))
    res = evaluate(ctx, t, sims=SIMS)
    for side in (res.a, res.b):
        assert list(side.decomposition) == list(COMPONENTS)
        assert sum(side.decomposition.values()) == pytest.approx(side.score, abs=1e-3)


def test_two_for_one_fills_and_cuts_roster_spots(ctx):
    give = top(ctx, 1, "WR", 2)
    get = top(ctx, 2, "RB")
    cfg, moves = apply_trade(ctx, Trade(1, 2, give, get))
    size = ctx.league.roster_size
    for rid, move in zip((1, 2), moves, strict=True):
        t = ctx.league.team(rid)
        reserve = {ctx.index[p] for p in t.reserve}
        active = [i for i in cfg.rosters[ctx.team_index(rid)] if i not in reserve]
        assert len(active) <= size
        if rid == 2:
            assert move.drops and not set(move.drops) & set(give)  # never cut who you got
    # team 1 had a full roster, gave 2 got 1 → picked up a free agent
    assert moves[0].adds and all(ctx.index[p] not in cfg.fa_pool for p in moves[0].adds)
    assert all(ctx.index[p] in cfg.fa_pool for p in moves[1].drops)


def test_three_for_one(ctx):
    give = top(ctx, 1, "RB", 3)
    get = top(ctx, 2, "QB")
    cfg, moves = apply_trade(ctx, Trade(1, 2, give, get))
    assert len(moves[0].adds) == 2 and len(moves[1].drops) == 2


def test_trading_someone_you_dont_own_fails(ctx):
    with pytest.raises(TradeError):
        apply_trade(ctx, Trade(1, 2, top(ctx, 2, "WR"), top(ctx, 2, "RB")))
    with pytest.raises(TradeError):
        apply_trade(ctx, Trade(1, 1, top(ctx, 1, "WR"), []))
    with pytest.raises(TradeError):
        apply_trade(ctx, Trade(1, 2, [], []))


def test_verdict_thresholds():
    assert verdict(Delta(0.02, 0.004)) == "Accept"
    assert verdict(Delta(0.02, 0.02)) == "Roughly even"  # not 2 SE clear
    assert verdict(Delta(0.003, 0.0001)) == "Roughly even"  # under 0.5 pp
    assert verdict(Delta(-0.02, 0.004)) == "Decline"


def test_upgrade_at_a_starting_spot_helps(ctx):
    # team 1 gets team 2's best WR for team 1's worst WR: team 1 must gain points
    worst = top(ctx, 1, "WR", 10)[-1]
    best = top(ctx, 2, "WR")[0]
    res = evaluate(ctx, Trade(1, 2, [worst], [best]), sims=SIMS, decomposition=False)
    assert res.a.points_per_week.value > 0 > res.b.points_per_week.value
    assert np.isfinite(res.a.title.se)


def test_duplicate_or_unknown_team_rejected(ctx):
    p = top(ctx, 1, "WR")
    with pytest.raises(TradeError):
        apply_trade(ctx, Trade(1, 2, p + p, []))
    with pytest.raises(TradeError):
        apply_trade(ctx, Trade(1, 999, p, []))


def test_cuts_are_bench_players(ctx):
    from ffcalc.lineup import optimal_lineup

    give = top(ctx, 1, "WR", 2)
    cfg, moves = apply_trade(ctx, Trade(1, 2, give, top(ctx, 2, "RB")))
    team2 = ctx.base.rosters[ctx.team_index(2)]
    for p in moves[1].drops:
        i = ctx.index[p]
        for w in range(ctx.inputs.expected.shape[1]):
            e = ctx.inputs.expected[team2, w]
            _, picks = optimal_lineup(list(e), list(ctx.inputs.pos[team2]), ctx.league.slots)
            starters = {int(team2[k]) for k, _ in picks}
            assert i not in starters or e[list(team2).index(i)] == 0


def test_traded_ir_player_keeps_ir_spot(ctx):
    owner = next(t for t in ctx.league.teams if t.reserve)
    ir = owner.reserve[0]
    other = next(t for t in ctx.league.teams if not t.reserve and t.roster_id != owner.roster_id)
    _, moves = apply_trade(ctx, Trade(owner.roster_id, other.roster_id, [ir], []))
    # the receiver has a free IR slot, so taking an IR player forces no cut
    assert moves[1].drops == []
