"""GOAL.md bug-hunt edge cases not covered by the per-module tests."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ffcalc import engine, projections
from ffcalc.scoring import STANDARD_PPR, points, score_lines
from ffcalc.simulate import (
    placeholder_tables,
    simulate_league,
    simulate_players,
    team_points_one,
)
from ffcalc.trade import Trade, evaluate
from tests.fakes import fake_sources
from tests.test_simulate import SLOTS, synthetic

SIMS = 1000


@pytest.fixture(scope="module")
def ctx():
    import os
    import tempfile

    os.environ["FFCALC_DATA"] = tempfile.mkdtemp()
    return engine.prepare("1400221038771437568", src=fake_sources(), build_history=False)


def top(ctx, roster_id, position, n=1):
    t = ctx.league.team(roster_id)
    pool = [p for p in t.players if p not in t.reserve]
    rows = ctx.table.filter(pl.col("sleeper_id").is_in(pool) & (pl.col("position") == position))
    ids = rows["sleeper_id"].to_list()
    ids.sort(key=lambda p: -ctx.inputs.expected[ctx.index[p]].sum())
    return ids[:n]


def with_week(ctx, current_week, results=None):
    """The same league, but `current_week` weeks in (inputs rebuilt for the shorter horizon)."""
    league = replace(ctx.league, current_week=current_week, results=results or ctx.league.results)
    inputs, base, index = engine.build_inputs(league, ctx.table, ctx.proj, ctx.tables)
    return replace(ctx, league=league, inputs=inputs, base=base, index=index)


def check_valid(res, league):
    for s in (res.before, res.after):
        assert s.playoff_prob.sum() == pytest.approx(league.playoff_teams)
        assert s.title_prob.sum() == pytest.approx(1.0)
        assert not np.isnan(s.team_points).any()
        assert ((s.playoff_prob >= 0) & (s.playoff_prob <= 1)).all()
    for side in (res.a, res.b):
        vals = [side.playoff_before, side.playoff_after, side.title_before, side.title_after,
                side.playoff.value, side.playoff.se, side.title.value, side.title.se,
                side.points_per_week.value, side.points_per_week.se, side.score]  # fmt: skip
        assert np.isfinite(vals).all()


# 1. final regular-season week ------------------------------------------------------------


def test_trade_in_final_regular_season_week(ctx):
    c = with_week(ctx, ctx.league.last_regular_week)
    assert c.inputs.weeks[0] == c.league.last_regular_week
    t = Trade(1, 2, top(c, 1, "WR"), top(c, 2, "WR"))
    check_valid(evaluate(c, t, sims=SIMS, decomposition=False), c.league)


# 2. playoffs already under way -----------------------------------------------------------


def test_trade_while_playoffs_are_in_progress(ctx):
    lg = ctx.league
    wk = lg.playoff_week_start + 1  # quarterfinals (week 15) already played
    played = {t.roster_id: 80.0 + 3 * i for i, t in enumerate(lg.teams)}
    c = with_week(ctx, wk, {**lg.results, lg.playoff_week_start: played})
    assert c.league.current_week > c.league.last_regular_week
    t = Trade(1, 2, top(c, 1, "RB"), top(c, 2, "RB"))
    check_valid(evaluate(c, t, sims=SIMS, decomposition=False), c.league)


# 3. superflex ----------------------------------------------------------------------------


def test_superflex_starts_a_second_qb():
    league, inp, cfg = synthetic()
    qb = np.flatnonzero(inp.pos == 0)
    mean = np.full_like(inp.mean, 5.0)
    mean[qb] = 18.0  # every QB beats every other player
    inp = replace(inp, mean=mean, p_play=np.ones_like(inp.p_play))
    out = simulate_players(inp, placeholder_tables(), 300, 3)
    roster = cfg.rosters[0]
    qbs = [i for i in roster if inp.pos[i] == 0]
    one_qb = np.array([i for i in roster if i != qbs[1]])
    sf = [*SLOTS, "SUPER_FLEX"]

    def pts(r, slots):
        return team_points_one(out, inp, r, cfg.fa_pool, slots).mean()

    assert pts(roster, sf) > pts(one_qb, sf) + 5  # QB2 fills the superflex slot
    assert abs(pts(roster, SLOTS) - pts(one_qb, SLOTS)) < 1  # only injury cover in a 1-QB league
    # the whole season plays out with the extra slot
    lg = replace(league, slots=sf)
    res = simulate_league(out, inp, lg, cfg)
    assert res.title_prob.sum() == pytest.approx(1.0)
    assert res.playoff_prob.sum() == pytest.approx(lg.playoff_teams)


# 4. TE premium ---------------------------------------------------------------------------


def test_te_premium_scores_te_lines_higher():
    te = {"rec": 5, "rec_yd": 50, "bonus_rec_te": 5}
    prem = STANDARD_PPR | {"bonus_rec_te": 0.5}
    assert points(te, prem) == pytest.approx(points(te, STANDARD_PPR) + 2.5)
    rows = [("te", "rec", 5.0), ("te", "rec_yd", 50.0), ("te", "bonus_rec_te", 5.0),
            ("wr", "rec", 5.0), ("wr", "rec_yd", 50.0)]  # fmt: skip
    lines = pl.DataFrame(rows, schema=["sleeper_id", "stat", "value"], orient="row").with_columns(
        pl.lit(7).alias("week")
    )
    by_id = lambda sc: dict(score_lines(lines, sc).select("sleeper_id", "points").iter_rows())  # noqa: E731
    a, b = by_id(prem), by_id(STANDARD_PPR)
    assert a["te"] == pytest.approx(b["te"] + 2.5)
    assert a["wr"] == pytest.approx(b["wr"])  # no premium for other positions


def test_te_premium_flows_into_blended_projections(ctx):
    weeks = [7, 8]
    args = (fake_sources(), 2026, 7, weeks)
    plain = projections.build(*args, STANDARD_PPR, ctx.table, weights={}).points
    prem = projections.build(*args, STANDARD_PPR | {"bonus_rec_te": 0.5}, ctx.table, weights={})
    pos = ctx.table.select("sleeper_id", "position")
    j = (
        plain.join(prem.points, on=["season", "week", "sleeper_id"], suffix="_p")
        .join(pos, on="sleeper_id")
        .with_columns((pl.col("mean_p") - pl.col("mean")).alias("d"))
    )
    assert j.filter(pl.col("position") == "TE")["d"].min() >= -1e-9
    assert j.filter(pl.col("position") == "TE")["d"].max() > 0.5  # some TE gains real points
    assert j.filter(pl.col("position") != "TE")["d"].abs().max() < 1e-6


# 5. league sizes -------------------------------------------------------------------------


@pytest.mark.parametrize("n_teams", [10, 12, 14])
@pytest.mark.parametrize("playoff_teams", [4, 6, 8])
def test_league_sizes_and_playoff_formats(n_teams, playoff_teams):
    league, inp, cfg = synthetic(n_teams=n_teams, playoff_teams=playoff_teams)
    out = simulate_players(inp, placeholder_tables(), 300, 5)
    res = simulate_league(out, inp, league, cfg)
    assert res.playoff_prob.sum() == pytest.approx(playoff_teams)
    assert res.title_prob.sum() == pytest.approx(1.0)
    assert (res.made_playoffs | ~res.champion).all()
    assert (np.sort(res.seed, axis=1) == np.arange(1, n_teams + 1)).all()


# 6. odd playoff sizes: byes go to the top seeds ------------------------------------------


@pytest.mark.parametrize(("playoff_teams", "n_teams"), [(7, 12), (10, 12)])
def test_odd_playoff_sizes_give_byes_to_top_seeds(playoff_teams, n_teams):
    league, inp, cfg = synthetic(n_teams=n_teams, playoff_teams=playoff_teams, current_week=16)
    for i, t in enumerate(league.teams):
        t.wins, t.points_for = 10 - i, 1500.0 - i  # seed = roster order
    # round one (week 15, already played): seeds 1 and 2 scored nothing, everyone else 100
    league.results[15] = {
        t.roster_id: (0.0 if i < 2 else 100.0) for i, t in enumerate(league.teams)
    }
    out = simulate_players(inp, placeholder_tables(), 600, 4)
    res = simulate_league(out, inp, league, cfg)
    assert res.playoff_prob.sum() == pytest.approx(playoff_teams)
    assert res.title_prob.sum() == pytest.approx(1.0)
    assert (res.seed[:, :playoff_teams] == np.arange(1, playoff_teams + 1)).all()
    # byes: a 0-point round one still advances seeds 1 and 2 (7 teams: only seed 1 has the
    # bye, so only check seed 1 there)
    assert res.champion[:, 0].any()
    if playoff_teams == 10:
        assert res.champion[:, 1].any()
    # seeds outside the bracket never win
    assert not res.champion[:, playoff_teams:].any()


# 7. a player nobody projects -------------------------------------------------------------


def test_player_with_no_projections_never_scores_or_starts(ctx):
    warned = next(w for w in ctx.warnings if w.startswith("No projections"))
    blank = [
        p for t in ctx.league.teams for p in t.players
        if p in ctx.index and ctx.inputs.mean[ctx.index[p]].sum() == 0
    ]  # fmt: skip
    assert blank and all(ctx.name(p) in warned for p in blank)
    out = simulate_players(ctx.inputs, ctx.tables, 200, 3)
    for p in blank:
        i = ctx.index[p]
        assert (out.points[:, :, i] == 0).all()
        owner = next(k for k, r in enumerate(ctx.base.rosters) if i in r)
        r = ctx.base.rosters[owner]
        args = (out, ctx.inputs)
        with_him = team_points_one(*args, r, ctx.base.fa_pool, ctx.league.slots)
        without = team_points_one(*args, r[r != i], ctx.base.fa_pool, ctx.league.slots)
        assert np.array_equal(with_him, without)  # he was never in a lineup


# 9. a gift: one side gives nothing -------------------------------------------------------


def test_gift_trade_does_not_help_the_giver(ctx):
    give = top(ctx, 1, "WR")
    res = evaluate(ctx, Trade(1, 2, give, []), sims=SIMS, decomposition=False)
    check_valid(res, ctx.league)
    assert res.a.points_per_week.value <= 3 * res.a.points_per_week.se
    assert res.b.points_per_week.value >= -3 * res.b.points_per_week.se


# 10. determinism across separately built universes --------------------------------------


@pytest.mark.slow
def test_same_trade_identical_from_two_separately_built_contexts():
    results = []
    for _ in range(2):
        c = engine.prepare("1400221038771437568", src=fake_sources(), build_history=False)
        t = Trade(1, 2, top(c, 1, "RB") + top(c, 1, "WR"), top(c, 2, "QB"))
        results.append(evaluate(c, t, sims=SIMS))
    r1, r2 = results
    for s1, s2 in ((r1.a, r2.a), (r1.b, r2.b)):
        assert s1.title == s2.title and s1.playoff == s2.playoff
        assert s1.points_per_week == s2.points_per_week and s1.score == s2.score
        assert s1.decomposition == s2.decomposition and s1.moves == s2.moves
    # inputs differ at ~1e-15 between builds (polars sum order), so no bit-for-bit check
    assert np.allclose(r1.after.team_points, r2.after.team_points, atol=1e-3)
