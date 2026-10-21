import time

import numpy as np
import pytest

from ffcalc import engine
from ffcalc.models import POSITIONS, League, Team
from ffcalc.simulate import (
    Config,
    SimInputs,
    bracket_order,
    paired_delta,
    placeholder_tables,
    playoff_rounds,
    simulate_league,
    simulate_players,
)
from tests.fakes import fake_sources

SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
PER_TEAM = {"QB": 2, "RB": 5, "WR": 5, "TE": 2, "K": 1, "DEF": 1}


def round_robin(n: int, weeks: range) -> dict[int, list[tuple[int, int]]]:
    ids = list(range(1, n + 1))
    out = {}
    for wk in weeks:
        k = wk % (n - 1)
        r = ids[:1] + ids[1:][k:] + ids[1:][:k]
        out[wk] = [(r[i], r[n - 1 - i]) for i in range(n // 2)]
    return out


def synthetic(n_teams=12, playoff_teams=6, round_type=0, median=False, current_week=7, seed=0):
    """A league and inputs with known structure (no fixtures)."""
    rng = np.random.default_rng(seed)
    league = League(
        "x", "x", 2026, n_teams, SLOTS, 6, 1, {}, playoff_teams, 15, round_type, median, 11,
        current_week,
        [Team(i + 1, None, f"T{i + 1}", [], wins=3, losses=3, points_for=600.0 + i)
         for i in range(n_teams)],
        round_robin(n_teams, range(1, 15)), {},
    )  # fmt: skip
    weeks = list(range(current_week, playoff_rounds(league)[-1][-1] + 1))
    pos = np.array(
        [POSITIONS.index(p) for _ in range(n_teams) for p, c in PER_TEAM.items() for _ in range(c)]
        + [POSITIONS.index(p) for p in POSITIONS for _ in range(6)]
    )
    n = len(pos)
    base = np.array([18, 11, 11, 8, 8, 7.0])[pos]
    mean = np.repeat((rng.gamma(4, 1, n) / 4 * base)[:, None], len(weeks), 1)
    nfl = rng.integers(0, 32, n)
    p_play = np.ones_like(mean)
    bye = rng.integers(0, len(weeks), 32)
    p_play[np.arange(n), bye[nfl]] = 0.0
    role = np.full(mean.shape, -1)
    role[pos == 0] = 0
    role[pos == 2] = 3
    role[pos == 3] = 6
    inp = SimInputs([str(i) for i in range(n)], pos, weeks, mean, p_play,
                    np.repeat((nfl // 2)[:, None], len(weeks), 1),
                    np.repeat(nfl[:, None], len(weeks), 1), role)  # fmt: skip
    per = sum(PER_TEAM.values())
    cfg = Config([np.arange(t * per, (t + 1) * per) for t in range(n_teams)],
                 np.arange(n_teams * per, n))  # fmt: skip
    return league, inp, cfg


@pytest.fixture(scope="module")
def outcome():
    league, inp, cfg = synthetic()
    return league, inp, cfg, simulate_players(inp, placeholder_tables(), 2000, 11)


def test_same_seed_same_outcomes_and_different_seed_differs():
    league, inp, cfg = synthetic()
    t = placeholder_tables()
    a = simulate_players(inp, t, 300, 5)
    b = simulate_players(inp, t, 300, 5)
    c = simulate_players(inp, t, 300, 6)
    assert np.array_equal(a.points, b.points) and np.array_equal(a.known_out, b.known_out)
    assert not np.array_equal(a.points, c.points)


def test_probabilities_sum_to_spots(outcome):
    league, inp, cfg, out = outcome
    res = simulate_league(out, inp, league, cfg)
    assert res.playoff_prob.sum() == pytest.approx(league.playoff_teams)
    assert res.title_prob.sum() == pytest.approx(1.0)
    assert ((res.playoff_prob >= 0) & (res.playoff_prob <= 1)).all()
    assert not np.isnan(res.team_points).any()
    # every champion made the playoffs
    assert (res.made_playoffs | ~res.champion).all()


def test_byes_score_zero_and_points_are_skewed(outcome):
    league, inp, cfg, out = outcome
    for w in range(len(inp.weeks)):
        bye = inp.p_play[:, w] == 0
        assert (out.points[:, w, bye] == 0).all()
        assert out.known_out[:, w, bye].all()
    wr = out.points[:, :, inp.pos == 2]
    played = wr[wr > 0]
    assert np.mean(played) > np.median(played)  # right-skewed weekly scores


def test_injuries_persist_across_weeks(outcome):
    league, inp, cfg, out = outcome
    ko = out.known_out & (inp.p_play.T[None, :, :] > 0)  # out for injury, not a bye
    # someone hurt in week w is often still out in week w+1
    persist = (ko[:, 1:-1] & ko[:, 2:]).sum() / max(ko[:, 1:-1].sum(), 1)
    assert 0.2 < persist < 0.9
    assert ko[:, 0].sum() == 0  # nobody starts the sim injured (no statuses given)


def test_same_game_correlation(outcome):
    league, inp, cfg, out = outcome
    # find a QB and a WR sharing an NFL team in week 0, both playing
    qbs = np.flatnonzero((inp.pos == 0) & (inp.p_play[:, 0] > 0))
    for q in qbs:
        wrs = np.flatnonzero((inp.pos == 2) & (inp.team[:, 0] == inp.team[q, 0]))
        if len(wrs):
            w = wrs[0]
            both = (out.points[:, 0, q] > 0) & (out.points[:, 0, w] > 0)
            r = np.corrcoef(out.points[both, 0, q], out.points[both, 0, w])[0, 1]
            assert r > 0.1
            return
    pytest.skip("no QB-WR teammates in this synthetic draw")


def test_bracket_order_and_rounds():
    assert bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    assert bracket_order(4) == [1, 4, 2, 3]
    league, _, _ = synthetic(playoff_teams=6)
    assert playoff_rounds(league) == [[15], [16], [17]]
    league, _, _ = synthetic(playoff_teams=4, round_type=1)
    assert playoff_rounds(league) == [[15], [16, 17]]
    league, _, _ = synthetic(playoff_teams=4, round_type=2)
    assert playoff_rounds(league) == [[15, 16], [17, 18]]


def test_six_team_bracket_gives_top_two_seeds_byes(outcome):
    league, inp, cfg, out = outcome
    res = simulate_league(out, inp, league, cfg)
    # a 1 or 2 seed never loses in round one, so they win the title more often than 3-6
    s = res.seed
    top = res.champion[s <= 2].mean()
    mid = res.champion[(s >= 3) & (s <= 6)].mean()
    assert top > mid


def test_two_week_rounds_sum_both_weeks():
    league, inp, cfg = synthetic(playoff_teams=2, round_type=1)  # one 2-week final
    out = simulate_players(inp, placeholder_tables(), 500, 1)
    res = simulate_league(out, inp, league, cfg)
    col = {w: i for i, w in enumerate(inp.weeks)}
    final = res.team_points[:, :, [col[15], col[16]]].sum(axis=2)
    finalists = np.argsort(res.seed, axis=1)[:, :2]
    pa = np.take_along_axis(final, finalists[:, :1], 1)[:, 0]
    pb = np.take_along_axis(final, finalists[:, 1:2], 1)[:, 0]
    winner = np.where(pa >= pb, finalists[:, 0], finalists[:, 1])
    assert (res.champion[np.arange(500), winner]).all()


def test_median_wins_add_one_win_per_top_half_team():
    league, inp, cfg = synthetic(median=True, current_week=14)  # one regular week left
    out = simulate_players(inp, placeholder_tables(), 200, 2)
    res = simulate_league(out, inp, league, cfg)
    base = np.array([t.wins for t in league.teams])
    # 6 head-to-head wins + 6 median wins handed out among 12 teams
    assert (res.wins - base).sum(axis=1) == pytest.approx(np.full(200, 12.0))


def test_points_for_breaks_ties():
    league, inp, cfg = synthetic(current_week=15)  # regular season over
    for i, t in enumerate(league.teams):
        t.wins = 7
        t.points_for = 1000.0 + i  # last team has the most points
    out = simulate_players(inp, placeholder_tables(), 50, 3)
    res = simulate_league(out, inp, league, cfg)
    assert (res.seed[:, -1] == 1).all() and (res.seed[:, 0] == 12).all()


def test_paired_delta_is_antisymmetric():
    a = np.array([1.0, 0.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 1.0, 0.0])
    d1, s1 = paired_delta(a, b)
    d2, s2 = paired_delta(b, a)
    assert d1 == -d2 and s1 == s2


def test_fixture_league_end_to_end():
    ctx = engine.prepare("1400221038771437568", src=fake_sources(), build_history=False)
    out = simulate_players(ctx.inputs, ctx.tables, 500, 7)
    res = simulate_league(out, ctx.inputs, ctx.league, ctx.base)
    assert ctx.league.num_teams == 14
    assert res.playoff_prob.sum() == pytest.approx(8.0)
    assert res.title_prob.sum() == pytest.approx(1.0)
    weekly = res.team_points[:, :, : ctx.league.last_regular_week - ctx.league.current_week + 1]
    assert 70 < weekly.mean() < 160  # plausible PPR team scores


@pytest.mark.slow
def test_12_team_10k_sims_under_20_seconds():
    league, inp, cfg = synthetic(n_teams=12)
    t0 = time.perf_counter()
    out = simulate_players(inp, placeholder_tables(), 10_000, 7)
    simulate_league(out, inp, league, cfg)
    assert time.perf_counter() - t0 < 20


@pytest.mark.slow
def test_14_team_fixture_league_10k_sims_under_20_seconds():
    ctx = engine.prepare("1400221038771437568", src=fake_sources(), build_history=False)
    t0 = time.perf_counter()
    out = simulate_players(ctx.inputs, ctx.tables, 10_000, 7)
    simulate_league(out, ctx.inputs, ctx.league, ctx.base)
    assert time.perf_counter() - t0 < 20


def test_draws_do_not_depend_on_who_else_is_simulated():
    from dataclasses import replace

    league, inp, cfg = synthetic()
    t = placeholder_tables()
    keep = np.arange(1, len(inp.ids))  # drop player 0 from the universe
    small = replace(inp, ids=[inp.ids[i] for i in keep], pos=inp.pos[keep], mean=inp.mean[keep],
                    p_play=inp.p_play[keep], game=inp.game[keep], team=inp.team[keep],
                    role=np.full_like(inp.role[keep], -1))  # fmt: skip
    big = replace(inp, role=np.full_like(inp.role, -1))  # no copula: draws must match exactly
    a = simulate_players(big, t, 200, 9)
    b = simulate_players(small, t, 200, 9)
    assert np.array_equal(a.points[:, :, keep], b.points)


def test_injury_clock_stops_on_a_bye():
    from dataclasses import replace

    league, inp, cfg = synthetic()
    t = placeholder_tables()
    t.hazard = np.zeros_like(t.hazard)
    i = 0
    p_play = inp.p_play.copy()
    p_play[i] = 1.0
    p_play[i, 1] = 0.0  # bye in sim week 1
    game = inp.game.copy()
    game[i, 1] = -1
    mean = inp.mean.copy()
    mean[i, 1] = 0.0
    out_weeks = np.zeros(len(inp.ids), int)
    out_weeks[i] = 1  # misses one team game
    inp2 = replace(inp, p_play=p_play, game=game, mean=mean, out_weeks=out_weeks)
    out = simulate_players(inp2, t, 100, 1)
    assert out.known_out[:, 0, i].all()  # out week 0
    assert out.known_out[:, 1, i].all()  # bye
    assert not out.known_out[:, 2, i].any()  # back after one *game*


def test_playoffs_already_started_use_real_scores():
    league, inp, cfg = synthetic(playoff_teams=4, current_week=16)
    for i, t in enumerate(league.teams):
        t.wins, t.points_for = 10 - i, 1500.0 - i  # seeds 1..12 in order
    # week 15 (semifinals) already played: seeds 3 and 4 won big
    league.results[15] = {
        t.roster_id: (300.0 if i in (2, 3) else 50.0) for i, t in enumerate(league.teams)
    }
    out = simulate_players(inp, placeholder_tables(), 300, 4)
    res = simulate_league(out, inp, league, cfg)
    # bracket order [1,4,2,3]: seed 4 beat 1, seed 3 beat 2 → only seeds 3/4 can win
    assert res.champion[:, [2, 3]].all(axis=1).sum() == 0
    assert res.champion[:, [2, 3]].any(axis=1).all()


def test_no_playoffs_means_no_champion():
    league, inp, cfg = synthetic(playoff_teams=0)
    out = simulate_players(inp, placeholder_tables(), 50, 2)
    res = simulate_league(out, inp, league, cfg)
    assert not res.champion.any() and not res.made_playoffs.any()


def test_persistence_links_a_players_weeks_without_changing_any_one_week():
    from dataclasses import replace

    _, inp, _ = synthetic()
    inp = replace(inp, role=np.full_like(inp.role, -1), p_play=np.ones_like(inp.p_play))
    t0 = replace(placeholder_tables(), hazard=np.zeros(len(POSITIONS)))  # no injuries
    t1 = replace(t0, persist=np.full(len(POSITIONS), 0.3))
    a = simulate_players(inp, t0, 4000, 5).points
    b = simulate_players(inp, t1, 4000, 5).points
    rbs = np.flatnonzero(inp.pos == POSITIONS.index("RB"))

    def week_corr(x):  # mean correlation of a player's week 1 and week 2 scores
        return np.mean([np.corrcoef(x[:, 0, i], x[:, 1, i])[0, 1] for i in rbs])

    assert abs(week_corr(a)) < 0.05
    assert 0.2 < week_corr(b) < 0.4  # 0.3 in normal-score space, a bit less after the floor
    # each week's spread is the same; only the season total gets wider
    assert np.allclose(a.std(0)[0, rbs], b.std(0)[0, rbs], rtol=0.08)
    assert (b.sum(1)[:, rbs].std(0) > 1.2 * a.sum(1)[:, rbs].std(0)).all()
