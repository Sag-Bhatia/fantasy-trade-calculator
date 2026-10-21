"""history.py: player-week table, residual / correlation / injury / status tables (spec §4, §6)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ffcalc import history
from ffcalc.history import (
    SHRINK,
    assign_roles,
    build_tables,
    correlation_tables,
    injury_tables,
    load_tables,
    player_weeks,
    residual_tables,
    save_tables,
    status_tables,
)
from ffcalc.models import POSITIONS, ROLES
from ffcalc.scoring import STANDARD_PPR, points
from ffcalc.simulate import NPOS, NROLES, QGRID
from tests.fakes import FakeNflverse, FakeSleeper

WEEKS = range(1, 5)
SEASONS = [2025]


@pytest.fixture(scope="module")
def sleeper() -> FakeSleeper:
    return FakeSleeper()


@pytest.fixture
def pw(sleeper) -> pl.DataFrame:
    return player_weeks(sleeper, STANDARD_PPR, SEASONS, weeks=WEEKS)


def _raw(sleeper, kind: str) -> dict[tuple[str, int], dict]:
    fetch = sleeper.projections if kind == "proj" else sleeper.stats
    return {(r["player_id"], w): r for w in WEEKS for r in fetch(2025, w)}


# ---------------------------------------------------------------- player_weeks


def test_one_row_per_projected_or_played_player_week(sleeper, pw):
    assert pw.select("season", "week", "sleeper_id").is_duplicated().sum() == 0
    proj = {
        k
        for k, r in _raw(sleeper, "proj").items()
        if (r["player"] or {}).get("position") in POSITIONS and r["stats"].get("pts_ppr", 0) >= 0.5
    }
    played = {
        k
        for k, r in _raw(sleeper, "stats").items()
        if (r["player"] or {}).get("position") in POSITIONS
        and (r["stats"].get("gp") or (r["player"]["position"] == "DEF" and r.get("game_id")))
    }
    got = set(zip(pw["sleeper_id"], pw["week"], strict=True))
    assert got == proj | played
    assert set(pw["position"]) == set(POSITIONS)
    # played but never projected -> proj is 0, not a dropped row
    only_played = pw.filter(pl.col("proj") == 0)
    assert set(zip(only_played["sleeper_id"], only_played["week"], strict=True)) == played - proj
    assert only_played["played"].all() and only_played.height > 0


def test_actual_is_points_of_raw_stat_line(sleeper, pw):
    raw = _raw(sleeper, "stats")
    played = pw.filter(pl.col("played"))
    assert played.height > 1000
    for r in played.iter_rows(named=True):
        want = points(raw[(r["sleeper_id"], r["week"])]["stats"], STANDARD_PPR)
        assert r["actual"] == pytest.approx(want, abs=1e-3), r


def test_known_players_exact(pw):
    # A.J. Brown wk 1: 1 rec for 8 yd -> 1.8 pts (Sleeper's own pts_ppr agrees)
    row = pw.filter((pl.col("sleeper_id") == "5859") & (pl.col("week") == 1)).row(0, named=True)
    assert row["actual"] == pytest.approx(1.8, abs=1e-3)
    assert (row["position"], row["team"], row["opponent"]) == ("WR", "PHI", "DAL")
    assert row["game_id"] == "202510126" and row["played"] and row["team_played"]


def test_unplayed_rows(pw):
    # projected but no gp in the stat feed: K 8670 wk1, WR 11617 wk2, RB 11643 wk3
    bad = pw.filter(~pl.col("played"))
    assert set(zip(bad["sleeper_id"], bad["week"], strict=True)) == {
        ("8670", 1),
        ("11617", 2),
        ("11643", 3),
    }
    assert (bad["actual"] == 0.0).all()
    assert bad["team_played"].all()  # his team played; he didn't


def test_defense_scores_points_allowed_bucket(sleeper, pw):
    raw = _raw(sleeper, "stats")
    d = pw.filter((pl.col("position") == "DEF") & pl.col("played"))
    assert d.height == 128
    bucketed = 0
    for r in d.iter_rows(named=True):
        stats = raw[(r["sleeper_id"], r["week"])]["stats"]
        bucket = [k for k in stats if k.startswith("pts_allow_")]
        assert len(bucket) == 1
        no_bucket = points({k: v for k, v in stats.items() if k != bucket[0]}, STANDARD_PPR)
        full = points(stats, STANDARD_PPR)
        assert r["actual"] == pytest.approx(full, abs=1e-3)
        assert full - no_bucket == pytest.approx(STANDARD_PPR[bucket[0]])
        bucketed += STANDARD_PPR[bucket[0]] != 0
    assert bucketed > 0  # the bucket really moved some scores


def test_player_weeks_skips_unavailable_weeks(sleeper):
    out = player_weeks(sleeper, STANDARD_PPR, SEASONS, weeks=[1, 9])  # wk 9 has no fixture
    assert set(out["week"]) == {1}
    with pytest.raises(RuntimeError):
        player_weeks(sleeper, STANDARD_PPR, [2019], weeks=[1])


# ------------------------------------------------------------ residual_tables


def test_residual_tables_shapes_and_order(pw):
    edges, resid_q, floor, scored = residual_tables(pw)
    assert edges.shape == (NPOS, history.TIERS - 1)
    assert resid_q.shape == (NPOS, history.TIERS, QGRID)
    assert floor.shape == (NPOS,)
    assert (np.diff(edges, axis=1) > 0).all()  # tier edges increase for every position
    assert (np.diff(resid_q, axis=2) >= 0).all()  # quantiles never decrease
    assert np.isfinite(resid_q).all() and scored["z"].is_finite().all()
    rel = history._relevant(pw).filter(pl.col("played"))
    for p, name in enumerate(POSITIONS):
        actual = rel.filter(pl.col("position") == name)["actual"].to_numpy()
        assert floor[p] <= np.median(actual)


def test_residual_tier_edges_are_projection_quantiles(pw):
    edges, _, _, _ = residual_tables(pw)
    qb = history._relevant(pw).filter((pl.col("position") == "QB") & pl.col("played"))["proj"]
    assert edges[0, 1] == pytest.approx(qb.quantile(0.4, interpolation="linear"))


def test_residual_tables_small_sample_falls_back_to_wide_normal(pw):
    tiny = pw.filter(pl.col("position") == "QB").head(10)
    edges, resid_q, floor, _ = residual_tables(tiny)
    assert (edges[0] == 0).all() and floor[0] == 0
    row = resid_q[0, 0]
    assert (np.diff(row) >= 0).all() and row[QGRID // 2] == pytest.approx(0, abs=0.2)
    assert row[-1] > 5  # roughly +2.9 sd of a 6-point normal


# --------------------------------------------------------------- assign_roles


def _roster_frame() -> pl.DataFrame:
    rows = []

    def add(team, pos, projs, week=1):
        rows.extend((week, team, pos, f"{team}{pos}{i}w{week}", p) for i, p in enumerate(projs))

    for wk in (1, 2):
        add("A", "QB", [10, 20], wk)
        add("A", "RB", [5, 15, 10], wk)
        add("A", "WR", [6, 12, 9, 3], wk)
        add("A", "TE", [4, 7], wk)
        add("A", "K", [8], wk)
        add("A", "DEF", [9], wk)
    add("B", "WR", [1.0], 1)
    return pl.DataFrame(
        rows, schema=["week", "team", "position", "sleeper_id", "proj"], orient="row"
    )


def test_assign_roles_by_projection_rank():
    out = assign_roles(_roster_frame(), ["week"])
    got = {r["sleeper_id"]: ROLES[r["role"]] if r["role"] >= 0 else None for r in out.to_dicts()}
    assert got["AQB1w1"] == "QB1" and got["AQB0w1"] is None  # 20 beats 10
    assert got["ARB1w1"] == "RB1" and got["ARB2w1"] == "RB2" and got["ARB0w1"] is None
    assert got["AWR1w1"] == "WR1" and got["AWR2w1"] == "WR2" and got["AWR0w1"] == "WR3"
    assert got["AWR3w1"] is None  # fourth WR -> -1
    assert got["ATE1w1"] == "TE1" and got["ATE0w1"] is None
    assert got["AK0w1"] == "K" and got["ADEF0w1"] == "DEF"
    assert got["BWR0w1"] == "WR1"  # ranks are per team
    assert out.filter(pl.col("sleeper_id") == "AWR3w1")["role"].item() == -1
    # weeks are ranked independently when `week` is in `by`
    assert got["AQB1w2"] == "QB1"


def test_assign_roles_without_by_key_collapses_weeks():
    out = assign_roles(_roster_frame(), [])  # both weeks pooled: two QB1 candidates rank 1..4
    qb = out.filter(pl.col("position") == "QB")
    assert (qb["role"] >= 0).sum() == 1


# --------------------------------------------------------- correlation_tables


def test_correlation_tables_symmetric_bounded(pw):
    _, _, _, scored = residual_tables(pw)
    same, opp, n_pairs = correlation_tables(scored, pw)
    assert same.shape == opp.shape == (NROLES, NROLES)
    assert n_pairs > 1000
    for m in (same, opp):
        np.testing.assert_allclose(m, m.T)
        assert m.min() >= -0.9 and m.max() <= 0.9
    np.testing.assert_array_equal(np.diag(same), 0.0)
    assert np.abs(same).max() > 0 and np.abs(opp).max() > 0
    qb1, wr1 = ROLES.index("QB1"), ROLES.index("WR1")
    assert same[qb1, wr1] > 0  # QB and his own WR1 move together


def test_correlation_shrinks_toward_zero_by_sample_size():
    n = SHRINK  # 150 identical pairs, raw r = 1 -> shrunk to n / (n + SHRINK) = 0.5
    pw = pl.DataFrame(
        {
            "season": [2025] * (2 * n),
            "week": [w for w in range(n) for _ in range(2)],
            "sleeper_id": ["qb", "wr"] * n,
            "team": ["A"] * (2 * n),
            "position": ["QB", "WR"] * n,
            "proj": [20.0, 12.0] * n,
        }
    )
    scored = pw.select("season", "week", "sleeper_id", "team", pl.lit(1.0).alias("z")).with_columns(
        game_id=pl.col("week").cast(pl.String)
    )
    same, opp, n_pairs = correlation_tables(scored, pw)
    qb1, wr1 = ROLES.index("QB1"), ROLES.index("WR1")
    assert n_pairs == n
    assert same[qb1, wr1] == pytest.approx(0.5) and same[wr1, qb1] == pytest.approx(0.5)
    assert (opp == 0).all()


# ------------------------------------------------------------- injury_tables


def _pw(rows: list[tuple]) -> pl.DataFrame:
    """rows: (sleeper_id, position, team, season, week, played, team_played, proj)."""
    return pl.DataFrame(
        rows,
        schema=[
            "sleeper_id",
            "position",
            "team",
            "season",
            "week",
            "played",
            "team_played",
            "proj",
        ],
        orient="row",
    )


def _team_games(team: str, weeks, season: int = 2025) -> list[tuple]:
    """A teammate nobody cares about (proj 1) whose rows say which weeks `team` played."""
    return [(f"fill_{team}", "K", team, season, w, True, True, 1.0) for w in weeks]


def _wr(pid, team, season, weeks_played, weeks_rows, proj=10.0) -> list[tuple]:
    return [(pid, "WR", team, season, w, w in weeks_played, True, proj) for w in weeks_rows]


def test_injury_event_duration_and_hazard():
    # plays 1-2, misses 3-4 (explicit played=False rows), plays 5
    x = _wr("x", "T", 2025, {1, 2, 5}, range(1, 6))
    hazard, pmf = injury_tables(_pw(x + _team_games("T", range(1, 6))))
    wr = POSITIONS.index("WR")
    # chances: wk1 (next game played), wk2 (next game missed) -> 1 event / 2 chances
    assert hazard[wr] == pytest.approx(0.5)
    assert pmf[wr, 1] == 1.0 and pmf[wr].sum() == 1.0  # exactly 2 games missed
    other = np.delete(np.arange(NPOS), wr)
    assert (hazard[other] == 0).all()
    assert (pmf[other, 0] == 1.0).all()  # no data -> default one-game pmf


def test_missing_rows_count_as_missed_games():
    # Sleeper drops ruled-out players from its projections, so a missed week is a MISSING
    # row (real shape: Justin Fields, rows wks 1,2,4 only, Out wk 3)
    qb = [("f", "QB", "NYJ", 2025, w, True, True, 18.0) for w in (1, 2, 4)]
    hazard, pmf = injury_tables(_pw(qb + _team_games("NYJ", [1, 2, 3, 4])))
    i = POSITIONS.index("QB")
    assert hazard[i] == pytest.approx(0.5)  # chances wk1 (played wk2) and wk2 (missed wk3)
    assert pmf[i, 0] == 1.0


def test_bye_weeks_are_not_misses():
    y = _wr("y", "T", 2025, {1, 3}, [1, 3])
    bye_row = [("y", "WR", "T", 2025, 2, False, False, 10.0)]  # row exists but team_played False
    games = _team_games("T", [1, 3])  # nobody on T played wk 2
    hazard, _ = injury_tables(_pw(y + bye_row + games))
    assert hazard[POSITIONS.index("WR")] == 0.0
    # same player, but his team did play wk 2 -> one chance (wk 1), one missed game
    hazard, pmf = injury_tables(_pw(y + _team_games("T", [1, 2, 3])))
    assert hazard[POSITIONS.index("WR")] == 1.0
    assert pmf[POSITIONS.index("WR"), 0] == 1.0


def test_injury_tables_ignore_irrelevant_players_and_cap_duration():
    low = _wr("x", "T", 2025, {1, 2, 5}, range(1, 6), proj=1.0)  # below MIN_PROJ["WR"]
    hazard, _ = injury_tables(_pw(low + _team_games("T", range(1, 6))))
    assert hazard.sum() == 0
    rb = [("z", "RB", "T", 2025, w, w in (1, 12), True, 10.0) for w in range(1, 13)]  # out 10
    _, pmf = injury_tables(_pw(rb + _team_games("T", range(1, 13))))
    assert pmf[POSITIONS.index("RB"), history.MAX_OUT - 1] == 1.0


def test_injury_tables_do_not_chain_across_seasons():
    # 2025 wk 18 is his only game that season; 2026 wk 1 he is absent. No chance, no event.
    rows = _wr("s", "T", 2025, {18}, [18]) + _team_games("T", [18]) + _team_games("T", [1], 2026)
    hazard, _ = injury_tables(_pw(rows))
    assert hazard.sum() == 0.0


# -------------------------------------------------------------- status_tables


def _status_inputs():
    """Week 1: everyone plays at 10 projected points (so all are 'regular'). Week 2 has the
    game statuses; absent players are simply missing from pw (Sleeper drops them)."""
    rows: list[
        tuple
    ] = []  # sleeper_id, position, team, season, week, played, team_played, proj, actual
    inj: list[tuple] = []  # sleeper_id, status (week 2)

    def week1(pid, pos, team="T"):
        rows.append((pid, pos, team, 2025, 1, True, True, 10.0, 10.0))

    for i in range(40):  # 30 of 40 Questionable WRs play, at 80% of projection
        week1(f"w{i}", "WR")
        inj.append((f"w{i}", "Questionable"))
        if i < 30:
            rows.append((f"w{i}", "WR", "T", 2025, 2, True, True, 10.0, 8.0))
        elif i < 35:
            rows.append((f"w{i}", "WR", "T", 2025, 2, False, True, 10.0, 0.0))  # explicit miss
        # i >= 35: no row at all
    for i in range(5):  # 5 Questionable RBs, none play, no rows
        week1(f"r{i}", "RB")
        inj.append((f"r{i}", "Questionable"))
    for i in range(10):  # healthy WRs: full projection, no report ("Probable" is ignored)
        week1(f"h{i}", "WR")
        rows.append((f"h{i}", "WR", "T", 2025, 2, True, True, 10.0, 10.0))
    inj.append(("h0", "Probable"))
    for i in range(3):  # Doubtful WRs: 2 of 3 play at half projection
        week1(f"d{i}", "WR")
        inj.append((f"d{i}", "Doubtful"))
        if i < 2:
            rows.append((f"d{i}", "WR", "T", 2025, 2, True, True, 10.0, 5.0))
    for i in range(3):  # Doubtful RBs: none play
        week1(f"dr{i}", "RB")
        inj.append((f"dr{i}", "Doubtful"))
    # not regular: sat out week 1, so his wk 2 'Questionable' must not count
    rows.append(("irr", "WR", "T", 2025, 2, True, True, 10.0, 10.0))
    inj.append(("irr", "Questionable"))
    # team on a bye in week 2: a report on a bye-week player is not a missed game
    week1("bye1", "WR", "B")
    inj.append(("bye1", "Questionable"))
    pw = pl.DataFrame(
        rows,
        schema=["sleeper_id", "position", "team", "season", "week", "played", "team_played",
                "proj", "actual"],
        orient="row",
    )  # fmt: skip
    injuries = pl.DataFrame(
        [(2025, 2, f"g_{pid}", st) for pid, st in inj],
        schema=["season", "week", "gsis_id", "report_status"],
        orient="row",
    )
    ids = [r[0] for r in rows] + [pid for pid, _ in inj]
    table = pl.DataFrame({"gsis_id": [f"g_{p}" for p in ids], "sleeper_id": ids}).unique()
    return pw, injuries, table


def test_status_play_probabilities_and_pooling():
    pw, inj, table = _status_inputs()
    play, _ = status_tables(pw, inj, table)
    wr, rb, qb = (POSITIONS.index(p) for p in ("WR", "RB", "QB"))
    q = play["Questionable"]
    assert q[wr] == pytest.approx(30 / 40)  # cell >= 30 rows: used as is; irregular/bye excluded
    assert q[rb] == pytest.approx(30 / 45)  # 5 rows < 30: pooled over all positions
    assert q[qb] == pytest.approx(30 / 45)  # empty cell: pooled too
    assert all(v == pytest.approx(2 / 6) for v in play["Doubtful"])  # small cells -> pooled
    assert play["Out"] == [0.0] * NPOS  # no data


def test_status_multiplier_is_relative_to_healthy_players():
    pw, inj, table = _status_inputs()
    _, mult = status_tables(pw, inj, table)
    wr, rb = POSITIONS.index("WR"), POSITIONS.index("RB")
    assert mult["Questionable"][wr] == pytest.approx(0.8)  # 80% of projection vs 100% healthy
    assert mult["Questionable"][rb] == pytest.approx(0.8)  # small cell -> pooled numerator and base
    assert mult["Doubtful"][wr] == pytest.approx(0.5)  # pooled cell, played at half projection
    assert mult["Out"] == [1.0] * NPOS


# --------------------------------------------------- build_tables / save / load


@pytest.fixture
def injuries_and_ids():
    nfl = FakeNflverse()
    ids = nfl.ff_playerids().with_columns(pl.col("sleeper_id").cast(pl.String))
    return nfl.injuries([2025]), ids


def test_build_tables_end_to_end(sleeper, injuries_and_ids):
    inj, ids = injuries_and_ids
    t, summary = build_tables(sleeper, STANDARD_PPR, SEASONS, injuries=inj, table=ids, weeks=WEEKS)
    assert t.tier_edges.shape == (NPOS, history.TIERS - 1)
    assert t.resid_q.shape == (NPOS, history.TIERS, QGRID)
    assert t.floor.shape == (NPOS,)
    assert t.corr_same.shape == t.corr_opp.shape == (NROLES, NROLES)
    assert t.hazard.shape == (NPOS,) and t.duration_pmf.shape == (NPOS, history.MAX_OUT)
    np.testing.assert_allclose(t.duration_pmf.sum(axis=1), 1.0)
    assert t.source == "sleeper [2025]"
    assert (t.hazard[:4] > 0).all()  # skill players do miss games in 4 real weeks
    for status in history.STATUSES:
        assert len(t.status_play[status]) == len(t.status_mult[status]) == NPOS
        assert all(0.0 <= v <= 1.0 for v in t.status_play[status])
    assert summary["player_weeks"] == player_weeks(sleeper, STANDARD_PPR, SEASONS, WEEKS).height
    assert summary["game_pairs"] > 0


def test_build_tables_without_injuries_keeps_placeholder_status(sleeper):
    t, _ = build_tables(sleeper, STANDARD_PPR, SEASONS, weeks=WEEKS)
    assert t.status_play["Questionable"] == [0.85] * NPOS


def test_save_load_round_trip(sleeper, injuries_and_ids):
    inj, ids = injuries_and_ids
    t, _ = build_tables(sleeper, STANDARD_PPR, SEASONS, injuries=inj, table=ids, weeks=WEEKS)
    assert load_tables(STANDARD_PPR, SEASONS) is None  # nothing saved yet
    save_tables(t, STANDARD_PPR, SEASONS)
    back = load_tables(STANDARD_PPR, SEASONS)
    assert back is not None and back.source == t.source
    for f in ("tier_edges", "resid_q", "floor", "corr_same", "corr_opp", "hazard", "duration_pmf"):
        np.testing.assert_allclose(getattr(back, f), getattr(t, f))
    assert back.status_play == t.status_play and back.status_mult == t.status_mult
    # a different scoring dict or season list is a different key
    assert load_tables({**STANDARD_PPR, "rec": 0.5}, SEASONS) is None
    assert load_tables(STANDARD_PPR, [2024, 2025]) is None
    # key ignores dict ordering
    assert load_tables(dict(reversed(list(STANDARD_PPR.items()))), SEASONS) is not None
