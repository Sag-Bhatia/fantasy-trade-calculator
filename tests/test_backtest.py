import polars as pl
import pytest

from ffcalc import backtest as bt


def test_brier_and_reliability():
    df = pl.DataFrame({"p": [0.9, 0.05, 0.5, 0.95], "made": [True, False, False, True]})
    assert bt.brier(df) == pytest.approx((0.01 + 0.0025 + 0.25 + 0.0025) / 4)
    rel = bt.reliability(df)
    assert rel["model said"].to_list() == ["0–10%", "50–60%", "90–100%"]
    assert rel["n"].to_list() == [1, 1, 2]
    assert rel["actually made playoffs"].to_list() == [0.0, 0.0, 1.0]


def test_baselines_use_only_earlier_games():
    pw = pl.DataFrame(
        {
            "sleeper_id": ["a"] * 5,
            "season": [2024] * 5,
            "week": [1, 2, 3, 4, 5],
            "actual": [10.0, 20.0, 30.0, 0.0, 40.0],
            "played": [True, True, True, False, True],
        }
    )
    b = bt.baselines(pw).sort("week")
    assert b["season_avg"].to_list() == [None, 10.0, 15.0, 20.0]  # weeks 1, 2, 3, 5
    assert b["last3"].to_list()[-1] == pytest.approx(20.0)  # mean of 10, 20, 30


def test_markdown_table_renders_missing_values():
    df = pl.DataFrame({"position": ["K"], "espn": [None], "sleeper": [3.76]}, strict=False)
    out = bt._md(df, {"sleeper": ".2f"})
    assert "| K | — | 3.76 |" in out


def test_snapshot_league_counts_only_earlier_weeks():
    import numpy as np

    from ffcalc.models import League, Team

    teams = [Team(1, None, "A", []), Team(2, None, "B", [])]
    sched = {1: [(1, 2)], 2: [(1, 2)], 3: [(1, 2)]}
    base = League("x", "x", 2024, 2, ["QB"], 0, 0, {}, 2, 4, 0, False, 0, 1, teams, sched, {})
    tp = np.array([[100.0, 50.0, 10.0], [90.0, 60.0, 99.0]])  # team × week
    lg = bt._snapshot_league(base, tp, 3)  # standings before week 3
    a, b = lg.teams
    assert (a.wins, a.losses, b.wins, b.losses) == (1, 1, 1, 1)
    assert a.points_for == pytest.approx(150.0) and lg.current_week == 3


def test_draft_does_not_depend_on_row_order():
    import numpy as np

    positions = ["QB"] * 30 + ["RB"] * 60 + ["WR"] * 60 + ["TE"] * 30 + ["K"] * 20 + ["DEF"] * 20
    pool = pl.DataFrame(
        {
            "sleeper_id": [f"p{i:03d}" for i in range(len(positions))],
            "position": positions,
            "w1": [5.0 + (i * 7919 % 97) / 5 for i in range(len(positions))],
        }
    )
    a = bt.draft(pool, np.random.default_rng(3))
    b = bt.draft(pool.reverse(), np.random.default_rng(3))
    assert a == b and all(len(r) == len(bt.SLOTS) + bt.BENCH for r in a)
