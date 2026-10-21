"""model.py: Vegas implied totals, as-of features, ridge training and prediction (spec §4)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ffcalc import model
from ffcalc.model import (
    MIN_GAMES,
    STATS,
    TARGETS,
    USAGE,
    actual_wide,
    features_as_of,
    implied_totals,
    predict,
    train,
)
from tests.fakes import FakeNflverse, FakeSleeper

# ------------------------------------------------------------- implied_totals


def test_implied_totals_sum_to_total_line():
    sched = FakeNflverse().schedules([2025])
    reg = sched.filter(pl.col("game_type") == "REG")
    imp = implied_totals(sched)
    assert imp.height == 2 * reg.height  # one row per team per game, playoffs dropped
    # PHI (home) vs DAL, week 1: total 47.5, home favoured by 8.5 -> 28.0 / 19.5
    wk1 = imp.filter(pl.col("week") == 1)
    phi = wk1.filter(pl.col("team") == "PHI").row(0, named=True)
    dal = wk1.filter(pl.col("team") == "DAL").row(0, named=True)
    assert (phi["opponent"], dal["opponent"]) == ("DAL", "PHI")
    assert phi["implied"] == pytest.approx(28.0) and dal["implied"] == pytest.approx(19.5)


def test_implied_totals_pairs_add_up_for_every_game():
    sched = FakeNflverse().schedules([2025])
    imp = implied_totals(sched)
    reg = sched.filter(pl.col("game_type") == "REG")
    fix = model.TEAM_FIX
    for g in reg.iter_rows(named=True):
        h, a = fix.get(g["home_team"], g["home_team"]), fix.get(g["away_team"], g["away_team"])
        game = imp.filter((pl.col("week") == g["week"]) & (pl.col("team") == h))
        away = imp.filter((pl.col("week") == g["week"]) & (pl.col("team") == a))
        assert game["opponent"].item() == a and away["opponent"].item() == h
        assert game["implied"].item() + away["implied"].item() == pytest.approx(g["total_line"])
        assert game["implied"].item() - away["implied"].item() == pytest.approx(g["spread_line"])


def test_implied_totals_maps_team_codes_to_sleeper():
    imp = implied_totals(FakeNflverse().schedules([2025]))
    teams = set(imp["team"])
    assert "LAR" in teams and "LA" not in teams
    # week 2: TEN (home) vs LA, total 41.5, spread -5.5 (away favoured) -> LAR 23.5, TEN 18.0
    wk2 = imp.filter(pl.col("week") == 2)
    lar = wk2.filter(pl.col("team") == "LAR").row(0, named=True)
    assert lar["opponent"] == "TEN" and lar["implied"] == pytest.approx(23.5)
    assert wk2.filter(pl.col("team") == "TEN")["implied"].item() == pytest.approx(18.0)


def test_implied_totals_edge_cases():
    sched = pl.DataFrame(
        {
            "season": [2025] * 3,
            "week": [1, 1, 19],
            "game_type": ["REG", "REG", "WC"],
            "home_team": ["WSH", "JAC", "KC"],
            "away_team": ["LVR", "LA", "BUF"],
            "total_line": [40.0, None, 50.0],
            "spread_line": [3.0, 1.0, 1.0],
        }
    )
    imp = implied_totals(sched)
    assert set(imp["team"]) == {"WAS", "LV", "JAX", "LAR"}  # playoffs dropped, codes mapped
    assert imp.filter(pl.col("team") == "WAS")["implied"].item() == pytest.approx(21.5)
    assert imp.filter(pl.col("team") == "LV")["implied"].item() == pytest.approx(18.5)
    assert imp.filter(pl.col("team") == "JAX")["implied"].item() is None  # no line -> null
    assert imp.schema["season"] == pl.Int64 and imp.schema["week"] == pl.Int64


# ---------------------------------------------------------------- actual_wide


@pytest.fixture(scope="module")
def act() -> pl.DataFrame:
    return actual_wide(FakeSleeper(), 2025, range(1, 5))


def test_actual_wide_one_row_per_played_skill_player_week(act):
    assert set(act["position"]) == {"QB", "RB", "WR", "TE"}  # no K / DEF
    assert act.select("season", "week", "sleeper_id").is_duplicated().sum() == 0
    assert set(act["week"]) == {1.0, 2.0, 3.0, 4.0}
    assert all(c in act.columns for c in STATS)


def test_actual_wide_known_player_and_snap_share(act):
    # A.J. Brown wk 1: 1 rec on 1 target for 8 yd, 55 of 63 team snaps
    r = act.filter((pl.col("sleeper_id") == "5859") & (pl.col("week") == 1)).row(0, named=True)
    assert (r["rec"], r["rec_tgt"], r["rec_yd"]) == (1.0, 1.0, 8.0)
    assert r["snap_share"] == pytest.approx(55 / 63, rel=1e-5)
    assert (act["snap_share"] >= 0).all() and (act["snap_share"] <= 1.0).all()


def test_actual_wide_skips_missing_weeks():
    out = actual_wide(FakeSleeper(), 2025, [1, 9])  # wk 9 has no fixture
    assert set(out["week"]) == {1.0}
    assert actual_wide(FakeSleeper(), 2019, [1]).is_empty()


# ------------------------------------------------------------- features_as_of


def _act_rows(rows: list[dict]) -> pl.DataFrame:
    """Hand-built actuals: every USAGE/STATS column present, default 0."""
    base = {c: 0.0 for c in {*STATS, *USAGE}}
    return pl.DataFrame(
        [{"season": 2025, "position": "WR", "team": "T", **base, **r} for r in rows]
    )


def test_features_as_of_uses_only_earlier_weeks():
    act = _act_rows(
        [
            {"sleeper_id": "a", "week": 1, "rec_tgt": 10.0},
            {"sleeper_id": "a", "week": 2, "rec_tgt": 10.0},
            {"sleeper_id": "a", "week": 3, "rec_tgt": 1000.0},  # must not leak into wk 3
            {"sleeper_id": "a", "week": 4, "rec_tgt": 2000.0},
        ]
    )
    f3 = features_as_of(act, 2025, 3).row(0, named=True)
    assert f3["games"] == 2 and f3["std_rec_tgt"] == 10.0 and f3["l3_rec_tgt"] == 10.0
    f4 = features_as_of(act, 2025, 4).row(0, named=True)
    assert f4["games"] == 3 and f4["std_rec_tgt"] == pytest.approx(340.0)
    # last-3 window drops the oldest game
    f5 = features_as_of(act, 2025, 5).row(0, named=True)
    assert f5["games"] == 4
    assert f5["l3_rec_tgt"] == pytest.approx((10 + 1000 + 2000) / 3)
    assert f5["std_rec_tgt"] == pytest.approx((10 + 10 + 1000 + 2000) / 4)


def test_features_as_of_requires_min_games_and_same_season():
    act = _act_rows(
        [
            {"sleeper_id": "one", "week": 1},  # a single prior game
            {"sleeper_id": "two", "week": 1},
            {"sleeper_id": "two", "week": 2},
            {"sleeper_id": "old", "week": 1, "season": 2024},
            {"sleeper_id": "old", "week": 2, "season": 2024},
        ]
    )
    assert MIN_GAMES == 2
    f = features_as_of(act, 2025, 3)
    assert f["sleeper_id"].to_list() == ["two"]  # other season doesn't count
    assert features_as_of(act, 2025, 1).is_empty()  # nothing before week 1


def test_features_as_of_on_real_fixture(act):
    f = features_as_of(act, 2025, 3)
    assert f.height > 100 and (f["games"] == 2).all()  # only wks 1-2 can be seen
    assert features_as_of(act, 2025, 2).is_empty()  # one game isn't enough


# --------------------------------------------------------------- train/predict

LEVELS = [2.0, 5.0, 8.0, 11.0]


def _synthetic():
    """Four players per position with a constant target level; every usage stat scales with
    the level (+ seeded noise), so more targets must predict more targets."""
    rng = np.random.default_rng(7)
    rows = []
    for pos in TARGETS:
        for k, level in enumerate(LEVELS):
            for w in range(1, 19):
                stats = {c: level * (1.0 + 0.1 * i) + rng.normal(0, 0.3)
                         for i, c in enumerate(sorted({*STATS, *USAGE}))}  # fmt: skip
                stats["rec_tgt"] = level + rng.normal(0, 0.3)
                stats["snap_share"] = 0.5
                rows.append(
                    {"season": 2025, "week": w, "sleeper_id": f"{pos}{k}", "position": pos,
                     "team": f"T{k}", **stats}
                )  # fmt: skip
    act = pl.DataFrame(rows)
    teams = [f"T{k}" for k in range(len(LEVELS))]
    implied = pl.DataFrame(
        [{"season": 2025, "week": w, "team": t, "opponent": "X", "implied": 20.0 + k}
         for w in range(1, 19) for k, t in enumerate(teams)]
    )  # fmt: skip
    return act, implied


@pytest.fixture(scope="module")
def trained():
    act, implied = _synthetic()
    return act, implied, train(act, implied, [2025])


def test_train_shapes(trained):
    _, implied, m = trained
    assert set(m["coef"]) == set(TARGETS) and all(n > 0 for n in m["n"].values())
    assert m["mean_implied"] == pytest.approx(implied["implied"].mean())
    nfeat = len(model.FEATS)
    for pos, targets in TARGETS.items():
        c = m["coef"][pos]
        assert len(c["mu"]) == len(c["sd"]) == nfeat
        assert set(c["beta"]) == set(targets) and all(
            len(b) == nfeat + 1 for b in c["beta"].values()
        )


def _predict_week(trained, week=18):
    act, implied, m = trained
    feats = features_as_of(act, 2025, week)
    games = (
        implied.filter(pl.col("week") == week)
        .join(feats.select("sleeper_id", "team"), on="team")
        .select("sleeper_id", "season", "week", "implied")
    )
    return predict(m, feats, games)


def test_predict_is_nonnegative_and_orders_by_usage(trained):
    out = _predict_week(trained)
    assert out.height > 0 and set(out["source"]) == {"inhouse"}
    assert (out["value"] > 0).all()  # clipped at 0, zeros dropped
    assert set(out["week"]) == {18} and set(out["season"]) == {2025}
    tgt = out.filter(pl.col("stat") == "rec_tgt")
    for pos in ("RB", "WR", "TE"):
        v = [tgt.filter(pl.col("sleeper_id") == f"{pos}{k}")["value"].item() for k in range(4)]
        assert v == sorted(v) and v[-1] > v[0] + 3  # more targets in -> more targets out
        assert v[0] == pytest.approx(LEVELS[0], abs=1.5)
        assert v[-1] == pytest.approx(LEVELS[-1], abs=1.5)
    # QBs have no rec_tgt target at all
    assert tgt.filter(pl.col("sleeper_id").str.starts_with("QB")).is_empty()


def test_predict_is_deterministic(trained):
    keys = ["sleeper_id", "stat"]
    assert _predict_week(trained).sort(keys).equals(_predict_week(trained).sort(keys))


def test_predict_nothing_to_predict_returns_empty_schema(trained):
    act, implied, m = trained
    feats = features_as_of(act, 2025, 18)
    games = pl.DataFrame(
        schema={"sleeper_id": pl.String, "season": pl.Int64, "week": pl.Int64,
                "implied": pl.Float64}
    )  # fmt: skip
    out = predict(m, feats, games)
    assert out.is_empty() and out.columns == [
        "source",
        "season",
        "week",
        "sleeper_id",
        "stat",
        "value",
    ]


def test_predict_null_implied_falls_back_to_mean(trained):
    act, implied, m = trained
    feats = features_as_of(act, 2025, 18)
    one = feats.filter(pl.col("sleeper_id") == "WR1")
    mk = lambda v: pl.DataFrame(  # noqa: E731
        {"sleeper_id": ["WR1"], "season": [2025], "week": [18], "implied": [v]}
    )
    null = predict(m, one, mk(None))
    mean = predict(m, one, mk(m["mean_implied"]))
    assert null.sort("stat").equals(mean.sort("stat"))
