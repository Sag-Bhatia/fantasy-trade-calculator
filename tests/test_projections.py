"""projections.py: weighted blend of stat lines, weights file, build() with fake sources."""

from __future__ import annotations

import json

import polars as pl
import pytest

from ffcalc import projections
from ffcalc.projections import LINE_SCHEMA, blend
from ffcalc.scoring import STANDARD_PPR, points
from ffcalc.sources import Sources
from tests.fakes import FakeSleeper


def _lines(rows: list[tuple]) -> pl.DataFrame:
    """rows: (source, week, sleeper_id, stat, value); season fixed at 2025."""
    return pl.DataFrame(
        [(s, 2025, w, pid, st, float(v)) for s, w, pid, st, v in rows],
        schema=LINE_SCHEMA,
        orient="row",
    )


def _pos(**kw: str) -> pl.DataFrame:
    return pl.DataFrame({"sleeper_id": list(kw), "position": list(kw.values())})


def _val(df: pl.DataFrame, pid: str, stat: str, week: int = 1) -> float:
    return df.filter(
        (pl.col("sleeper_id") == pid) & (pl.col("stat") == stat) & (pl.col("week") == week)
    )["value"].item()


# ---------------------------------------------------------------------- blend


def test_blend_weighted_mean_per_position():
    lines = _lines(
        [
            ("sleeper", 1, "w1", "rec_yd", 100),
            ("espn", 1, "w1", "rec_yd", 60),
            ("sleeper", 1, "r1", "rec_yd", 100),
            ("espn", 1, "r1", "rec_yd", 60),
        ]
    )
    pos = pl.DataFrame({"sleeper_id": ["w1", "r1"], "position": ["WR", "RB"]})
    out = blend(lines, pos, {"WR": {"sleeper": 3, "espn": 1}})
    assert _val(out, "w1", "rec_yd") == pytest.approx((3 * 100 + 60) / 4)
    # no RB weights -> equal weights
    assert _val(out, "r1", "rec_yd") == pytest.approx(80.0)
    assert (out["n_sources"] == 2).all()


def test_blend_empty_weights_is_plain_mean():
    lines = _lines(
        [
            ("sleeper", 1, "w1", "rec", 6),
            ("espn", 1, "w1", "rec", 4),
            ("inhouse", 1, "w1", "rec", 5),
        ]
    )
    out = blend(lines, _pos(w1="WR"), {})
    assert out.height == 1 and out["value"].item() == pytest.approx(5.0)
    assert out["n_sources"].item() == 3


def test_blend_single_source_player_keeps_that_line():
    lines = _lines(
        [
            ("sleeper", 1, "both", "rec_yd", 100),
            ("espn", 1, "both", "rec_yd", 60),
            ("espn", 1, "solo", "rec_yd", 44),
            ("espn", 1, "solo", "rec", 5),
        ]
    )
    out = blend(lines, _pos(both="WR", solo="WR"), {"WR": {"sleeper": 3, "espn": 1}})
    # renormalized over the sources present: espn alone gets weight 1/1, not 1/4
    assert _val(out, "solo", "rec_yd") == 44 and _val(out, "solo", "rec") == 5
    assert out.filter(pl.col("sleeper_id") == "solo")["n_sources"].to_list() == [1, 1]
    assert out.filter(pl.col("sleeper_id") == "both")["n_sources"].to_list() == [2]


def test_blend_stat_missing_from_a_present_source_counts_as_zero():
    lines = _lines(
        [
            ("sleeper", 1, "w1", "rec_yd", 100),
            ("sleeper", 1, "w1", "rec_td", 1),  # espn projects rec_td, just not for w1
            ("espn", 1, "w1", "rec_yd", 60),
            ("espn", 1, "w2", "rec_td", 0.5),
        ]
    )
    out = blend(lines, _pos(w1="WR", w2="WR"), {"WR": {"sleeper": 3, "espn": 1}})
    assert _val(out, "w1", "rec_td") == pytest.approx(3 * 1 / 4)  # (3*1 + 1*0) / 4


def test_blend_stat_a_source_never_models_uses_the_others():
    lines = _lines(
        [
            ("sleeper", 1, "w1", "rec_yd", 100),
            ("sleeper", 1, "w1", "bonus_rec_yd_100", 0.4),  # espn has no bonus keys at all
            ("espn", 1, "w1", "rec_yd", 60),
        ]
    )
    out = blend(lines, _pos(w1="WR"), {"WR": {"sleeper": 3, "espn": 1}})
    assert _val(out, "w1", "bonus_rec_yd_100") == pytest.approx(0.4)
    assert _val(out, "w1", "rec_yd") == pytest.approx((300 + 60) / 4)
    assert set(out["n_sources"].to_list()) == {2}


def test_blend_weeks_and_players_are_independent():
    lines = _lines(
        [
            ("sleeper", 1, "w1", "rec", 6),
            ("espn", 1, "w1", "rec", 4),
            ("sleeper", 2, "w1", "rec", 8),  # week 2: only one source
            ("espn", 1, "w2", "rec", 1),
        ]
    )
    out = blend(lines, _pos(w1="WR", w2="WR"), {})
    assert _val(out, "w1", "rec", week=1) == pytest.approx(5.0)
    assert _val(out, "w1", "rec", week=2) == 8.0
    assert out.filter(pl.col("week") == 2)["n_sources"].item() == 1
    assert _val(out, "w2", "rec") == 1.0


def test_blend_unknown_position_gets_equal_weights():
    lines = _lines([("sleeper", 1, "ghost", "rec", 6), ("espn", 1, "ghost", "rec", 2)])
    out = blend(lines, _pos(other="WR"), {"WR": {"sleeper": 9, "espn": 1}})
    assert out["value"].item() == pytest.approx(4.0)


def test_blend_leaves_inputs_untouched():
    lines = _lines([("sleeper", 1, "w1", "rec", 6), ("espn", 1, "w1", "rec", 2)])
    before = lines.clone()
    blend(lines, _pos(w1="WR"), {"WR": {"sleeper": 1, "espn": 3}})
    assert lines.equals(before)


# -------------------------------------------------------------- weights file


def test_load_weights_missing_or_corrupt_means_equal_weights():
    assert projections.load_weights() == {}
    path = projections.weights_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json")
    assert projections.load_weights() == {}


def test_load_weights_reads_backtest_file():
    path = projections.weights_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"WR": {"sleeper": 2.0, "espn": 1.0}}))
    assert projections.load_weights() == {"WR": {"sleeper": 2.0, "espn": 1.0}}


# ------------------------------------------------------------- build / games


def _table() -> pl.DataFrame:
    rows = FakeSleeper().projections(2025, 1)
    return pl.DataFrame(
        [(r["player_id"], r["team"], r["player"]["position"]) for r in rows],
        schema=["sleeper_id", "team", "position"],
        orient="row",
    )


def test_build_with_only_sleeper_scores_the_stat_line():
    src = Sources(sleeper=FakeSleeper(), fantasycalc=None)
    p = projections.build(src, 2025, 1, [1, 2], STANDARD_PPR, _table(), weights={})
    assert set(p.points["week"]) == {1, 2} and (p.points["n_sources"] == 1).all()
    assert p.points.select("season", "week", "sleeper_id").is_duplicated().sum() == 0
    assert (p.points["mean"] >= 0).all()
    raw = {r["player_id"]: r for r in FakeSleeper().projections(2025, 1)}
    row = p.points.filter((pl.col("sleeper_id") == "5859") & (pl.col("week") == 1))
    assert row["mean"].item() == pytest.approx(points(raw["5859"]["stats"], STANDARD_PPR), abs=1e-3)
    # by_source carries the same number, tagged with its source
    one = p.by_source.filter((pl.col("sleeper_id") == "5859") & (pl.col("week") == 1))
    assert one["source"].to_list() == ["sleeper"]
    assert one["points"].item() == pytest.approx(row["mean"].item())
    assert p.games.is_empty()


def test_build_drops_lines_for_a_team_on_bye():
    class Nfl:  # PHI plays wk 1 only (bye wk 2); DAL and KC play both
        def schedules(self, seasons):
            return pl.DataFrame(
                {
                    "season": [2025, 2025, 2025],
                    "week": [1, 1, 2],
                    "game_type": ["REG"] * 3,
                    "home_team": ["PHI", "KC", "DAL"],
                    "away_team": ["DAL", "LAC", "KC"],
                    "total_line": [47.5, 45.0, 44.0],
                    "spread_line": [8.5, 3.0, -1.0],
                }
            ).with_columns(pl.col("season", "week").cast(pl.Int32))

    src = Sources(sleeper=FakeSleeper(), fantasycalc=None, nflverse=Nfl())
    p = projections.build(src, 2025, 1, [1, 2], STANDARD_PPR, _table(), weights={})
    table = _table()
    team = dict(zip(table["sleeper_id"], table["team"], strict=True))
    got = {(team[pid], w) for pid, w in zip(p.points["sleeper_id"], p.points["week"], strict=True)}
    assert ("PHI", 1) in got and ("PHI", 2) not in got  # bye -> no projection at all
    assert ("DAL", 1) in got and ("DAL", 2) in got and ("KC", 2) in got
    assert ("SF", 1) not in got  # team missing from the schedule has no game
    assert set(p.by_source["week"]) == {1, 2}
    phi = p.games.filter(pl.col("team") == "PHI")
    assert phi["week"].to_list() == [1] and phi["implied"].item() == pytest.approx(28.0)
