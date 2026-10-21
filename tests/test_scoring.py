import polars as pl
import pytest

from ffcalc.scoring import STANDARD_PPR, complete, points, score_lines

WR = {"rec": 5, "rec_yd": 80, "rec_td": 1, "bonus_rec_wr": 5, "rec_tgt": 8}


def test_points_is_dot_product_over_shared_keys():
    assert points(WR, STANDARD_PPR) == pytest.approx(5 + 8 + 6)


def test_te_premium_uses_league_setting():
    assert points(WR | {"bonus_rec_te": 5}, STANDARD_PPR | {"bonus_rec_te": 0.5}) == pytest.approx(
        21.5
    )


def test_kicker_buckets_derived_only_when_missing():
    proj = {"fgm_50p": 0.2, "fgmiss_40_49": 0.1, "fgmiss_50p": 0.1}
    assert complete(proj)["fgmiss"] == pytest.approx(0.2)
    assert complete(proj)["fgm_50_59"] == pytest.approx(0.2)
    actual = {"fgmiss": 2, "fgmiss_50p": 1, "fgm_50_59": 1, "fgm_50p": 1}
    assert complete(actual) == actual  # never double-counted


def test_score_lines_matches_points():
    stats = WR | {"fgmiss_50p": 1.0, "fgm_50p": 1.0}
    lines = pl.DataFrame(
        {"sleeper_id": ["1"] * len(stats), "week": [7] * len(stats),
         "stat": list(stats), "value": [float(v) for v in stats.values()]}
    )  # fmt: skip
    sc = {"rec": 1, "rec_yd": 0.1, "rec_td": 6, "fgmiss": -1, "fgm_50_59": 5}
    out = score_lines(lines, sc)
    assert out.height == 1
    assert out["points"][0] == pytest.approx(points(stats, sc))


def test_sixty_yard_field_goal_not_double_counted():
    # Sleeper actuals: a 61-yard make sets fgm_50p and fgm_60p but not fgm_50_59
    line = {"fgm_50p": 1.0, "fgm_60p": 1.0}
    assert points(line, {"fgm_50_59": 5, "fgm_60p": 6}) == pytest.approx(6)
