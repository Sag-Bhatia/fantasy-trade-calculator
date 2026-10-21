import io
from pathlib import Path

import polars as pl
import pytest

from ffcalc.clients.fantasypros_csv import bucket_stats, infer_position, load
from ffcalc.ids import player_table
from tests.fakes import FakeNflverse, sample_sleeper_players

FP = Path(__file__).parent / "fixtures" / "fantasypros"


@pytest.fixture
def table():
    return player_table(sample_sleeper_players(), FakeNflverse().ff_playerids())


def stats(matched: pl.DataFrame, sid: str) -> dict[str, float]:
    return dict(matched.filter(pl.col("sleeper_id") == sid).select("stat", "value").rows())


def test_qb_duplicate_headers_split_passing_and_rushing(table):
    matched, unmatched = load(FP / "qb.csv", "QB", table, season=2026)
    mahomes = stats(matched, "4046")
    assert mahomes["pass_att"] == 560.1
    assert mahomes["pass_cmp"] == 372.4
    assert mahomes["pass_yd"] == 4210.5  # "4,210.5" thousands separator
    assert mahomes["pass_td"] == 30.2
    assert mahomes["pass_int"] == 11.0
    assert mahomes["rush_att"] == 62.3
    assert mahomes["rush_yd"] == 301.4
    assert mahomes["rush_td"] == 2.1
    assert mahomes["fum_lost"] == 2.0
    assert unmatched.rows() == [("Fake Quarterback", "XYZ", "QB")]
    assert set(matched["source"]) == {"fantasypros"}
    assert set(matched["season"]) == {2026}
    assert matched["week"].null_count() == matched.height


def test_fpts_never_stored(table):
    for name, pos in [("qb", "QB"), ("rb", "RB"), ("wr", "WR"), ("te", "TE"), ("k", "K")]:
        matched, _ = load(FP / f"{name}.csv", pos, table)
        assert "fpts" not in {s.lower() for s in matched["stat"]}
    matched, _ = load(FP / "dst.csv", "DEF", table)
    assert "fpts" not in set(matched["stat"])


def test_rb_receiving_is_second_block(table):
    matched, unmatched = load(FP / "rb.csv", "RB", table)
    barkley = stats(matched, "4866")
    assert barkley["rush_att"] == 290.0
    assert barkley["rush_yd"] == 1400.5
    assert barkley["rec"] == 45.0
    assert barkley["rec_yd"] == 350.0
    assert barkley["rec_td"] == 2.0
    # Kenneth Walker III (FP) → Kenneth Walker (Sleeper), RB on KC, not the free-agent WR namesake
    assert stats(matched, "8151")["rush_att"] == 250.0
    assert unmatched.height == 0


def test_wr_receiving_is_first_block_and_name_normalization(table):
    matched, unmatched = load(FP / "wr.csv", "WR", table)
    chase = stats(matched, "7564")  # "Ja'Marr Chase" with idmap-independent matching
    assert (chase["rec"], chase["rec_yd"], chase["rec_td"]) == (110.0, 1380.0, 10.0)
    assert (chase["rush_att"], chase["rush_yd"]) == (2.0, 10.0)
    assert stats(matched, "4983")["rec_yd"] == 1050.0  # "D.J. Moore" == "DJ Moore"
    assert unmatched["name"].to_list() == ["Not A Real Receiver"]


def test_te_k_dst(table):
    te, un = load(FP / "te.csv", "TE", table)
    assert stats(te, "1466")["rec"] == 85.0 and un.height == 0
    k, un = load(FP / "k.csv", "K", table)
    assert stats(k, "4227") == {"fgm": 30.0, "fga": 35.0, "xpm": 40.0}
    assert un.rows() == [("Nobody Kicker", "ZZZ", "K")]
    d, un = load(FP / "dst.csv", "DEF", table)
    assert stats(d, "DEN") == {
        "sack": 46.0, "int": 14.0, "fum_rec": 10.0, "ff": 12.0,
        "def_td": 2.0, "safe": 0.5, "pts_allow": 330.0, "yds_allow": 5300.0,
    }  # fmt: skip
    assert un["name"].to_list() == ["Mystery Defense"]


def test_position_inferred_from_headers(table):
    for name, pos in [("qb", "QB"), ("rb", "RB"), ("wr", "WR"), ("te", "TE"),
                      ("k", "K"), ("dst", "DEF")]:  # fmt: skip
        headers = (FP / f"{name}.csv").read_text().splitlines()[0].replace('"', "").split(",")
        assert infer_position(headers) == pos
        matched, _ = load(FP / f"{name}.csv", None, table)
        assert matched.height > 0
    with pytest.raises(ValueError):
        infer_position(["Player", "Team", "FPTS"])


def test_name_with_team_and_no_team_column(table):
    matched, unmatched = load(FP / "qb_name_team.csv", None, table)
    assert stats(matched, "4046")["pass_yd"] == 4210.5
    assert stats(matched, "4984")["rush_td"] == 11.5
    assert unmatched.height == 0


def test_wrong_team_is_unmatched_not_guessed(table):
    csv = "Player,Team,REC,YDS,TDS,FL,FPTS\nTravis Kelce,DAL,80,800,6,1,150\n"
    matched, unmatched = load(io.StringIO(csv), "TE", table)
    assert matched.height == 0
    assert unmatched.rows() == [("Travis Kelce", "DAL", "TE")]


def test_bytes_buffer_and_alias_team_codes(table):
    csv = b'\xef\xbb\xbf"Player","Team","FG","FGA","XPT"\n"Brandon Aubrey","DAL","32","36","42"\n'
    matched, _ = load(io.BytesIO(csv), "K", table, week=3)
    assert set(matched["week"]) == {3}
    assert stats(matched, "11533")["fgm"] == 32.0


DST_HEAD = "Player,Team,SACK,INT,FR,FF,TD,SAFETY,PA,YDS_AGN,FPTS\n"


def test_dst_name_must_agree_with_team(table):
    csv = DST_HEAD + (
        "Denver Broncos,KC,2,1,1,1,0,0,20,300,5\n"  # name says DEN, team says KC
        "Broncos,DEN,2,1,1,1,0,0,20,300,5\n"
        "Philadelphia,,2,1,1,1,0,0,20,300,5\n"  # bare city, no team column value
        "Kansas City Chiefs D/ST,KC,2,1,1,1,0,0,20,300,5\n"
    )
    matched, unmatched = load(io.StringIO(csv), "DEF", table)
    assert sorted(set(matched["sleeper_id"])) == ["DEN", "KC", "PHI"]
    assert unmatched.rows() == [("Denver Broncos", "KC", "DEF")]


def test_dst_points_and_yards_allowed_buckets_for_weekly_files(table):
    csv = DST_HEAD + "Denver Broncos,DEN,2.4,0.7,0.5,0.7,0.1,0,23.75,412.0,6.5\n"
    weekly, _ = load(io.StringIO(csv), "DEF", table, week=7)
    s = stats(weekly, "DEN")
    assert s["pts_allow"] == 23.75 and s["pts_allow_21_27"] == 1.0
    assert s["yds_allow_400_449"] == 1.0
    season, _ = load(io.StringIO(csv), "DEF", table)
    assert not [k for k in stats(season, "DEN") if k.startswith(("pts_allow_", "yds_allow_"))]


@pytest.mark.parametrize(
    ("pts", "key"),
    [(0, "pts_allow_0"), (3, "pts_allow_1_6"), (10, "pts_allow_7_13"), (17, "pts_allow_14_20"),
     (30, "pts_allow_28_34"), (41, "pts_allow_35p")],
)  # fmt: skip
def test_pts_allow_buckets(pts, key):
    assert bucket_stats(pts, None) == {key: 1.0}


def test_duplicate_player_rows_not_double_counted(table):
    csv = (
        "Player,Team,REC,YDS,TDS,FL,FPTS\n"
        "Travis Kelce,KC,85,900,7,1,1\n"
        "Travis Kelce,KC,10,100,1,0,1\n"
    )
    matched, unmatched = load(io.StringIO(csv), "TE", table)
    assert stats(matched, "1466")["rec"] == 85.0  # first row wins
    assert matched.filter(pl.col("stat") == "rec").height == 1
    assert unmatched.rows() == [("Travis Kelce", "KC", "TE")]
