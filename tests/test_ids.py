import polars as pl
import pytest

from ffcalc.ids import lookup, normalize_name, player_table
from tests.fakes import FakeNflverse, sample_sleeper_players


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("D.J. Moore", "DJ Moore"),
        ("Kenneth Walker III", "Kenneth Walker"),
        ("Marvin Harrison Jr.", "marvin harrison"),
        ("Ja'Marr Chase", "JaMarr Chase"),
        ("Amon-Ra St. Brown", "Amon Ra St Brown"),
        ("José Álvarez", "Jose Alvarez"),
    ],
)
def test_normalize_name_equal(a, b):
    assert normalize_name(a) == normalize_name(b)


def test_normalize_name_keeps_distinct_and_lone_suffix():
    assert normalize_name("Josh Allen") != normalize_name("Josh Jacobs")
    assert normalize_name("V") == "v"  # never reduce a name to nothing
    assert normalize_name("Patrick Mahomes II") == "patrickmahomes"


@pytest.fixture
def table():
    return player_table(sample_sleeper_players(), FakeNflverse().ff_playerids())


def test_table_shape_and_def_rows(table):
    assert table.columns == [
        "sleeper_id", "name", "position", "team", "injury_status", "status",
        "age", "years_exp", "espn_id", "gsis_id", "fantasypros_id", "search_name",
    ]  # fmt: skip
    assert table["sleeper_id"].is_unique().all()
    assert set(table["position"]) <= {"QB", "RB", "WR", "TE", "K", "DEF"}
    kc = table.filter(pl.col("sleeper_id") == "KC").row(0, named=True)
    assert (kc["position"], kc["team"], kc["name"]) == ("DEF", "KC", "Kansas City Chiefs")
    assert all(t == pl.String for t in [table.schema[c] for c in ("espn_id", "gsis_id")])


def test_ids_come_from_idmap_and_fall_back_to_sleeper(table):
    chase = table.filter(pl.col("sleeper_id") == "7564").row(0, named=True)
    # Sleeper has no espn_id for Chase; the idmap fills it
    assert chase["espn_id"] == "4362628"
    assert chase["gsis_id"] == "00-0036900"
    assert chase["fantasypros_id"] == "19788"

    ids = FakeNflverse().ff_playerids().filter(pl.col("sleeper_id") != 4046)
    mahomes = player_table(sample_sleeper_players(), ids).filter(pl.col("sleeper_id") == "4046")
    assert mahomes["espn_id"][0] == "3139477"  # Sleeper's own field
    assert mahomes["fantasypros_id"][0] is None


def test_non_fantasy_positions_dropped():
    players = {"1": {"position": "LB", "full_name": "A Backer"}, "2": {"position": "QB"}}
    t = player_table(players, FakeNflverse().ff_playerids())
    assert t["sleeper_id"].to_list() == ["2"]


def test_lookup(table):
    assert lookup(table, "4046") == "4046"
    assert lookup(table, "patrick mahomes") == "4046"
    assert lookup(table, "DJ Moore") == "4983"
    assert lookup(table, "Patrick Maho") == "4046"  # unique prefix
    assert lookup(table, "KC") == "KC"
    assert lookup(table, "Kansas City Chiefs") == "KC"
    assert lookup(table, "Nobody At All") is None
    assert lookup(table, "") is None


def test_lookup_ambiguous_prefix_is_none(table):
    # "ja" starts Ja'Marr Chase, Jalen Hurts and Jahmyr Gibbs: never guess
    assert lookup(table, "Ja") is None
    assert lookup(table, "Kenneth Walker") == "8151"  # namesake tie-break is the only exception


def test_lookup_namesakes_prefer_rostered(table):
    # two active "Kenneth Walker"s; the RB is on a team, the WR is not
    assert lookup(table, "Kenneth Walker III") == "8151"
