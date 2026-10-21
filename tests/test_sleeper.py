import polars as pl
import pytest

from ffcalc import cache
from ffcalc.clients import sleeper as sl
from ffcalc.clients.sleeper import SleeperClient, game_frame, stat_frame
from ffcalc.league import LeagueNotFound
from tests.fakes import FakeSleeper

ROWS = [
    {
        "season": "2026",
        "week": 7,
        "player_id": "7611",
        "team": "NE",
        "opponent": "CHI",
        "game_id": "202610706",
        "stats": {
            "rush_yd": 41.46,
            "rec": 2.23,
            "fum": 0.0,
            "pts_ppr": 11.09,
            "adp_dd_ppr": 71.0,
            "pos_adp_dd_ppr": 24.0,
            "pos_rank_ppr": 5.0,
            "rank_ppr": 9.0,
            "gp": 1.0,
        },
    },
    {"season": "2026", "week": 7, "player_id": "KC", "team": "KC", "stats": {}},
]


def test_stat_frame_drops_derived_and_zero_keys():
    df = stat_frame(ROWS, "sleeper")
    assert df.columns == ["source", "season", "week", "sleeper_id", "stat", "value"]
    assert df.schema["value"] == pl.Float64
    assert sorted(df["stat"]) == ["gp", "rec", "rush_yd"]
    assert set(df["source"]) == {"sleeper"} and set(df["season"]) == {2026}


def test_game_frame_keeps_missing_opponent():
    df = game_frame(ROWS)
    assert df.columns == ["season", "week", "sleeper_id", "team", "opponent", "game_id"]
    assert df.height == 2 and df["game_id"][1] is None


def test_fixture_projections_shape():
    rows = FakeSleeper().projections(2026, 7)
    df = stat_frame(rows, "sleeper")
    assert df.height > 1000
    assert {"rush_yd", "pass_td"} <= set(df["stat"])
    assert not set(df["stat"]) & {"pts_ppr", "pts_std", "pts_half_ppr"}
    assert not any(s.startswith("adp_") for s in df["stat"])
    assert game_frame(rows).height == len(rows)


def test_stat_frame_keeps_defense_points_allowed_buckets():
    from ffcalc.scoring import STANDARD_PPR, points, score_lines

    row = {
        "season": "2025",
        "week": 3,
        "player_id": "GB",
        "stats": {"pts_allow_35p": 1.0, "pts_allow": 38.0, "sack": 2.0, "pts_std": -2.0},
    }
    df = stat_frame([row], "sleeper")
    assert "pts_allow_35p" in set(df["stat"])
    got = score_lines(df.drop("source"), STANDARD_PPR)["points"][0]
    assert got == points(row["stats"], STANDARD_PPR) == -2.0


def test_unknown_week_is_empty():
    f = FakeSleeper()
    assert f.projections(2026, 99) == [] and f.matchups("x", 99) == []


def test_projections_snapshot_and_params(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None):
        calls.append((url, params))
        if url.endswith("/state/nfl"):
            return {"season": "2026", "week": 7}
        return ROWS

    monkeypatch.setattr(cache, "get_json", fake_get)
    c = SleeperClient()
    assert c.projections(2026, 7) == ROWS
    url, params = [c for c in calls if "projections" in c[0]][0]
    assert url == f"{sl.PROJ_API}/projections/nfl/2026/7"
    assert [v for k, v in params if k == "position[]"] == list(sl.FANTASY_POSITIONS)
    assert (cache.data_dir() / "snapshots" / "sleeper" / "2026" / "7.parquet").exists()
    n = len(calls)
    c.projections(2026, 7)  # served from cache
    assert len(calls) == n


def _spy(monkeypatch, state_week=7):
    calls = []

    def fake_get(url, params=None, headers=None):
        calls.append(url)
        return {"season": "2026", "week": state_week} if url.endswith("/state/nfl") else []

    monkeypatch.setattr(cache, "get_json", fake_get)
    return calls


def test_input_validation_blocks_requests(monkeypatch):
    calls = _spy(monkeypatch)
    c = SleeperClient()
    for bad in ("../x", "abc", "1/2", ""):
        with pytest.raises(LeagueNotFound):
            c.league(bad)
        with pytest.raises(LeagueNotFound):
            c.matchups(bad, 1)
    for bad in ("a b", "x/../y", "", "a" * 41):
        with pytest.raises(ValueError, match="username"):
            c.user(bad)
    assert calls == []
    c.user("Some_user-1.x")
    assert calls == [f"{sl.API}/user/Some_user-1.x"]


def test_finished_weeks_use_history_ttl_but_latest_stays_fresh(monkeypatch):
    _spy(monkeypatch, state_week=7)
    seen = {}
    real = cache.cached_json

    def spy_cached(key, ttl, fetch):
        seen[key] = ttl
        return real(key, ttl, fetch)

    monkeypatch.setattr(cache, "cached_json", spy_cached)
    c = SleeperClient()
    c.matchups("123", 3)
    c.matchups("123", 6)  # latest finished week: corrections may still land
    c.transactions("123", 5)
    c.transactions("123", 7)
    assert seen["sleeper/matchups_123_3"] == cache.TTL["history"]
    assert seen["sleeper/matchups_123_6"] == cache.TTL["league"]
    assert seen["sleeper/transactions_123_5"] == cache.TTL["history"]
    assert seen["sleeper/transactions_123_7"] == cache.TTL["league"]
