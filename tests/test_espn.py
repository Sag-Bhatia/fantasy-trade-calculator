import polars as pl
import pytest

from ffcalc import cache
from ffcalc.clients import espn
from tests.fakes import FakeEspn, fake_sources


def _table(raw: dict, skip: int = 2) -> pl.DataFrame:
    """ESPN id e -> sleeper id 's<e>'; the last `skip` players are left unmapped."""
    ids = [str(e["id"]) for e in raw["players"]]
    return pl.DataFrame(
        {"sleeper_id": [f"s{i}" for i in ids], "espn_id": ids},
        schema=pl.Schema({"sleeper_id": pl.String, "espn_id": pl.String}),
    ).head(len(ids) - skip)


@pytest.fixture
def client():
    return FakeEspn()


@pytest.fixture
def table(client):
    return _table(client.season_raw(2026))


def test_columns_and_id_mapping(client, table):
    df = client.projections(2026, table)
    assert df.schema == pl.Schema(espn.SCHEMA)
    assert set(df["source"]) == {"espn"} and set(df["season"]) == {2026}
    assert set(df["sleeper_id"]) <= set(table["sleeper_id"]) and df.height > 1000
    assert set(df["stat"]) <= set(espn.STAT_MAP.values())


def test_unmapped_players_dropped_and_counted(client, table):
    client.projections(2026, table)
    raw_ids = {str(e["id"]) for e in client.season_raw(2026)["players"]}
    assert {str(i) for i in client.unmapped} == raw_ids - set(table["espn_id"])
    assert len(client.unmapped) == 2


def test_unknown_stat_ids_and_zeros_ignored(client, table):
    raw = client.season_raw(2026)
    # fixture lines carry unmapped stat ids; none may leak into the frame
    assert any(
        set(s["stats"]) - set(espn.STAT_MAP) for e in raw["players"] for s in e["player"]["stats"]
    )
    df = client.projections(2026, table)
    assert (df["value"] != 0).all()


def test_weeks_split_and_other_seasons_ignored(client, table):
    df = client.projections(2026, table)
    assert df["week"].min() >= 1 and df["week"].max() <= 18
    raw = client.season_raw(2026)
    assert any(s["seasonId"] == 2025 for e in raw["players"] for s in e["player"]["stats"])
    first = raw["players"][0]
    n = len(
        [
            s for s in first["player"]["stats"]
            if s["statSourceId"] == 1 and s["seasonId"] == 2026 and s["statSplitTypeId"] == 1
        ]
    )  # fmt: skip
    got = df.filter(pl.col("sleeper_id") == f"s{first['id']}")["week"].n_unique()
    assert 0 < got <= n


def test_projections_vs_actuals_separate(client, table):
    proj, act = client.projections(2026, table), client.actuals(2026, table)
    assert set(act["source"]) == {"espn_actual"}
    assert act["week"].max() == 6 and proj["week"].max() > 6  # only played weeks have actuals
    # a player's week-1 projection and actual differ

    def one(df):
        return df.filter((pl.col("week") == 1) & (pl.col("stat") == "rush_yd")).sort("sleeper_id")

    assert one(proj)["value"].to_list() != one(act)["value"].to_list()


def test_season_level_rows_excluded_2024(client):
    raw = client.season_raw(2024)
    assert any(s["statSplitTypeId"] == 0 for e in raw["players"] for s in e["player"]["stats"])
    table = _table(raw, skip=0)
    act = client.actuals(2024, table)
    assert act["week"].min() == 1 and act.height > 500
    assert client.projections(2024, table)["week"].n_unique() >= 17


def test_snapshots_only_for_unplayed_weeks_of_current_season(client, table):
    client.projections(2026, table)
    snaps = sorted((cache.data_dir() / "snapshots" / "espn" / "2026").glob("*.parquet"))
    assert min(int(p.stem) for p in snaps) == 7  # weeks 1-6 already have actuals
    client.projections(2024, _table(client.season_raw(2024), skip=0))
    assert not (cache.data_dir() / "snapshots" / "espn" / "2024").exists()


def test_fake_sources_has_espn():
    assert isinstance(fake_sources().espn, FakeEspn)


def test_live_client_request(monkeypatch):
    seen = []

    def fake_get(url, params=None, headers=None):
        seen.append((url, params, headers))
        return {"players": []}

    monkeypatch.setattr(cache, "get_json", fake_get)
    espn.EspnClient().season_raw(2024)
    url, params, headers = seen[0]
    assert "/seasons/2024/segments/0/leaguedefaults/3" in url
    assert params == {"view": "kona_player_info"} and "filterSlotIds" in headers["X-Fantasy-Filter"]


def test_unmapped_survives_the_actuals_pass(table):
    """A player with projections but no actuals must still be counted as unmapped."""
    c = FakeEspn()
    raw = c.season_raw(2026)
    ghost = raw["players"][-1]  # unmapped in `table`
    ghost["player"]["stats"] = [s for s in ghost["player"]["stats"] if s["statSourceId"] == 1]
    c.season_raw = lambda season: raw  # type: ignore[method-assign]
    c.projections(2026, table)
    assert str(ghost["id"]) in {str(i) for i in c.unmapped}
    c.actuals(2026, table)  # each call reports only its own drops
    assert str(ghost["id"]) not in {str(i) for i in c.unmapped}


def test_current_season_is_injectable_not_wall_clock(table):
    other = FakeEspn()
    other._season_now = 2030  # 2026 is now a past season: no snapshots, history TTL
    other.projections(2026, table)
    assert not (cache.data_dir() / "snapshots" / "espn" / "2026").exists()
