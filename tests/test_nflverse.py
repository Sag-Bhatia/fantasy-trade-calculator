import datetime as dt

import polars as pl
import pytest

from ffcalc.clients import nflverse
from ffcalc.clients.nflverse import NflverseClient, current_season
from tests.fakes import FakeNflverse


def test_fake_matches_real_interface():
    fake = FakeNflverse()
    for method in ("player_stats", "schedules", "injuries", "snap_counts"):
        df = getattr(fake, method)([2025])
        assert df.height > 0 and set(df["season"]) == {2025}
        assert getattr(fake, method)([2024]).height == 0
    ids = fake.ff_playerids()
    assert {"sleeper_id", "espn_id", "gsis_id", "fantasypros_id"} <= set(ids.columns)
    assert set(fake.player_stats([2025])["week"]) == {1, 2, 3}
    assert {"spread_line", "total_line"} <= set(fake.schedules([2025]).columns)


def test_current_season():
    assert current_season(dt.date(2026, 10, 21)) == 2026
    assert current_season(dt.date(2027, 2, 1)) == 2026


def test_client_caches_per_season_and_skips_unpublished(monkeypatch):
    calls = []

    def fake_injuries(seasons):
        calls.append(seasons)
        if seasons == [2099]:
            raise ValueError("Season must be between 2009 and 2026")
        return FakeNflverse().injuries(seasons)

    monkeypatch.setattr(nflverse.nflreadpy, "load_injuries", fake_injuries)
    client = NflverseClient()
    df = client.injuries([2025, 2099])
    assert set(df["season"]) == {2025}
    client.injuries([2025])
    assert calls.count([2025]) == 1  # second call served from disk cache
    assert client.injuries([2099]).equals(pl.DataFrame())


def test_player_stats_filters_regular_season(monkeypatch):
    raw = FakeNflverse().player_stats([2025]).with_columns(season_type=pl.lit("REG"))
    post = raw.head(2).with_columns(season_type=pl.lit("POST"))
    monkeypatch.setattr(
        nflverse.nflreadpy, "load_player_stats", lambda seasons: pl.concat([raw, post])
    )
    assert set(NflverseClient().player_stats([2025])["season_type"]) == {"REG"}


def test_other_errors_propagate_without_cache_and_fall_back_with_it(monkeypatch):
    def down(seasons):
        raise ConnectionError("offline")

    monkeypatch.setattr(nflverse.nflreadpy, "load_schedules", down)
    with pytest.raises(ConnectionError):
        NflverseClient().schedules([2026])

    good = FakeNflverse().schedules([2025])
    monkeypatch.setattr(nflverse.nflreadpy, "load_schedules", lambda seasons: good)
    NflverseClient().schedules([2025])
    monkeypatch.setattr(nflverse.nflreadpy, "load_schedules", down)
    monkeypatch.setattr(nflverse.cache, "_fresh", lambda p, ttl: False)  # force a refetch
    assert NflverseClient().schedules([2025]).height == good.height  # stale copy used
