import polars as pl

from ffcalc import cache
from ffcalc.clients.fantasycalc import FantasyCalcClient, values_frame
from tests.fakes import FakeFantasyCalc


def test_frame_columns_and_rows():
    df = FakeFantasyCalc().values(14, 1, 1.0)
    assert df.columns == [
        "sleeper_id", "name", "position", "team", "value",
        "overall_rank", "position_rank", "trend_30d",
    ]  # fmt: skip
    assert df.height > 100 and df["sleeper_id"].null_count() == 0
    assert df.schema["value"] == pl.Int64 and df["value"].max() > 0


def test_rows_without_sleeper_id_dropped():
    rows = [
        {"player": {"sleeperId": "9221", "name": "A", "position": "RB"}, "value": 5},
        {"player": {"name": "B", "position": "WR"}, "value": 4},
    ]
    assert values_frame(rows)["sleeper_id"].to_list() == ["9221"]


def test_client_params_and_cache(monkeypatch):
    seen = []

    def fake_get(url, params=None, headers=None):
        seen.append(params)
        return [{"player": {"sleeperId": "1", "name": "A", "position": "QB"}, "value": 1}]

    monkeypatch.setattr(cache, "get_json", fake_get)
    c = FantasyCalcClient()
    assert c.values(12, 2, 0.5).height == 1
    c.values(12, 2, 0.5)
    assert seen == [{"isDynasty": "false", "numQbs": 2, "numTeams": 12, "ppr": 0.5}]
