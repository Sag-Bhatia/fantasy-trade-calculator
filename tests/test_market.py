from dataclasses import replace

import polars as pl
import pytest

from ffcalc import engine
from ffcalc.clients.fantasycalc import SCHEMA
from ffcalc.market import (
    acceptance,
    fc_params,
    league_trades,
    market_check,
    values,
)
from ffcalc.sources import Sources
from ffcalc.trade import Trade
from tests.fakes import fake_sources


@pytest.fixture(scope="module")
def league(tmp_path_factory):
    import os

    os.environ["FFCALC_DATA"] = str(tmp_path_factory.mktemp("data"))
    return engine.prepare("1400221038771437568", src=fake_sources(), build_history=False).league


def test_fc_params(league):
    assert fc_params(league) == {"num_teams": 14, "num_qbs": 1, "ppr": 1}
    half = replace(league, scoring={"rec": 0.5})
    assert fc_params(half)["ppr"] == 0.5
    assert fc_params(replace(league, scoring={}))["ppr"] == 0
    assert fc_params(replace(league, scoring={"rec": 0.9}))["ppr"] == 1
    sf = replace(league, slots=[*league.slots, "SUPER_FLEX"])
    assert fc_params(sf)["num_qbs"] == 2
    assert fc_params(replace(league, slots=["QB", "QB", "RB"]))["num_qbs"] == 2


def frame(**vals):
    return pl.DataFrame(
        [(k, k, "WR", "X", v, 1, 1, 0) for k, v in vals.items()], schema=SCHEMA, orient="row"
    )


def test_market_check_math_and_unknown_players():
    vals = frame(a1=100, a2=50, b1=200)
    mc = market_check(Trade(1, 2, ["a1", "a2"], ["b1", "zz"]), vals, {"ppr": 1})
    assert mc.a_gives.total == 150 and mc.b_gives.total == 200
    assert mc.b_gives.values == {"b1": 200, "zz": 0}
    assert mc.gap == pytest.approx(50 / 150)
    assert not mc.overpay and mc.params == {"ppr": 1}


def test_overpay_boundary():
    def run(give, get):
        return market_check(Trade(1, 2, ["a"], ["b"]), frame(a=give, b=get))

    assert not run(115, 100).overpay  # exactly 15% more: not an overpay
    assert run(116, 100).overpay
    assert run(100, 100).gap == 0
    assert run(100, 50).gap == pytest.approx(-0.5)
    assert run(10, 0).overpay  # receives nothing


def test_values_empty_on_failure(league):
    class Boom:
        def values(self, **kw):
            raise RuntimeError("down")

    out = values(Sources(sleeper=None, fantasycalc=Boom()), league)
    assert out.height == 0 and out.columns == list(SCHEMA)
    assert values(fake_sources(), league).height > 0


class TxSleeper:
    def __init__(self, by_week):
        self.by_week = by_week

    def transactions(self, league_id, week):
        return self.by_week.get(week, [])


def test_league_trades(league):
    def tx(type_, status, created, adds, drops, rids):
        return {"type": type_, "status": status, "created": created, "adds": adds,
                "drops": drops, "roster_ids": rids, "draft_picks": []}  # fmt: skip

    # roster 1 sends a,b and gets c; roster 2 the reverse
    two_for_one = tx(
        "trade", "complete", 100, {"c": 1, "a": 2, "b": 2}, {"a": 1, "b": 1, "c": 2}, [1, 2]
    )
    later = tx("trade", "complete", 300, {"x": 3}, {"x": 4}, [3, 4])
    src = Sources(
        sleeper=TxSleeper(
            {
                1: [two_for_one, tx("waiver", "complete", 50, {"z": 5}, None, [5])],
                2: [tx("trade", "failed", 200, {"y": 3}, {"y": 4}, [3, 4]), later],
            }
        ),
        fantasycalc=None,
    )
    lg = replace(league, current_week=3)
    res = league_trades(src, lg, frame(a=100, b=50, c=120, x=7))
    assert [t.created for t in res] == [300, 100]  # newest first, waiver and failed trade dropped
    t = res[1]
    assert t.week == 1
    assert t.sides == [(1, ["c"], ["a", "b"]), (2, ["a", "b"], ["c"])]
    assert t.value_by_roster == {1: 120, 2: 150}
    assert res[0].value_by_roster == {3: 7, 4: 0}


def test_acceptance():
    assert acceptance(0, 0) == 0.5
    assert acceptance(1, 0) < acceptance(2, 0) < acceptance(3, 0)
    assert acceptance(0, -0.2) < acceptance(0, 0) < acceptance(0, 0.2)
    assert acceptance(100, 5) == 0.98
    assert acceptance(-100, -5) == 0.02
    assert acceptance(-1e6, -1e6) == 0.02
