import pytest

from ffcalc.league import LeagueNotFound, find_leagues, load_league
from tests.fakes import FakeSleeper

LID = "1400221038771437568"


@pytest.fixture
def league():
    return load_league(LID, FakeSleeper())


def test_settings(league):
    assert league.slots == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    assert (league.bench_size, league.ir_slots, league.num_teams) == (6, 1, 14)
    assert (league.playoff_teams, league.playoff_week_start, league.playoff_round_type) == (
        8,
        15,
        0,
    )
    assert league.trade_deadline == 11 and not league.median_win
    assert league.current_week == 7 and league.previous_league_id is None
    assert league.scoring["rec"] == 1.0 and league.scoring["pass_td"] == 4.0
    assert league.num_qbs == 1 and league.last_regular_week == 14


def test_teams(league):
    assert len(league.teams) == 14
    t = league.team(1)
    assert t.name and t.players and len(set(t.players)) == len(t.players)
    assert set(t.reserve) <= set(t.players)
    assert t.points_for == 840.40 and t.wins == 5


def test_schedule_and_results(league):
    assert sorted(league.schedule) == list(range(1, 15))
    assert all(len(p) == 7 for p in league.schedule.values())
    for t in league.teams:
        for w, pairs in league.schedule.items():
            assert sum(t.roster_id in p for p in pairs) == 1, (t.roster_id, w)
    assert sorted(league.results) == list(range(1, 7))
    assert all(len(r) == 14 for r in league.results.values())
    assert sum(league.results[1].values()) > 0


def test_missing_league():
    class Gone(FakeSleeper):
        def league(self, league_id):
            return None

    with pytest.raises(LeagueNotFound):
        load_league("123", Gone())


def test_find_leagues():
    out = find_leagues("manager01", 2026, FakeSleeper())
    assert out == [{"league_id": LID, "name": out[0]["name"], "total_rosters": 14}]
    assert find_leagues("manager01", 2020, FakeSleeper()) == []


def test_find_leagues_unknown_user():
    assert find_leagues("nobody_here", 2026, FakeSleeper()) == []


@pytest.mark.parametrize("bad", ["nope", "", "12 34", "../etc", "123abc", "١٢٣"])
def test_bad_league_id_is_rejected_before_any_request(bad):
    with pytest.raises(LeagueNotFound, match="league ID"):
        load_league(bad, FakeSleeper())


@pytest.mark.parametrize("bad", ["", "a b", "x/y", "a" * 41, "name?x=1"])
def test_bad_username_is_rejected(bad):
    with pytest.raises(ValueError, match="username"):
        find_leagues(bad, 2026, FakeSleeper())
