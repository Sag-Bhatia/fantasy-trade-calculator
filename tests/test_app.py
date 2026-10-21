"""Streamlit AppTest smoke tests (spec §10): the whole app, offline, on the fixture league."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from ffcalc import engine, ui
from tests.fakes import FakeSleeper, fake_sources

APP = str(Path(__file__).resolve().parent.parent / "app.py")
LEAGUE = "1400221038771437568"
QUICK = "Quick (3,000 seasons)"


@pytest.fixture(autouse=True)
def offline_ui(monkeypatch):
    # AppTest runs the script in this process, so patching ui reaches it.
    monkeypatch.setattr(ui, "source_factory", fake_sources)
    monkeypatch.setattr(ui, "build_history", False)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    import os

    os.environ["FFCALC_DATA"] = str(tmp_path_factory.mktemp("data"))
    return engine.prepare(LEAGUE, src=fake_sources(), build_history=False)


def run_app(league=LEAGUE, timeout=120) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    if league:
        at.query_params["league"] = league
    return at.run(timeout=timeout)


def texts(elements) -> str:
    return "\n".join(str(e.value) for e in elements)


def test_no_league_shows_welcome():
    at = run_app(league=None)
    assert not at.exception
    assert [t.value for t in at.title] == ["Grade a fantasy football trade"]
    assert not at.selectbox  # no team picker before a league is chosen


def test_league_via_query_param_loads_14_teams():
    at = run_app()
    assert not at.exception
    team = next(s for s in at.sidebar.selectbox if s.label == "Your team")
    assert len(team.options) == 14
    assert at.sidebar.text_input[0].value == LEAGUE


def test_league_via_sidebar_text_input():
    at = run_app(league=None)
    at.sidebar.text_input[0].set_value(LEAGUE).run(timeout=120)
    assert not at.exception
    assert len(at.sidebar.selectbox[0].options) == 14


def test_bad_league_id_is_a_friendly_error(monkeypatch):
    monkeypatch.setattr(FakeSleeper, "league", lambda self, league_id: {})
    at = run_app(league="123")
    assert not at.exception  # no stack trace
    assert len(at.error) == 1
    assert "league ID" in at.error[0].value


def test_non_numeric_league_id_is_a_friendly_error():
    at = run_app(league="not-a-number")
    assert not at.exception
    assert at.error and "league ID" in at.error[0].value


def test_trade_options_are_exactly_the_rosters(ctx):
    at = run_app()
    me = at.session_state["team"]
    partner = at.selectbox(key="partner").value
    give, get = at.multiselect(key="give"), at.multiselect(key=f"get_{partner}")
    assert give.label == "You give" and get.label.startswith("You get from")
    assert me != partner
    assert list(give.options) == [ui.player_label(ctx, p) for p in ui.roster_options(ctx, me)]
    assert list(get.options) == [ui.player_label(ctx, p) for p in ui.roster_options(ctx, partner)]
    # nobody appears on both sides: you can't offer a player you don't own
    assert not set(give.options) & set(get.options)


def test_grade_a_trade(ctx):
    at = run_app()
    me, partner = at.session_state["team"], at.selectbox(key="partner").value
    mine, theirs = ui.roster_options(ctx, me), ui.roster_options(ctx, partner)
    assert at.button[0].disabled  # nothing picked yet
    at.multiselect(key="give").select(ui.player_label(ctx, mine[len(mine) // 2]))
    at.multiselect(key=f"get_{partner}").select(ui.player_label(ctx, theirs[len(theirs) // 2]))
    at.radio[0].set_value(QUICK)
    at.run(timeout=120)
    btn = next(b for b in at.button if b.label == "Grade this trade")
    assert not btn.disabled
    btn.click().run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    html = texts(at.get("html"))
    assert any(v in html for v in ui.VERDICT_TEXT.values())
    assert "Title odds" in html
    assert "Why it helps or hurts" in [s.value for s in at.subheader]
    assert len(at.tabs) == 4
    assert [t.label for t in at.tabs] == [
        "Week by week",
        "Market check",
        "Roster moves",
        "The numbers",
    ]


def test_league_page(ctx):
    at = run_app()
    at.switch_page("views/league.py").run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert next(m for m in at.metric if m.label == "Teams").value == "14"
    standings = at.dataframe[0].value
    assert len(standings) == 14


def test_accuracy_page():
    at = run_app()
    at.switch_page("views/accuracy.py").run(timeout=120)
    assert not at.exception
    report = Path(APP).parent / "REPORT.md"
    if report.exists():
        assert "accuracy and calibration" in texts(at.markdown)
    else:
        assert at.info and "backtest" in at.info[0].value


def test_finder_page_has_button_and_no_trades_message(monkeypatch):
    monkeypatch.setattr(ui, "run_finder", lambda league_id, _ctx, team, _progress=None: [])
    at = run_app()
    at.switch_page("views/finder.py").run(timeout=120)
    assert not at.exception
    assert [t.value for t in at.title] == ["Find trades"]
    btn = next(b for b in at.button if b.label == "Find trades for my team")
    btn.click().run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert any("No trade clearly helps both teams" in i.value for i in at.info)


def test_prefill_from_finder_sets_trade_page(ctx):
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["league"] = LEAGUE
    me = ctx.league.teams[0].roster_id
    at.query_params["team"] = str(me)
    partner = ctx.league.teams[1].roster_id
    give, get = ui.roster_options(ctx, me)[:1], ui.roster_options(ctx, partner)[:2]
    at.session_state["prefill"] = (partner, give, get)
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="partner").value == partner
    assert at.multiselect(key="give").value == give
    assert at.multiselect(key=f"get_{partner}").value == get
    assert "prefill" not in at.session_state  # consumed once


def test_bad_team_query_param_falls_back_to_first_team():
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["league"] = LEAGUE
    at.query_params["team"] = "abc"
    at.run()
    assert not at.exception
    team = next(s for s in at.sidebar.selectbox if s.label == "Your team")
    assert team.index == 0


def test_source_down_shows_friendly_error(monkeypatch):
    import httpx

    def down(self, league_id):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(FakeSleeper, "league", down)
    at = run_app()
    assert not at.exception
    errors = texts(at.error)
    assert "didn't answer" in errors and "Traceback" not in errors
