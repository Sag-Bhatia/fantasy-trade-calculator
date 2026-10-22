"""HTTP API tests (docs/api.md): FastAPI TestClient, offline, on the fixture league."""

import json
from collections import Counter
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ffcalc import api, engine
from ffcalc.lineup import optimal_lineup
from ffcalc.trade import Trade, evaluate
from tests.fakes import FakeSleeper, fake_sources

LEAGUE = "1400221038771437568"
ME, PARTNER = 1, 6
SIMS = 3000


@pytest.fixture(autouse=True)
def offline_api(monkeypatch):
    monkeypatch.setattr(api, "source_factory", fake_sources)
    monkeypatch.setattr(api, "build_history", False)
    api._cache.clear()
    api._fp.clear()
    yield
    api._cache.clear()
    api._fp.clear()


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def ctx():
    return engine.prepare(LEAGUE, src=fake_sources(), build_history=False)


def roster(client, roster_id) -> list[dict]:
    r = client.get(f"/api/league/{LEAGUE}/team/{roster_id}")
    assert r.status_code == 200
    return r.json()["players"]


def a_trade(client) -> dict:
    """A 2-for-1: my two best non-K/DEF bench-or-starter players for one of theirs."""
    mine = [p for p in roster(client, ME) if p["pos"] in ("RB", "WR")][:2]
    theirs = [p for p in roster(client, PARTNER) if p["pos"] in ("RB", "WR")][:1]
    return {
        "league_id": LEAGUE,
        "team": ME,
        "partner": PARTNER,
        "give": [p["id"] for p in mine],
        "get": [p["id"] for p in theirs],
    }


def ndjson(resp) -> list[dict]:
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in resp.text.splitlines()]


def test_league_summary(client):
    r = client.get(f"/api/league/{LEAGUE}")
    assert r.status_code == 200
    j = r.json()
    assert (j["num_teams"], j["season"], j["week"], j["bench"]) == (14, 2026, 7, 6)
    assert j["slots"] == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    assert j["scoring"] == {"label": "Full PPR", "per_catch": 1, "pass_td": 4, "te_premium": 0}
    assert (j["playoff_teams"], j["playoff_week_start"], j["trade_deadline"]) == (8, 15, 11)
    assert j["deadline_passed"] is False
    assert len(j["teams"]) == 14 and j["teams"][0]["roster_id"] == 1
    assert set(j["teams"][0]) == {"roster_id", "name", "wins", "losses", "ties", "points_for"}
    assert j["teams"][0]["points_for"] == 840.4
    assert isinstance(j["warnings"], list)


def test_user_leagues(client):
    # the fixture's state() season is 2026; users are matched on display name
    name = FakeSleeper().users("")[0]["display_name"]
    r = client.get(f"/api/users/{name}/leagues")
    assert r.status_code == 200
    assert r.json()[0]["league_id"] == LEAGUE and r.json()[0]["total_rosters"] == 14
    assert client.get("/api/users/bad name/leagues").status_code == 422


def test_team_roster_slots(client, ctx):
    r = client.get(f"/api/league/{LEAGUE}/team/{ME}")
    j = r.json()
    assert j["roster_id"] == ME and j["record"].count("-") >= 1
    players = j["players"]
    assert {
        "id",
        "name",
        "pos",
        "nfl_team",
        "headshot",
        "slot",
        "injury",
        "week_pts",
        "ros_pts",
    } <= (set(players[0]))
    slots = [p["slot"] for p in players]
    starters = Counter(s for s in slots if s not in ("BN", "IR"))
    assert not starters - Counter(ctx.league.slots)  # never more players than slots
    # a slot stays empty only when nobody left on the roster is eligible and expected to score
    from ffcalc.models import SLOT_ELIGIBILITY

    for slot in Counter(ctx.league.slots) - starters:
        left = [p for p in players if p["slot"] == "BN" and p["pos"] in SLOT_ELIGIBILITY[slot]]
        assert all(p["week_pts"] <= 0 for p in left)
    assert len(starters) >= 6
    assert Counter(slots)["IR"] == len(ctx.league.team(ME).reserve)
    assert {p["id"] for p in players} <= set(ctx.league.team(ME).players)
    order = {s: i for i, s in reversed(list(enumerate(ctx.league.slots)))}
    order |= {"BN": 98, "IR": 99}
    keys = [(order[p["slot"]], -p["week_pts"]) for p in players]
    assert keys == sorted(keys)
    assert all(p["headshot"].startswith("https://sleepercdn.com/") for p in players)
    d = next(p for p in players if p["pos"] == "DEF")
    assert d["headshot"].endswith(f"/team_logos/nfl/{d['nfl_team'].lower()}.png")
    assert client.get(f"/api/league/{LEAGUE}/team/99").status_code == 404


def test_odds(client, ctx):
    j = client.get(f"/api/league/{LEAGUE}/odds").json()
    rows = j["rows"]
    assert j["sims"] == 3000 and len(rows) == 14
    assert sum(r["playoff"] for r in rows) == pytest.approx(ctx.league.playoff_teams, abs=0.01)
    assert sum(r["title"] for r in rows) == pytest.approx(1.0, abs=0.01)
    assert [r["playoff"] for r in rows] == sorted((r["playoff"] for r in rows), reverse=True)
    assert all(0 <= r["title"] <= 1 for r in rows)
    assert client.get(f"/api/league/{LEAGUE}/odds").json() == j  # cached


def test_league_trades(client):
    j = client.get(f"/api/league/{LEAGUE}/trades").json()
    assert j, "fixture league has trades"
    side = j[0]["sides"][0]
    assert set(side) == {"roster_id", "name", "got", "value"}
    assert all(isinstance(g, str) for g in side["got"])


def test_preview_matches_direct_lineup(client, ctx):
    body = a_trade(client)
    j = client.post("/api/preview", json=body).json()
    assert j["shape"] == "2 for 1" and "full grade" in j["note"]

    def best(roster):
        idx = [ctx.index[p] for p in roster]
        e = ctx.inputs.expected[:, 0]
        return optimal_lineup(
            [float(e[i]) for i in idx], [int(ctx.inputs.pos[i]) for i in idx], ctx.league.slots
        )[0]

    mine, theirs = ctx.league.team(ME).players, ctx.league.team(PARTNER).players
    give, get = body["give"], body["get"]
    me_after = [p for p in mine if p not in give] + get
    them_after = [p for p in theirs if p not in get] + give
    # rosters of players the engine doesn't know about are ignored on both sides
    known = lambda ps: [p for p in ps if p in ctx.index]  # noqa: E731
    assert j["me"] == pytest.approx(best(known(me_after)) - best(known(mine)), abs=0.006)
    assert j["them"] == pytest.approx(best(known(them_after)) - best(known(theirs)), abs=0.006)


def test_preview_validation(client):
    body = a_trade(client)
    assert client.post("/api/preview", json={**body, "partner": ME}).status_code == 422
    r = client.post("/api/preview", json={"league_id": LEAGUE})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)


@pytest.mark.slow
def test_grade_stream_matches_engine(client, ctx):
    body = {**a_trade(client), "sims": SIMS}
    lines = ndjson(client.post("/api/grade", json=body))
    assert lines[-1]["type"] == "result" and len(lines) >= 3
    assert all(m["type"] == "progress" and m["message"] for m in lines[:-1])
    assert any("Simulating 3,000" in m["message"] for m in lines[:-1])
    res = lines[-1]["result"]
    direct = evaluate(ctx, Trade(ME, PARTNER, body["give"], body["get"]), sims=SIMS, seed=7)
    assert (res["sims"], res["seed"], res["verdict"]) == (SIMS, 7, direct.verdict)
    assert res["weeks"] == ctx.inputs.weeks and res["playoff_week_start"] == 15
    for key, d in (("a", direct.a), ("b", direct.b)):
        t = res[key]
        assert t["roster_id"] == d.roster_id
        assert t["title_after"] == pytest.approx(d.title_after, abs=1e-4)
        assert t["playoff_before"] == pytest.approx(d.playoff_before, abs=1e-4)
        assert t["title"]["value"] == pytest.approx(d.title.value, abs=1e-4)
        assert t["ppw"]["value"] == pytest.approx(d.points_per_week.value, abs=0.01)
        assert t["score"] == pytest.approx(d.score, abs=0.01)
        assert [x["part"] for x in t["decomposition"]] == list(d.decomposition)
        assert sum(x["value"] for x in t["decomposition"]) == pytest.approx(t["score"], abs=0.05)
        assert len(t["weekly_delta"]) == len(res["weeks"])
        assert set(t["moves"]) == {"drops"}
    assert res["market"]["give"] == direct.market.a_gives.total
    assert res["market"]["get"] == direct.market.b_gives.total
    assert 0 <= res["acceptance"] <= 1 and isinstance(res["warnings"], list)


def test_grade_validation_errors(client):
    body = a_trade(client)
    unknown = client.post("/api/grade", json={**body, "get": ["not-a-player"]})
    assert unknown.status_code == 422 and "roster" in unknown.json()["detail"]
    third = roster(client, 2)[0]["id"]  # on neither side's roster
    wrong = client.post("/api/grade", json={**body, "get": [third]})
    assert wrong.status_code == 422 and "roster" in wrong.json()["detail"]
    dup = client.post("/api/grade", json={**body, "get": [body["give"][0]]})
    assert dup.status_code == 422
    assert client.post("/api/grade", json={**body, "sims": 500}).status_code == 422
    assert client.post("/api/grade", json={**body, "team": 99}).status_code == 422
    assert client.post("/api/grade", json={**body, "give": [], "get": []}).status_code == 422


def test_grade_errors_come_as_a_line(client, monkeypatch):
    body = a_trade(client)

    def down(*a, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(api, "evaluate", down)
    lines = ndjson(client.post("/api/grade", json=body))
    assert lines[-1] == {
        "type": "error",
        "message": "A data source didn't answer. Try again in a minute.",
    }

    def boom(*a, **k):
        raise api.TradeError("This trade can't work.")

    monkeypatch.setattr(api, "evaluate", boom)
    assert ndjson(client.post("/api/grade", json=body))[-1]["message"] == "This trade can't work."


def test_unknown_league_is_404(client, monkeypatch):
    assert client.get("/api/league/not-a-number").status_code == 404
    monkeypatch.setattr(FakeSleeper, "league", lambda self, league_id: {})
    r = client.get("/api/league/123")
    assert r.status_code == 404 and "league ID" in r.json()["detail"]


def test_source_down_is_503_without_traceback(client, monkeypatch):
    def down(self, league_id):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(FakeSleeper, "league", down)
    r = client.get(f"/api/league/{LEAGUE}")
    assert r.status_code == 503
    assert r.json() == {"detail": "A data source didn't answer. Try again in a minute."}
    assert "Traceback" not in r.text


def test_context_prepared_once_per_league(client, monkeypatch):
    calls = []
    real = engine.prepare
    monkeypatch.setattr(engine, "prepare", lambda *a, **k: calls.append(a) or real(*a, **k))
    client.get(f"/api/league/{LEAGUE}")
    client.get(f"/api/league/{LEAGUE}/team/{ME}")
    assert len(calls) == 1


@pytest.mark.slow
def test_find_stream(client, ctx):
    lines = ndjson(client.post("/api/find", json={"league_id": LEAGUE, "team": ME}))
    assert lines[-1]["type"] == "result", lines[-1]
    assert any(m["type"] == "progress" for m in lines[:-1])
    res = lines[-1]["result"]
    assert res["rosters"] == 13 and res["checked"] > 0
    for o in res["offers"]:
        assert o["partner"]["roster_id"] != ME and o["give"] and o["get"]
        assert set(o) >= {"me", "them", "me_points", "them_points", "market_gap", "moves"}


@pytest.mark.slow
def test_balance_stream_describes_each_change(client):
    b = a_trade(client)
    lines = ndjson(client.post("/api/balance", json=b))
    assert lines[-1]["type"] == "result", lines[-1]
    res = lines[-1]["result"]
    assert res["tried"] > 0
    for o in res["offers"]:
        c = o["change"]
        assert sum(len(v) for v in c.values()) == 1  # exactly one player added or removed
        give = {p["id"] for p in o["give"]}
        assert give == (set(b["give"]) | {p["id"] for p in c["give_added"]}) - {
            p["id"] for p in c["give_removed"]
        }
        assert set(o) >= {"partner", "get", "me", "them", "moves"}


def test_fantasypros_remove_with_a_bad_league_id_is_404(client):
    assert client.delete("/api/league/abc/fantasypros").status_code == 404


def test_balance_rejects_a_bad_trade(client):
    body = {"league_id": LEAGUE, "team": ME, "partner": ME, "give": [], "get": []}
    assert client.post("/api/balance", json=body).status_code == 422


def test_find_unknown_team_is_404(client):
    assert client.post("/api/find", json={"league_id": LEAGUE, "team": 99}).status_code == 404


def test_spa_fallback(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DIST", tmp_path / "dist")  # not built
    r = client.get("/")
    assert r.status_code == 200 and "npm run build" in r.text
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>the app</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    r = client.get("/")
    assert "the app" in r.text and r.headers["cache-control"] == "no-store"
    assert "etag" not in r.headers  # a file-date ETag kept stale pages after a deploy
    assert "the app" in client.get("/league/123/trade").text  # client-side route
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/assets/old-build.js").status_code == 404  # not the page
    r = client.get("/api/nope")
    assert r.status_code == 404 and "detail" in r.json()
    assert "root:" not in client.get("/%2e%2e/%2e%2e/%2e%2e/etc/passwd").text


def test_fantasypros_upload_rebuilds_the_league_and_can_be_removed(client, monkeypatch):
    prepared = []
    real = engine.prepare
    monkeypatch.setattr(
        engine, "prepare", lambda *a, **k: prepared.append(k.get("fp_lines")) or real(*a, **k)
    )
    url = f"/api/league/{LEAGUE}/fantasypros"
    csv = (Path(__file__).parent / "fixtures" / "fantasypros" / "qb.csv").read_bytes()

    assert client.post(url, params={"week": 7}, content=b" ").status_code == 422
    assert client.post(url, params={"week": 19}, content=csv).status_code == 422
    bad = client.post(url, params={"week": 7}, content=b"hello,world")
    assert bad.status_code == 422 and "FantasyPros projections CSV" in bad.json()["detail"]
    r = client.post(url, params={"week": 7}, content=csv)
    assert r.status_code == 200, r.text
    assert r.json()["matched"] > 0
    assert "Fake Quarterback" in r.json()["unmatched"]
    assert client.get(f"/api/league/{LEAGUE}").json()["fantasypros"] is True
    assert prepared[-1] is not None and set(prepared[-1]["source"]) == {"fantasypros"}

    assert client.delete(url).status_code == 200
    assert client.get(f"/api/league/{LEAGUE}").json()["fantasypros"] is False
    assert prepared[-1] is None


def test_hosted_copy_turns_off_fantasypros_uploads(client, monkeypatch):
    monkeypatch.setattr(api, "HOSTED", True)
    assert client.get(f"/api/league/{LEAGUE}").json()["fantasypros_upload"] is False
    r = client.post(f"/api/league/{LEAGUE}/fantasypros", params={"week": 7}, content=b"x,y")
    assert r.status_code == 403 and "your own computer" in r.json()["detail"]


def test_heavy_jobs_wait_their_turn(client, monkeypatch):
    import threading

    monkeypatch.setattr(api, "_heavy", threading.BoundedSemaphore(1))
    api._heavy.acquire()  # someone else's search is running
    body = {**a_trade(client), "sims": 3000}
    out: list = []
    t = threading.Thread(target=lambda: out.append(ndjson(client.post("/api/grade", json=body))))
    t.start()
    t.join(1.0)
    assert t.is_alive()  # queued behind the running search
    api._heavy.release()
    t.join(60)
    lines = out[0]
    assert lines[0] == {"type": "progress", "message": "Waiting for another search to finish"}
    assert lines[-1]["type"] == "result"


def test_vercel_entry_moves_build_data_to_tmp(tmp_path, monkeypatch):
    import os
    import runpy
    import shutil
    import tempfile

    repo = tmp_path / "repo"
    cached = repo / "data" / "cache" / "players.json"
    cached.parent.mkdir(parents=True)
    cached.write_text("{}")
    os.utime(cached, (1, 1))
    cached.parent.chmod(0o555)  # the deployed bundle is read-only
    shutil.copy(Path(__file__).resolve().parent.parent / "app.py", repo)
    env = {k: v for k, v in os.environ.items() if not k.startswith("FFCALC_")}
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(os, "access", lambda *a, **k: False)  # read-only, like Vercel
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    assert runpy.run_path(str(repo / "app.py"))["app"] is api.app
    work = tmp_path / "tmp" / "ffcalc"
    assert env["FFCALC_DATA"] == str(work) and env["FFCALC_HOSTED"] == "1"
    # file age kept, so a players list from build time still counts as stale after a day
    assert (work / "cache" / "players.json").stat().st_mtime == 1
    (work / "cache" / "new.json").write_text("{}")  # and the copy can take new caches
