"""HTTP API for the web app (docs/api.md). Thin: every number comes from the engine."""

from __future__ import annotations

import io
import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import engine, finder, market
from .clients import fantasypros_csv
from .clients.sleeper import check_league_id
from .league import LeagueNotFound, find_leagues
from .lineup import optimal_lineup
from .models import Team
from .simulate import simulate_league, simulate_players
from .trade import Delta, TeamResult, Trade, TradeError, TradeResult, evaluate

log = logging.getLogger(__name__)

# Tests swap these for offline fakes, exactly like ffcalc/ui.py.
source_factory: Callable[[], Any] | None = None  # None → live sources
build_history = True

DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
CACHE_SECONDS = 600
SIM_CHOICES = (3_000, 10_000)
ODDS_SIMS = 3_000
DOWN = "A data source didn't answer. Try again in a minute."
NETWORK_ERRORS = (httpx.HTTPError, ConnectionError, OSError)

app = FastAPI(title="Fantasy football trade calculator")


# ------------------------------------------------------------------ league cache


@dataclass
class Entry:
    ctx: engine.Context
    at: float = field(default_factory=time.monotonic)
    info: dict[str, tuple[str, str, str | None, str | None]] = field(default_factory=dict)
    odds: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)  # guards odds


_cache: dict[str, Entry] = {}
_locks: dict[str, threading.Lock] = {}  # one per league so it is never prepared twice at once
_guard = threading.Lock()
# shortcut: in memory, so a restart forgets an added FantasyPros file; persist it if people ask
_fp: dict[str, pl.DataFrame] = {}  # league → FantasyPros stat lines the user added


def _lock(league_id: str) -> tuple[str, threading.Lock]:
    key = check_league_id(league_id.strip())
    with _guard:
        return key, _locks.setdefault(key, threading.Lock())


def _entry(league_id: str) -> Entry:
    key, lock = _lock(league_id)
    with lock:
        e = _cache.get(key)
        if e is None or time.monotonic() - e.at > CACHE_SECONDS:
            src = source_factory() if source_factory else None
            ctx = engine.prepare(key, src=src, fp_lines=_fp.get(key), build_history=build_history)
            e = _cache[key] = Entry(ctx)
            e.info = {
                sid: (name, pos, team, inj)
                for sid, name, pos, team, inj in ctx.table.select(
                    "sleeper_id", "name", "position", "team", "injury_status"
                ).iter_rows()
            }
        return e


@contextmanager
def friendly() -> Iterator[None]:
    """Turn failures into plain-English HTTP errors, never a stack trace."""
    try:
        yield
    except HTTPException:
        raise
    except LeagueNotFound as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except NETWORK_ERRORS as e:
        raise HTTPException(503, DOWN) from e


def _load(league_id: str) -> Entry:
    with friendly():
        return _entry(league_id)


def _team(ctx: engine.Context, roster_id: int) -> Team:
    try:
        return ctx.league.team(roster_id)
    except StopIteration:
        raise HTTPException(404, f"There's no team {roster_id} in this league.") from None


# ------------------------------------------------------------------ shapes


def _record(t: Team) -> str:
    return f"{t.wins}-{t.losses}" + (f"-{t.ties}" if t.ties else "")


def _brief(e: Entry, sid: str) -> dict[str, Any]:
    name, pos, team, _ = e.info.get(sid) or (e.ctx.name(sid), "", None, None)
    if pos == "DEF":
        head = f"https://sleepercdn.com/images/team_logos/nfl/{(team or sid).lower()}.png"
    else:
        head = f"https://sleepercdn.com/content/nfl/players/thumb/{sid}.jpg"
    return {"id": sid, "name": name, "pos": pos, "nfl_team": team, "headshot": head}


def _delta(d: Delta, nd: int = 4) -> dict[str, float]:
    return {"value": round(float(d.value), nd), "se": round(float(d.se), nd)}


def _team_result(e: Entry, res: TradeResult, r: TeamResult) -> dict[str, Any]:
    t = e.ctx.team_index(r.roster_id)
    weekly = (res.after.team_points[:, t, :] - res.before.team_points[:, t, :]).mean(axis=0)
    return {
        "roster_id": r.roster_id,
        "name": r.name,
        "title_before": round(r.title_before, 4),
        "title_after": round(r.title_after, 4),
        "title": _delta(r.title),
        "playoff_before": round(r.playoff_before, 4),
        "playoff_after": round(r.playoff_after, 4),
        "playoff": _delta(r.playoff),
        "ppw": _delta(r.points_per_week, 2),
        "score": round(r.score, 2),
        "decomposition": [{"part": k, "value": round(v, 2)} for k, v in r.decomposition.items()],
        "weekly_delta": [round(float(x), 2) for x in weekly],
        "moves": {
            "adds": [_brief(e, p) for p in r.moves.adds],
            "drops": [_brief(e, p) for p in r.moves.drops],
        },
    }


def _market(e: Entry, res: TradeResult) -> dict[str, Any] | None:
    m = res.market
    if m is None:
        return None
    unvalued = [
        e.ctx.name(p) for side in (m.a_gives, m.b_gives) for p, v in side.values.items() if v == 0
    ]
    return {
        "give": m.a_gives.total,
        "get": m.b_gives.total,
        "gap": round(m.gap, 4),
        "overpay": m.overpay,
        "unvalued": unvalued,
    }


def _grade_result(e: Entry, res: TradeResult) -> dict[str, Any]:
    lg = e.ctx.league
    return {
        "verdict": res.verdict,
        "sims": res.sims,
        "seed": res.seed,
        "weeks": list(e.ctx.inputs.weeks),
        "playoff_week_start": lg.playoff_week_start,
        "a": _team_result(e, res, res.a),
        "b": _team_result(e, res, res.b),
        "market": _market(e, res),
        "acceptance": None if res.acceptance is None else round(res.acceptance, 4),
        "warnings": res.warnings,
    }


# ------------------------------------------------------------------ requests


class FindBody(BaseModel):
    league_id: str
    team: int


class PreviewBody(FindBody):
    partner: int
    give: list[str] = []
    get: list[str] = []


class GradeBody(PreviewBody):
    sims: int = 10_000


@app.exception_handler(RequestValidationError)
async def _bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    where = ", ".join(".".join(str(x) for x in err["loc"][1:]) or "body" for err in exc.errors())
    return JSONResponse({"detail": f"That request wasn't understood (check: {where})."}, 422)


def _trade(ctx: engine.Context, b: PreviewBody) -> Trade:
    """Check a trade request against the league's rosters (422 with a plain message)."""
    ids = {t.roster_id for t in ctx.league.teams}
    if b.team not in ids or b.partner not in ids:
        raise HTTPException(422, "That team isn't in this league.")
    if b.team == b.partner:
        raise HTTPException(422, "Pick two different teams.")
    if not b.give and not b.get:
        raise HTTPException(422, "Pick at least one player to trade.")
    if len({*b.give, *b.get}) != len(b.give) + len(b.get):
        raise HTTPException(422, "A player is listed twice.")
    for owner, players in ((b.team, b.give), (b.partner, b.get)):
        t = ctx.league.team(owner)
        for p in players:
            if p not in ctx.index or p not in t.players:
                raise HTTPException(422, f"{ctx.name(p)} isn't on {t.name}'s roster.")
    return Trade(b.team, b.partner, list(b.give), list(b.get))


# ------------------------------------------------------------------ routes


@app.get("/api/users/{username}/leagues")
def user_leagues(username: str) -> list[dict]:
    with friendly():
        src = source_factory() if source_factory else None
        if src is None:
            from .sources import live

            src = live()
        try:
            return find_leagues(username, int(src.sleeper.state()["season"]), src.sleeper)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                msg = "No Sleeper user with that name. Check the spelling."
                raise HTTPException(404, msg) from e
            raise


@app.get("/api/league/{league_id}")
def league(league_id: str) -> dict[str, Any]:
    ctx = _load(league_id).ctx
    lg = ctx.league
    sc = lg.scoring
    ppr = sc.get("rec", 0)
    label = {1: "Full PPR", 0.5: "Half PPR", 0: "Standard"}.get(ppr, f"{ppr:g} per catch")
    return {
        "league_id": lg.league_id,
        "name": lg.name if lg.name and lg.name.lower() != "league" else "Your league",
        "season": lg.season,
        "week": lg.current_week,
        "num_teams": lg.num_teams,
        "scoring": {
            "label": label,
            "per_catch": ppr,
            "pass_td": sc.get("pass_td", 4),
            "te_premium": sc.get("bonus_rec_te", 0),
        },
        "slots": lg.slots,
        "bench": lg.bench_size,
        "playoff_teams": lg.playoff_teams,
        "playoff_week_start": lg.playoff_week_start,
        "trade_deadline": lg.trade_deadline,
        "deadline_passed": bool(lg.trade_deadline and lg.current_week > lg.trade_deadline),
        "teams": [
            {
                "roster_id": t.roster_id,
                "name": t.name,
                "wins": t.wins,
                "losses": t.losses,
                "ties": t.ties,
                "points_for": round(t.points_for, 2),
            }
            for t in lg.teams
        ],
        "warnings": ctx.warnings,
        "fantasypros": _lock(league_id)[0] in _fp,
    }


@app.post("/api/league/{league_id}/fantasypros")
async def add_fantasypros(
    league_id: str, request: Request, week: int = Query(ge=1, le=18)
) -> dict[str, Any]:
    """Blend a FantasyPros projections CSV (the raw file is the request body) into this league."""
    body = await request.body()
    if not body.strip():
        raise HTTPException(422, "That file is empty.")
    return await run_in_threadpool(_add_fantasypros, league_id, body, week)


def _add_fantasypros(league_id: str, body: bytes, week: int) -> dict[str, Any]:
    e = _load(league_id)
    try:
        lines, unmatched = fantasypros_csv.load(
            io.BytesIO(body), None, e.ctx.table, season=e.ctx.league.season, week=week
        )
    except (ValueError, UnicodeDecodeError) as err:
        raise HTTPException(
            422, "That doesn't look like a FantasyPros projections CSV. Export one from a "
            "position's projections page on fantasypros.com and try again."
        ) from err  # fmt: skip
    if not lines.height:
        raise HTTPException(422, "No row in that file matched a player. Is it a projections CSV?")
    key, lock = _lock(league_id)
    with lock:
        _fp[key] = lines
        _cache.pop(key, None)  # the next load rebuilds the projections with the file blended in
    return {"matched": lines.height, "unmatched": unmatched["name"].to_list()}


@app.delete("/api/league/{league_id}/fantasypros")
def remove_fantasypros(league_id: str) -> dict[str, Any]:
    key, lock = _lock(league_id)
    with lock:
        if _fp.pop(key, None) is not None:
            _cache.pop(key, None)
    return {"ok": True}


@app.get("/api/league/{league_id}/team/{roster_id}")
def team(league_id: str, roster_id: int) -> dict[str, Any]:
    e = _load(league_id)
    ctx, lg = e.ctx, e.ctx.league
    t = _team(ctx, roster_id)
    inp = ctx.inputs
    active = [p for p in t.players if p in ctx.index and p not in t.reserve]
    _, picks = optimal_lineup(
        [float(inp.expected[ctx.index[p], 0]) for p in active],
        [int(inp.pos[ctx.index[p]]) for p in active],
        lg.slots,
    )
    slot_of = {active[i]: s for i, s in picks}
    for p in t.reserve:
        slot_of[p] = "IR"
    order: dict[str, int] = {}
    for i, s in enumerate(lg.slots):
        order.setdefault(s, i)
    order["BN"], order["IR"] = len(lg.slots), len(lg.slots) + 1
    rows = []
    for p in (p for p in t.players if p in ctx.index):
        i = ctx.index[p]
        slot = slot_of.get(p, "BN")
        rows.append(
            {
                **_brief(e, p),
                "slot": slot,
                "injury": e.info.get(p, ("", "", None, None))[3],
                "week_pts": round(float(inp.expected[i, 0]), 2),
                "ros_pts": round(float(inp.expected[i].sum()), 2),
            }
        )
    rows.sort(key=lambda r: (order.get(r["slot"], len(order)), -float(r["week_pts"])))
    return {"roster_id": t.roster_id, "name": t.name, "record": _record(t), "players": rows}


@app.get("/api/league/{league_id}/odds")
def odds(league_id: str) -> dict[str, Any]:
    e = _load(league_id)
    with e.lock, friendly():
        if e.odds is None:
            ctx, lg = e.ctx, e.ctx.league
            out = simulate_players(ctx.inputs, ctx.tables, ODDS_SIMS, 7)
            res = simulate_league(out, ctx.inputs, lg, ctx.base)
            weeks = [i for i, w in enumerate(ctx.inputs.weeks) if w <= lg.last_regular_week]
            ppw = (
                res.team_points[:, :, weeks].mean(axis=(0, 2)) if weeks else np.zeros(len(lg.teams))
            )
            rows: list[dict[str, Any]] = [
                {
                    "roster_id": t.roster_id,
                    "name": t.name,
                    "record": _record(t),
                    "points_for": round(t.points_for, 1),
                    "ppw": round(float(ppw[i]), 2),
                    "playoff": round(float(res.playoff_prob[i]), 4),
                    "title": round(float(res.title_prob[i]), 4),
                }
                for i, t in enumerate(lg.teams)
            ]
            rows.sort(key=lambda r: -float(r["playoff"]))
            e.odds = {"sims": ODDS_SIMS, "rows": rows}
        return e.odds


@app.get("/api/league/{league_id}/trades")
def trades(league_id: str) -> list[dict[str, Any]]:
    ctx = _load(league_id).ctx
    with friendly():
        found = market.league_trades(ctx.src, ctx.league, ctx.market)
    return [
        {
            "week": t.week,
            "sides": [
                {
                    "roster_id": rid,
                    "name": ctx.league.team(rid).name,
                    "got": [ctx.name(p) for p in got],
                    "value": t.value_by_roster.get(rid, 0),
                }
                for rid, got, _sent in t.sides
            ],
        }
        for t in found
    ]


def _this_week(ctx: engine.Context, roster: list[int]) -> float:
    e, pos = ctx.inputs.expected[:, 0], ctx.inputs.pos
    return optimal_lineup(
        [float(e[i]) for i in roster], [int(pos[i]) for i in roster], ctx.league.slots
    )[0]


@app.post("/api/preview")
def preview(b: PreviewBody) -> dict[str, Any]:
    ctx = _load(b.league_id).ctx
    if not b.give and not b.get:
        return {"me": 0.0, "them": 0.0, "shape": "0 for 0",
                "note": "Pick players on either side to see the change."}  # fmt: skip
    tr = _trade(ctx, b)
    out = []
    sides = ((b.team, tr.a_gives, tr.b_gives), (b.partner, tr.b_gives, tr.a_gives))
    for owner, lose, gain in sides:
        now = [ctx.index[p] for p in ctx.league.team(owner).players if p in ctx.index]
        after = [i for i in now if ctx.inputs.ids[i] not in lose] + [ctx.index[p] for p in gain]
        out.append(round(_this_week(ctx, after) - _this_week(ctx, now), 2))
    n_give, n_get = len(b.give), len(b.get)
    partner = ctx.league.team(b.partner).name
    if n_get > n_give:
        note = (
            "You receive more players than you send, so the full grade cuts your least useful "
            f"bench player, and {partner} gets an open spot filled by the best free agent."
        )
    elif n_get < n_give:
        note = (
            f"You send more players than you receive, so the full grade adds the best free "
            f"agent to your open spot, and {partner} cuts its least useful bench player."
        )
    else:
        note = "Same number of players each way, so no roster moves are needed."
    note += " This preview only looks at this week's best lineup."
    return {"me": out[0], "them": out[1], "shape": f"{n_give} for {n_get}", "note": note}


def _stream(work: Callable[[Callable[[str], None]], dict[str, Any]]) -> StreamingResponse:
    """Run `work` in a thread; stream its progress messages, then its result or an error."""
    q: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def run() -> None:
        try:
            result = work(lambda m: q.put({"type": "progress", "message": m}))
            q.put({"type": "result", "result": result})
        except TradeError as e:
            q.put({"type": "error", "message": str(e)})
        except NETWORK_ERRORS:
            q.put({"type": "error", "message": DOWN})
        except Exception:
            log.exception("request failed")
            q.put({"type": "error", "message": "Something went wrong. Try again in a minute."})
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    def lines() -> Iterator[str]:
        while (item := q.get()) is not None:
            yield json.dumps(item) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.post("/api/grade")
def grade(b: GradeBody) -> StreamingResponse:
    e = _load(b.league_id)
    if b.sims not in SIM_CHOICES:
        raise HTTPException(422, "Choose 3,000 or 10,000 simulated seasons.")
    tr = _trade(e.ctx, b)
    return _stream(lambda progress: _grade_result(
        e, evaluate(e.ctx, tr, sims=b.sims, progress=progress)))  # fmt: skip


def _offer(e: Entry, s: finder.Suggestion) -> dict[str, Any]:
    partner = e.ctx.league.team(s.trade.team_b)
    return {
        "partner": {
            "roster_id": partner.roster_id,
            "name": partner.name,
            "record": _record(partner),
        },
        "give": [_brief(e, p) for p in s.trade.a_gives],
        "get": [_brief(e, p) for p in s.trade.b_gives],
        "me": _delta(s.a_title),
        "them": _delta(s.b_title),
        "me_points": round(s.a_points, 2),
        "them_points": round(s.b_points, 2),
        "market_gap": None if s.market_gap is None else round(s.market_gap, 4),
        "moves": [
            {
                "roster_id": m.roster_id,
                "adds": [_brief(e, p) for p in m.adds],
                "drops": [_brief(e, p) for p in m.drops],
            }
            for m in s.moves
            if m.adds or m.drops
        ],
    }


@app.post("/api/find")
def find(b: FindBody) -> StreamingResponse:
    e = _load(b.league_id)
    ctx = e.ctx
    _team(ctx, b.team)

    def work(progress: Callable[[str], None]) -> dict[str, Any]:
        checked = len(finder.candidates(ctx, b.team))
        offers = [_offer(e, s) for s in finder.find_trades(ctx, b.team, progress=progress)]
        return {"checked": checked, "rosters": len(ctx.league.teams) - 1, "offers": offers}

    return _stream(work)


@app.post("/api/balance")
def balance(b: PreviewBody) -> StreamingResponse:
    """Small changes to a trade that make it a win for both teams ("Make it fair")."""
    e = _load(b.league_id)
    tr = _trade(e.ctx, b)

    def work(progress: Callable[[str], None]) -> dict[str, Any]:
        offers = []
        for s in finder.balance(e.ctx, tr, progress=progress):
            o = _offer(e, s)
            o["change"] = {  # what differs from the trade the user built
                "give_added": [_brief(e, p) for p in s.trade.a_gives if p not in tr.a_gives],
                "give_removed": [_brief(e, p) for p in tr.a_gives if p not in s.trade.a_gives],
                "get_added": [_brief(e, p) for p in s.trade.b_gives if p not in tr.b_gives],
                "get_removed": [_brief(e, p) for p in tr.b_gives if p not in s.trade.b_gives],
            }
            offers.append(o)
        return {"tried": len(finder.tweaks(e.ctx, tr)), "offers": offers}

    return _stream(work)


@app.get("/api/report")
def report() -> dict[str, str]:
    path = Path(__file__).resolve().parent.parent / "REPORT.md"
    if not path.exists():
        raise HTTPException(404, "The accuracy report hasn't been generated yet.")
    return {"markdown": path.read_text()}


# ------------------------------------------------------------------ the built web app

NOT_BUILT = (
    "<!doctype html><title>Trade calculator</title><body style='font-family:system-ui;"
    "max-width:36rem;margin:3rem auto'><h1>The web app isn't built yet</h1>"
    "<p>In a terminal, run:</p><pre>cd web &amp;&amp; npm install &amp;&amp; npm run build</pre>"
    "<p>Then reload this page.</p></body>"
)


@app.get("/{path:path}", include_in_schema=False)
def web(path: str) -> Any:
    if path == "api" or path.startswith("api/"):
        raise HTTPException(404, "There's no such address.")
    index = DIST / "index.html"
    if not index.exists():
        return HTMLResponse(NOT_BUILT)
    f = (DIST / path).resolve()
    if path and f.is_file() and f.is_relative_to(DIST.resolve()):
        return FileResponse(f)
    return FileResponse(index)  # any other page: the single-page app handles it
