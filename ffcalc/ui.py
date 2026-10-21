"""Streamlit building blocks shared by the pages (spec §10). Pages only compose these."""

from __future__ import annotations

import html
import io
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import altair as alt
import httpx
import numpy as np
import polars as pl
import streamlit as st

from . import engine, market
from .league import LeagueNotFound
from .simulate import simulate_league, simulate_players
from .trade import COMPONENTS, TeamResult, Trade, TradeError, TradeResult, evaluate

GAIN = "#2E8B57"
LOSS = "#C8372D"


def palette() -> tuple[str, str]:
    """Gain/loss colors with >= 4.5:1 text contrast on the active theme's background."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    return ("#5CC48C", "#FF7A6E") if dark else ("#1E6B43", "#B4281F")


# Tests swap these for offline fakes (AppTest runs the app in this same process).
source_factory: Callable[[], Any] | None = None  # None → live sources
build_history = True

SIM_CHOICES = {"Quick (3,000 seasons)": 3_000, "Full (10,000 seasons)": 10_000}

HELP = {
    "title": "Title odds: the share of simulated seasons your team wins the championship.",
    "playoff": "Playoff odds: the share of simulated seasons your team makes the playoffs.",
    "ppw": "Starter points per week: what your best lineup is expected to score, rest of season.",
    "se": "± is the standard error. Differences smaller than about twice this are noise.",
    "score": "Trade score: the change in expected starter points over the rest of the season, "
    "with playoff weeks counted double.",
    "replacement": "Replacement level: an open roster spot is filled with the best free agent "
    "actually available in your league, not a generic player.",
}

CSS = """
<style>
.ff-board{background:#123524;color:#F3F5F1;border-radius:6px;padding:1.4rem 1.6rem 1.2rem;
  border:1px solid #2F5A41;box-shadow:inset 0 3px 0 #2E8B57;margin:.4rem 0 1.2rem}
.ff-verdict{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:2.5rem;
  line-height:1;letter-spacing:.01em;margin:0 0 .35rem}
.ff-sub{color:#B9C6BD;font-size:1rem;margin:0 0 1.1rem;max-width:62ch}
.ff-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.2rem}
.ff-team{border-top:2px solid #2F5A41;padding-top:.7rem}
.ff-name{font-weight:600;font-size:1rem;color:#DDE5DF;margin-bottom:.2rem;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.ff-big{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:3.6rem;line-height:.95;
  font-variant-numeric:tabular-nums}
.ff-unit{font-size:1.1rem;font-weight:600;margin-left:.25rem;color:#B9C6BD}
.ff-gain{color:#7FD3A3}.ff-loss{color:#FF8A7E}.ff-even{color:#F3F5F1}
.ff-line{color:#B9C6BD;font-size:.92rem;margin-top:.35rem;font-variant-numeric:tabular-nums}
@media (max-width:640px){.ff-row{grid-template-columns:1fr}.ff-big{font-size:3rem}
  .ff-verdict{font-size:2rem}.ff-board{padding:1.1rem}}
</style>
"""


def style() -> None:
    st.html(CSS)


# ------------------------------------------------------------------ loading


@contextmanager
def friendly_errors() -> Any:
    """Show what went wrong and what to do, never a stack trace."""
    try:
        yield
    except LeagueNotFound:
        st.error(
            "Sleeper doesn't know that league ID. Open your league in the Sleeper app or on "
            "sleeper.com and copy the long number from the league's web address."
        )
        st.stop()
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except (ConnectionError, OSError, RuntimeError, httpx.HTTPError) as e:
        st.error(
            "A data source didn't answer, so the numbers can't be built right now. "
            f"Check your internet connection and try again in a minute. ({type(e).__name__})"
        )
        st.stop()


@st.cache_resource(show_spinner=False, ttl=600, max_entries=4)
def load_context(
    league_id: str,
    fp_csv: bytes | None = None,
    fp_week: int | None = None,
    _progress: Callable[[str], None] = lambda _: None,
) -> engine.Context:
    fp_lines = None
    if fp_csv:
        fp_lines = _fp_lines(league_id, fp_csv, fp_week)
    src = source_factory() if source_factory else None
    return engine.prepare(
        league_id, src=src, fp_lines=fp_lines, build_history=build_history, progress=_progress
    )


def _fp_lines(league_id: str, fp_csv: bytes, week: int | None) -> pl.DataFrame:
    from .clients import fantasypros_csv

    base = load_context(league_id)
    lines, unmatched = fantasypros_csv.load(
        io.BytesIO(fp_csv), None, base.table, season=base.league.season, week=week
    )
    st.session_state["fp_unmatched"] = unmatched
    return lines


def context(league_id: str, status_label: str = "Loading your league") -> engine.Context:
    fp = st.session_state.get("fp_csv")
    fp_week = st.session_state.get("fp_week")
    with friendly_errors(), st.status(status_label, expanded=False) as s:
        ctx = load_context(league_id, fp, fp_week, lambda m: s.update(label=f"{m}…"))
        s.update(label="League loaded", state="complete")
    return ctx


@st.cache_resource(show_spinner=False, ttl=600, max_entries=4)
def league_odds(ctx_id: int, _ctx: engine.Context, sims: int = 3_000) -> pl.DataFrame:
    """Current playoff/title odds for every team (no trade)."""
    out = simulate_players(_ctx.inputs, _ctx.tables, sims, 7)
    res = simulate_league(out, _ctx.inputs, _ctx.league, _ctx.base)
    lg = _ctx.league
    weeks = [i for i, w in enumerate(_ctx.inputs.weeks) if w <= lg.last_regular_week]
    return pl.DataFrame(
        {
            "Team": [t.name for t in lg.teams],
            "roster_id": [t.roster_id for t in lg.teams],
            "Record": [f"{t.wins}-{t.losses}" + (f"-{t.ties}" if t.ties else "") for t in lg.teams],
            "Points for": [round(t.points_for, 1) for t in lg.teams],
            "Projected points / week": res.team_points[:, :, weeks].mean(axis=(0, 2)).round(1)
            if weeks
            else np.zeros(len(lg.teams)),
            "Playoff odds": res.playoff_prob,
            "Title odds": res.title_prob,
        }
    ).sort("Title odds", descending=True)


@st.cache_resource(show_spinner=False, ttl=600, max_entries=8)
def run_trade(ctx_id: int, _ctx: engine.Context, team_a: int, team_b: int,
              give: tuple[str, ...], get: tuple[str, ...], sims: int,
              _progress: Callable[[str], None] = lambda _: None) -> TradeResult:  # fmt: skip
    return evaluate(
        _ctx, Trade(team_a, team_b, list(give), list(get)), sims=sims, progress=_progress
    )


def evaluate_trade(ctx: engine.Context, team_a: int, team_b: int, give: list[str], get: list[str],
                   sims: int) -> TradeResult | None:  # fmt: skip
    with st.status("Grading the trade", expanded=True) as s:
        try:
            res = run_trade(id(ctx), ctx, team_a, team_b, tuple(sorted(give)),
                            tuple(sorted(get)), sims, lambda m: s.write(m))  # fmt: skip
        except TradeError as e:
            s.update(label="This trade can't be graded", state="error")
            st.error(str(e))
            return None
        except (ConnectionError, OSError, RuntimeError, httpx.HTTPError):
            s.update(label="This trade can't be graded right now", state="error")
            st.error("A data source didn't answer while grading. Try again in a minute.")
            return None
        s.update(label="Trade graded", state="complete", expanded=False)
    return res


# ------------------------------------------------------------------ players


def player_label(ctx: engine.Context, sid: str) -> str:
    row = ctx.table.filter(pl.col("sleeper_id") == sid)
    if row.is_empty():
        return sid
    r = row.row(0, named=True)
    team = r["team"] or "free agent"
    tag = f"  {r['injury_status']}" if r["injury_status"] else ""
    ros = ""
    if sid in ctx.index:
        pts = float(ctx.inputs.expected[ctx.index[sid]].sum())
        ros = f"  {pts:.0f} pts"
    return f"{r['name']} ({r['position']}, {team}){tag}{ros}"


def roster_options(ctx: engine.Context, roster_id: int) -> list[str]:
    """A team's players, best rest-of-season first."""
    t = ctx.league.team(roster_id)
    known = [p for p in t.players if p in ctx.index]
    return sorted(known, key=lambda p: -float(ctx.inputs.expected[ctx.index[p]].sum()))


def team_label(ctx: engine.Context, roster_id: int) -> str:
    t = ctx.league.team(roster_id)
    return f"{t.name} ({t.wins}-{t.losses})"


# ------------------------------------------------------------------ results


def _pp(x: float) -> str:
    return f"{x * 100:+.1f}"


def _tone(delta_pp: float, se_pp: float) -> str:
    if abs(delta_pp) < max(0.5, 2 * se_pp):
        return "ff-even"
    return "ff-gain" if delta_pp > 0 else "ff-loss"


def _team_block(r: TeamResult) -> str:
    d, se = r.title.value * 100, r.title.se * 100
    return (
        f'<div class="ff-team"><div class="ff-name">{html.escape(r.name)}</div>'
        f'<div class="ff-big {_tone(d, se)}">{d:+.1f}<span class="ff-unit">%</span></div>'
        f'<div class="ff-line">Title odds {r.title_before:.1%} to {r.title_after:.1%} (± {se:.1f})</div>'
        f'<div class="ff-line">Playoff odds {r.playoff_before:.1%} to {r.playoff_after:.1%}</div>'
        f'<div class="ff-line">Starters {r.points_per_week.value:+.1f} points a week</div></div>'
    )


VERDICT_TEXT = {
    "Accept": "Take this trade",
    "Decline": "Pass on this trade",
    "Roughly even": "This trade is close to even",
}


def scoreboard(res: TradeResult) -> None:
    a = res.a
    d, se = a.title.value * 100, a.title.se * 100
    if res.verdict == "Accept":
        why = f"Your title odds go up {d:.1f} percentage points, well beyond the simulation noise."
    elif res.verdict == "Decline":
        why = (
            f"Your title odds go down {-d:.1f} percentage points, well beyond the simulation noise."
        )
    else:
        why = f"The change in your title odds ({d:+.1f} percentage points, ± {se:.1f}) is too small to call."
    st.html(
        f'<div class="ff-board" role="region" aria-label="Trade verdict">'
        f'<div class="ff-verdict">{VERDICT_TEXT[res.verdict]}</div>'
        f'<p class="ff-sub">{html.escape(why)} The ± is the simulation&rsquo;s margin of error.</p>'
        f'<div class="ff-row">{_team_block(res.a)}{_team_block(res.b)}</div></div>'
    )


def drive_chart(r: TeamResult, height: int = 260) -> alt.Chart:
    """Why it helps or hurts: gains run right, losses left, gridlines every 10 points."""
    df = pl.DataFrame(
        {"part": list(r.decomposition), "points": [round(v, 1) for v in r.decomposition.values()]}
    ).with_columns(
        pl.when(pl.col("points") >= 0).then(pl.lit("gain")).otherwise(pl.lit("loss")).alias("sign")
    )
    lim = max(10.0, float(np.abs(df["points"].to_numpy()).max(initial=0)) * 1.25)
    step = 10 if lim <= 80 else 20 if lim <= 200 else 50
    x = alt.X(
        "points:Q",
        title="Expected starter points, rest of season",
        scale=alt.Scale(domain=[-lim, lim]),
        axis=alt.Axis(
            values=list(np.arange(-np.ceil(lim / step) * step, lim + 1, step)), grid=True
        ),
    )
    y = alt.Y("part:N", sort=list(COMPONENTS), title=None, axis=alt.Axis(labelLimit=120))
    gain, loss = palette()
    color = alt.Color(
        "sign:N", scale=alt.Scale(domain=["gain", "loss"], range=[gain, loss]), legend=None
    )
    bars = (
        alt.Chart(df.to_pandas())
        .mark_bar(cornerRadius=2, height=22)
        .encode(
            x=x,
            y=y,
            color=color,
            tooltip=[
                alt.Tooltip("part:N", title="Part"),
                alt.Tooltip("points:Q", title="Points", format="+.1f"),
            ],
        )
    )
    labels = (
        alt.Chart(df.to_pandas())
        .mark_text(align="left", dx=6, fontSize=13, fontWeight=600)
        .encode(
            x=alt.X("points:Q", scale=alt.Scale(domain=[-lim, lim])),
            y=y,
            text=alt.Text("points:Q", format="+.1f"),
            color=color,
        )
        .transform_filter("datum.points >= 0")
    )
    neg = (
        alt.Chart(df.to_pandas())
        .mark_text(align="right", dx=-6, fontSize=13, fontWeight=600)
        .encode(
            x=alt.X("points:Q", scale=alt.Scale(domain=[-lim, lim])),
            y=y,
            text=alt.Text("points:Q", format="+.1f"),
            color=color,
        )
        .transform_filter("datum.points < 0")
    )
    zero = alt.Chart().mark_rule(strokeWidth=2, opacity=0.7).encode(x=alt.datum(0))
    return (bars + labels + neg + zero).properties(height=height)


def weekly_band(ctx: engine.Context, res: TradeResult, team_index: int) -> alt.Chart:
    """Median and middle-80% range of a team's weekly score, before and after."""
    rows = []
    for name, sr in (("Without the trade", res.before), ("With the trade", res.after)):
        tp = sr.team_points[:, team_index, :]
        for w, wk in enumerate(ctx.inputs.weeks):
            q = np.percentile(tp[:, w], [10, 50, 90])
            rows.append((name, wk, *map(float, q)))
    df = pl.DataFrame(rows, schema=["case", "week", "p10", "p50", "p90"], orient="row").to_pandas()
    cases = ["Without the trade", "With the trade"]
    color = alt.Color("case:N", scale=alt.Scale(domain=cases, range=["#8C9A90", palette()[0]]), title=None,
                      legend=alt.Legend(orient="top", symbolType="stroke", symbolStrokeWidth=3))  # fmt: skip
    dash = alt.StrokeDash(
        "case:N", scale=alt.Scale(domain=cases, range=[[5, 4], [1, 0]]), legend=None
    )
    x = alt.X("week:O", title="Week", axis=alt.Axis(labelAngle=0))
    base = alt.Chart(df).encode(x=x, color=color)
    band = base.mark_area(opacity=0.16).encode(
        y=alt.Y("p10:Q", title="Starter points", scale=alt.Scale(zero=False)), y2="p90:Q"
    )
    line = base.mark_line(point=True, strokeWidth=2.5).encode(
        y="p50:Q",
        strokeDash=dash,
        tooltip=[alt.Tooltip("case:N", title=" "), alt.Tooltip("week:O", title="Week"),
                 alt.Tooltip("p50:Q", title="Typical week", format=".1f"),
                 alt.Tooltip("p10:Q", title="Bad week (10%)", format=".1f"),
                 alt.Tooltip("p90:Q", title="Great week (90%)", format=".1f")],
    )  # fmt: skip
    layers = band + line
    start = ctx.league.playoff_week_start
    if start in ctx.inputs.weeks:
        rule = (
            alt.Chart(pl.DataFrame({"week": [start]}).to_pandas())
            .mark_rule(strokeWidth=1, opacity=0.5)
            .encode(x=alt.X("week:O"))
        )
        label = (
            alt.Chart(pl.DataFrame({"week": [start], "t": ["Playoffs"]}).to_pandas())
            .mark_text(align="left", dx=4, dy=-130, fontSize=12, opacity=0.8)
            .encode(x=alt.X("week:O"), text="t:N")
        )
        layers = layers + rule + label
    return layers.properties(height=300)


def market_panel(ctx: engine.Context, res: TradeResult) -> None:
    if res.market is None:
        st.info(
            "Market values from FantasyCalc aren't available right now, so there's no market check."
        )
        return
    m = res.market
    c1, c2, c3 = st.columns(3)
    c1.metric("Market value you give", f"{m.a_gives.total:,}")
    c2.metric(
        "Market value you get", f"{m.b_gives.total:,}", delta=f"{m.gap:+.0%} vs what you give"
    )
    if res.acceptance is not None:
        c3.metric(
            f"Chance {res.b.name} says yes (rough heuristic)",
            f"{res.acceptance:.0%}",
            help="A rough rule of thumb, not a measurement: it rises when the trade helps their "
            "title odds and when they get more market value. There's no data on rejected "
            "trades to check it against.",
        )
    if m.overpay:
        st.warning("By market value you're giving up more than 15% extra.")
    rows = [(ctx.name(p), "You give", v) for p, v in m.a_gives.values.items()] + [
        (ctx.name(p), "You get", v) for p, v in m.b_gives.values.items()
    ]
    st.dataframe(
        pl.DataFrame(rows, schema=["Player", "Side", "FantasyCalc value"], orient="row"),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "FantasyCalc values come from real trades on fantasy sites; 0 means the player isn't in their list."
    )


def moves_panel(ctx: engine.Context, res: TradeResult) -> None:
    any_moves = False
    for r in (res.a, res.b):
        if r.moves.adds or r.moves.drops:
            any_moves = True
            adds = ", ".join(player_label(ctx, p) for p in r.moves.adds) or "nobody"
            drops = ", ".join(player_label(ctx, p) for p in r.moves.drops) or "nobody"
            st.markdown(f"**{r.name}** picks up {adds} and drops {drops}.")
    if not any_moves:
        st.markdown("Both rosters stay legal without any pickups or drops.")
    st.caption(HELP["replacement"])


def league_trades_panel(ctx: engine.Context) -> None:
    if ctx.src is None or ctx.market.is_empty():
        return
    try:
        trades = market.league_trades(ctx.src, ctx.league, ctx.market)
    except Exception:
        st.caption("This league's trade history couldn't be loaded right now.")
        return
    if not trades:
        st.caption("No trades in this league yet this season.")
        return
    for t in trades[:8]:
        parts = []
        for rid, got, _sent in t.sides:
            names = ", ".join(ctx.name(p) for p in got) or "nothing"
            parts.append(
                f"**{ctx.league.team(rid).name}** got {names} ({t.value_by_roster.get(rid, 0):,})"
            )
        st.markdown(f"Week {t.week}: " + "; ".join(parts))
    st.caption(
        "Numbers in parentheses are today's FantasyCalc market values of what each team received."
    )


def plain_summary(ctx: engine.Context, res: TradeResult) -> str:
    """One paragraph in plain English about the biggest reasons."""
    parts = sorted(res.a.decomposition.items(), key=lambda kv: -abs(kv[1]))
    top = [(k, v) for k, v in parts if abs(v) >= 1][:2]
    if not top:
        return "None of the parts of this trade move your expected points much."
    says = {
        "Starters": "who you'd start",
        "Depth": "bench depth for injuries",
        "Waiver wire": "the free agent you'd pick up or the player you'd cut",
        "Schedule & byes": "matchups and bye weeks",
        "Injury risk": "current injury news",
        "Playoff weeks": "playoff-week matchups (counted double)",
    }
    bits = [f"{says[k]} ({v:+.0f} points)" for k, v in top]
    return "The biggest reasons are " + " and ".join(bits) + "."


# ------------------------------------------------------------------ sidebar / onboarding

DEFAULT_LEAGUE = ""


def sidebar() -> engine.Context | None:
    """League and team pickers. Returns the loaded league, or None before one is chosen."""
    qp = st.query_params
    with st.sidebar:
        st.header("Your league")
        league_id = st.text_input(
            "Sleeper league ID",
            value=qp.get("league", st.session_state.get("league_id", DEFAULT_LEAGUE)),
            placeholder="e.g. 1400221038771437568",
            help="The long number in your league's web address on sleeper.com.",
        ).strip()
        with st.expander("Don't know your league ID?"):
            username = st.text_input("Sleeper username", key="username")
            if username:
                find_league(username)
        if not league_id:
            return None
        st.session_state["league_id"] = league_id
        qp["league"] = league_id
        ctx = context(league_id)
        ids = [t.roster_id for t in ctx.league.teams]
        raw = qp.get("team", st.session_state.get("team", ids[0]))
        saved = int(raw) if str(raw).isdigit() else ids[0]
        team = st.selectbox(
            "Your team",
            ids,
            index=ids.index(saved) if saved in ids else 0,
            format_func=lambda r: team_label(ctx, r),
        )
        st.session_state["team"] = team
        qp["team"] = str(team)
        lg = ctx.league
        st.caption(f"{lg.num_teams} teams, week {lg.current_week} of the {lg.season} season")
        for w in ctx.warnings:
            st.caption(w)
    return ctx


def find_league(username: str) -> None:
    from .league import find_leagues
    from .sources import live

    try:
        src = source_factory() if source_factory else live()
        season = int(src.sleeper.state()["season"])
        leagues = find_leagues(username, season, src.sleeper)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception:
        st.error("Sleeper didn't answer. Try again in a minute.")
        return
    if not leagues:
        st.write("No leagues found for that username this season.")
    for lg in leagues:
        st.code(lg["league_id"], language=None)
        st.caption(f"{lg['name']} ({lg['total_rosters']} teams)")


def welcome() -> None:
    st.title("Grade a fantasy football trade")
    st.markdown(
        "Paste your Sleeper league ID in the sidebar. The calculator plays out the rest of "
        "your season 10,000 times with and without a trade, using your league's real "
        "rosters, scoring, schedule, playoff format and free agents, then tells you how your "
        "title and playoff odds change and why."
    )
    st.markdown(
        "**Finding your league ID:** open your league on sleeper.com. The web address ends in "
        "a long number, like `sleeper.com/leagues/1400221038771437568`. That number is the ID. "
        "In the app, it's under League settings."
    )


def numbers_table(res: TradeResult) -> None:
    rows = []
    for r in (res.a, res.b):
        rows += [
            (r.name, "Title odds", f"{r.title_before:.1%}", f"{r.title_after:.1%}",
             f"{_pp(r.title.value)} ± {r.title.se * 100:.1f}"),
            (r.name, "Playoff odds", f"{r.playoff_before:.1%}", f"{r.playoff_after:.1%}",
             f"{_pp(r.playoff.value)} ± {r.playoff.se * 100:.1f}"),
            (r.name, "Starter points / week", "", "",
             f"{r.points_per_week.value:+.2f} ± {r.points_per_week.se:.2f}"),
            (r.name, "Trade score (playoff weeks ×2)", "", "", f"{r.score:+.1f} points"),
        ]  # fmt: skip
    st.dataframe(
        pl.DataFrame(rows, schema=["Team", "Measure", "Without", "With", "Change"], orient="row"),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Odds changes are in percentage points. {res.sims:,} simulated seasons, "
        f"seed {res.seed}. {HELP['se']} {HELP['score']}"
    )


@st.cache_resource(show_spinner=False, ttl=600, max_entries=8)
def run_finder(ctx_id: int, _ctx: engine.Context, team: int,
               _progress: Callable[[str], None] = lambda _: None) -> list[Any]:  # fmt: skip
    from .finder import find_trades

    return find_trades(_ctx, team, progress=_progress)


def suggestion_card(ctx: engine.Context, s: Any, i: int) -> None:
    partner = ctx.league.team(s.trade.team_b).name
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 3, 2])
        c1.markdown("**You give**\n\n" + "\n\n".join(player_label(ctx, p) for p in s.trade.a_gives))
        c2.markdown(
            f"**You get from {partner}**\n\n"
            + "\n\n".join(player_label(ctx, p) for p in s.trade.b_gives)
        )
        c3.metric("Your title odds", f"{s.a_title.value * 100:+.1f}%", help=HELP["title"])
        c3.metric(f"{partner}'s title odds", f"{s.b_title.value * 100:+.1f}%")
        extra = []
        if s.market_gap is not None:
            extra.append(f"Market value {s.market_gap:+.0%} for you")
        for m in s.moves:
            if m.adds:
                extra.append(
                    f"{ctx.league.team(m.roster_id).name} adds {', '.join(ctx.name(p) for p in m.adds)}"
                )
            if m.drops:
                extra.append(
                    f"{ctx.league.team(m.roster_id).name} drops {', '.join(ctx.name(p) for p in m.drops)}"
                )
        st.caption(
            f"Starter points a week: you {s.a_points:+.1f}, them {s.b_points:+.1f}. "
            + "; ".join(extra)
            + ("." if extra else "")
        )
        if st.button("Grade this trade in detail", key=f"open_{i}"):
            st.session_state["prefill"] = (
                s.trade.team_b,
                list(s.trade.a_gives),
                list(s.trade.b_gives),
            )
            st.switch_page("views/trade.py")


SLOT_NAMES = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "Defense",
    "FLEX": "Flex (RB/WR/TE)", "WRRB_FLEX": "Flex (RB/WR)", "REC_FLEX": "Flex (WR/TE)",
    "SUPER_FLEX": "Superflex (QB/RB/WR/TE)",
}  # fmt: skip


def league_header(ctx: engine.Context) -> None:
    lg = ctx.league
    title = lg.name if lg.name and lg.name.strip().lower() != "league" else "Your league"
    st.title(title)
    sc = lg.scoring
    ppr = sc.get("rec", 0)
    fmt = {1: "Full PPR", 0.5: "Half PPR", 0: "Standard"}.get(ppr, f"{ppr:g} per catch")
    te = sc.get("bonus_rec_te", 0)
    detail = (
        f"{ppr:g} point{'s' if ppr != 1 else ''} per catch, {sc.get('pass_td', 4):g} per passing TD"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teams", lg.num_teams)
    c2.metric("Scoring", fmt, help=detail + (f", tight ends +{te:g} per catch" if te else ""))
    c3.metric(
        "Playoffs", f"{lg.playoff_teams} teams", help=f"Start in week {lg.playoff_week_start}"
    )
    c4.metric("Trade deadline", f"Week {lg.trade_deadline}" if lg.trade_deadline else "None")
    st.caption("Starting lineup: " + ", ".join(SLOT_NAMES.get(x, x) for x in lg.slots)
               + f", plus {lg.bench_size} bench spots.")  # fmt: skip


def report_text() -> str | None:
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "REPORT.md"
    return path.read_text() if path.exists() else None
