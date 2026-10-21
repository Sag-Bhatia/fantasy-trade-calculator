import streamlit as st

from ffcalc import ui

ctx = st.session_state.get("ctx")
if ctx is None:
    ui.welcome()
    st.stop()

me = st.session_state["team"]
st.title("Grade a trade")
others = [t.roster_id for t in ctx.league.teams if t.roster_id != me]
prefill = st.session_state.pop("prefill", None)  # from the trade finder
if prefill:
    st.session_state["partner"] = prefill[0]
    st.session_state["give"] = prefill[1]
    st.session_state[f"get_{prefill[0]}"] = prefill[2]
partner = st.selectbox(
    "Trade with", others, format_func=lambda r: ui.team_label(ctx, r), key="partner"
)
c1, c2 = st.columns(2)
give = c1.multiselect(
    "You give",
    ui.roster_options(ctx, me),
    format_func=lambda p: ui.player_label(ctx, p),
    placeholder="Search your players",
    key="give",
    help="Points are each player's projected total for the rest of the season.",
)
get = c2.multiselect(
    f"You get from {ctx.league.team(partner).name}",
    ui.roster_options(ctx, partner),
    format_func=lambda p: ui.player_label(ctx, p),
    placeholder="Search their players",
    key=f"get_{partner}",
)
c3, c4 = st.columns([2, 1], vertical_alignment="bottom")
depth = c3.radio("How many seasons to simulate", list(ui.SIM_CHOICES), horizontal=True,
                 help="Full is more precise and takes about half a minute.")  # fmt: skip
go = c4.button("Grade this trade", type="primary", disabled=not (give or get), width="stretch")

key = (
    ctx.league.league_id,
    me,
    partner,
    tuple(sorted(give)),
    tuple(sorted(get)),
    ui.SIM_CHOICES[depth],
)
if go:
    res = ui.evaluate_trade(ctx, me, partner, give, get, ui.SIM_CHOICES[depth])
    if res is not None:
        st.session_state["last_trade"] = (key, res)
last = st.session_state.get("last_trade")
if not last or last[0] != key:
    if not (give or get):
        st.caption("Pick players on either side, then grade the trade.")
    st.stop()

res = last[1]
ui.scoreboard(res)
st.markdown(ui.plain_summary(ctx, res))
for w in res.warnings:
    st.caption(w)

st.subheader("Why it helps or hurts")
side = st.segmented_control("Show", ["You", ctx.league.team(partner).name], default="You",
                            label_visibility="collapsed")  # fmt: skip
r = res.b if side and side != "You" else res.a
st.altair_chart(ui.drive_chart(r), width="stretch")
st.caption(
    "Each bar is how much one part of the picture changes the trade's value, in expected "
    "starter points for the rest of the season. They're added one at a time in this order "
    f"and sum to the trade score of {r.score:+.1f} points. "
    "Starters: who you'd start in an average week. Depth: your bench covering injuries. "
    "Waiver wire: the free agent you'd add or the player you'd cut. Schedule & byes: real "
    "matchups and bye weeks. Injury risk: current injury news. Playoff weeks: playoff-week "
    "points counted double."
)

t1, t2, t3, t4 = st.tabs(["Week by week", "Market check", "Roster moves", "The numbers"])
with t1:
    st.altair_chart(ui.weekly_band(ctx, res, ctx.team_index(me)), width="stretch")
    st.caption(
        "Lines: a typical week, with the trade (green) and without it (grey, dashed). Shaded "
        "band: from a bad week to a great week (the middle 80% of simulated weeks)."
    )
with t2:
    ui.market_panel(ctx, res)
with t3:
    ui.moves_panel(ctx, res)
with t4:
    ui.numbers_table(res)
