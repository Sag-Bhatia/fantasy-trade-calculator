import streamlit as st

from ffcalc import ui

ctx = st.session_state.get("ctx")
if ctx is None:
    ui.welcome()
    st.stop()

me = st.session_state["team"]
st.title("Find trades")
st.markdown(
    "Searches every other roster for one- and two-player deals that raise **both** teams' "
    "title odds, so the other manager has a reason to say yes. It checks thousands of "
    "combinations and takes about a minute."
)
if st.button("Find trades for my team", type="primary"):
    with ui.friendly_errors(), st.status("Searching", expanded=True) as s:
        found = ui.run_finder(ctx.key, ctx, me, lambda m: s.write(m))
        n = len(found)
        s.update(label=f"Found {n} trade{'s' if n != 1 else ''}", state="complete", expanded=False)
    st.session_state["found"] = (ctx.league.league_id, me, found)

last = st.session_state.get("found")
if last and last[:2] == (ctx.league.league_id, me):
    found = last[2]
    if not found:
        st.info(
            "No trade clearly helps both teams right now. That's common when rosters are "
            "balanced; try again after injuries or waiver moves shake things up."
        )
    for i, sug in enumerate(found):
        ui.suggestion_card(ctx, sug, i)
    if found:
        st.caption("Title odds changes come from 10,000 simulated seasons; each one beats its "
                   "own simulation noise. " + ui.HELP["se"])  # fmt: skip
