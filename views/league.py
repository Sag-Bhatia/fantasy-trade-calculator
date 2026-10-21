import polars as pl
import streamlit as st

from ffcalc import ui

ctx = st.session_state.get("ctx")
if ctx is None:
    ui.welcome()
    st.stop()

lg = ctx.league
ui.league_header(ctx)

st.subheader("Where everyone stands")
with st.spinner("Simulating the rest of the season"):
    odds = ui.league_odds(ctx.key, ctx)
st.dataframe(
    odds.drop("roster_id").with_columns(pl.col("Playoff odds") * 100, pl.col("Title odds") * 100),
    height=36 * (odds.height + 1) + 3,
    hide_index=True,
    width="stretch",
    column_config={
        "Playoff odds": st.column_config.ProgressColumn(
            "Playoff odds", format="%.1f%%", min_value=0, max_value=100, help=ui.HELP["playoff"]
        ),
        "Title odds": st.column_config.ProgressColumn(
            "Title odds", format="%.1f%%", min_value=0, max_value=100, help=ui.HELP["title"]
        ),
    },  # fmt: skip
)
st.caption(
    "3,000 simulated seasons from today's rosters, using each team's real remaining schedule."
)

st.subheader("Trades made in this league")
ui.league_trades_panel(ctx)

st.subheader("Add FantasyPros projections (optional)")
st.markdown(
    "FantasyPros has no free data feed, so if you want their consensus in the blend, export a "
    "weekly projections CSV from fantasypros.com and drop it here."
)
up = st.file_uploader("FantasyPros projections CSV", type="csv")
week = st.number_input(
    "Week it projects", min_value=1, max_value=18, value=min(lg.current_week, 18)
)
if up is not None and st.button("Use this file"):
    st.session_state["fp_csv"] = up.getvalue()
    st.session_state["fp_week"] = int(week)
    st.rerun()
if st.session_state.get("fp_csv"):
    if st.button("Stop using the FantasyPros file"):
        st.session_state.pop("fp_csv", None)
        st.session_state.pop("fp_unmatched", None)
        st.rerun()
    unmatched = st.session_state.get("fp_unmatched")
    st.success("FantasyPros projections are part of the blend.")
    if isinstance(unmatched, pl.DataFrame) and unmatched.height:
        st.caption(f"{unmatched.height} rows didn't match a player and were left out:")
        st.dataframe(unmatched, hide_index=True)
