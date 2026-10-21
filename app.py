"""Streamlit entry point: `uv run streamlit run app.py`."""

import logging

import streamlit as st

import ffcalc  # noqa: F401  (TLS setup)
from ffcalc import ui

log = logging.getLogger("ffcalc.app")

st.set_page_config(page_title="Trade Calculator", page_icon="🏈", layout="wide")
ui.style()
page = st.navigation(
    [
        st.Page(
            "views/trade.py", title="Grade a trade", icon=":material/swap_horiz:", default=True
        ),  # fmt: skip
        st.Page("views/league.py", title="Your league", icon=":material/leaderboard:"),
        st.Page("views/finder.py", title="Find trades", icon=":material/search:"),
        st.Page("views/accuracy.py", title="How accurate is it?", icon=":material/fact_check:"),
    ]
)
# Streamlit's own stop/rerun signals are BaseExceptions, so they pass straight through.
try:
    st.session_state["ctx"] = ui.sidebar()
    page.run()
except Exception:
    log.exception("page failed")
    st.error(
        "Something went wrong while building this page. Reload it to try again. If it keeps "
        "happening, one of the data sources may have changed; the details are in the terminal "
        "window running the app."
    )
