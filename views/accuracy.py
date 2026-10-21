import streamlit as st

from ffcalc import ui

text = ui.report_text()
if text:
    st.markdown(text)
else:
    st.title("How accurate is it?")
    st.info(
        "The accuracy report hasn't been generated yet. Run `uv run python -m ffcalc.backtest`."
    )
