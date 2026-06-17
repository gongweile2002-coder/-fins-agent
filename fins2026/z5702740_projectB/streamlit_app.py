# ruff: noqa
"""FinTech Project - your investment app (starter).

This thin starter proves the app deploys and loads the hosted data. Build your real
dashboard on top of it: a fund picker, each fund's fact sheet (growth of $1,
drawdown, Sharpe, holdings), an allocation control, and your sentiment analytics.

Run locally:   streamlit run streamlit_app.py
Deploy:        push this folder to a public GitHub repo, then connect it on
               share.streamlit.io with entrypoint streamlit_app.py (see brief App. D).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st
from src import data_access

st.set_page_config(page_title="Systematic Funds", layout="wide")
st.title("Systematic Multi-Asset Funds")
st.caption("Starter app - replace this with your fund dashboard.")


@st.cache_data(ttl=86_400, show_spinner="Loading data...")
def _equities():
    return data_access.load_equity_prices()


tab_funds, tab_sentiment, tab_data = st.tabs(["Funds", "Sentiment", "Data"])

with tab_funds:
    st.subheader("Funds")
    st.info("TODO: build your fund picker, fact sheets, and allocation control here.")

with tab_sentiment:
    st.subheader("Sentiment")
    st.info("TODO: show your sector sentiment indices over time.")

with tab_data:
    eq = _equities()
    st.write(f"Equity prices: {eq.shape[0]:,} rows, {eq['ticker'].nunique()} tickers, "
             f"{eq['date'].min().date()} to {eq['date'].max().date()}")
    st.dataframe(eq.head(20), width="stretch")
