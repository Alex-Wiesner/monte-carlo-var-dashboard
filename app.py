import pandas as pd
import time
import streamlit as st
import numpy as np
from src.data_fetch import cached_fetch_price_data
from src.plot import plot_data
from src.engine import (
    simulate_portfolio,
    var_es,
)

st.set_page_config(page_title="Monte Carlo VaR Dashboard", layout="wide")

st.title("Monte Carlo VaR Dashboard")
st.caption("Fast, interactive risk analytics for a custom stock basket")

st.sidebar.header("Portfolio configuration")

default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
tickers_input = st.sidebar.text_input(
    "Comma‑separated tickers", value=", ".join(default_tickers)
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

weights_input = st.sidebar.text_input(
    "Comma‑separated weights (must sum to 1)",
    value=", ".join([f"{1 / len(tickers)}"] * len(tickers)),
)
weights = np.array([float(w) for w in weights_input.split(",")])

if not np.isclose(weights.sum(), 1.0):
    st.sidebar.error("Weights must sum to 1.0")
    st.stop()

horizon = st.sidebar.slider("Simulation horizon (days)", 1, 180, 30)
n_sims = st.sidebar.selectbox(
    "Number of Monte Carlo paths", [5000, 10000, 20000], index=1
)

st.sidebar.subheader("Stress scenario")
stress_shift = (
    st.sidebar.slider("Mean‑return shift (bps)", -200, 200, 0, step=10) / 10000
)


prices = cached_fetch_price_data(tickers, start="2020-01-01")
log_ret = np.log(prices / prices.shift(1)).dropna()
mu = log_ret.mean().values + stress_shift
cov = log_ret.cov().values

st.success(f"Loaded {len(tickers)} tickers, {len(prices)} days of history.")

with st.spinner("Running Monte Carlo simulation…"):
    start = time.process_time()
    pnl = simulate_portfolio(
        weights, mu, cov, horizon_days=horizon, n_sims=n_sims)
    elapsed = time.process_time() - start

st.write(f"Simulation ran in {elapsed * 1000:,.2f} ms")

var_95, es_95 = var_es(pnl, conf=0.95)
var_99, es_99 = var_es(pnl, conf=0.99)

col1, col2 = st.columns(2)
col1.metric("VaR 95 %", f"{var_95 * 100:,.2f}%")
col1.metric("ES 95 %", f"{es_95 * 100:,.2f}%")
col2.metric("VaR 99 %", f"{var_99 * 100:,.2f}%")
col2.metric("ES 99 %", f"{es_99 * 100:,.2f}%")

st.subheader("Distribution of simulated portfolio P&L")
counts, edges = np.histogram(pnl, bins=20)
centres = (edges[:-1] + edges[1:]) * 50
st.bar_chart(pd.DataFrame({"Frequency": counts}, index=np.round(centres, 2)))

if st.checkbox("Show sample price trajectories"):
    st.altair_chart(plot_data(weights, mu, cov, horizon),
                    use_container_width=True)
