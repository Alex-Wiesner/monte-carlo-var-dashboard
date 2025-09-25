import yfinance as yf
import pandas as pd
import streamlit as st


def _fetch_price_data(tickers: list[str], start: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    data = data.ffill().dropna()
    return data["Close"]


@st.cache_data()
def cached_fetch_price_data(tickers: list[str], start: str) -> pd.DataFrame:
    return _fetch_price_data(tickers, start)
