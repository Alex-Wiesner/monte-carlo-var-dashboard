import yfinance as yf
import pandas as pd


def fetch_price_data(tickers: list[str], start: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    data = data.ffill().dropna()
    return data["Close"]
