import yfinance as yf
import pandas as pd
import os

tickers = [
    'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
    'HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS'
]

df = yf.download(tickers, start='2020-01-01', end='2024-01-01', auto_adjust=True)['Close']

# Remove Jan 1 — NSE is always closed on New Year
df = df[~((df.index.month == 1) & (df.index.day == 1))]

# Drop stocks with too many missing values, forward fill the rest
df = df.dropna(thresh=int(0.95 * len(df)), axis=1)
df = df.ffill()

os.makedirs('data', exist_ok=True)
df.to_csv('data/nse_prices.csv')
print(f"Saved: {df.shape[0]} rows, {df.shape[1]} stocks")
print(f"Range: {df.index[0].date()} to {df.index[-1].date()}")