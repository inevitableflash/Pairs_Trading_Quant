import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import os

def generate_signals():
    df = pd.read_csv('data/nse_prices.csv', index_col='Date', parse_dates=True)
    
    train_df = df.loc['2020-01-01':'2022-12-31']
    
    best_p = 1.0
    best_pair = (None, None)
    cols = train_df.columns
    n = len(cols)
    
    for i in range(n):
        for j in range(i+1, n):
            _, p_val, _ = coint(train_df[cols[i]], train_df[cols[j]])
            if p_val < best_p:
                best_p = p_val
                best_pair = (cols[i], cols[j])

    s1, s2 = best_pair
    print(f"Best pair: {s1} / {s2} | train p-val: {best_p:.5f}")
    
    # OLS on training window only — hedge ratio and intercept are frozen here
    X_train = sm.add_constant(train_df[s2])
    ols = sm.OLS(train_df[s1], X_train).fit()
    hedge_ratio = ols.params.iloc[1]
    intercept   = ols.params.iloc[0]

    # Spread computed on full dataset using locked parameters
    spread = df[s1] - (hedge_ratio * df[s2]) - intercept

    # Rolling stats computed on spread itself (not split) — this is fine
    # because the 30-day window is a signal parameter, not a model parameter.
    # Hedge ratio and intercept are the only things that must be train-only.
    roll_mean = spread.rolling(window=30).mean()
    roll_std  = spread.rolling(window=30).std()
    z_score   = (spread - roll_mean) / roll_std

    out = pd.DataFrame({'Spread': spread, 'Z_Score': z_score}, index=df.index)
    os.makedirs('data', exist_ok=True)
    out.to_csv('data/signals.csv')
    print("Signals written to data/signals.csv")

if __name__ == "__main__":
    generate_signals()