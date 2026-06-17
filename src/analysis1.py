import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

def run_scanner():
    # Parse dates explicitly to ensure time-series alignment
    df = pd.read_csv('data/nse_prices.csv', index_col='Date', parse_dates=True)
    
    # Train/Test split boundary 
    train_df = df.loc['2020-01-01':'2022-12-31']
    
    best_p = 1.0
    best_pair = (None, None)
    cols = train_df.columns
    
    # O(N^2) scan for minimum Engle-Granger p-value
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            _, p_val, _ = coint(train_df[cols[i]], train_df[cols[j]])
            if p_val < best_p:
                best_p = p_val
                best_pair = (cols[i], cols[j])
                
    s1, s2 = best_pair
    print(f"Optimal Pair: {s1} / {s2} (p: {best_p:.5f})")
    
    # Calculate OLS hedge ratio (must include constant for accurate spread)
    X = sm.add_constant(train_df[s2])
    model = sm.OLS(train_df[s1], X).fit()
    hedge = model.params.iloc[1]
    intercept = model.params.iloc[0]
    
    # Verify spread stationarity via ADF
    spread = train_df[s1] - (hedge * train_df[s2]) - intercept
    adf_p = adfuller(spread.dropna())[1]
    
    print(f"Hedge Ratio: {hedge:.4f}")
    print(f"ADF p-value: {adf_p:.5f}")
    
    if adf_p < 0.05:
        print("PASS: Spread is stationary on training data. Good candidate for mean reversion.")
    else:
        print("FAIL: Spread is non-stationary on training data. Pick a different pair.")

if __name__ == "__main__":
    run_scanner()