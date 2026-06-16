import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import os

def scan_universe(train_data):
    cols = train_data.columns
    n = len(cols)
    pairs = []
    
    # O(N^2) scan for stationary spreads via Engle-Granger
    for i in range(n):
        for j in range(i+1, n):
            _, p_val, _ = ts.coint(train_data[cols[i]], train_data[cols[j]])
            
            # Filter strictly for stationary bounds
            if p_val < 0.05:
                pairs.append((cols[i], cols[j], p_val))
                
    # Rank by strongest mean-reversion probability
    return sorted(pairs, key=lambda x: x[2])

if __name__ == "__main__":
    # Ensure datetime index for accurate time-series slicing
    df = pd.read_csv('data/nse_prices.csv', index_col='Date', parse_dates=True)
    os.makedirs('results', exist_ok=True)
    
    # Restrict scan to training data to completely eliminate look-ahead bias
    train_df = df.loc['2020-01-01':'2022-12-31']
    
    print("Generating train-set correlation matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(train_df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
    plt.savefig('results/correlation_heatmap.png')
    plt.close()
    
    print("Executing cointegration scan...")
    viable_pairs = scan_universe(train_df)
    
    print("\n--- STATISTICALLY STATIONARY PAIRS (p < 0.05) ---")
    if not viable_pairs:
        print("No viable mean-reverting pairs found in training data.")
    else:
        for s1, s2, p in viable_pairs[:5]:
            print(f"{s1} / {s2} | p-val: {p:.5f}")