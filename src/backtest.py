import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_backtest():
    df = pd.read_csv('data/signals.csv', index_col='Date', parse_dates=True)
    test_df = df.loc['2023-01-01':].copy()

    position   = 0
    entry_pnl  = 0.0   # cumulative PnL at entry, to measure per-trade result
    capital    = 100_000
    tc         = 0.001

    daily_pnl  = []
    trade_log  = []   # list of (entry_date, exit_date, trade_pnl)
    entry_date = None
    entry_cum  = 0.0
    cum_so_far = 0.0

    dates = list(test_df.index)

    for idx, date in enumerate(dates):
        z      = test_df.at[date, 'Z_Score']
        spread = test_df.at[date, 'Spread']
        dpnl   = 0.0

        if pd.isna(z):
            daily_pnl.append(0.0)
            continue

        # Mark-to-market: compare to previous trading day (not calendar day)
        if idx > 0:
            prev_spread = test_df.at[dates[idx - 1], 'Spread']
        else:
            prev_spread = spread

        if position == 0:
            if z < -2.0:
                position   = 1
                entry_date = date
                entry_cum  = cum_so_far
                dpnl      -= capital * tc
            elif z > 2.0:
                position   = -1
                entry_date = date
                entry_cum  = cum_so_far
                dpnl      -= capital * tc

        elif position == 1:
            dpnl = (spread - prev_spread) * (capital / max(abs(prev_spread), 1e-9))
            if z >= 0 or z < -3.0:
                dpnl      -= capital * tc
                trade_log.append((entry_date, date, cum_so_far + dpnl - entry_cum))
                position   = 0

        elif position == -1:
            dpnl = (prev_spread - spread) * (capital / max(abs(prev_spread), 1e-9))
            if z <= 0 or z > 3.0:
                dpnl      -= capital * tc
                trade_log.append((entry_date, date, cum_so_far + dpnl - entry_cum))
                position   = 0

        daily_pnl.append(dpnl)
        cum_so_far += dpnl

    test_df['Daily_PnL']      = daily_pnl
    test_df['Cumulative_PnL'] = test_df['Daily_PnL'].cumsum()

    returns  = test_df['Daily_PnL'] / capital
    sharpe   = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    drawdown = (test_df['Cumulative_PnL'] - test_df['Cumulative_PnL'].cummax()).min()
    total    = test_df['Cumulative_PnL'].iloc[-1]

    n_trades  = len(trade_log)
    wins      = sum(1 for _, _, p in trade_log if p > 0)
    win_rate  = (wins / n_trades * 100) if n_trades > 0 else 0
    durations = [(ex - en).days for en, ex, _ in trade_log]
    avg_dur   = np.mean(durations) if durations else 0

    print(f"Net P&L:    Rs {total:,.2f}")
    print(f"Sharpe:     {sharpe:.2f}")
    print(f"Max DD:     Rs {drawdown:,.2f}")
    print(f"Trades:     {n_trades}  |  Win Rate: {win_rate:.1f}%  |  Avg Duration: {avg_dur:.1f} days")

    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, test_df['Cumulative_PnL'], color='#2E6FAD', linewidth=1.5)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.title('Out-of-Sample Equity Curve (Net of TC)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative P&L (Rs)')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('results/equity_curve.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    run_backtest()