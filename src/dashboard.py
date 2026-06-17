import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dashboard():
    # Load signals - ensuring index is datetime so the x-axis renders correctly
    df = pd.read_csv('data/signals.csv', index_col='Date', parse_dates=True)
    
    # Lock to out-of-sample period only to visualize actual test executions
    test_df = df.loc['2023-01-01':].copy()
    
    # Isolate timestamps where we crossed the 2-sigma threshold for scatter overlays
    buys = test_df[test_df['Z_Score'] < -2.0]
    sells = test_df[test_df['Z_Score'] > 2.0]
    
    # Link x-axes so zooming on the spread chart zooms the z-score simultaneously
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    # Top panel: Raw spread
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Spread'], name='Spread', line=dict(color='#1f77b4')), row=1, col=1)
    
    # Mark entries. Using up/down triangles to make trade direction instantly obvious
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Spread'], mode='markers', name='Buy Trigger', marker=dict(symbol='triangle-up', color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Spread'], mode='markers', name='Sell Trigger', marker=dict(symbol='triangle-down', color='red', size=10)), row=1, col=1)

    # Bottom panel: Z-score tracking against mean reversion thresholds
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Z_Score'], name='Z-Score', line=dict(color='purple')), row=2, col=1)
    
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="black", row=2, col=1)

    # Clean UI styling
    fig.update_layout(height=750, title_text="StatArb Execution Analysis", template="plotly_white", hovermode="x unified")
    fig.show()

if __name__ == "__main__":
    create_dashboard()