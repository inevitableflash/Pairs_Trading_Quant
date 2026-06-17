import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def run_ml_filter():
    df = pd.read_csv('data/signals.csv', index_col='Date', parse_dates=True)

    # Features: spread velocity, volatility clustering, z-score momentum
    df['Spread_ROC_3'] = df['Spread'].diff(3)
    df['Spread_Vol_7'] = df['Spread'].rolling(window=7).std()
    df['Z_Momentum']   = df['Z_Score'].diff(2)

    # Target: will spread be higher 1 day from now?
    df['Target'] = np.where(df['Spread'].shift(-1) > df['Spread'], 1, 0)
    df = df.dropna()

    features = ['Z_Score', 'Spread_ROC_3', 'Spread_Vol_7', 'Z_Momentum']
    X, y = df[features], df['Target']

    # Chronological split — never shuffle time series data
    train_mask = df.index < '2023-01-01'
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        random_state=42, eval_metric='logloss'
    )

    # Cross-validate on training set using time-aware splits
    tscv   = TimeSeriesSplit(n_splits=5)
    cv_acc = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
    print(f"CV Accuracy: {cv_acc.mean()*100:.1f}% ± {cv_acc.std()*100:.1f}%")

    model.fit(X_train, y_train)

    # Use probabilities, not hard labels — only trade when model is confident
    proba = model.predict_proba(X_test)[:, 1]
    hard  = model.predict(X_test)

    print(f"OOS Accuracy (hard threshold 0.5): {accuracy_score(y_test, hard)*100:.2f}%")
    
    # High-confidence signals only
    confident = (proba > 0.60) | (proba < 0.40)
    print(f"High-confidence signals (prob > 0.6 or < 0.4): {confident.sum()} of {len(proba)} test days")

    # Save ML signals for comparison backtest
    ml_signals = pd.Series(0, index=X_test.index)
    ml_signals[proba > 0.60] = 1    # predict spread up → short signal
    ml_signals[proba < 0.40] = -1   # predict spread down → long signal
    ml_signals.to_csv('data/ml_signals.csv', header=True)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 4))
    importance.plot(kind='barh', color='#1A7A6E')
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=150)
    plt.close()
    print("Feature importance chart saved.")

if __name__ == "__main__":
    run_ml_filter()