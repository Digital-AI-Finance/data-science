"""Market Direction Prediction - Finance application"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Market Direction Prediction with Logistic Regression', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Feature engineering for market prediction
ax1 = axes[0, 0]
ax1.axis('off')

features = '''
FEATURES FOR MARKET DIRECTION

Target Variable:
----------------
y = 1 if tomorrow's return > 0 (Up)
y = 0 if tomorrow's return <= 0 (Down)

Momentum Features:
------------------
- returns_lag1    : Yesterday's return
- returns_lag5    : 5-day return
- ma_cross        : Price vs 20-day MA (1 if above)
- rsi             : Relative Strength Index

Volatility Features:
--------------------
- volatility_20d  : 20-day rolling std
- vix_level       : VIX index
- range_ratio     : (High-Low)/Close

Volume Features:
----------------
- volume_ratio    : Volume vs 20-day avg
- volume_trend    : 5-day volume change

Sentiment Features:
-------------------
- news_sentiment  : News sentiment score
- put_call_ratio  : Options put/call ratio

IMPORTANT NOTES:
----------------
- Use ONLY information available BEFORE prediction
- No data leakage! (future information)
- Returns are notoriously hard to predict
- Accuracy > 55% is considered good
'''

ax1.text(0.02, 0.98, features, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Feature Engineering', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Model performance over time
ax2 = axes[0, 1]

# Simulated rolling accuracy
months = 24
dates = pd.date_range('2022-01-01', periods=months, freq='M')
rolling_accuracy = 0.52 + np.random.uniform(-0.05, 0.05, months) + 0.02 * np.sin(np.arange(months)/3)
rolling_accuracy = np.clip(rolling_accuracy, 0.45, 0.60)

ax2.plot(dates, rolling_accuracy, color=MLBLUE, linewidth=2, marker='o', markersize=4)
ax2.axhline(0.5, color=MLRED, linewidth=2, linestyle='--', label='Random Guess (50%)')
ax2.axhline(rolling_accuracy.mean(), color=MLGREEN, linewidth=2, linestyle='-.',
            label=f'Mean Accuracy: {rolling_accuracy.mean():.1%}')

ax2.fill_between(dates, 0.5, rolling_accuracy,
                  where=rolling_accuracy > 0.5, alpha=0.3, color=MLGREEN, label='Above random')
ax2.fill_between(dates, 0.5, rolling_accuracy,
                  where=rolling_accuracy < 0.5, alpha=0.3, color=MLRED, label='Below random')

ax2.set_title('Rolling 3-Month Prediction Accuracy', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Accuracy', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Feature importance
ax3 = axes[1, 0]

features_list = ['MA Cross', 'Momentum 5d', 'Volume Ratio', 'Volatility', 'RSI', 'Sentiment', 'VIX']
importance = [0.32, 0.25, 0.18, -0.15, 0.12, 0.22, -0.28]  # Coefficients

y_pos = np.arange(len(features_list))
colors = [MLGREEN if i > 0 else MLRED for i in importance]

bars = ax3.barh(y_pos, importance, color=colors, edgecolor='black', linewidth=0.5)
ax3.axvline(0, color='black', linewidth=1.5)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(features_list)
ax3.set_title('Feature Coefficients (Impact on P(Up))', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Coefficient', fontsize=10)
ax3.grid(alpha=0.3, axis='x')

# Add interpretation
for bar, coef in zip(bars, importance):
    if coef > 0:
        label = f'+{coef:.2f} (bullish)'
    else:
        label = f'{coef:.2f} (bearish)'
    x_pos = coef + 0.02 if coef > 0 else coef - 0.02
    ha = 'left' if coef > 0 else 'right'
    ax3.text(x_pos, bar.get_y() + bar.get_height()/2, label,
             va='center', ha=ha, fontsize=8)

# Plot 4: Complete pipeline code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Complete Market Direction Pipeline

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

# 1. Load price data
df = pd.read_csv('spy_prices.csv', index_col='date', parse_dates=True)

# 2. Create features (use ONLY past data!)
df['returns'] = df['close'].pct_change()
df['returns_lag1'] = df['returns'].shift(1)
df['returns_lag5'] = df['returns'].rolling(5).sum().shift(1)
df['ma_20'] = df['close'].rolling(20).mean()
df['ma_cross'] = (df['close'].shift(1) > df['ma_20'].shift(1)).astype(int)
df['volatility'] = df['returns'].rolling(20).std().shift(1)
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

# 3. Create target (next day direction)
df['target'] = (df['returns'].shift(-1) > 0).astype(int)

# 4. Drop NaN and prepare data
df = df.dropna()
X = df[['returns_lag1', 'returns_lag5', 'ma_cross', 'volatility', 'volume_ratio']]
y = df['target']

# 5. Time series cross-validation (NO random split!)
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(C=0.1)
    model.fit(X_train_scaled, y_train)
    scores.append(accuracy_score(y_test, model.predict(X_test_scaled)))

print(f"Mean Accuracy: {np.mean(scores):.2%} +/- {np.std(scores):.2%}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Pipeline Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
