"""Finance Application - Trees in finance"""
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
fig.suptitle('Decision Trees in Finance Applications', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Trading signal generation tree
ax1 = axes[0, 0]
ax1.axis('off')

trading_tree = '''
TRADING SIGNAL TREE EXAMPLE

|--- RSI <= 30.00 (Oversold)
|   |--- 50_MA_Cross = True (Upward momentum)
|   |   |--- Volume_Ratio > 1.5 (High volume)
|   |   |   |--- SIGNAL: STRONG BUY (Confidence: 78%)
|   |   |--- Volume_Ratio <= 1.5
|   |   |   |--- SIGNAL: BUY (Confidence: 62%)
|   |--- 50_MA_Cross = False
|   |   |--- SIGNAL: HOLD (Confidence: 55%)
|--- RSI > 30.00
|   |--- RSI > 70.00 (Overbought)
|   |   |--- Mom_5d < 0 (Losing momentum)
|   |   |   |--- SIGNAL: SELL (Confidence: 72%)
|   |   |--- Mom_5d >= 0
|   |   |   |--- SIGNAL: HOLD (Confidence: 58%)
|   |--- RSI <= 70.00
|   |   |--- SIGNAL: HOLD (Confidence: 60%)


INTERPRETATION:
---------------
"Buy oversold stocks with upward momentum
 and high volume conviction"

This is essentially a quantitative trading rule
extracted from historical data!
'''

ax1.text(0.02, 0.98, trading_tree, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Trading Signal Decision Tree', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Backtest performance
ax2 = axes[0, 1]

# Simulated cumulative returns
days = 252
dates = pd.date_range('2023-01-01', periods=days, freq='B')

np.random.seed(42)
market_returns = np.random.normal(0.0004, 0.012, days)
tree_returns = market_returns + np.random.normal(0.0002, 0.005, days)

cumulative_market = np.cumprod(1 + market_returns) - 1
cumulative_tree = np.cumprod(1 + tree_returns) - 1

ax2.plot(dates, cumulative_market * 100, color=MLBLUE, linewidth=2, label='Buy & Hold (S&P 500)')
ax2.plot(dates, cumulative_tree * 100, color=MLGREEN, linewidth=2, label='Random Forest Strategy')

ax2.fill_between(dates, cumulative_market * 100, cumulative_tree * 100,
                  where=cumulative_tree > cumulative_market, alpha=0.3, color=MLGREEN)
ax2.fill_between(dates, cumulative_market * 100, cumulative_tree * 100,
                  where=cumulative_tree < cumulative_market, alpha=0.3, color=MLRED)

ax2.axhline(0, color='gray', linewidth=1, linestyle='--')

ax2.set_title('Strategy Backtest (1 Year)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Cumulative Return (%)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Add performance stats
ax2.text(0.02, 0.98, f'Strategy: {cumulative_tree[-1]*100:.1f}%\nBenchmark: {cumulative_market[-1]*100:.1f}%',
         transform=ax2.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 3: Feature importance for trading
ax3 = axes[1, 0]

features = ['RSI', 'MACD', '50-Day MA Cross', 'Volume Ratio', 'Volatility 20d',
            'Momentum 5d', 'Bollinger Band', 'Earnings Surprise']
importance = [0.18, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.07]

y_pos = np.arange(len(features))
colors = plt.cm.Greens(np.linspace(0.9, 0.3, len(features)))

bars = ax3.barh(y_pos, importance, color=colors, edgecolor='black', linewidth=0.5)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(features)
ax3.set_title('Feature Importance (Random Forest Trading Model)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Importance', fontsize=10)
ax3.invert_yaxis()
ax3.grid(alpha=0.3, axis='x')

# Add values
for bar, imp in zip(bars, importance):
    ax3.text(imp + 0.005, bar.get_y() + bar.get_height()/2, f'{imp:.0%}',
             va='center', fontsize=9)

# Plot 4: Complete pipeline code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
COMPLETE TRADING MODEL PIPELINE

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# 1. Feature Engineering
def create_features(df):
    df['returns'] = df['close'].pct_change()
    df['RSI'] = compute_rsi(df['close'], 14)
    df['ma_cross'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volatility'] = df['returns'].rolling(20).std()
    df['mom_5d'] = df['close'].pct_change(5)
    return df

# 2. Target: Next-day direction
df['target'] = (df['returns'].shift(-1) > 0).astype(int)

# 3. Time Series CV (prevent look-ahead bias!)
tscv = TimeSeriesSplit(n_splits=5)

# 4. Train Model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,       # Prevent overfitting
    min_samples_leaf=20,
    random_state=42
)

# 5. Walk-Forward Backtest
for train_idx, test_idx in tscv.split(X):
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = rf.predict(X.iloc[test_idx])
    # Calculate trading returns...

# 6. Generate Live Signals
latest_features = create_features(latest_data)
signal = rf.predict(latest_features.iloc[-1:])
prob = rf.predict_proba(latest_features.iloc[-1:])[0, 1]
print(f"Signal: {'BUY' if signal[0]==1 else 'SELL'} (Prob: {prob:.0%})")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Pipeline', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
