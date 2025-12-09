"""Time Series Patterns - Common patterns in financial data"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)
n = 252  # One year of trading days

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Common Time Series Patterns in Finance', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Mean Reversion
ax1 = axes[0, 0]
mean_level = 100
theta = 0.1  # Speed of reversion
sigma = 2
prices_mr = [mean_level]
for _ in range(n-1):
    dp = theta * (mean_level - prices_mr[-1]) + sigma * np.random.randn()
    prices_mr.append(prices_mr[-1] + dp)

ax1.plot(prices_mr, color=MLBLUE, linewidth=1.5)
ax1.axhline(mean_level, color=MLRED, linestyle='--', linewidth=2, label=f'Mean = {mean_level}')
ax1.fill_between(range(n), mean_level - 5, mean_level + 5, alpha=0.2, color=MLRED)

ax1.set_xlabel('Trading Day', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title('Mean Reversion: Price returns to average', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Momentum / Trend Following
ax2 = axes[0, 1]
drift = 0.0008
vol = 0.015
prices_trend = 100 * np.exp(np.cumsum(drift + vol * np.random.randn(n)))

# Add momentum indicator
ma_fast = pd.Series(prices_trend).rolling(10).mean()
ma_slow = pd.Series(prices_trend).rolling(50).mean()

ax2.plot(prices_trend, color='gray', linewidth=1, alpha=0.5, label='Price')
ax2.plot(ma_fast, color=MLGREEN, linewidth=2, label='MA(10) Fast')
ax2.plot(ma_slow, color=MLORANGE, linewidth=2, label='MA(50) Slow')

# Mark crossovers
for i in range(51, n):
    if ma_fast.iloc[i-1] < ma_slow.iloc[i-1] and ma_fast.iloc[i] > ma_slow.iloc[i]:
        ax2.axvline(i, color=MLGREEN, alpha=0.3, linewidth=2)
        ax2.scatter([i], [prices_trend[i]], color=MLGREEN, s=100, zorder=5, marker='^')
    elif ma_fast.iloc[i-1] > ma_slow.iloc[i-1] and ma_fast.iloc[i] < ma_slow.iloc[i]:
        ax2.axvline(i, color=MLRED, alpha=0.3, linewidth=2)
        ax2.scatter([i], [prices_trend[i]], color=MLRED, s=100, zorder=5, marker='v')

ax2.set_xlabel('Trading Day', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.set_title('Momentum: Trend following with MA crossover', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Volatility Clustering
ax3 = axes[1, 0]
# Generate GARCH-like returns
returns = np.zeros(n)
vol_series = np.zeros(n)
vol_series[0] = 0.01
alpha, beta = 0.1, 0.85

for i in range(1, n):
    vol_series[i] = np.sqrt(0.00001 + alpha * returns[i-1]**2 + beta * vol_series[i-1]**2)
    returns[i] = vol_series[i] * np.random.randn()

colors = [MLGREEN if r >= 0 else MLRED for r in returns]
ax3.bar(range(n), returns * 100, color=colors, alpha=0.7, width=1)
ax3.plot(vol_series * 100 * 2, color=MLPURPLE, linewidth=2, label='Volatility envelope')
ax3.plot(-vol_series * 100 * 2, color=MLPURPLE, linewidth=2)

ax3.set_xlabel('Trading Day', fontsize=10)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.set_title('Volatility Clustering: High vol follows high vol', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Regime Changes
ax4 = axes[1, 1]
# Create regime-switching data
regime1 = np.random.randn(80) * 1 + 0.05  # Low vol, positive drift
regime2 = np.random.randn(60) * 3 - 0.1   # High vol, negative drift
regime3 = np.random.randn(112) * 1.5 + 0.02  # Medium vol, slight positive

returns_regime = np.concatenate([regime1, regime2, regime3])
prices_regime = 100 * np.exp(np.cumsum(returns_regime / 100))

ax4.plot(prices_regime, color=MLBLUE, linewidth=1.5)

# Shade regimes
ax4.axvspan(0, 80, alpha=0.2, color=MLGREEN, label='Bull (low vol)')
ax4.axvspan(80, 140, alpha=0.2, color=MLRED, label='Bear (high vol)')
ax4.axvspan(140, 252, alpha=0.2, color=MLORANGE, label='Recovery')

# Add regime lines
ax4.axvline(80, color=MLRED, linewidth=2, linestyle='--')
ax4.axvline(140, color=MLGREEN, linewidth=2, linestyle='--')

ax4.set_xlabel('Trading Day', fontsize=10)
ax4.set_ylabel('Price ($)', fontsize=10)
ax4.set_title('Regime Changes: Different market conditions', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
