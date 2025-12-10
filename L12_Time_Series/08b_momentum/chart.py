"""Momentum - Trend following with MA crossover"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)
n = 252
drift = 0.0008
vol = 0.015
prices_trend = 100 * np.exp(np.cumsum(drift + vol * np.random.randn(n)))

ma_fast = pd.Series(prices_trend).rolling(10).mean()
ma_slow = pd.Series(prices_trend).rolling(50).mean()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(prices_trend, color='gray', linewidth=1, alpha=0.5, label='Price')
ax.plot(ma_fast, color=MLGREEN, linewidth=2, label='MA(10) Fast')
ax.plot(ma_slow, color=MLORANGE, linewidth=2, label='MA(50) Slow')

for i in range(51, n):
    if ma_fast.iloc[i-1] < ma_slow.iloc[i-1] and ma_fast.iloc[i] > ma_slow.iloc[i]:
        ax.scatter([i], [prices_trend[i]], color=MLGREEN, s=100, zorder=5, marker='^')
    elif ma_fast.iloc[i-1] > ma_slow.iloc[i-1] and ma_fast.iloc[i] < ma_slow.iloc[i]:
        ax.scatter([i], [prices_trend[i]], color=MLRED, s=100, zorder=5, marker='v')

ax.set_xlabel('Trading Day', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title('Momentum: Trend following with MA crossover', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
