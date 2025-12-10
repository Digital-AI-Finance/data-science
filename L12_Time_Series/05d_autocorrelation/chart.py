"""Autocorrelation - Lagged correlation analysis"""
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
MLBLUE = '#0066CC'
MLGREEN = '#2CA02C'

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 100 + np.cumsum(np.random.randn(100) * 2)
df = pd.DataFrame({'Price': prices}, index=dates)

fig, ax = plt.subplots(figsize=(10, 6))

lags = range(1, 11)
correlations = [df['Price'].corr(df['Price'].shift(lag)) for lag in lags]

bars = ax.bar(lags, correlations, color=MLBLUE, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linewidth=1)

for i, (lag, corr) in enumerate(zip(lags, correlations)):
    if abs(corr) > 0.5:
        bars[i].set_color(MLGREEN)
    ax.text(lag, corr + 0.02, f'{corr:.2f}', ha='center', fontsize=9)

ax.set_xlabel('Lag (days)', fontsize=10)
ax.set_ylabel('Autocorrelation', fontsize=10)
ax.set_title('Autocorrelation: Price.corr(Price.shift(lag))', fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.set_xticks(lags)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
