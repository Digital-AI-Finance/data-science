"""Shift Demo - Lag and lead operations"""
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
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30, freq='D')
prices = 100 + np.cumsum(np.random.randn(30) * 2)
df = pd.DataFrame({'Price': prices}, index=dates)
df['Lag_1'] = df['Price'].shift(1)
df['Lead_1'] = df['Price'].shift(-1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df.index, df['Price'], 'o-', color=MLBLUE, linewidth=2, markersize=6, label='Original')
ax.plot(df.index, df['Lag_1'], 's--', color=MLORANGE, linewidth=2, markersize=5, label='shift(1) - Lag')
ax.plot(df.index, df['Lead_1'], '^--', color=MLGREEN, linewidth=2, markersize=5, label='shift(-1) - Lead')

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title("df['Price'].shift(n) - Move data forward/backward", fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
