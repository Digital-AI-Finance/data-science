"""Bollinger Bands - MA with standard deviation bands"""
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
MLRED = '#D62728'

np.random.seed(42)
n = 252
dates = pd.date_range('2024-01-01', periods=n, freq='B')
prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015))
df = pd.DataFrame({'Price': prices}, index=dates)

fig, ax = plt.subplots(figsize=(10, 6))

ma20 = df['Price'].rolling(20).mean()
std20 = df['Price'].rolling(20).std()
upper = ma20 + 2 * std20
lower = ma20 - 2 * std20

ax.plot(df.index, df['Price'], color='gray', linewidth=1, label='Price')
ax.plot(df.index, ma20, color=MLPURPLE, linewidth=2, label='MA(20)')
ax.fill_between(df.index, lower, upper, color=MLBLUE, alpha=0.2, label='2-Std Band')
ax.plot(df.index, upper, color=MLBLUE, linewidth=1, linestyle='--')
ax.plot(df.index, lower, color=MLBLUE, linewidth=1, linestyle='--')

above = df['Price'] > upper
below = df['Price'] < lower
ax.scatter(df.index[above], df['Price'][above], color=MLRED, s=30, zorder=5, label='Above band')
ax.scatter(df.index[below], df['Price'][below], color=MLGREEN, s=30, zorder=5, label='Below band')

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title('Bollinger Bands: MA +/- 2*rolling.std()', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
