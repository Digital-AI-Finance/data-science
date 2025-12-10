"""Rolling Channel - Min/Max price bands"""
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

rolling_max = df['Price'].rolling(20).max()
rolling_min = df['Price'].rolling(20).min()

ax.plot(df.index, df['Price'], color='gray', linewidth=1, label='Price')
ax.plot(df.index, rolling_max, color=MLGREEN, linewidth=1.5, label='20-day High')
ax.plot(df.index, rolling_min, color=MLRED, linewidth=1.5, label='20-day Low')
ax.fill_between(df.index, rolling_min, rolling_max, color=MLBLUE, alpha=0.1)

new_high = df['Price'] == rolling_max
ax.scatter(df.index[new_high], df['Price'][new_high], color=MLGREEN, s=20, marker='^', zorder=5)

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title('Rolling Channel: rolling(20).max/min()', fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
