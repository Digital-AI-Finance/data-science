"""Returns via Shift - Computing daily returns"""
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
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30, freq='D')
prices = 100 + np.cumsum(np.random.randn(30) * 2)
df = pd.DataFrame({'Price': prices}, index=dates)
df['Return'] = (df['Price'] - df['Price'].shift(1)) / df['Price'].shift(1) * 100

fig, ax = plt.subplots(figsize=(10, 6))

colors = [MLGREEN if r >= 0 else MLRED for r in df['Return'].dropna()]
ax.bar(df.index[1:], df['Return'].dropna(), color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linewidth=1)
ax.axhline(df['Return'].mean(), color=MLPURPLE, linestyle='--', linewidth=2,
           label=f"Mean: {df['Return'].mean():.2f}%")

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Daily Return (%)', fontsize=10)
ax.set_title("Returns: (Price - Price.shift(1)) / Price.shift(1)", fontsize=11,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
