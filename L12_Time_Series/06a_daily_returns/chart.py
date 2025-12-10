"""Daily Returns - pct_change bar chart"""
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
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.015))
df = pd.DataFrame({'Price': prices}, index=dates)
df['Daily_Pct'] = df['Price'].pct_change() * 100

fig, ax = plt.subplots(figsize=(10, 6))

returns = df['Daily_Pct'].dropna()
colors = [MLGREEN if r >= 0 else MLRED for r in returns]
ax.bar(returns.index, returns, color=colors, alpha=0.7, width=0.8)
ax.axhline(0, color='black', linewidth=1)
ax.axhline(returns.mean(), color=MLPURPLE, linestyle='--', linewidth=2,
           label=f'Mean: {returns.mean():.2f}%')

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Daily Return (%)', fontsize=10)
ax.set_title("df['Price'].pct_change() * 100", fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
