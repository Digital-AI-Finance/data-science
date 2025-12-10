"""Rolling Volatility - Annualized standard deviation"""
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
MLRED = '#D62728'

np.random.seed(42)
n = 252
dates = pd.date_range('2024-01-01', periods=n, freq='B')
prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015))
df = pd.DataFrame({'Price': prices}, index=dates)
returns = df['Price'].pct_change()

fig, ax = plt.subplots(figsize=(10, 6))

rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100

ax.fill_between(df.index, 0, rolling_vol, color=MLRED, alpha=0.3)
ax.plot(df.index, rolling_vol, color=MLRED, linewidth=2)
ax.axhline(rolling_vol.mean(), color=MLPURPLE, linestyle='--', linewidth=2,
           label=f'Mean Vol: {rolling_vol.mean():.1f}%')

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Annualized Volatility (%)', fontsize=10)
ax.set_title("Rolling Vol: returns.rolling(20).std() * sqrt(252)", fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
