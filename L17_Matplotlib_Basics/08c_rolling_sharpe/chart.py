"""Rolling Sharpe Ratio - Performance over time"""
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

fig, ax = plt.subplots(figsize=(10, 6))

returns = np.random.normal(0.0004, 0.015, 504)

window = 60
rolling_mean = pd.Series(returns).rolling(window).mean()
rolling_std = pd.Series(returns).rolling(window).std()
rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)

ax.plot(rolling_sharpe, color=MLBLUE, linewidth=2)
ax.fill_between(range(len(rolling_sharpe)), 0, rolling_sharpe,
                where=rolling_sharpe > 0, color=MLGREEN, alpha=0.3)
ax.fill_between(range(len(rolling_sharpe)), 0, rolling_sharpe,
                where=rolling_sharpe < 0, color=MLRED, alpha=0.3)
ax.axhline(0, color='black', linewidth=1)
ax.axhline(1, color=MLGREEN, linestyle='--', linewidth=1.5, alpha=0.7, label='Good (>1)')
ax.axhline(-1, color=MLRED, linestyle='--', linewidth=1.5, alpha=0.7, label='Poor (<-1)')

ax.set_title('Rolling 60-day Sharpe Ratio', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Day', fontsize=10)
ax.set_ylabel('Sharpe Ratio', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
