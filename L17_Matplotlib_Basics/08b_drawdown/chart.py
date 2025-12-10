"""Drawdown Chart - Visualizing losses from peak"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.015))
cummax = np.maximum.accumulate(prices)
drawdown = (prices - cummax) / cummax * 100

ax.fill_between(range(len(drawdown)), 0, drawdown, color=MLRED, alpha=0.5)
ax.plot(drawdown, color=MLRED, linewidth=1.5)
ax.axhline(drawdown.min(), color='black', linestyle='--', linewidth=1,
           label=f'Max DD: {drawdown.min():.1f}%')

ax.set_title('Drawdown Chart', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Day', fontsize=10)
ax.set_ylabel('Drawdown (%)', fontsize=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
