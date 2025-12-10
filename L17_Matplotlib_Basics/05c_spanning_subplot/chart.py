"""Spanning Subplot - Full width time series"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLBLUE = '#0066CC'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

days = 252
prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.015))
ax.plot(prices, color=MLBLUE, linewidth=2)
ax.fill_between(range(len(prices)), prices.min(), prices, alpha=0.2, color=MLBLUE)
ax.set_title('fig.add_subplot(2, 1, 2) - Spans full width', fontsize=12,
             fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Trading Day', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
