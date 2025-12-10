"""Axis Customization - Limits, ticks, labels"""
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
MLRED = '#D62728'
MLORANGE = '#FF7F0E'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

prices = 100 + np.cumsum(np.random.randn(100) * 2)
ax.plot(prices, color=MLBLUE, linewidth=2)

ax.set_xlim(0, 100)
ax.set_ylim(80, 130)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct', 'Dec'])
ax.set_xlabel('Month', fontsize=10, fontweight='bold')
ax.set_ylabel('Price ($)', fontsize=10, fontweight='bold')
ax.set_title('Axis Limits, Ticks, Labels', fontsize=12, fontweight='bold', color=MLPURPLE)

ax.axhline(100, color=MLRED, linestyle='--', linewidth=1.5, label='Starting price')
ax.axvspan(60, 80, alpha=0.2, color=MLORANGE, label='Highlight region')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
