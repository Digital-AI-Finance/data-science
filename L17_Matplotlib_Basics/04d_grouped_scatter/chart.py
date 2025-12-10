"""Grouped Scatter - Asset class comparison"""
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
MLGREEN = '#2CA02C'
MLORANGE = '#FF7F0E'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

# Three asset classes
for name, color, mean_ret, mean_risk in [('Stocks', MLBLUE, 8, 18),
                                          ('Bonds', MLGREEN, 3, 6),
                                          ('Commodities', MLORANGE, 5, 22)]:
    risk = np.random.normal(mean_risk, 3, 20)
    returns = np.random.normal(mean_ret, 2, 20)
    ax.scatter(risk, returns, c=color, s=60, alpha=0.7, edgecolors='black', label=name)

ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_title('Asset Class Comparison', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Risk (%)', fontsize=10)
ax.set_ylabel('Return (%)', fontsize=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
