"""Cumulative Histogram - With frequency and CDF"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLORANGE = '#FF7F0E'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

returns = np.random.normal(0.05, 2, 500)

ax.hist(returns, bins=40, color=MLORANGE, alpha=0.7, edgecolor='black', label='Frequency')

ax_twin = ax.twinx()
ax_twin.hist(returns, bins=40, cumulative=True, density=True, histtype='step',
             color=MLPURPLE, linewidth=2.5, label='Cumulative')
ax_twin.set_ylabel('Cumulative Probability', fontsize=10, color=MLPURPLE)

ax.set_title('Histogram with Cumulative', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Return (%)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax_twin.legend(loc='center right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
