"""Comparing Distributions - Multiple overlapping histograms"""
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

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

low_vol = np.random.normal(0, 1, 500)
high_vol = np.random.normal(0, 2.5, 500)

ax.hist(low_vol, bins=40, alpha=0.5, color=MLBLUE, edgecolor='black', label=f'Low Vol (std={np.std(low_vol):.2f})')
ax.hist(high_vol, bins=40, alpha=0.5, color=MLRED, edgecolor='black', label=f'High Vol (std={np.std(high_vol):.2f})')

ax.set_title('Comparing Distributions', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Return (%)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
