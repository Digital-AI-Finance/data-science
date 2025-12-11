"""Gini Impurity"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))


p = np.linspace(0, 1, 100)
gini = 2 * p * (1 - p)

ax.plot(p, gini, color=MLBLUE, linewidth=3, label='Gini = 2p(1-p)')
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5)
ax.scatter([0.5], [0.5], c=MLRED, s=100, zorder=5, label='Max impurity at p=0.5')
ax.scatter([0, 1], [0, 0], c=MLGREEN, s=100, zorder=5, label='Pure nodes at p=0 or 1')

ax.set_title('Gini Impurity: Measure of Node Purity', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Proportion of Class 1 (p)', fontsize=10)
ax.set_ylabel('Gini Impurity', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.02, 0.55)


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
