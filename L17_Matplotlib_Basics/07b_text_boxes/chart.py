"""Text Boxes - bbox styles for annotations"""
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
MLRED = '#D62728'
MLORANGE = '#FF7F0E'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

prices = 100 + np.cumsum(np.random.randn(100) * 2)
ax.plot(prices, color=MLBLUE, linewidth=2)

# Different bbox styles
ax.text(10, prices.max() - 5, 'Bull Market', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=MLGREEN, alpha=0.7))

ax.text(60, prices.min() + 5, 'Correction', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='square', facecolor=MLRED, alpha=0.7, edgecolor='black'))

ax.text(80, np.mean(prices), 'Recovery', fontsize=10,
        bbox=dict(boxstyle='rarrow', facecolor=MLORANGE, alpha=0.7))

ax.set_title('Text Boxes (bbox)', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
