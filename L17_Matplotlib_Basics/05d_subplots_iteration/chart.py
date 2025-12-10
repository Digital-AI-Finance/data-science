"""plt.subplots Iteration - Accessing axes by index"""
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

data = np.random.randn(100)
ax.hist(data, bins=20, color=MLBLUE, alpha=0.7, edgecolor='black')
ax.set_title('plt.subplots(rows, cols): ax[row, col] indexing', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Value', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.grid(alpha=0.3)

# Add annotation explaining the pattern
ax.text(0.95, 0.95, 'Access: axes[0,0], axes[0,1], ...\nor axes.flat for iteration',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
