"""Underfitting - Model Too Simple"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))


# Generate simple data with noise
n = 15
x = np.linspace(0, 10, n)
y_true = 2 + 0.5 * x
y = y_true + np.random.normal(0, 1.2, n)

ax.scatter(x, y, c=MLBLUE, s=80, edgecolors='black', zorder=5, label='Training data')
ax.axhline(y.mean(), color=MLRED, linewidth=2.5, label=f'y = {y.mean():.1f} (constant)')

ax.set_title('UNDERFITTING: Model Too Simple', fontsize=12, fontweight='bold', color=MLRED)
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, 'High bias\nHigh error on both\ntrain and test',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
