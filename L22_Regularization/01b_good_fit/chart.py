"""Good Fit - Right Complexity"""
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
x_plot = np.linspace(-0.5, 10.5, 200)

ax.scatter(x, y, c=MLBLUE, s=80, edgecolors='black', zorder=5, label='Training data')

slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
intercept = y.mean() - slope * x.mean()
ax.plot(x_plot, intercept + slope * x_plot, color=MLGREEN, linewidth=2.5,
        label=f'y = {intercept:.1f} + {slope:.2f}x')

ax.set_title('GOOD FIT: Right Complexity', fontsize=12, fontweight='bold', color=MLGREEN)
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(-0.5, 10.5)
ax.text(0.05, 0.95, 'Low bias, Low variance\nGeneralizes well',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
