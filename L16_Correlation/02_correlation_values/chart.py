"""Correlation Values - Understanding the range"""
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

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Correlation Coefficient Values: -1 to +1', fontsize=14, fontweight='bold', color=MLPURPLE)

correlations = [1.0, 0.7, 0.3, -0.3, -0.7, -1.0]
titles = ['Perfect Positive\nr = 1.0', 'Strong Positive\nr = 0.7', 'Weak Positive\nr = 0.3',
          'Weak Negative\nr = -0.3', 'Strong Negative\nr = -0.7', 'Perfect Negative\nr = -1.0']
colors = [MLGREEN, MLGREEN, MLBLUE, MLORANGE, MLRED, MLRED]

for ax, r, title, color in zip(axes.flat, correlations, titles, colors):
    n = 100
    x = np.random.normal(0, 1, n)

    if r == 1.0:
        y = x
    elif r == -1.0:
        y = -x
    else:
        # Generate correlated data
        noise_var = 1 - r**2
        y = r * x + np.sqrt(noise_var) * np.random.normal(0, 1, n)

    ax.scatter(x, y, color=color, alpha=0.6, s=40, edgecolors='black')

    # Add regression line
    if abs(r) > 0:
        z = np.polyfit(x, y, 1)
        ax.plot(np.sort(x), np.poly1d(z)(np.sort(x)), color='black', linewidth=2)

    ax.set_title(title, fontsize=11, fontweight='bold', color=color)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)

    # Add actual correlation
    actual_r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'Actual: {actual_r:.2f}', transform=ax.transAxes,
            fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
