"""Regression Concept - Finding the best-fit line with residuals"""
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

# Generate data: market risk vs expected return
n = 30
x = np.random.uniform(5, 25, n)
y_true = 2 + 0.4 * x
y = y_true + np.random.normal(0, 1.5, n)

# Fit line
slope, intercept = np.polyfit(x, y, 1)
y_pred = intercept + slope * x
x_line = np.linspace(5, 25, 100)

# Plot data points
ax.scatter(x, y, c=MLBLUE, s=80, alpha=0.7, edgecolors='black', label='Data points')

# Plot regression line
ax.plot(x_line, intercept + slope * x_line, color=MLGREEN, linewidth=2.5, label='Best-fit line')

# Draw residuals
for xi, yi, ypi in zip(x, y, y_pred):
    ax.plot([xi, xi], [yi, ypi], color=MLRED, linewidth=1.5, alpha=0.6)

# Add equation
ax.text(0.95, 0.05, f'$y = {intercept:.2f} + {slope:.2f}x$',
        transform=ax.transAxes, fontsize=12, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLGREEN, linewidth=2))

ax.set_title('Linear Regression: Finding the Best-Fit Line', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Risk (%)', fontsize=10)
ax.set_ylabel('Return (%)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

# Add annotation for residuals
ax.text(0.05, 0.95, 'Red lines = residuals\n(prediction errors)',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
