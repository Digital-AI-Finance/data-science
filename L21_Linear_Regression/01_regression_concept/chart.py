"""Regression Concept - Finding the best-fit line"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Linear Regression: Finding the Best-Fit Line', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data: market risk vs expected return
n = 30
x = np.random.uniform(5, 25, n)  # Risk (volatility)
y_true = 2 + 0.4 * x  # True relationship
y = y_true + np.random.normal(0, 1.5, n)  # Add noise

# Plot 1: The problem - finding a pattern
ax1 = axes[0, 0]
ax1.scatter(x, y, c=MLBLUE, s=80, alpha=0.7, edgecolors='black')

ax1.set_title('The Problem: Is There a Pattern?', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Risk (%)', fontsize=10)
ax1.set_ylabel('Return (%)', fontsize=10)
ax1.grid(alpha=0.3)

ax1.text(0.05, 0.95, 'Question: How does\nrisk relate to return?',
         transform=ax1.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 2: Multiple possible lines
ax2 = axes[0, 1]
ax2.scatter(x, y, c=MLBLUE, s=80, alpha=0.7, edgecolors='black')

# Draw several possible lines
x_line = np.linspace(5, 25, 100)
lines = [
    (0.3, 4, MLRED, 'Line 1'),
    (0.5, 0, MLORANGE, 'Line 2'),
    (0.4, 2, MLGREEN, 'Best Fit?')
]

for slope, intercept, color, label in lines:
    ax2.plot(x_line, intercept + slope * x_line, color=color, linewidth=2,
             linestyle='--' if color != MLGREEN else '-', label=label)

ax2.set_title('Which Line Fits Best?', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Risk (%)', fontsize=10)
ax2.set_ylabel('Return (%)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Show residuals
ax3 = axes[1, 0]
ax3.scatter(x, y, c=MLBLUE, s=80, alpha=0.7, edgecolors='black')

# Fit line
slope, intercept = np.polyfit(x, y, 1)
y_pred = intercept + slope * x
ax3.plot(x_line, intercept + slope * x_line, color=MLGREEN, linewidth=2.5, label='Best fit line')

# Draw residuals
for xi, yi, ypi in zip(x, y, y_pred):
    ax3.plot([xi, xi], [yi, ypi], color=MLRED, linewidth=1.5, alpha=0.7)

ax3.set_title('Residuals: Vertical Distances', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Risk (%)', fontsize=10)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

ax3.text(0.95, 0.05, 'Red lines = residuals\n(prediction errors)',
         transform=ax3.transAxes, fontsize=9, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 4: The regression equation
ax4 = axes[1, 1]
ax4.scatter(x, y, c=MLBLUE, s=80, alpha=0.7, edgecolors='black')
ax4.plot(x_line, intercept + slope * x_line, color=MLGREEN, linewidth=2.5)

# Add equation
ax4.text(0.5, 0.95, f'$y = {intercept:.2f} + {slope:.2f}x$',
         transform=ax4.transAxes, fontsize=14, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLGREEN, linewidth=2))

# Add interpretation
ax4.text(0.05, 0.05,
         f'Interpretation:\n'
         f'- Intercept ({intercept:.1f}): Base return when risk = 0\n'
         f'- Slope ({slope:.2f}): For each 1% more risk,\n'
         f'  expect {slope:.2f}% more return',
         transform=ax4.transAxes, fontsize=9, va='bottom',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

ax4.set_title('The Regression Line Equation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Risk (%)', fontsize=10)
ax4.set_ylabel('Return (%)', fontsize=10)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
