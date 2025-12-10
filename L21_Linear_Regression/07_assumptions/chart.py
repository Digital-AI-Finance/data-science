"""Assumptions - Good vs violated linearity comparison"""
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
    'figure.figsize': (12, 5),
    'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

# Create side-by-side comparison (2 panels, but same concept - good vs bad)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

n = 40

# Left panel: GOOD - Linear relationship
x_good = np.random.uniform(5, 25, n)
y_good = 3 + 0.5 * x_good + np.random.normal(0, 1.5, n)

slope = np.sum((x_good - x_good.mean()) * (y_good - y_good.mean())) / np.sum((x_good - x_good.mean())**2)
intercept = y_good.mean() - slope * x_good.mean()

ax1.scatter(x_good, y_good, c=MLBLUE, s=60, alpha=0.7, edgecolors='black')
ax1.plot(np.sort(x_good), intercept + slope * np.sort(x_good), color=MLGREEN, linewidth=2.5)
ax1.set_title('Linearity: GOOD', fontsize=12, fontweight='bold', color=MLGREEN)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, 'Linear relationship\nRegression is valid', transform=ax1.transAxes, fontsize=10,
         va='top', color=MLGREEN, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Right panel: VIOLATED - Curved relationship
x_bad = np.random.uniform(0, 10, n)
y_bad = 2 + 0.5 * x_bad + 0.12 * x_bad**2 + np.random.normal(0, 0.8, n)

slope_bad = np.sum((x_bad - x_bad.mean()) * (y_bad - y_bad.mean())) / np.sum((x_bad - x_bad.mean())**2)
intercept_bad = y_bad.mean() - slope_bad * x_bad.mean()

ax2.scatter(x_bad, y_bad, c=MLBLUE, s=60, alpha=0.7, edgecolors='black')
ax2.plot(np.sort(x_bad), intercept_bad + slope_bad * np.sort(x_bad), color=MLRED, linewidth=2.5)
ax2.set_title('Linearity: VIOLATED', fontsize=12, fontweight='bold', color=MLRED)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.grid(alpha=0.3)
ax2.text(0.05, 0.95, 'Curved pattern!\nNeed transformation', transform=ax2.transAxes, fontsize=10,
         va='top', color=MLRED, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Key Assumption: Is the Relationship Linear?', fontsize=13, fontweight='bold', color=MLPURPLE, y=1.02)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
