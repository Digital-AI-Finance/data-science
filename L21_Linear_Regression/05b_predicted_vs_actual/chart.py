"""Predicted vs Actual - Model diagnostic"""
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

# Generate data and fit
n = 50
x = np.random.uniform(5, 25, n)
y_actual = 2 + 0.45 * x + np.random.normal(0, 1.5, n)

slope = np.sum((x - x.mean()) * (y_actual - y_actual.mean())) / np.sum((x - x.mean())**2)
intercept = y_actual.mean() - slope * x.mean()
y_pred = intercept + slope * x

# Calculate R-squared
r_squared = 1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - y_actual.mean())**2)

ax.scatter(y_actual, y_pred, c=MLBLUE, s=60, alpha=0.7, edgecolors='black', label='Predictions')

# Perfect prediction line (45-degree)
min_val = min(y_actual.min(), y_pred.min())
max_val = max(y_actual.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color=MLRED, linewidth=2,
        linestyle='--', label='Perfect prediction')

ax.set_title('Predicted vs Actual Values', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Actual Return (%)', fontsize=10)
ax.set_ylabel('Predicted Return (%)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
# Non-square aspect for slide compatibility

# Add R-squared
ax.text(0.95, 0.05, f'R-squared = {r_squared:.3f}', transform=ax.transAxes,
        fontsize=11, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
