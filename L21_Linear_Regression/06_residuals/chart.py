"""Residuals - Checking model quality via residuals vs fitted"""
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

# Generate data
n = 50
x = np.random.uniform(5, 25, n)
y = 3 + 0.5 * x + np.random.normal(0, 1.5, n)

# Fit model
slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
intercept = y.mean() - slope * x.mean()
y_pred = intercept + slope * x
residuals = y - y_pred

# Color by residual sign
colors = [MLGREEN if r >= 0 else MLRED for r in residuals]
ax.scatter(y_pred, residuals, c=colors, s=70, alpha=0.7, edgecolors='black')

# Add horizontal line at 0
ax.axhline(0, color='gray', linewidth=2, linestyle='--', label='Zero line')

ax.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Fitted Values (Predicted)', fontsize=10)
ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=10)
ax.grid(alpha=0.3)

# Add interpretation box
interpretation = 'Good model indicators:\n- Random scatter around 0\n- No funnel shape (constant variance)\n- No curved pattern (linearity)'
ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Add stats
stats_text = f'Mean residual: {residuals.mean():.4f}\nStd residual: {residuals.std():.2f}'
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLGREEN, linewidth=1))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
