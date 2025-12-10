"""Residual Histogram - Checking normality"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
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

# Generate data and residuals
n = 50
x = np.random.uniform(5, 25, n)
y = 3 + 0.5 * x + np.random.normal(0, 1.5, n)

slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
intercept = y.mean() - slope * x.mean()
y_pred = intercept + slope * x
residuals = y - y_pred

# Histogram
ax.hist(residuals, bins=12, color=MLBLUE, alpha=0.7, edgecolor='black', density=True, label='Residuals')

# Normal curve overlay
x_norm = np.linspace(residuals.min() - 1, residuals.max() + 1, 100)
ax.plot(x_norm, stats.norm.pdf(x_norm, 0, residuals.std()), color=MLRED,
        linewidth=2.5, label='Normal distribution')

ax.axvline(0, color='gray', linewidth=2, linestyle='--', label='Zero')
ax.axvline(residuals.mean(), color=MLORANGE, linewidth=2, label=f'Mean: {residuals.mean():.3f}')

ax.set_title('Residual Distribution (Should Be Normal)', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Residual Value', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
