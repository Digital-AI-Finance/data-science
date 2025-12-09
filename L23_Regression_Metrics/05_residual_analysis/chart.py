"""Residual Analysis - Diagnostic plots"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Residual Analysis: Four Key Diagnostic Plots', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate well-behaved data
n = 100
x = np.random.uniform(5, 25, n)
y_true = 3 + 0.5 * x
y = y_true + np.random.normal(0, 2, n)

# Fit model
slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
intercept = y.mean() - slope * x.mean()
y_pred = intercept + slope * x
residuals = y - y_pred

# Plot 1: Residuals vs Fitted
ax1 = axes[0, 0]

ax1.scatter(y_pred, residuals, c=MLBLUE, s=50, alpha=0.7, edgecolors='black')
ax1.axhline(0, color=MLRED, linewidth=2, linestyle='--')

# Add lowess smoothing line (simulated)
sorted_idx = np.argsort(y_pred)
window = 15
smooth_y = np.convolve(residuals[sorted_idx], np.ones(window)/window, mode='valid')
ax1.plot(np.sort(y_pred)[window//2:-window//2+1], smooth_y, color=MLORANGE, linewidth=2)

ax1.set_title('1. Residuals vs Fitted', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Fitted Values', fontsize=10)
ax1.set_ylabel('Residuals', fontsize=10)
ax1.grid(alpha=0.3)

ax1.text(0.02, 0.98, 'Check: Random scatter\naround 0 (no pattern)',
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 2: Q-Q Plot (normality check)
ax2 = axes[0, 1]

stats.probplot(residuals, dist="norm", plot=ax2)
ax2.get_lines()[0].set_color(MLBLUE)
ax2.get_lines()[0].set_markersize(6)
ax2.get_lines()[1].set_color(MLRED)
ax2.get_lines()[1].set_linewidth(2)

ax2.set_title('2. Q-Q Plot (Normality)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3)

ax2.text(0.02, 0.98, 'Check: Points follow\ndiagonal line',
         transform=ax2.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 3: Scale-Location (homoscedasticity)
ax3 = axes[1, 0]

sqrt_std_resid = np.sqrt(np.abs(residuals / residuals.std()))
ax3.scatter(y_pred, sqrt_std_resid, c=MLBLUE, s=50, alpha=0.7, edgecolors='black')

# Add smooth line
smooth_sqrt = np.convolve(sqrt_std_resid[sorted_idx], np.ones(window)/window, mode='valid')
ax3.plot(np.sort(y_pred)[window//2:-window//2+1], smooth_sqrt, color=MLORANGE, linewidth=2)

ax3.set_title('3. Scale-Location (Constant Variance)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Fitted Values', fontsize=10)
ax3.set_ylabel('Sqrt(|Standardized Residuals|)', fontsize=10)
ax3.grid(alpha=0.3)

ax3.text(0.02, 0.98, 'Check: Horizontal band\n(no funnel shape)',
         transform=ax3.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 4: Residuals vs Order (independence)
ax4 = axes[1, 1]

ax4.plot(range(len(residuals)), residuals, color=MLBLUE, linewidth=1.5, marker='o', markersize=4, alpha=0.7)
ax4.axhline(0, color=MLRED, linewidth=2, linestyle='--')

ax4.set_title('4. Residuals vs Order (Independence)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Observation Order', fontsize=10)
ax4.set_ylabel('Residuals', fontsize=10)
ax4.grid(alpha=0.3)

ax4.text(0.02, 0.98, 'Check: No trend or\npattern over time',
         transform=ax4.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
