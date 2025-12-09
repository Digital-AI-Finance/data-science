"""Residuals - Understanding prediction errors"""
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
fig.suptitle('Residual Analysis: Checking Model Quality', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
n = 50
x = np.random.uniform(5, 25, n)
y = 3 + 0.5 * x + np.random.normal(0, 1.5, n)

# Fit model
slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
intercept = y.mean() - slope * x.mean()
y_pred = intercept + slope * x
residuals = y - y_pred

# Plot 1: What are residuals?
ax1 = axes[0, 0]

ax1.scatter(x, y, c=MLBLUE, s=80, alpha=0.7, edgecolors='black')
x_line = np.linspace(5, 25, 100)
ax1.plot(x_line, intercept + slope * x_line, color=MLGREEN, linewidth=2.5)

# Show residuals for a few points
for i in [5, 15, 25, 35]:
    color = MLRED if residuals[i] < 0 else MLGREEN
    ax1.plot([x[i], x[i]], [y[i], y_pred[i]], color=color, linewidth=2.5, alpha=0.8)
    ax1.annotate(f'{residuals[i]:.1f}', xy=(x[i], (y[i] + y_pred[i])/2),
                 xytext=(5, 0), textcoords='offset points', fontsize=9, color=color)

ax1.set_title('Residuals = Actual - Predicted', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.grid(alpha=0.3)

# Add formula
ax1.text(0.95, 0.05, r'$e_i = y_i - \hat{y}_i$', transform=ax1.transAxes,
         fontsize=12, ha='right', bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 2: Residuals vs Fitted (check for patterns)
ax2 = axes[0, 1]

colors = [MLGREEN if r >= 0 else MLRED for r in residuals]
ax2.scatter(y_pred, residuals, c=colors, s=60, alpha=0.7, edgecolors='black')
ax2.axhline(0, color='gray', linewidth=2, linestyle='--')

ax2.set_title('Residuals vs Fitted Values', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Fitted Values', fontsize=10)
ax2.set_ylabel('Residuals', fontsize=10)
ax2.grid(alpha=0.3)

# Add interpretation
ax2.text(0.05, 0.95, 'Good: Random scatter around 0\nBad: Pattern (curved, funnel)',
         transform=ax2.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 3: Residual histogram (check normality)
ax3 = axes[1, 0]

ax3.hist(residuals, bins=15, color=MLBLUE, alpha=0.7, edgecolor='black', density=True)

# Add normal curve
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
ax3.plot(x_norm, stats.norm.pdf(x_norm, 0, residuals.std()), color=MLRED,
         linewidth=2.5, label='Normal distribution')

ax3.axvline(0, color='gray', linewidth=2, linestyle='--')
ax3.axvline(residuals.mean(), color=MLORANGE, linewidth=2, label=f'Mean: {residuals.mean():.3f}')

ax3.set_title('Residual Distribution', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Residual Value', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(fontsize=8)

# Add stats
ax3.text(0.95, 0.95, f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}',
         transform=ax3.transAxes, fontsize=9, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 4: Residual diagnostics summary
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate diagnostic stats
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuals))

diagnostics = f'''
RESIDUAL DIAGNOSTIC CHECKLIST

1. MEAN SHOULD BE NEAR ZERO
   Mean residual: {residuals.mean():.4f}
   Status: {"PASS" if abs(residuals.mean()) < 0.1 else "CHECK"}

2. VARIANCE SHOULD BE CONSTANT
   (Check Residuals vs Fitted plot)
   Look for: Funnel shape = heteroscedasticity
   Status: Check visually

3. NO PATTERN IN RESIDUALS
   (Check Residuals vs Fitted plot)
   Look for: Curves, waves = non-linearity
   Status: Check visually

4. NORMALLY DISTRIBUTED
   (Check histogram and Q-Q plot)
   Status: Check visually

KEY METRICS:
- MSE: {mse:.4f}
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}
- Residual Std: {residuals.std():.4f}
'''

ax4.text(0.02, 0.95, diagnostics, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Diagnostic Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
