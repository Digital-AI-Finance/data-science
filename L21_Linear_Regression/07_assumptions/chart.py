"""Assumptions - Linear regression requirements"""
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

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Linear Regression Assumptions: Checking Model Validity', fontsize=14, fontweight='bold', color=MLPURPLE)

# Good data
n = 50
x_good = np.random.uniform(5, 25, n)
y_good = 3 + 0.5 * x_good + np.random.normal(0, 1.5, n)

# Fit good model
slope = np.sum((x_good - x_good.mean()) * (y_good - y_good.mean())) / np.sum((x_good - x_good.mean())**2)
intercept = y_good.mean() - slope * x_good.mean()
residuals_good = y_good - (intercept + slope * x_good)

# Plot 1: Linearity - GOOD
ax1 = axes[0, 0]
ax1.scatter(x_good, y_good, c=MLBLUE, s=60, alpha=0.7, edgecolors='black')
ax1.plot(np.sort(x_good), intercept + slope * np.sort(x_good), color=MLGREEN, linewidth=2.5)
ax1.set_title('1. Linearity: GOOD', fontsize=11, fontweight='bold', color=MLGREEN)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Linearity - VIOLATED
ax2 = axes[0, 1]
x_nonlin = np.random.uniform(0, 10, n)
y_nonlin = 2 + 0.5 * x_nonlin + 0.1 * x_nonlin**2 + np.random.normal(0, 0.8, n)
slope_bad = np.sum((x_nonlin - x_nonlin.mean()) * (y_nonlin - y_nonlin.mean())) / np.sum((x_nonlin - x_nonlin.mean())**2)
intercept_bad = y_nonlin.mean() - slope_bad * x_nonlin.mean()

ax2.scatter(x_nonlin, y_nonlin, c=MLBLUE, s=60, alpha=0.7, edgecolors='black')
ax2.plot(np.sort(x_nonlin), intercept_bad + slope_bad * np.sort(x_nonlin), color=MLRED, linewidth=2.5)
ax2.set_title('1. Linearity: VIOLATED', fontsize=11, fontweight='bold', color=MLRED)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.grid(alpha=0.3)
ax2.text(0.05, 0.95, 'Curved pattern!', transform=ax2.transAxes, fontsize=10,
         va='top', color=MLRED, fontweight='bold')

# Plot 3: Homoscedasticity - GOOD
ax3 = axes[0, 2]
y_pred_good = intercept + slope * x_good
ax3.scatter(y_pred_good, residuals_good, c=MLGREEN, s=60, alpha=0.7, edgecolors='black')
ax3.axhline(0, color='gray', linewidth=2, linestyle='--')
ax3.set_title('2. Constant Variance: GOOD', fontsize=11, fontweight='bold', color=MLGREEN)
ax3.set_xlabel('Fitted Values', fontsize=10)
ax3.set_ylabel('Residuals', fontsize=10)
ax3.grid(alpha=0.3)
ax3.text(0.05, 0.95, 'Equal spread', transform=ax3.transAxes, fontsize=10,
         va='top', color=MLGREEN, fontweight='bold')

# Plot 4: Homoscedasticity - VIOLATED (heteroscedasticity)
ax4 = axes[1, 0]
x_het = np.random.uniform(5, 25, n)
y_het = 3 + 0.5 * x_het + np.random.normal(0, x_het * 0.15, n)  # Variance increases with x
slope_het = np.sum((x_het - x_het.mean()) * (y_het - y_het.mean())) / np.sum((x_het - x_het.mean())**2)
intercept_het = y_het.mean() - slope_het * x_het.mean()
residuals_het = y_het - (intercept_het + slope_het * x_het)

ax4.scatter(intercept_het + slope_het * x_het, residuals_het, c=MLRED, s=60, alpha=0.7, edgecolors='black')
ax4.axhline(0, color='gray', linewidth=2, linestyle='--')
ax4.set_title('2. Constant Variance: VIOLATED', fontsize=11, fontweight='bold', color=MLRED)
ax4.set_xlabel('Fitted Values', fontsize=10)
ax4.set_ylabel('Residuals', fontsize=10)
ax4.grid(alpha=0.3)
ax4.text(0.05, 0.95, 'Funnel shape!', transform=ax4.transAxes, fontsize=10,
         va='top', color=MLRED, fontweight='bold')

# Plot 5: Normality - Q-Q plot GOOD
ax5 = axes[1, 1]
stats.probplot(residuals_good, dist="norm", plot=ax5)
ax5.get_lines()[0].set_color(MLBLUE)
ax5.get_lines()[0].set_markersize(8)
ax5.get_lines()[1].set_color(MLGREEN)
ax5.get_lines()[1].set_linewidth(2)
ax5.set_title('3. Normality (Q-Q Plot): GOOD', fontsize=11, fontweight='bold', color=MLGREEN)
ax5.grid(alpha=0.3)

# Plot 6: Summary of assumptions
ax6 = axes[1, 2]
ax6.axis('off')

summary = '''
LINEAR REGRESSION ASSUMPTIONS

1. LINEARITY
   - Relationship between X and Y is linear
   - Check: Scatter plot, residuals vs fitted
   - Fix: Transform variables, use polynomial

2. HOMOSCEDASTICITY
   - Constant variance of residuals
   - Check: Residuals vs fitted plot
   - Fix: Log transform, weighted least squares

3. NORMALITY OF RESIDUALS
   - Residuals are normally distributed
   - Check: Histogram, Q-Q plot
   - Note: Less critical for large samples

4. INDEPENDENCE
   - Observations are independent
   - Check: Durbin-Watson test (time series)
   - Fix: Time series models if violated

5. NO MULTICOLLINEARITY (multiple regression)
   - Predictors not highly correlated
   - Check: VIF, correlation matrix
   - Fix: Remove or combine variables
'''

ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax6.set_title('Assumptions Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
