"""Homoscedasticity - Constant variance check"""
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

n = 50

# Left: GOOD - Constant variance
x_good = np.random.uniform(5, 25, n)
y_good = 3 + 0.5 * x_good + np.random.normal(0, 1.5, n)  # Constant noise

slope = np.sum((x_good - x_good.mean()) * (y_good - y_good.mean())) / np.sum((x_good - x_good.mean())**2)
intercept = y_good.mean() - slope * x_good.mean()
y_pred_good = intercept + slope * x_good
residuals_good = y_good - y_pred_good

ax1.scatter(y_pred_good, residuals_good, c=MLGREEN, s=60, alpha=0.7, edgecolors='black')
ax1.axhline(0, color='gray', linewidth=2, linestyle='--')
ax1.set_title('Constant Variance: GOOD', fontsize=12, fontweight='bold', color=MLGREEN)
ax1.set_xlabel('Fitted Values', fontsize=10)
ax1.set_ylabel('Residuals', fontsize=10)
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, 'Equal spread\nacross all values', transform=ax1.transAxes, fontsize=10,
         va='top', color=MLGREEN, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Right: VIOLATED - Funnel shape (heteroscedasticity)
x_bad = np.random.uniform(5, 25, n)
y_bad = 3 + 0.5 * x_bad + np.random.normal(0, x_bad * 0.12, n)  # Variance increases with x

slope_bad = np.sum((x_bad - x_bad.mean()) * (y_bad - y_bad.mean())) / np.sum((x_bad - x_bad.mean())**2)
intercept_bad = y_bad.mean() - slope_bad * x_bad.mean()
y_pred_bad = intercept_bad + slope_bad * x_bad
residuals_bad = y_bad - y_pred_bad

ax2.scatter(y_pred_bad, residuals_bad, c=MLRED, s=60, alpha=0.7, edgecolors='black')
ax2.axhline(0, color='gray', linewidth=2, linestyle='--')
ax2.set_title('Constant Variance: VIOLATED', fontsize=12, fontweight='bold', color=MLRED)
ax2.set_xlabel('Fitted Values', fontsize=10)
ax2.set_ylabel('Residuals', fontsize=10)
ax2.grid(alpha=0.3)
ax2.text(0.05, 0.95, 'Funnel shape!\n(Heteroscedasticity)', transform=ax2.transAxes, fontsize=10,
         va='top', color=MLRED, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Homoscedasticity Check: Is Variance Constant?', fontsize=13, fontweight='bold', color=MLPURPLE, y=1.02)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
