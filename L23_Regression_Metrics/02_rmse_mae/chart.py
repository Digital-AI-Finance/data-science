"""RMSE and MAE - Interpretable error metrics"""
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
fig.suptitle('RMSE and MAE: Interpretable Error Metrics', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate predictions with different error patterns
n = 50
y_true = np.random.uniform(50, 150, n)

# Model 1: Small consistent errors
errors_small = np.random.normal(0, 5, n)
y_pred_small = y_true + errors_small

# Model 2: Few large outlier errors
errors_outlier = np.random.normal(0, 3, n)
errors_outlier[0:5] = np.random.uniform(15, 25, 5)  # Add outliers
y_pred_outlier = y_true + errors_outlier

# Plot 1: RMSE vs MAE formulas
ax1 = axes[0, 0]
ax1.axis('off')

formulas = r'''
RMSE (Root Mean Squared Error)

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- Same units as y (interpretable!)
- Typical error magnitude
- Penalizes outliers more


MAE (Mean Absolute Error)

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- Same units as y
- Average error magnitude
- More robust to outliers


Key Relationship:
MAE <= RMSE always
(equality only if all errors equal)
'''

ax1.text(0.05, 0.95, formulas, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('RMSE and MAE Formulas', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Compare RMSE vs MAE on two models
ax2 = axes[0, 1]

# Calculate metrics
rmse_small = np.sqrt(np.mean(errors_small**2))
mae_small = np.mean(np.abs(errors_small))
rmse_outlier = np.sqrt(np.mean(errors_outlier**2))
mae_outlier = np.mean(np.abs(errors_outlier))

models = ['Consistent\nErrors', 'With\nOutliers']
x = np.arange(len(models))
width = 0.35

bars1 = ax2.bar(x - width/2, [rmse_small, rmse_outlier], width, label='RMSE', color=MLBLUE, edgecolor='black')
bars2 = ax2.bar(x + width/2, [mae_small, mae_outlier], width, label='MAE', color=MLGREEN, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_title('RMSE vs MAE: Outlier Sensitivity', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Error ($)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis='y')

# Add annotations
ax2.annotate(f'RMSE-MAE gap:\n{rmse_outlier - mae_outlier:.1f}', xy=(1, rmse_outlier),
             xytext=(1.3, rmse_outlier + 1), fontsize=9,
             arrowprops=dict(arrowstyle='->', color=MLRED))

# Plot 3: Error distribution comparison
ax3 = axes[1, 0]

ax3.hist(errors_small, bins=15, alpha=0.6, color=MLBLUE, edgecolor='black', label='Consistent errors')
ax3.hist(errors_outlier, bins=15, alpha=0.6, color=MLORANGE, edgecolor='black', label='With outliers')

ax3.axvline(0, color='gray', linewidth=1.5, linestyle='--')

ax3.set_title('Error Distributions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Error ($)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: When to use which
ax4 = axes[1, 1]
ax4.axis('off')

guide = '''
WHEN TO USE WHICH METRIC

RMSE (Root Mean Squared Error):
- Standard choice for regression
- Same units as target variable
- Penalizes large errors more
- Good when: outliers are meaningful
  and should be avoided
- Example: Stock price prediction
  (big errors are costly)

MAE (Mean Absolute Error):
- More robust to outliers
- Easier to interpret
- All errors weighted equally
- Good when: outliers may be noise
  or data quality issues
- Example: House price prediction
  (some extreme sales are anomalies)

PRACTICAL TIP:
If RMSE >> MAE, you have outliers.
Investigate them before deciding
which metric to optimize.

sklearn Code:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
'''

ax4.text(0.02, 0.98, guide, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Choosing Between RMSE and MAE', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
