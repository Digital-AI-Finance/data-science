"""MSE Formula - Mean Squared Error explained"""
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
fig.suptitle('Mean Squared Error (MSE): The Fundamental Metric', fontsize=14, fontweight='bold', color=MLPURPLE)

# Simple example data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2.1, 3.8, 5.5, 6.9, 8.2, 10.1, 11.5, 13.2])
y_pred = 1.0 + 1.5 * x  # Predicted values
residuals = y - y_pred

# Plot 1: Visual representation of errors
ax1 = axes[0, 0]

ax1.scatter(x, y, c=MLBLUE, s=100, edgecolors='black', zorder=5, label='Actual')
ax1.plot(x, y_pred, color=MLGREEN, linewidth=2.5, label='Predicted')

# Draw residuals and squared error boxes
for xi, yi, ypi, ri in zip(x, y, y_pred, residuals):
    # Draw residual line
    ax1.plot([xi, xi], [yi, ypi], color=MLRED, linewidth=2)
    # Draw squared box (scaled for visualization)
    size = abs(ri) * 0.4
    rect_bottom = min(yi, ypi)
    rect = plt.Rectangle((xi - size/2, rect_bottom), size, abs(ri),
                         fill=True, color=MLRED, alpha=0.2, edgecolor=MLRED)
    ax1.add_patch(rect)

ax1.set_title('Squared Errors Visualized', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: MSE formula
ax2 = axes[0, 1]
ax2.axis('off')

formula = r'''
MEAN SQUARED ERROR (MSE)

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- n = number of observations
- $y_i$ = actual value
- $\hat{y}_i$ = predicted value
- $(y_i - \hat{y}_i)$ = residual (error)

Steps:
1. Calculate each error: $e_i = y_i - \hat{y}_i$
2. Square each error: $e_i^2$
3. Take the mean: $\frac{1}{n}\sum e_i^2$

Properties:
- Always non-negative
- Penalizes large errors more (squared)
- Same units as $y^2$ (not y)
- Lower is better
'''

ax2.text(0.05, 0.95, formula, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('MSE Formula', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Step-by-step calculation
ax3 = axes[1, 0]
ax3.axis('off')

# Create calculation table
mse = np.mean(residuals**2)

calc = f'''
STEP-BY-STEP CALCULATION

Actual (y):    {y}
Predicted:     {y_pred}

Residuals (e): {np.round(residuals, 2)}
Squared (e2):  {np.round(residuals**2, 2)}

Sum of squared errors: {np.sum(residuals**2):.4f}
Number of points (n):  {len(y)}

MSE = {np.sum(residuals**2):.4f} / {len(y)} = {mse:.4f}

Interpretation:
- On average, squared error is {mse:.4f}
- Typical error magnitude: sqrt(MSE) = {np.sqrt(mse):.3f}
  (This is RMSE - more interpretable!)
'''

ax3.text(0.02, 0.95, calc, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Worked Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Why squared errors?
ax4 = axes[1, 1]

# Compare absolute vs squared error penalty
error_range = np.linspace(-3, 3, 100)
absolute_penalty = np.abs(error_range)
squared_penalty = error_range**2

ax4.plot(error_range, absolute_penalty, color=MLBLUE, linewidth=2.5, label='|error| (MAE)')
ax4.plot(error_range, squared_penalty, color=MLRED, linewidth=2.5, label='error-squared (MSE)')

ax4.axvline(0, color='gray', linewidth=1)
ax4.axhline(0, color='gray', linewidth=1)

# Mark example errors
ax4.scatter([1, 2], [1, 4], c=MLRED, s=100, zorder=5, edgecolors='black')
ax4.scatter([1, 2], [1, 2], c=MLBLUE, s=100, zorder=5, edgecolors='black')

ax4.annotate('Error=2: MSE=4, MAE=2\nMSE penalizes more!', xy=(2, 4),
             xytext=(2.3, 5), fontsize=9,
             arrowprops=dict(arrowstyle='->', color=MLRED))

ax4.set_title('Why Squared? Large Errors Penalized More', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Error', fontsize=10)
ax4.set_ylabel('Penalty', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
ax4.set_xlim(-3.5, 3.5)
ax4.set_ylim(-0.5, 10)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
