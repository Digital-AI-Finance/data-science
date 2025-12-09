"""OLS Formula - Ordinary Least Squares derivation"""
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
fig.suptitle('Ordinary Least Squares (OLS): The Math', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate simple data
n = 20
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.1, 3.8, 5.2, 7.1, 8.5, 10.2, 11.8, 14.1, 15.5, 17.2])

# Calculate OLS components
x_mean = np.mean(x)
y_mean = np.mean(y)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - slope * x_mean

# Plot 1: Squared residuals visualization
ax1 = axes[0, 0]
y_pred = intercept + slope * x
residuals = y - y_pred

ax1.scatter(x, y, c=MLBLUE, s=100, edgecolors='black', zorder=5)
ax1.plot(x, y_pred, color=MLGREEN, linewidth=2.5)

# Draw squared residual boxes
for xi, yi, ypi, ri in zip(x, y, y_pred, residuals):
    # Draw residual line
    ax1.plot([xi, xi], [yi, ypi], color=MLRED, linewidth=2)
    # Draw square (approximate)
    if abs(ri) > 0.5:
        size = abs(ri)
        rect = plt.Rectangle((xi, min(yi, ypi)), size, size,
                             fill=True, color=MLRED, alpha=0.2, edgecolor=MLRED)
        ax1.add_patch(rect)

ax1.set_title('OLS Minimizes Squared Residuals', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: OLS formulas
ax2 = axes[0, 1]
ax2.axis('off')

formulas = r'''
OLS Formulas

The Goal: Minimize $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

Model: $\hat{y} = \beta_0 + \beta_1 x$

Slope:
$$\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

Intercept:
$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

Where:
$\bar{x}$ = mean of x
$\bar{y}$ = mean of y
'''

ax2.text(0.1, 0.9, formulas, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('OLS Formulas', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Step-by-step calculation
ax3 = axes[1, 0]
ax3.axis('off')

calculation = f'''
Step-by-Step Calculation

Given: x = [1, 2, 3, ..., 10]
       y = [2.1, 3.8, 5.2, ..., 17.2]

Step 1: Calculate means
   x_bar = {x_mean:.1f}
   y_bar = {y_mean:.2f}

Step 2: Calculate slope numerator
   sum((x-x_bar)*(y-y_bar)) = {numerator:.2f}

Step 3: Calculate slope denominator
   sum((x-x_bar)^2) = {denominator:.1f}

Step 4: Slope
   b1 = {numerator:.2f} / {denominator:.1f} = {slope:.3f}

Step 5: Intercept
   b0 = {y_mean:.2f} - {slope:.3f} * {x_mean:.1f} = {intercept:.3f}

Result: y = {intercept:.2f} + {slope:.2f}x
'''

ax3.text(0.05, 0.95, calculation, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Worked Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Final result
ax4 = axes[1, 1]
ax4.scatter(x, y, c=MLBLUE, s=100, edgecolors='black', zorder=5, label='Data points')
ax4.plot(x, y_pred, color=MLGREEN, linewidth=3, label=f'y = {intercept:.2f} + {slope:.2f}x')

# Mark the means
ax4.axvline(x_mean, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax4.axhline(y_mean, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax4.scatter([x_mean], [y_mean], c=MLORANGE, s=150, marker='*', zorder=6,
            edgecolors='black', label='Mean point')

ax4.set_title('Final OLS Regression Line', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('X', fontsize=10)
ax4.set_ylabel('Y', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Add R-squared
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
r_squared = 1 - (ss_res / ss_tot)
ax4.text(0.95, 0.05, f'R-squared = {r_squared:.4f}',
         transform=ax4.transAxes, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLGREEN))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
