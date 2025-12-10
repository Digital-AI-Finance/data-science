"""OLS Formula - Visualizing squared residuals"""
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

# Generate simple data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.1, 3.8, 5.2, 7.1, 8.5, 10.2, 11.8, 14.1, 15.5, 17.2])

# Calculate OLS
x_mean = np.mean(x)
y_mean = np.mean(y)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - slope * x_mean

y_pred = intercept + slope * x
residuals = y - y_pred

# Plot data and line
ax.scatter(x, y, c=MLBLUE, s=100, edgecolors='black', zorder=5, label='Data points')
ax.plot(x, y_pred, color=MLGREEN, linewidth=2.5, label=f'Best fit: y = {intercept:.2f} + {slope:.2f}x')

# Draw squared residual boxes
for xi, yi, ypi, ri in zip(x, y, y_pred, residuals):
    # Draw residual line
    ax.plot([xi, xi], [yi, ypi], color=MLRED, linewidth=2)
    # Draw square (approximate visualization)
    if abs(ri) > 0.3:
        size = abs(ri) * 0.4  # Scale for visibility
        rect = plt.Rectangle((xi - size/2, min(yi, ypi)), size, abs(ri),
                             fill=True, color=MLRED, alpha=0.2, edgecolor=MLRED, linewidth=1)
        ax.add_patch(rect)

ax.set_title('OLS Minimizes Sum of Squared Residuals', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

# Add formula box
formula_text = r'Minimize: $\sum(y_i - \hat{y}_i)^2$' + f'\nSSE = {np.sum(residuals**2):.2f}'
ax.text(0.95, 0.05, formula_text, transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
