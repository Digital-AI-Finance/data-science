"""R-squared - Coefficient of determination"""
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
fig.suptitle('R-squared: Proportion of Variance Explained', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data with different R-squared values
n = 50
x = np.linspace(0, 10, n)

# High R-squared (0.9)
y_high = 2 + 1.5 * x + np.random.normal(0, 1, n)

# Medium R-squared (0.5)
y_med = 2 + 1.5 * x + np.random.normal(0, 4, n)

# Low R-squared (0.1)
y_low = 2 + 1.5 * x + np.random.normal(0, 10, n)

def calc_r2(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

# Plot 1: Visual comparison of R-squared values
ax1 = axes[0, 0]

for y_data, label, color in [(y_high, 'High R-sq', MLGREEN),
                               (y_med, 'Medium R-sq', MLORANGE),
                               (y_low, 'Low R-sq', MLRED)]:
    # Fit line
    slope = np.sum((x - x.mean()) * (y_data - y_data.mean())) / np.sum((x - x.mean())**2)
    intercept = y_data.mean() - slope * x.mean()
    y_pred = intercept + slope * x
    r2 = calc_r2(y_data, y_pred)

    ax1.scatter(x, y_data, c=color, s=30, alpha=0.5)
    ax1.plot(x, y_pred, color=color, linewidth=2, label=f'{label}: R-sq={r2:.2f}')

ax1.set_title('Same Slope, Different R-squared', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: R-squared formula
ax2 = axes[0, 1]
ax2.axis('off')

formula = r'''
R-SQUARED (Coefficient of Determination)

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Where:
$$SS_{res} = \sum(y_i - \hat{y}_i)^2$$ (unexplained variance)
$$SS_{tot} = \sum(y_i - \bar{y})^2$$ (total variance)

Interpretation:
- R-sq = 0.75 means model explains 75% of variance
- Remaining 25% is unexplained (noise, missing variables)

Range:
- R-sq = 1.0: Perfect fit (all variance explained)
- R-sq = 0.0: Model no better than mean
- R-sq < 0.0: Worse than mean (possible with test data)

Key Insight:
R-sq tells you FIT, not prediction quality!
High R-sq doesn't guarantee good predictions.
'''

ax2.text(0.02, 0.95, formula, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('R-squared Formula', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: SS_tot vs SS_res visualization
ax3 = axes[1, 0]

# Use medium R-squared data
y = y_med
y_mean = np.mean(y)
slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
intercept = y.mean() - slope * x.mean()
y_pred = intercept + slope * x

ax3.scatter(x, y, c=MLBLUE, s=50, edgecolors='black', zorder=5)
ax3.plot(x, y_pred, color=MLGREEN, linewidth=2.5, label='Regression line')
ax3.axhline(y_mean, color=MLRED, linestyle='--', linewidth=2, label=f'Mean: {y_mean:.1f}')

# Show total variance (distance from mean)
for xi, yi, ypi in zip(x[::5], y[::5], y_pred[::5]):
    ax3.plot([xi, xi], [yi, y_mean], color=MLRED, linewidth=1, alpha=0.5)  # SS_tot
    ax3.plot([xi + 0.15, xi + 0.15], [yi, ypi], color=MLGREEN, linewidth=1.5)  # SS_res

ax3.set_title('SS_tot (red) vs SS_res (green)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Y', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Calculate and display
ss_tot = np.sum((y - y_mean)**2)
ss_res = np.sum((y - y_pred)**2)
r2 = 1 - ss_res / ss_tot

ax3.text(0.95, 0.05, f'SS_tot = {ss_tot:.1f}\nSS_res = {ss_res:.1f}\nR-sq = {r2:.3f}',
         transform=ax3.transAxes, fontsize=9, ha='right',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLPURPLE))

# Plot 4: R-squared interpretation guide
ax4 = axes[1, 1]

# R-squared benchmarks
r2_values = [0.1, 0.3, 0.5, 0.7, 0.9]
labels = ['Very Weak', 'Weak', 'Moderate', 'Strong', 'Very Strong']
colors = [MLRED, MLORANGE, MLORANGE, MLGREEN, MLGREEN]

bars = ax4.barh(labels, r2_values, color=colors, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, r2_values):
    ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.0%}',
             va='center', fontsize=10, fontweight='bold')

ax4.set_xlim(0, 1.1)
ax4.set_title('R-squared Interpretation Guide', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('R-squared', fontsize=10)
ax4.grid(alpha=0.3, axis='x')

# Add domain context
ax4.text(0.5, -0.15, 'Note: "Good" R-sq varies by domain.\nIn finance, R-sq > 0.05 can be valuable!',
         transform=ax4.transAxes, fontsize=9, ha='center', style='italic')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
