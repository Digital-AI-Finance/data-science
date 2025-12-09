"""Prediction Line - Making and visualizing predictions"""
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
fig.suptitle('Making Predictions with Linear Regression', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate training data
n = 40
x_train = np.random.uniform(5, 25, n)
y_train = 2 + 0.45 * x_train + np.random.normal(0, 1.5, n)

# Fit model
slope = np.sum((x_train - x_train.mean()) * (y_train - y_train.mean())) / np.sum((x_train - x_train.mean())**2)
intercept = y_train.mean() - slope * x_train.mean()

# Plot 1: Prediction at a single point
ax1 = axes[0, 0]

ax1.scatter(x_train, y_train, c=MLBLUE, s=60, alpha=0.6, edgecolors='black')
x_line = np.linspace(0, 30, 100)
ax1.plot(x_line, intercept + slope * x_line, color=MLGREEN, linewidth=2.5)

# Predict for x = 20
x_pred = 20
y_pred = intercept + slope * x_pred

ax1.scatter([x_pred], [y_pred], c=MLORANGE, s=200, marker='*', zorder=5, edgecolors='black')
ax1.plot([x_pred, x_pred], [0, y_pred], color=MLORANGE, linestyle='--', linewidth=1.5)
ax1.plot([0, x_pred], [y_pred, y_pred], color=MLORANGE, linestyle='--', linewidth=1.5)

ax1.annotate(f'Prediction: ({x_pred}, {y_pred:.1f})', xy=(x_pred, y_pred),
             xytext=(x_pred + 3, y_pred + 2), fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLORANGE))

ax1.set_title('Point Prediction', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Risk (%)', fontsize=10)
ax1.set_ylabel('Expected Return (%)', fontsize=10)
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 18)
ax1.grid(alpha=0.3)

# Plot 2: Multiple predictions
ax2 = axes[0, 1]

ax2.scatter(x_train, y_train, c=MLBLUE, s=60, alpha=0.6, edgecolors='black', label='Training data')
ax2.plot(x_line, intercept + slope * x_line, color=MLGREEN, linewidth=2.5, label='Fitted line')

# Multiple predictions
x_new = np.array([8, 12, 18, 22, 27])
y_new = intercept + slope * x_new

ax2.scatter(x_new, y_new, c=MLORANGE, s=150, marker='D', zorder=5,
            edgecolors='black', label='Predictions')

# Add prediction values
for xi, yi in zip(x_new, y_new):
    ax2.annotate(f'{yi:.1f}', xy=(xi, yi), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=9, color=MLORANGE)

ax2.set_title('Multiple Predictions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Risk (%)', fontsize=10)
ax2.set_ylabel('Expected Return (%)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Prediction vs Actual (training)
ax3 = axes[1, 0]

y_train_pred = intercept + slope * x_train

ax3.scatter(y_train, y_train_pred, c=MLBLUE, s=60, alpha=0.7, edgecolors='black')

# Perfect prediction line
min_val, max_val = min(y_train.min(), y_train_pred.min()), max(y_train.max(), y_train_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], color=MLRED, linewidth=2, linestyle='--',
         label='Perfect prediction')

ax3.set_title('Predicted vs Actual', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Actual Return (%)', fontsize=10)
ax3.set_ylabel('Predicted Return (%)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.set_aspect('equal', adjustable='box')

# Add R-squared
r_squared = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - y_train.mean())**2)
ax3.text(0.05, 0.95, f'R-squared = {r_squared:.3f}', transform=ax3.transAxes,
         fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 4: Extrapolation warning
ax4 = axes[1, 1]

ax4.scatter(x_train, y_train, c=MLBLUE, s=60, alpha=0.6, edgecolors='black')

# Shade training range
ax4.axvspan(x_train.min(), x_train.max(), alpha=0.1, color=MLGREEN, label='Training range')

# Show extrapolation
x_extended = np.linspace(0, 40, 100)
y_extended = intercept + slope * x_extended

ax4.plot(x_extended[x_extended <= x_train.max()],
         y_extended[x_extended <= x_train.max()],
         color=MLGREEN, linewidth=2.5, label='Interpolation (safe)')
ax4.plot(x_extended[x_extended >= x_train.max()],
         y_extended[x_extended >= x_train.max()],
         color=MLRED, linewidth=2.5, linestyle='--', label='Extrapolation (risky)')

# Mark dangerous prediction
x_danger = 35
y_danger = intercept + slope * x_danger
ax4.scatter([x_danger], [y_danger], c=MLRED, s=150, marker='X', zorder=5, edgecolors='black')
ax4.annotate('Unreliable!\n(Outside training data)', xy=(x_danger, y_danger),
             xytext=(x_danger - 8, y_danger + 2), fontsize=9, color=MLRED,
             arrowprops=dict(arrowstyle='->', color=MLRED))

ax4.set_title('Extrapolation Warning', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Risk (%)', fontsize=10)
ax4.set_ylabel('Expected Return (%)', fontsize=10)
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
