"""Prediction Line - Making predictions with the model"""
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

# Generate training data
n = 40
x_train = np.random.uniform(5, 25, n)
y_train = 2 + 0.45 * x_train + np.random.normal(0, 1.5, n)

# Fit model
slope = np.sum((x_train - x_train.mean()) * (y_train - y_train.mean())) / np.sum((x_train - x_train.mean())**2)
intercept = y_train.mean() - slope * x_train.mean()

# Plot training data and fit line
ax.scatter(x_train, y_train, c=MLBLUE, s=60, alpha=0.6, edgecolors='black', label='Training data')
x_line = np.linspace(0, 30, 100)
ax.plot(x_line, intercept + slope * x_line, color=MLGREEN, linewidth=2.5, label='Fitted line')

# Predict for a new x value
x_pred = 18
y_pred = intercept + slope * x_pred

# Show prediction with projection lines
ax.scatter([x_pred], [y_pred], c=MLORANGE, s=200, marker='*', zorder=5, edgecolors='black', label='Prediction')
ax.plot([x_pred, x_pred], [0, y_pred], color=MLORANGE, linestyle='--', linewidth=1.5)
ax.plot([0, x_pred], [y_pred, y_pred], color=MLORANGE, linestyle='--', linewidth=1.5)

# Annotate prediction
ax.annotate(f'Prediction:\nx = {x_pred} -> y = {y_pred:.1f}', xy=(x_pred, y_pred),
            xytext=(x_pred + 4, y_pred + 2), fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=MLORANGE))

ax.set_title('Making Predictions: model.predict(X_new)', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Risk (%)', fontsize=10)
ax.set_ylabel('Expected Return (%)', fontsize=10)
ax.set_xlim(0, 32)
ax.set_ylim(0, 18)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

# Add equation
ax.text(0.95, 0.05, f'$\\hat{{y}} = {intercept:.2f} + {slope:.2f}x$',
        transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
