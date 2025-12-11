"""RMSE - Root Mean Squared Error"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))


# Generate predictions
n = 30
y_true = np.random.uniform(50, 150, n)
y_pred = y_true + np.random.normal(0, 10, n)

errors = y_true - y_pred
mse = np.mean(errors**2)
rmse = np.sqrt(mse)

ax.scatter(range(n), y_true, c=MLBLUE, s=60, label='Actual', alpha=0.7)
ax.scatter(range(n), y_pred, c=MLORANGE, s=60, label='Predicted', marker='x')

for i in range(n):
    ax.plot([i, i], [y_true[i], y_pred[i]], color=MLRED, alpha=0.3, linewidth=1)

ax.set_title(f'RMSE = ${rmse:.2f}$ (same units as y)', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Observation', fontsize=10)
ax.set_ylabel('Value ($)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax.text(0.02, 0.98, f'RMSE = sqrt(MSE)\nPenalizes large errors more', transform=ax.transAxes,
        fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
