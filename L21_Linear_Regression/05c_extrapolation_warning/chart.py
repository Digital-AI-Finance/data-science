"""Extrapolation Warning - Danger of predicting outside training range"""
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

# Training data
n = 40
x_train = np.random.uniform(5, 25, n)
y_train = 2 + 0.45 * x_train + np.random.normal(0, 1.5, n)

slope = np.sum((x_train - x_train.mean()) * (y_train - y_train.mean())) / np.sum((x_train - x_train.mean())**2)
intercept = y_train.mean() - slope * x_train.mean()

ax.scatter(x_train, y_train, c=MLBLUE, s=60, alpha=0.6, edgecolors='black', label='Training data')

# Shade training range
ax.axvspan(x_train.min(), x_train.max(), alpha=0.1, color=MLGREEN, label='Safe zone (interpolation)')

# Extended line
x_extended = np.linspace(0, 40, 100)
y_extended = intercept + slope * x_extended

# Interpolation (safe)
mask_safe = x_extended <= x_train.max()
ax.plot(x_extended[mask_safe], y_extended[mask_safe], color=MLGREEN, linewidth=2.5)

# Extrapolation (risky)
mask_risky = x_extended >= x_train.max()
ax.plot(x_extended[mask_risky], y_extended[mask_risky], color=MLRED, linewidth=2.5,
        linestyle='--', label='Danger zone (extrapolation)')

# Mark dangerous prediction
x_danger = 35
y_danger = intercept + slope * x_danger
ax.scatter([x_danger], [y_danger], c=MLRED, s=150, marker='X', zorder=5, edgecolors='black')
ax.annotate('Unreliable!', xy=(x_danger, y_danger), xytext=(x_danger - 6, y_danger + 2),
            fontsize=10, color=MLRED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=MLRED))

ax.set_title('Extrapolation Warning: Stay Within Training Range', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Risk (%)', fontsize=10)
ax.set_ylabel('Expected Return (%)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
