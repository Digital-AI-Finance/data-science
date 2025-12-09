"""Overfitting Visual - Why regularization is needed"""
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
fig.suptitle('Overfitting: The Problem Regularization Solves', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate simple data with noise
n = 15
x = np.linspace(0, 10, n)
y_true = 2 + 0.5 * x  # True linear relationship
y = y_true + np.random.normal(0, 1.2, n)  # Noisy observations

x_plot = np.linspace(-0.5, 10.5, 200)

# Plot 1: Underfitting (too simple)
ax1 = axes[0, 0]
ax1.scatter(x, y, c=MLBLUE, s=80, edgecolors='black', zorder=5, label='Training data')

# Horizontal line (constant model)
ax1.axhline(y.mean(), color=MLRED, linewidth=2.5, label=f'y = {y.mean():.1f} (constant)')

ax1.set_title('UNDERFITTING: Model Too Simple', fontsize=11, fontweight='bold', color=MLRED)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, 'High bias\nHigh error on both\ntrain and test',
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 2: Good fit (just right)
ax2 = axes[0, 1]
ax2.scatter(x, y, c=MLBLUE, s=80, edgecolors='black', zorder=5, label='Training data')

# Linear fit
slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
intercept = y.mean() - slope * x.mean()
ax2.plot(x_plot, intercept + slope * x_plot, color=MLGREEN, linewidth=2.5,
         label=f'y = {intercept:.1f} + {slope:.2f}x')

ax2.set_title('GOOD FIT: Right Complexity', fontsize=11, fontweight='bold', color=MLGREEN)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xlim(-0.5, 10.5)
ax2.text(0.05, 0.95, 'Low bias, Low variance\nGeneralizes well',
         transform=ax2.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 3: Overfitting (too complex)
ax3 = axes[1, 0]
ax3.scatter(x, y, c=MLBLUE, s=80, edgecolors='black', zorder=5, label='Training data')

# High-degree polynomial (overfits)
degree = 12
coeffs = np.polyfit(x, y, degree)
y_overfit = np.polyval(coeffs, x_plot)

# Clip extreme values for visualization
y_overfit = np.clip(y_overfit, -5, 15)

ax3.plot(x_plot, y_overfit, color=MLRED, linewidth=2.5, label=f'Degree {degree} polynomial')

ax3.set_title('OVERFITTING: Model Too Complex', fontsize=11, fontweight='bold', color=MLRED)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Y', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.set_xlim(-0.5, 10.5)
ax3.set_ylim(-2, 12)
ax3.text(0.05, 0.95, 'Low bias, High variance\nMemorizes noise\nFails on new data',
         transform=ax3.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 4: Bias-Variance tradeoff
ax4 = axes[1, 1]

complexity = np.linspace(0, 10, 100)
bias_squared = 5 * np.exp(-0.5 * complexity)
variance = 0.5 * np.exp(0.3 * complexity)
total_error = bias_squared + variance + 0.5  # + irreducible error

ax4.plot(complexity, bias_squared, color=MLBLUE, linewidth=2.5, label='Bias squared')
ax4.plot(complexity, variance, color=MLORANGE, linewidth=2.5, label='Variance')
ax4.plot(complexity, total_error, color=MLRED, linewidth=3, linestyle='--', label='Total Error')

# Mark optimal
optimal_idx = np.argmin(total_error)
ax4.scatter([complexity[optimal_idx]], [total_error[optimal_idx]], c=MLGREEN, s=150,
            zorder=5, edgecolors='black', label='Optimal complexity')
ax4.axvline(complexity[optimal_idx], color='gray', linestyle=':', linewidth=1.5)

ax4.set_title('Bias-Variance Tradeoff', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Model Complexity', fontsize=10)
ax4.set_ylabel('Error', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Add region labels
ax4.text(1, 4.5, 'Underfitting', fontsize=10, color=MLBLUE, fontweight='bold')
ax4.text(7, 4.5, 'Overfitting', fontsize=10, color=MLORANGE, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
