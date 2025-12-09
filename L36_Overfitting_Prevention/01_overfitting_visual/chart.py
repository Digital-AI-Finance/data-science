"""Overfitting Visualization - Understanding the Problem"""
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
fig.suptitle('Overfitting: When Your Model Learns Too Much', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Visual comparison of fitting
ax1 = axes[0, 0]

# Generate data
x = np.linspace(0, 1, 20)
y_true = np.sin(2 * np.pi * x)
y_noisy = y_true + np.random.randn(20) * 0.3

x_smooth = np.linspace(0, 1, 100)
y_smooth = np.sin(2 * np.pi * x_smooth)

# Underfit (linear)
z_under = np.polyfit(x, y_noisy, 1)
y_under = np.polyval(z_under, x_smooth)

# Good fit (polynomial degree 3)
z_good = np.polyfit(x, y_noisy, 3)
y_good = np.polyval(z_good, x_smooth)

# Overfit (high degree polynomial)
z_over = np.polyfit(x, y_noisy, 15)
y_over = np.polyval(z_over, x_smooth)

ax1.scatter(x, y_noisy, color='black', s=60, zorder=5, label='Training data')
ax1.plot(x_smooth, y_smooth, 'k--', linewidth=1.5, alpha=0.5, label='True function')
ax1.plot(x_smooth, y_under, color=MLBLUE, linewidth=2, label='Underfit')
ax1.plot(x_smooth, y_good, color=MLGREEN, linewidth=2, label='Good fit')
ax1.plot(x_smooth, y_over, color=MLRED, linewidth=2, label='Overfit')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Underfitting vs Good Fit vs Overfitting', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.set_ylim(-2, 2)
ax1.grid(alpha=0.3)

# Plot 2: Training vs validation loss
ax2 = axes[0, 1]

epochs = np.arange(100)

# Training loss (always decreases)
train_loss = 1.5 * np.exp(-0.03 * epochs) + 0.1 + np.random.randn(100) * 0.02

# Validation loss (decreases then increases)
val_loss = np.concatenate([
    1.6 * np.exp(-0.03 * epochs[:40]) + 0.15 + np.random.randn(40) * 0.03,
    0.22 + 0.003 * (epochs[40:] - 40) + np.random.randn(60) * 0.03
])

ax2.plot(epochs, train_loss, color=MLBLUE, linewidth=2, label='Training Loss')
ax2.plot(epochs, val_loss, color=MLRED, linewidth=2, label='Validation Loss')

# Mark overfitting region
ax2.axvline(40, color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(epochs[40:], 0, 2, alpha=0.1, color=MLRED)
ax2.text(60, 0.9, 'OVERFITTING\nZONE', fontsize=10, ha='center', color=MLRED, fontweight='bold')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Overfitting in Training Curves', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 2)

# Plot 3: The overfitting problem explained
ax3 = axes[1, 0]
ax3.axis('off')

explanation = '''
UNDERSTANDING OVERFITTING

WHAT IS IT?
-----------
Model learns training data TOO WELL, including noise.
Memorizes instead of generalizing.


SYMPTOMS:
---------
- Training loss very low
- Validation loss much higher
- Gap between train/val grows
- Perfect on training, poor on new data


CAUSES:
-------
1. Model too complex (too many parameters)
2. Training too long
3. Not enough training data
4. No regularization


ANALOGY:
--------
Studying for exam by memorizing answers
vs understanding concepts.

Memorization works on practice tests,
fails on new questions.


THE BIAS-VARIANCE TRADEOFF:
---------------------------
Underfit: High bias, low variance
          (too simple, misses patterns)

Overfit: Low bias, high variance
         (too complex, fits noise)

Goal: Find the sweet spot!


DETECTION:
----------
ALWAYS use a validation set!
- If val_loss >> train_loss: overfitting
- If both high: underfitting
- If both low and similar: good fit
'''

ax3.text(0.02, 0.98, explanation, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Understanding Overfitting', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Model complexity vs error
ax4 = axes[1, 1]

complexity = np.arange(1, 20)

# Training error (decreases with complexity)
train_error = 1.5 / (complexity + 0.5) + 0.05

# Test error (U-shaped)
test_error = 0.3 / complexity + 0.015 * (complexity - 5)**2 + 0.1

ax4.plot(complexity, train_error, 'o-', color=MLBLUE, linewidth=2, markersize=6, label='Training Error')
ax4.plot(complexity, test_error, 's-', color=MLRED, linewidth=2, markersize=6, label='Test Error')

# Mark optimal complexity
optimal = np.argmin(test_error) + 1
ax4.axvline(optimal, color=MLGREEN, linestyle='--', linewidth=2, label=f'Optimal (complexity={optimal})')
ax4.scatter([optimal], [test_error[optimal-1]], color=MLGREEN, s=150, zorder=5)

# Shade regions
ax4.fill_between(complexity[:optimal-1], 0, 1.5, alpha=0.1, color=MLBLUE)
ax4.fill_between(complexity[optimal:], 0, 1.5, alpha=0.1, color=MLRED)

ax4.text(3, 1.2, 'Underfit', fontsize=10, ha='center', color=MLBLUE)
ax4.text(15, 1.2, 'Overfit', fontsize=10, ha='center', color=MLRED)

ax4.set_xlabel('Model Complexity (parameters, layers, etc.)')
ax4.set_ylabel('Error')
ax4.set_title('Bias-Variance Tradeoff', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(alpha=0.3)
ax4.set_ylim(0, 1.5)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
