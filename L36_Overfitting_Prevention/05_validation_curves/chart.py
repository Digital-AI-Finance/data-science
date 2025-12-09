"""Validation Curves - Diagnosing Model Performance"""
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
fig.suptitle('Validation Curves: Diagnosing Your Model', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Good training
ax1 = axes[0, 0]

epochs = np.arange(100)
train_good = 0.8 * np.exp(-0.03 * epochs) + 0.15 + np.random.randn(100) * 0.01
val_good = 0.85 * np.exp(-0.025 * epochs) + 0.2 + np.random.randn(100) * 0.015

ax1.plot(epochs, train_good, color=MLBLUE, linewidth=2, label='Training')
ax1.plot(epochs, val_good, color=MLORANGE, linewidth=2, label='Validation')

ax1.fill_between(epochs, train_good, val_good, alpha=0.2, color='gray')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('GOOD: Converging, Small Gap', fontsize=11, fontweight='bold', color=MLGREEN)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax1.text(60, 0.7, 'Small gap = Good generalization', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 2: Overfitting
ax2 = axes[0, 1]

train_over = 1.0 * np.exp(-0.04 * epochs) + 0.05 + np.random.randn(100) * 0.01
val_over = np.concatenate([
    1.05 * np.exp(-0.03 * epochs[:35]) + 0.15 + np.random.randn(35) * 0.02,
    0.22 + 0.004 * (epochs[35:] - 35) + np.random.randn(65) * 0.02
])

ax2.plot(epochs, train_over, color=MLBLUE, linewidth=2, label='Training')
ax2.plot(epochs, val_over, color=MLRED, linewidth=2, label='Validation')

ax2.fill_between(epochs[35:], train_over[35:], val_over[35:], alpha=0.3, color=MLRED)
ax2.axvline(35, color='gray', linestyle='--', alpha=0.5)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('OVERFITTING: Val Loss Increases', fontsize=11, fontweight='bold', color=MLRED)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

ax2.text(55, 0.35, 'Growing gap = Overfitting\nFix: Dropout, Early Stopping', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Underfitting
ax3 = axes[1, 0]

train_under = 0.9 - 0.002 * epochs + np.random.randn(100) * 0.015
val_under = 0.95 - 0.002 * epochs + np.random.randn(100) * 0.02

ax3.plot(epochs, train_under, color=MLBLUE, linewidth=2, label='Training')
ax3.plot(epochs, val_under, color=MLORANGE, linewidth=2, label='Validation')

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('UNDERFITTING: Both High, Not Learning', fontsize=11, fontweight='bold', color=MLBLUE)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

ax3.text(40, 0.75, 'Both losses high = Underfitting\nFix: More capacity, longer training', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 4: Diagnostic guide
ax4 = axes[1, 1]
ax4.axis('off')

guide = '''
VALIDATION CURVE DIAGNOSTIC GUIDE

PATTERN               | DIAGNOSIS      | FIX
----------------------|----------------|---------------------------
Both curves high,     | UNDERFITTING   | - Add more layers/neurons
barely decreasing     |                | - Train longer
                      |                | - Reduce regularization

Train low, val high   | OVERFITTING    | - Add dropout
(growing gap)         |                | - Early stopping
                      |                | - More training data
                      |                | - L2 regularization

Both curves low,      | GOOD FIT       | - You're done!
small stable gap      |                | - Maybe tune hyperparams

Both curves           | NOISY DATA     | - Data quality issue
very noisy            | or BAD LR      | - Try lower learning rate
                      |                | - Larger batch size

Val loss oscillates   | LR TOO HIGH    | - Reduce learning rate
                      |                | - Try learning rate schedule


WHAT TO LOOK FOR:
-----------------
1. FINAL VALUES: Are they acceptable?
2. GAP SIZE: Train-val gap should be small
3. TREND: Val should not increase
4. CONVERGENCE: Both should plateau


RULE OF THUMB:
--------------
If gap > 10% of train loss: too much overfitting
If both > 2x expected: underfitting


KERAS VISUALIZATION:
--------------------
import matplotlib.pyplot as plt

# After training
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''

ax4.text(0.02, 0.98, guide, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Diagnostic Guide', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
