"""Training Curves - Monitoring the Learning Process"""
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
fig.suptitle('Training Curves: Monitoring Neural Network Training', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Normal training curves
ax1 = axes[0, 0]

epochs = np.arange(100)

# Good training
train_loss = 1.5 * np.exp(-0.03 * epochs) + 0.15 + np.random.randn(100) * 0.02
val_loss = 1.6 * np.exp(-0.025 * epochs) + 0.2 + np.random.randn(100) * 0.03

ax1.plot(epochs, train_loss, color=MLBLUE, linewidth=2, label='Training Loss')
ax1.plot(epochs, val_loss, color=MLORANGE, linewidth=2, label='Validation Loss')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Good Training: Loss Curves', fontsize=11, fontweight='bold', color=MLGREEN)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Add annotation
ax1.annotate('Converging nicely', xy=(70, 0.3), fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8))

# Plot 2: Overfitting vs underfitting
ax2 = axes[0, 1]

# Underfitting
train_under = 0.8 - 0.002 * epochs + np.random.randn(100) * 0.01
val_under = 0.85 - 0.002 * epochs + np.random.randn(100) * 0.015

# Overfitting
train_over = 1.2 * np.exp(-0.05 * epochs) + 0.05 + np.random.randn(100) * 0.01
val_over = np.concatenate([
    1.3 * np.exp(-0.05 * epochs[:30]) + 0.1 + np.random.randn(30) * 0.02,
    0.25 + 0.005 * (epochs[30:] - 30) + np.random.randn(70) * 0.02
])

ax2.plot(epochs, train_under, color=MLBLUE, linewidth=2, linestyle=':', label='Underfit Train', alpha=0.7)
ax2.plot(epochs, val_under, color=MLORANGE, linewidth=2, linestyle=':', label='Underfit Val', alpha=0.7)

ax2.plot(epochs, train_over, color=MLBLUE, linewidth=2, label='Overfit Train')
ax2.plot(epochs, val_over, color=MLRED, linewidth=2, label='Overfit Val')

ax2.axvline(30, color='gray', linestyle='--', alpha=0.5)
ax2.text(32, 0.9, 'Val loss starts\nincreasing', fontsize=8)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Underfitting vs Overfitting', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.3)

# Plot 3: Diagnosing training curves
ax3 = axes[1, 0]
ax3.axis('off')

diagnosis = '''
DIAGNOSING TRAINING CURVES

GOOD TRAINING:
--------------
- Both curves decrease and plateau
- Small gap between train and val
- Validation follows training closely


OVERFITTING:
------------
- Training loss keeps decreasing
- Validation loss starts INCREASING
- Growing gap between curves
FIX: More data, dropout, early stopping


UNDERFITTING:
-------------
- Both losses stay high
- Model not learning patterns
- Curves flat or barely decreasing
FIX: More capacity (layers/neurons), longer training


OSCILLATING:
------------
- Loss jumps up and down
- Unstable training
FIX: Lower learning rate, larger batch size


PLATEAUING EARLY:
-----------------
- Both curves flatten too soon
- Not reaching optimal performance
FIX: Higher learning rate, different architecture


WHAT TO MONITOR:
----------------
1. Training Loss: Should decrease
2. Validation Loss: Should decrease, not diverge
3. Gap: Should be small and stable
4. Accuracy: Should increase (for classification)


KERAS HISTORY:
--------------
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=100)

# Access metrics
train_loss = history.history['loss']
val_loss = history.history['val_loss']
'''

ax3.text(0.02, 0.98, diagnosis, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Diagnosing Training Curves', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Finance example - stock prediction
ax4 = axes[1, 1]

# Simulated training for stock return prediction
epochs = np.arange(150)

# Model 1: Simple MLP
train1 = 0.08 * np.exp(-0.02 * epochs) + 0.025 + np.random.randn(150) * 0.002
val1 = 0.09 * np.exp(-0.015 * epochs) + 0.035 + np.random.randn(150) * 0.003

# Model 2: With dropout (more regularized)
train2 = 0.075 * np.exp(-0.018 * epochs) + 0.028 + np.random.randn(150) * 0.002
val2 = 0.085 * np.exp(-0.016 * epochs) + 0.032 + np.random.randn(150) * 0.0025

ax4.plot(epochs, train1, color=MLBLUE, linewidth=1.5, alpha=0.6, label='Simple Train')
ax4.plot(epochs, val1, color=MLBLUE, linewidth=2, linestyle='--', label='Simple Val')
ax4.plot(epochs, train2, color=MLGREEN, linewidth=1.5, alpha=0.6, label='Regularized Train')
ax4.plot(epochs, val2, color=MLGREEN, linewidth=2, linestyle='--', label='Regularized Val')

# Mark early stopping point
ax4.axvline(80, color=MLRED, linestyle=':', linewidth=2, label='Early Stop')
ax4.text(85, 0.07, 'Early\nStopping', fontsize=8, color=MLRED)

ax4.set_xlabel('Epoch')
ax4.set_ylabel('MSE Loss')
ax4.set_title('Finance: Return Prediction Training', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(alpha=0.3)

ax4.text(10, 0.03, 'Regularized model:\nSmaller train-val gap',
         fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
