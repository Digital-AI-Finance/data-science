"""Loss Functions - Measuring Prediction Error"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Loss Functions: Measuring Prediction Error', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: MSE vs MAE
ax1 = axes[0, 0]

error = np.linspace(-3, 3, 100)

# MSE
mse = error**2

# MAE
mae = np.abs(error)

# Huber (smooth combination)
delta = 1.0
huber = np.where(np.abs(error) <= delta, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))

ax1.plot(error, mse, color=MLBLUE, linewidth=2.5, label='MSE (L2)')
ax1.plot(error, mae, color=MLGREEN, linewidth=2.5, label='MAE (L1)')
ax1.plot(error, huber, color=MLORANGE, linewidth=2.5, linestyle='--', label='Huber')

ax1.set_xlabel('Error (y - y_hat)')
ax1.set_ylabel('Loss')
ax1.set_title('Regression Loss Functions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 6)

ax1.text(1.5, 5, 'MSE: penalizes large errors more\nMAE: robust to outliers',
         fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

# Plot 2: Binary cross-entropy
ax2 = axes[0, 1]

y_hat = np.linspace(0.001, 0.999, 100)

# BCE when y=1
bce_y1 = -np.log(y_hat)

# BCE when y=0
bce_y0 = -np.log(1 - y_hat)

ax2.plot(y_hat, bce_y1, color=MLBLUE, linewidth=2.5, label='y=1: -log(p)')
ax2.plot(y_hat, bce_y0, color=MLRED, linewidth=2.5, label='y=0: -log(1-p)')

ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

ax2.set_xlabel('Predicted Probability (y_hat)')
ax2.set_ylabel('Loss')
ax2.set_title('Binary Cross-Entropy Loss', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 5)

ax2.text(0.05, 4, 'If y=1: want p close to 1\nIf y=0: want p close to 0',
         fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Loss functions summary
ax3 = axes[1, 0]
ax3.axis('off')

summary = '''
LOSS FUNCTIONS SUMMARY

REGRESSION:
-----------
MSE (Mean Squared Error):
    L = (1/n) * sum((y - y_hat)^2)
    Keras: loss='mse'
    - Penalizes large errors heavily
    - Sensitive to outliers

MAE (Mean Absolute Error):
    L = (1/n) * sum(|y - y_hat|)
    Keras: loss='mae'
    - Robust to outliers
    - Non-differentiable at 0


BINARY CLASSIFICATION:
----------------------
Binary Cross-Entropy:
    L = -[y*log(y_hat) + (1-y)*log(1-y_hat)]
    Keras: loss='binary_crossentropy'
    - Output: sigmoid
    - y in {0, 1}


MULTI-CLASS CLASSIFICATION:
---------------------------
Categorical Cross-Entropy:
    L = -sum(y_k * log(y_hat_k))
    Keras: loss='categorical_crossentropy'
    - Output: softmax
    - y is one-hot encoded

Sparse Categorical Cross-Entropy:
    Same formula, but y is integer (0, 1, 2, ...)
    Keras: loss='sparse_categorical_crossentropy'


CHOOSING LOSS:
--------------
Task            | Loss                    | Output Activation
Regression      | mse or mae              | linear (none)
Binary class.   | binary_crossentropy     | sigmoid
Multi-class     | categorical_crossentropy| softmax
'''

ax3.text(0.02, 0.98, summary, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Loss Functions Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Finance example
ax4 = axes[1, 1]

# Simulated loss curves for different loss functions
np.random.seed(42)
epochs = np.arange(100)

# MSE training
mse_train = 0.5 * np.exp(-0.03 * epochs) + 0.02 + np.random.randn(100) * 0.01
mse_val = 0.55 * np.exp(-0.025 * epochs) + 0.04 + np.random.randn(100) * 0.015

# MAE training (more robust)
mae_train = 0.4 * np.exp(-0.035 * epochs) + 0.015 + np.random.randn(100) * 0.008
mae_val = 0.42 * np.exp(-0.03 * epochs) + 0.03 + np.random.randn(100) * 0.01

ax4.plot(epochs, mse_train, color=MLBLUE, linewidth=1.5, alpha=0.7, label='MSE Train')
ax4.plot(epochs, mse_val, color=MLBLUE, linewidth=2, linestyle='--', label='MSE Val')
ax4.plot(epochs, mae_train, color=MLGREEN, linewidth=1.5, alpha=0.7, label='MAE Train')
ax4.plot(epochs, mae_val, color=MLGREEN, linewidth=2, linestyle='--', label='MAE Val')

ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.set_title('Finance Example: Return Prediction', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

ax4.text(50, 0.4, 'MAE often better for\nfinancial data with outliers',
         fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
