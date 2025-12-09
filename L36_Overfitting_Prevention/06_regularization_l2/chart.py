"""L2 Regularization - Weight Decay"""
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
fig.suptitle('L2 Regularization (Weight Decay)', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: L2 regularization concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
L2 REGULARIZATION (RIDGE)

THE IDEA:
---------
Add a penalty for large weights to the loss function.

Loss_total = Loss_original + lambda * sum(w^2)

lambda: Regularization strength (hyperparameter)


WHY IT WORKS:
-------------
- Penalizes large weights
- Forces weights to be small
- Prevents any single feature from dominating
- Makes model simpler (more generalizable)


INTUITION:
----------
Without regularization:
  Model can use huge weights to fit noise

With L2:
  "Keep weights small unless really needed"
  -> Smoother decision boundaries
  -> Less sensitive to small changes in input


THE MATH:
---------
Original loss: L(y, y_hat)

With L2:
L_reg = L(y, y_hat) + (lambda/2) * sum(W_ij^2)

Gradient update becomes:
W := W - lr * (dL/dW + lambda * W)
   = W * (1 - lr * lambda) - lr * dL/dW

This is why it's called "weight decay"!
Each update shrinks weights by factor (1 - lr * lambda)


L2 vs L1:
---------
L2 (Ridge): Shrinks weights, keeps all features
L1 (Lasso): Can zero out weights (feature selection)
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('L2 Regularization Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Effect on weights
ax2 = axes[0, 1]

# Simulate weight distributions
np.random.seed(42)
weights_no_reg = np.random.randn(1000) * 2  # Wide distribution
weights_l2_weak = np.random.randn(1000) * 1  # Narrower
weights_l2_strong = np.random.randn(1000) * 0.3  # Much narrower

ax2.hist(weights_no_reg, bins=30, alpha=0.5, color=MLBLUE, label='No regularization', density=True)
ax2.hist(weights_l2_weak, bins=30, alpha=0.5, color=MLGREEN, label='L2 (lambda=0.001)', density=True)
ax2.hist(weights_l2_strong, bins=30, alpha=0.5, color=MLRED, label='L2 (lambda=0.01)', density=True)

ax2.set_xlabel('Weight Value')
ax2.set_ylabel('Density')
ax2.set_title('Weight Distributions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax2.text(0, 0.8, 'L2 pulls weights\ntoward zero', fontsize=9, ha='center',
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Keras implementation
ax3 = axes[1, 0]
ax3.axis('off')

keras_code = '''
KERAS L2 REGULARIZATION

from tensorflow import keras
from tensorflow.keras import layers, regularizers


# METHOD 1: Per-layer regularization
model = keras.Sequential([
    layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),  # L2 on weights
        bias_regularizer=regularizers.l2(0.001),    # L2 on biases (optional)
        input_shape=(10,)
    ),
    layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    ),
    layers.Dense(1, activation='sigmoid')
])


# METHOD 2: Combined L1/L2 (Elastic Net)
model = keras.Sequential([
    layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
        input_shape=(10,)
    ),
    layers.Dense(1, activation='sigmoid')
])


# COMMON LAMBDA VALUES:
# 0.0001 (1e-4): Light regularization
# 0.001 (1e-3): Medium (good starting point)
# 0.01 (1e-2): Strong regularization


# TIP: kernel = weights, bias = biases
# Usually only regularize kernel (weights)


# COMPILE AND TRAIN (loss automatically includes L2)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=100, validation_split=0.2)
'''

ax3.text(0.02, 0.98, keras_code, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Keras Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Effect of lambda on validation performance
ax4 = axes[1, 1]

lambdas = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
lambda_labels = ['0', '1e-4', '5e-4', '1e-3', '5e-3', '1e-2', '5e-2', '1e-1']

# Simulated performance
train_acc = [0.99, 0.97, 0.95, 0.92, 0.88, 0.83, 0.75, 0.65]
val_acc = [0.78, 0.84, 0.88, 0.90, 0.88, 0.84, 0.76, 0.62]

ax4.plot(range(len(lambdas)), train_acc, 'o-', color=MLBLUE, linewidth=2, markersize=8, label='Training Acc')
ax4.plot(range(len(lambdas)), val_acc, 's-', color=MLORANGE, linewidth=2, markersize=8, label='Validation Acc')

# Mark optimal
optimal_idx = np.argmax(val_acc)
ax4.axvline(optimal_idx, color=MLGREEN, linestyle='--', linewidth=2)
ax4.scatter([optimal_idx], [val_acc[optimal_idx]], color=MLGREEN, s=150, zorder=5)

ax4.set_xticks(range(len(lambdas)))
ax4.set_xticklabels(lambda_labels, fontsize=8)
ax4.set_xlabel('Lambda (regularization strength)')
ax4.set_ylabel('Accuracy')
ax4.set_title('Effect of L2 Strength', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Add regions
ax4.fill_between([0, 2], 0.5, 1, alpha=0.1, color=MLRED)
ax4.fill_between([5, 7], 0.5, 1, alpha=0.1, color=MLBLUE)

ax4.text(1, 0.55, 'Under-\nregularized', fontsize=8, ha='center', color=MLRED)
ax4.text(6, 0.55, 'Over-\nregularized', fontsize=8, ha='center', color=MLBLUE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
