"""Keras Dropout Implementation"""
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
fig.suptitle('Dropout in Keras: Implementation and Usage', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic dropout syntax
ax1 = axes[0, 0]
ax1.axis('off')

syntax = '''
KERAS DROPOUT SYNTAX

from tensorflow import keras
from tensorflow.keras import layers


# BASIC USAGE:
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),         # Drop 30% of neurons
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),         # Drop 30% of neurons
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),         # Drop 20% of neurons
    layers.Dense(1, activation='sigmoid')
])


# WHERE TO PUT DROPOUT:
- AFTER activation layers (Dense with activation)
- BEFORE output layer is optional
- NOT on input layer (usually)
- NOT on output layer


# AUTOMATIC BEHAVIOR:
- Training: Dropout is ACTIVE (neurons dropped)
- Inference: Dropout is INACTIVE (all neurons used)
- Keras handles this automatically!


# model.predict() and model.evaluate()
# automatically disable dropout
'''

ax1.text(0.02, 0.98, syntax, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Dropout Syntax', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Comparison with and without dropout
ax2 = axes[0, 1]

epochs = np.arange(100)

# Without dropout (overfitting)
train_no_drop = 0.6 * np.exp(-0.05 * epochs) + 0.05 + np.random.randn(100) * 0.01
val_no_drop = np.concatenate([
    0.65 * np.exp(-0.04 * epochs[:30]) + 0.1 + np.random.randn(30) * 0.015,
    0.18 + 0.002 * (epochs[30:] - 30) + np.random.randn(70) * 0.015
])

# With dropout (better generalization)
train_drop = 0.55 * np.exp(-0.04 * epochs) + 0.08 + np.random.randn(100) * 0.01
val_drop = 0.6 * np.exp(-0.035 * epochs) + 0.1 + np.random.randn(100) * 0.012

ax2.plot(epochs, train_no_drop, color=MLBLUE, linewidth=1.5, alpha=0.7, label='Train (no dropout)')
ax2.plot(epochs, val_no_drop, color=MLBLUE, linewidth=2, linestyle='--', label='Val (no dropout)')
ax2.plot(epochs, train_drop, color=MLGREEN, linewidth=1.5, alpha=0.7, label='Train (with dropout)')
ax2.plot(epochs, val_drop, color=MLGREEN, linewidth=2, linestyle='--', label='Val (with dropout)')

ax2.axvline(30, color='gray', linestyle=':', alpha=0.5)
ax2.text(35, 0.5, 'Without dropout:\nval loss increases', fontsize=8, color=MLBLUE)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Effect of Dropout on Training', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Complete example
ax3 = axes[1, 0]
ax3.axis('off')

example = '''
COMPLETE EXAMPLE WITH DROPOUT

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. PREPARE DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 2. BUILD MODEL WITH DROPOUT
model = keras.Sequential([
    # First hidden layer
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.4),  # 40% dropout

    # Second hidden layer
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # 30% dropout

    # Third hidden layer
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),  # 20% dropout

    # Output layer (no dropout!)
    layers.Dense(1, activation='sigmoid')
])


# 3. COMPILE AND TRAIN
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)


# 4. EVALUATE (dropout automatically disabled)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2%}")
'''

ax3.text(0.02, 0.98, example, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Dropout best practices
ax4 = axes[1, 1]
ax4.axis('off')

practices = '''
DROPOUT BEST PRACTICES

RECOMMENDED RATES:
------------------
Layer Type         | Dropout Rate
-------------------|-------------
After input        | 0.0 - 0.2
Hidden (small net) | 0.2 - 0.3
Hidden (large net) | 0.3 - 0.5
Before output      | 0.0 - 0.2


COMMON PATTERNS:
----------------
Pattern 1: Constant
  Dropout(0.3) after every hidden layer

Pattern 2: Decreasing
  0.4 -> 0.3 -> 0.2 (decrease toward output)

Pattern 3: Increasing
  0.2 -> 0.3 -> 0.4 (increase toward output)


WHEN TO USE MORE DROPOUT:
-------------------------
- Large model (many parameters)
- Small dataset
- Seeing large train/val gap
- Complex problem


WHEN TO USE LESS DROPOUT:
-------------------------
- Small model
- Large dataset
- Model underfitting
- Already using other regularization


COMBINING WITH OTHER TECHNIQUES:
--------------------------------
Dropout works well with:
- Early stopping
- L2 regularization (but usually pick one)
- Data augmentation
- Batch normalization (place after BN)


DEBUGGING TIP:
--------------
If training is slow or accuracy is low,
try REDUCING dropout rate first!
'''

ax4.text(0.02, 0.98, practices, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Best Practices', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
