"""Early Stopping - Knowing When to Stop Training"""
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
fig.suptitle('Early Stopping: Knowing When to Stop Training', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Early stopping visualization
ax1 = axes[0, 0]

epochs = np.arange(150)

# Training loss (keeps decreasing)
train_loss = 1.5 * np.exp(-0.02 * epochs) + 0.1 + np.random.randn(150) * 0.015

# Validation loss (decreases then increases)
val_loss = np.concatenate([
    1.6 * np.exp(-0.025 * epochs[:50]) + 0.2 + np.random.randn(50) * 0.02,
    0.25 + 0.003 * (epochs[50:] - 50) + np.random.randn(100) * 0.02
])

ax1.plot(epochs, train_loss, color=MLBLUE, linewidth=2, label='Training Loss')
ax1.plot(epochs, val_loss, color=MLORANGE, linewidth=2, label='Validation Loss')

# Mark early stopping point
best_epoch = np.argmin(val_loss)
ax1.axvline(best_epoch, color=MLGREEN, linestyle='--', linewidth=2, label=f'Best epoch: {best_epoch}')
ax1.scatter([best_epoch], [val_loss[best_epoch]], color=MLGREEN, s=150, zorder=5)

# Mark patience region
patience = 20
ax1.axvspan(best_epoch, best_epoch + patience, alpha=0.2, color=MLORANGE)
ax1.annotate('Patience: 20 epochs', xy=(best_epoch + patience/2, 0.6),
            fontsize=9, ha='center')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Early Stopping in Action', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Keras implementation
ax2 = axes[0, 1]
ax2.axis('off')

code = '''
KERAS EARLY STOPPING

from tensorflow.keras.callbacks import EarlyStopping

# CREATE CALLBACK
early_stop = EarlyStopping(
    monitor='val_loss',         # Metric to monitor
    patience=20,                # Epochs without improvement
    restore_best_weights=True,  # Go back to best model
    min_delta=0.001,            # Minimum improvement threshold
    verbose=1                   # Print when stopping
)


# USE IN TRAINING
history = model.fit(
    X_train, y_train,
    epochs=500,                 # Set high, early stop will kick in
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],     # Add callback here
    verbose=1
)

# Training stops when val_loss doesn't improve
# for 'patience' consecutive epochs


KEY PARAMETERS:
---------------
monitor: What to watch
  - 'val_loss' (most common)
  - 'val_accuracy'
  - Any metric in model.compile()

patience: How long to wait
  - Too low: stop too early
  - Too high: train too long
  - Typical: 10-50 epochs

restore_best_weights: IMPORTANT!
  - True: Return to best model
  - False: Keep final model (often overfit)

min_delta: Minimum improvement
  - Ignore tiny improvements
  - Typical: 0 or 0.001
'''

ax2.text(0.02, 0.98, code, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Keras Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Patience comparison
ax3 = axes[1, 0]

np.random.seed(42)

# Base validation loss curve
base_val = np.concatenate([
    1.0 * np.exp(-0.03 * epochs[:40]) + 0.2,
    0.22 + 0.002 * (epochs[40:] - 40)
]) + np.random.randn(150) * 0.015

# Different patience stopping points
patience_values = [5, 20, 50]
colors = [MLRED, MLGREEN, MLBLUE]

ax3.plot(epochs, base_val, color='gray', linewidth=2, alpha=0.5, label='Val Loss')

best_idx = np.argmin(base_val[:60])
ax3.scatter([best_idx], [base_val[best_idx]], color='black', s=100, zorder=5, marker='*', label='True Best')

for patience, color in zip(patience_values, colors):
    # Find where early stopping would trigger
    for i in range(best_idx, len(base_val) - patience):
        if all(base_val[i:i+patience] >= base_val[best_idx] - 0.01):
            stop_idx = i + patience
            break
    else:
        stop_idx = len(base_val) - 1

    ax3.axvline(stop_idx, color=color, linestyle='--', linewidth=2, alpha=0.7)
    ax3.scatter([stop_idx], [base_val[min(stop_idx, len(base_val)-1)]], color=color, s=80, zorder=5)
    ax3.text(stop_idx + 2, 0.6 - 0.1*patience_values.index(patience),
             f'patience={patience}\nstop@{stop_idx}', fontsize=8, color=color)

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Validation Loss')
ax3.set_title('Effect of Patience Value', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(alpha=0.3)

# Plot 4: Complete example with multiple callbacks
ax4 = axes[1, 1]
ax4.axis('off')

complete = '''
COMPLETE EXAMPLE WITH CALLBACKS

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# BUILD MODEL
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# SETUP CALLBACKS
callbacks = [
    # Stop when validation loss stops improving
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),

    # Save best model to file
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True
    ),

    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,        # Multiply lr by this
        patience=10,
        min_lr=1e-6
    )
]


# TRAIN
history = model.fit(
    X_train, y_train,
    epochs=500,            # High, callbacks will stop early
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)


# LOAD BEST MODEL
best_model = keras.models.load_model('best_model.keras')


TIPS:
-----
- Always use restore_best_weights=True
- Set epochs high, let early stopping decide
- Combine with model checkpointing for safety
'''

ax4.text(0.02, 0.98, complete, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
