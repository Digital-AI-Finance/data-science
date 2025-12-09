"""Keras Model Saving - Deep Learning Persistence"""
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
fig.suptitle('Keras Model Saving: Deep Learning Persistence', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Keras saving overview
ax1 = axes[0, 0]
ax1.axis('off')

overview = '''
KERAS MODEL SAVING

WHAT KERAS SAVES:
-----------------
- Model architecture (layers, connections)
- Trained weights
- Optimizer state
- Compilation configuration

THREE SAVING APPROACHES:
------------------------
1. SavedModel format (TensorFlow native)
2. HDF5 format (.h5)
3. Weights only


1. SAVEDMODEL (RECOMMENDED):
----------------------------
model.save('my_model')  # Creates folder

# Structure:
my_model/
  saved_model.pb
  variables/
    variables.data-00000-of-00001
    variables.index

# Load
model = keras.models.load_model('my_model')


2. HDF5 FORMAT:
---------------
model.save('model.h5')

# Single file, smaller
# Good for sharing

model = keras.models.load_model('model.h5')


3. WEIGHTS ONLY:
----------------
# Save only weights (need architecture to reload)
model.save_weights('weights.h5')

# Load into same architecture
new_model = create_model()  # Must match!
new_model.load_weights('weights.h5')


WHICH TO USE:
-------------
- Deployment: SavedModel
- Sharing: HDF5
- Checkpoints: Weights only
'''

ax1.text(0.02, 0.98, overview, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Keras Saving Overview', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Complete example
ax2 = axes[0, 1]
ax2.axis('off')

example = '''
COMPLETE KERAS SAVE/LOAD EXAMPLE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 1. BUILD MODEL
def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# 2. TRAIN
model = create_model()
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)


# 3. SAVE (SavedModel format)
model.save('stock_predictor')
print("Model saved!")


# 4. SAVE (HDF5 format)
model.save('stock_predictor.h5')


# 5. LOAD AND USE
loaded_model = keras.models.load_model('stock_predictor')

# Verify it works
predictions = loaded_model.predict(X_test)
print(f"Predictions: {predictions[:5]}")


# 6. SAVE WITH CALLBACKS (during training)
checkpoint = keras.callbacks.ModelCheckpoint(
    'checkpoints/epoch_{epoch:02d}.h5',
    save_best_only=True,
    monitor='val_loss'
)

model.fit(X, y, epochs=50, callbacks=[checkpoint])
'''

ax2.text(0.02, 0.98, example, transform=ax2.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Format comparison
ax3 = axes[1, 0]

formats = ['SavedModel', 'HDF5', 'Weights\nOnly', 'ONNX']
file_size = [150, 45, 40, 35]
load_time = [500, 200, 100, 150]
portability = [3, 4, 2, 5]

x = np.arange(len(formats))
width = 0.25

bars1 = ax3.bar(x - width, [s/150 for s in file_size], width, label='Relative Size', color=MLBLUE, edgecolor='black')
bars2 = ax3.bar(x, [t/500 for t in load_time], width, label='Load Time', color=MLGREEN, edgecolor='black')
bars3 = ax3.bar(x + width, [p/5 for p in portability], width, label='Portability', color=MLORANGE, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(formats, fontsize=9)
ax3.set_ylabel('Normalized Score')
ax3.set_title('Format Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(alpha=0.3, axis='y')

# Add recommendations
recs = ['TF Deploy', 'Share', 'Checkpoint', 'Cross-platform']
for i, rec in enumerate(recs):
    ax3.text(i, 0.05, rec, fontsize=7, ha='center', style='italic')

# Plot 4: Checkpointing during training
ax4 = axes[1, 1]
ax4.axis('off')

checkpointing = '''
CHECKPOINTING DURING TRAINING

WHY CHECKPOINT?
---------------
- Training can crash
- Save best model automatically
- Resume training from checkpoint


BASIC CHECKPOINT:
-----------------
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_{epoch:02d}.h5',
    save_freq='epoch'  # Save every epoch
)

model.fit(X, y, epochs=100, callbacks=[checkpoint_callback])


SAVE BEST ONLY:
---------------
best_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    save_best_only=True,      # Only save if improved
    monitor='val_accuracy',   # What to monitor
    mode='max'                # max for accuracy, min for loss
)


EARLY STOPPING + CHECKPOINT:
----------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

callbacks = [best_checkpoint, early_stop]
model.fit(X, y, epochs=100, callbacks=callbacks)


RESUME TRAINING:
----------------
# Load checkpoint
model = keras.models.load_model('checkpoints/model_25.h5')

# Continue training
model.fit(X, y, initial_epoch=25, epochs=100)


CHECKPOINT FOLDER STRUCTURE:
----------------------------
checkpoints/
  model_01.h5
  model_02.h5
  ...
  model_25.h5    <- Resume from here
  best_model.h5  <- Best validation score
'''

ax4.text(0.02, 0.98, checkpointing, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Checkpointing During Training', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
