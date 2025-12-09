"""Keras Sequential Model - Building MLPs"""
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
fig.suptitle('Building MLPs with Keras Sequential API', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic Sequential model
ax1 = axes[0, 0]
ax1.axis('off')

basic_code = '''
KERAS SEQUENTIAL API

from tensorflow import keras
from tensorflow.keras import layers

# METHOD 1: List of layers
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# METHOD 2: Add layers one by one
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# View model summary
model.summary()


# Output:
# Layer (type)          Output Shape      Param #
# ================================================
# dense (Dense)         (None, 64)        704
# dense_1 (Dense)       (None, 32)        2080
# dense_2 (Dense)       (None, 1)         33
# ================================================
# Total params: 2,817
'''

ax1.text(0.02, 0.98, basic_code, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Basic Sequential Model', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Compile and fit
ax2 = axes[0, 1]
ax2.axis('off')

compile_fit = '''
COMPILE AND TRAIN

# COMPILE: Define loss, optimizer, metrics
model.compile(
    optimizer='adam',              # Optimizer
    loss='binary_crossentropy',    # Loss function
    metrics=['accuracy']           # Metrics to track
)


# FIT: Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,                     # Training iterations
    batch_size=32,                 # Samples per batch
    validation_split=0.2,          # Validation data
    verbose=1                      # Print progress
)


# EVALUATE: Test performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")


# PREDICT: Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)


# ACCESS HISTORY
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.show()
'''

ax2.text(0.02, 0.98, compile_fit, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Compile and Train', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Common configurations
ax3 = axes[1, 0]
ax3.axis('off')

configs = '''
COMMON CONFIGURATIONS

BINARY CLASSIFICATION:
----------------------
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


MULTI-CLASS CLASSIFICATION:
---------------------------
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


REGRESSION:
-----------
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # No activation = linear
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


KEY POINTS:
-----------
- Hidden layers: ReLU activation
- Output: sigmoid (binary), softmax (multi-class), linear (regression)
- Loss must match output activation!
'''

ax3.text(0.02, 0.98, configs, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Common Configurations', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete example
ax4 = axes[1, 1]
ax4.axis('off')

example = '''
COMPLETE EXAMPLE

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


# 2. BUILD MODEL
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# 3. COMPILE
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# 4. TRAIN
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)


# 5. EVALUATE
print(f"Test: {model.evaluate(X_test, y_test, verbose=0)[1]:.2%}")
'''

ax4.text(0.02, 0.98, example, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
