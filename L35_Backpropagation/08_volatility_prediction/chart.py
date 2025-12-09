"""Finance Application - Volatility Prediction with Neural Networks"""
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
fig.suptitle('Finance Application: Volatility Prediction with MLP', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Problem description
ax1 = axes[0, 0]
ax1.axis('off')

problem = '''
VOLATILITY PREDICTION

GOAL:
-----
Predict tomorrow's volatility (standard deviation
of returns) using today's market features.


WHY IT MATTERS:
---------------
- Risk management
- Option pricing
- Portfolio allocation
- VaR calculation


FEATURES (INPUT):
-----------------
- Past realized volatility (5, 20, 60 day)
- Daily return
- Volume
- VIX level
- Momentum indicators
- Day of week


TARGET (OUTPUT):
----------------
Next-day realized volatility (or squared return)


MODEL ARCHITECTURE:
-------------------
Input (10 features)
-> Dense(64, ReLU)
-> Dropout(0.3)
-> Dense(32, ReLU)
-> Dropout(0.2)
-> Dense(1, Linear)  # Positive output

Loss: MSE (regression task)
'''

ax1.text(0.02, 0.98, problem, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Problem Setup', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Training curves
ax2 = axes[0, 1]

epochs = np.arange(150)

# Simulated training for volatility prediction
train_loss = 0.002 * np.exp(-0.02 * epochs) + 0.0003 + np.random.randn(150) * 0.00005
val_loss = 0.0025 * np.exp(-0.015 * epochs) + 0.0005 + np.random.randn(150) * 0.00008

ax2.plot(epochs, train_loss * 1000, color=MLBLUE, linewidth=2, label='Training MSE')
ax2.plot(epochs, val_loss * 1000, color=MLORANGE, linewidth=2, label='Validation MSE')

# Early stopping point
best_epoch = np.argmin(val_loss)
ax2.axvline(best_epoch, color=MLGREEN, linestyle='--', linewidth=2, label=f'Best: epoch {best_epoch}')
ax2.scatter([best_epoch], [val_loss[best_epoch] * 1000], color=MLGREEN, s=100, zorder=5)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss (x1000)')
ax2.set_title('Training History', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Prediction vs actual
ax3 = axes[1, 0]

# Generate synthetic volatility data
days = 100
actual_vol = np.abs(0.02 + 0.01 * np.sin(np.linspace(0, 4*np.pi, days)) + np.random.randn(days) * 0.005)

# MLP predictions (slightly smoothed)
predicted_vol = actual_vol * (1 + np.random.randn(days) * 0.15)
predicted_vol = np.convolve(predicted_vol, np.ones(3)/3, mode='same')

# GARCH baseline (simpler model)
garch_vol = np.abs(actual_vol + np.random.randn(days) * 0.008)

ax3.plot(range(days), actual_vol * 100, color='black', linewidth=2, label='Actual Volatility')
ax3.plot(range(days), predicted_vol * 100, color=MLBLUE, linewidth=1.5, alpha=0.8, label='MLP Prediction')
ax3.plot(range(days), garch_vol * 100, color=MLRED, linewidth=1.5, alpha=0.6, linestyle='--', label='GARCH Baseline')

ax3.set_xlabel('Day')
ax3.set_ylabel('Daily Volatility (%)')
ax3.set_title('Out-of-Sample Predictions', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Calculate R2
ss_res_mlp = np.sum((actual_vol - predicted_vol)**2)
ss_res_garch = np.sum((actual_vol - garch_vol)**2)
ss_tot = np.sum((actual_vol - np.mean(actual_vol))**2)
r2_mlp = 1 - ss_res_mlp / ss_tot
r2_garch = 1 - ss_res_garch / ss_tot

ax3.text(10, 3.5, f'MLP R2: {r2_mlp:.3f}\nGARCH R2: {r2_garch:.3f}',
         fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

# Plot 4: Complete code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
COMPLETE VOLATILITY PREDICTION CODE

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. PREPARE FEATURES
features = ['vol_5d', 'vol_20d', 'vol_60d', 'return_1d',
            'volume', 'vix', 'momentum_20d', 'day_of_week']
target = 'vol_next_day'

X = df[features].values
y = df[target].values

# 2. SPLIT AND SCALE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # Time series!
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. BUILD MODEL
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(8,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)  # Linear for regression
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 4. TRAIN WITH EARLY STOPPING
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# 5. EVALUATE
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# 6. PREDICT
predictions = model.predict(X_test)
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=6.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
