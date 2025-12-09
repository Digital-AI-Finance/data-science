"""Finance Application - Regularization for Stock Prediction"""
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
fig.suptitle('Finance Application: Preventing Overfitting in Stock Prediction', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Problem description
ax1 = axes[0, 0]
ax1.axis('off')

problem = '''
THE OVERFITTING PROBLEM IN FINANCE

WHY FINANCE IS PRONE TO OVERFITTING:
------------------------------------
1. Low signal-to-noise ratio
   - Stock returns are very noisy
   - True patterns are subtle

2. Non-stationary data
   - Market regimes change
   - Past patterns may not repeat

3. Multiple testing
   - Many features available
   - Temptation to try many models

4. Limited data
   - Historical data is finite
   - Rare events (crashes) underrepresented


CONSEQUENCES:
-------------
- Strategy looks great in backtest
- Fails in live trading
- Massive losses possible


SOLUTIONS WE'LL USE:
--------------------
1. Dropout (0.3-0.5)
2. Early stopping (patience=20)
3. L2 regularization (lambda=0.001)
4. Proper train/val/test splits
5. Walk-forward validation


CRITICAL RULE:
--------------
Never evaluate on data from the past!
Always use future (unseen) data for testing.
'''

ax1.text(0.02, 0.98, problem, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('The Problem', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Comparison of regularized vs unregularized
ax2 = axes[0, 1]

epochs = np.arange(200)

# Unregularized model (overfits badly)
train_unreg = 0.03 * np.exp(-0.02 * epochs) + 0.002 + np.random.randn(200) * 0.001
val_unreg = np.concatenate([
    0.035 * np.exp(-0.015 * epochs[:50]) + 0.008 + np.random.randn(50) * 0.002,
    0.012 + 0.0002 * (epochs[50:] - 50) + np.random.randn(150) * 0.002
])

# Regularized model (generalizes better)
train_reg = 0.028 * np.exp(-0.015 * epochs) + 0.006 + np.random.randn(200) * 0.001
val_reg = 0.032 * np.exp(-0.012 * epochs) + 0.008 + np.random.randn(200) * 0.0015

ax2.plot(epochs, train_unreg, color=MLBLUE, linewidth=1.5, alpha=0.6, label='No Reg Train')
ax2.plot(epochs, val_unreg, color=MLRED, linewidth=2, linestyle='--', label='No Reg Val')
ax2.plot(epochs, train_reg, color=MLGREEN, linewidth=1.5, alpha=0.6, label='Regularized Train')
ax2.plot(epochs, val_reg, color=MLGREEN, linewidth=2, linestyle='--', label='Regularized Val')

# Early stopping point for regularized
best_epoch = 80
ax2.axvline(best_epoch, color='gray', linestyle=':', linewidth=2, label=f'Early Stop @ {best_epoch}')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Regularized vs Unregularized', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(alpha=0.3)

# Plot 3: Out-of-sample performance comparison
ax3 = axes[1, 0]

# Simulated backtest vs live performance
models = ['No\nRegularization', 'Dropout\nOnly', 'L2\nOnly', 'Dropout +\nEarly Stop', 'Full\nRegularization']
backtest_sharpe = [2.5, 1.8, 1.9, 1.4, 1.2]
live_sharpe = [0.3, 0.6, 0.5, 0.9, 1.0]

x = np.arange(len(models))
width = 0.35

bars1 = ax3.bar(x - width/2, backtest_sharpe, width, label='Backtest', color=MLBLUE, edgecolor='black')
bars2 = ax3.bar(x + width/2, live_sharpe, width, label='Live Trading', color=MLORANGE, edgecolor='black')

ax3.set_ylabel('Sharpe Ratio')
ax3.set_title('Backtest vs Live Performance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=8)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')

# Add percentage drop annotations
for i, (bt, live) in enumerate(zip(backtest_sharpe, live_sharpe)):
    drop = (bt - live) / bt * 100
    ax3.text(i, max(bt, live) + 0.1, f'-{drop:.0f}%', ha='center', fontsize=8, color=MLRED)

ax3.text(4, 2.2, 'Lower backtest\n= More realistic', fontsize=9, ha='center',
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 4: Complete code example
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
COMPLETE REGULARIZED STOCK PREDICTION MODEL

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# 1. PREPARE DATA (walk-forward)
train_end = '2020-12-31'
val_end = '2021-12-31'
test_end = '2022-12-31'

X_train = df[df.index <= train_end][features]
X_val = df[(df.index > train_end) & (df.index <= val_end)][features]
X_test = df[(df.index > val_end) & (df.index <= test_end)][features]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# 2. BUILD REGULARIZED MODEL
model = keras.Sequential([
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(n_features,)),
    layers.Dropout(0.4),
    layers.Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)  # Predict return
])


# 3. COMPILE
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)


# 4. TRAIN WITH CALLBACKS
callbacks = [
    EarlyStopping(monitor='val_loss', patience=30,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=15, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=0
)


# 5. EVALUATE ON TEST (truly out-of-sample)
test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Out-of-sample MSE: {test_mse:.6f}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=6.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
