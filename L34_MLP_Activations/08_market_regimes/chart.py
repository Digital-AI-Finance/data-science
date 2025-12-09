"""Finance Application - Market Regime Detection with MLPs"""
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
fig.suptitle('Finance Application: Market Regime Detection with MLP', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Problem description
ax1 = axes[0, 0]
ax1.axis('off')

problem = '''
MARKET REGIME DETECTION

GOAL:
-----
Classify the current market into one of
several "regimes" based on features.


REGIMES:
--------
0: Low volatility, uptrend (Bull)
1: High volatility, downtrend (Bear)
2: High volatility, sideways (Crisis)
3: Low volatility, sideways (Calm)


FEATURES:
---------
- Realized volatility (20-day)
- Return momentum (5, 20, 60 day)
- VIX level
- Yield curve slope
- Credit spread


WHY MLP?
--------
- Non-linear regime boundaries
- Can capture complex interactions
- Flexible multi-class output


ARCHITECTURE:
-------------
Input (8 features)
-> Dense(32, ReLU)
-> Dense(16, ReLU)
-> Dense(4, Softmax)

Output: Probabilities for each regime
'''

ax1.text(0.02, 0.98, problem, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Problem Setup', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visualize regime data
ax2 = axes[0, 1]

# Generate synthetic regime data
n_per_regime = 50

# Bull: low vol, positive return
bull_vol = np.random.randn(n_per_regime) * 3 + 12
bull_ret = np.random.randn(n_per_regime) * 2 + 8

# Bear: high vol, negative return
bear_vol = np.random.randn(n_per_regime) * 4 + 28
bear_ret = np.random.randn(n_per_regime) * 3 - 10

# Crisis: very high vol, mixed return
crisis_vol = np.random.randn(n_per_regime) * 5 + 40
crisis_ret = np.random.randn(n_per_regime) * 8

# Calm: low vol, near zero return
calm_vol = np.random.randn(n_per_regime) * 2 + 10
calm_ret = np.random.randn(n_per_regime) * 1

ax2.scatter(bull_vol, bull_ret, c=MLGREEN, s=40, alpha=0.6, label='Bull', edgecolors='black', linewidths=0.3)
ax2.scatter(bear_vol, bear_ret, c=MLRED, s=40, alpha=0.6, label='Bear', edgecolors='black', linewidths=0.3)
ax2.scatter(crisis_vol, crisis_ret, c=MLORANGE, s=40, alpha=0.6, label='Crisis', edgecolors='black', linewidths=0.3)
ax2.scatter(calm_vol, calm_ret, c=MLBLUE, s=40, alpha=0.6, label='Calm', edgecolors='black', linewidths=0.3)

ax2.set_title('Market Regimes: Volatility vs Return', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('20-Day Volatility (%)')
ax2.set_ylabel('20-Day Return (%)')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax2.axhline(0, color='black', linewidth=0.5)

# Plot 3: Model output example
ax3 = axes[1, 0]

# Simulated regime probabilities over time
days = np.arange(100)
probs = np.random.dirichlet([2, 0.5, 0.3, 1], size=100)

# Smooth probabilities
for i in range(1, len(probs)):
    probs[i] = 0.7 * probs[i] + 0.3 * probs[i-1]

regime_names = ['Bull', 'Bear', 'Crisis', 'Calm']
colors = [MLGREEN, MLRED, MLORANGE, MLBLUE]

ax3.stackplot(days, probs.T, labels=regime_names, colors=colors, alpha=0.8)

ax3.set_title('MLP Regime Probabilities Over Time', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Day')
ax3.set_ylabel('Probability')
ax3.legend(loc='upper right', fontsize=8)
ax3.set_ylim(0, 1)

# Plot 4: Complete code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
COMPLETE MLP REGIME DETECTION

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. PREPARE FEATURES
features = ['vol_20', 'ret_5', 'ret_20', 'ret_60',
            'vix', 'yield_slope', 'credit_spread', 'momentum']
X = df[features].values
y = df['regime'].values  # 0, 1, 2, 3

# 2. PREPROCESS
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode targets
y_train_oh = keras.utils.to_categorical(y_train, 4)
y_test_oh = keras.utils.to_categorical(y_test, 4)

# 3. BUILD MODEL
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(8,)),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. TRAIN
history = model.fit(X_train, y_train_oh, epochs=100,
                    validation_split=0.2, verbose=0)

# 5. PREDICT REGIME PROBABILITIES
probs = model.predict(X_test)
predicted_regime = np.argmax(probs, axis=1)
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
