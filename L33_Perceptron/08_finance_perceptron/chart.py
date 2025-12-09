"""Finance Application - Perceptron for Market Direction"""
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
fig.suptitle('Finance Application: Perceptron for Market Direction', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Problem setup
ax1 = axes[0, 0]
ax1.axis('off')

setup = '''
MARKET DIRECTION PREDICTION

GOAL:
-----
Predict whether tomorrow's market
will be UP (1) or DOWN (0)


FEATURES (Technical Indicators):
--------------------------------
x1: 5-day moving average momentum
x2: RSI (Relative Strength Index)
x3: MACD signal
x4: Volume ratio
x5: Volatility


TARGET:
-------
y = 1 if tomorrow's return > 0 (UP)
y = 0 if tomorrow's return <= 0 (DOWN)


WHY PERCEPTRON?
---------------
- Simple baseline model
- Interpretable weights
- Fast training
- Good for understanding features


LIMITATIONS:
------------
- Markets may not be linearly separable
- Need more complex models in practice
- This is for learning, not trading advice!
'''

ax1.text(0.02, 0.98, setup, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Problem Setup', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Simulated data visualization
ax2 = axes[0, 1]

# Generate synthetic financial features
n_up = 100
n_down = 100

# UP days: higher momentum, mid-range RSI
momentum_up = np.random.randn(n_up) * 0.3 + 0.5
rsi_up = np.random.randn(n_up) * 10 + 55

# DOWN days: lower momentum, extreme RSI
momentum_down = np.random.randn(n_down) * 0.3 - 0.3
rsi_down = np.random.randn(n_down) * 10 + 45

ax2.scatter(momentum_down, rsi_down, c=MLRED, s=30, alpha=0.6, label='DOWN days', edgecolors='black', linewidths=0.3)
ax2.scatter(momentum_up, rsi_up, c=MLGREEN, s=30, alpha=0.6, label='UP days', edgecolors='black', linewidths=0.3)

# Decision boundary (simulated learned)
x_line = np.linspace(-1, 1.5, 100)
y_line = 50 + 10 * x_line
ax2.plot(x_line, y_line, color=MLPURPLE, linewidth=2, linestyle='--', label='Perceptron boundary')

ax2.set_title('Market Direction Data', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('5-Day Momentum')
ax2.set_ylabel('RSI')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Results and interpretation
ax3 = axes[1, 0]
ax3.axis('off')

results = '''
SAMPLE RESULTS

LEARNED WEIGHTS:
----------------
Feature          |  Weight  | Interpretation
-----------------|----------|----------------
5-day momentum   |   0.42   | Positive momentum -> UP
RSI              |   0.15   | Higher RSI -> UP
MACD signal      |   0.28   | Positive MACD -> UP
Volume ratio     |  -0.08   | High volume -> DOWN?
Volatility       |  -0.22   | High vol -> DOWN
Bias             |  -0.35   |


INTERPRETATION:
---------------
Positive weights: Feature predicts UP
Negative weights: Feature predicts DOWN

Magnitude: How important the feature is


PERFORMANCE (Simulated):
------------------------
Training Accuracy: 58%
Test Accuracy:     53%

Better than random (50%), but not great!


KEY INSIGHT:
------------
Markets are HARD to predict.
Even simple patterns are weak.
But perceptron gives us a baseline
and interpretable weights.
'''

ax3.text(0.02, 0.98, results, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Results & Interpretation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
COMPLETE FINANCE EXAMPLE

import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. LOAD DATA
df = pd.read_csv('stock_data.csv')

# 2. CREATE FEATURES
df['momentum'] = df['close'].pct_change(5)
df['rsi'] = compute_rsi(df['close'], 14)
df['macd'] = compute_macd(df['close'])
df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
df['volatility'] = df['close'].pct_change().rolling(10).std()

# 3. CREATE TARGET
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna()

# 4. PREPARE DATA
features = ['momentum', 'rsi', 'macd', 'vol_ratio', 'volatility']
X = df[features].values
y = df['target'].values

# 5. SPLIT (respecting time order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 6. SCALE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. TRAIN PERCEPTRON
model = Perceptron(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 8. EVALUATE
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train: {train_acc:.2%}, Test: {test_acc:.2%}")

# 9. INTERPRET
for feat, weight in zip(features, model.coef_[0]):
    print(f"{feat}: {weight:.3f}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
