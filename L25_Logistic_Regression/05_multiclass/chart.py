"""Multiclass Logistic Regression - Beyond binary"""
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
fig.suptitle('Multiclass Classification with Logistic Regression', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Multiclass strategies
ax1 = axes[0, 0]
ax1.axis('off')

strategies = '''
MULTICLASS STRATEGIES

1. ONE-VS-REST (OvR) / One-vs-All
   ----------------------------------
   For K classes, train K binary classifiers

   Classifier 1: Class 0 vs (1, 2, 3)
   Classifier 2: Class 1 vs (0, 2, 3)
   Classifier 3: Class 2 vs (0, 1, 3)
   Classifier 4: Class 3 vs (0, 1, 2)

   Predict: Choose class with highest probability

   sklearn: multi_class='ovr'

2. MULTINOMIAL (Softmax)
   ----------------------------------
   Single model with K outputs
   Uses softmax function:

   P(y=k) = exp(z_k) / sum(exp(z_j))

   All probabilities sum to 1

   sklearn: multi_class='multinomial'

3. ONE-VS-ONE (OvO)
   ----------------------------------
   Train K*(K-1)/2 binary classifiers
   Each compares two classes directly

   Predict: Majority vote

   sklearn: Not default, but available
'''

ax1.text(0.02, 0.98, strategies, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Multiclass Strategies', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: 3-class visualization
ax2 = axes[0, 1]

# Generate 3-class data
n = 100
X0 = np.random.multivariate_normal([2, 6], [[0.8, 0], [0, 0.8]], n)
X1 = np.random.multivariate_normal([5, 2], [[0.8, 0], [0, 0.8]], n)
X2 = np.random.multivariate_normal([8, 6], [[0.8, 0], [0, 0.8]], n)

ax2.scatter(X0[:, 0], X0[:, 1], c=MLBLUE, s=40, alpha=0.6, label='Bearish', edgecolors='black')
ax2.scatter(X1[:, 0], X1[:, 1], c=MLGREEN, s=40, alpha=0.6, label='Neutral', edgecolors='black')
ax2.scatter(X2[:, 0], X2[:, 1], c=MLORANGE, s=40, alpha=0.6, label='Bullish', edgecolors='black')

# Draw approximate decision boundaries
x_line = np.linspace(0, 10, 100)
ax2.plot(x_line, -0.5*x_line + 6, color=MLRED, linewidth=2, linestyle='--', alpha=0.7)
ax2.plot(x_line, 2*x_line - 6, color=MLRED, linewidth=2, linestyle='--', alpha=0.7)

ax2.set_title('3-Class Problem: Market Regime', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1 (Momentum)', fontsize=10)
ax2.set_ylabel('Feature 2 (Volatility)', fontsize=10)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Softmax probabilities
ax3 = axes[1, 0]

samples = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5']
bearish_prob = [0.82, 0.15, 0.33, 0.05, 0.45]
neutral_prob = [0.10, 0.75, 0.34, 0.20, 0.35]
bullish_prob = [0.08, 0.10, 0.33, 0.75, 0.20]

x = np.arange(len(samples))
width = 0.25

bars1 = ax3.bar(x - width, bearish_prob, width, label='Bearish', color=MLBLUE, edgecolor='black', linewidth=0.5)
bars2 = ax3.bar(x, neutral_prob, width, label='Neutral', color=MLGREEN, edgecolor='black', linewidth=0.5)
bars3 = ax3.bar(x + width, bullish_prob, width, label='Bullish', color=MLORANGE, edgecolor='black', linewidth=0.5)

ax3.set_xticks(x)
ax3.set_xticklabels(samples)
ax3.set_title('predict_proba() for Multiclass', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Probability', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')

# Add predicted class
predictions = ['Bearish', 'Neutral', 'Neutral', 'Bullish', 'Bearish']
for i, pred in enumerate(predictions):
    ax3.text(i, 0.9, f'Pred: {pred}', ha='center', fontsize=8, fontweight='bold')

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Multiclass Logistic Regression in sklearn

from sklearn.linear_model import LogisticRegression

# Classes: 0=Bearish, 1=Neutral, 2=Bullish

# Method 1: One-vs-Rest (default for binary solvers)
model_ovr = LogisticRegression(
    multi_class='ovr',      # One-vs-Rest
    solver='liblinear'      # Works with OvR
)
model_ovr.fit(X_train, y_train)

# Method 2: Multinomial (Softmax)
model_multi = LogisticRegression(
    multi_class='multinomial',  # Softmax
    solver='lbfgs'              # Required for multinomial
)
model_multi.fit(X_train, y_train)

# Get probabilities (sums to 1 for each sample)
probs = model_multi.predict_proba(X_test)
print(probs[0])  # [0.15, 0.75, 0.10] -> Neutral

# Coefficients shape: (n_classes, n_features)
print(model_multi.coef_.shape)  # (3, 5) for 3 classes, 5 features

# Each row is coefficients for one class
for i, class_name in enumerate(['Bearish', 'Neutral', 'Bullish']):
    print(f"{class_name}: {model_multi.coef_[i]}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
