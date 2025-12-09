"""Regularization in Logistic Regression"""
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
fig.suptitle('Regularization in Logistic Regression', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: C parameter effect on coefficients
ax1 = axes[0, 0]

# Simulated coefficient paths as C increases
C_values = np.logspace(-3, 2, 50)
n_features = 5
feature_names = ['Momentum', 'Volume', 'Volatility', 'RSI', 'Sentiment']

# Simulate coefficients converging as C increases
coef_paths = []
for i in range(n_features):
    final_coef = np.random.uniform(-1, 1)
    path = final_coef * (1 - np.exp(-C_values * 0.5))
    coef_paths.append(path)

colors = [MLBLUE, MLORANGE, MLGREEN, MLRED, MLPURPLE]
for path, name, color in zip(coef_paths, feature_names, colors):
    ax1.plot(C_values, path, linewidth=2, label=name, color=color)

ax1.axhline(0, color='gray', linewidth=1, linestyle='--')
ax1.set_xscale('log')
ax1.set_title('Coefficient Values vs C (Inverse Regularization)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('C (higher = less regularization)', fontsize=10)
ax1.set_ylabel('Coefficient Value', fontsize=10)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(alpha=0.3)

ax1.annotate('Strong regularization\n(coefficients shrunk)', xy=(0.002, 0), fontsize=8, ha='center')
ax1.annotate('Weak regularization\n(full coefficients)', xy=(50, 0.5), fontsize=8, ha='center')

# Plot 2: L1 vs L2 regularization
ax2 = axes[0, 1]
ax2.axis('off')

comparison = '''
L1 vs L2 REGULARIZATION

L2 (Ridge) - penalty='l2'
---------------------------
Loss = BCE + lambda * sum(beta^2)

- Shrinks all coefficients
- No coefficient becomes exactly 0
- Handles multicollinearity
- Default in sklearn

L1 (Lasso) - penalty='l1'
---------------------------
Loss = BCE + lambda * sum(|beta|)

- Some coefficients become exactly 0
- Automatic feature selection!
- Sparse models
- Needs solver='liblinear' or 'saga'

ElasticNet - penalty='elasticnet'
----------------------------------
Loss = BCE + lambda * (r*L1 + (1-r)*L2)

- Mix of L1 and L2
- l1_ratio parameter (0 to 1)
- Needs solver='saga'

SKLEARN C PARAMETER:
--------------------
C = 1/lambda

C=0.01 -> Strong regularization
C=1    -> Moderate regularization
C=100  -> Weak regularization
'''

ax2.text(0.02, 0.98, comparison, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('L1 vs L2 Regularization', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Train vs Test accuracy vs C
ax3 = axes[1, 0]

C_values = np.logspace(-3, 2, 20)
# Simulated accuracies
train_acc = 0.70 + 0.15 * (1 - np.exp(-C_values * 0.3)) + np.random.uniform(-0.01, 0.01, 20)
test_acc = 0.68 + 0.10 * (1 - np.exp(-C_values * 0.2)) - 0.05 * (C_values > 10).astype(float) * (C_values - 10) / 90

ax3.plot(C_values, train_acc, color=MLBLUE, linewidth=2.5, marker='o', markersize=4, label='Train Accuracy')
ax3.plot(C_values, test_acc, color=MLORANGE, linewidth=2.5, marker='s', markersize=4, label='Test Accuracy')

# Mark optimal
optimal_idx = np.argmax(test_acc)
ax3.scatter([C_values[optimal_idx]], [test_acc[optimal_idx]], c=MLGREEN, s=200,
            marker='*', zorder=5, edgecolors='black', label=f'Optimal C={C_values[optimal_idx]:.2f}')
ax3.axvline(C_values[optimal_idx], color='gray', linestyle='--', alpha=0.5)

ax3.set_xscale('log')
ax3.set_title('Accuracy vs Regularization Strength', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('C (inverse regularization)', fontsize=10)
ax3.set_ylabel('Accuracy', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Add zones
ax3.text(0.003, 0.72, 'Underfitting', fontsize=9, color=MLRED)
ax3.text(30, 0.72, 'Overfitting', fontsize=9, color=MLRED)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Regularization in sklearn LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# L2 (Ridge) Regularization
model_l2 = LogisticRegression(
    penalty='l2',
    C=1.0,          # Try: 0.01, 0.1, 1, 10, 100
    solver='lbfgs'
)

# L1 (Lasso) Regularization
model_l1 = LogisticRegression(
    penalty='l1',
    C=1.0,
    solver='liblinear'  # Required for L1
)

# ElasticNet Regularization
model_elastic = LogisticRegression(
    penalty='elasticnet',
    C=1.0,
    l1_ratio=0.5,   # Mix of L1/L2 (0=L2, 1=L1)
    solver='saga'   # Required for ElasticNet
)

# Find optimal C with cross-validation
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(
    LogisticRegression(penalty='l2'),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"Best C: {grid_search.best_params_['C']}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
