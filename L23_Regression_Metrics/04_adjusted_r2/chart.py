"""Adjusted R-squared - Penalizing complexity"""
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
fig.suptitle('Adjusted R-squared: Penalizing Model Complexity', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Problem with R-squared - always increases with features
ax1 = axes[0, 0]

n_features = np.arange(1, 11)
n_samples = 100

# R-squared always increases (or stays same) with more features
r2_values = 0.3 + 0.07 * n_features - 0.003 * n_features**2 + np.random.uniform(0, 0.02, len(n_features))
r2_values = np.cumsum(np.abs(np.diff(np.insert(r2_values, 0, 0.2)))) * 0.8

# Adjusted R-squared penalizes
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

adj_r2_values = [adjusted_r2(r2, n_samples, p) for r2, p in zip(r2_values, n_features)]

ax1.plot(n_features, r2_values, color=MLBLUE, linewidth=2.5, marker='o', markersize=8, label='R-squared')
ax1.plot(n_features, adj_r2_values, color=MLORANGE, linewidth=2.5, marker='s', markersize=8, label='Adjusted R-squared')

ax1.set_title('R-squared Always Increases with Features', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Number of Features', fontsize=10)
ax1.set_ylabel('Score', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Mark optimal
optimal_idx = np.argmax(adj_r2_values)
ax1.scatter([n_features[optimal_idx]], [adj_r2_values[optimal_idx]], c=MLGREEN, s=200,
            marker='*', zorder=5, edgecolors='black', label='Optimal features')
ax1.axvline(n_features[optimal_idx], color='gray', linestyle='--', linewidth=1)

# Plot 2: Adjusted R-squared formula
ax2 = axes[0, 1]
ax2.axis('off')

formula = r'''
ADJUSTED R-SQUARED

$$\bar{R}^2 = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}$$

Where:
- R-sq = regular R-squared
- n = number of observations
- p = number of predictors (features)

The penalty factor: $\frac{n-1}{n-p-1}$
- Increases as p increases
- Penalizes adding useless features
- Can decrease if new feature doesn't help

Example (n=100):
- p=1:  penalty = 99/98 = 1.01
- p=5:  penalty = 99/94 = 1.05
- p=10: penalty = 99/89 = 1.11

When to Use:
- Comparing models with different # features
- Feature selection (choose max Adj R-sq)
- Avoiding overfitting to noise
'''

ax2.text(0.02, 0.95, formula, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Adjusted R-squared Formula', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Example comparison
ax3 = axes[1, 0]

models = ['1 Feature\n(relevant)', '3 Features\n(+ 2 noise)', '5 Features\n(+ 4 noise)', '10 Features\n(+ 9 noise)']
r2 = [0.45, 0.52, 0.58, 0.65]
adj_r2 = [0.44, 0.50, 0.54, 0.58]  # Adjusted values

x = np.arange(len(models))
width = 0.35

bars1 = ax3.bar(x - width/2, r2, width, label='R-squared', color=MLBLUE, edgecolor='black', linewidth=0.5)
bars2 = ax3.bar(x + width/2, adj_r2, width, label='Adjusted R-squared', color=MLORANGE, edgecolor='black', linewidth=0.5)

ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=9)
ax3.set_title('Adding Noise Features: R-sq vs Adj R-sq', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Score', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')

# Add annotations showing gap
for i, (r, ar) in enumerate(zip(r2, adj_r2)):
    gap = r - ar
    ax3.annotate(f'-{gap:.2f}', xy=(i + width/2, ar), xytext=(i + width/2 + 0.15, ar - 0.03),
                fontsize=8, color=MLRED)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Calculating Adjusted R-squared

# sklearn doesn't have built-in Adj R-sq
# Easy to calculate manually:

from sklearn.metrics import r2_score

def adjusted_r2(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2

# Example usage:
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2(y_test, y_pred, X_test.shape[1])

print(f"R-squared:     {r2:.4f}")
print(f"Adj R-squared: {adj_r2:.4f}")

# Model comparison
for name, n_feat in [('Model A', 5), ('Model B', 20)]:
    adj = adjusted_r2(y_test, y_pred, n_feat)
    print(f"{name} ({n_feat} features): {adj:.4f}")
'''

ax4.text(0.02, 0.95, code, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Code Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
