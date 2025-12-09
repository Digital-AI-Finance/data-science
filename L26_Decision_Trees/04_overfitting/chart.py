"""Overfitting in Decision Trees"""
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
fig.suptitle('Overfitting in Decision Trees', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate sample data
np.random.seed(42)
n = 150
X = np.random.uniform(0, 10, (n, 2))
y = ((X[:, 0] + X[:, 1]) > 10).astype(int)
# Add some noise
noise_idx = np.random.choice(n, 15, replace=False)
y[noise_idx] = 1 - y[noise_idx]

# Plot 1: Shallow tree (underfitting)
ax1 = axes[0, 0]

# Create meshgrid
xx, yy = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))

# Shallow decision boundary (depth=1)
# Simple: x1 > 5
boundary = (xx > 5).astype(float)
ax1.contourf(xx, yy, boundary, levels=[0, 0.5, 1], colors=[MLBLUE, MLORANGE], alpha=0.3)

ax1.scatter(X[y==0, 0], X[y==0, 1], c=MLBLUE, s=40, alpha=0.7, label='Class 0', edgecolors='black')
ax1.scatter(X[y==1, 0], X[y==1, 1], c=MLORANGE, s=40, alpha=0.7, label='Class 1', edgecolors='black')

ax1.set_title('max_depth=1 (Underfitting)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Feature 1', fontsize=10)
ax1.set_ylabel('Feature 2', fontsize=10)
ax1.legend(fontsize=9)

ax1.text(0.05, 0.95, 'Train: 65%\nTest: 64%', transform=ax1.transAxes, fontsize=10,
         va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: Good tree (balanced)
ax2 = axes[0, 1]

# Moderate depth boundary approximating diagonal
boundary = ((xx + yy) > 10).astype(float)
ax2.contourf(xx, yy, boundary, levels=[0, 0.5, 1], colors=[MLBLUE, MLORANGE], alpha=0.3)

ax2.scatter(X[y==0, 0], X[y==0, 1], c=MLBLUE, s=40, alpha=0.7, label='Class 0', edgecolors='black')
ax2.scatter(X[y==1, 0], X[y==1, 1], c=MLORANGE, s=40, alpha=0.7, label='Class 1', edgecolors='black')

ax2.set_title('max_depth=4 (Good Fit)', fontsize=11, fontweight='bold', color=MLGREEN)
ax2.set_xlabel('Feature 1', fontsize=10)
ax2.set_ylabel('Feature 2', fontsize=10)
ax2.legend(fontsize=9)

ax2.text(0.05, 0.95, 'Train: 92%\nTest: 88%', transform=ax2.transAxes, fontsize=10,
         va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 3: Deep tree (overfitting)
ax3 = axes[1, 0]

# Complex boundary with noise fit
# Simulate overfitting by adding irregular regions around noise points
boundary = ((xx + yy) > 10).astype(float)
# Add some irregular "islands"
for i in range(5):
    cx, cy = np.random.uniform(2, 8, 2)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    boundary[dist < 0.5] = 1 - boundary[dist < 0.5].mean()

ax3.contourf(xx, yy, boundary, levels=[0, 0.5, 1], colors=[MLBLUE, MLORANGE], alpha=0.3)

ax3.scatter(X[y==0, 0], X[y==0, 1], c=MLBLUE, s=40, alpha=0.7, label='Class 0', edgecolors='black')
ax3.scatter(X[y==1, 0], X[y==1, 1], c=MLORANGE, s=40, alpha=0.7, label='Class 1', edgecolors='black')

ax3.set_title('max_depth=None (Overfitting)', fontsize=11, fontweight='bold', color=MLRED)
ax3.set_xlabel('Feature 1', fontsize=10)
ax3.set_ylabel('Feature 2', fontsize=10)
ax3.legend(fontsize=9)

ax3.text(0.05, 0.95, 'Train: 100%\nTest: 72%', transform=ax3.transAxes, fontsize=10,
         va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 4: How to prevent overfitting
ax4 = axes[1, 1]
ax4.axis('off')

prevention = '''
PREVENTING OVERFITTING IN DECISION TREES

1. LIMIT TREE DEPTH (max_depth)
   - Most effective method
   - Start with max_depth=3-5
   - Use cross-validation to tune

2. MINIMUM SAMPLES TO SPLIT (min_samples_split)
   - Requires at least N samples to make a split
   - Default: 2 (will split any node)
   - Try: 5, 10, 20, 50

3. MINIMUM SAMPLES IN LEAF (min_samples_leaf)
   - Every leaf must have at least N samples
   - Prevents single-sample leaves
   - Try: 1, 5, 10, 20

4. MAXIMUM LEAF NODES (max_leaf_nodes)
   - Limits total number of leaves
   - Alternative to max_depth

5. PRUNING (built into sklearn)
   - ccp_alpha: Cost-complexity pruning
   - Higher value = more pruning
   - Use cross-validation to find optimal

6. USE RANDOM FOREST
   - Ensemble of trees with bagging
   - Much more robust to overfitting
   - Best practice for production

RULE OF THUMB:
--------------
Single tree: Good for interpretability
Random Forest: Good for accuracy
'''

ax4.text(0.02, 0.98, prevention, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Preventing Overfitting', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
