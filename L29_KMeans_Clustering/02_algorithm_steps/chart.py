"""K-Means Algorithm Steps - Iterative Process"""
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

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('K-Means Algorithm: Step by Step', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
n_per_cluster = 40
cluster1 = np.random.randn(n_per_cluster, 2) * 0.7 + [2, 2]
cluster2 = np.random.randn(n_per_cluster, 2) * 0.7 + [-2, 2]
cluster3 = np.random.randn(n_per_cluster, 2) * 0.7 + [0, -2]
X = np.vstack([cluster1, cluster2, cluster3])

colors = [MLBLUE, MLRED, MLGREEN]

# Step 1: Initial data
ax1 = axes[0, 0]
ax1.scatter(X[:, 0], X[:, 1], c='gray', s=40, alpha=0.6)
ax1.set_title('Step 1: Raw Data', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, 'No labels yet', transform=ax1.transAxes, fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

# Step 2: Random centroid initialization
ax2 = axes[0, 1]
ax2.scatter(X[:, 0], X[:, 1], c='gray', s=40, alpha=0.6)

# Random initial centroids (poor initialization)
init_centroids = np.array([[0, 0], [1, 1], [-1, -1]])
for i, c in enumerate(init_centroids):
    ax2.scatter(c[0], c[1], c=colors[i], s=200, marker='X', edgecolors='black', linewidths=2, zorder=5)

ax2.set_title('Step 2: Initialize K Centroids', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.grid(alpha=0.3)
ax2.text(0.05, 0.95, 'K=3 random centroids', transform=ax2.transAxes, fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

# Step 3: Assign points to nearest centroid
ax3 = axes[0, 2]

# Compute assignments based on initial centroids
distances = np.zeros((len(X), 3))
for i, c in enumerate(init_centroids):
    distances[:, i] = np.sqrt(np.sum((X - c)**2, axis=1))
labels = np.argmin(distances, axis=1)

for i in range(3):
    mask = labels == i
    ax3.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=40, alpha=0.6)
    ax3.scatter(init_centroids[i, 0], init_centroids[i, 1], c=colors[i], s=200,
                marker='X', edgecolors='black', linewidths=2, zorder=5)

ax3.set_title('Step 3: Assign to Nearest Centroid', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.grid(alpha=0.3)
ax3.text(0.05, 0.95, 'Color = cluster assignment', transform=ax3.transAxes, fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

# Step 4: Recompute centroids
ax4 = axes[1, 0]

# New centroids
new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(3)])

for i in range(3):
    mask = labels == i
    ax4.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=40, alpha=0.6)
    # Old centroid (faded)
    ax4.scatter(init_centroids[i, 0], init_centroids[i, 1], c=colors[i], s=150,
                marker='X', alpha=0.3, zorder=4)
    # New centroid
    ax4.scatter(new_centroids[i, 0], new_centroids[i, 1], c=colors[i], s=200,
                marker='X', edgecolors='black', linewidths=2, zorder=5)
    # Arrow showing movement
    ax4.annotate('', xy=new_centroids[i], xytext=init_centroids[i],
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax4.set_title('Step 4: Recompute Centroids', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Feature 1')
ax4.set_ylabel('Feature 2')
ax4.grid(alpha=0.3)
ax4.text(0.05, 0.95, 'Centroids move to cluster mean', transform=ax4.transAxes, fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

# Step 5: Repeat until convergence
ax5 = axes[1, 1]

# Final state (true clusters)
final_centroids = np.array([[2, 2], [-2, 2], [0, -2]])
distances_final = np.zeros((len(X), 3))
for i, c in enumerate(final_centroids):
    distances_final[:, i] = np.sqrt(np.sum((X - c)**2, axis=1))
labels_final = np.argmin(distances_final, axis=1)

for i in range(3):
    mask = labels_final == i
    ax5.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=40, alpha=0.6)
    ax5.scatter(final_centroids[i, 0], final_centroids[i, 1], c=colors[i], s=200,
                marker='X', edgecolors='black', linewidths=2, zorder=5)

ax5.set_title('Step 5: Converged!', fontsize=11, fontweight='bold', color=MLGREEN)
ax5.set_xlabel('Feature 1')
ax5.set_ylabel('Feature 2')
ax5.grid(alpha=0.3)
ax5.text(0.05, 0.95, 'Centroids no longer move', transform=ax5.transAxes, fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

# Step 6: Algorithm summary
ax6 = axes[1, 2]
ax6.axis('off')

summary = '''
K-MEANS ALGORITHM

1. INITIALIZE
   Choose K random centroids

2. ASSIGN
   Assign each point to nearest
   centroid (Euclidean distance)

3. UPDATE
   Recompute centroids as mean
   of assigned points

4. REPEAT
   Steps 2-3 until convergence
   (centroids stop moving)


CONVERGENCE:
------------
- Guaranteed to converge
- But may find local minimum!
- Run multiple times with
  different initializations


COMPLEXITY:
-----------
O(n * k * i * d)
- n = number of points
- k = number of clusters
- i = iterations
- d = dimensions
'''

ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax6.set_title('Algorithm Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
