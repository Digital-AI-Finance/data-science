"""K-Means Clustering - Core Concept"""
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
fig.suptitle('K-Means Clustering: Core Concept', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is clustering?
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT IS K-MEANS CLUSTERING?

GOAL: Partition n observations into k clusters
      where each observation belongs to the
      cluster with the nearest centroid (mean).


UNSUPERVISED LEARNING:
----------------------
- No labels needed!
- Algorithm discovers structure in data
- Groups similar observations together


KEY CHARACTERISTICS:
-------------------
- K = number of clusters (you choose)
- Centroid = center of each cluster
- Assignment based on distance
- Iterative optimization


CLUSTERING VS CLASSIFICATION:
-----------------------------
Classification: Labels are known (supervised)
               "Train model to predict fraud/not fraud"

Clustering: Labels are unknown (unsupervised)
           "Find natural groupings in data"


FINANCE APPLICATIONS:
--------------------
- Customer segmentation
- Asset grouping by behavior
- Market regime detection
- Portfolio construction
- Risk profiling
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('K-Means Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual example - unclustered vs clustered
ax2 = axes[0, 1]

# Generate clustered data
n_per_cluster = 50
cluster1 = np.random.randn(n_per_cluster, 2) * 0.8 + [2, 2]
cluster2 = np.random.randn(n_per_cluster, 2) * 0.8 + [-2, 2]
cluster3 = np.random.randn(n_per_cluster, 2) * 0.8 + [0, -2]

X = np.vstack([cluster1, cluster2, cluster3])

# Show clustered data with colors
ax2.scatter(cluster1[:, 0], cluster1[:, 1], c=MLBLUE, s=50, alpha=0.7, label='Cluster 1')
ax2.scatter(cluster2[:, 0], cluster2[:, 1], c=MLRED, s=50, alpha=0.7, label='Cluster 2')
ax2.scatter(cluster3[:, 0], cluster3[:, 1], c=MLGREEN, s=50, alpha=0.7, label='Cluster 3')

# Show centroids
centroids = np.array([[2, 2], [-2, 2], [0, -2]])
ax2.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X',
            edgecolors='white', linewidths=2, label='Centroids', zorder=5)

ax2.set_title('K-Means Groups Similar Points', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1', fontsize=10)
ax2.set_ylabel('Feature 2', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Distance to centroid
ax3 = axes[1, 0]

# Show one cluster with distances
ax3.scatter(cluster1[:, 0], cluster1[:, 1], c=MLBLUE, s=50, alpha=0.7)
centroid = np.array([2, 2])
ax3.scatter(centroid[0], centroid[1], c='black', s=200, marker='X',
            edgecolors='white', linewidths=2, zorder=5)

# Draw lines to some points
for i in range(0, len(cluster1), 5):
    ax3.plot([centroid[0], cluster1[i, 0]], [centroid[1], cluster1[i, 1]],
             'k--', alpha=0.3, linewidth=1)

ax3.set_title('Points Assigned by Distance to Centroid', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Feature 1', fontsize=10)
ax3.set_ylabel('Feature 2', fontsize=10)
ax3.grid(alpha=0.3)

# Add formula
ax3.text(0.05, 0.95, 'Distance = sqrt((x1-c1)^2 + (x2-c2)^2)',
         transform=ax3.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 4: K-means objective
ax4 = axes[1, 1]
ax4.axis('off')

objective = '''
K-MEANS OBJECTIVE FUNCTION

MINIMIZE: Within-Cluster Sum of Squares (WCSS)

          k    n_j
WCSS = SUM   SUM  ||x_i - c_j||^2
         j=1  i=1

Where:
- k = number of clusters
- n_j = points in cluster j
- x_i = data point
- c_j = centroid of cluster j
- ||.|| = Euclidean distance


IN PLAIN ENGLISH:
-----------------
"Make points close to their assigned centroid"

Lower WCSS = Tighter clusters = Better fit


IMPORTANT NOTES:
---------------
1. K-means minimizes WCSS, not cluster count
2. More clusters = Lower WCSS (trivially)
3. Need methods to choose optimal K
   (Elbow method, Silhouette score)


THE TRADEOFF:
-------------
K=1: All in one cluster (max WCSS)
K=n: Each point is its own cluster (WCSS=0)
Optimal K: Balance between them
'''

ax4.text(0.02, 0.98, objective, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('K-Means Objective', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
