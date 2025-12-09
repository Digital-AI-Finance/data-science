"""Cluster Comparison - K-Means vs Hierarchical"""
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

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Comparing K-Means vs Hierarchical Clustering', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate different types of data
from sklearn.datasets import make_blobs, make_moons

# Dataset 1: Spherical clusters (good for K-Means)
X1, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# Dataset 2: Non-spherical (challenging for K-Means)
X2, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
X2 = X2 * 3  # Scale up

from sklearn.cluster import KMeans, AgglomerativeClustering

colors = [MLBLUE, MLRED, MLGREEN]

# Plot 1: Spherical - K-Means
ax1 = axes[0, 0]
kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels1 = kmeans1.fit_predict(X1)
for i in range(3):
    mask = labels1 == i
    ax1.scatter(X1[mask, 0], X1[mask, 1], c=colors[i], s=30, alpha=0.7)
ax1.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1],
            c='black', s=150, marker='X', edgecolors='white', zorder=5)
ax1.set_title('K-Means: Spherical Data', fontsize=11, fontweight='bold', color=MLGREEN)
ax1.grid(alpha=0.3)

# Plot 2: Spherical - Hierarchical
ax2 = axes[0, 1]
agg1 = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels2 = agg1.fit_predict(X1)
for i in range(3):
    mask = labels2 == i
    ax2.scatter(X1[mask, 0], X1[mask, 1], c=colors[i], s=30, alpha=0.7)
ax2.set_title('Hierarchical: Spherical Data', fontsize=11, fontweight='bold', color=MLGREEN)
ax2.grid(alpha=0.3)

# Plot 3: Summary for spherical
ax3 = axes[0, 2]
ax3.axis('off')

summary1 = '''
SPHERICAL CLUSTERS

Both methods work well!

K-MEANS:
+ Fast and scalable
+ Clear centroids
+ Good for large data

HIERARCHICAL:
+ Deterministic
+ Shows hierarchy
+ More exploration

VERDICT: Use K-Means for speed,
         Hierarchical for insight
'''

ax3.text(0.1, 0.95, summary1, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Spherical Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Moons - K-Means (fails)
ax4 = axes[1, 0]
kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels3 = kmeans2.fit_predict(X2)
for i in range(2):
    mask = labels3 == i
    ax4.scatter(X2[mask, 0], X2[mask, 1], c=colors[i], s=30, alpha=0.7)
ax4.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1],
            c='black', s=150, marker='X', edgecolors='white', zorder=5)
ax4.set_title('K-Means: Moon Data (FAILS)', fontsize=11, fontweight='bold', color=MLRED)
ax4.grid(alpha=0.3)

# Plot 5: Moons - Hierarchical (works with single linkage)
ax5 = axes[1, 1]
agg2 = AgglomerativeClustering(n_clusters=2, linkage='single')
labels4 = agg2.fit_predict(X2)
for i in range(2):
    mask = labels4 == i
    ax5.scatter(X2[mask, 0], X2[mask, 1], c=colors[i], s=30, alpha=0.7)
ax5.set_title('Hierarchical (Single): Moon Data', fontsize=11, fontweight='bold', color=MLGREEN)
ax5.grid(alpha=0.3)

# Plot 6: Decision guide
ax6 = axes[1, 2]
ax6.axis('off')

guide = '''
WHEN TO USE WHAT

USE K-MEANS WHEN:
-----------------
+ Large dataset (>10K points)
+ Spherical cluster shapes
+ Know approximate K
+ Need speed
+ Want interpretable centroids


USE HIERARCHICAL WHEN:
----------------------
+ Small/medium dataset
+ Want to explore structure
+ Need the hierarchy itself
+ Non-spherical clusters
+ Correlation-based distance


LINKAGE CHOICE:
---------------
Ward:     Most common, spherical
Complete: Compact clusters
Average:  Balanced
Single:   Chain-like, non-spherical


VALIDATION:
-----------
Always use silhouette score
to compare results!
'''

ax6.text(0.02, 0.98, guide, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax6.set_title('Decision Guide', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
