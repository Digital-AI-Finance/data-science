"""Linkage Methods - How to Measure Cluster Distance"""
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
fig.suptitle('Linkage Methods: How to Measure Distance Between Clusters', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate two clusters
cluster1 = np.array([[1, 2], [1.5, 2.5], [2, 2]])
cluster2 = np.array([[4, 3], [4.5, 3.5], [5, 3]])

def plot_clusters(ax, c1, c2, title, lines=None, highlight=None):
    ax.scatter(c1[:, 0], c1[:, 1], c=MLBLUE, s=100, label='Cluster A')
    ax.scatter(c2[:, 0], c2[:, 1], c=MLRED, s=100, label='Cluster B')

    if lines is not None:
        for p1, p2 in lines:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.3)

    if highlight is not None:
        for p1, p2 in highlight:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=MLGREEN, linewidth=3)

    ax.set_title(title, fontsize=11, fontweight='bold', color=MLPURPLE)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 6)
    ax.set_ylim(1, 5)

# Plot 1: Single Linkage
ax1 = axes[0, 0]

# All pairwise distances
all_lines = [(p1, p2) for p1 in cluster1 for p2 in cluster2]

# Find minimum distance
min_dist = float('inf')
min_pair = None
for p1 in cluster1:
    for p2 in cluster2:
        d = np.sqrt(np.sum((p1 - p2)**2))
        if d < min_dist:
            min_dist = d
            min_pair = (p1, p2)

plot_clusters(ax1, cluster1, cluster2, 'Single Linkage (Minimum)',
              lines=all_lines, highlight=[min_pair])
ax1.text(3, 1.5, f'd = {min_dist:.2f}', fontsize=10, color=MLGREEN, fontweight='bold')

# Plot 2: Complete Linkage
ax2 = axes[0, 1]

# Find maximum distance
max_dist = 0
max_pair = None
for p1 in cluster1:
    for p2 in cluster2:
        d = np.sqrt(np.sum((p1 - p2)**2))
        if d > max_dist:
            max_dist = d
            max_pair = (p1, p2)

plot_clusters(ax2, cluster1, cluster2, 'Complete Linkage (Maximum)',
              lines=all_lines, highlight=[max_pair])
ax2.text(3, 1.5, f'd = {max_dist:.2f}', fontsize=10, color=MLGREEN, fontweight='bold')

# Plot 3: Average Linkage
ax3 = axes[0, 2]

# Calculate average distance
total_dist = 0
count = 0
for p1 in cluster1:
    for p2 in cluster2:
        total_dist += np.sqrt(np.sum((p1 - p2)**2))
        count += 1
avg_dist = total_dist / count

plot_clusters(ax3, cluster1, cluster2, 'Average Linkage',
              lines=all_lines, highlight=None)
ax3.text(3, 1.5, f'd = {avg_dist:.2f} (avg)', fontsize=10, color=MLGREEN, fontweight='bold')

# Plot 4: Ward Linkage explanation
ax4 = axes[1, 0]
ax4.axis('off')

ward_text = '''
WARD'S METHOD

CONCEPT:
--------
Minimize the increase in total
within-cluster variance when merging.

FORMULA:
--------
d(A,B) = increase in WCSS
         if A and B were merged


WHY USE WARD:
-------------
- Tends to create compact, spherical clusters
- Similar to K-means objective
- Often gives best results for
  well-separated clusters


COMPARISON:
-----------
Single:   Can create long, "chained" clusters
Complete: Creates compact but possibly unequal
Average:  Compromise between single/complete
Ward:     Compact, equal-sized clusters
'''

ax4.text(0.02, 0.98, ward_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title("Ward's Method", fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 5: Linkage comparison summary
ax5 = axes[1, 1]
ax5.axis('off')

summary = '''
LINKAGE METHOD SUMMARY

METHOD    | DISTANCE         | BEHAVIOR
----------|------------------|------------------
Single    | min(all pairs)   | Chaining effect
Complete  | max(all pairs)   | Compact clusters
Average   | mean(all pairs)  | Balanced
Ward      | variance incr.   | Equal-sized


CHAINING EFFECT (Single Linkage):
---------------------------------
Points get added one-by-one to existing cluster
Can create long, snake-like clusters
May not find expected groups


RECOMMENDATION:
---------------
Start with Ward's method (most common)
Try complete if Ward gives poor results
Avoid single unless you expect chains


SKLEARN DEFAULT: Ward
'''

ax5.text(0.02, 0.98, summary, transform=ax5.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax5.set_title('Linkage Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 6: Code
ax6 = axes[1, 2]
ax6.axis('off')

code = '''
LINKAGE IN SCIPY AND SKLEARN

from scipy.cluster.hierarchy import linkage

# Compute linkage matrix
# method: 'single', 'complete', 'average', 'ward'
Z = linkage(X, method='ward')


from sklearn.cluster import AgglomerativeClustering

# Using different linkage methods
models = {
    'single': AgglomerativeClustering(
        n_clusters=3, linkage='single'),
    'complete': AgglomerativeClustering(
        n_clusters=3, linkage='complete'),
    'average': AgglomerativeClustering(
        n_clusters=3, linkage='average'),
    'ward': AgglomerativeClustering(
        n_clusters=3, linkage='ward')
}

for name, model in models.items():
    labels = model.fit_predict(X)
    print(f"{name}: {np.bincount(labels)}")


# Note: Ward requires euclidean distance
# Other methods can use different metrics
'''

ax6.text(0.02, 0.98, code, transform=ax6.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax6.set_title('Python Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
