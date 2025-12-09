"""Dendrogram - Visualizing the Hierarchy"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage

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
fig.suptitle('Dendrogram: Visualizing the Cluster Hierarchy', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate sample data
np.random.seed(42)
n_samples = 20
X = np.vstack([
    np.random.randn(5, 2) * 0.5 + [0, 0],
    np.random.randn(7, 2) * 0.5 + [3, 3],
    np.random.randn(8, 2) * 0.5 + [6, 0]
])

# Compute linkage
Z = linkage(X, method='ward')

# Plot 1: Full dendrogram
ax1 = axes[0, 0]

dendrogram(Z, ax=ax1, leaf_rotation=90, leaf_font_size=8)
ax1.set_title('Complete Dendrogram', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Distance')
ax1.grid(alpha=0.3, axis='y')

# Plot 2: Dendrogram with cut line
ax2 = axes[0, 1]

dendrogram(Z, ax=ax2, leaf_rotation=90, leaf_font_size=8, color_threshold=4)
ax2.axhline(y=4, color=MLRED, linestyle='--', linewidth=2, label='Cut at d=4')
ax2.set_title('Dendrogram with Cut Line (3 clusters)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Distance')
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

# Annotate clusters
ax2.text(3, 5, 'Cluster 1', fontsize=10, color=MLGREEN, fontweight='bold')
ax2.text(10, 5, 'Cluster 2', fontsize=10, color=MLRED, fontweight='bold')
ax2.text(16, 5, 'Cluster 3', fontsize=10, color=MLBLUE, fontweight='bold')

# Plot 3: How to read a dendrogram
ax3 = axes[1, 0]
ax3.axis('off')

explanation = '''
HOW TO READ A DENDROGRAM

STRUCTURE:
----------
- X-axis: Individual samples (leaves)
- Y-axis: Distance at which clusters merge
- Vertical lines: Cluster merges
- Horizontal lines: Distance level


READING THE TREE:
-----------------
1. Start at bottom (each point is a cluster)
2. Move up to see merges
3. Height of merge = distance between clusters
4. Closer merges = more similar


CUTTING THE TREE:
-----------------
Draw horizontal line at height h
Count clusters below the line

Cut high -> Few clusters (broad groups)
Cut low  -> Many clusters (fine groups)


KEY INSIGHTS:
-------------
- Long vertical lines = distinct clusters
- Short vertical lines = similar points
- Gaps in heights = natural cluster boundaries


FINDING OPTIMAL K:
------------------
Look for large gaps in merge heights
Cut just below the gap
This separates distinct groups
'''

ax3.text(0.02, 0.98, explanation, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Reading Dendrograms', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Code for dendrograms
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
DENDROGRAM IN SCIPY

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt

# 1. COMPUTE LINKAGE MATRIX
Z = linkage(X, method='ward')


# 2. PLOT BASIC DENDROGRAM
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()


# 3. CUSTOMIZED DENDROGRAM
dendrogram(
    Z,
    truncate_mode='lastp',  # Show only last p merges
    p=12,                    # Number of clusters to show
    leaf_rotation=90,        # Rotate labels
    leaf_font_size=10,       # Label size
    color_threshold=5,       # Color threshold for clusters
    above_threshold_color='gray'
)


# 4. GET CLUSTER LABELS BY CUTTING
# Cut by number of clusters
labels = fcluster(Z, t=3, criterion='maxclust')

# Cut by distance
labels = fcluster(Z, t=4.0, criterion='distance')

print(f"Cluster labels: {labels}")
print(f"Cluster sizes: {np.bincount(labels)}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
