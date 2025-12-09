"""Hierarchical Clustering - Core Concept"""
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
fig.suptitle('Hierarchical Clustering: Core Concept', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is hierarchical clustering?
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT IS HIERARCHICAL CLUSTERING?

GOAL: Build a HIERARCHY of clusters,
      from individual points to one big cluster.


TWO APPROACHES:
---------------
AGGLOMERATIVE (Bottom-Up):
  Start: Each point is its own cluster
  Step:  Merge closest clusters
  End:   One cluster containing all

DIVISIVE (Top-Down):
  Start: All points in one cluster
  Step:  Split into sub-clusters
  End:   Each point is its own cluster


KEY ADVANTAGE OVER K-MEANS:
---------------------------
- No need to specify K upfront!
- See the FULL hierarchy
- Choose K by cutting the tree
- Natural tree visualization


DENDROGRAM:
-----------
Tree diagram showing merge/split sequence
Height = distance at which clusters merged
Cut at any height to get K clusters


WHEN TO USE:
------------
- Want to explore cluster structure
- Don't know optimal K
- Need hierarchical relationships
- Small to medium datasets
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Hierarchical Clustering Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual comparison
ax2 = axes[0, 1]

# Generate simple data
np.random.seed(42)
X = np.array([[1, 1], [1.5, 1.2], [3, 3], [3.5, 3.2], [5, 1], [5.5, 1.3]])

# Draw points
ax2.scatter(X[:, 0], X[:, 1], c=MLBLUE, s=100, zorder=5, edgecolors='black')

# Label points
for i, (x, y) in enumerate(X):
    ax2.annotate(f'P{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

# Draw some connecting lines (to show distances)
ax2.plot([X[0, 0], X[1, 0]], [X[0, 1], X[1, 1]], 'k--', alpha=0.5)
ax2.plot([X[2, 0], X[3, 0]], [X[2, 1], X[3, 1]], 'k--', alpha=0.5)
ax2.plot([X[4, 0], X[5, 0]], [X[4, 1], X[5, 1]], 'k--', alpha=0.5)

ax2.set_title('Sample Data Points', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 7)
ax2.set_ylim(0, 5)

# Add text
ax2.text(3.5, 4.5, 'Which points should\nbe clustered together?', fontsize=10, ha='center',
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Agglomerative steps visualization
ax3 = axes[1, 0]
ax3.axis('off')

steps = '''
AGGLOMERATIVE CLUSTERING STEPS

DATA: 6 points (P1, P2, ..., P6)

STEP 0: Each point is its own cluster
        {P1}, {P2}, {P3}, {P4}, {P5}, {P6}

STEP 1: Merge closest pair (P1, P2)
        {P1,P2}, {P3}, {P4}, {P5}, {P6}

STEP 2: Merge next closest (P4, P5)
        {P1,P2}, {P3}, {P4,P5}, {P6}

STEP 3: Merge (P3, P6)
        {P1,P2}, {P3,P6}, {P4,P5}

STEP 4: Merge two clusters
        {P1,P2,P3,P6}, {P4,P5}

STEP 5: Merge remaining
        {P1,P2,P3,P4,P5,P6}


RESULT: Full merge history!
        Can choose to stop at any step
        to get desired number of clusters.


AT STEP 3: 3 clusters
AT STEP 4: 2 clusters
AT STEP 5: 1 cluster
'''

ax3.text(0.02, 0.98, steps, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Agglomerative Steps', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: K-Means vs Hierarchical
ax4 = axes[1, 1]
ax4.axis('off')

comparison = '''
K-MEANS vs HIERARCHICAL CLUSTERING

                  K-MEANS          HIERARCHICAL
                  -------          ------------
Specify K         YES (required)   NO (choose later)

Output            K clusters       Full tree + K clusters

Speed             O(n*k*i)         O(n^2) or O(n^2 log n)

Large data        YES              Limited (memory)

Reproducible      No (random init) YES (deterministic)

Cluster shape     Spherical        Any shape

Interpretable     Centroids        Dendrogram


USE K-MEANS WHEN:
-----------------
- Large dataset (>10,000 points)
- Know or can estimate K
- Spherical clusters expected

USE HIERARCHICAL WHEN:
----------------------
- Small/medium dataset
- Want to explore structure
- Need the hierarchy itself
- Comparing many possible K values
'''

ax4.text(0.02, 0.98, comparison, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('K-Means vs Hierarchical', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
