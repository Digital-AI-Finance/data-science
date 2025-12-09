"""Cutting the Dendrogram Tree - Choosing Number of Clusters"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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
fig.suptitle('Cutting the Dendrogram: Choosing Number of Clusters', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate sample data
X = np.vstack([
    np.random.randn(15, 2) * 0.6 + [0, 0],
    np.random.randn(15, 2) * 0.6 + [4, 4],
    np.random.randn(15, 2) * 0.6 + [8, 0]
])

Z = linkage(X, method='ward')

colors_list = [MLBLUE, MLRED, MLGREEN, MLORANGE, MLPURPLE]

# Plot 1: Dendrogram with 2 clusters
ax1 = axes[0, 0]
dendrogram(Z, ax=ax1, color_threshold=9)
ax1.axhline(y=9, color='black', linestyle='--', linewidth=2)
ax1.set_title('Cut for 2 Clusters', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Distance')

# Plot 2: Dendrogram with 3 clusters
ax2 = axes[0, 1]
dendrogram(Z, ax=ax2, color_threshold=5)
ax2.axhline(y=5, color='black', linestyle='--', linewidth=2)
ax2.set_title('Cut for 3 Clusters', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Distance')

# Plot 3: Dendrogram with 5 clusters
ax3 = axes[0, 2]
dendrogram(Z, ax=ax3, color_threshold=2.5)
ax3.axhline(y=2.5, color='black', linestyle='--', linewidth=2)
ax3.set_title('Cut for 5 Clusters', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Distance')

# Plot 4: Scatter - 2 clusters
ax4 = axes[1, 0]
labels_2 = fcluster(Z, t=2, criterion='maxclust')
for i in range(1, 3):
    mask = labels_2 == i
    ax4.scatter(X[mask, 0], X[mask, 1], c=colors_list[i-1], s=40, alpha=0.7)
ax4.set_title('Data with 2 Clusters', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3)

# Plot 5: Scatter - 3 clusters
ax5 = axes[1, 1]
labels_3 = fcluster(Z, t=3, criterion='maxclust')
for i in range(1, 4):
    mask = labels_3 == i
    ax5.scatter(X[mask, 0], X[mask, 1], c=colors_list[i-1], s=40, alpha=0.7)
ax5.set_title('Data with 3 Clusters (Optimal)', fontsize=11, fontweight='bold', color=MLGREEN)
ax5.grid(alpha=0.3)

# Plot 6: How to choose
ax6 = axes[1, 2]
ax6.axis('off')

guidance = '''
HOW TO CHOOSE THE CUT

METHOD 1: VISUAL INSPECTION
---------------------------
Look for longest vertical lines
(gaps in the dendrogram)
Cut just below the gap


METHOD 2: DOMAIN KNOWLEDGE
--------------------------
Do you need 3 categories?
5 risk levels?
Let business needs guide you


METHOD 3: SILHOUETTE SCORE
--------------------------
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

best_k, best_score = 2, -1
for k in range(2, 10):
    labels = fcluster(Z, t=k, criterion='maxclust')
    score = silhouette_score(X, labels)
    if score > best_score:
        best_k, best_score = k, score


METHOD 4: ELBOW ON MERGE HEIGHTS
--------------------------------
# Heights are in Z[:, 2]
heights = Z[:, 2]
# Look for jumps in heights


CRITERION OPTIONS:
------------------
fcluster(Z, t=3, criterion='maxclust')  # t = number of clusters
fcluster(Z, t=5.0, criterion='distance')  # t = cut height
'''

ax6.text(0.02, 0.98, guidance, transform=ax6.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax6.set_title('Choosing the Cut', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
