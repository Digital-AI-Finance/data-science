"""Agglomerative Clustering in scikit-learn"""
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
fig.suptitle('Agglomerative Clustering in scikit-learn', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate sample data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.8, random_state=42)

# Plot 1: Basic sklearn code
ax1 = axes[0, 0]
ax1.axis('off')

code = '''
AGGLOMERATIVE CLUSTERING IN SKLEARN

from sklearn.cluster import AgglomerativeClustering

# Create model
agg = AgglomerativeClustering(
    n_clusters=4,          # Number of clusters
    linkage='ward',        # Linkage method
    # affinity='euclidean' # Distance metric (deprecated, use metric)
)

# Fit and get labels
labels = agg.fit_predict(X)


KEY PARAMETERS:
---------------
n_clusters: int or None
    Number of clusters to find.
    If None, must set distance_threshold.

linkage: {'ward', 'complete', 'average', 'single'}
    Which linkage criterion to use.
    'ward' minimizes variance (default).

distance_threshold: float or None
    If set, n_clusters must be None.
    Clusters are merged until this threshold.


EXAMPLE WITH DISTANCE THRESHOLD:
--------------------------------
agg = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=5.0,
    linkage='ward'
)
labels = agg.fit_predict(X)
n_clusters_found = agg.n_clusters_
'''

ax1.text(0.02, 0.98, code, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Basic Usage', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Clustering result
ax2 = axes[0, 1]

from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = agg.fit_predict(X)

colors = [MLBLUE, MLRED, MLGREEN, MLORANGE]
for i in range(4):
    mask = labels == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=40, alpha=0.7, label=f'Cluster {i}')

ax2.set_title('Agglomerative Clustering Result', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Comparison of linkage methods
ax3 = axes[1, 0]

from sklearn.cluster import AgglomerativeClustering

# Create a challenging dataset
np.random.seed(42)
X_chain = np.vstack([
    np.random.randn(30, 2) * 0.5 + [0, 0],
    np.random.randn(30, 2) * 0.5 + [3, 0],
    np.random.randn(30, 2) * 0.5 + [1.5, 2],
])

linkage_methods = ['ward', 'complete', 'average', 'single']
fig3_colors = [MLBLUE, MLRED, MLGREEN, MLORANGE]

# Just show ward result
agg_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_ward = agg_ward.fit_predict(X_chain)

for i in range(3):
    mask = labels_ward == i
    ax3.scatter(X_chain[mask, 0], X_chain[mask, 1], c=fig3_colors[i], s=40, alpha=0.7)

ax3.set_title("Ward's Linkage (Recommended)", fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.grid(alpha=0.3)

# Plot 4: Complete pipeline
ax4 = axes[1, 1]
ax4.axis('off')

pipeline = '''
COMPLETE HIERARCHICAL CLUSTERING PIPELINE

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 1. LOAD AND PREPARE DATA
df = pd.read_csv('stock_features.csv')
features = ['volatility', 'return', 'volume']
X = df[features].values

# 2. SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. EXPLORE WITH DENDROGRAM
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=20)
plt.axhline(y=5, color='r', linestyle='--')
plt.title('Dendrogram - Cut at distance 5?')
plt.show()

# 4. CHOOSE K AND CLUSTER
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['cluster'] = agg.fit_predict(X_scaled)

# 5. ANALYZE CLUSTERS
cluster_summary = df.groupby('cluster')[features].mean()
print(cluster_summary)

# 6. VALIDATE WITH SILHOUETTE
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, df['cluster'])
print(f"Silhouette Score: {score:.4f}")
'''

ax4.text(0.02, 0.98, pipeline, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Pipeline', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
