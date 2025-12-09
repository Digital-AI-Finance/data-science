"""K-Means in scikit-learn"""
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
fig.suptitle('K-Means Clustering in scikit-learn', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic sklearn code
ax1 = axes[0, 0]
ax1.axis('off')

code = '''
K-MEANS IN SKLEARN

from sklearn.cluster import KMeans

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,        # Number of clusters (K)
    init='k-means++',    # Smart initialization
    n_init=10,           # Run 10 times, pick best
    max_iter=300,        # Max iterations per run
    random_state=42      # Reproducibility
)

# Fit the model
kmeans.fit(X)

# Get results
labels = kmeans.labels_           # Cluster assignment (0, 1, 2)
centroids = kmeans.cluster_centers_  # Centroid coordinates
inertia = kmeans.inertia_         # WCSS (lower = better)


# Predict new points
new_labels = kmeans.predict(X_new)


# Fit and predict in one step
labels = kmeans.fit_predict(X)


KEY PARAMETERS:
---------------
n_clusters: Number of clusters to form
init: 'k-means++' (default, smart) or 'random'
n_init: Number of runs with different seeds
max_iter: Maximum iterations per run
'''

ax1.text(0.02, 0.98, code, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Basic Usage', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visualization of clustering
ax2 = axes[0, 1]

# Generate sample data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Simulated K-means result
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

colors = [MLBLUE, MLRED, MLGREEN, MLORANGE]
for i in range(4):
    mask = labels == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=30, alpha=0.6, label=f'Cluster {i}')
    ax2.scatter(centroids[i, 0], centroids[i, 1], c=colors[i], s=200, marker='X',
                edgecolors='black', linewidths=2, zorder=5)

ax2.set_title(f'K-Means Result (WCSS={kmeans.inertia_:.1f})', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: k-means++ initialization
ax3 = axes[1, 0]
ax3.axis('off')

kmeanspp = '''
K-MEANS++ INITIALIZATION

PROBLEM WITH RANDOM INIT:
-------------------------
- Poor initial centroids = bad clusters
- May converge to local minimum
- Inconsistent results


K-MEANS++ SOLUTION:
-------------------
1. Choose first centroid randomly

2. For each remaining centroid:
   - Compute distance D(x) from each point
     to its nearest centroid
   - Choose next centroid with probability
     proportional to D(x)^2
   - Points far from existing centroids
     are more likely to be chosen

3. Repeat until K centroids selected


WHY IT WORKS:
-------------
- Spreads initial centroids apart
- Better coverage of data space
- More likely to find global optimum
- Default in sklearn (init='k-means++')


COMPARISON:
-----------
Random init: May need 100+ runs
k-means++:   Usually 10 runs sufficient
'''

ax3.text(0.02, 0.98, kmeanspp, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('K-Means++ Initialization', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete example
ax4 = axes[1, 1]
ax4.axis('off')

example = '''
COMPLETE K-MEANS EXAMPLE

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load financial data
df = pd.read_csv('stock_features.csv')
X = df[['volatility', 'avg_return', 'volume']].values

# IMPORTANT: Scale features first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal K (we'll see how later)
k_optimal = 4

# Fit K-Means
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_stats = df.groupby('cluster').agg({
    'volatility': 'mean',
    'avg_return': 'mean',
    'volume': 'mean',
    'ticker': 'count'
}).rename(columns={'ticker': 'n_stocks'})

print(cluster_stats)


# Interpret centroids (unscaled)
centroids_unscaled = scaler.inverse_transform(
    kmeans.cluster_centers_
)
print("Cluster centroids:")
print(pd.DataFrame(
    centroids_unscaled,
    columns=['volatility', 'avg_return', 'volume']
))
'''

ax4.text(0.02, 0.98, example, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
