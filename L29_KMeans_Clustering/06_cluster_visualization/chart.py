"""Cluster Visualization Techniques"""
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
fig.suptitle('Cluster Visualization Techniques', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

colors = [MLBLUE, MLRED, MLGREEN, MLORANGE]

# Plot 1: Scatter plot with centroids
ax1 = axes[0, 0]

for i in range(4):
    mask = labels == i
    ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=40, alpha=0.6, label=f'Cluster {i}')

ax1.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X',
            edgecolors='white', linewidths=2, zorder=5, label='Centroids')

ax1.set_title('Scatter Plot with Centroids', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Voronoi regions (decision boundaries)
ax2 = axes[0, 1]

# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# Predict cluster for each point in mesh
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
from matplotlib.colors import ListedColormap
cmap = ListedColormap([MLBLUE, MLRED, MLGREEN, MLORANGE])
ax2.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
ax2.contour(xx, yy, Z, colors='black', linewidths=0.5)

# Plot data points
for i in range(4):
    mask = labels == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=40, alpha=0.8, edgecolors='black', linewidths=0.5)

ax2.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X',
            edgecolors='white', linewidths=2, zorder=5)

ax2.set_title('Voronoi Decision Boundaries', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.grid(alpha=0.3)

# Plot 3: Cluster profile (bar chart)
ax3 = axes[1, 0]

# Simulate cluster characteristics
cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
features = ['Volatility', 'Return', 'Volume']

# Centroid values (normalized for display)
cluster_profiles = np.array([
    [0.3, 0.5, 0.7],   # Cluster 0
    [0.8, 0.2, 0.4],   # Cluster 1
    [0.5, 0.8, 0.3],   # Cluster 2
    [0.6, 0.4, 0.9],   # Cluster 3
])

x = np.arange(len(features))
width = 0.2

for i, (profile, color) in enumerate(zip(cluster_profiles, colors)):
    ax3.bar(x + i*width, profile, width, label=f'Cluster {i}', color=color, edgecolor='black')

ax3.set_xticks(x + 1.5*width)
ax3.set_xticklabels(features)
ax3.set_title('Cluster Profiles (Centroid Values)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Normalized Value')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')

# Plot 4: Visualization code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
CLUSTER VISUALIZATION CODE

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Fit K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_


# 1. BASIC SCATTER PLOT
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red',
            marker='X', s=200, edgecolors='black')
plt.title('K-Means Clusters')
plt.show()


# 2. DECISION BOUNDARIES (Voronoi)
from matplotlib.colors import ListedColormap

# Create mesh
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
    np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200)
)

# Predict on mesh
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolors='black')
plt.show()


# 3. CLUSTER PROFILES
df_clustered = pd.DataFrame(X, columns=['f1', 'f2'])
df_clustered['cluster'] = labels

profile = df_clustered.groupby('cluster').mean()
profile.T.plot(kind='bar')
plt.title('Feature Means by Cluster')
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Visualization Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
