"""Elbow Method - Choosing Optimal K"""
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
fig.suptitle('Elbow Method: Choosing Optimal K', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data with 4 true clusters
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Plot 1: Elbow curve
ax1 = axes[0, 0]

from sklearn.cluster import KMeans

k_range = range(1, 11)
wcss = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

ax1.plot(k_range, wcss, 'o-', color=MLBLUE, linewidth=2, markersize=8)
ax1.axvline(4, color=MLRED, linestyle='--', linewidth=2, label='Elbow at K=4')

# Highlight the elbow
ax1.scatter([4], [wcss[3]], c=MLRED, s=150, zorder=5, edgecolors='black')

ax1.set_title('The Elbow Method', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Number of Clusters (K)', fontsize=10)
ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Annotate
ax1.annotate('Elbow point', xy=(4, wcss[3]), xytext=(6, wcss[3]+200),
             fontsize=10, arrowprops=dict(arrowstyle='->', color=MLRED))

# Plot 2: Explanation
ax2 = axes[0, 1]
ax2.axis('off')

explanation = '''
THE ELBOW METHOD

CONCEPT:
--------
Plot WCSS vs K and look for an "elbow"
where the rate of decrease sharply changes.


HOW TO READ:
------------
K=1: One cluster (high WCSS)
K=2: Two clusters (lower WCSS)
...
K=n: Each point is a cluster (WCSS=0)

The elbow = diminishing returns point


WHY AN ELBOW?
-------------
- Before elbow: Adding clusters helps a lot
  (captures real structure)

- After elbow: Adding clusters helps little
  (just splitting existing groups)


THE TRADEOFF:
-------------
- Too few K: Underfitting (miss structure)
- Too many K: Overfitting (noise as clusters)
- Elbow K: Just right!


LIMITATIONS:
------------
- Not always a clear elbow
- Subjective interpretation
- May need domain knowledge
- Consider Silhouette score too
'''

ax2.text(0.02, 0.98, explanation, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Understanding the Elbow', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Visual comparison of different K
ax3 = axes[1, 0]

# Show K=2, K=4, K=8 side by side
from sklearn.cluster import KMeans

colors = [MLBLUE, MLRED, MLGREEN, MLORANGE, MLPURPLE, '#808080', '#FF69B4', '#00CED1']

# K=4 (optimal)
kmeans4 = KMeans(n_clusters=4, random_state=42, n_init=10)
labels4 = kmeans4.fit_predict(X)

for i in range(4):
    mask = labels4 == i
    ax3.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=30, alpha=0.6)

ax3.scatter(kmeans4.cluster_centers_[:, 0], kmeans4.cluster_centers_[:, 1],
           c='black', s=150, marker='X', edgecolors='white', linewidths=2, zorder=5)

ax3.set_title(f'K=4 (Optimal, WCSS={kmeans4.inertia_:.0f})', fontsize=11, fontweight='bold', color=MLGREEN)
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.grid(alpha=0.3)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
ELBOW METHOD IN PYTHON

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_elbow(X, k_range=range(1, 11)):
    """Plot elbow curve for K selection."""
    wcss = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, 'bo-', linewidth=2)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.grid(True, alpha=0.3)
    plt.show()

    return wcss


# Usage
wcss = plot_elbow(X_scaled, k_range=range(1, 11))


# Automated elbow detection (optional)
# Using kneed library
from kneed import KneeLocator

kl = KneeLocator(
    range(1, 11), wcss,
    curve='convex',
    direction='decreasing'
)
optimal_k = kl.elbow
print(f"Optimal K: {optimal_k}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
