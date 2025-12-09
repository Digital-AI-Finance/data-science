"""Silhouette Score - Cluster Quality Metric"""
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
fig.suptitle('Silhouette Score: Measuring Cluster Quality', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Plot 1: Silhouette concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
SILHOUETTE SCORE

MEASURES: How similar points are to their own cluster
          vs other clusters.

FOR EACH POINT:
---------------
a(i) = average distance to points in SAME cluster
       (cohesion - want this SMALL)

b(i) = average distance to points in NEAREST other cluster
       (separation - want this LARGE)


SILHOUETTE COEFFICIENT:
-----------------------
          b(i) - a(i)
s(i) = ---------------
        max(a(i), b(i))


INTERPRETATION:
---------------
s(i) = +1 : Point is well inside its cluster
s(i) =  0 : Point is on cluster boundary
s(i) = -1 : Point is in wrong cluster!


OVERALL SCORE:
--------------
Average s(i) across all points

Good clustering: score > 0.5
Reasonable:      score > 0.25
Poor:            score < 0.25
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Silhouette Score Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Silhouette scores for different K
ax2 = axes[0, 1]

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_range = range(2, 11)
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

ax2.plot(k_range, silhouette_scores, 'o-', color=MLBLUE, linewidth=2, markersize=8)

# Highlight best K
best_idx = np.argmax(silhouette_scores)
best_k = list(k_range)[best_idx]
ax2.axvline(best_k, color=MLGREEN, linestyle='--', linewidth=2, label=f'Best K={best_k}')
ax2.scatter([best_k], [silhouette_scores[best_idx]], c=MLGREEN, s=150, zorder=5, edgecolors='black')

ax2.set_title('Silhouette Score vs K', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Number of Clusters (K)', fontsize=10)
ax2.set_ylabel('Silhouette Score', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1)

# Add interpretation bands
ax2.axhspan(0.5, 1.0, alpha=0.1, color=MLGREEN)
ax2.axhspan(0.25, 0.5, alpha=0.1, color=MLORANGE)
ax2.axhspan(0, 0.25, alpha=0.1, color=MLRED)

ax2.text(9.5, 0.75, 'Good', fontsize=8, color=MLGREEN)
ax2.text(9.5, 0.37, 'OK', fontsize=8, color=MLORANGE)
ax2.text(9.5, 0.12, 'Poor', fontsize=8, color=MLRED)

# Plot 3: Silhouette diagram
ax3 = axes[1, 0]

from sklearn.metrics import silhouette_samples

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
silhouette_vals = silhouette_samples(X, labels)

colors = [MLBLUE, MLRED, MLGREEN, MLORANGE]
y_lower = 10

for i in range(4):
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()

    size = len(cluster_silhouette_vals)
    y_upper = y_lower + size

    ax3.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                      facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
    ax3.text(-0.05, y_lower + 0.5 * size, str(i), fontsize=10, fontweight='bold')

    y_lower = y_upper + 10

avg_score = silhouette_score(X, labels)
ax3.axvline(avg_score, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_score:.3f}')

ax3.set_title('Silhouette Diagram (K=4)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Silhouette Coefficient', fontsize=10)
ax3.set_ylabel('Cluster', fontsize=10)
ax3.set_xlim(-0.1, 1)
ax3.legend(fontsize=9)

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
SILHOUETTE SCORE IN SKLEARN

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Find best K using silhouette
def find_best_k(X, k_range=range(2, 11)):
    """Find optimal K using silhouette score."""
    scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"K={k}: Silhouette = {score:.4f}")

    best_k = list(k_range)[np.argmax(scores)]
    return best_k, scores


best_k, scores = find_best_k(X_scaled)
print(f"Optimal K: {best_k}")


# Per-sample silhouette (for diagnosis)
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X)

sample_scores = silhouette_samples(X, labels)

# Find poorly assigned points
poor_fit = sample_scores < 0
print(f"Points in wrong cluster: {poor_fit.sum()}")


# ELBOW + SILHOUETTE TOGETHER:
# ----------------------------
# 1. Elbow method: suggests possible K values
# 2. Silhouette: confirms best K from candidates
# 3. Domain knowledge: final decision
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
