"""PCA - Principal Component Analysis Concept"""
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
fig.suptitle('Principal Component Analysis (PCA): Core Concept', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is PCA?
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT IS PCA?

PROBLEM: Too many features (dimensions)
- Hard to visualize
- Slow computation
- Curse of dimensionality
- Redundant/correlated features


SOLUTION: Dimensionality Reduction
Find a smaller set of features that
captures most of the information.


PCA DOES THIS BY:
-----------------
1. Finding directions of maximum variance
2. Projecting data onto these directions
3. Keeping only the most important ones


PRINCIPAL COMPONENTS:
--------------------
PC1: Direction of maximum variance
PC2: Direction of 2nd max variance
     (perpendicular to PC1)
...and so on


KEY INSIGHT:
------------
If features are correlated,
a few PCs can capture most information
with far fewer dimensions.


USE CASES:
----------
- Visualization (reduce to 2D/3D)
- Noise reduction
- Feature extraction
- Preprocessing for ML
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('PCA Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual intuition - 2D data with principal components
ax2 = axes[0, 1]

# Generate correlated 2D data
mean = [0, 0]
cov = [[3, 2], [2, 2]]  # Correlated
X = np.random.multivariate_normal(mean, cov, 100)

ax2.scatter(X[:, 0], X[:, 1], c=MLBLUE, s=30, alpha=0.6)

# Calculate principal components
from numpy.linalg import eig
cov_matrix = np.cov(X.T)
eigenvalues, eigenvectors = eig(cov_matrix)

# Sort by eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Plot principal component directions
center = X.mean(axis=0)
for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    color = MLRED if i == 0 else MLORANGE
    label = f'PC{i+1} (var={eigval:.2f})'
    scale = np.sqrt(eigval) * 2
    ax2.arrow(center[0], center[1], eigvec[0]*scale, eigvec[1]*scale,
              head_width=0.2, head_length=0.1, fc=color, ec=color, linewidth=2)
    ax2.text(center[0] + eigvec[0]*scale*1.2, center[1] + eigvec[1]*scale*1.2,
             label, fontsize=9, color=color, fontweight='bold')

ax2.set_title('Principal Components = Directions of Max Variance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.grid(alpha=0.3)
ax2.set_aspect('equal')

# Plot 3: Before and after PCA
ax3 = axes[1, 0]

# Project data onto PC1
pc1 = eigenvectors[:, 0]
X_projected = X @ pc1

# Show projection
ax3.scatter(X[:, 0], X[:, 1], c=MLBLUE, s=30, alpha=0.3, label='Original 2D')

# Plot projection line (PC1)
t = np.linspace(-5, 5, 100)
ax3.plot(t * pc1[0], t * pc1[1], color=MLRED, linewidth=2, label='PC1 axis')

# Show projected points on PC1
X_on_pc1 = np.outer(X_projected, pc1)
ax3.scatter(X_on_pc1[:, 0], X_on_pc1[:, 1], c=MLGREEN, s=30, alpha=0.7, label='Projected (1D)')

# Draw projection lines for a few points
for i in range(0, len(X), 10):
    ax3.plot([X[i, 0], X_on_pc1[i, 0]], [X[i, 1], X_on_pc1[i, 1]], 'k:', alpha=0.3)

ax3.set_title('Projection onto PC1 (2D -> 1D)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)
ax3.set_xlim(-5, 5)
ax3.set_ylim(-4, 4)

# Plot 4: Why PCA works
ax4 = axes[1, 1]
ax4.axis('off')

why = '''
WHY PCA WORKS

CORRELATED FEATURES:
-------------------
If Feature1 and Feature2 are correlated,
they contain redundant information.

One direction captures most variation,
the other captures little.


EXAMPLE: Stock Returns
----------------------
100 stocks, daily returns for 1 year
= 100 dimensions

But stocks move together (market effect)!

PCA finds:
- PC1: Market direction (explains 50%+)
- PC2-5: Sector effects
- Rest: Noise

Can reduce 100D -> 10D with minimal loss!


VARIANCE PRESERVATION:
---------------------
PCA keeps directions with HIGH variance
(= important information)

Discards directions with LOW variance
(= noise or redundancy)


MATHEMATICAL GUARANTEE:
-----------------------
PCA gives the BEST linear reduction
in terms of reconstruction error.
'''

ax4.text(0.02, 0.98, why, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Why PCA Works', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
