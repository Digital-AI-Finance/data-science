"""Eigenvalues and Eigenvectors in PCA"""
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
fig.suptitle('Eigenvalues and Eigenvectors: The Math Behind PCA', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What are eigenvalues/eigenvectors?
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
EIGENVALUES AND EIGENVECTORS

DEFINITION:
-----------
For a matrix A, if:
    A * v = lambda * v

Then:
- v is an EIGENVECTOR
- lambda is the EIGENVALUE


INTUITION:
----------
An eigenvector v is a direction that
the matrix A only SCALES (doesn't rotate).

The eigenvalue lambda tells you HOW MUCH
it gets scaled.


IN PCA:
-------
A = Covariance matrix of your data

Eigenvectors = Principal components
               (directions)

Eigenvalues = Variance along each PC
              (importance)


PCA ALGORITHM:
--------------
1. Center data (subtract mean)
2. Compute covariance matrix
3. Find eigenvalues & eigenvectors
4. Sort by eigenvalue (descending)
5. Keep top k eigenvectors
6. Project data onto these k directions
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Eigenvalues & Eigenvectors', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual demonstration
ax2 = axes[0, 1]

# Generate data and compute covariance
mean = [0, 0]
cov = [[4, 2], [2, 2]]
X = np.random.multivariate_normal(mean, cov, 200)
X_centered = X - X.mean(axis=0)

# Covariance matrix
cov_matrix = np.cov(X_centered.T)

# Eigendecomposition
from numpy.linalg import eig
eigenvalues, eigenvectors = eig(cov_matrix)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

ax2.scatter(X_centered[:, 0], X_centered[:, 1], c=MLBLUE, s=20, alpha=0.5)

# Plot eigenvectors scaled by eigenvalues
colors = [MLRED, MLORANGE]
for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    scale = np.sqrt(eigval) * 2
    ax2.arrow(0, 0, eigvec[0]*scale, eigvec[1]*scale,
              head_width=0.15, head_length=0.1, fc=colors[i], ec=colors[i], linewidth=2.5)
    ax2.text(eigvec[0]*scale*1.15, eigvec[1]*scale*1.15,
             f'PC{i+1}\n(lambda={eigval:.2f})', fontsize=9, color=colors[i], fontweight='bold')

ax2.set_title('Eigenvectors of Covariance Matrix', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Feature 1 (centered)')
ax2.set_ylabel('Feature 2 (centered)')
ax2.grid(alpha=0.3)
ax2.set_aspect('equal')
ax2.set_xlim(-5, 5)
ax2.set_ylim(-4, 4)

# Plot 3: Eigenvalue magnitudes
ax3 = axes[1, 0]

# Example with more dimensions
np.random.seed(42)
n_features = 8
eigenvalues_example = np.array([5.2, 2.1, 0.8, 0.4, 0.2, 0.15, 0.1, 0.05])

bars = ax3.bar(range(1, n_features+1), eigenvalues_example, color=MLBLUE, edgecolor='black')

# Color first few bars differently
for i in range(3):
    bars[i].set_color(MLGREEN)

ax3.axhline(y=1, color=MLRED, linestyle='--', label='Kaiser criterion (lambda=1)')

ax3.set_title('Eigenvalues (Variance per Component)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Principal Component')
ax3.set_ylabel('Eigenvalue (Variance)')
ax3.set_xticks(range(1, n_features+1))
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

ax3.text(5.5, 4, 'Keep these\n(high variance)', fontsize=9, color=MLGREEN)
ax3.text(5.5, 0.5, 'Discard these\n(low variance)', fontsize=9, color=MLBLUE)

# Plot 4: Step by step calculation
ax4 = axes[1, 1]
ax4.axis('off')

calculation = '''
PCA STEP BY STEP

1. CENTER THE DATA
   X_centered = X - X.mean(axis=0)


2. COMPUTE COVARIANCE MATRIX
   Cov = (X_centered.T @ X_centered) / (n-1)

   Example 2x2:
   Cov = [[var(x1),    cov(x1,x2)]
          [cov(x2,x1), var(x2)   ]]


3. FIND EIGENVALUES & EIGENVECTORS
   Cov @ v = lambda @ v

   Solve: det(Cov - lambda*I) = 0


4. SORT BY EIGENVALUE (DESCENDING)
   lambda_1 >= lambda_2 >= ... >= lambda_d


5. SELECT TOP k COMPONENTS
   Based on:
   - Explained variance ratio
   - Scree plot / elbow
   - Kaiser criterion (lambda >= 1)


6. PROJECT DATA
   X_pca = X_centered @ eigenvectors[:, :k]


Result: d dimensions -> k dimensions
'''

ax4.text(0.02, 0.98, calculation, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Step-by-Step Calculation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
