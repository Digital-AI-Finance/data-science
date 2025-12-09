"""PCA Visualization - 2D and 3D Projections"""
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

fig = plt.figure(figsize=(14, 10))
fig.suptitle('PCA for Data Visualization', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=400, n_features=15, n_informative=5,
                           n_redundant=5, n_classes=4, random_state=42,
                           n_clusters_per_class=1)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot 1: 2D PCA visualization
ax1 = fig.add_subplot(2, 2, 1)

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

colors = [MLBLUE, MLRED, MLGREEN, MLORANGE]
for i in range(4):
    mask = y == i
    ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i], s=40, alpha=0.6,
                label=f'Class {i}', edgecolors='black', linewidths=0.3)

ax1.set_title(f'2D PCA ({pca_2d.explained_variance_ratio_.sum()*100:.1f}% variance)',
              fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: 3D PCA visualization
ax2 = fig.add_subplot(2, 2, 2, projection='3d')

pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_scaled)

for i in range(4):
    mask = y == i
    ax2.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                c=colors[i], s=30, alpha=0.6, label=f'Class {i}')

ax2.set_title(f'3D PCA ({pca_3d.explained_variance_ratio_.sum()*100:.1f}% variance)',
              fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel(f'PC1')
ax2.set_ylabel(f'PC2')
ax2.set_zlabel(f'PC3')
ax2.legend(fontsize=7)

# Plot 3: Biplot
ax3 = fig.add_subplot(2, 2, 3)

# Scatter plot
for i in range(4):
    mask = y == i
    ax3.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i], s=30, alpha=0.4)

# Loading vectors (first 5 features)
loadings = pca_2d.components_.T
feature_names = [f'F{i+1}' for i in range(15)]

scale = 4  # Scale factor for visibility
for i in range(5):
    ax3.arrow(0, 0, loadings[i, 0]*scale, loadings[i, 1]*scale,
              head_width=0.15, head_length=0.1, fc=MLPURPLE, ec=MLPURPLE, linewidth=1.5)
    ax3.text(loadings[i, 0]*scale*1.1, loadings[i, 1]*scale*1.1,
             feature_names[i], fontsize=9, color=MLPURPLE, fontweight='bold')

ax3.set_title('Biplot (Data + Loadings)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.grid(alpha=0.3)
ax3.set_xlim(-6, 6)
ax3.set_ylim(-6, 6)

# Plot 4: Code
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

code = '''
PCA VISUALIZATION CODE

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prepare and fit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# 2D SCATTER PLOT
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter, label='Class')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA Visualization')
plt.show()


# 3D PLOT
from mpl_toolkits.mplot3d import Axes3D

pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='viridis')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


# BIPLOT (data + loadings)
# Shows both points AND feature contributions
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Visualization Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
