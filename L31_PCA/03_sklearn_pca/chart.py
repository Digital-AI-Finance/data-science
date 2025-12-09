"""PCA in scikit-learn"""
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
fig.suptitle('PCA in scikit-learn', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic sklearn code
ax1 = axes[0, 0]
ax1.axis('off')

code = '''
PCA IN SKLEARN

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. PREPARE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 2. FIT PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)


KEY ATTRIBUTES:
---------------
pca.components_           # Principal component directions
pca.explained_variance_   # Variance (eigenvalues)
pca.explained_variance_ratio_  # Proportion of total variance
pca.n_components_         # Number of components kept


EXAMPLE:
--------
print(pca.explained_variance_ratio_)
# [0.72, 0.23]  -> PC1 explains 72%, PC2 explains 23%

print(pca.explained_variance_ratio_.sum())
# 0.95  -> Total: 95% of variance explained


IMPORTANT:
----------
Always scale your data before PCA!
PCA is sensitive to feature scales.
'''

ax1.text(0.02, 0.98, code, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Basic sklearn Usage', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual example
ax2 = axes[0, 1]

# Generate high-dimensional-ish data with structure
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=300, n_features=10, n_informative=3,
                           n_redundant=5, n_classes=3, random_state=42,
                           n_clusters_per_class=1)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

colors = [MLBLUE, MLRED, MLGREEN]
for i in range(3):
    mask = y == i
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], s=30, alpha=0.6, label=f'Class {i}')

ax2.set_title(f'10D -> 2D (Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)',
              fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: n_components options
ax3 = axes[1, 0]
ax3.axis('off')

options = '''
N_COMPONENTS OPTIONS

1. INTEGER: Exact number of components
   pca = PCA(n_components=5)
   # Keep exactly 5 components


2. FLOAT (0-1): Variance to preserve
   pca = PCA(n_components=0.95)
   # Keep enough components for 95% variance


3. 'mle': Maximum Likelihood Estimation
   pca = PCA(n_components='mle')
   # Automatically estimate optimal number


4. None: Keep all components
   pca = PCA()
   # Useful for analysis


RECOMMENDED APPROACH:
--------------------
# Start by keeping all, analyze
pca_full = PCA()
pca_full.fit(X_scaled)

# Check cumulative variance
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
k = np.argmax(cumsum >= 0.95) + 1
print(f"Need {k} components for 95% variance")

# Then fit with chosen k
pca = PCA(n_components=k)
X_reduced = pca.fit_transform(X_scaled)
'''

ax3.text(0.02, 0.98, options, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Choosing n_components', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete pipeline example
ax4 = axes[1, 1]
ax4.axis('off')

pipeline = '''
COMPLETE PCA PIPELINE

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Create pipeline with PCA
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', LogisticRegression())
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)


# Access PCA results from pipeline
pca_step = pipe.named_steps['pca']
print(f"Components kept: {pca_step.n_components_}")
print(f"Variance explained: {pca_step.explained_variance_ratio_.sum():.2f}")


# FOR VISUALIZATION ONLY:
pca_viz = PCA(n_components=2)
X_viz = pca_viz.fit_transform(StandardScaler().fit_transform(X))

plt.scatter(X_viz[:, 0], X_viz[:, 1], c=y, cmap='viridis')
plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})')
plt.title('PCA Visualization')
'''

ax4.text(0.02, 0.98, pipeline, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Pipeline', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
