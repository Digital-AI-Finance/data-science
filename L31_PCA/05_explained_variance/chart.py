"""Explained Variance - Understanding PCA Output"""
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
fig.suptitle('Explained Variance: Understanding PCA Output', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate high-dimensional data
from sklearn.datasets import make_classification
X, _ = make_classification(n_samples=500, n_features=20, n_informative=5,
                           n_redundant=10, n_classes=2, random_state=42)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)

# Plot 1: What is explained variance?
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
EXPLAINED VARIANCE

TOTAL VARIANCE:
--------------
Sum of variances across all features.
For standardized data: = number of features.


EXPLAINED VARIANCE (per PC):
----------------------------
How much of total variance is captured
by each principal component.

pca.explained_variance_
# Actual variance (eigenvalue)

pca.explained_variance_ratio_
# Proportion of total (eigenvalue / sum)


EXAMPLE:
--------
20 features (standardized) -> Total var = 20

PC1: explained_variance_ = 8.5
     explained_variance_ratio_ = 8.5/20 = 0.425

Interpretation: PC1 captures 42.5% of
the total information in the data.


CUMULATIVE VARIANCE:
-------------------
Sum of explained_variance_ratio_ for
first k components.

If cumsum[4] = 0.90, then first 5 PCs
explain 90% of the total variance.


INFORMATION LOSS:
-----------------
If you keep k components with 95% cumvar,
you lose 5% of the variance (information).
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Explained Variance Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Explained variance ratio
ax2 = axes[0, 1]

n_components = len(pca_full.explained_variance_ratio_)
x = range(1, n_components + 1)

bars = ax2.bar(x, pca_full.explained_variance_ratio_ * 100, color=MLBLUE,
               edgecolor='black', alpha=0.7)

# Color first few components
for i in range(5):
    bars[i].set_color(MLGREEN)

ax2.set_title('Explained Variance Ratio per Component', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Explained Variance (%)')
ax2.grid(alpha=0.3, axis='y')

# Annotate top components
for i in range(3):
    ax2.text(i+1, pca_full.explained_variance_ratio_[i]*100 + 1,
             f'{pca_full.explained_variance_ratio_[i]*100:.1f}%',
             ha='center', fontsize=8, fontweight='bold')

# Plot 3: Cumulative variance
ax3 = axes[1, 0]

cumsum = np.cumsum(pca_full.explained_variance_ratio_) * 100

ax3.plot(x, cumsum, 'o-', color=MLBLUE, linewidth=2, markersize=6)
ax3.fill_between(x, cumsum, alpha=0.2, color=MLBLUE)

# Threshold lines
ax3.axhline(95, color=MLGREEN, linestyle='--', linewidth=2, label='95% threshold')
ax3.axhline(90, color=MLORANGE, linestyle='--', linewidth=2, label='90% threshold')
ax3.axhline(80, color=MLRED, linestyle='--', linewidth=2, label='80% threshold')

# Find component counts
k_95 = np.argmax(cumsum >= 95) + 1
k_90 = np.argmax(cumsum >= 90) + 1
k_80 = np.argmax(cumsum >= 80) + 1

ax3.scatter([k_80, k_90, k_95], [80, 90, 95], c=[MLRED, MLORANGE, MLGREEN],
            s=100, zorder=5, edgecolors='black')

ax3.text(k_95 + 0.5, 93, f'k={k_95}', fontsize=9, color=MLGREEN)
ax3.text(k_90 + 0.5, 88, f'k={k_90}', fontsize=9, color=MLORANGE)
ax3.text(k_80 + 0.5, 78, f'k={k_80}', fontsize=9, color=MLRED)

ax3.set_title('Cumulative Explained Variance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Number of Components')
ax3.set_ylabel('Cumulative Variance (%)')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)
ax3.set_ylim(0, 105)

# Plot 4: Practical interpretation
ax4 = axes[1, 1]
ax4.axis('off')

interpretation = f'''
PRACTICAL INTERPRETATION

OUR DATA: 20 features -> {len(pca_full.components_)} components

SUMMARY:
--------
Components for 80% variance: {k_80}
Components for 90% variance: {k_90}
Components for 95% variance: {k_95}

Reduction: 20D -> {k_95}D (for 95%)


WHAT THIS MEANS:
----------------
- Original: 20 features
- After PCA (95%): {k_95} features
- Compression: {(1 - k_95/20)*100:.0f}% fewer features
- Information retained: 95%


TRADE-OFF:
----------
More components -> More variance retained
                   More complex
                   Less compression

Fewer components -> More compression
                    More information loss
                    Simpler model


RECOMMENDATION:
---------------
1. For visualization: 2-3 components
2. For ML preprocessing: 90-95% variance
3. For noise reduction: 80-90% variance
'''

ax4.text(0.02, 0.98, interpretation, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Practical Interpretation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
