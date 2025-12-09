"""Scree Plot - Choosing Number of Components"""
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
fig.suptitle('Scree Plot: Choosing the Number of Principal Components', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate example eigenvalues
eigenvalues = np.array([5.5, 2.3, 1.2, 0.7, 0.4, 0.25, 0.15, 0.1, 0.08, 0.05])
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot 1: Basic scree plot
ax1 = axes[0, 0]

ax1.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', color=MLBLUE, linewidth=2, markersize=8)

# Highlight elbow
ax1.scatter([3], [eigenvalues[2]], c=MLRED, s=200, zorder=5, edgecolors='black', linewidths=2)
ax1.axvline(3, color=MLRED, linestyle='--', alpha=0.5, label='Elbow at k=3')

ax1.set_title('Scree Plot (Eigenvalues)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Eigenvalue')
ax1.set_xticks(range(1, len(eigenvalues)+1))
ax1.legend()
ax1.grid(alpha=0.3)

# Add annotation
ax1.annotate('Elbow point', xy=(3, eigenvalues[2]), xytext=(5, eigenvalues[2]+1),
             fontsize=10, arrowprops=dict(arrowstyle='->', color=MLRED))

# Plot 2: Cumulative variance plot
ax2 = axes[0, 1]

ax2.bar(range(1, len(eigenvalues)+1), explained_variance_ratio * 100, color=MLBLUE,
        edgecolor='black', alpha=0.7, label='Individual')
ax2.plot(range(1, len(eigenvalues)+1), cumulative_variance * 100, 'o-', color=MLRED,
         linewidth=2, markersize=8, label='Cumulative')

# Threshold lines
ax2.axhline(95, color=MLGREEN, linestyle='--', alpha=0.7, label='95% threshold')
ax2.axhline(90, color=MLORANGE, linestyle=':', alpha=0.7, label='90% threshold')

# Find where we reach 95%
k_95 = np.argmax(cumulative_variance >= 0.95) + 1
ax2.axvline(k_95, color=MLGREEN, linestyle='--', alpha=0.5)

ax2.set_title('Explained Variance Ratio', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Explained Variance (%)')
ax2.set_xticks(range(1, len(eigenvalues)+1))
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis='y')

ax2.text(k_95+0.2, 70, f'k={k_95} for\n95% var', fontsize=9, color=MLGREEN)

# Plot 3: Methods to choose k
ax3 = axes[1, 0]
ax3.axis('off')

methods = '''
METHODS TO CHOOSE NUMBER OF COMPONENTS

1. ELBOW METHOD (Scree Plot)
   --------------------------
   Look for "elbow" where eigenvalues
   start to level off.
   Keep components before the elbow.


2. VARIANCE THRESHOLD
   -------------------
   Keep enough components to explain
   a target variance (e.g., 95%).

   cumsum = np.cumsum(pca.explained_variance_ratio_)
   k = np.argmax(cumsum >= 0.95) + 1


3. KAISER CRITERION
   -----------------
   Keep components with eigenvalue >= 1
   (for standardized data).

   k = np.sum(pca.explained_variance_ >= 1)


4. CROSS-VALIDATION
   -----------------
   Use downstream task performance
   to select optimal k.


RECOMMENDATION:
---------------
- Start with variance threshold (95%)
- Check scree plot for natural elbow
- Validate with downstream task
- Consider interpretability needs
'''

ax3.text(0.02, 0.98, methods, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Selection Methods', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Code for scree plot
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
SCREE PLOT CODE

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Fit PCA (keep all components)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)


# Scree plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Eigenvalues
ax1.plot(range(1, len(pca.explained_variance_)+1),
         pca.explained_variance_, 'bo-')
ax1.set_xlabel('Component')
ax1.set_ylabel('Eigenvalue')
ax1.set_title('Scree Plot')

# Cumulative variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
ax2.bar(range(1, len(cumsum)+1),
        pca.explained_variance_ratio_ * 100, alpha=0.7)
ax2.plot(range(1, len(cumsum)+1), cumsum * 100, 'ro-')
ax2.axhline(95, color='g', linestyle='--')
ax2.set_xlabel('Component')
ax2.set_ylabel('Variance Explained (%)')
ax2.set_title('Cumulative Variance')

plt.tight_layout()
plt.show()


# Find k for 95% variance
k = np.argmax(cumsum >= 0.95) + 1
print(f"Need {k} components for 95% variance")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
