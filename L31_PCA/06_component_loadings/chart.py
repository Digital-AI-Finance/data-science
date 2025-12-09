"""Component Loadings - Interpreting Principal Components"""
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
fig.suptitle('Component Loadings: Interpreting Principal Components', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What are loadings?
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT ARE COMPONENT LOADINGS?

DEFINITION:
-----------
Loadings = weights of original features
           in each principal component.

pca.components_ shape: (n_components, n_features)

Each row = one PC
Each column = one original feature


INTERPRETATION:
---------------
High positive loading: Feature contributes
                      positively to PC

High negative loading: Feature contributes
                      negatively (opposite)

Near-zero loading: Feature doesn't matter
                   for this PC


EXAMPLE (Stock Data):
---------------------
PC1 loadings: [0.4, 0.35, 0.3, 0.35, 0.4, ...]
              AAPL  MSFT  GOOGL AMZN  META

All positive, similar magnitude
-> PC1 = "Market factor" (all stocks move together)

PC2 loadings: [0.5, 0.4, 0.3, -0.4, -0.3, ...]
              Tech stocks  ...  Oil stocks

Positive for tech, negative for oil
-> PC2 = "Tech vs Energy factor"
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Loadings Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Loading heatmap
ax2 = axes[0, 1]

# Simulated loadings for financial data
features = ['Return', 'Volatility', 'Volume', 'P/E', 'Market Cap', 'Debt/Equity']
n_features = len(features)
n_pcs = 4

# Create realistic-looking loadings
loadings = np.array([
    [0.5, 0.3, 0.4, 0.3, 0.5, 0.2],    # PC1: Size/Quality
    [0.1, 0.6, -0.1, -0.5, 0.2, 0.5],  # PC2: Risk
    [-0.3, 0.2, 0.7, 0.3, -0.1, 0.4],  # PC3: Liquidity
    [0.4, -0.2, 0.1, 0.5, -0.3, -0.5], # PC4: Value
])

im = ax2.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-0.7, vmax=0.7)

ax2.set_xticks(range(n_features))
ax2.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(n_pcs))
ax2.set_yticklabels([f'PC{i+1}' for i in range(n_pcs)])

# Add text annotations
for i in range(n_pcs):
    for j in range(n_features):
        color = 'white' if abs(loadings[i, j]) > 0.4 else 'black'
        ax2.text(j, i, f'{loadings[i, j]:.2f}', ha='center', va='center',
                fontsize=8, color=color)

ax2.set_title('Loadings Heatmap', fontsize=11, fontweight='bold', color=MLPURPLE)
plt.colorbar(im, ax=ax2, shrink=0.8, label='Loading')

# Plot 3: Bar chart for one component
ax3 = axes[1, 0]

# PC1 loadings
pc1_loadings = loadings[0]

colors = [MLGREEN if l > 0 else MLRED for l in pc1_loadings]
bars = ax3.barh(range(n_features), pc1_loadings, color=colors, edgecolor='black')

ax3.set_yticks(range(n_features))
ax3.set_yticklabels(features)
ax3.axvline(0, color='black', linewidth=1)
ax3.set_title('PC1 Loadings (Interpretation)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Loading')
ax3.grid(alpha=0.3, axis='x')

# Add interpretation
ax3.text(0.55, 2, 'PC1 = "Market Quality"\nAll positive: stocks that\ndo well together', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 4: Code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
ACCESSING AND INTERPRETING LOADINGS

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Fit PCA
pca = PCA(n_components=4)
pca.fit(X_scaled)


# Get loadings
loadings = pca.components_
print(f"Shape: {loadings.shape}")
# (n_components, n_features)


# Create interpretable DataFrame
loading_df = pd.DataFrame(
    loadings.T,
    index=feature_names,
    columns=[f'PC{i+1}' for i in range(4)]
)
print(loading_df.round(3))


# Find most important features for PC1
pc1_loadings = pd.Series(loadings[0], index=feature_names)
print("\\nTop features for PC1:")
print(pc1_loadings.abs().sort_values(ascending=False)[:5])


# Visualize loadings
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(loading_df, cmap='RdBu_r', center=0, annot=True)
plt.title('PCA Component Loadings')
plt.tight_layout()
plt.show()


# Name components based on loadings
# PC1: All positive -> "Market factor"
# PC2: Tech+, Oil- -> "Tech vs Energy"
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
