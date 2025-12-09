"""Correlation-Based Clustering - Financial Assets"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

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
fig.suptitle('Correlation-Based Hierarchical Clustering for Assets', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate synthetic stock return data
n_days = 252
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
           'JPM', 'BAC', 'GS', 'C',  # Banks
           'XOM', 'CVX', 'COP',  # Energy
           'JNJ', 'PFE', 'MRK']  # Healthcare

n_stocks = len(tickers)

# Create correlated returns
market = np.random.randn(n_days) * 0.01

# Tech stocks (high correlation within group)
tech_factor = np.random.randn(n_days) * 0.01
tech_returns = [market + tech_factor + np.random.randn(n_days) * 0.005 for _ in range(5)]

# Bank stocks
bank_factor = np.random.randn(n_days) * 0.01
bank_returns = [market + bank_factor + np.random.randn(n_days) * 0.005 for _ in range(4)]

# Energy stocks
energy_factor = np.random.randn(n_days) * 0.015
energy_returns = [market + energy_factor + np.random.randn(n_days) * 0.006 for _ in range(3)]

# Healthcare
health_factor = np.random.randn(n_days) * 0.008
health_returns = [market + health_factor + np.random.randn(n_days) * 0.005 for _ in range(3)]

returns = np.column_stack(tech_returns + bank_returns + energy_returns + health_returns)

# Calculate correlation matrix
corr_matrix = np.corrcoef(returns.T)

# Plot 1: Correlation heatmap
ax1 = axes[0, 0]

im = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax1.set_xticks(range(n_stocks))
ax1.set_yticks(range(n_stocks))
ax1.set_xticklabels(tickers, rotation=45, ha='right', fontsize=7)
ax1.set_yticklabels(tickers, fontsize=7)
ax1.set_title('Correlation Matrix (Before Clustering)', fontsize=11, fontweight='bold', color=MLPURPLE)
plt.colorbar(im, ax=ax1, shrink=0.8, label='Correlation')

# Plot 2: Dendrogram using correlation distance
ax2 = axes[0, 1]

# Convert correlation to distance: d = 1 - corr
dist_matrix = 1 - corr_matrix
np.fill_diagonal(dist_matrix, 0)

# Use condensed form for linkage
dist_condensed = squareform(dist_matrix)
Z = linkage(dist_condensed, method='average')

dendrogram(Z, ax=ax2, labels=tickers, leaf_rotation=45, leaf_font_size=8)
ax2.set_title('Dendrogram (Correlation Distance)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Distance (1 - correlation)')

# Add sector boxes
ax2.axhline(y=0.4, color=MLRED, linestyle='--', alpha=0.5)

# Plot 3: Reordered correlation matrix
ax3 = axes[1, 0]

# Get ordering from dendrogram
from scipy.cluster.hierarchy import leaves_list
order = leaves_list(Z)

# Reorder correlation matrix
corr_reordered = corr_matrix[order][:, order]
tickers_reordered = [tickers[i] for i in order]

im3 = ax3.imshow(corr_reordered, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax3.set_xticks(range(n_stocks))
ax3.set_yticks(range(n_stocks))
ax3.set_xticklabels(tickers_reordered, rotation=45, ha='right', fontsize=7)
ax3.set_yticklabels(tickers_reordered, fontsize=7)
ax3.set_title('Correlation Matrix (After Clustering)', fontsize=11, fontweight='bold', color=MLGREEN)
plt.colorbar(im3, ax=ax3, shrink=0.8, label='Correlation')

# Plot 4: Code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
CORRELATION-BASED CLUSTERING

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# 1. COMPUTE RETURNS
returns = prices.pct_change().dropna()

# 2. CORRELATION MATRIX
corr = returns.corr()

# 3. CONVERT TO DISTANCE
# Options:
# d = 1 - corr       (correlation distance)
# d = sqrt(2*(1-corr))  (angular distance)
dist = 1 - corr
np.fill_diagonal(dist.values, 0)

# 4. HIERARCHICAL CLUSTERING
dist_condensed = squareform(dist)
Z = linkage(dist_condensed, method='average')

# 5. PLOT DENDROGRAM
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=corr.columns, leaf_rotation=45)
plt.title('Asset Clusters by Return Correlation')
plt.ylabel('Distance (1 - correlation)')
plt.show()

# 6. GET CLUSTER LABELS
labels = fcluster(Z, t=4, criterion='maxclust')
cluster_df = pd.DataFrame({
    'ticker': corr.columns,
    'cluster': labels
})
print(cluster_df.sort_values('cluster'))


# USE CASE: DIVERSIFICATION
# Select one stock from each cluster for a diversified portfolio
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
