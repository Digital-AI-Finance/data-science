"""Portfolio Clustering - Complete Finance Application"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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
fig.suptitle('Portfolio Clustering: Complete Finance Application', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate synthetic portfolio data
n_days = 252
assets = ['SPY', 'QQQ', 'IWM',  # US Equity
          'EFA', 'VWO',  # International
          'TLT', 'IEF', 'AGG',  # Bonds
          'GLD', 'SLV',  # Commodities
          'VNQ', 'REIT']  # Real Estate

n_assets = len(assets)

# Create correlated returns by asset class
market = np.random.randn(n_days) * 0.01

# US Equity (correlated)
us_eq = [market + np.random.randn(n_days) * 0.008 for _ in range(3)]

# International (somewhat correlated with US)
intl = [market * 0.7 + np.random.randn(n_days) * 0.01 for _ in range(2)]

# Bonds (negatively correlated)
bonds = [-market * 0.3 + np.random.randn(n_days) * 0.004 for _ in range(3)]

# Commodities (low correlation)
commod = [np.random.randn(n_days) * 0.015 for _ in range(2)]

# Real Estate (moderate correlation)
reit = [market * 0.5 + np.random.randn(n_days) * 0.012 for _ in range(2)]

returns = np.column_stack(us_eq + intl + bonds + commod + reit)

# Calculate correlation and distance
corr_matrix = np.corrcoef(returns.T)
dist_matrix = 1 - corr_matrix
np.fill_diagonal(dist_matrix, 0)
dist_condensed = squareform(dist_matrix)
Z = linkage(dist_condensed, method='average')

# Plot 1: Dendrogram with asset labels
ax1 = axes[0, 0]

dendrogram(Z, ax=ax1, labels=assets, leaf_rotation=45, leaf_font_size=9)
ax1.set_title('Asset Hierarchy by Return Correlation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Distance (1 - correlation)')
ax1.axhline(y=0.6, color=MLRED, linestyle='--', linewidth=2, label='Cut for 4 clusters')
ax1.legend(fontsize=8)

# Plot 2: Cluster assignments
ax2 = axes[0, 1]

labels = fcluster(Z, t=4, criterion='maxclust')
cluster_names = {1: 'US/Intl Equity', 2: 'Bonds', 3: 'Commodities', 4: 'Real Estate'}

# Bar chart showing cluster membership
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]
cluster_counts = {i: [] for i in range(1, 5)}
for asset, label in zip(assets, labels):
    cluster_counts[label].append(asset)

y_pos = 0
for cluster_id, cluster_assets in cluster_counts.items():
    for asset in cluster_assets:
        ax2.barh(y_pos, 1, color=colors[cluster_id-1], edgecolor='black')
        ax2.text(0.5, y_pos, asset, ha='center', va='center', fontsize=9, fontweight='bold')
        y_pos += 1
    y_pos += 0.5  # Gap between clusters

ax2.set_xlim(0, 1)
ax2.set_ylim(-0.5, y_pos)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('Cluster Assignments', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add cluster labels
y_positions = [1.5, 5.5, 9, 11.5]
for i, (pos, name) in enumerate(zip(y_positions, cluster_names.values())):
    ax2.text(1.1, pos, name, fontsize=9, color=colors[i], fontweight='bold', va='center')

# Plot 3: Risk/Return by cluster
ax3 = axes[1, 0]

# Calculate annual stats
annual_returns = returns.mean(axis=0) * 252 * 100
annual_vol = returns.std(axis=0) * np.sqrt(252) * 100

for i in range(1, 5):
    mask = labels == i
    ax3.scatter(annual_vol[mask], annual_returns[mask],
                c=colors[i-1], s=100, alpha=0.8, label=cluster_names[i], edgecolors='black')

# Label each point
for j, asset in enumerate(assets):
    ax3.annotate(asset, (annual_vol[j], annual_returns[j]),
                 xytext=(3, 3), textcoords='offset points', fontsize=7)

ax3.set_title('Risk-Return by Cluster', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Volatility (%)')
ax3.set_ylabel('Expected Return (%)')
ax3.legend(fontsize=7, loc='upper left')
ax3.grid(alpha=0.3)

# Plot 4: Portfolio construction code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
DIVERSIFIED PORTFOLIO FROM CLUSTERS

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 1. LOAD PRICE DATA
prices = pd.read_csv('etf_prices.csv', index_col=0)
returns = prices.pct_change().dropna()

# 2. CLUSTER BY CORRELATION
corr = returns.corr()
dist = 1 - corr
Z = linkage(squareform(dist), method='average')

# 3. GET CLUSTER ASSIGNMENTS
n_clusters = 4
labels = fcluster(Z, t=n_clusters, criterion='maxclust')

cluster_df = pd.DataFrame({
    'asset': corr.columns,
    'cluster': labels,
    'return': returns.mean() * 252,
    'vol': returns.std() * np.sqrt(252),
    'sharpe': (returns.mean() * 252) / (returns.std() * np.sqrt(252))
})

# 4. SELECT BEST FROM EACH CLUSTER
portfolio = cluster_df.groupby('cluster').apply(
    lambda x: x.nlargest(1, 'sharpe')
)['asset'].tolist()

print(f"Diversified portfolio: {portfolio}")


# 5. EQUAL WEIGHT ALLOCATION
weights = {asset: 1/len(portfolio) for asset in portfolio}
print(f"Weights: {weights}")


# BENEFIT: Assets from different clusters
# have low correlation -> diversification!
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Portfolio Construction Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
