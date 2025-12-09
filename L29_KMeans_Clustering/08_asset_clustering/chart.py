"""Asset Clustering - Complete Finance Example"""
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
fig.suptitle('Asset Clustering: Complete Finance Example', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate synthetic stock data
n_stocks = 50
tickers = [f'STK{i:02d}' for i in range(n_stocks)]

# Create realistic clusters
# Cluster 0: Large cap, stable (10 stocks)
volatility_0 = np.random.uniform(0.15, 0.25, 10)
returns_0 = np.random.uniform(0.05, 0.10, 10)

# Cluster 1: Small cap, volatile (15 stocks)
volatility_1 = np.random.uniform(0.35, 0.50, 15)
returns_1 = np.random.uniform(0.10, 0.25, 15)

# Cluster 2: Medium growth (15 stocks)
volatility_2 = np.random.uniform(0.25, 0.35, 15)
returns_2 = np.random.uniform(0.08, 0.15, 15)

# Cluster 3: Low return, moderate vol (10 stocks)
volatility_3 = np.random.uniform(0.20, 0.30, 10)
returns_3 = np.random.uniform(0.02, 0.06, 10)

volatility = np.concatenate([volatility_0, volatility_1, volatility_2, volatility_3])
returns = np.concatenate([returns_0, returns_1, returns_2, returns_3])
true_labels = np.array([0]*10 + [1]*15 + [2]*15 + [3]*10)

X = np.column_stack([volatility, returns])

# Apply K-Means
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

colors = [MLBLUE, MLRED, MLGREEN, MLORANGE]
cluster_names = ['Blue Chips', 'High Risk', 'Growth', 'Defensive']

# Plot 1: Risk-Return scatter
ax1 = axes[0, 0]

for i in range(4):
    mask = labels == i
    ax1.scatter(volatility[mask]*100, returns[mask]*100, c=colors[i], s=60,
                alpha=0.7, label=cluster_names[i], edgecolors='black', linewidths=0.5)

# Plot centroids (inverse transformed)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax1.scatter(centroids_original[:, 0]*100, centroids_original[:, 1]*100,
            c='black', s=200, marker='X', edgecolors='white', linewidths=2,
            zorder=5, label='Centroids')

ax1.set_title('Stock Clusters: Risk vs Return', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Volatility (%)')
ax1.set_ylabel('Expected Return (%)')
ax1.legend(fontsize=7, loc='upper left')
ax1.grid(alpha=0.3)

# Plot 2: Cluster distribution
ax2 = axes[0, 1]

cluster_counts = np.bincount(labels)
bars = ax2.bar(range(4), cluster_counts, color=colors, edgecolor='black')

ax2.set_xticks(range(4))
ax2.set_xticklabels(cluster_names, fontsize=9)
ax2.set_title('Stocks per Cluster', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Number of Stocks')
ax2.grid(alpha=0.3, axis='y')

# Add count labels
for bar, count in zip(bars, cluster_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(count), ha='center', fontsize=10, fontweight='bold')

# Plot 3: Cluster characteristics table
ax3 = axes[1, 0]
ax3.axis('off')

# Calculate stats per cluster
stats_text = '''
CLUSTER CHARACTERISTICS

Cluster         |  N  | Volatility | Return | Sharpe
----------------|-----|------------|--------|--------
'''

for i in range(4):
    mask = labels == i
    vol = volatility[mask].mean() * 100
    ret = returns[mask].mean() * 100
    sharpe = (ret - 2) / vol  # Assuming 2% risk-free rate
    n = mask.sum()
    stats_text += f'{cluster_names[i]:15} | {n:3d} |   {vol:5.1f}%   |  {ret:4.1f}% |  {sharpe:.2f}\n'

stats_text += '''
----------------|-----|------------|--------|--------

INTERPRETATION:
---------------
Blue Chips:   Low risk, steady returns
              Large-cap, established companies

High Risk:    High volatility, high return potential
              Small-cap, speculative

Growth:       Moderate risk, good returns
              Mid-cap growth stocks

Defensive:    Low returns, moderate volatility
              Utilities, consumer staples
'''

ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Cluster Statistics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete pipeline code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
COMPLETE ASSET CLUSTERING PIPELINE

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. LOAD DATA
df = pd.read_csv('stock_characteristics.csv')
features = ['volatility', 'avg_return', 'pe_ratio', 'market_cap']
X = df[features].values

# 2. SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. FIND OPTIMAL K
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}: Silhouette = {score:.4f}")

# 4. FIT FINAL MODEL
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. INTERPRET CLUSTERS
cluster_stats = df.groupby('cluster')[features].mean()
print(cluster_stats.round(3))

# 6. NAME CLUSTERS
names = {0: 'Blue Chips', 1: 'Growth', 2: 'Value', 3: 'Speculative'}
df['cluster_name'] = df['cluster'].map(names)

# 7. PORTFOLIO USE
# Select one stock from each cluster for diversification
diversified = df.groupby('cluster').apply(
    lambda x: x.nlargest(1, 'sharpe_ratio')
)
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Pipeline', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
