"""Centroid Interpretation - Understanding Clusters"""
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
fig.suptitle('Interpreting Cluster Centroids', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Finance example - stock characteristics
ax1 = axes[0, 0]

# Simulated centroid values for 4 stock clusters
features = ['Volatility', 'Avg Return', 'P/E Ratio', 'Market Cap\n(log)', 'Volume\n(log)']
centroids_scaled = np.array([
    [0.2, 0.3, 0.6, 0.9, 0.7],   # Cluster 0: Large cap, stable
    [0.9, 0.7, 0.3, 0.2, 0.4],   # Cluster 1: Small cap, volatile
    [0.3, 0.8, 0.4, 0.6, 0.5],   # Cluster 2: Growth stocks
    [0.4, 0.1, 0.8, 0.7, 0.3],   # Cluster 3: Value stocks
])

colors = [MLBLUE, MLRED, MLGREEN, MLORANGE]
cluster_names = ['Large Cap\nStable', 'Small Cap\nVolatile', 'Growth', 'Value']

x = np.arange(len(features))
width = 0.2

for i, (centroid, color, name) in enumerate(zip(centroids_scaled, colors, cluster_names)):
    ax1.bar(x + i*width, centroid, width, label=name, color=color, edgecolor='black')

ax1.set_xticks(x + 1.5*width)
ax1.set_xticklabels(features, fontsize=8)
ax1.set_title('Stock Cluster Profiles', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Scaled Value (0-1)')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(alpha=0.3, axis='y')

# Plot 2: Radar chart concept
ax2 = axes[0, 1]

# Create radar chart for one cluster
categories = features
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Plot each cluster
for i, (centroid, color, name) in enumerate(zip(centroids_scaled, colors, cluster_names)):
    values = list(centroid) + [centroid[0]]
    ax2.plot(angles, values, 'o-', linewidth=2, color=color, label=name.replace('\n', ' '))
    ax2.fill(angles, values, alpha=0.1, color=color)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(features, fontsize=8)
ax2.set_ylim(0, 1)
ax2.set_title('Radar Chart Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1))
ax2.grid(alpha=0.3)

# Plot 3: Interpretation guidelines
ax3 = axes[1, 0]
ax3.axis('off')

guidelines = '''
INTERPRETING CENTROIDS

WHAT CENTROIDS TELL YOU:
------------------------
- Center (average) of each cluster
- Typical characteristics of cluster members
- What makes clusters different


INTERPRETATION PROCESS:
-----------------------
1. Scale features first (StandardScaler)
2. Run K-Means
3. Get centroids: kmeans.cluster_centers_
4. Inverse transform to original scale
5. Compare centroid values across clusters


NAMING CLUSTERS:
----------------
Look at distinguishing features:

Example (Stocks):
- High volatility + High return -> "Growth Stocks"
- Low volatility + High market cap -> "Blue Chips"
- High P/E + Moderate return -> "Value Stocks"


IMPORTANT CONSIDERATIONS:
-------------------------
1. Use domain knowledge for interpretation
2. Check cluster sizes (avoid tiny clusters)
3. Validate with subject matter experts
4. Centroids are AVERAGES - variation exists!


DON'T:
------
- Over-interpret small differences
- Assume all cluster members are identical
- Ignore outliers within clusters
'''

ax3.text(0.02, 0.98, guidelines, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Interpretation Guidelines', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Code for interpretation
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
CENTROID INTERPRETATION CODE

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Prepare data
features = ['volatility', 'avg_return', 'pe_ratio', 'market_cap']
X = df[features].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)


# Get centroids in ORIGINAL scale
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Create interpretable DataFrame
centroid_df = pd.DataFrame(
    centroids_original,
    columns=features,
    index=['Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3']
)
print(centroid_df.round(2))


# Cluster statistics
cluster_stats = df.groupby('cluster').agg({
    'volatility': ['mean', 'std'],
    'avg_return': ['mean', 'std'],
    'ticker': 'count'  # Count per cluster
}).round(3)
print(cluster_stats)


# Name clusters based on characteristics
cluster_names = {
    0: 'Blue Chips',
    1: 'Growth Stocks',
    2: 'Value Plays',
    3: 'High Risk'
}
df['cluster_name'] = df['cluster'].map(cluster_names)
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
