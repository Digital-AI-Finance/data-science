"""Multi-Column GroupBy - Grouping by multiple columns"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

# Create sample data
np.random.seed(42)
sectors = ['Tech', 'Tech', 'Finance', 'Finance']
sizes = ['Large', 'Small', 'Large', 'Small']
n_per = 20

data = []
for sector, size in zip(sectors, sizes):
    base_return = 0.08 if sector == 'Tech' else 0.05
    vol = 0.15 if size == 'Small' else 0.10
    returns = np.random.normal(base_return, vol, n_per)
    for r in returns:
        data.append({'Sector': sector, 'Size': size, 'Return': r})

df = pd.DataFrame(data)

# Multi-column groupby
multi_group = df.groupby(['Sector', 'Size'])['Return'].agg(['mean', 'std', 'count'])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Multi-Column GroupBy: df.groupby(['Sector', 'Size'])", fontsize=14,
             fontweight='bold', color=MLPURPLE)

# Plot 1: Grouped bar chart - Mean returns
ax1 = axes[0, 0]
x = np.arange(2)  # Tech, Finance
width = 0.35
large_means = [multi_group.loc[('Tech', 'Large'), 'mean'] * 100,
               multi_group.loc[('Finance', 'Large'), 'mean'] * 100]
small_means = [multi_group.loc[('Tech', 'Small'), 'mean'] * 100,
               multi_group.loc[('Finance', 'Small'), 'mean'] * 100]

bars1 = ax1.bar(x - width/2, large_means, width, label='Large Cap', color=MLBLUE, alpha=0.7)
bars2 = ax1.bar(x + width/2, small_means, width, label='Small Cap', color=MLORANGE, alpha=0.7)

ax1.set_ylabel('Mean Return (%)', fontsize=10)
ax1.set_title('Mean Return by Sector and Size', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xticks(x)
ax1.set_xticklabels(['Tech', 'Finance'])
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.5)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

# Plot 2: Volatility comparison
ax2 = axes[0, 1]
large_std = [multi_group.loc[('Tech', 'Large'), 'std'] * 100,
             multi_group.loc[('Finance', 'Large'), 'std'] * 100]
small_std = [multi_group.loc[('Tech', 'Small'), 'std'] * 100,
             multi_group.loc[('Finance', 'Small'), 'std'] * 100]

bars3 = ax2.bar(x - width/2, large_std, width, label='Large Cap', color=MLBLUE, alpha=0.7)
bars4 = ax2.bar(x + width/2, small_std, width, label='Small Cap', color=MLORANGE, alpha=0.7)

ax2.set_ylabel('Std Deviation (%)', fontsize=10)
ax2.set_title('Volatility by Sector and Size', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xticks(x)
ax2.set_xticklabels(['Tech', 'Finance'])
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Heatmap style
ax3 = axes[1, 0]
pivot = df.pivot_table(values='Return', index='Sector', columns='Size', aggfunc='mean') * 100
im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=15)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Large', 'Small'])
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Finance', 'Tech'])
ax3.set_title('Mean Return Heatmap (%)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add text annotations
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{pivot.iloc[i, j]:.1f}%', ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')
plt.colorbar(im, ax=ax3, label='Return (%)')

# Plot 4: Count by group
ax4 = axes[1, 1]
counts = multi_group['count'].values
labels = ['Tech\nLarge', 'Tech\nSmall', 'Finance\nLarge', 'Finance\nSmall']
colors = [MLBLUE, MLBLUE, MLORANGE, MLORANGE]
alphas = [0.9, 0.5, 0.9, 0.5]

bars5 = ax4.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
for bar, alpha in zip(bars5, alphas):
    bar.set_alpha(alpha)

ax4.set_ylabel('Count', fontsize=10)
ax4.set_title('Observations per Group', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(axis='y', alpha=0.3)

for bar in bars5:
    height = bar.get_height()
    ax4.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
