"""Aggregation Functions - Comparing different aggregation methods"""
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

# Create sample stock data
np.random.seed(42)
sectors = ['Tech', 'Finance', 'Healthcare', 'Energy']
n_per_sector = [15, 12, 10, 8]

data = []
for sector, n in zip(sectors, n_per_sector):
    if sector == 'Tech':
        returns = np.random.normal(0.08, 0.15, n)
    elif sector == 'Finance':
        returns = np.random.normal(0.05, 0.10, n)
    elif sector == 'Healthcare':
        returns = np.random.normal(0.06, 0.12, n)
    else:
        returns = np.random.normal(0.03, 0.18, n)

    for r in returns:
        data.append({'Sector': sector, 'Return': r})

df = pd.DataFrame(data)

# Calculate various aggregations
agg_results = df.groupby('Sector')['Return'].agg(['mean', 'std', 'min', 'max', 'count'])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GroupBy Aggregation Functions', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Mean by sector
ax1 = axes[0, 0]
colors = [MLBLUE, MLORANGE, MLGREEN, MLRED]
bars1 = ax1.bar(agg_results.index, agg_results['mean'] * 100, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Mean Return (%)', fontsize=10)
ax1.set_title(".groupby('Sector')['Return'].mean()", fontsize=11, color=MLBLUE, family='monospace')
ax1.axhline(0, color='black', linewidth=0.5)
ax1.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

# Plot 2: Standard deviation
ax2 = axes[0, 1]
bars2 = ax2.bar(agg_results.index, agg_results['std'] * 100, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Std Deviation (%)', fontsize=10)
ax2.set_title(".groupby('Sector')['Return'].std()", fontsize=11, color=MLORANGE, family='monospace')
ax2.grid(axis='y', alpha=0.3)
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

# Plot 3: Min-Max range
ax3 = axes[1, 0]
x = np.arange(len(sectors))
width = 0.35
bars_min = ax3.bar(x - width/2, agg_results['min'] * 100, width, label='Min', color=MLRED, alpha=0.7)
bars_max = ax3.bar(x + width/2, agg_results['max'] * 100, width, label='Max', color=MLGREEN, alpha=0.7)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.set_title(".groupby('Sector')['Return'].agg(['min', 'max'])", fontsize=10, color=MLGREEN, family='monospace')
ax3.set_xticks(x)
ax3.set_xticklabels(sectors)
ax3.axhline(0, color='black', linewidth=0.5)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Count
ax4 = axes[1, 1]
bars4 = ax4.bar(agg_results.index, agg_results['count'], color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title(".groupby('Sector')['Return'].count()", fontsize=11, color=MLPURPLE, family='monospace')
ax4.grid(axis='y', alpha=0.3)
for bar in bars4:
    height = bar.get_height()
    ax4.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

# Add summary text
total = agg_results['count'].sum()
ax4.text(0.5, 0.95, f'Total stocks: {int(total)}', transform=ax4.transAxes,
         fontsize=9, va='top', ha='center', style='italic', color='gray')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
