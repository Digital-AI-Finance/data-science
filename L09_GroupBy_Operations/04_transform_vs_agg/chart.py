"""Transform vs Aggregate - Understanding the difference"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
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

# Sample data
data = {
    'Stock': ['AAPL', 'MSFT', 'NVDA', 'JPM', 'GS'],
    'Sector': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance'],
    'Return': [0.05, 0.08, 0.12, 0.03, 0.06]
}
df = pd.DataFrame(data)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left panel: Aggregate
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.text(5, 9.5, 'agg() - Aggregation', ha='center', fontsize=14,
         fontweight='bold', color=MLBLUE)
ax1.text(5, 8.8, 'Reduces each group to a single value', ha='center',
         fontsize=10, style='italic', color='gray')

# Original data
orig_box = FancyBboxPatch((0.5, 5.5), 4, 3, boxstyle="round,pad=0.1",
                          edgecolor=MLPURPLE, facecolor='white', linewidth=2)
ax1.add_patch(orig_box)
ax1.text(2.5, 8.2, 'Original (5 rows)', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)

headers = ['Stock', 'Sector', 'Return']
for i, h in enumerate(headers):
    ax1.text(1 + i*1.3, 7.7, h, ha='center', fontsize=8, fontweight='bold', color='gray')
for i, (stock, sector, ret) in enumerate(zip(df['Stock'], df['Sector'], df['Return'])):
    y = 7.3 - i*0.35
    ax1.text(1, y, stock, ha='center', fontsize=8)
    ax1.text(2.3, y, sector, ha='center', fontsize=8)
    ax1.text(3.6, y, f'{ret:.0%}', ha='center', fontsize=8)

# Arrow
ax1.annotate('', xy=(7, 6.5), xytext=(5, 6.5),
            arrowprops=dict(arrowstyle='->', color=MLBLUE, lw=3))
ax1.text(6, 7, '.groupby("Sector")\n["Return"].mean()', ha='center',
         fontsize=9, family='monospace', color=MLBLUE)

# Result
result_box = FancyBboxPatch((5.5, 5.5), 4, 1.8, boxstyle="round,pad=0.1",
                            edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax1.add_patch(result_box)
ax1.text(7.5, 7, 'Result (2 rows)', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
ax1.text(6.3, 6.5, 'Sector', ha='center', fontsize=8, fontweight='bold', color='gray')
ax1.text(8.3, 6.5, 'Mean', ha='center', fontsize=8, fontweight='bold', color='gray')
ax1.text(6.3, 6.1, 'Tech', ha='center', fontsize=9)
ax1.text(8.3, 6.1, '8.3%', ha='center', fontsize=9, fontweight='bold', color=MLBLUE)
ax1.text(6.3, 5.7, 'Finance', ha='center', fontsize=9)
ax1.text(8.3, 5.7, '4.5%', ha='center', fontsize=9, fontweight='bold', color=MLBLUE)

# Use case
ax1.text(5, 4.5, 'Use Case: Summary statistics, reports', ha='center',
         fontsize=10, color=MLBLUE, fontweight='bold')

# Code example
code_box = FancyBboxPatch((1, 2.5), 8, 1.5, boxstyle="round,pad=0.1",
                          edgecolor=MLBLUE, facecolor=MLLAVENDER, alpha=0.2)
ax1.add_patch(code_box)
ax1.text(5, 3.5, "sector_avg = df.groupby('Sector')['Return'].mean()", ha='center',
         fontsize=9, family='monospace')

# Right panel: Transform
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

ax2.text(5, 9.5, 'transform() - Transformation', ha='center', fontsize=14,
         fontweight='bold', color=MLORANGE)
ax2.text(5, 8.8, 'Returns same shape as input', ha='center',
         fontsize=10, style='italic', color='gray')

# Original data (same)
orig_box2 = FancyBboxPatch((0.5, 5.5), 4, 3, boxstyle="round,pad=0.1",
                           edgecolor=MLPURPLE, facecolor='white', linewidth=2)
ax2.add_patch(orig_box2)
ax2.text(2.5, 8.2, 'Original (5 rows)', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)

for i, h in enumerate(headers):
    ax2.text(1 + i*1.3, 7.7, h, ha='center', fontsize=8, fontweight='bold', color='gray')
for i, (stock, sector, ret) in enumerate(zip(df['Stock'], df['Sector'], df['Return'])):
    y = 7.3 - i*0.35
    ax2.text(1, y, stock, ha='center', fontsize=8)
    ax2.text(2.3, y, sector, ha='center', fontsize=8)
    ax2.text(3.6, y, f'{ret:.0%}', ha='center', fontsize=8)

# Arrow
ax2.annotate('', xy=(7, 6.5), xytext=(5, 6.5),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=3))
ax2.text(6, 7, '.groupby("Sector")\n["Return"].transform("mean")', ha='center',
         fontsize=9, family='monospace', color=MLORANGE)

# Result - same rows
result_box2 = FancyBboxPatch((5.5, 4.5), 4, 3, boxstyle="round,pad=0.1",
                             edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=2)
ax2.add_patch(result_box2)
ax2.text(7.5, 7.2, 'Result (5 rows)', ha='center', fontsize=10, fontweight='bold', color=MLORANGE)
ax2.text(6.3, 6.7, 'Stock', ha='center', fontsize=8, fontweight='bold', color='gray')
ax2.text(8.3, 6.7, 'SectorMean', ha='center', fontsize=8, fontweight='bold', color='gray')

transformed = ['8.3%', '8.3%', '8.3%', '4.5%', '4.5%']
for i, (stock, val) in enumerate(zip(df['Stock'], transformed)):
    y = 6.3 - i*0.35
    ax2.text(6.3, y, stock, ha='center', fontsize=8)
    ax2.text(8.3, y, val, ha='center', fontsize=9, fontweight='bold', color=MLORANGE)

# Use case
ax2.text(5, 3.8, 'Use Case: Adding group-level stats to each row', ha='center',
         fontsize=10, color=MLORANGE, fontweight='bold')

# Code example
code_box2 = FancyBboxPatch((0.5, 2.2), 9, 1.5, boxstyle="round,pad=0.1",
                           edgecolor=MLORANGE, facecolor='#FFF5E6', alpha=0.3)
ax2.add_patch(code_box2)
ax2.text(5, 3.2, "df['SectorMean'] = df.groupby('Sector')['Return'].transform('mean')", ha='center',
         fontsize=8, family='monospace')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
