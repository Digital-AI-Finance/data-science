"""Seaborn Introduction - Enhanced matplotlib visualizations"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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
fig.suptitle('Seaborn: Statistical Visualization Made Easy', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: matplotlib vs seaborn comparison - histogram
ax1 = axes[0, 0]
returns = np.random.normal(0.05, 2, 500)

# Seaborn makes it easy
sns.histplot(returns, kde=True, ax=ax1, color=MLBLUE, alpha=0.7,
             edgecolor='black', linewidth=0.5)

ax1.set_title('Seaborn: histplot with KDE', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Return (%)', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)

# Plot 2: Built-in themes
ax2 = axes[0, 1]

# Create sample data
categories = ['Tech', 'Finance', 'Health', 'Energy', 'Consumer']
values = [15.2, 8.5, 12.1, 6.3, 9.8]

bars = ax2.bar(categories, values, color=[MLBLUE, MLGREEN, MLORANGE, MLRED, MLPURPLE],
               edgecolor='black', linewidth=0.5)

# Add value labels
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_title('Sector Returns with Seaborn Styling', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Return (%)', fontsize=10)
ax2.set_ylim(0, 18)

# Plot 3: Seaborn color palettes
ax3 = axes[1, 0]

# Show different palette options
palettes = ['deep', 'muted', 'pastel', 'bright', 'dark']
n_colors = 6

for i, palette in enumerate(palettes):
    colors = sns.color_palette(palette, n_colors)
    for j, color in enumerate(colors):
        ax3.barh(i, 1, left=j, color=color, height=0.8, edgecolor='white', linewidth=0.5)

ax3.set_yticks(range(len(palettes)))
ax3.set_yticklabels(palettes, fontsize=10)
ax3.set_xlim(0, n_colors)
ax3.set_xticks([])
ax3.set_title('Seaborn Color Palettes', fontsize=11, fontweight='bold', color=MLPURPLE)

# Add palette names annotation
ax3.text(n_colors/2, -0.8, 'Built-in palettes for consistent, attractive colors',
         ha='center', fontsize=9, style='italic')

# Plot 4: Integration with pandas DataFrames
ax4 = axes[1, 1]

# Create a finance dataset
df = pd.DataFrame({
    'Asset': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * 4,
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'] * 5,
    'Return': np.random.normal(3, 5, 20)
})

# Seaborn works directly with DataFrames
sns.barplot(data=df, x='Asset', y='Return', hue='Quarter', ax=ax4,
            palette=[MLBLUE, MLGREEN, MLORANGE, MLRED])

ax4.set_title('DataFrame Integration: Quarterly Returns', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Stock', fontsize=10)
ax4.set_ylabel('Return (%)', fontsize=10)
ax4.legend(title='Quarter', fontsize=8, title_fontsize=9, loc='upper right')
ax4.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
