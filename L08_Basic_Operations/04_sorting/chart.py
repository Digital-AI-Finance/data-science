"""Sorting Operations - Demonstrating sort_values() and sort_index()"""
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

# Create sample data
np.random.seed(42)
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
returns = np.random.uniform(-5, 8, len(stocks))
volumes = np.random.randint(1, 20, len(stocks)) * 1000000
pe_ratios = np.random.uniform(15, 45, len(stocks))

df = pd.DataFrame({
    'Stock': stocks,
    'Return': returns,
    'Volume': volumes,
    'PE_Ratio': pe_ratios
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sorting DataFrames with sort_values()', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Unsorted (original)
ax1 = axes[0, 0]
colors = [MLGREEN if r > 0 else MLRED for r in returns]
bars1 = ax1.barh(stocks, returns, color=colors, alpha=0.7, edgecolor='black')
ax1.axvline(0, color='black', linewidth=0.5)
ax1.set_xlabel('Return (%)', fontsize=10)
ax1.set_title('Original Order (Unsorted)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Sorted by returns (descending)
ax2 = axes[0, 1]
df_sorted = df.sort_values('Return', ascending=False)
colors_sorted = [MLGREEN if r > 0 else MLRED for r in df_sorted['Return']]
ax2.barh(df_sorted['Stock'], df_sorted['Return'], color=colors_sorted, alpha=0.7, edgecolor='black')
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlabel('Return (%)', fontsize=10)
ax2.set_title("df.sort_values('Return', ascending=False)", fontsize=10,
              fontweight='bold', color=MLBLUE, family='monospace')
ax2.grid(axis='x', alpha=0.3)

# Add rank annotations
for i, (stock, ret) in enumerate(zip(df_sorted['Stock'], df_sorted['Return'])):
    ax2.annotate(f'#{i+1}', xy=(ret, stock), xytext=(5, 0), textcoords='offset points',
                fontsize=8, color=MLPURPLE, fontweight='bold', va='center')

# Plot 3: Sorted by volume
ax3 = axes[1, 0]
df_vol_sorted = df.sort_values('Volume', ascending=True)
ax3.barh(df_vol_sorted['Stock'], df_vol_sorted['Volume']/1e6, color=MLBLUE, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Volume (Millions)', fontsize=10)
ax3.set_title("df.sort_values('Volume', ascending=True)", fontsize=10,
              fontweight='bold', color=MLBLUE, family='monospace')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Multi-column sort visualization
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

ax4.text(5, 9.5, 'Multi-Column Sorting', ha='center', fontsize=12,
         fontweight='bold', color=MLPURPLE)

# Code examples
examples = [
    ("# Sort by single column", "df.sort_values('Price')", MLBLUE),
    ("# Sort descending", "df.sort_values('Price', ascending=False)", MLORANGE),
    ("# Sort by multiple columns", "df.sort_values(['Sector', 'Return'],\n                  ascending=[True, False])", MLGREEN),
    ("# Sort by index", "df.sort_index()", MLRED),
    ("# Reset index after sort", "df.sort_values('Price').reset_index(drop=True)", MLPURPLE),
]

y_pos = 8.5
for comment, code, color in examples:
    box = FancyBboxPatch((0.5, y_pos-0.9), 9, 1.2, boxstyle="round,pad=0.05",
                         edgecolor=color, facecolor='white', alpha=0.5, linewidth=1.5)
    ax4.add_patch(box)
    ax4.text(0.7, y_pos, comment, fontsize=9, color='gray', style='italic')
    ax4.text(0.7, y_pos-0.5, code, fontsize=9, family='monospace', color=color)
    y_pos -= 1.7

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
