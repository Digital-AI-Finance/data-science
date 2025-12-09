"""Multi-Source Integration - Combining data from multiple sources"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
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

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(8, 9.5, 'Multi-Source Data Integration Pipeline', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE)

# Source 1: Price Data
src1_box = FancyBboxPatch((0.5, 7), 3, 2, boxstyle="round,pad=0.1",
                          edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax.add_patch(src1_box)
ax.text(2, 8.7, 'Yahoo Finance', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
ax.text(2, 8.2, 'df_prices', ha='center', fontsize=9, family='monospace')
ax.text(2, 7.6, 'Ticker, Date, OHLCV', ha='center', fontsize=8, color='gray')

# Source 2: Fundamentals
src2_box = FancyBboxPatch((0.5, 4), 3, 2, boxstyle="round,pad=0.1",
                          edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=2)
ax.add_patch(src2_box)
ax.text(2, 5.7, 'SEC Filings', ha='center', fontsize=10, fontweight='bold', color=MLORANGE)
ax.text(2, 5.2, 'df_fundamentals', ha='center', fontsize=9, family='monospace')
ax.text(2, 4.6, 'Ticker, Quarter, EPS, Rev', ha='center', fontsize=8, color='gray')

# Source 3: Sector Data
src3_box = FancyBboxPatch((0.5, 1), 3, 2, boxstyle="round,pad=0.1",
                          edgecolor=MLGREEN, facecolor='#E6FFE6', linewidth=2)
ax.add_patch(src3_box)
ax.text(2, 2.7, 'Reference Data', ha='center', fontsize=10, fontweight='bold', color=MLGREEN)
ax.text(2, 2.2, 'df_sectors', ha='center', fontsize=9, family='monospace')
ax.text(2, 1.6, 'Ticker, Sector, Industry', ha='center', fontsize=8, color='gray')

# Merge steps
merge1_box = FancyBboxPatch((5, 5.5), 3, 2, boxstyle="round,pad=0.15",
                            edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.5, linewidth=2)
ax.add_patch(merge1_box)
ax.text(6.5, 7, 'Step 1: Merge', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)
ax.text(6.5, 6.3, "merge(df_prices,\ndf_fundamentals,\non='Ticker')", ha='center',
        fontsize=8, family='monospace')

merge2_box = FancyBboxPatch((9.5, 5.5), 3, 2, boxstyle="round,pad=0.15",
                            edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.5, linewidth=2)
ax.add_patch(merge2_box)
ax.text(11, 7, 'Step 2: Merge', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)
ax.text(11, 6.3, "merge(result,\ndf_sectors,\non='Ticker')", ha='center',
        fontsize=8, family='monospace')

# Final result
result_box = FancyBboxPatch((12, 1), 3.5, 3.5, boxstyle="round,pad=0.1",
                            edgecolor=MLRED, facecolor='#FFE6E6', linewidth=2)
ax.add_patch(result_box)
ax.text(13.75, 4.2, 'Final Dataset', ha='center', fontsize=11, fontweight='bold', color=MLRED)
ax.text(13.75, 3.6, 'df_complete', ha='center', fontsize=10, family='monospace')

columns = ['Ticker', 'Date', 'Price', 'Volume', 'EPS', 'Revenue', 'Sector', 'Industry']
for i, col in enumerate(columns):
    ax.text(13.75, 3.0 - i*0.25, col, ha='center', fontsize=7, color='gray')

# Arrows
ax.annotate('', xy=(5, 8), xytext=(3.5, 8),
            arrowprops=dict(arrowstyle='->', color=MLBLUE, lw=2))
ax.annotate('', xy=(5, 5), xytext=(3.5, 5),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
ax.annotate('', xy=(9.5, 6.5), xytext=(8, 6.5),
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))
ax.annotate('', xy=(9.5, 2.5), xytext=(3.5, 2),
            arrowprops=dict(arrowstyle='->', color=MLGREEN, lw=2,
                           connectionstyle='arc3,rad=0.3'))
ax.annotate('', xy=(12, 2.5), xytext=(11, 5.5),
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))

# Code snippet at bottom
code_box = FancyBboxPatch((4, 0.5), 8, 1.8, boxstyle="round,pad=0.1",
                          edgecolor=MLPURPLE, facecolor='white', linewidth=1.5)
ax.add_patch(code_box)
ax.text(8, 2.0, 'Complete Pipeline:', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)
ax.text(8, 1.3, "df = (df_prices.merge(df_fundamentals, on='Ticker')\n               .merge(df_sectors, on='Ticker'))",
        ha='center', fontsize=9, family='monospace')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
