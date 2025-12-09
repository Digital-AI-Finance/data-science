"""Generated chart with course color palette"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Wedge
import numpy as np
import pandas as pd
import seaborn as sns
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

# Course colors already used in this lesson
COLOR_PRIMARY = MLPURPLE
COLOR_SECONDARY = MLPURPLE
COLOR_ACCENT = MLBLUE
COLOR_LIGHT = MLLAVENDER
COLOR_GREEN = MLGREEN
COLOR_ORANGE = MLORANGE
COLOR_RED = MLRED


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

ax.text(5, 7.5, 'Chained Filtering with query()', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Traditional
rect1 = patches.FancyBboxPatch((0.5, 4.5), 4.5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#F5F5F5', edgecolor='gray', linewidth=1.5)
ax.add_patch(rect1)
ax.text(2.75, 6.5, 'Traditional', fontsize=11, fontweight='bold', ha='center', color='gray')
ax.text(2.75, 5.5, 'df[(df["AAPL"] > 185) &', fontsize=9, fontfamily='monospace', ha='center')
ax.text(2.75, 5, '   (df["Volume"] > 1e6)]', fontsize=9, fontfamily='monospace', ha='center')

# Query method
rect2 = patches.FancyBboxPatch((5.5, 4.5), 4, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor=MLGREEN, linewidth=2)
ax.add_patch(rect2)
ax.text(7.5, 6.5, 'query() Method', fontsize=11, fontweight='bold', ha='center', color=MLGREEN)
ax.text(7.5, 5.3, 'df.query("AAPL > 185 and', fontsize=9, fontfamily='monospace', ha='center')
ax.text(7.5, 4.8, '         Volume > 1e6")', fontsize=9, fontfamily='monospace', ha='center')

ax.text(5, 3.5, 'query() is more readable for complex filters', fontsize=10, ha='center', style='italic')

# isin example
rect3 = patches.FancyBboxPatch((2, 1), 6, 1.8, boxstyle="round,pad=0.1",
                                facecolor=MLLAVENDER, edgecolor=MLPURPLE, linewidth=1.5)
ax.add_patch(rect3)
ax.text(5, 2.3, 'Membership: isin()', fontsize=11, fontweight='bold', ha='center', color=MLPURPLE)
ax.text(5, 1.5, 'df[df["Symbol"].isin(["AAPL", "MSFT", "GOOGL"])]', fontsize=9, fontfamily='monospace', ha='center')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
