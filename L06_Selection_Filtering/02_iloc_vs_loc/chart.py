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

ax.text(5, 7.5, 'iloc vs loc', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# iloc
rect1 = patches.FancyBboxPatch((0.5, 3), 4, 4, boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
ax.add_patch(rect1)
ax.text(2.5, 6.5, 'iloc (Integer Location)', fontsize=11, fontweight='bold', ha='center', color=MLBLUE)
ax.text(2.5, 5.7, 'Position-based indexing', fontsize=9, ha='center')
ax.text(2.5, 4.8, 'df.iloc[0]', fontsize=10, fontfamily='monospace', ha='center')
ax.text(2.5, 4.2, 'df.iloc[0:5, 1:3]', fontsize=10, fontfamily='monospace', ha='center')
ax.text(2.5, 3.5, 'Uses: 0, 1, 2, ...', fontsize=9, ha='center', color='gray')

# loc
rect2 = patches.FancyBboxPatch((5.5, 3), 4, 4, boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor=MLGREEN, linewidth=2)
ax.add_patch(rect2)
ax.text(7.5, 6.5, 'loc (Label Location)', fontsize=11, fontweight='bold', ha='center', color=MLGREEN)
ax.text(7.5, 5.7, 'Label-based indexing', fontsize=9, ha='center')
ax.text(7.5, 4.8, "df.loc['2024-01-02']", fontsize=10, fontfamily='monospace', ha='center')
ax.text(7.5, 4.2, "df.loc[:, 'AAPL']", fontsize=10, fontfamily='monospace', ha='center')
ax.text(7.5, 3.5, 'Uses: dates, names', fontsize=9, ha='center', color='gray')

ax.text(5, 2, 'iloc: exclusive end | loc: inclusive end', fontsize=10, ha='center', style='italic')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
