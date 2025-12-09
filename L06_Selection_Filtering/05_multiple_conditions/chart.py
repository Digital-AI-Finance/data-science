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

ax.text(5, 7.5, 'Multiple Conditions', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# AND condition
rect1 = patches.FancyBboxPatch((0.5, 4), 4, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
ax.add_patch(rect1)
ax.text(2.5, 6, 'AND: &', fontsize=12, fontweight='bold', ha='center', color=MLBLUE)
ax.text(2.5, 5.2, '(df["AAPL"] > 185) &', fontsize=9, fontfamily='monospace', ha='center')
ax.text(2.5, 4.6, '(df["MSFT"] > 380)', fontsize=9, fontfamily='monospace', ha='center')

# OR condition
rect2 = patches.FancyBboxPatch((5.5, 4), 4, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#FFF3E0', edgecolor=MLORANGE, linewidth=2)
ax.add_patch(rect2)
ax.text(7.5, 6, 'OR: |', fontsize=12, fontweight='bold', ha='center', color=MLORANGE)
ax.text(7.5, 5.2, '(df["AAPL"] > 200) |', fontsize=9, fontfamily='monospace', ha='center')
ax.text(7.5, 4.6, '(df["MSFT"] > 400)', fontsize=9, fontfamily='monospace', ha='center')

# NOT
rect3 = patches.FancyBboxPatch((2.5, 1), 5, 2, boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor=MLRED, linewidth=2)
ax.add_patch(rect3)
ax.text(5, 2.5, 'NOT: ~', fontsize=12, fontweight='bold', ha='center', color=MLRED)
ax.text(5, 1.7, '~(df["AAPL"] > 190)', fontsize=9, fontfamily='monospace', ha='center')

ax.text(5, 0.3, 'Always use parentheses around each condition!', fontsize=10, ha='center', style='italic')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
