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

ax.text(5, 7.5, 'Index and Columns', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# DataFrame representation
rect = patches.FancyBboxPatch((2, 2), 6, 4, boxstyle="round,pad=0.1",
                               facecolor='white', edgecolor=MLPURPLE, linewidth=2)
ax.add_patch(rect)

# Columns bar
rect_cols = patches.FancyBboxPatch((2, 5.5), 6, 0.5, boxstyle="round,pad=0.02",
                                    facecolor=MLBLUE, edgecolor=MLBLUE, linewidth=1)
ax.add_patch(rect_cols)
ax.text(5, 5.75, 'df.columns: ["Date", "AAPL", "MSFT", "GOOGL"]', fontsize=9,
        ha='center', color='white', fontfamily='monospace')

# Index bar
rect_idx = patches.FancyBboxPatch((2, 2), 0.8, 3.5, boxstyle="round,pad=0.02",
                                   facecolor=MLORANGE, edgecolor=MLORANGE, linewidth=1)
ax.add_patch(rect_idx)
ax.text(2.4, 3.75, 'df.index', fontsize=8, rotation=90, ha='center', va='center', color='white')

ax.text(5, 3.5, 'Data', fontsize=12, ha='center', va='center', color='gray')

ax.text(5, 1.2, 'df.shape: (252, 4)  |  df.dtypes: column data types', fontsize=10, ha='center')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
