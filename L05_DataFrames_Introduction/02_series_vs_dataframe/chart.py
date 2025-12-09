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

ax.text(5, 7.5, 'Series vs DataFrame', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Series
rect1 = patches.FancyBboxPatch((0.5, 2), 3.5, 4.5, boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
ax.add_patch(rect1)
ax.text(2.25, 6, 'Series (1D)', fontsize=12, fontweight='bold', ha='center', color=MLBLUE)
ax.text(2.25, 5.2, 'Single column', fontsize=10, ha='center')

series_data = [('0', '185.2'), ('1', '184.8'), ('2', '186.1')]
for i, (idx, val) in enumerate(series_data):
    ax.text(1.5, 4.3 - i*0.6, idx, fontsize=9, fontfamily='monospace')
    ax.text(2.8, 4.3 - i*0.6, val, fontsize=9, fontfamily='monospace')

# DataFrame
rect2 = patches.FancyBboxPatch((5.5, 2), 4, 4.5, boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor=MLGREEN, linewidth=2)
ax.add_patch(rect2)
ax.text(7.5, 6, 'DataFrame (2D)', fontsize=12, fontweight='bold', ha='center', color=MLGREEN)
ax.text(7.5, 5.2, 'Multiple columns', fontsize=10, ha='center')

ax.text(6.2, 4.5, 'AAPL', fontsize=8, fontweight='bold')
ax.text(7.5, 4.5, 'MSFT', fontsize=8, fontweight='bold')
ax.text(8.5, 4.5, 'VOL', fontsize=8, fontweight='bold')

df_data = [('185.2', '376.1', '1.2M'), ('184.8', '374.2', '1.1M'), ('186.1', '378.5', '1.3M')]
for i, (a, m, v) in enumerate(df_data):
    ax.text(6.2, 4 - i*0.5, a, fontsize=8, fontfamily='monospace')
    ax.text(7.5, 4 - i*0.5, m, fontsize=8, fontfamily='monospace')
    ax.text(8.5, 4 - i*0.5, v, fontsize=8, fontfamily='monospace')

ax.text(5, 1, 'DataFrame = Collection of Series sharing an index', fontsize=10, ha='center', style='italic')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
