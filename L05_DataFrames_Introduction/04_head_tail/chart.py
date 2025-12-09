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

ax.text(5, 7.5, 'Viewing Data: head() and tail()', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# head()
rect1 = patches.FancyBboxPatch((0.5, 3.5), 4, 3.5, boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
ax.add_patch(rect1)
ax.text(2.5, 6.5, 'df.head(3)', fontsize=11, fontfamily='monospace', ha='center', color=MLBLUE)
ax.text(2.5, 5.8, 'First 3 rows', fontsize=9, ha='center', style='italic')

head_data = ['2024-01-02  185.2', '2024-01-03  184.8', '2024-01-04  186.1']
for i, row in enumerate(head_data):
    ax.text(2.5, 5 - i*0.5, row, fontsize=9, fontfamily='monospace', ha='center')

# tail()
rect2 = patches.FancyBboxPatch((5.5, 3.5), 4, 3.5, boxstyle="round,pad=0.1",
                                facecolor='#FFF3E0', edgecolor=MLORANGE, linewidth=2)
ax.add_patch(rect2)
ax.text(7.5, 6.5, 'df.tail(3)', fontsize=11, fontfamily='monospace', ha='center', color=MLORANGE)
ax.text(7.5, 5.8, 'Last 3 rows', fontsize=9, ha='center', style='italic')

tail_data = ['2024-12-27  195.8', '2024-12-30  196.2', '2024-12-31  197.1']
for i, row in enumerate(tail_data):
    ax.text(7.5, 5 - i*0.5, row, fontsize=9, fontfamily='monospace', ha='center')

ax.text(5, 2.5, 'Default: 5 rows | Customize: head(10), tail(20)', fontsize=10, ha='center')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
