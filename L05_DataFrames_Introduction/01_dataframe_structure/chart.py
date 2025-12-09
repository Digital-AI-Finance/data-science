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

ax.text(5, 7.5, 'DataFrame Structure', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Column headers
cols = ['Index', 'Date', 'AAPL', 'MSFT', 'GOOGL']
for i, col in enumerate(cols):
    rect = patches.FancyBboxPatch((i*1.8 + 0.5, 5.5), 1.6, 0.8, boxstyle="round,pad=0.02",
                                   facecolor=MLPURPLE, edgecolor=MLPURPLE, linewidth=1)
    ax.add_patch(rect)
    ax.text(i*1.8 + 1.3, 5.9, col, fontsize=9, ha='center', color='white', fontweight='bold')

# Data rows
data_rows = [
    ['0', '2024-01-02', '185.2', '376.1', '140.9'],
    ['1', '2024-01-03', '184.8', '374.2', '139.5'],
    ['2', '2024-01-04', '186.1', '378.5', '141.2'],
]

for row_i, row in enumerate(data_rows):
    for col_i, val in enumerate(row):
        color = '#E8E8FF' if row_i % 2 == 0 else '#F5F5FF'
        rect = patches.FancyBboxPatch((col_i*1.8 + 0.5, 4.5 - row_i*0.9), 1.6, 0.7,
                                       boxstyle="round,pad=0.02", facecolor=color,
                                       edgecolor=MLLAVENDER, linewidth=1)
        ax.add_patch(rect)
        ax.text(col_i*1.8 + 1.3, 4.85 - row_i*0.9, val, fontsize=8, ha='center')

ax.annotate('Columns\n(features)', xy=(5, 6.3), xytext=(7.5, 6.8), fontsize=9,
            arrowprops=dict(arrowstyle='->', color=MLBLUE), color=MLBLUE)
ax.annotate('Rows\n(observations)', xy=(0.3, 4), xytext=(-0.5, 3), fontsize=9,
            arrowprops=dict(arrowstyle='->', color=MLORANGE), color=MLORANGE)

ax.text(5, 1.5, '2D labeled data structure with rows and columns', fontsize=10, ha='center', style='italic')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
