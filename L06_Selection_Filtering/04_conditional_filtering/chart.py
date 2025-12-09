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

ax.text(5, 7.5, 'Conditional Filtering Flow', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Flow
boxes = [
    (1, 4.5, 'DataFrame\n(252 rows)', '#E8E8FF'),
    (4, 4.5, 'Condition\nprice > 190', '#FFE0B2'),
    (7, 4.5, 'Filtered\n(45 rows)', '#C8E6C9')
]

for x, y, text, color in boxes:
    rect = patches.FancyBboxPatch((x, y), 2, 1.8, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor=MLPURPLE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+1, y+0.9, text, fontsize=10, ha='center', va='center')

ax.annotate('', xy=(4, 5.4), xytext=(3, 5.4), arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))
ax.annotate('', xy=(7, 5.4), xytext=(6, 5.4), arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))

ax.text(5, 3, 'df_filtered = df[df["AAPL"] > 190]', fontsize=11, fontfamily='monospace', ha='center')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
