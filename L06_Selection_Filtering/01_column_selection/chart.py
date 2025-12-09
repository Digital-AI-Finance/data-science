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

ax.text(5, 7.5, 'Column Selection Methods', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

methods = [
    ("df['AAPL']", 'Single column (Series)', MLBLUE),
    ("df[['AAPL', 'MSFT']]", 'Multiple columns (DataFrame)', MLGREEN),
    ("df.AAPL", 'Attribute access (simple names)', MLORANGE),
    ("df.loc[:, 'AAPL':'GOOGL']", 'Range of columns', MLPURPLE)
]

for i, (code, desc, color) in enumerate(methods):
    y = 6 - i * 1.3
    rect = patches.FancyBboxPatch((1, y - 0.3), 4, 0.8, boxstyle="round,pad=0.05",
                                   facecolor='#F5F5F5', edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(3, y + 0.1, code, fontsize=10, ha='center', fontfamily='monospace')
    ax.text(6, y + 0.1, desc, fontsize=10, ha='left', color='gray')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
