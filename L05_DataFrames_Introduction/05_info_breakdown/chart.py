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

ax.text(5, 7.5, 'DataFrame Info: df.info()', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

info_lines = [
    '<class pandas.DataFrame>',
    'RangeIndex: 252 entries, 0 to 251',
    'Data columns (5 columns):',
    '  Date    252 non-null datetime64',
    '  AAPL    252 non-null float64',
    '  MSFT    252 non-null float64',
    '  GOOGL   250 non-null float64  (2 missing)',
    'memory usage: 10.0 KB'
]

rect = patches.FancyBboxPatch((1, 1), 8, 5.5, boxstyle="round,pad=0.1",
                               facecolor='#F5F5F5', edgecolor=MLPURPLE, linewidth=1.5)
ax.add_patch(rect)

for i, line in enumerate(info_lines):
    color = MLRED if 'missing' in line else 'black'
    ax.text(1.5, 6 - i*0.6, line, fontsize=9, fontfamily='monospace', color=color)

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

