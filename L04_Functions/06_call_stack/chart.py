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

ax.text(5, 7.5, 'Function Call Stack', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Stack frames
frames = [
    (3, 'main()', '#E8E8FF'),
    (2, 'calculate_portfolio()', '#D0D0FF'),
    (1, 'get_returns()', '#B8B8FF'),
    (0, 'fetch_price()', '#A0A0FF')
]

for i, (level, name, color) in enumerate(frames):
    rect = patches.FancyBboxPatch((2, level * 1.5 + 1), 4, 1.2,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor=MLPURPLE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(4, level * 1.5 + 1.6, name, fontsize=10, ha='center', fontfamily='monospace')

ax.annotate('', xy=(6.5, 5.5), xytext=(6.5, 1.5),
            arrowprops=dict(arrowstyle='<->', color=MLGREEN, lw=2))
ax.text(7, 3.5, 'Call\nStack', fontsize=10, ha='left', color=MLGREEN)

ax.text(8.5, 5.5, 'First In', fontsize=9, color='gray')
ax.text(8.5, 1.5, 'Last Out', fontsize=9, color='gray')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
