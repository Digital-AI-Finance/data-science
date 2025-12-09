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

ax.text(5, 7.5, 'Return Value Flow', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Boxes
boxes = [
    (1, 5, 'Function Call', MLLAVENDER),
    (4, 5, 'Execute Body', '#FFE0B2'),
    (7, 5, 'Return Value', '#C8E6C9')
]
for x, y, label, color in boxes:
    rect = patches.FancyBboxPatch((x, y), 2, 1.2, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor=MLPURPLE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+1, y+0.6, label, fontsize=10, ha='center', va='center')

# Arrows
ax.annotate('', xy=(4, 5.6), xytext=(3, 5.6), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
ax.annotate('', xy=(7, 5.6), xytext=(6, 5.6), arrowprops=dict(arrowstyle='->', color=MLPURPLE))

# Examples
ax.text(1, 3.5, 'Single return:', fontsize=10, fontweight='bold')
ax.text(1, 2.8, 'return price * 1.05', fontsize=10, fontfamily='monospace')

ax.text(5, 3.5, 'Multiple returns:', fontsize=10, fontweight='bold')
ax.text(5, 2.8, 'return mean, std', fontsize=10, fontfamily='monospace')

ax.text(1, 1.5, 'No return (None):', fontsize=10, fontweight='bold')
ax.text(1, 0.8, 'print("Hello")', fontsize=10, fontfamily='monospace')

ax.text(5, 1.5, 'Early return:', fontsize=10, fontweight='bold')
ax.text(5, 0.8, 'if x < 0: return 0', fontsize=10, fontfamily='monospace')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
