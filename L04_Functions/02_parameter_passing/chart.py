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

ax.text(5, 7.5, 'Parameter Passing', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Caller side
rect1 = patches.FancyBboxPatch((0.5, 4), 3.5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#E8E8FF', edgecolor=MLPURPLE, linewidth=2)
ax.add_patch(rect1)
ax.text(2.25, 6, 'Caller', fontsize=11, fontweight='bold', ha='center', color=MLPURPLE)
ax.text(2.25, 5.2, 'price = 100', fontsize=10, fontfamily='monospace', ha='center')
ax.text(2.25, 4.5, 'ret = calc(price)', fontsize=10, fontfamily='monospace', ha='center')

# Arrow
ax.annotate('', xy=(6, 5.25), xytext=(4, 5.25),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
ax.text(5, 5.8, 'value copied', fontsize=9, ha='center', color=MLORANGE)

# Function side
rect2 = patches.FancyBboxPatch((6, 4), 3.5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#E8FFE8', edgecolor=MLGREEN, linewidth=2)
ax.add_patch(rect2)
ax.text(7.75, 6, 'Function', fontsize=11, fontweight='bold', ha='center', color=MLGREEN)
ax.text(7.75, 5.2, 'def calc(p):', fontsize=10, fontfamily='monospace', ha='center')
ax.text(7.75, 4.5, '  return p * 0.05', fontsize=10, fontfamily='monospace', ha='center')

# Types
ax.text(2.25, 2.5, 'Positional: func(a, b)', fontsize=10, fontfamily='monospace', ha='center')
ax.text(2.25, 1.8, 'Keyword: func(x=1, y=2)', fontsize=10, fontfamily='monospace', ha='center')
ax.text(7.75, 2.5, 'Default: def f(x=10)', fontsize=10, fontfamily='monospace', ha='center')
ax.text(7.75, 1.8, '*args, **kwargs', fontsize=10, fontfamily='monospace', ha='center')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
