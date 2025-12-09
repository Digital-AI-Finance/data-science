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

ax.text(5, 7.5, 'Pure vs Impure Functions', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Pure function
rect1 = patches.FancyBboxPatch((0.5, 3.5), 4, 3.5, boxstyle="round,pad=0.1",
                                facecolor='#C8E6C9', edgecolor=MLGREEN, linewidth=2)
ax.add_patch(rect1)
ax.text(2.5, 6.5, 'Pure Function', fontsize=12, fontweight='bold', ha='center', color=MLGREEN)
ax.text(2.5, 5.8, 'Same input -> Same output', fontsize=9, ha='center')
ax.text(2.5, 5.1, 'No side effects', fontsize=9, ha='center')
ax.text(2.5, 4.2, 'def add(a, b):', fontsize=10, fontfamily='monospace', ha='center')
ax.text(2.5, 3.7, '  return a + b', fontsize=10, fontfamily='monospace', ha='center')

# Impure function
rect2 = patches.FancyBboxPatch((5.5, 3.5), 4, 3.5, boxstyle="round,pad=0.1",
                                facecolor='#FFCDD2', edgecolor=MLRED, linewidth=2)
ax.add_patch(rect2)
ax.text(7.5, 6.5, 'Impure Function', fontsize=12, fontweight='bold', ha='center', color=MLRED)
ax.text(7.5, 5.8, 'Modifies external state', fontsize=9, ha='center')
ax.text(7.5, 5.1, 'May have side effects', fontsize=9, ha='center')
ax.text(7.5, 4.2, 'def update(lst, x):', fontsize=10, fontfamily='monospace', ha='center')
ax.text(7.5, 3.7, '  lst.append(x)', fontsize=10, fontfamily='monospace', ha='center')

ax.text(5, 2.5, 'Prefer pure functions for predictable, testable code', fontsize=10, ha='center', style='italic')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
