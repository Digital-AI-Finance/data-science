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

ax.text(5, 7.5, 'Variable Scope: Local vs Global', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Global scope
rect_global = patches.FancyBboxPatch((0.5, 1), 9, 5.5, boxstyle="round,pad=0.1",
                                      facecolor='#FFF3E0', edgecolor=MLORANGE, linewidth=2)
ax.add_patch(rect_global)
ax.text(5, 6, 'Global Scope', fontsize=12, fontweight='bold', ha='center', color=MLORANGE)
ax.text(1.5, 5.2, 'tax_rate = 0.15', fontsize=10, fontfamily='monospace')

# Local scope
rect_local = patches.FancyBboxPatch((2, 2), 6, 2.5, boxstyle="round,pad=0.1",
                                     facecolor='#E3F2FD', edgecolor=MLBLUE, linewidth=2)
ax.add_patch(rect_local)
ax.text(5, 4, 'Local Scope (inside function)', fontsize=11, fontweight='bold', ha='center', color=MLBLUE)
ax.text(3, 3.2, 'def calc_tax(income):', fontsize=10, fontfamily='monospace')
ax.text(3.5, 2.5, 'tax = income * tax_rate', fontsize=10, fontfamily='monospace')

ax.text(5, 0.5, 'Local variables exist only during function execution', fontsize=10, ha='center', style='italic')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
