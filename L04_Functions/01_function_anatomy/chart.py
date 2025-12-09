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

# Function box
rect = patches.FancyBboxPatch((1, 2), 8, 4, boxstyle="round,pad=0.1",
                               facecolor=MLLAVENDER, edgecolor=MLPURPLE, linewidth=2)
ax.add_patch(rect)

# Labels
ax.text(5, 7, 'Function Anatomy', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)
ax.text(1.5, 5.5, 'def calculate_return(price_old, price_new):', fontsize=11, fontfamily='monospace')
ax.text(2, 4.5, '"""Calculate percentage return."""', fontsize=10, fontfamily='monospace', color='gray')
ax.text(2, 3.5, 'return (price_new - price_old) / price_old * 100', fontsize=10, fontfamily='monospace')

ax.annotate('keyword', xy=(1.5, 5.5), xytext=(0.5, 6.5), fontsize=9, color=MLORANGE,
            arrowprops=dict(arrowstyle='->', color=MLORANGE))
ax.annotate('parameters', xy=(6, 5.5), xytext=(7, 6.5), fontsize=9, color=MLBLUE,
            arrowprops=dict(arrowstyle='->', color=MLBLUE))
ax.annotate('docstring', xy=(3, 4.5), xytext=(6.5, 4.8), fontsize=9, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('return value', xy=(2, 3.5), xytext=(0.5, 2.8), fontsize=9, color=MLGREEN,
            arrowprops=dict(arrowstyle='->', color=MLGREEN))

ax.text(5, 1, 'Functions encapsulate reusable logic', fontsize=10, ha='center', style='italic')
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
