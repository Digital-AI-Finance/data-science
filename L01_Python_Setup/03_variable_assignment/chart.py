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

# Map old colors to course colors
COLOR_PRIMARY = MLPURPLE
COLOR_SECONDARY = MLPURPLE
COLOR_ACCENT = MLBLUE
COLOR_LIGHT = MLLAVENDER
COLOR_GREEN = MLGREEN
COLOR_ORANGE = MLORANGE
COLOR_RED = MLRED


fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Variable Assignment Process', ha='center', va='top',
        fontsize=16, fontweight='bold', color=COLOR_SECONDARY)

# Step boxes
steps = [
    (8.5, 'Write variable name', COLOR_PRIMARY),
    (7.3, 'Use = operator', COLOR_PRIMARY),
    (6.1, 'Provide value', COLOR_PRIMARY),
    (4.9, 'Python stores in memory', COLOR_ACCENT),
    (3.7, 'Variable ready to use', COLOR_GREEN),
]

for y, text, color in steps:
    box = FancyBboxPatch((2, y - 0.4), 6, 0.8, boxstyle="round,pad=0.1",
                         edgecolor=color, facecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y, text, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)

    # Arrow to next step
    if y > 4:
        ax.annotate('', xy=(5, y - 0.5), xytext=(5, y - 1.1),
                   arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

# Example
example_box = FancyBboxPatch((1.5, 1.5), 7, 1.8, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(example_box)
ax.text(5, 3.1, 'Example: Stock Price', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)
ax.text(2, 2.7, 'stock_price = 150.50', ha='left', va='top',
        fontsize=11, family='monospace', color='black')
ax.text(2, 2.3, 'print(stock_price)  # Output: 150.50', ha='left', va='top',
        fontsize=11, family='monospace', color='#808080')
ax.text(2, 1.9, 'new_price = stock_price * 1.05', ha='left', va='top',
        fontsize=11, family='monospace', color='black')

plt.tight_layout()
output_path = 'chart.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
