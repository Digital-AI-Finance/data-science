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


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Python Data Types Hierarchy', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Root
root = FancyBboxPatch((3.5, 7.5), 3, 0.8, boxstyle="round,pad=0.1",
                      edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(root)
ax.text(5, 7.9, 'Data Types', ha='center', va='center',
        fontsize=14, fontweight='bold', color=COLOR_SECONDARY)

# Numeric types
numeric = FancyBboxPatch((0.5, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                         edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(numeric)
ax.text(1.5, 5.9, 'Numeric', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

# Text
text_box = FancyBboxPatch((3, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(text_box)
ax.text(4, 5.9, 'Text', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

# Boolean
bool_box = FancyBboxPatch((5.5, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(bool_box)
ax.text(6.5, 5.9, 'Boolean', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

# Sequence
seq_box = FancyBboxPatch((8, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                         edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(seq_box)
ax.text(9, 5.9, 'Sequence', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

# Subtypes
subtypes = [
    (0.2, 3.5, 'int\n150'),
    (1.3, 3.5, 'float\n150.50'),
    (2.4, 3.5, 'complex\n1+2j'),
    (3, 3.5, 'str\n"AAPL"'),
    (5.5, 3.5, 'bool\nTrue/False'),
    (7.5, 3.5, 'list\n[1,2,3]'),
    (8.6, 3.5, 'tuple\n(1,2)'),
]

for x, y, text in subtypes:
    subtype_box = FancyBboxPatch((x, y), 1, 0.9, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_ACCENT, facecolor='#F0F0F0', linewidth=1.5)
    ax.add_patch(subtype_box)
    ax.text(x + 0.5, y + 0.45, text, ha='center', va='center',
            fontsize=9, family='monospace', color='black')

# Arrows
arrow_props = dict(arrowstyle='->', color=COLOR_SECONDARY, lw=1.5)
ax.annotate('', xy=(1.5, 5.5), xytext=(5, 7.5), arrowprops=arrow_props)
ax.annotate('', xy=(4, 5.5), xytext=(5, 7.5), arrowprops=arrow_props)
ax.annotate('', xy=(6.5, 5.5), xytext=(5, 7.5), arrowprops=arrow_props)
ax.annotate('', xy=(9, 5.5), xytext=(5, 7.5), arrowprops=arrow_props)

# Finance note
note_box = FancyBboxPatch((2, 1.0), 6, 1.2, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(note_box)
ax.text(5, 1.9, 'Finance Application', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_GREEN)
ax.text(5, 1.5, 'price = 150.50  # float for stock price', ha='center', va='top',
        fontsize=9, family='monospace', color='black')
ax.text(5, 1.2, 'ticker = "AAPL"  # string for stock symbol', ha='center', va='top',
        fontsize=9, family='monospace', color='black')

plt.tight_layout()
output_path = 'chart.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
