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


fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Choosing the Right Data Structure', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Decision tree
ax.text(5, 8.8, 'Need to store data?', ha='center', va='top',
        fontsize=12, fontweight='bold', color='black')

# Branch 1: Ordered sequence
branch1_box = FancyBboxPatch((0.3, 6.5), 3.5, 1.8, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(branch1_box)
ax.text(2.05, 8.0, 'Ordered sequence?', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
ax.text(0.6, 7.5, 'Use LIST', ha='left', va='top',
        fontsize=10, fontweight='bold', color=COLOR_GREEN)
ax.text(0.6, 7.1, 'Examples:', ha='left', va='top',
        fontsize=9, fontweight='bold', color='black')
ax.text(0.8, 6.7, '- Stock prices over time', ha='left', va='top',
        fontsize=8, color='black')
ax.text(0.8, 6.4, '- List of tickers', ha='left', va='top',
        fontsize=8, color='black')

# Branch 2: Key-value mapping
branch2_box = FancyBboxPatch((6.2, 6.5), 3.5, 1.8, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(branch2_box)
ax.text(7.95, 8.0, 'Key-value mapping?', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_ACCENT)
ax.text(6.5, 7.5, 'Use DICTIONARY', ha='left', va='top',
        fontsize=10, fontweight='bold', color=COLOR_GREEN)
ax.text(6.5, 7.1, 'Examples:', ha='left', va='top',
        fontsize=9, fontweight='bold', color='black')
ax.text(6.7, 6.7, '- Stock ticker → price', ha='left', va='top',
        fontsize=8, color='black')
ax.text(6.7, 6.4, '- Portfolio holdings', ha='left', va='top',
        fontsize=8, color='black')

# Arrows
ax.annotate('', xy=(2, 6.5), xytext=(4.5, 8.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=2))
ax.annotate('', xy=(8, 6.5), xytext=(5.5, 8.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

# Comparison table
table_box = FancyBboxPatch((0.5, 0.5), 9, 5.5, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(table_box)
ax.text(5, 5.8, 'Feature Comparison', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

# Table headers
ax.text(2.5, 5.2, 'List', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
ax.text(5.0, 5.2, 'Feature', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
ax.text(7.5, 5.2, 'Dictionary', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_ACCENT)

# Separator
ax.plot([0.7, 9.3], [5.0, 5.0], color=COLOR_SECONDARY, lw=1.5)

# Comparison rows
comparisons = [
    ('Ordered', 'Order', 'Unordered', 4.6),
    ('prices[0]', 'Access', 'portfolio["AAPL"]', 4.0),
    ('O(n) search', 'Speed', 'O(1) lookup', 3.4),
    ('Duplicates OK', 'Duplicates', 'Unique keys', 2.8),
    ('Integer indices', 'Keys', 'Any immutable', 2.2),
    ('append(), sort()', 'Methods', 'get(), keys()', 1.6),
    ('[1,2,3]', 'Syntax', '{"a": 1, "b": 2}', 1.0),
]

for list_val, feature, dict_val, y in comparisons:
    ax.text(2.5, y, list_val, ha='center', va='center',
            fontsize=9, family='monospace', color=COLOR_PRIMARY, fontweight='bold')
    ax.text(5.0, y, feature, ha='center', va='center',
            fontsize=9, color='black')
    ax.text(7.5, y, dict_val, ha='center', va='center',
            fontsize=9, family='monospace', color=COLOR_ACCENT, fontweight='bold')

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
