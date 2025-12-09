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


fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Common List Methods', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

methods = [
    ('append()', 'Add item to end', 'prices.append(175.00)', '[150, 165, 175]', COLOR_GREEN, 8.2),
    ('insert()', 'Add item at position', 'prices.insert(1, 160)', '[150, 160, 165]', COLOR_GREEN, 7.0),
    ('remove()', 'Remove first occurrence', 'prices.remove(165)', '[150]', COLOR_ORANGE, 5.8),
    ('pop()', 'Remove and return item', 'prices.pop()', 'Returns: 165', COLOR_ORANGE, 4.6),
    ('sort()', 'Sort list in place', 'prices.sort()', '[150, 165, 175]', COLOR_ACCENT, 3.4),
    ('reverse()', 'Reverse list order', 'prices.reverse()', '[165, 150]', COLOR_ACCENT, 2.2),
    ('count()', 'Count occurrences', 'prices.count(150)', '1', COLOR_PRIMARY, 1.0),
]

for method, desc, example, result, color, y in methods:
    # Method name
    method_box = FancyBboxPatch((0.5, y - 0.35), 1.5, 0.6, boxstyle="round,pad=0.05",
                                edgecolor=color, facecolor='white', linewidth=2)
    ax.add_patch(method_box)
    ax.text(1.25, y, method, ha='center', va='center',
            fontsize=10, family='monospace', fontweight='bold', color=color)

    # Description
    ax.text(2.3, y, desc, ha='left', va='center',
            fontsize=9, color='black')

    # Example code
    example_box = FancyBboxPatch((4.5, y - 0.35), 3, 0.6, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=1.5)
    ax.add_patch(example_box)
    ax.text(6.0, y, example, ha='center', va='center',
            fontsize=8, family='monospace', color='black')

    # Result
    result_box = FancyBboxPatch((7.8, y - 0.35), 2, 0.6, boxstyle="round,pad=0.05",
                                edgecolor=color, facecolor='white', linewidth=1.5)
    ax.add_patch(result_box)
    ax.text(8.8, y, result, ha='center', va='center',
            fontsize=8, family='monospace', color=color, fontweight='bold')

# Headers
ax.text(1.25, 9.0, 'Method', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
ax.text(3.3, 9.0, 'Description', ha='left', va='center',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
ax.text(6.0, 9.0, 'Example', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
ax.text(8.8, 9.0, 'Result', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
