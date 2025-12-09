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

ax.text(5, 9.5, 'for vs while: Loop Comparison', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Headers
header_y = 8.7
ax.text(2.5, header_y, 'for Loop', ha='center', va='center',
        fontsize=13, fontweight='bold', color=COLOR_PRIMARY)
ax.text(5.0, header_y, 'Aspect', ha='center', va='center',
        fontsize=13, fontweight='bold', color=COLOR_SECONDARY)
ax.text(7.5, header_y, 'while Loop', ha='center', va='center',
        fontsize=13, fontweight='bold', color=COLOR_ACCENT)

# Separator
ax.plot([0.5, 9.5], [8.4, 8.4], color=COLOR_SECONDARY, lw=2)

# Comparison rows
comparisons = [
    ('Iterate over sequence', 'Use Case', 'Repeat until condition', 7.8),
    ('Known iterations', 'Duration', 'Unknown iterations', 7.1),
    ('for x in sequence:', 'Syntax', 'while condition:', 6.4),
    ('Automatic', 'Increment', 'Manual', 5.7),
    ('List, range(), dict', 'Common With', 'Counters, flags', 5.0),
    ('More readable', 'Readability', 'More flexible', 4.3),
    ('Portfolio analysis', 'Finance Example', 'Price convergence', 3.6),
]

for for_val, aspect, while_val, y in comparisons:
    # for column
    for_box = FancyBboxPatch((0.5, y - 0.25), 3.5, 0.5, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=1.5)
    ax.add_patch(for_box)
    ax.text(2.25, y, for_val, ha='center', va='center',
            fontsize=9, color=COLOR_PRIMARY, fontweight='bold')

    # Aspect column
    aspect_box = FancyBboxPatch((4.2, y - 0.25), 1.6, 0.5, boxstyle="round,pad=0.05",
                                edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=1.5)
    ax.add_patch(aspect_box)
    ax.text(5.0, y, aspect, ha='center', va='center',
            fontsize=9, color='black', fontweight='bold')

    # while column
    while_box = FancyBboxPatch((6.0, y - 0.25), 3.5, 0.5, boxstyle="round,pad=0.05",
                               edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
    ax.add_patch(while_box)
    ax.text(7.75, y, while_val, ha='center', va='center',
            fontsize=9, color=COLOR_ACCENT, fontweight='bold')

# Code examples
for_code_box = FancyBboxPatch((0.5, 0.3), 4.3, 2.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_PRIMARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(for_code_box)
ax.text(2.65, 3.0, 'for Example', ha='center', va='top',
        fontsize=10, fontweight='bold', color=COLOR_PRIMARY)

for_lines = [
    'prices = [150, 165, 148]',
    'total = 0',
    'for price in prices:',
    '    total += price',
    'avg = total / len(prices)',
    '',
    '# 3 iterations (known)',
]

y_pos = 2.6
for line in for_lines:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.8, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.35

while_code_box = FancyBboxPatch((5.2, 0.3), 4.3, 2.8, boxstyle="round,pad=0.1",
                                edgecolor=COLOR_ACCENT, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(while_code_box)
ax.text(7.35, 3.0, 'while Example', ha='center', va='top',
        fontsize=10, fontweight='bold', color=COLOR_ACCENT)

while_lines = [
    'price = 100',
    'target = 150',
    'while price < target:',
    '    price *= 1.05',
    '    years += 1',
    '',
    '# Unknown iterations',
]

y_pos = 2.6
for line in while_lines:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.35

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
