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


fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'List Indexing in Python', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# List representation
prices = [150.00, 165.50, 148.25, 172.00, 169.75]
n = len(prices)
box_width = 1.3
start_x = 2.0

# Positive indices (top)
ax.text(1.0, 7.5, 'Positive Indexing:', ha='left', va='center',
        fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

for i, price in enumerate(prices):
    x = start_x + i * box_width
    # Box
    box = FancyBboxPatch((x, 6.5), box_width-0.1, 1, boxstyle="round,pad=0.05",
                         edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(box)
    ax.text(x + (box_width-0.1)/2, 7.0, f'{price}', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

    # Index label (top)
    ax.text(x + (box_width-0.1)/2, 7.8, f'[{i}]', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_ACCENT)

# Variable name
ax.text(start_x - 0.5, 7.0, 'prices =', ha='right', va='center',
        fontsize=11, family='monospace', color='black')

# Negative indices (bottom)
ax.text(1.0, 5.5, 'Negative Indexing:', ha='left', va='center',
        fontsize=12, fontweight='bold', color=COLOR_PRIMARY)

for i, price in enumerate(prices):
    x = start_x + i * box_width
    # Box
    box = FancyBboxPatch((x, 4.5), box_width-0.1, 1, boxstyle="round,pad=0.05",
                         edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(x + (box_width-0.1)/2, 5.0, f'{price}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='black')

    # Index label (bottom)
    neg_idx = i - n
    ax.text(x + (box_width-0.1)/2, 4.2, f'[{neg_idx}]', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_ORANGE)

# Examples
examples_box = FancyBboxPatch((1.0, 1.0), 8, 2.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(examples_box)
ax.text(5, 3.6, 'Access Examples', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

examples = [
    ('prices[0]', '150.0', 'First element'),
    ('prices[2]', '148.25', 'Third element'),
    ('prices[-1]', '169.75', 'Last element'),
    ('prices[-2]', '172.0', 'Second from end'),
]

y_pos = 3.0
for code, result, desc in examples:
    ax.text(1.5, y_pos, code, ha='left', va='top',
            fontsize=10, family='monospace', color='black', fontweight='bold')
    ax.text(3.5, y_pos, '→', ha='center', va='top',
            fontsize=12, color=COLOR_ACCENT)
    ax.text(4.0, y_pos, result, ha='left', va='top',
            fontsize=10, family='monospace', color=COLOR_GREEN, fontweight='bold')
    ax.text(5.5, y_pos, f'# {desc}', ha='left', va='top',
            fontsize=9, family='monospace', color='#808080')
    y_pos -= 0.5

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
