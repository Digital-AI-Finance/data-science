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


fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Boolean Logic & Truth Tables', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# AND operator
and_box = FancyBboxPatch((0.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                         edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(and_box)
ax.text(2.5, 8.7, 'AND Operator', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

# AND truth table
and_table_data = [
    ('A', 'B', 'A and B'),
    ('True', 'True', 'True'),
    ('True', 'False', 'False'),
    ('False', 'True', 'False'),
    ('False', 'False', 'False'),
]

y_pos = 8.2
for i, (a, b, result) in enumerate(and_table_data):
    if i == 0:
        color = COLOR_SECONDARY
        weight = 'bold'
    else:
        color = 'black'
        weight = 'normal'

    ax.text(1.0, y_pos, a, ha='center', va='center',
            fontsize=9, fontweight=weight, color=color)
    ax.text(2.0, y_pos, b, ha='center', va='center',
            fontsize=9, fontweight=weight, color=color)
    ax.text(3.5, y_pos, result, ha='center', va='center',
            fontsize=9, fontweight=weight,
            color=COLOR_GREEN if result == 'True' and i > 0 else color)
    y_pos -= 0.4

# OR operator
or_box = FancyBboxPatch((5.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                        edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(or_box)
ax.text(7.5, 8.7, 'OR Operator', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

# OR truth table
or_table_data = [
    ('A', 'B', 'A or B'),
    ('True', 'True', 'True'),
    ('True', 'False', 'True'),
    ('False', 'True', 'True'),
    ('False', 'False', 'False'),
]

y_pos = 8.2
for i, (a, b, result) in enumerate(or_table_data):
    if i == 0:
        color = COLOR_SECONDARY
        weight = 'bold'
    else:
        color = 'black'
        weight = 'normal'

    ax.text(6.0, y_pos, a, ha='center', va='center',
            fontsize=9, fontweight=weight, color=color)
    ax.text(7.0, y_pos, b, ha='center', va='center',
            fontsize=9, fontweight=weight, color=color)
    ax.text(8.5, y_pos, result, ha='center', va='center',
            fontsize=9, fontweight=weight,
            color=COLOR_GREEN if result == 'True' and i > 0 else color)
    y_pos -= 0.4

# NOT operator
not_box = FancyBboxPatch((3, 3.5), 4, 2.0, boxstyle="round,pad=0.1",
                         edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(not_box)
ax.text(5, 5.2, 'NOT Operator', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

# NOT truth table
not_table_data = [
    ('A', 'not A'),
    ('True', 'False'),
    ('False', 'True'),
]

y_pos = 4.7
for i, (a, result) in enumerate(not_table_data):
    if i == 0:
        color = COLOR_SECONDARY
        weight = 'bold'
    else:
        color = 'black'
        weight = 'normal'

    ax.text(4.0, y_pos, a, ha='center', va='center',
            fontsize=9, fontweight=weight, color=color)
    ax.text(6.0, y_pos, result, ha='center', va='center',
            fontsize=9, fontweight=weight, color=color)
    y_pos -= 0.4

# Finance example
finance_box = FancyBboxPatch((1, 1.0), 8, 2.0, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(finance_box)
ax.text(5, 2.8, 'Finance Example: Buy Signal', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_GREEN)
ax.text(1.5, 2.3, 'price = 145.00', ha='left', va='top',
        fontsize=9, family='monospace', color='black')
ax.text(1.5, 1.9, 'volume = 1000000', ha='left', va='top',
        fontsize=9, family='monospace', color='black')
ax.text(1.5, 1.5, 'buy = (price < 150) and (volume > 500000)  # True', ha='left', va='top',
        fontsize=9, family='monospace', color='black')
ax.text(1.5, 1.1, 'print(f"Buy signal: {buy}")  # Buy signal: True', ha='left', va='top',
        fontsize=9, family='monospace', color='#808080')

plt.tight_layout()
output_path = 'chart.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
