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


fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Type Conversion (Casting)', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Type boxes in circle
types = [
    ('int', 5, 7, 0),
    ('float', 7.5, 5, 1),
    ('str', 5, 3, 2),
    ('bool', 2.5, 5, 3),
]

type_colors = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_ORANGE, COLOR_GREEN]

for type_name, x, y, idx in types:
    type_box = FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8,
                              boxstyle="round,pad=0.1",
                              edgecolor=type_colors[idx],
                              facecolor='white', linewidth=3)
    ax.add_patch(type_box)
    ax.text(x, y, type_name, ha='center', va='center',
            fontsize=14, fontweight='bold', color=type_colors[idx])

# Conversion arrows and labels
conversions = [
    # (from_idx, to_idx, label, curve)
    (0, 1, 'float()', 0.3),  # int to float
    (1, 0, 'int()', -0.3),   # float to int
    (0, 2, 'str()', 0.2),    # int to str
    (2, 0, 'int()', -0.2),   # str to int
    (1, 2, 'str()', 0.2),    # float to str
    (2, 1, 'float()', -0.2), # str to float
    (3, 0, 'int()', 0.2),    # bool to int
    (0, 3, 'bool()', -0.2),  # int to bool
]

for from_idx, to_idx, label, _ in conversions:
    from_type = types[from_idx]
    to_type = types[to_idx]

    # Draw curved arrow
    ax.annotate('', xy=(to_type[1], to_type[2]),
               xytext=(from_type[1], from_type[2]),
               arrowprops=dict(arrowstyle='->', color=type_colors[from_idx],
                             lw=1.5, alpha=0.6,
                             connectionstyle='arc3,rad=0.3'))

# Examples table
examples_box = FancyBboxPatch((0.5, 0.2), 9, 2.0, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(examples_box)
ax.text(5, 2.0, 'Conversion Examples', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

examples = [
    'int("150")      # "150" -> 150',
    'float("150.50") # "150.50" -> 150.5',
    'str(150)        # 150 -> "150"',
    'int(150.99)     # 150.99 -> 150 (truncates!)',
    'bool(0)         # 0 -> False',
    'bool(150)       # 150 -> True',
]

y_pos = 1.6
x_left = 1.0
x_right = 5.5

for i, example in enumerate(examples):
    x = x_left if i < 3 else x_right
    y_offset = i if i < 3 else i - 3
    ax.text(x, y_pos - y_offset * 0.35, example, ha='left', va='top',
            fontsize=9, family='monospace', color='black')

plt.tight_layout()
output_path = 'chart.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
