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

ax.text(5, 9.5, 'String Operations in Python', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

operations = [
    ('Concatenation', 'ticker1 + ticker2', '"AAPL" + "MSFT"', '"AAPLMSFT"', 8.0),
    ('Repetition', 'ticker * 3', '"XYZ" * 3', '"XYZXYZXYZ"', 6.8),
    ('Upper/Lower', 'ticker.upper()', '"aapl".upper()', '"AAPL"', 5.6),
    ('Slicing', 'ticker[0:2]', '"APPLE"[0:2]', '"AP"', 4.4),
    ('Length', 'len(ticker)', 'len("AAPL")', '4', 3.2),
    ('Format', 'f-string', 'f"Price: ${price}"', '"Price: $150.50"', 2.0),
]

for op_name, syntax, example, result, y in operations:
    # Operation box
    op_box = FancyBboxPatch((0.5, y - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(op_box)
    ax.text(1.5, y, op_name, ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

    # Syntax box
    syntax_box = FancyBboxPatch((2.8, y - 0.4), 2.2, 0.7, boxstyle="round,pad=0.05",
                                edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
    ax.add_patch(syntax_box)
    ax.text(3.9, y, syntax, ha='center', va='center',
            fontsize=9, family='monospace', color='black')

    # Example box
    example_box = FancyBboxPatch((5.3, y - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
    ax.add_patch(example_box)
    ax.text(6.3, y, example, ha='center', va='center',
            fontsize=8, family='monospace', color='#808080')

    # Result box
    result_box = FancyBboxPatch((7.5, y - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                                edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
    ax.add_patch(result_box)
    ax.text(8.5, y, result, ha='center', va='center',
            fontsize=9, family='monospace', color=COLOR_GREEN)

# Headers
ax.text(1.5, 9.0, 'Operation', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
ax.text(3.9, 9.0, 'Syntax', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
ax.text(6.3, 9.0, 'Example', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
ax.text(8.5, 9.0, 'Result', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

plt.tight_layout()
output_path = 'chart.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
