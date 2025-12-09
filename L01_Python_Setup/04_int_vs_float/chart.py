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

ax.text(5, 9.5, 'Integer vs Float: Key Differences', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Left column: Integer
int_box = FancyBboxPatch((0.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                         edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(int_box)
ax.text(2.5, 8.7, 'Integer (int)', ha='center', va='top',
        fontsize=14, fontweight='bold', color=COLOR_SECONDARY)
ax.text(0.8, 8.2, '- Whole numbers only', ha='left', va='top',
        fontsize=10, color='black')
ax.text(0.8, 7.8, '- No decimal point', ha='left', va='top',
        fontsize=10, color='black')
ax.text(0.8, 7.4, '- Exact counting', ha='left', va='top',
        fontsize=10, color='black')
ax.text(0.8, 7.0, '- Examples:', ha='left', va='top',
        fontsize=10, fontweight='bold', color='black')

int_examples = FancyBboxPatch((1, 6.2), 2.8, 0.6, boxstyle="round,pad=0.05",
                              edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
ax.add_patch(int_examples)
ax.text(2.4, 6.5, 'shares = 100', ha='center', va='center',
        fontsize=9, family='monospace', color='black')

# Right column: Float
float_box = FancyBboxPatch((5.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(float_box)
ax.text(7.5, 8.7, 'Float (float)', ha='center', va='top',
        fontsize=14, fontweight='bold', color=COLOR_SECONDARY)
ax.text(5.8, 8.2, '- Decimal numbers', ha='left', va='top',
        fontsize=10, color='black')
ax.text(5.8, 7.8, '- Has decimal point', ha='left', va='top',
        fontsize=10, color='black')
ax.text(5.8, 7.4, '- Precise measurements', ha='left', va='top',
        fontsize=10, color='black')
ax.text(5.8, 7.0, '- Examples:', ha='left', va='top',
        fontsize=10, fontweight='bold', color='black')

float_examples = FancyBboxPatch((6, 6.2), 3, 0.6, boxstyle="round,pad=0.05",
                                edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
ax.add_patch(float_examples)
ax.text(7.5, 6.5, 'price = 150.50', ha='center', va='center',
        fontsize=9, family='monospace', color='black')

# Comparison table
table_box = FancyBboxPatch((1, 2.5), 8, 3.5, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(table_box)
ax.text(5, 5.8, 'Finance Use Cases', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

# Table headers
ax.text(2.5, 5.3, 'Integer', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
ax.text(7.5, 5.3, 'Float', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)

# Draw separator line
ax.plot([1.5, 8.5], [5.1, 5.1], color=COLOR_SECONDARY, lw=1.5)
ax.plot([5, 5], [5.1, 2.7], color=COLOR_SECONDARY, lw=1.5)

# Table content
int_uses = ['Number of shares', 'Days in period', 'Number of trades']
float_uses = ['Stock price', 'Portfolio value', 'Return percentage']

y_pos = 4.7
for int_use, float_use in zip(int_uses, float_uses):
    ax.text(2.5, y_pos, int_use, ha='center', va='center',
            fontsize=10, color='black')
    ax.text(7.5, y_pos, float_use, ha='center', va='center',
            fontsize=10, color='black')
    y_pos -= 0.6

# Code example
code_box = FancyBboxPatch((1.5, 1.0), 7, 1.2, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_GREEN, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(code_box)
ax.text(5, 2.0, 'value = shares * price  # int * float = float', ha='center', va='top',
        fontsize=10, family='monospace', color='black')
ax.text(5, 1.6, 'value = 100 * 150.50  # = 15050.0', ha='center', va='top',
        fontsize=10, family='monospace', color='black')
ax.text(5, 1.2, 'type(value)  # <class \'float\'>', ha='center', va='top',
        fontsize=10, family='monospace', color='#808080')

plt.tight_layout()
output_path = 'chart.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
