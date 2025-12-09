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

ax.text(5, 9.5, 'Jupyter Notebook Interface', ha='center', va='top',
        fontsize=20, fontweight='bold', color=COLOR_SECONDARY)

# Menu bar
menu_bar = FancyBboxPatch((0.5, 8.5), 9, 0.6, boxstyle="round,pad=0.05",
                          edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(menu_bar)
ax.text(1, 8.8, 'File  Edit  View  Insert  Cell  Kernel  Help',
        va='center', fontsize=11, color=COLOR_SECONDARY, fontweight='bold')

# Code cell
code_cell = FancyBboxPatch((0.5, 5.5), 9, 2.5, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_PRIMARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(code_cell)
ax.text(0.7, 7.8, 'In [1]:', va='top', fontsize=11, color=COLOR_SECONDARY,
        fontweight='bold', family='monospace')
ax.text(1.5, 7.8, '# Calculate stock price change', va='top', fontsize=10,
        color='#808080', family='monospace')
ax.text(1.5, 7.4, 'initial_price = 150.00', va='top', fontsize=10,
        color='black', family='monospace')
ax.text(1.5, 7.0, 'final_price = 165.50', va='top', fontsize=10,
        color='black', family='monospace')
ax.text(1.5, 6.6, 'change = final_price - initial_price', va='top', fontsize=10,
        color='black', family='monospace')
ax.text(1.5, 6.2, 'print(f"Change: ${change:.2f}")', va='top', fontsize=10,
        color='black', family='monospace')

# Output cell
output_cell = FancyBboxPatch((0.5, 4.0), 9, 1.2, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
ax.add_patch(output_cell)
ax.text(0.7, 5.0, 'Out[1]:', va='top', fontsize=11, color=COLOR_ACCENT,
        fontweight='bold', family='monospace')
ax.text(1.5, 5.0, 'Change: $15.50', va='top', fontsize=10,
        color='black', family='monospace')

# Annotations
ax.annotate('Input Code', xy=(0.5, 6.5), xytext=(0.2, 3.0),
            fontsize=10, color=COLOR_PRIMARY, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=2))
ax.annotate('Output Result', xy=(0.5, 4.5), xytext=(0.2, 2.0),
            fontsize=10, color=COLOR_ACCENT, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

plt.tight_layout()
output_path = 'chart.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
