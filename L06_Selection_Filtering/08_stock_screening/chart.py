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

# Course colors already used in this lesson
COLOR_PRIMARY = MLPURPLE
COLOR_SECONDARY = MLPURPLE
COLOR_ACCENT = MLBLUE
COLOR_LIGHT = MLLAVENDER
COLOR_GREEN = MLGREEN
COLOR_ORANGE = MLORANGE
COLOR_RED = MLRED


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

ax.text(5, 7.5, 'Stock Screening Workflow', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Workflow steps
steps = [
    (1, 5.5, '1. Load Data', 'pd.read_csv("stocks.csv")', '#E3F2FD'),
    (1, 4, '2. Price Filter', 'df[df["price"] > 100]', '#E8F5E9'),
    (1, 2.5, '3. Volume Filter', 'df[df["volume"] > 1e6]', '#FFF3E0'),
    (5.5, 5.5, '4. Select Columns', 'df[["symbol","price"]]', '#F3E5F5'),
    (5.5, 4, '5. Sort', 'df.sort_values("price")', '#FFEBEE'),
    (5.5, 2.5, '6. Export', 'df.to_csv("screened.csv")', '#E0F7FA')
]

for x, y, title, code, color in steps:
    rect = patches.FancyBboxPatch((x, y), 3.8, 1.2, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor=MLPURPLE, linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 1.9, y + 0.85, title, fontsize=9, fontweight='bold', ha='center')
    ax.text(x + 1.9, y + 0.35, code, fontsize=8, fontfamily='monospace', ha='center')

# Arrows
ax.annotate('', xy=(1.9, 4), xytext=(1.9, 5.5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
ax.annotate('', xy=(1.9, 2.5), xytext=(1.9, 4), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
ax.annotate('', xy=(5.5, 5.5), xytext=(4.8, 5.5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
ax.annotate('', xy=(7.4, 4), xytext=(7.4, 5.5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
ax.annotate('', xy=(7.4, 2.5), xytext=(7.4, 4), arrowprops=dict(arrowstyle='->', color=MLPURPLE))

ax.text(5, 1, 'Combine filters to build powerful stock screeners', fontsize=10, ha='center', style='italic')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("\nL06 COMPLETE: 8/8 charts generated")

# =============================================================================
# Main execution
# =============================================================================
