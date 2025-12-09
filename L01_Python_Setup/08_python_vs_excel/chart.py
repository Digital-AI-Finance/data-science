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

ax.text(5, 9.5, 'Python vs Excel for Finance', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Comparison table
categories = [
    ('Feature', 'Excel', 'Python', 8.5),
    ('Data Size', 'Limited (1M rows)', 'Unlimited', 7.8),
    ('Automation', 'Manual/Macros', 'Full Scripts', 7.1),
    ('Reproducibility', 'Low', 'High', 6.4),
    ('Version Control', 'Difficult', 'Git Integration', 5.7),
    ('Visualization', 'Built-in Charts', 'Custom Libraries', 5.0),
    ('Speed', 'Slow (large data)', 'Fast', 4.3),
    ('Learning Curve', 'Easy', 'Moderate', 3.6),
]

# Draw table header
header_box = FancyBboxPatch((1, 8.2), 8, 0.6, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(header_box)

for category, excel, python, y in categories:
    if y == 8.5:  # Header
        ax.text(2, y, category, ha='center', va='center',
               fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
        ax.text(5, y, excel, ha='center', va='center',
               fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
        ax.text(8, y, python, ha='center', va='center',
               fontsize=11, fontweight='bold', color=COLOR_SECONDARY)
    else:
        # Category
        cat_box = FancyBboxPatch((1, y - 0.25), 2, 0.5, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=1.5)
        ax.add_patch(cat_box)
        ax.text(2, y, category, ha='center', va='center',
               fontsize=9, fontweight='bold', color=COLOR_PRIMARY)

        # Excel
        excel_box = FancyBboxPatch((3.5, y - 0.25), 2.5, 0.5, boxstyle="round,pad=0.05",
                                   edgecolor=COLOR_ACCENT, facecolor='#F5F5F5', linewidth=1.5)
        ax.add_patch(excel_box)
        ax.text(4.75, y, excel, ha='center', va='center',
               fontsize=9, color='black')

        # Python
        python_box = FancyBboxPatch((6.5, y - 0.25), 2.5, 0.5, boxstyle="round,pad=0.05",
                                    edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
        ax.add_patch(python_box)
        ax.text(7.75, y, python, ha='center', va='center',
               fontsize=9, color=COLOR_GREEN, fontweight='bold')

# Best use cases
excel_use = FancyBboxPatch((0.5, 0.8), 4.2, 2.2, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(excel_use)
ax.text(2.6, 2.8, 'Best for Excel:', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_ACCENT)
excel_uses = [
    '- Quick calculations',
    '- Small datasets (<10K rows)',
    '- Visual exploration',
    '- Ad-hoc analysis',
]
y_pos = 2.4
for use in excel_uses:
    ax.text(0.8, y_pos, use, ha='left', va='top',
            fontsize=9, color='black')
    y_pos -= 0.35

python_use = FancyBboxPatch((5.3, 0.8), 4.2, 2.2, boxstyle="round,pad=0.1",
                            edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(python_use)
ax.text(7.4, 2.8, 'Best for Python:', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_GREEN)
python_uses = [
    '- Large datasets (>100K rows)',
    '- Automated workflows',
    '- Machine learning',
    '- Production systems',
]
y_pos = 2.4
for use in python_uses:
    ax.text(5.6, y_pos, use, ha='left', va='top',
            fontsize=9, color='black')
    y_pos -= 0.35

plt.tight_layout()
output_path = 'chart.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
