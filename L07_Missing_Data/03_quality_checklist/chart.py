"""Data Quality Checklist - Visual flowchart for data quality assessment"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
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

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Data Quality Checklist for Financial Data',
        ha='center', fontsize=16, fontweight='bold', color=MLPURPLE)

# Define checklist items with status
checklist = [
    ('1. Check Missing Values', 'df.isnull().sum()', 'Critical', MLRED),
    ('2. Detect Duplicates', 'df.duplicated().sum()', 'High', MLORANGE),
    ('3. Validate Data Types', 'df.dtypes', 'Medium', MLBLUE),
    ('4. Check Value Ranges', 'df.describe()', 'High', MLORANGE),
    ('5. Verify Date Continuity', 'date gaps check', 'Critical', MLRED),
    ('6. Outlier Detection', 'IQR / Z-score', 'Medium', MLBLUE),
]

y_positions = [8.0, 6.8, 5.6, 4.4, 3.2, 2.0]

for i, (item, method, priority, color) in enumerate(checklist):
    y = y_positions[i]

    # Checkbox
    checkbox = FancyBboxPatch((0.5, y-0.25), 0.4, 0.4, boxstyle="round,pad=0.02",
                              edgecolor=MLPURPLE, facecolor='white', linewidth=2)
    ax.add_patch(checkbox)
    ax.text(0.7, y-0.05, 'X', ha='center', va='center', fontsize=12,
            fontweight='bold', color=MLGREEN)

    # Item text
    ax.text(1.1, y, item, fontsize=11, fontweight='bold', va='center', color=MLPURPLE)

    # Method box
    method_box = FancyBboxPatch((5.0, y-0.25), 2.5, 0.4, boxstyle="round,pad=0.05",
                                edgecolor=MLBLUE, facecolor='#F0F0FF', linewidth=1.5)
    ax.add_patch(method_box)
    ax.text(6.25, y-0.05, method, ha='center', va='center', fontsize=9,
            family='monospace', color=MLBLUE)

    # Priority indicator
    priority_box = FancyBboxPatch((8.0, y-0.2), 1.2, 0.35, boxstyle="round,pad=0.05",
                                  edgecolor=color, facecolor=color, linewidth=1)
    ax.add_patch(priority_box)
    ax.text(8.6, y-0.02, priority, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')

# Column headers
ax.text(0.7, 8.7, 'Status', ha='center', fontsize=10, fontweight='bold', color='gray')
ax.text(3.0, 8.7, 'Check Item', ha='center', fontsize=10, fontweight='bold', color='gray')
ax.text(6.25, 8.7, 'Python Method', ha='center', fontsize=10, fontweight='bold', color='gray')
ax.text(8.6, 8.7, 'Priority', ha='center', fontsize=10, fontweight='bold', color='gray')

# Summary box at bottom
summary_box = FancyBboxPatch((1.5, 0.3), 7, 1.2, boxstyle="round,pad=0.1",
                             edgecolor=MLGREEN, facecolor='#F0FFF0', linewidth=2)
ax.add_patch(summary_box)
ax.text(5, 1.1, 'Quality Score: 6/6 Checks Passed', ha='center', va='center',
        fontsize=12, fontweight='bold', color=MLGREEN)
ax.text(5, 0.6, 'Data ready for analysis after quality validation', ha='center', va='center',
        fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
