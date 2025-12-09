"""GroupBy Workflow - Step-by-step groupby process"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
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
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'GroupBy Workflow in pandas', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE)

# Workflow steps
steps = [
    {
        'num': '1', 'title': 'Select Column to Group By',
        'code': "grouped = df.groupby('Sector')",
        'note': 'Creates GroupBy object (lazy evaluation)',
        'color': MLBLUE
    },
    {
        'num': '2', 'title': 'Select Column to Aggregate',
        'code': "grouped['Return']",
        'note': 'Choose which column(s) to operate on',
        'color': MLORANGE
    },
    {
        'num': '3', 'title': 'Apply Aggregation Function',
        'code': ".mean()  # or sum(), count(), std()...",
        'note': 'Actually performs the computation',
        'color': MLGREEN
    },
    {
        'num': '4', 'title': 'Get Results',
        'code': "# Returns Series or DataFrame",
        'note': 'Indexed by group keys',
        'color': MLRED
    },
]

y_start = 8.0
for i, step in enumerate(steps):
    y = y_start - i * 1.8

    # Number circle
    circle = plt.Circle((1, y), 0.4, facecolor=step['color'], edgecolor='white', linewidth=2)
    ax.add_patch(circle)
    ax.text(1, y, step['num'], ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')

    # Step box
    box = FancyBboxPatch((1.8, y-0.5), 5.5, 1, boxstyle="round,pad=0.1",
                         edgecolor=step['color'], facecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(4.55, y+0.2, step['title'], ha='center', fontsize=11,
            fontweight='bold', color=step['color'])
    ax.text(4.55, y-0.2, step['code'], ha='center', fontsize=9,
            family='monospace', color='black')

    # Note
    ax.text(7.8, y, step['note'], va='center', fontsize=9, style='italic', color='gray')

    # Arrow to next
    if i < len(steps) - 1:
        ax.annotate('', xy=(1, y-0.6), xytext=(1, y-1.2),
                    arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))

# Complete example on right side
example_box = FancyBboxPatch((9.5, 3.5), 4, 5, boxstyle="round,pad=0.15",
                             edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.2)
ax.add_patch(example_box)
ax.text(11.5, 8.2, 'Complete Example', ha='center', fontsize=12,
        fontweight='bold', color=MLPURPLE)

code_lines = [
    "# Group by sector and calculate",
    "# mean return for each",
    "",
    "result = df.groupby('Sector')\\",
    "           ['Return']\\",
    "           .mean()",
    "",
    "# Output:",
    "# Sector",
    "# Finance    0.052",
    "# Tech       0.078",
    "# Energy     0.034",
]
for i, line in enumerate(code_lines):
    color = 'gray' if line.startswith('#') else 'black'
    ax.text(9.8, 7.8 - i*0.35, line, fontsize=8, family='monospace', color=color)

# Quick reference at bottom
ref_box = FancyBboxPatch((0.5, 0.5), 13, 1.5, boxstyle="round,pad=0.1",
                         edgecolor=MLPURPLE, facecolor='white', linewidth=1.5)
ax.add_patch(ref_box)
ax.text(7, 1.7, 'Common Aggregation Functions:', ha='center', fontsize=10,
        fontweight='bold', color=MLPURPLE)
funcs = ['mean()', 'sum()', 'count()', 'std()', 'min()', 'max()', 'first()', 'last()']
ax.text(7, 1.1, '  |  '.join(funcs), ha='center', fontsize=9, family='monospace')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
