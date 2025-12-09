"""Merge Types - Inner, Outer, Left, Right joins"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

merge_types = [
    {'ax': axes[0, 0], 'type': 'inner', 'title': 'Inner Join', 'color': MLGREEN,
     'desc': 'Only matching keys (intersection)', 'highlight': 'center'},
    {'ax': axes[0, 1], 'type': 'outer', 'title': 'Outer Join (Full)', 'color': MLPURPLE,
     'desc': 'All keys from both (union)', 'highlight': 'all'},
    {'ax': axes[1, 0], 'type': 'left', 'title': 'Left Join', 'color': MLBLUE,
     'desc': 'All keys from left + matches from right', 'highlight': 'left'},
    {'ax': axes[1, 1], 'type': 'right', 'title': 'Right Join', 'color': MLORANGE,
     'desc': 'All keys from right + matches from left', 'highlight': 'right'},
]

for item in merge_types:
    ax = item['ax']
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, item['title'], ha='center', fontsize=14,
            fontweight='bold', color=item['color'])
    ax.text(5, 8.8, f"merge(df1, df2, how='{item['type']}')", ha='center',
            fontsize=10, family='monospace', color='gray')

    # Venn diagram circles
    # Left circle (df1)
    left_circle = Circle((3.5, 5.5), 2, facecolor=MLBLUE, alpha=0.3, edgecolor=MLBLUE, linewidth=2)
    ax.add_patch(left_circle)
    ax.text(2.3, 5.5, 'df1', ha='center', fontsize=11, fontweight='bold', color=MLBLUE)

    # Right circle (df2)
    right_circle = Circle((6.5, 5.5), 2, facecolor=MLORANGE, alpha=0.3, edgecolor=MLORANGE, linewidth=2)
    ax.add_patch(right_circle)
    ax.text(7.7, 5.5, 'df2', ha='center', fontsize=11, fontweight='bold', color=MLORANGE)

    # Highlight based on join type
    if item['highlight'] == 'center':
        # Intersection
        from matplotlib.patches import Wedge
        wedge1 = Wedge((3.5, 5.5), 2, -60, 60, facecolor=item['color'], alpha=0.6, edgecolor='none')
        wedge2 = Wedge((6.5, 5.5), 2, 120, 240, facecolor=item['color'], alpha=0.6, edgecolor='none')
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)
    elif item['highlight'] == 'all':
        # Union - highlight both circles more
        left_circle.set_alpha(0.5)
        right_circle.set_alpha(0.5)
    elif item['highlight'] == 'left':
        left_circle.set_facecolor(item['color'])
        left_circle.set_alpha(0.5)
    elif item['highlight'] == 'right':
        right_circle.set_facecolor(item['color'])
        right_circle.set_alpha(0.5)

    # Description
    ax.text(5, 2.5, item['desc'], ha='center', fontsize=10, style='italic', color='gray')

    # Sample data
    ax.text(1.5, 1.5, 'df1: A, B, C', fontsize=9, color=MLBLUE)
    ax.text(6.5, 1.5, 'df2: B, C, D', fontsize=9, color=MLORANGE)

    # Result indicator
    if item['type'] == 'inner':
        result = 'Result: B, C'
    elif item['type'] == 'outer':
        result = 'Result: A, B, C, D'
    elif item['type'] == 'left':
        result = 'Result: A, B, C'
    else:
        result = 'Result: B, C, D'

    ax.text(5, 0.8, result, ha='center', fontsize=10, fontweight='bold', color=item['color'])

fig.suptitle('Types of Merge Operations', fontsize=16, fontweight='bold', color=MLPURPLE, y=0.98)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
