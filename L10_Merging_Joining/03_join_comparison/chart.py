"""Join Comparison - merge vs join vs concat"""
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

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(8, 9.5, 'Comparing Join Methods in pandas', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE)

# Table headers
headers = ['Method', 'Syntax', 'Join On', 'Best For']
x_positions = [1.5, 5, 9.5, 13.5]
for x, h in zip(x_positions, headers):
    ax.text(x, 8.5, h, ha='center', fontsize=11, fontweight='bold', color=MLPURPLE)

ax.axhline(y=8.2, xmin=0.03, xmax=0.97, color=MLPURPLE, linewidth=2)

# Methods comparison
methods = [
    {
        'name': 'pd.merge()',
        'syntax': "pd.merge(df1, df2,\n  on='key', how='inner')",
        'join_on': 'Column values\n(most flexible)',
        'best_for': 'SQL-style joins,\nmultiple keys',
        'color': MLBLUE
    },
    {
        'name': 'df.join()',
        'syntax': "df1.join(df2,\n  how='left')",
        'join_on': 'Index values\n(fast)',
        'best_for': 'Index-based joins,\ntime series alignment',
        'color': MLORANGE
    },
    {
        'name': 'pd.concat()',
        'syntax': "pd.concat([df1, df2],\n  axis=0)",
        'join_on': 'Position\n(stack rows/cols)',
        'best_for': 'Combining similar\nDataFrames',
        'color': MLGREEN
    },
]

y_pos = 7.5
for method in methods:
    # Background box
    box = FancyBboxPatch((0.3, y_pos - 1.7), 15.4, 2, boxstyle="round,pad=0.1",
                         edgecolor=method['color'], facecolor='white', linewidth=2)
    ax.add_patch(box)

    # Method name
    ax.text(x_positions[0], y_pos - 0.5, method['name'], ha='center',
            fontsize=11, fontweight='bold', color=method['color'])

    # Syntax
    lines = method['syntax'].split('\n')
    for i, line in enumerate(lines):
        ax.text(x_positions[1], y_pos - 0.3 - i*0.4, line, ha='center',
                fontsize=9, family='monospace', color='black')

    # Join on
    lines = method['join_on'].split('\n')
    for i, line in enumerate(lines):
        ax.text(x_positions[2], y_pos - 0.3 - i*0.4, line, ha='center',
                fontsize=9, color='black')

    # Best for
    lines = method['best_for'].split('\n')
    for i, line in enumerate(lines):
        ax.text(x_positions[3], y_pos - 0.3 - i*0.4, line, ha='center',
                fontsize=9, color='black')

    y_pos -= 2.3

# Decision guide at bottom
guide_box = FancyBboxPatch((0.5, 0.5), 15, 1.5, boxstyle="round,pad=0.1",
                           edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3)
ax.add_patch(guide_box)
ax.text(8, 1.7, 'Quick Decision Guide:', ha='center', fontsize=11,
        fontweight='bold', color=MLPURPLE)
ax.text(8, 1.1, 'Join on columns? -> merge()  |  Join on index? -> join()  |  Stack same structure? -> concat()',
        ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
