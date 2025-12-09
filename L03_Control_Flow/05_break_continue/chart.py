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


fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'break vs continue: Loop Control', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Break section
break_box = FancyBboxPatch((0.3, 5.0), 4.5, 3.8, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_RED, facecolor='white', linewidth=2)
ax.add_patch(break_box)
ax.text(2.55, 8.6, 'break: Exit Loop Immediately', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_RED)

break_code = [
    'prices = [150, 165, 148, 175, 162]',
    '',
    'for price in prices:',
    '    if price < 150:',
    '        print(f"Stop! Low: ${price}")',
    '        break  # Exit loop',
    '    print(f"OK: ${price}")',
    '',
    '# Output:',
    '# OK: $150',
    '# OK: $165',
    '# Stop! Low: $148',
    '# (loop ends, 175 and 162 not processed)',
]

y_pos = 8.2
for line in break_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.6, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.26

# Continue section
continue_box = FancyBboxPatch((5.2, 5.0), 4.5, 3.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(continue_box)
ax.text(7.45, 8.6, 'continue: Skip to Next Iteration', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_ORANGE)

continue_code = [
    'prices = [150, 165, 148, 175, 162]',
    '',
    'for price in prices:',
    '    if price < 150:',
    '        print(f"Skip: ${price}")',
    '        continue  # Skip rest',
    '    print(f"Process: ${price}")',
    '',
    '# Output:',
    '# Process: $150',
    '# Process: $165',
    '# Skip: $148',
    '# Process: $175',
    '# Process: $162',
]

y_pos = 8.2
for line in continue_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.26

# Comparison
comp_box = FancyBboxPatch((1, 0.8), 8, 3.7, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(comp_box)
ax.text(5, 4.3, 'Key Differences', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

diff_table = [
    ('break', 'Exits loop completely', 'Use when condition met', 3.6),
    ('continue', 'Skips current iteration', 'Use to filter items', 3.0),
    ('break', 'No more iterations', 'Loop terminates', 2.4),
    ('continue', 'Continues with next', 'Loop continues', 1.8),
]

for keyword, action, use_case, y in diff_table:
    if keyword == 'break':
        color = COLOR_RED
    else:
        color = COLOR_ORANGE

    ax.text(2, y, keyword, ha='center', va='center',
            fontsize=10, family='monospace', fontweight='bold', color=color)
    ax.text(4.5, y, action, ha='left', va='center',
            fontsize=9, color='black')
    ax.text(7, y, use_case, ha='left', va='center',
            fontsize=9, color='#808080', style='italic')

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
