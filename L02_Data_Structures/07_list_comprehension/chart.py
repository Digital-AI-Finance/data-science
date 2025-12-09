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

ax.text(5, 9.5, 'List Comprehension: Concise List Creation', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Traditional approach
trad_box = FancyBboxPatch((0.5, 6.5), 4.3, 2.5, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(trad_box)
ax.text(2.65, 8.8, 'Traditional Loop', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_ORANGE)

trad_code = [
    'prices = [150, 165, 148, 172]',
    'doubled = []',
    'for price in prices:',
    '    doubled.append(price * 2)',
    '',
    '# Result: [300, 330, 296, 344]',
]

y_pos = 8.3
for line in trad_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.8, y_pos, line, ha='left', va='top',
            fontsize=9, family='monospace', color=color)
    y_pos -= 0.35

# Arrow
ax.annotate('', xy=(5.0, 7.7), xytext=(4.9, 7.7),
            arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=3))
ax.text(4.95, 8.2, 'More Concise', ha='center', va='bottom',
        fontsize=10, fontweight='bold', color=COLOR_ACCENT)

# List comprehension
comp_box = FancyBboxPatch((5.2, 6.5), 4.3, 2.5, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(comp_box)
ax.text(7.35, 8.8, 'List Comprehension', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_GREEN)

comp_code = [
    'prices = [150, 165, 148, 172]',
    'doubled = [price * 2',
    '           for price in prices]',
    '',
    '',
    '# Result: [300, 330, 296, 344]',
]

y_pos = 8.3
for line in comp_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=9, family='monospace', color=color, fontweight='bold')
    y_pos -= 0.35

# More examples
examples_box = FancyBboxPatch((0.5, 1.0), 9, 4.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(examples_box)
ax.text(5, 5.6, 'List Comprehension Examples', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

examples = [
    ('Basic transformation', '[x * 2 for x in prices]', 'Double all prices'),
    ('With condition (filter)', '[x for x in prices if x > 150]', 'Only prices > 150'),
    ('String manipulation', '[t.lower() for t in tickers]', 'Lowercase all tickers'),
    ('Math operations', '[x**2 for x in [1,2,3,4]]', 'Squares: [1,4,9,16]'),
    ('With if-else', '[x if x > 150 else 0 for x in prices]', 'Set low prices to 0'),
]

y_pos = 5.0
for title, code, desc in examples:
    ax.text(1.0, y_pos, f'{title}:', ha='left', va='top',
            fontsize=9, fontweight='bold', color=COLOR_PRIMARY)
    ax.text(1.5, y_pos - 0.35, code, ha='left', va='top',
            fontsize=9, family='monospace', color='black')
    ax.text(1.5, y_pos - 0.65, f'# {desc}', ha='left', va='top',
            fontsize=8, family='monospace', color='#808080')
    y_pos -= 0.95

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
