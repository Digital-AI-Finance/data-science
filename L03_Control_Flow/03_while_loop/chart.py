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


fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'while Loop: Conditional Iteration', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Flowchart
start_box = FancyBboxPatch((3.5, 8.2), 3, 0.6, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(start_box)
ax.text(5, 8.5, 'Start: price = 100', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

# Condition
cond_box = FancyBboxPatch((3.2, 6.8), 3.6, 0.9, boxstyle="round,pad=0.05",
                          edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(cond_box)
ax.text(5, 7.25, 'while price < 150?', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_ACCENT)

# Loop body
loop_body = FancyBboxPatch((0.5, 5.0), 3, 1.2, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(loop_body)
ax.text(2, 5.9, 'Increase price:', ha='center', va='top',
        fontsize=10, fontweight='bold', color=COLOR_GREEN)
ax.text(2, 5.5, 'price *= 1.05', ha='center', va='top',
        fontsize=9, family='monospace', color='black')
ax.text(2, 5.15, '(5% increase)', ha='center', va='top',
        fontsize=8, color='#808080')

# Exit
exit_box = FancyBboxPatch((6.5, 5.0), 3, 1.2, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(exit_box)
ax.text(8, 5.9, 'Exit loop:', ha='center', va='top',
        fontsize=10, fontweight='bold', color=COLOR_ORANGE)
ax.text(8, 5.5, 'price >= 150', ha='center', va='top',
        fontsize=9, family='monospace', color='black')
ax.text(8, 5.15, 'Continue program', ha='center', va='top',
        fontsize=8, color='#808080')

# Arrows
ax.annotate('', xy=(5, 6.8), xytext=(5, 8.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

ax.annotate('', xy=(2, 6.2), xytext=(3.5, 7.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
ax.text(2.5, 6.8, 'True', ha='center', va='center',
        fontsize=9, color=COLOR_GREEN, fontweight='bold')

ax.annotate('', xy=(8, 6.2), xytext=(6.5, 7.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))
ax.text(7.5, 6.8, 'False', ha='center', va='center',
        fontsize=9, color=COLOR_ORANGE, fontweight='bold')

# Loop back arrow
ax.annotate('', xy=(3.2, 7.5), xytext=(0.5, 6.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=2,
                          connectionstyle='arc3,rad=0.3'))
ax.text(1.5, 7.5, 'Repeat', ha='center', va='center',
        fontsize=9, color=COLOR_PRIMARY, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_PRIMARY))

# Code example
code_box = FancyBboxPatch((0.5, 0.3), 9, 4.2, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(code_box)
ax.text(5, 4.3, 'while Loop Example: Growth Simulation', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    '# Simulate stock price growth until target',
    'price = 100.00',
    'target = 150.00',
    'years = 0',
    'growth_rate = 0.05  # 5% per year',
    '',
    'while price < target:',
    '    price *= (1 + growth_rate)',
    '    years += 1',
    '    print(f"Year {years}: ${price:.2f}")',
    '',
    'print(f"Reached ${target} in {years} years")',
    '',
    '# Output: Reached $150.00 in 9 years',
]

y_pos = 3.9
for line in code_lines:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(1, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.26

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
# For brevity, I'll create simplified versions of remaining charts
