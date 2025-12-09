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

ax.text(5, 9.5, 'if-elif-else Decision Flow', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Start
start_box = FancyBboxPatch((3.5, 8.2), 3, 0.6, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(start_box)
ax.text(5, 8.5, 'Start: Get stock price', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

# First condition
cond1_box = FancyBboxPatch((3.2, 7.0), 3.6, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(cond1_box)
ax.text(5, 7.4, 'if price > 160?', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_ACCENT)

# True branch 1
true1_box = FancyBboxPatch((0.5, 5.8), 2.5, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(true1_box)
ax.text(1.75, 6.2, 'action = "SELL"', ha='center', va='center',
        fontsize=9, family='monospace', color=COLOR_GREEN, fontweight='bold')

# Second condition
cond2_box = FancyBboxPatch((5.5, 5.8), 3.6, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(cond2_box)
ax.text(7.3, 6.2, 'elif price < 140?', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_ACCENT)

# True branch 2
true2_box = FancyBboxPatch((5.5, 4.6), 2.5, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(true2_box)
ax.text(6.75, 5.0, 'action = "BUY"', ha='center', va='center',
        fontsize=9, family='monospace', color=COLOR_GREEN, fontweight='bold')

# Else branch
else_box = FancyBboxPatch((8.5, 4.6), 1.3, 0.8, boxstyle="round,pad=0.05",
                          edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(else_box)
ax.text(9.15, 5.0, 'HOLD', ha='center', va='center',
        fontsize=9, family='monospace', color=COLOR_ORANGE, fontweight='bold')

# End
end_box = FancyBboxPatch((3.5, 3.0), 3, 0.6, boxstyle="round,pad=0.1",
                         edgecolor=COLOR_SECONDARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(end_box)
ax.text(5, 3.3, 'End: Execute action', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

# Arrows with labels
ax.annotate('', xy=(5, 7.0), xytext=(5, 8.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

ax.annotate('', xy=(1.75, 6.6), xytext=(3.5, 7.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
ax.text(2.5, 7.0, 'True', ha='center', va='bottom',
        fontsize=9, color=COLOR_GREEN, fontweight='bold')

ax.annotate('', xy=(7.3, 5.8), xytext=(6.5, 7.0),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))
ax.text(6.8, 6.5, 'False', ha='center', va='center',
        fontsize=9, color=COLOR_ORANGE, fontweight='bold')

ax.annotate('', xy=(6.75, 4.6), xytext=(7.3, 5.8),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
ax.text(7.5, 5.2, 'True', ha='center', va='center',
        fontsize=9, color=COLOR_GREEN, fontweight='bold')

ax.annotate('', xy=(9.15, 4.6), xytext=(8.9, 5.8),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))
ax.text(9.5, 5.2, 'False', ha='center', va='center',
        fontsize=9, color=COLOR_ORANGE, fontweight='bold')

# All paths to end
ax.annotate('', xy=(5, 3.6), xytext=(1.75, 5.8),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=1.5, alpha=0.5))
ax.annotate('', xy=(5, 3.6), xytext=(6.75, 4.6),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=1.5, alpha=0.5))
ax.annotate('', xy=(5, 3.6), xytext=(9.15, 4.6),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=1.5, alpha=0.5))

# Code example
code_box = FancyBboxPatch((1, 0.5), 8, 2, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(code_box)
ax.text(5, 2.3, 'Python Code', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    'price = 155.00',
    'if price > 160:',
    '    action = "SELL"',
    'elif price < 140:',
    '    action = "BUY"',
    'else:',
    '    action = "HOLD"',
]

y_pos = 1.9
for line in code_lines:
    ax.text(1.5, y_pos, line, ha='left', va='top',
            fontsize=9, family='monospace', color='black')
    y_pos -= 0.2

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
