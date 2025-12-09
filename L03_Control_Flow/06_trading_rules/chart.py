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

ax.text(5, 9.5, 'Trading Rules: Decision Tree', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Root
root_box = FancyBboxPatch((3.5, 8.2), 3, 0.7, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(root_box)
ax.text(5, 8.55, 'Check Price & Volume', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_SECONDARY)

# Level 1: Price check
price_check = FancyBboxPatch((3.2, 6.8), 3.6, 0.8, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(price_check)
ax.text(5, 7.2, 'price < buy_threshold?', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_ACCENT)

# Left branch: Check volume
vol_check = FancyBboxPatch((0.5, 5.3), 3.2, 0.8, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(vol_check)
ax.text(2.1, 5.7, 'volume > min_vol?', ha='center', va='center',
        fontsize=9, fontweight='bold', color=COLOR_ACCENT)

# Buy action
buy_box = FancyBboxPatch((0.3, 3.8), 1.8, 0.9, boxstyle="round,pad=0.05",
                         edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(buy_box)
ax.text(1.2, 4.5, 'BUY', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_GREEN)
ax.text(1.2, 4.1, 'Good price\n& volume', ha='center', va='center',
        fontsize=7, color='black')

# Wait action 1
wait1_box = FancyBboxPatch((2.5, 3.8), 1.5, 0.9, boxstyle="round,pad=0.05",
                           edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(wait1_box)
ax.text(3.25, 4.5, 'WAIT', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_ORANGE)
ax.text(3.25, 4.1, 'Low\nvolume', ha='center', va='center',
        fontsize=7, color='black')

# Right branch: Check sell threshold
sell_check = FancyBboxPatch((6.2, 5.3), 3.2, 0.8, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(sell_check)
ax.text(7.8, 5.7, 'price > sell_threshold?', ha='center', va='center',
        fontsize=9, fontweight='bold', color=COLOR_ACCENT)

# Sell action
sell_box = FancyBboxPatch((6.0, 3.8), 1.8, 0.9, boxstyle="round,pad=0.05",
                          edgecolor=COLOR_RED, facecolor='#FFEBEE', linewidth=2)
ax.add_patch(sell_box)
ax.text(6.9, 4.5, 'SELL', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_RED)
ax.text(6.9, 4.1, 'High\nprice', ha='center', va='center',
        fontsize=7, color='black')

# Hold action
hold_box = FancyBboxPatch((8.2, 3.8), 1.5, 0.9, boxstyle="round,pad=0.05",
                          edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(hold_box)
ax.text(8.95, 4.5, 'HOLD', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
ax.text(8.95, 4.1, 'Wait for\nbetter', ha='center', va='center',
        fontsize=7, color='black')

# Arrows
ax.annotate('', xy=(5, 6.8), xytext=(5, 8.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

ax.annotate('', xy=(2.1, 6.1), xytext=(3.5, 6.9),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
ax.text(2.5, 6.5, 'True', ha='center', va='center',
        fontsize=8, color=COLOR_GREEN, fontweight='bold')

ax.annotate('', xy=(7.8, 6.1), xytext=(6.5, 6.9),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))
ax.text(7.3, 6.5, 'False', ha='center', va='center',
        fontsize=8, color=COLOR_ORANGE, fontweight='bold')

ax.annotate('', xy=(1.2, 4.7), xytext=(1.5, 5.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5))
ax.annotate('', xy=(3.25, 4.7), xytext=(2.7, 5.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=1.5))

ax.annotate('', xy=(6.9, 4.7), xytext=(7.2, 5.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5))
ax.annotate('', xy=(8.95, 4.7), xytext=(8.4, 5.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=1.5))

# Code
code_box = FancyBboxPatch((0.5, 0.3), 9, 3.0, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor='#F5F5F5', linewidth=2)
ax.add_patch(code_box)
ax.text(5, 3.1, 'Python Implementation', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    'price, volume = 145.00, 1200000',
    'buy_threshold, sell_threshold = 150.00, 170.00',
    'min_volume = 1000000',
    '',
    'if price < buy_threshold:',
    '    if volume > min_volume:',
    '        action = "BUY"',
    '    else:',
    '        action = "WAIT"',
    'elif price > sell_threshold:',
    '    action = "SELL"',
    'else:',
    '    action = "HOLD"',
]

y_pos = 2.7
for line in code_lines:
    ax.text(1, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color='black')
    y_pos -= 0.19

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
