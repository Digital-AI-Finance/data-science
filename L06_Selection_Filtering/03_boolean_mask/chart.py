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

# Course colors already used in this lesson
COLOR_PRIMARY = MLPURPLE
COLOR_SECONDARY = MLPURPLE
COLOR_ACCENT = MLBLUE
COLOR_LIGHT = MLLAVENDER
COLOR_GREEN = MLGREEN
COLOR_ORANGE = MLORANGE
COLOR_RED = MLRED


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

ax.text(5, 7.5, 'Boolean Masking', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Original data
ax.text(1.5, 6.5, 'AAPL', fontsize=10, fontweight='bold', ha='center')
prices = [185, 190, 188, 195, 182]
for i, p in enumerate(prices):
    ax.text(1.5, 5.8 - i*0.6, str(p), fontsize=10, ha='center', fontfamily='monospace')

# Condition
ax.text(4, 6.5, 'df["AAPL"] > 188', fontsize=10, fontfamily='monospace', ha='center', color=MLBLUE)

# Boolean mask
ax.text(6.5, 6.5, 'Mask', fontsize=10, fontweight='bold', ha='center')
masks = ['False', 'True', 'False', 'True', 'False']
colors = [MLRED, MLGREEN, MLRED, MLGREEN, MLRED]
for i, (m, c) in enumerate(zip(masks, colors)):
    ax.text(6.5, 5.8 - i*0.6, m, fontsize=10, ha='center', fontfamily='monospace', color=c)

# Result
ax.text(8.5, 6.5, 'Result', fontsize=10, fontweight='bold', ha='center')
ax.text(8.5, 5.8, '190', fontsize=10, ha='center', fontfamily='monospace', color=MLGREEN)
ax.text(8.5, 5.2, '195', fontsize=10, ha='center', fontfamily='monospace', color=MLGREEN)

# Arrows
ax.annotate('', xy=(3.5, 5), xytext=(2.5, 5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
ax.annotate('', xy=(6, 5), xytext=(5, 5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))
ax.annotate('', xy=(8, 5), xytext=(7, 5), arrowprops=dict(arrowstyle='->', color=MLPURPLE))

ax.text(5, 2.5, 'Boolean mask filters rows where condition is True', fontsize=10, ha='center', style='italic')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
