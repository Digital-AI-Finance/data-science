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

ax.text(5, 7.5, 'Summary Statistics: df.describe()', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Stats table
stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
values_aapl = ['252', '189.5', '8.2', '175.1', '183.4', '188.9', '195.2', '210.3']
values_msft = ['252', '385.2', '12.5', '355.8', '375.6', '384.1', '394.8', '420.5']

# Headers
ax.text(2, 6.5, 'Stat', fontsize=10, fontweight='bold', ha='center')
ax.text(4.5, 6.5, 'AAPL', fontsize=10, fontweight='bold', ha='center', color=MLBLUE)
ax.text(7, 6.5, 'MSFT', fontsize=10, fontweight='bold', ha='center', color=MLGREEN)

for i, (stat, aapl, msft) in enumerate(zip(stats, values_aapl, values_msft)):
    y = 6 - i*0.6
    ax.text(2, y, stat, fontsize=9, ha='center')
    ax.text(4.5, y, aapl, fontsize=9, ha='center', fontfamily='monospace')
    ax.text(7, y, msft, fontsize=9, ha='center', fontfamily='monospace')

ax.axhline(y=6.3, xmin=0.1, xmax=0.9, color=MLLAVENDER, linewidth=1)

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

