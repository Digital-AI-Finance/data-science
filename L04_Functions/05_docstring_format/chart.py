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

ax.text(5, 7.5, 'Docstring Best Practices', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

code = '''def calculate_sharpe(returns, rf_rate=0.02):
"""
Calculate the Sharpe ratio for a series of returns.

Parameters:
    returns (array): Daily return values
    rf_rate (float): Risk-free rate (default: 0.02)

Returns:
    float: Annualized Sharpe ratio
"""
excess = returns.mean() - rf_rate/252
return excess / returns.std() * np.sqrt(252)'''

rect = patches.FancyBboxPatch((0.5, 0.5), 9, 6.5, boxstyle="round,pad=0.1",
                               facecolor='#F5F5F5', edgecolor=MLPURPLE, linewidth=1.5)
ax.add_patch(rect)

y_pos = 6.5
for line in code.split('\n'):
    ax.text(1, y_pos, line, fontsize=9, fontfamily='monospace')
    y_pos -= 0.45

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
