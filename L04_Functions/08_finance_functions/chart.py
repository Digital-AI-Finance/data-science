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

ax.text(5, 7.5, 'Essential Finance Functions', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

functions = [
    ('calculate_return(p1, p2)', 'Price change %'),
    ('annualize_return(daily_ret)', 'Convert to yearly'),
    ('calculate_volatility(returns)', 'Standard deviation'),
    ('sharpe_ratio(ret, rf)', 'Risk-adjusted return'),
    ('max_drawdown(prices)', 'Largest peak-to-trough'),
    ('beta(stock, market)', 'Market sensitivity')
]

for i, (func, desc) in enumerate(functions):
    y = 6.5 - i * 1
    rect = patches.FancyBboxPatch((0.5, y - 0.3), 5, 0.7, boxstyle="round,pad=0.02",
                                   facecolor=MLLAVENDER, edgecolor=MLPURPLE, linewidth=1)
    ax.add_patch(rect)
    ax.text(0.7, y, func, fontsize=9, fontfamily='monospace', va='center')
    ax.text(6, y, desc, fontsize=9, va='center', color='gray')

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("\nL04 COMPLETE: 8/8 charts generated")

# =============================================================================
# L05: DataFrames Introduction
# =============================================================================