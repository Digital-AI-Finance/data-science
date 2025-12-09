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

ax.text(5, 9.5, 'List Slicing: [start:stop:step]', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Original list
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'TSLA', 'NVDA']
n = len(tickers)
box_width = 1.0
start_x = 1.5

ax.text(0.5, 8.2, 'tickers =', ha='left', va='center',
        fontsize=11, family='monospace', fontweight='bold', color='black')

for i, ticker in enumerate(tickers):
    x = start_x + i * box_width
    box = FancyBboxPatch((x, 7.5), box_width-0.1, 0.8, boxstyle="round,pad=0.05",
                         edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(box)
    ax.text(x + (box_width-0.1)/2, 7.9, ticker, ha='center', va='center',
            fontsize=9, fontweight='bold', color=COLOR_SECONDARY)
    # Index
    ax.text(x + (box_width-0.1)/2, 7.2, f'[{i}]', ha='center', va='center',
            fontsize=8, color=COLOR_ACCENT)

# Slicing examples
slicing_examples = [
    ('tickers[0:3]', [0, 1, 2], ['AAPL', 'MSFT', 'GOOGL'], 'First 3 items', 6.0),
    ('tickers[2:5]', [2, 3, 4], ['GOOGL', 'AMZN', 'SPY'], 'Items 2 to 5', 4.5),
    ('tickers[::2]', [0, 2, 4, 6], ['AAPL', 'GOOGL', 'SPY', 'NVDA'], 'Every 2nd item', 3.0),
    ('tickers[-3:]', [4, 5, 6], ['SPY', 'TSLA', 'NVDA'], 'Last 3 items', 1.5),
]

for slice_code, indices, result_list, desc, y in slicing_examples:
    # Slice code
    ax.text(0.5, y + 0.5, slice_code, ha='left', va='center',
            fontsize=10, family='monospace', fontweight='bold', color='black')

    # Result visualization
    for j, (idx, val) in enumerate(zip(indices, result_list)):
        x = start_x + j * box_width
        box = FancyBboxPatch((x, y), box_width-0.1, 0.6, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + (box_width-0.1)/2, y + 0.3, val, ha='center', va='center',
                fontsize=8, fontweight='bold', color=COLOR_GREEN)

    # Description
    ax.text(9.0, y + 0.3, desc, ha='right', va='center',
            fontsize=9, color='#808080', style='italic')

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
