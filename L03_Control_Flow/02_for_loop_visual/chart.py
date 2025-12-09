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

ax.text(5, 9.5, 'for Loop: Iteration Over Sequence', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# List to iterate over
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
prices = [150.50, 340.00, 125.75, 165.00]

# Original list
ax.text(1, 8.2, 'tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]', ha='left', va='top',
        fontsize=10, family='monospace', color='black', fontweight='bold')

# Loop visualization
box_width = 1.8
start_x = 1.5

for i, (ticker, price) in enumerate(zip(tickers, prices)):
    x = start_x + i * box_width

    # Iteration number
    iter_circle = Circle((x + 0.7, 7.2), 0.3, edgecolor=COLOR_PRIMARY,
                         facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(iter_circle)
    ax.text(x + 0.7, 7.2, str(i), ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

    # Item box
    item_box = FancyBboxPatch((x, 6.0), 1.4, 0.7, boxstyle="round,pad=0.05",
                              edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
    ax.add_patch(item_box)
    ax.text(x + 0.7, 6.35, f'"{ticker}"', ha='center', va='center',
            fontsize=9, family='monospace', color='black', fontweight='bold')

    # Arrow
    ax.annotate('', xy=(x + 0.7, 6.0), xytext=(x + 0.7, 6.9),
                arrowprops=dict(arrowstyle='->', color=COLOR_SECONDARY, lw=2))

    # Process box
    process_box = FancyBboxPatch((x, 4.8), 1.4, 0.9, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
    ax.add_patch(process_box)
    ax.text(x + 0.7, 5.5, 'Process', ha='center', va='center',
            fontsize=8, fontweight='bold', color=COLOR_GREEN)
    ax.text(x + 0.7, 5.15, f'${price}', ha='center', va='center',
            fontsize=9, family='monospace', color='black')

# Loop label
ax.text(0.5, 6.35, 'for ticker in tickers:', ha='left', va='center',
        fontsize=10, family='monospace', color=COLOR_PRIMARY, fontweight='bold')

# Code example
code_box = FancyBboxPatch((0.5, 0.5), 9, 3.8, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(code_box)
ax.text(5, 4.1, 'Loop Example: Calculate Total Portfolio Value', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    'portfolio = {"AAPL": 150.50, "MSFT": 340.00, "GOOGL": 125.75}',
    'shares = {"AAPL": 100, "MSFT": 50, "GOOGL": 75}',
    '',
    'total_value = 0',
    'for ticker in portfolio:',
    '    price = portfolio[ticker]',
    '    num_shares = shares[ticker]',
    '    value = price * num_shares',
    '    total_value += value',
    '    print(f"{ticker}: ${value:,.2f}")',
    '',
    'print(f"Total: ${total_value:,.2f}")',
    '# Output: Total: $31,481.25',
]

y_pos = 3.7
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
