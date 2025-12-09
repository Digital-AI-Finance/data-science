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


fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Nested Loops: Loop Within a Loop', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Outer loop
outer_box = FancyBboxPatch((0.5, 6.0), 9, 2.8, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
ax.add_patch(outer_box)
ax.text(1, 8.6, 'Outer Loop: for ticker in ["AAPL", "MSFT", "GOOGL"]', ha='left', va='top',
        fontsize=10, family='monospace', fontweight='bold', color=COLOR_SECONDARY)

# Inner loop
inner_box = FancyBboxPatch((1.5, 6.5), 7, 1.8, boxstyle="round,pad=0.1",
                           edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(inner_box)
ax.text(2, 8.1, 'Inner Loop: for day in range(5)', ha='left', va='top',
        fontsize=10, family='monospace', fontweight='bold', color=COLOR_ACCENT)

# Process
process_box = FancyBboxPatch((2.5, 6.8), 5, 1.2, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=1.5)
ax.add_patch(process_box)
ax.text(5, 7.6, 'Process each ticker for each day', ha='center', va='center',
        fontsize=9, fontweight='bold', color=COLOR_GREEN)
ax.text(5, 7.2, 'Total iterations: 3 tickers × 5 days = 15', ha='center', va='center',
        fontsize=9, color='black')

# Example output
example_box = FancyBboxPatch((0.5, 0.5), 9, 5.0, boxstyle="round,pad=0.1",
                             edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(example_box)
ax.text(5, 5.3, 'Nested Loop Example: Price Matrix', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

code_lines = [
    'tickers = ["AAPL", "MSFT", "GOOGL"]',
    'days = ["Mon", "Tue", "Wed", "Thu", "Fri"]',
    '',
    'for ticker in tickers:                    # Outer loop (3 iterations)',
    '    print(f"\\n{ticker} prices:")',
    '    for day in days:                      # Inner loop (5 iterations)',
    '        price = get_price(ticker, day)    # Called 15 times total',
    '        print(f"  {day}: ${price:.2f}")',
    '',
    '# Output:',
    '# AAPL prices:',
    '#   Mon: $150.50',
    '#   Tue: $151.25',
    '#   ...',
]

y_pos = 4.9
for line in code_lines:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(1, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color)
    y_pos -= 0.3

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
