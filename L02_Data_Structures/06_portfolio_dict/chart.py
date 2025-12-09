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

ax.text(5, 9.5, 'Portfolio Representation: Dictionary', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Portfolio data
portfolio = {
    'AAPL': {'shares': 100, 'buy_price': 145.00, 'current_price': 150.50},
    'MSFT': {'shares': 50, 'buy_price': 320.00, 'current_price': 340.00},
    'GOOGL': {'shares': 75, 'buy_price': 120.00, 'current_price': 125.75}
}

# Visual table
ax.text(5, 8.5, 'portfolio = {', ha='center', va='top',
        fontsize=11, family='monospace', fontweight='bold', color='black')

y_pos = 7.8
for ticker, data in portfolio.items():
    # Stock ticker
    ticker_box = FancyBboxPatch((1.0, y_pos - 0.3), 1.2, 0.6, boxstyle="round,pad=0.05",
                                edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(ticker_box)
    ax.text(1.6, y_pos, f'"{ticker}"', ha='center', va='center',
            fontsize=9, family='monospace', fontweight='bold', color=COLOR_SECONDARY)

    # Details
    shares = data['shares']
    buy_price = data['buy_price']
    current_price = data['current_price']
    gain_loss = (current_price - buy_price) * shares
    gain_pct = ((current_price - buy_price) / buy_price) * 100

    details = f'shares: {shares}, buy: ${buy_price}, current: ${current_price}'
    details_box = FancyBboxPatch((2.5, y_pos - 0.3), 6.5, 0.6, boxstyle="round,pad=0.05",
                                 edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
    ax.add_patch(details_box)
    ax.text(5.75, y_pos, details, ha='center', va='center',
            fontsize=8, family='monospace', color='black')

    y_pos -= 0.9

ax.text(5, y_pos + 0.3, '}', ha='center', va='top',
        fontsize=11, family='monospace', fontweight='bold', color='black')

# Calculations
calc_box = FancyBboxPatch((0.5, 1.0), 9, 3.8, boxstyle="round,pad=0.1",
                          edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(calc_box)
ax.text(5, 4.6, 'Portfolio Calculations', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_GREEN)

calculations = [
    '# Calculate total value',
    'total_value = 0',
    'for ticker in portfolio:',
    '    shares = portfolio[ticker]["shares"]',
    '    price = portfolio[ticker]["current_price"]',
    '    total_value += shares * price',
    '',
    'print(f"Total portfolio value: ${total_value:,.2f}")',
    '# Output: Total portfolio value: $31,481.25',
]

y_pos = 4.2
for line in calculations:
    if line.startswith('#') or 'Output' in line:
        color = '#808080'
        weight = 'normal'
    else:
        color = 'black'
        weight = 'normal'

    ax.text(1.0, y_pos, line, ha='left', va='top',
            fontsize=8, family='monospace', color=color, fontweight=weight)
    y_pos -= 0.35

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
