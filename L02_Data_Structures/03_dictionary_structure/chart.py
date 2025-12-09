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

ax.text(5, 9.5, 'Dictionary: Key-Value Pairs', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Dictionary representation
portfolio = {
    'AAPL': 150.50,
    'MSFT': 340.00,
    'GOOGL': 125.75,
    'AMZN': 165.00
}

# Visual representation
ax.text(5, 8.7, 'portfolio = { }', ha='center', va='top',
        fontsize=12, family='monospace', fontweight='bold', color='black')

y_pos = 7.8
for ticker, price in portfolio.items():
    # Key box
    key_box = FancyBboxPatch((2.0, y_pos - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                             edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(key_box)
    ax.text(3.0, y_pos, f'"{ticker}"', ha='center', va='center',
            fontsize=10, family='monospace', fontweight='bold', color=COLOR_SECONDARY)

    # Arrow
    ax.text(4.3, y_pos, ':', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLOR_ACCENT)

    # Value box
    value_box = FancyBboxPatch((4.8, y_pos - 0.4), 2, 0.7, boxstyle="round,pad=0.05",
                               edgecolor=COLOR_GREEN, facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(value_box)
    ax.text(5.8, y_pos, f'{price}', ha='center', va='center',
            fontsize=10, family='monospace', fontweight='bold', color=COLOR_GREEN)

    # Labels (only for first row)
    if y_pos == 7.8:
        ax.text(3.0, y_pos + 0.6, 'KEY', ha='center', va='center',
                fontsize=9, fontweight='bold', color=COLOR_PRIMARY)
        ax.text(5.8, y_pos + 0.6, 'VALUE', ha='center', va='center',
                fontsize=9, fontweight='bold', color=COLOR_GREEN)

    y_pos -= 1.0

# Access examples
examples_box = FancyBboxPatch((1.0, 1.0), 8, 2.8, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(examples_box)
ax.text(5, 3.6, 'Dictionary Operations', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

operations = [
    ('portfolio["AAPL"]', '150.5', 'Access value by key'),
    ('portfolio["AAPL"] = 155.00', 'None', 'Update value'),
    ('portfolio["TSLA"] = 250.00', 'None', 'Add new key-value pair'),
    ('"MSFT" in portfolio', 'True', 'Check if key exists'),
]

y_pos = 3.0
for code, result, desc in operations:
    ax.text(1.5, y_pos, code, ha='left', va='top',
            fontsize=9, family='monospace', color='black', fontweight='bold')
    if result != 'None':
        ax.text(5.5, y_pos, '→', ha='center', va='top',
                fontsize=11, color=COLOR_ACCENT)
        ax.text(6.0, y_pos, result, ha='left', va='top',
                fontsize=9, family='monospace', color=COLOR_GREEN, fontweight='bold')
    ax.text(7.2, y_pos, f'# {desc}', ha='left', va='top',
            fontsize=8, family='monospace', color='#808080')
    y_pos -= 0.6

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
