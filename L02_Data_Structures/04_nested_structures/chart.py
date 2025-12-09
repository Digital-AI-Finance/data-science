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

ax.text(5, 9.5, 'Nested Data Structures', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Portfolio with nested structure
ax.text(5, 8.8, 'portfolio = {', ha='center', va='top',
        fontsize=11, family='monospace', fontweight='bold', color='black')

stocks = [
    ('AAPL', 150.50, 100, 15050.00),
    ('MSFT', 340.00, 50, 17000.00),
    ('GOOGL', 125.75, 75, 9431.25),
]

y_pos = 8.0
for ticker, price, shares, value in stocks:
    # Outer key
    outer_box = FancyBboxPatch((1.5, y_pos - 0.3), 1.5, 0.6, boxstyle="round,pad=0.05",
                               edgecolor=COLOR_PRIMARY, facecolor=COLOR_LIGHT, linewidth=2)
    ax.add_patch(outer_box)
    ax.text(2.25, y_pos, f'"{ticker}"', ha='center', va='center',
            fontsize=9, family='monospace', fontweight='bold', color=COLOR_SECONDARY)

    ax.text(3.2, y_pos, ':', ha='center', va='center',
            fontsize=12, fontweight='bold', color='black')

    # Nested dictionary
    nested_box = FancyBboxPatch((3.5, y_pos - 0.5), 5.5, 0.9, boxstyle="round,pad=0.05",
                                edgecolor=COLOR_ACCENT, facecolor='white', linewidth=1.5)
    ax.add_patch(nested_box)

    nested_text = f'{{"price": {price}, "shares": {shares}, "value": {value}}}'
    ax.text(6.25, y_pos, nested_text, ha='center', va='center',
            fontsize=8, family='monospace', color='black')

    y_pos -= 1.2

ax.text(5, y_pos, '}', ha='center', va='top',
        fontsize=11, family='monospace', fontweight='bold', color='black')

# Access examples
examples_box = FancyBboxPatch((0.5, 1.0), 9, 3.2, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_SECONDARY, facecolor='white', linewidth=2)
ax.add_patch(examples_box)
ax.text(5, 4.0, 'Accessing Nested Data', ha='center', va='top',
        fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

access_examples = [
    ('portfolio["AAPL"]', '{"price": 150.5, "shares": 100, ...}', 'Full nested dict'),
    ('portfolio["AAPL"]["price"]', '150.5', 'Specific value'),
    ('portfolio["MSFT"]["shares"]', '50', 'Shares for MSFT'),
    ('portfolio["GOOGL"]["value"]', '9431.25', 'Total value'),
]

y_pos = 3.4
for code, result, desc in access_examples:
    ax.text(1.0, y_pos, code, ha='left', va='top',
            fontsize=9, family='monospace', color='black', fontweight='bold')
    ax.text(4.8, y_pos, '→', ha='center', va='top',
            fontsize=11, color=COLOR_ACCENT)
    ax.text(5.2, y_pos, result, ha='left', va='top',
            fontsize=8, family='monospace', color=COLOR_GREEN, fontweight='bold')
    ax.text(7.5, y_pos, f'# {desc}', ha='left', va='top',
            fontsize=8, family='monospace', color='#808080')
    y_pos -= 0.6

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
