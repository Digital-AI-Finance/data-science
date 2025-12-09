"""Column Creation - Demonstrating different ways to create DataFrame columns"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
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

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Creating New Columns in pandas', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE)

# Method 1: Direct assignment
box1 = FancyBboxPatch((0.5, 7.0), 6, 2, boxstyle="round,pad=0.1",
                      edgecolor=MLBLUE, facecolor='#F0F8FF', linewidth=2)
ax.add_patch(box1)
ax.text(3.5, 8.6, '1. Direct Assignment', ha='center', fontsize=11,
        fontweight='bold', color=MLBLUE)
ax.text(3.5, 8.0, "df['Returns'] = df['Close'].pct_change()", ha='center',
        fontsize=9, family='monospace', color='black')
ax.text(3.5, 7.4, 'Simple, fast, most common method', ha='center',
        fontsize=9, style='italic', color='gray')

# Method 2: assign()
box2 = FancyBboxPatch((7.5, 7.0), 6, 2, boxstyle="round,pad=0.1",
                      edgecolor=MLORANGE, facecolor='#FFF8F0', linewidth=2)
ax.add_patch(box2)
ax.text(10.5, 8.6, '2. Using assign()', ha='center', fontsize=11,
        fontweight='bold', color=MLORANGE)
ax.text(10.5, 8.0, "df = df.assign(Returns=df['Close'].pct_change())", ha='center',
        fontsize=9, family='monospace', color='black')
ax.text(10.5, 7.4, 'Returns new DataFrame, chainable', ha='center',
        fontsize=9, style='italic', color='gray')

# Method 3: Arithmetic operations
box3 = FancyBboxPatch((0.5, 4.0), 6, 2, boxstyle="round,pad=0.1",
                      edgecolor=MLGREEN, facecolor='#F0FFF0', linewidth=2)
ax.add_patch(box3)
ax.text(3.5, 5.6, '3. Arithmetic Operations', ha='center', fontsize=11,
        fontweight='bold', color=MLGREEN)
ax.text(3.5, 5.0, "df['Spread'] = df['High'] - df['Low']", ha='center',
        fontsize=9, family='monospace', color='black')
ax.text(3.5, 4.4, 'Element-wise operations on columns', ha='center',
        fontsize=9, style='italic', color='gray')

# Method 4: Conditional assignment
box4 = FancyBboxPatch((7.5, 4.0), 6, 2, boxstyle="round,pad=0.1",
                      edgecolor=MLRED, facecolor='#FFF0F0', linewidth=2)
ax.add_patch(box4)
ax.text(10.5, 5.6, '4. Conditional (np.where)', ha='center', fontsize=11,
        fontweight='bold', color=MLRED)
ax.text(10.5, 5.0, "df['Signal'] = np.where(df['Returns']>0, 'Up', 'Down')", ha='center',
        fontsize=8, family='monospace', color='black')
ax.text(10.5, 4.4, 'Conditional column based on logic', ha='center',
        fontsize=9, style='italic', color='gray')

# Example DataFrame
example_box = FancyBboxPatch((2, 0.8), 10, 2.5, boxstyle="round,pad=0.1",
                             edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3)
ax.add_patch(example_box)
ax.text(7, 3.0, 'Example: Stock DataFrame After Column Creation', ha='center',
        fontsize=11, fontweight='bold', color=MLPURPLE)

# Table headers
headers = ['Date', 'Close', 'Returns', 'Spread', 'Signal']
x_pos = [2.8, 4.5, 6.5, 8.5, 10.8]
for x, h in zip(x_pos, headers):
    ax.text(x, 2.5, h, ha='center', fontsize=9, fontweight='bold', color=MLPURPLE)

# Sample data rows
data_rows = [
    ['2024-01-01', '$150.00', '-', '$3.50', '-'],
    ['2024-01-02', '$152.50', '+1.67%', '$4.20', 'Up'],
    ['2024-01-03', '$151.00', '-0.98%', '$3.80', 'Down'],
]
for i, row in enumerate(data_rows):
    y = 2.0 - i * 0.5
    for x, val in zip(x_pos, row):
        ax.text(x, y, val, ha='center', fontsize=8, color='black')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
