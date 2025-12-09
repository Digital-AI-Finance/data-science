"""Apply Function - Demonstrating apply() for custom transformations"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left panel: Concept diagram
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.text(5, 9.5, 'How apply() Works', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE)

# Input column
input_box = FancyBboxPatch((0.5, 5), 2.5, 3.5, boxstyle="round,pad=0.1",
                           edgecolor=MLBLUE, facecolor='#F0F8FF', linewidth=2)
ax1.add_patch(input_box)
ax1.text(1.75, 8.2, 'Input Column', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
values = ['100.0', '105.5', '103.2', '108.7', '107.0']
for i, v in enumerate(values):
    ax1.text(1.75, 7.5 - i*0.5, v, ha='center', fontsize=9, color='black')

# Function box
func_box = FancyBboxPatch((3.5, 6), 3, 1.5, boxstyle="round,pad=0.1",
                          edgecolor=MLORANGE, facecolor='#FFF8F0', linewidth=2)
ax1.add_patch(func_box)
ax1.text(5, 7.2, 'Function', ha='center', fontsize=10, fontweight='bold', color=MLORANGE)
ax1.text(5, 6.5, 'lambda x: x * 1.1', ha='center', fontsize=9, family='monospace')

# Output column
output_box = FancyBboxPatch((7, 5), 2.5, 3.5, boxstyle="round,pad=0.1",
                            edgecolor=MLGREEN, facecolor='#F0FFF0', linewidth=2)
ax1.add_patch(output_box)
ax1.text(8.25, 8.2, 'Output Column', ha='center', fontsize=10, fontweight='bold', color=MLGREEN)
out_values = ['110.0', '116.1', '113.5', '119.6', '117.7']
for i, v in enumerate(out_values):
    ax1.text(8.25, 7.5 - i*0.5, v, ha='center', fontsize=9, color='black')

# Arrows
ax1.annotate('', xy=(3.5, 6.75), xytext=(3.0, 6.75),
             arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))
ax1.annotate('', xy=(7.0, 6.75), xytext=(6.5, 6.75),
             arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))

# Code example
ax1.text(5, 3.5, "df['Price_Adjusted'] = df['Price'].apply(lambda x: x * 1.1)",
         ha='center', fontsize=9, family='monospace', color=MLPURPLE,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.3))

# Types of apply
ax1.text(5, 2.5, 'Types of apply():', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)
ax1.text(5, 2.0, 'Column-wise: df["col"].apply(func)', ha='center', fontsize=9, color='black')
ax1.text(5, 1.5, 'Row-wise: df.apply(func, axis=1)', ha='center', fontsize=9, color='black')
ax1.text(5, 1.0, 'Element-wise: df.applymap(func)', ha='center', fontsize=9, color='black')

# Right panel: Real example with stock data
ax2 = axes[1]

# Generate sample stock data
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(20) * 2)

# Apply different functions
categories = []
for p in prices:
    if p < 100:
        categories.append('Undervalued')
    elif p > 110:
        categories.append('Overvalued')
    else:
        categories.append('Fair Value')

colors_cat = [MLGREEN if c == 'Undervalued' else MLRED if c == 'Overvalued' else MLBLUE for c in categories]

ax2.bar(range(len(prices)), prices, color=colors_cat, alpha=0.7, edgecolor='black')
ax2.axhline(100, color=MLGREEN, linestyle='--', linewidth=2, label='Undervalued < $100')
ax2.axhline(110, color=MLRED, linestyle='--', linewidth=2, label='Overvalued > $110')

ax2.set_xlabel('Trading Day', fontsize=11)
ax2.set_ylabel('Stock Price ($)', fontsize=11)
ax2.set_title('apply() Example: Price Categorization', fontsize=12,
              fontweight='bold', color=MLPURPLE)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Add code snippet
code_text = "def categorize(price):\n    if price < 100: return 'Undervalued'\n    elif price > 110: return 'Overvalued'\n    else: return 'Fair Value'\n\ndf['Category'] = df['Price'].apply(categorize)"
ax2.text(0.95, 0.05, code_text, transform=ax2.transAxes, fontsize=8,
         family='monospace', va='bottom', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
