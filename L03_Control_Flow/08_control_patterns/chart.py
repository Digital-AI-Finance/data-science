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


fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Common Control Flow Patterns', ha='center', va='top',
        fontsize=18, fontweight='bold', color=COLOR_SECONDARY)

# Pattern 1: Guard clause
pattern1_box = FancyBboxPatch((0.3, 6.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(pattern1_box)
ax.text(2.55, 8.9, 'Pattern 1: Guard Clause', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)

guard_code = [
    'def buy_stock(price, balance):',
    '    # Guard: Check preconditions',
    '    if price <= 0:',
    '        return "Invalid price"',
    '    if balance < price:',
    '        return "Insufficient funds"',
    '',
    '    # Main logic',
    '    execute_buy(price)',
]

y_pos = 8.5
for line in guard_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.6, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 2: Accumulator
pattern2_box = FancyBboxPatch((5.2, 6.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_ACCENT, facecolor='white', linewidth=2)
ax.add_patch(pattern2_box)
ax.text(7.45, 8.9, 'Pattern 2: Accumulator', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_ACCENT)

accum_code = [
    'prices = [150, 165, 148, 172]',
    'total = 0  # Accumulator',
    '',
    'for price in prices:',
    '    total += price',
    '',
    'average = total / len(prices)',
    'print(f"Avg: ${average:.2f}")',
]

y_pos = 8.5
for line in accum_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 3: Search/Find
pattern3_box = FancyBboxPatch((0.3, 3.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_GREEN, facecolor='white', linewidth=2)
ax.add_patch(pattern3_box)
ax.text(2.55, 5.9, 'Pattern 3: Search & Break', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_GREEN)

search_code = [
    'prices = [150, 165, 148, 172]',
    'found = False',
    '',
    'for price in prices:',
    '    if price < 150:',
    '        print(f"Found: ${price}")',
    '        found = True',
    '        break  # Stop searching',
]

y_pos = 5.5
for line in search_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.6, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 4: Filter
pattern4_box = FancyBboxPatch((5.2, 3.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_ORANGE, facecolor='white', linewidth=2)
ax.add_patch(pattern4_box)
ax.text(7.45, 5.9, 'Pattern 4: Filter Pattern', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_ORANGE)

filter_code = [
    'all_prices = [150, 165, 148, 172]',
    'high_prices = []  # Filtered list',
    '',
    'for price in all_prices:',
    '    if price > 160:',
    '        high_prices.append(price)',
    '',
    '# Result: [165, 172]',
]

y_pos = 5.5
for line in filter_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 5: Counter
pattern5_box = FancyBboxPatch((0.3, 0.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_RED, facecolor='white', linewidth=2)
ax.add_patch(pattern5_box)
ax.text(2.55, 2.9, 'Pattern 5: Counter', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_RED)

counter_code = [
    'prices = [150, 165, 148, 172, 145]',
    'count_low = 0  # Counter',
    '',
    'for price in prices:',
    '    if price < 150:',
    '        count_low += 1',
    '',
    'print(f"{count_low} low prices")',
]

y_pos = 2.5
for line in counter_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(0.6, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

# Pattern 6: Min/Max
pattern6_box = FancyBboxPatch((5.2, 0.8), 4.5, 2.3, boxstyle="round,pad=0.1",
                              edgecolor=COLOR_PRIMARY, facecolor='white', linewidth=2)
ax.add_patch(pattern6_box)
ax.text(7.45, 2.9, 'Pattern 6: Find Min/Max', ha='center', va='top',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY)

minmax_code = [
    'prices = [150, 165, 148, 172]',
    'max_price = prices[0]  # Initialize',
    '',
    'for price in prices:',
    '    if price > max_price:',
    '        max_price = price',
    '',
    'print(f"Max: ${max_price}")',
]

y_pos = 2.5
for line in minmax_code:
    if line.startswith('#'):
        color = '#808080'
    else:
        color = 'black'
    ax.text(5.5, y_pos, line, ha='left', va='top',
            fontsize=7, family='monospace', color=color)
    y_pos -= 0.22

plt.tight_layout()
plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('\nL03 COMPLETE: 8/8 charts generated')
