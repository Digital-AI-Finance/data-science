"""pandas Operations Cheat Sheet - Visual reference guide"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
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

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(8, 9.7, 'pandas Basic Operations - Quick Reference', ha='center',
        fontsize=18, fontweight='bold', color=MLPURPLE)

# Categories
categories = [
    {
        'title': 'Column Creation',
        'color': MLBLUE,
        'x': 0.3, 'y': 7.5,
        'items': [
            ("df['new'] = values", "Direct assignment"),
            ("df.assign(new=values)", "Returns new DataFrame"),
            ("df.insert(loc, 'col', values)", "Insert at position"),
        ]
    },
    {
        'title': 'Arithmetic',
        'color': MLORANGE,
        'x': 5.5, 'y': 7.5,
        'items': [
            ("df['A'] + df['B']", "Column addition"),
            ("df['A'] * 2", "Scalar multiplication"),
            ("df['A'].pct_change()", "Percentage change"),
        ]
    },
    {
        'title': 'Transformations',
        'color': MLGREEN,
        'x': 10.7, 'y': 7.5,
        'items': [
            ("df['A'].apply(func)", "Apply function"),
            ("df.applymap(func)", "Element-wise"),
            ("df.transform(func)", "Same shape output"),
        ]
    },
    {
        'title': 'Sorting',
        'color': MLRED,
        'x': 0.3, 'y': 4.0,
        'items': [
            ("df.sort_values('col')", "Sort by column"),
            ("df.sort_values(ascending=False)", "Descending order"),
            ("df.sort_index()", "Sort by index"),
        ]
    },
    {
        'title': 'Aggregation',
        'color': MLPURPLE,
        'x': 5.5, 'y': 4.0,
        'items': [
            ("df['A'].value_counts()", "Frequency count"),
            ("df['A'].unique()", "Unique values"),
            ("df['A'].nunique()", "Count unique"),
        ]
    },
    {
        'title': 'Rolling Stats',
        'color': '#008080',  # Teal
        'x': 10.7, 'y': 4.0,
        'items': [
            ("df.rolling(N).mean()", "Moving average"),
            ("df.rolling(N).std()", "Rolling std dev"),
            ("df.expanding().sum()", "Expanding sum"),
        ]
    },
]

for cat in categories:
    # Category box
    box = FancyBboxPatch((cat['x'], cat['y']-2), 4.8, 2.5, boxstyle="round,pad=0.1",
                         edgecolor=cat['color'], facecolor='white', linewidth=2)
    ax.add_patch(box)

    # Category title
    ax.text(cat['x'] + 2.4, cat['y'] + 0.3, cat['title'], ha='center',
            fontsize=11, fontweight='bold', color=cat['color'])

    # Items
    for i, (code, desc) in enumerate(cat['items']):
        y_pos = cat['y'] - 0.4 - i * 0.6
        ax.text(cat['x'] + 0.2, y_pos, code, fontsize=8, family='monospace',
                color='black')
        ax.text(cat['x'] + 0.2, y_pos - 0.25, desc, fontsize=7, color='gray',
                style='italic')

# Quick tips box at bottom
tips_box = FancyBboxPatch((0.5, 0.3), 15, 1.2, boxstyle="round,pad=0.1",
                          edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3)
ax.add_patch(tips_box)
ax.text(8, 1.2, 'Finance Tips', ha='center', fontsize=11, fontweight='bold', color=MLPURPLE)
tips = [
    "Returns: df['Price'].pct_change()",
    "Log Returns: np.log(df['Price']/df['Price'].shift(1))",
    "Cumulative: (1 + returns).cumprod() - 1",
    "Volatility: returns.rolling(20).std()",
]
ax.text(8, 0.7, '  |  '.join(tips), ha='center', fontsize=8, family='monospace', color='black')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
