"""GroupBy Patterns - Common patterns and use cases"""
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
ax.text(8, 9.7, 'Common GroupBy Patterns', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE)

# Pattern boxes
patterns = [
    {
        'title': 'Basic Aggregation',
        'code': "df.groupby('Sector')['Return'].mean()",
        'desc': 'Single column, single function',
        'color': MLBLUE,
        'x': 0.3, 'y': 7.2
    },
    {
        'title': 'Multiple Functions',
        'code': "df.groupby('Sector')['Return'].agg(['mean', 'std', 'count'])",
        'desc': 'Multiple statistics at once',
        'color': MLORANGE,
        'x': 5.5, 'y': 7.2
    },
    {
        'title': 'Multiple Columns',
        'code': "df.groupby('Sector')[['Return', 'Volume']].mean()",
        'desc': 'Aggregate multiple columns',
        'color': MLGREEN,
        'x': 10.7, 'y': 7.2
    },
    {
        'title': 'Named Aggregation',
        'code': "df.groupby('Sector').agg(\n    avg_return=('Return', 'mean'),\n    total_vol=('Volume', 'sum'))",
        'desc': 'Custom column names',
        'color': MLRED,
        'x': 0.3, 'y': 4.2
    },
    {
        'title': 'Custom Function',
        'code': "df.groupby('Sector')['Return'].apply(\n    lambda x: x.quantile(0.95))",
        'desc': 'Use any function',
        'color': MLPURPLE,
        'x': 5.5, 'y': 4.2
    },
    {
        'title': 'Filter Groups',
        'code': "df.groupby('Sector').filter(\n    lambda x: x['Return'].mean() > 0.05)",
        'desc': 'Keep groups meeting condition',
        'color': '#008080',
        'x': 10.7, 'y': 4.2
    },
]

for pat in patterns:
    # Box
    box = FancyBboxPatch((pat['x'], pat['y'] - 2), 4.8, 2.8, boxstyle="round,pad=0.1",
                         edgecolor=pat['color'], facecolor='white', linewidth=2)
    ax.add_patch(box)

    # Title
    ax.text(pat['x'] + 2.4, pat['y'] + 0.5, pat['title'], ha='center',
            fontsize=11, fontweight='bold', color=pat['color'])

    # Code
    lines = pat['code'].split('\n')
    for i, line in enumerate(lines):
        ax.text(pat['x'] + 0.2, pat['y'] - 0.3 - i*0.4, line,
                fontsize=8, family='monospace', color='black')

    # Description
    ax.text(pat['x'] + 2.4, pat['y'] - 1.6, pat['desc'], ha='center',
            fontsize=9, style='italic', color='gray')

# Tips section at bottom
tips_box = FancyBboxPatch((0.5, 0.5), 15, 1.3, boxstyle="round,pad=0.1",
                          edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3)
ax.add_patch(tips_box)
ax.text(8, 1.5, 'Pro Tips', ha='center', fontsize=11, fontweight='bold', color=MLPURPLE)
tips = [
    "as_index=False: Keep grouping column as regular column",
    "sort=False: Maintain original order (faster)",
    "observed=True: Only show categories present in data",
]
ax.text(8, 1.0, '  |  '.join(tips), ha='center', fontsize=8, color='black')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
