"""Datetime Parsing - Converting strings to datetime"""
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

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(8, 9.5, 'Datetime Parsing in pandas', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE)

# Common date formats
formats = [
    {'raw': '"2024-01-15"', 'format': '%Y-%m-%d', 'desc': 'ISO format (default)', 'color': MLBLUE},
    {'raw': '"01/15/2024"', 'format': '%m/%d/%Y', 'desc': 'US format', 'color': MLORANGE},
    {'raw': '"15-Jan-2024"', 'format': '%d-%b-%Y', 'desc': 'Day-Month-Year', 'color': MLGREEN},
    {'raw': '"Jan 15, 2024"', 'format': '%b %d, %Y', 'desc': 'Written format', 'color': MLRED},
]

ax.text(2, 8.5, 'String Input', ha='center', fontsize=11, fontweight='bold', color=MLPURPLE)
ax.text(6, 8.5, 'Format Code', ha='center', fontsize=11, fontweight='bold', color=MLPURPLE)
ax.text(11, 8.5, 'Description', ha='center', fontsize=11, fontweight='bold', color=MLPURPLE)

for i, fmt in enumerate(formats):
    y = 7.5 - i * 1.2

    # Input box
    box1 = FancyBboxPatch((0.5, y-0.3), 3, 0.6, boxstyle="round,pad=0.05",
                          edgecolor=fmt['color'], facecolor='white', linewidth=2)
    ax.add_patch(box1)
    ax.text(2, y, fmt['raw'], ha='center', va='center', fontsize=10, family='monospace')

    # Arrow
    ax.annotate('', xy=(4.2, y), xytext=(3.7, y),
                arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))

    # Format box
    box2 = FancyBboxPatch((4.5, y-0.3), 3, 0.6, boxstyle="round,pad=0.05",
                          edgecolor=fmt['color'], facecolor=MLLAVENDER, alpha=0.3, linewidth=1.5)
    ax.add_patch(box2)
    ax.text(6, y, fmt['format'], ha='center', va='center', fontsize=10, family='monospace', color=fmt['color'])

    # Description
    ax.text(9, y, fmt['desc'], va='center', fontsize=10, color='gray')

# Parsing methods section
ax.text(8, 3.5, 'Parsing Methods', ha='center', fontsize=12, fontweight='bold', color=MLPURPLE)

methods = [
    ("pd.to_datetime('2024-01-15')", "Auto-detect format", MLBLUE),
    ("pd.to_datetime(df['date'], format='%Y-%m-%d')", "Specify format (faster)", MLORANGE),
    ("pd.read_csv('file.csv', parse_dates=['date'])", "Parse during import", MLGREEN),
]

for i, (code, desc, color) in enumerate(methods):
    y = 2.8 - i * 0.8

    box = FancyBboxPatch((1, y-0.25), 9, 0.5, boxstyle="round,pad=0.05",
                         edgecolor=color, facecolor='white', linewidth=1.5)
    ax.add_patch(box)
    ax.text(5.5, y, code, ha='center', va='center', fontsize=9, family='monospace', color=color)
    ax.text(11.5, y, desc, va='center', fontsize=9, color='gray')

# Format codes reference
ref_box = FancyBboxPatch((0.5, 0.3), 15, 0.8, boxstyle="round,pad=0.1",
                         edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3)
ax.add_patch(ref_box)
ax.text(8, 0.7, 'Common Format Codes: %Y=year(4) %y=year(2) %m=month %d=day %H=hour %M=min %b=month(abbr)',
        ha='center', fontsize=9, family='monospace')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
