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

# Course colors already used in this lesson
COLOR_PRIMARY = MLPURPLE
COLOR_SECONDARY = MLPURPLE
COLOR_ACCENT = MLBLUE
COLOR_LIGHT = MLLAVENDER
COLOR_GREEN = MLGREEN
COLOR_ORANGE = MLORANGE
COLOR_RED = MLRED


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

ax.text(5, 7.5, 'Selection Methods Comparison', fontsize=14, fontweight='bold', ha='center', color=MLPURPLE)

# Table
headers = ['Method', 'Use Case', 'Returns']
data = [
    ['df["col"]', 'Single column', 'Series'],
    ['df[["col1","col2"]]', 'Multiple columns', 'DataFrame'],
    ['df.iloc[0]', 'Row by position', 'Series'],
    ['df.loc["date"]', 'Row by label', 'Series'],
    ['df[df.col > x]', 'Filter rows', 'DataFrame']
]

# Draw headers
for i, h in enumerate(headers):
    ax.text(1.5 + i*3, 6.5, h, fontsize=10, fontweight='bold', ha='center', color=MLPURPLE)

ax.axhline(y=6.2, xmin=0.1, xmax=0.9, color=MLLAVENDER, linewidth=2)

# Draw data
for row_i, row in enumerate(data):
    y = 5.7 - row_i * 0.8
    for col_i, val in enumerate(row):
        font = 'monospace' if col_i == 0 else 'sans-serif'
        ax.text(1.5 + col_i*3, y, val, fontsize=9, ha='center', fontfamily=font)

plt.savefig('chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
