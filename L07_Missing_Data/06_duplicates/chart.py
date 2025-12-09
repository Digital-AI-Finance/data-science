"""Duplicate Detection - Visualization of duplicate handling in stock data"""
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

# Create sample data with duplicates
np.random.seed(42)
dates = ['2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03',
         '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-05']
prices = [100.0, 101.5, 101.5, 102.0, 102.3, 102.0, 103.0, 104.5, 104.5]
volumes = [1000, 1500, 1500, 2000, 2100, 2000, 1800, 2200, 2200]

df = pd.DataFrame({'Date': dates, 'Price': prices, 'Volume': volumes})

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left: Table visualization showing duplicates
ax1 = axes[0]
ax1.axis('off')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)

# Title
ax1.text(5, 11.5, 'Detecting Duplicates in Stock Data', ha='center',
         fontsize=14, fontweight='bold', color=MLPURPLE)

# Column headers
headers = ['Index', 'Date', 'Price', 'Volume', 'Duplicate?']
x_positions = [1, 2.5, 4.5, 6.5, 8.5]
for x, header in zip(x_positions, headers):
    ax1.text(x, 10.5, header, ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)

# Draw header line
ax1.axhline(y=10.2, xmin=0.05, xmax=0.95, color=MLPURPLE, linewidth=2)

# Data rows
duplicates = df.duplicated(keep='first')
for i, (idx, row) in enumerate(df.iterrows()):
    y = 9.5 - i * 1.0
    is_dup = duplicates[idx]

    # Background color
    if is_dup:
        rect = FancyBboxPatch((0.3, y-0.35), 9.4, 0.7, boxstyle="round,pad=0.02",
                              facecolor=MLRED, alpha=0.2, edgecolor=MLRED)
        ax1.add_patch(rect)

    # Data values
    ax1.text(x_positions[0], y, str(idx), ha='center', fontsize=9)
    ax1.text(x_positions[1], y, row['Date'], ha='center', fontsize=9)
    ax1.text(x_positions[2], y, f"${row['Price']:.1f}", ha='center', fontsize=9)
    ax1.text(x_positions[3], y, f"{row['Volume']:,}", ha='center', fontsize=9)
    ax1.text(x_positions[4], y, 'Yes' if is_dup else 'No', ha='center', fontsize=9,
             color=MLRED if is_dup else MLGREEN, fontweight='bold')

# Summary box
summary_box = FancyBboxPatch((0.5, 0.3), 9, 1.2, boxstyle="round,pad=0.1",
                             edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3)
ax1.add_patch(summary_box)
ax1.text(5, 1.1, f'Total Rows: {len(df)}  |  Duplicates: {duplicates.sum()}  |  Unique: {len(df) - duplicates.sum()}',
         ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)
ax1.text(5, 0.5, 'df.duplicated() marks all rows after first occurrence as duplicates',
         ha='center', fontsize=9, style='italic', color='gray')

# Right: Methods comparison
ax2 = axes[1]
methods = ['duplicated()', 'duplicated(keep="last")', 'duplicated(keep=False)',
           'drop_duplicates()', 'drop_duplicates(subset=["Date"])']
descriptions = [
    'Marks duplicates (keeps first)',
    'Marks duplicates (keeps last)',
    'Marks ALL duplicates',
    'Removes duplicate rows',
    'Removes by Date only'
]
results = [
    f'{duplicates.sum()} duplicates found',
    f'{df.duplicated(keep="last").sum()} duplicates found',
    f'{df.duplicated(keep=False).sum()} involved in duplication',
    f'{len(df.drop_duplicates())} rows remain',
    f'{len(df.drop_duplicates(subset=["Date"]))} rows remain'
]

ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

ax2.text(5, 9.5, 'Duplicate Handling Methods', ha='center',
         fontsize=14, fontweight='bold', color=MLPURPLE)

for i, (method, desc, result) in enumerate(zip(methods, descriptions, results)):
    y = 8.0 - i * 1.5

    # Method box
    method_box = FancyBboxPatch((0.3, y-0.3), 4.5, 0.6, boxstyle="round,pad=0.05",
                                edgecolor=MLBLUE, facecolor='#F0F0FF', linewidth=1.5)
    ax2.add_patch(method_box)
    ax2.text(2.55, y, method, ha='center', va='center', fontsize=9,
             family='monospace', color=MLBLUE)

    # Result
    ax2.text(5.5, y, '->', ha='center', va='center', fontsize=12, color='gray')
    ax2.text(7.5, y, result, ha='center', va='center', fontsize=9,
             color=MLPURPLE, fontweight='bold')

    # Description
    ax2.text(5, y-0.5, desc, ha='center', va='center', fontsize=8,
             style='italic', color='gray')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
