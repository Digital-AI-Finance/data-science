"""Merge Workflow - Step-by-step merge process"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
ax.text(8, 9.5, 'Merge Workflow: Stock Prices + Company Info', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE)

# DataFrame 1: Stock Prices
df1_box = FancyBboxPatch((0.5, 5.5), 4.5, 3.5, boxstyle="round,pad=0.1",
                         edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax.add_patch(df1_box)
ax.text(2.75, 8.7, 'df_prices', ha='center', fontsize=12, fontweight='bold', color=MLBLUE)
ax.text(2.75, 8.2, '(Stock price data)', ha='center', fontsize=9, style='italic', color='gray')

# Headers
headers1 = ['Ticker', 'Date', 'Price']
for i, h in enumerate(headers1):
    ax.text(1 + i*1.4, 7.6, h, ha='center', fontsize=9, fontweight='bold', color=MLBLUE)

# Data
data1 = [
    ('AAPL', '2024-01', '$150'),
    ('MSFT', '2024-01', '$350'),
    ('GOOGL', '2024-01', '$140'),
]
for i, (t, d, p) in enumerate(data1):
    y = 7.1 - i*0.5
    ax.text(1, y, t, ha='center', fontsize=8)
    ax.text(2.4, y, d, ha='center', fontsize=8)
    ax.text(3.8, y, p, ha='center', fontsize=8)

# DataFrame 2: Company Info
df2_box = FancyBboxPatch((0.5, 1), 4.5, 3.5, boxstyle="round,pad=0.1",
                         edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=2)
ax.add_patch(df2_box)
ax.text(2.75, 4.2, 'df_info', ha='center', fontsize=12, fontweight='bold', color=MLORANGE)
ax.text(2.75, 3.7, '(Company metadata)', ha='center', fontsize=9, style='italic', color='gray')

# Headers
headers2 = ['Ticker', 'Sector', 'MarketCap']
for i, h in enumerate(headers2):
    ax.text(1 + i*1.4, 3.1, h, ha='center', fontsize=9, fontweight='bold', color=MLORANGE)

# Data
data2 = [
    ('AAPL', 'Tech', '$2.8T'),
    ('MSFT', 'Tech', '$2.5T'),
    ('JPM', 'Finance', '$450B'),
]
for i, (t, s, m) in enumerate(data2):
    y = 2.6 - i*0.5
    ax.text(1, y, t, ha='center', fontsize=8)
    ax.text(2.4, y, s, ha='center', fontsize=8)
    ax.text(3.8, y, m, ha='center', fontsize=8)

# Merge operation box
merge_box = FancyBboxPatch((5.5, 4), 3.5, 2, boxstyle="round,pad=0.15",
                           edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.5, linewidth=2)
ax.add_patch(merge_box)
ax.text(7.25, 5.5, 'pd.merge()', ha='center', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.text(7.25, 4.9, "on='Ticker'", ha='center', fontsize=10, family='monospace')
ax.text(7.25, 4.4, "how='inner'", ha='center', fontsize=10, family='monospace')

# Arrows
ax.annotate('', xy=(5.5, 5), xytext=(5, 7),
            arrowprops=dict(arrowstyle='->', color=MLBLUE, lw=2))
ax.annotate('', xy=(5.5, 5), xytext=(5, 3),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
ax.annotate('', xy=(10, 5), xytext=(9, 5),
            arrowprops=dict(arrowstyle='->', color=MLGREEN, lw=3))

# Result DataFrame
result_box = FancyBboxPatch((10.5, 3), 5, 4, boxstyle="round,pad=0.1",
                            edgecolor=MLGREEN, facecolor='#E6FFE6', linewidth=2)
ax.add_patch(result_box)
ax.text(13, 6.7, 'Result (Inner Join)', ha='center', fontsize=12, fontweight='bold', color=MLGREEN)

# Headers
headers_r = ['Ticker', 'Date', 'Price', 'Sector', 'MCap']
for i, h in enumerate(headers_r):
    ax.text(10.9 + i*0.95, 6.1, h, ha='center', fontsize=8, fontweight='bold', color=MLGREEN)

# Result data (only matching tickers)
result_data = [
    ('AAPL', '2024-01', '$150', 'Tech', '$2.8T'),
    ('MSFT', '2024-01', '$350', 'Tech', '$2.5T'),
]
for i, row in enumerate(result_data):
    y = 5.5 - i*0.6
    for j, val in enumerate(row):
        ax.text(10.9 + j*0.95, y, val, ha='center', fontsize=8)

# Notes
ax.text(13, 3.7, 'GOOGL dropped (no match in df_info)', ha='center',
        fontsize=9, style='italic', color=MLRED)
ax.text(13, 3.2, 'JPM dropped (no match in df_prices)', ha='center',
        fontsize=9, style='italic', color=MLRED)

# Key highlight
key_box = FancyBboxPatch((0.5, 0.2), 15, 0.6, boxstyle="round,pad=0.05",
                         edgecolor=MLPURPLE, facecolor='white', linewidth=1)
ax.add_patch(key_box)
ax.text(8, 0.5, "Key: 'Ticker' column is the common identifier used to match rows between DataFrames",
        ha='center', fontsize=10, color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
