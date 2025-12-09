"""Concat - Combining DataFrames vertically and horizontally"""
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

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left panel: Vertical concat (axis=0)
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.text(5, 9.5, 'pd.concat([df1, df2], axis=0)', ha='center',
         fontsize=13, fontweight='bold', color=MLBLUE, family='monospace')
ax1.text(5, 8.9, 'Vertical Stacking (Row-wise)', ha='center',
         fontsize=11, style='italic', color='gray')

# DataFrame 1
df1_box = FancyBboxPatch((1, 6), 3.5, 2, boxstyle="round,pad=0.1",
                         edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax1.add_patch(df1_box)
ax1.text(2.75, 7.7, 'df1 (Q1 Data)', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
df1_data = [('AAPL', 150), ('MSFT', 350)]
for i, (s, p) in enumerate(df1_data):
    ax1.text(1.5, 7.0 - i*0.5, f'{s}', fontsize=9)
    ax1.text(3.5, 7.0 - i*0.5, f'${p}', fontsize=9)

# DataFrame 2
df2_box = FancyBboxPatch((5.5, 6), 3.5, 2, boxstyle="round,pad=0.1",
                         edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=2)
ax1.add_patch(df2_box)
ax1.text(7.25, 7.7, 'df2 (Q2 Data)', ha='center', fontsize=10, fontweight='bold', color=MLORANGE)
df2_data = [('GOOGL', 140), ('AMZN', 180)]
for i, (s, p) in enumerate(df2_data):
    ax1.text(6, 7.0 - i*0.5, f'{s}', fontsize=9)
    ax1.text(8, 7.0 - i*0.5, f'${p}', fontsize=9)

# Arrows
ax1.annotate('', xy=(5, 4.5), xytext=(2.75, 5.8),
            arrowprops=dict(arrowstyle='->', color=MLBLUE, lw=2))
ax1.annotate('', xy=(5, 4.5), xytext=(7.25, 5.8),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))

# Result
result_box = FancyBboxPatch((2.5, 1), 5, 3, boxstyle="round,pad=0.1",
                            edgecolor=MLGREEN, facecolor='#E6FFE6', linewidth=2)
ax1.add_patch(result_box)
ax1.text(5, 3.7, 'Result (4 rows)', ha='center', fontsize=10, fontweight='bold', color=MLGREEN)
result_data = [('AAPL', 150), ('MSFT', 350), ('GOOGL', 140), ('AMZN', 180)]
for i, (s, p) in enumerate(result_data):
    y = 3.1 - i*0.5
    color = MLBLUE if i < 2 else MLORANGE
    ax1.text(3.2, y, f'{s}', fontsize=9, color=color)
    ax1.text(6, y, f'${p}', fontsize=9, color=color)

# Right panel: Horizontal concat (axis=1)
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

ax2.text(5, 9.5, 'pd.concat([df1, df2], axis=1)', ha='center',
         fontsize=13, fontweight='bold', color=MLPURPLE, family='monospace')
ax2.text(5, 8.9, 'Horizontal Stacking (Column-wise)', ha='center',
         fontsize=11, style='italic', color='gray')

# DataFrame 1 (prices)
df1h_box = FancyBboxPatch((0.5, 6), 2.5, 2, boxstyle="round,pad=0.1",
                          edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax2.add_patch(df1h_box)
ax2.text(1.75, 7.7, 'Prices', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
ax2.text(1, 7.2, 'Stock', fontsize=8, fontweight='bold', color='gray')
ax2.text(2.3, 7.2, 'Price', fontsize=8, fontweight='bold', color='gray')
for i, (s, p) in enumerate([('AAPL', 150), ('MSFT', 350)]):
    ax2.text(1, 6.7 - i*0.4, s, fontsize=8)
    ax2.text(2.3, 6.7 - i*0.4, f'${p}', fontsize=8)

# DataFrame 2 (volumes)
df2h_box = FancyBboxPatch((3.5, 6), 2.5, 2, boxstyle="round,pad=0.1",
                          edgecolor=MLORANGE, facecolor='#FFF5E6', linewidth=2)
ax2.add_patch(df2h_box)
ax2.text(4.75, 7.7, 'Volumes', ha='center', fontsize=10, fontweight='bold', color=MLORANGE)
ax2.text(4, 7.2, 'Stock', fontsize=8, fontweight='bold', color='gray')
ax2.text(5.3, 7.2, 'Volume', fontsize=8, fontweight='bold', color='gray')
for i, (s, v) in enumerate([('AAPL', '10M'), ('MSFT', '15M')]):
    ax2.text(4, 6.7 - i*0.4, s, fontsize=8)
    ax2.text(5.3, 6.7 - i*0.4, v, fontsize=8)

# Arrow
ax2.annotate('', xy=(5, 4.5), xytext=(1.75, 5.8),
            arrowprops=dict(arrowstyle='->', color=MLBLUE, lw=2))
ax2.annotate('', xy=(5, 4.5), xytext=(4.75, 5.8),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))

# Result
resulth_box = FancyBboxPatch((1.5, 1.5), 7, 2.5, boxstyle="round,pad=0.1",
                             edgecolor=MLGREEN, facecolor='#E6FFE6', linewidth=2)
ax2.add_patch(resulth_box)
ax2.text(5, 3.7, 'Result (4 columns)', ha='center', fontsize=10, fontweight='bold', color=MLGREEN)
headers = ['Stock', 'Price', 'Stock', 'Volume']
colors_h = [MLBLUE, MLBLUE, MLORANGE, MLORANGE]
x_pos_h = [2, 3.5, 5.5, 7]
for x, h, c in zip(x_pos_h, headers, colors_h):
    ax2.text(x, 3.2, h, fontsize=8, fontweight='bold', color=c)
for i in range(2):
    ax2.text(2, 2.7 - i*0.5, ['AAPL', 'MSFT'][i], fontsize=8)
    ax2.text(3.5, 2.7 - i*0.5, ['$150', '$350'][i], fontsize=8)
    ax2.text(5.5, 2.7 - i*0.5, ['AAPL', 'MSFT'][i], fontsize=8)
    ax2.text(7, 2.7 - i*0.5, ['10M', '15M'][i], fontsize=8)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
