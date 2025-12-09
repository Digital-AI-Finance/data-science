"""Broadcasting - NumPy broadcasting rules"""
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

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left: Broadcasting concept
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.text(5, 9.5, 'Broadcasting: Array + Scalar', ha='center',
         fontsize=14, fontweight='bold', color=MLPURPLE)

# Array (shape 5,)
arr = [100, 150, 200, 250, 300]
for i, val in enumerate(arr):
    rect = plt.Rectangle((1 + i*1.5, 7), 1.2, 0.8, facecolor=MLLAVENDER, edgecolor=MLPURPLE, linewidth=2)
    ax1.add_patch(rect)
    ax1.text(1.6 + i*1.5, 7.4, f'{val}', ha='center', va='center', fontsize=10)

ax1.text(0.3, 7.4, 'prices', va='center', fontsize=10, fontweight='bold', color=MLPURPLE)
ax1.text(9, 7.4, 'shape: (5,)', va='center', fontsize=9, color='gray')

# Scalar
ax1.text(5, 5.5, '+   10   (scalar)', ha='center', fontsize=14, fontweight='bold', color=MLORANGE)

# Arrow
ax1.annotate('', xy=(5, 4.5), xytext=(5, 5),
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))
ax1.text(6, 4.7, 'Broadcasts to match shape', fontsize=9, style='italic', color='gray')

# Result
result = [110, 160, 210, 260, 310]
for i, val in enumerate(result):
    rect = plt.Rectangle((1 + i*1.5, 3), 1.2, 0.8, facecolor='#E6FFE6', edgecolor=MLGREEN, linewidth=2)
    ax1.add_patch(rect)
    ax1.text(1.6 + i*1.5, 3.4, f'{val}', ha='center', va='center', fontsize=10)

ax1.text(0.3, 3.4, 'result', va='center', fontsize=10, fontweight='bold', color=MLGREEN)

# Finance example
ax1.text(5, 1.5, 'Finance: Add $10 fee to all prices', ha='center',
         fontsize=10, fontweight='bold', color=MLPURPLE)
ax1.text(5, 0.9, 'prices + 10  # No loop needed!', ha='center',
         fontsize=10, family='monospace', color=MLBLUE)

# Right: 2D Broadcasting
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

ax2.text(5, 9.5, 'Broadcasting: Matrix + Row', ha='center',
         fontsize=14, fontweight='bold', color=MLPURPLE)

# 3x3 matrix (stock returns)
matrix_vals = [[0.05, 0.03, 0.02], [0.04, 0.01, 0.03], [-0.02, 0.02, 0.01]]
for i in range(3):
    for j in range(3):
        rect = plt.Rectangle((1 + j*1.2, 7 - i*0.8), 1, 0.6, facecolor=MLLAVENDER, edgecolor=MLPURPLE)
        ax2.add_patch(rect)
        ax2.text(1.5 + j*1.2, 7.3 - i*0.8, f'{matrix_vals[i][j]:.0%}', ha='center', va='center', fontsize=8)

ax2.text(0.3, 6.5, 'returns\n(3 stocks,\n3 days)', va='center', fontsize=8, color=MLPURPLE)

# Row vector (weights)
ax2.text(5, 5, '+   weights (1x3):', ha='center', fontsize=11, fontweight='bold', color=MLORANGE)
weights = [0.5, 0.3, 0.2]
for j, w in enumerate(weights):
    rect = plt.Rectangle((5.5 + j*1.2, 4.2), 1, 0.6, facecolor='#FFF5E6', edgecolor=MLORANGE)
    ax2.add_patch(rect)
    ax2.text(6 + j*1.2, 4.5, f'{w}', ha='center', va='center', fontsize=9)

# Arrow
ax2.annotate('', xy=(5, 3.5), xytext=(5, 4),
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))

# Result (conceptual)
ax2.text(5, 2.5, 'Result: Each row added to weights', ha='center',
         fontsize=10, color=MLGREEN, fontweight='bold')

# Rules box
ax2.text(5, 1.2, 'Broadcasting Rules:', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)
ax2.text(5, 0.7, '1. Dimensions compared right-to-left\n2. Sizes must match OR one must be 1',
         ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
