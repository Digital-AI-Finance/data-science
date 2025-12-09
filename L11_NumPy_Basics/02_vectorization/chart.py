"""Vectorization - Speed of vectorized operations"""
import matplotlib.pyplot as plt
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

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left: Concept diagram
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.text(5, 9.5, 'Loop vs Vectorization', ha='center',
         fontsize=14, fontweight='bold', color=MLPURPLE)

# Loop approach (slow)
ax1.text(2.5, 8.5, 'Loop Approach', ha='center', fontsize=11, fontweight='bold', color=MLRED)

# Loop boxes
for i in range(5):
    y = 7.5 - i * 1.2
    # Element box
    rect = plt.Rectangle((1, y-0.3), 0.8, 0.6, facecolor=MLLAVENDER, edgecolor=MLPURPLE)
    ax1.add_patch(rect)
    ax1.text(1.4, y, f'x[{i}]', ha='center', va='center', fontsize=8)
    # Operation
    ax1.text(2.2, y, f'* 2 =', ha='center', va='center', fontsize=8)
    # Result
    rect2 = plt.Rectangle((2.8, y-0.3), 0.8, 0.6, facecolor='#FFE6E6', edgecolor=MLRED)
    ax1.add_patch(rect2)
    ax1.text(3.2, y, f'y[{i}]', ha='center', va='center', fontsize=8)

ax1.text(2.5, 1.5, 'Sequential\n(one at a time)', ha='center', fontsize=9, color=MLRED)

# Vectorized approach (fast)
ax1.text(7.5, 8.5, 'Vectorized', ha='center', fontsize=11, fontweight='bold', color=MLGREEN)

# Single array operation
rect_x = plt.Rectangle((5.5, 4), 1.5, 3, facecolor=MLLAVENDER, edgecolor=MLPURPLE, linewidth=2)
ax1.add_patch(rect_x)
ax1.text(6.25, 5.5, 'x\n[0,1,2,3,4]', ha='center', va='center', fontsize=9)

ax1.text(7.3, 5.5, '* 2 =', ha='center', va='center', fontsize=12, fontweight='bold')

rect_y = plt.Rectangle((8, 4), 1.5, 3, facecolor='#E6FFE6', edgecolor=MLGREEN, linewidth=2)
ax1.add_patch(rect_y)
ax1.text(8.75, 5.5, 'y\n[0,2,4,6,8]', ha='center', va='center', fontsize=9)

ax1.text(7.5, 1.5, 'Parallel\n(all at once)', ha='center', fontsize=9, color=MLGREEN)

# Right: Performance demo with actual data
ax2 = axes[1]

# Generate data
np.random.seed(42)
n = 10000
stock_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

# Calculate returns using vectorization
returns = (stock_prices[1:] - stock_prices[:-1]) / stock_prices[:-1] * 100

ax2.plot(returns, color=MLPURPLE, alpha=0.7, linewidth=0.5)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.fill_between(range(len(returns)), 0, returns,
                 where=returns > 0, color=MLGREEN, alpha=0.3, label='Gains')
ax2.fill_between(range(len(returns)), 0, returns,
                 where=returns < 0, color=MLRED, alpha=0.3, label='Losses')

ax2.set_xlabel('Trading Day', fontsize=11)
ax2.set_ylabel('Daily Return (%)', fontsize=11)
ax2.set_title('10,000 Returns Calculated Instantly', fontsize=12, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Add code annotation
ax2.text(0.02, 0.98, "returns = np.diff(prices) / prices[:-1] * 100",
         transform=ax2.transAxes, fontsize=9, family='monospace', va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
