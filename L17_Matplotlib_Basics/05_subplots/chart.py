"""Subplots - Multiple plots in one figure"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

# Main figure with 2x2 layout showing different subplot arrangements
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Subplot Layouts in matplotlib', fontsize=14, fontweight='bold', color=MLPURPLE)

# Example 1: Basic 2x2 (top-left quadrant shown as annotation)
ax1 = fig.add_subplot(2, 2, 1)
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), color=MLBLUE, linewidth=2)
ax1.set_title('fig.add_subplot(2, 2, 1)', fontsize=10, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('x', fontsize=9)
ax1.set_ylabel('sin(x)', fontsize=9)
ax1.grid(alpha=0.3)

# Example 2: Second plot
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, np.cos(x), color=MLGREEN, linewidth=2)
ax2.set_title('fig.add_subplot(2, 2, 2)', fontsize=10, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('x', fontsize=9)
ax2.set_ylabel('cos(x)', fontsize=9)
ax2.grid(alpha=0.3)

# Example 3: Spanning columns (bottom half)
ax3 = fig.add_subplot(2, 1, 2)  # This creates a subplot that spans the bottom half
# Time series data
days = 252
prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.015))
ax3.plot(prices, color=MLBLUE, linewidth=2)
ax3.fill_between(range(len(prices)), prices.min(), prices, alpha=0.2, color=MLBLUE)
ax3.set_title('fig.add_subplot(2, 1, 2) - Spans full width', fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Trading Day', fontsize=9)
ax3.set_ylabel('Price ($)', fontsize=9)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Create a second figure showing common patterns
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
fig2.suptitle('Common Subplot Patterns: plt.subplots(rows, cols)', fontsize=14,
              fontweight='bold', color=MLPURPLE)

data = np.random.randn(100)
patterns = ['ax[0,0]', 'ax[0,1]', 'ax[0,2]', 'ax[1,0]', 'ax[1,1]', 'ax[1,2]']
colors = [MLBLUE, MLGREEN, MLRED, MLORANGE, MLPURPLE, MLBLUE]

for ax, pattern, color in zip(axes2.flat, patterns, colors):
    ax.hist(data * np.random.uniform(0.5, 2), bins=20, color=color, alpha=0.7, edgecolor='black')
    ax.set_title(pattern, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
# Save with different name to show multiple examples
