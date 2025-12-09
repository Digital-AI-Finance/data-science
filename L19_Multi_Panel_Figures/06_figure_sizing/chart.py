"""Figure Sizing - DPI, dimensions, and aspect ratios"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Figure Sizing and Aspect Ratios', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate sample data
days = 100
prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.015))

# Plot 1: Standard aspect ratio
ax1 = axes[0, 0]
ax1.plot(prices, color=MLBLUE, linewidth=2)
ax1.set_title('Default Aspect Ratio', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Day', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.grid(alpha=0.3)
ax1.text(0.95, 0.05, 'aspect="auto"\n(default)', transform=ax1.transAxes,
         ha='right', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 2: Equal aspect ratio
ax2 = axes[0, 1]
risk = np.random.uniform(5, 25, 50)
ret = np.random.uniform(2, 15, 50)
ax2.scatter(risk, ret, c=MLGREEN, s=60, alpha=0.7, edgecolors='black')
ax2.set_aspect('equal', adjustable='box')
ax2.set_title('Equal Aspect (Square)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Risk (%)', fontsize=10)
ax2.set_ylabel('Return (%)', fontsize=10)
ax2.grid(alpha=0.3)
ax2.text(0.95, 0.05, 'aspect="equal"', transform=ax2.transAxes,
         ha='right', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

# Plot 3: Sizing guide
ax3 = axes[1, 0]
ax3.axis('off')

sizing_text = '''Figure Sizing Guide:

figsize=(width, height)  # in inches
dpi=150                   # dots per inch

Common Sizes:
- Presentation: (12, 8) or (16, 9)
- Report: (8, 6) or (10, 6)
- Journal: (6, 4) or (3.5, 2.5)

Output Resolution:
- Screen: 72-150 dpi
- Print: 300 dpi
- High-quality: 600 dpi

Example:
fig, ax = plt.subplots(figsize=(10, 6))
plt.savefig('chart.pdf', dpi=300,
            bbox_inches='tight')
'''

ax3.text(0.05, 0.95, sizing_text, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Sizing Parameters', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Tight layout demonstration
ax4 = axes[1, 1]

# Show different padding examples
returns = np.random.normal(0.5, 2, 100)
ax4.hist(returns, bins=25, color=MLORANGE, alpha=0.7, edgecolor='black')
ax4.set_title('Layout Control', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Return (%)', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.grid(alpha=0.3)

# Add annotation about tight_layout
layout_text = '''plt.tight_layout()
- Auto-adjusts padding
- Prevents label overlap

plt.subplots_adjust(
    left=0.1, right=0.9,
    top=0.9, bottom=0.1,
    wspace=0.3, hspace=0.3
)'''
ax4.text(0.95, 0.95, layout_text, transform=ax4.transAxes,
         ha='right', va='top', fontsize=8, fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLPURPLE, alpha=0.9))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
