"""Shared Axes - Aligned scales across subplots"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
fig.suptitle('Shared Axes: sharex and sharey Parameters', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate price data for multiple assets
days = 100
time = np.arange(days)
aapl = 150 + np.cumsum(np.random.randn(days) * 2)
msft = 300 + np.cumsum(np.random.randn(days) * 3)
googl = 130 + np.cumsum(np.random.randn(days) * 2.5)

# Plot 1: No shared axes (default)
ax1 = axes[0, 0]
ax1.plot(time, aapl, color=MLBLUE, linewidth=2, label='AAPL')
ax1.set_title('No Shared Axes (default)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Day', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Next to plot 1, different scale
ax2 = axes[0, 1]
ax2.plot(time, msft, color=MLGREEN, linewidth=2, label='MSFT')
ax2.set_title('Different Y Scale', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Day', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Create a separate figure to demonstrate shared axes
# Plot 3: Demonstrating sharex concept
ax3 = axes[1, 0]

# Show normalized prices
aapl_norm = aapl / aapl[0] * 100
msft_norm = msft / msft[0] * 100
googl_norm = googl / googl[0] * 100

ax3.plot(time, aapl_norm, color=MLBLUE, linewidth=2, label='AAPL')
ax3.plot(time, msft_norm, color=MLGREEN, linewidth=2, label='MSFT')
ax3.plot(time, googl_norm, color=MLORANGE, linewidth=2, label='GOOGL')
ax3.axhline(100, color='gray', linestyle='--', linewidth=1)
ax3.set_title('Normalized Prices (Same Scale)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Day', fontsize=10)
ax3.set_ylabel('Price (indexed to 100)', fontsize=10)
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(alpha=0.3)

# Plot 4: Code example annotation
ax4 = axes[1, 1]
ax4.axis('off')

code_text = '''Shared Axes Syntax:

# Share X axis between columns
fig, axes = plt.subplots(2, 2, sharex='col')

# Share Y axis between rows
fig, axes = plt.subplots(2, 2, sharey='row')

# Share both axes across all subplots
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

Benefits:
- Consistent scale for comparison
- Cleaner appearance (fewer tick labels)
- Easy to spot relative differences
'''

ax4.text(0.1, 0.9, code_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))

ax4.set_title('Shared Axes Syntax Reference', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
