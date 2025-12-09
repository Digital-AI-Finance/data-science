"""Chart Customization - Styling and formatting"""
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
fig.suptitle('Chart Customization Options', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
x = np.linspace(0, 10, 50)
y = np.sin(x) + np.random.normal(0, 0.1, 50)

# Plot 1: Line styles and markers
ax1 = axes[0, 0]
ax1.plot(x[:20], y[:20], 'o-', color=MLBLUE, linewidth=2, markersize=8, label='Solid + circle')
ax1.plot(x[:20], y[:20]+1, 's--', color=MLGREEN, linewidth=2, markersize=6, label='Dashed + square')
ax1.plot(x[:20], y[:20]+2, '^:', color=MLRED, linewidth=2, markersize=6, label='Dotted + triangle')
ax1.plot(x[:20], y[:20]+3, 'D-.', color=MLORANGE, linewidth=2, markersize=6, label='Dash-dot + diamond')

ax1.set_title('Line Styles and Markers', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Colors and transparency
ax2 = axes[0, 1]
for alpha, offset in [(1.0, 0), (0.7, 0.5), (0.4, 1.0), (0.2, 1.5)]:
    ax2.fill_between(x, y + offset, y + offset + 0.4, alpha=alpha, color=MLBLUE,
                     label=f'alpha = {alpha}')

ax2.set_title('Transparency (alpha)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Axis customization
ax3 = axes[1, 0]
prices = 100 + np.cumsum(np.random.randn(100) * 2)
ax3.plot(prices, color=MLBLUE, linewidth=2)

ax3.set_xlim(0, 100)
ax3.set_ylim(80, 130)
ax3.set_xticks([0, 25, 50, 75, 100])
ax3.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct', 'Dec'])
ax3.set_xlabel('Month', fontsize=10, fontweight='bold')
ax3.set_ylabel('Price ($)', fontsize=10, fontweight='bold')
ax3.set_title('Axis Limits, Ticks, Labels', fontsize=11, fontweight='bold', color=MLPURPLE)

ax3.axhline(100, color=MLRED, linestyle='--', linewidth=1.5, label='Starting price')
ax3.axvspan(60, 80, alpha=0.2, color=MLORANGE, label='Highlight region')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Text and annotations
ax4 = axes[1, 1]
ax4.plot(prices, color=MLBLUE, linewidth=2)

# Find max and min
max_idx = np.argmax(prices)
min_idx = np.argmin(prices)

ax4.scatter([max_idx], [prices[max_idx]], color=MLGREEN, s=100, zorder=5)
ax4.scatter([min_idx], [prices[min_idx]], color=MLRED, s=100, zorder=5)

ax4.annotate(f'Max: ${prices[max_idx]:.0f}', xy=(max_idx, prices[max_idx]),
             xytext=(max_idx - 20, prices[max_idx] + 8),
             fontsize=10, fontweight='bold', color=MLGREEN,
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

ax4.annotate(f'Min: ${prices[min_idx]:.0f}', xy=(min_idx, prices[min_idx]),
             xytext=(min_idx + 10, prices[min_idx] - 8),
             fontsize=10, fontweight='bold', color=MLRED,
             arrowprops=dict(arrowstyle='->', color=MLRED))

ax4.text(0.5, 0.95, 'Annotations highlight key points',
         transform=ax4.transAxes, fontsize=9, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

ax4.set_title('Annotations and Text', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
