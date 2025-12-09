"""Annotations - Adding context and insights to charts"""
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
fig.suptitle('Effective Annotations for Storytelling', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
days = 252
dates = pd.date_range('2024-01-01', periods=days, freq='B')
prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.015))

# Plot 1: Event annotations
ax1 = axes[0, 0]
ax1.plot(dates, prices, color=MLBLUE, linewidth=2)

# Add event markers
events = [
    (50, 'Fed Rate Decision', MLRED),
    (120, 'Earnings Beat', MLGREEN),
    (180, 'Market Correction', MLRED),
    (220, 'Recovery', MLGREEN)
]

for idx, label, color in events:
    ax1.axvline(dates[idx], color=color, linestyle='--', linewidth=1, alpha=0.7)
    ax1.scatter([dates[idx]], [prices[idx]], color=color, s=80, zorder=5, edgecolors='black')
    ax1.annotate(label, xy=(dates[idx], prices[idx]),
                 xytext=(10, 15 if prices[idx] > np.mean(prices) else -25),
                 textcoords='offset points',
                 fontsize=9, fontweight='bold', color=color,
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

ax1.set_title('Event Annotations', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(alpha=0.3)

# Plot 2: Statistical annotations
ax2 = axes[0, 1]
returns = np.diff(np.log(prices)) * 100

ax2.hist(returns, bins=35, color=MLBLUE, alpha=0.7, edgecolor='black')

# Add statistical markers
mean_ret = np.mean(returns)
std_ret = np.std(returns)
var_95 = np.percentile(returns, 5)

ax2.axvline(mean_ret, color=MLGREEN, linewidth=2.5, label=f'Mean: {mean_ret:.3f}%')
ax2.axvline(mean_ret + 2*std_ret, color=MLORANGE, linewidth=2, linestyle='--')
ax2.axvline(mean_ret - 2*std_ret, color=MLORANGE, linewidth=2, linestyle='--')
ax2.axvline(var_95, color=MLRED, linewidth=2.5, linestyle=':', label=f'VaR 95%: {var_95:.2f}%')

# Add stats box
stats_text = f'Statistics:\nMean: {mean_ret:.3f}%\nStd Dev: {std_ret:.2f}%\nSkew: {0.1:.2f}'
ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLPURPLE, alpha=0.9))

ax2.set_title('Statistical Annotations', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Daily Return (%)', fontsize=10)
ax2.legend(fontsize=8, loc='upper left')

# Plot 3: Highlight regions
ax3 = axes[1, 0]
ax3.plot(dates, prices, color=MLBLUE, linewidth=2)

# Add shaded regions for market conditions
ax3.axvspan(dates[40], dates[80], alpha=0.2, color=MLGREEN, label='Bull Market')
ax3.axvspan(dates[160], dates[200], alpha=0.2, color=MLRED, label='Bear Market')

# Add reference lines
ax3.axhline(prices[0], color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.text(dates[5], prices[0] + 2, 'Starting Price', fontsize=9, color='gray')

# Mark max/min
max_idx = np.argmax(prices)
min_idx = np.argmin(prices)
ax3.scatter([dates[max_idx]], [prices[max_idx]], color=MLGREEN, s=100, zorder=5, marker='^')
ax3.scatter([dates[min_idx]], [prices[min_idx]], color=MLRED, s=100, zorder=5, marker='v')

ax3.annotate(f'Peak: ${prices[max_idx]:.0f}', xy=(dates[max_idx], prices[max_idx]),
             xytext=(0, 15), textcoords='offset points', ha='center',
             fontsize=9, fontweight='bold', color=MLGREEN)
ax3.annotate(f'Trough: ${prices[min_idx]:.0f}', xy=(dates[min_idx], prices[min_idx]),
             xytext=(0, -20), textcoords='offset points', ha='center',
             fontsize=9, fontweight='bold', color=MLRED)

ax3.set_title('Region Highlights + Extremes', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Price ($)', fontsize=10)
ax3.legend(fontsize=8, loc='upper left')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(alpha=0.3)

# Plot 4: Annotation best practices
ax4 = axes[1, 1]
ax4.axis('off')

practices = '''ANNOTATION BEST PRACTICES

1. LABEL KEY POINTS
   - Peaks, troughs, inflection points
   - Events that caused changes
   - Target/threshold crossings

2. USE VISUAL HIERARCHY
   - Bold for most important
   - Arrows point to data
   - Color matches context (red=bad, green=good)

3. AVOID CLUTTER
   - Max 3-5 annotations per chart
   - Remove unnecessary labels
   - Use callout boxes sparingly

4. POSITION STRATEGICALLY
   - Don't overlap with data
   - Place near but not on point
   - Align consistently (left/right)

5. INCLUDE CONTEXT
   - "Why" not just "what"
   - Comparison to baseline
   - Time reference when needed
'''

ax4.text(0.05, 0.95, practices, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Best Practices', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
