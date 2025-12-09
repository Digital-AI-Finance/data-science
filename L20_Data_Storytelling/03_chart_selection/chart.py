"""Chart Selection - Choosing the right visualization"""
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

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Chart Selection Guide: Match Chart to Purpose', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Comparison -> Bar Chart
ax1 = axes[0, 0]
categories = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
values = [18.5, 15.2, 12.8, 8.4, 22.1]
colors = [MLGREEN if v > 15 else MLBLUE for v in values]

ax1.barh(categories, values, color=colors, edgecolor='black', linewidth=0.5)
ax1.axvline(15, color=MLRED, linestyle='--', linewidth=1.5, label='Benchmark')
ax1.set_title('COMPARISON: Bar Chart', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('YTD Return (%)', fontsize=10)
ax1.legend(fontsize=8)

# Plot 2: Trend -> Line Chart
ax2 = axes[0, 1]
days = np.arange(100)
price = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.015))

ax2.plot(days, price, color=MLBLUE, linewidth=2)
ax2.fill_between(days, price.min(), price, alpha=0.2, color=MLBLUE)
ax2.set_title('TREND: Line Chart', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Day', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: Distribution -> Histogram
ax3 = axes[0, 2]
returns = np.random.normal(0.5, 2, 500)

ax3.hist(returns, bins=30, color=MLGREEN, alpha=0.7, edgecolor='black')
ax3.axvline(np.mean(returns), color=MLRED, linewidth=2.5, linestyle='--', label=f'Mean: {np.mean(returns):.2f}%')
ax3.set_title('DISTRIBUTION: Histogram', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Return (%)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.legend(fontsize=8)

# Plot 4: Composition -> Pie/Donut Chart
ax4 = axes[1, 0]
sectors = ['Tech', 'Finance', 'Health', 'Energy']
weights = [40, 25, 20, 15]
colors_pie = [MLBLUE, MLGREEN, MLORANGE, MLRED]

wedges, texts, autotexts = ax4.pie(weights, labels=sectors, colors=colors_pie,
                                   autopct='%1.0f%%', startangle=90,
                                   wedgeprops=dict(width=0.6, edgecolor='white'),
                                   textprops={'fontsize': 9})
ax4.set_title('COMPOSITION: Donut Chart', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 5: Relationship -> Scatter Plot
ax5 = axes[1, 1]
risk = np.random.uniform(5, 30, 40)
ret = 2 + 0.35 * risk + np.random.normal(0, 2, 40)

ax5.scatter(risk, ret, c=MLBLUE, s=60, alpha=0.7, edgecolors='black')
z = np.polyfit(risk, ret, 1)
ax5.plot(np.sort(risk), np.poly1d(z)(np.sort(risk)), color=MLRED, linewidth=2, linestyle='--')
ax5.set_title('RELATIONSHIP: Scatter Plot', fontsize=11, fontweight='bold', color=MLPURPLE)
ax5.set_xlabel('Risk (%)', fontsize=10)
ax5.set_ylabel('Return (%)', fontsize=10)
ax5.grid(alpha=0.3)

# Plot 6: Selection guide
ax6 = axes[1, 2]
ax6.axis('off')

guide = '''CHART SELECTION QUICK GUIDE

PURPOSE              CHART TYPE
-----------------    ------------------
Compare values       Bar, Column
Show trends          Line, Area
Part of whole        Pie, Stacked Bar
Distribution         Histogram, Box
Relationship         Scatter, Bubble
Flow/Process         Sankey, Waterfall

TIPS:
- When in doubt, use bar chart
- Line charts need continuous x-axis
- Limit pie charts to 5-6 segments
- Use scatter for 2+ variables
'''

ax6.text(0.05, 0.95, guide, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax6.set_title('Quick Reference', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
