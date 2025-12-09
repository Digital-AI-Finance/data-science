"""Executive Summary - Dashboard-style summaries"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                       height_ratios=[0.4, 1, 0.8])

# Main title
fig.suptitle('Q4 2024 Investment Portfolio Summary', fontsize=16, fontweight='bold', color=MLPURPLE, y=0.98)

# Row 0: Key metrics (KPI cards style)
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.axis('off')

# KPI data
kpis = [
    ('Total AUM', '$2.4B', '+$180M', MLGREEN),
    ('Q4 Return', '+4.2%', 'vs +3.1% benchmark', MLGREEN),
    ('Sharpe Ratio', '1.42', 'Above target 1.0', MLBLUE),
    ('Max Drawdown', '-8.5%', 'Within -10% limit', MLORANGE)
]

for i, (label, value, subtitle, color) in enumerate(kpis):
    x = 0.12 + i * 0.22
    ax_kpi.text(x, 0.75, label, fontsize=10, ha='center', color='gray')
    ax_kpi.text(x, 0.4, value, fontsize=24, ha='center', fontweight='bold', color=color)
    ax_kpi.text(x, 0.1, subtitle, fontsize=9, ha='center', color='gray', style='italic')

# Row 1, Col 0-1: Performance chart
ax_perf = fig.add_subplot(gs[1, :2])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
portfolio = np.cumsum([1.2, 0.8, -0.5, 2.1, 1.5, -0.3, 1.8, 0.5, -1.2, 1.5, 0.8, 1.4])
benchmark = np.cumsum([1.0, 0.5, -0.8, 1.5, 1.2, 0.2, 1.2, 0.3, -0.5, 1.0, 0.5, 1.0])

ax_perf.plot(months, portfolio, color=MLBLUE, linewidth=2.5, marker='o', markersize=6, label='Portfolio')
ax_perf.plot(months, benchmark, color='gray', linewidth=2, linestyle='--', marker='s', markersize=5, label='Benchmark')
ax_perf.fill_between(months, portfolio, benchmark,
                      where=[p > b for p, b in zip(portfolio, benchmark)],
                      alpha=0.3, color=MLGREEN, label='Outperformance')
ax_perf.axhline(0, color='gray', linewidth=1)
ax_perf.set_title('Cumulative Performance vs Benchmark', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_perf.set_ylabel('Cumulative Return (%)', fontsize=10)
ax_perf.legend(fontsize=8, loc='upper left')
ax_perf.grid(alpha=0.3)

# Row 1, Col 2: Allocation pie
ax_alloc = fig.add_subplot(gs[1, 2])
sectors = ['Equities', 'Fixed Income', 'Alternatives', 'Cash']
weights = [55, 28, 12, 5]
colors_sec = [MLBLUE, MLGREEN, MLORANGE, MLLAVENDER]
wedges, texts, autotexts = ax_alloc.pie(weights, labels=sectors, colors=colors_sec,
                                         autopct='%1.0f%%', startangle=90,
                                         wedgeprops=dict(edgecolor='white', linewidth=2),
                                         textprops={'fontsize': 9})
ax_alloc.set_title('Asset Allocation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Row 1, Col 3: Top/Bottom performers
ax_contrib = fig.add_subplot(gs[1, 3])
positions = ['NVDA', 'MSFT', 'AAPL', 'XOM', 'CVX']
contributions = [0.85, 0.42, 0.31, -0.18, -0.25]
colors_contrib = [MLGREEN if c > 0 else MLRED for c in contributions]

bars = ax_contrib.barh(positions, contributions, color=colors_contrib, edgecolor='black', linewidth=0.5)
ax_contrib.axvline(0, color='gray', linewidth=1)

for i, (pos, val) in enumerate(zip(positions, contributions)):
    ax_contrib.text(val + 0.03 if val > 0 else val - 0.03,
                    i, f'{val:+.2f}%',
                    va='center', ha='left' if val > 0 else 'right',
                    fontsize=9, fontweight='bold',
                    color=MLGREEN if val > 0 else MLRED)

ax_contrib.set_title('Top/Bottom Contributors', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_contrib.set_xlabel('Contribution (%)', fontsize=10)
ax_contrib.set_xlim(-0.5, 1.1)

# Row 2: Key insights and action items
ax_insights = fig.add_subplot(gs[2, :2])
ax_insights.axis('off')

insights = '''KEY INSIGHTS:
+ Portfolio outperformed benchmark by 110 bps in Q4
+ Tech sector allocation (+5% OW) was primary driver
+ Fixed income rebalancing added 25 bps vs benchmark
- Energy underweight cost 15 bps as oil rallied

RECOMMENDATION: Maintain equity overweight into Q1;
consider reducing tech exposure if valuations extend.'''

ax_insights.text(0.02, 0.9, insights, transform=ax_insights.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))
ax_insights.set_title('Key Insights & Recommendations', fontsize=11, fontweight='bold', color=MLPURPLE)

# Row 2: Risk metrics
ax_risk = fig.add_subplot(gs[2, 2:])
metrics = ['Volatility', 'VaR (95%)', 'Tracking Error', 'Beta']
values = [12.5, 2.1, 3.2, 1.05]
targets = [15.0, 2.5, 4.0, 1.0]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax_risk.bar(x - width/2, values, width, label='Actual', color=MLBLUE, edgecolor='black', linewidth=0.5)
bars2 = ax_risk.bar(x + width/2, targets, width, label='Target', color=MLLAVENDER, edgecolor='black', linewidth=0.5)

ax_risk.set_xticks(x)
ax_risk.set_xticklabels(metrics, fontsize=9)
ax_risk.set_title('Risk Metrics vs Targets', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_risk.legend(fontsize=8)
ax_risk.grid(alpha=0.3, axis='y')

plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
