"""Dashboard Layout - Professional multi-panel designs"""
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
fig.suptitle('Dashboard Layout Design Patterns', fontsize=14, fontweight='bold', color=MLPURPLE, y=0.98)

# Create outer gridspec for two dashboard examples
outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15)

# === LEFT: Executive Dashboard ===
left_gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[0], hspace=0.35, wspace=0.25)

# KPI Cards row (simulated with bar)
ax_kpi = fig.add_subplot(left_gs[0, :])
kpis = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
values = [18.5, 1.42, -12.3, 58]
units = ['%', '', '%', '%']
colors_kpi = [MLGREEN, MLBLUE, MLRED, MLORANGE]

for i, (kpi, val, unit, color) in enumerate(zip(kpis, values, units, colors_kpi)):
    ax_kpi.bar(i, 1, color=color, alpha=0.8, width=0.8)
    ax_kpi.text(i, 0.5, f'{val:+.1f}{unit}' if kpi != 'Sharpe Ratio' else f'{val:.2f}',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax_kpi.text(i, 1.1, kpi, ha='center', va='bottom', fontsize=9)

ax_kpi.set_xlim(-0.5, 3.5)
ax_kpi.set_ylim(0, 1.5)
ax_kpi.axis('off')
ax_kpi.set_title('KPI Summary Cards', fontsize=10, fontweight='bold', color=MLPURPLE)

# Main chart
ax_main = fig.add_subplot(left_gs[1, :])
days = 252
prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.015))
ax_main.plot(prices, color=MLBLUE, linewidth=2)
ax_main.fill_between(range(len(prices)), prices.min(), prices, alpha=0.2, color=MLBLUE)
ax_main.axhline(100, color='gray', linestyle='--', linewidth=1)
ax_main.set_title('Portfolio Value', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_main.set_ylabel('Value ($)', fontsize=9)
ax_main.grid(alpha=0.3)

# Supporting charts
ax_alloc = fig.add_subplot(left_gs[2, 0])
sectors = ['Tech', 'Finance', 'Health', 'Energy']
weights = [0.35, 0.25, 0.25, 0.15]
ax_alloc.pie(weights, labels=sectors, colors=[MLBLUE, MLGREEN, MLORANGE, MLRED],
             autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
ax_alloc.set_title('Allocation', fontsize=10, fontweight='bold', color=MLPURPLE)

ax_perf = fig.add_subplot(left_gs[2, 1])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
monthly_ret = [2.1, -1.5, 3.2, 1.8, -0.5, 2.3]
colors_ret = [MLGREEN if r > 0 else MLRED for r in monthly_ret]
ax_perf.bar(months, monthly_ret, color=colors_ret, edgecolor='black', linewidth=0.5)
ax_perf.axhline(0, color='gray', linewidth=1)
ax_perf.set_title('Monthly Returns', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_perf.set_ylabel('Return (%)', fontsize=9)
ax_perf.tick_params(axis='x', labelsize=8)

# === RIGHT: Analytics Dashboard ===
right_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], hspace=0.3, wspace=0.25)

# Correlation heatmap
ax_corr = fig.add_subplot(right_gs[0, 0])
assets = ['SPY', 'AGG', 'GLD', 'VIX']
corr = np.array([[1.0, -0.2, 0.1, -0.7],
                 [-0.2, 1.0, 0.3, 0.1],
                 [0.1, 0.3, 1.0, 0.0],
                 [-0.7, 0.1, 0.0, 1.0]])
im = ax_corr.imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax_corr.set_xticks(range(4))
ax_corr.set_yticks(range(4))
ax_corr.set_xticklabels(assets, fontsize=8)
ax_corr.set_yticklabels(assets, fontsize=8)
for i in range(4):
    for j in range(4):
        ax_corr.text(j, i, f'{corr[i,j]:.1f}', ha='center', va='center', fontsize=8)
ax_corr.set_title('Correlation', fontsize=10, fontweight='bold', color=MLPURPLE)

# Risk contribution
ax_risk = fig.add_subplot(right_gs[0, 1])
risk_contrib = [45, 25, 20, 10]
ax_risk.barh(assets, risk_contrib, color=[MLBLUE, MLGREEN, MLORANGE, MLRED])
ax_risk.set_xlabel('Risk Contribution (%)', fontsize=9)
ax_risk.set_title('Risk Attribution', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_risk.invert_yaxis()

# Distribution
ax_dist = fig.add_subplot(right_gs[1, 0])
returns = np.random.normal(0.5, 2, 500)
ax_dist.hist(returns, bins=30, color=MLBLUE, alpha=0.7, edgecolor='black', density=True)
ax_dist.axvline(np.percentile(returns, 5), color=MLRED, linestyle='--', linewidth=2, label='VaR 95%')
ax_dist.set_title('Return Distribution', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_dist.set_xlabel('Return (%)', fontsize=9)
ax_dist.legend(fontsize=8)

# Rolling metrics
ax_roll = fig.add_subplot(right_gs[1, 1])
roll_sharpe = np.random.normal(1.0, 0.5, 60)
roll_sharpe = np.cumsum(roll_sharpe) / np.arange(1, 61)
ax_roll.plot(roll_sharpe, color=MLBLUE, linewidth=2)
ax_roll.axhline(1, color=MLGREEN, linestyle='--', linewidth=1.5, label='Target')
ax_roll.fill_between(range(60), 0, roll_sharpe, where=roll_sharpe > 1, alpha=0.3, color=MLGREEN)
ax_roll.fill_between(range(60), 0, roll_sharpe, where=roll_sharpe <= 1, alpha=0.3, color=MLRED)
ax_roll.set_title('Rolling Sharpe', fontsize=10, fontweight='bold', color=MLPURPLE)
ax_roll.set_xlabel('Days', fontsize=9)
ax_roll.legend(fontsize=8)

plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
