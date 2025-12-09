"""Finance Dashboard - Complete portfolio analysis view"""
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

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                       height_ratios=[0.3, 1.5, 1, 1])

fig.suptitle('Portfolio Analysis Dashboard', fontsize=16, fontweight='bold', color=MLPURPLE, y=0.98)

# Generate comprehensive data
days = 252
dates = pd.date_range('2024-01-01', periods=days, freq='B')
portfolio_value = 1000000 * np.exp(np.cumsum(np.random.randn(days) * 0.012))
benchmark_value = 1000000 * np.exp(np.cumsum(np.random.randn(days) * 0.010))
returns = np.diff(np.log(portfolio_value)) * 100

# Row 0: KPI Summary (spans full width)
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.axis('off')

# Calculate KPIs
total_return = (portfolio_value[-1] / portfolio_value[0] - 1) * 100
excess_return = total_return - (benchmark_value[-1] / benchmark_value[0] - 1) * 100
volatility = np.std(returns) * np.sqrt(252)
sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
max_dd = np.min((portfolio_value - np.maximum.accumulate(portfolio_value)) / np.maximum.accumulate(portfolio_value)) * 100

kpis = [
    ('Total Return', f'{total_return:+.1f}%', MLGREEN if total_return > 0 else MLRED),
    ('Excess Return', f'{excess_return:+.1f}%', MLGREEN if excess_return > 0 else MLRED),
    ('Volatility', f'{volatility:.1f}%', MLORANGE),
    ('Sharpe Ratio', f'{sharpe:.2f}', MLBLUE),
    ('Max Drawdown', f'{max_dd:.1f}%', MLRED)
]

for i, (name, value, color) in enumerate(kpis):
    x = 0.1 + i * 0.18
    ax_kpi.text(x, 0.7, name, fontsize=10, ha='center', color='gray')
    ax_kpi.text(x, 0.2, value, fontsize=18, fontweight='bold', ha='center', color=color)

# Row 1: Main Performance Chart (spans 3 columns)
ax_perf = fig.add_subplot(gs[1, :3])
ax_perf.plot(dates, portfolio_value / 1e6, color=MLBLUE, linewidth=2.5, label='Portfolio')
ax_perf.plot(dates, benchmark_value / 1e6, color='gray', linewidth=1.5, linestyle='--', label='Benchmark')
ax_perf.fill_between(dates, portfolio_value / 1e6, benchmark_value / 1e6,
                      where=portfolio_value > benchmark_value, alpha=0.3, color=MLGREEN, label='Outperformance')
ax_perf.fill_between(dates, portfolio_value / 1e6, benchmark_value / 1e6,
                      where=portfolio_value < benchmark_value, alpha=0.3, color=MLRED, label='Underperformance')
ax_perf.set_title('Portfolio vs Benchmark Performance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_perf.set_ylabel('Value ($M)', fontsize=10)
ax_perf.legend(loc='upper left', fontsize=8)
ax_perf.grid(alpha=0.3)
ax_perf.tick_params(axis='x', rotation=45)

# Row 1, Col 3: Allocation Pie
ax_alloc = fig.add_subplot(gs[1, 3])
sectors = ['Equities', 'Fixed Income', 'Alternatives', 'Cash']
weights = [55, 25, 15, 5]
colors_sec = [MLBLUE, MLGREEN, MLORANGE, MLLAVENDER]
wedges, texts, autotexts = ax_alloc.pie(weights, labels=sectors, colors=colors_sec,
                                         autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9})
ax_alloc.set_title('Asset Allocation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Row 2: Drawdown Chart (spans 2 columns)
ax_dd = fig.add_subplot(gs[2, :2])
cummax = np.maximum.accumulate(portfolio_value)
drawdown = (portfolio_value - cummax) / cummax * 100
ax_dd.fill_between(dates, 0, drawdown, color=MLRED, alpha=0.5)
ax_dd.plot(dates, drawdown, color=MLRED, linewidth=1)
ax_dd.axhline(max_dd, color='black', linestyle='--', linewidth=1.5, label=f'Max DD: {max_dd:.1f}%')
ax_dd.set_title('Drawdown Analysis', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_dd.set_ylabel('Drawdown (%)', fontsize=10)
ax_dd.legend(fontsize=8)
ax_dd.grid(alpha=0.3)
ax_dd.tick_params(axis='x', rotation=45)

# Row 2, Cols 2-3: Monthly Returns Heatmap
ax_monthly = fig.add_subplot(gs[2, 2:])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_returns = np.random.normal(0.8, 2.5, 12)
colors_monthly = [MLGREEN if r > 0 else MLRED for r in monthly_returns]
bars = ax_monthly.bar(months, monthly_returns, color=colors_monthly, edgecolor='black', linewidth=0.5)
ax_monthly.axhline(0, color='gray', linewidth=1)
ax_monthly.set_title('Monthly Returns YTD', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_monthly.set_ylabel('Return (%)', fontsize=10)
ax_monthly.tick_params(axis='x', labelsize=8, rotation=45)
ax_monthly.grid(alpha=0.3, axis='y')

# Row 3: Return Distribution
ax_dist = fig.add_subplot(gs[3, :2])
ax_dist.hist(returns, bins=40, color=MLBLUE, alpha=0.7, edgecolor='white', density=True)
var_95 = np.percentile(returns, 5)
cvar_95 = np.mean(returns[returns <= var_95])
ax_dist.axvline(np.mean(returns), color=MLGREEN, linewidth=2.5, linestyle='-', label=f'Mean: {np.mean(returns):.2f}%')
ax_dist.axvline(var_95, color=MLRED, linewidth=2.5, linestyle='--', label=f'VaR 95%: {var_95:.2f}%')
ax_dist.axvline(cvar_95, color=MLRED, linewidth=2.5, linestyle=':', label=f'CVaR 95%: {cvar_95:.2f}%')
ax_dist.set_title('Return Distribution', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_dist.set_xlabel('Daily Return (%)', fontsize=10)
ax_dist.legend(fontsize=8)
ax_dist.grid(alpha=0.3)

# Row 3: Rolling Metrics
ax_roll = fig.add_subplot(gs[3, 2:])
window = 60
rolling_mean = pd.Series(returns).rolling(window).mean() * 252
rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
rolling_sharpe = rolling_mean / rolling_vol

ax_roll.plot(dates[window:], rolling_sharpe[window:], color=MLBLUE, linewidth=2)
ax_roll.fill_between(dates[window:], 0, rolling_sharpe[window:],
                      where=rolling_sharpe[window:] > 0, alpha=0.3, color=MLGREEN)
ax_roll.fill_between(dates[window:], 0, rolling_sharpe[window:],
                      where=rolling_sharpe[window:] < 0, alpha=0.3, color=MLRED)
ax_roll.axhline(1, color=MLGREEN, linestyle='--', linewidth=1.5, alpha=0.7, label='Target Sharpe = 1')
ax_roll.axhline(0, color='gray', linewidth=1)
ax_roll.set_title('Rolling 60-Day Sharpe Ratio', fontsize=11, fontweight='bold', color=MLPURPLE)
ax_roll.set_xlabel('Date', fontsize=10)
ax_roll.legend(fontsize=8)
ax_roll.grid(alpha=0.3)
ax_roll.tick_params(axis='x', rotation=45)

plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
