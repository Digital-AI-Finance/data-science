"""Finance-Specific Statistics - Risk and Return metrics"""
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
fig.suptitle('Finance-Specific Statistics', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate realistic stock data
n = 252
returns_spy = np.random.normal(0.04, 1.2, n)  # S&P 500
returns_stock = np.random.normal(0.08, 2.0, n)  # Individual stock
rf = 0.01  # Daily risk-free rate (annualized ~2.5%)

# Plot 1: Sharpe Ratio visualization
ax1 = axes[0, 0]
mean_ret = np.mean(returns_stock)
std_ret = np.std(returns_stock)
sharpe = (mean_ret - rf) / std_ret * np.sqrt(252)  # Annualized

ax1.hist(returns_stock, bins=40, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')
ax1.axvline(mean_ret, color=MLGREEN, linewidth=2.5, label=f'Mean: {mean_ret:.2f}%')
ax1.axvline(rf, color=MLORANGE, linewidth=2, linestyle='--', label=f'Risk-free: {rf:.2f}%')

ax1.set_title(f'Sharpe Ratio: {sharpe:.2f}', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Daily Return (%)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Add formula
ax1.text(0.95, 0.95, 'Sharpe = (R - Rf) / Std', transform=ax1.transAxes,
         fontsize=9, ha='right', va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.7))

# Plot 2: VaR and CVaR
ax2 = axes[0, 1]
sorted_returns = np.sort(returns_stock)
var_95 = np.percentile(returns_stock, 5)
cvar_95 = np.mean(sorted_returns[sorted_returns <= var_95])

ax2.hist(returns_stock, bins=40, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')
ax2.axvline(var_95, color=MLRED, linewidth=2.5, label=f'VaR 95%: {var_95:.2f}%')
ax2.axvline(cvar_95, color=MLORANGE, linewidth=2.5, linestyle='--', label=f'CVaR 95%: {cvar_95:.2f}%')

# Shade the tail
ax2.fill_betweenx([0, 0.4], min(returns_stock), var_95, alpha=0.3, color=MLRED)

ax2.set_title('Value at Risk (VaR) and CVaR', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Daily Return (%)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.annotate('5% worst days', xy=(var_95 - 2, 0.15), fontsize=9, color=MLRED)

# Plot 3: Maximum Drawdown
ax3 = axes[1, 0]
prices = 100 * np.exp(np.cumsum(returns_stock / 100))
cummax = np.maximum.accumulate(prices)
drawdown = (prices - cummax) / cummax * 100

ax3.plot(prices, color=MLBLUE, linewidth=2, label='Price')
ax3.plot(cummax, color=MLGREEN, linewidth=1.5, linestyle='--', label='Running Max')

ax3_dd = ax3.twinx()
ax3_dd.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.3, color=MLRED)
ax3_dd.set_ylabel('Drawdown (%)', fontsize=10, color=MLRED)
ax3_dd.tick_params(axis='y', colors=MLRED)

max_dd = np.min(drawdown)
max_dd_idx = np.argmin(drawdown)
ax3.scatter([max_dd_idx], [prices[max_dd_idx]], color=MLRED, s=100, zorder=5)

ax3.set_title(f'Maximum Drawdown: {max_dd:.1f}%', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Trading Day', fontsize=10)
ax3.set_ylabel('Price ($)', fontsize=10)
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(alpha=0.3)

# Plot 4: Key metrics summary
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate all metrics
metrics = {
    'Annualized Return': f'{np.mean(returns_stock) * 252:.1f}%',
    'Annualized Volatility': f'{np.std(returns_stock) * np.sqrt(252):.1f}%',
    'Sharpe Ratio': f'{sharpe:.2f}',
    'VaR (95%)': f'{var_95:.2f}%',
    'CVaR (95%)': f'{cvar_95:.2f}%',
    'Max Drawdown': f'{max_dd:.1f}%',
    'Skewness': f'{pd.Series(returns_stock).skew():.2f}',
    'Kurtosis': f'{pd.Series(returns_stock).kurtosis() + 3:.2f}',
}

ax4.text(0.5, 0.95, 'Portfolio Statistics Summary', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

y = 0.82
for i, (metric, value) in enumerate(metrics.items()):
    col = i % 2
    row = i // 2
    x = 0.15 + col * 0.4
    y_pos = 0.82 - row * 0.18

    ax4.text(x, y_pos, metric + ':', fontsize=11, color='gray', transform=ax4.transAxes)
    ax4.text(x + 0.22, y_pos, value, fontsize=11, fontweight='bold', color=MLPURPLE,
             transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
