"""Finance Charts - Common financial visualizations"""
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
fig.suptitle('Common Finance Charts', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Price chart with volume
ax1 = axes[0, 0]
n = 100
prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015))
volume = np.random.randint(100, 500, n).astype(float)
volume[prices > np.roll(prices, 1)] *= 1.2  # Higher volume on up days

ax1.plot(prices, color=MLBLUE, linewidth=2)
ax1_vol = ax1.twinx()
colors = [MLGREEN if prices[i] > prices[i-1] else MLRED for i in range(1, n)]
colors = [MLGREEN] + colors
ax1_vol.bar(range(n), volume, color=colors, alpha=0.3, width=1)
ax1_vol.set_ylabel('Volume', fontsize=10, color='gray')
ax1_vol.set_ylim(0, volume.max() * 3)

ax1.set_title('Price + Volume', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Day', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Drawdown chart
ax2 = axes[0, 1]
prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.015))
cummax = np.maximum.accumulate(prices)
drawdown = (prices - cummax) / cummax * 100

ax2.fill_between(range(len(drawdown)), 0, drawdown, color=MLRED, alpha=0.5)
ax2.plot(drawdown, color=MLRED, linewidth=1.5)
ax2.axhline(drawdown.min(), color='black', linestyle='--', linewidth=1,
            label=f'Max DD: {drawdown.min():.1f}%')

ax2.set_title('Drawdown Chart', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Day', fontsize=10)
ax2.set_ylabel('Drawdown (%)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Rolling Sharpe
ax3 = axes[1, 0]
returns = np.random.normal(0.0004, 0.015, 504)
prices = 100 * np.exp(np.cumsum(returns))

window = 60
rolling_mean = pd.Series(returns).rolling(window).mean()
rolling_std = pd.Series(returns).rolling(window).std()
rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)

ax3.plot(rolling_sharpe, color=MLBLUE, linewidth=2)
ax3.fill_between(range(len(rolling_sharpe)), 0, rolling_sharpe,
                 where=rolling_sharpe > 0, color=MLGREEN, alpha=0.3)
ax3.fill_between(range(len(rolling_sharpe)), 0, rolling_sharpe,
                 where=rolling_sharpe < 0, color=MLRED, alpha=0.3)
ax3.axhline(0, color='black', linewidth=1)
ax3.axhline(1, color=MLGREEN, linestyle='--', linewidth=1.5, alpha=0.7, label='Good (>1)')
ax3.axhline(-1, color=MLRED, linestyle='--', linewidth=1.5, alpha=0.7, label='Poor (<-1)')

ax3.set_title('Rolling 60-day Sharpe Ratio', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Day', fontsize=10)
ax3.set_ylabel('Sharpe Ratio', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Cumulative returns comparison
ax4 = axes[1, 1]
dates = pd.date_range('2023-01-01', periods=252, freq='B')

# Multiple strategies
strategies = {
    'Portfolio': np.random.normal(0.0004, 0.01, 252),
    'Benchmark': np.random.normal(0.0003, 0.012, 252),
    'Risk-Free': np.full(252, 0.00015),
}

for name, returns in strategies.items():
    cumulative = (1 + returns).cumprod()
    color = MLBLUE if name == 'Portfolio' else (MLORANGE if name == 'Benchmark' else MLGREEN)
    style = '-' if name == 'Portfolio' else ('--' if name == 'Benchmark' else ':')
    ax4.plot(dates, cumulative, color=color, linewidth=2, linestyle=style,
             label=f'{name}: {(cumulative[-1]-1)*100:.1f}%')

ax4.axhline(1, color='gray', linestyle='-', linewidth=1)
ax4.set_title('Cumulative Returns Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Cumulative Return', fontsize=10)
ax4.tick_params(axis='x', rotation=45)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
