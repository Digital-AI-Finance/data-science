"""NumPy Finance - Complete financial analysis example"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

# Complete financial analysis using NumPy
np.random.seed(42)

# Simulate stock prices
n_days = 504  # 2 years
drift = 0.0004  # Daily expected return
volatility = 0.018
prices = 100 * np.exp(np.cumsum(drift + volatility * np.random.randn(n_days)))

# Calculate returns
returns = np.diff(prices) / prices[:-1]
log_returns = np.log(prices[1:] / prices[:-1])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NumPy Financial Analysis Dashboard', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Price with Bollinger Bands
ax1 = axes[0, 0]
window = 20
rolling_mean = np.convolve(prices, np.ones(window)/window, mode='valid')
rolling_std = np.array([np.std(prices[i:i+window]) for i in range(len(prices)-window+1)])
upper_band = rolling_mean + 2 * rolling_std
lower_band = rolling_mean - 2 * rolling_std

ax1.plot(prices, color='gray', alpha=0.5, linewidth=1, label='Price')
ax1.plot(range(window-1, len(prices)), rolling_mean, color=MLBLUE, linewidth=2, label='20-day MA')
ax1.fill_between(range(window-1, len(prices)), lower_band, upper_band,
                 alpha=0.2, color=MLBLUE, label='Bollinger Bands')

ax1.set_xlabel('Trading Day', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title('Bollinger Bands: np.convolve + np.std', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Return distribution with statistics
ax2 = axes[0, 1]
ax2.hist(returns * 100, bins=50, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')

# Normal distribution overlay
x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
mean_ret = np.mean(returns) * 100
std_ret = np.std(returns) * 100
normal = (1/(std_ret * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mean_ret)/std_ret)**2)
ax2.plot(x, normal, color=MLRED, linewidth=2, label='Normal fit')

ax2.axvline(mean_ret, color=MLGREEN, linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.2f}%')
ax2.axvline(np.percentile(returns * 100, 5), color=MLRED, linestyle='--', linewidth=2,
            label=f'VaR 95%: {np.percentile(returns * 100, 5):.2f}%')

ax2.set_xlabel('Daily Return (%)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('Return Distribution', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Cumulative returns
ax3 = axes[1, 0]
cumulative = np.cumprod(1 + returns) - 1
drawdown = cumulative - np.maximum.accumulate(cumulative)
max_dd = np.min(drawdown)

ax3.fill_between(range(len(cumulative)), 0, cumulative * 100,
                 where=cumulative >= 0, color=MLGREEN, alpha=0.3)
ax3.fill_between(range(len(cumulative)), 0, cumulative * 100,
                 where=cumulative < 0, color=MLRED, alpha=0.3)
ax3.plot(cumulative * 100, color=MLPURPLE, linewidth=2)

# Drawdown
ax3_twin = ax3.twinx()
ax3_twin.fill_between(range(len(drawdown)), 0, drawdown * 100, alpha=0.3, color=MLRED)
ax3_twin.set_ylabel('Drawdown (%)', fontsize=10, color=MLRED)

ax3.set_xlabel('Trading Day', fontsize=10)
ax3.set_ylabel('Cumulative Return (%)', fontsize=10)
ax3.set_title(f'Performance: Max Drawdown = {max_dd*100:.1f}%', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.grid(alpha=0.3)

# Plot 4: Summary statistics
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate all stats using NumPy
stats = {
    'Total Return': f'{(prices[-1]/prices[0] - 1) * 100:.1f}%',
    'Annual Return': f'{np.mean(returns) * 252 * 100:.1f}%',
    'Annual Volatility': f'{np.std(returns) * np.sqrt(252) * 100:.1f}%',
    'Sharpe Ratio': f'{(np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)):.2f}',
    'Max Drawdown': f'{max_dd * 100:.1f}%',
    'VaR (95%)': f'{np.percentile(returns, 5) * 100:.2f}%',
    'Skewness': f'{np.mean((returns - np.mean(returns))**3) / np.std(returns)**3:.2f}',
    'Kurtosis': f'{np.mean((returns - np.mean(returns))**4) / np.std(returns)**4:.2f}',
}

ax4.text(0.5, 0.95, 'Portfolio Statistics (NumPy)', ha='center', va='top',
         fontsize=14, fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

y = 0.85
for key, value in stats.items():
    ax4.text(0.3, y, f'{key}:', ha='right', va='center', fontsize=11,
             transform=ax4.transAxes, color='gray')
    ax4.text(0.35, y, value, ha='left', va='center', fontsize=11,
             transform=ax4.transAxes, fontweight='bold', color=MLPURPLE)
    y -= 0.1

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
