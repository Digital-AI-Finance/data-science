"""Returns Calculation - Computing and visualizing stock returns"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Generate stock price data
np.random.seed(42)
n_days = 100
dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))

# Calculate returns
simple_returns = pd.Series(prices).pct_change() * 100  # Simple returns in %
log_returns = np.log(prices[1:] / prices[:-1]) * 100  # Log returns in %
log_returns = np.insert(log_returns, 0, np.nan)

# Cumulative returns
cumulative_simple = (1 + simple_returns/100).cumprod() - 1
cumulative_log = np.exp(np.nancumsum(log_returns/100)) - 1

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Stock Returns: Calculation Methods', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Price and returns comparison
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
ax1.plot(dates, prices, color=MLBLUE, linewidth=2, label='Price')
ax1_twin.bar(dates, simple_returns, color=MLORANGE, alpha=0.5, width=0.8, label='Simple Return')
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10, color=MLBLUE)
ax1_twin.set_ylabel('Daily Return (%)', fontsize=10, color=MLORANGE)
ax1.set_title("Simple Return: df['Price'].pct_change()", fontsize=11, color=MLBLUE, family='monospace')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc='upper left', fontsize=9)
ax1_twin.legend(loc='upper right', fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Simple vs Log returns
ax2 = axes[0, 1]
ax2.scatter(simple_returns[1:], log_returns[1:], alpha=0.5, color=MLPURPLE, s=30)
ax2.plot([-8, 8], [-8, 8], 'k--', alpha=0.5, label='y = x (perfect match)')
ax2.set_xlabel('Simple Return (%)', fontsize=10)
ax2.set_ylabel('Log Return (%)', fontsize=10)
ax2.set_title('Simple vs Log Returns', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_aspect('equal')

# Add correlation text
corr = np.corrcoef(simple_returns[1:], log_returns[1:])[0, 1]
ax2.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax2.transAxes,
         fontsize=10, va='top', color=MLPURPLE, fontweight='bold')

# Plot 3: Return distribution
ax3 = axes[1, 0]
ax3.hist(simple_returns.dropna(), bins=30, alpha=0.7, color=MLBLUE, edgecolor='black', label='Simple')
ax3.hist(log_returns[1:], bins=30, alpha=0.5, color=MLORANGE, edgecolor='black', label='Log')
ax3.axvline(simple_returns.mean(), color=MLBLUE, linestyle='--', linewidth=2,
            label=f'Simple Mean: {simple_returns.mean():.2f}%')
ax3.axvline(np.nanmean(log_returns), color=MLORANGE, linestyle='--', linewidth=2,
            label=f'Log Mean: {np.nanmean(log_returns):.2f}%')
ax3.set_xlabel('Return (%)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Return Distribution', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Cumulative returns
ax4 = axes[1, 1]
ax4.plot(dates, cumulative_simple * 100, color=MLBLUE, linewidth=2, label='Cumulative (Simple)')
ax4.plot(dates, cumulative_log * 100, color=MLORANGE, linewidth=2, linestyle='--', label='Cumulative (Log)')
ax4.axhline(0, color='black', linewidth=0.5)
ax4.fill_between(dates, 0, cumulative_simple * 100,
                 where=cumulative_simple > 0, alpha=0.3, color=MLGREEN, label='Gain')
ax4.fill_between(dates, 0, cumulative_simple * 100,
                 where=cumulative_simple < 0, alpha=0.3, color=MLRED, label='Loss')
ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Cumulative Return (%)', fontsize=10)
ax4.set_title('Cumulative Returns Over Time', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.tick_params(axis='x', rotation=45)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Add final return annotation
final_return = cumulative_simple.iloc[-1] * 100
ax4.annotate(f'Final: {final_return:.1f}%', xy=(dates[-1], final_return),
             xytext=(-50, 20), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color=MLPURPLE),
             fontsize=10, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
