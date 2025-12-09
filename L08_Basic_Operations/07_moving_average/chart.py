"""Moving Average - Rolling calculations for trend analysis"""
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
n_days = 200
dates = pd.date_range('2024-01-01', periods=n_days, freq='D')

# Price with trend and noise
trend = np.linspace(100, 130, n_days)
seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, n_days))
noise = np.cumsum(np.random.randn(n_days) * 0.5)
prices = trend + seasonal + noise

# Calculate moving averages
ma_5 = pd.Series(prices).rolling(window=5).mean()
ma_20 = pd.Series(prices).rolling(window=20).mean()
ma_50 = pd.Series(prices).rolling(window=50).mean()

# Calculate rolling statistics
rolling_std = pd.Series(prices).rolling(window=20).std()
upper_band = ma_20 + 2 * rolling_std
lower_band = ma_20 - 2 * rolling_std

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Moving Averages and Rolling Statistics', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Multiple MAs
ax1 = axes[0, 0]
ax1.plot(dates, prices, color='gray', alpha=0.5, linewidth=1, label='Price')
ax1.plot(dates, ma_5, color=MLBLUE, linewidth=1.5, label='MA(5) - Short')
ax1.plot(dates, ma_20, color=MLORANGE, linewidth=2, label='MA(20) - Medium')
ax1.plot(dates, ma_50, color=MLGREEN, linewidth=2.5, label='MA(50) - Long')
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title("df['Price'].rolling(window=N).mean()", fontsize=11, color=MLBLUE, family='monospace')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Bollinger Bands
ax2 = axes[0, 1]
ax2.plot(dates, prices, color='gray', alpha=0.7, linewidth=1, label='Price')
ax2.plot(dates, ma_20, color=MLPURPLE, linewidth=2, label='MA(20)')
ax2.fill_between(dates, lower_band, upper_band, alpha=0.2, color=MLBLUE, label='Bollinger Bands')
ax2.plot(dates, upper_band, color=MLBLUE, linewidth=1, linestyle='--')
ax2.plot(dates, lower_band, color=MLBLUE, linewidth=1, linestyle='--')
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.set_title('Bollinger Bands (MA +/- 2*STD)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Rolling volatility
ax3 = axes[1, 0]
returns = pd.Series(prices).pct_change() * 100
rolling_vol = returns.rolling(window=20).std()
ax3.fill_between(dates, 0, rolling_vol, alpha=0.5, color=MLORANGE)
ax3.plot(dates, rolling_vol, color=MLORANGE, linewidth=2)
ax3.axhline(rolling_vol.mean(), color=MLRED, linestyle='--', linewidth=2,
            label=f'Mean Volatility: {rolling_vol.mean():.2f}%')
ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Rolling Volatility (%)', fontsize=10)
ax3.set_title("returns.rolling(20).std() - Rolling Volatility", fontsize=11, color=MLORANGE, family='monospace')
ax3.tick_params(axis='x', rotation=45)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Golden/Death Cross strategy
ax4 = axes[1, 1]
ax4.plot(dates, prices, color='gray', alpha=0.5, linewidth=1, label='Price')
ax4.plot(dates, ma_20, color=MLBLUE, linewidth=2, label='MA(20) - Fast')
ax4.plot(dates, ma_50, color=MLORANGE, linewidth=2, label='MA(50) - Slow')

# Find crossovers
ma_diff = ma_20 - ma_50
golden_cross = (ma_diff > 0) & (ma_diff.shift(1) <= 0)
death_cross = (ma_diff < 0) & (ma_diff.shift(1) >= 0)

# Mark crossovers
for i, (gc, dc) in enumerate(zip(golden_cross, death_cross)):
    if gc:
        ax4.scatter(dates[i], prices[i], color=MLGREEN, s=150, marker='^', zorder=5)
        ax4.annotate('Golden\nCross', xy=(dates[i], prices[i]), xytext=(10, 20),
                    textcoords='offset points', fontsize=8, color=MLGREEN, fontweight='bold')
    if dc:
        ax4.scatter(dates[i], prices[i], color=MLRED, s=150, marker='v', zorder=5)

ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Price ($)', fontsize=10)
ax4.set_title('Moving Average Crossover Strategy', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.tick_params(axis='x', rotation=45)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
