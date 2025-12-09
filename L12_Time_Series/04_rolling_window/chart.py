"""Rolling Window - Moving statistics"""
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

# Generate stock data
np.random.seed(42)
n = 252
dates = pd.date_range('2024-01-01', periods=n, freq='B')
prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015))

df = pd.DataFrame({'Price': prices}, index=dates)
returns = df['Price'].pct_change()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Rolling Window Operations', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Multiple moving averages
ax1 = axes[0, 0]
ax1.plot(df.index, df['Price'], color='gray', alpha=0.5, linewidth=1, label='Price')
ax1.plot(df.index, df['Price'].rolling(5).mean(), color=MLGREEN, linewidth=1.5, label='MA(5)')
ax1.plot(df.index, df['Price'].rolling(20).mean(), color=MLBLUE, linewidth=2, label='MA(20)')
ax1.plot(df.index, df['Price'].rolling(50).mean(), color=MLORANGE, linewidth=2.5, label='MA(50)')

ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title("Moving Averages: df['Price'].rolling(N).mean()", fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Rolling volatility
ax2 = axes[0, 1]
rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100  # Annualized

ax2.fill_between(df.index, 0, rolling_vol, color=MLRED, alpha=0.3)
ax2.plot(df.index, rolling_vol, color=MLRED, linewidth=2)
ax2.axhline(rolling_vol.mean(), color=MLPURPLE, linestyle='--', linewidth=2,
            label=f'Mean Vol: {rolling_vol.mean():.1f}%')

ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Annualized Volatility (%)', fontsize=10)
ax2.set_title("Rolling Vol: returns.rolling(20).std() * sqrt(252)", fontsize=9,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Bollinger Bands
ax3 = axes[1, 0]
ma20 = df['Price'].rolling(20).mean()
std20 = df['Price'].rolling(20).std()
upper = ma20 + 2 * std20
lower = ma20 - 2 * std20

ax3.plot(df.index, df['Price'], color='gray', linewidth=1, label='Price')
ax3.plot(df.index, ma20, color=MLPURPLE, linewidth=2, label='MA(20)')
ax3.fill_between(df.index, lower, upper, color=MLBLUE, alpha=0.2, label='2-Std Band')
ax3.plot(df.index, upper, color=MLBLUE, linewidth=1, linestyle='--')
ax3.plot(df.index, lower, color=MLBLUE, linewidth=1, linestyle='--')

# Highlight touches
above = df['Price'] > upper
below = df['Price'] < lower
ax3.scatter(df.index[above], df['Price'][above], color=MLRED, s=30, zorder=5, label='Above band')
ax3.scatter(df.index[below], df['Price'][below], color=MLGREEN, s=30, zorder=5, label='Below band')

ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Price ($)', fontsize=10)
ax3.set_title('Bollinger Bands: MA +/- 2*rolling.std()', fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax3.tick_params(axis='x', rotation=45)
ax3.legend(fontsize=8, ncol=2)
ax3.grid(alpha=0.3)

# Plot 4: Rolling min/max
ax4 = axes[1, 1]
rolling_max = df['Price'].rolling(20).max()
rolling_min = df['Price'].rolling(20).min()

ax4.plot(df.index, df['Price'], color='gray', linewidth=1, label='Price')
ax4.plot(df.index, rolling_max, color=MLGREEN, linewidth=1.5, label='20-day High')
ax4.plot(df.index, rolling_min, color=MLRED, linewidth=1.5, label='20-day Low')
ax4.fill_between(df.index, rolling_min, rolling_max, color=MLBLUE, alpha=0.1)

# Highlight new highs
new_high = df['Price'] == rolling_max
ax4.scatter(df.index[new_high], df['Price'][new_high], color=MLGREEN, s=20, marker='^', zorder=5)

ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Price ($)', fontsize=10)
ax4.set_title('Rolling Channel: rolling(20).max/min()', fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax4.tick_params(axis='x', rotation=45)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
