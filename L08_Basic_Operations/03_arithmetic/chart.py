"""Arithmetic Operations - DataFrame arithmetic and broadcasting"""
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

# Create sample stock data
np.random.seed(42)
n_days = 30
dates = pd.date_range('2024-01-01', periods=n_days, freq='D')

high = 100 + np.cumsum(np.random.randn(n_days) * 2)
low = high - np.random.uniform(1, 5, n_days)
close = (high + low) / 2 + np.random.randn(n_days)
open_price = close.shift(1) if hasattr(close, 'shift') else np.roll(close, 1)
open_price = np.roll(close, 1)
open_price[0] = close[0] - 0.5

# Calculate derived columns using arithmetic
spread = high - low  # Column - Column
pct_change = (close - open_price) / open_price * 100  # Complex formula
midpoint = (high + low) / 2  # Column / scalar
range_pct = spread / close * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Arithmetic Operations on Stock Data', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: High-Low Spread
ax1 = axes[0, 0]
ax1.fill_between(dates, low, high, alpha=0.3, color=MLBLUE, label='High-Low Range')
ax1.plot(dates, spread, color=MLORANGE, linewidth=2, label='Spread (High - Low)')
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title("df['Spread'] = df['High'] - df['Low']", fontsize=11, color=MLBLUE, family='monospace')
ax1.legend(fontsize=9)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(alpha=0.3)

# Plot 2: Percentage Change
ax2 = axes[0, 1]
colors = [MLGREEN if x > 0 else MLRED for x in pct_change]
ax2.bar(dates, pct_change, color=colors, alpha=0.7, width=0.8)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Daily Change (%)', fontsize=10)
ax2.set_title("df['Pct_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100", fontsize=9, color=MLGREEN, family='monospace')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Midpoint calculation
ax3 = axes[1, 0]
ax3.plot(dates, high, 'o-', color=MLRED, alpha=0.5, markersize=3, label='High')
ax3.plot(dates, low, 'o-', color=MLBLUE, alpha=0.5, markersize=3, label='Low')
ax3.plot(dates, midpoint, 's-', color=MLPURPLE, markersize=4, linewidth=2, label='Midpoint')
ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Price ($)', fontsize=10)
ax3.set_title("df['Midpoint'] = (df['High'] + df['Low']) / 2", fontsize=11, color=MLPURPLE, family='monospace')
ax3.legend(fontsize=9)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(alpha=0.3)

# Plot 4: Range as percentage
ax4 = axes[1, 1]
ax4.fill_between(dates, 0, range_pct, alpha=0.5, color=MLORANGE)
ax4.plot(dates, range_pct, color=MLORANGE, linewidth=2)
ax4.axhline(range_pct.mean(), color=MLPURPLE, linestyle='--', linewidth=2,
            label=f'Mean: {range_pct.mean():.1f}%')
ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Range (%)', fontsize=10)
ax4.set_title("df['Range_Pct'] = df['Spread'] / df['Close'] * 100", fontsize=11, color=MLORANGE, family='monospace')
ax4.legend(fontsize=9)
ax4.tick_params(axis='x', rotation=45)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
