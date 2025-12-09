"""Shift and Lag Operations - Creating lagged features"""
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

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30, freq='D')
prices = 100 + np.cumsum(np.random.randn(30) * 2)
df = pd.DataFrame({'Price': prices}, index=dates)

# Create lagged features
df['Lag_1'] = df['Price'].shift(1)
df['Lag_2'] = df['Price'].shift(2)
df['Lead_1'] = df['Price'].shift(-1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Shift and Lag Operations', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Shift demonstration
ax1 = axes[0, 0]
ax1.plot(df.index, df['Price'], 'o-', color=MLBLUE, linewidth=2, markersize=6, label='Original')
ax1.plot(df.index, df['Lag_1'], 's--', color=MLORANGE, linewidth=2, markersize=5, label='shift(1) - Lag')
ax1.plot(df.index, df['Lead_1'], '^--', color=MLGREEN, linewidth=2, markersize=5, label='shift(-1) - Lead')

ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title("df['Price'].shift(n) - Move data forward/backward", fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Multiple lags visualization
ax2 = axes[0, 1]
colors = [MLBLUE, MLORANGE, MLGREEN, MLRED]
for i, lag in enumerate([0, 1, 2, 3]):
    lagged = df['Price'].shift(lag)
    ax2.plot(df.index[5:15], lagged[5:15], 'o-', color=colors[i],
             linewidth=2, markersize=8-i, label=f'shift({lag})', alpha=0.8)

ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.set_title('Creating Multiple Lag Features', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Daily returns using shift
ax3 = axes[1, 0]
df['Return'] = (df['Price'] - df['Price'].shift(1)) / df['Price'].shift(1) * 100

colors = [MLGREEN if r >= 0 else MLRED for r in df['Return'].dropna()]
ax3.bar(df.index[1:], df['Return'].dropna(), color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(0, color='black', linewidth=1)
ax3.axhline(df['Return'].mean(), color=MLPURPLE, linestyle='--', linewidth=2,
            label=f"Mean: {df['Return'].mean():.2f}%")

ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Daily Return (%)', fontsize=10)
ax3.set_title("Returns: (Price - Price.shift(1)) / Price.shift(1)", fontsize=9,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax3.tick_params(axis='x', rotation=45)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Lagged correlation for prediction
ax4 = axes[1, 1]
# Calculate correlations with lagged values
lags = range(1, 11)
correlations = [df['Price'].corr(df['Price'].shift(lag)) for lag in lags]

ax4.bar(lags, correlations, color=MLBLUE, alpha=0.7, edgecolor='black')
ax4.axhline(0, color='black', linewidth=1)

# Highlight significant correlations
for i, (lag, corr) in enumerate(zip(lags, correlations)):
    if abs(corr) > 0.5:
        ax4.bar(lag, corr, color=MLGREEN, alpha=0.7, edgecolor='black')
    ax4.text(lag, corr + 0.02, f'{corr:.2f}', ha='center', fontsize=8)

ax4.set_xlabel('Lag (days)', fontsize=10)
ax4.set_ylabel('Autocorrelation', fontsize=10)
ax4.set_title('Autocorrelation: Price.corr(Price.shift(lag))', fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax4.set_xticks(lags)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
