"""Resampling - Changing time series frequency"""
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

# Generate daily data
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
df = pd.DataFrame({'Price': prices}, index=dates)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Time Series Resampling', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Original daily data
ax1 = axes[0, 0]
ax1.plot(df.index, df['Price'], color=MLBLUE, linewidth=1, alpha=0.7)
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title(f'Daily Data: {len(df)} observations', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(alpha=0.3)

# Plot 2: Weekly resampling
ax2 = axes[0, 1]
weekly = df.resample('W').agg({'Price': ['mean', 'min', 'max']})
weekly.columns = ['Mean', 'Min', 'Max']

ax2.fill_between(weekly.index, weekly['Min'], weekly['Max'], alpha=0.3, color=MLBLUE, label='Range')
ax2.plot(weekly.index, weekly['Mean'], color=MLBLUE, linewidth=2, label='Weekly Mean')
ax2.scatter(weekly.index, weekly['Mean'], color=MLBLUE, s=30)

ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.set_title(f"Weekly: df.resample('W').mean() - {len(weekly)} obs", fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Monthly OHLC
ax3 = axes[1, 0]
monthly = df.resample('M').agg({'Price': ['first', 'max', 'min', 'last']})
monthly.columns = ['Open', 'High', 'Low', 'Close']

# Candlestick-style visualization
for i, (date, row) in enumerate(monthly.iterrows()):
    color = MLGREEN if row['Close'] >= row['Open'] else MLRED
    # Body
    ax3.bar(date, abs(row['Close'] - row['Open']),
            bottom=min(row['Open'], row['Close']),
            width=15, color=color, alpha=0.7)
    # Wick
    ax3.plot([date, date], [row['Low'], row['High']], color='black', linewidth=1)

ax3.set_xlabel('Month', fontsize=10)
ax3.set_ylabel('Price ($)', fontsize=10)
ax3.set_title("Monthly OHLC: resample('M').agg(['first','max','min','last'])", fontsize=9,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(alpha=0.3)

# Plot 4: Quarterly comparison
ax4 = axes[1, 1]
quarterly = df.resample('Q').agg({'Price': ['mean', 'std']})
quarterly.columns = ['Mean', 'Std']
quarterly['Quarter'] = ['Q1', 'Q2', 'Q3', 'Q4']

x = np.arange(len(quarterly))
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]
bars = ax4.bar(quarterly['Quarter'], quarterly['Mean'], yerr=quarterly['Std'],
               color=colors, alpha=0.7, capsize=5, edgecolor='black')

ax4.set_xlabel('Quarter', fontsize=10)
ax4.set_ylabel('Avg Price ($)', fontsize=10)
ax4.set_title("Quarterly: resample('Q').agg(['mean','std'])", fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax4.annotate(f'${height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
