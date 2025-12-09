"""Time Series Basics - Introduction to time series data"""
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

# Generate time series data
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
n = len(dates)

# Create price series with trend, seasonality, and noise
trend = np.linspace(100, 150, n)
seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, n))
noise = np.cumsum(np.random.randn(n) * 0.5)
prices = trend + seasonal + noise

df = pd.DataFrame({'Date': dates, 'Price': prices})
df.set_index('Date', inplace=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Time Series Data Fundamentals', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic time series
ax1 = axes[0, 0]
ax1.plot(df.index, df['Price'], color=MLBLUE, linewidth=1.5)
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title('Stock Price Time Series', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(alpha=0.3)

# Annotate components
ax1.annotate('Trend', xy=(dates[300], prices[300]), xytext=(dates[200], prices[300]+30),
             arrowprops=dict(arrowstyle='->', color=MLGREEN), fontsize=9, color=MLGREEN)
ax1.annotate('Seasonality', xy=(dates[500], prices[500]), xytext=(dates[400], prices[500]-20),
             arrowprops=dict(arrowstyle='->', color=MLORANGE), fontsize=9, color=MLORANGE)

# Plot 2: DatetimeIndex features
ax2 = axes[0, 1]
monthly = df.resample('ME').mean()
ax2.bar(monthly.index, monthly['Price'], width=20, color=MLPURPLE, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Month', fontsize=10)
ax2.set_ylabel('Avg Price ($)', fontsize=10)
ax2.set_title('Monthly Aggregation: df.resample("M").mean()', fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Subsetting by date
ax3 = axes[1, 0]
# Full series in light color
ax3.plot(df.index, df['Price'], color='gray', alpha=0.3, linewidth=1)
# Highlight 2024
df_2024 = df.loc['2024']
ax3.plot(df_2024.index, df_2024['Price'], color=MLGREEN, linewidth=2, label='2024 data')
# Highlight specific month
df_jul = df.loc['2023-07']
ax3.plot(df_jul.index, df_jul['Price'], color=MLRED, linewidth=3, label='July 2023')

ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Price ($)', fontsize=10)
ax3.set_title("Date Slicing: df['2024'] or df['2023-07']", fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax3.tick_params(axis='x', rotation=45)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Time-based attributes
ax4 = axes[1, 1]
df['DayOfWeek'] = df.index.dayofweek
df['Month'] = df.index.month

# Average price by day of week
dow_avg = df.groupby('DayOfWeek')['Price'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

bars = ax4.bar(days, dow_avg, color=[MLBLUE]*5 + [MLGREEN]*2, alpha=0.7, edgecolor='black')
ax4.axhline(dow_avg.mean(), color=MLRED, linestyle='--', linewidth=2, label='Overall mean')

ax4.set_xlabel('Day of Week', fontsize=10)
ax4.set_ylabel('Avg Price ($)', fontsize=10)
ax4.set_title('df.index.dayofweek - Extract Time Attributes', fontsize=10,
              fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# Annotate weekend
ax4.annotate('Weekend', xy=(5.5, dow_avg[5]), xytext=(5.5, dow_avg[5]+3),
             fontsize=9, ha='center', color=MLGREEN, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
