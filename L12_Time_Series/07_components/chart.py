"""Time Series Components - Trend, Seasonality, and Residuals"""
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

# Generate synthetic time series with clear components
np.random.seed(42)
n = 365 * 2  # 2 years of daily data
dates = pd.date_range('2023-01-01', periods=n, freq='D')

# Components
trend = np.linspace(100, 180, n)  # Linear trend
seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 365)  # Annual seasonality
weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly pattern
noise = np.random.randn(n) * 3  # Random noise

# Combined series
series = trend + seasonal + weekly + noise

df = pd.DataFrame({'Value': series}, index=dates)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Time Series Decomposition', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Original series with components highlighted
ax1 = axes[0, 0]
ax1.plot(df.index, df['Value'], color=MLBLUE, linewidth=1, alpha=0.7, label='Original')
ax1.plot(df.index, trend, color=MLGREEN, linewidth=2.5, label='Trend', linestyle='--')

ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Value', fontsize=10)
ax1.set_title('Original Series = Trend + Seasonality + Noise', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax1.tick_params(axis='x', rotation=45)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Seasonal component
ax2 = axes[0, 1]
# Detrend first
detrended = df['Value'] - trend

# Monthly average to show seasonality
monthly = df.copy()
monthly['Month'] = monthly.index.month
monthly['Detrended'] = detrended
monthly_avg = monthly.groupby('Month')['Detrended'].mean()

ax2.bar(monthly_avg.index, monthly_avg, color=MLPURPLE, alpha=0.7, edgecolor='black')
ax2.axhline(0, color='black', linewidth=1)

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_labels, fontsize=8)
ax2.set_xlabel('Month', fontsize=10)
ax2.set_ylabel('Seasonal Effect', fontsize=10)
ax2.set_title('Seasonal Pattern (Monthly Average)', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax2.grid(axis='y', alpha=0.3)

# Annotate peaks
max_month = monthly_avg.idxmax()
min_month = monthly_avg.idxmin()
ax2.annotate(f'Peak: {month_labels[max_month-1]}',
             xy=(max_month, monthly_avg[max_month]),
             xytext=(max_month, monthly_avg[max_month] + 3),
             ha='center', fontsize=9, color=MLGREEN, fontweight='bold')

# Plot 3: Residuals analysis
ax3 = axes[1, 0]
# Simple decomposition: remove trend and seasonal
residuals = detrended - seasonal

ax3.plot(df.index, residuals, color=MLRED, linewidth=0.8, alpha=0.7)
ax3.axhline(0, color='black', linewidth=1)
ax3.axhline(residuals.mean() + 2*residuals.std(), color=MLORANGE, linestyle='--',
            linewidth=1.5, label='+/- 2 Std')
ax3.axhline(residuals.mean() - 2*residuals.std(), color=MLORANGE, linestyle='--',
            linewidth=1.5)

ax3.fill_between(df.index, residuals.mean() - 2*residuals.std(),
                 residuals.mean() + 2*residuals.std(),
                 color=MLORANGE, alpha=0.1)

ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Residual', fontsize=10)
ax3.set_title(f'Residuals: Std = {residuals.std():.2f}', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax3.tick_params(axis='x', rotation=45)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Component summary (stacked view)
ax4 = axes[1, 1]

# Show decomposition conceptually
components = {
    'Trend': trend[-100:],
    'Seasonal': seasonal[-100:],
    'Weekly': weekly[-100:],
    'Noise': noise[-100:]
}

y_offset = 0
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]
for (name, data), color in zip(components.items(), colors):
    # Normalize for display
    normalized = (data - data.mean()) / (data.max() - data.min()) * 20
    ax4.plot(range(100), normalized + y_offset, color=color, linewidth=2, label=name)
    ax4.axhline(y_offset, color='gray', linewidth=0.5, linestyle=':')
    y_offset += 25

ax4.set_xlabel('Days (last 100)', fontsize=10)
ax4.set_ylabel('Component (normalized)', fontsize=10)
ax4.set_title('Decomposition: Additive Model', fontsize=11,
              fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=9, loc='upper right')
ax4.set_yticks([])
ax4.grid(alpha=0.3)

# Add formula
ax4.text(50, 85, 'Y = Trend + Seasonal + Cyclical + Noise',
         ha='center', fontsize=11, fontweight='bold', color=MLPURPLE,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.3))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
