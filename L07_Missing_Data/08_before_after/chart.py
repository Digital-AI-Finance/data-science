"""Before/After Comparison - Data cleaning transformation visualization"""
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

# Generate messy data (before)
np.random.seed(42)
n = 50
dates = pd.date_range('2024-01-01', periods=n, freq='D')
prices = 100 + np.cumsum(np.random.randn(n) * 2)

# Create messy version
messy_prices = prices.copy()
messy_prices[10:15] = np.nan  # Missing values
messy_prices[30] = 200  # Outlier
messy_prices[31] = 200  # Duplicate value pattern

messy_volume = np.random.randint(1000, 5000, n).astype(float)
messy_volume[20:25] = np.nan

# Create clean version
clean_prices = pd.Series(messy_prices).interpolate().values
clean_prices[30] = np.median(clean_prices[28:33])  # Fix outlier
clean_volume = pd.Series(messy_volume).fillna(pd.Series(messy_volume).mean()).values

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Before vs After Data Cleaning', fontsize=16, fontweight='bold', color=MLPURPLE)

# Top left: Price before
ax1 = axes[0, 0]
ax1.plot(dates, messy_prices, 'o-', color=MLRED, alpha=0.7, markersize=4)
ax1.scatter(dates[10:15], [np.nan]*5, color=MLRED, s=100, marker='x', label='Missing')
ax1.axvline(dates[30], color=MLORANGE, linestyle='--', alpha=0.5)
ax1.annotate('Outlier', xy=(dates[30], messy_prices[30]), xytext=(dates[35], 190),
             arrowprops=dict(arrowstyle='->', color=MLORANGE), fontsize=9, color=MLORANGE)
ax1.set_title('BEFORE: Stock Price (Dirty)', fontsize=12, fontweight='bold', color=MLRED)
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(alpha=0.3)

# Missing values highlighted
missing_idx = np.where(np.isnan(messy_prices))[0]
for idx in missing_idx:
    left_idx = max(0, idx - 1)
    right_idx = min(n - 1, idx + 1)
    ax1.axvspan(dates[left_idx], dates[right_idx], alpha=0.2, color=MLRED)

# Top right: Price after
ax2 = axes[0, 1]
ax2.plot(dates, clean_prices, 'o-', color=MLGREEN, alpha=0.7, markersize=4)
ax2.set_title('AFTER: Stock Price (Clean)', fontsize=12, fontweight='bold', color=MLGREEN)
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(alpha=0.3)
ax2.text(0.5, 0.95, 'Missing values interpolated, outlier corrected',
         transform=ax2.transAxes, fontsize=9, style='italic', color='gray',
         ha='center', va='top')

# Bottom left: Volume before
ax3 = axes[1, 0]
ax3.bar(dates, messy_volume, color=MLRED, alpha=0.6, width=0.8)
ax3.set_title('BEFORE: Trading Volume (Dirty)', fontsize=12, fontweight='bold', color=MLRED)
ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Volume', fontsize=10)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

# Highlight missing
for idx in np.where(np.isnan(messy_volume))[0]:
    ax3.axvspan(dates[max(0, idx-1)], dates[min(n-1, idx+1)], alpha=0.3, color=MLRED)
ax3.text(0.5, 0.95, f'Missing values: {np.isnan(messy_volume).sum()}',
         transform=ax3.transAxes, fontsize=9, color=MLRED, fontweight='bold',
         ha='center', va='top')

# Bottom right: Volume after
ax4 = axes[1, 1]
ax4.bar(dates, clean_volume, color=MLGREEN, alpha=0.6, width=0.8)
ax4.set_title('AFTER: Trading Volume (Clean)', fontsize=12, fontweight='bold', color=MLGREEN)
ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Volume', fontsize=10)
ax4.tick_params(axis='x', rotation=45)
ax4.grid(axis='y', alpha=0.3)
ax4.text(0.5, 0.95, 'Missing values filled with mean',
         transform=ax4.transAxes, fontsize=9, style='italic', color='gray',
         ha='center', va='top')

# Add statistics comparison
stats_before = f"Before: {np.nanmean(messy_prices):.1f} mean, {np.nanstd(messy_prices):.1f} std"
stats_after = f"After: {np.mean(clean_prices):.1f} mean, {np.std(clean_prices):.1f} std"

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
