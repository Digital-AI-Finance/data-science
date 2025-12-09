"""Fillna Methods - Comparison of different fill strategies for stock prices"""
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

# Generate synthetic stock price data with gaps
np.random.seed(42)
n_days = 30
dates = pd.date_range('2024-01-01', periods=n_days, freq='D')

# Base price with trend
prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
original = prices.copy()

# Create gaps
gap_indices = [5, 6, 7, 15, 16, 25]
prices_with_gaps = prices.copy()
prices_with_gaps[gap_indices] = np.nan

# Apply different fill methods
df = pd.DataFrame({'Original': original, 'With_Gaps': prices_with_gaps}, index=dates)
df['Forward_Fill'] = df['With_Gaps'].ffill()
df['Backward_Fill'] = df['With_Gaps'].bfill()
df['Mean_Fill'] = df['With_Gaps'].fillna(df['With_Gaps'].mean())
df['Interpolate'] = df['With_Gaps'].interpolate(method='linear')

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparison of fillna() Methods for Stock Prices', fontsize=14, fontweight='bold', color=MLPURPLE)

methods = [
    ('Forward_Fill', 'Forward Fill (ffill)', 'Last known value carried forward', MLBLUE),
    ('Backward_Fill', 'Backward Fill (bfill)', 'Next known value carried back', MLORANGE),
    ('Mean_Fill', 'Mean Fill', 'Column mean used for all gaps', MLGREEN),
    ('Interpolate', 'Linear Interpolation', 'Linear estimate between points', MLPURPLE)
]

for ax, (col, title, desc, color) in zip(axes.flat, methods):
    # Plot original as dotted
    ax.plot(dates, original, 'o-', color=MLLAVENDER, alpha=0.5, label='Original', markersize=4)

    # Plot filled data
    ax.plot(dates, df[col], 'o-', color=color, label=title, markersize=5, linewidth=2)

    # Highlight filled points
    filled_mask = df['With_Gaps'].isna()
    ax.scatter(dates[filled_mask], df[col][filled_mask],
               color=MLRED, s=100, zorder=5, marker='s', label='Filled Values')

    ax.set_title(f'{title}', fontsize=11, fontweight='bold', color=MLPURPLE)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Stock Price ($)', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.tick_params(axis='x', rotation=45)

    # Add description
    ax.text(0.02, 0.02, desc, transform=ax.transAxes, fontsize=8,
            style='italic', color='gray', va='bottom')

    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
