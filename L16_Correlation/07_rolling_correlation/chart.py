"""Rolling Correlation - Time-varying relationships"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

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

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Rolling Correlation: Time-Varying Relationships', fontsize=14,
             fontweight='bold', color=MLPURPLE)

# Generate correlated time series with changing relationship
n = 504  # 2 years
dates = pd.date_range('2023-01-01', periods=n, freq='B')

# Base market returns
market = np.random.normal(0.0004, 0.01, n)

# Stock with changing beta
stock = np.zeros(n)
for i in range(n):
    if i < 126:  # Q1: High correlation
        stock[i] = 0.9 * market[i] + np.random.normal(0, 0.005)
    elif i < 252:  # Q2-Q3: Lower correlation
        stock[i] = 0.4 * market[i] + np.random.normal(0, 0.01)
    elif i < 378:  # Q4: Negative correlation (crisis)
        stock[i] = -0.3 * market[i] + np.random.normal(0, 0.015)
    else:  # Recovery
        stock[i] = 0.7 * market[i] + np.random.normal(0, 0.008)

df = pd.DataFrame({'Market': market, 'Stock': stock}, index=dates)

# Plot 1: Raw returns
ax1 = axes[0, 0]
ax1.plot(df.index, df['Market'] * 100, color=MLBLUE, alpha=0.7, linewidth=1, label='Market')
ax1.plot(df.index, df['Stock'] * 100, color=MLGREEN, alpha=0.7, linewidth=1, label='Stock')
ax1.set_title('Daily Returns', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Return (%)', fontsize=10)
ax1.tick_params(axis='x', rotation=45)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Rolling correlation
ax2 = axes[0, 1]
windows = [20, 60, 120]
colors = [MLBLUE, MLORANGE, MLRED]

for window, color in zip(windows, colors):
    rolling_corr = df['Market'].rolling(window).corr(df['Stock'])
    ax2.plot(df.index, rolling_corr, color=color, linewidth=2, label=f'{window}-day')

ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.axhline(df['Market'].corr(df['Stock']), color=MLPURPLE, linestyle=':',
            linewidth=2, label=f"Full period: {df['Market'].corr(df['Stock']):.2f}")

ax2.set_title('Rolling Correlation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Correlation', fontsize=10)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(-1, 1)

# Annotate regimes
ax2.axvspan(dates[0], dates[125], alpha=0.1, color=MLGREEN)
ax2.axvspan(dates[252], dates[377], alpha=0.1, color=MLRED)
ax2.text(dates[60], 0.8, 'Normal', fontsize=8, color=MLGREEN)
ax2.text(dates[300], -0.6, 'Crisis', fontsize=8, color=MLRED)

# Plot 3: Scatter by regime
ax3 = axes[1, 0]
regimes = [
    (0, 126, 'Pre-Crisis', MLGREEN),
    (252, 378, 'Crisis', MLRED),
    (378, n, 'Recovery', MLBLUE)
]

for start, end, label, color in regimes:
    ax3.scatter(df['Market'].iloc[start:end] * 100, df['Stock'].iloc[start:end] * 100,
                color=color, alpha=0.5, s=30, label=label)
    # Add regression line
    z = np.polyfit(df['Market'].iloc[start:end], df['Stock'].iloc[start:end], 1)
    x_line = np.linspace(df['Market'].min(), df['Market'].max(), 50)
    ax3.plot(x_line * 100, np.poly1d(z)(x_line) * 100, color=color, linewidth=2)

ax3.axhline(0, color='gray', linewidth=1)
ax3.axvline(0, color='gray', linewidth=1)
ax3.set_title('Scatter by Market Regime', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Market Return (%)', fontsize=10)
ax3.set_ylabel('Stock Return (%)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Implications
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'Why Rolling Correlation Matters', ha='center', fontsize=14,
         fontweight='bold', color=MLPURPLE, transform=ax4.transAxes)

implications = [
    ('Correlations Change Over Time', 'Static correlation masks regime shifts'),
    ('Crisis Correlation Spike', 'Assets become more correlated in crashes'),
    ('Diversification Illusion', 'Low normal correlation may not protect in crisis'),
    ('Window Size Matters', 'Shorter = noisier, Longer = more lag'),
]

y = 0.75
for title, desc in implications:
    ax4.text(0.1, y, title, fontsize=11, fontweight='bold', color=MLBLUE, transform=ax4.transAxes)
    ax4.text(0.1, y - 0.07, desc, fontsize=9, color='gray', transform=ax4.transAxes)
    y -= 0.17

ax4.text(0.5, 0.1, "df['A'].rolling(window).corr(df['B'])",
         ha='center', fontsize=10, family='monospace', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
