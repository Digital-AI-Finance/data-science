"""Percentage Change - Computing returns and growth rates"""
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
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.015))
df = pd.DataFrame({'Price': prices}, index=dates)

# Calculate various returns
df['Daily_Pct'] = df['Price'].pct_change() * 100
df['Weekly_Pct'] = df['Price'].pct_change(periods=5) * 100
df['Monthly_Pct'] = df['Price'].pct_change(periods=21) * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Percentage Change Operations', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Daily returns
ax1 = axes[0, 0]
returns = df['Daily_Pct'].dropna()
colors = [MLGREEN if r >= 0 else MLRED for r in returns]
ax1.bar(returns.index, returns, color=colors, alpha=0.7, width=0.8)
ax1.axhline(0, color='black', linewidth=1)
ax1.axhline(returns.mean(), color=MLPURPLE, linestyle='--', linewidth=2,
            label=f'Mean: {returns.mean():.2f}%')

ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Daily Return (%)', fontsize=10)
ax1.set_title("df['Price'].pct_change() * 100", fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Different periods comparison
ax2 = axes[0, 1]
ax2.plot(df.index, df['Daily_Pct'], color=MLBLUE, alpha=0.5, linewidth=1, label='Daily')
ax2.plot(df.index, df['Weekly_Pct'], color=MLORANGE, linewidth=2, label='Weekly (5-day)')
ax2.plot(df.index, df['Monthly_Pct'], color=MLGREEN, linewidth=2.5, label='Monthly (21-day)')
ax2.axhline(0, color='black', linewidth=1)

ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Return (%)', fontsize=10)
ax2.set_title("pct_change(periods=N) - Different lookbacks", fontsize=10,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Return distribution
ax3 = axes[1, 0]
ax3.hist(returns, bins=30, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')

# Normal distribution overlay
x = np.linspace(returns.min(), returns.max(), 100)
mu, std = returns.mean(), returns.std()
normal = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/std)**2)
ax3.plot(x, normal, color=MLRED, linewidth=2, label='Normal fit')

# Mark key percentiles
for pct, color, label in [(5, MLRED, 'VaR 95%'), (50, MLPURPLE, 'Median')]:
    val = np.percentile(returns, pct)
    ax3.axvline(val, color=color, linestyle='--', linewidth=2, label=f'{label}: {val:.2f}%')

ax3.set_xlabel('Daily Return (%)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title('Return Distribution', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Cumulative returns
ax4 = axes[1, 1]
cumulative = (1 + df['Daily_Pct']/100).cumprod() - 1
cumulative = cumulative * 100

ax4.fill_between(df.index, 0, cumulative, where=cumulative >= 0,
                 color=MLGREEN, alpha=0.3, interpolate=True)
ax4.fill_between(df.index, 0, cumulative, where=cumulative < 0,
                 color=MLRED, alpha=0.3, interpolate=True)
ax4.plot(df.index, cumulative, color=MLPURPLE, linewidth=2)

ax4.axhline(0, color='black', linewidth=1)
ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Cumulative Return (%)', fontsize=10)
ax4.set_title("Cumulative: (1 + pct_change()).cumprod() - 1", fontsize=9,
              fontweight='bold', color=MLPURPLE, family='monospace')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(alpha=0.3)

# Add final return annotation
final_ret = cumulative.iloc[-1]
ax4.annotate(f'Total: {final_ret:.1f}%', xy=(df.index[-1], final_ret),
             xytext=(-60, 20), textcoords='offset points',
             fontsize=10, fontweight='bold', color=MLPURPLE,
             arrowprops=dict(arrowstyle='->', color=MLPURPLE))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
