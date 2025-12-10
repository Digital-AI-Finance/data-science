"""Cumulative Returns - Total return over time"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.015))
df = pd.DataFrame({'Price': prices}, index=dates)
df['Daily_Pct'] = df['Price'].pct_change() * 100

fig, ax = plt.subplots(figsize=(10, 6))

cumulative = (1 + df['Daily_Pct']/100).cumprod() - 1
cumulative = cumulative * 100

ax.fill_between(df.index, 0, cumulative, where=cumulative >= 0,
                color=MLGREEN, alpha=0.3, interpolate=True)
ax.fill_between(df.index, 0, cumulative, where=cumulative < 0,
                color=MLRED, alpha=0.3, interpolate=True)
ax.plot(df.index, cumulative, color=MLPURPLE, linewidth=2)

ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Cumulative Return (%)', fontsize=10)
ax.set_title("Cumulative: (1 + pct_change()).cumprod() - 1", fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3)

final_ret = cumulative.iloc[-1]
ax.annotate(f'Total: {final_ret:.1f}%', xy=(df.index[-1], final_ret),
            xytext=(-60, 20), textcoords='offset points',
            fontsize=11, fontweight='bold', color=MLPURPLE,
            arrowprops=dict(arrowstyle='->', color=MLPURPLE))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
