"""Periods Comparison - Different lookback windows"""
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
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.015))
df = pd.DataFrame({'Price': prices}, index=dates)
df['Daily_Pct'] = df['Price'].pct_change() * 100
df['Weekly_Pct'] = df['Price'].pct_change(periods=5) * 100
df['Monthly_Pct'] = df['Price'].pct_change(periods=21) * 100

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df.index, df['Daily_Pct'], color=MLBLUE, alpha=0.5, linewidth=1, label='Daily')
ax.plot(df.index, df['Weekly_Pct'], color=MLORANGE, linewidth=2, label='Weekly (5-day)')
ax.plot(df.index, df['Monthly_Pct'], color=MLGREEN, linewidth=2.5, label='Monthly (21-day)')
ax.axhline(0, color='black', linewidth=1)

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Return (%)', fontsize=10)
ax.set_title("pct_change(periods=N) - Different lookbacks", fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
