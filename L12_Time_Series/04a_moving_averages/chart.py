"""Moving Averages - Multiple window sizes"""
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
n = 252
dates = pd.date_range('2024-01-01', periods=n, freq='B')
prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015))
df = pd.DataFrame({'Price': prices}, index=dates)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df.index, df['Price'], color='gray', alpha=0.5, linewidth=1, label='Price')
ax.plot(df.index, df['Price'].rolling(5).mean(), color=MLGREEN, linewidth=1.5, label='MA(5)')
ax.plot(df.index, df['Price'].rolling(20).mean(), color=MLBLUE, linewidth=2, label='MA(20)')
ax.plot(df.index, df['Price'].rolling(50).mean(), color=MLORANGE, linewidth=2.5, label='MA(50)')

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title("Moving Averages: df['Price'].rolling(N).mean()", fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
