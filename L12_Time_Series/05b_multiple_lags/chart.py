"""Multiple Lags - Creating lag features"""
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
MLRED = '#D62728'

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30, freq='D')
prices = 100 + np.cumsum(np.random.randn(30) * 2)
df = pd.DataFrame({'Price': prices}, index=dates)

fig, ax = plt.subplots(figsize=(10, 6))

colors = [MLBLUE, MLORANGE, MLGREEN, MLRED]
for i, lag in enumerate([0, 1, 2, 3]):
    lagged = df['Price'].shift(lag)
    ax.plot(df.index[5:15], lagged[5:15], 'o-', color=colors[i],
            linewidth=2, markersize=8-i, label=f'shift({lag})', alpha=0.8)

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title('Creating Multiple Lag Features', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
