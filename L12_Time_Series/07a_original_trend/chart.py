"""Original Series with Trend - Time series decomposition start"""
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
MLGREEN = '#2CA02C'

np.random.seed(42)
n = 365 * 2
dates = pd.date_range('2023-01-01', periods=n, freq='D')
trend = np.linspace(100, 180, n)
seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.randn(n) * 3
series = trend + seasonal + noise

df = pd.DataFrame({'Value': series}, index=dates)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df.index, df['Value'], color=MLBLUE, linewidth=1, alpha=0.7, label='Original')
ax.plot(df.index, trend, color=MLGREEN, linewidth=2.5, label='Trend', linestyle='--')

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Value', fontsize=10)
ax.set_title('Original Series = Trend + Seasonality + Noise', fontsize=12,
             fontweight='bold', color=MLPURPLE)
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
