"""Monthly Aggregation - Resampling time series data"""
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
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
n = len(dates)

trend = np.linspace(100, 150, n)
seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, n))
noise = np.cumsum(np.random.randn(n) * 0.5)
prices = trend + seasonal + noise

df = pd.DataFrame({'Date': dates, 'Price': prices})
df.set_index('Date', inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))

monthly = df.resample('ME').mean()
ax.bar(monthly.index, monthly['Price'], width=20, color=MLPURPLE, alpha=0.7, edgecolor='black')
ax.set_xlabel('Month', fontsize=10)
ax.set_ylabel('Avg Price ($)', fontsize=10)
ax.set_title('Monthly Aggregation: df.resample("M").mean()', fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
