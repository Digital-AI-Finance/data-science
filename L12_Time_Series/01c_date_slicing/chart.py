"""Date Slicing - Subsetting time series by date"""
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

# Full series in light color
ax.plot(df.index, df['Price'], color='gray', alpha=0.3, linewidth=1)
# Highlight 2024
df_2024 = df.loc['2024']
ax.plot(df_2024.index, df_2024['Price'], color=MLGREEN, linewidth=2, label='2024 data')
# Highlight specific month
df_jul = df.loc['2023-07']
ax.plot(df_jul.index, df_jul['Price'], color=MLRED, linewidth=3, label='July 2023')

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title("Date Slicing: df['2024'] or df['2023-07']", fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
