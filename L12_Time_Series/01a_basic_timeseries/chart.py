"""Basic Time Series - Stock price with trend and seasonality"""
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

ax.plot(df.index, df['Price'], color=MLBLUE, linewidth=1.5)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title('Stock Price Time Series', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3)

ax.annotate('Trend', xy=(dates[300], prices[300]), xytext=(dates[200], prices[300]+30),
            arrowprops=dict(arrowstyle='->', color=MLGREEN), fontsize=10, color=MLGREEN)
ax.annotate('Seasonality', xy=(dates[500], prices[500]), xytext=(dates[400], prices[500]-20),
            arrowprops=dict(arrowstyle='->', color=MLORANGE), fontsize=10, color=MLORANGE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
