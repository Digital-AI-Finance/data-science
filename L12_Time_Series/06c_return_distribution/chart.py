"""Return Distribution - Histogram with normal fit"""
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
MLRED = '#D62728'

np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.015))
df = pd.DataFrame({'Price': prices}, index=dates)
returns = df['Price'].pct_change().dropna() * 100

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(returns, bins=30, density=True, alpha=0.7, color=MLBLUE, edgecolor='black')

x = np.linspace(returns.min(), returns.max(), 100)
mu, std = returns.mean(), returns.std()
normal = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/std)**2)
ax.plot(x, normal, color=MLRED, linewidth=2, label='Normal fit')

val_5 = np.percentile(returns, 5)
ax.axvline(val_5, color=MLRED, linestyle='--', linewidth=2, label=f'VaR 95%: {val_5:.2f}%')
ax.axvline(returns.median(), color=MLPURPLE, linestyle='--', linewidth=2, label=f'Median: {returns.median():.2f}%')

ax.set_xlabel('Daily Return (%)', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Return Distribution', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
