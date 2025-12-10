"""Volatility Clustering - High vol follows high vol"""
import matplotlib.pyplot as plt
import numpy as np
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
n = 252

returns = np.zeros(n)
vol_series = np.zeros(n)
vol_series[0] = 0.01
alpha, beta = 0.1, 0.85

for i in range(1, n):
    vol_series[i] = np.sqrt(0.00001 + alpha * returns[i-1]**2 + beta * vol_series[i-1]**2)
    returns[i] = vol_series[i] * np.random.randn()

fig, ax = plt.subplots(figsize=(10, 6))

colors = [MLGREEN if r >= 0 else MLRED for r in returns]
ax.bar(range(n), returns * 100, color=colors, alpha=0.7, width=1)
ax.plot(vol_series * 100 * 2, color=MLPURPLE, linewidth=2, label='Volatility envelope')
ax.plot(-vol_series * 100 * 2, color=MLPURPLE, linewidth=2)

ax.set_xlabel('Trading Day', fontsize=10)
ax.set_ylabel('Return (%)', fontsize=10)
ax.set_title('Volatility Clustering: High vol follows high vol', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
