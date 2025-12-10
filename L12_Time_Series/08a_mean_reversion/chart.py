"""Mean Reversion - Price returns to average"""
import matplotlib.pyplot as plt
import numpy as np
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
n = 252
mean_level = 100
theta = 0.1
sigma = 2
prices_mr = [mean_level]

for _ in range(n-1):
    dp = theta * (mean_level - prices_mr[-1]) + sigma * np.random.randn()
    prices_mr.append(prices_mr[-1] + dp)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(prices_mr, color=MLBLUE, linewidth=1.5)
ax.axhline(mean_level, color=MLRED, linestyle='--', linewidth=2, label=f'Mean = {mean_level}')
ax.fill_between(range(n), mean_level - 5, mean_level + 5, alpha=0.2, color=MLRED)

ax.set_xlabel('Trading Day', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title('Mean Reversion: Price returns to average', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
