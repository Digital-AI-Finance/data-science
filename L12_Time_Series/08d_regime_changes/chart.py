"""Regime Changes - Different market conditions"""
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
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

regime1 = np.random.randn(80) * 1 + 0.05
regime2 = np.random.randn(60) * 3 - 0.1
regime3 = np.random.randn(112) * 1.5 + 0.02

returns_regime = np.concatenate([regime1, regime2, regime3])
prices_regime = 100 * np.exp(np.cumsum(returns_regime / 100))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(prices_regime, color=MLBLUE, linewidth=1.5)

ax.axvspan(0, 80, alpha=0.2, color=MLGREEN, label='Bull (low vol)')
ax.axvspan(80, 140, alpha=0.2, color=MLRED, label='Bear (high vol)')
ax.axvspan(140, 252, alpha=0.2, color=MLORANGE, label='Recovery')

ax.axvline(80, color=MLRED, linewidth=2, linestyle='--')
ax.axvline(140, color=MLGREEN, linewidth=2, linestyle='--')

ax.set_xlabel('Trading Day', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title('Regime Changes: Different market conditions', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
