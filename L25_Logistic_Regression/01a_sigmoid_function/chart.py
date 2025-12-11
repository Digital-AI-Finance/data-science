"""Sigmoid Function"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))


z = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-z))

ax.plot(z, sigmoid, color=MLBLUE, linewidth=3, label='Sigmoid: $\\sigma(z) = 1/(1+e^{-z})$')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision threshold')
ax.axvline(0, color='gray', linestyle=':', linewidth=1)

ax.fill_between(z[z < 0], sigmoid[z < 0], alpha=0.3, color=MLRED, label='Predict 0')
ax.fill_between(z[z >= 0], sigmoid[z >= 0], alpha=0.3, color=MLGREEN, label='Predict 1')

ax.set_title('Sigmoid Function: Maps Any Value to (0, 1)', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('z = $\\beta_0 + \\beta_1 x$', fontsize=10)
ax.set_ylabel('P(y=1)', fontsize=10)
ax.legend(fontsize=9, loc='right')
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
