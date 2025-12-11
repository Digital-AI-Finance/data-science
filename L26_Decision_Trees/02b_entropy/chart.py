"""Entropy"""
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


p = np.linspace(0.001, 0.999, 100)
entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)

ax.plot(p, entropy, color=MLORANGE, linewidth=3, label='Entropy = -p*log(p) - (1-p)*log(1-p)')
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5)
ax.scatter([0.5], [1.0], c=MLRED, s=100, zorder=5, label='Max entropy at p=0.5')

ax.set_title('Entropy: Information-Based Impurity', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Proportion of Class 1 (p)', fontsize=10)
ax.set_ylabel('Entropy (bits)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.1)


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
