"""Density Histogram - Normalized with PDF overlay"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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

fig, ax = plt.subplots(figsize=(10, 6))

data = np.random.normal(0, 2, 1000)

ax.hist(data, bins=40, density=True, color=MLGREEN, alpha=0.5, edgecolor='black', label='Data')

# Overlay fitted normal PDF
x = np.linspace(data.min(), data.max(), 100)
pdf = stats.norm.pdf(x, np.mean(data), np.std(data))
ax.plot(x, pdf, color=MLRED, linewidth=2.5, label='Normal PDF')

ax.set_title('Density Histogram with PDF', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Value', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
