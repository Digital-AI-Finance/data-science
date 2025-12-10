"""Basic Histogram - Distribution with mean and median"""
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
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))

data = np.random.normal(100, 15, 1000)

ax.hist(data, bins=30, color=MLBLUE, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(data), color=MLRED, linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.1f}')
ax.axvline(np.median(data), color=MLGREEN, linestyle=':', linewidth=2, label=f'Median: {np.median(data):.1f}')

ax.set_title('Basic Histogram', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Value', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
