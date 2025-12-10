"""Bias-Variance Tradeoff"""
import matplotlib.pyplot as plt
import numpy as np
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

fig, ax = plt.subplots(figsize=(10, 6))


complexity = np.linspace(0, 10, 100)
bias_squared = 5 * np.exp(-0.5 * complexity)
variance = 0.5 * np.exp(0.3 * complexity)
total_error = bias_squared + variance + 0.5

ax.plot(complexity, bias_squared, color=MLBLUE, linewidth=2.5, label='Bias squared')
ax.plot(complexity, variance, color=MLORANGE, linewidth=2.5, label='Variance')
ax.plot(complexity, total_error, color=MLRED, linewidth=3, linestyle='--', label='Total Error')

optimal_idx = np.argmin(total_error)
ax.scatter([complexity[optimal_idx]], [total_error[optimal_idx]], c=MLGREEN, s=150,
           zorder=5, edgecolors='black', label='Optimal complexity')
ax.axvline(complexity[optimal_idx], color='gray', linestyle=':', linewidth=1.5)

ax.set_title('Bias-Variance Tradeoff', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Model Complexity', fontsize=10)
ax.set_ylabel('Error', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax.text(1, 4.5, 'Underfitting', fontsize=11, color=MLBLUE, fontweight='bold')
ax.text(7, 4.5, 'Overfitting', fontsize=11, color=MLORANGE, fontweight='bold')


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
