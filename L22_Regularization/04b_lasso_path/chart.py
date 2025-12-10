"""Lasso Coefficient Path"""
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


lambdas = np.logspace(-3, 3, 100)
ols_coefs = {'Market': 1.5, 'Size': 0.8, 'Value': 0.6, 'Momentum': -0.4, 'Quality': 0.3, 'Noise': 0.1}
colors = [MLBLUE, MLGREEN, MLORANGE, MLRED, MLPURPLE, 'gray']

for (name, ols_val), color in zip(ols_coefs.items(), colors):
    lasso_path = np.sign(ols_val) * np.maximum(0, abs(ols_val) - lambdas * 0.15)
    ax.plot(lambdas, lasso_path, color=color, linewidth=2, label=name)

ax.axhline(0, color='gray', linewidth=1, linestyle='--')
ax.set_xscale('log')
ax.set_title('Lasso Path: Sparse Selection', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel(r'$\lambda$ (log scale)', fontsize=10)
ax.set_ylabel('Coefficient Value', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.annotate('Coefficients hit 0\nat different lambdas', xy=(5, 0), xytext=(0.5, 0.8),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
