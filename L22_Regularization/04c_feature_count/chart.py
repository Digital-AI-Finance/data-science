"""Lasso Feature Selection"""
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

n_nonzero = []
for lam in lambdas:
    count = sum(1 for ols_val in ols_coefs.values() if abs(ols_val) - lam * 0.15 > 0)
    n_nonzero.append(count)

ax.plot(lambdas, n_nonzero, color=MLORANGE, linewidth=3)
ax.fill_between(lambdas, 0, n_nonzero, color=MLORANGE, alpha=0.3)

ax.set_xscale('log')
ax.set_title('Lasso: Number of Selected Features', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel(r'$\lambda$ (log scale)', fontsize=10)
ax.set_ylabel('Number of Non-Zero Coefficients', fontsize=10)
ax.set_ylim(0, len(ols_coefs) + 0.5)
ax.grid(alpha=0.3)

for lambda_val in [0.001, 1, 10]:
    n = sum(1 for ols_val in ols_coefs.values() if abs(ols_val) - lambda_val * 0.15 > 0)
    ax.scatter([lambda_val], [n], c=MLBLUE, s=100, zorder=5, edgecolors='black')


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
