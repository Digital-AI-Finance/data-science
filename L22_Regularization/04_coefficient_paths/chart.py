"""Coefficient Paths - How coefficients change with lambda"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Regularization Paths: Coefficients vs Lambda', fontsize=14, fontweight='bold', color=MLPURPLE)

# Simulated coefficient paths
lambdas = np.logspace(-3, 3, 100)

# Feature coefficients at lambda=0 (OLS)
ols_coefs = {
    'Market': 1.5,
    'Size': 0.8,
    'Value': 0.6,
    'Momentum': -0.4,
    'Quality': 0.3,
    'Noise': 0.1
}

colors = [MLBLUE, MLGREEN, MLORANGE, MLRED, MLPURPLE, 'gray']

# Plot 1: Ridge coefficient path
ax1 = axes[0, 0]

for (name, ols_val), color in zip(ols_coefs.items(), colors):
    # Ridge: coefficients shrink smoothly
    ridge_path = ols_val / (1 + lambdas * 0.5)
    ax1.plot(lambdas, ridge_path, color=color, linewidth=2, label=name)

ax1.axhline(0, color='gray', linewidth=1, linestyle='--')
ax1.set_xscale('log')
ax1.set_title('Ridge Path: Smooth Shrinkage', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel(r'$\lambda$ (log scale)', fontsize=10)
ax1.set_ylabel('Coefficient Value', fontsize=10)
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(alpha=0.3)

# Add annotation
ax1.annotate('All coefs shrink\nbut never reach 0', xy=(100, 0.1), xytext=(10, 0.5),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

# Plot 2: Lasso coefficient path
ax2 = axes[0, 1]

for (name, ols_val), color in zip(ols_coefs.items(), colors):
    # Lasso: coefficients hit 0 at different lambda values
    threshold = abs(ols_val) * 2  # When coefficient hits 0
    lasso_path = np.sign(ols_val) * np.maximum(0, abs(ols_val) - lambdas * 0.15)
    ax2.plot(lambdas, lasso_path, color=color, linewidth=2, label=name)

ax2.axhline(0, color='gray', linewidth=1, linestyle='--')
ax2.set_xscale('log')
ax2.set_title('Lasso Path: Sparse Selection', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel(r'$\lambda$ (log scale)', fontsize=10)
ax2.set_ylabel('Coefficient Value', fontsize=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.3)

# Add annotation
ax2.annotate('Coefficients hit 0\nat different lambdas', xy=(5, 0), xytext=(0.5, 0.8),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

# Plot 3: Number of non-zero coefficients (Lasso)
ax3 = axes[1, 0]

# Count non-zero coefficients at each lambda
n_nonzero = []
for lam in lambdas:
    count = sum(1 for ols_val in ols_coefs.values()
                if abs(ols_val) - lam * 0.15 > 0)
    n_nonzero.append(count)

ax3.plot(lambdas, n_nonzero, color=MLORANGE, linewidth=3)
ax3.fill_between(lambdas, 0, n_nonzero, color=MLORANGE, alpha=0.3)

ax3.set_xscale('log')
ax3.set_title('Lasso: Number of Selected Features', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel(r'$\lambda$ (log scale)', fontsize=10)
ax3.set_ylabel('Number of Non-Zero Coefficients', fontsize=10)
ax3.set_ylim(0, len(ols_coefs) + 0.5)
ax3.grid(alpha=0.3)

# Add markers for key lambda values
for lambda_val, n_feat in [(0.001, 6), (1, 4), (10, 2)]:
    ax3.scatter([lambda_val], [sum(1 for ols_val in ols_coefs.values()
                                    if abs(ols_val) - lambda_val * 0.15 > 0)],
                c=MLBLUE, s=100, zorder=5, edgecolors='black')

# Plot 4: Ridge vs Lasso comparison
ax4 = axes[1, 1]
ax4.axis('off')

comparison = '''
RIDGE vs LASSO COEFFICIENT PATHS

RIDGE (L2):
- Coefficients shrink smoothly toward 0
- Never exactly reach 0
- All features stay in model
- Good when: all features are relevant

LASSO (L1):
- Coefficients hit 0 at different lambda values
- Automatic feature selection
- Sparse solutions
- Good when: some features are irrelevant

CHOOSING LAMBDA:
- Small lambda: More features, potential overfit
- Large lambda: Fewer features, potential underfit
- Use cross-validation to find optimal lambda

ELASTIC NET (combines both):
- Mix of L1 and L2 penalties
- sklearn: ElasticNet(alpha=1.0, l1_ratio=0.5)
- l1_ratio: 0 = Ridge, 1 = Lasso, 0.5 = balanced

In Finance:
- Ridge: Factor models (keep all factors)
- Lasso: Variable selection (find key factors)
'''

ax4.text(0.02, 0.98, comparison, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Ridge vs Lasso Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
