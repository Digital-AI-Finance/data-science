"""Lasso Regression Concept - L1 regularization"""
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
fig.suptitle('Lasso Regression (L1 Regularization)', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Lasso formula
ax1 = axes[0, 0]
ax1.axis('off')

formula = r'''
LASSO REGRESSION
(Least Absolute Shrinkage and Selection Operator)

Lasso adds L1 penalty:
$$\min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

Where:
- First term: Fit the data
- Second term: Sum of ABSOLUTE values
- Lambda: Regularization strength

Key Property:
Lasso can set coefficients EXACTLY to 0
(automatic feature selection!)

When to use Lasso:
- Many features, some are irrelevant
- Want interpretable sparse models
- Feature selection is important
'''

ax1.text(0.05, 0.95, formula, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('The Lasso Formula', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: L1 penalty visualization (diamond)
ax2 = axes[0, 1]

# Create diamond (L1 constraint)
r = 2
diamond_x = [r, 0, -r, 0, r]
diamond_y = [0, r, 0, -r, 0]
ax2.plot(diamond_x, diamond_y, color=MLORANGE, linewidth=2.5, label='L1 constraint')
ax2.fill(diamond_x, diamond_y, color=MLORANGE, alpha=0.1)

# Create contours (OLS loss)
b1 = np.linspace(-4, 4, 100)
b2 = np.linspace(-4, 4, 100)
B1, B2 = np.meshgrid(b1, b2)

ols_optimal = (3, 0.5)
loss = 0.5 * (B1 - ols_optimal[0])**2 + 0.8 * (B2 - ols_optimal[1])**2
ax2.contour(B1, B2, loss, levels=[1, 2, 4, 8, 16], colors='gray', alpha=0.5, linestyles='--')

# Mark OLS optimal
ax2.scatter([ols_optimal[0]], [ols_optimal[1]], c=MLRED, s=150, zorder=5,
            edgecolors='black', label='OLS solution')

# Lasso solution hits corner (sparse!)
ax2.scatter([r], [0], c=MLGREEN, s=150, zorder=5, marker='*',
            edgecolors='black', label='Lasso solution')

ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axvline(0, color='gray', linewidth=0.5)

ax2.set_title('L1 Constraint: Diamond (Corners = Sparse)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel(r'$\beta_1$', fontsize=10)
ax2.set_ylabel(r'$\beta_2$', fontsize=10)
ax2.legend(fontsize=8, loc='upper left')
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.set_aspect('equal')
ax2.grid(alpha=0.3)

ax2.annotate('Corner = one coef is 0!', xy=(r, 0), xytext=(r+1, 1.5),
             fontsize=9, arrowprops=dict(arrowstyle='->', color=MLGREEN))

# Plot 3: Feature selection with Lasso
ax3 = axes[1, 0]

features = ['Market', 'Size', 'Value', 'Mom', 'Quality', 'Vol', 'Liq', 'Noise1', 'Noise2', 'Noise3']
n_features = len(features)

# OLS keeps all features
ols_coefs = np.array([1.5, 0.8, 0.6, -0.4, 0.3, 0.2, 0.15, 0.08, -0.05, 0.03])

# Lasso zeroes out noise
lasso_coefs = np.array([1.3, 0.6, 0.4, -0.3, 0.2, 0.1, 0, 0, 0, 0])

x_pos = np.arange(n_features)
width = 0.35

bars1 = ax3.bar(x_pos - width/2, ols_coefs, width, label='OLS (all features)', color=MLBLUE, edgecolor='black', linewidth=0.5)
bars2 = ax3.bar(x_pos + width/2, lasso_coefs, width, label='Lasso (sparse)', color=MLORANGE, edgecolor='black', linewidth=0.5)

ax3.axhline(0, color='gray', linewidth=1)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(features, fontsize=8, rotation=45, ha='right')
ax3.set_title('Lasso Zeros Out Irrelevant Features', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Coefficient Value', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')

# Highlight zeroed features
for i, (ols, lasso) in enumerate(zip(ols_coefs, lasso_coefs)):
    if lasso == 0:
        ax3.annotate('0', xy=(i + width/2, 0.02), fontsize=9, ha='center', color=MLRED, fontweight='bold')

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
sklearn Lasso Regression

from sklearn.linear_model import Lasso

# Create Lasso model
model = Lasso(alpha=1.0)  # alpha = lambda

# Fit to data
model.fit(X_train, y_train)

# Sparse coefficients (some are 0!)
print(model.coef_)
# [1.3, 0.6, 0.4, -0.3, 0.2, 0.1, 0, 0, 0, 0]

# Count non-zero features
n_selected = np.sum(model.coef_ != 0)
print(f"Features selected: {n_selected}")

# Cross-validation for best alpha
from sklearn.linear_model import LassoCV

model_cv = LassoCV(cv=5)
model_cv.fit(X_train, y_train)
print(f"Best alpha: {model_cv.alpha_}")
print(f"Non-zero coefs: {np.sum(model_cv.coef_ != 0)}")
'''

ax4.text(0.02, 0.95, code, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
