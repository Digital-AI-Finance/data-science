"""Ridge Regression Concept - L2 regularization"""
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
fig.suptitle('Ridge Regression (L2 Regularization)', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Ridge formula
ax1 = axes[0, 0]
ax1.axis('off')

formula = r'''
RIDGE REGRESSION

Ordinary Least Squares (OLS):
$$\min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Ridge Regression adds L2 penalty:
$$\min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

Where:
- First term: Fit the data (minimize errors)
- Second term: Keep coefficients small
- Lambda: Regularization strength
  - Lambda = 0: Same as OLS
  - Lambda large: Shrinks coefficients toward 0

Key Property:
Ridge SHRINKS coefficients but never sets them exactly to 0
'''

ax1.text(0.05, 0.95, formula, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('The Ridge Formula', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: L2 penalty visualization
ax2 = axes[0, 1]

# Create circle (L2 constraint)
theta = np.linspace(0, 2*np.pi, 100)
r = 2
ax2.plot(r * np.cos(theta), r * np.sin(theta), color=MLBLUE, linewidth=2.5, label='L2 constraint')

# Create contours (OLS loss)
b1 = np.linspace(-4, 4, 100)
b2 = np.linspace(-4, 4, 100)
B1, B2 = np.meshgrid(b1, b2)

# Simulated OLS contours (ellipses centered at optimal)
ols_optimal = (3, 2.5)
loss = 0.5 * (B1 - ols_optimal[0])**2 + 0.8 * (B2 - ols_optimal[1])**2
ax2.contour(B1, B2, loss, levels=[1, 2, 4, 8, 16], colors='gray', alpha=0.5, linestyles='--')

# Mark OLS optimal
ax2.scatter([ols_optimal[0]], [ols_optimal[1]], c=MLRED, s=150, zorder=5,
            edgecolors='black', label='OLS solution')

# Mark Ridge optimal (on constraint circle)
ridge_angle = np.arctan2(ols_optimal[1], ols_optimal[0])
ridge_optimal = (r * np.cos(ridge_angle), r * np.sin(ridge_angle))
ax2.scatter([ridge_optimal[0]], [ridge_optimal[1]], c=MLGREEN, s=150, zorder=5,
            marker='*', edgecolors='black', label='Ridge solution')

ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axvline(0, color='gray', linewidth=0.5)

ax2.set_title('L2 Constraint: Circle', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel(r'$\beta_1$', fontsize=10)
ax2.set_ylabel(r'$\beta_2$', fontsize=10)
ax2.legend(fontsize=8, loc='upper left')
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.set_aspect('equal')
ax2.grid(alpha=0.3)

# Plot 3: Effect of lambda
ax3 = axes[1, 0]

# Simulated coefficients for different lambda values
lambdas = [0, 0.1, 1, 10, 100]
features = ['Market', 'Size', 'Value', 'Momentum', 'Quality']
n_features = len(features)

# OLS coefficients (lambda=0)
ols_coefs = np.array([1.5, 0.8, 0.6, -0.4, 0.3])

x_pos = np.arange(n_features)
width = 0.15

for i, lam in enumerate(lambdas):
    # Ridge shrinks toward 0, more shrinkage with higher lambda
    ridge_coefs = ols_coefs / (1 + lam * 0.3)
    offset = (i - 2) * width
    color = plt.cm.viridis(i / len(lambdas))
    ax3.bar(x_pos + offset, ridge_coefs, width, label=f'lambda={lam}', color=color, edgecolor='black', linewidth=0.5)

ax3.axhline(0, color='gray', linewidth=1)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(features, fontsize=9)
ax3.set_title('Ridge: Coefficients Shrink with Lambda', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Coefficient Value', fontsize=10)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(alpha=0.3, axis='y')

# Plot 4: sklearn code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
sklearn Ridge Regression

from sklearn.linear_model import Ridge

# Create Ridge model with regularization
model = Ridge(alpha=1.0)  # alpha = lambda

# Fit to data
model.fit(X_train, y_train)

# Coefficients are shrunk
print(model.coef_)
print(model.intercept_)

# Predictions
y_pred = model.predict(X_test)

# Choose lambda via cross-validation
from sklearn.linear_model import RidgeCV

model_cv = RidgeCV(
    alphas=[0.1, 1.0, 10.0],
    cv=5
)
model_cv.fit(X_train, y_train)
print(f"Best alpha: {model_cv.alpha_}")
'''

ax4.text(0.02, 0.95, code, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
