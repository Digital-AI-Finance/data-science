"""Lambda Tuning - Finding optimal regularization strength"""
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
fig.suptitle('Tuning Lambda: Finding Optimal Regularization', fontsize=14, fontweight='bold', color=MLPURPLE)

# Simulated cross-validation results
lambdas = np.logspace(-4, 2, 50)

# Train error decreases with lower lambda (more overfit)
train_error = 0.1 + 0.05 * np.log10(lambdas + 0.001) + 0.3 * np.exp(-lambdas * 10)

# Test error is U-shaped
test_error = 0.15 + 0.02 * (np.log10(lambdas) - 0)**2 + 0.05 * np.exp(-lambdas * 10) + np.random.normal(0, 0.02, len(lambdas))
test_error = np.clip(test_error, 0.1, 0.6)

# Plot 1: Train vs Test error
ax1 = axes[0, 0]

ax1.plot(lambdas, train_error, color=MLBLUE, linewidth=2.5, label='Training Error')
ax1.plot(lambdas, test_error, color=MLRED, linewidth=2.5, label='Test Error')

# Mark optimal lambda
optimal_idx = np.argmin(test_error)
optimal_lambda = lambdas[optimal_idx]

ax1.scatter([optimal_lambda], [test_error[optimal_idx]], c=MLGREEN, s=200, zorder=5,
            marker='*', edgecolors='black', label=f'Optimal: lambda={optimal_lambda:.3f}')
ax1.axvline(optimal_lambda, color='gray', linestyle='--', linewidth=1.5)

ax1.set_xscale('log')
ax1.set_title('Train vs Test Error', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel(r'$\lambda$ (log scale)', fontsize=10)
ax1.set_ylabel('Mean Squared Error', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Add region labels
ax1.text(1e-4, 0.4, 'Overfit\n(low lambda)', fontsize=9, color=MLRED)
ax1.text(10, 0.4, 'Underfit\n(high lambda)', fontsize=9, color=MLBLUE)

# Plot 2: Cross-validation scores with confidence interval
ax2 = axes[0, 1]

# Simulated CV mean and std
cv_mean = test_error
cv_std = 0.03 + 0.01 * np.abs(np.log10(lambdas))

ax2.plot(lambdas, cv_mean, color=MLBLUE, linewidth=2.5, label='CV Mean Error')
ax2.fill_between(lambdas, cv_mean - cv_std, cv_mean + cv_std, alpha=0.3, color=MLBLUE, label='CI (1 std)')

ax2.scatter([optimal_lambda], [cv_mean[optimal_idx]], c=MLGREEN, s=150, zorder=5,
            marker='*', edgecolors='black')
ax2.axvline(optimal_lambda, color='gray', linestyle='--', linewidth=1.5, label=f'Best: {optimal_lambda:.3f}')

ax2.set_xscale('log')
ax2.set_title('Cross-Validation Error with Confidence', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel(r'$\lambda$ (log scale)', fontsize=10)
ax2.set_ylabel('CV Error', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Plot 3: Grid search visualization
ax3 = axes[1, 0]

# Show grid of lambda values tested
lambda_grid = np.logspace(-3, 2, 20)
errors = 0.15 + 0.02 * (np.log10(lambda_grid))**2 + np.random.normal(0, 0.015, len(lambda_grid))

colors = [MLGREEN if e == errors.min() else MLBLUE for e in errors]
ax3.scatter(lambda_grid, errors, c=colors, s=100, edgecolors='black', zorder=5)
ax3.plot(lambda_grid, errors, color='gray', linewidth=1, linestyle='-', alpha=0.5)

ax3.set_xscale('log')
ax3.set_title('Grid Search: Evaluate Each Lambda', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel(r'$\lambda$ (log scale)', fontsize=10)
ax3.set_ylabel('CV Error', fontsize=10)
ax3.grid(alpha=0.3)

# Add annotation
best_lambda = lambda_grid[np.argmin(errors)]
ax3.annotate(f'Best: {best_lambda:.4f}', xy=(best_lambda, errors.min()),
             xytext=(best_lambda * 5, errors.min() + 0.05), fontsize=10,
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

# Plot 4: sklearn code for lambda tuning
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Lambda Tuning with sklearn

# Method 1: RidgeCV / LassoCV (auto tune)
from sklearn.linear_model import RidgeCV, LassoCV

# Define lambda values to try
alphas = np.logspace(-4, 2, 50)

# Ridge with built-in CV
model = RidgeCV(alphas=alphas, cv=5)
model.fit(X_train, y_train)
print(f"Best alpha: {model.alpha_}")

# Lasso with built-in CV
model = LassoCV(alphas=alphas, cv=5)
model.fit(X_train, y_train)
print(f"Best alpha: {model.alpha_}")

# Method 2: GridSearchCV (more control)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {'alpha': alphas}
grid = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid.fit(X_train, y_train)

print(f"Best alpha: {grid.best_params_['alpha']}")
print(f"Best score: {-grid.best_score_:.4f}")
'''

ax4.text(0.02, 0.95, code, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
