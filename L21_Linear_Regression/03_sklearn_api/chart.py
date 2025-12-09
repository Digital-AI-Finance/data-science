"""sklearn API - Linear regression with scikit-learn"""
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
fig.suptitle('sklearn LinearRegression: The API', fontsize=14, fontweight='bold', color=MLPURPLE)

# Generate data
n = 50
X = np.random.uniform(5, 30, n).reshape(-1, 1)
y = 3 + 0.5 * X.flatten() + np.random.normal(0, 2, n)

# Plot 1: Code example
ax1 = axes[0, 0]
ax1.axis('off')

code = '''
# sklearn Linear Regression

from sklearn.linear_model import LinearRegression

# Step 1: Create the model
model = LinearRegression()

# Step 2: Fit to data
# X must be 2D: (n_samples, n_features)
X = risk.values.reshape(-1, 1)
y = returns.values
model.fit(X, y)

# Step 3: Access coefficients
slope = model.coef_[0]      # 0.52
intercept = model.intercept_ # 2.87

# Step 4: Make predictions
y_pred = model.predict(X)

# Step 5: Evaluate
r_squared = model.score(X, y)  # 0.76
'''

ax1.text(0.02, 0.98, code, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('sklearn Code Pattern', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Fit visualization
ax2 = axes[0, 1]

# Simulate sklearn fit
slope = np.sum((X.flatten() - X.mean()) * (y - y.mean())) / np.sum((X.flatten() - X.mean())**2)
intercept = y.mean() - slope * X.mean()
y_pred = intercept + slope * X.flatten()

ax2.scatter(X, y, c=MLBLUE, s=60, alpha=0.7, edgecolors='black')
ax2.plot(np.sort(X.flatten()), intercept + slope * np.sort(X.flatten()),
         color=MLGREEN, linewidth=2.5, label='model.fit(X, y)')

ax2.set_title('After model.fit()', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Risk (%)', fontsize=10)
ax2.set_ylabel('Return (%)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Add coefficient box
coef_text = f'model.coef_[0] = {slope:.3f}\nmodel.intercept_ = {intercept:.3f}'
ax2.text(0.95, 0.05, coef_text, transform=ax2.transAxes, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=MLGREEN))

# Plot 3: Prediction
ax3 = axes[1, 0]

# New data for prediction
X_new = np.array([10, 15, 20, 25, 30]).reshape(-1, 1)
y_new_pred = intercept + slope * X_new.flatten()

ax3.scatter(X, y, c=MLBLUE, s=50, alpha=0.5, edgecolors='black', label='Training data')
ax3.plot(np.sort(X.flatten()), intercept + slope * np.sort(X.flatten()),
         color=MLGREEN, linewidth=2)

# Plot predictions
ax3.scatter(X_new, y_new_pred, c=MLORANGE, s=120, marker='*',
            edgecolors='black', zorder=5, label='model.predict(X_new)')

for xi, yi in zip(X_new.flatten(), y_new_pred):
    ax3.annotate(f'{yi:.1f}', xy=(xi, yi), xytext=(5, 5),
                 textcoords='offset points', fontsize=9, color=MLORANGE)

ax3.set_title('model.predict(X_new)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Risk (%)', fontsize=10)
ax3.set_ylabel('Return (%)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Model evaluation
ax4 = axes[1, 1]
ax4.axis('off')

r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

evaluation = f'''
Model Evaluation Methods

# R-squared (coefficient of determination)
r2 = model.score(X, y)
# Result: {r_squared:.4f}

Interpretation:
- R-squared = {r_squared:.2%} of variance explained
- Range: 0 (no fit) to 1 (perfect fit)

# Residuals analysis
residuals = y - model.predict(X)

# Mean Squared Error (manual)
mse = np.mean(residuals ** 2)
# Result: {np.mean((y - y_pred)**2):.3f}

# RMSE (in same units as y)
rmse = np.sqrt(mse)
# Result: {np.sqrt(np.mean((y - y_pred)**2)):.3f}

# Or use sklearn.metrics
from sklearn.metrics import mean_squared_error, r2_score
'''

ax4.text(0.02, 0.95, evaluation, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('model.score() and Metrics', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
