"""Feature Selection - Using Lasso for sparse models"""
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
fig.suptitle('Feature Selection with Lasso', fontsize=14, fontweight='bold', color=MLPURPLE)

# Define features (some relevant, some noise)
all_features = ['Market', 'Size', 'Value', 'Mom', 'Quality', 'Vol',
                'Noise1', 'Noise2', 'Noise3', 'Noise4', 'Noise5']
n_features = len(all_features)

# True relevant features
relevant = [True, True, True, True, True, True, False, False, False, False, False]

# OLS coefficients (keeps all, including noise)
ols_coefs = np.array([1.5, 0.8, 0.6, -0.4, 0.3, 0.2, 0.12, -0.08, 0.05, -0.03, 0.02])

# Lasso coefficients (selects only relevant)
lasso_coefs = np.array([1.3, 0.6, 0.4, -0.3, 0.2, 0.1, 0, 0, 0, 0, 0])

# Plot 1: OLS vs Lasso coefficients
ax1 = axes[0, 0]

x = np.arange(n_features)
width = 0.35

colors_ols = [MLBLUE if rel else 'gray' for rel in relevant]
colors_lasso = [MLGREEN if coef != 0 else MLRED for coef in lasso_coefs]

bars1 = ax1.bar(x - width/2, ols_coefs, width, color=colors_ols, edgecolor='black',
                linewidth=0.5, label='OLS (all features)')
bars2 = ax1.bar(x + width/2, lasso_coefs, width, color=colors_lasso, edgecolor='black',
                linewidth=0.5, label='Lasso (selected)')

ax1.axhline(0, color='gray', linewidth=1)
ax1.set_xticks(x)
ax1.set_xticklabels(all_features, fontsize=8, rotation=45, ha='right')
ax1.set_title('OLS Keeps Noise, Lasso Removes It', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Coefficient', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, axis='y')

# Add "Noise Features" bracket
ax1.annotate('', xy=(6, -0.15), xytext=(10, -0.15),
             arrowprops=dict(arrowstyle='|-|', color='gray'))
ax1.text(8, -0.22, 'Noise Features', ha='center', fontsize=9, color='gray')

# Plot 2: Feature importance ranking
ax2 = axes[0, 1]

# Sort by absolute Lasso coefficient
sort_idx = np.argsort(np.abs(lasso_coefs))[::-1]
sorted_features = [all_features[i] for i in sort_idx]
sorted_coefs = lasso_coefs[sort_idx]

colors = [MLGREEN if c != 0 else MLRED for c in sorted_coefs]
bars = ax2.barh(sorted_features, np.abs(sorted_coefs), color=colors, edgecolor='black', linewidth=0.5)

ax2.axvline(0, color='gray', linewidth=1)
ax2.set_title('Feature Importance (|Lasso Coefficient|)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('|Coefficient|', fontsize=10)
ax2.invert_yaxis()

# Mark selected vs not selected
ax2.text(1.35, 5.5, f'Selected: {sum(lasso_coefs != 0)}', fontsize=10, color=MLGREEN, fontweight='bold')
ax2.text(1.35, 6.5, f'Removed: {sum(lasso_coefs == 0)}', fontsize=10, color=MLRED, fontweight='bold')

# Plot 3: Selection path - which features drop off
ax3 = axes[1, 0]

lambdas = np.logspace(-3, 1, 50)

for i, (feat, ols_coef, rel) in enumerate(zip(all_features[:6], ols_coefs[:6], relevant[:6])):
    # Relevant features stay longer
    threshold = 0.5 if rel else 0.1
    path = np.sign(ols_coef) * np.maximum(0, abs(ols_coef) - lambdas * threshold)
    color = MLBLUE if rel else 'gray'
    ax3.plot(lambdas, path, color=color, linewidth=2 if rel else 1, label=feat)

ax3.axhline(0, color='gray', linewidth=1, linestyle='--')
ax3.set_xscale('log')
ax3.set_title('Lasso Path: Relevant Features Persist', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel(r'$\lambda$ (log scale)', fontsize=10)
ax3.set_ylabel('Coefficient', fontsize=10)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(alpha=0.3)

# Plot 4: Summary and code
ax4 = axes[1, 1]
ax4.axis('off')

summary = '''
LASSO FOR FEATURE SELECTION

Why use Lasso for feature selection?
- Automatically identifies relevant features
- Removes noise without manual filtering
- Produces interpretable sparse models

Process:
1. Fit Lasso with cross-validated lambda
2. Features with coef = 0 are removed
3. Features with coef != 0 are selected

Code:
from sklearn.linear_model import LassoCV

# Fit with automatic lambda selection
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

# Get selected features
selected = X.columns[lasso.coef_ != 0]
print(f"Selected features: {list(selected)}")

# Feature importance
importance = pd.Series(
    np.abs(lasso.coef_),
    index=X.columns
).sort_values(ascending=False)

Finance Application:
- Start with many potential factors
- Lasso selects truly predictive ones
- Build parsimonious factor model
'''

ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Feature Selection Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
