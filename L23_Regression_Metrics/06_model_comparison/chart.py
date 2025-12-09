"""Model Comparison - Comparing multiple models"""
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
fig.suptitle('Model Comparison: Choosing the Best Model', fontsize=14, fontweight='bold', color=MLPURPLE)

# Model comparison data
models = ['OLS', 'Ridge', 'Lasso', 'ElasticNet']
train_rmse = [2.15, 2.25, 2.30, 2.28]
test_rmse = [2.85, 2.45, 2.48, 2.42]
train_r2 = [0.72, 0.70, 0.68, 0.69]
test_r2 = [0.58, 0.67, 0.66, 0.68]
n_features = [10, 10, 6, 8]

# Plot 1: Train vs Test RMSE
ax1 = axes[0, 0]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, train_rmse, width, label='Train RMSE', color=MLBLUE, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, test_rmse, width, label='Test RMSE', color=MLORANGE, edgecolor='black', linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_title('Train vs Test RMSE (Lower is Better)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('RMSE', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3, axis='y')

# Highlight best test RMSE
best_idx = np.argmin(test_rmse)
ax1.annotate('Best', xy=(best_idx + width/2, test_rmse[best_idx]),
             xytext=(best_idx + 0.5, test_rmse[best_idx] - 0.15), fontsize=10, color=MLGREEN, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

# Plot 2: Overfitting analysis (gap between train and test)
ax2 = axes[0, 1]

gaps = [tr - te for tr, te in zip(test_rmse, train_rmse)]
colors = [MLRED if g > 0.4 else MLGREEN for g in gaps]

bars = ax2.bar(models, gaps, color=colors, edgecolor='black', linewidth=0.5)
ax2.axhline(0.3, color='gray', linestyle='--', linewidth=1.5, label='Acceptable gap')

ax2.set_title('Overfitting: Test-Train Gap', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('RMSE Gap (Test - Train)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis='y')

# Add gap values
for bar, gap in zip(bars, gaps):
    ax2.text(bar.get_x() + bar.get_width()/2, gap + 0.02, f'{gap:.2f}',
             ha='center', fontsize=10, fontweight='bold')

# Plot 3: R-squared comparison
ax3 = axes[1, 0]

x = np.arange(len(models))
width = 0.35

bars1 = ax3.bar(x - width/2, train_r2, width, label='Train R-sq', color=MLBLUE, edgecolor='black', linewidth=0.5)
bars2 = ax3.bar(x + width/2, test_r2, width, label='Test R-sq', color=MLORANGE, edgecolor='black', linewidth=0.5)

ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.set_title('Train vs Test R-squared (Higher is Better)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('R-squared', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')
ax3.set_ylim(0.5, 0.8)

# Highlight best test R-squared
best_idx = np.argmax(test_r2)
ax3.annotate('Best', xy=(best_idx + width/2, test_r2[best_idx]),
             xytext=(best_idx + 0.5, test_r2[best_idx] + 0.03), fontsize=10, color=MLGREEN, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

# Create summary table
summary = '''
MODEL COMPARISON SUMMARY

Model        Test RMSE  Test R-sq  Features  Verdict
-------------------------------------------------------
OLS          2.85       0.58       10        Overfits
Ridge        2.45       0.67       10        Good
Lasso        2.48       0.66       6         Good (sparse)
ElasticNet   2.42       0.68       8         Best

SELECTION CRITERIA:

1. PRIMARY: Test set performance (not train!)
   - Lower RMSE / Higher R-sq on test data

2. SECONDARY: Generalization gap
   - Small gap = model generalizes well
   - Large gap = overfitting

3. TERTIARY: Interpretability
   - Fewer features = more interpretable
   - Lasso/ElasticNet for feature selection

WINNER: ElasticNet
- Best test R-sq (0.68)
- Best test RMSE (2.42)
- Reasonable # features (8)
'''

ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Selection Summary', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
