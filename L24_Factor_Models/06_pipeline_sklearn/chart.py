"""Complete sklearn Pipeline - Factor model workflow"""
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
fig.suptitle('Complete Factor Model Pipeline with sklearn', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Pipeline flowchart
ax1 = axes[0, 0]
ax1.axis('off')

pipeline = '''
FACTOR MODEL PIPELINE

1. DATA PREPARATION
   +------------------------+
   | Load factor data (FF)  |
   | Load stock returns     |
   | Merge on dates         |
   | Calculate excess ret   |
   +------------------------+
              |
              v
2. FEATURE ENGINEERING
   +------------------------+
   | Scale factors (opt.)   |
   | Handle missing data    |
   | Check collinearity     |
   +------------------------+
              |
              v
3. MODEL FITTING
   +------------------------+
   | Train/Test split       |
   | Fit LinearRegression   |
   | or Ridge/Lasso         |
   +------------------------+
              |
              v
4. EVALUATION
   +------------------------+
   | R-squared              |
   | Alpha significance     |
   | Residual analysis      |
   +------------------------+
              |
              v
5. INTERPRETATION
   +------------------------+
   | Factor exposures       |
   | Risk attribution       |
   | Performance decomp     |
   +------------------------+
'''

ax1.text(0.02, 0.98, pipeline, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Pipeline Overview', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Train vs Test performance
ax2 = axes[0, 1]

# Simulated results for different model complexities
factors_used = [1, 2, 3, 4, 5, 6, 8, 10]
train_r2 = [0.62, 0.68, 0.73, 0.76, 0.78, 0.79, 0.82, 0.85]
test_r2 = [0.61, 0.67, 0.71, 0.73, 0.74, 0.74, 0.72, 0.68]

ax2.plot(factors_used, train_r2, color=MLBLUE, linewidth=2.5, marker='o', markersize=8, label='Train R-sq')
ax2.plot(factors_used, test_r2, color=MLORANGE, linewidth=2.5, marker='s', markersize=8, label='Test R-sq')

# Mark optimal
optimal_idx = np.argmax(test_r2)
ax2.scatter([factors_used[optimal_idx]], [test_r2[optimal_idx]], c=MLGREEN, s=200,
            marker='*', zorder=5, edgecolors='black', label='Optimal')
ax2.axvline(factors_used[optimal_idx], color='gray', linestyle='--', alpha=0.5)

ax2.set_title('Model Complexity: Train vs Test R-squared', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Number of Factors', fontsize=10)
ax2.set_ylabel('R-squared', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

ax2.text(8.5, 0.75, 'Overfitting\nzone', fontsize=9, color=MLRED, style='italic')

# Plot 3: sklearn code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
Full sklearn Pipeline Code

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load and prepare data
stock_ret = pd.read_csv('stock_returns.csv', index_col='date')
ff_factors = pd.read_csv('ff_factors.csv', index_col='date')
data = pd.merge(stock_ret, ff_factors, left_index=True, right_index=True)

# 2. Define features and target
X = data[['Mkt-RF', 'SMB', 'HML', 'MOM']]
y = data['stock_ret'] - data['RF']  # Excess return

# 3. Time series split (NOT random!)
tscv = TimeSeriesSplit(n_splits=5)

# 4. Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Optional
    ('regressor', Ridge(alpha=0.1))
])

# 5. Cross-validate
scores = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)
    score = r2_score(y_test, pipeline.predict(X_test))
    scores.append(score)

print(f"CV R-sq: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('sklearn Pipeline Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Cross-validation results
ax4 = axes[1, 1]

cv_folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
cv_scores = [0.71, 0.73, 0.72, 0.75, 0.74]

bars = ax4.bar(cv_folds, cv_scores, color=MLBLUE, edgecolor='black', linewidth=0.5)
ax4.axhline(np.mean(cv_scores), color=MLRED, linewidth=2.5, linestyle='--',
            label=f'Mean: {np.mean(cv_scores):.3f}')

# Add std band
ax4.axhspan(np.mean(cv_scores) - np.std(cv_scores),
            np.mean(cv_scores) + np.std(cv_scores),
            alpha=0.2, color=MLRED, label=f'+/- Std: {np.std(cv_scores):.3f}')

ax4.set_title('Time Series Cross-Validation Results', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_ylabel('R-squared', fontsize=10)
ax4.set_ylim(0.65, 0.80)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3, axis='y')

# Add values on bars
for bar, score in zip(bars, cv_scores):
    ax4.text(bar.get_x() + bar.get_width()/2, score + 0.005, f'{score:.3f}',
             ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
