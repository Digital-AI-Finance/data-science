"""sklearn Logistic Regression - Implementation"""
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
fig.suptitle('Logistic Regression in sklearn', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Basic sklearn workflow
ax1 = axes[0, 0]
ax1.axis('off')

workflow = '''
LOGISTIC REGRESSION WORKFLOW

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. PREPARE DATA
X = df[['returns_lag1', 'volume', 'volatility']]
y = df['market_direction']  # 0 = down, 1 = up

# 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. SCALE FEATURES (important for regularization!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. FIT MODEL
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 5. PREDICT
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 6. EVALUATE
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
'''

ax1.text(0.02, 0.98, workflow, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('sklearn Workflow', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Key parameters
ax2 = axes[0, 1]
ax2.axis('off')

params = '''
KEY LOGISTICREGRESSION PARAMETERS

LogisticRegression(
    penalty='l2',        # Regularization type
    C=1.0,               # Inverse of regularization strength
    solver='lbfgs',      # Optimization algorithm
    max_iter=100,        # Max iterations
    random_state=42,     # Reproducibility
    class_weight=None    # Handle imbalanced classes
)

PENALTY OPTIONS:
----------------
'l2'     : Ridge-like (default, shrinks all)
'l1'     : Lasso-like (feature selection)
'elasticnet': Mix of L1 and L2
None     : No regularization

C PARAMETER:
------------
C = 1/lambda (inverse of regularization)
- High C (100): Low regularization, may overfit
- Low C (0.01): High regularization, may underfit
- Default C=1: Balanced

SOLVER OPTIONS:
---------------
'lbfgs'    : Default, works for small datasets
'liblinear': Good for small datasets, L1 penalty
'sag'/'saga': Faster for large datasets
'newton-cg': Second-order optimization

CLASS_WEIGHT:
-------------
None              : All classes equal
'balanced'        : Adjust for imbalanced data
{0: 1, 1: 10}     : Custom weights
'''

ax2.text(0.02, 0.98, params, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Key Parameters', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Feature coefficients visualization
ax3 = axes[1, 0]

features = ['Returns (t-1)', 'Volume', 'Volatility', 'RSI', 'MA Cross']
coefficients = [0.45, 0.22, -0.35, 0.15, 0.38]
colors = [MLGREEN if c > 0 else MLRED for c in coefficients]

y_pos = np.arange(len(features))
bars = ax3.barh(y_pos, coefficients, color=colors, edgecolor='black', linewidth=0.5)
ax3.axvline(0, color='black', linewidth=1.5)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(features)
ax3.set_title('Feature Coefficients (model.coef_)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Coefficient Value', fontsize=10)
ax3.grid(alpha=0.3, axis='x')

# Add values
for bar, coef in zip(bars, coefficients):
    x_pos = coef + 0.02 if coef > 0 else coef - 0.02
    ha = 'left' if coef > 0 else 'right'
    ax3.text(x_pos, bar.get_y() + bar.get_height()/2, f'{coef:.2f}',
             va='center', ha=ha, fontsize=10, fontweight='bold')

ax3.text(0.02, 0.02, 'Positive coef: increases P(Up)\nNegative coef: decreases P(Up)',
         transform=ax3.transAxes, fontsize=8, style='italic')

# Plot 4: Model outputs
ax4 = axes[1, 1]

# Simulated predictions
n_samples = 20
y_prob = np.random.uniform(0.1, 0.9, n_samples)
y_prob = np.sort(y_prob)
y_pred = (y_prob >= 0.5).astype(int)
y_true = np.array([0,0,0,0,0,0,0,1,0,1,0,1,1,0,1,1,1,1,1,1])

ax4.scatter(range(n_samples), y_prob, c=[MLGREEN if p else MLRED for p in (y_pred == y_true)],
            s=100, edgecolors='black', zorder=3)

ax4.axhline(0.5, color=MLPURPLE, linewidth=2.5, linestyle='--', label='Decision Threshold')
ax4.fill_between(range(n_samples), 0, 0.5, alpha=0.1, color=MLBLUE, label='Predict Class 0')
ax4.fill_between(range(n_samples), 0.5, 1, alpha=0.1, color=MLORANGE, label='Predict Class 1')

ax4.set_title('predict_proba() Output', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlabel('Sample Index', fontsize=10)
ax4.set_ylabel('P(Class 1)', fontsize=10)
ax4.set_ylim(0, 1)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MLGREEN, label='Correct'),
                   Patch(facecolor=MLRED, label='Wrong')]
ax4.legend(handles=legend_elements, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
