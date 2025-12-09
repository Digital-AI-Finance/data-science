"""Model Planning - Designing Your ML Approach"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Planning: Designing Your ML Approach', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Model selection guide
ax1 = axes[0, 0]
ax1.axis('off')

model_guide = '''
MODEL SELECTION GUIDE

REGRESSION (Predict numbers):
-----------------------------
Target: Price, returns, volatility

Models to try:
1. Linear Regression (baseline)
2. Ridge/Lasso Regression
3. Random Forest Regressor
4. Neural Network (MLP)

Metrics: MSE, MAE, R-squared


CLASSIFICATION (Predict categories):
------------------------------------
Target: Up/Down, Fraud/Normal, Sector

Models to try:
1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest Classifier
4. Neural Network (MLP)

Metrics: Accuracy, Precision, Recall, AUC


CLUSTERING (Group similar items):
---------------------------------
No target variable

Models to try:
1. K-Means
2. Hierarchical Clustering
3. DBSCAN

Metrics: Silhouette score, Inertia


MINIMUM REQUIREMENT:
--------------------
- At least 3 different models
- Baseline model (simple)
- 2+ more complex models
- Clear comparison of results
'''

ax1.text(0.02, 0.98, model_guide, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Model Selection Guide', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Model comparison framework
ax2 = axes[0, 1]

models = ['Linear Reg', 'Ridge', 'Random Forest', 'Neural Net']
complexity = [1, 2, 4, 5]
interpretability = [5, 4, 2, 1]
performance = [2, 3, 4, 4.5]

x = np.arange(len(models))
width = 0.25

bars1 = ax2.bar(x - width, complexity, width, label='Complexity', color=MLBLUE, alpha=0.7)
bars2 = ax2.bar(x, interpretability, width, label='Interpretability', color=MLGREEN, alpha=0.7)
bars3 = ax2.bar(x + width, performance, width, label='Typical Performance', color=MLORANGE, alpha=0.7)

ax2.set_ylabel('Score (1=Low, 5=High)')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9)
ax2.legend(fontsize=8)
ax2.set_ylim(0, 6)
ax2.set_title('Model Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3, axis='y')

# Plot 3: Feature engineering ideas
ax3 = axes[1, 0]
ax3.axis('off')

feature_eng = '''
FEATURE ENGINEERING IDEAS

TIME SERIES FEATURES:
---------------------
- Lag features (t-1, t-5, t-20)
- Rolling mean (5, 20, 50 days)
- Rolling std (volatility)
- Daily returns
- Cumulative returns
- Day of week
- Month of year


TECHNICAL INDICATORS:
---------------------
- Moving averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD
- Bollinger Bands
- Volume ratios


TEXT FEATURES (if using NLP):
-----------------------------
- Sentiment scores
- Word counts
- TF-IDF vectors
- Keyword presence


CROSS-SECTIONAL FEATURES:
-------------------------
- Relative performance
- Sector averages
- Market correlation


FEATURE SELECTION TIPS:
-----------------------
1. Start with domain knowledge
2. Check correlations
3. Remove redundant features
4. Use feature importance
5. Keep it simple initially!
'''

ax3.text(0.02, 0.98, feature_eng, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Feature Engineering Ideas', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: ML pipeline visual
ax4 = axes[1, 1]
ax4.axis('off')

# Draw pipeline
pipeline_steps = [
    ('DATA', 'Raw data\nfeatures', 0.08, MLBLUE),
    ('SPLIT', 'Train/Test\n80/20', 0.25, MLORANGE),
    ('SCALE', 'Normalize\nfeatures', 0.42, MLGREEN),
    ('TRAIN', 'Fit models\nCV=5', 0.59, MLPURPLE),
    ('TUNE', 'Grid search\nparams', 0.76, MLRED),
    ('EVAL', 'Test set\nmetrics', 0.93, MLGREEN)
]

for name, desc, x, color in pipeline_steps:
    ax4.add_patch(plt.Rectangle((x-0.06, 0.5), 0.12, 0.35, facecolor=color, alpha=0.3))
    ax4.text(x, 0.78, name, fontsize=10, ha='center', fontweight='bold', color=color)
    ax4.text(x, 0.6, desc, fontsize=7, ha='center')

# Arrows
for i in range(len(pipeline_steps)-1):
    x1 = pipeline_steps[i][2] + 0.06
    x2 = pipeline_steps[i+1][2] - 0.06
    ax4.annotate('', xy=(x2, 0.67), xytext=(x1, 0.67),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax4.text(0.5, 0.95, 'ML PIPELINE', fontsize=12, ha='center', fontweight='bold')

# Warnings
ax4.text(0.5, 0.38, 'COMMON MISTAKES TO AVOID:', fontsize=10, ha='center', fontweight='bold', color=MLRED)
warnings = [
    '- Data leakage (scaling before split)',
    '- Overfitting (too complex model)',
    '- Not enough cross-validation',
    '- Ignoring class imbalance'
]
for j, warning in enumerate(warnings):
    ax4.text(0.5, 0.3 - j*0.06, warning, fontsize=9, ha='center')

ax4.set_xlim(0, 1)
ax4.set_ylim(0.05, 1)
ax4.set_title('ML Pipeline', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
