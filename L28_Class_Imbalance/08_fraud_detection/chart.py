"""Fraud Detection - Complete imbalanced classification example"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
fig.suptitle('Fraud Detection: Complete Imbalanced Classification Pipeline', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Model comparison
ax1 = axes[0, 0]

methods = ['Baseline\n(No treatment)', 'SMOTE', 'Class\nWeights', 'SMOTE +\nWeights', 'Cost-Sensitive\nThreshold']
accuracy = [0.99, 0.95, 0.96, 0.94, 0.93]
fraud_recall = [0.10, 0.78, 0.72, 0.82, 0.88]
fraud_precision = [0.90, 0.25, 0.35, 0.30, 0.22]
f1 = [2*p*r/(p+r+0.001) for p, r in zip(fraud_precision, fraud_recall)]

x = np.arange(len(methods))
width = 0.2

bars1 = ax1.bar(x - 1.5*width, accuracy, width, label='Accuracy', color=MLBLUE, edgecolor='black')
bars2 = ax1.bar(x - 0.5*width, fraud_recall, width, label='Fraud Recall', color=MLRED, edgecolor='black')
bars3 = ax1.bar(x + 0.5*width, fraud_precision, width, label='Fraud Precision', color=MLORANGE, edgecolor='black')
bars4 = ax1.bar(x + 1.5*width, f1, width, label='Fraud F1', color=MLGREEN, edgecolor='black')

ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=8)
ax1.set_title('Method Comparison (1% Fraud Rate)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_ylabel('Score', fontsize=10)
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(alpha=0.3, axis='y')

# Plot 2: Business metrics
ax2 = axes[0, 1]

methods_short = ['Baseline', 'SMOTE', 'Weights', 'Combined', 'Cost-Opt']
total_cost = [95000, 32000, 38000, 28000, 22000]
frauds_caught = [10, 78, 72, 82, 88]

x = np.arange(len(methods_short))
width = 0.35

ax2.bar(x - width/2, [c/1000 for c in total_cost], width, label='Total Cost ($K)', color=MLRED, edgecolor='black')

ax2_twin = ax2.twinx()
ax2_twin.bar(x + width/2, frauds_caught, width, label='Frauds Caught (%)', color=MLGREEN, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(methods_short, fontsize=9)
ax2.set_title('Business Impact', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_ylabel('Cost ($K)', fontsize=10, color=MLRED)
ax2_twin.set_ylabel('Fraud Recall (%)', fontsize=10, color=MLGREEN)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

# Plot 3: Complete pipeline
ax3 = axes[1, 0]
ax3.axis('off')

pipeline = '''
COMPLETE FRAUD DETECTION PIPELINE

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, make_scorer

# 1. CREATE PIPELINE
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # Double protection!
        random_state=42
    ))
])

# 2. STRATIFIED CROSS-VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. EVALUATE WITH APPROPRIATE METRIC
from sklearn.metrics import f1_score, average_precision_score
f1_scorer = make_scorer(f1_score, pos_label=1)  # Fraud = 1
ap_scorer = make_scorer(average_precision_score)

f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=f1_scorer)
ap_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=ap_scorer)

print(f"F1 Score: {f1_scores.mean():.3f} +/- {f1_scores.std():.3f}")
print(f"Avg Precision: {ap_scores.mean():.3f} +/- {ap_scores.std():.3f}")

# 4. FINAL MODEL + THRESHOLD TUNING
pipeline.fit(X_train, y_train)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Find cost-optimal threshold
best_t = find_optimal_threshold(y_test, y_prob, cost_fp=100, cost_fn=10000)
y_pred = (y_prob >= best_t).astype(int)

print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
'''

ax3.text(0.02, 0.98, pipeline, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Complete Pipeline Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Key takeaways
ax4 = axes[1, 1]
ax4.axis('off')

takeaways = '''
KEY TAKEAWAYS: IMBALANCED CLASSIFICATION

1. DON'T TRUST ACCURACY
   - Use F1, Precision, Recall, PR-AUC
   - Always report minority class metrics

2. COMBINE TECHNIQUES
   - SMOTE + Class Weights often best
   - Stratified CV always

3. TUNE THE THRESHOLD
   - Default 0.5 is rarely optimal
   - Use cost-sensitive threshold

4. CONSIDER BUSINESS IMPACT
   - Different errors have different costs
   - Optimize for profit, not just metrics

5. VALIDATE PROPERLY
   - Never resample test data
   - Use stratified splits
   - Check stability across folds


RECOMMENDED APPROACH:
---------------------
1. Start with class_weight='balanced'
2. Add SMOTE if needed (imblearn pipeline)
3. Use StratifiedKFold CV
4. Evaluate with F1 and PR-AUC
5. Tune threshold based on costs
6. Monitor in production (drift!)


WHEN TO USE EACH TECHNIQUE:
---------------------------
- Slight imbalance (80/20): Class weights
- Moderate (95/5): SMOTE + weights
- Extreme (99.9/0.1): Anomaly detection
'''

ax4.text(0.02, 0.98, takeaways, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Key Takeaways', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
