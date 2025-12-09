"""Feature Importance - Which features matter?"""
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
fig.suptitle('Feature Importance in Decision Trees and Random Forests', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: How importance is calculated
ax1 = axes[0, 0]
ax1.axis('off')

explanation = '''
HOW FEATURE IMPORTANCE WORKS

GINI IMPORTANCE (Default in sklearn)
------------------------------------
For each feature:

1. Find all nodes that split on this feature

2. Calculate impurity decrease at each split:
   decrease = n_samples * (parent_impurity
              - left_weight * left_impurity
              - right_weight * right_impurity)

3. Sum all decreases for this feature

4. Normalize so all importances sum to 1


INTERPRETATION:
---------------
- Higher = more important for predictions
- 0 = never used in any split
- Does NOT indicate direction (positive/negative)

PROS:
- Fast to compute (built into training)
- Works for both trees and forests

CONS:
- Biased toward high-cardinality features
- Biased toward features with many possible splits
- Can be misleading with correlated features
'''

ax1.text(0.02, 0.98, explanation, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Gini Importance Explained', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Feature importance bar chart
ax2 = axes[0, 1]

features = ['Momentum 5d', 'Volume Ratio', 'RSI', 'MA Cross', 'Volatility',
            'Sentiment', 'VIX', 'Earnings Gap']
importance = np.array([0.22, 0.18, 0.15, 0.14, 0.12, 0.10, 0.06, 0.03])

# Sort by importance
idx = np.argsort(importance)[::-1]
features_sorted = [features[i] for i in idx]
importance_sorted = importance[idx]

colors = plt.cm.Blues(np.linspace(0.9, 0.3, len(features)))
bars = ax2.barh(range(len(features)), importance_sorted, color=colors, edgecolor='black', linewidth=0.5)

ax2.set_yticks(range(len(features)))
ax2.set_yticklabels(features_sorted)
ax2.set_title('Feature Importance (Random Forest)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Importance', fontsize=10)
ax2.invert_yaxis()
ax2.grid(alpha=0.3, axis='x')

# Add values
for bar, imp in zip(bars, importance_sorted):
    ax2.text(imp + 0.005, bar.get_y() + bar.get_height()/2, f'{imp:.2%}',
             va='center', fontsize=9)

# Plot 3: Permutation importance comparison
ax3 = axes[1, 0]

features_short = ['Mom5d', 'Volume', 'RSI', 'MA', 'Vol', 'Sent', 'VIX', 'Earn']
gini_imp = [0.22, 0.18, 0.15, 0.14, 0.12, 0.10, 0.06, 0.03]
perm_imp = [0.18, 0.20, 0.12, 0.16, 0.08, 0.14, 0.03, 0.02]

x = np.arange(len(features_short))
width = 0.35

bars1 = ax3.bar(x - width/2, gini_imp, width, label='Gini Importance', color=MLBLUE, edgecolor='black')
bars2 = ax3.bar(x + width/2, perm_imp, width, label='Permutation Importance', color=MLORANGE, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(features_short, fontsize=9)
ax3.set_title('Gini vs Permutation Importance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Importance', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')

# Highlight differences
ax3.annotate('Rankings can\ndiffer!', xy=(5, max(gini_imp[5], perm_imp[5])),
             xytext=(5.5, 0.20), fontsize=9, color=MLRED,
             arrowprops=dict(arrowstyle='->', color=MLRED))

# Plot 4: Code for feature importance
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
EXTRACTING FEATURE IMPORTANCE

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd

# Fit model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Method 1: Gini Importance (built-in)
gini_importance = rf.feature_importances_

# Display
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': gini_importance
}).sort_values('importance', ascending=False)
print(importance_df)

# Method 2: Permutation Importance (more reliable)
perm_result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_importance = perm_result.importances_mean
perm_std = perm_result.importances_std

# Plot
import matplotlib.pyplot as plt
plt.barh(feature_names, gini_importance)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
