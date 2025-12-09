"""Explainability - Making ML Models Interpretable"""
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
fig.suptitle('Explainability: Making ML Models Interpretable', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why explainability matters
ax1 = axes[0, 0]
ax1.axis('off')

why_explain = '''
WHY EXPLAINABILITY MATTERS

REGULATORY REQUIREMENTS:
------------------------
- GDPR: "Right to explanation"
- Financial regulations require
  justification for decisions
- Audit trails for compliance


TRUST & ADOPTION:
-----------------
- Users need to trust predictions
- Managers need to approve models
- Clients want to understand advice
- Debugging requires understanding


RISK MANAGEMENT:
----------------
- Identify model weaknesses
- Detect unexpected behavior
- Validate against domain knowledge
- Prevent costly mistakes


BUSINESS VALUE:
---------------
- Explain recommendations to clients
- Justify trading decisions
- Support investment committees
- Enable human oversight


THE BLACK BOX PROBLEM:
----------------------
Complex models (Neural Networks,
Ensembles) are hard to explain.

Input --> [???] --> Output

vs. Interpretable models:

Input --> [Clear Rules] --> Output


KEY QUESTION:
-------------
"Why did the model predict THIS?"
'''

ax1.text(0.02, 0.98, why_explain, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Why Explainability?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Model interpretability spectrum
ax2 = axes[0, 1]

models = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest', 'Gradient\nBoosting', 'Neural\nNetwork']
interpretability = [95, 85, 50, 40, 20]
performance = [60, 70, 85, 90, 92]
colors = [MLGREEN, MLGREEN, MLORANGE, MLORANGE, MLRED]

ax2.scatter(interpretability, performance, s=300, c=colors, alpha=0.7)

for i, model in enumerate(models):
    ax2.annotate(model, (interpretability[i], performance[i]),
                textcoords='offset points', xytext=(0, 15),
                ha='center', fontsize=8)

ax2.set_xlabel('Interpretability (%)')
ax2.set_ylabel('Typical Performance (%)')
ax2.set_xlim(0, 100)
ax2.set_ylim(50, 100)
ax2.set_title('Interpretability vs Performance Tradeoff', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3)

# Add trade-off line
ax2.plot([100, 20], [60, 92], 'k--', alpha=0.3)
ax2.text(50, 80, 'Trade-off zone', fontsize=9, style='italic', rotation=-20)

# Plot 3: Explainability methods
ax3 = axes[1, 0]
ax3.axis('off')

methods = '''
EXPLAINABILITY METHODS

INHERENTLY INTERPRETABLE:
-------------------------
Linear Models:
  y = w1*x1 + w2*x2 + ... + b
  Coefficients show feature importance

Decision Trees:
  IF price > 100 AND volume > 1M
  THEN predict "BUY"
  Simple rules to follow


POST-HOC EXPLANATIONS:
----------------------
Feature Importance (Global):
  - Which features matter most overall?
  - Random Forest .feature_importances_

Partial Dependence Plots:
  - How does changing one feature
    affect predictions on average?

SHAP Values (Local):
  - Why THIS specific prediction?
  - Contribution of each feature

LIME (Local):
  - Approximate model locally
    with interpretable model


CODE EXAMPLE:
-------------
# Feature importance
importances = model.feature_importances_
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.3f}")

# SHAP values
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
'''

ax3.text(0.02, 0.98, methods, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Explainability Methods', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Feature importance visualization example
ax4 = axes[1, 1]

features = ['Price Momentum', 'Trading Volume', 'Volatility', 'Market Cap',
            'P/E Ratio', 'RSI', 'Moving Avg', 'Sector']
importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]

colors = [MLGREEN if imp > 0.15 else MLBLUE if imp > 0.08 else MLORANGE for imp in importance]

y_pos = np.arange(len(features))
bars = ax4.barh(y_pos, importance, color=colors, alpha=0.7)

ax4.set_yticks(y_pos)
ax4.set_yticklabels(features)
ax4.set_xlabel('Feature Importance')
ax4.set_title('Example: Stock Prediction Feature Importance', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_xlim(0, 0.35)

for bar, val in zip(bars, importance):
    ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.0%}', va='center', fontsize=9)

# Legend
ax4.text(0.25, 7, 'High', color=MLGREEN, fontsize=8, fontweight='bold')
ax4.text(0.25, 6.5, 'Medium', color=MLBLUE, fontsize=8, fontweight='bold')
ax4.text(0.25, 6, 'Low', color=MLORANGE, fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
