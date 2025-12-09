"""SHAP Values - Understanding Individual Predictions"""
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
fig.suptitle('SHAP Values: Understanding Individual Predictions', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What are SHAP values
ax1 = axes[0, 0]
ax1.axis('off')

shap_intro = '''
WHAT ARE SHAP VALUES?

DEFINITION:
-----------
SHAP = SHapley Additive exPlanations

Based on game theory concept:
"How much does each feature
 contribute to the prediction?"


THE KEY IDEA:
-------------
Prediction = Base value + SHAP values

Base value: Average prediction
SHAP value: Each feature's contribution

Example:
  Base prediction: 0.50
  + Price effect:  +0.15
  + Volume effect: +0.10
  + Sector effect: -0.05
  = Final:          0.70


WHY SHAP?
---------
1. Consistent: Same contribution = same value
2. Local: Explains individual predictions
3. Additive: Values sum to prediction
4. Unified: Works for any model


KEY INSIGHT:
------------
Positive SHAP = pushes prediction UP
Negative SHAP = pushes prediction DOWN

For binary classification:
  Positive = more likely class 1
  Negative = more likely class 0


SHAP IN PYTHON:
---------------
import shap

# Create explainer
explainer = shap.Explainer(model)

# Calculate SHAP values
shap_values = explainer(X)

# Visualize
shap.plots.waterfall(shap_values[0])
'''

ax1.text(0.02, 0.98, shap_intro, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What are SHAP Values?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Waterfall plot example
ax2 = axes[0, 1]

# Simulate waterfall plot
features = ['Price Momentum', 'Volume', 'Volatility', 'P/E Ratio', 'Market Cap']
shap_values = [0.15, 0.10, -0.08, 0.05, -0.02]
base_value = 0.50
final_value = base_value + sum(shap_values)

# Create waterfall
y_pos = np.arange(len(features) + 2)
values = [base_value] + shap_values + [final_value]
labels = ['Base\nValue'] + features + ['Final\nPrediction']

cumulative = [base_value]
for sv in shap_values:
    cumulative.append(cumulative[-1] + sv)
cumulative.append(final_value)

# Draw bars
for i in range(len(features)):
    start = cumulative[i]
    end = cumulative[i+1]
    color = MLGREEN if shap_values[i] > 0 else MLRED
    ax2.barh(i+1, abs(shap_values[i]), left=min(start, end), color=color, alpha=0.7)
    ax2.text(end + 0.02, i+1, f'{shap_values[i]:+.2f}', va='center', fontsize=9)

# Base and final
ax2.barh(0, 0.01, left=base_value-0.005, color=MLBLUE, alpha=0.7)
ax2.text(base_value + 0.02, 0, f'{base_value:.2f}', va='center', fontsize=9)
ax2.barh(len(features)+1, 0.01, left=final_value-0.005, color=MLPURPLE, alpha=0.7)
ax2.text(final_value + 0.02, len(features)+1, f'{final_value:.2f}', va='center', fontsize=9)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlabel('Prediction Value')
ax2.set_xlim(0.3, 0.9)
ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('SHAP Waterfall Plot (Example)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.grid(alpha=0.3, axis='x')

# Plot 3: SHAP summary plot style
ax3 = axes[1, 0]

# Simulate summary plot data
np.random.seed(42)
n_samples = 50
features_list = ['Price Momentum', 'Volume', 'Volatility', 'P/E Ratio', 'Market Cap']

for i, feature in enumerate(features_list):
    # Generate SHAP values and feature values
    shap_vals = np.random.randn(n_samples) * (0.15 - i*0.02)
    feature_vals = np.random.rand(n_samples)

    # Color by feature value
    colors = plt.cm.RdBu_r(feature_vals)

    y = np.ones(n_samples) * (len(features_list) - i - 1) + np.random.randn(n_samples) * 0.1
    ax3.scatter(shap_vals, y, c=feature_vals, cmap='RdBu_r', alpha=0.6, s=30)

ax3.set_yticks(range(len(features_list)))
ax3.set_yticklabels(features_list[::-1], fontsize=9)
ax3.set_xlabel('SHAP Value (impact on prediction)')
ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
ax3.set_title('SHAP Summary Plot (Example)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Colorbar
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0, 1))
cbar = plt.colorbar(sm, ax=ax3, label='Feature Value')
cbar.ax.set_yticklabels(['Low', '', '', '', 'High'])

# Plot 4: SHAP interpretation guide
ax4 = axes[1, 1]
ax4.axis('off')

interpretation = '''
INTERPRETING SHAP VALUES

READING A WATERFALL PLOT:
-------------------------
- Start from base value (average)
- Each bar shows feature contribution
- Green/positive: increases prediction
- Red/negative: decreases prediction
- Final value = sum of all


READING A SUMMARY PLOT:
-----------------------
- Each dot = one prediction
- X-axis: SHAP value (impact)
- Color: feature value (red=high, blue=low)
- Spread: feature importance


COMMON PATTERNS:
----------------
High feature value -> High SHAP:
  "Higher price momentum = BUY signal"

High feature value -> Low SHAP:
  "Higher P/E ratio = SELL signal"

Wide spread:
  "This feature matters a lot"

Narrow spread:
  "This feature matters less"


FOR YOUR PROJECT:
-----------------
1. Train your model
2. Create SHAP explainer
3. Generate SHAP values
4. Include plots in presentation
5. Explain key features

"My model predicts BUY because
 the price momentum is strong (+0.15)
 and volume is increasing (+0.10)."


LIMITATIONS:
------------
- Computationally expensive
- Approximate for complex models
- Doesn't show causation
'''

ax4.text(0.02, 0.98, interpretation, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Interpreting SHAP', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
