"""Odds Ratio - Interpreting logistic coefficients"""
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
fig.suptitle('Odds Ratios: Interpreting Logistic Regression Coefficients', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Odds ratio concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = r'''
ODDS AND ODDS RATIOS

PROBABILITY vs ODDS:
--------------------
Probability P = 0.75 (75% chance)
Odds = P / (1-P) = 0.75 / 0.25 = 3

"3 to 1 odds" means 3 times more likely
to happen than not happen.

ODDS RATIO (OR):
----------------
Compares odds between two groups

OR = (Odds with feature) / (Odds without feature)

OR = 2.0 means:
"The odds of success DOUBLE when the
 feature increases by 1 unit"

FROM LOGISTIC COEFFICIENTS:
---------------------------
log(odds) = beta_0 + beta_1*x_1 + ...

Therefore:
    Odds Ratio = exp(beta)

Example: beta = 0.693
    OR = exp(0.693) = 2.0

INTERPRETATION:
---------------
OR > 1 : Feature INCREASES odds of outcome
OR = 1 : Feature has NO effect
OR < 1 : Feature DECREASES odds of outcome
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Odds Ratio Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Odds ratio visualization
ax2 = axes[0, 1]

features = ['High Volume', 'Positive Sentiment', 'Above MA', 'High Volatility', 'Earnings Beat']
odds_ratios = [1.85, 2.30, 1.45, 0.65, 3.10]
coefficients = [np.log(or_) for or_ in odds_ratios]

y_pos = np.arange(len(features))
colors = [MLGREEN if or_ > 1 else MLRED for or_ in odds_ratios]

bars = ax2.barh(y_pos, odds_ratios, color=colors, edgecolor='black', linewidth=0.5)
ax2.axvline(1, color='black', linewidth=2.5, linestyle='--', label='OR = 1 (no effect)')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(features)
ax2.set_title('Odds Ratios for Market Direction', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Odds Ratio', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, axis='x')

# Add annotations
for bar, or_ in zip(bars, odds_ratios):
    interpretation = f'{or_:.2f}x' if or_ > 1 else f'{or_:.2f}x'
    ax2.text(or_ + 0.1, bar.get_y() + bar.get_height()/2, interpretation,
             va='center', fontsize=10, fontweight='bold')

# Plot 3: Coefficient to OR conversion
ax3 = axes[1, 0]

beta_values = np.linspace(-2, 2, 100)
or_values = np.exp(beta_values)

ax3.plot(beta_values, or_values, color=MLBLUE, linewidth=3)
ax3.axhline(1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Add reference points
ref_betas = [-1, -0.5, 0, 0.5, 1, 1.5]
for b in ref_betas:
    or_ = np.exp(b)
    ax3.scatter([b], [or_], c=MLORANGE, s=80, zorder=5, edgecolors='black')
    ax3.annotate(f'beta={b}\nOR={or_:.2f}', xy=(b, or_), xytext=(b+0.15, or_+0.3),
                 fontsize=8)

ax3.set_title('Coefficient to Odds Ratio: OR = exp(beta)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('Coefficient (beta)', fontsize=10)
ax3.set_ylabel('Odds Ratio', fontsize=10)
ax3.set_ylim(0, 5)
ax3.grid(alpha=0.3)

# Plot 4: Code example
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
Computing Odds Ratios in Python

import numpy as np
from sklearn.linear_model import LogisticRegression

# Fit model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get coefficients
coefficients = model.coef_[0]
feature_names = X_train.columns

# Calculate odds ratios
odds_ratios = np.exp(coefficients)

# Display results
print("Feature              Coef     OR      Interpretation")
print("-" * 60)
for name, coef, or_ in zip(feature_names, coefficients, odds_ratios):
    if or_ > 1:
        interp = f"+{(or_-1)*100:.0f}% odds"
    else:
        interp = f"-{(1-or_)*100:.0f}% odds"
    print(f"{name:20} {coef:+.3f}  {or_:.2f}x   {interp}")

# Example output:
# Feature              Coef     OR      Interpretation
# ------------------------------------------------------------
# volume               +0.616  1.85x   +85% odds
# sentiment            +0.833  2.30x   +130% odds
# above_ma             +0.372  1.45x   +45% odds
# volatility           -0.431  0.65x   -35% odds
# earnings_beat        +1.131  3.10x   +210% odds
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
