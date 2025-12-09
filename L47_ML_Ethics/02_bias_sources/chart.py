"""Bias Sources - Where Bias Comes From in ML"""
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
fig.suptitle('Bias in ML: Sources and Types', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Types of bias
ax1 = axes[0, 0]
ax1.axis('off')

bias_types = '''
TYPES OF BIAS IN ML

1. DATA COLLECTION BIAS:
------------------------
- Who collected the data?
- What was included/excluded?
- Historical inequalities baked in
Example: Training on bull market only


2. SELECTION BIAS:
------------------
- Non-random sampling
- Survivorship bias
- Missing populations
Example: Only successful companies in data


3. MEASUREMENT BIAS:
--------------------
- How features are measured
- Proxy variables
- Different standards
Example: Different accounting standards


4. ALGORITHMIC BIAS:
--------------------
- Model assumptions
- Optimization objectives
- Feature weights
Example: Overweighting recent data


5. CONFIRMATION BIAS:
---------------------
- Researcher expectations
- Cherry-picking results
- Ignoring contradictions
Example: Only reporting good backtests


6. DEPLOYMENT BIAS:
-------------------
- Context differs from training
- Population shift
- Feedback loops
Example: Model causes market change
'''

ax1.text(0.02, 0.98, bias_types, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Types of Bias', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: ML Pipeline bias points
ax2 = axes[0, 1]
ax2.axis('off')

# Draw pipeline with bias points
pipeline_steps = ['Data\nCollection', 'Data\nPreparation', 'Feature\nEngineering',
                  'Model\nTraining', 'Evaluation', 'Deployment']

for i, step in enumerate(pipeline_steps):
    x = 0.08 + i * 0.15
    ax2.add_patch(plt.Rectangle((x, 0.5), 0.12, 0.25, facecolor=MLBLUE, alpha=0.3))
    ax2.text(x+0.06, 0.62, step, fontsize=8, ha='center', va='center')

    if i < len(pipeline_steps) - 1:
        ax2.annotate('', xy=(x+0.15, 0.62), xytext=(x+0.12, 0.62),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray'))

# Bias entry points (red triangles)
bias_points = [
    (0.14, 0.45, 'Historical\nbias'),
    (0.29, 0.45, 'Selection\nbias'),
    (0.44, 0.45, 'Proxy\nbias'),
    (0.59, 0.45, 'Algorithm\nbias'),
    (0.74, 0.45, 'Metric\nbias'),
    (0.89, 0.45, 'Feedback\nbias')
]

for x, y, label in bias_points:
    ax2.plot(x, y+0.05, 'v', markersize=12, color=MLRED)
    ax2.text(x, y-0.08, label, fontsize=7, ha='center', color=MLRED)

ax2.text(0.5, 0.95, 'BIAS ENTRY POINTS IN ML PIPELINE', fontsize=10, ha='center', fontweight='bold')
ax2.text(0.5, 0.15, 'Red markers show where bias can enter your model', fontsize=9, ha='center', style='italic')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Pipeline Bias Points', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Finance-specific bias examples
ax3 = axes[1, 0]
ax3.axis('off')

finance_bias = '''
BIAS IN FINANCE ML

SURVIVORSHIP BIAS:
------------------
Problem: Training only on existing companies
         (failed companies not in data)
Impact:  Overoptimistic performance estimates
Fix:     Include delisted/bankrupt companies


LOOK-AHEAD BIAS:
----------------
Problem: Using future information in features
         (data not available at prediction time)
Impact:  Unrealistic backtesting results
Fix:     Strict point-in-time data handling


SECTOR/REGION BIAS:
-------------------
Problem: Model trained on US tech stocks
         applied to emerging markets
Impact:  Poor generalization, wrong signals
Fix:     Diverse training data, sector controls


TIME PERIOD BIAS:
-----------------
Problem: Training only on bull markets
         or specific regimes
Impact:  Fails during market stress
Fix:     Include multiple market cycles


LIQUIDITY BIAS:
---------------
Problem: Backtesting ignores market impact
         of trading recommendations
Impact:  Unrealistic expected returns
Fix:     Include transaction costs, slippage


DATA SNOOPING BIAS:
-------------------
Problem: Testing many strategies until one works
Impact:  Overfitting to historical data
Fix:     Out-of-sample testing, walk-forward
'''

ax3.text(0.02, 0.98, finance_bias, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Finance-Specific Bias', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Bias detection methods
ax4 = axes[1, 1]

methods = ['Data\nAudit', 'Feature\nAnalysis', 'Subgroup\nTesting', 'Cross-\nValidation', 'External\nReview']
effectiveness = [75, 80, 90, 85, 95]
effort = [40, 50, 70, 60, 85]

x = np.arange(len(methods))
width = 0.35

bars1 = ax4.bar(x - width/2, effectiveness, width, label='Effectiveness', color=MLGREEN, alpha=0.7)
bars2 = ax4.bar(x + width/2, effort, width, label='Effort Required', color=MLORANGE, alpha=0.7)

ax4.set_ylabel('Score (%)')
ax4.set_xticks(x)
ax4.set_xticklabels(methods, fontsize=9)
ax4.legend(fontsize=8)
ax4.set_ylim(0, 110)
ax4.set_title('Bias Detection Methods', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars1, effectiveness):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
             ha='center', fontsize=8, color=MLGREEN)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
