"""Topic Selection - Choosing Your Project Topic"""
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
fig.suptitle('Topic Selection: Finding Your Project Focus', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Example topics
ax1 = axes[0, 0]
ax1.axis('off')

topics = '''
SUGGESTED PROJECT TOPICS

PREDICTION FOCUSED:
-------------------
1. Stock Volatility Prediction
   - Predict VIX or stock volatility
   - Use historical prices, volume
   - Classification or regression

2. Forex Exchange Rate Forecast
   - Predict currency movements
   - Multiple currency pairs
   - Time series features

3. Crypto Price Movement
   - Bitcoin/Ethereum direction
   - On-chain metrics
   - Social sentiment


ANALYSIS FOCUSED:
-----------------
4. Portfolio Optimization Visualizer
   - Efficient frontier calculation
   - Risk-return tradeoffs
   - Interactive asset selection

5. ESG Sentiment Analysis
   - Analyze ESG news
   - Company sustainability scores
   - Sector comparisons

6. Anomaly Detection Dashboard
   - Detect unusual patterns
   - Market manipulation signals
   - Risk monitoring


REMEMBER:
---------
- NO credit scoring / lending data
- Finance domain only
- Must be deployable
'''

ax1.text(0.02, 0.98, topics, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Example Topics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Topic selection criteria
ax2 = axes[0, 1]
ax2.axis('off')

# Draw criteria framework
criteria = [
    ('Data Available?', 'Can you get the data easily?\nYahoo Finance, Kaggle, etc.', MLBLUE),
    ('Scope Feasible?', '2 weeks of work\nNot too big, not too small', MLGREEN),
    ('Interesting?', 'Will you stay motivated?\nSomething you care about', MLORANGE),
    ('Deployable?', 'Can it run in Streamlit?\nReasonable compute needs', MLPURPLE)
]

for i, (title, desc, color) in enumerate(criteria):
    y = 0.85 - i * 0.22
    ax2.add_patch(plt.Rectangle((0.05, y-0.08), 0.9, 0.18, facecolor=color, alpha=0.2))
    ax2.text(0.08, y+0.05, title, fontsize=11, fontweight='bold', color=color)
    ax2.text(0.08, y-0.03, desc, fontsize=9)

ax2.text(0.5, 0.95, 'TOPIC SELECTION CHECKLIST', fontsize=11, ha='center', fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Selection Criteria', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Topic difficulty comparison
ax3 = axes[1, 0]

project_types = ['Stock Volatility', 'Portfolio Optimizer', 'Sentiment Analysis',
                 'Anomaly Detection', 'Forex Forecast', 'Crypto Analysis']
data_difficulty = [2, 2, 3, 3, 2, 3]
model_complexity = [3, 4, 4, 3, 3, 3]
deploy_difficulty = [2, 3, 3, 3, 2, 2]

x = np.arange(len(project_types))
width = 0.25

bars1 = ax3.bar(x - width, data_difficulty, width, label='Data Acquisition', color=MLBLUE, alpha=0.7)
bars2 = ax3.bar(x, model_complexity, width, label='Model Complexity', color=MLGREEN, alpha=0.7)
bars3 = ax3.bar(x + width, deploy_difficulty, width, label='Deployment', color=MLORANGE, alpha=0.7)

ax3.set_ylabel('Difficulty (1=Easy, 5=Hard)')
ax3.set_xticks(x)
ax3.set_xticklabels(project_types, fontsize=8, rotation=15, ha='right')
ax3.legend(fontsize=8, loc='upper right')
ax3.set_ylim(0, 5)
ax3.set_title('Difficulty by Project Type', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.grid(alpha=0.3, axis='y')
ax3.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
ax3.text(5.5, 3.1, 'Medium', fontsize=8, color='gray')

# Plot 4: Decision flowchart
ax4 = axes[1, 1]
ax4.axis('off')

flowchart = '''
TOPIC DECISION FLOWCHART

START: What interests you most?
         |
         v
    +----+----+
    |         |
PREDICTION  ANALYSIS
    |         |
    v         v
Time series? Portfolio?
Stock/Forex  Clustering?
Crypto?      Sentiment?
    |         |
    v         v
Check data   Check data
availability availability
    |         |
    v         v
    +----+----+
         |
         v
    Feasible in
    2 weeks?
         |
    +----+----+
    |         |
   YES        NO
    |         |
    v         v
  PROCEED    Simplify
  with       scope or
  topic      choose
             different
             topic


TIPS:
-----
- Start simple, add complexity later
- Better to finish simple project
  than abandon complex one
- Ask instructor if unsure!
'''

ax4.text(0.02, 0.98, flowchart, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Decision Process', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
