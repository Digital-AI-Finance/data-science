"""Finance Ethics - Specific Ethical Considerations for Finance ML"""
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
fig.suptitle('Finance Ethics: Specific Considerations for Finance ML', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Finance-specific ethical issues
ax1 = axes[0, 0]
ax1.axis('off')

finance_ethics = '''
FINANCE-SPECIFIC ETHICAL ISSUES

MARKET MANIPULATION:
--------------------
- Algorithms that game the system
- Spoofing and layering
- Front-running signals
- Creating artificial volatility
Rule: Don't build to manipulate


INFORMATION ASYMMETRY:
----------------------
- Using non-public information
- Unfair data advantages
- Exploiting retail investors
- Insider trading signals
Rule: Fair information access


SYSTEMIC RISK:
--------------
- Herding behavior (all models similar)
- Feedback loops
- Flash crashes
- Contagion effects
Rule: Consider market-wide impact


FIDUCIARY DUTY:
---------------
- Acting in client's interest
- Suitable recommendations
- Honest about performance
- Clear about conflicts
Rule: Client interests first


FINANCIAL INCLUSION:
--------------------
- Equal service quality
- Fair pricing models
- Accessible products
- Serving underserved
Rule: Don't exclude unfairly


TRANSPARENCY:
-------------
- Disclose AI use
- Explain recommendations
- Clear about limitations
- Honest marketing
Rule: No hidden AI decisions
'''

ax1.text(0.02, 0.98, finance_ethics, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Finance-Specific Issues', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Case studies
ax2 = axes[0, 1]
ax2.axis('off')

cases = '''
ETHICS CASE STUDIES

CASE 1: FLASH CRASH (2010)
--------------------------
What: Dow dropped 1000 points in minutes
Cause: Algorithmic trading cascade
Lesson: Need circuit breakers
        and human oversight


CASE 2: ALGO HERDING
--------------------
What: Many quant funds failed together
Cause: Similar models, similar trades
Lesson: Diversity in approaches
        reduces systemic risk


CASE 3: BIASED TRADING MODELS
-----------------------------
What: Models favored certain stocks
Cause: Data reflected past biases
Lesson: Audit training data
        for historical bias


CASE 4: MISLEADING BACKTESTS
----------------------------
What: Funds showed amazing returns
Cause: Overfitting, cherry-picking
Lesson: Out-of-sample testing
        is essential


WHAT WE LEARN:
--------------
1. Technical excellence != ethical
2. Unintended consequences happen
3. Oversight matters
4. Transparency builds trust
5. Think beyond your model


FOR YOUR PROJECT:
-----------------
Include a "Limitations & Ethics"
section discussing potential
misuse and safeguards.
'''

ax2.text(0.02, 0.98, cases, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Case Studies', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Ethics checklist for projects
ax3 = axes[1, 0]

checklist_items = [
    'Data source is documented',
    'No sensitive/private data used',
    'Bias testing performed',
    'Limitations clearly stated',
    'Not for actual investment',
    'Model explainability provided',
    'Potential misuse considered',
    'Honest about performance'
]

# Check status
status = [1, 1, 1, 1, 1, 1, 1, 1]  # All should be checked

y_pos = np.arange(len(checklist_items))

for i, (item, checked) in enumerate(zip(checklist_items, status)):
    color = MLGREEN if checked else MLRED
    ax3.add_patch(plt.Rectangle((0, i-0.4), 0.05, 0.8, facecolor=color, alpha=0.5))
    ax3.text(0.07, i, item, fontsize=10, va='center')
    if checked:
        ax3.text(0.025, i, 'Y', fontsize=10, va='center', ha='center', fontweight='bold')

ax3.set_xlim(0, 1)
ax3.set_ylim(-0.5, len(checklist_items)-0.5)
ax3.invert_yaxis()
ax3.axis('off')
ax3.set_title('Project Ethics Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Ethics statement template
ax4 = axes[1, 1]
ax4.axis('off')

template = '''
PROJECT ETHICS STATEMENT TEMPLATE

Include this in your README and presentation:


ETHICAL CONSIDERATIONS
======================

DATA ETHICS:
------------
"Data was obtained from [source],
which is [publicly available/licensed].
No personal or private data was used."


BIAS ASSESSMENT:
----------------
"Model performance was tested across
[groups]. Performance varied by [X%]
between groups, which is [acceptable/
being investigated]."


LIMITATIONS:
------------
"This model is for educational purposes
only. Key limitations include:
- Trained on historical data only
- Does not account for [factors]
- Should not be used for [purpose]"


INTENDED USE:
-------------
"This model demonstrates [concept].
It is NOT intended for actual
investment decisions."


POTENTIAL MISUSE:
-----------------
"This model could be misused for
[scenario]. To prevent this,
[safeguard] has been implemented."


NOTE: Adapt this template to your
specific project. Be honest!
'''

ax4.text(0.02, 0.98, template, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Ethics Statement Template', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
