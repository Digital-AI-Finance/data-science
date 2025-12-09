"""Regulatory Requirements - Legal and Compliance Aspects"""
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
fig.suptitle('Regulatory Requirements: Legal and Compliance in Finance ML', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Key regulations
ax1 = axes[0, 0]
ax1.axis('off')

regulations = '''
KEY REGULATIONS FOR FINANCE ML

GDPR (Europe):
--------------
- Right to explanation
- Data minimization
- Purpose limitation
- Consent requirements
Impact: Must explain automated decisions


EU AI ACT (Coming):
-------------------
- Risk-based classification
- High-risk AI requirements
- Transparency obligations
- Human oversight
Impact: Finance ML often "high-risk"


MiFID II (Europe):
------------------
- Algorithm testing
- Risk controls
- Record keeping
- Best execution
Impact: Trading algorithms regulated


SEC (United States):
--------------------
- Market manipulation rules
- Disclosure requirements
- Fiduciary duty
- Fair access
Impact: Investment advice rules


FINRA (United States):
----------------------
- Suitability requirements
- Communications rules
- Supervision obligations
Impact: Must supervise ML recommendations


KEY PRINCIPLE:
--------------
"If a human would be regulated
 for making this decision,
 the ML model is too."
'''

ax1.text(0.02, 0.98, regulations, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Key Regulations', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Risk classification (EU AI Act style)
ax2 = axes[0, 1]
ax2.axis('off')

# Draw risk pyramid
levels = [
    ('UNACCEPTABLE\nRISK', 0.85, MLRED, 'Banned applications'),
    ('HIGH\nRISK', 0.6, MLORANGE, 'Strict requirements'),
    ('LIMITED\nRISK', 0.35, MLBLUE, 'Transparency rules'),
    ('MINIMAL\nRISK', 0.1, MLGREEN, 'Few requirements')
]

for label, y, color, desc in levels:
    width = 0.8 * (1 - y + 0.2)
    x = 0.5 - width/2
    ax2.add_patch(plt.Rectangle((x, y), width, 0.2, facecolor=color, alpha=0.5))
    ax2.text(0.5, y + 0.1, label, fontsize=9, ha='center', va='center', fontweight='bold')
    ax2.text(0.92, y + 0.1, desc, fontsize=8, va='center')

ax2.text(0.5, 0.95, 'EU AI ACT RISK LEVELS', fontsize=11, ha='center', fontweight='bold')

# Finance examples
ax2.text(0.08, 0.55, 'Finance examples:', fontsize=8, fontweight='bold')
ax2.text(0.08, 0.45, '- Trading algorithms', fontsize=7)
ax2.text(0.08, 0.40, '- Investment advice', fontsize=7)
ax2.text(0.08, 0.35, '- Risk assessment', fontsize=7)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Risk Classification', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Compliance requirements
ax3 = axes[1, 0]
ax3.axis('off')

compliance = '''
COMPLIANCE REQUIREMENTS

DOCUMENTATION:
--------------
- Model development process
- Data sources and quality
- Feature selection rationale
- Training methodology
- Validation results
- Known limitations


MODEL GOVERNANCE:
-----------------
- Version control
- Change management
- Approval workflows
- Audit trails
- Regular review cycles


TESTING REQUIREMENTS:
---------------------
- Backtesting results
- Stress testing
- Sensitivity analysis
- Out-of-sample validation
- Fairness testing


MONITORING:
-----------
- Performance tracking
- Drift detection
- Alert mechanisms
- Incident response
- Regular reporting


FOR YOUR PROJECT:
-----------------
Document these in README:
1. Data source and period
2. Model choice rationale
3. Evaluation methodology
4. Known limitations
5. Intended use cases


SIMPLE COMPLIANCE STATEMENT:
----------------------------
"This model is for educational
purposes only. Not intended for
actual investment decisions.
Performance based on historical
data; past results do not
guarantee future returns."
'''

ax3.text(0.02, 0.98, compliance, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Compliance Requirements', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Compliance maturity model
ax4 = axes[1, 1]

stages = ['Ad-hoc', 'Basic', 'Managed', 'Optimized', 'Leading']
documentation = [1, 2, 3, 4, 5]
governance = [1, 2, 3, 4, 4.5]
testing = [1, 1.5, 3, 4, 5]
monitoring = [0.5, 1.5, 2.5, 4, 5]

x = np.arange(len(stages))
width = 0.2

ax4.bar(x - 1.5*width, documentation, width, label='Documentation', color=MLBLUE, alpha=0.7)
ax4.bar(x - 0.5*width, governance, width, label='Governance', color=MLGREEN, alpha=0.7)
ax4.bar(x + 0.5*width, testing, width, label='Testing', color=MLORANGE, alpha=0.7)
ax4.bar(x + 1.5*width, monitoring, width, label='Monitoring', color=MLPURPLE, alpha=0.7)

ax4.set_ylabel('Maturity Level (1-5)')
ax4.set_xticks(x)
ax4.set_xticklabels(stages, fontsize=8)
ax4.legend(fontsize=7, loc='upper left')
ax4.set_ylim(0, 5.5)
ax4.set_title('ML Compliance Maturity Model', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3, axis='y')

# Target for student projects
ax4.axhline(y=2, color=MLRED, linestyle='--', alpha=0.5)
ax4.text(4.5, 2.2, 'Project\ntarget', fontsize=7, ha='center', color=MLRED)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
