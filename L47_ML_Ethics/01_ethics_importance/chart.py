"""Ethics Importance - Why ML Ethics Matters"""
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
fig.suptitle('ML Ethics: Why It Matters in Finance', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why ethics matters
ax1 = axes[0, 0]
ax1.axis('off')

importance = '''
WHY ML ETHICS MATTERS

THE STAKES ARE HIGH:
--------------------
ML models in finance affect:
- Investment decisions
- Trading strategies
- Risk assessments
- Wealth distribution
- Market stability


REAL CONSEQUENCES:
------------------
- Flash crashes from algorithms
- Biased trading recommendations
- Unfair risk scoring
- Market manipulation
- Systemic risk amplification


WHO IS AFFECTED:
----------------
- Individual investors
- Pension funds
- Retirement savings
- Market participants
- Entire economies


ETHICAL ML DEVELOPER:
---------------------
1. Considers impact of predictions
2. Questions data sources
3. Tests for bias
4. Documents limitations
5. Prioritizes transparency


YOUR RESPONSIBILITY:
--------------------
As a data scientist, you will:
- Build models that affect people
- Make decisions with consequences
- Need to justify your choices
- Be accountable for outcomes


"With great power comes
 great responsibility."
'''

ax1.text(0.02, 0.98, importance, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Why Ethics Matters', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Key ethical principles
ax2 = axes[0, 1]
ax2.axis('off')

# Draw principles as connected boxes
principles = [
    ('FAIRNESS', 'Equal treatment\nNo discrimination', MLBLUE),
    ('TRANSPARENCY', 'Explainable decisions\nClear methodology', MLORANGE),
    ('ACCOUNTABILITY', 'Ownership of outcomes\nError correction', MLGREEN),
    ('PRIVACY', 'Data protection\nConfidentiality', MLPURPLE),
    ('SAFETY', 'Risk assessment\nHarm prevention', MLRED)
]

for i, (name, desc, color) in enumerate(principles):
    row = i // 3
    col = i % 3
    x = 0.17 + col * 0.33
    y = 0.65 - row * 0.45

    ax2.add_patch(plt.Rectangle((x-0.13, y-0.15), 0.26, 0.35, facecolor=color, alpha=0.2))
    ax2.text(x, y+0.12, name, fontsize=10, ha='center', fontweight='bold', color=color)
    ax2.text(x, y-0.02, desc, fontsize=8, ha='center')

ax2.text(0.5, 0.95, 'KEY ETHICAL PRINCIPLES', fontsize=11, ha='center', fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Ethical Principles', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Ethics timeline in ML
ax3 = axes[1, 0]

events = [
    (2010, 'Flash Crash\n(algorithmic trading)', MLRED),
    (2016, 'Biased algorithms\n(documented cases)', MLORANGE),
    (2018, 'GDPR\n(right to explanation)', MLBLUE),
    (2020, 'AI Ethics guidelines\n(industry adoption)', MLGREEN),
    (2023, 'AI Act (EU)\n(risk-based regulation)', MLPURPLE)
]

years = [e[0] for e in events]
y_pos = np.arange(len(events))

ax3.scatter(years, y_pos, s=200, c=[e[2] for e in events], alpha=0.7, zorder=5)
ax3.plot(years, y_pos, 'k--', alpha=0.3, zorder=1)

for i, (year, event, color) in enumerate(events):
    ax3.annotate(event, (year, i), textcoords='offset points',
                xytext=(10, 0), fontsize=8, va='center')

ax3.set_yticks([])
ax3.set_xlabel('Year')
ax3.set_xlim(2008, 2025)
ax3.set_title('ML Ethics Timeline', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.grid(alpha=0.3, axis='x')

# Plot 4: Stakeholder impact
ax4 = axes[1, 1]

stakeholders = ['Investors', 'Regulators', 'Society', 'Companies', 'Employees']
positive = [70, 60, 65, 75, 50]
negative = [30, 40, 35, 25, 50]

x = np.arange(len(stakeholders))
width = 0.35

bars1 = ax4.bar(x - width/2, positive, width, label='Benefit from ethical ML', color=MLGREEN, alpha=0.7)
bars2 = ax4.bar(x + width/2, negative, width, label='Risk from unethical ML', color=MLRED, alpha=0.7)

ax4.set_ylabel('Impact Level (%)')
ax4.set_xticks(x)
ax4.set_xticklabels(stakeholders, fontsize=9)
ax4.legend(fontsize=8)
ax4.set_ylim(0, 100)
ax4.set_title('Stakeholder Impact', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
