"""Correlation vs Causation"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
fig.suptitle('Correlation Does NOT Imply Causation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Causal relationship diagram
ax1 = axes[0, 0]
ax1.axis('off')

ax1.text(0.5, 0.95, 'Possible Explanations for Correlation', ha='center', fontsize=12,
         fontweight='bold', color=MLPURPLE, transform=ax1.transAxes)

# X causes Y
ax1.text(0.15, 0.75, 'X', fontsize=14, fontweight='bold', color=MLBLUE, transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER))
ax1.annotate('', xy=(0.35, 0.75), xytext=(0.2, 0.75), transform=ax1.transAxes,
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax1.text(0.4, 0.75, 'Y', fontsize=14, fontweight='bold', color=MLBLUE, transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER))
ax1.text(0.6, 0.75, '1. X causes Y', fontsize=10, transform=ax1.transAxes)

# Y causes X
ax1.text(0.15, 0.55, 'X', fontsize=14, fontweight='bold', color=MLGREEN, transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER))
ax1.annotate('', xy=(0.2, 0.55), xytext=(0.35, 0.55), transform=ax1.transAxes,
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax1.text(0.4, 0.55, 'Y', fontsize=14, fontweight='bold', color=MLGREEN, transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER))
ax1.text(0.6, 0.55, '2. Y causes X (reverse)', fontsize=10, transform=ax1.transAxes)

# Confounding
ax1.text(0.27, 0.4, 'Z', fontsize=14, fontweight='bold', color=MLORANGE, transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER))
ax1.text(0.15, 0.25, 'X', fontsize=14, fontweight='bold', color=MLORANGE, transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER))
ax1.text(0.4, 0.25, 'Y', fontsize=14, fontweight='bold', color=MLORANGE, transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER))
ax1.text(0.6, 0.3, '3. Z causes both\n   (confounding)', fontsize=10, transform=ax1.transAxes)

# Coincidence
ax1.text(0.15, 0.08, 'X', fontsize=14, fontweight='bold', color=MLRED, transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER))
ax1.text(0.4, 0.08, 'Y', fontsize=14, fontweight='bold', color=MLRED, transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER))
ax1.text(0.6, 0.08, '4. Pure coincidence', fontsize=10, transform=ax1.transAxes)

# Plot 2: Finance example
ax2 = axes[0, 1]
ax2.axis('off')

ax2.text(0.5, 0.95, 'Finance Examples', ha='center', fontsize=12,
         fontweight='bold', color=MLPURPLE, transform=ax2.transAxes)

examples = [
    ('Low VIX vs High Returns', 'Correlation, not causation!\nBoth reflect calm markets', MLBLUE),
    ('Analyst Upgrades vs Price', 'Does upgrade cause price rise?\nOr did price rise prompt upgrade?', MLGREEN),
    ('GDP Growth vs Stock Returns', 'Many confounding factors:\ninterest rates, policy, sentiment', MLORANGE),
]

y = 0.75
for title, explanation, color in examples:
    ax2.text(0.1, y, title, fontsize=10, fontweight='bold', color=color, transform=ax2.transAxes)
    ax2.text(0.1, y - 0.1, explanation, fontsize=9, color='gray', transform=ax2.transAxes)
    y -= 0.28

# Plot 3: Establishing causation
ax3 = axes[1, 0]
ax3.axis('off')

ax3.text(0.5, 0.95, 'How to Establish Causation', ha='center', fontsize=12,
         fontweight='bold', color=MLPURPLE, transform=ax3.transAxes)

criteria = [
    ('1. Temporal Order', 'Cause must precede effect'),
    ('2. Correlation', 'Must be correlated (necessary but not sufficient)'),
    ('3. No Confounders', 'Control for other variables'),
    ('4. Mechanism', 'Plausible explanation exists'),
    ('5. Experiment', 'Randomized controlled trial (gold standard)'),
]

y = 0.78
for criterion, explanation in criteria:
    ax3.text(0.1, y, criterion, fontsize=10, fontweight='bold', color=MLBLUE, transform=ax3.transAxes)
    ax3.text(0.4, y, explanation, fontsize=9, color='gray', transform=ax3.transAxes)
    y -= 0.14

ax3.text(0.5, 0.08, 'In finance: True experiments are rare, be skeptical!',
         ha='center', fontsize=10, style='italic', color=MLRED, transform=ax3.transAxes)

# Plot 4: Key takeaway
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.7, 'CORRELATION', ha='center', fontsize=20,
         fontweight='bold', color=MLBLUE, transform=ax4.transAxes)
ax4.text(0.5, 0.5, '!=', ha='center', fontsize=40,
         fontweight='bold', color=MLRED, transform=ax4.transAxes)
ax4.text(0.5, 0.3, 'CAUSATION', ha='center', fontsize=20,
         fontweight='bold', color=MLGREEN, transform=ax4.transAxes)

ax4.text(0.5, 0.08, 'Always ask: What else could explain this relationship?',
         ha='center', fontsize=11, style='italic', color=MLPURPLE, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.5))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
