"""Probability Interpretation"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 6))


# Credit score example
scores = np.array([580, 620, 650, 700, 750, 800])
# Simulated default probabilities
probs = 1 / (1 + np.exp(0.03 * (scores - 680)))

ax.bar(scores, probs, width=30, color=MLBLUE, edgecolor='black', alpha=0.7)
ax.axhline(0.5, color=MLRED, linestyle='--', linewidth=2, label='50% threshold')

for score, prob in zip(scores, probs):
    ax.text(score, prob + 0.03, f'{prob:.0%}', ha='center', fontsize=9)

ax.set_title('Default Probability by Credit Score', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Credit Score', fontsize=10)
ax.set_ylabel('P(Default)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 1)


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
