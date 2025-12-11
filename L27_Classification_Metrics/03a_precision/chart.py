"""Precision - Quality of Positive Predictions"""
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


# Visual: out of predicted positives, how many are correct?
tp, fp = 70, 30
precision = tp / (tp + fp)

ax.barh(['True Positives', 'False Positives'], [tp, fp], color=[MLGREEN, MLRED], edgecolor='black')
ax.axvline(0, color='black', linewidth=1)

ax.set_title(f'Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {precision:.1%}',
             fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Count', fontsize=10)
ax.grid(alpha=0.3, axis='x')

ax.text(0.95, 0.05, 'When you predict positive,\nhow often are you correct?',
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
