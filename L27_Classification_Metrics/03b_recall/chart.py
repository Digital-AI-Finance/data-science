"""Recall - Coverage of Actual Positives"""
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


# Visual: out of actual positives, how many did we find?
tp, fn = 70, 30
recall = tp / (tp + fn)

ax.barh(['True Positives', 'False Negatives'], [tp, fn], color=[MLGREEN, MLORANGE], edgecolor='black')
ax.axvline(0, color='black', linewidth=1)

ax.set_title(f'Recall = TP/(TP+FN) = {tp}/({tp}+{fn}) = {recall:.1%}',
             fontsize=12, fontweight='bold', color=MLPURPLE)
ax.set_xlabel('Count', fontsize=10)
ax.grid(alpha=0.3, axis='x')

ax.text(0.95, 0.05, 'Out of all actual positives,\nhow many did you find?',
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
