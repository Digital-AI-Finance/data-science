"""Confusion Matrix Structure"""
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


from matplotlib.patches import Rectangle

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Draw 2x2 grid
colors = [[MLGREEN, MLRED], [MLORANGE, MLBLUE]]
labels = [['TN\n(Correct Reject)', 'FP\n(False Alarm)'],
          ['FN\n(Missed)', 'TP\n(Correct Hit)']]
values = [[850, 50], [30, 70]]

for i in range(2):
    for j in range(2):
        rect = Rectangle((j*5, (1-i)*5), 5, 5, facecolor=colors[i][j], alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(j*5 + 2.5, (1-i)*5 + 3, labels[i][j], ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(j*5 + 2.5, (1-i)*5 + 1.5, f'n={values[i][j]}', ha='center', va='center', fontsize=11)

ax.text(2.5, 10.3, 'Predicted 0', ha='center', fontsize=10, fontweight='bold')
ax.text(7.5, 10.3, 'Predicted 1', ha='center', fontsize=10, fontweight='bold')
ax.text(-0.5, 7.5, 'Actual 0', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
ax.text(-0.5, 2.5, 'Actual 1', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)

ax.set_title('Confusion Matrix: The Foundation of Classification Metrics', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.axis('off')


plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
