"""F1 Score - Harmonic mean of precision and recall"""
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

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('F1 Score: Balancing Precision and Recall', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: F1 formula and intuition
ax1 = axes[0, 0]
ax1.axis('off')

formula = r'''
F1 SCORE FORMULA

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

This is the HARMONIC MEAN of P and R.

WHY HARMONIC MEAN?
------------------
Arithmetic mean: (P + R) / 2
  Problem: Can be high even if one is low
  Example: P=1.0, R=0.1 -> AM = 0.55

Harmonic mean: 2*P*R / (P + R)
  Penalizes extreme imbalance
  Example: P=1.0, R=0.1 -> F1 = 0.18


INTERPRETATION:
---------------
F1 = 1.0: Perfect (P=R=1)
F1 = 0.5: Either P or R is low
F1 ~ 0:   Very poor (P or R near 0)

F1 is HIGH only when BOTH P and R are high!


WHEN TO USE F1:
---------------
- Need single metric for comparison
- Care equally about P and R
- Imbalanced classes
- Model selection/tuning
'''

ax1.text(0.02, 0.98, formula, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('F1 Formula & Intuition', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: F1 surface plot
ax2 = axes[0, 1]

# Create F1 surface
p = np.linspace(0.01, 1, 50)
r = np.linspace(0.01, 1, 50)
P, R = np.meshgrid(p, r)
F1 = 2 * P * R / (P + R)

contour = ax2.contourf(P, R, F1, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax2, label='F1 Score')

# Add contour lines
ax2.contour(P, R, F1, levels=[0.2, 0.4, 0.6, 0.8], colors='white', linewidths=1)

# Add points
points = [(0.9, 0.9, 'High P, High R'),
          (0.9, 0.2, 'High P, Low R'),
          (0.2, 0.9, 'Low P, High R'),
          (0.5, 0.5, 'Balanced')]

for px, py, label in points:
    f1 = 2 * px * py / (px + py)
    ax2.scatter([px], [py], s=100, c='white', edgecolors='black', zorder=5)
    ax2.annotate(f'{label}\nF1={f1:.2f}', xy=(px, py), xytext=(px+0.05, py-0.1), fontsize=7, color='white')

ax2.set_title('F1 Score Surface', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Precision', fontsize=10)
ax2.set_ylabel('Recall', fontsize=10)

# Plot 3: Comparing models
ax3 = axes[1, 0]

models = ['Model A', 'Model B', 'Model C', 'Model D']
precision = [0.95, 0.70, 0.85, 0.60]
recall = [0.40, 0.90, 0.80, 0.95]
f1 = [2*p*r/(p+r) for p, r in zip(precision, recall)]

x = np.arange(len(models))
width = 0.25

bars1 = ax3.bar(x - width, precision, width, label='Precision', color=MLBLUE, edgecolor='black')
bars2 = ax3.bar(x, recall, width, label='Recall', color=MLORANGE, edgecolor='black')
bars3 = ax3.bar(x + width, f1, width, label='F1', color=MLGREEN, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.set_title('Model Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylabel('Score', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')
ax3.set_ylim(0, 1.1)

# Highlight best F1
best_idx = np.argmax(f1)
ax3.annotate('Best F1!', xy=(best_idx + width, f1[best_idx]), xytext=(best_idx + 0.3, f1[best_idx] + 0.1),
             fontsize=10, color=MLGREEN, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=MLGREEN))

# Plot 4: F-beta score
ax4 = axes[1, 1]
ax4.axis('off')

fbeta = '''
F-BETA SCORE: WEIGHTED F1

$$F_\\beta = (1 + \\beta^2) \\times \\frac{Precision \\times Recall}{\\beta^2 \\times Precision + Recall}$$

BETA CONTROLS THE WEIGHT:
-------------------------
beta < 1: Weight PRECISION more
beta = 1: Equal weight (standard F1)
beta > 1: Weight RECALL more

COMMON VALUES:
--------------
F0.5: Precision is 2x as important as Recall
  Use: Email spam (don't lose good emails)

F1: Equal importance (default)
  Use: General purpose

F2: Recall is 2x as important as Precision
  Use: Medical screening (don't miss cases)


SKLEARN CODE:
-------------
from sklearn.metrics import fbeta_score

f05 = fbeta_score(y_test, y_pred, beta=0.5)
f1 = fbeta_score(y_test, y_pred, beta=1)
f2 = fbeta_score(y_test, y_pred, beta=2)

# Or use make_scorer for GridSearchCV
from sklearn.metrics import make_scorer
f2_scorer = make_scorer(fbeta_score, beta=2)
'''

ax4.text(0.02, 0.98, fbeta, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('F-Beta Score (Weighted F1)', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
