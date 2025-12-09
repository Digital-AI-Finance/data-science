"""Gini and Entropy - Splitting criteria"""
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
fig.suptitle('Gini Impurity and Entropy: How Trees Choose Splits', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Gini vs Entropy curves
ax1 = axes[0, 0]

p = np.linspace(0.001, 0.999, 100)

# Gini impurity: 1 - sum(p_i^2) = 2*p*(1-p) for binary
gini = 2 * p * (1 - p)

# Entropy: -sum(p_i * log2(p_i))
entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)

# Misclassification error
misclass = np.minimum(p, 1-p)

ax1.plot(p, gini, color=MLBLUE, linewidth=2.5, label='Gini Impurity')
ax1.plot(p, entropy/2, color=MLORANGE, linewidth=2.5, label='Entropy (scaled)')
ax1.plot(p, misclass, color=MLGREEN, linewidth=2.5, label='Misclassification', linestyle='--')

ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.7)
ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7)

ax1.set_title('Impurity Measures (Binary Classification)', fontsize=11, fontweight='bold', color=MLPURPLE)
ax1.set_xlabel('Probability of Class 1 (p)', fontsize=10)
ax1.set_ylabel('Impurity', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax1.annotate('Maximum impurity\n(p=0.5, most uncertain)', xy=(0.5, 0.5), xytext=(0.65, 0.35),
             fontsize=9, arrowprops=dict(arrowstyle='->', color=MLRED))

# Plot 2: Formulas
ax2 = axes[0, 1]
ax2.axis('off')

formulas = r'''
IMPURITY MEASURES

GINI IMPURITY
-------------
$Gini = 1 - \sum_{i=1}^{C} p_i^2$

For binary (C=2):
$Gini = 1 - p^2 - (1-p)^2 = 2p(1-p)$

- Range: [0, 0.5] for binary
- 0 = pure node (all one class)
- 0.5 = maximum impurity (50/50 split)


ENTROPY
-------
$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$

For binary:
$H = -p\log_2(p) - (1-p)\log_2(1-p)$

- Range: [0, 1] for binary
- 0 = pure node
- 1 = maximum entropy (50/50)


INFORMATION GAIN
----------------
$IG = H(parent) - \sum \frac{n_j}{n} H(child_j)$

Choose split that MAXIMIZES information gain
(or MINIMIZES weighted child impurity)
'''

ax2.text(0.02, 0.98, formulas, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Mathematical Formulas', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Split example
ax3 = axes[1, 0]
ax3.axis('off')

# Draw split example
split_example = '''
EXAMPLE: CHOOSING THE BEST SPLIT

Parent Node: 100 samples
- Class 0: 60 samples (60%)
- Class 1: 40 samples (40%)

Gini(parent) = 2 * 0.6 * 0.4 = 0.48

SPLIT A: Volume > 1M
------------------------
Left (70 samples):  45 Class 0, 25 Class 1
Gini(L) = 2 * (45/70) * (25/70) = 0.459

Right (30 samples): 15 Class 0, 15 Class 1
Gini(R) = 2 * (15/30) * (15/30) = 0.50

Weighted Gini = (70/100)*0.459 + (30/100)*0.50 = 0.471
Reduction = 0.48 - 0.471 = 0.009

SPLIT B: Returns > 0
------------------------
Left (55 samples):  50 Class 0, 5 Class 1
Gini(L) = 2 * (50/55) * (5/55) = 0.165

Right (45 samples): 10 Class 0, 35 Class 1
Gini(R) = 2 * (10/45) * (35/45) = 0.346

Weighted Gini = (55/100)*0.165 + (45/100)*0.346 = 0.247
Reduction = 0.48 - 0.247 = 0.233

WINNER: SPLIT B (larger reduction in impurity!)
'''

ax3.text(0.02, 0.98, split_example, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Split Selection Example', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Visual split comparison
ax4 = axes[1, 1]

# Show two different splits
split_names = ['Split A\n(Volume)', 'Split B\n(Returns)']
parent_gini = 0.48
weighted_ginis = [0.471, 0.247]
reductions = [parent_gini - g for g in weighted_ginis]

x = np.arange(len(split_names))
width = 0.35

bars1 = ax4.bar(x - width/2, weighted_ginis, width, label='Weighted Child Gini', color=MLBLUE, edgecolor='black')
bars2 = ax4.bar(x + width/2, reductions, width, label='Gini Reduction', color=MLGREEN, edgecolor='black')

ax4.axhline(parent_gini, color=MLRED, linewidth=2.5, linestyle='--', label=f'Parent Gini: {parent_gini}')

ax4.set_xticks(x)
ax4.set_xticklabels(split_names)
ax4.set_title('Comparing Two Potential Splits', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.set_ylabel('Gini Value', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3, axis='y')

# Highlight winner
ax4.annotate('BEST SPLIT!', xy=(1, reductions[1] + 0.02), fontsize=10,
             color=MLGREEN, fontweight='bold', ha='center')

# Add values on bars
for bar, val in zip(bars1, weighted_ginis):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
             ha='center', fontsize=9)
for bar, val in zip(bars2, reductions):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
             ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
