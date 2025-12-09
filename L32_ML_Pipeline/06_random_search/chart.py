"""Randomized Search - Efficient Hyperparameter Tuning"""
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
fig.suptitle('Randomized Search: Efficient Hyperparameter Tuning', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Grid vs Random comparison
ax1 = axes[0, 0]

# Grid search points
grid_x = np.array([0.1, 0.5, 0.9])
grid_y = np.array([0.1, 0.5, 0.9])
gx, gy = np.meshgrid(grid_x, grid_y)
ax1.scatter(gx.flatten(), gy.flatten(), c=MLBLUE, s=100, marker='s',
            label='Grid Search (9 points)', edgecolors='black', zorder=3)

# Random search points
random_x = np.random.uniform(0, 1, 9)
random_y = np.random.uniform(0, 1, 9)
ax1.scatter(random_x, random_y, c=MLRED, s=100, marker='o',
            label='Random Search (9 points)', edgecolors='black', zorder=3)

# Optimal region
circle = plt.Circle((0.7, 0.3), 0.15, color=MLGREEN, alpha=0.3, label='Optimal region')
ax1.add_patch(circle)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel('Hyperparameter 1')
ax1.set_ylabel('Hyperparameter 2')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(alpha=0.3)
ax1.set_title('Grid vs Random Search Coverage', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Why random is often better
ax2 = axes[0, 1]
ax2.axis('off')

why_random = '''
WHY RANDOM SEARCH?

GRID SEARCH PROBLEM:
-------------------
- Exhaustive: tries ALL combinations
- Exponential: 5 params x 10 values = 100,000 combos!
- Wasteful: many redundant evaluations


RANDOM SEARCH ADVANTAGES:
-------------------------
1. EFFICIENCY
   Fixed budget (e.g., 100 iterations)
   regardless of number of parameters

2. COVERAGE
   Better explores continuous spaces
   Grid misses values between grid points

3. PROBABILITY
   With enough samples, likely to hit
   good regions of parameter space

4. FLEXIBILITY
   Easy to use probability distributions
   (uniform, log-uniform, etc.)


RESEARCH FINDING (Bergstra & Bengio, 2012):
-------------------------------------------
Random search finds good parameters
with fewer iterations than grid search
when some parameters matter more than others.


RULE OF THUMB:
--------------
60 random iterations is often enough
to find parameters within 5% of optimal.
'''

ax2.text(0.02, 0.98, why_random, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Why Random Search?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: RandomizedSearchCV code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
RANDOMIZEDSEARCHCV CODE

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform, randint

# Parameter distributions (not just lists!)
param_distributions = {
    'pca__n_components': randint(5, 50),     # Integer uniform
    'svc__C': loguniform(0.01, 100),         # Log-uniform
    'svc__gamma': loguniform(0.001, 1),      # Log-uniform
    'svc__kernel': ['rbf', 'linear', 'poly'] # Categorical
}


# Randomized search
random_search = RandomizedSearchCV(
    pipe,
    param_distributions,
    n_iter=100,           # Number of random samples
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)


# Results (same as GridSearchCV)
print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")


# Common distributions
from scipy.stats import uniform, loguniform, randint

uniform(0, 1)        # Uniform [0, 1]
loguniform(1e-4, 1)  # Log-uniform (good for learning rates)
randint(1, 100)      # Integer uniform [1, 100)
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('RandomizedSearchCV Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Comparison summary
ax4 = axes[1, 1]

# Simulated performance vs iterations
iterations = np.arange(1, 101)
grid_perf = 0.70 + 0.15 * (1 - np.exp(-iterations/30)) + np.random.randn(100) * 0.01
random_perf = 0.70 + 0.18 * (1 - np.exp(-iterations/20)) + np.random.randn(100) * 0.01

# Cumulative best
grid_best = np.maximum.accumulate(grid_perf)
random_best = np.maximum.accumulate(random_perf)

ax4.plot(iterations, grid_best, color=MLBLUE, linewidth=2, label='Grid Search (best so far)')
ax4.plot(iterations, random_best, color=MLRED, linewidth=2, label='Random Search (best so far)')

ax4.axhline(0.88, color=MLGREEN, linestyle='--', alpha=0.7, label='Optimal performance')

ax4.set_xlabel('Number of Iterations')
ax4.set_ylabel('Best Validation Score')
ax4.set_title('Convergence: Grid vs Random', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 100)
ax4.set_ylim(0.65, 0.92)

# Annotate
ax4.annotate('Random reaches\ngood solution faster!', xy=(40, 0.85), xytext=(60, 0.78),
             fontsize=9, arrowprops=dict(arrowstyle='->', color=MLRED))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
