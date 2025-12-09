"""Perceptron Convergence Theorem"""
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
fig.suptitle('Perceptron Convergence: Guaranteed to Learn!', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Convergence theorem
ax1 = axes[0, 0]
ax1.axis('off')

theorem = '''
PERCEPTRON CONVERGENCE THEOREM

THEOREM (Novikoff, 1962):
-------------------------
If the training data is linearly separable,
the perceptron algorithm will converge
to a solution in a finite number of steps.


REQUIREMENTS:
-------------
1. Data must be linearly separable
2. Learning rate > 0
3. Enough iterations


BOUND ON ITERATIONS:
-------------------
Let R = max ||x_i|| (max distance of points)
Let gamma = margin (distance to boundary)

Number of updates <= (R / gamma)^2


IMPLICATIONS:
-------------
+ GUARANTEED convergence (if separable)
+ Simple algorithm, works!
+ Foundation for all neural networks

- Doesn't work if NOT separable
- No unique solution (depends on order)
- Doesn't maximize margin


WHAT IF NOT SEPARABLE?
----------------------
Algorithm never converges!
Will keep making updates forever.

Solution: Use variant like
- Pocket algorithm
- Averaged perceptron
- Or use different model
'''

ax1.text(0.02, 0.98, theorem, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Convergence Theorem', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Convergence plot (errors over epochs)
ax2 = axes[0, 1]

# Simulated error curve
epochs = np.arange(1, 21)
errors_separable = np.array([10, 7, 5, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
errors_not_sep = np.array([10, 8, 6, 5, 4, 5, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3])

ax2.plot(epochs, errors_separable, 'o-', color=MLGREEN, linewidth=2, markersize=6,
         label='Linearly separable')
ax2.plot(epochs, errors_not_sep, 'o-', color=MLRED, linewidth=2, markersize=6,
         label='NOT separable')

ax2.axhline(0, color='black', linewidth=0.5)

ax2.set_title('Convergence: Separable vs Non-Separable', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Number of Errors')
ax2.legend()
ax2.grid(alpha=0.3)

# Annotate
ax2.annotate('Converged!', xy=(9, 0), xytext=(12, 2),
             fontsize=10, color=MLGREEN, arrowprops=dict(arrowstyle='->', color=MLGREEN))
ax2.annotate('Never\nconverges', xy=(20, 3), xytext=(17, 6),
             fontsize=10, color=MLRED, arrowprops=dict(arrowstyle='->', color=MLRED))

# Plot 3: Margin concept
ax3 = axes[1, 0]

# Generate well-separated data
class0 = np.random.randn(20, 2) * 0.4 + [-1.5, 0]
class1 = np.random.randn(20, 2) * 0.4 + [1.5, 0]

ax3.scatter(class0[:, 0], class0[:, 1], c=MLBLUE, s=50, edgecolors='black')
ax3.scatter(class1[:, 0], class1[:, 1], c=MLRED, s=50, edgecolors='black')

# Decision boundary
ax3.axvline(0, color=MLGREEN, linewidth=2, label='Decision boundary')

# Margin lines
ax3.axvline(-0.8, color=MLGREEN, linewidth=1, linestyle='--', alpha=0.7)
ax3.axvline(0.8, color=MLGREEN, linewidth=1, linestyle='--', alpha=0.7)

# Draw margin arrow
ax3.annotate('', xy=(0.8, 1.5), xytext=(0, 1.5),
             arrowprops=dict(arrowstyle='<->', color=MLORANGE, lw=2))
ax3.text(0.4, 1.7, 'margin', fontsize=10, color=MLORANGE, fontweight='bold')

ax3.set_title('Margin: Distance to Nearest Points', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.grid(alpha=0.3)
ax3.set_xlim(-3, 3)
ax3.set_ylim(-2, 2)

ax3.text(-2.5, -1.5, 'Larger margin\n= Faster convergence', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 4: sklearn perceptron
ax4 = axes[1, 1]
ax4.axis('off')

sklearn_code = '''
PERCEPTRON IN SKLEARN

from sklearn.linear_model import Perceptron

# Create perceptron
model = Perceptron(
    penalty=None,           # No regularization
    alpha=0.0001,           # Regularization strength
    fit_intercept=True,     # Learn bias term
    max_iter=1000,          # Max epochs
    tol=1e-3,               # Stopping tolerance
    shuffle=True,           # Shuffle each epoch
    random_state=42
)

# Fit
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Access learned parameters
weights = model.coef_        # Shape: (1, n_features)
bias = model.intercept_      # Shape: (1,)


# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Note: sklearn's Perceptron uses
# hinge loss variant for online learning
# Slightly different from classic perceptron


# For more control, use:
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='perceptron')
'''

ax4.text(0.02, 0.98, sklearn_code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('sklearn Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
