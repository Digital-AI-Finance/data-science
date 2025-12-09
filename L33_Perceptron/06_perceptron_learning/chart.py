"""Perceptron Learning Algorithm"""
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
fig.suptitle('Perceptron Learning Algorithm', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Learning algorithm
ax1 = axes[0, 0]
ax1.axis('off')

algorithm = '''
PERCEPTRON LEARNING ALGORITHM

GOAL: Find weights w, b that correctly
      classify all training points.


INITIALIZATION:
---------------
w = [0, 0, ..., 0]  (or random)
b = 0


FOR EACH EPOCH:
---------------
For each training point (x, y_true):

  1. Compute prediction:
     y_pred = step(w . x + b)

  2. If CORRECT (y_pred == y_true):
     Do nothing

  3. If WRONG (y_pred != y_true):
     Update weights:

     If y_true = 1, y_pred = 0 (missed positive):
        w = w + learning_rate * x
        b = b + learning_rate

     If y_true = 0, y_pred = 1 (false positive):
        w = w - learning_rate * x
        b = b - learning_rate


REPEAT until no errors (or max epochs)


LEARNING RATE:
--------------
Typically eta = 1 for perceptron
(or small value like 0.1)
'''

ax1.text(0.02, 0.98, algorithm, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Learning Algorithm', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Learning visualization
ax2 = axes[0, 1]

# Generate simple data
class0 = np.array([[-2, -1], [-1, -2], [-2, -2], [-1.5, -1.5]])
class1 = np.array([[1, 1], [2, 1], [1, 2], [1.5, 1.5]])

ax2.scatter(class0[:, 0], class0[:, 1], c=MLBLUE, s=100, label='Class 0', edgecolors='black')
ax2.scatter(class1[:, 0], class1[:, 1], c=MLRED, s=100, label='Class 1', edgecolors='black')

# Show evolution of decision boundary
boundaries = [
    ([0.1, 0.1], -0.1, 'Epoch 1', 'lightgray'),
    ([0.3, 0.2], 0.0, 'Epoch 5', 'gray'),
    ([0.5, 0.5], 0.2, 'Final', MLGREEN),
]

x_line = np.linspace(-3, 3, 100)
for w, b, label, color in boundaries:
    if w[1] != 0:
        y_line = -(w[0]/w[1]) * x_line - b/w[1]
        linestyle = '--' if color != MLGREEN else '-'
        ax2.plot(x_line, y_line, color=color, linewidth=2, linestyle=linestyle, label=label)

ax2.set_title('Decision Boundary Evolution', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)

# Plot 3: Update rule intuition
ax3 = axes[1, 0]
ax3.axis('off')

intuition = '''
INTUITION BEHIND THE UPDATE RULE

CASE 1: False Negative (y_true=1, y_pred=0)
-----------------------------------------
We MISSED a positive point.
w . x + b was too negative.

Fix: ADD x to w, ADD 1 to b
     Makes w . x larger for this point
     Pushes boundary to include it


CASE 2: False Positive (y_true=0, y_pred=1)
-----------------------------------------
We wrongly classified a negative point.
w . x + b was too positive.

Fix: SUBTRACT x from w, SUBTRACT 1 from b
     Makes w . x smaller for this point
     Pushes boundary away from it


GEOMETRIC VIEW:
---------------
Each update "pushes" the decision boundary
in the direction of the misclassified point.

False negative -> Push boundary toward point
False positive -> Push boundary away from point


THE BEAUTY:
-----------
Simple rule, provably converges!
(if data is linearly separable)
'''

ax3.text(0.02, 0.98, intuition, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Update Rule Intuition', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
PERCEPTRON IN PYTHON

import numpy as np

class Perceptron:
    def __init__(self, learning_rate=1.0, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs

    def fit(self, X, y):
        # Initialize weights
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.errors = []

        for epoch in range(self.n_epochs):
            n_errors = 0
            for xi, yi in zip(X, y):
                y_pred = self.predict_one(xi)
                error = yi - y_pred

                if error != 0:  # Misclassification
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    n_errors += 1

            self.errors.append(n_errors)
            if n_errors == 0:
                break

        return self

    def predict_one(self, x):
        return 1 if np.dot(x, self.w) + self.b >= 0 else 0

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# Usage
perceptron = Perceptron(learning_rate=1.0)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Python Implementation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
