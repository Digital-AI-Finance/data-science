"""ML Pipeline - Core Concept"""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ML Pipeline: Streamlining the Machine Learning Workflow', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is a pipeline?
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT IS AN ML PIPELINE?

DEFINITION:
-----------
A pipeline chains multiple data processing
steps into a single estimator.


WITHOUT PIPELINE (manual):
--------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


WITH PIPELINE (clean):
----------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)


BENEFITS:
---------
1. Cleaner code
2. Prevents data leakage
3. Easy cross-validation
4. Simple deployment
5. Reproducible workflow


COMMON STEPS:
-------------
1. Imputation (missing values)
2. Scaling (StandardScaler)
3. Feature selection
4. Dimensionality reduction (PCA)
5. Model (classifier/regressor)
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Pipeline Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Pipeline flow visualization
ax2 = axes[0, 1]
ax2.axis('off')

# Draw pipeline flow
steps = ['Raw\nData', 'Imputer', 'Scaler', 'PCA', 'Classifier', 'Predictions']
colors = [MLBLUE, MLORANGE, MLGREEN, MLPURPLE, MLRED, MLBLUE]

y_pos = 0.5
x_positions = np.linspace(0.05, 0.95, len(steps))

for i, (x, step, color) in enumerate(zip(x_positions, steps, colors)):
    # Draw box
    bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black')
    ax2.text(x, y_pos, step, ha='center', va='center', fontsize=10,
             fontweight='bold', color='white', bbox=bbox, transform=ax2.transAxes)

    # Draw arrow
    if i < len(steps) - 1:
        ax2.annotate('', xy=(x_positions[i+1]-0.06, y_pos), xytext=(x+0.06, y_pos),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Add fit/transform labels
ax2.text(0.5, 0.75, 'fit_transform() / transform()', ha='center', fontsize=10,
         transform=ax2.transAxes, style='italic')

ax2.text(0.5, 0.25, 'Pipeline chains all steps into ONE estimator', ha='center',
         fontsize=11, fontweight='bold', color=MLPURPLE, transform=ax2.transAxes)

ax2.set_title('Pipeline Flow', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Data leakage prevention
ax3 = axes[1, 0]
ax3.axis('off')

leakage = '''
DATA LEAKAGE PREVENTION

PROBLEM WITHOUT PIPELINE:
-------------------------
# WRONG! Scaling before split
scaler.fit(X)           # Uses ALL data
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled, ...)

# Test data influenced by training data!
# Model sees "future" information


CORRECT WITH PIPELINE:
----------------------
X_train, X_test = train_test_split(X, ...)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)  # Scaler fits ONLY on train
pipe.predict(X_test)        # Scaler transforms test

# No leakage! Test data properly isolated


IN CROSS-VALIDATION:
-------------------
Without pipeline: Scale all data -> split -> leak!
With pipeline: Split -> scale each fold -> no leak!


RULE:
-----
Any transformation that uses data statistics
MUST be inside the pipeline.
'''

ax3.text(0.02, 0.98, leakage, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Preventing Data Leakage', fontsize=11, fontweight='bold', color=MLRED)

# Plot 4: Basic code
ax4 = axes[1, 1]
ax4.axis('off')

code = '''
BASIC PIPELINE CODE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])


# Fit pipeline (fits all steps)
pipe.fit(X_train, y_train)


# Predict (transforms + predicts)
y_pred = pipe.predict(X_test)


# Get probability predictions
y_prob = pipe.predict_proba(X_test)


# Access individual steps
scaler = pipe.named_steps['scaler']
model = pipe.named_steps['classifier']


# Cross-validation with pipeline
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} +/- {scores.std():.3f}")
'''

ax4.text(0.02, 0.98, code, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Basic Code', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
