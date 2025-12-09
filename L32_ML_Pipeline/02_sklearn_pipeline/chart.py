"""sklearn Pipeline - Detailed Usage"""
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
fig.suptitle('sklearn Pipeline: Building Blocks', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Pipeline structure
ax1 = axes[0, 0]
ax1.axis('off')

structure = '''
PIPELINE STRUCTURE

from sklearn.pipeline import Pipeline, make_pipeline


METHOD 1: Named steps (recommended)
-----------------------------------
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier())
])


METHOD 2: make_pipeline (auto-naming)
-------------------------------------
pipe = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    PCA(n_components=10),
    RandomForestClassifier()
)
# Names: simpleimputer, standardscaler, pca, ...


ACCESSING STEPS:
----------------
pipe.named_steps['scaler']    # Get step
pipe.steps[1]                  # (name, estimator) tuple
pipe[1]                        # Direct indexing


STEP REQUIREMENTS:
------------------
All steps except the last: must have transform()
Last step: can be any estimator (fit/predict)
'''

ax1.text(0.02, 0.98, structure, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Pipeline Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Pipeline methods
ax2 = axes[0, 1]
ax2.axis('off')

methods = '''
PIPELINE METHODS

FIT:
----
pipe.fit(X_train, y_train)

Calls fit_transform on all steps except last
Calls fit on last step


TRANSFORM (if last step is transformer):
-----------------------------------------
X_transformed = pipe.transform(X)

Calls transform on ALL steps


PREDICT (if last step is predictor):
------------------------------------
y_pred = pipe.predict(X_test)

Calls transform on all but last
Calls predict on last step


PREDICT_PROBA:
--------------
y_prob = pipe.predict_proba(X_test)

Same as predict, but calls predict_proba


SCORE:
------
score = pipe.score(X_test, y_test)

Uses last step's scoring method


FIT_PREDICT:
------------
y_pred = pipe.fit_predict(X_train, y_train)

Fits then predicts on training data
'''

ax2.text(0.02, 0.98, methods, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Pipeline Methods', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: ColumnTransformer
ax3 = axes[1, 0]
ax3.axis('off')

column_trans = '''
COLUMNTRANSFORMER: Different Features, Different Processing

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define transformations for different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'income', 'amount']),
        ('cat', OneHotEncoder(), ['category', 'region'])
    ],
    remainder='drop'  # or 'passthrough'
)


# Use in pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


REMAINDER OPTIONS:
------------------
'drop': Discard unspecified columns
'passthrough': Keep unspecified columns as-is


USE CASE:
---------
- Numerical features: Scale
- Categorical features: One-hot encode
- Text features: Vectorize
- All in one pipeline!
'''

ax3.text(0.02, 0.98, column_trans, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('ColumnTransformer', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete example
ax4 = axes[1, 1]
ax4.axis('off')

example = '''
COMPLETE PIPELINE EXAMPLE

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Define columns
num_features = ['amount', 'age', 'balance']
cat_features = ['category', 'region']

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Full pipeline
full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y)
full_pipe.fit(X_train, y_train)
print(f"Test score: {full_pipe.score(X_test, y_test):.3f}")
'''

ax4.text(0.02, 0.98, example, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
