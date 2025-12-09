"""Production Pipeline - Saving and Deploying Models"""
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
fig.suptitle('Production Pipeline: Saving and Deploying ML Models', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why pipelines for production
ax1 = axes[0, 0]
ax1.axis('off')

why = '''
WHY PIPELINES FOR PRODUCTION

PROBLEM WITHOUT PIPELINE:
-------------------------
# Training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model.fit(X_scaled, y_train)

# Deployment: MUST apply same preprocessing!
# Need to save: scaler, encoder, imputer, model...
# Easy to forget something or apply in wrong order


WITH PIPELINE:
--------------
# Training
pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipe.fit(X_train, y_train)

# Deployment: ONE object does everything
joblib.dump(pipe, 'model.joblib')

# In production
pipe = joblib.load('model.joblib')
prediction = pipe.predict(new_data)  # Just works!


BENEFITS:
---------
1. Single artifact to save/load
2. Guaranteed consistent preprocessing
3. Easier versioning and tracking
4. Simpler deployment
5. Reproducible predictions
'''

ax1.text(0.02, 0.98, why, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Why Pipelines for Production', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Save/Load code
ax2 = axes[0, 1]
ax2.axis('off')

save_load = '''
SAVING AND LOADING PIPELINES

OPTION 1: joblib (recommended for sklearn)
------------------------------------------
import joblib

# Save
joblib.dump(pipe, 'pipeline.joblib')

# Load
pipe = joblib.load('pipeline.joblib')


OPTION 2: pickle (standard library)
-----------------------------------
import pickle

# Save
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Load
with open('pipeline.pkl', 'rb') as f:
    pipe = pickle.load(f)


BEST PRACTICES:
---------------
1. Include sklearn version in filename
   pipeline_sklearn1.2.0.joblib

2. Save training metadata alongside
   metadata = {
       'features': feature_names,
       'target': 'price',
       'training_date': '2024-01-15',
       'cv_score': 0.85,
       'sklearn_version': sklearn.__version__
   }
   joblib.dump({'pipe': pipe, 'meta': metadata}, 'model.joblib')

3. Test loaded model before deployment
   loaded = joblib.load('model.joblib')
   assert loaded.predict(X_test[:5]).shape == (5,)
'''

ax2.text(0.02, 0.98, save_load, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Save/Load Pipelines', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Production workflow
ax3 = axes[1, 0]
ax3.axis('off')

# Draw workflow diagram
workflow_steps = [
    ('Train', MLBLUE, 0.15),
    ('Validate', MLORANGE, 0.35),
    ('Save', MLGREEN, 0.55),
    ('Deploy', MLRED, 0.75),
    ('Monitor', MLPURPLE, 0.95)
]

for name, color, x in workflow_steps:
    bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black')
    ax3.text(x, 0.7, name, ha='center', va='center', fontsize=10,
             fontweight='bold', color='white', bbox=bbox, transform=ax3.transAxes)

# Arrows
for i in range(len(workflow_steps) - 1):
    ax3.annotate('', xy=(workflow_steps[i+1][2]-0.08, 0.7),
                 xytext=(workflow_steps[i][2]+0.08, 0.7),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Descriptions
descriptions = [
    'Fit pipeline\non training data',
    'CV score,\ntest metrics',
    'joblib.dump()\nwith metadata',
    'API or\napplication',
    'Track\nperformance'
]

for i, (name, color, x) in enumerate(workflow_steps):
    ax3.text(x, 0.4, descriptions[i], ha='center', va='top', fontsize=8,
             transform=ax3.transAxes)

ax3.set_title('Production Workflow', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Complete example
ax4 = axes[1, 1]
ax4.axis('off')

example = '''
COMPLETE PRODUCTION EXAMPLE

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sklearn

# Define preprocessing
num_features = ['amount', 'days_since_last']
cat_features = ['category', 'region']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

# Full pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train and validate
pipe.fit(X_train, y_train)
cv_score = cross_val_score(pipe, X_train, y_train, cv=5).mean()

# Save with metadata
model_package = {
    'pipeline': pipe,
    'features': num_features + cat_features,
    'cv_score': cv_score,
    'sklearn_version': sklearn.__version__
}
joblib.dump(model_package, 'fraud_model_v1.joblib')
'''

ax4.text(0.02, 0.98, example, transform=ax4.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Complete Example', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
