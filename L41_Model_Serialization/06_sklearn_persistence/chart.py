"""Sklearn Model Persistence - Official Methods"""
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
fig.suptitle('Sklearn Model Persistence: Official Methods', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Sklearn persistence overview
ax1 = axes[0, 0]
ax1.axis('off')

overview = '''
SKLEARN MODEL PERSISTENCE

OFFICIAL RECOMMENDATION:
------------------------
sklearn recommends JOBLIB for model persistence.

"For sklearn models, use joblib's dump & load
 for better performance with large numpy arrays."
 - sklearn documentation


BASIC PATTERN:
--------------
from sklearn.ensemble import RandomForestClassifier
import joblib

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'model.joblib')

# Load
model = joblib.load('model.joblib')


SAVING FULL PIPELINES:
----------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'pipeline.joblib')

# Loads with preprocessing included!
pipeline = joblib.load('pipeline.joblib')
predictions = pipeline.predict(X_new)


KEY INSIGHT:
------------
Save PIPELINES, not just models.
Preprocessing must match exactly!
'''

ax1.text(0.02, 0.98, overview, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Sklearn Persistence Overview', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Different model types
ax2 = axes[0, 1]
ax2.axis('off')

model_types = '''
SAVING DIFFERENT MODEL TYPES

LINEAR MODELS:
--------------
from sklearn.linear_model import LogisticRegression, Ridge

model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, 'logistic.joblib')

# Coefficients are saved automatically
loaded = joblib.load('logistic.joblib')
print(loaded.coef_)  # Works!


TREE MODELS:
------------
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
joblib.dump(rf, 'random_forest.joblib', compress=3)

# Can be large! Use compression


SVM MODELS:
-----------
from sklearn.svm import SVC

svm = SVC(kernel='rbf')
svm.fit(X, y)
joblib.dump(svm, 'svm.joblib')


COMPLEX PIPELINES:
------------------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X, y)
joblib.dump(pipeline, 'complex_pipeline.joblib')

# ALL transformers saved and loaded together!
'''

ax2.text(0.02, 0.98, model_types, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Different Model Types', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Model sizes
ax3 = axes[1, 0]

model_names = ['Logistic\nRegression', 'SVM\n(RBF)', 'Random\nForest\n(100)', 'Gradient\nBoosting', 'XGBoost\n(100)']
sizes_mb = [0.002, 5.2, 45, 12, 8]
load_times_ms = [1, 50, 200, 80, 60]

x = np.arange(len(model_names))
width = 0.35

ax3_size = ax3
ax3_time = ax3.twinx()

bars1 = ax3_size.bar(x - width/2, sizes_mb, width, label='File Size (MB)', color=MLBLUE, edgecolor='black')
bars2 = ax3_time.bar(x + width/2, load_times_ms, width, label='Load Time (ms)', color=MLGREEN, edgecolor='black')

ax3_size.set_xticks(x)
ax3_size.set_xticklabels(model_names, fontsize=8)
ax3_size.set_ylabel('File Size (MB)', color=MLBLUE)
ax3_time.set_ylabel('Load Time (ms)', color=MLGREEN)
ax3_size.set_title('Model Size & Load Time', fontsize=11, fontweight='bold', color=MLPURPLE)

ax3_size.set_yscale('log')
ax3_time.set_yscale('log')

lines1, labels1 = ax3_size.get_legend_handles_labels()
lines2, labels2 = ax3_time.get_legend_handles_labels()
ax3_size.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

ax3_size.grid(alpha=0.3, axis='y')

# Plot 4: Version compatibility
ax4 = axes[1, 1]
ax4.axis('off')

compatibility = '''
VERSION COMPATIBILITY

THE PROBLEM:
------------
Model saved in sklearn 1.2 may not load in sklearn 1.4!

sklearn.__version__: 1.2.0
UserWarning: Trying to unpickle estimator
from version 1.2.0 when using version 1.4.0


BEST PRACTICES:
---------------
1. Pin sklearn version in requirements.txt
   scikit-learn==1.3.0

2. Save version info with model
   metadata['sklearn_version'] = sklearn.__version__

3. Test loading after sklearn upgrades

4. Use virtual environments


CHECKING COMPATIBILITY:
-----------------------
import warnings
import sklearn
import joblib

# Load with warning capture
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model = joblib.load('old_model.joblib')

    if len(w) > 0:
        print(f"Warning: {w[0].message}")
        print("Model may not work correctly!")


ONNX FOR PORTABILITY:
---------------------
For cross-version compatibility, consider ONNX:

from skl2onnx import convert_sklearn

onnx_model = convert_sklearn(
    model,
    initial_types=[('input', FloatTensorType([None, n_features]))]
)
onnx.save(onnx_model, 'model.onnx')

# ONNX models work across versions and languages!


MIGRATION STRATEGY:
-------------------
1. Keep old model + metadata
2. Retrain on new sklearn version
3. Compare predictions
4. Deploy new model only if equivalent
'''

ax4.text(0.02, 0.98, compatibility, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Version Compatibility', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
