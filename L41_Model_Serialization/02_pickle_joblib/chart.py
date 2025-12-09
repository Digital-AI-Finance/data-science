"""Pickle and Joblib for Model Saving"""
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
fig.suptitle('Pickle and Joblib for Model Saving', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Pickle basics
ax1 = axes[0, 0]
ax1.axis('off')

pickle_code = '''
PICKLE - PYTHON'S SERIALIZATION

WHAT IS PICKLE?
---------------
Built-in Python module for serializing
any Python object to bytes.


BASIC USAGE:
------------
import pickle

# Train your model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# SAVE MODEL
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 'wb' = write binary mode


# LOAD MODEL
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 'rb' = read binary mode


# USE LOADED MODEL
predictions = loaded_model.predict(X_test)


PICKLE PROTOCOLS:
-----------------
Protocol 0: ASCII (oldest, compatible)
Protocol 4: Default in Python 3.8+
Protocol 5: Best for large objects

pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


WARNING - SECURITY:
-------------------
NEVER load pickle files from untrusted sources!
Pickle can execute arbitrary code on load.
'''

ax1.text(0.02, 0.98, pickle_code, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Pickle Basics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Joblib
ax2 = axes[0, 1]
ax2.axis('off')

joblib_code = '''
JOBLIB - OPTIMIZED FOR LARGE ARRAYS

WHY JOBLIB?
-----------
- Optimized for numpy arrays
- Better compression
- Faster for sklearn models
- sklearn's recommended method


INSTALLATION:
-------------
pip install joblib


BASIC USAGE:
------------
import joblib

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# SAVE MODEL
joblib.dump(model, 'model.joblib')

# With compression (slower save, smaller file)
joblib.dump(model, 'model.joblib', compress=3)


# LOAD MODEL
loaded_model = joblib.load('model.joblib')


# USE LOADED MODEL
predictions = loaded_model.predict(X_test)


COMPRESSION OPTIONS:
--------------------
compress=0   No compression (default)
compress=3   Good balance (recommended)
compress=9   Maximum compression (slow)

# Or specify algorithm
joblib.dump(model, 'model.joblib', compress=('lz4', 3))


FILE SIZE COMPARISON:
---------------------
Model Type      | pickle | joblib | joblib+comp
----------------|--------|--------|------------
LogisticReg     | 2 KB   | 2 KB   | 1 KB
RandomForest    | 50 MB  | 45 MB  | 15 MB
Large ensemble  | 500 MB | 400 MB | 100 MB
'''

ax2.text(0.02, 0.98, joblib_code, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Joblib Basics', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Speed comparison
ax3 = axes[1, 0]

# Simulated benchmark data
operations = ['Small Model\nSave', 'Small Model\nLoad', 'Large Model\nSave', 'Large Model\nLoad']
pickle_times = [0.01, 0.005, 2.5, 1.8]
joblib_times = [0.008, 0.004, 1.5, 1.0]
joblib_comp_times = [0.015, 0.006, 2.0, 1.2]

x = np.arange(len(operations))
width = 0.25

bars1 = ax3.bar(x - width, pickle_times, width, label='pickle', color=MLBLUE, edgecolor='black')
bars2 = ax3.bar(x, joblib_times, width, label='joblib', color=MLGREEN, edgecolor='black')
bars3 = ax3.bar(x + width, joblib_comp_times, width, label='joblib (compressed)', color=MLORANGE, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(operations, fontsize=8)
ax3.set_ylabel('Time (seconds)')
ax3.set_title('Speed Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')
ax3.set_yscale('log')

# Add annotation
ax3.annotate('joblib faster\nfor large models', xy=(2, 1.5), xytext=(2.5, 0.5),
            fontsize=8, arrowprops=dict(arrowstyle='->', color=MLGREEN))

# Plot 4: Best practices
ax4 = axes[1, 1]
ax4.axis('off')

best_practices = '''
BEST PRACTICES

1. ALWAYS USE JOBLIB FOR SKLEARN:
---------------------------------
It's officially recommended by sklearn.

from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'model.joblib')


2. SAVE PREPROCESSING TOO:
--------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X, y)
joblib.dump(pipeline, 'full_pipeline.joblib')


3. CHECK VERSION COMPATIBILITY:
-------------------------------
# Save with version info
import sklearn
metadata = {
    'model': model,
    'sklearn_version': sklearn.__version__,
    'python_version': '3.9'
}
joblib.dump(metadata, 'model_with_meta.joblib')


4. USE MEANINGFUL FILENAMES:
----------------------------
# Bad
joblib.dump(model, 'model.joblib')

# Good
joblib.dump(model, 'rf_classifier_v2_2024-01-15.joblib')


5. VERIFY AFTER LOADING:
------------------------
loaded = joblib.load('model.joblib')
test_pred = loaded.predict(X_test[:5])
print(f"Model loaded, test predictions: {test_pred}")


6. HANDLE LARGE MODELS:
-----------------------
# For very large models, use memmap
loaded = joblib.load('large_model.joblib', mmap_mode='r')
'''

ax4.text(0.02, 0.98, best_practices, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Best Practices', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
