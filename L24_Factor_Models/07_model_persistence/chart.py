"""Model Persistence - Saving and loading models"""
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
fig.suptitle('Model Persistence: Save, Load, and Deploy', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why persistence matters
ax1 = axes[0, 0]
ax1.axis('off')

why_persist = '''
WHY SAVE MODELS?

1. REPRODUCIBILITY
   - Same predictions every time
   - Audit trail for compliance
   - Version control models

2. DEPLOYMENT
   - Train once, predict many times
   - No need to re-fit on server
   - Faster predictions

3. SHARING
   - Share models with team
   - Model registry (MLflow)
   - API serving

COMMON FORMATS:

pickle / joblib
---------------
+ Fast, easy
+ Works with sklearn
- Python version sensitive
- Security risk (arbitrary code)

ONNX
----
+ Framework agnostic
+ Production-ready
- More complex setup

JSON (coefficients only)
------------------------
+ Human readable
+ Framework agnostic
- Manual reconstruction
'''

ax1.text(0.02, 0.98, why_persist, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Why Model Persistence?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: File size comparison
ax2 = axes[0, 1]

methods = ['pickle', 'joblib', 'joblib\n(compressed)', 'ONNX', 'JSON\n(coefs)']
sizes = [45.2, 45.5, 12.3, 8.7, 0.5]  # KB
speeds = [1.0, 1.2, 3.5, 2.1, 0.3]  # Relative save time

x = np.arange(len(methods))
width = 0.35

ax2_twin = ax2.twinx()

bars1 = ax2.bar(x - width/2, sizes, width, label='File Size (KB)', color=MLBLUE, edgecolor='black', linewidth=0.5)
bars2 = ax2_twin.bar(x + width/2, speeds, width, label='Save Time (rel.)', color=MLORANGE, edgecolor='black', linewidth=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=9)
ax2.set_ylabel('File Size (KB)', fontsize=10, color=MLBLUE)
ax2_twin.set_ylabel('Save Time (relative)', fontsize=10, color=MLORANGE)
ax2.set_title('Storage Format Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)

ax2.legend(loc='upper left', fontsize=8)
ax2_twin.legend(loc='upper right', fontsize=8)
ax2.grid(alpha=0.3, axis='y')

# Plot 3: pickle/joblib code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
Saving and Loading sklearn Models

import pickle
import joblib
from sklearn.linear_model import Ridge

# ===================
# METHOD 1: pickle
# ===================
# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# ===================
# METHOD 2: joblib (recommended for sklearn)
# ===================
# Save (uncompressed)
joblib.dump(model, 'model.joblib')

# Save (compressed - smaller file)
joblib.dump(model, 'model.joblib.gz', compress=3)

# Load
loaded_model = joblib.load('model.joblib')

# ===================
# Use loaded model
# ===================
predictions = loaded_model.predict(X_new)

# Verify it's the same
assert np.allclose(model.coef_, loaded_model.coef_)
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('pickle & joblib Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Best practices
ax4 = axes[1, 1]
ax4.axis('off')

best_practices = '''
MODEL PERSISTENCE BEST PRACTICES

1. VERSION EVERYTHING
   model_v1.0.0_2024-01-15.joblib
   - Include version number
   - Include training date
   - Track with git/DVC

2. SAVE METADATA
   {
     "model_type": "Ridge",
     "features": ["MKT", "SMB", "HML"],
     "train_r2": 0.73,
     "train_date": "2024-01-15",
     "sklearn_version": "1.3.0",
     "python_version": "3.10.12"
   }

3. SAVE PREPROCESSING TOO
   - StandardScaler parameters
   - Feature names and order
   - Or use Pipeline (saves everything)

4. SECURITY
   - Never load untrusted pickles!
   - Pickle can execute arbitrary code
   - Use joblib with trusted sources only

5. TESTING
   # After loading:
   assert model.feature_names_in_ == expected_features
   assert model.predict(X_test[:1]).shape == (1,)
'''

ax4.text(0.02, 0.98, best_practices, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Best Practices', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
