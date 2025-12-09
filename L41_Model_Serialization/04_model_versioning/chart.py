"""Model Versioning - Tracking Model Changes"""
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
fig.suptitle('Model Versioning: Tracking Model Changes', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why version models
ax1 = axes[0, 0]
ax1.axis('off')

why_version = '''
WHY VERSION YOUR MODELS?

SCENARIO:
---------
Monday: Model works great, deploy to production
Tuesday: Train new model, deploy update
Wednesday: Users complain - predictions are wrong!
Thursday: "Which model was working on Monday??"


WITHOUT VERSIONING:
-------------------
- Can't rollback to previous model
- Don't know what changed
- Can't reproduce results
- Debugging nightmare


WITH VERSIONING:
----------------
+ Instant rollback to any version
+ Track what changed and when
+ Reproduce any result
+ Compare model performance
+ Audit trail for compliance


WHAT TO VERSION:
----------------
1. Model file (.joblib)
2. Training code (git)
3. Training data reference
4. Hyperparameters
5. Performance metrics
6. Feature list
7. Dependencies (requirements.txt)


VERSIONING STRATEGIES:
----------------------
1. Simple: Timestamp in filename
2. Better: Semantic versioning (v1.2.3)
3. Best: Full MLOps (MLflow, DVC)
'''

ax1.text(0.02, 0.98, why_version, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Why Version Models?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Versioning scheme
ax2 = axes[0, 1]
ax2.axis('off')

versioning = '''
VERSIONING SCHEMES

1. TIMESTAMP-BASED:
-------------------
model_20240115_143022.joblib
model_20240116_091545.joblib

Pros: Simple, automatic ordering
Cons: Not semantic, long names


2. SEMANTIC VERSIONING:
-----------------------
model_v1.0.0.joblib  # Initial release
model_v1.0.1.joblib  # Bug fix
model_v1.1.0.joblib  # New feature
model_v2.0.0.joblib  # Breaking change

MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes


3. HYBRID APPROACH (RECOMMENDED):
---------------------------------
models/
  v1/
    model_v1.0.0_20240115.joblib
    metadata.json
  v2/
    model_v2.0.0_20240120.joblib
    metadata.json


4. FOLDER STRUCTURE:
--------------------
models/
  stock_classifier/
    v1.0.0/
      model.joblib
      features.joblib
      metrics.json
      config.yaml
    v1.1.0/
      ...
    latest -> v1.1.0  # symlink


CODE EXAMPLE:
-------------
import os
from datetime import datetime

def save_model_versioned(model, name, version, base_dir='models'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    version_dir = f"{base_dir}/{name}/v{version}"
    os.makedirs(version_dir, exist_ok=True)

    model_path = f"{version_dir}/model_{timestamp}.joblib"
    joblib.dump(model, model_path)
    return model_path
'''

ax2.text(0.02, 0.98, versioning, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Versioning Schemes', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Version timeline visualization
ax3 = axes[1, 0]

# Create version timeline
versions = ['v1.0.0', 'v1.0.1', 'v1.1.0', 'v2.0.0', 'v2.0.1']
dates = [0, 1, 3, 6, 7]
accuracy = [0.82, 0.82, 0.85, 0.88, 0.89]
colors = [MLBLUE, MLBLUE, MLGREEN, MLORANGE, MLORANGE]

# Plot timeline
ax3.scatter(dates, accuracy, c=colors, s=200, zorder=5, edgecolor='black')

# Connect with lines
ax3.plot(dates, accuracy, color='gray', linestyle='--', alpha=0.5)

# Add version labels
for d, v, a in zip(dates, versions, accuracy):
    ax3.annotate(f'{v}\n{a:.0%}', xy=(d, a), xytext=(d, a+0.03),
                fontsize=9, ha='center', fontweight='bold')

# Add events
events = [
    (0, 'Initial\nrelease'),
    (1, 'Bug\nfix'),
    (3, 'New\nfeatures'),
    (6, 'Retrained\non 2024 data'),
    (7, 'Hyperparameter\ntuning')
]
for d, event in events:
    ax3.text(d, 0.77, event, fontsize=7, ha='center', va='top', style='italic')

ax3.set_xlabel('Weeks Since Initial Release')
ax3.set_ylabel('Test Accuracy')
ax3.set_title('Model Version Timeline', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.set_ylim(0.75, 0.95)
ax3.grid(alpha=0.3)

# Add legend for version types
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=MLBLUE, label='v1.0.x (Bug fixes)'),
    Patch(facecolor=MLGREEN, label='v1.1.x (Features)'),
    Patch(facecolor=MLORANGE, label='v2.x.x (Major)')
]
ax3.legend(handles=legend_elements, loc='lower right', fontsize=8)

# Plot 4: Version comparison
ax4 = axes[1, 1]

# Compare different versions
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
v1_scores = [0.82, 0.80, 0.78, 0.79, 0.85]
v2_scores = [0.89, 0.87, 0.86, 0.865, 0.91]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x - width/2, v1_scores, width, label='v1.0.0', color=MLBLUE, edgecolor='black')
bars2 = ax4.bar(x + width/2, v2_scores, width, label='v2.0.0', color=MLGREEN, edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.set_ylabel('Score')
ax4.set_title('Version Performance Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=9)
ax4.set_ylim(0.7, 1.0)
ax4.grid(alpha=0.3, axis='y')

# Add improvement annotations
for i, (v1, v2) in enumerate(zip(v1_scores, v2_scores)):
    improvement = (v2 - v1) / v1 * 100
    ax4.text(i, v2 + 0.02, f'+{improvement:.0f}%', fontsize=8, ha='center', color=MLGREEN)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
