"""Metadata Tracking - Recording Model Information"""
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
fig.suptitle('Metadata Tracking: Recording Model Information', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What metadata to track
ax1 = axes[0, 0]
ax1.axis('off')

metadata_types = '''
WHAT METADATA TO TRACK

ESSENTIAL METADATA:
-------------------
1. Model Information
   - Algorithm name
   - Hyperparameters
   - Feature names
   - Target variable

2. Training Information
   - Training date/time
   - Training data path
   - Number of samples
   - Train/test split

3. Performance Metrics
   - Accuracy, F1, AUC, etc.
   - Cross-validation scores
   - Training time

4. Environment
   - Python version
   - Library versions
   - Hardware used


NICE TO HAVE:
-------------
- Git commit hash
- Author/team
- Description/notes
- Data preprocessing steps
- Training logs
- Feature importance


EXAMPLE METADATA DICT:
----------------------
metadata = {
    "model_name": "stock_classifier",
    "version": "2.0.0",
    "algorithm": "RandomForestClassifier",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    },
    "features": ["price", "volume", "momentum"],
    "target": "direction",
    "training_date": "2024-01-15T14:30:00",
    "training_samples": 10000,
    "metrics": {
        "accuracy": 0.89,
        "f1_score": 0.87
    },
    "sklearn_version": "1.3.0"
}
'''

ax1.text(0.02, 0.98, metadata_types, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What Metadata to Track', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Saving metadata
ax2 = axes[0, 1]
ax2.axis('off')

saving_code = '''
SAVING METADATA

import json
import joblib
from datetime import datetime
import sklearn
import numpy as np


def create_metadata(model, X_train, y_train, metrics, params):
    \"\"\"Create comprehensive metadata dict.\"\"\"
    metadata = {
        # Model info
        'model_type': type(model).__name__,
        'hyperparameters': params,

        # Features
        'features': X_train.columns.tolist(),
        'n_features': X_train.shape[1],
        'target': 'direction',

        # Training info
        'training_date': datetime.now().isoformat(),
        'n_samples': len(X_train),

        # Performance
        'metrics': metrics,

        # Environment
        'sklearn_version': sklearn.__version__,
        'numpy_version': np.__version__
    }
    return metadata


def save_model_with_metadata(model, metadata, path):
    \"\"\"Save model and metadata together.\"\"\"
    # Save model
    joblib.dump(model, f'{path}/model.joblib')

    # Save metadata as JSON (human-readable)
    with open(f'{path}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save features separately for easy loading
    joblib.dump(metadata['features'], f'{path}/features.joblib')


# USAGE
metrics = {'accuracy': 0.89, 'f1': 0.87, 'auc': 0.91}
params = {'n_estimators': 100, 'max_depth': 10}

metadata = create_metadata(model, X_train, y_train, metrics, params)
save_model_with_metadata(model, metadata, 'models/v2')


# LOAD AND USE METADATA
with open('models/v2/metadata.json', 'r') as f:
    loaded_meta = json.load(f)

print(f"Model trained: {loaded_meta['training_date']}")
print(f"Accuracy: {loaded_meta['metrics']['accuracy']}")
'''

ax2.text(0.02, 0.98, saving_code, transform=ax2.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Saving Metadata', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Metadata file visualization
ax3 = axes[1, 0]
ax3.axis('off')

# Draw folder structure with metadata
ax3.text(0.5, 0.95, 'MODEL FOLDER STRUCTURE', fontsize=11, ha='center', fontweight='bold')

folder_structure = '''
models/
  stock_classifier/
    v2.0.0/
      model.joblib           <- Trained model
      features.joblib        <- Feature names list
      metadata.json          <- Human-readable info
      training_history.csv   <- Training metrics
      config.yaml            <- Hyperparameters
'''

ax3.text(0.1, 0.75, folder_structure, fontsize=10, fontfamily='monospace',
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Show metadata.json content
metadata_json = '''
metadata.json:
{
  "model_name": "stock_classifier",
  "version": "2.0.0",
  "algorithm": "RandomForestClassifier",
  "training_date": "2024-01-15T14:30:00",
  "n_samples": 10000,
  "n_features": 15,
  "metrics": {
    "accuracy": 0.89,
    "f1_score": 0.87,
    "auc_roc": 0.91
  },
  "sklearn_version": "1.3.0"
}
'''

ax3.text(0.55, 0.65, metadata_json, fontsize=8, fontfamily='monospace',
         verticalalignment='top', bbox=dict(facecolor=MLLAVENDER, alpha=0.8))

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title('Folder Structure', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Model registry concept
ax4 = axes[1, 1]

# Create table showing model registry
registry_data = [
    ['v1.0.0', '2024-01-01', '0.82', '0.80', 'Initial release'],
    ['v1.1.0', '2024-01-08', '0.85', '0.83', 'Added momentum'],
    ['v2.0.0', '2024-01-15', '0.89', '0.87', 'Retrained on 2024'],
    ['v2.0.1', '2024-01-20', '0.89', '0.88', 'Tuned thresholds']
]

columns = ['Version', 'Date', 'Accuracy', 'F1', 'Notes']

table = ax4.table(cellText=registry_data, colLabels=columns,
                  loc='center', cellLoc='center',
                  colColours=[MLLAVENDER]*5)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2)

# Highlight production version
for j in range(5):
    table[3, j].set_facecolor('#E8F5E9')  # Light green for v2.0.0

ax4.axis('off')
ax4.set_title('Model Registry (Production: v2.0.0)', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
