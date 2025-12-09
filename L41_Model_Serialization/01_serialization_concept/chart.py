"""Model Serialization Concept - Why Save Models"""
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
fig.suptitle('Model Serialization: Why Save Models?', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: What is serialization
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
WHAT IS MODEL SERIALIZATION?

DEFINITION:
-----------
Converting a trained model from memory
into a format that can be stored on disk
and loaded later.

Also called: Model persistence, saving models


WHY SERIALIZE MODELS?
---------------------
1. AVOID RETRAINING
   Training can take hours/days.
   Save once, use many times.

2. DEPLOYMENT
   Move models from development to production.
   Different machines, same model.

3. REPRODUCIBILITY
   Keep exact model that produced results.
   Scientific integrity.

4. VERSION CONTROL
   Track model versions over time.
   Roll back if needed.

5. SHARING
   Share models with colleagues.
   Publish trained models.


WHAT GETS SAVED?
----------------
- Model architecture/structure
- Learned parameters (weights)
- Hyperparameters
- Preprocessing steps (sometimes)
- Training configuration


WHAT DOESN'T GET SAVED:
-----------------------
- Training data
- Training history (usually)
- GPU state
- Random seeds (depends)
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('What is Model Serialization?', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Visual workflow
ax2 = axes[0, 1]
ax2.axis('off')

# Draw workflow diagram
# Training phase
ax2.add_patch(plt.Rectangle((0.05, 0.7), 0.25, 0.2, facecolor=MLBLUE, alpha=0.3))
ax2.text(0.175, 0.8, 'TRAINING\n\nData + Algorithm\n= Trained Model', fontsize=9,
         ha='center', va='center')

# Serialization
ax2.annotate('', xy=(0.4, 0.8), xytext=(0.32, 0.8),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
ax2.text(0.36, 0.85, 'Save', fontsize=9, ha='center', color=MLORANGE)

# Disk storage
ax2.add_patch(plt.Rectangle((0.42, 0.7), 0.2, 0.2, facecolor=MLGREEN, alpha=0.3))
ax2.text(0.52, 0.8, 'DISK\n\nmodel.pkl\n(serialized)', fontsize=9,
         ha='center', va='center')

# Loading
ax2.annotate('', xy=(0.72, 0.8), xytext=(0.64, 0.8),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
ax2.text(0.68, 0.85, 'Load', fontsize=9, ha='center', color=MLORANGE)

# Deployment phase
ax2.add_patch(plt.Rectangle((0.74, 0.7), 0.22, 0.2, facecolor=MLPURPLE, alpha=0.3))
ax2.text(0.85, 0.8, 'PRODUCTION\n\nLoaded Model\n= Predictions', fontsize=9,
         ha='center', va='center')

# Time savings illustration
ax2.add_patch(plt.Rectangle((0.1, 0.2), 0.35, 0.35, facecolor=MLRED, alpha=0.2))
ax2.text(0.275, 0.45, 'WITHOUT SERIALIZATION:', fontsize=9, ha='center', fontweight='bold')
ax2.text(0.275, 0.35, 'Train: 2 hours\nEvery time you restart!', fontsize=9, ha='center')
ax2.text(0.275, 0.22, 'Total: 2+ hours each run', fontsize=8, ha='center', color=MLRED)

ax2.add_patch(plt.Rectangle((0.55, 0.2), 0.35, 0.35, facecolor=MLGREEN, alpha=0.2))
ax2.text(0.725, 0.45, 'WITH SERIALIZATION:', fontsize=9, ha='center', fontweight='bold')
ax2.text(0.725, 0.35, 'Train once: 2 hours\nLoad: <1 second', fontsize=9, ha='center')
ax2.text(0.725, 0.22, 'Total: <1 second!', fontsize=8, ha='center', color=MLGREEN)

ax2.set_xlim(0, 1)
ax2.set_ylim(0.1, 1)
ax2.set_title('Serialization Workflow', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: File formats
ax3 = axes[1, 0]
ax3.axis('off')

formats = '''
SERIALIZATION FORMATS

PICKLE (.pkl, .pickle):
-----------------------
Python's built-in serialization.
+ Easy to use
+ Handles any Python object
- Python-version specific
- Security risk (don't load untrusted!)

import pickle
pickle.dump(model, open('model.pkl', 'wb'))


JOBLIB (.joblib):
-----------------
Optimized for numpy arrays.
+ Faster for large arrays
+ Better compression
+ Handles large models well
- Still Python-specific

import joblib
joblib.dump(model, 'model.joblib')


ONNX (.onnx):
-------------
Open Neural Network Exchange.
+ Cross-platform
+ Works with many frameworks
+ Fast inference
- More complex setup

import onnx
onnx.save(model_onnx, 'model.onnx')


HDF5 (.h5):
-----------
Hierarchical Data Format.
+ Keras/TensorFlow native
+ Handles large models
+ Cross-platform
- Needs h5py library


RECOMMENDATION:
---------------
sklearn models: joblib
keras models: .h5 or SavedModel
Production: ONNX for deployment
'''

ax3.text(0.02, 0.98, formats, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('File Formats', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Format comparison
ax4 = axes[1, 1]

formats_names = ['pickle', 'joblib', 'ONNX', 'HDF5', 'SavedModel']
ease_of_use = [5, 4.5, 3, 4, 4]
speed = [3, 5, 5, 4, 4]
portability = [2, 2, 5, 4, 4]

x = np.arange(len(formats_names))
width = 0.25

bars1 = ax4.bar(x - width, ease_of_use, width, label='Ease of Use', color=MLBLUE, edgecolor='black')
bars2 = ax4.bar(x, speed, width, label='Save/Load Speed', color=MLGREEN, edgecolor='black')
bars3 = ax4.bar(x + width, portability, width, label='Portability', color=MLORANGE, edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(formats_names, fontsize=9)
ax4.set_ylabel('Score (1-5)')
ax4.set_title('Format Comparison', fontsize=11, fontweight='bold', color=MLPURPLE)
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(alpha=0.3, axis='y')
ax4.set_ylim(0, 6)

# Add best use case labels
use_cases = ['Quick\nprototype', 'sklearn\nmodels', 'Cross-\nplatform', 'Keras\nmodels', 'TensorFlow\nproduction']
for i, use in enumerate(use_cases):
    ax4.text(i, 0.3, use, fontsize=7, ha='center', va='bottom', style='italic')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
