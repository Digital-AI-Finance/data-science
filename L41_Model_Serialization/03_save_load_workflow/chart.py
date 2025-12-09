"""Complete Save and Load Workflow"""
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
fig.suptitle('Complete Save and Load Workflow', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Training script
ax1 = axes[0, 0]
ax1.axis('off')

train_script = '''
TRAINING SCRIPT (train_model.py)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from datetime import datetime


# 1. LOAD DATA
df = pd.read_csv('stock_data.csv')
X = df.drop('target', axis=1)
y = df['target']


# 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. CREATE PIPELINE
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ))
])


# 4. TRAIN
pipeline.fit(X_train, y_train)


# 5. EVALUATE
train_acc = accuracy_score(y_train, pipeline.predict(X_train))
test_acc = accuracy_score(y_test, pipeline.predict(X_test))
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


# 6. SAVE MODEL
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
model_path = f'models/pipeline_{timestamp}.joblib'
joblib.dump(pipeline, model_path)
print(f"Model saved to: {model_path}")


# 7. SAVE FEATURE NAMES (important!)
feature_names = X.columns.tolist()
joblib.dump(feature_names, f'models/features_{timestamp}.joblib')
'''

ax1.text(0.02, 0.98, train_script, transform=ax1.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Training Script', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Prediction script
ax2 = axes[0, 1]
ax2.axis('off')

predict_script = '''
PREDICTION SCRIPT (predict.py)

import pandas as pd
import joblib
import sys


def load_model(model_path):
    \"\"\"Load saved model pipeline.\"\"\"
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)


def load_features(features_path):
    \"\"\"Load expected feature names.\"\"\"
    return joblib.load(features_path)


def predict(model, data, feature_names):
    \"\"\"Make predictions on new data.\"\"\"
    # Ensure correct column order
    data = data[feature_names]

    # Make predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    return predictions, probabilities


# MAIN USAGE
if __name__ == '__main__':
    # Load model
    model = load_model('models/pipeline_20240115_1430.joblib')
    features = load_features('models/features_20240115_1430.joblib')

    # Load new data
    new_data = pd.read_csv('new_stocks.csv')

    # Predict
    preds, probs = predict(model, new_data, features)

    # Output results
    results = new_data.copy()
    results['prediction'] = preds
    results['confidence'] = probs.max(axis=1)

    print(results[['prediction', 'confidence']])
    results.to_csv('predictions.csv', index=False)
'''

ax2.text(0.02, 0.98, predict_script, transform=ax2.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Prediction Script', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Visual workflow
ax3 = axes[1, 0]
ax3.axis('off')

# Draw development vs production workflow
# Development side
ax3.text(0.25, 0.95, 'DEVELOPMENT', fontsize=12, ha='center', fontweight='bold', color=MLBLUE)
ax3.add_patch(plt.Rectangle((0.05, 0.65), 0.4, 0.25, facecolor=MLBLUE, alpha=0.2))

steps_dev = ['1. Load training data', '2. Preprocess + train', '3. Evaluate model',
             '4. Save model + features']
for i, step in enumerate(steps_dev):
    ax3.text(0.25, 0.85 - i*0.05, step, fontsize=9, ha='center')

# Arrow to disk
ax3.annotate('', xy=(0.55, 0.77), xytext=(0.47, 0.77),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=3))
ax3.text(0.51, 0.82, 'joblib.dump()', fontsize=8, ha='center', color=MLORANGE)

# Disk
ax3.add_patch(plt.Circle((0.6, 0.77), 0.05, facecolor=MLGREEN, alpha=0.4))
ax3.text(0.6, 0.77, 'Disk', fontsize=9, ha='center')

# Arrow from disk
ax3.annotate('', xy=(0.72, 0.77), xytext=(0.65, 0.77),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=3))
ax3.text(0.69, 0.82, 'joblib.load()', fontsize=8, ha='center', color=MLORANGE)

# Production side
ax3.text(0.82, 0.95, 'PRODUCTION', fontsize=12, ha='center', fontweight='bold', color=MLGREEN)
ax3.add_patch(plt.Rectangle((0.62, 0.65), 0.35, 0.25, facecolor=MLGREEN, alpha=0.2))

steps_prod = ['1. Load saved model', '2. Receive new data', '3. Make predictions',
              '4. Return results']
for i, step in enumerate(steps_prod):
    ax3.text(0.8, 0.85 - i*0.05, step, fontsize=9, ha='center')

# Files saved
ax3.text(0.5, 0.55, 'FILES SAVED:', fontsize=10, ha='center', fontweight='bold')
files = ['models/\n  pipeline_v1.joblib\n  features_v1.joblib\n  metadata_v1.json']
ax3.text(0.5, 0.35, files[0], fontsize=9, ha='center', fontfamily='monospace',
         bbox=dict(facecolor='white', alpha=0.8))

ax3.set_xlim(0, 1)
ax3.set_ylim(0.15, 1)
ax3.set_title('Workflow Diagram', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Common pitfalls
ax4 = axes[1, 1]
ax4.axis('off')

pitfalls = '''
COMMON PITFALLS AND SOLUTIONS

1. FEATURE ORDER MISMATCH:
--------------------------
Problem: New data has different column order
Solution: Save and enforce feature names

# When saving
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'features.joblib')

# When loading
features = joblib.load('features.joblib')
X_new = X_new[features]  # Enforce order


2. MISSING PREPROCESSING:
-------------------------
Problem: Forgot to save scaler/encoder
Solution: Use sklearn Pipeline!

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
joblib.dump(pipeline, 'full_pipeline.joblib')


3. VERSION INCOMPATIBILITY:
---------------------------
Problem: Different sklearn versions
Solution: Document versions, use virtual environments

# Save version info
metadata = {
    'sklearn': sklearn.__version__,
    'numpy': np.__version__
}


4. LARGE FILE SIZES:
--------------------
Problem: Model files are huge
Solution: Use compression

joblib.dump(model, 'model.joblib', compress=3)


5. LOST MODEL FILES:
--------------------
Problem: Can't find the right model version
Solution: Use organized folder structure + git

models/
  v1/
    model.joblib
    features.joblib
    metrics.json
  v2/
    ...
'''

ax4.text(0.02, 0.98, pitfalls, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Common Pitfalls', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
