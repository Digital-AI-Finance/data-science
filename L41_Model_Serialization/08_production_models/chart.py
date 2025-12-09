"""Production Models - Finance Application"""
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
fig.suptitle('Production Models: Finance Application', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Production checklist
ax1 = axes[0, 0]
ax1.axis('off')

checklist = '''
PRODUCTION MODEL CHECKLIST

BEFORE DEPLOYMENT:
------------------
[x] Model saved with joblib/keras
[x] Preprocessing pipeline included
[x] Feature names documented
[x] Metadata saved (metrics, version)
[x] Version pinned in requirements.txt
[x] Model tested on hold-out data
[x] Predictions verified manually


DEPLOYMENT ARTIFACTS:
---------------------
production/
  stock_classifier/
    model.joblib          <- Trained model
    features.joblib       <- Expected features
    metadata.json         <- Model info
    requirements.txt      <- Dependencies
    predict.py            <- Prediction script
    test_predictions.py   <- Validation tests


REQUIREMENTS.TXT:
-----------------
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2


VALIDATION TESTS:
-----------------
def test_model_loads():
    model = joblib.load('model.joblib')
    assert model is not None

def test_prediction_shape():
    X_sample = pd.DataFrame(...)
    pred = model.predict(X_sample)
    assert len(pred) == len(X_sample)

def test_prediction_values():
    pred = model.predict(known_input)
    assert pred[0] == expected_output


RUN BEFORE DEPLOY:
------------------
pytest test_predictions.py -v
'''

ax1.text(0.02, 0.98, checklist, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Production Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Complete production code
ax2 = axes[0, 1]
ax2.axis('off')

production_code = '''
PRODUCTION PREDICTION SERVICE

import joblib
import pandas as pd
from pathlib import Path
import json


class StockPredictor:
    \"\"\"Production stock prediction service.\"\"\"

    def __init__(self, model_dir='production/stock_classifier'):
        self.model_dir = Path(model_dir)
        self.model = None
        self.features = None
        self.metadata = None
        self._load()


    def _load(self):
        \"\"\"Load model and supporting files.\"\"\"
        self.model = joblib.load(self.model_dir / 'model.joblib')
        self.features = joblib.load(self.model_dir / 'features.joblib')

        with open(self.model_dir / 'metadata.json') as f:
            self.metadata = json.load(f)

        print(f"Loaded model v{self.metadata['version']}")


    def predict(self, data: pd.DataFrame) -> dict:
        \"\"\"Make predictions on new data.\"\"\"
        # Validate features
        missing = set(self.features) - set(data.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Ensure correct order
        X = data[self.features]

        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'model_version': self.metadata['version']
        }


# USAGE
predictor = StockPredictor()
result = predictor.predict(new_stock_data)
print(result['predictions'])
'''

ax2.text(0.02, 0.98, production_code, transform=ax2.transAxes, fontsize=7,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('Production Prediction Service', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Model lifecycle visualization
ax3 = axes[1, 0]
ax3.axis('off')

# Draw model lifecycle
stages = [
    ('DEVELOP', MLBLUE, 0.1, 'Train\nEvaluate\nTest'),
    ('SAVE', MLORANGE, 0.3, 'joblib.dump()\nMetadata\nVersion'),
    ('VALIDATE', MLGREEN, 0.5, 'Unit tests\nIntegration\nStaging'),
    ('DEPLOY', MLPURPLE, 0.7, 'Production\nMonitor\nAlert'),
    ('MAINTAIN', MLRED, 0.9, 'Retrain\nUpdate\nRollback')
]

for name, color, x, desc in stages:
    ax3.add_patch(plt.Circle((x, 0.7), 0.08, facecolor=color, alpha=0.4))
    ax3.text(x, 0.7, name, fontsize=8, ha='center', va='center', fontweight='bold')
    ax3.text(x, 0.45, desc, fontsize=7, ha='center', va='top')

# Arrows
for i in range(len(stages)-1):
    ax3.annotate('', xy=(stages[i+1][2]-0.1, 0.7), xytext=(stages[i][2]+0.1, 0.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

# Feedback loop
ax3.annotate('', xy=(0.15, 0.55), xytext=(0.85, 0.55),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1,
                          connectionstyle='arc3,rad=-0.3'))
ax3.text(0.5, 0.25, 'Feedback Loop: Monitor -> Retrain -> Deploy', fontsize=9, ha='center')

ax3.set_xlim(0, 1)
ax3.set_ylim(0.1, 0.9)
ax3.set_title('Model Lifecycle', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Finance-specific considerations
ax4 = axes[1, 1]
ax4.axis('off')

finance_specific = '''
FINANCE-SPECIFIC CONSIDERATIONS

DATA SENSITIVITY:
-----------------
- Never save training data with model
- Exclude proprietary features from logs
- Secure model files (encryption)


REGULATORY COMPLIANCE:
----------------------
- Audit trail for all predictions
- Model explainability required
- Document model decisions
- Version everything


MONITORING IN PRODUCTION:
-------------------------
# Log every prediction
def predict_with_logging(model, X, request_id):
    pred = model.predict(X)

    log_entry = {
        'request_id': request_id,
        'timestamp': datetime.now().isoformat(),
        'model_version': model.version,
        'input_hash': hash(X.tobytes()),
        'prediction': pred.tolist()
    }
    logger.info(json.dumps(log_entry))

    return pred


MODEL DRIFT DETECTION:
----------------------
Track prediction distribution over time.
Alert if significant shift detected.

# Check distribution
recent_preds = get_predictions(last_7_days)
baseline_preds = get_predictions(training_period)

# Statistical test
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(recent_preds, baseline_preds)

if p_value < 0.05:
    alert("Model drift detected!")


ROLLBACK STRATEGY:
------------------
- Keep last 3 model versions
- Instant rollback capability
- A/B testing before full deploy
'''

ax4.text(0.02, 0.98, finance_specific, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Finance Considerations', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
