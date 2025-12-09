"""Documentation - Writing Clear Project Documentation"""
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
fig.suptitle('Documentation: Writing Clear Project Documentation', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: README template
ax1 = axes[0, 0]
ax1.axis('off')

readme = '''
README.MD TEMPLATE

# Project Title

Short description of what your project does.

## Live Demo
[Click here](https://your-app.streamlit.app)

## Features
- Feature 1: Description
- Feature 2: Description
- Feature 3: Description

## Screenshots
![Dashboard](screenshots/dashboard.png)

## Data Source
- Source: [Yahoo Finance](https://finance.yahoo.com)
- Period: 2020-2024
- Assets: AAPL, MSFT, GOOGL

## Models Used
1. Linear Regression (baseline)
2. Random Forest Classifier
3. Neural Network (MLP)

## Results
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Baseline | 0.65 | 0.62 |
| RF | 0.82 | 0.80 |
| MLP | 0.85 | 0.83 |

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author
Your Name - Course Name - Date
'''

ax1.text(0.02, 0.98, readme, transform=ax1.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('README Template', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Documentation quality levels
ax2 = axes[0, 1]

levels = ['Poor', 'Basic', 'Good', 'Excellent']
components = ['README', 'Code Comments', 'Docstrings', 'Usage Examples']

quality_matrix = np.array([
    [1, 2, 3, 4],  # README
    [1, 2, 3, 4],  # Comments
    [1, 2, 3, 4],  # Docstrings
    [1, 2, 3, 4]   # Examples
])

descriptions = [
    ['None', 'Title only', 'Key sections', 'Complete'],
    ['None', 'Few', 'Complex parts', 'All logic'],
    ['None', 'Main func', 'Most func', 'All func'],
    ['None', 'Basic', 'Multiple', 'Edge cases']
]

colors_map = [MLRED, MLORANGE, MLBLUE, MLGREEN]

for i, comp in enumerate(components):
    for j, level in enumerate(levels):
        color = colors_map[j]
        ax2.add_patch(plt.Rectangle((j*0.24+0.02, 0.75-i*0.2), 0.22, 0.15,
                                     facecolor=color, alpha=0.3 + j*0.15))
        ax2.text(j*0.24+0.13, 0.82-i*0.2, descriptions[i][j], fontsize=7, ha='center')

# Labels
for i, comp in enumerate(components):
    ax2.text(-0.05, 0.82-i*0.2, comp, fontsize=9, ha='right', fontweight='bold')
for j, level in enumerate(levels):
    ax2.text(j*0.24+0.13, 0.92, level, fontsize=9, ha='center', fontweight='bold',
             color=colors_map[j])

ax2.set_xlim(-0.15, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('Documentation Quality Levels', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Code comments examples
ax3 = axes[1, 0]
ax3.axis('off')

comments = '''
CODE COMMENTS EXAMPLES

FUNCTION DOCSTRING:
-------------------
def predict_stock_direction(df, model):
    """
    Predict stock price direction (up/down).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    model : sklearn model
        Trained classifier

    Returns:
    --------
    np.array
        Predictions (1=up, 0=down)
    """
    features = prepare_features(df)
    return model.predict(features)


INLINE COMMENTS:
----------------
# Calculate 20-day rolling volatility
df['volatility'] = df['returns'].rolling(20).std()

# Remove rows with missing values from feature calculation
df = df.dropna()

# Scale features to [0,1] range for neural network
X_scaled = scaler.fit_transform(X)


SECTION COMMENTS:
-----------------
# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================

# ... code ...

# ============================================
# MODEL TRAINING
# ============================================


WHEN TO COMMENT:
----------------
- Complex calculations
- Business logic decisions
- Non-obvious code
- Workarounds or hacks
- Important assumptions

WHEN NOT TO COMMENT:
--------------------
- Obvious code (x = x + 1)
- What the code does (use clear names)
- Commented-out code (delete it!)
'''

ax3.text(0.02, 0.98, comments, transform=ax3.transAxes, fontsize=7.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('Code Comments', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Documentation checklist
ax4 = axes[1, 1]

checklist_items = [
    'README.md with project description',
    'Installation instructions',
    'Usage examples',
    'Live demo URL',
    'Screenshots included',
    'Data source documented',
    'Model results table',
    'Limitations mentioned',
    'Code has docstrings',
    'Complex code has comments'
]

importance = [100, 90, 85, 95, 70, 80, 85, 75, 60, 70]
colors = [MLGREEN if v >= 85 else MLBLUE if v >= 70 else MLORANGE for v in importance]

y_pos = np.arange(len(checklist_items))
bars = ax4.barh(y_pos, importance, color=colors, alpha=0.7)

ax4.set_yticks(y_pos)
ax4.set_yticklabels(checklist_items, fontsize=8)
ax4.set_xlabel('Importance (%)')
ax4.set_xlim(0, 110)
ax4.set_title('Documentation Checklist', fontsize=11, fontweight='bold', color=MLPURPLE)

for bar, val in zip(bars, importance):
    ax4.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val}%',
             va='center', fontsize=8)

ax4.axvline(x=85, color='gray', linestyle='--', alpha=0.5)
ax4.text(86, 9.5, 'Essential', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
