"""Common Preprocessing Steps in Pipelines"""
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
fig.suptitle('Common Preprocessing Steps in ML Pipelines', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Imputation
ax1 = axes[0, 0]
ax1.axis('off')

imputation = '''
1. IMPUTATION (Missing Values)

from sklearn.impute import SimpleImputer

# Strategy options
SimpleImputer(strategy='mean')      # Numerical
SimpleImputer(strategy='median')    # Robust to outliers
SimpleImputer(strategy='most_frequent')  # Categorical
SimpleImputer(strategy='constant', fill_value=0)


# KNN Imputation (smarter)
from sklearn.impute import KNNImputer
KNNImputer(n_neighbors=5)


WHEN TO USE:
------------
- mean: No outliers, roughly symmetric
- median: Outliers present, skewed data
- most_frequent: Categorical features
- KNN: Complex patterns in missingness


BEST PRACTICE:
--------------
- Always check % missing first
- Consider if missingness is informative
- Add indicator column for missing values:
  SimpleImputer(add_indicator=True)
'''

ax1.text(0.02, 0.98, imputation, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('1. Imputation', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Scaling
ax2 = axes[0, 1]
ax2.axis('off')

scaling = '''
2. SCALING (Normalization)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


STANDARDSCALER (most common):
-----------------------------
z = (x - mean) / std
Output: mean=0, std=1

StandardScaler()


MINMAXSCALER:
-------------
x_scaled = (x - min) / (max - min)
Output: range [0, 1]

MinMaxScaler()


ROBUSTSCALER:
-------------
Uses median and IQR instead
Robust to outliers

RobustScaler()


WHEN TO USE:
------------
StandardScaler: Linear models, neural networks
MinMaxScaler: When you need bounded range
RobustScaler: When outliers present


REMEMBER:
---------
Fit on training data ONLY!
That's why we use pipelines.
'''

ax2.text(0.02, 0.98, scaling, transform=ax2.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax2.set_title('2. Scaling', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 3: Encoding
ax3 = axes[1, 0]
ax3.axis('off')

encoding = '''
3. CATEGORICAL ENCODING

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)


ONEHOTENCODER (nominal categories):
-----------------------------------
Category -> Binary columns

OneHotEncoder(
    handle_unknown='ignore',  # For unseen categories
    sparse_output=False       # Dense array output
)

'Red', 'Blue', 'Green' -> [1,0,0], [0,1,0], [0,0,1]


ORDINALENCODER (ordered categories):
------------------------------------
Category -> Integer

OrdinalEncoder(categories=[['Low', 'Med', 'High']])

'Low', 'Med', 'High' -> 0, 1, 2


LABELENCODER (for target variable):
-----------------------------------
For y, not X

LabelEncoder()


USE IN COLUMNTRANSFORMER:
-------------------------
ColumnTransformer([
    ('cat', OneHotEncoder(), ['color', 'size']),
    ('ord', OrdinalEncoder(), ['rating']),
    ('num', StandardScaler(), ['price', 'qty'])
])
'''

ax3.text(0.02, 0.98, encoding, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('3. Encoding', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Feature selection & transformation
ax4 = axes[1, 1]
ax4.axis('off')

other = '''
4. FEATURE SELECTION & TRANSFORMATION

FEATURE SELECTION:
------------------
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    RFE
)

# Select top k features
SelectKBest(k=10)

# Recursive feature elimination
from sklearn.feature_selection import RFE
RFE(estimator=LogisticRegression(), n_features_to_select=10)


DIMENSIONALITY REDUCTION:
-------------------------
from sklearn.decomposition import PCA

PCA(n_components=10)       # Keep 10 components
PCA(n_components=0.95)     # Keep 95% variance


POLYNOMIAL FEATURES:
-------------------
from sklearn.preprocessing import PolynomialFeatures

PolynomialFeatures(degree=2, include_bias=False)

[a, b] -> [a, b, a^2, ab, b^2]


TYPICAL PIPELINE ORDER:
-----------------------
1. Imputation
2. Encoding
3. Scaling
4. Feature selection / PCA
5. Model
'''

ax4.text(0.02, 0.98, other, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('4. Selection & Transformation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
