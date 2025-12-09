"""Time Series Split - Temporal Cross-Validation"""
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
fig.suptitle('Time Series Cross-Validation: Respecting Temporal Order', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Why standard CV fails for time series
ax1 = axes[0, 0]
ax1.axis('off')

problem = '''
WHY STANDARD CV FAILS FOR TIME SERIES

PROBLEM: Look-Ahead Bias
-------------------------
Standard K-Fold randomly shuffles data.
This mixes past and future observations!

Example (stock prediction):
- Fold 1 train: Jan, Mar, May
- Fold 1 test: Feb, Apr

Model trains on FUTURE (Mar, May)
to predict PAST (Feb)!

This is cheating - unrealistic performance.


FINANCIAL DATA ISSUES:
----------------------
1. Autocorrelation
   Today's price depends on yesterday's

2. Regime changes
   Market conditions evolve over time

3. Information leakage
   Future information shouldn't influence past


SOLUTION: Time Series Split
---------------------------
Always train on PAST, test on FUTURE.
Never look ahead!


RULE:
-----
For any temporal data (stocks, weather, sales),
use TimeSeriesSplit, not KFold.
'''

ax1.text(0.02, 0.98, problem, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('The Problem with Standard CV', fontsize=11, fontweight='bold', color=MLRED)

# Plot 2: Time series split visualization
ax2 = axes[0, 1]
ax2.axis('off')

# Draw time series split
n_splits = 5
n_samples = 100

y_positions = np.linspace(0.85, 0.25, n_splits)
bar_height = 0.1

for split_idx, y_pos in enumerate(y_positions):
    # Calculate split sizes
    train_end = int(20 + (split_idx + 1) * 16)
    test_start = train_end
    test_end = test_start + 16

    # Draw timeline
    total_width = 0.8
    train_width = (train_end / n_samples) * total_width
    test_width = ((test_end - test_start) / n_samples) * total_width
    unused_width = total_width - train_width - test_width

    x_start = 0.1

    # Training portion
    rect_train = plt.Rectangle((x_start, y_pos), train_width, bar_height,
                                facecolor=MLBLUE, edgecolor='black', alpha=0.7)
    ax2.add_patch(rect_train)

    # Test portion
    rect_test = plt.Rectangle((x_start + train_width, y_pos), test_width, bar_height,
                               facecolor=MLRED, edgecolor='black', alpha=0.7)
    ax2.add_patch(rect_test)

    # Unused (future)
    if unused_width > 0:
        rect_unused = plt.Rectangle((x_start + train_width + test_width, y_pos),
                                     unused_width, bar_height,
                                     facecolor='lightgray', edgecolor='black', alpha=0.5)
        ax2.add_patch(rect_unused)

    # Split label
    ax2.text(0.05, y_pos + bar_height/2, f'Split {split_idx + 1}',
             fontsize=9, va='center', fontweight='bold')

# Time arrow
ax2.annotate('', xy=(0.9, 0.15), xytext=(0.1, 0.15),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax2.text(0.5, 0.08, 'Time', ha='center', fontsize=10, fontweight='bold')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('TimeSeriesSplit (5 splits)', fontsize=11, fontweight='bold', color=MLPURPLE)

# Legend
ax2.text(0.25, 0.95, 'Train (past)', fontsize=10, color=MLBLUE, fontweight='bold')
ax2.text(0.55, 0.95, 'Test (future)', fontsize=10, color=MLRED, fontweight='bold')

# Plot 3: Code
ax3 = axes[1, 0]
ax3.axis('off')

code = '''
TIMESERIESSPLIT IN SKLEARN

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score

# Create time series CV splitter
tscv = TimeSeriesSplit(
    n_splits=5,
    gap=0,          # Gap between train and test (optional)
    max_train_size=None  # Limit training window (optional)
)


# Use with cross_val_score
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

scores = cross_val_score(pipe, X, y, cv=tscv)
print(f"CV Score: {scores.mean():.3f} +/- {scores.std():.3f}")


# Visualize splits
for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Split {i+1}:")
    print(f"  Train: {train_idx[0]} to {train_idx[-1]}")
    print(f"  Test:  {test_idx[0]} to {test_idx[-1]}")


# With gap (e.g., 5 days)
tscv_gap = TimeSeriesSplit(n_splits=5, gap=5)


# With max_train_size (rolling window)
tscv_rolling = TimeSeriesSplit(n_splits=5, max_train_size=252)
'''

ax3.text(0.02, 0.98, code, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax3.set_title('TimeSeriesSplit Code', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 4: Finance example
ax4 = axes[1, 1]
ax4.axis('off')

finance = '''
FINANCE APPLICATION

STOCK RETURN PREDICTION:
------------------------
# Data: Daily returns, 5 years
# Goal: Predict next day's return

from sklearn.model_selection import TimeSeriesSplit

# Use 1 year for each test period
tscv = TimeSeriesSplit(n_splits=4)

# Gap of 1 day to avoid look-ahead
# (if using lagged features)
tscv = TimeSeriesSplit(n_splits=4, gap=1)


PIPELINE FOR FINANCE:
--------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

# Time series CV
scores = cross_val_score(
    pipe, X, y,
    cv=tscv,
    scoring='neg_mean_squared_error'
)


TYPICAL SPLIT (5 years):
------------------------
Split 1: Train 2019, Test 2020
Split 2: Train 2019-2020, Test 2021
Split 3: Train 2019-2021, Test 2022
Split 4: Train 2019-2022, Test 2023


KEY PRINCIPLE:
--------------
Never train on future data!
Test performance should reflect
real trading conditions.
'''

ax4.text(0.02, 0.98, finance, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Finance Application', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
