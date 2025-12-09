"""Data Augmentation - Creating More Training Data"""
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
fig.suptitle('Data Augmentation: Creating More Training Data', fontsize=14, fontweight='bold', color=MLPURPLE)

# Plot 1: Concept
ax1 = axes[0, 0]
ax1.axis('off')

concept = '''
DATA AUGMENTATION

THE IDEA:
---------
Create NEW training examples by modifying
existing ones in ways that preserve the label.

More data -> Better generalization -> Less overfitting


FOR IMAGES (common):
--------------------
- Rotation
- Flipping (horizontal/vertical)
- Zooming
- Cropping
- Color adjustments
- Adding noise


FOR TABULAR DATA (finance):
---------------------------
- Adding small noise to features
- SMOTE for class imbalance
- Synthetic data generation
- Bootstrap sampling with noise


FOR TIME SERIES:
----------------
- Window slicing (different start points)
- Time warping
- Magnitude scaling
- Adding noise
- Jittering


WHY IT WORKS:
-------------
1. Increases effective dataset size
2. Teaches model invariances
3. Reduces overfitting
4. Acts as regularization


KEY PRINCIPLE:
--------------
Augmented data must be PLAUSIBLE.
Don't create unrealistic examples!
'''

ax1.text(0.02, 0.98, concept, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax1.set_title('Data Augmentation Concept', fontsize=11, fontweight='bold', color=MLPURPLE)

# Plot 2: Tabular data augmentation visualization
ax2 = axes[0, 1]

# Original data points
np.random.seed(42)
n_orig = 20
x_orig = np.random.randn(n_orig)
y_orig = np.random.randn(n_orig)

# Augmented (with noise)
n_aug = 60
noise_scale = 0.15
x_aug = np.repeat(x_orig, 3) + np.random.randn(n_aug) * noise_scale
y_aug = np.repeat(y_orig, 3) + np.random.randn(n_aug) * noise_scale

ax2.scatter(x_aug, y_aug, color=MLBLUE, alpha=0.3, s=30, label='Augmented (with noise)')
ax2.scatter(x_orig, y_orig, color=MLRED, s=80, edgecolors='black', linewidths=1, label='Original')

# Draw connections
for i in range(n_orig):
    for j in range(3):
        ax2.plot([x_orig[i], x_aug[i*3 + j]], [y_orig[i], y_aug[i*3 + j]],
                color='gray', linewidth=0.5, alpha=0.3)

ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('Noise-Based Augmentation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax2.text(-2, 2, f'Original: {n_orig}\nAugmented: {n_aug}', fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Plot 3: Time series augmentation
ax3 = axes[1, 0]

# Original time series
t = np.linspace(0, 4*np.pi, 100)
original = np.sin(t) + 0.5 * np.sin(3*t)

# Augmented versions
jittered = original + np.random.randn(100) * 0.1
scaled = original * 1.2
warped = np.sin(t * 1.1) + 0.5 * np.sin(3*t * 1.1)  # Time warping

ax3.plot(t, original, color='black', linewidth=2.5, label='Original')
ax3.plot(t, jittered, color=MLBLUE, linewidth=1.5, alpha=0.7, label='Jittered')
ax3.plot(t, scaled, color=MLGREEN, linewidth=1.5, alpha=0.7, label='Scaled')
ax3.plot(t, warped, color=MLORANGE, linewidth=1.5, alpha=0.7, label='Time Warped')

ax3.set_xlabel('Time')
ax3.set_ylabel('Value')
ax3.set_title('Time Series Augmentation', fontsize=11, fontweight='bold', color=MLPURPLE)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(alpha=0.3)

# Plot 4: Finance-specific augmentation
ax4 = axes[1, 1]
ax4.axis('off')

finance = '''
AUGMENTATION FOR FINANCIAL DATA

NOISE INJECTION:
----------------
# Add small Gaussian noise to features
def augment_with_noise(X, noise_factor=0.01):
    noise = np.random.randn(*X.shape) * noise_factor
    return X + noise * np.std(X, axis=0)


BOOTSTRAP WITH NOISE:
---------------------
def bootstrap_augment(X, y, n_samples):
    indices = np.random.choice(len(X), n_samples, replace=True)
    X_boot = X[indices] + np.random.randn(n_samples, X.shape[1]) * 0.01
    y_boot = y[indices]
    return X_boot, y_boot


SYNTHETIC RETURNS:
------------------
# Generate synthetic returns with same statistics
def generate_synthetic_returns(returns, n_synthetic):
    mu = np.mean(returns)
    sigma = np.std(returns)
    return np.random.normal(mu, sigma, n_synthetic)


WINDOW SLICING (TIME SERIES):
-----------------------------
# Create multiple training samples from one series
def window_slice(series, window_size, stride=1):
    windows = []
    for i in range(0, len(series) - window_size, stride):
        windows.append(series[i:i+window_size])
    return np.array(windows)


IMPORTANT CAUTIONS:
-------------------
1. Don't augment test data!
2. Keep augmented data realistic
3. Preserve temporal order for time series
4. Don't leak future information
5. Validate on REAL data only


WHEN TO USE:
------------
- Small dataset (< 1000 samples)
- High overfitting despite regularization
- Class imbalance (use SMOTE)
- Need more robust model
'''

ax4.text(0.02, 0.98, finance, transform=ax4.transAxes, fontsize=8,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=MLLAVENDER, alpha=0.8))
ax4.set_title('Finance-Specific Augmentation', fontsize=11, fontweight='bold', color=MLPURPLE)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
