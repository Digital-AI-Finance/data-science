"""Imputation Techniques - Comparison of imputation methods for financial data"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Standard matplotlib configuration
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

# Course colors
MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

# Generate synthetic return data
np.random.seed(42)
n = 100
returns = np.random.normal(0.001, 0.02, n)  # Daily returns
returns_with_na = returns.copy()

# Introduce missing values
missing_idx = np.random.choice(n, 15, replace=False)
returns_with_na[missing_idx] = np.nan

# Create DataFrame
df = pd.DataFrame({'Original': returns, 'With_Missing': returns_with_na})

# Different imputation methods
df['Mean'] = df['With_Missing'].fillna(df['With_Missing'].mean())
df['Median'] = df['With_Missing'].fillna(df['With_Missing'].median())
df['Zero'] = df['With_Missing'].fillna(0)
df['Rolling_Mean'] = df['With_Missing'].fillna(df['With_Missing'].rolling(5, min_periods=1).mean())

# Calculate error metrics for each method
methods = ['Mean', 'Median', 'Zero', 'Rolling_Mean']
errors = {}
for method in methods:
    imputed_vals = df.loc[missing_idx, method]
    true_vals = df.loc[missing_idx, 'Original']
    mae = np.abs(imputed_vals - true_vals).mean()
    rmse = np.sqrt(((imputed_vals - true_vals)**2).mean())
    errors[method] = {'MAE': mae, 'RMSE': rmse}

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Bar chart of errors
ax1 = axes[0]
x = np.arange(len(methods))
width = 0.35

mae_vals = [errors[m]['MAE'] * 100 for m in methods]
rmse_vals = [errors[m]['RMSE'] * 100 for m in methods]

bars1 = ax1.bar(x - width/2, mae_vals, width, label='MAE', color=MLBLUE)
bars2 = ax1.bar(x + width/2, rmse_vals, width, label='RMSE', color=MLORANGE)

ax1.set_xlabel('Imputation Method', fontsize=11)
ax1.set_ylabel('Error (% points)', fontsize=11)
ax1.set_title('Imputation Error Comparison', fontsize=12, fontweight='bold', color=MLPURPLE)
ax1.set_xticks(x)
ax1.set_xticklabels(['Mean', 'Median', 'Zero', 'Rolling\nMean'], fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

# Right plot: Distribution comparison
ax2 = axes[1]
bins = np.linspace(-0.06, 0.06, 25)

ax2.hist(df['Original'], bins=bins, alpha=0.7, label='True Distribution',
         color=MLPURPLE, density=True)
ax2.hist(df['Mean'], bins=bins, alpha=0.5, label='After Mean Imputation',
         color=MLBLUE, density=True, histtype='step', linewidth=2)
ax2.hist(df['Median'], bins=bins, alpha=0.5, label='After Median Imputation',
         color=MLORANGE, density=True, histtype='step', linewidth=2)

ax2.axvline(df['Original'].mean(), color=MLPURPLE, linestyle='--',
            label=f'True Mean: {df["Original"].mean()*100:.2f}%')
ax2.set_xlabel('Daily Returns', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Distribution After Imputation', fontsize=12, fontweight='bold', color=MLPURPLE)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle('Imputation Method Comparison for Stock Returns', fontsize=14,
             fontweight='bold', color=MLPURPLE, y=1.02)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
