"""Residuals Analysis - What remains after decomposition"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

MLPURPLE = '#3333B2'
MLRED = '#D62728'
MLORANGE = '#FF7F0E'

np.random.seed(42)
n = 365 * 2
dates = pd.date_range('2023-01-01', periods=n, freq='D')
trend = np.linspace(100, 180, n)
seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.randn(n) * 3
series = trend + seasonal + noise

df = pd.DataFrame({'Value': series}, index=dates)
detrended = df['Value'] - trend
residuals = detrended - seasonal

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df.index, residuals, color=MLRED, linewidth=0.8, alpha=0.7)
ax.axhline(0, color='black', linewidth=1)
ax.axhline(residuals.mean() + 2*residuals.std(), color=MLORANGE, linestyle='--', linewidth=1.5, label='+/- 2 Std')
ax.axhline(residuals.mean() - 2*residuals.std(), color=MLORANGE, linestyle='--', linewidth=1.5)
ax.fill_between(df.index, residuals.mean() - 2*residuals.std(), residuals.mean() + 2*residuals.std(),
                color=MLORANGE, alpha=0.1)

ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Residual', fontsize=10)
ax.set_title(f'Residuals: Std = {residuals.std():.2f}', fontsize=12, fontweight='bold', color=MLPURPLE)
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
