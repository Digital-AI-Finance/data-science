"""Quarterly Comparison - Bar chart with error bars"""
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
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D62728'

np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
df = pd.DataFrame({'Price': prices}, index=dates)

fig, ax = plt.subplots(figsize=(10, 6))

quarterly = df.resample('Q').agg({'Price': ['mean', 'std']})
quarterly.columns = ['Mean', 'Std']
quarterly['Quarter'] = ['Q1', 'Q2', 'Q3', 'Q4']

colors = [MLBLUE, MLGREEN, MLORANGE, MLRED]
bars = ax.bar(quarterly['Quarter'], quarterly['Mean'], yerr=quarterly['Std'],
              color=colors, alpha=0.7, capsize=5, edgecolor='black')

ax.set_xlabel('Quarter', fontsize=10)
ax.set_ylabel('Avg Price ($)', fontsize=10)
ax.set_title("Quarterly: resample('Q').agg(['mean','std'])", fontsize=12,
             fontweight='bold', color=MLPURPLE, family='monospace')
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'${height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
