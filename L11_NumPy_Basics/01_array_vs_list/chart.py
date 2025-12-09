"""Array vs List - Performance comparison"""
import matplotlib.pyplot as plt
import numpy as np
import time
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

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left panel: Performance comparison
ax1 = axes[0]
sizes = [1000, 10000, 100000, 1000000]
list_times = [0.1, 1.2, 12, 120]  # Simulated times in ms
numpy_times = [0.01, 0.05, 0.5, 5]

x = np.arange(len(sizes))
width = 0.35

bars1 = ax1.bar(x - width/2, list_times, width, label='Python List', color=MLRED, alpha=0.7)
bars2 = ax1.bar(x + width/2, numpy_times, width, label='NumPy Array', color=MLGREEN, alpha=0.7)

ax1.set_xlabel('Array Size', fontsize=11)
ax1.set_ylabel('Time (milliseconds)', fontsize=11)
ax1.set_title('Performance: List vs NumPy Array', fontsize=12, fontweight='bold', color=MLPURPLE)
ax1.set_xticks(x)
ax1.set_xticklabels(['1K', '10K', '100K', '1M'])
ax1.legend(fontsize=10)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# Add speedup annotations
for i, (lt, nt) in enumerate(zip(list_times, numpy_times)):
    speedup = lt / nt
    ax1.annotate(f'{speedup:.0f}x faster', xy=(x[i], nt), xytext=(0, -25),
                textcoords='offset points', ha='center', fontsize=8, color=MLGREEN, fontweight='bold')

# Right panel: Feature comparison
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

ax2.text(5, 9.5, 'NumPy Array vs Python List', ha='center',
         fontsize=14, fontweight='bold', color=MLPURPLE)

# Comparison table
features = [
    ('Element Types', 'Mixed types', 'Single type (homogeneous)'),
    ('Memory', 'Scattered', 'Contiguous block'),
    ('Math Operations', 'Loop required', 'Vectorized (fast)'),
    ('Broadcasting', 'Not supported', 'Supported'),
    ('Size After Creation', 'Dynamic (mutable)', 'Fixed'),
]

# Headers
ax2.text(1, 8.5, 'Feature', fontsize=10, fontweight='bold', color=MLPURPLE)
ax2.text(4, 8.5, 'Python List', fontsize=10, fontweight='bold', color=MLRED)
ax2.text(7.5, 8.5, 'NumPy Array', fontsize=10, fontweight='bold', color=MLGREEN)

for i, (feat, list_val, numpy_val) in enumerate(features):
    y = 7.8 - i * 1.2
    ax2.text(1, y, feat, fontsize=9, color='black')
    ax2.text(4, y, list_val, fontsize=9, color=MLRED)
    ax2.text(7.5, y, numpy_val, fontsize=9, color=MLGREEN)

# Code example
ax2.text(5, 2.5, 'Example: Element-wise multiplication', ha='center',
         fontsize=10, fontweight='bold', color=MLPURPLE)
ax2.text(2.5, 1.8, "# List (slow):\n[x*2 for x in my_list]", fontsize=9,
         family='monospace', color=MLRED, ha='left')
ax2.text(7.5, 1.8, "# NumPy (fast):\nmy_array * 2", fontsize=9,
         family='monospace', color=MLGREEN, ha='left')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
