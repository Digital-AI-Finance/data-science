"""Split-Apply-Combine - GroupBy conceptual diagram"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
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

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Title
ax.text(8, 8.7, 'GroupBy: Split-Apply-Combine Pattern', ha='center',
        fontsize=16, fontweight='bold', color=MLPURPLE)

# Step 1: Original Data
orig_box = FancyBboxPatch((0.5, 3), 3, 4.5, boxstyle="round,pad=0.1",
                          edgecolor=MLPURPLE, facecolor='white', linewidth=2)
ax.add_patch(orig_box)
ax.text(2, 7.2, 'Original Data', ha='center', fontsize=11, fontweight='bold', color=MLPURPLE)

# Sample data
data = [
    ('AAPL', 'Tech', 150),
    ('MSFT', 'Tech', 350),
    ('JPM', 'Finance', 140),
    ('GS', 'Finance', 380),
    ('NVDA', 'Tech', 480),
]
headers = ['Stock', 'Sector', 'Price']
for i, h in enumerate(headers):
    ax.text(0.8 + i*0.9, 6.7, h, ha='center', fontsize=8, fontweight='bold', color='gray')
for i, (stock, sector, price) in enumerate(data):
    y = 6.2 - i*0.6
    color = MLBLUE if sector == 'Tech' else MLORANGE
    ax.text(0.8, y, stock, ha='center', fontsize=8)
    ax.text(1.7, y, sector, ha='center', fontsize=8, color=color, fontweight='bold')
    ax.text(2.6, y, f'${price}', ha='center', fontsize=8)

# Arrow to Split
ax.annotate('', xy=(4.5, 5.5), xytext=(3.5, 5.5),
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))
ax.text(4, 6, '1. SPLIT', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)

# Step 2: Split Groups
# Tech group
tech_box = FancyBboxPatch((5, 5.5), 2.5, 2, boxstyle="round,pad=0.1",
                          edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax.add_patch(tech_box)
ax.text(6.25, 7.2, 'Tech Group', ha='center', fontsize=10, fontweight='bold', color=MLBLUE)
tech_data = [('AAPL', 150), ('MSFT', 350), ('NVDA', 480)]
for i, (s, p) in enumerate(tech_data):
    ax.text(5.4, 6.7 - i*0.4, f'{s}: ${p}', fontsize=8)

# Finance group
fin_box = FancyBboxPatch((5, 2.5), 2.5, 2, boxstyle="round,pad=0.1",
                         edgecolor=MLORANGE, facecolor='#FFF0E6', linewidth=2)
ax.add_patch(fin_box)
ax.text(6.25, 4.2, 'Finance Group', ha='center', fontsize=10, fontweight='bold', color=MLORANGE)
fin_data = [('JPM', 140), ('GS', 380)]
for i, (s, p) in enumerate(fin_data):
    ax.text(5.4, 3.7 - i*0.4, f'{s}: ${p}', fontsize=8)

# Arrow to Apply
ax.annotate('', xy=(8.5, 6.5), xytext=(7.5, 6.5),
            arrowprops=dict(arrowstyle='->', color=MLBLUE, lw=2))
ax.annotate('', xy=(8.5, 3.5), xytext=(7.5, 3.5),
            arrowprops=dict(arrowstyle='->', color=MLORANGE, lw=2))
ax.text(8, 5.2, '2. APPLY', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)
ax.text(8, 4.8, 'mean()', ha='center', fontsize=9, family='monospace')

# Step 3: Applied results
tech_result = FancyBboxPatch((9, 6), 2, 1, boxstyle="round,pad=0.1",
                             edgecolor=MLBLUE, facecolor='#E6F0FF', linewidth=2)
ax.add_patch(tech_result)
ax.text(10, 6.5, f'Tech: ${(150+350+480)/3:.0f}', ha='center', fontsize=10,
        fontweight='bold', color=MLBLUE)

fin_result = FancyBboxPatch((9, 3), 2, 1, boxstyle="round,pad=0.1",
                            edgecolor=MLORANGE, facecolor='#FFF0E6', linewidth=2)
ax.add_patch(fin_result)
ax.text(10, 3.5, f'Finance: ${(140+380)/2:.0f}', ha='center', fontsize=10,
        fontweight='bold', color=MLORANGE)

# Arrow to Combine
ax.annotate('', xy=(12.5, 5), xytext=(11, 6.5),
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))
ax.annotate('', xy=(12.5, 5), xytext=(11, 3.5),
            arrowprops=dict(arrowstyle='->', color=MLPURPLE, lw=2))
ax.text(11.75, 5.5, '3. COMBINE', ha='center', fontsize=10, fontweight='bold', color=MLPURPLE)

# Final result
final_box = FancyBboxPatch((12.5, 3.5), 3, 3, boxstyle="round,pad=0.1",
                           edgecolor=MLGREEN, facecolor='#E6FFE6', linewidth=2)
ax.add_patch(final_box)
ax.text(14, 6.2, 'Final Result', ha='center', fontsize=11, fontweight='bold', color=MLGREEN)
ax.text(13, 5.5, 'Sector', ha='center', fontsize=9, fontweight='bold', color='gray')
ax.text(14.7, 5.5, 'Avg Price', ha='center', fontsize=9, fontweight='bold', color='gray')
ax.text(13, 5.0, 'Tech', ha='center', fontsize=9, color=MLBLUE)
ax.text(14.7, 5.0, '$327', ha='center', fontsize=9)
ax.text(13, 4.5, 'Finance', ha='center', fontsize=9, color=MLORANGE)
ax.text(14.7, 4.5, '$260', ha='center', fontsize=9)

# Code example at bottom
code_box = FancyBboxPatch((3, 0.5), 10, 1.5, boxstyle="round,pad=0.1",
                          edgecolor=MLPURPLE, facecolor=MLLAVENDER, alpha=0.3)
ax.add_patch(code_box)
ax.text(8, 1.6, "df.groupby('Sector')['Price'].mean()", ha='center',
        fontsize=12, family='monospace', color=MLPURPLE, fontweight='bold')
ax.text(8, 1.0, 'GroupBy = Split + Apply + Combine', ha='center',
        fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()
